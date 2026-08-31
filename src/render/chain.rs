//! FilterChain: ordered, mixed GLSL/ONNX stages over one shared GlContext.
//!
//! GLSL->GLSL stays GPU-resident; the CPU round-trip happens only at
//! GLSL<->ONNX boundaries. Expensive resources (ONNX sessions, parsed
//! shaders) are cached by path in the factory so live chain swaps are cheap.

use super::gl::{Dtype, GlContext, GpuTex};
use super::glsl_engine::GlslEngine;
use super::mpv::UserShader;
use super::onnx_stage::{OnnxProvider, OnnxStage, PreparedDmlSharedOutput};
use super::{vulkan_multipass, vulkan_onepass};
use crate::core::config::{OnnxBackendPreference, StageKind, StageSpec, resolve_path};
use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::rc::Rc;
// ONNX stages are Arc<Mutex<…>> (not Rc<RefCell<…>>) so frame-interpolation
// inference can run on a worker thread while the engine keeps presenting.
use std::sync::{Arc, Mutex};

const BUILTIN_NEOFLOW_ENABLED: bool = false;

thread_local! {
    // v653: zero-resize identity pass used when a cross-GPU DirectML
    // interpolation output has no post-GLSL stages. Vulkan still imports the
    // app-owned D3D12 NCHW buffer on the selected compute GPU and performs the
    // NCHW->RGBA conversion there, so the DirectML output boundary remains
    // resident even for a RIFE-only chain. The final Vulkan->CPU->presentation
    // bridge remains unchanged and is the only cross-adapter copy.
    static DML_VULKAN_IDENTITY_SHADER: Rc<UserShader> = Rc::new(UserShader::parse(
        "DmlVulkanIdentity.glsl",
        r#"//!HOOK RGB
//!BIND HOOKED
//!DESC DML Vulkan identity handoff
vec4 hook()
{
    return HOOKED_tex(HOOKED_pos);
}
"#,
    ));
}

fn dml_vulkan_identity_shader() -> Rc<UserShader> {
    DML_VULKAN_IDENTITY_SHADER.with(Clone::clone)
}

// DirectML temporal restoration models have shown corrupted output and a very
// steep cost increase at high input resolutions. Keep only the temporal ONNX
// inference at <=1080 pixels high, preserving aspect ratio; following stages
// can upscale again normally. TensorRT/CUDA, single-frame ONNX and frame
// interpolation are intentionally untouched.
const DIRECTML_TEMPORAL_MAX_HEIGHT: i32 = 1080;

fn upload_vulkan_bridge_rgba8(
    gc: &mut GlContext,
    dtype: Dtype,
    width: i32,
    height: i32,
    rgba8: &[u8],
) -> GpuTex {
    match dtype {
        Dtype::U8 => gc.upload_rgba8(width, height, rgba8),
        Dtype::F16 => {
            let mut rgba16 = Vec::with_capacity(rgba8.len() * 2);
            for &value in rgba8 {
                rgba16.extend_from_slice(
                    &half::f16::from_f32(value as f32 * (1.0 / 255.0))
                        .to_bits()
                        .to_ne_bytes(),
                );
            }
            gc.upload_rgba16f(width, height, &rgba16)
        }
        Dtype::U32 => unreachable!("Vulkan GLSL bridge rejects U32 input"),
    }
}

pub enum Stage {
    Glsl {
        name: String,
        key: String,
        shader: Rc<UserShader>,
    },
    Onnx {
        name: String,
        key: String,
        stage: Arc<Mutex<OnnxStage>>,
        // Read once when the session is created. Querying this through the
        // mutex on every render tick serialized GL presentation behind the
        // DirectML RIFE worker and also delayed live chain edits.
        is_interp: bool,
    },
    /// built-in GPU frame interpolation (handled by the engine)
    Flow {
        name: String,
        external_source: Option<Rc<String>>,
    },
}

/// What the engine uses to synthesize in-between frames.
#[derive(Clone)]
pub enum InterpHandle {
    Onnx {
        name: String,
        stage: Arc<Mutex<OnnxStage>>,
    },
    Flow {
        name: String,
        external_source: Option<Rc<String>>,
    },
}

impl Stage {
    pub fn name(&self) -> &str {
        match self {
            Stage::Glsl { name, .. } => name,
            Stage::Onnx { name, .. } => name,
            Stage::Flow { name, .. } => name,
        }
    }
    pub fn kind(&self) -> StageKind {
        match self {
            Stage::Glsl { .. } => StageKind::Glsl,
            Stage::Onnx { .. } => StageKind::Onnx,
            Stage::Flow { .. } => StageKind::Flow,
        }
    }
    pub fn is_interp(&self) -> bool {
        match self {
            Stage::Flow { .. } => true,
            Stage::Onnx { is_interp, .. } => *is_interp,
            Stage::Glsl { .. } => false,
        }
    }

    pub fn metrics_name(&self) -> String {
        match self {
            Stage::Onnx { name, stage, .. } => {
                let provider = match stage.lock().unwrap().provider {
                    OnnxProvider::TensorRT => "TensorRT",
                    OnnxProvider::DirectML => "DirectML",
                    OnnxProvider::Cuda => "CUDA",
                };
                format!("{name} [{provider}]")
            }
            _ => self.name().to_string(),
        }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct OnnxCacheKey {
    path: String,
    preference: OnnxBackendPreference,
    dml_adapter: Option<i32>,
    trt_device: Option<i32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OnnxBackendUsage {
    pub tensorrt: usize,
    pub cuda: usize,
    pub directml: usize,
    pub directml_fallback: usize,
}

pub struct StageFactory {
    base_dir: PathBuf,
    gpu_adapter: Option<i32>,
    onnx_preference: OnnxBackendPreference,
    trt_device_id: Option<i32>,
    trt_cache_root: PathBuf,
    shaders: HashMap<String, Rc<UserShader>>,
    onnx: HashMap<OnnxCacheKey, (Arc<Mutex<OnnxStage>>, bool)>,
    onnx_lru: VecDeque<OnnxCacheKey>,
}

impl StageFactory {
    pub fn new(base_dir: PathBuf) -> Self {
        Self {
            trt_cache_root: base_dir.join("cache").join("TensorRT"),
            base_dir,
            gpu_adapter: None,
            onnx_preference: OnnxBackendPreference::DirectML,
            trt_device_id: None,
            shaders: HashMap::new(),
            onnx: HashMap::new(),
            onnx_lru: VecDeque::new(),
        }
    }

    pub fn set_gpu_adapter(&mut self, gpu_adapter: Option<i32>) {
        if self.gpu_adapter != gpu_adapter {
            log::info!(
                "gpu-selection: old={:?} new={:?}; clearing DirectML sessions",
                self.gpu_adapter,
                gpu_adapter
            );
            self.gpu_adapter = gpu_adapter;
            self.clear_onnx_cache();
        }
    }

    pub fn set_onnx_backend(
        &mut self,
        preference: OnnxBackendPreference,
        trt_device_id: Option<i32>,
        trt_cache_root: PathBuf,
    ) {
        if self.onnx_preference != preference
            || self.trt_device_id != trt_device_id
            || self.trt_cache_root != trt_cache_root
        {
            self.onnx_preference = preference;
            self.trt_device_id = trt_device_id;
            self.trt_cache_root = trt_cache_root;
            self.clear_onnx_cache();
        }
    }

    pub fn onnx_preference(&self) -> OnnxBackendPreference {
        self.onnx_preference
    }

    pub fn fork_for_backend(
        &self,
        preference: OnnxBackendPreference,
        trt_device_id: Option<i32>,
        trt_cache_root: PathBuf,
    ) -> Self {
        // Backend switching is transactional, but rebuilding the candidate from
        // an empty cache needlessly throws away a warm TensorRT session. Carry
        // compatible Arc-backed sessions into the candidate so DirectML ->
        // TensorRT -> DirectML/Stop -> TensorRT can reuse the same provider
        // session when the physical GPU, CUDA device and runtime cache root are
        // unchanged. A real GPU/runtime change still drops TensorRT entries.
        let same_trt_identity =
            self.trt_device_id == trt_device_id && self.trt_cache_root == trt_cache_root;
        let mut onnx = self.onnx.clone();
        if !same_trt_identity {
            onnx.retain(|key, _| key.preference != OnnxBackendPreference::TensorRT);
        }
        let remaining: HashSet<OnnxCacheKey> = onnx.keys().cloned().collect();
        let mut onnx_lru = self.onnx_lru.clone();
        onnx_lru.retain(|key| remaining.contains(key));
        let carried_tensorrt = onnx
            .keys()
            .filter(|key| key.preference == OnnxBackendPreference::TensorRT)
            .count();
        if carried_tensorrt > 0 {
            log::info!(
                "onnx-session-backend-carry: target={preference:?} kept_tensorrt={} policy=same-gpu-runtime",
                carried_tensorrt
            );
        }
        Self {
            base_dir: self.base_dir.clone(),
            gpu_adapter: self.gpu_adapter,
            onnx_preference: preference,
            trt_device_id,
            trt_cache_root,
            shaders: self.shaders.clone(),
            onnx,
            onnx_lru,
        }
    }

    pub fn build(&mut self, spec: &StageSpec) -> Result<Stage> {
        let path = resolve_path(&self.base_dir, &spec.path);
        let key = path.to_string_lossy().into_owned();
        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| spec.path.clone());
        match spec.kind {
            StageKind::Flow => {
                if spec.path.starts_with("builtin:") && !BUILTIN_NEOFLOW_ENABLED {
                    anyhow::bail!("built-in NeoFlow is temporarily disabled");
                }
                let external_source = if spec.path.starts_with("builtin:") {
                    None
                } else {
                    let source = std::fs::read_to_string(&path)
                        .with_context(|| format!("read external NeoFlow {}", path.display()))?;
                    if !source.contains("NEOFLOW_PASS_COMPOSITE")
                        && !(source.contains("NEOFLOW_PASS_OWNER_CLEAR")
                            && source.contains("NEOFLOW_PASS_FINAL"))
                        && !((source.contains("NeoFlow GameDIS")
                            || source.contains("NeoFlow GameMesh")
                            || source.contains("NeoFlow HybridCadence")
                            || source.contains("NeoFlow CausalStable"))
                            && source.contains("NF_PASS_FINAL"))
                    {
                        anyhow::bail!("external NeoFlow pass markers not found");
                    }
                    Some(Rc::new(source))
                };
                Ok(Stage::Flow {
                    name: if external_source.is_some() {
                        name
                    } else {
                        builtin_flow_name(&spec.path).into()
                    },
                    external_source,
                })
            }
            StageKind::Glsl => {
                let base_shader = match self.shaders.get(&key) {
                    Some(s) => s.clone(),
                    None => {
                        let shader =
                            UserShader::load(&key).map_err(|e| anyhow!("load {key}: {e}"))?;
                        log::info!(
                            "glsl-load: name={} passes={} rgb={} chroma_emulation={} compute={} post={}",
                            shader.name(),
                            shader.passes.len(),
                            shader.is_rgb,
                            shader.uses_chroma,
                            shader.is_compute,
                            shader.is_post
                        );
                        let s = Rc::new(shader);
                        self.shaders.insert(key.clone(), s.clone());
                        s
                    }
                };
                let shader = if spec.params.is_empty() {
                    base_shader
                } else {
                    let mut configured = (*base_shader).clone();
                    for param in &mut configured.params {
                        if let Some(value) = spec.params.get(&param.name) {
                            param.value = value.clamp(param.min, param.max);
                        }
                    }
                    Rc::new(configured)
                };
                Ok(Stage::Glsl { name, key, shader })
            }
            StageKind::Onnx => {
                let cache_key = OnnxCacheKey {
                    path: key.clone(),
                    preference: self.onnx_preference,
                    dml_adapter: self.gpu_adapter,
                    trt_device: self.trt_device_id,
                };
                let (stage, is_interp) = match self.onnx.get(&cache_key) {
                    Some((s, is_interp)) => {
                        log::info!(
                            "onnx-session-cache-hit: model={} backend={:?} policy=warm-restart",
                            name,
                            self.onnx_preference
                        );
                        (s.clone(), *is_interp)
                    }
                    None => {
                        let s = Arc::new(Mutex::new(OnnxStage::load_with_preference(
                            &path,
                            self.onnx_preference,
                            self.gpu_adapter,
                            self.trt_device_id,
                            &self.trt_cache_root,
                        )?));
                        let (is_interp, provider, fallback_reason) = {
                            let stage = s.lock().unwrap();
                            (
                                stage.interp != super::onnx_stage::InterpKind::None,
                                stage.provider_desc.clone(),
                                stage.fallback_reason.clone(),
                            )
                        };
                        log::info!(
                            "onnx-backend-stage: model={} provider={} interpolation={} fallback_reason={:?}",
                            name,
                            provider,
                            is_interp,
                            fallback_reason
                        );
                        self.onnx.insert(cache_key.clone(), (s.clone(), is_interp));
                        (s, is_interp)
                    }
                };
                self.touch_onnx_cache_key(&cache_key);
                Ok(Stage::Onnx {
                    name,
                    key,
                    stage,
                    is_interp,
                })
            }
        }
    }

    fn touch_onnx_cache_key(&mut self, key: &OnnxCacheKey) {
        self.onnx_lru.retain(|existing| existing != key);
        self.onnx_lru.push_back(key.clone());
    }

    fn prune_onnx_to(&mut self, stages: &[Stage]) {
        let active_paths: HashSet<&str> = stages
            .iter()
            .filter_map(|st| match st {
                Stage::Onnx { key, .. } => Some(key.as_str()),
                _ => None,
            })
            .collect();

        // DirectML sessions are cheap to recreate and keep the historical
        // single-chain cache policy. TensorRT is different: the serialized
        // engine may already exist on disk while recreating the ORT/TRT
        // execution session still makes the first live invocation visibly
        // cold. Keep a very small MRU set so preset A -> B -> A does not throw
        // away the already-warm TensorRT session. The bound prevents long-run
        // VRAM/process growth when users browse many presets.
        const MAX_TENSORRT_WARM_SESSIONS: usize = 3;
        if self.onnx_preference != OnnxBackendPreference::TensorRT {
            self.onnx
                .retain(|key, _| active_paths.contains(key.path.as_str()));
        } else {
            let mut keep: HashSet<OnnxCacheKey> = self
                .onnx
                .keys()
                .filter(|key| active_paths.contains(key.path.as_str()))
                .cloned()
                .collect();
            let target = MAX_TENSORRT_WARM_SESSIONS.max(keep.len());
            for key in self.onnx_lru.iter().rev() {
                if keep.len() >= target {
                    break;
                }
                if key.preference == OnnxBackendPreference::TensorRT && self.onnx.contains_key(key)
                {
                    keep.insert(key.clone());
                }
            }
            let before = self.onnx.len();
            self.onnx.retain(|key, _| keep.contains(key));
            let dropped = before.saturating_sub(self.onnx.len());
            if dropped > 0 {
                log::info!(
                    "onnx-session-preset-cache: kept={} dropped={} limit={} policy=tensorrt-mru",
                    self.onnx.len(),
                    dropped,
                    MAX_TENSORRT_WARM_SESSIONS
                );
            }
        }
        let remaining: HashSet<OnnxCacheKey> = self.onnx.keys().cloned().collect();
        self.onnx_lru.retain(|key| remaining.contains(key));
    }

    fn evict_onnx_path(&mut self, path: &str) -> usize {
        let before = self.onnx.len();
        self.onnx.retain(|key, _| key.path != path);
        let remaining: HashSet<OnnxCacheKey> = self.onnx.keys().cloned().collect();
        self.onnx_lru.retain(|key| remaining.contains(key));
        before.saturating_sub(self.onnx.len())
    }

    pub fn clear_onnx_cache(&mut self) {
        self.onnx.clear();
        self.onnx_lru.clear();
    }

    /// Explicit-GPU live interpolation replacement must be able to emulate the
    /// DirectML part of Stop -> Start without throwing away warm TensorRT
    /// Sessions. Drop every cached DirectML stage so the replacement chain gets
    /// a fresh ORT/DML provider after the old worker and bridges are retired.
    pub fn drop_directml_sessions(&mut self) -> usize {
        let before = self.onnx.len();
        self.onnx.retain(|_key, (stage, _)| {
            let provider = stage
                .lock()
                .unwrap_or_else(|poisoned| {
                    log::error!(
                        "onnx-stage-lock-poisoned: action=recover-for-explicit-gpu-session-drop"
                    );
                    poisoned.into_inner()
                })
                .provider;
            provider != OnnxProvider::DirectML
        });
        let remaining: HashSet<OnnxCacheKey> = self.onnx.keys().cloned().collect();
        self.onnx_lru.retain(|key| remaining.contains(key));
        let dropped = before.saturating_sub(self.onnx.len());
        if dropped > 0 {
            log::info!(
                "onnx-session-explicit-gpu-handoff: dropped_directml={} kept_non_directml={} policy=fresh-dml-preserve-tensorrt",
                dropped,
                self.onnx.len()
            );
        }
        dropped
    }

    /// Capture Stop must release per-capture GL/CUDA/DML bridges, but a TensorRT
    /// ORT session is intentionally kept warm across Stop -> Start. Preset
    /// changes keep only a bounded recent TensorRT MRU set via `prune_onnx_to`,
    /// while real backend/GPU/geometry changes continue to clear everything.
    pub fn retain_tensorrt_sessions_for_capture_restart(&mut self) -> usize {
        let before = self.onnx.len();
        self.onnx
            .retain(|key, _| key.preference == OnnxBackendPreference::TensorRT);
        let remaining: HashSet<OnnxCacheKey> = self.onnx.keys().cloned().collect();
        self.onnx_lru.retain(|key| remaining.contains(key));
        let kept = self.onnx.len();
        if before != kept || kept > 0 {
            log::info!(
                "onnx-session-restart-cache: kept_tensorrt={} dropped_non_tensorrt={} policy=warm-restart",
                kept,
                before.saturating_sub(kept)
            );
        }
        kept
    }
}

pub fn builtin_flow_name(_path: &str) -> &'static str {
    "NeoFlow"
}

fn disambiguate_stage_metric_labels(names: &[String]) -> Vec<String> {
    let mut totals: HashMap<&str, usize> = HashMap::new();
    for name in names {
        *totals.entry(name.as_str()).or_default() += 1;
    }

    let mut seen: HashMap<&str, usize> = HashMap::new();
    names
        .iter()
        .map(|name| {
            let occurrence = seen.entry(name.as_str()).or_default();
            *occurrence += 1;
            if totals.get(name.as_str()).copied().unwrap_or(1) > 1 {
                format!("{name} #{}", *occurrence)
            } else {
                name.clone()
            }
        })
        .collect()
}

#[derive(Clone)]
struct VulkanResidentBatch {
    start: usize,
    end: usize,
    shader: Rc<UserShader>,
    stage_names: Vec<String>,
    pass_counts: Vec<usize>,
}

fn vulkan_batch_builtin_resource(name: &str) -> bool {
    matches!(
        name.to_ascii_uppercase().as_str(),
        "MAIN"
            | "HOOKED"
            | "NATIVE"
            | "MAINPRESUB"
            | "OUTPUT"
            | "SCALED"
            | "PREKERNEL"
            | "POSTKERNEL"
            | "RGB"
    )
}

fn vulkan_batch_shader_isolated(shader: &UserShader) -> bool {
    if shader.passes.is_empty()
        || shader.name() == "NeoDeint.glsl"
        || !shader.is_rgb
        || shader.uses_chroma
        || shader.is_compute
        || shader.is_post
        || shader.textures.iter().any(|texture| texture.storage)
    {
        return false;
    }

    // A standalone shader cannot legally consume a private resource created
    // by the preceding shader. Reject such ambiguous chains so combining
    // shader graphs cannot accidentally make an otherwise-missing BIND/HOOK
    // resolve from a previous stage.
    let mut local_resources = shader
        .textures
        .iter()
        .map(|texture| texture.name.clone())
        .collect::<HashSet<_>>();
    for pass in &shader.passes {
        for hook in &pass.hooks {
            if !vulkan_batch_builtin_resource(hook) && !local_resources.contains(hook) {
                return false;
            }
        }
        for bind in &pass.binds {
            if !vulkan_batch_builtin_resource(bind) && !local_resources.contains(bind) {
                return false;
            }
        }
        if let Some(save) = &pass.save {
            if !vulkan_batch_builtin_resource(save) {
                local_resources.insert(save.clone());
            }
        }
    }
    true
}

fn build_vulkan_resident_batch(
    stages: &[Stage],
    start: usize,
    end: usize,
    min_stage_count: usize,
) -> Option<VulkanResidentBatch> {
    if start >= end || end > stages.len() || end - start < min_stage_count.max(1) {
        return None;
    }
    let first_shader = match &stages[start] {
        Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader) => shader,
        _ => return None,
    };
    if stages[start..end].iter().any(|stage| {
        !matches!(stage, Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader))
    }) {
        return None;
    }

    let mut passes = Vec::new();
    let mut params = Vec::new();
    let mut textures = Vec::new();
    let mut param_names = HashSet::new();
    let mut texture_names = HashSet::new();
    let mut stage_names = Vec::new();
    let mut pass_counts = Vec::new();
    let mut hasher = DefaultHasher::new();

    for stage in &stages[start..end] {
        let Stage::Glsl { shader, .. } = stage else {
            unreachable!()
        };
        shader.source_hash.hash(&mut hasher);
        stage_names.push(shader.name());
        pass_counts.push(shader.passes.len().max(1));
        passes.extend(shader.passes.iter().cloned());
        for param in &shader.params {
            if !param_names.insert(param.name.clone()) {
                return None;
            }
            params.push(param.clone());
        }
        for texture in &shader.textures {
            if !texture_names.insert(texture.name.clone()) {
                return None;
            }
            textures.push(texture.clone());
        }
    }

    let shader = UserShader {
        path: format!("VulkanResidentChain[{}]", stage_names.join("+")),
        source_hash: hasher.finish(),
        passes,
        is_rgb: first_shader.is_rgb,
        uses_chroma: false,
        is_compute: false,
        is_post: false,
        params,
        textures,
    };
    Some(VulkanResidentBatch {
        start,
        end,
        shader: Rc::new(shader),
        stage_names,
        pass_counts,
    })
}

fn build_vulkan_resident_batches(stages: &[Stage]) -> Vec<VulkanResidentBatch> {
    let mut batches = Vec::new();
    let mut start = 0usize;
    while start < stages.len() {
        if !matches!(
            &stages[start],
            Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader)
        ) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < stages.len()
            && matches!(
                &stages[end],
                Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader)
            )
        {
            end += 1;
        }
        if let Some(batch) = build_vulkan_resident_batch(stages, start, end, 2) {
            batches.push(batch);
        }
        start = end;
    }
    batches
}

pub struct FilterChain {
    pub stages: Vec<Stage>,
    neodeint_bypass: bool,
    dml_temporal_limit_log_keys: HashSet<(usize, i32, i32, i32, i32)>,
    vulkan_resident_batches: Vec<VulkanResidentBatch>,
    dml_vulkan_resident_log_keys: HashSet<u64>,
    dml_vulkan_image_handoff_disabled: bool,
}

fn directml_temporal_limited_size(w: i32, h: i32) -> Option<(i32, i32)> {
    if w <= 0 || h <= DIRECTML_TEMPORAL_MAX_HEIGHT {
        return None;
    }
    let scaled = (w as f64 * DIRECTML_TEMPORAL_MAX_HEIGHT as f64 / h as f64).round() as i32;
    let limited_w = ((scaled.max(2) + 1) / 2) * 2;
    Some((limited_w, DIRECTML_TEMPORAL_MAX_HEIGHT))
}

fn log_directml_temporal_height_limit(
    keys: &mut HashSet<(usize, i32, i32, i32, i32)>,
    stage_index: usize,
    name: &str,
    source: (i32, i32),
    limited: (i32, i32),
) {
    let key = (stage_index, source.0, source.1, limited.0, limited.1);
    if keys.insert(key) {
        log::info!(
            "directml-temporal-height-limit: model={} source={}x{} limited={}x{} max_height={} stage_index={} scaler=spline36 policy=always-for-directml-temporal",
            name,
            source.0,
            source.1,
            limited.0,
            limited.1,
            DIRECTML_TEMPORAL_MAX_HEIGHT,
            stage_index
        );
    }
}

impl FilterChain {
    /// Empty placeholder used only while an explicit-GPU live interpolation
    /// handoff destroys the old provider generation before constructing the
    /// replacement. Keeping a real DirectML stage alive here would recreate the
    /// old/new Session overlap that Stop -> Start naturally avoids.
    pub fn empty() -> Self {
        Self {
            stages: Vec::new(),
            neodeint_bypass: false,
            dml_temporal_limit_log_keys: HashSet::new(),
            vulkan_resident_batches: Vec::new(),
            dml_vulkan_resident_log_keys: HashSet::new(),
            dml_vulkan_image_handoff_disabled: false,
        }
    }

    /// Drain and detach GPU-shared ONNX outputs before a resize, chain swap, or
    /// session teardown. OpenGL must release its imported memory object before
    /// the owning DirectML allocation is dropped.
    pub fn prepare_gpu_transition(&mut self, gc: &mut GlContext) {
        gc.finish();
        for stage in &mut self.stages {
            if let Stage::Onnx { stage, .. } = stage {
                let mut stage = stage.lock().unwrap_or_else(|poisoned| {
                    log::error!("onnx-stage-lock-poisoned: action=recover-for-gpu-retirement");
                    poisoned.into_inner()
                });
                stage.retire_tensorrt_gpu_bridge(gc);
                stage.retire_tensorrt_temporal_gpu_bridge(gc);
                stage.retire_dml_temporal_gpu_bridge(gc);
                stage.retire_interp_gpu_bridges(gc);
                stage.retire_direct_output(gc);
            }
        }
    }

    /// Recreate only ONNX sessions after a real source-dimension change.
    ///
    /// DirectML may retain shape-specific compiled execution state inside an
    /// ORT session. Reusing the same session after a live 1280x720 -> 960x540
    /// transition was observed to make a smaller input about 6x slower until
    /// Stop/Start recreated the session. GLSL stages stay intact and continue
    /// using the resident shader cache; only ONNX stage objects are replaced.
    pub fn rebuild_onnx_sessions_for_geometry(
        &mut self,
        factory: &mut StageFactory,
    ) -> Result<usize> {
        let onnx_specs = self
            .stages
            .iter()
            .enumerate()
            .filter_map(|(index, stage)| match stage {
                Stage::Onnx { key, .. } => Some((
                    index,
                    StageSpec {
                        kind: StageKind::Onnx,
                        path: key.clone(),
                        enabled: true,
                        params: Default::default(),
                    },
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        if onnx_specs.is_empty() {
            return Ok(0);
        }

        // The factory cache owns Arc references to the old shape-specialized
        // sessions. Drop those references before loading replacements.
        factory.clear_onnx_cache();
        let mut replacements = Vec::with_capacity(onnx_specs.len());
        for (index, spec) in onnx_specs {
            let replacement = factory.build(&spec).with_context(|| {
                format!("rebuild ONNX stage after geometry change: {}", spec.path)
            })?;
            anyhow::ensure!(
                matches!(&replacement, Stage::Onnx { .. }),
                "geometry rebuild returned a non-ONNX stage"
            );
            replacements.push((index, replacement));
        }
        let count = replacements.len();
        for (index, replacement) in replacements {
            self.stages[index] = replacement;
        }
        Ok(count)
    }

    pub fn reset_backend_runtime_state(&mut self) {
        for stage in &mut self.stages {
            if let Stage::Onnx { stage, .. } = stage {
                stage.lock().unwrap().reset_backend_runtime_state();
            }
        }
    }

    /// OUTPUT/SCALED-hook shaders (sharpeners): the engine runs these on the
    /// final display-size image after downscaling. Return the same stable
    /// per-stage metric label used by the ordinary in-chain GLSL path so post
    /// shaders participate in GPU timer statistics as well.
    pub fn post_shaders_with_metric_labels(&self) -> Vec<(String, Rc<UserShader>)> {
        let metric_names = self
            .stages
            .iter()
            .map(Stage::metrics_name)
            .collect::<Vec<_>>();
        let metric_labels = disambiguate_stage_metric_labels(&metric_names);
        self.stages
            .iter()
            .enumerate()
            .filter_map(|(index, st)| match st {
                Stage::Glsl { shader, .. } if shader.is_post => {
                    Some((metric_labels[index].clone(), shader.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// First frame-interpolation stage in the chain (engine handles it
    /// specially: it consumes multiple frames and emits in-between frames).
    pub fn interp_stage(&self) -> Option<InterpHandle> {
        self.stages.iter().find_map(|st| match st {
            Stage::Onnx {
                name,
                stage,
                is_interp: true,
                ..
            } => Some(InterpHandle::Onnx {
                name: {
                    let provider = match stage.lock().unwrap().provider {
                        OnnxProvider::TensorRT => "TensorRT",
                        OnnxProvider::DirectML => "DirectML",
                        OnnxProvider::Cuda => "CUDA",
                    };
                    format!("{name} [{provider}]")
                },
                stage: stage.clone(),
            }),
            Stage::Flow {
                name,
                external_source,
            } => Some(InterpHandle::Flow {
                name: name.clone(),
                external_source: external_source.clone(),
            }),
            _ => None,
        })
    }

    pub fn has_interp(&self) -> bool {
        self.stages.iter().any(Stage::is_interp)
    }

    /// True when starting this chain can execute any ONNX inference, including
    /// interpolation stages that are driven by the engine outside process_range.
    pub fn has_onnx(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| matches!(stage, Stage::Onnx { .. }))
    }

    pub fn has_glsl(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| matches!(stage, Stage::Glsl { .. }))
    }

    pub fn has_neodeint(&self) -> bool {
        self.stages
            .iter()
            .any(|stage| matches!(stage, Stage::Glsl { name, .. } if name == "NeoDeint.glsl"))
    }

    /// Bypass Anime IVTC for a frame that the host has identified as already
    /// progressive. Skipping the complete stage is essential: local shader
    /// masks can mistake legitimate diagonal ink for combing and create jaggies.
    pub fn set_neodeint_bypass(&mut self, bypass: bool) {
        self.neodeint_bypass = bypass;
    }

    pub fn neodeint_bypass(&self) -> bool {
        self.neodeint_bypass
    }

    pub fn interp_index(&self) -> Option<usize> {
        self.stages.iter().position(Stage::is_interp)
    }

    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Statistics must follow the user's declared chain order, not the order
    /// in which asynchronous timing probes happen to complete.
    pub fn metric_stage_order(&self) -> Vec<String> {
        let names = self
            .stages
            .iter()
            .map(Stage::metrics_name)
            .collect::<Vec<_>>();
        disambiguate_stage_metric_labels(&names)
    }

    pub fn has_onnx_in_range(&self, start_index: usize) -> bool {
        self.stages.iter().skip(start_index).any(|stage| {
            matches!(
                stage,
                Stage::Onnx {
                    is_interp: false,
                    ..
                }
            )
        })
    }

    pub fn has_glsl_in_range(&self, start_index: usize) -> bool {
        self.stages
            .iter()
            .skip(start_index)
            .any(|stage| matches!(stage, Stage::Glsl { .. }))
    }

    pub fn interp_provider(&self) -> Option<OnnxProvider> {
        self.stages.iter().find_map(|stage| match stage {
            Stage::Onnx {
                stage,
                is_interp: true,
                ..
            } => Some(
                stage
                    .lock()
                    .unwrap_or_else(|poisoned| {
                        log::error!("onnx-stage-lock-poisoned: action=recover-for-interp-provider");
                        poisoned.into_inner()
                    })
                    .provider,
            ),
            _ => None,
        })
    }

    /// Recreate only the active DirectML interpolation session after a live
    /// chain edit changes the stages that feed it. DirectML/ORT can retain
    /// shape-specialized execution state even after the per-capture GPU bridge
    /// has been rebuilt for the new input size; Stop/Start fixed that state by
    /// dropping the non-TensorRT session. Keep TensorRT warm and leave unrelated
    /// ONNX stages untouched.
    pub fn rebuild_directml_interpolation_session(
        &mut self,
        factory: &mut StageFactory,
    ) -> Result<bool> {
        let Some(index) = self.interp_index() else {
            return Ok(false);
        };
        let key = match &self.stages[index] {
            Stage::Onnx {
                key,
                stage,
                is_interp: true,
                ..
            } => {
                let provider = stage
                    .lock()
                    .unwrap_or_else(|poisoned| {
                        log::error!(
                            "onnx-stage-lock-poisoned: action=recover-for-directml-interp-rebuild"
                        );
                        poisoned.into_inner()
                    })
                    .provider;
                if provider != OnnxProvider::DirectML {
                    return Ok(false);
                }
                key.clone()
            }
            _ => return Ok(false),
        };

        let evicted = factory.evict_onnx_path(&key);
        let spec = StageSpec {
            kind: StageKind::Onnx,
            path: key.clone(),
            enabled: true,
            params: Default::default(),
        };
        let replacement = factory.build(&spec).with_context(|| {
            format!(
                "rebuild DirectML interpolation session after live pre-chain route change: {}",
                spec.path
            )
        })?;
        anyhow::ensure!(
            matches!(
                &replacement,
                Stage::Onnx {
                    is_interp: true,
                    ..
                }
            ),
            "DirectML interpolation rebuild returned a non-interpolation stage"
        );
        self.stages[index] = replacement;
        log::info!(
            "directml-interp-session-rebuild: model={} cache_evicted={} result=recreated",
            key,
            evicted
        );
        Ok(true)
    }

    pub fn interpolation_plan(&self) -> Option<(Vec<String>, String, Vec<String>)> {
        let index = self.interp_index()?;
        let pre = self.stages[..index]
            .iter()
            .map(|stage| stage.name().to_string())
            .collect::<Vec<_>>();
        let interp = self.stages[index].name().to_string();
        let post = self.stages[index + 1..]
            .iter()
            .map(|stage| stage.name().to_string())
            .collect::<Vec<_>>();

        // The temporal engine executes exactly these three contiguous ranges.
        // Keep a development-time invariant so future special-case changes
        // cannot silently reorder the user's GUI chain.
        let declared = self
            .stages
            .iter()
            .map(|stage| stage.name().to_string())
            .collect::<Vec<_>>();
        let mut effective = pre.clone();
        effective.push(interp.clone());
        effective.extend(post.iter().cloned());
        debug_assert_eq!(effective, declared, "interpolation execution order changed");

        Some((pre, interp, post))
    }

    pub fn interp_key(&self) -> Option<String> {
        self.stages.iter().find_map(|st| match st {
            Stage::Onnx {
                key,
                stage,
                is_interp: true,
                ..
            } => {
                let provider = stage.lock().unwrap().provider;
                Some(format!("onnx:{key}:{provider:?}"))
            }
            Stage::Flow { name, .. } => Some(format!("flow:{name}")),
            _ => None,
        })
    }

    pub fn onnx_backend_usage(&self) -> OnnxBackendUsage {
        let mut usage = OnnxBackendUsage::default();
        for stage in &self.stages {
            let Stage::Onnx { stage, .. } = stage else {
                continue;
            };
            let stage = stage.lock().unwrap();
            match stage.provider {
                OnnxProvider::TensorRT => usage.tensorrt += 1,
                OnnxProvider::Cuda => usage.cuda += 1,
                OnnxProvider::DirectML => usage.directml += 1,
            }
            if stage.fallback_reason.is_some() {
                usage.directml_fallback += 1;
            }
        }
        usage
    }

    pub fn warmup_candidate(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
        out_size: (i32, i32),
        preserve: &[GpuTex],
    ) -> Result<OnnxBackendUsage> {
        anyhow::ensure!(w > 0 && h > 0, "warmup input size is empty");
        anyhow::ensure!(
            rgba.len() == (w as usize) * (h as usize) * 4,
            "warmup RGBA buffer size mismatch"
        );
        let result = (|| {
            let input = gc.upload_rgba8(w, h, rgba);
            let final_tex = if let Some(interp_index) = self.interp_index() {
                let pre = self.process_range(gc, input, out_size, 0, interp_index, None)?;
                let rgb = gc.download_rgb8(pre);
                let (iw, ih, interpolated) = match &mut self.stages[interp_index] {
                    Stage::Onnx { stage, .. } => {
                        let mut stage = stage.lock().unwrap();
                        let frame_count = stage.interp_frames();
                        let frames =
                            std::iter::repeat_n(rgb.as_slice(), frame_count).collect::<Vec<_>>();
                        stage.process_interp(pre.w(), pre.h(), &frames, 0.5)?
                    }
                    Stage::Flow { .. } => (pre.w(), pre.h(), rgb),
                    Stage::Glsl { .. } => unreachable!(),
                };
                anyhow::ensure!(
                    iw > 0 && ih > 0 && interpolated.len() == (iw as usize) * (ih as usize) * 3,
                    "interpolation warmup returned an invalid image"
                );
                let interpolated = gc.upload_rgb8(iw, ih, &interpolated);
                self.process_range(
                    gc,
                    interpolated,
                    out_size,
                    interp_index + 1,
                    self.stages.len(),
                    None,
                )?
            } else {
                self.process(gc, input, out_size, None)?
            };
            let output = gc.download_rgba8(final_tex);
            anyhow::ensure!(
                final_tex.w() > 0
                    && final_tex.h() > 0
                    && output.len() == (final_tex.w() as usize) * (final_tex.h() as usize) * 4,
                "candidate chain warmup returned an invalid image"
            );
            for stage in &mut self.stages {
                if let Stage::Onnx { stage, .. } = stage {
                    stage.lock().unwrap().finalize_provider_after_warmup();
                }
            }
            Ok(self.onnx_backend_usage())
        })();
        gc.release_frame(preserve);
        result
    }
}

impl FilterChain {
    pub fn from_specs(factory: &mut StageFactory, specs: &[StageSpec]) -> (Self, Vec<String>) {
        let mut stages = Vec::new();
        let mut errors = Vec::new();
        let mut interp_claimed = false;
        for spec in specs.iter().filter(|s| s.enabled) {
            if spec.kind == StageKind::Flow {
                if interp_claimed {
                    errors.push(skip_extra_interp_message(&spec.path));
                    continue;
                }
            } else if interp_claimed
                && spec.kind == StageKind::Onnx
                && looks_like_interp_path(&spec.path)
            {
                errors.push(skip_extra_interp_message(&spec.path));
                continue;
            }
            match factory.build(spec) {
                Ok(st) => {
                    if st.is_interp() {
                        if interp_claimed {
                            errors.push(skip_extra_interp_message(st.name()));
                            continue;
                        }
                        interp_claimed = true;
                    }
                    stages.push(st);
                }
                Err(e) => errors.push(format!("{}: {e:#}", spec.path)),
            }
        }
        factory.prune_onnx_to(&stages);
        // Fail safe on chain creation. A static progressive source can be
        // reprocessed immediately after a GUI toggle, before the engine has
        // received another capture frame on which to run the comb detector.
        // Starting in bypass prevents one incorrect bobbed/jagged frame.
        let neodeint_bypass = stages
            .iter()
            .any(|stage| matches!(stage, Stage::Glsl { name, .. } if name == "NeoDeint.glsl"));
        let vulkan_resident_batches = build_vulkan_resident_batches(&stages);
        (
            Self {
                stages,
                neodeint_bypass,
                dml_temporal_limit_log_keys: HashSet::new(),
                vulkan_resident_batches,
                dml_vulkan_resident_log_keys: HashSet::new(),
                dml_vulkan_image_handoff_disabled: false,
            },
            errors,
        )
    }

    /// Run the chain. `probe` (name, kind, ms) is called per stage only when
    /// stats are enabled — pass None for zero overhead.
    pub fn process(
        &mut self,
        gc: &mut GlContext,
        input: GpuTex,
        out_size: (i32, i32),
        probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Result<GpuTex> {
        self.process_from(gc, input, out_size, 0, probe)
    }

    /// v652 ordinary DirectML -> Vulkan cross-GPU resident handoff.
    ///
    /// v664 uses this whenever production GLSL is routed to Vulkan and the
    /// leading DirectML stage shares that selected LUID.  This covers both the
    /// original cross-GPU case and the same-GPU `[Vulkan]` selector, avoiding an
    /// unnecessary DirectML -> OpenGL -> CPU -> Vulkan round trip.
    /// Keep the ONNX result on that GPU, import it directly into the complete
    /// following GLSL range (one or more compatible shaders), and return only
    /// the final Vulkan result to presentation GL. Failure disables this
    /// optimization for the current chain and leaves the established path
    /// available immediately.
    pub fn process_first_onnx_vulkan_resident_rgba8(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
        out_size: (i32, i32),
        probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Result<Option<(GpuTex, String, f64)>> {
        if self.dml_vulkan_image_handoff_disabled || w <= 0 || h <= 0 {
            return Ok(None);
        }
        let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() else {
            return Ok(None);
        };
        if !vulkan_multipass::production_requested() {
            return Ok(None);
        }
        // v664: `production_selected_luid()` is populated only when GLSL is
        // actually routed to Vulkan.  That includes the explicit `[Vulkan]`
        // selector on the same physical GPU as presentation OpenGL.  v652 kept
        // this handoff cross-GPU-only, which forced same-GPU `[Vulkan]` through
        // DirectML -> OpenGL -> CPU readback -> Vulkan even though DirectML and
        // Vulkan already share the selected LUID.  Allow the same D3D12 ->
        // Vulkan resident handoff here as well.  Ordinary same-GPU explicit
        // selection without `[Vulkan]` still has no production Vulkan LUID and
        // therefore never reaches this path.
        let range_end = self.stages.len();
        if range_end <= 1 || !self.can_process_range_from_dml_shared(1, range_end) {
            return Ok(None);
        }

        let prepared = {
            let Some(Stage::Onnx {
                name,
                stage,
                is_interp: false,
                ..
            }) = self.stages.first_mut()
            else {
                return Ok(None);
            };
            let mut stage = stage.lock().unwrap_or_else(|poisoned| {
                log::error!(
                    "onnx-stage-lock-poisoned: action=recover-for-dml-vulkan-image-handoff"
                );
                poisoned.into_inner()
            });
            if stage.provider != OnnxProvider::DirectML {
                return Ok(None);
            }
            let metrics_name = format!("{name} [DirectML]");
            stage
                .process_rgba8_dml_shared_output(w, h, rgba)
                .map(|output| output.map(|(shared, ms)| (shared, metrics_name, ms)))
        };

        let (shared, metrics_name, onnx_ms) = match prepared {
            Ok(Some(output)) => output,
            Ok(None) => return Ok(None),
            Err(error) => {
                self.dml_vulkan_image_handoff_disabled = true;
                let line = format!(
                    "dml-vulkan-image-handoff: result=fallback requested_luid={luid:016x} reason={error:#} fallback=legacy-chain disabled_for_chain=true"
                );
                log::warn!("{line}");
                crate::render::vulkan_gpu::record_probe_result(&line);
                return Ok(None);
            }
        };

        match self.process_range_from_dml_shared(gc, &shared, out_size, 1, range_end, probe) {
            Ok(Some(output)) => Ok(Some((output, metrics_name, onnx_ms))),
            Ok(None) => {
                self.dml_vulkan_image_handoff_disabled = true;
                let line = format!(
                    "dml-vulkan-image-handoff: result=fallback requested_luid={luid:016x} reason=vulkan-shared-range-unavailable fallback=legacy-chain disabled_for_chain=true"
                );
                log::warn!("{line}");
                crate::render::vulkan_gpu::record_probe_result(&line);
                Ok(None)
            }
            Err(error) => {
                self.dml_vulkan_image_handoff_disabled = true;
                let line = format!(
                    "dml-vulkan-image-handoff: result=fallback requested_luid={luid:016x} reason={error:#} fallback=legacy-chain disabled_for_chain=true"
                );
                log::warn!("{line}");
                crate::render::vulkan_gpu::record_probe_result(&line);
                Ok(None)
            }
        }
    }

    /// Fast capture path for a non-interpolation ONNX stage at chain index 0.
    /// Returns the first stage's GPU output and measured time. Call
    /// `process_from(..., 1, ...)` for the remaining stages.
    pub fn process_first_onnx_rgba8(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(GpuTex, String, f64)>> {
        let Some(Stage::Onnx {
            name,
            stage,
            is_interp: false,
            ..
        }) = self.stages.first_mut()
        else {
            return Ok(None);
        };
        let t0 = std::time::Instant::now();
        let mut stage = stage.lock().unwrap();
        let metrics_name = match stage.provider {
            OnnxProvider::TensorRT => format!("{name} [TensorRT]"),
            OnnxProvider::DirectML => format!("{name} [DirectML]"),
            OnnxProvider::Cuda => format!("{name} [CUDA]"),
        };
        // Let the ordinary GPU chain own over-limit DirectML temporal input so
        // its Spline36 1080p safety cap is applied before this first stage.
        // Returning None only bypasses this specialized raw-RGBA fast path.
        if stage.is_directml_temporal_filter() && directml_temporal_limited_size(w, h).is_some() {
            return Ok(None);
        }
        stage.prepare_tensorrt_input_shape(w, h);
        super::onnx_stage::mark_tensorrt_model_started(&stage.name);
        let tex = if stage.should_try_tensorrt_gpu_texture(w, h) {
            let source = gc.upload_rgba8(w, h, rgba);
            if let Some(texture) = stage.process_gpu_texture(gc, source)? {
                texture
            } else {
                let (ow, oh, out) = stage.process_rgba8_native_output(w, h, rgba)?;
                gc.upload_rgba8(ow, oh, &out)
            }
        } else if let Some((_ow, _oh, texture)) =
            stage.try_process_rgba8_neo_texture(gc, w, h, rgba)?
        {
            texture
        } else if let Some((_ow, _oh, texture)) = stage.process_rgba8_gpu_output(gc, w, h, rgba) {
            texture
        } else {
            let (ow, oh, out) = stage.process_rgba8_native_output(w, h, rgba)?;
            gc.upload_rgba8(ow, oh, &out)
        };
        if stage.complete_tensorrt_input_shape(w, h) {
            super::onnx_stage::mark_tensorrt_model_completed(&stage.name);
        }
        Ok(Some((
            tex,
            metrics_name,
            t0.elapsed().as_secs_f64() * 1000.0,
        )))
    }

    /// v651 raw-capture fast path for the leading v648 resident Vulkan batch.
    ///
    /// WGC already owns an SDR frame as CPU RGBA8.  Previously the generic
    /// pre-GL Vulkan route consumed stage 0 by itself and uploaded that result
    /// to OpenGL, which meant `process_from(..., 1, ...)` could no longer match
    /// the v648 batch `[0..N)`.  A chain such as Anime4K Restore + Upscale
    /// therefore crossed CPU/OpenGL between the two shaders even though a
    /// merged resident graph had been prepared.
    ///
    /// Execute the complete leading batch directly from the WGC byte buffer,
    /// keep every intermediate pass on the explicitly selected Vulkan GPU, and
    /// upload only the final batch result to the presentation OpenGL context.
    /// Returning `None` preserves the established per-stage pre-GL fallback.
    pub fn process_leading_vulkan_resident_rgba8(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
        out_size: (i32, i32),
        mut probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Option<(GpuTex, usize)> {
        let luid = crate::render::vulkan_gpu::production_selected_luid()?;
        if !vulkan_multipass::production_requested() || w <= 0 || h <= 0 {
            return None;
        }
        // v655: an explicit cross-GPU selection means compatible GLSL must
        // actually execute on the selected Vulkan adapter even when there is
        // only one leading shader. v648 intentionally prebuilt only 2+ stage
        // resident batches, leaving a single leading GLSL dependent on a
        // separate per-stage route. Build the maximal compatible leading span
        // here with a minimum of one stage so AMD-selected sessions cannot
        // silently fall back to the presentation GPU merely because the chain
        // contains one shader.
        let leading_end = self
            .stages
            .iter()
            .take_while(|stage| {
                matches!(stage, Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader))
            })
            .count();
        let batch = self
            .vulkan_resident_batches
            .iter()
            .find(|batch| batch.start == 0 && batch.end == leading_end)
            .cloned()
            .or_else(|| build_vulkan_resident_batch(&self.stages, 0, leading_end, 1))?;

        let expected = (w as usize)
            .checked_mul(h as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if expected != Some(rgba.len()) {
            let line = format!(
                "vulkan-resident-chain: result=fallback route=wgc-pre-gl requested_luid={luid:016x} stages=[{}] reason=input-rgba-size-mismatch expected={:?} actual={} fallback=per-stage",
                batch.stage_names.join(", "),
                expected,
                rgba.len(),
            );
            vulkan_onepass::record_route_once(
                format!(
                    "resident-chain-wgc-input:{luid:016x}:{}:{}",
                    batch.start, batch.end
                ),
                &line,
            );
            return None;
        }

        let started = std::time::Instant::now();
        let result = match vulkan_multipass::process_rgba8(
            luid,
            &batch.shader,
            w as u32,
            h as u32,
            out_size.0.max(1) as u32,
            out_size.1.max(1) as u32,
            rgba,
        ) {
            Ok(Some(result)) => result,
            Ok(None) => return None,
            Err(error) => {
                let line = format!(
                    "vulkan-resident-chain: result=fallback route=wgc-pre-gl requested_luid={luid:016x} stages=[{}] reason={:#} fallback=per-stage",
                    batch.stage_names.join(", "),
                    error,
                );
                vulkan_onepass::record_route_once(
                    format!(
                        "resident-chain-wgc-runtime:{luid:016x}:{}:{}",
                        batch.start, batch.end
                    ),
                    &line,
                );
                return None;
            }
        };

        let output_w = result.output_width as i32;
        let output_h = result.output_height as i32;
        let output = gc.upload_rgba8(output_w, output_h, &result.output_rgba8);
        let chain_ms = started.elapsed().as_secs_f64() * 1000.0;

        if let Some(p) = probe.as_deref_mut() {
            let metric_names = self
                .stages
                .iter()
                .map(Stage::metrics_name)
                .collect::<Vec<_>>();
            let metric_labels = disambiguate_stage_metric_labels(&metric_names);
            let total_passes = batch.pass_counts.iter().sum::<usize>().max(1);
            for (offset, pass_count) in batch.pass_counts.iter().enumerate() {
                let index = batch.start + offset;
                p(
                    &metric_labels[index],
                    StageKind::Glsl,
                    chain_ms * (*pass_count as f64 / total_passes as f64),
                );
            }
        }

        if result.first_frame_active {
            let line = format!(
                "vulkan-resident-chain: result=active route=wgc-pre-gl requested_luid={:016x} gpu='{}' stages=[{}] stage_count={} passes={} input={}x{} output={}x{} transfer=wgc-cpu-to-vulkan-resident-chain-to-cpu-to-gl gl_readback=false inter_stage_cpu_copies=0 intermediate_gpu_resident=true vulkan_ms={:.3} chain_ms={:.3} metrics=pass-weighted-estimate fallback=per-stage-on-error",
                luid,
                result.gpu_name,
                batch.stage_names.join(", "),
                batch.stage_names.len(),
                result.active_passes,
                w,
                h,
                output_w,
                output_h,
                result.elapsed_ms,
                chain_ms,
            );
            log::info!("{line}");
            crate::render::vulkan_gpu::record_probe_result(&line);
        }

        Some((output, batch.end))
    }

    pub fn process_from(
        &mut self,
        gc: &mut GlContext,
        input: GpuTex,
        out_size: (i32, i32),
        start_index: usize,
        probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Result<GpuTex> {
        let end_index = self.stages.len();
        self.process_range(gc, input, out_size, start_index, end_index, probe)
    }

    pub(crate) fn can_process_range_from_dml_shared(
        &self,
        start_index: usize,
        end_index: usize,
    ) -> bool {
        let range_end = end_index.min(self.stages.len());
        if crate::render::vulkan_gpu::production_selected_luid().is_none()
            || !vulkan_multipass::production_requested()
        {
            return false;
        }

        // An empty post range is still useful for RIFE-only chains: Vulkan
        // performs the D3D12 NCHW -> RGBA handoff on the selected GPU.
        if start_index == range_end {
            return true;
        }
        if start_index > range_end {
            return false;
        }

        // v665: do not use the older resident-batch admission as the gate for
        // DirectML input.  A shader can be fully valid as a standalone Vulkan
        // plan yet intentionally fail raw-pass concatenation (FSRCNNX is the
        // important LUMA example).  Exact consecutive GLSL ranges are eligible
        // whenever every stage is production-admitted; one stage uses its
        // ordinary plan and two or more use the boundary-preserving sequence
        // planner.
        self.stages[start_index..range_end]
            .iter()
            .all(|stage| match stage {
                Stage::Glsl { shader, .. } => {
                    !shader.is_post
                        && shader.name() != "NeoDeint.glsl"
                        && vulkan_multipass::production_shader_admitted(shader)
                }
                _ => false,
            })
    }

    /// DirectML -> Vulkan shared-output handoff. v649 introduced this for
    /// interpolation outputs; v652 also uses it for an ordinary leading image
    /// model. DirectML and Vulkan must resolve to the same selected-GPU LUID.
    /// One or more isolated GLSL stages may follow; incompatible ranges return
    /// None and retain the proven CPU/OpenGL fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn process_range_from_dml_shared(
        &mut self,
        gc: &mut GlContext,
        shared: &PreparedDmlSharedOutput,
        out_size: (i32, i32),
        start_index: usize,
        end_index: usize,
        mut probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Result<Option<GpuTex>> {
        let range_end = end_index.min(self.stages.len());
        let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() else {
            return Ok(None);
        };
        if u64::from_le_bytes(shared.luid) != luid {
            return Ok(None);
        }
        if !vulkan_multipass::production_requested() {
            return Ok(None);
        }

        let passthrough_only = start_index == range_end;
        if start_index > range_end {
            return Ok(None);
        }

        let metric_names = self
            .stages
            .iter()
            .map(Stage::metrics_name)
            .collect::<Vec<_>>();
        let metric_labels = disambiguate_stage_metric_labels(&metric_names);

        // v665: DirectML shared input no longer depends on the conservative
        // raw-pass resident-batch builder.  Preserve every shader's standalone
        // boundary semantics, especially LUMA -> RGB reconstruction in
        // FSRCNNX, while keeping the whole post range in one Vulkan runtime.
        let sequence_shaders = if passthrough_only {
            Vec::new()
        } else {
            let shaders = self.stages[start_index..range_end]
                .iter()
                .filter_map(|stage| match stage {
                    Stage::Glsl { shader, .. }
                        if !shader.is_post
                            && shader.name() != "NeoDeint.glsl"
                            && vulkan_multipass::production_shader_admitted(shader) =>
                    {
                        Some(Rc::clone(shader))
                    }
                    _ => None,
                })
                .collect::<Vec<_>>();
            if shaders.len() != range_end.saturating_sub(start_index) {
                return Ok(None);
            }
            shaders
        };

        let stage_names = sequence_shaders
            .iter()
            .map(|shader| shader.name())
            .collect::<Vec<_>>();
        let pass_counts = sequence_shaders
            .iter()
            .map(|shader| shader.passes.len().max(1))
            .collect::<Vec<_>>();

        let stage_started = std::time::Instant::now();
        let output_ref_w = if passthrough_only {
            shared.size.0.max(1) as u32
        } else {
            out_size.0.max(1) as u32
        };
        let output_ref_h = if passthrough_only {
            shared.size.1.max(1) as u32
        } else {
            out_size.1.max(1) as u32
        };

        let result = if passthrough_only {
            let identity = dml_vulkan_identity_shader();
            vulkan_multipass::process_d3d12_nchw_to_gl(
                luid,
                identity.as_ref(),
                shared.size.0.max(1) as u32,
                shared.size.1.max(1) as u32,
                output_ref_w,
                output_ref_h,
                shared.key,
                shared.handle.0 as isize,
                shared.byte_len,
                shared.padded.0.max(1) as u32,
                shared.padded.1.max(1) as u32,
                shared.fp16,
                gc,
            )?
        } else if sequence_shaders.len() == 1 {
            vulkan_multipass::process_d3d12_nchw_to_gl(
                luid,
                sequence_shaders[0].as_ref(),
                shared.size.0.max(1) as u32,
                shared.size.1.max(1) as u32,
                output_ref_w,
                output_ref_h,
                shared.key,
                shared.handle.0 as isize,
                shared.byte_len,
                shared.padded.0.max(1) as u32,
                shared.padded.1.max(1) as u32,
                shared.fp16,
                gc,
            )?
        } else {
            let shader_refs = sequence_shaders
                .iter()
                .map(|shader| shader.as_ref())
                .collect::<Vec<_>>();
            vulkan_multipass::process_d3d12_nchw_sequence_to_gl(
                luid,
                &shader_refs,
                shared.size.0.max(1) as u32,
                shared.size.1.max(1) as u32,
                output_ref_w,
                output_ref_h,
                shared.key,
                shared.handle.0 as isize,
                shared.byte_len,
                shared.padded.0.max(1) as u32,
                shared.padded.1.max(1) as u32,
                shared.fp16,
                gc,
            )?
        };
        let Some(result) = result else {
            return Ok(None);
        };

        let output_w = result.output_width as i32;
        let output_h = result.output_height as i32;
        let output = result.output;
        let elapsed_ms = stage_started.elapsed().as_secs_f64() * 1000.0;

        if !passthrough_only {
            if let Some(p) = probe.as_deref_mut() {
                let total_passes = pass_counts.iter().sum::<usize>().max(1);
                for (offset, pass_count) in pass_counts.iter().enumerate() {
                    let index = start_index + offset;
                    p(
                        &metric_labels[index],
                        StageKind::Glsl,
                        elapsed_ms * (*pass_count as f64 / total_passes as f64),
                    );
                }
            }
        }

        if result.first_frame_active || self.dml_vulkan_resident_log_keys.insert(shared.key) {
            let route = if passthrough_only {
                "identity"
            } else if sequence_shaders.len() == 1 {
                "single-plan"
            } else {
                "sequence-plan"
            };
            let output_transfer = if result.output_external_buffer {
                "d3d12-directml-to-vulkan-device-local-input-to-d3d12-external-rgba8-to-gl"
            } else {
                "d3d12-directml-to-vulkan-device-local-input-to-mapped-readback-to-gl"
            };
            let line = format!(
                "dml-vulkan-resident-chain: result=active requested_luid={:016x} gpu='{}' shared_key={} allocation_bytes={} stages=[{}] stage_count={} passes={} input={}x{} padded={}x{} output={}x{} fp16={} plan={} transfer={} dml_to_vulkan_cpu_readback=0 dml_to_vulkan_cpu_upload=0 gl_input_readback=0 vulkan_to_gl_cpu_readback={} vulkan_to_gl_cpu_upload={} nchw_input_device_local=true intermediate_gpu_resident=true output_vec_copy=false output_external_buffer={} workgroup=16x8 external_sync_ms={:.3} vulkan_ms={:.3} gl_bridge_ms={:.3} total_ms={:.3} fallback=legacy-on-error",
                luid,
                result.gpu_name,
                shared.key,
                shared.heap_byte_len,
                if passthrough_only {
                    "<RIFE-output-handoff>".to_string()
                } else {
                    stage_names.join(", ")
                },
                stage_names.len(),
                result.active_passes,
                shared.size.0,
                shared.size.1,
                shared.padded.0,
                shared.padded.1,
                output_w,
                output_h,
                shared.fp16,
                route,
                output_transfer,
                if result.output_external_buffer { 0 } else { 1 },
                if result.output_external_buffer { 0 } else { 1 },
                result.output_external_buffer,
                result.external_sync_ms,
                result.vulkan_ms,
                result.gl_upload_ms,
                elapsed_ms,
            );
            log::info!("{line}");
            crate::render::vulkan_gpu::record_probe_result(&line);
        }
        Ok(Some(output))
    }

    /// Execute an exact, half-open section of the GUI chain. This is used to
    /// preserve the visible Pre -> Interpolation -> Post ordering: captured
    /// endpoints run the pre range once, while each interpolation output runs
    /// only the post range.
    pub fn process_range(
        &mut self,
        gc: &mut GlContext,
        input: GpuTex,
        out_size: (i32, i32),
        start_index: usize,
        end_index: usize,
        mut probe: Option<&mut dyn FnMut(&str, StageKind, f64)>,
    ) -> Result<GpuTex> {
        let mut cur = input;
        // Use stable per-chain labels. Metrics previously used only the file
        // name as the key, so identical stages overwrote one another even
        // though every stage was actually executed.
        let metric_names = self
            .stages
            .iter()
            .map(Stage::metrics_name)
            .collect::<Vec<_>>();
        let metric_labels = disambiguate_stage_metric_labels(&metric_names);
        // Timer-query results normally become available a few frames after
        // submission. Polling availability is non-blocking, so statistics do
        // not insert glFinish or otherwise disturb the measured frame rate.
        if let Some(p) = probe.as_deref_mut() {
            for (label, gpu_ms) in gc.poll_gpu_timers() {
                p(&label, StageKind::Glsl, gpu_ms);
            }
        }
        // mpv `frame` builtin: one tick per processed frame (interpolated
        // in-betweens each count as a frame, like mpv's own interpolation)
        crate::render::glsl_engine::advance_frame();
        let range_end = end_index.min(self.stages.len());

        // v662: a second consecutive GLSL stage used to fall out of the
        // conservative v648 resident-batch gate whenever the first shader was
        // LUMA-based (FSRCNNX is the common case).  Each shader then executed
        // through its own GL->CPU->Vulkan->CPU->GL bridge.  At 2560x1440 and
        // especially 5120x2880 that staging cost dwarfs the actual Vulkan GPU
        // work.  Build each shader's normal Vulkan plan independently and
        // concatenate the *plans* instead of their raw passes, preserving
        // LUMA->RGB stage boundaries while keeping all inter-stage images on
        // Vulkan.  On any incompatibility, fall through to the proven v661
        // per-stage path unchanged.
        if let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() {
            if vulkan_multipass::production_requested() && start_index < range_end {
                let sequence_end = self
                    .stages
                    .iter()
                    .enumerate()
                    .take(range_end)
                    .skip(start_index)
                    .take_while(|(_, stage)| {
                        matches!(
                            stage,
                            Stage::Glsl { shader, .. }
                                if !shader.is_post
                                    && shader.name() != "NeoDeint.glsl"
                                    && vulkan_multipass::production_shader_admitted(shader)
                        )
                    })
                    .map(|(index, _)| index + 1)
                    .last()
                    .unwrap_or(start_index);
                if sequence_end.saturating_sub(start_index) >= 2 {
                    if cur.has_offset() {
                        cur = crate::render::scaler::align_offset(gc, cur).context(
                            "failed to align pending mpv shader OFFSET before Vulkan sequence chain",
                        )?;
                    }
                    let unsupported_reason = if cur.comps() != 4 {
                        Some(format!("input-components-{}", cur.comps()))
                    } else if cur.has_offset() {
                        Some("pending-mpv-offset".to_string())
                    } else if !matches!(cur.key.dtype, Dtype::U8 | Dtype::F16) {
                        Some(format!("input-dtype-{:?}", cur.key.dtype))
                    } else {
                        None
                    };
                    if unsupported_reason.is_none() {
                        let sequence_shaders = self.stages[start_index..sequence_end]
                            .iter()
                            .filter_map(|stage| match stage {
                                Stage::Glsl { shader, .. } => Some(Rc::clone(shader)),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        if sequence_shaders.len() == sequence_end - start_index {
                            let shader_refs = sequence_shaders
                                .iter()
                                .map(|shader| shader.as_ref())
                                .collect::<Vec<_>>();
                            let stage_started = std::time::Instant::now();
                            let input_dtype = cur.key.dtype;
                            let input_w = cur.w();
                            let input_h = cur.h();
                            let rgba = gc.download_rgba8(cur);
                            match vulkan_multipass::process_rgba8_sequence_to_gl(
                                luid,
                                &shader_refs,
                                input_w as u32,
                                input_h as u32,
                                out_size.0.max(1) as u32,
                                out_size.1.max(1) as u32,
                                &rgba,
                                gc,
                            ) {
                                Ok(Some(result)) => {
                                    let output_w = result.output_width as i32;
                                    let output_h = result.output_height as i32;
                                    let output = if input_dtype == Dtype::U8 {
                                        result.output
                                    } else {
                                        // Preserve the established bridge's
                                        // dtype contract for uncommon F16
                                        // callers; production WGC/DML image
                                        // chains are U8 here.
                                        let rgba = gc.download_rgba8(result.output);
                                        gc.recycle(result.output);
                                        upload_vulkan_bridge_rgba8(
                                            gc,
                                            input_dtype,
                                            output_w,
                                            output_h,
                                            &rgba,
                                        )
                                    };
                                    let stage_elapsed_ms =
                                        stage_started.elapsed().as_secs_f64() * 1000.0;
                                    if let Some(p) = probe.as_deref_mut() {
                                        let pass_counts = sequence_shaders
                                            .iter()
                                            .map(|shader| shader.passes.len().max(1))
                                            .collect::<Vec<_>>();
                                        let total_passes = pass_counts.iter().sum::<usize>().max(1);
                                        for (offset, pass_count) in pass_counts.iter().enumerate() {
                                            let index = start_index + offset;
                                            let estimated_ms = stage_elapsed_ms
                                                * (*pass_count as f64 / total_passes as f64);
                                            p(&metric_labels[index], StageKind::Glsl, estimated_ms);
                                        }
                                    }
                                    if result.first_frame_active {
                                        let names = sequence_shaders
                                            .iter()
                                            .map(|shader| shader.name())
                                            .collect::<Vec<_>>();
                                        let output_transfer = if result.output_external_buffer {
                                            "gl-to-cpu-to-vulkan-sequence-to-d3d12-external-rgba8-to-gl"
                                        } else {
                                            "gl-to-cpu-to-vulkan-sequence-mapped-readback-to-gl"
                                        };
                                        let line = format!(
                                            "vulkan-sequence-chain: result=active requested_luid={:016x} gpu='{}' stages=[{}] stage_count={} passes={} input={}x{} output={}x{} input_dtype={:?} transfer={} input_cpu_readback=1 output_cpu_readback={} output_cpu_upload={} inter_stage_cpu_copies=0 intermediate_gpu_resident=true output_vec_copy=false output_external_buffer={} workgroup=16x8 external_sync_ms={:.3} vulkan_ms={:.3} gl_bridge_ms={:.3} chain_ms={:.3} fallback=per-stage-on-error",
                                            luid,
                                            result.gpu_name,
                                            names.join(", "),
                                            names.len(),
                                            result.active_passes,
                                            input_w,
                                            input_h,
                                            output_w,
                                            output_h,
                                            input_dtype,
                                            output_transfer,
                                            if result.output_external_buffer { 0 } else { 1 },
                                            if result.output_external_buffer { 0 } else { 1 },
                                            result.output_external_buffer,
                                            result.external_sync_ms,
                                            result.vulkan_ms,
                                            result.gl_upload_ms,
                                            stage_elapsed_ms,
                                        );
                                        log::info!("{line}");
                                        crate::render::vulkan_gpu::record_probe_result(&line);
                                    }
                                    if sequence_end < range_end {
                                        return self.process_range(
                                            gc,
                                            output,
                                            out_size,
                                            sequence_end,
                                            range_end,
                                            probe,
                                        );
                                    }
                                    return Ok(output);
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    let names = sequence_shaders
                                        .iter()
                                        .map(|shader| shader.name())
                                        .collect::<Vec<_>>();
                                    let line = format!(
                                        "vulkan-sequence-chain: result=fallback requested_luid={luid:016x} stages=[{}] reason={:#} fallback=per-stage",
                                        names.join(", "),
                                        error,
                                    );
                                    vulkan_onepass::record_route_once(
                                        format!(
                                            "sequence-chain-runtime:{luid:016x}:{}:{}",
                                            start_index, sequence_end
                                        ),
                                        &line,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }

        // v648: keep a consecutive ordinary GLSL section resident on the
        // explicitly selected Vulkan GPU. v647 crossed the bridge for every
        // shader independently (GL -> CPU -> Vulkan -> CPU -> GL), which made
        // staging dominate the actual Vulkan compute time. A prebuilt merged
        // shader graph lets the complete section use one Vulkan runtime, one
        // upload, and one final readback/upload while every intermediate pass
        // remains a Vulkan image. If the conservative merge gate or runtime
        // fails, fall through to the proven per-stage path unchanged.
        if let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() {
            if vulkan_multipass::production_requested() {
                // v655 selected-GPU resident-prefix scheduler. The old gate
                // required the resident batch to consume the *entire* requested
                // range. In mixed chains that meant a valid GLSL prefix followed
                // by ONNX/temporal work could miss the resident path and later
                // execute piecemeal or fall back to presentation OpenGL. Consume
                // the maximal compatible GLSL prefix (1+ stages) on the selected
                // Vulkan GPU, then continue the remaining range from batch.end.
                let prefix_end = self
                    .stages
                    .iter()
                    .enumerate()
                    .take(range_end)
                    .skip(start_index)
                    .take_while(|(_, stage)| {
                        matches!(stage, Stage::Glsl { shader, .. } if vulkan_batch_shader_isolated(shader))
                    })
                    .map(|(index, _)| index + 1)
                    .last()
                    .unwrap_or(start_index);
                let resident_batch = self
                    .vulkan_resident_batches
                    .iter()
                    .filter(|batch| {
                        batch.start == start_index
                            && batch.end <= range_end
                            && batch.end == prefix_end
                    })
                    .max_by_key(|batch| batch.end)
                    .cloned()
                    .or_else(|| {
                        build_vulkan_resident_batch(&self.stages, start_index, prefix_end, 1)
                    });
                if let Some(batch) = resident_batch {
                    if cur.has_offset() {
                        cur = crate::render::scaler::align_offset(gc, cur).context(
                            "failed to align pending mpv shader OFFSET before resident Vulkan GLSL chain",
                        )?;
                    }
                    let unsupported_reason = if !matches!(cur.comps(), 3 | 4) {
                        Some(format!("input-components-{}", cur.comps()))
                    } else if cur.has_offset() {
                        Some("pending-mpv-offset".to_string())
                    } else if !matches!(cur.key.dtype, Dtype::U8 | Dtype::F16) {
                        Some(format!("input-dtype-{:?}", cur.key.dtype))
                    } else {
                        None
                    };
                    if let Some(reason) = unsupported_reason {
                        let line = format!(
                            "vulkan-resident-chain: result=fallback requested_luid={luid:016x} stages=[{}] reason={} fallback=per-stage",
                            batch.stage_names.join(", "),
                            reason,
                        );
                        vulkan_onepass::record_route_once(
                            format!(
                                "resident-chain-gate:{luid:016x}:{}:{}:{}",
                                batch.start, batch.end, reason
                            ),
                            &line,
                        );
                    } else {
                        let stage_started = std::time::Instant::now();
                        let input_dtype = cur.key.dtype;
                        let input_w = cur.w();
                        let input_h = cur.h();
                        let rgba = gc.download_rgba8(cur);
                        match vulkan_multipass::process_rgba8(
                            luid,
                            &batch.shader,
                            input_w as u32,
                            input_h as u32,
                            out_size.0.max(1) as u32,
                            out_size.1.max(1) as u32,
                            &rgba,
                        ) {
                            Ok(Some(result)) => {
                                let output_w = result.output_width as i32;
                                let output_h = result.output_height as i32;
                                let output = upload_vulkan_bridge_rgba8(
                                    gc,
                                    input_dtype,
                                    output_w,
                                    output_h,
                                    &result.output_rgba8,
                                );
                                let stage_elapsed_ms =
                                    stage_started.elapsed().as_secs_f64() * 1000.0;
                                if let Some(p) = probe.as_deref_mut() {
                                    let total_passes =
                                        batch.pass_counts.iter().sum::<usize>().max(1);
                                    for (offset, pass_count) in batch.pass_counts.iter().enumerate()
                                    {
                                        let index = batch.start + offset;
                                        let estimated_ms = stage_elapsed_ms
                                            * (*pass_count as f64 / total_passes as f64);
                                        p(&metric_labels[index], StageKind::Glsl, estimated_ms);
                                    }
                                }
                                if result.first_frame_active {
                                    let line = format!(
                                        "vulkan-resident-chain: result=active requested_luid={:016x} gpu='{}' stages=[{}] stage_count={} passes={} input={}x{} output={}x{} input_dtype={:?} transfer=gl-to-cpu-to-vulkan-resident-chain-to-cpu-to-gl cpu_bridge_roundtrips=1 intermediate_gpu_resident=true vulkan_ms={:.3} chain_ms={:.3} metrics=pass-weighted-estimate fallback=per-stage-on-error",
                                        luid,
                                        result.gpu_name,
                                        batch.stage_names.join(", "),
                                        batch.stage_names.len(),
                                        result.active_passes,
                                        input_w,
                                        input_h,
                                        output_w,
                                        output_h,
                                        input_dtype,
                                        result.elapsed_ms,
                                        stage_elapsed_ms,
                                    );
                                    log::info!("{line}");
                                    crate::render::vulkan_gpu::record_probe_result(&line);
                                }
                                if batch.end < range_end {
                                    return self.process_range(
                                        gc, output, out_size, batch.end, range_end, probe,
                                    );
                                }
                                return Ok(output);
                            }
                            Ok(None) => {}
                            Err(error) => {
                                let line = format!(
                                    "vulkan-resident-chain: result=fallback requested_luid={:016x} stages=[{}] reason={:#} fallback=per-stage",
                                    luid,
                                    batch.stage_names.join(", "),
                                    error,
                                );
                                vulkan_onepass::record_route_once(
                                    format!(
                                        "resident-chain-runtime:{luid:016x}:{}:{}",
                                        batch.start, batch.end
                                    ),
                                    &line,
                                );
                            }
                        }
                    }
                }
            }
        }

        // When the engine enters the chain after its interpolation stage, the
        // input texture is already a GPU-resident interpolated frame.  Preserve
        // that residency for following DirectML image models instead of forcing
        // a GPU -> CPU -> DML -> CPU -> GPU round-trip.
        let after_interpolation = self
            .interp_index()
            .is_some_and(|interp_index| start_index > interp_index);
        let neodeint_bypass = self.neodeint_bypass;
        let dml_temporal_limit_log_keys = &mut self.dml_temporal_limit_log_keys;
        for (stage_index, stage) in self
            .stages
            .iter_mut()
            .enumerate()
            .take(range_end)
            .skip(start_index)
        {
            if neodeint_bypass
                && matches!(
                    stage,
                    Stage::Glsl { name, .. } if name == "NeoDeint.glsl"
                )
            {
                continue;
            }
            let handled_by_engine = stage.is_interp();
            let stage_kind = stage.kind();
            let t0 = (probe.is_some() && !handled_by_engine && stage_kind != StageKind::Glsl)
                .then(std::time::Instant::now);
            match stage {
                Stage::Glsl { shader, .. } if shader.is_post => {
                    // post-stage (OUTPUT/SCALED hook): applied after the
                    // final downscale, not here
                }
                Stage::Glsl { shader, .. } => {
                    // Consume mpv OFFSET before crossing the Vulkan bridge.
                    // This keeps a preceding GLSL's phase correction intact
                    // without forcing the following user filter back to OpenGL.
                    if cur.has_offset()
                        && crate::render::vulkan_gpu::production_selected_luid().is_some()
                    {
                        cur = crate::render::scaler::align_offset(gc, cur).context(
                            "failed to align pending mpv shader OFFSET before Vulkan GLSL",
                        )?;
                    }
                    // v636 selected-GPU chain routing. v635 could execute the
                    // first GLSL stage through Vulkan before the frame entered
                    // OpenGL, but GLSL after ONNX (and later GLSL stages) fell
                    // back to the render GPU. Route every ordinary GLSL stage
                    // through the selected Vulkan adapter when one is explicit.
                    // The bridge is deliberately CPU-staged for correctness;
                    // the Vulkan runtime cache keeps every active shader warm.
                    let mut vulkan_applied = false;
                    if vulkan_multipass::production_requested()
                        && vulkan_multipass::production_shader_admitted(shader)
                    {
                        if let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() {
                            let unsupported_reason = if cur.comps() != 4 {
                                Some(format!("input-components-{}", cur.comps()))
                            } else if cur.has_offset() {
                                Some("pending-mpv-offset".to_string())
                            } else if !matches!(cur.key.dtype, Dtype::U8 | Dtype::F16) {
                                Some(format!("input-dtype-{:?}", cur.key.dtype))
                            } else {
                                None
                            };
                            if let Some(reason) = unsupported_reason {
                                let line = format!(
                                    "vulkan-multipass-glsl: result=fallback route=mixed-chain shader='{}' requested_luid={luid:016x} reason={} input_dtype={:?} comps={} offset={} fallback=OpenGL",
                                    shader.name(),
                                    reason,
                                    cur.key.dtype,
                                    cur.comps(),
                                    cur.has_offset(),
                                );
                                vulkan_onepass::record_route_once(
                                    format!(
                                        "multipass-chain-gate:{luid:016x}:{}:{reason}",
                                        shader.name()
                                    ),
                                    &line,
                                );
                            } else {
                                let stage_started = std::time::Instant::now();
                                let input_dtype = cur.key.dtype;
                                let input_w = cur.w();
                                let input_h = cur.h();
                                let rgba = gc.download_rgba8(cur);
                                match vulkan_multipass::process_rgba8(
                                    luid,
                                    shader,
                                    input_w as u32,
                                    input_h as u32,
                                    out_size.0.max(1) as u32,
                                    out_size.1.max(1) as u32,
                                    &rgba,
                                ) {
                                    Ok(Some(result)) => {
                                        let output_w = result.output_width as i32;
                                        let output_h = result.output_height as i32;
                                        let output = upload_vulkan_bridge_rgba8(
                                            gc,
                                            input_dtype,
                                            output_w,
                                            output_h,
                                            &result.output_rgba8,
                                        );
                                        let stage_elapsed_ms =
                                            stage_started.elapsed().as_secs_f64() * 1000.0;
                                        if let Some(p) = probe.as_deref_mut() {
                                            p(
                                                &metric_labels[stage_index],
                                                StageKind::Glsl,
                                                stage_elapsed_ms,
                                            );
                                        }
                                        if result.first_frame_active {
                                            let line = format!(
                                                "vulkan-multipass-glsl: result=active route=mixed-chain shader='{}' requested_luid={:016x} gpu='{}' input={}x{} output={}x{} passes={} input_dtype={:?} transfer=gl-to-cpu-to-vulkan-to-cpu-to-gl display=Vulkan-result vulkan_ms={:.3} stage_ms={:.3} runtime_cache=multi fallback=OpenGL-on-error",
                                                shader.name(),
                                                luid,
                                                result.gpu_name,
                                                input_w,
                                                input_h,
                                                output_w,
                                                output_h,
                                                result.active_passes,
                                                input_dtype,
                                                result.elapsed_ms,
                                                stage_elapsed_ms,
                                            );
                                            log::info!("{line}");
                                            crate::render::vulkan_gpu::record_probe_result(&line);
                                        }
                                        cur = output;
                                        vulkan_applied = true;
                                    }
                                    Ok(None) => {}
                                    Err(error) => {
                                        let line = format!(
                                            "vulkan-multipass-glsl: result=fallback route=mixed-chain shader='{}' requested_luid={:016x} reason={:#} fallback=OpenGL",
                                            shader.name(),
                                            luid,
                                            error
                                        );
                                        log::warn!("{line}");
                                        crate::render::vulkan_gpu::record_probe_result(&line);
                                    }
                                }
                            }
                        }
                    }

                    // Retain the older one-pass compatibility route as a
                    // fallback for shaders the multi-pass path cannot execute.
                    if !vulkan_applied
                        && vulkan_onepass::production_one_pass_requested()
                        && vulkan_onepass::production_shader_admitted(shader)
                    {
                        if let Some(luid) = crate::render::vulkan_gpu::production_selected_luid() {
                            let route_key = format!(
                                "candidate:{luid:016x}:{}:{}x{}:{:?}:{}:{}",
                                shader.name(),
                                cur.w(),
                                cur.h(),
                                cur.key.dtype,
                                cur.comps(),
                                cur.has_offset(),
                            );
                            let candidate = format!(
                                "vulkan-production-glsl: phase=candidate shader='{}' requested_luid={luid:016x} size={}x{} input_dtype={:?} comps={} offset={} transfer=cpu-staging",
                                shader.name(),
                                cur.w(),
                                cur.h(),
                                cur.key.dtype,
                                cur.comps(),
                                cur.has_offset(),
                            );
                            vulkan_onepass::record_route_once(route_key, &candidate);

                            let unsupported_reason = if cur.comps() != 4 {
                                Some(format!("input-components-{}", cur.comps()))
                            } else if cur.has_offset() {
                                Some("pending-mpv-offset".to_string())
                            } else if !matches!(cur.key.dtype, Dtype::U8 | Dtype::F16) {
                                Some(format!("input-dtype-{:?}", cur.key.dtype))
                            } else {
                                None
                            };

                            if let Some(reason) = unsupported_reason {
                                let line = format!(
                                    "vulkan-production-glsl: result=fallback shader='{}' requested_luid={luid:016x} reason={} input_dtype={:?} comps={} offset={} fallback=OpenGL",
                                    shader.name(),
                                    reason,
                                    cur.key.dtype,
                                    cur.comps(),
                                    cur.has_offset(),
                                );
                                vulkan_onepass::record_route_once(
                                    format!("gate:{luid:016x}:{}:{reason}", shader.name()),
                                    &line,
                                );
                            } else {
                                let production_started = std::time::Instant::now();
                                let input_dtype = cur.key.dtype;
                                let rgba = gc.download_rgba8(cur);
                                match vulkan_onepass::process_rgba8(
                                    luid,
                                    shader,
                                    cur.w() as u32,
                                    cur.h() as u32,
                                    &rgba,
                                ) {
                                    Ok(Some(result)) => {
                                        // Preserve the OpenGL chain's storage type. GLSL
                                        // intermediates are normally F16; uploading the
                                        // Vulkan RGBA8 result back as F16 avoids changing
                                        // downstream type expectations during this bridge test.
                                        let output = upload_vulkan_bridge_rgba8(
                                            gc,
                                            input_dtype,
                                            cur.w(),
                                            cur.h(),
                                            &result.output_rgba8,
                                        );
                                        let stage_elapsed_ms =
                                            production_started.elapsed().as_secs_f64() * 1000.0;
                                        if let Some(p) = probe.as_deref_mut() {
                                            p(
                                                &metric_labels[stage_index],
                                                StageKind::Glsl,
                                                stage_elapsed_ms,
                                            );
                                        }
                                        if result.first_frame_verified {
                                            let line = format!(
                                                "vulkan-production-glsl: result=active shader='{}' requested_luid={:016x} gpu='{}' size={}x{} input_dtype={:?} output_dtype={:?} validation={} visual_check_required={} verified={}/{} tolerance={} transfer=cpu-staging persistent_runtime=true display=Vulkan-result vulkan_ms={:.3} stage_ms={:.3} fallback=OpenGL-on-error",
                                                shader.name(),
                                                luid,
                                                result.gpu_name,
                                                cur.w(),
                                                cur.h(),
                                                input_dtype,
                                                input_dtype,
                                                result.validation_mode,
                                                result.visual_check_required,
                                                result.verified_pixels,
                                                (cur.w() as usize) * (cur.h() as usize),
                                                result.tolerance,
                                                result.elapsed_ms,
                                                stage_elapsed_ms,
                                            );
                                            log::info!("{line}");
                                            crate::render::vulkan_gpu::record_probe_result(&line);
                                        }
                                        cur = output;
                                        vulkan_applied = true;
                                    }
                                    Ok(None) => {
                                        let line = format!(
                                            "vulkan-production-glsl: result=fallback shader='{}' requested_luid={luid:016x} reason=onepass-subset-not-admitted fallback=OpenGL",
                                            shader.name(),
                                        );
                                        vulkan_onepass::record_route_once(
                                            format!("not-admitted:{luid:016x}:{}", shader.name()),
                                            &line,
                                        );
                                    }
                                    Err(error) => {
                                        let line = format!(
                                            "vulkan-production-glsl: result=fallback shader='{}' requested_luid={:016x} reason={:#} fallback=OpenGL",
                                            shader.name(),
                                            luid,
                                            error
                                        );
                                        log::warn!("{line}");
                                        crate::render::vulkan_gpu::record_probe_result(&line);
                                    }
                                }
                            }
                        } else {
                            let line = format!(
                                "vulkan-production-glsl: result=fallback shader='{}' reason=explicit-luid-missing fallback=OpenGL",
                                shader.name(),
                            );
                            vulkan_onepass::record_route_once(
                                format!("luid-missing:{}", shader.name()),
                                &line,
                            );
                        }
                    }
                    if !vulkan_applied {
                        let timer = probe
                            .is_some()
                            .then(|| gc.begin_gpu_timer(&metric_labels[stage_index]))
                            .flatten();
                        let applied = GlslEngine::apply(gc, shader, cur, out_size);
                        if let Some(query) = timer {
                            gc.end_gpu_timer(query, metric_labels[stage_index].clone());
                        }
                        cur = applied.with_context(|| {
                            format!("GLSL stage failed: {} ({})", shader.name(), shader.path)
                        })?;
                    }
                }
                Stage::Onnx {
                    is_interp: true, ..
                } => {
                    // interpolation stages are handled by the engine, not as a
                    // per-frame image filter: pass through here
                }
                Stage::Flow { .. } => {
                    // handled by the engine (needs two frames)
                }
                Stage::Onnx { stage, .. } => {
                    // mpv user-shader OFFSET metadata is meant to be consumed
                    // by the next scaler. ONNX is outside mpv's hook pipeline,
                    // so normalize the phase before handing pixels to the model
                    // rather than silently dropping the pending correction.
                    if cur.has_offset() {
                        cur = crate::render::scaler::align_offset(gc, cur)
                            .context("failed to align pending mpv shader OFFSET before ONNX")?;
                    }
                    let mut stage = stage.lock().unwrap();
                    if stage.is_directml_temporal_filter() {
                        if let Some((limited_w, limited_h)) =
                            directml_temporal_limited_size(cur.w(), cur.h())
                        {
                            let source_size = (cur.w(), cur.h());
                            cur = crate::render::scaler::resample(
                                gc,
                                cur,
                                limited_w,
                                limited_h,
                                crate::render::scaler::Kernel::Spline36,
                            )
                            .with_context(|| {
                                format!(
                                    "DirectML temporal height-limit resample failed for {}: {}x{} -> {}x{}",
                                    stage.name, source_size.0, source_size.1, limited_w, limited_h
                                )
                            })?;
                            log_directml_temporal_height_limit(
                                dml_temporal_limit_log_keys,
                                stage_index,
                                &stage.name,
                                source_size,
                                (limited_w, limited_h),
                            );
                        }
                    }
                    let input_size = (cur.w(), cur.h());
                    stage.prepare_tensorrt_input_shape(input_size.0, input_size.1);
                    super::onnx_stage::mark_tensorrt_model_started(&stage.name);
                    let gpu_texture = if after_interpolation {
                        stage.process_gpu_texture_after_interpolation(gc, cur)?
                    } else {
                        stage.process_gpu_texture(gc, cur)?
                    };
                    if let Some(texture) = gpu_texture {
                        cur = texture;
                    } else {
                        let rgb = gc.download_rgb8(cur);
                        let (ow, oh, out) = stage.process(cur.w(), cur.h(), &rgb)?;
                        cur = gc.upload_rgb8(ow, oh, &out);
                    }
                    if stage.complete_tensorrt_input_shape(input_size.0, input_size.1) {
                        super::onnx_stage::mark_tensorrt_model_completed(&stage.name);
                    }
                }
            }
            if let (Some(t0), Some(p)) = (t0, probe.as_deref_mut()) {
                // ONNX providers expose completed inference timing here.
                // GLSL stages use asynchronous GL_TIME_ELAPSED queries above.
                p(
                    &metric_labels[stage_index],
                    stage.kind(),
                    t0.elapsed().as_secs_f64() * 1000.0,
                );
            }
        }
        Ok(cur)
    }
}

fn skip_extra_interp_message(name: &str) -> String {
    format!(
        "Only one frame-interpolation filter can be active. Additional filter '{name}' was skipped. Choose x2/x3/x4/x5 with Frame interpolation."
    )
}

fn looks_like_interp_path(path: &str) -> bool {
    let lower = PathBuf::from(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_else(|| path.to_ascii_lowercase());
    [
        "rife",
        "drba",
        "gmfss",
        "dain",
        "ifrnet",
        "interpolation",
        "frameinterp",
        "frame_interpolation",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flow_spec(path: &str) -> StageSpec {
        StageSpec {
            kind: StageKind::Flow,
            path: path.into(),
            enabled: true,
            params: Default::default(),
        }
    }

    #[test]
    fn directml_temporal_height_limit_caps_only_above_1080() {
        assert_eq!(
            directml_temporal_limited_size(2304, 1296),
            Some((1920, 1080))
        );
        assert_eq!(
            directml_temporal_limited_size(3840, 2160),
            Some((1920, 1080))
        );
        assert_eq!(
            directml_temporal_limited_size(2560, 1440),
            Some((1920, 1080))
        );
        assert_eq!(directml_temporal_limited_size(1920, 1080), None);
        assert_eq!(directml_temporal_limited_size(1152, 648), None);
    }

    #[test]
    fn duplicate_stage_metrics_get_separate_stable_rows() {
        let labels = disambiguate_stage_metric_labels(&[
            "FSRCNNX_x2.glsl".to_string(),
            "FSRCNNX_x2.glsl".to_string(),
            "Anime4K.glsl".to_string(),
        ]);
        assert_eq!(
            labels,
            vec!["FSRCNNX_x2.glsl #1", "FSRCNNX_x2.glsl #2", "Anime4K.glsl",]
        );
    }

    #[test]
    fn duplicate_provider_suffix_is_preserved_in_metrics_label() {
        let labels = disambiguate_stage_metric_labels(&[
            "model.onnx [DirectML]".to_string(),
            "model.onnx [DirectML]".to_string(),
        ]);
        assert_eq!(
            labels,
            vec!["model.onnx [DirectML] #1", "model.onnx [DirectML] #2",]
        );
    }

    #[test]
    fn builtin_neoflow_is_frozen() {
        let mut factory = StageFactory::new(PathBuf::from("."));
        let error = factory.build(&flow_spec("builtin:NeoFlow")).err().unwrap();
        assert!(error.to_string().contains("temporarily disabled"));
    }

    #[test]
    fn applies_per_stage_resize_scale() {
        let mut factory = StageFactory::new(PathBuf::from("."));
        let mut params = std::collections::BTreeMap::new();
        params.insert("RESIZE_SCALE".to_string(), 1.25);
        let stage = factory
            .build(&StageSpec {
                kind: StageKind::Glsl,
                path: "shaders/Resize/Bilinear_Neo.glsl".into(),
                enabled: true,
                params,
            })
            .unwrap();
        let Stage::Glsl { shader, .. } = stage else {
            panic!("expected GLSL stage");
        };
        let scale = shader
            .params
            .iter()
            .find(|param| param.name == "RESIZE_SCALE")
            .unwrap();
        assert!((scale.value - 1.25).abs() < f32::EPSILON);
    }

    #[test]
    #[ignore = "built-in NeoFlow is intentionally frozen"]
    fn keeps_only_first_frame_interpolator() {
        let mut factory = StageFactory::new(PathBuf::from("."));
        let (chain, errors) = FilterChain::from_specs(
            &mut factory,
            &[flow_spec("builtin:NeoFlow"), flow_spec("x")],
        );

        assert_eq!(chain.stages.len(), 1);
        assert!(matches!(chain.stages[0], Stage::Flow { .. }));
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("frame-interpolation"));
    }

    #[test]
    #[ignore = "built-in NeoFlow is intentionally frozen"]
    fn keeps_builtin_flow_name() {
        let mut factory = StageFactory::new(PathBuf::from("."));
        let stage = factory.build(&flow_spec("builtin:NeoFlowSigma")).unwrap();
        assert_eq!(stage.name(), "NeoFlow");
        // legacy saved presets must resolve to Σ, not break
        let stage = factory.build(&flow_spec("builtin:NeoFlowKari")).unwrap();
        assert_eq!(stage.name(), "NeoFlow");
    }

    #[test]
    #[ignore = "built-in NeoFlow is intentionally frozen"]
    fn skips_obvious_second_interp_onnx_before_loading() {
        let mut factory = StageFactory::new(PathBuf::from("."));
        let specs = [
            flow_spec("builtin:NeoFlow"),
            StageSpec {
                kind: StageKind::Onnx,
                path: "models/rife_v4.25_lite.onnx".into(),
                enabled: true,
                params: Default::default(),
            },
        ];
        let (chain, errors) = FilterChain::from_specs(&mut factory, &specs);

        assert_eq!(chain.stages.len(), 1);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("rife_v4.25_lite.onnx"));
    }
}
