//! FilterChain: ordered, mixed GLSL/ONNX stages over one shared GlContext.
//!
//! GLSL->GLSL stays GPU-resident; the CPU round-trip happens only at
//! GLSL<->ONNX boundaries. Expensive resources (ONNX sessions, parsed
//! shaders) are cached by path in the factory so live chain swaps are cheap.

use super::gl::{GlContext, GpuTex};
use super::glsl_engine::GlslEngine;
use super::mpv::UserShader;
use super::onnx_stage::{OnnxProvider, OnnxStage};
use crate::core::config::{OnnxBackendPreference, StageKind, StageSpec, resolve_path};
use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::rc::Rc;
// ONNX stages are Arc<Mutex<…>> (not Rc<RefCell<…>>) so frame-interpolation
// inference can run on a worker thread while the engine keeps presenting.
use std::sync::{Arc, Mutex};

const BUILTIN_NEOFLOW_ENABLED: bool = false;

// DirectML temporal restoration models have shown corrupted output and a very
// steep cost increase at high input resolutions. Keep only the temporal ONNX
// inference at <=1080 pixels high, preserving aspect ratio; following stages
// can upscale again normally. TensorRT/CUDA, single-frame ONNX and frame
// interpolation are intentionally untouched.
const DIRECTML_TEMPORAL_MAX_HEIGHT: i32 = 1080;

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
        Self {
            base_dir: self.base_dir.clone(),
            gpu_adapter: self.gpu_adapter,
            onnx_preference: preference,
            trt_device_id,
            trt_cache_root,
            shaders: self.shaders.clone(),
            onnx: HashMap::new(),
            onnx_lru: VecDeque::new(),
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

pub struct FilterChain {
    pub stages: Vec<Stage>,
    neodeint_bypass: bool,
    dml_temporal_limit_log_keys: HashSet<(usize, i32, i32, i32, i32)>,
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
                        log::error!(
                            "onnx-stage-lock-poisoned: action=recover-for-interp-provider"
                        );
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
        (
            Self {
                stages,
                neodeint_bypass,
                dml_temporal_limit_log_keys: HashSet::new(),
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
        if stage.is_directml_temporal_filter()
            && directml_temporal_limited_size(w, h).is_some()
        {
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
        assert_eq!(directml_temporal_limited_size(2304, 1296), Some((1920, 1080)));
        assert_eq!(directml_temporal_limited_size(3840, 2160), Some((1920, 1080)));
        assert_eq!(directml_temporal_limited_size(2560, 1440), Some((1920, 1080)));
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
