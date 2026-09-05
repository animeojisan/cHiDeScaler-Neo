//! Clean-room Libretro Slang preset reader/runtime for externally supplied files.
//!
//! This module intentionally contains no shader code or preset data from
//! RetroArch/RetroCrisis. It implements a small, generic compatibility layer
//! from the documented .slangp/.slang concepts to Neo's existing OpenGL
//! full-screen pass runner. Users provide their own shader packs at runtime.

use super::gl::{Dtype, GlContext, GpuTex};
use anyhow::{Context, Result, anyhow, bail};
use glow::HasContext;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScaleType {
    Source,
    Viewport,
    Absolute,
}

impl ScaleType {
    fn parse(value: Option<&str>) -> Self {
        match value
            .unwrap_or("source")
            .trim()
            .trim_matches('"')
            .to_ascii_lowercase()
            .as_str()
        {
            "viewport" => Self::Viewport,
            "absolute" => Self::Absolute,
            _ => Self::Source,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WrapMode {
    ClampEdge,
    ClampBorder,
    Repeat,
    Mirror,
}

impl WrapMode {
    fn parse(value: Option<&str>) -> Self {
        match value
            .unwrap_or("clamp_to_edge")
            .trim()
            .trim_matches('"')
            .to_ascii_lowercase()
            .as_str()
        {
            "repeat" => Self::Repeat,
            "mirrored_repeat" | "mirror" | "mirrored" => Self::Mirror,
            "clamp_to_border" => Self::ClampBorder,
            _ => Self::ClampEdge,
        }
    }

    fn gl(self) -> u32 {
        match self {
            Self::Repeat => glow::REPEAT,
            Self::Mirror => glow::MIRRORED_REPEAT,
            Self::ClampBorder => glow::CLAMP_TO_BORDER,
            Self::ClampEdge => glow::CLAMP_TO_EDGE,
        }
    }
}

#[derive(Clone, Debug)]
struct ParameterDecl {
    default: f32,
    min: f32,
    max: f32,
}

#[derive(Clone, Debug)]
struct UniformField {
    gl_name: String,
    field_name: String,
    ty: String,
}

#[derive(Clone, Debug)]
struct ShaderSource {
    vertex: String,
    fragment: String,
    uniforms: Vec<UniformField>,
    samplers: Vec<String>,
    parameters: BTreeMap<String, ParameterDecl>,
    f16_output: bool,
}

#[derive(Clone, Debug)]
struct PassSpec {
    path: PathBuf,
    alias: Option<String>,
    filter_linear: bool,
    wrap: WrapMode,
    mipmap_input: bool,
    float_framebuffer: bool,
    scale_x_type: ScaleType,
    scale_y_type: ScaleType,
    scale_x: f32,
    scale_y: f32,
    frame_count_mod: Option<u32>,
    shader: ShaderSource,
}

#[derive(Clone, Debug)]
struct TextureSpec {
    name: String,
    path: PathBuf,
    linear: bool,
    wrap: WrapMode,
    mipmap: bool,
}

#[derive(Clone, Debug)]
pub struct SlangPreset {
    path: PathBuf,
    display_name: String,
    passes: Vec<PassSpec>,
    textures: Vec<TextureSpec>,
    values: BTreeMap<String, f32>,
    fingerprint: u64,
    feedback_pass: Option<usize>,
    feedback_references: HashSet<usize>,
    original_history_depth: usize,
}

#[derive(Clone, Debug)]
pub struct SlangStage {
    preset: SlangPreset,
    frame_count: u32,
    logged_active: bool,
}

#[derive(Clone, Debug)]
pub struct SlangValidation {
    pub passes: usize,
    pub textures: usize,
    pub parameters: usize,
    pub history: usize,
}

/// Parse and fully expand an externally supplied preset without creating GL
/// resources. Used by the compatibility inventory; no third-party source is
/// copied into Neo or emitted by this API.
pub fn validate_external_preset(path: &Path) -> Result<SlangValidation> {
    let preset = SlangPreset::load(path, &BTreeMap::new())?;
    Ok(SlangValidation {
        passes: preset.passes.len(),
        textures: preset.textures.len(),
        parameters: preset.values.len(),
        history: preset.original_history_depth,
    })
}

impl SlangStage {
    pub fn load(path: &Path, instance_values: &BTreeMap<String, f32>) -> Result<Self> {
        let preset = SlangPreset::load(path, instance_values)?;
        log::info!(
            "slangp-load: preset='{}' passes={} textures={} feedback_pass={:?} feedback_refs={:?} history={} path={}",
            preset.display_name,
            preset.passes.len(),
            preset.textures.len(),
            preset.feedback_pass,
            preset.feedback_references,
            preset.original_history_depth,
            preset.path.display(),
        );
        Ok(Self {
            preset,
            frame_count: 0,
            logged_active: false,
        })
    }

    pub fn display_name(&self) -> &str {
        &self.preset.display_name
    }

    pub fn apply(
        &mut self,
        gc: &mut GlContext,
        input: GpuTex,
        viewport: (i32, i32),
        advance_frame: bool,
    ) -> Result<GpuTex> {
        if self.preset.passes.is_empty() {
            return Ok(input);
        }
        if input.has_offset() {
            bail!("SLANGP cannot consume a pending mpv OFFSET; align before this stage");
        }

        let original = input;
        let mut previous = input;
        let mut outputs: Vec<GpuTex> = Vec::with_capacity(self.preset.passes.len());
        let mut aliases: HashMap<String, GpuTex> = HashMap::new();
        let mut lut_textures: HashMap<String, GpuTex> = HashMap::new();
        for texture in &self.preset.textures {
            let tex = load_external_texture(gc, self.preset.fingerprint, texture)
                .with_context(|| format!("SLANGP LUT '{}'", texture.path.display()))?;
            lut_textures.insert(texture.name.clone(), tex);
        }

        for (pass_index, pass) in self.preset.passes.iter().enumerate() {
            let source = previous;
            let (out_w, out_h) = pass_output_size(pass, source, viewport);
            let dtype = if pass.float_framebuffer || pass.shader.f16_output {
                Dtype::F16
            } else {
                Dtype::U8
            };
            let output = gc.make_tex(out_w, out_h, 4, dtype);

            let mut resources: HashMap<String, GpuTex> = HashMap::new();
            resources.insert("Source".into(), source);
            resources.insert("Original".into(), original);
            for (index, tex) in outputs.iter().enumerate() {
                resources.insert(format!("PassOutput{index}"), *tex);
            }
            for (name, tex) in &aliases {
                resources.insert(name.clone(), *tex);
            }
            for (name, tex) in &lut_textures {
                resources.insert(name.clone(), *tex);
            }

            // Generic Libretro pass feedback. The preset-wide feedback_pass is
            // accepted as well as explicit PassFeedback#/AliasFeedback sampler
            // references discovered from the shader source.
            for feedback_index in &self.preset.feedback_references {
                if let Some(tex) = gc.persistent_texture_by_key(&feedback_key(
                    self.preset.fingerprint,
                    *feedback_index,
                )) {
                    resources.insert(format!("PassFeedback{feedback_index}"), tex);
                    if let Some(alias) = self
                        .preset
                        .passes
                        .get(*feedback_index)
                        .and_then(|p| p.alias.as_ref())
                    {
                        resources.insert(format!("{alias}Feedback"), tex);
                    }
                }
            }
            if let Some(index) = self.preset.feedback_pass {
                if let Some(tex) =
                    gc.persistent_texture_by_key(&feedback_key(self.preset.fingerprint, index))
                {
                    resources.insert(format!("PassFeedback{index}"), tex);
                    if let Some(alias) =
                        self.preset.passes.get(index).and_then(|p| p.alias.as_ref())
                    {
                        resources.insert(format!("{alias}Feedback"), tex);
                    }
                }
            }
            // The first frame has no feedback. Slang implementations expose a
            // valid texture; using the current source is deterministic and avoids
            // undefined samples during warm-up.
            let fallback_feedback = source;
            for sampler in &pass.shader.samplers {
                if (sampler.starts_with("PassFeedback") || sampler.ends_with("Feedback"))
                    && !resources.contains_key(sampler)
                {
                    resources.insert(sampler.clone(), fallback_feedback);
                }
            }

            for history in 0..self.preset.original_history_depth {
                let key = original_history_key(self.preset.fingerprint, history);
                if let Some(tex) = gc.persistent_texture_by_key(&key) {
                    resources.insert(format!("OriginalHistory{history}"), tex);
                } else {
                    resources.insert(format!("OriginalHistory{history}"), original);
                }
            }

            if pass.mipmap_input {
                set_texture_sampling(gc, source, true, pass.wrap, true);
            } else {
                set_texture_sampling(gc, source, pass.filter_linear, pass.wrap, false);
            }

            let cache_key = format!(
                "slangp:{}:{}:{}",
                self.preset.fingerprint,
                pass_index,
                source_fingerprint(&pass.shader.vertex, &pass.shader.fragment)
            );
            let program = gc
                .program_pair_named(&cache_key, &pass.shader.vertex, &pass.shader.fragment)
                .map_err(|error| {
                    anyhow!(
                        "SLANGP pass {} compile failed ({}): {}",
                        pass_index,
                        pass.path.display(),
                        error
                    )
                })?;
            gc.bind_target(output);
            draw_pass(
                gc,
                program,
                &pass.shader,
                &resources,
                &self.preset.values,
                source,
                original,
                output,
                viewport,
                self.frame_count,
                pass.frame_count_mod,
            )?;
            gc.unbind_target();
            restore_pool_sampling(gc, source);

            previous = output;
            outputs.push(output);
            if let Some(alias) = &pass.alias {
                if !alias.is_empty() {
                    aliases.insert(alias.clone(), output);
                }
            }
        }

        // Persist only pass outputs which are actually requested as feedback.
        let mut persist_feedback = self.preset.feedback_references.clone();
        if let Some(index) = self.preset.feedback_pass {
            persist_feedback.insert(index);
        }
        for index in persist_feedback {
            if let Some(output) = outputs.get(index).copied() {
                gc.copy_to_persistent_rgba(&feedback_key(self.preset.fingerprint, index), output)
                    .map_err(anyhow::Error::msg)?;
            }
        }
        if self.preset.original_history_depth > 0 {
            for history in (1..self.preset.original_history_depth).rev() {
                if let Some(previous_history) = gc.persistent_texture_by_key(&original_history_key(
                    self.preset.fingerprint,
                    history - 1,
                )) {
                    gc.copy_to_persistent_rgba(
                        &original_history_key(self.preset.fingerprint, history),
                        previous_history,
                    )
                    .map_err(anyhow::Error::msg)?;
                }
            }
            gc.copy_to_persistent_rgba(&original_history_key(self.preset.fingerprint, 0), original)
                .map_err(anyhow::Error::msg)?;
        }

        if !self.logged_active {
            self.logged_active = true;
            log::info!(
                "slangp-active: preset='{}' passes={} input={}x{} output={}x{} renderer=OpenGL clean_room_loader=true external_content=true",
                self.preset.display_name,
                self.preset.passes.len(),
                input.w(),
                input.h(),
                previous.w(),
                previous.h(),
            );
        }
        if advance_frame {
            self.frame_count = self.frame_count.wrapping_add(1);
        }
        Ok(previous)
    }
}

impl SlangPreset {
    fn load(path: &Path, instance_values: &BTreeMap<String, f32>) -> Result<Self> {
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        let values_text = load_preset_with_references(&canonical)?;
        let count = values_text
            .get("shaders")
            .and_then(|v| parse_usize(v))
            .ok_or_else(|| {
                anyhow!(
                    "SLANGP missing valid 'shaders' count: {}",
                    canonical.display()
                )
            })?;
        if count == 0 || count > 64 {
            bail!(
                "SLANGP shader count {} is outside Neo's 1..64 safety range",
                count
            );
        }
        let root_hint = locate_shader_root(&canonical);
        let mut passes = Vec::with_capacity(count);
        let mut declared_parameters: BTreeMap<String, ParameterDecl> = BTreeMap::new();
        let mut feedback_references = HashSet::new();
        let mut original_history_depth = 0usize;

        for index in 0..count {
            let shader_key = format!("shader{index}");
            let shader_value = values_text
                .get(&shader_key)
                .ok_or_else(|| anyhow!("SLANGP missing {shader_key}: {}", canonical.display()))?;
            let shader_path =
                resolve_external_path(&canonical, shader_value, root_hint.as_deref())?;
            let shader = load_shader_source(&shader_path)?;
            declared_parameters.extend(shader.parameters.clone());
            for sampler in &shader.samplers {
                if let Some(index) = sampler
                    .strip_prefix("PassFeedback")
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    feedback_references.insert(index);
                }
                if let Some(index) = sampler
                    .strip_prefix("OriginalHistory")
                    .and_then(|n| n.parse::<usize>().ok())
                {
                    original_history_depth = original_history_depth.max(index + 1);
                }
            }
            let generic_scale_type = values_text.get(&format!("scale_type{index}"));
            let generic_scale = values_text
                .get(&format!("scale{index}"))
                .and_then(|v| parse_f32(v))
                .unwrap_or(1.0);
            let has_explicit_scale = [
                format!("scale_type{index}"),
                format!("scale_type_x{index}"),
                format!("scale_type_y{index}"),
                format!("scale{index}"),
                format!("scale_x{index}"),
                format!("scale_y{index}"),
            ]
            .iter()
            .any(|key| values_text.contains_key(key));
            // Libretro Slang treats an otherwise-unspecified final pass as the
            // presentation pass and stretches it to the final viewport. All
            // earlier unspecified passes remain source 1x.
            let implicit_final_viewport = index + 1 == count && !has_explicit_scale;
            let scale_x_type = if implicit_final_viewport {
                ScaleType::Viewport
            } else {
                ScaleType::parse(
                    values_text
                        .get(&format!("scale_type_x{index}"))
                        .map(String::as_str)
                        .or_else(|| generic_scale_type.map(String::as_str)),
                )
            };
            let scale_y_type = if implicit_final_viewport {
                ScaleType::Viewport
            } else {
                ScaleType::parse(
                    values_text
                        .get(&format!("scale_type_y{index}"))
                        .map(String::as_str)
                        .or_else(|| generic_scale_type.map(String::as_str)),
                )
            };
            let scale_x = values_text
                .get(&format!("scale_x{index}"))
                .and_then(|v| parse_f32(v))
                .unwrap_or(generic_scale);
            let scale_y = values_text
                .get(&format!("scale_y{index}"))
                .and_then(|v| parse_f32(v))
                .unwrap_or(generic_scale);
            if values_text
                .get(&format!("srgb_framebuffer{index}"))
                .is_some_and(|v| parse_bool(v))
            {
                bail!(
                    "SLANGP pass {index} requests srgb_framebuffer=true; v693 intentionally rejects this until the framebuffer color-space path is verified"
                );
            }
            passes.push(PassSpec {
                path: shader_path,
                alias: values_text
                    .get(&format!("alias{index}"))
                    .map(|s| unquote(s))
                    .filter(|s| !s.is_empty()),
                filter_linear: values_text
                    .get(&format!("filter_linear{index}"))
                    .is_some_and(|v| parse_bool(v)),
                wrap: WrapMode::parse(
                    values_text
                        .get(&format!("wrap_mode{index}"))
                        .map(String::as_str),
                ),
                mipmap_input: values_text
                    .get(&format!("mipmap_input{index}"))
                    .is_some_and(|v| parse_bool(v)),
                float_framebuffer: values_text
                    .get(&format!("float_framebuffer{index}"))
                    .is_some_and(|v| parse_bool(v)),
                scale_x_type,
                scale_y_type,
                scale_x,
                scale_y,
                frame_count_mod: values_text
                    .get(&format!("frame_count_mod{index}"))
                    .and_then(|v| parse_u32(v))
                    .filter(|v| *v > 0),
                shader,
            });
        }

        let mut values = BTreeMap::new();
        for (name, decl) in declared_parameters {
            let preset_value = values_text.get(&name).and_then(|v| parse_f32(v));
            let instance_value = instance_values.get(&name).copied();
            let value = instance_value.or(preset_value).unwrap_or(decl.default);
            values.insert(name, value.clamp(decl.min, decl.max));
        }
        // Some presets override compile/runtime values not repeated as a pragma
        // in every pass. Retain every numeric key so UBO fields can resolve it.
        for (key, raw) in &values_text {
            if let Some(value) = parse_f32(raw) {
                values.entry(key.clone()).or_insert(value);
            }
        }
        for (key, value) in instance_values {
            values.insert(key.clone(), *value);
        }

        let textures = load_texture_specs(&canonical, &values_text, root_hint.as_deref())?;
        let feedback_pass = values_text
            .get("feedback_pass")
            .and_then(|v| parse_usize(v));
        if let Some(index) = feedback_pass {
            if index >= count {
                bail!(
                    "SLANGP feedback_pass {} exceeds shader count {}",
                    index,
                    count
                );
            }
            feedback_references.insert(index);
        }
        // AliasFeedback references need reverse resolution to their pass index.
        for pass in &passes {
            for sampler in &pass.shader.samplers {
                if let Some(alias) = sampler.strip_suffix("Feedback") {
                    if let Some((index, _)) = passes
                        .iter()
                        .enumerate()
                        .find(|(_, candidate)| candidate.alias.as_deref() == Some(alias))
                    {
                        feedback_references.insert(index);
                    }
                }
            }
        }

        let mut hasher = DefaultHasher::new();
        canonical.to_string_lossy().hash(&mut hasher);
        values_text.iter().for_each(|pair| pair.hash(&mut hasher));
        let fingerprint = hasher.finish();
        let display_name = canonical
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("SLANGP preset")
            .to_string();
        Ok(Self {
            path: canonical,
            display_name,
            passes,
            textures,
            values,
            fingerprint,
            feedback_pass,
            feedback_references,
            original_history_depth,
        })
    }
}

fn load_preset_with_references(path: &Path) -> Result<BTreeMap<String, String>> {
    fn recurse(
        path: &Path,
        root_hint: Option<&Path>,
        visiting: &mut HashSet<PathBuf>,
        depth: usize,
    ) -> Result<BTreeMap<String, String>> {
        if depth > 24 {
            bail!("SLANGP #reference depth exceeded at {}", path.display());
        }
        let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if !visiting.insert(canonical.clone()) {
            bail!(
                "SLANGP #reference cycle detected at {}",
                canonical.display()
            );
        }
        let text = std::fs::read_to_string(&canonical)
            .with_context(|| format!("read SLANGP {}", canonical.display()))?;
        let mut merged = BTreeMap::new();
        let mut references = Vec::new();
        for raw in text.lines() {
            let line = raw.trim();
            if line.starts_with("#reference") {
                let value = line["#reference".len()..].trim();
                if !value.is_empty() {
                    references.push(value.to_string());
                }
            }
        }
        let discovered_root = locate_shader_root(&canonical);
        let local_root = root_hint.or(discovered_root.as_deref());
        for reference in references {
            let reference_path = resolve_external_path(&canonical, &reference, local_root)?;
            merged.extend(recurse(&reference_path, local_root, visiting, depth + 1)?);
        }
        for raw in text.lines() {
            let line = strip_preset_comment(raw).trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((key, value)) = line.split_once('=') else {
                continue;
            };
            let key = key.trim();
            if !key.is_empty() {
                merged.insert(key.to_string(), value.trim().to_string());
            }
        }
        visiting.remove(&canonical);
        Ok(merged)
    }

    let mut visiting = HashSet::new();
    let root = locate_shader_root(path);
    recurse(path, root.as_deref(), &mut visiting, 0)
}

fn strip_preset_comment(line: &str) -> &str {
    // Quotes in preset paths may legally contain spaces but not unescaped '#'
    // in the packs targeted by Neo. Preserve #reference separately above.
    line.split('#').next().unwrap_or(line)
}

fn load_texture_specs(
    preset: &Path,
    values: &BTreeMap<String, String>,
    root_hint: Option<&Path>,
) -> Result<Vec<TextureSpec>> {
    let Some(list) = values.get("textures") else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for name in unquote(list)
        .split(';')
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        let raw_path = values
            .get(name)
            .ok_or_else(|| anyhow!("SLANGP texture '{name}' has no path"))?;
        out.push(TextureSpec {
            name: name.to_string(),
            path: resolve_external_path(preset, raw_path, root_hint)?,
            linear: values
                .get(&format!("{name}_linear"))
                .is_some_and(|v| parse_bool(v)),
            wrap: WrapMode::parse(values.get(&format!("{name}_wrap_mode")).map(String::as_str)),
            mipmap: values
                .get(&format!("{name}_mipmap"))
                .is_some_and(|v| parse_bool(v)),
        });
    }
    Ok(out)
}

fn locate_shader_root(path: &Path) -> Option<PathBuf> {
    // Neo v693 has exactly one external Slang root: <app>/slangp/. The
    // user may keep a complete RetroArch-style `shaders_slang/` dependency
    // tree *inside* that folder, but Neo never scans an app-level
    // `shaders_slang/` directory.
    for ancestor in path.ancestors() {
        if ancestor
            .file_name()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("slangp"))
        {
            return Some(ancestor.to_path_buf());
        }
    }
    None
}

fn resolve_external_path(owner: &Path, raw: &str, root_hint: Option<&Path>) -> Result<PathBuf> {
    let raw = unquote(raw).replace('\\', "/");
    let candidate = PathBuf::from(&raw);
    if candidate.is_absolute() && candidate.exists() {
        return Ok(candidate);
    }
    if let Some(parent) = owner.parent() {
        let relative = parent.join(&candidate);
        if relative.exists() {
            return Ok(relative.canonicalize().unwrap_or(relative));
        }
    }
    if let Some(root) = root_hint {
        // RetroArch preset packs are commonly copied out of their original
        // `config/` tree while retaining references such as
        // `../../../shaders_slang/crt/...`.  Rebase the stable dependency
        // anchor instead of letting a different directory depth escape Neo's
        // external slangp root.  The shader data remains user supplied.
        if let Some(anchored) = dependency_anchor_suffix(&candidate) {
            let rooted = root.join(anchored);
            if rooted.exists() {
                return Ok(rooted.canonicalize().unwrap_or(rooted));
            }
        }
        // A reference such as `shaders_slang/crt/...` resolves below Neo's
        // dedicated slangp root, i.e. <app>/slangp/shaders_slang/crt/....
        let rooted = root.join(&candidate);
        if rooted.exists() {
            return Ok(rooted.canonicalize().unwrap_or(rooted));
        }
    }
    // Search only for Neo's dedicated slangp root. This is intentionally
    // bounded and deterministic; there is no global filesystem crawling.
    for ancestor in owner.ancestors().take(8) {
        let base = ancestor.join("slangp");
        if !base.is_dir() {
            continue;
        }
        let rooted = base.join(&candidate);
        if rooted.exists() {
            return Ok(rooted.canonicalize().unwrap_or(rooted));
        }
    }
    bail!(
        "external SLANGP dependency not found: '{}' referenced by '{}' (copy the complete external shader pack/dependency tree under <app>/slangp/)",
        raw,
        owner.display()
    )
}

fn dependency_anchor_suffix(path: &Path) -> Option<PathBuf> {
    let components: Vec<_> = path.components().collect();
    let anchor = components.iter().position(|component| {
        component
            .as_os_str()
            .to_str()
            .is_some_and(|name| name.eq_ignore_ascii_case("shaders_slang"))
    })?;
    let mut suffix = PathBuf::new();
    for component in &components[anchor..] {
        suffix.push(component.as_os_str());
    }
    Some(suffix)
}

fn load_shader_source(path: &Path) -> Result<ShaderSource> {
    let mut visiting = HashSet::new();
    let expanded = expand_includes(path, &mut visiting, 0)?;
    let parameters = parse_parameters(&expanded);
    let f16_output = expanded.lines().any(|line| {
        line.trim().starts_with("#pragma format")
            && line.to_ascii_uppercase().contains("16")
            && line.to_ascii_uppercase().contains("FLOAT")
    });
    let (common, vertex, fragment) = split_stages(&expanded)?;
    let vertex_raw = format!("{common}\n{vertex}");
    let fragment_raw = format!("{common}\n{fragment}");
    let mut uniforms = Vec::new();
    let vertex = rewrite_slang_glsl(&vertex_raw, &mut uniforms)?;
    let fragment = rewrite_slang_glsl(&fragment_raw, &mut uniforms)?;
    dedup_uniforms(&mut uniforms);
    let samplers = parse_samplers(&fragment);
    Ok(ShaderSource {
        vertex,
        fragment,
        uniforms,
        samplers,
        parameters,
        f16_output,
    })
}

fn expand_includes(path: &Path, visiting: &mut HashSet<PathBuf>, depth: usize) -> Result<String> {
    if depth > 32 {
        bail!("SLANG include depth exceeded: {}", path.display());
    }
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    if !visiting.insert(canonical.clone()) {
        bail!("SLANG include cycle: {}", canonical.display());
    }
    let text = std::fs::read_to_string(&canonical)
        .with_context(|| format!("read SLANG shader {}", canonical.display()))?;
    let mut out = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#include") {
            let include_raw = trimmed["#include".len()..].trim();
            let include = resolve_external_path(
                &canonical,
                include_raw,
                locate_shader_root(&canonical).as_deref(),
            )?;
            out.push_str(&expand_includes(&include, visiting, depth + 1)?);
            out.push('\n');
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    visiting.remove(&canonical);
    Ok(out)
}

fn parse_parameters(text: &str) -> BTreeMap<String, ParameterDecl> {
    let mut out = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with("#pragma parameter") {
            continue;
        }
        let rest = trimmed["#pragma parameter".len()..].trim();
        let Some((name, tail)) = take_token(rest) else {
            continue;
        };
        let tail = skip_quoted_token(tail.trim());
        let nums: Vec<f32> = tail
            .split_whitespace()
            .filter_map(|token| token.parse::<f32>().ok())
            .take(4)
            .collect();
        if nums.len() >= 3 {
            out.insert(
                name.to_string(),
                ParameterDecl {
                    default: nums[0],
                    min: nums[1],
                    max: nums[2],
                },
            );
        }
    }
    out
}

fn take_token(input: &str) -> Option<(&str, &str)> {
    let index = input.find(char::is_whitespace).unwrap_or(input.len());
    (!input[..index].is_empty()).then(|| (&input[..index], &input[index..]))
}

fn skip_quoted_token(input: &str) -> &str {
    let input = input.trim_start();
    if let Some(rest) = input.strip_prefix('"') {
        if let Some(end) = rest.find('"') {
            return &rest[end + 1..];
        }
    }
    input
        .find(char::is_whitespace)
        .map_or("", |index| &input[index..])
}

fn split_stages(text: &str) -> Result<(String, String, String)> {
    enum Section {
        Common,
        Vertex,
        Fragment,
    }
    let mut section = Section::Common;
    let mut common = String::new();
    let mut vertex = String::new();
    let mut fragment = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed == "#pragma stage vertex" {
            section = Section::Vertex;
            continue;
        }
        if trimmed == "#pragma stage fragment" {
            section = Section::Fragment;
            continue;
        }
        if trimmed.starts_with("#pragma parameter")
            || trimmed.starts_with("#pragma name")
            || trimmed.starts_with("#pragma format")
        {
            continue;
        }
        match section {
            Section::Common => {
                common.push_str(line);
                common.push('\n');
            }
            Section::Vertex => {
                vertex.push_str(line);
                vertex.push('\n');
            }
            Section::Fragment => {
                fragment.push_str(line);
                fragment.push('\n');
            }
        }
    }
    if vertex.trim().is_empty() || fragment.trim().is_empty() {
        bail!("SLANG source does not contain both vertex and fragment stages");
    }
    Ok((common, vertex, fragment))
}

fn rewrite_slang_glsl(text: &str, uniforms: &mut Vec<UniformField>) -> Result<String> {
    let mut source = normalize_version(text);
    source = rewrite_layout_set_binding(&source);
    source = rewrite_uniform_blocks(&source, uniforms)?;
    // Vulkan permits zero-based location declarations exactly like modern GL.
    // Descriptor set syntax was removed above; bindings remain legal GL 4.3.
    Ok(source)
}

fn normalize_version(text: &str) -> String {
    let mut out = String::new();
    let mut version_seen = false;
    for line in text.lines() {
        if line.trim_start().starts_with("#version") {
            if !version_seen {
                out.push_str("#version 430 core\n");
                version_seen = true;
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    if !version_seen {
        format!("#version 430 core\n{out}")
    } else {
        out
    }
}

fn rewrite_layout_set_binding(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        let mut line = line.to_string();
        if let Some(layout_start) = line.find("layout(") {
            if let Some(close_rel) = line[layout_start..].find(')') {
                let close = layout_start + close_rel;
                let inner = &line[layout_start + 7..close];
                if inner.contains("set") {
                    let kept: Vec<&str> = inner
                        .split(',')
                        .map(str::trim)
                        .filter(|part| !part.starts_with("set") && !part.starts_with("set "))
                        .collect();
                    let replacement = if kept.is_empty() {
                        String::new()
                    } else {
                        format!("layout({})", kept.join(", "))
                    };
                    line.replace_range(layout_start..=close, &replacement);
                }
            }
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

fn rewrite_uniform_blocks(text: &str, uniforms: &mut Vec<UniformField>) -> Result<String> {
    let mut source = text.to_string();
    let mut search_from = 0usize;
    loop {
        let Some(layout_rel) = source[search_from..].find("layout(") else {
            break;
        };
        let layout_start = search_from + layout_rel;
        let Some(layout_close_rel) = source[layout_start..].find(')') else {
            break;
        };
        let layout_close = layout_start + layout_close_rel;
        let after_layout = &source[layout_close + 1..];
        let Some(uniform_rel) = after_layout.find("uniform") else {
            break;
        };
        // A layout qualifier belongs to the immediately following declaration.
        // If another statement ends first, this is not the block we seek.
        if after_layout[..uniform_rel].contains(';') {
            search_from = layout_close + 1;
            continue;
        }
        let uniform_start = layout_close + 1 + uniform_rel;
        let after_uniform = &source[uniform_start + "uniform".len()..];
        let semicolon_rel = after_uniform.find(';');
        let brace_rel = after_uniform.find('{');
        let Some(brace_rel) = brace_rel else {
            search_from = semicolon_rel
                .map(|offset| uniform_start + "uniform".len() + offset + 1)
                .unwrap_or(layout_close + 1);
            continue;
        };
        if semicolon_rel.is_some_and(|semi| semi < brace_rel) {
            search_from = uniform_start + "uniform".len() + semicolon_rel.unwrap() + 1;
            continue;
        }
        let brace_open = uniform_start + "uniform".len() + brace_rel;
        let Some(brace_close) = matching_brace(&source, brace_open) else {
            bail!("unterminated SLANG uniform block");
        };
        let Some(semi_rel) = source[brace_close + 1..].find(';') else {
            bail!("unterminated SLANG uniform block instance");
        };
        let semi = brace_close + 1 + semi_rel;
        let instance_region = source[brace_close + 1..semi].trim();
        let instance = instance_region
            .split_whitespace()
            .last()
            .unwrap_or("global")
            .trim_matches(|c: char| c == '[' || c == ']')
            .to_string();
        let body = source[brace_open + 1..brace_close].to_string();
        let fields = parse_uniform_fields(&body, &instance)?;
        let declarations = fields
            .iter()
            .map(|field| format!("uniform {} {};\n", field.ty, field.gl_name))
            .collect::<String>();
        source.replace_range(layout_start..=semi, &declarations);
        for field in &fields {
            source = source.replace(
                &format!("{}.{}", instance, field.field_name),
                &field.gl_name,
            );
        }
        uniforms.extend(fields);
        search_from = layout_start + declarations.len();
    }
    Ok(source)
}

fn parse_uniform_fields(body: &str, instance: &str) -> Result<Vec<UniformField>> {
    let mut out = Vec::new();
    for statement in body.split(';') {
        let statement = statement.trim();
        if statement.is_empty() || statement.starts_with('#') {
            continue;
        }
        let mut parts = statement.split_whitespace();
        let Some(ty) = parts.next() else { continue };
        let names = parts.collect::<Vec<_>>().join(" ");
        if names.is_empty() || names.contains('{') || names.contains('}') {
            bail!("unsupported nested/anonymous SLANG uniform declaration: {statement}");
        }
        for raw_name in names.split(',') {
            let name = raw_name
                .trim()
                .trim_matches(|c: char| c == '\n' || c == '\r');
            if name.is_empty() {
                continue;
            }
            if name.contains('[') {
                bail!("uniform arrays are not supported yet: {statement}");
            }
            let field_name = name.to_string();
            let safe_instance: String = instance
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            out.push(UniformField {
                gl_name: format!("neo_{safe_instance}_{field_name}"),
                field_name,
                ty: ty.to_string(),
            });
        }
    }
    Ok(out)
}

fn matching_brace(text: &str, open: usize) -> Option<usize> {
    let mut depth = 0i32;
    for (offset, ch) in text[open..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(open + offset);
                }
            }
            _ => {}
        }
    }
    None
}

fn dedup_uniforms(fields: &mut Vec<UniformField>) {
    let mut seen = HashSet::new();
    fields.retain(|field| seen.insert(field.gl_name.clone()));
}

fn parse_samplers(source: &str) -> Vec<String> {
    let mut samplers = Vec::new();
    for statement in source.split(';') {
        let normalized = statement.replace('\n', " ");
        if !(normalized.contains("sampler2D") || normalized.contains("sampler3D")) {
            continue;
        }
        let tokens: Vec<&str> = normalized.split_whitespace().collect();
        if let Some(position) = tokens
            .iter()
            .position(|token| token.contains("sampler2D") || token.contains("sampler3D"))
        {
            if let Some(name) = tokens.get(position + 1) {
                let clean = name
                    .trim()
                    .trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '_');
                if !clean.is_empty() && !samplers.iter().any(|existing| existing == clean) {
                    samplers.push(clean.to_string());
                }
            }
        }
    }
    samplers
}

fn pass_output_size(pass: &PassSpec, source: GpuTex, viewport: (i32, i32)) -> (i32, i32) {
    let axis = |kind: ScaleType, scale: f32, source: i32, viewport: i32| -> i32 {
        let value = match kind {
            ScaleType::Source => source as f32 * scale,
            ScaleType::Viewport => viewport.max(1) as f32 * scale,
            ScaleType::Absolute => scale,
        };
        value.round().clamp(1.0, 16384.0) as i32
    };
    (
        axis(pass.scale_x_type, pass.scale_x, source.w(), viewport.0),
        axis(pass.scale_y_type, pass.scale_y, source.h(), viewport.1),
    )
}

fn draw_pass(
    gc: &GlContext,
    program: glow::Program,
    shader: &ShaderSource,
    resources: &HashMap<String, GpuTex>,
    values: &BTreeMap<String, f32>,
    source: GpuTex,
    original: GpuTex,
    output: GpuTex,
    viewport: (i32, i32),
    frame_count: u32,
    frame_count_mod: Option<u32>,
) -> Result<()> {
    let gl = gc.gl.clone();
    unsafe {
        gl.use_program(Some(program));
        let mut unit = 0u32;
        for sampler in &shader.samplers {
            let Some(texture) = resources.get(sampler).copied() else {
                // Optimized-out samplers have no location and need no resource.
                if gl.get_uniform_location(program, sampler).is_none() {
                    continue;
                }
                bail!("SLANGP sampler '{sampler}' has no bound resource");
            };
            gl.active_texture(glow::TEXTURE0 + unit);
            gl.bind_texture(texture.target(), Some(texture.tex));
            if let Some(location) = gl.get_uniform_location(program, sampler) {
                gl.uniform_1_i32(Some(&location), unit as i32);
            }
            unit += 1;
        }
        for field in &shader.uniforms {
            let Some(location) = gl.get_uniform_location(program, &field.gl_name) else {
                continue;
            };
            set_uniform_field(
                &gl,
                &location,
                field,
                resources,
                values,
                source,
                original,
                output,
                viewport,
                frame_count_mod.map_or(frame_count, |m| frame_count % m),
            );
        }
        gc.draw_fullscreen();
        gl.use_program(None);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn set_uniform_field(
    gl: &glow::Context,
    location: &glow::UniformLocation,
    field: &UniformField,
    resources: &HashMap<String, GpuTex>,
    values: &BTreeMap<String, f32>,
    source: GpuTex,
    original: GpuTex,
    output: GpuTex,
    viewport: (i32, i32),
    frame_count: u32,
) {
    unsafe {
        let name = field.field_name.as_str();
        if name == "MVP" && field.ty.starts_with("mat4") {
            let identity: [f32; 16] = [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            ];
            gl.uniform_matrix_4_f32_slice(Some(location), false, &identity);
            return;
        }
        if name == "FrameCount" {
            if field.ty == "uint" {
                gl.uniform_1_u32(Some(location), frame_count);
            } else {
                gl.uniform_1_i32(Some(location), frame_count as i32);
            }
            return;
        }
        if name == "FinalViewportSize" {
            set_size_uniform(gl, location, &field.ty, viewport.0, viewport.1);
            return;
        }
        if let Some(texture_name) = name.strip_suffix("Size") {
            let texture = match texture_name {
                "Source" => Some(source),
                "Original" => Some(original),
                "Output" => Some(output),
                other => resources.get(other).copied(),
            };
            if let Some(texture) = texture {
                set_size_uniform(gl, location, &field.ty, texture.w(), texture.h());
                return;
            }
        }
        if name == "SourceSize" {
            set_size_uniform(gl, location, &field.ty, source.w(), source.h());
            return;
        }
        if name == "OriginalSize" {
            set_size_uniform(gl, location, &field.ty, original.w(), original.h());
            return;
        }
        if name == "OutputSize" {
            set_size_uniform(gl, location, &field.ty, output.w(), output.h());
            return;
        }
        let value = values.get(name).copied().unwrap_or_else(|| match name {
            "FrameDirection" => 1.0,
            "Rotation" => 0.0,
            "OriginalAspect" | "OriginalAspectRotated" => {
                original.w() as f32 / original.h().max(1) as f32
            }
            "TotalSubFrames" => 1.0,
            "CurrentSubFrame" => 0.0,
            _ => 0.0,
        });
        match field.ty.as_str() {
            "int" => gl.uniform_1_i32(Some(location), value.round() as i32),
            "uint" => gl.uniform_1_u32(Some(location), value.max(0.0).round() as u32),
            "bool" => gl.uniform_1_i32(Some(location), (value != 0.0) as i32),
            "vec2" => gl.uniform_2_f32(Some(location), value, value),
            "vec3" => gl.uniform_3_f32(Some(location), value, value, value),
            "vec4" => gl.uniform_4_f32(Some(location), value, value, value, value),
            "ivec2" => gl.uniform_2_i32(Some(location), value.round() as i32, value.round() as i32),
            "ivec3" => gl.uniform_3_i32(
                Some(location),
                value.round() as i32,
                value.round() as i32,
                value.round() as i32,
            ),
            "ivec4" => gl.uniform_4_i32(
                Some(location),
                value.round() as i32,
                value.round() as i32,
                value.round() as i32,
                value.round() as i32,
            ),
            _ => gl.uniform_1_f32(Some(location), value),
        }
    }
}

fn set_size_uniform(
    gl: &glow::Context,
    location: &glow::UniformLocation,
    ty: &str,
    w: i32,
    h: i32,
) {
    unsafe {
        let wf = w.max(1) as f32;
        let hf = h.max(1) as f32;
        match ty {
            "vec2" => gl.uniform_2_f32(Some(location), wf, hf),
            "vec3" => gl.uniform_3_f32(Some(location), wf, hf, 1.0 / wf),
            _ => gl.uniform_4_f32(Some(location), wf, hf, 1.0 / wf, 1.0 / hf),
        }
    }
}

fn set_texture_sampling(
    gc: &GlContext,
    texture: GpuTex,
    linear: bool,
    wrap: WrapMode,
    mipmap: bool,
) {
    let gl = &gc.gl;
    unsafe {
        gl.bind_texture(texture.target(), Some(texture.tex));
        let mag = if linear { glow::LINEAR } else { glow::NEAREST } as i32;
        let min = if mipmap {
            if linear {
                glow::LINEAR_MIPMAP_LINEAR
            } else {
                glow::NEAREST_MIPMAP_NEAREST
            }
        } else if linear {
            glow::LINEAR
        } else {
            glow::NEAREST
        } as i32;
        gl.tex_parameter_i32(texture.target(), glow::TEXTURE_MIN_FILTER, min);
        gl.tex_parameter_i32(texture.target(), glow::TEXTURE_MAG_FILTER, mag);
        gl.tex_parameter_i32(texture.target(), glow::TEXTURE_WRAP_S, wrap.gl() as i32);
        gl.tex_parameter_i32(texture.target(), glow::TEXTURE_WRAP_T, wrap.gl() as i32);
        if mipmap {
            gl.generate_mipmap(texture.target());
        }
    }
}

fn restore_pool_sampling(gc: &GlContext, texture: GpuTex) {
    let gl = &gc.gl;
    unsafe {
        gl.bind_texture(texture.target(), Some(texture.tex));
        gl.tex_parameter_i32(
            texture.target(),
            glow::TEXTURE_MIN_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            texture.target(),
            glow::TEXTURE_MAG_FILTER,
            glow::NEAREST as i32,
        );
        gl.tex_parameter_i32(
            texture.target(),
            glow::TEXTURE_WRAP_S,
            glow::CLAMP_TO_EDGE as i32,
        );
        gl.tex_parameter_i32(
            texture.target(),
            glow::TEXTURE_WRAP_T,
            glow::CLAMP_TO_EDGE as i32,
        );
    }
}

fn load_external_texture(gc: &mut GlContext, preset_id: u64, spec: &TextureSpec) -> Result<GpuTex> {
    let bytes = std::fs::read(&spec.path)
        .with_context(|| format!("read texture {}", spec.path.display()))?;
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| anyhow!("decode PNG header {}: {e}", spec.path.display()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| anyhow!("PNG output size unavailable: {}", spec.path.display()))?;
    let mut buffer = vec![0u8; size];
    let info = reader
        .next_frame(&mut buffer)
        .map_err(|e| anyhow!("decode PNG {}: {e}", spec.path.display()))?;
    buffer.truncate(info.buffer_size());
    let rgba = png_to_rgba(
        &buffer,
        info.color_type,
        info.bit_depth,
        info.width,
        info.height,
    )?;
    let key = format!("slangp:{preset_id}:lut:{}", spec.name);
    let texture = gc.persistent_texture(
        &key,
        info.width as i32,
        info.height as i32,
        1,
        4,
        Dtype::U8,
        spec.linear,
        spec.wrap.gl(),
        &rgba,
    );
    if spec.mipmap {
        set_texture_sampling(gc, texture, spec.linear, spec.wrap, true);
    }
    Ok(texture)
}

fn png_to_rgba(
    data: &[u8],
    color: png::ColorType,
    depth: png::BitDepth,
    width: u32,
    height: u32,
) -> Result<Vec<u8>> {
    if depth != png::BitDepth::Eight {
        bail!("SLANGP LUT PNG must decode to 8-bit channels (got {depth:?})");
    }
    let pixels = width as usize * height as usize;
    let mut out = Vec::with_capacity(pixels * 4);
    match color {
        png::ColorType::Rgba => return Ok(data.to_vec()),
        png::ColorType::Rgb => {
            for rgb in data.chunks_exact(3) {
                out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
            }
        }
        png::ColorType::Grayscale => {
            for &v in data.iter().take(pixels) {
                out.extend_from_slice(&[v, v, v, 255]);
            }
        }
        png::ColorType::GrayscaleAlpha => {
            for ga in data.chunks_exact(2) {
                out.extend_from_slice(&[ga[0], ga[0], ga[0], ga[1]]);
            }
        }
        png::ColorType::Indexed => bail!("indexed PNG LUT was not expanded by decoder"),
    }
    Ok(out)
}

fn feedback_key(preset: u64, pass: usize) -> String {
    format!("slangp:{preset}:feedback:{pass}")
}

fn original_history_key(preset: u64, history: usize) -> String {
    format!("slangp:{preset}:original-history:{history}")
}

fn source_fingerprint(vertex: &str, fragment: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    vertex.hash(&mut hasher);
    fragment.hash(&mut hasher);
    hasher.finish()
}

fn parse_bool(value: &str) -> bool {
    matches!(
        unquote(value).to_ascii_lowercase().as_str(),
        "true" | "1" | "yes" | "on"
    )
}

fn parse_f32(value: &str) -> Option<f32> {
    unquote(value).parse().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    unquote(value).parse().ok()
}

fn parse_usize(value: &str) -> Option<usize> {
    unquote(value).parse().ok()
}

fn unquote(value: &str) -> String {
    let value = value.trim();
    if value.len() >= 2 && value.starts_with('"') && value.ends_with('"') {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preset_bool_and_scale_parser() {
        assert!(parse_bool("true"));
        assert!(parse_bool("\"1\""));
        assert_eq!(ScaleType::parse(Some("viewport")), ScaleType::Viewport);
        assert_eq!(ScaleType::parse(Some("absolute")), ScaleType::Absolute);
    }

    #[test]
    fn slang_layout_set_is_removed_but_binding_kept() {
        let src = "layout(set = 0, binding = 2) uniform sampler2D Source;";
        let rewritten = rewrite_layout_set_binding(src);
        assert!(!rewritten.contains("set = 0"));
        assert!(rewritten.contains("binding = 2"));
    }

    #[test]
    fn split_common_vertex_fragment() {
        let src = "#version 450\n#define X 1\n#pragma stage vertex\nvoid v(){}\n#pragma stage fragment\nvoid f(){}\n";
        let (common, vertex, fragment) = split_stages(src).unwrap();
        assert!(common.contains("#define X"));
        assert!(vertex.contains("void v"));
        assert!(fragment.contains("void f"));
    }

    #[test]
    fn retroarch_dependency_anchor_is_rebased_without_parent_escape() {
        let suffix = dependency_anchor_suffix(Path::new(
            "../../../shaders_slang/crt/shaders/guest/advanced/stock.slang",
        ))
        .unwrap();
        assert_eq!(
            suffix,
            PathBuf::from("shaders_slang/crt/shaders/guest/advanced/stock.slang")
        );
    }
}
