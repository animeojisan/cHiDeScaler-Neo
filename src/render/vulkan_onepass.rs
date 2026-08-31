//! v632 experimental selected-GPU Vulkan one-pass backend.
//!
//! Safety contract:
//! - GPU=Auto never enters this module and does not initialize Vulkan.
//! - The production route is opt-in in v618 (`NEO_VULKAN_PRODUCTION_1PASS=1`).
//! - Normal production display still admits only the already-proven bundled `deint_swa.glsl`.
//! - v629 adds an opt-in candidate-validation mode for AXAA and Intel_CMAA2_lite.
//! - v631 compiles GLSL->SPIR-V only when a persistent runtime key is created/replaced.
//! - v632 adds diagnostic-safe //!PARAM support by compiling current parameter values as constants;
//!   //!TEXTURE and multi-pass remain rejected, and normal production admission is unchanged.
//!   Those candidates require manual visual validation and are never enabled by normal production mode.
//! - The generic structural/compiler compatibility layer remains diagnostic-only for all other shaders.
//! - Unsupported shaders or any Vulkan failure return to the existing OpenGL path.
//! - Vulkan device/pipeline/images/buffers are persistent across frames for the
//!   same selected LUID + shader + dimensions; they are not recreated per frame.
//! - v618 still bridges OpenGL <-> Vulkan through RGBA8 CPU staging. This is a
//!   correctness milestone, not the final zero-copy implementation.

use super::mpv::{ParamTy, PassOffset, UserShader};
use anyhow::{Context, Result, anyhow};
use ash::{Entry, vk};
use std::cell::RefCell;
use std::collections::HashSet;
use std::ffi::{CStr, CString};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct ProductionSteadyProfile {
    pub warmup_frames: u32,
    pub measured_frames: u32,
    pub avg_total_ms: f64,
    pub min_total_ms: f64,
    pub max_total_ms: f64,
    pub avg_host_upload_ms: f64,
    pub avg_submit_wait_ms: f64,
    pub avg_command_record_ms: f64,
    pub avg_queue_submit_ms: f64,
    pub avg_fence_wait_ms: f64,
    pub avg_host_readback_ms: f64,
    pub readback_host_cached: bool,
}

#[derive(Clone, Debug)]
pub struct ProductionOnePassResult {
    pub output_rgba8: Vec<u8>,
    pub gpu_name: String,
    pub elapsed_ms: f64,
    pub host_upload_ms: f64,
    pub submit_wait_ms: f64,
    pub command_record_ms: f64,
    pub queue_submit_ms: f64,
    pub fence_wait_ms: f64,
    pub host_readback_ms: f64,
    pub steady_profile: Option<ProductionSteadyProfile>,
    pub first_frame_verified: bool,
    pub verified_pixels: usize,
    pub tolerance: u8,
    pub validation_mode: &'static str,
    pub visual_check_required: bool,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct RuntimeKey {
    luid: u64,
    width: u32,
    height: u32,
    shader_hash: u64,
}

fn failed_keys() -> &'static Mutex<HashSet<RuntimeKey>> {
    static FAILED: OnceLock<Mutex<HashSet<RuntimeKey>>> = OnceLock::new();
    FAILED.get_or_init(|| Mutex::new(HashSet::new()))
}

fn route_diag_keys() -> &'static Mutex<HashSet<String>> {
    static KEYS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    KEYS.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Write a production-routing breadcrumb once per process. This is deliberately
/// independent of Neo's normal file-logging setting so opt-in diagnostics can
/// explain a safe OpenGL fallback even when ordinary file logging is disabled.
pub fn record_route_once(key: impl Into<String>, line: &str) {
    let key = key.into();
    if route_diag_keys().lock().unwrap().insert(key) {
        log::info!("{line}");
        super::vulkan_gpu::record_probe_result(line);
    }
}

static ABANDON_RUNTIME_ON_PROCESS_EXIT: AtomicBool = AtomicBool::new(false);

thread_local! {
    static RUNTIME: RefCell<Option<VulkanOnePassRuntime>> = const { RefCell::new(None) };
}

/// Application-close contract for the experimental selected-GPU Vulkan route.
/// The runtime lives in render-thread TLS. Calling vendor vkDestroy* functions
/// from a TLS destructor proved capable of keeping the process resident after
/// the GUI had already completed cleanup on AMD. Once application close begins,
/// no Vulkan resource will be reused, so let Windows reclaim those process-owned
/// objects instead of entering driver destruction from TLS. Runtime replacement
/// during an active session still performs the ordinary explicit destruction.
pub fn prepare_runtime_for_process_exit() {
    ABANDON_RUNTIME_ON_PROCESS_EXIT.store(true, Ordering::Release);
    record_route_once(
        "vulkan-runtime-process-exit",
        "vulkan-production-glsl: phase=vulkan-runtime-process-exit cleanup=os-process-reclaim tls-destroy=false",
    );
}

pub fn production_one_pass_requested() -> bool {
    std::env::var("NEO_VULKAN_PRODUCTION_1PASS")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

pub fn candidate_one_pass_validation_requested() -> bool {
    std::env::var("NEO_VULKAN_CANDIDATE_1PASS")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn candidate_test_name(name: &str) -> bool {
    name.eq_ignore_ascii_case("AXAA.glsl") || name.eq_ignore_ascii_case("Intel_CMAA2_lite.glsl")
}

/// Drop a persistent Vulkan runtime when a new session selects another route.
/// Calling this on Auto performs no Vulkan initialization.
pub fn reset_runtime() {
    RUNTIME.with(|slot| {
        *slot.borrow_mut() = None;
    });
}

fn luid_from_vk(bytes: &[u8; vk::LUID_SIZE]) -> u64 {
    u64::from_le_bytes(*bytes)
}

fn device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
        .to_string_lossy()
        .trim()
        .to_string()
}

fn find_memory_type_with_flags(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|index| {
        (type_bits & (1u32 << index)) != 0
            && properties.memory_types[*index as usize]
                .property_flags
                .contains(required)
    })
}

fn shader_hash(shader: &UserShader) -> u64 {
    let mut hasher = DefaultHasher::new();
    shader.path.hash(&mut hasher);
    for pass in &shader.passes {
        pass.code().hash(&mut hasher);
        pass.desc.hash(&mut hasher);
    }
    // v632: PARAM values are compiled into the Vulkan wrapper as constants.
    // Include them in the runtime key so changing a user parameter safely
    // rebuilds the SPIR-V/runtime instead of reusing stale constants.
    for param in &shader.params {
        param.name.hash(&mut hasher);
        param.value.to_bits().hash(&mut hasher);
        std::mem::discriminant(&param.ty).hash(&mut hasher);
    }
    hasher.finish()
}

fn generic_one_pass_eligible(shader: &UserShader) -> Result<&super::mpv::Pass> {
    if shader.passes.len() != 1 {
        return Err(anyhow!("requires exactly one pass"));
    }
    // v632: ordinary //!PARAM values are supported by compiling their current
    // values into the wrapper as typed constants. This is deliberately simpler
    // than adding another descriptor/uniform path while the backend remains
    // experimental. A parameter change changes shader_hash() and rebuilds the
    // runtime once. Embedded //!TEXTURE resources still need real Vulkan image
    // descriptors and therefore remain outside this safe one-pass subset.
    if !shader.textures.is_empty() {
        return Err(anyhow!("//!TEXTURE is not supported yet"));
    }
    if shader.uses_chroma || shader.is_compute || shader.is_post {
        return Err(anyhow!(
            "only ordinary in-chain RGB fragment-style hooks are supported"
        ));
    }
    let pass = &shader.passes[0];
    if pass.compute.is_some()
        || pass.save.is_some()
        || pass.width.is_some()
        || pass.height.is_some()
        || pass.components != 4
        || pass.offset != PassOffset::None
    {
        return Err(anyhow!(
            "requires no COMPUTE/SAVE/resize/OFFSET and four output components"
        ));
    }
    if pass.binds.len() != 1 || !pass.binds[0].eq_ignore_ascii_case("HOOKED") {
        return Err(anyhow!("requires exactly one //!BIND HOOKED"));
    }
    let code = pass.code();
    for unsupported in [
        "HOOKED_raw",
        "texelFetch",
        "textureGrad",
        "textureLod",
        "dFdx",
        "dFdy",
        "fwidth",
        "gl_FragCoord",
        // mpv injects this per-frame symbol. The compute wrapper deliberately
        // does not emulate temporal/random state yet.
        "random",
    ] {
        if code.contains(unsupported) {
            return Err(anyhow!("unsupported compute-wrapper token `{unsupported}`"));
        }
    }
    Ok(pass)
}

pub fn production_shader_admitted(shader: &UserShader) -> bool {
    shader.name().eq_ignore_ascii_case("deint_swa.glsl")
        || (candidate_one_pass_validation_requested() && candidate_test_name(&shader.name()))
}

fn production_one_pass_eligible(shader: &UserShader) -> Result<&super::mpv::Pass> {
    if !production_shader_admitted(shader) {
        return Err(anyhow!(
            "v629 normal production admits only deint_swa.glsl; AXAA/Intel_CMAA2_lite require NEO_VULKAN_CANDIDATE_1PASS=1"
        ));
    }
    generic_one_pass_eligible(shader)
}

#[derive(Clone, Debug)]
pub struct OnePassCompatibility {
    pub compatible: bool,
    pub reason: String,
    pub spirv_words: usize,
}

/// Frontend-only compatibility check used while growing GLSL coverage.
/// This never initializes Vulkan and therefore cannot alter the selected GPU,
/// the stable OpenGL path, or presentation. A shader is marked compatible only
/// if it fits the current ordinary one-pass subset AND Naga accepts the actual
/// wrapper that production would use.
pub fn analyze_one_pass_compatibility(
    shader: &UserShader,
    width: u32,
    height: u32,
) -> OnePassCompatibility {
    let pass = match generic_one_pass_eligible(shader) {
        Ok(pass) => pass,
        Err(error) => {
            return OnePassCompatibility {
                compatible: false,
                reason: error.to_string(),
                spirv_words: 0,
            };
        }
    };
    match compile_to_spirv(shader, pass, width.max(1), height.max(1)) {
        Ok(words) => OnePassCompatibility {
            compatible: true,
            reason: "structural-and-naga-compatible".to_string(),
            spirv_words: words.len(),
        },
        Err(error) => OnePassCompatibility {
            compatible: false,
            reason: format!("naga-compile-failed: {error:#}"),
            spirv_words: 0,
        },
    }
}

pub fn compatibility_scan_requested() -> bool {
    std::env::var("NEO_VULKAN_COMPAT_SCAN")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

fn collect_glsl_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_glsl_files(&path, out);
        } else if path
            .extension()
            .and_then(|v| v.to_str())
            .map(|v| v.eq_ignore_ascii_case("glsl"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
}

fn clean_diag_text(value: &str) -> String {
    value
        .replace('\r', " ")
        .replace('\n', " ")
        .replace('\'', "_")
}

/// Scan bundled GLSL files using exactly the parser + compute wrapper intended
/// for future production admission. This is diagnostic-only: compatible shaders
/// are NOT automatically enabled for display.
pub fn scan_bundled_compatibility(app_dir: &Path) {
    let shader_root = app_dir.join("shaders");
    let mut files = Vec::new();
    collect_glsl_files(&shader_root, &mut files);
    files.sort();

    let mut compatible = 0usize;
    let mut rejected = 0usize;
    let mut load_failed = 0usize;
    for path in &files {
        let relative = path
            .strip_prefix(app_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        match UserShader::load(path.to_string_lossy().as_ref()) {
            Ok(shader) => {
                let result = analyze_one_pass_compatibility(&shader, 448, 252);
                if result.compatible {
                    compatible += 1;
                    let line = format!(
                        "vulkan-compat-scan: result=compatible shader='{}' passes={} spirv_words={} production=not-admitted-yet",
                        clean_diag_text(&relative),
                        shader.passes.len(),
                        result.spirv_words
                    );
                    log::info!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                } else {
                    rejected += 1;
                    let line = format!(
                        "vulkan-compat-scan: result=rejected shader='{}' passes={} reason='{}'",
                        clean_diag_text(&relative),
                        shader.passes.len(),
                        clean_diag_text(&result.reason)
                    );
                    log::debug!("{line}");
                    super::vulkan_gpu::record_probe_result(&line);
                }
            }
            Err(error) => {
                load_failed += 1;
                let line = format!(
                    "vulkan-compat-scan: result=load-failed shader='{}' reason='{}'",
                    clean_diag_text(&relative),
                    clean_diag_text(&error.to_string())
                );
                log::warn!("{line}");
                super::vulkan_gpu::record_probe_result(&line);
            }
        }
    }
    let summary = format!(
        "vulkan-compat-scan: result=summary total={} compatible={} rejected={} load_failed={} param_mode=compile-current-value textures=unsupported production_admitted=1 production_shader='deint_swa.glsl' vulkan_init=false display=OpenGL-unchanged",
        files.len(),
        compatible,
        rejected,
        load_failed
    );
    log::info!("{summary}");
    super::vulkan_gpu::record_probe_result(&summary);
}

fn glsl_float_literal(value: f32) -> String {
    if !value.is_finite() {
        return "0.0".to_string();
    }
    let mut text = format!("{value:?}");
    // GLSL accepts exponent notation, but a plain integer-looking token must
    // carry a decimal point when emitted as a float constant.
    if !text.contains('.') && !text.contains('e') && !text.contains('E') {
        text.push_str(".0");
    }
    text
}

fn parameter_prelude(shader: &UserShader) -> String {
    let mut out = String::new();
    for param in &shader.params {
        let name = &param.name;
        let value = param.value;
        match param.ty {
            ParamTy::Define => {
                if value.fract() == 0.0 && value.abs() < 1e9 {
                    out.push_str(&format!("#define {name} {}\n", value as i64));
                } else {
                    out.push_str(&format!("#define {name} {}\n", glsl_float_literal(value)));
                }
            }
            ParamTy::Int | ParamTy::ConstInt => {
                out.push_str(&format!("const int {name} = {};\n", value.round() as i64));
            }
            ParamTy::Uint | ParamTy::ConstUint => {
                out.push_str(&format!(
                    "const uint {name} = {}u;\n",
                    value.round().max(0.0) as u64
                ));
            }
            ParamTy::Float | ParamTy::ConstFloat => {
                out.push_str(&format!(
                    "const float {name} = {};\n",
                    glsl_float_literal(value)
                ));
            }
        }
    }
    out
}

fn compute_source(shader: &UserShader, pass: &super::mpv::Pass, width: u32, height: u32) -> String {
    let body = pass.code();
    let params = parameter_prelude(shader);
    format!(
        r#"#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(rgba8, set = 0, binding = 0) uniform image2D src_img;
layout(rgba8, set = 0, binding = 1) uniform image2D dst_img;

{params}
const vec2 HOOKED_size = vec2({width}.0, {height}.0);
const vec2 HOOKED_pt = vec2(1.0 / {width}.0, 1.0 / {height}.0);
#define HOOKED_pos ((vec2(gl_GlobalInvocationID.xy) + vec2(0.5)) * HOOKED_pt)
#define HOOKED_tex(pos) imageLoad(src_img, clamp(ivec2((pos) * HOOKED_size), ivec2(0), ivec2({width_minus}, {height_minus})))
#define HOOKED_texOff(off) imageLoad(src_img, clamp(ivec2(gl_GlobalInvocationID.xy) + ivec2(off), ivec2(0), ivec2({width_minus}, {height_minus})))
#define HOOKED_raw src_img

{body}

void main() {{
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x < {width} && p.y < {height}) {{
        imageStore(dst_img, p, clamp(hook(), 0.0, 1.0));
    }}
}}
"#,
        width_minus = width.saturating_sub(1),
        height_minus = height.saturating_sub(1),
    )
}

fn compile_to_spirv(
    shader: &UserShader,
    pass: &super::mpv::Pass,
    width: u32,
    height: u32,
) -> Result<Vec<u32>> {
    let source = compute_source(shader, pass, width, height);
    let mut frontend = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options::from(naga::ShaderStage::Compute);
    let module = frontend
        .parse(&options, &source)
        .map_err(|error| anyhow!("Naga production user-GLSL parse failed: {error:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|error| anyhow!("Naga production user-GLSL validation failed: {error:?}"))?;
    let spv_options = naga::back::spv::Options {
        lang_version: (1, 3),
        ..Default::default()
    };
    naga::back::spv::write_vec(&module, &info, &spv_options, None)
        .map_err(|error| anyhow!("Naga production SPIR-V generation failed: {error:?}"))
}

fn deint_swa_expected(input: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut out = vec![0u8; input.len()];
    let w = width as usize;
    let h = height as usize;
    for y in 0..h {
        let ya = y.saturating_sub(1);
        let yb = (y + 1).min(h - 1);
        for x in 0..w {
            for c in 0..4 {
                let cur = input[(y * w + x) * 4 + c] as f32;
                let above = input[(ya * w + x) * 4 + c] as f32;
                let below = input[(yb * w + x) * 4 + c] as f32;
                let value = cur * 0.5 + above * 0.25 + below * 0.25;
                out[(y * w + x) * 4 + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

fn record_onepass_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    initialized: bool,
    upload_buffer: vk::Buffer,
    readback_buffer: vk::Buffer,
    src_image: vk::Image,
    dst_image: vk::Image,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    width: u32,
    height: u32,
    byte_count: usize,
) -> Result<()> {
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);
    let layers = vk::ImageSubresourceLayers::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);
    let copy = vk::BufferImageCopy::default()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(layers)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(extent);
    let byte_size = byte_count as vk::DeviceSize;
    // No ONE_TIME_SUBMIT flag: the steady command buffer is intentionally
    // re-submitted only after the prior fence has completed.
    let begin = vk::CommandBufferBeginInfo::default();

    unsafe {
        device
            .begin_command_buffer(command_buffer, &begin)
            .context("vkBeginCommandBuffer failed")?;
        let upload_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(upload_buffer)
            .offset(0)
            .size(byte_size)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &upload_barrier,
            &[],
        );

        let src_old_layout = if initialized {
            vk::ImageLayout::GENERAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let src_access = if initialized {
            vk::AccessFlags::SHADER_READ
        } else {
            vk::AccessFlags::empty()
        };
        let src_stage = if initialized {
            vk::PipelineStageFlags::COMPUTE_SHADER
        } else {
            vk::PipelineStageFlags::TOP_OF_PIPE
        };
        let src_to_transfer = [vk::ImageMemoryBarrier::default()
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(src_old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src_image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &src_to_transfer,
        );
        device.cmd_copy_buffer_to_image(
            command_buffer,
            upload_buffer,
            src_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );

        let src_to_general = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(src_image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &src_to_general,
        );

        let dst_old_layout = if initialized {
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let dst_access = if initialized {
            vk::AccessFlags::TRANSFER_READ
        } else {
            vk::AccessFlags::empty()
        };
        let dst_stage = if initialized {
            vk::PipelineStageFlags::TRANSFER
        } else {
            vk::PipelineStageFlags::TOP_OF_PIPE
        };
        let dst_to_general = [vk::ImageMemoryBarrier::default()
            .src_access_mask(dst_access)
            .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
            .old_layout(dst_old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(dst_image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            dst_stage,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &dst_to_general,
        );

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_dispatch(command_buffer, width.div_ceil(8), height.div_ceil(8), 1);

        let dst_to_transfer = [vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(dst_image)
            .subresource_range(range)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &dst_to_transfer,
        );
        device.cmd_copy_image_to_buffer(
            command_buffer,
            dst_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            readback_buffer,
            &[copy],
        );
        let readback_barrier = [vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(readback_buffer)
            .offset(0)
            .size(byte_size)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &readback_barrier,
            &[],
        );
        device
            .end_command_buffer(command_buffer)
            .context("vkEndCommandBuffer failed")?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FirstFrameValidation {
    DeintCpuExact,
    CandidateVisual,
}

struct VulkanOnePassRuntime {
    _entry: Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    key: RuntimeKey,
    gpu_name: String,
    byte_count: usize,
    upload_buffer: vk::Buffer,
    upload_memory: vk::DeviceMemory,
    upload_ptr: *mut u8,
    readback_buffer: vk::Buffer,
    readback_memory: vk::DeviceMemory,
    readback_ptr: *mut u8,
    readback_host_cached: bool,
    src_image: vk::Image,
    src_memory: vk::DeviceMemory,
    src_view: vk::ImageView,
    dst_image: vk::Image,
    dst_memory: vk::DeviceMemory,
    dst_view: vk::ImageView,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    initial_command_buffer: vk::CommandBuffer,
    steady_command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    /// Set only after a successful queue submit and cleared after the bounded
    /// fence wait succeeds. Drop must never block indefinitely on a driver.
    submission_in_flight: bool,
    image_initialized: bool,
    verified_once: bool,
    validation_mode: FirstFrameValidation,
    profile_seen: u32,
    profile_samples: u32,
    profile_sum_total_ms: f64,
    profile_sum_host_upload_ms: f64,
    profile_sum_submit_wait_ms: f64,
    profile_sum_command_record_ms: f64,
    profile_sum_queue_submit_ms: f64,
    profile_sum_fence_wait_ms: f64,
    profile_sum_host_readback_ms: f64,
    profile_min_total_ms: f64,
    profile_max_total_ms: f64,
    profile_reported: bool,
}

impl VulkanOnePassRuntime {
    fn create(
        key: RuntimeKey,
        spirv: &[u32],
        validation_mode: FirstFrameValidation,
    ) -> Result<Self> {
        let entry = unsafe { Entry::load() }.context("Vulkan loader could not be loaded")?;
        let app_name = CString::new("cHiDeScaler-Neo Vulkan OnePass v629")?;
        let engine_name = CString::new("cHiDeScaler-Neo")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(1)
            .engine_name(&engine_name)
            .engine_version(1)
            .api_version(vk::API_VERSION_1_1);
        let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&create_info, None) }
            .context("vkCreateInstance failed for v615 production one-pass")?;

        let create_result = (|| -> Result<Self> {
            let physical_devices = unsafe { instance.enumerate_physical_devices() }
                .context("vkEnumeratePhysicalDevices failed")?;
            let mut selected = None;
            let mut seen = Vec::new();
            for physical_device in physical_devices {
                let mut id = vk::PhysicalDeviceIDProperties::default();
                let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
                unsafe {
                    instance.get_physical_device_properties2(physical_device, &mut properties2)
                };
                let properties = properties2.properties;
                let name = device_name(&properties);
                let luid =
                    (id.device_luid_valid == vk::TRUE).then(|| luid_from_vk(&id.device_luid));
                seen.push(format!(
                    "{}:{}",
                    name,
                    luid.map(|v| format!("{v:016x}"))
                        .unwrap_or_else(|| "no-luid".into())
                ));
                if luid != Some(key.luid) {
                    continue;
                }
                let queue_families = unsafe {
                    instance.get_physical_device_queue_family_properties(physical_device)
                };
                let queue_family_index = queue_families
                    .iter()
                    .enumerate()
                    .find(|(_, family)| {
                        family.queue_count > 0
                            && family.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .map(|(index, _)| index as u32)
                    .ok_or_else(|| anyhow!("selected Vulkan GPU has no compute queue"))?;
                selected = Some((physical_device, properties, name, queue_family_index));
                break;
            }
            let (physical_device, _properties, gpu_name, queue_family_index) = selected.ok_or_else(|| {
                anyhow!(
                    "no Vulkan physical device matched selected DXGI LUID {:016x}; enumerated=[{}]",
                    key.luid,
                    seen.join(", ")
                )
            })?;

            let priorities = [1.0f32];
            let queue_infos = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)];
            let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_infos);
            let device = unsafe { instance.create_device(physical_device, &device_info, None) }
                .context("vkCreateDevice failed for v615 production one-pass")?;
            let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
            let memory_properties =
                unsafe { instance.get_physical_device_memory_properties(physical_device) };
            let byte_count = (key.width as usize)
                .checked_mul(key.height as usize)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| anyhow!("v615 byte count overflow {}x{}", key.width, key.height))?;
            let byte_size = byte_count as vk::DeviceSize;

            let host_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let upload_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let upload_buffer = unsafe { device.create_buffer(&upload_info, None) }
                .context("vkCreateBuffer upload failed")?;
            let upload_req = unsafe { device.get_buffer_memory_requirements(upload_buffer) };
            let upload_type = find_memory_type_with_flags(
                &memory_properties,
                upload_req.memory_type_bits,
                host_flags,
            )
            .ok_or_else(|| anyhow!("no HOST_VISIBLE|HOST_COHERENT upload memory"))?;
            let upload_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(upload_req.size)
                .memory_type_index(upload_type);
            let upload_memory = unsafe { device.allocate_memory(&upload_alloc, None) }
                .context("vkAllocateMemory upload failed")?;
            unsafe { device.bind_buffer_memory(upload_buffer, upload_memory, 0) }
                .context("vkBindBufferMemory upload failed")?;
            let upload_ptr = unsafe {
                device.map_memory(upload_memory, 0, byte_size, vk::MemoryMapFlags::empty())
            }
            .context("vkMapMemory upload failed")?
            .cast::<u8>();

            let readback_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let readback_buffer = unsafe { device.create_buffer(&readback_info, None) }
                .context("vkCreateBuffer readback failed")?;
            let readback_req = unsafe { device.get_buffer_memory_requirements(readback_buffer) };
            // Readback is read heavily by the CPU every frame. Some AMD drivers expose
            // both a write-combined HOST_VISIBLE|HOST_COHERENT type and a CPU-cached
            // HOST_VISIBLE|HOST_COHERENT|HOST_CACHED type. Picking the first coherent
            // type made the mapped readback memcpy dominate the steady-state profile.
            // Prefer HOST_CACHED for readback, but retain the old coherent type as a
            // compatibility fallback on devices that do not expose a cached host heap.
            let cached_readback_flags = host_flags | vk::MemoryPropertyFlags::HOST_CACHED;
            let cached_readback_type = find_memory_type_with_flags(
                &memory_properties,
                readback_req.memory_type_bits,
                cached_readback_flags,
            );
            let readback_host_cached = cached_readback_type.is_some();
            let readback_type = cached_readback_type
                .or_else(|| {
                    find_memory_type_with_flags(
                        &memory_properties,
                        readback_req.memory_type_bits,
                        host_flags,
                    )
                })
                .ok_or_else(|| anyhow!("no HOST_VISIBLE|HOST_COHERENT readback memory"))?;
            let readback_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(readback_req.size)
                .memory_type_index(readback_type);
            let readback_memory = unsafe { device.allocate_memory(&readback_alloc, None) }
                .context("vkAllocateMemory readback failed")?;
            unsafe { device.bind_buffer_memory(readback_buffer, readback_memory, 0) }
                .context("vkBindBufferMemory readback failed")?;
            let readback_ptr = unsafe {
                device.map_memory(readback_memory, 0, byte_size, vk::MemoryMapFlags::empty())
            }
            .context("vkMapMemory readback failed")?
            .cast::<u8>();

            let extent = vk::Extent3D {
                width: key.width,
                height: key.height,
                depth: 1,
            };
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::STORAGE
                        | vk::ImageUsageFlags::TRANSFER_SRC
                        | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);

            let src_image = unsafe { device.create_image(&image_info, None) }
                .context("vkCreateImage src failed")?;
            let src_req = unsafe { device.get_image_memory_requirements(src_image) };
            let src_type = find_memory_type_with_flags(
                &memory_properties,
                src_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type_with_flags(
                    &memory_properties,
                    src_req.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or_else(|| anyhow!("no compatible src image memory"))?;
            let src_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(src_req.size)
                .memory_type_index(src_type);
            let src_memory = unsafe { device.allocate_memory(&src_alloc, None) }
                .context("vkAllocateMemory src image failed")?;
            unsafe { device.bind_image_memory(src_image, src_memory, 0) }
                .context("vkBindImageMemory src failed")?;

            let dst_image = unsafe { device.create_image(&image_info, None) }
                .context("vkCreateImage dst failed")?;
            let dst_req = unsafe { device.get_image_memory_requirements(dst_image) };
            let dst_type = find_memory_type_with_flags(
                &memory_properties,
                dst_req.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .or_else(|| {
                find_memory_type_with_flags(
                    &memory_properties,
                    dst_req.memory_type_bits,
                    vk::MemoryPropertyFlags::empty(),
                )
            })
            .ok_or_else(|| anyhow!("no compatible dst image memory"))?;
            let dst_alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(dst_req.size)
                .memory_type_index(dst_type);
            let dst_memory = unsafe { device.allocate_memory(&dst_alloc, None) }
                .context("vkAllocateMemory dst image failed")?;
            unsafe { device.bind_image_memory(dst_image, dst_memory, 0) }
                .context("vkBindImageMemory dst failed")?;

            let range = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1);
            let src_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(src_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(range),
                    None,
                )
            }
            .context("vkCreateImageView src failed")?;
            let dst_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(dst_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(vk::Format::R8G8B8A8_UNORM)
                        .subresource_range(range),
                    None,
                )
            }
            .context("vkCreateImageView dst failed")?;

            let bindings = [
                vk::DescriptorSetLayoutBinding::default()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
                vk::DescriptorSetLayoutBinding::default()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE),
            ];
            let descriptor_set_layout = unsafe {
                device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
            }
            .context("vkCreateDescriptorSetLayout failed")?;
            let set_layouts = [descriptor_set_layout];
            let pipeline_layout = unsafe {
                device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
                    None,
                )
            }
            .context("vkCreatePipelineLayout failed")?;
            let shader_module = unsafe {
                device
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(spirv), None)
            }
            .context("vkCreateShaderModule failed")?;
            let entry_name = CString::new("main")?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader_module)
                .name(&entry_name);
            let infos = [vk::ComputePipelineCreateInfo::default()
                .stage(stage)
                .layout(pipeline_layout)];
            let pipelines =
                unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &infos, None) }
                    .map_err(|(partial, error)| {
                        unsafe {
                            for pipeline in partial {
                                device.destroy_pipeline(pipeline, None);
                            }
                        }
                        anyhow!("vkCreateComputePipelines failed: {error:?}")
                    })?;
            let pipeline = pipelines[0];

            let pool_sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                descriptor_count: 2,
            }];
            let descriptor_pool = unsafe {
                device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&pool_sizes),
                    None,
                )
            }
            .context("vkCreateDescriptorPool failed")?;
            let descriptor_sets = unsafe {
                device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&set_layouts),
                )
            }
            .context("vkAllocateDescriptorSets failed")?;
            let descriptor_set = descriptor_sets[0];
            let src_info = [vk::DescriptorImageInfo::default()
                .image_view(src_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let dst_info = [vk::DescriptorImageInfo::default()
                .image_view(dst_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&src_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&dst_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]) };

            let command_pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(queue_family_index),
                    None,
                )
            }
            .context("vkCreateCommandPool failed")?;
            let command_buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(command_pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(2),
                )
            }
            .context("vkAllocateCommandBuffers failed")?;
            let initial_command_buffer = command_buffers[0];
            let steady_command_buffer = command_buffers[1];

            // v628: retain the v626 optimization: record both legal image-layout variants once. The first
            // submission starts from UNDEFINED layouts; every later submission
            // starts from the layouts left by the previous completed frame.
            // The steady command buffer is therefore reusable after each bounded
            // fence completion without resetting/re-recording the command pool.
            record_onepass_commands(
                &device,
                initial_command_buffer,
                false,
                upload_buffer,
                readback_buffer,
                src_image,
                dst_image,
                pipeline,
                pipeline_layout,
                descriptor_set,
                key.width,
                key.height,
                byte_count,
            )?;
            record_onepass_commands(
                &device,
                steady_command_buffer,
                true,
                upload_buffer,
                readback_buffer,
                src_image,
                dst_image,
                pipeline,
                pipeline_layout,
                descriptor_set,
                key.width,
                key.height,
                byte_count,
            )?;
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .context("vkCreateFence failed")?;

            Ok(Self {
                _entry: entry,
                instance,
                device,
                queue,
                key,
                gpu_name,
                byte_count,
                upload_buffer,
                upload_memory,
                upload_ptr,
                readback_buffer,
                readback_memory,
                readback_ptr,
                readback_host_cached,
                src_image,
                src_memory,
                src_view,
                dst_image,
                dst_memory,
                dst_view,
                descriptor_set_layout,
                descriptor_pool,
                pipeline_layout,
                shader_module,
                pipeline,
                command_pool,
                initial_command_buffer,
                steady_command_buffer,
                fence,
                submission_in_flight: false,
                image_initialized: false,
                verified_once: false,
                validation_mode,
                profile_seen: 0,
                profile_samples: 0,
                profile_sum_total_ms: 0.0,
                profile_sum_host_upload_ms: 0.0,
                profile_sum_submit_wait_ms: 0.0,
                profile_sum_command_record_ms: 0.0,
                profile_sum_queue_submit_ms: 0.0,
                profile_sum_fence_wait_ms: 0.0,
                profile_sum_host_readback_ms: 0.0,
                profile_min_total_ms: f64::INFINITY,
                profile_max_total_ms: 0.0,
                profile_reported: false,
            })
        })();

        // On a rare mid-construction failure the process keeps the loader/instance
        // alive until exit rather than risking destruction ahead of partially
        // created child handles. The failed key is disabled immediately by the
        // caller, so this cannot accumulate frame-by-frame.
        create_result
    }

    fn process(&mut self, input: &[u8]) -> Result<ProductionOnePassResult> {
        if input.len() != self.byte_count {
            return Err(anyhow!(
                "v629 runtime input byte mismatch: got {}, expected {}",
                input.len(),
                self.byte_count
            ));
        }
        let width = self.key.width;
        let height = self.key.height;
        let started = Instant::now();
        let host_upload_started = Instant::now();
        unsafe {
            std::ptr::copy_nonoverlapping(input.as_ptr(), self.upload_ptr, input.len());
        }
        let host_upload_ms = host_upload_started.elapsed().as_secs_f64() * 1000.0;
        let submit_wait_started = Instant::now();
        // v628: command buffers are pre-recorded in create(). Keep this metric
        // for apples-to-apples profiling; it now measures only the negligible
        // selection of the initial versus steady executable command buffer.
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .context("vkResetFences failed")?;
        }
        let command_record_started = Instant::now();
        let command_buffer = if self.image_initialized {
            self.steady_command_buffer
        } else {
            self.initial_command_buffer
        };
        let command_record_ms = command_record_started.elapsed().as_secs_f64() * 1000.0;

        let buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        let queue_submit_started = Instant::now();
        unsafe {
            self.device
                .queue_submit(self.queue, &submits, self.fence)
                .context("vkQueueSubmit failed")?;
        }
        let queue_submit_ms = queue_submit_started.elapsed().as_secs_f64() * 1000.0;
        self.submission_in_flight = true;

        let fence_wait_started = Instant::now();
        let fence_wait = unsafe {
            self.device
                .wait_for_fences(&[self.fence], true, 1_000_000_000)
        };
        let fence_wait_ms = fence_wait_started.elapsed().as_secs_f64() * 1000.0;
        if fence_wait.is_ok() {
            self.submission_in_flight = false;
        }
        fence_wait.context("v629 Vulkan one-pass timed out")?;
        let submit_wait_ms = submit_wait_started.elapsed().as_secs_f64() * 1000.0;
        self.image_initialized = true;
        let host_readback_started = Instant::now();
        let output =
            unsafe { std::slice::from_raw_parts(self.readback_ptr, self.byte_count) }.to_vec();
        let host_readback_ms = host_readback_started.elapsed().as_secs_f64() * 1000.0;

        let mut first_frame_verified = false;
        let mut verified_pixels = 0usize;
        let mut tolerance = 0u8;
        if !self.verified_once {
            match self.validation_mode {
                FirstFrameValidation::DeintCpuExact => {
                    tolerance = 1;
                    let expected = deint_swa_expected(input, width, height);
                    let pixels = (width as usize) * (height as usize);
                    for index in 0..pixels {
                        let base = index * 4;
                        if (0..4)
                            .all(|c| output[base + c].abs_diff(expected[base + c]) <= tolerance)
                        {
                            verified_pixels += 1;
                        }
                    }
                    if verified_pixels != pixels {
                        return Err(anyhow!(
                            "v629 deint first-frame verification failed: verified={verified_pixels}/{pixels} tolerance={tolerance}"
                        ));
                    }
                }
                FirstFrameValidation::CandidateVisual => {
                    // AXAA/CMAA2 do not have an independent CPU implementation in Neo.
                    // A successful Vulkan execution proves routing/compiler/runtime health,
                    // but output correctness must be visually compared before formal admission.
                    verified_pixels = 0;
                }
            }
            self.verified_once = true;
            first_frame_verified = true;
        }

        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;

        // v623: profile only steady-state frames. The first verified frame is
        // intentionally excluded, then ten additional frames are discarded as
        // warmup. The following sixty frames are averaged without any file I/O
        // in the hot loop; exactly one summary is returned to the caller.
        let mut steady_profile = None;
        if self.verified_once && !first_frame_verified && !self.profile_reported {
            const WARMUP_FRAMES: u32 = 10;
            const MEASURED_FRAMES: u32 = 60;
            self.profile_seen = self.profile_seen.saturating_add(1);
            if self.profile_seen > WARMUP_FRAMES && self.profile_samples < MEASURED_FRAMES {
                self.profile_samples += 1;
                self.profile_sum_total_ms += elapsed_ms;
                self.profile_sum_host_upload_ms += host_upload_ms;
                self.profile_sum_submit_wait_ms += submit_wait_ms;
                self.profile_sum_command_record_ms += command_record_ms;
                self.profile_sum_queue_submit_ms += queue_submit_ms;
                self.profile_sum_fence_wait_ms += fence_wait_ms;
                self.profile_sum_host_readback_ms += host_readback_ms;
                self.profile_min_total_ms = self.profile_min_total_ms.min(elapsed_ms);
                self.profile_max_total_ms = self.profile_max_total_ms.max(elapsed_ms);
                if self.profile_samples == MEASURED_FRAMES {
                    let n = self.profile_samples as f64;
                    steady_profile = Some(ProductionSteadyProfile {
                        warmup_frames: WARMUP_FRAMES,
                        measured_frames: self.profile_samples,
                        avg_total_ms: self.profile_sum_total_ms / n,
                        min_total_ms: self.profile_min_total_ms,
                        max_total_ms: self.profile_max_total_ms,
                        avg_host_upload_ms: self.profile_sum_host_upload_ms / n,
                        avg_submit_wait_ms: self.profile_sum_submit_wait_ms / n,
                        avg_command_record_ms: self.profile_sum_command_record_ms / n,
                        avg_queue_submit_ms: self.profile_sum_queue_submit_ms / n,
                        avg_fence_wait_ms: self.profile_sum_fence_wait_ms / n,
                        avg_host_readback_ms: self.profile_sum_host_readback_ms / n,
                        readback_host_cached: self.readback_host_cached,
                    });
                    self.profile_reported = true;
                }
            }
        }

        Ok(ProductionOnePassResult {
            output_rgba8: output,
            gpu_name: self.gpu_name.clone(),
            elapsed_ms,
            host_upload_ms,
            submit_wait_ms,
            command_record_ms,
            queue_submit_ms,
            fence_wait_ms,
            host_readback_ms,
            steady_profile,
            first_frame_verified,
            verified_pixels,
            tolerance,
            validation_mode: match self.validation_mode {
                FirstFrameValidation::DeintCpuExact => "cpu-exact",
                FirstFrameValidation::CandidateVisual => "execution-only",
            },
            visual_check_required: self.validation_mode == FirstFrameValidation::CandidateVisual,
        })
    }
}

impl Drop for VulkanOnePassRuntime {
    fn drop(&mut self) {
        if ABANDON_RUNTIME_ON_PROCESS_EXIT.load(Ordering::Acquire) {
            // Do not call into a vendor Vulkan destructor from render-thread TLS
            // after application close has started. Ash handles are plain wrappers;
            // returning here skips our explicit vkDestroy*/vkFree* calls and the
            // terminating Windows process reclaims the underlying driver objects.
            log::info!(
                "vulkan-production-cleanup: result=os-process-reclaim luid={:016x} reason=application-exit",
                self.key.luid
            );
            return;
        }
        // A successful process() call has already completed a bounded fence wait,
        // so vkDeviceWaitIdle is redundant here. More importantly, an unbounded
        // driver wait during Drop can keep Neo resident after its GUI is closed.
        // If the bounded fence wait ever failed, leave the still-live Vulkan
        // objects for OS process teardown instead of issuing unsafe destruction.
        if self.submission_in_flight {
            log::warn!(
                "vulkan-production-cleanup: result=deferred-to-process-exit reason=submission-still-in-flight luid={:016x}",
                self.key.luid
            );
            return;
        }
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader_module, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_image_view(self.src_view, None);
            self.device.destroy_image_view(self.dst_view, None);
            self.device.destroy_image(self.src_image, None);
            self.device.destroy_image(self.dst_image, None);
            self.device.free_memory(self.src_memory, None);
            self.device.free_memory(self.dst_memory, None);
            self.device.unmap_memory(self.upload_memory);
            self.device.unmap_memory(self.readback_memory);
            self.device.destroy_buffer(self.upload_buffer, None);
            self.device.destroy_buffer(self.readback_buffer, None);
            self.device.free_memory(self.upload_memory, None);
            self.device.free_memory(self.readback_memory, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Attempt the v618 production one-pass path. `Ok(None)` is an intentional,
/// safe OpenGL fallback (feature disabled, unsupported shader, or a key that
/// already failed once). `Err` is returned only on the first failure for a key
/// so the caller can log it once; the key is then disabled for the session.
pub fn process_rgba8(
    requested_luid: u64,
    shader: &UserShader,
    width: u32,
    height: u32,
    input: &[u8],
) -> Result<Option<ProductionOnePassResult>> {
    if !production_one_pass_requested() {
        return Ok(None);
    }
    let pass = match production_one_pass_eligible(shader) {
        Ok(pass) => pass,
        Err(_) => return Ok(None),
    };
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let expected_len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(4))
        .ok_or_else(|| anyhow!("v618 input size overflow"))?;
    if input.len() != expected_len {
        return Err(anyhow!(
            "v618 input bytes={} expected={expected_len}",
            input.len()
        ));
    }
    let key = RuntimeKey {
        luid: requested_luid,
        width,
        height,
        shader_hash: shader_hash(shader),
    };
    if failed_keys().lock().unwrap().contains(&key) {
        return Ok(None);
    }
    let validation_mode = if shader.name().eq_ignore_ascii_case("deint_swa.glsl") {
        FirstFrameValidation::DeintCpuExact
    } else {
        FirstFrameValidation::CandidateVisual
    };

    let result = RUNTIME.with(|slot| -> Result<ProductionOnePassResult> {
        let mut slot = slot.borrow_mut();
        let replace = slot.as_ref().map(|runtime| runtime.key != key).unwrap_or(true);
        if replace {
            // v631: compile the GLSL wrapper only when the persistent runtime
            // actually has to be created/replaced. v630 compiled the identical
            // shader to SPIR-V on every frame before checking the runtime key,
            // which wasted CPU time outside ProductionOnePassResult::elapsed_ms.
            let compile_started = Instant::now();
            let spirv = compile_to_spirv(shader, pass, width, height)?;
            let compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            let compile_line = format!(
                "vulkan-production-glsl: phase=runtime-compile shader='{}' requested_luid={:016x} size={}x{} spirv_words={} compile_ms={:.3} cache=runtime-key-once",
                shader.name(),
                requested_luid,
                width,
                height,
                spirv.len(),
                compile_ms,
            );
            log::info!("{compile_line}");
            super::vulkan_gpu::record_probe_result(&compile_line);
            *slot = Some(VulkanOnePassRuntime::create(key.clone(), &spirv, validation_mode)?);
        }
        slot.as_mut().unwrap().process(input)
    });
    match result {
        Ok(result) => Ok(Some(result)),
        Err(error) => {
            failed_keys().lock().unwrap().insert(key);
            RUNTIME.with(|slot| *slot.borrow_mut() = None);
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deint_swa_compiles_for_vulkan_compute_wrapper() {
        let source = include_str!("../../shaders/Deinterlace/deint_swa.glsl");
        let shader = UserShader::parse("deint_swa.glsl", source);
        let pass = production_one_pass_eligible(&shader).expect("deint_swa should be eligible");
        let words = compile_to_spirv(&shader, pass, 448, 252)
            .expect("Vulkan production wrapper should compile");
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x0723_0203);
    }

    #[test]
    fn generic_compatibility_is_not_bound_to_the_deint_filename() {
        let source = include_str!("../../shaders/Deinterlace/deint_swa.glsl");
        let shader = UserShader::parse("renamed-safe-onepass.glsl", source);
        let compatibility = analyze_one_pass_compatibility(&shader, 448, 252);
        assert!(compatibility.compatible, "{}", compatibility.reason);
        assert!(compatibility.spirv_words > 0);
        assert!(production_one_pass_eligible(&shader).is_err());
    }

    #[test]
    fn temporal_random_symbol_stays_out_of_the_generic_subset() {
        let source = include_str!("../../shaders/Filmgrain/filmgrain.glsl");
        let shader = UserShader::parse("filmgrain.glsl", source);
        let compatibility = analyze_one_pass_compatibility(&shader, 448, 252);
        assert!(!compatibility.compatible);
        assert!(compatibility.reason.contains("random"));
    }
}
