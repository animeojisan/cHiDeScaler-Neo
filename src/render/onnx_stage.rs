//! ONNX super-resolution stage via onnxruntime + DirectML (any DX12 GPU).
//!
//! onnxruntime.dll is loaded dynamically at runtime (`init_onnx`), so the app
//! itself stays tiny and runs fully without it — ONNX filters simply report
//! unavailable. Any NCHW image model (Compact/ESRGAN/CUGAN...) works drop-in;
//! fp16 vs fp32 is detected from the model input.

use anyhow::{Result, anyhow};
use half::f16;
use ort::{
    AsPointer,
    memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType},
    session::{IoBinding, RunOptions, Session, builder::GraphOptimizationLevel},
    value::{
        DynValue, PrimitiveTensorElementType, Shape, Tensor, TensorRef, TensorRefMut,
        TensorValueType,
    },
};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CString, c_void};
use std::mem::ManuallyDrop;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Mutex, OnceLock, Weak,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::{Duration, Instant, SystemTime};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, LUID},
        Graphics::{
            Direct3D12::{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, D3D12_FENCE_FLAG_NONE,
                D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_BARRIER, D3D12_RESOURCE_BARRIER_0,
                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_BARRIER_FLAG_NONE,
                D3D12_RESOURCE_BARRIER_TYPE_TRANSITION, D3D12_RESOURCE_DESC,
                D3D12_RESOURCE_DIMENSION_BUFFER, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATES, D3D12_RESOURCE_TRANSITION_BARRIER,
                D3D12_TEXTURE_LAYOUT_ROW_MAJOR, ID3D12CommandAllocator, ID3D12CommandList,
                ID3D12CommandQueue, ID3D12Device, ID3D12Fence, ID3D12GraphicsCommandList,
                ID3D12Resource,
            },
            Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        },
        System::Threading::{
            CreateEventW, INFINITE, OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
            WaitForSingleObject,
        },
    },
    core::{Interface, PCWSTR},
};

use crate::core::config::OnnxBackendPreference;
use crate::render::{
    cuda_interop::{CudaSharedBuffer, cuda_shared_stats},
    gl::{GlContext, GpuTex, InterpAuxPlane},
    onnx_accel::{
        Decision as NeoDecision, NeoAccelState, analyze_local_image_model, compare_output_patch,
        crop_rgba8, extract_output_patch, write_output_patch,
    },
};

static ORT_READY: OnceLock<std::result::Result<(), String>> = OnceLock::new();
static DIRECT_OUTPUT_KEY: AtomicU64 = AtomicU64::new(1);
static NEO_ACCEL_KEY: AtomicU64 = AtomicU64::new(1);
static TENSORRT_PROFILE_KEY: AtomicU64 = AtomicU64::new(1);
static TENSORRT_ACTIVE_CACHE_DIRS: OnceLock<std::sync::Mutex<HashMap<PathBuf, usize>>> =
    OnceLock::new();
static TENSORRT_CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ONNX_CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ACTIVE_ONNX_RUN_OPTIONS: OnceLock<Mutex<Vec<Weak<CancelableRunOptions>>>> = OnceLock::new();
static INTERP_GPU_INPUT_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_GPU_OUTPUT_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_FRAME_READBACKS: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_FRAME_UPLOADS: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_PACK_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_OUTPUT_CONVERSIONS: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_FALLBACK_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_RESIDENCY_VIOLATIONS: AtomicU64 = AtomicU64::new(0);
static DML_SHARED_ACTIVE_ALLOCATIONS: AtomicU64 = AtomicU64::new(0);
static DML_SHARED_ACTIVE_BYTES: AtomicU64 = AtomicU64::new(0);
static DML_SHARED_TOTAL_CREATED: AtomicU64 = AtomicU64::new(0);
static DML_SHARED_TOTAL_FREED: AtomicU64 = AtomicU64::new(0);
static DML_SHARED_RELEASE_FAILURES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Default)]
pub struct DmlSharedStats {
    pub active_allocations: u64,
    pub active_bytes: u64,
    pub total_created: u64,
    pub total_freed: u64,
    pub release_failures: u64,
}

pub fn dml_shared_stats() -> DmlSharedStats {
    DmlSharedStats {
        active_allocations: DML_SHARED_ACTIVE_ALLOCATIONS.load(Ordering::Relaxed),
        active_bytes: DML_SHARED_ACTIVE_BYTES.load(Ordering::Relaxed),
        total_created: DML_SHARED_TOTAL_CREATED.load(Ordering::Relaxed),
        total_freed: DML_SHARED_TOTAL_FREED.load(Ordering::Relaxed),
        release_failures: DML_SHARED_RELEASE_FAILURES.load(Ordering::Relaxed),
    }
}

fn dml_shared_created(bytes: u64) {
    DML_SHARED_ACTIVE_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    DML_SHARED_ACTIVE_BYTES.fetch_add(bytes, Ordering::Relaxed);
    DML_SHARED_TOTAL_CREATED.fetch_add(1, Ordering::Relaxed);
}

fn dml_shared_released(bytes: u64, released: bool) {
    if released {
        DML_SHARED_ACTIVE_ALLOCATIONS.fetch_sub(1, Ordering::Relaxed);
        DML_SHARED_ACTIVE_BYTES.fetch_sub(bytes, Ordering::Relaxed);
        DML_SHARED_TOTAL_FREED.fetch_add(1, Ordering::Relaxed);
    } else {
        DML_SHARED_RELEASE_FAILURES.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InterpResidencySnapshot {
    pub gpu_input_frames: u64,
    pub gpu_output_frames: u64,
    pub cpu_fallback_frames: u64,
    pub cpu_frame_readbacks: u64,
    pub cpu_frame_uploads: u64,
    pub cpu_pack_frames: u64,
    pub cpu_output_conversions: u64,
    pub violations: u64,
}

pub fn interp_residency_snapshot() -> InterpResidencySnapshot {
    InterpResidencySnapshot {
        gpu_input_frames: INTERP_GPU_INPUT_FRAMES.load(Ordering::Relaxed),
        gpu_output_frames: INTERP_GPU_OUTPUT_FRAMES.load(Ordering::Relaxed),
        cpu_fallback_frames: INTERP_CPU_FALLBACK_FRAMES.load(Ordering::Relaxed),
        cpu_frame_readbacks: INTERP_CPU_FRAME_READBACKS.load(Ordering::Relaxed),
        cpu_frame_uploads: INTERP_CPU_FRAME_UPLOADS.load(Ordering::Relaxed),
        cpu_pack_frames: INTERP_CPU_PACK_FRAMES.load(Ordering::Relaxed),
        cpu_output_conversions: INTERP_CPU_OUTPUT_CONVERSIONS.load(Ordering::Relaxed),
        violations: INTERP_RESIDENCY_VIOLATIONS.load(Ordering::Relaxed),
    }
}
#[derive(Default)]
struct TensorRtBuildProgress {
    models: Vec<String>,
    current: Option<String>,
    completed: std::collections::HashSet<String>,
    started: Option<Instant>,
    model_started: std::collections::HashMap<String, Instant>,
}

static TENSORRT_BUILD_PROGRESS: OnceLock<std::sync::Mutex<TensorRtBuildProgress>> = OnceLock::new();

fn tensorrt_progress_model_name(model: &Path) -> String {
    model
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("ONNX model")
        .to_string()
}

fn begin_tensorrt_build(model: &Path) {
    let name = tensorrt_progress_model_name(model);
    let first_build_epoch;
    {
        let mut progress = TENSORRT_BUILD_PROGRESS
            .get_or_init(Default::default)
            .lock()
            .unwrap();
        first_build_epoch = progress.started.is_none();
        if first_build_epoch {
            progress.started = Some(Instant::now());
        }
        if !progress.models.contains(&name) {
            progress.models.push(name.clone());
        }
        log::info!(
            "tensorrt-engine-build: registered model={} total={}",
            name,
            progress.models.len()
        );
    }
    if first_build_epoch {
        crate::input::InputSystem::set_tensorrt_build_cursor_guard(true);
    }
}

fn mark_tensorrt_model_failed(model: &Path, reason: &str) {
    let name = tensorrt_progress_model_name(model);
    let release_guard;
    {
        let mut progress = TENSORRT_BUILD_PROGRESS
            .get_or_init(Default::default)
            .lock()
            .unwrap();
        let before = progress.models.len();
        progress.models.retain(|candidate| candidate != &name);
        progress.completed.remove(&name);
        progress.model_started.remove(&name);
        if progress.current.as_deref() == Some(name.as_str()) {
            progress.current = progress
                .models
                .iter()
                .find(|model| !progress.completed.contains(*model))
                .cloned();
        }
        if progress.models.is_empty() || progress.completed.len() == progress.models.len() {
            if before > 0 {
                log::warn!(
                    "tensorrt-engine-build: failed model={} remaining=0 reason={}",
                    name,
                    reason
                );
            }
            *progress = TensorRtBuildProgress::default();
            release_guard = true;
        } else {
            if progress.models.len() != before {
                log::warn!(
                    "tensorrt-engine-build: failed model={} remaining={} reason={}",
                    name,
                    progress.models.len(),
                    reason
                );
            }
            release_guard = false;
        }
    }
    if release_guard {
        crate::input::InputSystem::set_tensorrt_build_cursor_guard(false);
    }
}

pub fn begin_tensorrt_start_request() {
    TENSORRT_CANCEL_REQUESTED.store(false, Ordering::Release);
    ONNX_CANCEL_REQUESTED.store(false, Ordering::Release);
}

/// Close the cooperative Stop cancellation epoch after the render thread has
/// crossed the Stop queue boundary and released the old session. Idle backend
/// switches happen without a new capture Start, so leaving these flags set
/// would incorrectly reject every later DirectML/TensorRT checkbox request.
pub fn finish_onnx_stop_request() {
    let onnx = ONNX_CANCEL_REQUESTED.swap(false, Ordering::AcqRel);
    let tensorrt = TENSORRT_CANCEL_REQUESTED.swap(false, Ordering::AcqRel);
    if onnx || tensorrt {
        log::info!(
            "onnx-stop-cancellation-cleared: onnx={} tensorrt={}",
            onnx,
            tensorrt
        );
    }
}

struct CancelableRunOptions {
    options: RunOptions,
    terminated: AtomicBool,
}

impl CancelableRunOptions {
    fn new() -> Result<Arc<Self>> {
        let options = Arc::new(Self {
            options: RunOptions::new().map_err(oerr)?,
            terminated: AtomicBool::new(false),
        });
        register_onnx_run_options(&options);
        Ok(options)
    }

    fn terminate(&self) -> Result<()> {
        self.terminated.store(true, Ordering::Release);
        self.options.terminate().map_err(oerr)
    }

    fn armed(&self) -> Result<&RunOptions> {
        if onnx_cancel_requested() {
            return Err(anyhow!("ONNX inference cancelled by Stop request"));
        }
        // A terminated RunOptions remains terminated until explicitly reset.
        // Reset only a previously-cancelled object; normal frames pay no FFI
        // cost and a new session always starts armed.
        if self.terminated.swap(false, Ordering::AcqRel) {
            self.options.unterminate().map_err(oerr)?;
        }
        // Close the tiny race between the first flag check and unterminate.
        if onnx_cancel_requested() {
            let _ = self.terminate();
            return Err(anyhow!("ONNX inference cancelled by Stop request"));
        }
        Ok(&self.options)
    }
}

fn register_onnx_run_options(options: &Arc<CancelableRunOptions>) {
    let mut active = ACTIVE_ONNX_RUN_OPTIONS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    active.retain(|entry| entry.strong_count() > 0);
    active.push(Arc::downgrade(options));
}

/// Request cooperative cancellation for every ONNX inference currently owned
/// by this Neo process. This function is safe to call from the GUI thread and
/// returns without waiting for the render/provider thread to unwind.
pub fn request_onnx_cancel() -> usize {
    ONNX_CANCEL_REQUESTED.store(true, Ordering::Release);
    let mut requested = 0usize;
    let mut active = ACTIVE_ONNX_RUN_OPTIONS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    active.retain(|entry| {
        let Some(options) = entry.upgrade() else {
            return false;
        };
        match options.terminate() {
            Ok(()) => requested = requested.saturating_add(1),
            Err(error) => log::warn!("onnx-cancel-request-failed: {error}"),
        }
        true
    });
    if requested > 0 {
        log::info!("onnx-cancel-requested: active_run_options={requested}");
    } else {
        log::debug!("onnx-cancel-requested: active_run_options=0");
    }
    requested
}

pub fn onnx_cancel_requested() -> bool {
    ONNX_CANCEL_REQUESTED.load(Ordering::Acquire)
}

pub fn request_tensorrt_cancel() -> bool {
    if tensorrt_build_progress().is_some()
        && TENSORRT_CANCEL_REQUESTED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    {
        log::info!("tensorrt-engine-build: cancellation requested");
        true
    } else {
        false
    }
}

pub fn tensorrt_cancel_requested() -> bool {
    TENSORRT_CANCEL_REQUESTED.load(Ordering::Acquire)
}

pub fn tensorrt_is_preparing() -> bool {
    tensorrt_build_progress().is_some()
}

pub fn mark_tensorrt_model_started(name: &str) {
    let mut progress = TENSORRT_BUILD_PROGRESS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !progress.models.iter().any(|model| model == name) {
        return;
    }
    // A completed pre-filter may keep running while a later interpolation
    // engine is being built. Never let those normal frames move the popup back
    // to an already-finished model.
    if progress.completed.contains(name) {
        return;
    }
    let changed = progress.current.as_deref() != Some(name);
    progress.current = Some(name.to_string());
    progress
        .model_started
        .entry(name.to_string())
        .or_insert_with(Instant::now);
    if changed {
        let index = progress
            .models
            .iter()
            .position(|model| model == name)
            .map(|index| index + 1)
            .unwrap_or(1);
        log::info!(
            "tensorrt-engine-build: active model={} index={}/{} completed={}",
            name,
            index,
            progress.models.len(),
            progress.completed.len()
        );
    }
}

pub fn mark_tensorrt_model_completed(name: &str) {
    let release_guard;
    {
        let mut progress = TENSORRT_BUILD_PROGRESS
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if !progress.models.iter().any(|model| model == name) {
            return;
        }
        if progress.completed.insert(name.to_string()) {
            let model_elapsed = progress
                .model_started
                .get(name)
                .map(|started| started.elapsed())
                .unwrap_or_default();
            log::info!(
                "tensorrt-engine-build: model-ready model={} completed={}/{} model_elapsed_ms={:.2}",
                name,
                progress.completed.len(),
                progress.models.len(),
                model_elapsed.as_secs_f64() * 1000.0
            );
        }
        if !progress.models.is_empty() && progress.completed.len() == progress.models.len() {
            if let Some(started) = progress.started {
                log::info!(
                    "tensorrt-engine-build: inference-ready completed={}/{} elapsed_ms={:.2}",
                    progress.completed.len(),
                    progress.models.len(),
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
            *progress = TensorRtBuildProgress::default();
            release_guard = true;
        } else {
            progress.current = progress
                .models
                .iter()
                .find(|model| !progress.completed.contains(*model))
                .cloned();
            release_guard = false;
        }
    }
    if release_guard {
        crate::input::InputSystem::set_tensorrt_build_cursor_guard(false);
    }
}

/// Current TensorRT build item. The final bool is true only after that model's
/// first real TensorRT inference has started; registration alone is not treated
/// as active engine creation.
pub fn tensorrt_build_progress() -> Option<(String, Duration, usize, usize, bool)> {
    let progress = TENSORRT_BUILD_PROGRESS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    progress.started?;
    let name = progress.current.clone().or_else(|| {
        progress
            .models
            .iter()
            .find(|model| !progress.completed.contains(*model))
            .cloned()
    })?;
    let model_started = progress.model_started.get(&name).copied();
    Some((
        name,
        model_started
            .map(|started| started.elapsed())
            .unwrap_or_default(),
        progress.completed.len(),
        progress.models.len(),
        model_started.is_some(),
    ))
}

/// Clear build progress when a capture/session is explicitly stopped.
/// Successful completion goes through `mark_tensorrt_model_completed`;
/// displaying a frame is not proof that a lazy TensorRT build has finished.
pub fn finish_tensorrt_build_progress() {
    {
        let mut progress = TENSORRT_BUILD_PROGRESS
            .get_or_init(Default::default)
            .lock()
            .unwrap();
        if let Some(started) = progress.started {
            log::info!(
                "tensorrt-engine-build: progress-cleared completed={}/{} elapsed_ms={:.2}",
                progress.completed.len(),
                progress.models.len(),
                started.elapsed().as_secs_f64() * 1000.0
            );
        }
        *progress = TensorRtBuildProgress::default();
    }
    crate::input::InputSystem::set_tensorrt_build_cursor_guard(false);
}
/// rife_v4.25_lite REQUIRES 128-multiple padding: 64 (1920x1088) fails inside
/// DML with an invalid-parameter on a Mul node (tested) — the lite arch has a
/// deeper internal downsampling pyramid than regular v4 models.
const RIFE_PAD_MULTIPLE: usize = 128;

/// Initialize the ONNX runtime from a specific dll path (once per process).
/// Looks for onnxruntime.dll next to the exe, then in ./backends/.
pub fn init_onnx() -> std::result::Result<(), String> {
    ORT_READY
        .get_or_init(|| {
            let exe_dir = std::env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()))
                .unwrap_or_else(|| ".".into());
            let candidates = [
                exe_dir.join("onnxruntime.dll"),
                exe_dir.join("backends").join("onnxruntime.dll"),
                // Read-only compatibility for older portable layouts. New
                // packages use `backends` so users do not confuse runtime
                // libraries with the ONNX model folders.
                exe_dir.join("onnx").join("onnxruntime.dll"),
            ];
            let dll = candidates.iter().find(|p| p.exists()).ok_or_else(|| {
                format!(
                    "onnxruntime.dll not found next to the app ({}) — ONNX filters disabled",
                    exe_dir.display()
                )
            })?;
            // Dependent DLLs sit next to onnxruntime.dll (possibly a backend
            // outside the normal search path) — preload them explicitly.
            for dep in ["DirectML.dll", "onnxruntime_providers_shared.dll"] {
                let p = dll.with_file_name(dep);
                if p.exists() {
                    unsafe {
                        let wide: Vec<u16> = p
                            .as_os_str()
                            .to_string_lossy()
                            .encode_utf16()
                            .chain(std::iter::once(0))
                            .collect();
                        let _ = windows::Win32::System::LibraryLoader::LoadLibraryW(
                            windows::core::PCWSTR(wide.as_ptr()),
                        );
                    }
                }
            }
            let builder =
                ort::init_from(dll).map_err(|e| format!("onnxruntime load failed: {e}"))?;
            builder.commit();
            Ok(())
        })
        .clone()
}

struct TensorRtDiagnosticDir {
    path: Option<PathBuf>,
}

impl TensorRtDiagnosticDir {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }

    fn cleanup(&mut self) {
        let Some(path) = self.path.take() else {
            return;
        };
        match std::fs::remove_dir_all(&path) {
            Ok(()) => log::debug!(
                "tensorrt-cache: diagnostic directory removed path={}",
                path.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                log::debug!(
                    "tensorrt-cache: diagnostic directory retained path={} error={error}",
                    path.display()
                );
                self.path = Some(path);
            }
        }
    }
}

impl Drop for TensorRtDiagnosticDir {
    fn drop(&mut self) {
        self.cleanup();
    }
}

pub struct OnnxStage {
    pub name: String,
    source_path: PathBuf,
    fallback_dml_adapter: Option<i32>,
    session: Session,
    run_options: Arc<CancelableRunOptions>,
    tensorrt_diagnostic_dir: Option<TensorRtDiagnosticDir>,
    in_name: String,
    out_name: String,
    fp16: bool,
    scratch_f32: Vec<f32>,
    scratch_f16: Vec<f16>,
    /// (pw, ph, channels, aux) of the constant planes (timestep grids /
    /// multiplier planes) already present in the scratch buffer — they only
    /// depend on the padded size, so refilling them per frame was pure waste
    pack_const_key: Option<(usize, usize, usize, usize)>,
    /// timestep value currently written to the scratch's timestep plane
    pack_t_cached: Option<f32>,
    /// Consecutive RIFE pairs are (A,B), then (B,C). Keep enough identity
    /// metadata to reuse B's already-packed planes as the next first input.
    pack_frame_key: Option<(usize, usize, usize, usize, usize)>,
    pack_second_ptr: Option<usize>,
    /// v655: log once when the conservative cross-GPU RIFE pack path disables
    /// pointer-only previous-frame reuse. This avoids treating allocator pointer
    /// reuse as frame identity on slow/asynchronous secondary-GPU sessions.
    cross_gpu_rife_full_repack_logged: bool,
    last_interp_profile: Option<InterpProfile>,
    last_upscale_profile: Option<UpscaleProfile>,
    last_upscale_input_size: Option<(i32, i32)>,
    direct_output: Option<DmlDirectOutput>,
    direct_input: Option<DmlDirectInput>,
    dml_temporal_bridge: Option<DmlTemporalGpuBridge>,
    direct_output_disabled: bool,
    tensorrt_device_id: Option<i32>,
    tensorrt_gpu_bridge: Option<TensorRtGpuBridge>,
    tensorrt_temporal_bridge: Option<TensorRtTemporalGpuBridge>,
    tensorrt_interp_bridge: Option<TensorRtInterpGpuBridge>,
    dml_interp_bridge: Option<DmlInterpGpuBridge>,
    /// Cross-GPU fallback: CPU-packed RIFE input still enters DirectML normally,
    /// but completed outputs stay in app-owned shareable D3D12 buffers so the
    /// same selected GPU can hand them directly to Vulkan without a CPU readback.
    dml_cpu_interp_shared_outputs: [Vec<DmlDirectOutput>; 2],
    dml_cpu_interp_shared_disabled: bool,
    interp_gpu_path_disabled: bool,
    tensorrt_gpu_path_disabled: bool,
    tensorrt_gpu_unsupported_logged: bool,
    tensorrt_shape_seen: std::collections::HashSet<(i32, i32)>,
    input_channels: Option<usize>,
    /// Adaptive exact-area inference for validated local image CNNs. Unsupported
    /// graphs never allocate this state and stay on the original DirectML path.
    neo_accel: Option<NeoAccelState>,
    /// Temporal restoration models concatenate N RGB frames on the channel
    /// axis (for example 15ch=5 frames, 21ch=7 frames).
    temporal_frames: Option<usize>,
    temporal_history: VecDeque<Vec<u8>>,
    temporal_size: Option<(i32, i32)>,
    pub provider_desc: String,
    pub provider: OnnxProvider,
    pub fallback_reason: Option<String>,
    tensorrt_cache_dir: Option<PathBuf>,
    tensorrt_engine_verified: bool,
    tensorrt_profiling_active: bool,
    /// Frame-interpolation model (RIFE-style): two image inputs, or one
    /// 6-channel input (prev+cur concatenated). Optional scalar "timestep".
    pub interp: InterpKind,
}

impl Drop for OnnxStage {
    fn drop(&mut self) {
        if let Some(cache_dir) = &self.tensorrt_cache_dir {
            unregister_tensorrt_cache(cache_dir);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxProvider {
    DirectML,
    TensorRT,
    Cuda,
}

#[derive(Clone, Debug, PartialEq)]
pub enum InterpKind {
    /// Ordinary 1-in/1-out image model.
    None,
    /// Two separate NCHW 3ch inputs (+ optional timestep input name).
    TwoInputs {
        second: String,
        timestep: Option<String>,
    },
    /// Single NCHW 6ch input: channels = [prev(3), cur(3)].
    SixChannel,
    /// vs-mlrt RIFE v1 layout: single NCHW input with `channels` channels:
    /// img0(3) + img1(3) + timestep(1) [+ hgrid(1) + vgrid(1) + 2/(w-1) + 2/(h-1)].
    /// Width/height are padded to a multiple of 128 (covers 32/64/128 reqs).
    RifeV1 { channels: usize },
    /// distilDRBA layout (14ch): [f(n-1), f(n), f(n+1), f(n+2)] (12ch) +
    /// timestep(1, between f(n) and f(n+1)) + scale placeholder = 1.0 (1).
    /// Requires a 64-multiple tile; interpolation runs one frame behind.
    Drba,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct InterpProfile {
    pub pack_ms: f64,
    pub run_ms: f64,
    pub out_ms: f64,
    pub padded_size: (usize, usize),
}

/// Stable description of one GPU-resident interpolation output slot.
/// The render thread can import a completed slot without locking the ONNX
/// stage while the worker continues computing later, distinct timesteps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedInterpGpuOutput {
    pub(crate) key: u64,
    pub(crate) size: (i32, i32),
    pub(crate) padded: (i32, i32),
    pub(crate) fp16: bool,
    /// Keep DirectML interpolation in floating point when a shader/image
    /// stage follows it. Quantizing RIFE/DRBA to RGBA8 before a CNN
    /// upscaler can turn tiny provider-specific errors into visible
    /// checkerboard/nearest-neighbour-like blocks.
    pub(crate) preserve_float: bool,
}

/// One completed DirectML interpolation slot exported as its app-owned
/// shareable D3D12 committed buffer. The Win32 handle is a fresh NT handle
/// created for this export and remains owned by this wrapper; Vulkan import
/// does not take ownership of Win32 handles.
pub(crate) struct PreparedDmlSharedOutput {
    pub(crate) key: u64,
    pub(crate) handle: HANDLE,
    pub(crate) byte_len: usize,
    pub(crate) heap_byte_len: u64,
    pub(crate) luid: [u8; 8],
    pub(crate) size: (i32, i32),
    pub(crate) padded: (i32, i32),
    pub(crate) fp16: bool,
}

// Moving an owned Win32 handle between the interpolation worker and render
// thread is safe; the underlying DirectML resource remains owned by OnnxStage.
unsafe impl Send for PreparedDmlSharedOutput {}

impl Drop for PreparedDmlSharedOutput {
    fn drop(&mut self) {
        if !self.handle.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.handle);
            }
            self.handle = HANDLE::default();
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct UpscaleProfile {
    pub pack_ms: f64,
    pub run_ms: f64,
    pub out_ms: f64,
    pub output_size: (usize, usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DmlGlInteropMode {
    ProviderOwnedCopy,
    Direct,
}

impl DmlGlInteropMode {
    fn label(self) -> &'static str {
        match self {
            Self::ProviderOwnedCopy => "ProviderOwnedCopyBridge",
            Self::Direct => "AppOwnedDirect",
        }
    }
}

fn resolve_dml_gl_interop_mode(
    configured: &str,
    after_interpolation: bool,
    conservative_driver: bool,
) -> Option<DmlGlInteropMode> {
    let configured = configured.trim().to_ascii_lowercase();
    match configured.as_str() {
        "direct" => Some(DmlGlInteropMode::Direct),
        "1" | "input" | "copy" | "auto" => Some(DmlGlInteropMode::ProviderOwnedCopy),
        "0" | "off" | "false" | "disable" | "disabled" => None,
        "" if after_interpolation && !conservative_driver => {
            Some(DmlGlInteropMode::ProviderOwnedCopy)
        }
        _ => None,
    }
}

impl OnnxStage {
    pub fn load(path: &Path) -> Result<Self> {
        Self::load_on_adapter(path, None)
    }

    pub fn load_on_adapter(path: &Path, adapter_id: Option<i32>) -> Result<Self> {
        Self::load_directml(path, adapter_id)
    }

    pub fn load_with_preference(
        path: &Path,
        preference: OnnxBackendPreference,
        dml_adapter_id: Option<i32>,
        trt_device_id: Option<i32>,
        cache_root: &Path,
    ) -> Result<Self> {
        match preference {
            OnnxBackendPreference::DirectML => Self::load_directml(path, dml_adapter_id),
            OnnxBackendPreference::TensorRT => {
                let device_id =
                    trt_device_id.ok_or_else(|| anyhow!("TensorRT CUDA device is unavailable"))?;
                match Self::load_tensorrt(path, device_id, dml_adapter_id, cache_root) {
                    Ok(stage) => Ok(stage),
                    Err(trt_error) => {
                        mark_tensorrt_model_failed(path, &format!("{trt_error:#}"));
                        log::warn!(
                            "onnx-backend-fallback: model={} TensorRT failed: {trt_error:#}; using DirectML",
                            path.display()
                        );
                        let mut stage = Self::load_directml(path, dml_adapter_id)?;
                        stage.fallback_reason = Some(format!("{trt_error:#}"));
                        stage.provider_desc.push_str(" fallback");
                        Ok(stage)
                    }
                }
            }
        }
    }

    fn load_directml(path: &Path, adapter_id: Option<i32>) -> Result<Self> {
        init_onnx().map_err(|e| anyhow!(e))?;
        let graph_optimization = match std::env::var("CHIDESCALER_DML_GRAPH_OPT").ok().as_deref() {
            Some("disable") => GraphOptimizationLevel::Disable,
            Some("basic") => GraphOptimizationLevel::Level1,
            Some("extended") => GraphOptimizationLevel::Level2,
            _ => GraphOptimizationLevel::All,
        };
        log::info!("directml-graph-optimization: level={graph_optimization:?}");
        let directml = match adapter_id {
            Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
            None => ort::ep::DirectML::default().with_performance_preference(
                ort::ep::directml::PerformancePreference::HighPerformance,
            ),
        };
        let session = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(graph_optimization)
            .map_err(oerr)?
            .with_execution_providers([directml.build()])
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?
            .commit_from_file(path)
            .map_err(oerr)?;
        let provider_desc = adapter_id
            .map(|id| format!("DirectML device {id}"))
            .unwrap_or_else(|| "DirectML auto/high-performance".into());
        Self::from_session(
            path,
            session,
            adapter_id,
            OnnxProvider::DirectML,
            provider_desc,
            None,
            None,
            false,
        )
    }

    fn load_tensorrt(
        path: &Path,
        device_id: i32,
        fallback_dml_adapter: Option<i32>,
        cache_root: &Path,
    ) -> Result<Self> {
        init_onnx().map_err(|e| anyhow!(e))?;
        let builder_level = tensorrt_builder_level();
        let workspace_mb = tensorrt_workspace_mb();
        let cache_variant = format!(
            "builder-{builder_level}_workspace-{}mb_fp16-1",
            workspace_mb
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".into())
        );
        let cache_dir = tensorrt_model_cache_dir(cache_root, path, cache_variant.as_bytes());
        migrate_legacy_tensorrt_cache(cache_root, path, cache_variant.as_bytes(), &cache_dir);
        maintain_tensorrt_cache_root(cache_root, Some(&cache_dir));
        let cache_enabled = match std::fs::create_dir_all(&cache_dir) {
            Ok(()) => {
                touch_tensorrt_cache(&cache_dir);
                true
            }
            Err(error) => {
                log::warn!(
                    "tensorrt-cache: model={} disabled reason={error}",
                    path.display()
                );
                false
            }
        };
        let cache_was_present = cache_enabled && Self::has_tensorrt_engine_files(&cache_dir);
        let provider_verification_cached =
            cache_was_present && tensorrt_provider_verification_ready(&cache_dir);
        if !cache_was_present {
            begin_tensorrt_build(path);
        }
        if cache_enabled {
            prepare_tensorrt_cache_attempt(&cache_dir, cache_was_present)?;
        }
        let mut tensorrt = ort::ep::TensorRT::default()
            .with_device_id(device_id)
            .with_fp16(true)
            .with_force_sequential_engine_build(true)
            .with_builder_optimization_level(builder_level);
        if let Some(workspace_mb) = workspace_mb {
            tensorrt = tensorrt
                .with_max_workspace_size(workspace_mb.saturating_mul(1024).saturating_mul(1024));
        }
        log::info!(
            "tensorrt-builder-config: model={} level={} workspace_mb={}",
            path.display(),
            builder_level,
            workspace_mb
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".into())
        );
        if cache_enabled {
            let cache_path = tensorrt_compatible_path(&cache_dir)?
                .to_string_lossy()
                .into_owned();
            tensorrt = tensorrt
                .with_engine_cache(true)
                .with_engine_cache_path(cache_path.clone())
                .with_timing_cache(true)
                .with_timing_cache_path(cache_path);
            log::info!(
                "tensorrt-cache: model={} path={} enabled=true",
                path.display(),
                cache_dir.display()
            );
        }
        let cuda = ort::ep::CUDA::default().with_device_id(device_id);
        let profile_key = TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed);
        let profile_root = if provider_verification_cached {
            log::info!(
                "tensorrt-cache: provider verification reused model={} path={}",
                path.display(),
                cache_dir.display()
            );
            None
        } else {
            let portable = cache_enabled
                .then(|| tensorrt_diagnostics_profile_root(cache_root, profile_key))
                .flatten()
                .and_then(|root| tensorrt_compatible_path(&root).ok());
            portable.or_else(|| {
                let fallback = std::env::temp_dir()
                    .join("cHiDeScaler-Neo")
                    .join("TensorRT-profiles")
                    .join(format!("session-{}", std::process::id()))
                    .join(format!("profile-{profile_key}"));
                std::fs::create_dir_all(&fallback).ok().map(|_| fallback)
            })
        };
        let profile_prefix = profile_root.as_ref().map(|root| {
            root.join(format!(
                "ort-provider-{}-{}",
                std::process::id(),
                profile_key
            ))
        });
        let diagnostic_guard = profile_root.clone().map(TensorRtDiagnosticDir::new);
        let mut builder = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(oerr)?
            .with_execution_providers([
                tensorrt.build().error_on_failure(),
                cuda.build().error_on_failure(),
            ])
            .map_err(oerr)?
            .with_disable_cpu_fallback()
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?;
        if let Some(profile_prefix) = &profile_prefix {
            builder = builder.with_profiling(profile_prefix).map_err(oerr)?;
        }
        let staged_path = stage_tensorrt_model(path, &cache_dir)?;
        let session_path = tensorrt_compatible_path(&staged_path)?;
        let session = match builder.commit_from_file(&session_path).map_err(oerr) {
            Ok(session) => session,
            Err(error) => {
                if cache_enabled && !cache_was_present {
                    cleanup_incomplete_tensorrt_cache(&cache_dir);
                }
                return Err(error);
            }
        };
        Self::from_session(
            path,
            session,
            fallback_dml_adapter,
            OnnxProvider::TensorRT,
            format!("TensorRT + CUDA device {device_id}"),
            cache_enabled.then_some(cache_dir),
            diagnostic_guard,
            profile_prefix.is_some(),
        )
        .map(|mut stage| {
            stage.tensorrt_device_id = Some(device_id);
            stage.tensorrt_engine_verified = provider_verification_cached;
            stage
        })
    }

    fn from_session(
        path: &Path,
        session: Session,
        fallback_dml_adapter: Option<i32>,
        provider: OnnxProvider,
        mut provider_desc: String,
        tensorrt_cache_dir: Option<PathBuf>,
        tensorrt_diagnostic_dir: Option<TensorRtDiagnosticDir>,
        tensorrt_profiling_active: bool,
    ) -> Result<Self> {
        // On early validation errors, close the ORT session before deleting its
        // profiling directory. The local declaration order guarantees that.
        let tensorrt_diagnostic_dir = tensorrt_diagnostic_dir;
        let session = session;
        let inp = session
            .inputs()
            .first()
            .ok_or_else(|| anyhow!("model has no inputs"))?;
        let in_name = inp.name().to_string();
        let fp16 = format!("{:?}", inp.dtype())
            .to_lowercase()
            .contains("float16")
            || format!("{}", inp.dtype()).to_lowercase().contains("f16");
        let out_name = session
            .outputs()
            .first()
            .ok_or_else(|| anyhow!("model has no outputs"))?
            .name()
            .to_string();
        // frame-interpolation model detection (RIFE-style)
        let interp = {
            let ins = session.inputs();
            let image_ins: Vec<&str> = ins
                .iter()
                .filter(|o| {
                    let d = format!("{}", o.dtype());
                    d.contains("3") || d.to_lowercase().contains("tensor")
                })
                .map(|o| o.name())
                .collect();
            let dtype0 = format!("{}", ins[0].dtype());
            if ins.len() >= 2 && image_ins.len() >= 2 {
                let timestep = ins
                    .iter()
                    .map(|o| o.name())
                    .find(|n| {
                        n.to_lowercase().contains("timestep") || n.to_lowercase().contains("factor")
                    })
                    .map(|s| s.to_string());
                InterpKind::TwoInputs {
                    second: ins[1].name().to_string(),
                    timestep,
                }
            } else if dtype0.contains(", 6,") {
                InterpKind::SixChannel
            } else if let Some(ch) = parse_channels(&dtype0) {
                // vs-mlrt style: RIFE = 3*2 frames + timestep + grids + mults
                // (7/11ch). distilDRBA = 4 frames + timestep + scale (14ch),
                // layout confirmed from the bundled vsmlrt DRBAMerge source.
                if ch == 7 || ch == 11 {
                    InterpKind::RifeV1 { channels: ch }
                } else if ch == 14 {
                    InterpKind::Drba
                } else {
                    InterpKind::None
                }
            } else {
                InterpKind::None
            }
        };
        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let input_channels = parse_channels(&format!("{}", session.inputs()[0].dtype()));
        let temporal_frames = input_channels
            .filter(|channels| *channels >= 9 && *channels % 3 == 0)
            .map(|channels| channels / 3);
        let neo_accel = if interp == InterpKind::None && temporal_frames.is_none() {
            match analyze_local_image_model(path) {
                Ok(Some(plan)) => {
                    log::info!(
                        "NeoAccel eligible: model='{}' convs={} halo={}x{} scale_hint={} alignment={}",
                        name,
                        plan.conv_count,
                        plan.halo_x,
                        plan.halo_y,
                        plan.scale_hint,
                        plan.alignment
                    );
                    Some(NeoAccelState::new(
                        plan,
                        format!(
                            "neo-onnx-accel:{}",
                            NEO_ACCEL_KEY.fetch_add(1, Ordering::Relaxed)
                        ),
                    ))
                }
                Ok(None) => None,
                Err(error) => {
                    log::warn!(
                        "NeoAccel model analysis failed for '{}'; DirectML fallback retained: {error:#}",
                        name
                    );
                    None
                }
            }
        } else {
            None
        };
        if neo_accel.is_some() {
            provider_desc.push_str(" + NeoAccel sparse-local");
        }
        if let Some(frames) = temporal_frames {
            provider_desc.push_str(&format!(" + temporal-{frames}f"));
        }
        let run_options = CancelableRunOptions::new()?;
        if onnx_cancel_requested() {
            // A Stop can arrive while Session::commit_from_file is still
            // building a provider. Make the newly-created stage inherit that
            // stop request instead of starting one late inference.
            run_options.terminate().map_err(oerr)?;
        }
        let stage = Self {
            name,
            source_path: path.to_path_buf(),
            fallback_dml_adapter,
            session,
            run_options,
            tensorrt_diagnostic_dir,
            in_name,
            out_name,
            fp16,
            scratch_f32: Vec::new(),
            scratch_f16: Vec::new(),
            pack_const_key: None,
            pack_t_cached: None,
            pack_frame_key: None,
            pack_second_ptr: None,
            cross_gpu_rife_full_repack_logged: false,
            last_interp_profile: None,
            last_upscale_profile: None,
            last_upscale_input_size: None,
            direct_output: None,
            direct_input: None,
            dml_temporal_bridge: None,
            direct_output_disabled: false,
            tensorrt_device_id: None,
            tensorrt_gpu_bridge: None,
            tensorrt_temporal_bridge: None,
            tensorrt_interp_bridge: None,
            dml_interp_bridge: None,
            dml_cpu_interp_shared_outputs: [Vec::new(), Vec::new()],
            dml_cpu_interp_shared_disabled: false,
            interp_gpu_path_disabled: false,
            tensorrt_gpu_path_disabled: false,
            tensorrt_gpu_unsupported_logged: false,
            tensorrt_shape_seen: Default::default(),
            input_channels,
            neo_accel,
            temporal_frames,
            temporal_history: VecDeque::new(),
            temporal_size: None,
            provider_desc,
            provider,
            fallback_reason: None,
            tensorrt_cache_dir,
            tensorrt_engine_verified: false,
            tensorrt_profiling_active,
            interp,
        };
        if let Some(cache_dir) = &stage.tensorrt_cache_dir {
            register_tensorrt_cache(cache_dir);
        }
        Ok(stage)
    }

    /// True for ordinary multi-frame temporal restoration models running on
    /// DirectML. The render chain uses this to apply the DirectML-only
    /// high-resolution safety cap without relying on a specific model name.
    pub fn is_directml_temporal_filter(&self) -> bool {
        self.provider == OnnxProvider::DirectML
            && self.temporal_frames.is_some()
            && self.interp == InterpKind::None
    }

    pub fn supports_dml_gl_bridge(&self) -> bool {
        self.provider == OnnxProvider::DirectML
    }

    fn has_tensorrt_engine_cache(&self) -> bool {
        let Some(cache_dir) = &self.tensorrt_cache_dir else {
            return false;
        };
        Self::has_tensorrt_engine_files(cache_dir)
    }

    pub fn prepare_tensorrt_input_shape(&mut self, width: i32, height: i32) {
        if self.provider != OnnxProvider::TensorRT {
            return;
        }
        let Some(cache_dir) = &self.tensorrt_cache_dir else {
            return;
        };
        if !self.tensorrt_shape_cache_ready(cache_dir, width, height) {
            begin_tensorrt_build(&self.source_path);
            mark_tensorrt_model_started(&self.name);
            log::info!(
                "tensorrt-shape-cache: model={} input={}x{} reused=false",
                self.name,
                width,
                height
            );
        } else if self.tensorrt_shape_seen.insert((width, height)) {
            log::debug!(
                "tensorrt-shape-cache: model={} input={}x{} reused=true",
                self.name,
                width,
                height
            );
        }
    }

    pub fn complete_tensorrt_input_shape(&self, width: i32, height: i32) -> bool {
        if self.provider != OnnxProvider::TensorRT {
            return false;
        }
        let Some(cache_dir) = &self.tensorrt_cache_dir else {
            return false;
        };
        let ready = self.tensorrt_shape_manifest_path(cache_dir, width, height);
        if !ready.exists() {
            let engines = tensorrt_engine_file_names(cache_dir);
            if engines.is_empty() {
                log::warn!(
                    "tensorrt-shape-cache: refusing ready manifest without engine artifacts model={} input={}x{}",
                    self.name,
                    width,
                    height
                );
                return false;
            }
            if let Some(parent) = ready.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let manifest = serde_json::json!({
                "schema": 1,
                "model": self.name,
                "model_kind": self.tensorrt_shape_kind(),
                "input_width": width,
                "input_height": height,
                "input_channels": self.input_channels,
                "builder_level": tensorrt_builder_level(),
                "workspace_mb": tensorrt_workspace_mb(),
                "fp16": self.fp16,
                "engine_files": engines,
                "verified": true
            });
            let temporary = ready.with_extension("json.tmp");
            let write_result = serde_json::to_vec_pretty(&manifest)
                .map_err(std::io::Error::other)
                .and_then(|bytes| std::fs::write(&temporary, bytes))
                .and_then(|_| std::fs::rename(&temporary, &ready));
            if let Err(error) = write_result {
                log::warn!(
                    "tensorrt-shape-cache: manifest write failed path={} error={error}",
                    ready.display()
                );
                return false;
            }
        }
        self.tensorrt_shape_cache_ready(cache_dir, width, height)
    }

    fn tensorrt_shape_kind(&self) -> String {
        if let Some(frames) = self.temporal_frames {
            format!(
                "temporal-{}f-{}ch",
                frames,
                self.input_channels.unwrap_or(0)
            )
        } else if self.interp != InterpKind::None {
            format!("interp-{}ch", self.input_channels.unwrap_or(0))
        } else {
            format!("image-{}ch", self.input_channels.unwrap_or(0))
        }
    }

    fn tensorrt_shape_manifest_path(&self, cache_dir: &Path, width: i32, height: i32) -> PathBuf {
        cache_dir.join("shapes").join(format!(
            "{}x{}-{}.json",
            width,
            height,
            self.tensorrt_shape_kind()
        ))
    }

    fn tensorrt_shape_cache_ready(&self, cache_dir: &Path, width: i32, height: i32) -> bool {
        let manifest = self.tensorrt_shape_manifest_path(cache_dir, width, height);
        let valid_manifest = std::fs::read(&manifest)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
            .is_some_and(|value| {
                value.get("schema").and_then(|v| v.as_u64()) == Some(1)
                    && value.get("verified").and_then(|v| v.as_bool()) == Some(true)
                    && value.get("input_width").and_then(|v| v.as_i64()) == Some(width as i64)
                    && value.get("input_height").and_then(|v| v.as_i64()) == Some(height as i64)
                    && value.get("model_kind").and_then(|v| v.as_str())
                        == Some(self.tensorrt_shape_kind().as_str())
                    && !tensorrt_engine_file_names(cache_dir).is_empty()
            });
        if valid_manifest {
            return true;
        }
        // One-time migration for legacy caches. Trust the old marker only when
        // a non-empty engine artifact is present, then replace it after the
        // next successful inference with the stronger manifest.
        tensorrt_shape_ready_path(cache_dir, width, height).exists()
            && !tensorrt_engine_file_names(cache_dir).is_empty()
    }

    fn has_tensorrt_engine_files(cache_dir: &Path) -> bool {
        std::fs::read_dir(cache_dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .is_some_and(|extension| {
                        matches!(
                            extension.to_ascii_lowercase().as_str(),
                            "engine" | "cache" | "timing"
                        )
                    })
            })
    }

    fn replace_with_directml_fallback(&mut self, reason: String) -> bool {
        mark_tensorrt_model_failed(&self.source_path, &reason);
        self.cleanup_tensorrt_diagnostic_dir();
        let failed_tensorrt_cache = self.tensorrt_cache_dir.clone();
        match Self::load_directml(&self.source_path, self.fallback_dml_adapter) {
            Ok(mut replacement) => {
                replacement.fallback_reason = Some(reason.clone());
                replacement.provider_desc.push_str(" fallback");
                log::warn!(
                    "onnx-backend-fallback: model={} TensorRT unavailable after warmup: {}; using {}",
                    self.source_path.display(),
                    reason,
                    replacement.provider_desc
                );
                *self = replacement;
                if let Some(cache_dir) = failed_tensorrt_cache {
                    cleanup_incomplete_tensorrt_cache(&cache_dir);
                    log::warn!(
                        "tensorrt-cache: removed failed engine artifacts model={} path={}",
                        self.source_path.display(),
                        cache_dir.display()
                    );
                }
                true
            }
            Err(error) => {
                self.fallback_reason = Some(format!(
                    "{reason}; DirectML fallback creation also failed: {error:#}"
                ));
                log::error!(
                    "onnx-backend-fallback: model={} DirectML fallback failed: {error:#}; retaining CUDA session",
                    self.source_path.display()
                );
                false
            }
        }
    }

    fn cleanup_tensorrt_diagnostic_dir(&mut self) {
        if let Some(mut diagnostic) = self.tensorrt_diagnostic_dir.take() {
            diagnostic.cleanup();
        }
    }

    pub fn finalize_provider_after_warmup(&mut self) {
        if self.provider != OnnxProvider::TensorRT {
            return;
        }
        let engine_cache = self.has_tensorrt_engine_cache();
        if self.tensorrt_engine_verified && !self.tensorrt_profiling_active {
            self.cleanup_tensorrt_diagnostic_dir();
            if let Some(cache_dir) = &self.tensorrt_cache_dir {
                mark_tensorrt_cache_ready(cache_dir);
                touch_tensorrt_cache(cache_dir);
            }
            log::info!(
                "onnx-backend-stage: model={} provider=TensorRT+CUDA verification=cached engine_cache={}",
                self.name,
                if engine_cache {
                    "present"
                } else {
                    "not-created"
                }
            );
            return;
        }
        let provider_evidence = if self.tensorrt_profiling_active {
            self.tensorrt_profiling_active = false;
            self.session
                .end_profiling()
                .map_err(oerr)
                .and_then(|path| {
                    let path = PathBuf::from(path);
                    let result = providers_from_ort_profile(&path);
                    if let Err(error) = std::fs::remove_file(&path) {
                        log::debug!(
                            "tensorrt-cache: temporary provider profile cleanup failed path={} error={error}",
                            path.display()
                        );
                    } else if let Some(parent) = path.parent() {
                        let _ = std::fs::remove_dir(parent);
                    }
                    result
                })
        } else {
            Err(anyhow!("ONNX Runtime provider profiling was unavailable"))
        };
        self.cleanup_tensorrt_diagnostic_dir();
        match provider_evidence {
            Ok((true, used_cuda)) => {
                self.tensorrt_engine_verified = true;
                if let Some(cache_dir) = &self.tensorrt_cache_dir {
                    mark_tensorrt_provider_verification_ready(cache_dir);
                    mark_tensorrt_cache_ready(cache_dir);
                    touch_tensorrt_cache(cache_dir);
                }
                log::info!(
                    "onnx-backend-stage: model={} provider=TensorRT+CUDA cuda_nodes={} engine_cache={}",
                    self.name,
                    used_cuda,
                    if engine_cache {
                        "present"
                    } else {
                        "not-created"
                    }
                );
            }
            Ok((false, true)) => {
                if !self.replace_with_directml_fallback(
                    "TensorRT used no graph nodes; CUDA-only execution was rejected".into(),
                ) {
                    self.provider = OnnxProvider::Cuda;
                    self.provider_desc = "CUDA fallback (DirectML unavailable)".into();
                }
            }
            Ok((false, false)) => {
                if !self.replace_with_directml_fallback(
                    "provider profile contained no TensorRT node execution".into(),
                ) {
                    self.provider = OnnxProvider::Cuda;
                    self.provider_desc = "CUDA fallback (DirectML unavailable)".into();
                }
            }
            Err(error) => {
                // Never label a session as TensorRT merely because a cache file
                // exists. The profile records which EP executed graph nodes.
                if !self.replace_with_directml_fallback(format!(
                    "TensorRT provider verification failed: {error:#}"
                )) {
                    self.provider = OnnxProvider::Cuda;
                    self.provider_desc = "CUDA fallback (DirectML unavailable)".into();
                }
            }
        }
    }

    pub fn reset_backend_runtime_state(&mut self) {
        self.pack_const_key = None;
        self.pack_t_cached = None;
        self.pack_frame_key = None;
        self.pack_second_ptr = None;
        self.cross_gpu_rife_full_repack_logged = false;
        self.last_interp_profile = None;
        self.last_upscale_profile = None;
        self.last_upscale_input_size = None;
        self.temporal_history.clear();
        self.temporal_size = None;
        self.direct_output_disabled = false;
        for bank in &mut self.dml_cpu_interp_shared_outputs {
            bank.clear();
        }
        self.dml_cpu_interp_shared_disabled = false;
        self.tensorrt_gpu_path_disabled = false;
        self.tensorrt_gpu_unsupported_logged = false;
    }

    pub fn last_interp_profile(&self) -> Option<InterpProfile> {
        self.last_interp_profile
    }

    pub fn last_upscale_profile(&self) -> Option<UpscaleProfile> {
        self.last_upscale_profile
    }

    pub fn reset_interp_pack_cache(&mut self) {
        self.pack_frame_key = None;
        self.pack_second_ptr = None;
    }

    pub(crate) fn disable_dml_cpu_interp_shared(&mut self, reason: &str) {
        if !self.dml_cpu_interp_shared_disabled {
            log::warn!(
                "dml-vulkan-resident-handoff: action=disable reason={} fallback=cpu-visible-interpolation",
                reason
            );
        }
        self.dml_cpu_interp_shared_disabled = true;
    }

    /// How many input frames the interpolation model consumes (1 = not an
    /// interpolation model).
    pub fn interp_frames(&self) -> usize {
        match &self.interp {
            InterpKind::Drba => 4,
            InterpKind::None => 1,
            _ => 2,
        }
    }

    /// DRBA interpolates between frames[1] and frames[2] (needs one frame of
    /// lookahead); everything else interpolates before the newest frame.
    pub fn interp_delayed(&self) -> bool {
        matches!(self.interp, InterpKind::Drba)
    }

    pub fn supports_interp_gpu(&self) -> bool {
        !matches!(
            std::env::var("CHIDESCALER_INTERP_GPU").ok().as_deref(),
            Some("disable")
        ) && !self.interp_gpu_path_disabled
            && matches!(
                self.interp,
                InterpKind::RifeV1 { .. } | InterpKind::SixChannel | InterpKind::Drba
            )
            && matches!(
                self.provider,
                OnnxProvider::TensorRT | OnnxProvider::DirectML
            )
    }

    pub fn disable_interp_gpu_path(&mut self, reason: &str) {
        if !self.interp_gpu_path_disabled {
            log::error!(
                "interp-gpu-path-poisoned: backend={} model={} reason={}",
                self.provider_desc,
                self.name,
                reason
            );
        }
        self.interp_gpu_path_disabled = true;
    }

    pub fn process_interp_gpu_textures(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
    ) -> Result<Option<Vec<GpuTex>>> {
        let Some(fence) = self.prepare_interp_gpu_textures(gc, frames, timesteps, generation)?
        else {
            return Ok(None);
        };
        let deadline = Instant::now() + std::time::Duration::from_millis(250);
        loop {
            if gc.poll_commands_fence(fence).map_err(anyhow::Error::msg)? {
                break;
            }
            if Instant::now() >= deadline {
                gc.cancel_commands_fence(fence);
                anyhow::bail!("OpenGL interpolation input handoff timed out after 250ms");
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.run_prepared_interp_gpu(timesteps.len())?;
        Ok(Some(self.finish_prepared_interp_gpu(gc, timesteps.len())?))
    }

    /// Issue all GL-side input packing and return a fence token. The caller
    /// must poll the token from later render-loop ticks before submitting the
    /// DirectML/TensorRT job. No GPU wait is performed here.
    pub fn prepare_interp_gpu_textures(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
    ) -> Result<Option<u64>> {
        self.prepare_interp_gpu_textures_with_factor(
            gc,
            frames,
            timesteps,
            generation,
            timesteps.len() + 1,
            false,
        )
    }

    /// `capacity_factor` is the user-selected interpolation factor, not merely
    /// the first cadence-limited batch size. Startup frequently begins with a
    /// single midpoint while cadence locks; sizing from that transient batch
    /// forced an immediate TensorRT bridge destroy/recreate on the next frame.
    pub fn prepare_interp_gpu_textures_with_factor(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
        capacity_factor: usize,
        stable_post_handoff: bool,
    ) -> Result<Option<u64>> {
        if !self.supports_interp_gpu() || timesteps.is_empty() {
            return Ok(None);
        }

        // v653: DirectML interpolation cannot use the OpenGL external-memory
        // bridge when presentation GL and the explicitly selected compute GPU
        // are different adapters. v652 still entered bridge construction first
        // and only discovered the LUID mismatch while importing the shared
        // buffers. That transaction is unnecessary and, on some drivers, can
        // leave the first RIFE session observing partially-visible/invalid
        // shared contents before the CPU-visible fallback takes over.
        //
        // Bypass that impossible route *before allocating/importing anything*.
        // The established CPU-packed DirectML path remains available, and when
        // compatible Vulkan GLSL follows RIFE the separate DML->Vulkan shared
        // output handoff can still keep the expensive output boundary resident
        // on the selected GPU.
        if self.provider == OnnxProvider::DirectML {
            if let (Some(gl_luid), Some(selected_luid)) = (
                gc.external_device_luid(),
                crate::render::vulkan_gpu::production_selected_luid(),
            ) {
                let gl_luid_u64 = u64::from_le_bytes(gl_luid);
                if gl_luid_u64 != selected_luid {
                    self.interp_gpu_path_disabled = true;
                    self.reset_interp_pack_cache();
                    log::info!(
                        "interp-gpu-path-bypass: backend=DirectML model={} reason=cross-gpu-gl-luid-mismatch selected_luid={:016x} gl_luid={:016x} route=cpu-input-directml{}",
                        self.name,
                        selected_luid,
                        gl_luid_u64,
                        if stable_post_handoff {
                            "->d3d12-vulkan-output-when-compatible"
                        } else {
                            "->cpu-visible-output"
                        }
                    );
                    return Ok(None);
                }
            }
        }
        let required = self.interp_frames();
        anyhow::ensure!(
            frames.len() >= required,
            "GPU interpolation history is incomplete"
        );
        let frames = &frames[frames.len() - required..];
        let size = (frames[0].w(), frames[0].h());
        self.prepare_tensorrt_input_shape(size.0, size.1);
        mark_tensorrt_model_started(&self.name);
        anyhow::ensure!(
            frames.iter().all(|frame| (frame.w(), frame.h()) == size),
            "GPU interpolation history dimensions differ"
        );
        let result = match self.provider {
            OnnxProvider::TensorRT => {
                self.process_tensorrt_interp_gpu(gc, frames, timesteps, generation, capacity_factor)
            }
            OnnxProvider::DirectML => {
                let Some(mode) = dml_interp_interop_mode(gc, stable_post_handoff) else {
                    log::info!(
                        "dml-interp-interop-policy: mode=nonresident reason=explicit-safe-fallback"
                    );
                    return Ok(None);
                };
                self.process_dml_interp_gpu(
                    gc,
                    frames,
                    timesteps,
                    generation,
                    capacity_factor,
                    mode,
                )
            }
            OnnxProvider::Cuda => Ok(None),
        };
        match result {
            Ok(Some(fence)) => {
                INTERP_GPU_INPUT_FRAMES.fetch_add(1, Ordering::Relaxed);
                Ok(Some(fence))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                self.retire_interp_gpu_bridges(gc);
                self.interp_gpu_path_disabled = true;
                // Do not carry input-pack identity/cache state across a failed
                // shared-resource transaction. The fallback must rebuild the
                // next RIFE invocation exactly like a clean CPU-visible session.
                self.reset_interp_pack_cache();
                if matches!(
                    std::env::var("CHIDESCALER_INTERP_GPU").ok().as_deref(),
                    Some("require")
                ) {
                    Err(anyhow!(
                        "interpolation GPU residency required but unavailable: backend={} model={} reason={error:#}",
                        self.provider_desc,
                        self.name
                    ))
                } else {
                    INTERP_CPU_FALLBACK_FRAMES.fetch_add(1, Ordering::Relaxed);
                    log::warn!(
                        "interp-gpu-path-disabled: backend={} model={} reason={error:#}; safe non-resident transfer fallback enabled for this session",
                        self.provider_desc,
                        self.name
                    );
                    Ok(None)
                }
            }
        }
    }

    pub fn run_prepared_interp_gpu(&mut self, count: usize) -> Result<()> {
        let mut total_ms = 0.0;
        let mut padded = (0, 0);
        for index in 0..count {
            let (run_ms, slot_padded) = self.run_prepared_interp_gpu_slot(index)?;
            total_ms += run_ms;
            padded = slot_padded;
        }
        self.last_interp_profile = Some(InterpProfile {
            pack_ms: 0.0,
            run_ms: total_ms,
            out_ms: 0.0,
            padded_size: (padded.0 as usize, padded.1 as usize),
        });
        Ok(())
    }

    /// Execute and synchronize exactly one unique interpolation timestep.
    /// Returning each slot independently lets the render thread present t=0.2,
    /// 0.4, 0.6 and 0.8 as they become ready instead of waiting for the whole
    /// x5 batch before showing the first generated frame.
    pub(crate) fn run_prepared_interp_gpu_slot(
        &mut self,
        index: usize,
    ) -> Result<(f64, (i32, i32))> {
        let started = Instant::now();
        let padded = match self.provider {
            OnnxProvider::TensorRT => {
                let model_name = self.name.clone();
                let (size, run_call_ms, output_sync_ms) = {
                    let bridge = self
                        .tensorrt_interp_bridge
                        .as_mut()
                        .ok_or_else(|| anyhow!("TensorRT interpolation bridge missing"))?;
                    let invocation = bridge
                        .invocations
                        .get_mut(index)
                        .ok_or_else(|| anyhow!("TensorRT interpolation slot {index} missing"))?;
                    let run_started = Instant::now();
                    self.session
                        .run_binding_with_options(&invocation.binding, self.run_options.armed()?)
                        .map_err(oerr)?;
                    let run_call_ms = run_started.elapsed().as_secs_f64() * 1000.0;
                    let sync_started = Instant::now();
                    invocation.binding.synchronize_outputs().map_err(oerr)?;
                    let output_sync_ms = sync_started.elapsed().as_secs_f64() * 1000.0;
                    (bridge.size, run_call_ms, output_sync_ms)
                };
                let provider_total_ms = run_call_ms + output_sync_ms;
                if provider_total_ms >= 50.0 {
                    log::warn!(
                        "tensorrt-interp-slot-slow: model={} input={}x{} slot={} run_call_ms={:.2} output_sync_ms={:.2} provider_total_ms={:.2}",
                        model_name,
                        size.0,
                        size.1,
                        index + 1,
                        run_call_ms,
                        output_sync_ms,
                        provider_total_ms
                    );
                } else if provider_total_ms >= 15.0 {
                    log::debug!(
                        "tensorrt-interp-slot-timing: model={} input={}x{} slot={} run_call_ms={:.2} output_sync_ms={:.2} provider_total_ms={:.2}",
                        model_name,
                        size.0,
                        size.1,
                        index + 1,
                        run_call_ms,
                        output_sync_ms,
                        provider_total_ms
                    );
                }
                self.complete_successful_tensorrt_shape(size.0, size.1);
                self.tensorrt_interp_bridge
                    .as_ref()
                    .map(|bridge| bridge.padded_size)
                    .unwrap_or((0, 0))
            }
            OnnxProvider::DirectML => {
                let bridge = self
                    .dml_interp_bridge
                    .as_mut()
                    .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
                let invocation = bridge
                    .invocations
                    .get_mut(index)
                    .ok_or_else(|| anyhow!("DirectML interpolation slot {index} missing"))?;
                let mut provider_outputs = self
                    .session
                    .run_binding_with_options(&invocation.output.binding, self.run_options.armed()?)
                    .map_err(oerr)?;
                invocation
                    .output
                    .binding
                    .synchronize_outputs()
                    .map_err(oerr)?;
                if bridge.interop_mode != DmlInterpInteropMode::Direct {
                    if self.fp16 {
                        let provider = provider_outputs
                            .remove(self.out_name.as_str())
                            .ok_or_else(|| anyhow!("DirectML provider output missing"))?
                            .downcast::<TensorValueType<f16>>()?;
                        anyhow::ensure!(
                            provider.memory_info().allocation_device().as_str() == "DML",
                            "DirectML interpolation output silently fell back to CPU"
                        );
                        let provider_dims: Vec<i64> = provider.shape().iter().copied().collect();
                        anyhow::ensure!(
                            provider_dims == invocation.output.dims,
                            "DirectML provider output shape changed: {:?} != {:?}",
                            provider_dims,
                            invocation.output.dims
                        );
                        let api = dml_api()?;
                        let mut raw_resource = std::ptr::null_mut();
                        ort_status(unsafe {
                            (api.GetD3D12ResourceFromAllocation)(
                                invocation.output.dml_allocator.ptr().cast_mut(),
                                provider.data_ptr().cast_mut(),
                                &mut raw_resource,
                            )
                        })?;
                        let borrowed = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
                            .ok_or_else(|| anyhow!("DirectML provider output resource is null"))?;
                        let provider_resource = borrowed.clone();
                        let _ = invocation.output.copy.copy_and_wait(
                            &provider_resource,
                            &invocation.output.resource,
                            invocation.output.byte_len,
                        )?;
                        drop(provider);
                    } else {
                        let provider = provider_outputs
                            .remove(self.out_name.as_str())
                            .ok_or_else(|| anyhow!("DirectML provider output missing"))?
                            .downcast::<TensorValueType<f32>>()?;
                        anyhow::ensure!(
                            provider.memory_info().allocation_device().as_str() == "DML",
                            "DirectML interpolation output silently fell back to CPU"
                        );
                        let provider_dims: Vec<i64> = provider.shape().iter().copied().collect();
                        anyhow::ensure!(
                            provider_dims == invocation.output.dims,
                            "DirectML provider output shape changed: {:?} != {:?}",
                            provider_dims,
                            invocation.output.dims
                        );
                        let api = dml_api()?;
                        let mut raw_resource = std::ptr::null_mut();
                        ort_status(unsafe {
                            (api.GetD3D12ResourceFromAllocation)(
                                invocation.output.dml_allocator.ptr().cast_mut(),
                                provider.data_ptr().cast_mut(),
                                &mut raw_resource,
                            )
                        })?;
                        let borrowed = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
                            .ok_or_else(|| anyhow!("DirectML provider output resource is null"))?;
                        let provider_resource = borrowed.clone();
                        let _ = invocation.output.copy.copy_and_wait(
                            &provider_resource,
                            &invocation.output.resource,
                            invocation.output.byte_len,
                        )?;
                        drop(provider);
                    }
                }
                drop(provider_outputs);
                bridge.padded_size
            }
            OnnxProvider::Cuda => anyhow::bail!("unsupported interpolation backend"),
        };
        Ok((started.elapsed().as_secs_f64() * 1000.0, padded))
    }

    pub(crate) fn prepared_interp_gpu_outputs(
        &self,
        count: usize,
        preserve_float_after_dml: bool,
    ) -> Result<Vec<PreparedInterpGpuOutput>> {
        let (size, padded, keys) = match self.provider {
            OnnxProvider::TensorRT => {
                let bridge = self
                    .tensorrt_interp_bridge
                    .as_ref()
                    .ok_or_else(|| anyhow!("TensorRT interpolation bridge missing"))?;
                (
                    bridge.size,
                    bridge.padded_size,
                    bridge
                        .invocations
                        .iter()
                        .take(count)
                        .map(|invocation| invocation.output.key)
                        .collect::<Vec<_>>(),
                )
            }
            OnnxProvider::DirectML => {
                let bridge = self
                    .dml_interp_bridge
                    .as_ref()
                    .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
                (
                    bridge.size,
                    bridge.padded_size,
                    bridge
                        .invocations
                        .iter()
                        .take(count)
                        .map(|invocation| invocation.output.key)
                        .collect::<Vec<_>>(),
                )
            }
            OnnxProvider::Cuda => anyhow::bail!("unsupported interpolation backend"),
        };
        anyhow::ensure!(
            keys.len() == count,
            "interpolation output slots are incomplete"
        );
        Ok(keys
            .into_iter()
            .map(|key| PreparedInterpGpuOutput {
                key,
                size,
                padded,
                fp16: self.fp16,
                preserve_float: preserve_float_after_dml
                    && matches!(self.provider, OnnxProvider::DirectML),
            })
            .collect())
    }

    /// Export the persistent app-owned DirectML output buffer for a completed
    /// interpolation slot. Cross-GPU selected-compute handoff uses this, and
    /// v661 also uses it to skip a redundant DML -> OpenGL -> CPU -> Vulkan
    /// loop when DirectML and a forced Vulkan post-chain share the same GPU.
    /// DirectML and Vulkan must resolve to the same selected-GPU LUID.
    /// TensorRT/other providers return None and continue through their existing
    /// OpenGL handoff unchanged.
    pub(crate) fn export_prepared_interp_dml_shared_output(
        &self,
        output: PreparedInterpGpuOutput,
    ) -> Result<Option<PreparedDmlSharedOutput>> {
        if !matches!(self.provider, OnnxProvider::DirectML) {
            return Ok(None);
        }
        let direct = if let Some(invocation) = self.dml_interp_bridge.as_ref().and_then(|bridge| {
            bridge
                .invocations
                .iter()
                .find(|invocation| invocation.output.key == output.key)
        }) {
            &invocation.output
        } else if let Some(slot) = self
            .dml_cpu_interp_shared_outputs
            .iter()
            .flat_map(|bank| bank.iter())
            .find(|slot| slot.key == output.key)
        {
            slot
        } else {
            anyhow::bail!(
                "DirectML interpolation output slot {} is no longer available",
                output.key
            );
        };
        anyhow::ensure!(
            direct.input_size == output.padded,
            "DirectML shared output geometry changed: {:?} != {:?}",
            direct.input_size,
            output.padded
        );
        Ok(Some(PreparedDmlSharedOutput {
            key: direct.key,
            handle: direct.shared_handle()?,
            byte_len: direct.byte_len,
            heap_byte_len: direct.heap_byte_len,
            luid: direct.luid,
            size: output.size,
            padded: output.padded,
            fp16: output.fp16,
        }))
    }

    pub(crate) fn finish_prepared_interp_gpu_output(
        gc: &mut GlContext,
        output: PreparedInterpGpuOutput,
    ) -> Result<GpuTex> {
        let texture = match (output.fp16, output.preserve_float) {
            (true, true) => gc.external_nchw_f16_to_rgba16f_crop(
                output.key,
                output.size.0,
                output.size.1,
                output.padded.0,
                output.padded.1,
            ),
            (false, true) => gc.external_nchw_f32_to_rgba16f_crop(
                output.key,
                output.size.0,
                output.size.1,
                output.padded.0,
                output.padded.1,
            ),
            (true, false) => gc.external_nchw_f16_to_rgba8_crop(
                output.key,
                output.size.0,
                output.size.1,
                output.padded.0,
                output.padded.1,
            ),
            (false, false) => gc.external_nchw_f32_to_rgba8_crop(
                output.key,
                output.size.0,
                output.size.1,
                output.padded.0,
                output.padded.1,
            ),
        }
        .map_err(anyhow::Error::msg)?;
        INTERP_GPU_OUTPUT_FRAMES.fetch_add(1, Ordering::Relaxed);
        Ok(texture)
    }

    pub fn finish_prepared_interp_gpu(
        &mut self,
        gc: &mut GlContext,
        count: usize,
    ) -> Result<Vec<GpuTex>> {
        let slots = self.prepared_interp_gpu_outputs(count, false)?;
        let mut outputs = Vec::with_capacity(count);
        for slot in slots {
            outputs.push(Self::finish_prepared_interp_gpu_output(gc, slot)?);
        }
        Ok(outputs)
    }

    fn interp_gpu_layout(&self, size: (i32, i32)) -> Result<(usize, usize, (i32, i32))> {
        let (channels, frames, multiple) = match self.interp {
            InterpKind::RifeV1 { channels } => (channels, 2, RIFE_PAD_MULTIPLE as i32),
            InterpKind::SixChannel => (6, 2, 1),
            InterpKind::Drba => (14, 4, 64),
            _ => anyhow::bail!("unsupported GPU interpolation descriptor"),
        };
        let padded = (
            ((size.0 + multiple - 1) / multiple) * multiple,
            ((size.1 + multiple - 1) / multiple) * multiple,
        );
        Ok((channels, frames, padded))
    }

    fn interp_slot_count(factor: usize) -> usize {
        match factor {
            0 | 1 | 2 => 3,
            3 => 5,
            4 => 6,
            _ => 7,
        }
    }

    fn interp_bridge_needs_growth(existing_slots: usize, required_slots: usize) -> bool {
        existing_slots < required_slots
    }

    fn pack_interp_invocation(
        interp: &InterpKind,
        gc: &mut GlContext,
        input_key: u64,
        history_keys: &[u64],
        size: (i32, i32),
        padded: (i32, i32),
        channels: usize,
        timestep: f32,
        fp16: bool,
    ) -> Result<()> {
        let plane = usize::try_from(padded.0)? * usize::try_from(padded.1)?;
        for (index, history) in history_keys.iter().copied().enumerate() {
            if fp16 {
                gc.copy_external_nchw_f16(history, input_key, 0, index * 3 * plane, 3 * plane)
            } else {
                gc.copy_external_nchw_f32(history, input_key, 0, index * 3 * plane, 3 * plane)
            }
            .map_err(anyhow::Error::msg)?;
        }
        let aux = history_keys.len() * 3;
        if channels > aux {
            (if fp16 {
                gc.fill_interp_aux_f16(
                    input_key,
                    padded,
                    aux,
                    InterpAuxPlane::Constant(timestep.clamp(0.0, 1.0)),
                )
            } else {
                gc.fill_interp_aux_f32(
                    input_key,
                    padded,
                    aux,
                    InterpAuxPlane::Constant(timestep.clamp(0.0, 1.0)),
                )
            })
            .map_err(anyhow::Error::msg)?;
        }
        if *interp == InterpKind::Drba {
            (if fp16 {
                gc.fill_interp_aux_f16(input_key, padded, 13, InterpAuxPlane::Constant(1.0))
            } else {
                gc.fill_interp_aux_f32(input_key, padded, 13, InterpAuxPlane::Constant(1.0))
            })
            .map_err(anyhow::Error::msg)?;
        } else if channels >= aux + 3 {
            (if fp16 {
                gc.fill_interp_aux_f16(input_key, padded, aux + 1, InterpAuxPlane::HorizontalGrid)
            } else {
                gc.fill_interp_aux_f32(input_key, padded, aux + 1, InterpAuxPlane::HorizontalGrid)
            })
            .map_err(anyhow::Error::msg)?;
            (if fp16 {
                gc.fill_interp_aux_f16(input_key, padded, aux + 2, InterpAuxPlane::VerticalGrid)
            } else {
                gc.fill_interp_aux_f32(input_key, padded, aux + 2, InterpAuxPlane::VerticalGrid)
            })
            .map_err(anyhow::Error::msg)?;
            if channels >= aux + 5 {
                (if fp16 {
                    gc.fill_interp_aux_f16(
                        input_key,
                        padded,
                        aux + 3,
                        InterpAuxPlane::Constant(2.0 / (padded.0 - 1).max(1) as f32),
                    )
                } else {
                    gc.fill_interp_aux_f32(
                        input_key,
                        padded,
                        aux + 3,
                        InterpAuxPlane::Constant(2.0 / (padded.0 - 1).max(1) as f32),
                    )
                })
                .map_err(anyhow::Error::msg)?;
                (if fp16 {
                    gc.fill_interp_aux_f16(
                        input_key,
                        padded,
                        aux + 4,
                        InterpAuxPlane::Constant(2.0 / (padded.1 - 1).max(1) as f32),
                    )
                } else {
                    gc.fill_interp_aux_f32(
                        input_key,
                        padded,
                        aux + 4,
                        InterpAuxPlane::Constant(2.0 / (padded.1 - 1).max(1) as f32),
                    )
                })
                .map_err(anyhow::Error::msg)?;
            }
        }
        let _ = size;
        Ok(())
    }

    /// Synthesize an in-between frame at time `t` (0..1 between the last two
    /// frames). `frames` are RGB8, oldest first, all the same size.
    pub fn process_interp(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        t: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        anyhow::ensure!(!frames.is_empty(), "no frames");
        // Backend-switch warmup uses this CPU-visible interpolation path even
        // when the provider is TensorRT. Announce the real RIFE/DRBA run here
        // so stacked-model progress advances from "waiting" to a timed second
        // engine instead of completing with a synthetic 0.0 ms duration.
        self.prepare_tensorrt_input_shape(w, h);
        mark_tensorrt_model_started(&self.name);
        let result = if let InterpKind::RifeV1 { channels } = self.interp.clone() {
            self.rife_multi(w, h, frames, channels, t)
        } else if self.interp == InterpKind::Drba {
            anyhow::ensure!(frames.len() >= 4, "DRBA needs 4 frames");
            self.drba(w, h, frames, t)
        } else {
            let cur = frames[frames.len() - 1];
            let prev = frames[frames.len().saturating_sub(2)];
            self.process_pair_t(w, h, prev, cur, t)
        };
        if result.is_ok() {
            self.complete_successful_tensorrt_shape(w, h);
        }
        result
    }

    /// Same as `process_interp`, but input frames are tightly packed RGBA8.
    /// This avoids a separate RGBA->RGB staging copy in the live RIFE path.
    pub fn process_interp_rgba8(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        t: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        anyhow::ensure!(!frames.is_empty(), "no frames");
        self.prepare_tensorrt_input_shape(w, h);
        mark_tensorrt_model_started(&self.name);
        let result = if let InterpKind::RifeV1 { channels } = self.interp.clone() {
            self.rife_multi_strided(w, h, frames, channels, t, 4)
        } else if self.interp == InterpKind::Drba {
            anyhow::ensure!(frames.len() >= 4, "DRBA needs 4 frames");
            self.drba_strided(w, h, frames, t, 4)
        } else {
            anyhow::bail!("RGBA direct interpolation is only supported for RIFE/DRBA");
        };
        if result.is_ok() {
            self.complete_successful_tensorrt_shape(w, h);
        }
        result
    }

    /// Generate all requested interpolation times. DRBA has a dynamic batch
    /// dimension, so x3/x4 can share one DirectML dispatch instead of running
    /// the same graph two or three times. Fixed-batch models keep the validated
    /// sequential path.
    pub fn process_interp_many_rgba8(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        ts: &[f32],
    ) -> Result<Vec<(i32, i32, Vec<u8>)>> {
        if self.interp == InterpKind::Drba && ts.len() > 1 {
            self.prepare_tensorrt_input_shape(w, h);
            mark_tensorrt_model_started(&self.name);
            let result = self.drba_batch_strided(w, h, frames, ts, 4);
            if result.is_ok() {
                self.complete_successful_tensorrt_shape(w, h);
            }
            return result;
        }
        ts.iter()
            .map(|t| self.process_interp_rgba8(w, h, frames, *t))
            .collect()
    }

    /// Cross-GPU DirectML -> Vulkan handoff for RIFE. Input packing intentionally
    /// remains CPU-visible because the capture/OpenGL GPU may be a different
    /// adapter. Only the expensive provider output boundary is made resident:
    /// DirectML writes on the selected GPU, a D3D12 GPU copy moves the provider
    /// tensor into an app-owned shareable buffer, and Vulkan imports that buffer
    /// on the same LUID. No DirectML output readback or Vulkan re-upload occurs.
    pub(crate) fn process_interp_many_rgba8_dml_shared(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        ts: &[f32],
        output_bank: usize,
    ) -> Result<Option<Vec<PreparedInterpGpuOutput>>> {
        if self.provider != OnnxProvider::DirectML
            || self.dml_cpu_interp_shared_disabled
            || ts.is_empty()
        {
            return Ok(None);
        }
        let InterpKind::RifeV1 { channels } = self.interp.clone() else {
            return Ok(None);
        };
        let output_bank = output_bank % 2;
        let attempt = (|| -> Result<Vec<PreparedInterpGpuOutput>> {
            anyhow::ensure!(!frames.is_empty(), "no frames");
            anyhow::ensure!(channels >= 3 * frames.len() + 1, "model wants more frames");
            let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
            anyhow::ensure!(wu > 0 && hu > 0, "empty frame");
            let pw = wu.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE;
            let ph = hu.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE;
            let pn = pw * ph;
            let aux = 3 * frames.len();
            let padded = (i32::try_from(pw)?, i32::try_from(ph)?);
            if self
                .dml_cpu_interp_shared_outputs
                .iter()
                .flat_map(|bank| bank.iter())
                .any(|slot| slot.input_size != padded)
            {
                for bank in &mut self.dml_cpu_interp_shared_outputs {
                    bank.clear();
                }
            }

            let pack_started = Instant::now();
            let mut total_run_ms = 0.0f64;
            let mut total_copy_ms = 0.0f64;
            let mut prepared = Vec::with_capacity(ts.len());
            let shape = vec![1i64, channels as i64, ph as i64, pw as i64];

            if self.fp16 {
                prepare_len_f16(&mut self.scratch_f16, pn * channels);
                let x = self.scratch_f16.as_mut_slice();
                fill_padded_rgb_planes_f16(x, frames, frames.len(), wu, hu, pw, ph, pn, 4);
                if channels >= aux + 3 {
                    let row: Vec<f16> = (0..pw)
                        .map(|j| f16::from_f32(2.0 * j as f32 / (pw - 1) as f32 - 1.0))
                        .collect();
                    use rayon::prelude::*;
                    x[(aux + 1) * pn..(aux + 2) * pn]
                        .par_chunks_mut(pw)
                        .for_each(|dst| dst.copy_from_slice(&row));
                    x[(aux + 2) * pn..(aux + 3) * pn]
                        .par_chunks_mut(pw)
                        .enumerate()
                        .for_each(|(i, row)| {
                            row.fill(f16::from_f32(2.0 * i as f32 / (ph - 1) as f32 - 1.0));
                        });
                }
                if channels >= aux + 5 {
                    x[(aux + 3) * pn..(aux + 4) * pn].fill(f16::from_f32(2.0 / (pw - 1) as f32));
                    x[(aux + 4) * pn..(aux + 5) * pn].fill(f16::from_f32(2.0 / (ph - 1) as f32));
                }
                for (index, timestep) in ts.iter().copied().enumerate() {
                    x[aux * pn..(aux + 1) * pn].fill(f16::from_f32(timestep.clamp(0.0, 1.0)));
                    let tin = TensorRef::from_array_view((shape.clone(), &*x)).map_err(oerr)?;
                    if self.dml_cpu_interp_shared_outputs[output_bank].len() <= index {
                        let output = create_direct_output(
                            &mut self.session,
                            &self.run_options,
                            &self.in_name,
                            &self.out_name,
                            &tin,
                            padded,
                        )?;
                        self.dml_cpu_interp_shared_outputs[output_bank].push(output);
                    }
                    let output = &mut self.dml_cpu_interp_shared_outputs[output_bank][index];
                    anyhow::ensure!(
                        output.dims.len() == 4
                            && output.dims[0] == 1
                            && output.dims[1] >= 3
                            && output.dims[2] == ph as i64
                            && output.dims[3] == pw as i64,
                        "DirectML shared RIFE output shape {:?} does not match padded {}x{}",
                        output.dims,
                        pw,
                        ph
                    );
                    let (run_ms, copy_ms) = run_cpu_tensor_to_dml_shared_output(
                        &mut self.session,
                        &self.run_options,
                        &self.in_name,
                        &self.out_name,
                        &tin,
                        output,
                    )?;
                    total_run_ms += run_ms;
                    total_copy_ms += copy_ms;
                    prepared.push(PreparedInterpGpuOutput {
                        key: output.key,
                        size: (w, h),
                        padded,
                        fp16: true,
                        preserve_float: true,
                    });
                }
            } else {
                prepare_len_f32(&mut self.scratch_f32, pn * channels);
                let x = self.scratch_f32.as_mut_slice();
                fill_padded_rgb_planes_f32(x, frames, frames.len(), wu, hu, pw, ph, pn, 4);
                if channels >= aux + 3 {
                    let row: Vec<f32> = (0..pw)
                        .map(|j| 2.0 * j as f32 / (pw - 1) as f32 - 1.0)
                        .collect();
                    use rayon::prelude::*;
                    x[(aux + 1) * pn..(aux + 2) * pn]
                        .par_chunks_mut(pw)
                        .for_each(|dst| dst.copy_from_slice(&row));
                    x[(aux + 2) * pn..(aux + 3) * pn]
                        .par_chunks_mut(pw)
                        .enumerate()
                        .for_each(|(i, row)| row.fill(2.0 * i as f32 / (ph - 1) as f32 - 1.0));
                }
                if channels >= aux + 5 {
                    x[(aux + 3) * pn..(aux + 4) * pn].fill(2.0 / (pw - 1) as f32);
                    x[(aux + 4) * pn..(aux + 5) * pn].fill(2.0 / (ph - 1) as f32);
                }
                for (index, timestep) in ts.iter().copied().enumerate() {
                    x[aux * pn..(aux + 1) * pn].fill(timestep.clamp(0.0, 1.0));
                    let tin = TensorRef::from_array_view((shape.clone(), &*x)).map_err(oerr)?;
                    if self.dml_cpu_interp_shared_outputs[output_bank].len() <= index {
                        let output = create_direct_output(
                            &mut self.session,
                            &self.run_options,
                            &self.in_name,
                            &self.out_name,
                            &tin,
                            padded,
                        )?;
                        self.dml_cpu_interp_shared_outputs[output_bank].push(output);
                    }
                    let output = &mut self.dml_cpu_interp_shared_outputs[output_bank][index];
                    anyhow::ensure!(
                        output.dims.len() == 4
                            && output.dims[0] == 1
                            && output.dims[1] >= 3
                            && output.dims[2] == ph as i64
                            && output.dims[3] == pw as i64,
                        "DirectML shared RIFE output shape {:?} does not match padded {}x{}",
                        output.dims,
                        pw,
                        ph
                    );
                    let (run_ms, copy_ms) = run_cpu_tensor_to_dml_shared_output(
                        &mut self.session,
                        &self.run_options,
                        &self.in_name,
                        &self.out_name,
                        &tin,
                        output,
                    )?;
                    total_run_ms += run_ms;
                    total_copy_ms += copy_ms;
                    prepared.push(PreparedInterpGpuOutput {
                        key: output.key,
                        size: (w, h),
                        padded,
                        fp16: false,
                        preserve_float: true,
                    });
                }
            }
            self.last_interp_profile = Some(InterpProfile {
                pack_ms: pack_started.elapsed().as_secs_f64() * 1000.0
                    - total_run_ms
                    - total_copy_ms,
                run_ms: total_run_ms,
                out_ms: total_copy_ms,
                padded_size: (pw, ph),
            });
            Ok(prepared)
        })();

        match attempt {
            Ok(prepared) => Ok(Some(prepared)),
            Err(error) => {
                self.dml_cpu_interp_shared_disabled = true;
                for bank in &mut self.dml_cpu_interp_shared_outputs {
                    bank.clear();
                }
                log::warn!(
                    "dml-vulkan-resident-handoff: action=disable reason={error:#} fallback=cpu-visible-interpolation"
                );
                Ok(None)
            }
        }
    }

    fn complete_successful_tensorrt_shape(&self, w: i32, h: i32) {
        if self.provider == OnnxProvider::TensorRT && self.complete_tensorrt_input_shape(w, h) {
            mark_tensorrt_model_completed(&self.name);
        }
    }

    /// Frame interpolation between `prev` and `cur` at time `t`.
    fn process_pair_t(
        &mut self,
        w: i32,
        h: i32,
        prev: &[u8],
        cur: &[u8],
        t: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        match self.interp.clone() {
            InterpKind::None => self.process(w, h, cur),
            InterpKind::RifeV1 { channels } => self.rife_multi(w, h, &[prev, cur], channels, t),
            InterpKind::Drba => self.drba(w, h, &[prev, prev, cur, cur], t),
            InterpKind::SixChannel => {
                // concat as 6ch NCHW = [prev(3), cur(3)]
                let (wu, hu) = (w as usize, h as usize);
                let n = wu * hu;
                let mut x = vec![0f32; n * 6];
                for (ci, src) in [(0usize, prev), (3, cur)] {
                    for c in 0..3 {
                        let plane = &mut x[(ci + c) * n..(ci + c + 1) * n];
                        for i in 0..n {
                            plane[i] = src[i * 3 + c] as f32 * (1.0 / 255.0);
                        }
                    }
                }
                let t = Tensor::from_array((vec![1i64, 6, h as i64, w as i64], x)).map_err(oerr)?;
                let outputs = self
                    .session
                    .run_with_options(
                        ort::inputs![self.in_name.as_str() => t],
                        self.run_options.armed()?,
                    )
                    .map_err(oerr)?;
                let out = outputs
                    .get(self.out_name.as_str())
                    .ok_or_else(|| anyhow!("no output"))?;
                let (s, v) = out.try_extract_tensor::<f32>().map_err(oerr)?;
                let dims: Vec<i64> = s[..].to_vec();
                let (ow, oh) = out_dims(&dims)?;
                let on = ow * oh;
                let mut rgb8 = vec![0u8; on * 3];
                for c in 0..3 {
                    let plane = &v[c * on..(c + 1) * on];
                    for i in 0..on {
                        rgb8[i * 3 + c] = (plane[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
                Ok((ow as i32, oh as i32, rgb8))
            }
            InterpKind::TwoInputs { second, timestep } => {
                let _ = t; // TwoInputs models take t via the timestep input below
                let (wu, hu) = (w as usize, h as usize);
                let n = wu * hu;
                let to_nchw = |src: &[u8]| {
                    let mut x = vec![0f32; n * 3];
                    for c in 0..3 {
                        for i in 0..n {
                            x[c * n + i] = src[i * 3 + c] as f32 * (1.0 / 255.0);
                        }
                    }
                    x
                };
                let shape = vec![1i64, 3, h as i64, w as i64];
                let t0 = Tensor::from_array((shape.clone(), to_nchw(prev))).map_err(oerr)?;
                let t1 = Tensor::from_array((shape, to_nchw(cur))).map_err(oerr)?;
                let outputs = if let Some(ts) = timestep {
                    let tt = Tensor::from_array((vec![1i64], vec![t])).map_err(oerr)?;
                    self.session
                        .run_with_options(
                            ort::inputs![
                                self.in_name.as_str() => t0,
                                second.as_str() => t1,
                                ts.as_str() => tt
                            ],
                            self.run_options.armed()?,
                        )
                        .map_err(oerr)?
                } else {
                    self.session
                        .run_with_options(
                            ort::inputs![
                                self.in_name.as_str() => t0,
                                second.as_str() => t1
                            ],
                            self.run_options.armed()?,
                        )
                        .map_err(oerr)?
                };
                let out = outputs
                    .get(self.out_name.as_str())
                    .ok_or_else(|| anyhow!("no output"))?;
                let (s, v) = out.try_extract_tensor::<f32>().map_err(oerr)?;
                let dims: Vec<i64> = s[..].to_vec();
                let (ow, oh) = out_dims(&dims)?;
                let on = ow * oh;
                let mut rgb8 = vec![0u8; on * 3];
                for c in 0..3 {
                    let plane = &v[c * on..(c + 1) * on];
                    for i in 0..on {
                        rgb8[i * 3 + c] = (plane[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
                Ok((ow as i32, oh as i32, rgb8))
            }
        }
    }

    /// RGB8 (tightly packed) in -> RGB8 out with new dimensions.
    pub fn process(&mut self, w: i32, h: i32, rgb: &[u8]) -> Result<(i32, i32, Vec<u8>)> {
        self.process_u8_strided(w, h, rgb, 3, 3)
    }

    /// Captured RGBA8 -> RGB model output without first uploading to OpenGL
    /// and reading the same frame back. This removes a GPU synchronization
    /// point for chains whose first stage is an image-upscaling ONNX model.
    pub fn process_rgba8(&mut self, w: i32, h: i32, rgba: &[u8]) -> Result<(i32, i32, Vec<u8>)> {
        self.process_u8_strided(w, h, rgba, 4, 3)
    }

    /// Captured RGBA8 -> RGBA8 model output. Keeping four output components
    /// avoids the driver's costly RGB-to-RGBA expansion during the following
    /// OpenGL upload.
    pub fn process_rgba8_native_output(
        &mut self,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<(i32, i32, Vec<u8>)> {
        self.process_u8_strided(w, h, rgba, 4, 4)
    }

    /// NeoAccel fast path for the first ordinary image model. It returns
    /// `None` whenever the frame is unsuitable, so the caller can immediately
    /// execute the unchanged GPU-direct DirectML path.
    pub fn try_process_rgba8_neo_texture(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(i32, i32, GpuTex)>> {
        let Some(mut accel) = self.neo_accel.take() else {
            return Ok(None);
        };
        let result = self.try_process_rgba8_neo_texture_inner(gc, w, h, rgba, &mut accel);
        if let Err(error) = &result {
            log::warn!(
                "NeoAccel disabled for '{}'; stable DirectML path restored: {error:#}",
                self.name
            );
            gc.remove_persistent_texture(&accel.texture_key);
            accel.disable_permanently();
        }
        self.neo_accel = Some(accel);
        match result {
            Ok(output) => Ok(output),
            Err(_) => Ok(None),
        }
    }

    fn try_process_rgba8_neo_texture_inner(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
        accel: &mut NeoAccelState,
    ) -> Result<Option<(i32, i32, GpuTex)>> {
        let (width, height) = (usize::try_from(w)?, usize::try_from(h)?);
        let decision = accel.decide(width, height, rgba);
        match decision {
            NeoDecision::Bypass => {
                if !accel.active && accel.cached_output.is_empty() {
                    gc.remove_persistent_texture(&accel.texture_key);
                }
                Ok(None)
            }
            NeoDecision::EstablishBaseline => {
                let (ow, oh, output) = self.process_u8_strided(w, h, rgba, 4, 4)?;
                let scale = exact_integer_scale(width, height, ow, oh)?;
                anyhow::ensure!(
                    scale == accel.plan.scale_hint,
                    "NeoAccel scale mismatch: graph hinted {}, runtime returned {}",
                    accel.plan.scale_hint,
                    scale
                );
                accel.cached_output = output;
                accel.output_size = Some((ow as usize, oh as usize));
                accel.active = true;
                accel.commit_input(rgba);
                self.retire_direct_output(gc);
                let texture = gc
                    .update_persistent_rgba8(&accel.texture_key, ow, oh, &accel.cached_output)
                    .map_err(|error| anyhow!(error))?;
                log::info!(
                    "NeoAccel baseline armed: model='{}' input={}x{} output={}x{} halo={}x{}",
                    self.name,
                    w,
                    h,
                    ow,
                    oh,
                    accel.plan.halo_x,
                    accel.plan.halo_y
                );
                Ok(Some((ow, oh, texture)))
            }
            NeoDecision::Reuse => {
                let Some((ow, oh)) = accel.output_size else {
                    accel.active = false;
                    return Ok(None);
                };
                let texture =
                    if let Some(texture) = gc.persistent_texture_by_key(&accel.texture_key) {
                        texture
                    } else {
                        gc.update_persistent_rgba8(
                            &accel.texture_key,
                            ow as i32,
                            oh as i32,
                            &accel.cached_output,
                        )
                        .map_err(|error| anyhow!(error))?
                    };
                Ok(Some((ow as i32, oh as i32, texture)))
            }
            NeoDecision::Patch(plan) => {
                let crop = crop_rgba8(rgba, width, plan.crop)?;
                let (crop_ow, crop_oh, crop_output) = self.process_u8_strided(
                    plan.crop.width() as i32,
                    plan.crop.height() as i32,
                    &crop,
                    4,
                    4,
                )?;
                let scale =
                    exact_integer_scale(plan.crop.width(), plan.crop.height(), crop_ow, crop_oh)?;
                let Some((full_ow, full_oh)) = accel.output_size else {
                    accel.active = false;
                    accel.cached_output.clear();
                    return Ok(None);
                };
                anyhow::ensure!(
                    full_ow == width.saturating_mul(scale)
                        && full_oh == height.saturating_mul(scale),
                    "NeoAccel cached output scale changed"
                );
                let patch = extract_output_patch(
                    &crop_output,
                    crop_ow as usize,
                    scale,
                    plan.crop,
                    plan.affected,
                )?;

                if accel.geometry_needs_proof(plan.geometry) {
                    // Require repeated quality-equivalence proofs and then
                    // re-check periodically. This catches model/driver behavior
                    // that depends on content rather than shape alone.
                    let (proof_ow, proof_oh, proof) = self.process_u8_strided(w, h, rgba, 4, 4)?;
                    anyhow::ensure!(
                        proof_ow as usize == full_ow && proof_oh as usize == full_oh,
                        "NeoAccel proof output dimensions changed"
                    );
                    let proof_report = compare_output_patch(
                        &proof,
                        full_ow,
                        scale,
                        plan.affected,
                        &patch,
                        accel.proof_tolerance,
                        accel.proof_max_mismatch_ratio,
                    );
                    if !proof_report.accepted {
                        log::warn!(
                            "NeoAccel quality proof failed: model='{}' crop={}x{} mismatched={}/{} max_delta={}; this model remains on full-frame DirectML",
                            self.name,
                            plan.crop.width(),
                            plan.crop.height(),
                            proof_report.mismatched_channels,
                            proof_report.compared_channels,
                            proof_report.max_delta
                        );
                        accel.disable_permanently();
                        accel.commit_input(rgba);
                        gc.remove_persistent_texture(&accel.texture_key);
                        self.retire_direct_output(gc);
                        let texture = gc.upload_rgba8(proof_ow, proof_oh, &proof);
                        return Ok(Some((proof_ow, proof_oh, texture)));
                    }
                    accel.record_geometry_proof(plan.geometry);
                    accel.cached_output = proof;
                    accel.output_size = Some((proof_ow as usize, proof_oh as usize));
                    accel.commit_input(rgba);
                    self.retire_direct_output(gc);
                    let texture = gc
                        .update_persistent_rgba8(
                            &accel.texture_key,
                            proof_ow,
                            proof_oh,
                            &accel.cached_output,
                        )
                        .map_err(|error| anyhow!(error))?;
                    log::info!(
                        "NeoAccel quality proof passed: model='{}' crop={}x{} affected={}x{} mismatched={}/{} max_delta={}",
                        self.name,
                        plan.crop.width(),
                        plan.crop.height(),
                        plan.affected.width(),
                        plan.affected.height(),
                        proof_report.mismatched_channels,
                        proof_report.compared_channels,
                        proof_report.max_delta
                    );
                    return Ok(Some((proof_ow, proof_oh, texture)));
                }

                accel.record_geometry_use(plan.geometry);
                write_output_patch(
                    &mut accel.cached_output,
                    full_ow,
                    scale,
                    plan.affected,
                    &patch,
                )?;
                accel.commit_input(rgba);
                self.retire_direct_output(gc);
                let patch_x = i32::try_from(plan.affected.x0.saturating_mul(scale))?;
                let patch_y = i32::try_from(plan.affected.y0.saturating_mul(scale))?;
                let patch_w = i32::try_from(plan.affected.width().saturating_mul(scale))?;
                let patch_h = i32::try_from(plan.affected.height().saturating_mul(scale))?;
                let texture = if gc.persistent_texture_by_key(&accel.texture_key).is_some() {
                    gc.update_persistent_rgba8_region(
                        &accel.texture_key,
                        patch_x,
                        patch_y,
                        patch_w,
                        patch_h,
                        &patch,
                    )
                    .map_err(|error| anyhow!(error))?
                } else {
                    gc.update_persistent_rgba8(
                        &accel.texture_key,
                        full_ow as i32,
                        full_oh as i32,
                        &accel.cached_output,
                    )
                    .map_err(|error| anyhow!(error))?
                };
                Ok(Some((full_ow as i32, full_oh as i32, texture)))
            }
        }
    }

    /// GPU-resident ordinary image-model path used between GLSL/ONNX stages.
    /// Unsupported models and an explicit opt-out preserve the CPU path.
    pub fn process_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<Option<GpuTex>> {
        self.process_gpu_texture_with_context(gc, input_texture, false)
    }

    /// Fast path for an ordinary ONNX stage that is executed *after* a GPU
    /// interpolation stage.  The stable v440 path downloaded the interpolated
    /// texture to RGB8, packed it again on the CPU, ran DirectML, converted the
    /// result back to RGB8 and uploaded it to GL.  That round-trip is especially
    /// expensive for RIFE/DRBA + ONNX upscaling.
    ///
    /// On non-conservative GL drivers we instead auto-select the already
    /// validated provider-owned-copy bridge: GL packs directly into an app-owned
    /// D3D12 input buffer, DirectML keeps ownership of its native output, and a
    /// D3D12 copy places the completed tensor in the GL-imported output buffer.
    /// Any setup/runtime failure falls back to the unchanged CPU handoff in the
    /// same frame.  Conservative drivers (currently AMD/ATI in GlContext) retain
    /// the v440 path unless CHIDESCALER_DML_GL_INTEROP explicitly opts in.
    pub fn process_gpu_texture_after_interpolation(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<Option<GpuTex>> {
        self.process_gpu_texture_with_context(gc, input_texture, true)
    }

    fn process_gpu_texture_with_context(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
        after_interpolation: bool,
    ) -> Result<Option<GpuTex>> {
        match self.provider {
            OnnxProvider::DirectML => {
                self.process_dml_gpu_texture(gc, input_texture, after_interpolation)
            }
            OnnxProvider::TensorRT => self.process_tensorrt_gpu_texture(gc, input_texture),
            OnnxProvider::Cuda => Ok(None),
        }
    }

    fn process_dml_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
        after_interpolation: bool,
    ) -> Result<Option<GpuTex>> {
        if self.temporal_frames.is_some()
            && self.interp == InterpKind::None
            && self.fp16
            && !self.direct_output_disabled
            && gc.external_device_luid().is_some()
        {
            return match self.process_dml_temporal_gpu_texture(gc, input_texture) {
                Ok(texture) => Ok(Some(texture)),
                Err(error) => {
                    self.retire_dml_temporal_gpu_bridge(gc);
                    self.direct_output_disabled = true;
                    log::warn!(
                        "dml-temporal-gpu-path disabled for '{}'; CPU temporal fallback restored: {error:#}",
                        self.name
                    );
                    Ok(None)
                }
            };
        }
        // The shared-input route stays opt-in for ordinary chains.  For the
        // specific interpolation -> ONNX case, v441 can automatically use the
        // safer provider-owned-copy bridge on non-conservative GL drivers.
        // This avoids the CPU readback/repack/output-conversion round-trip that
        // made a ~9-10 ms DirectML upscaler grow to ~16-17 ms when stacked after
        // RIFE in the v440 diagnostics.  Explicit environment settings always
        // win, including an explicit 0/off opt-out.
        let configured_interop = std::env::var("CHIDESCALER_DML_GL_INTEROP").unwrap_or_default();
        let interop_mode = resolve_dml_gl_interop_mode(
            &configured_interop,
            after_interpolation,
            gc.external_interop_conservative_recommended(),
        );
        let Some(interop_mode) = interop_mode else {
            return Ok(None);
        };
        if !self.fp16
            || self.input_channels != Some(3)
            || self.interp != InterpKind::None
            || self.temporal_frames.is_some()
            || self.direct_output_disabled
            || gc.external_device_luid().is_none()
            || (input_texture.w() as i64 * input_texture.h() as i64 * 3) & 1 != 0
        {
            return Ok(None);
        }
        match self.process_gpu_texture_inner(gc, input_texture, interop_mode) {
            Ok(texture) => Ok(Some(texture)),
            Err(error) => {
                self.retire_direct_output(gc);
                self.direct_output_disabled = true;
                log::warn!(
                    "ONNX GPU-resident path disabled for '{}'; CPU fallback restored: {error:#}",
                    self.name
                );
                Ok(None)
            }
        }
    }

    fn process_dml_temporal_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<GpuTex> {
        let frames = self
            .temporal_frames
            .ok_or_else(|| anyhow!("not a temporal model"))?;
        let size = (input_texture.w(), input_texture.h());
        let slot_elements = usize::try_from(size.0)? * usize::try_from(size.1)? * 3;
        anyhow::ensure!(slot_elements % 2 == 0, "unaligned DirectML temporal plane");
        if self
            .dml_temporal_bridge
            .as_ref()
            .is_some_and(|bridge| bridge.size != size || bridge.frame_count != frames)
        {
            self.retire_dml_temporal_gpu_bridge(gc);
        }
        if self.dml_temporal_bridge.is_none() {
            let channels = frames * 3;
            let zero = vec![f16::ZERO; slot_elements * frames];
            let discovery = TensorRef::from_array_view((
                Shape::new([1i64, channels as i64, size.1 as i64, size.0 as i64]),
                zero.as_slice(),
            ))
            .map_err(oerr)?;
            let output = create_direct_output(
                &mut self.session,
                &self.run_options,
                &self.in_name,
                &self.out_name,
                &discovery,
                size,
            )?;
            anyhow::ensure!(
                output.dims == vec![1, 3, size.1 as i64, size.0 as i64],
                "unexpected DirectML temporal output {:?}",
                output.dims
            );
            let packed_input =
                create_direct_input_channels(&output.device, output.luid, size, channels)?;
            let mut history = Vec::with_capacity(frames);
            for _ in 0..frames {
                history.push(create_direct_input(&output.device, output.luid, size)?);
            }
            let mut imported_keys = Vec::with_capacity(history.len() + 2);
            let import_result = (|| -> Result<()> {
                for buffer in history.iter().chain(std::iter::once(&packed_input)) {
                    gc.import_external_d3d12_buffer(
                        buffer.key,
                        buffer.shared_handle()?,
                        buffer.byte_len,
                        buffer.heap_byte_len,
                        buffer.luid,
                    )
                    .map_err(anyhow::Error::msg)?;
                    imported_keys.push(buffer.key);
                }
                gc.import_external_d3d12_buffer(
                    output.key,
                    output.shared_handle()?,
                    output.byte_len,
                    output.heap_byte_len,
                    output.luid,
                )
                .map_err(anyhow::Error::msg)?;
                imported_keys.push(output.key);
                Ok(())
            })();
            if let Err(error) = import_result {
                for key in imported_keys.into_iter().rev() {
                    gc.clear_external_buffer_key(key);
                }
                return Err(
                    error.context("DirectML temporal shared-buffer import transaction rolled back")
                );
            }
            self.dml_temporal_bridge = Some(DmlTemporalGpuBridge {
                history,
                packed_input,
                output,
                size,
                frame_count: frames,
                write_index: 0,
                valid_frames: 0,
            });
            log::info!(
                "dml-temporal-gpu-path-active: model={} frames={} channels={} input={}x{}",
                self.name,
                frames,
                channels,
                size.0,
                size.1
            );
        }
        let bridge = self.dml_temporal_bridge.as_mut().unwrap();
        let pack_start = Instant::now();
        let slot = bridge.write_index;
        gc.rgba_texture_to_external_nchw_f16(
            bridge.history[slot].key,
            input_texture,
            size.0,
            size.1,
        )
        .map_err(anyhow::Error::msg)?;
        bridge.write_index = (bridge.write_index + 1) % frames;
        bridge.valid_frames = (bridge.valid_frames + 1).min(frames);
        for (destination, source_slot) in
            temporal_history_slot_order(frames, bridge.valid_frames, bridge.write_index)
                .into_iter()
                .enumerate()
        {
            gc.copy_external_nchw_f16(
                bridge.history[source_slot].key,
                bridge.packed_input.key,
                0,
                destination * slot_elements,
                slot_elements,
            )
            .map_err(anyhow::Error::msg)?;
        }
        gc.finish();
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        gc.wait_external_buffer_idle(bridge.output.key)
            .map_err(anyhow::Error::msg)?;
        let memory = MemoryInfo::new(
            AllocationDevice::DIRECTML,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let input_value = unsafe {
            TensorRefMut::<f16>::from_raw(
                memory.clone(),
                bridge.packed_input.allocation,
                Shape::new(bridge.packed_input.dims),
            )?
        };
        let output_value = unsafe {
            TensorRefMut::<f16>::from_raw(
                memory,
                bridge.output.allocation,
                Shape::new(bridge.output.dims.iter().copied()),
            )?
        };
        let input_name = CString::new(self.in_name.as_bytes())?;
        let output_name = CString::new(self.out_name.as_bytes())?;
        ort_status(unsafe {
            (ort::api().BindInput)(
                bridge.output.binding.ptr().cast_mut(),
                input_name.as_ptr(),
                input_value.ptr(),
            )
        })?;
        ort_status(unsafe {
            (ort::api().BindOutput)(
                bridge.output.binding.ptr().cast_mut(),
                output_name.as_ptr(),
                output_value.ptr(),
            )
        })?;
        let run_start = Instant::now();
        let run_result = self
            .session
            .run_binding_with_options(&bridge.output.binding, self.run_options.armed()?)
            .map(|_| ())
            .map_err(oerr);
        if run_result.is_ok() {
            bridge.output.binding.synchronize_outputs().map_err(oerr)?;
        }
        bridge.output.binding.clear();
        run_result?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let out_start = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(bridge.output.key, size.0, size.1)
            .map_err(anyhow::Error::msg)?;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: out_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (size.0 as usize, size.1 as usize),
        });
        self.last_upscale_input_size = Some(size);
        Ok(texture)
    }

    pub(crate) fn retire_dml_temporal_gpu_bridge(&mut self, gc: &mut GlContext) {
        let Some(mut bridge) = self.dml_temporal_bridge.take() else {
            return;
        };
        gc.finish();
        bridge.output.binding.clear();
        for slot in &bridge.history {
            gc.clear_external_buffer_key(slot.key);
        }
        gc.clear_external_buffer_key(bridge.packed_input.key);
        gc.clear_external_buffer_key(bridge.output.key);
    }

    fn process_tensorrt_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<Option<GpuTex>> {
        let enabled = tensorrt_gpu_interop_enabled();
        if !enabled || self.tensorrt_gpu_path_disabled {
            return Ok(None);
        }
        if self.temporal_frames.is_some() && self.interp == InterpKind::None && self.fp16 {
            return match self.process_tensorrt_temporal_gpu_texture(gc, input_texture) {
                Ok(texture) => Ok(Some(texture)),
                Err(error) => {
                    self.retire_tensorrt_temporal_gpu_bridge(gc);
                    self.tensorrt_gpu_path_disabled = true;
                    log::warn!(
                        "trt-temporal-gpu-path disabled for '{}'; CPU temporal fallback restored: {error:#}",
                        self.name
                    );
                    Ok(None)
                }
            };
        }
        let unsupported = if !self.fp16 {
            Some("fp32")
        } else if self.input_channels != Some(3) {
            Some("unsupported-channel-count")
        } else if self.interp != InterpKind::None {
            Some("interpolation-model")
        } else if self.temporal_frames.is_some() {
            Some("temporal-input")
        } else if self.last_upscale_input_size != Some((input_texture.w(), input_texture.h())) {
            // The first frame discovers the dynamic output shape through the
            // validated CPU route. Following frames stay entirely on the GPU.
            return Ok(None);
        } else {
            None
        };
        if let Some(reason) = unsupported {
            if !self.tensorrt_gpu_unsupported_logged {
                self.tensorrt_gpu_unsupported_logged = true;
                log::info!(
                    "trt-gpu-path-unsupported: model={} reason={reason}",
                    self.name
                );
            }
            return Ok(None);
        }
        match self.process_tensorrt_gpu_texture_inner(gc, input_texture) {
            Ok(texture) => Ok(Some(texture)),
            Err(error) => {
                // RunOptions::terminate is the normal Stop path, not evidence
                // that TensorRT GPU interop failed. Preserve the bridge state
                // for orderly session teardown and let the engine classify the
                // cancellation without adding a yellow filter warning.
                if onnx_cancel_requested() {
                    log::info!(
                        "trt-gpu-run-cancelled-by-stop: model={} reason={error:#}",
                        self.name
                    );
                    return Err(error);
                }
                self.retire_tensorrt_gpu_bridge(gc);
                self.tensorrt_gpu_path_disabled = true;
                log::warn!(
                    "trt-gpu-bridge-disabled: model={} reason={error:#} fallback=stable-cpu-path",
                    self.name
                );
                Ok(None)
            }
        }
    }

    fn process_tensorrt_interp_gpu(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
        capacity_factor: usize,
    ) -> Result<Option<u64>> {
        let size = (frames[0].w(), frames[0].h());
        let (channels, frame_count, padded) = self.interp_gpu_layout(size)?;
        let slot_count = Self::interp_slot_count(capacity_factor.max(timesteps.len() + 1));
        if self.tensorrt_interp_bridge.as_ref().is_some_and(|bridge| {
            bridge.size != size
                || bridge.padded_size != padded
                || bridge.channels != channels
                || bridge.frame_count != frame_count
                // Fractional refresh-limited cadence legitimately alternates
                // output counts. Never shrink the bridge on the lower-count
                // frame; grow only when a larger user factor needs capacity.
                || Self::interp_bridge_needs_growth(bridge.slot_count, slot_count)
        }) {
            self.retire_tensorrt_interp_gpu_bridge(gc);
        }
        let device_id = self
            .tensorrt_device_id
            .ok_or_else(|| anyhow!("TensorRT CUDA device unavailable"))?;
        if self.tensorrt_interp_bridge.is_none() {
            let luid = gc
                .external_device_luid()
                .ok_or_else(|| anyhow!("OpenGL external-memory LUID unavailable"))?;
            let plane = usize::try_from(padded.0)? * usize::try_from(padded.1)?;
            let element_bytes = if self.fp16 { 2 } else { 4 };
            let history_bytes = plane * 3 * element_bytes;
            let input_bytes = plane * channels * element_bytes;
            let output_bytes = plane * 3 * element_bytes;
            let mut history = Vec::with_capacity(frame_count);
            for _ in 0..frame_count {
                history.push(CudaSharedBuffer::new(
                    DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                    history_bytes,
                    device_id,
                    luid,
                )?);
            }
            let memory = MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let mut invocations = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                let packed_input = CudaSharedBuffer::new(
                    DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                    input_bytes,
                    device_id,
                    luid,
                )?;
                let output = CudaSharedBuffer::new(
                    DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                    output_bytes,
                    device_id,
                    luid,
                )?;
                let input_shape =
                    Shape::new([1, channels as i64, padded.1 as i64, padded.0 as i64]);
                let output_shape = Shape::new([1, 3, padded.1 as i64, padded.0 as i64]);
                let input_value = unsafe {
                    if self.fp16 {
                        InterpTensorValue::F16(TensorRefMut::from_raw(
                            memory.clone(),
                            packed_input.device_ptr,
                            input_shape,
                        )?)
                    } else {
                        InterpTensorValue::F32(TensorRefMut::from_raw(
                            memory.clone(),
                            packed_input.device_ptr,
                            input_shape,
                        )?)
                    }
                };
                let output_value = unsafe {
                    if self.fp16 {
                        InterpTensorValue::F16(TensorRefMut::from_raw(
                            memory.clone(),
                            output.device_ptr,
                            output_shape,
                        )?)
                    } else {
                        InterpTensorValue::F32(TensorRefMut::from_raw(
                            memory.clone(),
                            output.device_ptr,
                            output_shape,
                        )?)
                    }
                };
                let input_name = CString::new(self.in_name.as_bytes())?;
                let output_name = CString::new(self.out_name.as_bytes())?;
                let binding = self.session.create_binding()?;
                ort_status(unsafe {
                    (ort::api().BindInput)(
                        binding.ptr().cast_mut(),
                        input_name.as_ptr(),
                        input_value.ptr(),
                    )
                })?;
                ort_status(unsafe {
                    (ort::api().BindOutput)(
                        binding.ptr().cast_mut(),
                        output_name.as_ptr(),
                        output_value.ptr(),
                    )
                })?;
                invocations.push(TensorRtInterpInvocation {
                    binding,
                    packed_input,
                    output,
                    _input_value: input_value,
                    _output_value: output_value,
                    _input_name: input_name,
                    _output_name: output_name,
                });
            }
            let mut imported_keys = Vec::with_capacity(history.len() + invocations.len() * 2);
            let import_result = (|| -> Result<()> {
                for buffer in history.iter().chain(
                    invocations
                        .iter()
                        .flat_map(|slot| [&slot.packed_input, &slot.output]),
                ) {
                    gc.import_external_d3d12_buffer(
                        buffer.key,
                        buffer.shared_handle()?,
                        buffer.byte_len,
                        buffer.allocation_byte_len,
                        buffer.luid,
                    )
                    .map_err(anyhow::Error::msg)?;
                    imported_keys.push(buffer.key);
                }
                Ok(())
            })();
            if let Err(error) = import_result {
                for key in imported_keys.into_iter().rev() {
                    gc.clear_external_buffer_key(key);
                }
                return Err(error.context(
                    "TensorRT interpolation shared-buffer import transaction rolled back",
                ));
            }
            self.tensorrt_interp_bridge = Some(TensorRtInterpGpuBridge {
                history,
                invocations,
                size,
                padded_size: padded,
                channels,
                frame_count,
                slot_count,
                generation,
            });
            log::info!(
                "interp-gpu-bridge-created: backend=TensorRT kind={:?} factor_capacity={} current_factor={} input={}x{} padded={}x{} worker=async input_payload=gpu-slot output_payload=gpu-slot slots={}",
                self.interp,
                capacity_factor,
                timesteps.len() + 1,
                size.0,
                size.1,
                padded.0,
                padded.1,
                slot_count
            );
            log::info!(
                "interp-gpu-slots: backend=TensorRT count={} peak_in_flight={} state=ready",
                slot_count,
                timesteps.len()
            );
            let cuda = cuda_shared_stats();
            log::info!(
                "interp-gpu-resource-state: backend=TensorRT gl_active={} gl_active_mb={:.1} gl_quarantine={} cuda_active_buffers={} cuda_active_mb={:.1} cuda_created={} cuda_freed={} cuda_release_failures={}",
                gc.external_import_active_count(),
                gc.external_import_active_bytes() as f64 / (1024.0 * 1024.0),
                gc.external_import_retired_count(),
                cuda.active_buffers,
                cuda.active_bytes as f64 / (1024.0 * 1024.0),
                cuda.total_created,
                cuda.total_freed,
                cuda.release_failures
            );
        }
        let bridge = self.tensorrt_interp_bridge.as_mut().unwrap();
        bridge.generation = generation;
        for (slot, frame) in bridge.history.iter().zip(frames.iter()) {
            (if self.fp16 {
                gc.pack_interp_rgb_f16(slot.key, *frame, size, padded, 0)
            } else {
                gc.pack_interp_rgb_f32(slot.key, *frame, size, padded, 0)
            })
            .map_err(anyhow::Error::msg)?;
        }
        let history_keys = bridge
            .history
            .iter()
            .map(|slot| slot.key)
            .collect::<Vec<_>>();
        for (invocation, timestep) in bridge.invocations.iter().zip(timesteps.iter()) {
            Self::pack_interp_invocation(
                &self.interp,
                gc,
                invocation.packed_input.key,
                &history_keys,
                size,
                padded,
                channels,
                *timestep,
                self.fp16,
            )?;
        }
        // Output slots are not reused until the previous GL conversion has
        // completed. This wait is bounded inside GlContext; the expensive GL
        // input handoff itself is represented by a non-blocking fence token.
        for invocation in bridge.invocations.iter().take(timesteps.len()) {
            gc.wait_external_buffer_idle(invocation.output.key)
                .map_err(anyhow::Error::msg)?;
        }
        let fence = gc.submit_commands_fence().map_err(anyhow::Error::msg)?;
        Ok(Some(fence))
    }

    fn retire_tensorrt_interp_gpu_bridge(&mut self, gc: &mut GlContext) {
        let Some(mut bridge) = self.tensorrt_interp_bridge.take() else {
            return;
        };
        if let Some(device_id) = self.tensorrt_device_id {
            if let Err(error) = CudaSharedBuffer::synchronize(device_id) {
                log::error!(
                    "interp-gpu-bridge-retire-sync-failed: backend=TensorRT reason={error:#}"
                );
            }
        }
        // Remove every ORT binding before GL/CUDA ownership is detached.
        // TensorRT/ORT may otherwise retain a raw pointer to a mapping whose
        // D3D12 view was already deleted during rapid factor/shape rebuilds.
        for invocation in &mut bridge.invocations {
            invocation.binding.clear();
        }
        let retired_keys = bridge.history.len() + bridge.invocations.len() * 2;
        let gl_before = gc.external_import_active_count();
        for buffer in bridge.history.iter().chain(
            bridge
                .invocations
                .iter()
                .flat_map(|slot| [&slot.packed_input, &slot.output]),
        ) {
            gc.clear_external_buffer_key(buffer.key);
        }
        drop(bridge);
        let cuda = cuda_shared_stats();
        log::info!(
            "interp-gpu-bridge-retired: backend=TensorRT keys={} gl_active_before={} gl_active_after={} cuda_active_buffers={} cuda_active_mb={:.1} cuda_created={} cuda_freed={} cuda_release_failures={}",
            retired_keys,
            gl_before,
            gc.external_import_active_count(),
            cuda.active_buffers,
            cuda.active_bytes as f64 / (1024.0 * 1024.0),
            cuda.total_created,
            cuda.total_freed,
            cuda.release_failures
        );
    }

    fn process_dml_interp_gpu(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
        capacity_factor: usize,
        interop_mode: DmlInterpInteropMode,
    ) -> Result<Option<u64>> {
        let size = (frames[0].w(), frames[0].h());
        let (channels, frame_count, padded) = self.interp_gpu_layout(size)?;
        let slot_count = Self::interp_slot_count(capacity_factor.max(timesteps.len() + 1));
        if self.dml_interp_bridge.as_ref().is_some_and(|bridge| {
            bridge.size != size
                || bridge.padded_size != padded
                || bridge.channels != channels
                || bridge.frame_count != frame_count
                // Fractional refresh-limited cadence legitimately alternates
                // output counts. Never shrink the bridge on the lower-count
                // frame; grow only when a larger user factor needs capacity.
                || Self::interp_bridge_needs_growth(bridge.slot_count, slot_count)
                || bridge.interop_mode != interop_mode
        }) {
            self.retire_dml_interp_gpu_bridge(gc);
        }
        if self.dml_interp_bridge.is_none() {
            let plane = usize::try_from(padded.0)? * usize::try_from(padded.1)?;
            // Shape discovery is initialization-only and contains no source
            // pixels. The steady-state path below never builds a CPU tensor.
            let shape = Shape::new([1, channels as i64, padded.1 as i64, padded.0 as i64]);
            let zero_f16 = if self.fp16 {
                vec![f16::ZERO; plane * channels]
            } else {
                Vec::new()
            };
            let zero_f32 = if self.fp16 {
                Vec::new()
            } else {
                vec![0f32; plane * channels]
            };
            let first_output = if self.fp16 {
                let discovery = TensorRef::from_array_view((shape.clone(), zero_f16.as_slice()))
                    .map_err(oerr)?;
                create_direct_output(
                    &mut self.session,
                    &self.run_options,
                    &self.in_name,
                    &self.out_name,
                    &discovery,
                    padded,
                )?
            } else {
                let discovery = TensorRef::from_array_view((shape.clone(), zero_f32.as_slice()))
                    .map_err(oerr)?;
                create_direct_output(
                    &mut self.session,
                    &self.run_options,
                    &self.in_name,
                    &self.out_name,
                    &discovery,
                    padded,
                )?
            };
            anyhow::ensure!(
                first_output.dims == vec![1, 3, padded.1 as i64, padded.0 as i64],
                "unexpected DirectML interpolation output {:?}",
                first_output.dims
            );
            let device = first_output.device.clone();
            let luid = first_output.luid;
            let mut history = Vec::with_capacity(frame_count);
            for _ in 0..frame_count {
                history.push(create_direct_input_channels_typed(
                    &device,
                    luid,
                    padded,
                    3,
                    if self.fp16 { 2 } else { 4 },
                )?);
            }
            let memory = MemoryInfo::new(
                AllocationDevice::DIRECTML,
                0,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let mut pending_outputs = VecDeque::from([first_output]);
            let mut invocations = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                let mut output = if let Some(output) = pending_outputs.pop_front() {
                    output
                } else {
                    if self.fp16 {
                        let d = TensorRef::from_array_view((shape.clone(), zero_f16.as_slice()))
                            .map_err(oerr)?;
                        create_direct_output(
                            &mut self.session,
                            &self.run_options,
                            &self.in_name,
                            &self.out_name,
                            &d,
                            padded,
                        )?
                    } else {
                        let d = TensorRef::from_array_view((shape.clone(), zero_f32.as_slice()))
                            .map_err(oerr)?;
                        create_direct_output(
                            &mut self.session,
                            &self.run_options,
                            &self.in_name,
                            &self.out_name,
                            &d,
                            padded,
                        )?
                    }
                };
                let input = create_direct_input_channels_typed(
                    &device,
                    luid,
                    padded,
                    channels,
                    if self.fp16 { 2 } else { 4 },
                )?;
                let safe_input = if interop_mode == DmlInterpInteropMode::ConservativeCopy {
                    Some(create_direct_input_channels_typed_state(
                        &device,
                        luid,
                        padded,
                        channels,
                        if self.fp16 { 2 } else { 4 },
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    )?)
                } else {
                    None
                };
                let bound_input = safe_input.as_ref().unwrap_or(&input);
                let input_value = unsafe {
                    if self.fp16 {
                        InterpTensorValue::F16(TensorRefMut::from_raw(
                            memory.clone(),
                            bound_input.allocation,
                            Shape::new(bound_input.dims),
                        )?)
                    } else {
                        InterpTensorValue::F32(TensorRefMut::from_raw(
                            memory.clone(),
                            bound_input.allocation,
                            Shape::new(bound_input.dims),
                        )?)
                    }
                };
                let output_value = if interop_mode == DmlInterpInteropMode::Direct {
                    Some(unsafe {
                        if self.fp16 {
                            InterpTensorValue::F16(TensorRefMut::from_raw(
                                memory.clone(),
                                output.allocation,
                                Shape::new(output.dims.iter().copied()),
                            )?)
                        } else {
                            InterpTensorValue::F32(TensorRefMut::from_raw(
                                memory.clone(),
                                output.allocation,
                                Shape::new(output.dims.iter().copied()),
                            )?)
                        }
                    })
                } else {
                    None
                };
                let input_name = CString::new(self.in_name.as_bytes())?;
                let output_name = CString::new(self.out_name.as_bytes())?;
                ort_status(unsafe {
                    (ort::api().BindInput)(
                        output.binding.ptr().cast_mut(),
                        input_name.as_ptr(),
                        input_value.ptr(),
                    )
                })?;
                if let Some(output_value) = output_value.as_ref() {
                    ort_status(unsafe {
                        (ort::api().BindOutput)(
                            output.binding.ptr().cast_mut(),
                            output_name.as_ptr(),
                            output_value.ptr(),
                        )
                    })?;
                } else {
                    output
                        .binding
                        .bind_output_to_device(self.out_name.clone(), &memory)
                        .map_err(oerr)?;
                }
                invocations.push(DmlInterpInvocation {
                    input,
                    output,
                    _input_value: input_value,
                    _output_value: output_value,
                    _input_name: input_name,
                    _output_name: output_name,
                    safe_input,
                });
            }
            let mut imported_keys = Vec::with_capacity(history.len() + invocations.len() * 2);
            let import_result = (|| -> Result<()> {
                for buffer in history.iter() {
                    gc.import_external_d3d12_buffer(
                        buffer.key,
                        buffer.shared_handle()?,
                        buffer.byte_len,
                        buffer.heap_byte_len,
                        buffer.luid,
                    )
                    .map_err(anyhow::Error::msg)?;
                    imported_keys.push(buffer.key);
                }
                for invocation in &invocations {
                    for (key, handle, byte_len, heap_len, item_luid) in [
                        (
                            invocation.input.key,
                            invocation.input.shared_handle()?,
                            invocation.input.byte_len,
                            invocation.input.heap_byte_len,
                            invocation.input.luid,
                        ),
                        (
                            invocation.output.key,
                            invocation.output.shared_handle()?,
                            invocation.output.byte_len,
                            invocation.output.heap_byte_len,
                            invocation.output.luid,
                        ),
                    ] {
                        gc.import_external_d3d12_buffer(key, handle, byte_len, heap_len, item_luid)
                            .map_err(anyhow::Error::msg)?;
                        imported_keys.push(key);
                    }
                }
                Ok(())
            })();
            if let Err(error) = import_result {
                for key in imported_keys.into_iter().rev() {
                    gc.clear_external_buffer_key(key);
                }
                return Err(error.context(
                    "DirectML interpolation shared-buffer import transaction rolled back",
                ));
            }
            self.dml_interp_bridge = Some(DmlInterpGpuBridge {
                history,
                invocations,
                size,
                padded_size: padded,
                channels,
                frame_count,
                slot_count,
                generation,
                interop_mode,
            });
            // A driver may reveal the unsafe DSA external-buffer path only
            // during the first import. Rebuild before any GL packing/inference
            // is submitted, so no potentially unsafe direct frame is shown.
            if interop_mode == DmlInterpInteropMode::Direct
                && gc.external_interop_conservative_recommended()
            {
                log::warn!(
                    "dml-interp-interop-upgrade: mode=provider-copy-safe reason=external-storage-driver-fallback vendor='{}' bound_storage_fallbacks={}",
                    gc.gl_vendor(),
                    gc.external_storage_fallback_count()
                );
                self.retire_dml_interp_gpu_bridge(gc);
                return self.process_dml_interp_gpu(
                    gc,
                    frames,
                    timesteps,
                    generation,
                    capacity_factor,
                    DmlInterpInteropMode::ConservativeCopy,
                );
            }
            log::info!(
                "interp-gpu-bridge-created: backend=DirectML kind={:?} factor_capacity={} current_factor={} input={}x{} padded={}x{} worker=async input_payload=gpu-slot output_payload=gpu-slot slots={} interop={} safe_sync={}",
                self.interp,
                capacity_factor,
                timesteps.len() + 1,
                size.0,
                size.1,
                padded.0,
                padded.1,
                slot_count,
                interop_mode.label(),
                match interop_mode {
                    DmlInterpInteropMode::Direct => "gl-fence+ort-sync",
                    DmlInterpInteropMode::ProviderOutputCopy => {
                        "gl-fence+ort-sync+d3d12-output-copy"
                    }
                    DmlInterpInteropMode::ConservativeCopy => {
                        "glFinish+d3d12-state-correct-copy-fence+immediate-ready"
                    }
                }
            );
            log::info!(
                "interp-gpu-slots: backend=DirectML count={} peak_in_flight={} state=ready",
                slot_count,
                timesteps.len()
            );
            if interop_mode == DmlInterpInteropMode::ConservativeCopy {
                log::info!(
                    "dml-interp-copy-state: queue=direct gl_input=common->copy_source->common dml_input=uav->copy_dest->uav provider_output=uav->copy_source->uav gl_output=common->copy_dest->common"
                );
            }
            let dml = dml_shared_stats();
            log::info!(
                "interp-gpu-resource-state: backend=DirectML gl_active={} gl_active_mb={:.1} gl_quarantine={} dml_active_allocations={} dml_active_mb={:.1} dml_created={} dml_freed={} dml_release_failures={}",
                gc.external_import_active_count(),
                gc.external_import_active_bytes() as f64 / (1024.0 * 1024.0),
                gc.external_import_retired_count(),
                dml.active_allocations,
                dml.active_bytes as f64 / (1024.0 * 1024.0),
                dml.total_created,
                dml.total_freed,
                dml.release_failures
            );
        }
        let bridge = self.dml_interp_bridge.as_mut().unwrap();
        bridge.generation = generation;
        for (slot, frame) in bridge.history.iter().zip(frames.iter()) {
            (if self.fp16 {
                gc.pack_interp_rgb_f16(slot.key, *frame, size, padded, 0)
            } else {
                gc.pack_interp_rgb_f32(slot.key, *frame, size, padded, 0)
            })
            .map_err(anyhow::Error::msg)?;
        }
        let history_keys = bridge
            .history
            .iter()
            .map(|slot| slot.key)
            .collect::<Vec<_>>();
        for (invocation, timestep) in bridge.invocations.iter().zip(timesteps.iter()) {
            Self::pack_interp_invocation(
                &self.interp,
                gc,
                invocation.input.key,
                &history_keys,
                size,
                padded,
                channels,
                *timestep,
                self.fp16,
            )?;
        }
        // Output slots are not reused until the previous GL conversion has
        // completed. On conservative drivers, fully complete the GL writes,
        // then copy the packed input into a DirectML-only D3D12 buffer.
        // This is still GPU-resident: only command completion is synchronized;
        // no frame pixels are read back to system memory.
        for invocation in bridge.invocations.iter().take(timesteps.len()) {
            gc.wait_external_buffer_idle(invocation.output.key)
                .map_err(anyhow::Error::msg)?;
        }
        if bridge.interop_mode == DmlInterpInteropMode::ConservativeCopy {
            // Preserve the proven refresh-aware interpolation cadence.  Only the
            // AMD-safe GPU transfer is optimized here: all packed timestep inputs
            // are copied by one state-correct D3D12 direct-queue submission and one fence wait.
            // Pixel data remains GPU-resident and presentation phase scheduling is
            // intentionally untouched.
            gc.finish();
            let copies = bridge
                .invocations
                .iter()
                .take(timesteps.len())
                .map(|invocation| {
                    let safe_input = invocation
                        .safe_input
                        .as_ref()
                        .ok_or_else(|| anyhow!("DirectML conservative input buffer missing"))?;
                    Ok((
                        invocation.input.resource.clone(),
                        safe_input.resource.clone(),
                        invocation.input.byte_len,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            if !copies.is_empty() {
                let copy = &mut bridge
                    .invocations
                    .first_mut()
                    .expect("non-empty conservative interpolation copy batch")
                    .output
                    .copy;
                let _ = copy.copy_many_and_wait(&copies)?;
            }
        }
        // ConservativeCopy has already completed all GL writes with glFinish
        // and then synchronously waited for the batched D3D12 input copy. A
        // second GL fence is therefore redundant. Returning token 0 is safe:
        // GlContext::poll_commands_fence treats an absent token as complete,
        // so the worker can start immediately without dropping either of the
        // visibility barriers that fixed AMD's green/flickering frames.
        let fence = if bridge.interop_mode == DmlInterpInteropMode::ConservativeCopy {
            0
        } else {
            gc.submit_commands_fence().map_err(anyhow::Error::msg)?
        };
        Ok(Some(fence))
    }

    fn retire_dml_interp_gpu_bridge(&mut self, gc: &mut GlContext) {
        let Some(mut bridge) = self.dml_interp_bridge.take() else {
            return;
        };
        for invocation in &mut bridge.invocations {
            invocation.output.binding.clear();
        }
        let retired_keys = bridge.history.len() + bridge.invocations.len() * 2;
        let gl_before = gc.external_import_active_count();
        for buffer in &bridge.history {
            gc.clear_external_buffer_key(buffer.key);
        }
        for invocation in &bridge.invocations {
            gc.clear_external_buffer_key(invocation.input.key);
            gc.clear_external_buffer_key(invocation.output.key);
        }
        drop(bridge);
        let dml = dml_shared_stats();
        log::info!(
            "interp-gpu-bridge-retired: backend=DirectML keys={} gl_active_before={} gl_active_after={} dml_active_allocations={} dml_active_mb={:.1} dml_created={} dml_freed={} dml_release_failures={}",
            retired_keys,
            gl_before,
            gc.external_import_active_count(),
            dml.active_allocations,
            dml.active_bytes as f64 / (1024.0 * 1024.0),
            dml.total_created,
            dml.total_freed,
            dml.release_failures
        );
    }

    pub fn retire_interp_gpu_bridges(&mut self, gc: &mut GlContext) {
        self.retire_tensorrt_interp_gpu_bridge(gc);
        self.retire_dml_interp_gpu_bridge(gc);
    }

    fn process_tensorrt_temporal_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<GpuTex> {
        let frames = self
            .temporal_frames
            .ok_or_else(|| anyhow!("not a temporal model"))?;
        let size = (input_texture.w(), input_texture.h());
        anyhow::ensure!(
            (size.0 as i64 * size.1 as i64 * 3) & 1 == 0,
            "unaligned temporal plane"
        );
        if self
            .tensorrt_temporal_bridge
            .as_ref()
            .is_some_and(|v| v.size != size || v.frame_count != frames)
        {
            self.retire_tensorrt_temporal_gpu_bridge(gc);
        }
        let device_id = self
            .tensorrt_device_id
            .ok_or_else(|| anyhow!("TensorRT CUDA device unavailable"))?;
        if self.tensorrt_temporal_bridge.is_none() {
            let luid = gc
                .external_device_luid()
                .ok_or_else(|| anyhow!("OpenGL external-memory LUID unavailable"))?;
            let plane_elements = usize::try_from(size.0)? * usize::try_from(size.1)? * 3;
            let slot_bytes = plane_elements * std::mem::size_of::<f16>();
            let mut history = Vec::with_capacity(frames);
            for _ in 0..frames {
                history.push(CudaSharedBuffer::new(
                    DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                    slot_bytes,
                    device_id,
                    luid,
                )?);
            }
            let packed_input = CudaSharedBuffer::new(
                DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                slot_bytes * frames,
                device_id,
                luid,
            )?;
            let output = CudaSharedBuffer::new(
                DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                slot_bytes,
                device_id,
                luid,
            )?;
            let mut binding = self.session.create_binding()?;
            let memory = MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let input_value: TensorRefMut<'static, f16> = unsafe {
                TensorRefMut::from_raw(
                    memory.clone(),
                    packed_input.device_ptr,
                    Shape::new([1, (frames * 3) as i64, size.1 as i64, size.0 as i64]),
                )?
            };
            let output_value: TensorRefMut<'static, f16> = unsafe {
                TensorRefMut::from_raw(
                    memory,
                    output.device_ptr,
                    Shape::new([1, 3, size.1 as i64, size.0 as i64]),
                )?
            };
            let input_name = CString::new(self.in_name.as_bytes())?;
            let output_name = CString::new(self.out_name.as_bytes())?;
            ort_status(unsafe {
                (ort::api().BindInput)(
                    binding.ptr().cast_mut(),
                    input_name.as_ptr(),
                    input_value.ptr(),
                )
            })?;
            ort_status(unsafe {
                (ort::api().BindOutput)(
                    binding.ptr().cast_mut(),
                    output_name.as_ptr(),
                    output_value.ptr(),
                )
            })?;
            let mut imported_keys = Vec::with_capacity(history.len() + 2);
            let import_result = (|| -> Result<()> {
                for buffer in history.iter().chain([&packed_input, &output].into_iter()) {
                    gc.import_external_d3d12_buffer(
                        buffer.key,
                        buffer.shared_handle()?,
                        buffer.byte_len,
                        buffer.allocation_byte_len,
                        buffer.luid,
                    )
                    .map_err(anyhow::Error::msg)?;
                    imported_keys.push(buffer.key);
                }
                Ok(())
            })();
            if let Err(error) = import_result {
                binding.clear();
                for key in imported_keys.into_iter().rev() {
                    gc.clear_external_buffer_key(key);
                }
                return Err(
                    error.context("TensorRT temporal shared-buffer import transaction rolled back")
                );
            }
            self.tensorrt_temporal_bridge = Some(TensorRtTemporalGpuBridge {
                binding,
                history,
                packed_input,
                output,
                _input_value: input_value,
                _output_value: output_value,
                _input_name: input_name,
                _output_name: output_name,
                size,
                frame_count: frames,
                write_index: 0,
                valid_frames: 0,
                timings: TensorRtGpuTimings::default(),
            });
            log::info!(
                "trt-temporal-gpu-path-active: model={} frames={} channels={} input={}x{}",
                self.name,
                frames,
                frames * 3,
                size.0,
                size.1
            );
        }
        let bridge = self.tensorrt_temporal_bridge.as_mut().unwrap();
        let total_start = Instant::now();
        let pack_start = Instant::now();
        let slot = bridge.write_index;
        gc.rgba_texture_to_external_nchw_f16(
            bridge.history[slot].key,
            input_texture,
            size.0,
            size.1,
        )
        .map_err(anyhow::Error::msg)?;
        bridge.write_index = (bridge.write_index + 1) % frames;
        bridge.valid_frames = (bridge.valid_frames + 1).min(frames);
        let ordered_slots =
            temporal_history_slot_order(frames, bridge.valid_frames, bridge.write_index);
        let slot_elements = usize::try_from(size.0)? * usize::try_from(size.1)? * 3;
        for (destination, source_slot) in ordered_slots.into_iter().enumerate() {
            gc.copy_external_nchw_f16(
                bridge.history[source_slot].key,
                bridge.packed_input.key,
                0,
                destination * slot_elements,
                slot_elements,
            )
            .map_err(anyhow::Error::msg)?;
        }
        gc.finish();
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        let run_start = Instant::now();
        self.session
            .run_binding_with_options(&bridge.binding, self.run_options.armed()?)
            .map_err(oerr)?;
        bridge.binding.synchronize_outputs().map_err(oerr)?;
        if tensorrt_sync_mode() == TensorRtSyncMode::Safe {
            CudaSharedBuffer::synchronize(device_id)?;
        }
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let out_start = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(bridge.output.key, size.0, size.1)
            .map_err(anyhow::Error::msg)?;
        let out_ms = out_start.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms,
            output_size: (size.0 as usize, size.1 as usize),
        });
        self.last_upscale_input_size = Some(size);
        bridge.timings.record(
            &self.name,
            pack_ms,
            run_ms,
            out_ms,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(texture)
    }

    pub(crate) fn retire_tensorrt_temporal_gpu_bridge(&mut self, gc: &mut GlContext) {
        let Some(mut bridge) = self.tensorrt_temporal_bridge.take() else {
            return;
        };
        gc.finish();
        if let Some(device_id) = self.tensorrt_device_id {
            let _ = CudaSharedBuffer::synchronize(device_id);
        }
        bridge.binding.clear();
        for slot in &bridge.history {
            gc.clear_external_buffer_key(slot.key);
        }
        gc.clear_external_buffer_key(bridge.packed_input.key);
        gc.clear_external_buffer_key(bridge.output.key);
    }

    pub fn should_try_tensorrt_gpu_texture(&self, width: i32, height: i32) -> bool {
        self.provider == OnnxProvider::TensorRT
            && tensorrt_gpu_interop_enabled()
            && !self.tensorrt_gpu_path_disabled
            && self.last_upscale_input_size == Some((width, height))
    }

    fn process_tensorrt_gpu_texture_inner(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<GpuTex> {
        anyhow::ensure!(
            std::env::var_os("CHIDESCALER_TRT_GPU_INTEROP_FORCE_FAIL").is_none(),
            "forced TensorRT GPU interop failure"
        );
        let input_size = (input_texture.w(), input_texture.h());
        let output_size = self
            .last_upscale_profile
            .as_ref()
            .map(|profile| (profile.output_size.0 as i32, profile.output_size.1 as i32))
            .ok_or_else(|| anyhow!("TensorRT output shape has not been discovered"))?;
        if self.tensorrt_gpu_bridge.as_ref().is_some_and(|bridge| {
            bridge.input_size != input_size || bridge.output_size != output_size
        }) {
            log::info!(
                "trt-gpu-bridge-resize: model={} old={:?} new={}x{}",
                self.name,
                self.tensorrt_gpu_bridge
                    .as_ref()
                    .map(|bridge| bridge.input_size),
                input_size.0,
                input_size.1
            );
            self.retire_tensorrt_gpu_bridge(gc);
        }
        let device_id = self
            .tensorrt_device_id
            .ok_or_else(|| anyhow!("TensorRT CUDA device id is unavailable"))?;
        if self.tensorrt_gpu_bridge.is_none() {
            let luid = gc
                .external_device_luid()
                .ok_or_else(|| anyhow!("OpenGL external-memory LUID is unavailable"))?;
            let input_bytes = usize::try_from(input_size.0)?
                .checked_mul(usize::try_from(input_size.1)?)
                .and_then(|value| value.checked_mul(3 * std::mem::size_of::<f16>()))
                .ok_or_else(|| anyhow!("TensorRT shared input size overflow"))?;
            let output_bytes = usize::try_from(output_size.0)?
                .checked_mul(usize::try_from(output_size.1)?)
                .and_then(|value| value.checked_mul(3 * std::mem::size_of::<f16>()))
                .ok_or_else(|| anyhow!("TensorRT shared output size overflow"))?;
            let input = CudaSharedBuffer::new(
                DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                input_bytes,
                device_id,
                luid,
            )?;
            let output = CudaSharedBuffer::new(
                DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
                output_bytes,
                device_id,
                luid,
            )?;
            let mut binding = self.session.create_binding()?;
            let memory = MemoryInfo::new(
                AllocationDevice::CUDA,
                device_id,
                AllocatorType::Device,
                MemoryType::Default,
            )?;
            let input_value: TensorRefMut<'static, f16> = unsafe {
                TensorRefMut::<f16>::from_raw(
                    memory.clone(),
                    input.device_ptr,
                    Shape::new([1, 3, input_size.1 as i64, input_size.0 as i64]),
                )?
            };
            let output_value: TensorRefMut<'static, f16> = unsafe {
                TensorRefMut::<f16>::from_raw(
                    memory,
                    output.device_ptr,
                    Shape::new([1, 3, output_size.1 as i64, output_size.0 as i64]),
                )?
            };
            let input_name = CString::new(self.in_name.as_bytes())?;
            let output_name = CString::new(self.out_name.as_bytes())?;
            ort_status(unsafe {
                (ort::api().BindInput)(
                    binding.ptr().cast_mut(),
                    input_name.as_ptr(),
                    input_value.ptr(),
                )
            })?;
            ort_status(unsafe {
                (ort::api().BindOutput)(
                    binding.ptr().cast_mut(),
                    output_name.as_ptr(),
                    output_value.ptr(),
                )
            })?;
            let mut imported_keys = Vec::with_capacity(2);
            let import_result = (|| -> Result<()> {
                for buffer in [&input, &output] {
                    gc.import_external_d3d12_buffer(
                        buffer.key,
                        buffer.shared_handle()?,
                        buffer.byte_len,
                        buffer.allocation_byte_len,
                        buffer.luid,
                    )
                    .map_err(anyhow::Error::msg)?;
                    imported_keys.push(buffer.key);
                }
                Ok(())
            })();
            if let Err(error) = import_result {
                binding.clear();
                for key in imported_keys.into_iter().rev() {
                    gc.clear_external_buffer_key(key);
                }
                return Err(error.context("TensorRT shared-buffer import transaction rolled back"));
            }
            self.tensorrt_gpu_bridge = Some(TensorRtGpuBridge {
                binding,
                input,
                output,
                _input_value: input_value,
                _output_value: output_value,
                _input_name: input_name,
                _output_name: output_name,
                input_size,
                output_size,
                timings: TensorRtGpuTimings::default(),
            });
            log::info!(
                "trt-gpu-bridge-ready: model={} input={}x{} output={}x{} cuda_device={} luid={luid:02x?} input_bytes={} output_bytes={}",
                self.name,
                input_size.0,
                input_size.1,
                output_size.0,
                output_size.1,
                device_id,
                input_bytes,
                output_bytes
            );
            log::info!(
                "trt-gpu-path-active: model={} mode=shared-d3d12-cuda",
                self.name
            );
        }
        let bridge = self.tensorrt_gpu_bridge.as_mut().unwrap();
        gc.wait_external_buffer_idle(bridge.output.key)
            .map_err(|error| anyhow!(error))?;
        let total_start = Instant::now();
        let pack_start = Instant::now();
        gc.rgba_texture_to_external_nchw_f16(
            bridge.input.key,
            input_texture,
            input_size.0,
            input_size.1,
        )
        .map_err(|error| anyhow!(error))?;
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        let run_start = Instant::now();
        let run_result = self
            .session
            .run_binding_with_options(&bridge.binding, self.run_options.armed()?)
            .map(|_| ())
            .map_err(oerr);
        if run_result.is_ok() {
            bridge.binding.synchronize_outputs().map_err(oerr)?;
        }
        run_result?;
        if tensorrt_sync_mode() == TensorRtSyncMode::Safe {
            CudaSharedBuffer::synchronize(device_id)?;
        }
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let unpack_start = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(bridge.output.key, output_size.0, output_size.1)
            .map_err(|error| anyhow!(error))?;
        let out_ms = unpack_start.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms,
            output_size: (output_size.0 as usize, output_size.1 as usize),
        });
        self.last_upscale_input_size = Some(input_size);
        bridge.timings.record(
            &self.name,
            pack_ms,
            run_ms,
            out_ms,
            total_start.elapsed().as_secs_f64() * 1000.0,
        );
        Ok(texture)
    }

    pub(crate) fn retire_tensorrt_gpu_bridge(&mut self, gc: &mut GlContext) {
        let Some(mut bridge) = self.tensorrt_gpu_bridge.take() else {
            return;
        };
        gc.finish();
        if let Some(device_id) = self.tensorrt_device_id {
            let _ = CudaSharedBuffer::synchronize(device_id);
        }
        bridge.binding.clear();
        gc.clear_external_buffer_key(bridge.input.key);
        gc.clear_external_buffer_key(bridge.output.key);
        drop(bridge);
    }

    fn process_gpu_texture_inner(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
        interop_mode: DmlGlInteropMode,
    ) -> Result<GpuTex> {
        let (w, h) = (input_texture.w(), input_texture.h());
        if self
            .direct_output
            .as_ref()
            .is_some_and(|s| s.input_size != (w, h))
            || self
                .direct_input
                .as_ref()
                .is_some_and(|s| s.input_size != (w, h))
        {
            self.retire_direct_output(gc);
        }
        if self.direct_output.is_none() {
            let elements = usize::try_from(w)?
                .checked_mul(usize::try_from(h)?)
                .and_then(|n| n.checked_mul(3))
                .ok_or_else(|| anyhow!("GPU-resident input size overflow"))?;
            let zero = vec![f16::ZERO; elements];
            let tensor = TensorRef::from_array_view((
                Shape::new([1i64, 3, h as i64, w as i64]),
                zero.as_slice(),
            ))
            .map_err(oerr)?;
            self.direct_output = Some(create_direct_output(
                &mut self.session,
                &self.run_options,
                &self.in_name,
                &self.out_name,
                &tensor,
                (w, h),
            )?);
            let output = self.direct_output.as_ref().unwrap();
            log::info!(
                "ONNX GPU-resident chain ready: model='{}' input={}x{} output={:?} mode={} safe_sync=glFinish fallback=cpu-on-error",
                self.name,
                w,
                h,
                output.dims,
                interop_mode.label()
            );
        }
        if self.direct_input.is_none() {
            let output = self.direct_output.as_ref().unwrap();
            self.direct_input = Some(create_direct_input(&output.device, output.luid, (w, h))?);
        }
        let input = self.direct_input.as_ref().unwrap();
        let output = self.direct_output.as_mut().unwrap();
        if !gc.has_external_import(input.key) {
            gc.import_external_d3d12_buffer(
                input.key,
                input.shared_handle()?,
                input.byte_len,
                input.heap_byte_len,
                input.luid,
            )
            .map_err(|error| anyhow!(error))?;
        }
        if !gc.has_external_import(output.key) {
            gc.import_external_d3d12_buffer(
                output.key,
                output.shared_handle()?,
                output.byte_len,
                output.heap_byte_len,
                output.luid,
            )
            .map_err(|error| anyhow!(error))?;
        }
        gc.wait_external_buffer_idle(output.key)
            .map_err(|e| anyhow!(e))?;
        let pack_start = Instant::now();
        gc.rgba_texture_to_external_nchw_f16(input.key, input_texture, w, h)
            .map_err(|e| anyhow!(e))?;
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        let memory = MemoryInfo::new(
            AllocationDevice::DIRECTML,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let input_value = unsafe {
            TensorRefMut::<f16>::from_raw(memory.clone(), input.allocation, Shape::new(input.dims))?
        };
        let input_name = CString::new(self.in_name.as_bytes())?;
        ort_status(unsafe {
            (ort::api().BindInput)(
                output.binding.ptr().cast_mut(),
                input_name.as_ptr(),
                input_value.ptr(),
            )
        })?;
        let run_start = Instant::now();
        let mut copy_ms = 0.0;
        let mut fence_wait_ms = 0.0;
        if interop_mode == DmlGlInteropMode::Direct {
            let output_value = unsafe {
                TensorRefMut::<f16>::from_raw(
                    memory,
                    output.allocation,
                    Shape::new(output.dims.iter().copied()),
                )?
            };
            let output_name = CString::new(self.out_name.as_bytes())?;
            ort_status(unsafe {
                (ort::api().BindOutput)(
                    output.binding.ptr().cast_mut(),
                    output_name.as_ptr(),
                    output_value.ptr(),
                )
            })?;
            let run_result = self
                .session
                .run_binding_with_options(&output.binding, self.run_options.armed()?)
                .map(|_| ())
                .map_err(oerr);
            if run_result.is_ok() {
                output.binding.synchronize_outputs().map_err(oerr)?;
            }
            output.binding.clear();
            run_result?;
        } else {
            output
                .binding
                .bind_output_to_device(self.out_name.clone(), &memory)
                .map_err(oerr)?;
            let mut provider_outputs = self
                .session
                .run_binding_with_options(&output.binding, self.run_options.armed()?)
                .map_err(oerr)?;
            output.binding.synchronize_outputs().map_err(oerr)?;
            let provider = provider_outputs
                .remove(self.out_name.as_str())
                .ok_or_else(|| anyhow!("DirectML provider output missing"))?
                .downcast::<TensorValueType<f16>>()?;
            let provider_dims: Vec<i64> = provider.shape().iter().copied().collect();
            anyhow::ensure!(
                provider_dims == output.dims,
                "DirectML provider output shape changed: {:?} != {:?}",
                provider_dims,
                output.dims
            );
            let api = dml_api()?;
            let mut raw_resource = std::ptr::null_mut();
            ort_status(unsafe {
                (api.GetD3D12ResourceFromAllocation)(
                    output.dml_allocator.ptr().cast_mut(),
                    provider.data_ptr().cast_mut(),
                    &mut raw_resource,
                )
            })?;
            let borrowed = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
                .ok_or_else(|| anyhow!("DirectML provider output resource is null"))?;
            let provider_resource = borrowed.clone();
            (copy_ms, fence_wait_ms) =
                output
                    .copy
                    .copy_and_wait(&provider_resource, &output.resource, output.byte_len)?;
            drop(provider);
            drop(provider_outputs);
            output.binding.clear();
        }
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let output_start = Instant::now();
        let ow = i32::try_from(output.dims[3])?;
        let oh = i32::try_from(output.dims[2])?;
        let texture = gc
            .external_nchw_f16_to_rgba8(output.key, ow, oh)
            .map_err(|e| anyhow!(e))?;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: output_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (ow as usize, oh as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        log::debug!(
            "DML GPU copy bridge: model='{}' output_bytes={} copy_ms={:.3} fence_wait_ms={:.3} selected_mode={}",
            self.name,
            output.byte_len,
            copy_ms,
            fence_wait_ms,
            interop_mode.label()
        );
        Ok(texture)
    }

    /// Optional zero-copy output path for the first ordinary FP16 ONNX stage.
    /// Any setup/runtime failure disables only this optimization and lets the
    /// caller use the unchanged CPU conversion/upload path in the same frame.
    pub fn process_rgba8_gpu_output(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Option<(i32, i32, GpuTex)> {
        if !self.supports_dml_gl_bridge()
            || !self.fp16
            || self.interp != InterpKind::None
            || self.temporal_frames.is_some()
            || self.direct_output_disabled
        {
            return None;
        }
        match self.process_rgba8_gpu_output_inner(gc, w, h, rgba) {
            Ok(output) => Some(output),
            Err(error) => {
                self.retire_direct_output(gc);
                if onnx_cancel_requested() {
                    // Stop cancellation is intentional teardown, not evidence
                    // that GPU-direct output is unstable. Do not poison the
                    // stage or emit a misleading fallback warning.
                    log::debug!(
                        "ONNX GPU-direct output retired during Stop for '{}': {error:#}",
                        self.name
                    );
                    return None;
                }
                self.direct_output_disabled = true;
                log::warn!(
                    "ONNX GPU-direct output disabled for '{}'; stable CPU path restored: {error:#}",
                    self.name
                );
                None
            }
        }
    }

    /// v652 cross-GPU ordinary-image handoff. Run the first ordinary FP16
    /// DirectML model into the same app-owned shareable D3D12 output buffer
    /// used by the proven interop code, but do not import it into OpenGL.
    /// The caller may hand this resource directly to Vulkan when DirectML and
    /// Vulkan share the selected GPU while presentation OpenGL lives on a
    /// different adapter. Any unsupported model returns None and preserves the
    /// established CPU/OpenGL path.
    pub(crate) fn process_rgba8_dml_shared_output(
        &mut self,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(PreparedDmlSharedOutput, f64)>> {
        if self.provider != OnnxProvider::DirectML
            || !self.fp16
            || self.interp != InterpKind::None
            || self.temporal_frames.is_some()
        {
            return Ok(None);
        }
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let n = wu
            .checked_mul(hu)
            .ok_or_else(|| anyhow!("DirectML shared output input size overflow"))?;
        anyhow::ensure!(rgba.len() >= n * 4, "RGBA input buffer too small");
        if self
            .direct_output
            .as_ref()
            .is_some_and(|state| state.input_size != (w, h))
        {
            log::info!(
                "ONNX DML->Vulkan shared resize: model='{}' {:?} -> {}x{}; retiring old shared output",
                self.name,
                self.direct_output.as_ref().map(|state| state.input_size),
                w,
                h
            );
            self.direct_output = None;
        }

        let started = Instant::now();
        let pack_start = Instant::now();
        prepare_len_f16(&mut self.scratch_f16, n * 3);
        let lut = u8_to_f16_lut();
        use rayon::prelude::*;
        self.scratch_f16
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(channel, plane)| {
                for i in 0..n {
                    plane[i] = lut[rgba[i * 4 + channel] as usize];
                }
            });
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        let input_shape = Shape::new([1i64, 3, h as i64, w as i64]);
        let input = TensorRef::from_array_view((input_shape, &*self.scratch_f16)).map_err(oerr)?;

        let created = self.direct_output.is_none();
        if created {
            self.direct_output = Some(create_direct_output(
                &mut self.session,
                &self.run_options,
                &self.in_name,
                &self.out_name,
                &input,
                (w, h),
            )?);
        }

        let state = self
            .direct_output
            .as_mut()
            .ok_or_else(|| anyhow!("DirectML shared output state disappeared"))?;
        let (run_ms, copy_ms) = run_cpu_tensor_to_dml_shared_output(
            &mut self.session,
            &self.run_options,
            &self.in_name,
            &self.out_name,
            &input,
            state,
        )?;
        let ow = i32::try_from(state.dims[3])?;
        let oh = i32::try_from(state.dims[2])?;
        let shared = PreparedDmlSharedOutput {
            key: state.key,
            handle: state.shared_handle()?,
            byte_len: state.byte_len,
            heap_byte_len: state.heap_byte_len,
            luid: state.luid,
            size: (ow, oh),
            padded: (ow, oh),
            fp16: true,
        };
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: copy_ms,
            output_size: (usize::try_from(ow)?, usize::try_from(oh)?),
        });
        self.last_upscale_input_size = Some((w, h));
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        if created {
            log::info!(
                "ONNX DML->Vulkan shared output ready: model='{}' shape={:?} bytes={} LUID={:02x?} pack_ms={:.3} run_ms={:.3} gpu_copy_ms={:.3}",
                self.name,
                state.dims,
                state.byte_len,
                state.luid,
                pack_ms,
                run_ms,
                copy_ms,
            );
        }
        Ok(Some((shared, total_ms)))
    }

    fn process_rgba8_gpu_output_inner(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<(i32, i32, GpuTex)> {
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let n = wu * hu;
        anyhow::ensure!(rgba.len() >= n * 4, "RGBA input buffer too small");
        if self
            .direct_output
            .as_ref()
            .is_some_and(|state| state.input_size != (w, h))
        {
            log::info!(
                "ONNX GPU-direct resize: model='{}' {:?} -> {}x{}; retiring old shared output",
                self.name,
                self.direct_output.as_ref().map(|state| state.input_size),
                w,
                h
            );
            self.retire_direct_output(gc);
        }
        let pack_start = Instant::now();
        prepare_len_f16(&mut self.scratch_f16, n * 3);
        let lut = u8_to_f16_lut();
        use rayon::prelude::*;
        self.scratch_f16
            .par_chunks_mut(n)
            .enumerate()
            .for_each(|(channel, plane)| {
                for i in 0..n {
                    plane[i] = lut[rgba[i * 4 + channel] as usize];
                }
            });
        let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
        let input_shape = Shape::new([1i64, 3, h as i64, w as i64]);
        let input = TensorRef::from_array_view((input_shape, &*self.scratch_f16)).map_err(oerr)?;

        if self.direct_output.is_none() {
            self.direct_output = Some(create_direct_output(
                &mut self.session,
                &self.run_options,
                &self.in_name,
                &self.out_name,
                &input,
                (w, h),
            )?);
            let state = self.direct_output.as_ref().unwrap();
            log::info!(
                "ONNX GPU-direct output ready: model='{}' shape={:?} bytes={} LUID={:02x?}",
                self.name,
                state.dims,
                state.byte_len,
                state.luid
            );
        }

        let state = self.direct_output.as_mut().unwrap();
        if !gc.has_external_import(state.key) {
            gc.import_external_d3d12_buffer(
                state.key,
                state.shared_handle()?,
                state.byte_len,
                state.heap_byte_len,
                state.luid,
            )
            .map_err(|error| anyhow!(error))?;
        }
        gc.wait_external_buffer_idle(state.key)
            .map_err(|error| anyhow!(error))?;

        let output = unsafe {
            TensorRefMut::<f16>::from_raw(
                MemoryInfo::new(
                    AllocationDevice::DIRECTML,
                    0,
                    AllocatorType::Device,
                    MemoryType::Default,
                )?,
                state.allocation,
                Shape::new(state.dims.iter().copied()),
            )?
        };
        let input_name = CString::new(self.in_name.as_bytes())?;
        let output_name = CString::new(self.out_name.as_bytes())?;
        ort_status(unsafe {
            (ort::api().BindInput)(
                state.binding.ptr().cast_mut(),
                input_name.as_ptr(),
                input.ptr(),
            )
        })?;
        ort_status(unsafe {
            (ort::api().BindOutput)(
                state.binding.ptr().cast_mut(),
                output_name.as_ptr(),
                output.ptr(),
            )
        })?;
        let run_start = Instant::now();
        let run_result = self
            .session
            .run_binding_with_options(&state.binding, self.run_options.armed()?)
            .map(|_| ())
            .map_err(oerr);
        if run_result.is_ok() {
            state.binding.synchronize_outputs().map_err(oerr)?;
        }
        state.binding.clear();
        run_result?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;

        let output_start = Instant::now();
        let ow = i32::try_from(state.dims[3])?;
        let oh = i32::try_from(state.dims[2])?;
        let texture = gc
            .external_nchw_f16_to_rgba8(state.key, ow, oh)
            .map_err(|error| anyhow!(error))?;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: output_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (ow as usize, oh as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        Ok((ow, oh, texture))
    }

    /// Detach OpenGL from the shared D3D12 allocation before DirectML frees it.
    /// The ordering matters on source resizes: freeing the allocation first can
    /// leave the GL memory object pointing at released driver memory.
    pub(crate) fn retire_direct_output(&mut self, gc: &mut GlContext) {
        if self.direct_output.is_none() && self.direct_input.is_none() {
            return;
        }
        gc.finish();
        if let Some(state) = self.direct_input.as_ref() {
            gc.clear_external_buffer_key(state.key);
        }
        if let Some(state) = self.direct_output.as_ref() {
            gc.clear_external_buffer_key(state.key);
        }
        self.direct_input = None;
        self.direct_output = None;
    }

    #[allow(dead_code)] // retained for explicit shared-output teardown diagnostics
    pub(crate) fn release_direct_output_after_gl_detach(&mut self) {
        self.direct_input = None;
        self.direct_output = None;
    }

    fn process_temporal_u8_strided(
        &mut self,
        w: i32,
        h: i32,
        pixels: &[u8],
        stride: usize,
        output_stride: usize,
        frames: usize,
    ) -> Result<(i32, i32, Vec<u8>)> {
        use rayon::prelude::*;
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let n = wu * hu;
        anyhow::ensure!(
            stride >= 3 && pixels.len() >= n * stride,
            "buffer too small"
        );
        anyhow::ensure!(
            output_stride == 3 || output_stride == 4,
            "invalid output stride"
        );
        if self.temporal_size != Some((w, h)) {
            self.temporal_history.clear();
            self.temporal_size = Some((w, h));
        }
        let mut rgb = vec![0u8; n * 3];
        rgb.par_chunks_mut(3).enumerate().for_each(|(i, px)| {
            px.copy_from_slice(&pixels[i * stride..i * stride + 3]);
        });
        self.temporal_history.push_back(rgb);
        while self.temporal_history.len() > frames {
            self.temporal_history.pop_front();
        }
        let first = self.temporal_history.front().cloned().unwrap_or_default();
        let history: Vec<&[u8]> =
            std::iter::repeat_n(first.as_slice(), frames - self.temporal_history.len())
                .chain(self.temporal_history.iter().map(Vec::as_slice))
                .collect();
        let channels = frames * 3;
        let shape = vec![1i64, channels as i64, h as i64, w as i64];
        let pack_start = Instant::now();
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            prepare_len_f16(&mut self.scratch_f16, n * channels);
            let lut = u8_to_f16_lut();
            self.scratch_f16
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(channel, plane)| {
                    let frame = history[channel / 3];
                    let component = channel % 3;
                    for i in 0..n {
                        plane[i] = lut[frame[i * 3 + component] as usize];
                    }
                });
            let input = TensorRef::from_array_view((shape, &*self.scratch_f16)).map_err(oerr)?;
            let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
            let run_start = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => input],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            (outputs, pack_ms, run_start.elapsed().as_secs_f64() * 1000.0)
        } else {
            prepare_len_f32(&mut self.scratch_f32, n * channels);
            self.scratch_f32
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(channel, plane)| {
                    let frame = history[channel / 3];
                    let component = channel % 3;
                    for i in 0..n {
                        plane[i] = frame[i * 3 + component] as f32 * (1.0 / 255.0);
                    }
                });
            let input = TensorRef::from_array_view((shape, &*self.scratch_f32)).map_err(oerr)?;
            let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
            let run_start = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => input],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            (outputs, pack_ms, run_start.elapsed().as_secs_f64() * 1000.0)
        };
        let out_start = Instant::now();
        let out = outputs
            .get(self.out_name.as_str())
            .ok_or_else(|| anyhow!("temporal model produced no output"))?;
        let (ow, oh, data) = if self.fp16 {
            let (shape, values) = out.try_extract_tensor::<f16>().map_err(oerr)?;
            let dims: Vec<i64> = shape[..].to_vec();
            let (ow, oh) = out_dims(&dims)?;
            let on = ow * oh;
            let lut = f16_unorm_to_u8_lut();
            let mut data = vec![0u8; on * output_stride];
            data.par_chunks_mut(output_stride)
                .enumerate()
                .for_each(|(i, px)| {
                    px[0] = lut[values[i].to_bits() as usize];
                    px[1] = lut[values[on + i].to_bits() as usize];
                    px[2] = lut[values[2 * on + i].to_bits() as usize];
                    if output_stride == 4 {
                        px[3] = 255;
                    }
                });
            (ow, oh, data)
        } else {
            let (shape, values) = out.try_extract_tensor::<f32>().map_err(oerr)?;
            let dims: Vec<i64> = shape[..].to_vec();
            let (ow, oh) = out_dims(&dims)?;
            let on = ow * oh;
            let mut data = vec![0u8; on * output_stride];
            data.par_chunks_mut(output_stride)
                .enumerate()
                .for_each(|(i, px)| {
                    px[0] = (values[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    px[1] = (values[on + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    px[2] = (values[2 * on + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    if output_stride == 4 {
                        px[3] = 255;
                    }
                });
            (ow, oh, data)
        };
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: out_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (ow, oh),
        });
        self.last_upscale_input_size = Some((w, h));
        Ok((ow as i32, oh as i32, data))
    }

    fn process_u8_strided(
        &mut self,
        w: i32,
        h: i32,
        pixels: &[u8],
        stride: usize,
        output_stride: usize,
    ) -> Result<(i32, i32, Vec<u8>)> {
        if let Some(frames) = self.temporal_frames {
            return self.process_temporal_u8_strided(w, h, pixels, stride, output_stride, frames);
        }
        let (w, h) = (w as usize, h as usize);
        let n = w * h;
        anyhow::ensure!(
            stride >= 3 && pixels.len() >= n * stride,
            "buffer too small"
        );
        anyhow::ensure!(
            output_stride == 3 || output_stride == 4,
            "invalid output stride"
        );
        let shape = vec![1i64, 3, h as i64, w as i64];

        use rayon::prelude::*;
        let pack_start = std::time::Instant::now();
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            prepare_len_f16(&mut self.scratch_f16, n * 3);
            let x = self.scratch_f16.as_mut_slice();
            let lut = u8_to_f16_lut();
            x.par_chunks_mut(n).enumerate().for_each(|(c, plane)| {
                for i in 0..n {
                    plane[i] = lut[pixels[i * stride + c] as usize];
                }
            });
            let t = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
            let run_start = std::time::Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => t],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        } else {
            prepare_len_f32(&mut self.scratch_f32, n * 3);
            let x = self.scratch_f32.as_mut_slice();
            x.par_chunks_mut(n).enumerate().for_each(|(c, plane)| {
                for i in 0..n {
                    plane[i] = pixels[i * stride + c] as f32 * (1.0 / 255.0);
                }
            });
            let t = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_start.elapsed().as_secs_f64() * 1000.0;
            let run_start = std::time::Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => t],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        };
        let out_start = std::time::Instant::now();
        // `SessionOutputs` keeps a mutable borrow of `self.session` alive.
        // Restrict every output/tensor view to this block so the borrow ends
        // before updating the stage profile and TensorRT shape-completion state.
        let (shape, ow, oh, data) = {
            let out = outputs
                .get(self.out_name.as_str())
                .ok_or_else(|| anyhow!("model produced no output"))?;

            if self.fp16 {
                let (s, v) = out.try_extract_tensor::<f16>().map_err(oerr)?;
                let dims: Vec<i64> = s[..].to_vec();
                let (ow, oh) = out_dims(&dims)?;
                let on = ow * oh;
                let mut rgb8 = vec![0u8; on * output_stride];
                let lut = f16_unorm_to_u8_lut();
                rgb8.par_chunks_mut(output_stride * 8192)
                    .enumerate()
                    .for_each(|(blk, o)| {
                        let base = blk * 8192;
                        for (j, px) in o.chunks_mut(output_stride).enumerate() {
                            let i = base + j;
                            px[0] = lut[v[i].to_bits() as usize];
                            px[1] = lut[v[on + i].to_bits() as usize];
                            px[2] = lut[v[2 * on + i].to_bits() as usize];
                            if output_stride == 4 {
                                px[3] = 255;
                            }
                        }
                    });
                (dims, ow, oh, rgb8)
            } else {
                let (s, v) = out.try_extract_tensor::<f32>().map_err(oerr)?;
                let dims: Vec<i64> = s[..].to_vec();
                let (ow, oh) = out_dims(&dims)?;
                let on = ow * oh;
                let mut rgb8 = vec![0u8; on * output_stride];
                rgb8.par_chunks_mut(output_stride * 8192)
                    .enumerate()
                    .for_each(|(blk, o)| {
                        let base = blk * 8192;
                        for (j, px) in o.chunks_mut(output_stride).enumerate() {
                            let i = base + j;
                            px[0] = (v[i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            px[1] = (v[on + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            px[2] = (v[2 * on + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                            if output_stride == 4 {
                                px[3] = 255;
                            }
                        }
                    });
                (dims, ow, oh, rgb8)
            }
        };
        drop(outputs);
        let _ = shape;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: out_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (ow, oh),
        });
        self.last_upscale_input_size = Some((w as i32, h as i32));
        self.complete_successful_tensorrt_shape(w as i32, h as i32);
        Ok((ow as i32, oh as i32, data))
    }
}

fn temporal_history_slot_order(frames: usize, valid: usize, next_write: usize) -> Vec<usize> {
    debug_assert!(frames > 0 && valid > 0 && valid <= frames);
    let first = if valid == frames { next_write } else { 0 };
    let repeated = frames - valid;
    (0..frames)
        .map(|destination| {
            if destination < repeated {
                first
            } else {
                (first + destination - repeated) % frames
            }
        })
        .collect()
}

fn exact_integer_scale(
    input_w: usize,
    input_h: usize,
    output_w: i32,
    output_h: i32,
) -> Result<usize> {
    let (output_w, output_h) = (usize::try_from(output_w)?, usize::try_from(output_h)?);
    anyhow::ensure!(input_w > 0 && input_h > 0, "NeoAccel empty input");
    anyhow::ensure!(
        output_w % input_w == 0 && output_h % input_h == 0,
        "NeoAccel requires an integer image scale"
    );
    let sx = output_w / input_w;
    let sy = output_h / input_h;
    anyhow::ensure!(sx == sy && (1..=8).contains(&sx), "NeoAccel invalid scale");
    Ok(sx)
}

struct DmlDirectOutput {
    key: u64,
    binding: IoBinding,
    resource: ID3D12Resource,
    device: ID3D12Device,
    allocation: *mut c_void,
    dims: Vec<i64>,
    byte_len: usize,
    heap_byte_len: u64,
    luid: [u8; 8],
    input_size: (i32, i32),
    copy: DmlCopyContext,
    dml_allocator: Allocator,
}

struct DmlCopyContext {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    value: u64,
}

struct DmlDirectInput {
    key: u64,
    resource: ID3D12Resource,
    device: ID3D12Device,
    allocation: *mut c_void,
    dims: [i64; 4],
    byte_len: usize,
    heap_byte_len: u64,
    luid: [u8; 8],
    input_size: (i32, i32),
}

struct DmlTemporalGpuBridge {
    // `output.binding` retains the current input/output OrtValues. Drop the
    // binding-owning output before freeing packed/history allocations.
    output: DmlDirectOutput,
    packed_input: DmlDirectInput,
    history: Vec<DmlDirectInput>,
    size: (i32, i32),
    frame_count: usize,
    write_index: usize,
    valid_frames: usize,
}

struct TensorRtGpuBridge {
    // ORT values retain raw pointers into the shared CUDA mappings. Rust drops
    // fields in declaration order, so bindings/values/names must die before
    // the CudaSharedBuffer mappings they reference.
    binding: IoBinding,
    _input_value: TensorRefMut<'static, f16>,
    _output_value: TensorRefMut<'static, f16>,
    _input_name: CString,
    _output_name: CString,
    input: CudaSharedBuffer,
    output: CudaSharedBuffer,
    input_size: (i32, i32),
    output_size: (i32, i32),
    timings: TensorRtGpuTimings,
}

struct TensorRtTemporalGpuBridge {
    binding: IoBinding,
    _input_value: TensorRefMut<'static, f16>,
    _output_value: TensorRefMut<'static, f16>,
    _input_name: CString,
    _output_name: CString,
    history: Vec<CudaSharedBuffer>,
    packed_input: CudaSharedBuffer,
    output: CudaSharedBuffer,
    size: (i32, i32),
    frame_count: usize,
    write_index: usize,
    valid_frames: usize,
    timings: TensorRtGpuTimings,
}

struct TensorRtInterpInvocation {
    binding: IoBinding,
    _input_value: InterpTensorValue,
    _output_value: InterpTensorValue,
    _input_name: CString,
    _output_name: CString,
    packed_input: CudaSharedBuffer,
    output: CudaSharedBuffer,
}

struct TensorRtInterpGpuBridge {
    history: Vec<CudaSharedBuffer>,
    invocations: Vec<TensorRtInterpInvocation>,
    size: (i32, i32),
    padded_size: (i32, i32),
    channels: usize,
    frame_count: usize,
    slot_count: usize,
    generation: u64,
}

unsafe impl Send for TensorRtInterpInvocation {}
unsafe impl Send for TensorRtInterpGpuBridge {}

impl Drop for TensorRtInterpInvocation {
    fn drop(&mut self) {
        self.binding.clear();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DmlInterpInteropMode {
    Direct,
    // Keep the fast direct-shared input path, but ask ORT/DirectML to own its
    // output allocation and copy that result into the GL-shared D3D12 buffer
    // before downstream GLSL samples it. This isolates the provider output
    // lifetime/visibility issue without adding CPU readback or glFinish.
    ProviderOutputCopy,
    ConservativeCopy,
}

impl DmlInterpInteropMode {
    fn label(self) -> &'static str {
        match self {
            Self::Direct => "direct-shared",
            Self::ProviderOutputCopy => "provider-output-copy",
            Self::ConservativeCopy => "provider-copy-safe",
        }
    }
}

struct DmlInterpInvocation {
    // Same lifetime rule as TensorRT: release ORT values before the DirectML
    // allocations backing their raw pointers.
    _input_value: InterpTensorValue,
    _output_value: Option<InterpTensorValue>,
    _input_name: CString,
    _output_name: CString,
    // Conservative mode stages GL writes here, then copies them on D3D12 into
    // `safe_input`, so DirectML never directly consumes a GL-imported buffer.
    input: DmlDirectInput,
    safe_input: Option<DmlDirectInput>,
    output: DmlDirectOutput,
}

struct DmlInterpGpuBridge {
    history: Vec<DmlDirectInput>,
    invocations: Vec<DmlInterpInvocation>,
    size: (i32, i32),
    padded_size: (i32, i32),
    channels: usize,
    frame_count: usize,
    slot_count: usize,
    generation: u64,
    interop_mode: DmlInterpInteropMode,
}

unsafe impl Send for DmlInterpInvocation {}
unsafe impl Send for DmlInterpGpuBridge {}

impl Drop for DmlInterpInvocation {
    fn drop(&mut self) {
        // IoBinding retains OrtValues that point into both `input` and
        // `output`. Clear it before Rust drops either backing allocation,
        // including constructor/import rollback paths.
        self.output.binding.clear();
    }
}

enum InterpTensorValue {
    F16(TensorRefMut<'static, f16>),
    F32(TensorRefMut<'static, f32>),
}

impl InterpTensorValue {
    fn ptr(&self) -> *const ort::sys::OrtValue {
        match self {
            Self::F16(v) => v.ptr(),
            Self::F32(v) => v.ptr(),
        }
    }
}

unsafe impl Send for TensorRtTemporalGpuBridge {}

#[derive(Default)]
struct TensorRtGpuTimings {
    samples: Vec<[f64; 4]>,
}

impl TensorRtGpuTimings {
    fn record(&mut self, model: &str, pack: f64, run: f64, unpack: f64, total: f64) {
        if std::env::var_os("CHIDESCALER_TRT_PER_FRAME_LOG").is_some() {
            log::debug!(
                "trt-gpu-timing: model={model} pack_ms={pack:.3} run_ms={run:.3} unpack_ms={unpack:.3} total_ms={total:.3}"
            );
        }
        self.samples.push([pack, run, unpack, total]);
        if self.samples.len() < 120 {
            return;
        }
        let percentile = |column: usize, quantile: f64| {
            let mut values = self
                .samples
                .iter()
                .map(|sample| sample[column])
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            values[((values.len() - 1) as f64 * quantile).round() as usize]
        };
        log::debug!(
            "trt-gpu-bridge-summary: model={} samples={} total_p50_ms={:.3} total_p95_ms={:.3} total_p99_ms={:.3} total_max_ms={:.3} pack_p50_ms={:.3} run_p50_ms={:.3} unpack_p50_ms={:.3}",
            model,
            self.samples.len(),
            percentile(3, 0.50),
            percentile(3, 0.95),
            percentile(3, 0.99),
            self.samples
                .iter()
                .map(|sample| sample[3])
                .fold(0.0, f64::max),
            percentile(0, 0.50),
            percentile(1, 0.50),
            percentile(2, 0.50),
        );
        self.samples.clear();
    }
}

impl Drop for TensorRtGpuBridge {
    fn drop(&mut self) {
        self.binding.clear();
    }
}

unsafe impl Send for TensorRtGpuBridge {}

// The stage itself is mutex-protected and all calls still happen on the
// render thread. The raw pointer is an ORT-owned DirectML allocation wrapper.
unsafe impl Send for DmlDirectOutput {}
unsafe impl Send for DmlDirectInput {}

impl DmlDirectOutput {
    fn shared_handle(&self) -> Result<HANDLE> {
        Ok(unsafe {
            self.device
                .CreateSharedHandle(&self.resource, None, 0x1000_0000, PCWSTR::null())?
        })
    }
}

impl Drop for DmlDirectOutput {
    fn drop(&mut self) {
        self.binding.clear();
        if !self.allocation.is_null() {
            let released = if let Ok(api) = dml_api() {
                let status = unsafe { (api.FreeGPUAllocation)(self.allocation) };
                match ort_status(status) {
                    Ok(()) => true,
                    Err(error) => {
                        log::warn!("DirectML shared output release failed: {error:#}");
                        false
                    }
                }
            } else {
                log::error!("DirectML shared output release failed: provider API unavailable");
                false
            };
            dml_shared_released(self.heap_byte_len, released);
            self.allocation = std::ptr::null_mut();
        }
    }
}

impl Drop for DmlCopyContext {
    fn drop(&mut self) {
        if !self.event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.event);
            }
            self.event = HANDLE::default();
        }
    }
}

impl DmlCopyContext {
    fn new(device: &ID3D12Device) -> Result<Self> {
        // v479: state transitions involving UAV resources are not legal work for
        // a COPY command queue.  Use a DIRECT queue so every transfer can state
        // exactly what DirectML and OpenGL own before/after the copy.
        let queue = unsafe {
            device.CreateCommandQueue::<ID3D12CommandQueue>(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                ..Default::default()
            })?
        };
        let allocator = unsafe {
            device
                .CreateCommandAllocator::<ID3D12CommandAllocator>(D3D12_COMMAND_LIST_TYPE_DIRECT)?
        };
        let list = unsafe {
            device.CreateCommandList::<_, _, ID3D12GraphicsCommandList>(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?
        };
        unsafe { list.Close()? };
        let fence = unsafe { device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE)? };
        let event = unsafe { CreateEventW(None, false, false, PCWSTR::null())? };
        Ok(Self {
            queue,
            allocator,
            list,
            fence,
            event,
            value: 0,
        })
    }

    fn transition(
        list: &ID3D12GraphicsCommandList,
        resource: &ID3D12Resource,
        before: D3D12_RESOURCE_STATES,
        after: D3D12_RESOURCE_STATES,
    ) {
        if before == after {
            return;
        }
        // windows-rs models the COM pointer inside the transition barrier as
        // ManuallyDrop.  Release our temporary clone immediately after command
        // recording so per-frame barriers do not retain the resource forever.
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(resource.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: before,
                    StateAfter: after,
                }),
            },
        };
        unsafe {
            list.ResourceBarrier(std::slice::from_ref(&barrier));
            let transition = &mut *barrier.Anonymous.Transition;
            ManuallyDrop::drop(&mut transition.pResource);
        }
    }

    fn wait_submitted(&mut self, copy_start: Instant) -> Result<(f64, f64)> {
        let copy_ms = copy_start.elapsed().as_secs_f64() * 1000.0;
        let wait_start = Instant::now();
        if unsafe { self.fence.GetCompletedValue() } < self.value {
            unsafe {
                self.fence.SetEventOnCompletion(self.value, self.event)?;
                let _ = WaitForSingleObject(self.event, INFINITE);
            }
        }
        Ok((copy_ms, wait_start.elapsed().as_secs_f64() * 1000.0))
    }

    /// Copy an ORT/DirectML-owned tensor into the OpenGL-shared buffer.
    /// DirectML allocator resources are kept in UAV state; GL-shared buffers
    /// are kept in COMMON whenever D3D12 is not actively copying them.
    fn copy_and_wait(
        &mut self,
        source: &ID3D12Resource,
        dest: &ID3D12Resource,
        bytes: usize,
    ) -> Result<(f64, f64)> {
        let copy_start = Instant::now();
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(
                &self.allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            Self::transition(
                &self.list,
                source,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            Self::transition(
                &self.list,
                dest,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            self.list.CopyBufferRegion(dest, 0, source, 0, bytes as u64);
            Self::transition(
                &self.list,
                source,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            Self::transition(
                &self.list,
                dest,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COMMON,
            );
            self.list.Close()?;
            let command: ID3D12CommandList = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(command)]);
            self.value = self.value.saturating_add(1);
            self.queue.Signal(&self.fence, self.value)?;
        }
        self.wait_submitted(copy_start)
    }

    /// Copy GL-packed COMMON buffers into DirectML-private UAV buffers.  The
    /// destination is returned to UAV before ORT starts inference, and the
    /// shared source is returned to COMMON before OpenGL sees it again.
    fn copy_many_and_wait(
        &mut self,
        copies: &[(ID3D12Resource, ID3D12Resource, usize)],
    ) -> Result<(f64, f64)> {
        if copies.is_empty() {
            return Ok((0.0, 0.0));
        }
        let copy_start = Instant::now();
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(
                &self.allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            for (source, dest, _) in copies {
                Self::transition(
                    &self.list,
                    source,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                Self::transition(
                    &self.list,
                    dest,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
            }
            for (source, dest, bytes) in copies {
                self.list
                    .CopyBufferRegion(dest, 0, source, 0, *bytes as u64);
            }
            for (source, dest, _) in copies {
                Self::transition(
                    &self.list,
                    source,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );
                Self::transition(
                    &self.list,
                    dest,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
            }
            self.list.Close()?;
            let command: ID3D12CommandList = self.list.cast()?;
            self.queue.ExecuteCommandLists(&[Some(command)]);
            self.value = self.value.saturating_add(1);
            self.queue.Signal(&self.fence, self.value)?;
        }
        self.wait_submitted(copy_start)
    }
}

impl DmlDirectInput {
    fn shared_handle(&self) -> Result<HANDLE> {
        Ok(unsafe {
            self.device
                .CreateSharedHandle(&self.resource, None, 0x1000_0000, PCWSTR::null())?
        })
    }
}

impl Drop for DmlDirectInput {
    fn drop(&mut self) {
        if !self.allocation.is_null() {
            let released = if let Ok(api) = dml_api() {
                let status = unsafe { (api.FreeGPUAllocation)(self.allocation) };
                match ort_status(status) {
                    Ok(()) => true,
                    Err(error) => {
                        log::warn!("DirectML shared input release failed: {error:#}");
                        false
                    }
                }
            } else {
                log::error!("DirectML shared input release failed: provider API unavailable");
                false
            };
            dml_shared_released(self.heap_byte_len, released);
            self.allocation = std::ptr::null_mut();
        }
    }
}

fn create_direct_input(
    device: &ID3D12Device,
    luid: [u8; 8],
    input_size: (i32, i32),
) -> Result<DmlDirectInput> {
    create_direct_input_channels(device, luid, input_size, 3)
}

fn create_direct_input_channels(
    device: &ID3D12Device,
    luid: [u8; 8],
    input_size: (i32, i32),
    channels: usize,
) -> Result<DmlDirectInput> {
    create_direct_input_channels_typed(
        device,
        luid,
        input_size,
        channels,
        std::mem::size_of::<f16>(),
    )
}

fn create_direct_input_channels_typed(
    device: &ID3D12Device,
    luid: [u8; 8],
    input_size: (i32, i32),
    channels: usize,
    element_bytes: usize,
) -> Result<DmlDirectInput> {
    create_direct_input_channels_typed_state(
        device,
        luid,
        input_size,
        channels,
        element_bytes,
        D3D12_RESOURCE_STATE_COMMON,
    )
}

fn create_direct_input_channels_typed_state(
    device: &ID3D12Device,
    luid: [u8; 8],
    input_size: (i32, i32),
    channels: usize,
    element_bytes: usize,
    initial_state: D3D12_RESOURCE_STATES,
) -> Result<DmlDirectInput> {
    let (w, h) = input_size;
    let elements = usize::try_from(w)?
        .checked_mul(usize::try_from(h)?)
        .and_then(|n| n.checked_mul(channels))
        .ok_or_else(|| anyhow!("input byte size overflow"))?;
    let byte_len = elements
        .checked_mul(element_bytes)
        .ok_or_else(|| anyhow!("input byte size overflow"))?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: byte_len as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let allocation_info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    anyhow::ensure!(
        allocation_info.SizeInBytes != 0 && allocation_info.SizeInBytes != u64::MAX,
        "invalid shared D3D12 input allocation size"
    );
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            initial_state,
            None,
            &mut resource,
        )?;
    }
    let resource = resource.ok_or_else(|| anyhow!("shared DirectML input creation failed"))?;
    let mut allocation = std::ptr::null_mut();
    let api = dml_api()?;
    let allocation_status =
        unsafe { (api.CreateGPUAllocationFromD3DResource)(resource.as_raw(), &mut allocation) };
    if let Err(error) = ort_status(allocation_status) {
        if !allocation.is_null() {
            let _ = ort_status(unsafe { (api.FreeGPUAllocation)(allocation) });
        }
        return Err(error);
    }
    anyhow::ensure!(
        !allocation.is_null(),
        "DirectML input allocation wrapper is null"
    );
    dml_shared_created(allocation_info.SizeInBytes);
    Ok(DmlDirectInput {
        key: DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
        resource,
        device: device.clone(),
        allocation,
        dims: [1, channels as i64, h as i64, w as i64],
        byte_len,
        heap_byte_len: allocation_info.SizeInBytes,
        luid,
        input_size,
    })
}

fn create_direct_output<T: PrimitiveTensorElementType + std::fmt::Debug>(
    session: &mut Session,
    run_options: &CancelableRunOptions,
    input_name: &str,
    output_name: &str,
    input: &TensorRef<'_, T>,
    input_size: (i32, i32),
) -> Result<DmlDirectOutput> {
    let memory = MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let dml_allocator = Allocator::new(
        session,
        MemoryInfo::new(
            AllocationDevice::DIRECTML,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?,
    )?;
    let mut discovery = session.create_binding()?;
    let input_c = CString::new(input_name.as_bytes())?;
    ort_status(unsafe {
        (ort::api().BindInput)(discovery.ptr().cast_mut(), input_c.as_ptr(), input.ptr())
    })?;
    discovery.bind_output_to_device(output_name.to_string(), &memory)?;
    let mut outputs = session
        .run_binding_with_options(&discovery, run_options.armed()?)
        .map_err(oerr)?;
    discovery.synchronize_outputs().map_err(oerr)?;
    let output = outputs
        .remove(output_name)
        .ok_or_else(|| anyhow!("DirectML discovery output missing"))?
        .downcast::<TensorValueType<T>>()?;
    anyhow::ensure!(
        output.memory_info().allocation_device().as_str() == "DML",
        "DirectML output silently fell back to CPU"
    );
    let dims: Vec<i64> = output.shape().iter().copied().collect();
    anyhow::ensure!(
        dims.len() == 4 && dims[0] == 1 && dims[1] >= 3,
        "unsupported output shape {dims:?}"
    );
    let elements = dims
        .iter()
        .try_fold(1usize, |total, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| total.checked_mul(dim))
        })
        .ok_or_else(|| anyhow!("invalid output shape {dims:?}"))?;

    let api = dml_api()?;
    let mut raw_resource = std::ptr::null_mut();
    ort_status(unsafe {
        (api.GetD3D12ResourceFromAllocation)(
            dml_allocator.ptr().cast_mut(),
            output.data_ptr().cast_mut(),
            &mut raw_resource,
        )
    })?;
    let borrowed = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
        .ok_or_else(|| anyhow!("DirectML returned a null D3D12 resource"))?;
    let discovered_resource = borrowed.clone();
    drop(output);
    drop(outputs);
    drop(discovery);

    let mut device: Option<ID3D12Device> = None;
    unsafe { discovered_resource.GetDevice(&mut device)? };
    let device = device.ok_or_else(|| anyhow!("DirectML D3D12 device unavailable"))?;
    let luid = luid_bytes(unsafe { device.GetAdapterLuid() });
    let byte_len = elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| anyhow!("output byte size overflow"))?;
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: byte_len as u64,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let allocation_info = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
    anyhow::ensure!(
        allocation_info.SizeInBytes != 0 && allocation_info.SizeInBytes != u64::MAX,
        "invalid shared D3D12 heap allocation size"
    );
    // A committed resource has a shareable implicit heap and can be imported
    // by OpenGL as GL_HANDLE_TYPE_D3D12_RESOURCE_EXT. AMD rejects binding a
    // raw shared heap as GL buffer storage even when the heap import succeeds.
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap,
            D3D12_HEAP_FLAG_SHARED,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )?;
    }
    let resource =
        resource.ok_or_else(|| anyhow!("shared DirectML committed buffer creation failed"))?;
    // Build every fallible Rust/Win32 helper before creating the raw DirectML
    // allocation wrapper. Once CreateGPUAllocationFromD3DResource succeeds,
    // construction must proceed directly into the RAII owner so an error path
    // cannot strand the provider allocation.
    let copy = DmlCopyContext::new(&device)?;
    let binding = session.create_binding()?;
    let mut allocation = std::ptr::null_mut();
    let allocation_status =
        unsafe { (api.CreateGPUAllocationFromD3DResource)(resource.as_raw(), &mut allocation) };
    if let Err(error) = ort_status(allocation_status) {
        if !allocation.is_null() {
            let _ = ort_status(unsafe { (api.FreeGPUAllocation)(allocation) });
        }
        return Err(error);
    }
    anyhow::ensure!(!allocation.is_null(), "DirectML allocation wrapper is null");
    dml_shared_created(allocation_info.SizeInBytes);
    Ok(DmlDirectOutput {
        key: DIRECT_OUTPUT_KEY.fetch_add(1, Ordering::Relaxed),
        binding,
        resource,
        device,
        allocation,
        dims,
        byte_len,
        heap_byte_len: allocation_info.SizeInBytes,
        luid,
        input_size,
        copy,
        dml_allocator,
    })
}

fn run_cpu_tensor_to_dml_shared_output<T: PrimitiveTensorElementType + std::fmt::Debug>(
    session: &mut Session,
    run_options: &CancelableRunOptions,
    input_name: &str,
    output_name: &str,
    input: &TensorRef<'_, T>,
    output: &mut DmlDirectOutput,
) -> Result<(f64, f64)> {
    let memory = MemoryInfo::new(
        AllocationDevice::DIRECTML,
        0,
        AllocatorType::Device,
        MemoryType::Default,
    )?;
    let input_c = CString::new(input_name.as_bytes())?;
    ort_status(unsafe {
        (ort::api().BindInput)(
            output.binding.ptr().cast_mut(),
            input_c.as_ptr(),
            input.ptr(),
        )
    })?;
    output
        .binding
        .bind_output_to_device(output_name.to_string(), &memory)
        .map_err(oerr)?;

    let run_started = Instant::now();
    let result = (|| -> Result<(f64, f64)> {
        let mut provider_outputs = session
            .run_binding_with_options(&output.binding, run_options.armed()?)
            .map_err(oerr)?;
        output.binding.synchronize_outputs().map_err(oerr)?;
        let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
        let copy_started = Instant::now();
        let provider = provider_outputs
            .remove(output_name)
            .ok_or_else(|| anyhow!("DirectML interpolation provider output missing"))?
            .downcast::<TensorValueType<T>>()?;
        anyhow::ensure!(
            provider.memory_info().allocation_device().as_str() == "DML",
            "DirectML interpolation output silently fell back to CPU"
        );
        let provider_dims: Vec<i64> = provider.shape().iter().copied().collect();
        anyhow::ensure!(
            provider_dims == output.dims,
            "DirectML interpolation provider output shape changed: {:?} != {:?}",
            provider_dims,
            output.dims
        );
        let api = dml_api()?;
        let mut raw_resource = std::ptr::null_mut();
        ort_status(unsafe {
            (api.GetD3D12ResourceFromAllocation)(
                output.dml_allocator.ptr().cast_mut(),
                provider.data_ptr().cast_mut(),
                &mut raw_resource,
            )
        })?;
        let borrowed = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
            .ok_or_else(|| anyhow!("DirectML interpolation provider output resource is null"))?;
        let provider_resource = borrowed.clone();
        let _ = output
            .copy
            .copy_and_wait(&provider_resource, &output.resource, output.byte_len)?;
        drop(provider);
        drop(provider_outputs);
        Ok((run_ms, copy_started.elapsed().as_secs_f64() * 1000.0))
    })();
    output.binding.clear();
    result
}

fn dml_interp_interop_mode(
    _gc: &GlContext,
    _stable_post_handoff: bool,
) -> Option<DmlInterpInteropMode> {
    let requested = std::env::var("CHIDESCALER_DML_INTERP_INTEROP")
        .unwrap_or_else(|_| "auto".into())
        .trim()
        .to_ascii_lowercase();
    match requested.as_str() {
        "0" | "off" | "disable" | "cpu" | "nonresident" => None,
        // Keep the old modes available only as explicit diagnostics.  The
        // default must not let DirectML consume a buffer that OpenGL has just
        // written directly.  On NVIDIA the GL fence can report completion while
        // DirectML still observes incompletely visible shared-buffer contents;
        // the result looks like nearest-neighbour blocks/mosaic before any
        // post-GLSL stage is applied.
        "direct" | "fast" => Some(DmlInterpInteropMode::Direct),
        "output-copy" | "post-safe" => Some(DmlInterpInteropMode::ProviderOutputCopy),
        "copy" | "safe" | "conservative" => Some(DmlInterpInteropMode::ConservativeCopy),
        _ => {
            // v479: isolate *both* sides of DirectML interpolation by default,
            // but now keep every D3D12 resource in an explicit legal state at
            // each ownership boundary. GL-shared staging/output stays COMMON,
            // DirectML-private/provider tensors stay UAV, and a DIRECT queue
            // performs state-correct GPU copies between them. No CPU frame
            // readback/upload is introduced.
            Some(DmlInterpInteropMode::ConservativeCopy)
        }
    }
}

fn dml_api() -> Result<&'static ort::sys::OrtDmlApi> {
    let mut ptr = std::ptr::null();
    ort_status(unsafe { (ort::api().GetExecutionProviderApi)(c"DML".as_ptr(), 0, &mut ptr) })?;
    anyhow::ensure!(!ptr.is_null(), "DirectML provider API unavailable");
    Ok(unsafe { &*ptr.cast::<ort::sys::OrtDmlApi>() })
}

fn ort_status(status: ort::sys::OrtStatusPtr) -> Result<()> {
    if status.0.is_null() {
        return Ok(());
    }
    let message = unsafe {
        std::ffi::CStr::from_ptr((ort::api().GetErrorMessage)(status.0))
            .to_string_lossy()
            .into_owned()
    };
    unsafe { (ort::api().ReleaseStatus)(status.0) };
    Err(anyhow!(message))
}

fn luid_bytes(luid: LUID) -> [u8; 8] {
    [
        luid.LowPart as u8,
        (luid.LowPart >> 8) as u8,
        (luid.LowPart >> 16) as u8,
        (luid.LowPart >> 24) as u8,
        luid.HighPart as u8,
        (luid.HighPart >> 8) as u8,
        (luid.HighPart >> 16) as u8,
        (luid.HighPart >> 24) as u8,
    ]
}

impl OnnxStage {
    /// Back-compat helper: midpoint between two frames.
    pub fn process_pair(
        &mut self,
        w: i32,
        h: i32,
        prev: &[u8],
        cur: &[u8],
    ) -> Result<(i32, i32, Vec<u8>)> {
        self.process_pair_t(w, h, prev, cur, 0.5)
    }

    /// vs-mlrt v1 layout: (1, 3*NF + 5, PH, PW) = imgs + timestep + coord
    /// grids + multiplier planes, edge-padded to the model tile multiple.
    /// fp16 models are fed f16.
    fn rife_multi(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        channels: usize,
        t: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        self.rife_multi_strided(w, h, frames, channels, t, 3)
    }

    fn rife_multi_strided(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        channels: usize,
        t: f32,
        stride: usize,
    ) -> Result<(i32, i32, Vec<u8>)> {
        let nf = frames.len();
        anyhow::ensure!(stride >= 3, "frame stride too small");
        anyhow::ensure!(channels >= 3 * nf + 1, "model wants more frames");
        let (w, h) = (w as usize, h as usize);
        anyhow::ensure!(w > 0 && h > 0, "empty frame");
        let pw = w.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE;
        let ph = h.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE;
        let pn = pw * ph;
        use rayon::prelude::*;
        let aux = 3 * nf;
        let shape = vec![1i64, channels as i64, ph as i64, pw as i64];
        let pack_t0 = Instant::now();
        let const_key = (pw, ph, channels, aux);
        let frame_key = (w, h, pw, ph, stride);
        let const_ok = self.pack_const_key == Some(const_key);
        let t_ok = const_ok && self.pack_t_cached == Some(t.clamp(0.0, 1.0));
        // v655 correctness guard for explicit cross-GPU compute. The historical
        // pack optimization identified the overlapping frame in consecutive
        // RIFE pairs by Vec data pointer alone: (A,B) -> (B,C). On a slower
        // asynchronous secondary GPU, allocator reuse can give a different new
        // frame the same address after the previous Arc/Vec is released. That
        // makes stale B planes become the next A planes and can corrupt only the
        // generated midpoint while real frames remain correct.
        //
        // production_selected_luid() is set only when explicit compute differs
        // from the presentation OpenGL adapter, so the known-good same-GPU RTX
        // DirectML path retains the fast reuse optimization unchanged.
        let cross_gpu_directml = self.provider == OnnxProvider::DirectML
            && crate::render::vulkan_gpu::production_selected_luid().is_some();
        if cross_gpu_directml && !self.cross_gpu_rife_full_repack_logged {
            log::info!(
                "rife-pack-safety: backend=DirectML route=cross-gpu-full-repack model={} reason=pointer-identity-not-sufficient same-gpu-fast-reuse=preserved",
                self.name
            );
            self.cross_gpu_rife_full_repack_logged = true;
        }
        let reuse_prev = !cross_gpu_directml
            && nf == 2
            && self.pack_frame_key == Some(frame_key)
            && self.pack_second_ptr == Some(frames[0].as_ptr() as usize);
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            prepare_len_f16(&mut self.scratch_f16, pn * channels);
            let x = self.scratch_f16.as_mut_slice();
            if reuse_prev {
                x.copy_within(3 * pn..6 * pn, 0);
                fill_padded_rgb_frame_f16(x, frames[1], 3, w, h, pw, ph, pn, stride);
            } else {
                fill_padded_rgb_planes_f16(x, frames, frames.len(), w, h, pw, ph, pn, stride);
            }
            if !t_ok {
                x[aux * pn..(aux + 1) * pn].fill(f16::from_f32(t.clamp(0.0, 1.0)));
            }
            if !const_ok && channels >= aux + 3 {
                let row: Vec<f16> = (0..pw)
                    .map(|j| f16::from_f32(2.0 * j as f32 / (pw - 1) as f32 - 1.0))
                    .collect();
                x[(aux + 1) * pn..(aux + 2) * pn]
                    .par_chunks_mut(pw)
                    .for_each(|dst| dst.copy_from_slice(&row));
                x[(aux + 2) * pn..(aux + 3) * pn]
                    .par_chunks_mut(pw)
                    .enumerate()
                    .for_each(|(i, row)| {
                        let v = f16::from_f32(2.0 * i as f32 / (ph - 1) as f32 - 1.0);
                        row.fill(v);
                    });
            }
            if !const_ok && channels >= aux + 5 {
                x[(aux + 3) * pn..(aux + 4) * pn].fill(f16::from_f32(2.0 / (pw - 1) as f32));
                x[(aux + 4) * pn..(aux + 5) * pn].fill(f16::from_f32(2.0 / (ph - 1) as f32));
            }
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_t0.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        } else {
            prepare_len_f32(&mut self.scratch_f32, pn * channels);
            let x = self.scratch_f32.as_mut_slice();
            if reuse_prev {
                x.copy_within(3 * pn..6 * pn, 0);
                fill_padded_rgb_frame_f32(x, frames[1], 3, w, h, pw, ph, pn, stride);
            } else {
                fill_padded_rgb_planes_f32(x, frames, frames.len(), w, h, pw, ph, pn, stride);
            }
            if !t_ok {
                x[aux * pn..(aux + 1) * pn].fill(t.clamp(0.0, 1.0));
            }
            if !const_ok && channels >= aux + 3 {
                let row: Vec<f32> = (0..pw)
                    .map(|j| 2.0 * j as f32 / (pw - 1) as f32 - 1.0)
                    .collect();
                x[(aux + 1) * pn..(aux + 2) * pn]
                    .par_chunks_mut(pw)
                    .for_each(|dst| dst.copy_from_slice(&row));
                x[(aux + 2) * pn..(aux + 3) * pn]
                    .par_chunks_mut(pw)
                    .enumerate()
                    .for_each(|(i, row)| {
                        let v = 2.0 * i as f32 / (ph - 1) as f32 - 1.0;
                        row.fill(v);
                    });
            }
            if !const_ok && channels >= aux + 5 {
                x[(aux + 3) * pn..(aux + 4) * pn].fill(2.0 / (pw - 1) as f32);
                x[(aux + 4) * pn..(aux + 5) * pn].fill(2.0 / (ph - 1) as f32);
            }
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_t0.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        };
        self.pack_const_key = Some(const_key);
        self.pack_t_cached = Some(t.clamp(0.0, 1.0));
        self.pack_frame_key = Some(frame_key);
        self.pack_second_ptr = frames.get(1).map(|frame| frame.as_ptr() as usize);
        let out_ms_t0 = Instant::now();
        let rgb8 = {
            let out = outputs
                .get(self.out_name.as_str())
                .ok_or_else(|| anyhow!("no output"))?;
            interp_output_to_rgb8(out, w, h, "interp")?
        };
        let out_ms = out_ms_t0.elapsed().as_secs_f64() * 1000.0;
        drop(outputs);
        self.last_interp_profile = Some(InterpProfile {
            pack_ms,
            run_ms,
            out_ms,
            padded_size: (pw, ph),
        });
        Ok((w as i32, h as i32, rgb8))
    }
}

impl OnnxStage {
    /// distilDRBA: (1, 14, PH, PW) = 4 frames + timestep + scale(=1.0),
    /// edge-padded to a 64 multiple. Interpolates frames[1]..frames[2] at `t`.
    fn drba(&mut self, w: i32, h: i32, frames: &[&[u8]], t: f32) -> Result<(i32, i32, Vec<u8>)> {
        self.drba_strided(w, h, frames, t, 3)
    }

    fn drba_strided(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        t: f32,
        stride: usize,
    ) -> Result<(i32, i32, Vec<u8>)> {
        anyhow::ensure!(stride >= 3, "frame stride too small");
        let (w, h) = (w as usize, h as usize);
        anyhow::ensure!(w > 0 && h > 0, "empty frame");
        let pw = w.div_ceil(64) * 64;
        let ph = h.div_ceil(64) * 64;
        let pn = pw * ph;
        let channels = 14usize;
        let shape = vec![1i64, channels as i64, ph as i64, pw as i64];
        let pack_t0 = Instant::now();
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            prepare_len_f16(&mut self.scratch_f16, pn * channels);
            let x = self.scratch_f16.as_mut_slice();
            fill_padded_rgb_planes_f16(x, frames, 4, w, h, pw, ph, pn, stride);
            x[12 * pn..13 * pn].fill(f16::from_f32(t.clamp(0.0, 1.0)));
            x[13 * pn..14 * pn].fill(f16::from_f32(1.0)); // scale placeholder
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_t0.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        } else {
            prepare_len_f32(&mut self.scratch_f32, pn * channels);
            let x = self.scratch_f32.as_mut_slice();
            fill_padded_rgb_planes_f32(x, frames, 4, w, h, pw, ph, pn, stride);
            x[12 * pn..13 * pn].fill(t.clamp(0.0, 1.0));
            x[13 * pn..14 * pn].fill(1.0); // scale placeholder
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            let run_ms = run_t0.elapsed().as_secs_f64() * 1000.0;
            (outputs, pack_ms, run_ms)
        };
        let out_ms_t0 = Instant::now();
        let rgb8 = {
            let out = outputs
                .get(self.out_name.as_str())
                .ok_or_else(|| anyhow!("no output"))?;
            interp_output_to_rgb8(out, w, h, "DRBA")?
        };
        let out_ms = out_ms_t0.elapsed().as_secs_f64() * 1000.0;
        drop(outputs);
        self.last_interp_profile = Some(InterpProfile {
            pack_ms,
            run_ms,
            out_ms,
            padded_size: (pw, ph),
        });
        Ok((w as i32, h as i32, rgb8))
    }

    fn drba_batch_strided(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        ts: &[f32],
        stride: usize,
    ) -> Result<Vec<(i32, i32, Vec<u8>)>> {
        anyhow::ensure!(frames.len() >= 4, "DRBA needs 4 frames");
        anyhow::ensure!(!ts.is_empty(), "DRBA batch needs a timestep");
        let (w, h) = (w as usize, h as usize);
        let pw = w.div_ceil(64) * 64;
        let ph = h.div_ceil(64) * 64;
        let pn = pw * ph;
        let channels = 14usize;
        let batch = ts.len();
        let shape = vec![batch as i64, channels as i64, ph as i64, pw as i64];
        let pack_t0 = Instant::now();
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            prepare_len_f16(&mut self.scratch_f16, batch * channels * pn);
            let x = self.scratch_f16.as_mut_slice();
            let item_len = channels * pn;
            let (first, rest) = x.split_at_mut(item_len);
            fill_padded_rgb_planes_f16(first, frames, 4, w, h, pw, ph, pn, stride);
            first[12 * pn..13 * pn].fill(f16::from_f32(ts[0].clamp(0.0, 1.0)));
            first[13 * pn..14 * pn].fill(f16::from_f32(1.0));
            for (item, timestep) in rest.chunks_exact_mut(item_len).zip(&ts[1..]) {
                item[..12 * pn].copy_from_slice(&first[..12 * pn]);
                item[12 * pn..13 * pn].fill(f16::from_f32(timestep.clamp(0.0, 1.0)));
                item[13 * pn..14 * pn].fill(f16::from_f32(1.0));
            }
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            (outputs, pack_ms, run_t0.elapsed().as_secs_f64() * 1000.0)
        } else {
            prepare_len_f32(&mut self.scratch_f32, batch * channels * pn);
            let x = self.scratch_f32.as_mut_slice();
            let item_len = channels * pn;
            let (first, rest) = x.split_at_mut(item_len);
            fill_padded_rgb_planes_f32(first, frames, 4, w, h, pw, ph, pn, stride);
            first[12 * pn..13 * pn].fill(ts[0].clamp(0.0, 1.0));
            first[13 * pn..14 * pn].fill(1.0);
            for (item, timestep) in rest.chunks_exact_mut(item_len).zip(&ts[1..]) {
                item[..12 * pn].copy_from_slice(&first[..12 * pn]);
                item[12 * pn..13 * pn].fill(timestep.clamp(0.0, 1.0));
                item[13 * pn..14 * pn].fill(1.0);
            }
            let tin = TensorRef::from_array_view((shape, &*x)).map_err(oerr)?;
            let pack_ms = pack_t0.elapsed().as_secs_f64() * 1000.0;
            let run_t0 = Instant::now();
            let outputs = self
                .session
                .run_with_options(
                    ort::inputs![self.in_name.as_str() => tin],
                    self.run_options.armed()?,
                )
                .map_err(oerr)?;
            (outputs, pack_ms, run_t0.elapsed().as_secs_f64() * 1000.0)
        };
        let out_t0 = Instant::now();
        let out = outputs
            .get(self.out_name.as_str())
            .ok_or_else(|| anyhow!("no output"))?;
        let images = interp_output_batch_to_rgb8(out, batch, w, h, "DRBA")?;
        let out_ms = out_t0.elapsed().as_secs_f64() * 1000.0;
        drop(outputs);
        self.last_interp_profile = Some(InterpProfile {
            pack_ms,
            run_ms,
            out_ms,
            padded_size: (pw, ph),
        });
        Ok(images
            .into_iter()
            .map(|image| (w as i32, h as i32, image))
            .collect())
    }
}

fn prepare_len_f32(buf: &mut Vec<f32>, len: usize) {
    if buf.len() != len {
        buf.resize(len, 0.0);
    }
}

fn prepare_len_f16(buf: &mut Vec<f16>, len: usize) {
    if buf.len() != len {
        buf.resize(len, f16::ZERO);
    }
}

fn fill_padded_rgb_planes_f32(
    x: &mut [f32],
    frames: &[&[u8]],
    max_frames: usize,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    pn: usize,
    stride: usize,
) {
    for (k, src) in frames.iter().take(max_frames).enumerate() {
        fill_padded_rgb_frame_f32(x, src, 3 * k, w, h, pw, ph, pn, stride);
    }
}

fn fill_padded_rgb_frame_f32(
    x: &mut [f32],
    src: &[u8],
    base: usize,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    pn: usize,
    stride: usize,
) {
    use rayon::prelude::*;
    x[base * pn..(base + 3) * pn]
        .par_chunks_mut(pn)
        .enumerate()
        .for_each(|(c, plane)| {
            for y in 0..ph {
                let sy = y.min(h - 1);
                for x in 0..pw {
                    let sx = x.min(w - 1);
                    plane[y * pw + x] = src[(sy * w + sx) * stride + c] as f32 * (1.0 / 255.0);
                }
            }
        });
}

/// u8 -> f16 unorm conversion table: f16::from_f32 per pixel is a software
/// float conversion and dominated the RIFE pack time (13M+ conversions per
/// 1080p pair); a 256-entry lookup makes it a byte-indexed copy.
fn u8_to_f16_lut() -> &'static [f16; 256] {
    static LUT: std::sync::OnceLock<[f16; 256]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [f16::ZERO; 256];
        for (i, v) in t.iter_mut().enumerate() {
            *v = f16::from_f32(i as f32 * (1.0 / 255.0));
        }
        t
    })
}

fn f16_unorm_to_u8_lut() -> &'static [u8; 65536] {
    static LUT: std::sync::OnceLock<[u8; 65536]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [0u8; 65536];
        for (bits, value) in table.iter_mut().enumerate() {
            let sample = f16::from_bits(bits as u16).to_f32();
            *value = (sample.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
        table
    })
}

fn fill_padded_rgb_planes_f16(
    x: &mut [f16],
    frames: &[&[u8]],
    max_frames: usize,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    pn: usize,
    stride: usize,
) {
    for (k, src) in frames.iter().take(max_frames).enumerate() {
        fill_padded_rgb_frame_f16(x, src, 3 * k, w, h, pw, ph, pn, stride);
    }
}

fn fill_padded_rgb_frame_f16(
    x: &mut [f16],
    src: &[u8],
    base: usize,
    w: usize,
    h: usize,
    pw: usize,
    ph: usize,
    pn: usize,
    stride: usize,
) {
    use rayon::prelude::*;
    let lut = u8_to_f16_lut();
    // Parallelize over rows of all three planes. Three-way plane-only
    // parallelism left most CPU cores idle at 1080p.
    x[base * pn..(base + 3) * pn]
        .par_chunks_mut(pw)
        .enumerate()
        .for_each(|(i, row)| {
            let c = i / ph;
            let sy = (i % ph).min(h - 1);
            let src_row = &src[sy * w * stride..(sy * w + w) * stride];
            for (dst, px) in row[..w].iter_mut().zip(src_row.chunks_exact(stride)) {
                *dst = lut[px[c] as usize];
            }
            let last = row[w - 1];
            row[w..].fill(last);
        });
}

fn interp_output_to_rgb8(
    out: &DynValue,
    want_w: usize,
    want_h: usize,
    label: &str,
) -> Result<Vec<u8>> {
    if let Ok((shape, v)) = out.try_extract_tensor::<f32>() {
        let dims: Vec<i64> = shape[..].to_vec();
        let (ow, oh) = interp_out_dims(&dims, label)?;
        return nchw_rgb_f32_to_rgb8(v, ow, oh, want_w, want_h, label);
    }
    let (shape, v) = out.try_extract_tensor::<f16>().map_err(oerr)?;
    let dims: Vec<i64> = shape[..].to_vec();
    let (ow, oh) = interp_out_dims(&dims, label)?;
    nchw_rgb_f16_to_rgb8(v, ow, oh, want_w, want_h, label)
}

fn interp_output_batch_to_rgb8(
    out: &DynValue,
    want_batch: usize,
    want_w: usize,
    want_h: usize,
    label: &str,
) -> Result<Vec<Vec<u8>>> {
    if let Ok((shape, v)) = out.try_extract_tensor::<f32>() {
        let dims: Vec<i64> = shape[..].to_vec();
        let (ow, oh) = interp_out_dims(&dims, label)?;
        anyhow::ensure!(
            dims[0] as usize >= want_batch,
            "short {label} batch {dims:?}"
        );
        let item_len = 3 * ow * oh;
        return (0..want_batch)
            .map(|index| {
                nchw_rgb_f32_to_rgb8(
                    &v[index * item_len..(index + 1) * item_len],
                    ow,
                    oh,
                    want_w,
                    want_h,
                    label,
                )
            })
            .collect();
    }
    let (shape, v) = out.try_extract_tensor::<f16>().map_err(oerr)?;
    let dims: Vec<i64> = shape[..].to_vec();
    let (ow, oh) = interp_out_dims(&dims, label)?;
    anyhow::ensure!(
        dims[0] as usize >= want_batch,
        "short {label} batch {dims:?}"
    );
    let item_len = 3 * ow * oh;
    (0..want_batch)
        .map(|index| {
            nchw_rgb_f16_to_rgb8(
                &v[index * item_len..(index + 1) * item_len],
                ow,
                oh,
                want_w,
                want_h,
                label,
            )
        })
        .collect()
}

fn interp_out_dims(dims: &[i64], label: &str) -> Result<(usize, usize)> {
    anyhow::ensure!(
        dims.len() == 4 && dims[1] >= 3,
        "unexpected {label} output {dims:?}"
    );
    let (ow, oh) = (dims[3] as usize, dims[2] as usize);
    anyhow::ensure!(ow > 0 && oh > 0, "unexpected {label} output {dims:?}");
    Ok((ow, oh))
}

fn nchw_rgb_f32_to_rgb8(
    v: &[f32],
    ow: usize,
    oh: usize,
    want_w: usize,
    want_h: usize,
    label: &str,
) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let on = ow * oh;
    anyhow::ensure!(v.len() >= 3 * on, "{label} output buffer too short");
    let mut rgb8 = vec![0u8; want_w * want_h * 3];
    rgb8.par_chunks_mut(3 * 8192)
        .enumerate()
        .for_each(|(blk, out)| {
            let base = blk * 8192;
            for (j, px) in out.chunks_mut(3).enumerate() {
                let p = base + j;
                let y = p / want_w;
                let x = p - y * want_w;
                let si = y.min(oh - 1) * ow + x.min(ow - 1);
                px[0] = unit_to_u8(v[si]);
                px[1] = unit_to_u8(v[on + si]);
                px[2] = unit_to_u8(v[2 * on + si]);
            }
        });
    Ok(rgb8)
}

fn nchw_rgb_f16_to_rgb8(
    v: &[f16],
    ow: usize,
    oh: usize,
    want_w: usize,
    want_h: usize,
    label: &str,
) -> Result<Vec<u8>> {
    use rayon::prelude::*;
    let on = ow * oh;
    anyhow::ensure!(v.len() >= 3 * on, "{label} output buffer too short");
    let mut rgb8 = vec![0u8; want_w * want_h * 3];
    let lut = f16_unorm_to_u8_lut();
    rgb8.par_chunks_mut(3 * 8192)
        .enumerate()
        .for_each(|(blk, out)| {
            let base = blk * 8192;
            for (j, px) in out.chunks_mut(3).enumerate() {
                let p = base + j;
                let y = p / want_w;
                let x = p - y * want_w;
                let si = y.min(oh - 1) * ow + x.min(ow - 1);
                px[0] = lut[v[si].to_bits() as usize];
                px[1] = lut[v[on + si].to_bits() as usize];
                px[2] = lut[v[2 * on + si].to_bits() as usize];
            }
        });
    Ok(rgb8)
}

#[inline]
fn unit_to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

fn tensorrt_model_cache_dir(cache_root: &Path, model: &Path, cache_variant: &[u8]) -> PathBuf {
    // Cache identity is content-based. Repacking or moving the portable app must
    // not create another engine folder when the ONNX bytes and builder settings
    // are unchanged.
    let mut hasher = Sha256::new();
    hasher.update(b"neo-trt-cache-v2");
    hasher.update(cache_variant);
    if let Ok(bytes) = std::fs::read(model) {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    } else {
        // The load will report the real model error later. Keep a deterministic
        // fallback key so cache maintenance itself never hides that error.
        hasher.update(model.to_string_lossy().as_bytes());
    }
    let digest = hasher.finalize();
    let hash = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    cache_root.join(hash)
}

fn legacy_tensorrt_model_cache_dir_v332(
    cache_root: &Path,
    model: &Path,
    cache_variant: &[u8],
) -> PathBuf {
    let mut hasher = Sha256::new();
    hasher.update(cache_variant);
    hasher.update(model.to_string_lossy().as_bytes());
    if let Ok(metadata) = std::fs::metadata(model) {
        hasher.update(metadata.len().to_le_bytes());
        if let Ok(modified) = metadata.modified()
            && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
        {
            hasher.update(duration.as_nanos().to_le_bytes());
        }
    }
    if let Ok(bytes) = std::fs::read(model) {
        hasher.update(&bytes);
    }
    let digest = hasher.finalize();
    let hash = digest[..16]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    cache_root.join(hash)
}

fn migrate_legacy_tensorrt_cache(
    cache_root: &Path,
    model: &Path,
    cache_variant: &[u8],
    cache_dir: &Path,
) {
    if cache_dir.exists() {
        return;
    }
    let legacy = legacy_tensorrt_model_cache_dir_v332(cache_root, model, cache_variant);
    if legacy == cache_dir || !legacy.exists() {
        return;
    }
    if let Err(error) = std::fs::create_dir_all(cache_root) {
        log::debug!(
            "tensorrt-cache: legacy migration root unavailable path={} error={error}",
            cache_root.display()
        );
        return;
    }
    match std::fs::rename(&legacy, cache_dir) {
        Ok(()) => log::info!(
            "tensorrt-cache: migrated legacy identity old={} new={}",
            legacy.display(),
            cache_dir.display()
        ),
        Err(error) => log::debug!(
            "tensorrt-cache: legacy identity retained old={} new={} error={error}",
            legacy.display(),
            cache_dir.display()
        ),
    }
}

const TENSORRT_BUILD_MARKER: &str = ".neo-build-in-progress";
const TENSORRT_PROVIDER_VERIFIED_MARKER: &str = ".neo-provider-verified";

fn tensorrt_gpu_interop_enabled() -> bool {
    matches!(
        std::env::var("CHIDESCALER_TRT_GPU_INTEROP")
            .unwrap_or_else(|_| "auto".into())
            .trim()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "copy" | "direct" | "auto"
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorRtSyncMode {
    Safe,
    Ort,
}

fn tensorrt_sync_mode() -> TensorRtSyncMode {
    match std::env::var("CHIDESCALER_TRT_SYNC_MODE")
        .unwrap_or_else(|_| "safe".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "ort" => TensorRtSyncMode::Ort,
        _ => TensorRtSyncMode::Safe,
    }
}

fn tensorrt_builder_level() -> u8 {
    std::env::var("CHIDESCALER_TRT_BUILDER_LEVEL")
        .ok()
        .and_then(|value| value.trim().parse::<u8>().ok())
        .filter(|value| *value <= 5)
        .unwrap_or(3)
}

fn tensorrt_workspace_mb() -> Option<usize> {
    let raw = std::env::var("CHIDESCALER_TRT_WORKSPACE_MB").ok()?;
    match raw.trim().parse::<usize>() {
        Ok(value) if (256..=32768).contains(&value) => Some(value),
        _ => {
            log::warn!(
                "tensorrt-builder-config: ignored invalid CHIDESCALER_TRT_WORKSPACE_MB={raw:?}"
            );
            None
        }
    }
}

fn tensorrt_shape_ready_path(cache_dir: &Path, width: i32, height: i32) -> PathBuf {
    cache_dir.join(format!("neo-shape-{width}x{height}.ready"))
}

fn tensorrt_engine_file_names(cache_dir: &Path) -> Vec<String> {
    std::fs::read_dir(cache_dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let path = entry.path();
            let supported = path
                .extension()
                .and_then(|extension| extension.to_str())
                .is_some_and(|extension| {
                    matches!(
                        extension.to_ascii_lowercase().as_str(),
                        "engine" | "cache" | "timing"
                    )
                });
            (supported && entry.metadata().ok().is_some_and(|meta| meta.len() > 0))
                .then(|| entry.file_name().to_string_lossy().into_owned())
        })
        .collect()
}

fn is_ort_provider_profile(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("ort-provider-") && name.ends_with(".json"))
}

fn ort_provider_profile_pid(path: &Path) -> Option<u32> {
    path.file_name()
        .and_then(|name| name.to_str())?
        .strip_prefix("ort-provider-")?
        .split('-')
        .next()?
        .parse::<u32>()
        .ok()
}

fn path_age(path: &Path) -> Option<Duration> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .elapsed()
        .ok()
}

fn remove_file_accounted(path: &Path, removed: &mut u64, removed_bytes: &mut u64) {
    let bytes = std::fs::metadata(path).map(|meta| meta.len()).unwrap_or(0);
    match std::fs::remove_file(path) {
        Ok(()) => {
            *removed = removed.saturating_add(1);
            *removed_bytes = removed_bytes.saturating_add(bytes);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => log::debug!(
            "tensorrt-cache: temporary file retained path={} error={error}",
            path.display()
        ),
    }
}

fn remove_empty_diagnostic_dirs(root: &Path) {
    let mut dirs = Vec::new();
    let mut pending = vec![root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        // A live ORT session may have created its profile directory but not the
        // JSON file yet. Never remove or descend into that session subtree.
        if dir != root && diagnostics_session_pid(&dir).is_some_and(process_is_alive) {
            continue;
        }
        dirs.push(dir.clone());
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            if entry.path().is_dir() {
                pending.push(entry.path());
            }
        }
    }
    dirs.sort_by_key(|dir| std::cmp::Reverse(dir.components().count()));
    for dir in dirs {
        if dir != root {
            let _ = std::fs::remove_dir(&dir);
        }
    }
}

#[cfg(windows)]
fn process_is_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    unsafe {
        match OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) {
            Ok(handle) => {
                let _ = CloseHandle(handle);
                true
            }
            Err(_) => false,
        }
    }
}

#[cfg(not(windows))]
fn process_is_alive(pid: u32) -> bool {
    pid == std::process::id()
}

fn diagnostics_session_pid(path: &Path) -> Option<u32> {
    path.file_name()
        .and_then(|name| name.to_str())?
        .strip_prefix("session-")?
        .parse::<u32>()
        .ok()
}

fn cleanup_tensorrt_diagnostics(cache_root: &Path) -> (u64, u64) {
    const STALE_AGE: Duration = Duration::from_secs(10 * 60);
    const HARD_LIMIT_BYTES: u64 = 32 * 1024 * 1024;
    const HARD_LIMIT_FILES: usize = 8;
    const ACTIVE_GRACE: Duration = Duration::from_secs(2 * 60);

    let diagnostics = cache_root.join("_diagnostics");
    let mut removed = 0u64;
    let mut removed_bytes = 0u64;
    if let Ok(sessions) = std::fs::read_dir(&diagnostics) {
        for session in sessions.flatten() {
            let path = session.path();
            if !path.is_dir() {
                continue;
            }
            let Some(pid) = diagnostics_session_pid(&path) else {
                continue;
            };
            if process_is_alive(pid) {
                continue;
            }
            let bytes = directory_size(&path);
            match std::fs::remove_dir_all(&path) {
                Ok(()) => {
                    removed = removed.saturating_add(1);
                    removed_bytes = removed_bytes.saturating_add(bytes);
                    log::info!(
                        "tensorrt-cache: abandoned diagnostic session removed pid={} path={} bytes={}",
                        pid,
                        path.display(),
                        bytes
                    );
                }
                Err(error) => log::debug!(
                    "tensorrt-cache: abandoned diagnostic session retained pid={} path={} error={error}",
                    pid,
                    path.display()
                ),
            }
        }
    }
    let mut files = Vec::<(PathBuf, u64, SystemTime)>::new();
    let mut pending = vec![diagnostics.clone()];
    while let Some(dir) = pending.pop() {
        if diagnostics_session_pid(&dir).is_some_and(process_is_alive) {
            continue;
        }
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            let metadata = entry.metadata().ok();
            let age = metadata
                .as_ref()
                .and_then(|meta| meta.modified().ok())
                .and_then(|modified| modified.elapsed().ok());
            let temporary = is_ort_provider_profile(&path)
                || path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        matches!(ext.to_ascii_lowercase().as_str(), "tmp" | "partial")
                    });
            if temporary && age.is_some_and(|age| age >= STALE_AGE) {
                remove_file_accounted(&path, &mut removed, &mut removed_bytes);
                continue;
            }
            if temporary {
                files.push((
                    path,
                    metadata.as_ref().map(|meta| meta.len()).unwrap_or(0),
                    metadata
                        .and_then(|meta| meta.modified().ok())
                        .unwrap_or(SystemTime::UNIX_EPOCH),
                ));
            }
        }
    }

    let mut total_bytes = files.iter().map(|(_, bytes, _)| *bytes).sum::<u64>();
    let mut total_files = files.len();
    files.sort_by_key(|(_, _, modified)| *modified);
    for (path, bytes, _) in files {
        if total_bytes <= HARD_LIMIT_BYTES && total_files <= HARD_LIMIT_FILES {
            break;
        }
        if path_age(&path).is_some_and(|age| age < ACTIVE_GRACE) {
            continue;
        }
        let before_removed = removed;
        remove_file_accounted(&path, &mut removed, &mut removed_bytes);
        if removed != before_removed {
            total_bytes = total_bytes.saturating_sub(bytes);
            total_files = total_files.saturating_sub(1);
        }
    }
    remove_empty_diagnostic_dirs(&diagnostics);
    let _ = std::fs::remove_dir(&diagnostics);
    (removed, removed_bytes)
}

fn cleanup_fallback_tensorrt_diagnostics() -> (u64, u64) {
    let root = std::env::temp_dir()
        .join("cHiDeScaler-Neo")
        .join("TensorRT-profiles");
    let mut removed = 0u64;
    let mut removed_bytes = 0u64;
    let Ok(sessions) = std::fs::read_dir(&root) else {
        return (0, 0);
    };
    for session in sessions.flatten() {
        let path = session.path();
        if !path.is_dir() {
            continue;
        }
        let Some(pid) = diagnostics_session_pid(&path) else {
            continue;
        };
        if process_is_alive(pid) {
            continue;
        }
        let bytes = directory_size(&path);
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                removed = removed.saturating_add(1);
                removed_bytes = removed_bytes.saturating_add(bytes);
            }
            Err(error) => log::debug!(
                "tensorrt-cache: fallback diagnostic session retained pid={} path={} error={error}",
                pid,
                path.display()
            ),
        }
    }
    let _ = std::fs::remove_dir(&root);
    if let Some(parent) = root.parent() {
        let _ = std::fs::remove_dir(parent);
    }
    (removed, removed_bytes)
}

fn cleanup_legacy_tensorrt_provider_profiles(cache_root: &Path) -> (u64, u64) {
    // Older builds placed profiles alongside reusable engines. Remove only
    // matching files; never remove model cache directories here.
    const STALE_AGE: Duration = Duration::from_secs(10 * 60);
    let mut removed = 0u64;
    let mut removed_bytes = 0u64;
    let mut pending = vec![cache_root.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if path.file_name().and_then(|name| name.to_str()) != Some("_diagnostics") {
                    pending.push(path);
                }
                continue;
            }
            if is_ort_provider_profile(&path) {
                let stale = path_age(&path).is_some_and(|age| age >= STALE_AGE);
                let abandoned =
                    ort_provider_profile_pid(&path).is_some_and(|pid| !process_is_alive(pid));
                if stale || abandoned {
                    remove_file_accounted(&path, &mut removed, &mut removed_bytes);
                }
            }
        }
    }
    (removed, removed_bytes)
}

fn is_model_cache_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.len() == 32 && name.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn directory_size(path: &Path) -> u64 {
    let mut total = 0u64;
    let mut pending = vec![path.to_path_buf()];
    while let Some(dir) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let child = entry.path();
            if child.is_dir() {
                pending.push(child);
            } else {
                total = total.saturating_add(entry.metadata().map(|meta| meta.len()).unwrap_or(0));
            }
        }
    }
    total
}

fn cache_access_marker(cache_dir: &Path) -> PathBuf {
    cache_dir.join(".neo-last-used")
}

fn cache_marker_process_alive(cache_dir: &Path) -> bool {
    let Ok(contents) = std::fs::read_to_string(cache_access_marker(cache_dir)) else {
        return false;
    };
    contents
        .lines()
        .find_map(|line| line.strip_prefix("pid="))
        .and_then(|pid| pid.parse::<u32>().ok())
        .is_some_and(process_is_alive)
}

fn touch_tensorrt_cache(cache_dir: &Path) {
    let marker = cache_access_marker(cache_dir);
    let _ = std::fs::write(
        marker,
        format!(
            "schema=1\npid={}\nunix={}\n",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|duration| duration.as_secs())
                .unwrap_or(0)
        ),
    );
}

fn register_tensorrt_cache(cache_dir: &Path) {
    let mut active = TENSORRT_ACTIVE_CACHE_DIRS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    *active.entry(cache_dir.to_path_buf()).or_insert(0) += 1;
    touch_tensorrt_cache(cache_dir);
}

fn unregister_tensorrt_cache(cache_dir: &Path) {
    let mut active = TENSORRT_ACTIVE_CACHE_DIRS
        .get_or_init(Default::default)
        .lock()
        .unwrap();
    if let Some(count) = active.get_mut(cache_dir) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            active.remove(cache_dir);
        }
    }
}

fn active_tensorrt_cache_dirs() -> HashSet<PathBuf> {
    TENSORRT_ACTIVE_CACHE_DIRS
        .get_or_init(Default::default)
        .lock()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

fn tensorrt_cache_limit_bytes() -> u64 {
    let mb = std::env::var("CHIDESCALER_TRT_CACHE_MAX_MB")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| (128..=16384).contains(value))
        .unwrap_or(512);
    mb.saturating_mul(1024 * 1024)
}

fn prune_tensorrt_model_caches(cache_root: &Path, protected: Option<&Path>) -> (u64, u64, u64) {
    const EVICT_MIN_AGE: Duration = Duration::from_secs(24 * 60 * 60);
    const ORPHAN_MIN_AGE: Duration = Duration::from_secs(60 * 60);
    let active = active_tensorrt_cache_dirs();
    let mut entries = Vec::<(PathBuf, u64, SystemTime, bool)>::new();
    let Ok(children) = std::fs::read_dir(cache_root) else {
        return (0, 0, 0);
    };
    for entry in children.flatten() {
        let path = entry.path();
        if !path.is_dir() || !is_model_cache_dir(&path) {
            continue;
        }
        let size = directory_size(&path);
        let access = std::fs::metadata(cache_access_marker(&path))
            .and_then(|meta| meta.modified())
            .or_else(|_| std::fs::metadata(&path).and_then(|meta| meta.modified()))
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let has_engine = OnnxStage::has_tensorrt_engine_files(&path);
        entries.push((path, size, access, has_engine));
    }

    let mut removed_dirs = 0u64;
    let mut removed_bytes = 0u64;
    for (path, size, access, has_engine) in &entries {
        let protected = protected.is_some_and(|current| current == path.as_path())
            || active.contains(path)
            || cache_marker_process_alive(path);
        let age = access.elapsed().unwrap_or_default();
        if !protected && !has_engine && age >= ORPHAN_MIN_AGE {
            if std::fs::remove_dir_all(path).is_ok() {
                removed_dirs += 1;
                removed_bytes = removed_bytes.saturating_add(*size);
            }
        }
    }

    entries.retain(|(path, _, _, _)| path.exists());
    let limit = tensorrt_cache_limit_bytes();
    let mut total = entries.iter().map(|(_, size, _, _)| *size).sum::<u64>();
    entries.sort_by_key(|(_, _, access, _)| *access);
    for (path, size, access, _) in entries {
        if total <= limit {
            break;
        }
        let protected = protected.is_some_and(|current| current == path.as_path())
            || active.contains(&path)
            || cache_marker_process_alive(&path);
        if protected || access.elapsed().unwrap_or_default() < EVICT_MIN_AGE {
            continue;
        }
        match std::fs::remove_dir_all(&path) {
            Ok(()) => {
                total = total.saturating_sub(size);
                removed_dirs += 1;
                removed_bytes = removed_bytes.saturating_add(size);
                log::info!(
                    "tensorrt-cache: LRU cache evicted path={} bytes={} remaining_bytes={} limit_bytes={}",
                    path.display(),
                    size,
                    total,
                    limit
                );
            }
            Err(error) => log::debug!(
                "tensorrt-cache: LRU cache retained path={} error={error}",
                path.display()
            ),
        }
    }
    if total > limit {
        log::info!(
            "tensorrt-cache: budget temporarily exceeded remaining_bytes={} limit_bytes={} reason=active-or-recent-cache-protected",
            total,
            limit
        );
    }
    (removed_dirs, removed_bytes, total)
}

fn maintain_tensorrt_cache_root(cache_root: &Path, protected: Option<&Path>) {
    if let Err(error) = std::fs::create_dir_all(cache_root) {
        log::debug!(
            "tensorrt-cache: maintenance skipped root={} error={error}",
            cache_root.display()
        );
        return;
    }
    let (diagnostic_files, diagnostic_bytes) = cleanup_tensorrt_diagnostics(cache_root);
    let (fallback_diagnostic_dirs, fallback_diagnostic_bytes) =
        cleanup_fallback_tensorrt_diagnostics();
    let (legacy_files, legacy_bytes) = cleanup_legacy_tensorrt_provider_profiles(cache_root);
    let (cache_dirs, cache_bytes, remaining_bytes) =
        prune_tensorrt_model_caches(cache_root, protected);
    if diagnostic_files + fallback_diagnostic_dirs + legacy_files + cache_dirs > 0 {
        log::info!(
            "tensorrt-cache-maintenance: diagnostics_files={} diagnostics_bytes={} fallback_diagnostic_dirs={} fallback_diagnostic_bytes={} legacy_profiles={} legacy_bytes={} cache_dirs={} cache_bytes={} remaining_mb={:.1} limit_mb={:.1}",
            diagnostic_files,
            diagnostic_bytes,
            fallback_diagnostic_dirs,
            fallback_diagnostic_bytes,
            legacy_files,
            legacy_bytes,
            cache_dirs,
            cache_bytes,
            remaining_bytes as f64 / (1024.0 * 1024.0),
            tensorrt_cache_limit_bytes() as f64 / (1024.0 * 1024.0)
        );
    }
}

pub fn maintain_tensorrt_cache(cache_root: &Path) {
    maintain_tensorrt_cache_root(cache_root, None);
}

fn tensorrt_diagnostics_profile_root(cache_root: &Path, profile_key: u64) -> Option<PathBuf> {
    let root = cache_root
        .join("_diagnostics")
        .join(format!("session-{}", std::process::id()))
        .join(format!("profile-{profile_key}"));
    std::fs::create_dir_all(&root).ok()?;
    Some(root)
}

fn tensorrt_provider_verification_ready(cache_dir: &Path) -> bool {
    cache_dir.join(TENSORRT_PROVIDER_VERIFIED_MARKER).exists()
}

fn mark_tensorrt_provider_verification_ready(cache_dir: &Path) {
    let marker = cache_dir.join(TENSORRT_PROVIDER_VERIFIED_MARKER);
    if let Err(error) = std::fs::write(
        &marker,
        "schema=1\nverified=true\nort_profile=TensorRTExecutionProvider\n",
    ) {
        log::warn!(
            "tensorrt-cache: provider verification marker write failed path={} error={error}",
            marker.display()
        );
    }
}

fn prepare_tensorrt_cache_attempt(cache_dir: &Path, cache_was_present: bool) -> Result<()> {
    // Cache maintenance must never be able to invalidate the path between the
    // initial probe and marker creation.
    std::fs::create_dir_all(cache_dir)?;
    let marker = cache_dir.join(TENSORRT_BUILD_MARKER);
    if marker.exists() && !cache_was_present {
        log::warn!(
            "tensorrt-cache: stale incomplete build detected path={}; removing incomplete cache",
            cache_dir.display()
        );
        cleanup_incomplete_tensorrt_cache(cache_dir);
        std::fs::create_dir_all(cache_dir)?;
    }
    if !cache_was_present {
        std::fs::write(marker, format!("pid={}\n", std::process::id()))?;
    }
    Ok(())
}

fn cleanup_incomplete_tensorrt_cache(cache_dir: &Path) {
    match std::fs::remove_dir_all(cache_dir) {
        Ok(()) => log::info!(
            "tensorrt-cache: incomplete cache removed path={}",
            cache_dir.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => log::warn!(
            "tensorrt-cache: incomplete cache cleanup failed path={} error={error}",
            cache_dir.display()
        ),
    }
}

fn mark_tensorrt_cache_ready(cache_dir: &Path) {
    let marker = cache_dir.join(TENSORRT_BUILD_MARKER);
    if marker.exists() {
        match std::fs::remove_file(&marker) {
            Ok(()) => log::info!(
                "tensorrt-cache: build verified path={} marker=cleared",
                cache_dir.display()
            ),
            Err(error) => log::warn!(
                "tensorrt-cache: build marker cleanup failed path={} error={error}",
                marker.display()
            ),
        }
    }
}

fn stage_tensorrt_model(model: &Path, cache_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(cache_dir)?;
    let staged = cache_dir.join("model.onnx");
    let source_len = std::fs::metadata(model)?.len();
    if std::fs::metadata(&staged).is_ok_and(|metadata| metadata.len() == source_len) {
        log::info!(
            "tensorrt-model-stage: source={} staged={} reused=true",
            model.display(),
            staged.display()
        );
        return Ok(staged);
    }
    let temporary = cache_dir.join("model.onnx.tmp");
    std::fs::copy(model, &temporary)?;
    if staged.exists() {
        std::fs::remove_file(&staged)?;
    }
    std::fs::rename(&temporary, &staged)?;
    log::info!(
        "tensorrt-model-stage: source={} staged={} reused=false bytes={source_len}",
        model.display(),
        staged.display()
    );
    Ok(staged)
}

#[cfg(windows)]
fn tensorrt_compatible_path(path: &Path) -> Result<PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows::Win32::Storage::FileSystem::GetShortPathNameW;
    use windows::core::PCWSTR;

    if path.as_os_str().to_string_lossy().is_ascii() {
        return Ok(path.to_path_buf());
    }
    if let Ok(current_dir) = std::env::current_dir()
        && let Ok(relative) = path.strip_prefix(&current_dir)
        && relative.as_os_str().to_string_lossy().is_ascii()
    {
        return Ok(relative.to_path_buf());
    }
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let needed = unsafe { GetShortPathNameW(PCWSTR(wide.as_ptr()), None) };
    anyhow::ensure!(
        needed > 0,
        "TensorRT requires an ASCII-compatible portable path, but Windows short paths are unavailable for {}",
        path.display()
    );
    let mut buffer = vec![0u16; needed as usize + 1];
    let written = unsafe { GetShortPathNameW(PCWSTR(wide.as_ptr()), Some(&mut buffer)) };
    anyhow::ensure!(
        written > 0,
        "Windows could not create a short path for {}",
        path.display()
    );
    buffer.truncate(written as usize);
    let short = PathBuf::from(std::ffi::OsString::from_wide(&buffer));
    anyhow::ensure!(
        short.as_os_str().to_string_lossy().is_ascii(),
        "Windows short path is not ASCII-compatible: {}",
        short.display()
    );
    Ok(short)
}

#[cfg(not(windows))]
fn tensorrt_compatible_path(path: &Path) -> Result<PathBuf> {
    Ok(path.to_path_buf())
}

fn providers_from_ort_profile(path: &Path) -> Result<(bool, bool)> {
    let profile: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let events = profile
        .as_array()
        .ok_or_else(|| anyhow!("ONNX Runtime profile root is not an array"))?;
    let mut tensorrt = false;
    let mut cuda = false;
    for event in events {
        let Some(provider) = event
            .get("args")
            .and_then(|args| args.get("provider"))
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let provider = provider.to_ascii_lowercase();
        tensorrt |= provider.contains("tensorrt");
        cuda |= provider.contains("cuda");
    }
    Ok((tensorrt, cuda))
}

fn parse_channels(dtype: &str) -> Option<usize> {
    // dtype Display looks like "Tensor<f32>(1, 11, dyn, dyn)"
    let inner = dtype.split('(').nth(1)?.split(')').next()?;
    let parts: Vec<&str> = inner.split(',').map(|s| s.trim()).collect();
    if parts.len() == 4 {
        parts[1].parse().ok()
    } else {
        None
    }
}

fn oerr<E: std::fmt::Display>(e: E) -> anyhow::Error {
    anyhow!("{e}")
}

fn out_dims(dims: &[i64]) -> Result<(usize, usize)> {
    anyhow::ensure!(
        dims.len() == 4 && dims[1] == 3,
        "unexpected output shape {dims:?} (want 1x3xHxW)"
    );
    Ok((dims[3] as usize, dims[2] as usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorrt_build_progress_tracks_stacked_models_in_order() {
        finish_tensorrt_build_progress();
        begin_tensorrt_build(Path::new("first.onnx"));
        begin_tensorrt_build(Path::new("second.onnx"));
        mark_tensorrt_model_started("first");
        let (name, _, completed, total, active) = tensorrt_build_progress().unwrap();
        assert_eq!(
            (name.as_str(), completed, total, active),
            ("first", 0, 2, true)
        );
        mark_tensorrt_model_completed("first");
        // Normal pre-filter frames must not move the popup back to the first
        // engine while the second engine is being prepared.
        mark_tensorrt_model_started("first");
        let (name, _, completed, total, active) = tensorrt_build_progress().unwrap();
        assert_eq!(
            (name.as_str(), completed, total, active),
            ("second", 1, 2, false)
        );
        mark_tensorrt_model_started("second");
        let (name, _, completed, total, active) = tensorrt_build_progress().unwrap();
        assert_eq!(
            (name.as_str(), completed, total, active),
            ("second", 1, 2, true)
        );
        mark_tensorrt_model_completed("second");
        assert!(tensorrt_build_progress().is_none());
        finish_tensorrt_build_progress();
        assert!(tensorrt_build_progress().is_none());
    }

    #[test]
    fn cpu_interpolation_entry_points_announce_tensorrt_progress() {
        let source = include_str!("onnx_stage.rs");
        for function in ["pub fn process_interp(", "pub fn process_interp_rgba8("] {
            let body = source
                .split(function)
                .nth(1)
                .expect("interpolation entry point")
                .split("let result =")
                .next()
                .expect("interpolation setup");
            assert!(body.contains("prepare_tensorrt_input_shape(w, h)"));
            assert!(body.contains("mark_tensorrt_model_started(&self.name)"));
        }
    }

    #[test]
    fn tensorrt_build_progress_clears_after_fallback_failure() {
        finish_tensorrt_build_progress();
        let model = Path::new("failed-model.onnx");
        begin_tensorrt_build(model);
        assert!(tensorrt_build_progress().is_some());
        mark_tensorrt_model_failed(model, "test failure");
        assert!(tensorrt_build_progress().is_none());
    }

    #[test]
    fn diagnostic_cleanup_never_removes_model_cache_directories() {
        let root = std::env::temp_dir().join(format!(
            "neo-trt-diagnostic-scope-test-{}-{}",
            std::process::id(),
            TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed)
        ));
        let model_cache = root.join("0123456789abcdef0123456789abcdef");
        std::fs::create_dir_all(&model_cache).unwrap();
        let abandoned_profile = model_cache.join("ort-provider-4294967294-1_test.json");
        std::fs::write(&abandoned_profile, b"[]").unwrap();
        cleanup_tensorrt_diagnostics(&root);
        cleanup_legacy_tensorrt_provider_profiles(&root);
        assert!(model_cache.is_dir());
        assert!(!abandoned_profile.exists());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn diagnostic_cleanup_preserves_live_empty_profile_directory() {
        let root = std::env::temp_dir().join(format!(
            "neo-trt-live-diagnostic-test-{}-{}",
            std::process::id(),
            TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed)
        ));
        let profile = root
            .join("_diagnostics")
            .join(format!("session-{}", std::process::id()))
            .join("profile-1");
        std::fs::create_dir_all(&profile).unwrap();
        cleanup_tensorrt_diagnostics(&root);
        assert!(profile.is_dir());
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn tensorrt_content_cache_key_ignores_model_path() {
        let root = std::env::temp_dir().join(format!(
            "neo-trt-content-key-test-{}-{}",
            std::process::id(),
            TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let first_model = root.join("first.onnx");
        let second_model = root.join("second.onnx");
        std::fs::write(&first_model, b"same-model-content").unwrap();
        std::fs::write(&second_model, b"same-model-content").unwrap();
        let first = tensorrt_model_cache_dir(&root, &first_model, b"builder-3");
        let second = tensorrt_model_cache_dir(&root, &second_model, b"builder-3");
        assert_eq!(first, second);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn f16_output_lut_matches_previous_conversion_for_every_bit_pattern() {
        let lut = f16_unorm_to_u8_lut();
        for bits in 0..=u16::MAX {
            let expected = unit_to_u8(f16::from_bits(bits).to_f32());
            assert_eq!(lut[bits as usize], expected, "f16 bits 0x{bits:04x}");
        }
    }

    #[test]
    fn u8_input_lut_matches_previous_conversion() {
        let lut = u8_to_f16_lut();
        for value in 0..=u8::MAX {
            assert_eq!(
                lut[value as usize],
                f16::from_f32(value as f32 * (1.0 / 255.0))
            );
        }
    }

    #[test]
    fn provider_profile_file_detection_accepts_ort_timestamp_suffix() {
        assert!(is_ort_provider_profile(Path::new(
            "ort-provider-18052-3_2026-08-03_21-05-07.json"
        )));
        assert!(!is_ort_provider_profile(Path::new("model.engine")));
        assert!(!is_ort_provider_profile(Path::new("shape.json")));
    }

    #[test]
    fn tensorrt_model_cache_key_is_stable_for_repeated_filter_toggles() {
        let root = std::env::temp_dir().join(format!(
            "neo-trt-cache-key-test-{}-{}",
            std::process::id(),
            TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&root).unwrap();
        let model = root.join("model.onnx");
        std::fs::write(&model, b"stable-model-bytes").unwrap();
        let first = tensorrt_model_cache_dir(&root, &model, b"builder-3");
        let second = tensorrt_model_cache_dir(&root, &model, b"builder-3");
        let changed_builder = tensorrt_model_cache_dir(&root, &model, b"builder-4");
        assert_eq!(first, second);
        assert_ne!(first, changed_builder);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn provider_profile_distinguishes_tensorrt_from_cuda_fallback() {
        let root = std::env::temp_dir().join(format!(
            "neo-ort-profile-test-{}-{}",
            std::process::id(),
            TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(
            &root,
            br#"[
                {"cat":"Node","args":{"provider":"TensorrtExecutionProvider"}},
                {"cat":"Node","args":{"provider":"CUDAExecutionProvider"}}
            ]"#,
        )
        .unwrap();
        assert_eq!(providers_from_ort_profile(&root).unwrap(), (true, true));
        std::fs::write(
            &root,
            br#"[{"cat":"Node","args":{"provider":"CUDAExecutionProvider"}}]"#,
        )
        .unwrap();
        assert_eq!(providers_from_ort_profile(&root).unwrap(), (false, true));
        let _ = std::fs::remove_file(root);
    }

    #[test]
    fn dml_post_interp_auto_bridge_is_safe_copy_only_on_non_conservative_driver() {
        assert_eq!(
            resolve_dml_gl_interop_mode("", true, false),
            Some(DmlGlInteropMode::ProviderOwnedCopy)
        );
        assert_eq!(resolve_dml_gl_interop_mode("", true, true), None);
        assert_eq!(resolve_dml_gl_interop_mode("", false, false), None);
    }

    #[test]
    fn dml_interop_environment_override_has_priority_over_auto_policy() {
        assert_eq!(
            resolve_dml_gl_interop_mode("direct", true, true),
            Some(DmlGlInteropMode::Direct)
        );
        assert_eq!(
            resolve_dml_gl_interop_mode("copy", false, true),
            Some(DmlGlInteropMode::ProviderOwnedCopy)
        );
        assert_eq!(resolve_dml_gl_interop_mode("0", true, false), None);
        assert_eq!(resolve_dml_gl_interop_mode("off", true, false), None);
    }

    #[test]
    fn interpolation_bridge_capacity_grows_but_never_shrinks_for_fractional_cadence() {
        assert!(OnnxStage::interp_bridge_needs_growth(3, 5));
        assert!(OnnxStage::interp_bridge_needs_growth(5, 7));
        assert!(!OnnxStage::interp_bridge_needs_growth(5, 3));
        assert!(!OnnxStage::interp_bridge_needs_growth(7, 5));
        assert!(!OnnxStage::interp_bridge_needs_growth(7, 7));
    }

    #[test]
    fn temporal_gpu_history_matches_cpu_front_padding_and_ring_order() {
        assert_eq!(temporal_history_slot_order(5, 1, 1), vec![0, 0, 0, 0, 0]);
        assert_eq!(temporal_history_slot_order(5, 2, 2), vec![0, 0, 0, 0, 1]);
        assert_eq!(temporal_history_slot_order(5, 3, 3), vec![0, 0, 0, 1, 2]);
        assert_eq!(temporal_history_slot_order(5, 5, 0), vec![0, 1, 2, 3, 4]);
        assert_eq!(temporal_history_slot_order(5, 5, 2), vec![2, 3, 4, 0, 1]);
    }
}
