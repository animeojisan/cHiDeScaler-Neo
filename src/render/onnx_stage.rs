//! ONNX super-resolution stage via onnxruntime + DirectML (any DX12 GPU).
//!
//! onnxruntime.dll is loaded dynamically at runtime (`init_onnx`), so the app
//! itself stays tiny and runs fully without it — ONNX filters simply report
//! unavailable. Any NCHW image model (Compact/ESRGAN/CUGAN...) works drop-in;
//! fp16 vs fp32 is detected from the model input.

#[path = "rife_dml_split.rs"]
mod rife_dml_split;

use anyhow::{Context, Result, anyhow};
use half::f16;
use ort::{
    AsPointer,
    ep::ArbitrarilyConfigurableExecutionProvider,
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
    neoarc_backend::NeoArcSession,
    neoamd_backend::{
        NEOAMD_CAP_D3D12_SHARED_BUFFER, NEOAMD_CAP_DYNAMIC_RESOLUTION, NEOAMD_CAP_FP16_NATIVE,
        NEOAMD_CAP_GFX12_WMMA, NEOAMD_CAP_STRUCTURAL_ROUTING, NEOAMD_RUN_ROUTE_DIRECT,
        NEOAMD_RUN_ROUTE_GRAPH_COMPUTE, NEOAMD_RUN_ROUTE_GRAPH_FULL, NEOAMD_RUN_ROUTE_RTMOSR_FP16,
        NEOAMD_RUN_ROUTE_RTMOSR_U8_FUSED, NEOAMD_SESSION_IMAGE, NEOAMD_SESSION_INTERPOLATION,
        NEOAMD_SESSION_TEMPORAL, NEOAMD_VENDOR_AMD, NeoAmdBackend, NeoAmdSession,
    },
    onnx_accel::{
        Decision as NeoDecision, NeoAccelState, analyze_local_image_model, compare_output_patch,
        crop_rgba8, extract_output_patch, inspect_local_image_model, inspect_rtmosr_image_model,
        inspect_spanf3_image_model, neoaccel_enabled, write_output_patch,
    },
};

static ORT_READY: OnceLock<std::result::Result<(), String>> = OnceLock::new();
static DIRECT_OUTPUT_KEY: AtomicU64 = AtomicU64::new(1);
static TENSORRT_PROFILE_KEY: AtomicU64 = AtomicU64::new(1);
static TENSORRT_ACTIVE_CACHE_DIRS: OnceLock<std::sync::Mutex<HashMap<PathBuf, usize>>> =
    OnceLock::new();
static TENSORRT_CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ONNX_CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ACTIVE_ONNX_RUN_OPTIONS: OnceLock<Mutex<Vec<Weak<CancelableRunOptions>>>> = OnceLock::new();
static INTERP_GPU_INPUT_FRAMES: AtomicU64 = AtomicU64::new(0);
static NEOAMD_STREAM_PREP_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static NEOAMD_STREAM_PHASE_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static NEOAMD_PROFILE_LOG_COUNT: AtomicU64 = AtomicU64::new(0);
static INTERP_GPU_OUTPUT_FRAMES: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_FRAME_READBACKS: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_FRAME_UPLOADS: AtomicU64 = AtomicU64::new(0);
static INTERP_CPU_PACK_FRAMES: AtomicU64 = AtomicU64::new(0);

#[inline]
fn should_log_neoamd_profile() -> bool {
    let n = NEOAMD_PROFILE_LOG_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
    n <= 32 || n % 120 == 0
}

fn neoamd_run_route_label(flags: Option<u32>) -> String {
    let Some(flags) = flags else {
        return "unreported".to_string();
    };
    let graph = if flags & NEOAMD_RUN_ROUTE_GRAPH_FULL != 0 {
        "graph-full"
    } else if flags & NEOAMD_RUN_ROUTE_GRAPH_COMPUTE != 0 {
        "graph-compute"
    } else if flags & NEOAMD_RUN_ROUTE_DIRECT != 0 {
        "direct"
    } else {
        "unknown"
    };
    let ingress = if flags & NEOAMD_RUN_ROUTE_RTMOSR_FP16 != 0 {
        "+rtmosr-staged-fp16"
    } else if flags & NEOAMD_RUN_ROUTE_RTMOSR_U8_FUSED != 0 {
        "+rtmosr-u8-fused"
    } else {
        ""
    };
    format!("{graph}{ingress}")
}

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
    // v681: time each model from the moment TensorRT engine preparation is
    // registered. `model_started` is intentionally kept separate because it
    // marks the first real inference, which can occur only after a long lazy
    // provider/engine build has already completed.
    model_build_started: std::collections::HashMap<String, Instant>,
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
        progress
            .model_build_started
            .entry(name.clone())
            .or_insert_with(Instant::now);
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
        progress.model_build_started.remove(&name);
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
                .model_build_started
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
    let model_build_started = progress.model_build_started.get(&name).copied();
    Some((
        name,
        model_build_started
            .map(|started| started.elapsed())
            .or_else(|| model_started.map(|started| started.elapsed()))
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

static DIRECTML_RIFE_BOOST: OnceLock<AtomicBool> = OnceLock::new();

fn directml_rife_boost_setting() -> &'static AtomicBool {
    DIRECTML_RIFE_BOOST.get_or_init(|| {
        AtomicBool::new(std::env::var("CHIDESCALER_DML_RIFE_BOOST").is_ok_and(|value| value == "1"))
    })
}

pub fn set_directml_rife_boost(enabled: bool) {
    directml_rife_boost_setting().store(enabled, Ordering::Relaxed);
}

/// v947: true when the most recently prepared DirectML pair runs the RIFE
/// boost split (one shared motion pass per pair). The engine's overload
/// admission must then treat the pair cost as mostly fixed, not per midpoint.
static DML_RIFE_SPLIT_PAIR: AtomicBool = AtomicBool::new(false);

pub fn last_prepared_pair_uses_rife_split() -> bool {
    DML_RIFE_SPLIT_PAIR.load(Ordering::Relaxed)
}

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
    /// v971: original FP32-IO RIFE model when DirectML runs an FP16-IO bridge
    /// of it (`source_path` then points at the cached bridge). Boost graphs are
    /// split from this original and bridged, keeping the FP32 motion state.
    dml_half_bridge_source: Option<PathBuf>,
    fallback_dml_adapter: Option<i32>,
    session: Session,
    /// Shape symbols of a dynamic DirectML RIFE input (batch, height, width).
    dml_rife_symbols: Option<[String; 3]>,
    dml_rife_shape: Option<(i32, i32)>,
    dml_host_interp_binding: Option<((usize, usize), IoBinding)>,
    dml_rife_cached_session: Option<((i32, i32), Session)>,
    dml_rife_specialization_disabled: bool,
    dml_rife_native_rejected: std::collections::HashSet<(i32, i32)>,
    dml_image_symbols: Option<[String; 3]>,
    dml_wgl_converter: Option<super::dml_gl_bridge::Converter>,
    dml_wgl_disabled: bool,
    dml_drba_sequential_logged: bool,
    dml_wgl_last_geometry: Option<((i32,i32),(usize,usize))>,
    dml_wgl_input_converter: Option<super::dml_gl_bridge::Converter>,
    dml_wgl_input_failed: Option<(i32,i32)>,
    dml_image_device_luid: Option<Option<u64>>,
    dml_image_budget_retry: Option<Instant>,
    dml_image_dynamic_session: Option<Session>,
    /// One previous small fixed geometry for quick 128↔256 style resizes.
    dml_image_cached_fixed_session: Option<((i32, i32), Session)>,
    dml_image_shape: Option<(i32, i32)>,
    dml_image_specialization_disabled: bool,
    run_options: Arc<CancelableRunOptions>,
    tensorrt_diagnostic_dir: Option<TensorRtDiagnosticDir>,
    in_name: String,
    out_name: String,
    fp16: bool,
    scratch_f32: Vec<f32>,
    scratch_f16: Vec<f16>,
    neoarc_pair_input: Vec<f16>,
    neoarc_pair_reuse: bool,
    neoarc_pair_packed: bool,
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
    interp_gpu_diag: GpuInterpDiagnostics,
    last_upscale_profile: Option<UpscaleProfile>,
    image_diag_last_log: Option<Instant>,
    image_diag_samples: std::collections::VecDeque<[f64;4]>,
    image_diag_shape: Option<(i32,i32)>,
    last_upscale_input_size: Option<(i32, i32)>,
    direct_output: Option<DmlDirectOutput>,
    direct_input: Option<DmlDirectInput>,
    dml_temporal_bridge: Option<DmlTemporalGpuBridge>,
    direct_output_disabled: bool,
    neoamd_shared_disabled: bool,
    neoamd_shared_key: Option<u64>,
    /// v978: extra NeoAMD interpolation output slots (Backend Pack v135+),
    /// tied to the slot-0 key in `neoamd_interp_ring_base`.
    neoamd_interp_ring: Vec<crate::render::neoamd_backend::NeoAmdSharedOutput>,
    neoamd_interp_ring_base: Option<u64>,
    neoamd_interp_ring_unavailable: bool,
    neoamd_shared_input_key: Option<u64>,
    neoamd_shared_input_size: Option<(i32, i32)>,
    tensorrt_device_id: Option<i32>,
    neoamd_session: Option<NeoAmdSession>,
    neoarc_session: Option<NeoArcSession>,
    neoarc_interp_gl: Option<NeoArcInterpGl>,
    tensorrt_gpu_bridge: Option<TensorRtGpuBridge>,
    tensorrt_temporal_bridge: Option<TensorRtTemporalGpuBridge>,
    tensorrt_interp_bridge: Option<TensorRtInterpGpuBridge>,
    dml_interp_bridge: Option<DmlInterpGpuBridge>,
    /// v933: FraZerion pair batch (models exported with `frazerion.pair_batch=1`).
    dml_pair_batch: Option<DmlPairBatch>,
    dml_pair_batch_capable: bool,
    dml_pair_batch_rows: usize,
    dml_pair_batch_failed: bool,
    dml_pair_batch_logged: bool,
    dml_pair_batch_skip_logged: bool,
    dml_split: Option<DmlSplit>,
    dml_split_paths: Option<(PathBuf, PathBuf)>,
    // v937: trunk/head graphs embedded in the model's own metadata (one .onnx file, no sidecars)
    dml_split_embedded: Option<std::sync::Arc<(Vec<u8>, Vec<u8>)>>,
    dml_split_state_channels: usize,
    dml_rife_boost: bool,
    fused_host_boost: Vec<((i32, i32, usize), Session)>,
    fused_host_failed: bool,
    dml_host_boost: Option<((i32, i32), Session, Session)>,
    dml_split_failed: bool,
    dml_split_logged: bool,
    dml_wgl_interp_bridge: Option<DmlWglInterpBridge>,
    dml_wgl_interp_failed: bool,
    /// Cross-GPU fallback: CPU-packed RIFE input still enters DirectML normally,
    /// but completed outputs stay in app-owned shareable D3D12 buffers so the
    /// same selected GPU can hand them directly to Vulkan without a CPU readback.
    dml_cpu_interp_shared_outputs: [Vec<DmlDirectOutput>; 2],
    dml_cpu_interp_shared_disabled: bool,
    /// Consecutive native NeoAMD DRBA intervals that exceed the practical
    /// DirectML-class throughput budget. Used only as a safety valve while the
    /// native DRBA kernels are tuned; a transient graph-capture/cold-start spike
    /// must not trigger fallback by itself.
    neoamd_drba_slow_streak: u8,
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
    neoarc_temporal_packed: bool,
    dml_temporal_cursor: Option<usize>,
    dml_temporal_mirror: bool,
    neoarc_gl_history_active: bool,
    temporal_history: VecDeque<Vec<u8>>,
    /// Pixel stride stored in temporal_history (3=RGB, 4=RGBA). NeoAMD accepts
    /// both, so production no longer repacks every live RGBA frame to a newly
    /// allocated RGB Vec before crossing the Backend Pack ABI.
    temporal_history_stride: usize,
    temporal_size: Option<(i32, i32)>,
    /// Keep a NeoAMD TemporalFix runtime failure local to the current W/H
    /// instead of mutating the cached stage's provider to DirectML. This covers
    /// allocation/runtime failures on the v052 dynamic path and lets a later
    /// geometry retry NeoAMD without contaminating OFF -> ON cache reuse.
    neoamd_temporal_rejected_size: Option<(i32, i32)>,
    pub provider_desc: String,
    pub provider: OnnxProvider,
    pub fallback_reason: Option<String>,
    tensorrt_cache_dir: Option<PathBuf>,
    tensorrt_cache_root: Option<PathBuf>,
    tensorrt_profile_shape: Option<(usize, i32, i32)>,
    tensorrt_engine_verified: bool,
    tensorrt_profiling_active: bool,
    /// v691: TensorRT actually executes this model path. For normal sessions it
    /// equals `source_path`; the NeoAccel Q/DQ experiment points at a bundled
    /// sidecar while keeping `source_path` as the user's original FP16 model.
    tensorrt_model_path: Option<PathBuf>,
    /// True only for the explicit-Q/DQ INT8 sidecar experiment. The normal
    /// TensorRT FP16 session never enables this flag.
    tensorrt_explicit_qdq: bool,
    /// Frame-interpolation model (RIFE-style): two image inputs, or one
    /// 6-channel input (prev+cur concatenated). Optional scalar "timestep".
    pub interp: InterpKind,
}

impl Drop for OnnxStage {
    fn drop(&mut self) {
        // Retire OpenCL/OpenVINO before Rust drops the earlier DirectML Session field.
        // The Intel driver can otherwise wait while tearing down overlapping APIs.
        self.retire_neoarc_gl();
        self.neoarc_interp_gl.take();
        self.neoarc_session.take();
        if let Some(cache_dir) = &self.tensorrt_cache_dir {
            unregister_tensorrt_cache(cache_dir);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OnnxProvider {
    DirectML,
    MigraphX,
    TensorRT,
    Cuda,
    NeoAMD,
    NeoRTX,
}

impl OnnxProvider {
    pub fn is_native_pack(self) -> bool {
        matches!(self, Self::NeoAMD | Self::NeoRTX)
    }
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

struct NeoArcInterpGl {
    io: super::neoarc_backend::GlInterpIo,
    boost_attempted: bool,
    capacity:usize,
    size: (i32,i32),
    padded: (i32,i32),
    context: (isize,isize),
    frames: Vec<u32>,
    phases: Vec<f32>,
    outputs: Vec<(String,GpuTex)>,
    /// v985: GL pixel buffers (RGBA FP16) the endpoints are copied into, so
    /// OpenCL reads buffers instead of GL images (Intel stalls 5-9 ms in the
    /// first kernel enqueue that reads acquired GL images). Empty = image path.
    input_buffers: Vec<glow::Buffer>,
    buffer_input_failed: bool,
    /// v992: compute pass that writes the endpoints into `input_buffers`.
    endpoint_program: Option<glow::Program>,
    endpoint_compute_failed: bool,
}

/// Stable description of one GPU-resident interpolation output slot.
/// The render thread can import a completed slot without locking the ONNX
/// stage while the worker continues computing later, distinct timesteps.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedInterpGpuOutput {
    pub(crate) key: u64,
    pub(crate) texture: Option<GpuTex>,
    pub(crate) size: (i32, i32),
    pub(crate) padded: (i32, i32),
    pub(crate) fp16: bool,
    /// v807: this output is a DirectML DRBA synthetic slot and may use the
    /// local-GL snapshot lane if this exact imported resource entered the
    /// bound-storage fallback. Other providers/interpolators remain unchanged.
    pub(crate) directml_drba_snapshot_candidate: bool,
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

/// v816: one completed NeoAMD RIFE midpoint left in the Backend Pack's shared
/// RGBA8 D3D12 buffer.  The handle is a fresh duplicate owned by this wrapper
/// until the render thread transfers it to GlContext.
pub(crate) struct PreparedNeoAmdSharedOutput {
    pub(crate) key: u64,
    pub(crate) handle: HANDLE,
    pub(crate) byte_len: usize,
    pub(crate) heap_byte_len: u64,
    pub(crate) luid: [u8; 8],
    pub(crate) size: (i32, i32),
}

unsafe impl Send for PreparedNeoAmdSharedOutput {}

impl PreparedNeoAmdSharedOutput {
    pub(crate) fn take_handle(&mut self) -> HANDLE {
        std::mem::take(&mut self.handle)
    }
}

impl Drop for PreparedNeoAmdSharedOutput {
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct NeoAmdPeerTensor {
    pub(crate) key: u64,
    pub(crate) width: i32,
    pub(crate) height: i32,
    pub(crate) byte_len: usize,
    pub(crate) allocation_byte_len: u64,
    pub(crate) adapter_luid: u64,
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
    pub(crate) fn metrics_provider_label(&self) -> &'static str {
        if self.neoarc_session.is_some() { return "NeoARC"; }
        match self.provider {
            OnnxProvider::TensorRT => "TensorRT",
            OnnxProvider::DirectML => "DirectML",
            OnnxProvider::MigraphX => "MIGraphX",
            OnnxProvider::Cuda => "CUDA",
            OnnxProvider::NeoAMD => "NeoAMD",
            OnnxProvider::NeoRTX => "NeoRTX",
        }
    }
    /// Diagnostic timing without the extra Vulkan and CPU reference passes.
    pub fn probe_shared_output_timing(
        &mut self,
        width: i32,
        height: i32,
    ) -> Result<Option<UpscaleProfile>> {
        let pixels = usize::try_from(width)?
            .checked_mul(usize::try_from(height)?)
            .ok_or_else(|| anyhow!("probe dimensions overflow"))?;
        anyhow::ensure!(pixels <= 1024 * 1024, "probe dimensions too large");
        let mut rgba = vec![0u8; pixels * 4];
        for (i, pixel) in rgba.chunks_exact_mut(4).enumerate() {
            pixel[0] = (i % 251) as u8;
            pixel[1] = ((i / 3) % 241) as u8;
            pixel[2] = ((i / 7) % 239) as u8;
            pixel[3] = 255;
        }
        let Some(_) = self.process_rgba8_dml_shared_output(width, height, &rgba)? else {
            return Ok(None);
        };
        Ok(self.last_upscale_profile)
    }

    /// Run a CPU RGBA8 input through DirectML and leave the final RGBA8 image
    /// in a shareable GPU buffer for a Vulkan presenter. No output readback or
    /// GL upload occurs here. Input packing is still CPU-side.
    pub fn process_rgba8_to_shared_rgba(
        &mut self,
        width: i32,
        height: i32,
        rgba: &[u8],
        shader: &crate::render::mpv::UserShader,
        output_ref_size: Option<(i32, i32)>,
    ) -> Result<Option<crate::render::vulkan_multipass::MultiPassSharedRgbaResult>> {
        let Some((shared, _)) = self.process_rgba8_dml_shared_output(width, height, rgba)? else {
            return Ok(None);
        };
        let output_ref_size = output_ref_size.unwrap_or(shared.size);
        crate::render::vulkan_multipass::process_d3d12_nchw_to_shared_rgba(
            u64::from_le_bytes(shared.luid),
            shader,
            shared.size.0 as u32,
            shared.size.1 as u32,
            output_ref_size.0.max(1) as u32,
            output_ref_size.1.max(1) as u32,
            shared.key,
            shared.handle.0 as isize,
            shared.byte_len,
            shared.padded.0 as u32,
            shared.padded.1 as u32,
            shared.fp16,
        )
    }

    /// Diagnostic readback of the same DirectML tensor is returned solely
    /// for pixel comparison; the shared output itself stays on GPU.
    pub fn probe_shared_rgba_output(
        &mut self,
        width: i32,
        height: i32,
        frame_seed: u8,
    ) -> Result<
        Option<(
            crate::render::vulkan_multipass::MultiPassSharedRgbaResult,
            Vec<u8>,
        )>,
    > {
        let pixels = usize::try_from(width)?
            .checked_mul(usize::try_from(height)?)
            .ok_or_else(|| anyhow!("probe dimensions overflow"))?;
        anyhow::ensure!(pixels <= 1024 * 1024, "probe dimensions too large");
        let mut rgba = vec![0u8; pixels * 4];
        for (i, pixel) in rgba.chunks_exact_mut(4).enumerate() {
            pixel[0] = ((i + frame_seed as usize * 37) % 251) as u8;
            pixel[1] = ((i / 3 + frame_seed as usize * 29) % 241) as u8;
            pixel[2] = ((i / 7 + frame_seed as usize * 17) % 239) as u8;
            pixel[3] = 255;
        }
        let shader = crate::render::mpv::UserShader::parse(
            "NeoArcDiagnosticIdentity.glsl",
            "//!HOOK RGB\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n",
        );
        let Some(output) = self.process_rgba8_to_shared_rgba(width, height, &rgba, &shader, None)? else {
            return Ok(None);
        };
        let state = self
            .direct_output
            .as_ref()
            .ok_or_else(|| anyhow!("DirectML shared output state disappeared"))?;
        let shared = PreparedDmlSharedOutput {
            key: state.key,
            handle: state.shared_handle()?,
            byte_len: state.byte_len,
            heap_byte_len: state.heap_byte_len,
            luid: state.luid,
            size: (i32::try_from(state.dims[3])?, i32::try_from(state.dims[2])?),
            padded: (i32::try_from(state.dims[3])?, i32::try_from(state.dims[2])?),
            fp16: true,
        };
        let reference = crate::render::vulkan_multipass::process_d3d12_nchw(
            u64::from_le_bytes(shared.luid),
            &shader,
            shared.size.0 as u32,
            shared.size.1 as u32,
            shared.size.0 as u32,
            shared.size.1 as u32,
            shared.key,
            shared.handle.0 as isize,
            shared.byte_len,
            shared.padded.0 as u32,
            shared.padded.1 as u32,
            shared.fp16,
        )?
        .ok_or_else(|| anyhow!("Vulkan diagnostic reference unavailable"))?;
        Ok(Some((output, reference.output_rgba8)))
    }

    /// Diagnostic inference through the production DirectML shared-output path.
    /// The returned NT handle remains owned here and is closed before return.
    pub fn probe_shared_output(
        &mut self,
        width: i32,
        height: i32,
    ) -> Result<Option<(u64, usize, f64, u64, u8)>> {
        let pixels = usize::try_from(width)?
            .checked_mul(usize::try_from(height)?)
            .ok_or_else(|| anyhow!("probe dimensions overflow"))?;
        anyhow::ensure!(pixels <= 1024 * 1024, "probe dimensions too large");
        let mut rgba = vec![0u8; pixels * 4];
        for (i, pixel) in rgba.chunks_exact_mut(4).enumerate() {
            pixel[0] = (i % 251) as u8;
            pixel[1] = ((i / 3) % 241) as u8;
            pixel[2] = ((i / 7) % 239) as u8;
            pixel[3] = 255;
        }
        let Some((shared, ms)) = self.process_rgba8_dml_shared_output(width, height, &rgba)? else {
            return Ok(None);
        };
        crate::render::vulkan_gpu::probe_existing_d3d12_buffer_import(
            u64::from_le_bytes(shared.luid),
            shared.handle.0 as isize,
            shared.byte_len,
        )?;
        let shader = crate::render::mpv::UserShader::parse(
            "NeoArcDiagnosticIdentity.glsl",
            "//!HOOK RGB\n//!BIND HOOKED\nvec4 hook() { return HOOKED_tex(HOOKED_pos); }\n",
        );
        let image = crate::render::vulkan_multipass::process_d3d12_nchw(
            u64::from_le_bytes(shared.luid),
            &shader,
            shared.size.0 as u32,
            shared.size.1 as u32,
            shared.size.0 as u32,
            shared.size.1 as u32,
            shared.key,
            shared.handle.0 as isize,
            shared.byte_len,
            shared.padded.0 as u32,
            shared.padded.1 as u32,
            shared.fp16,
        )?
        .ok_or_else(|| anyhow!("Vulkan shared output conversion unavailable"))?;
        anyhow::ensure!(
            image.output_rgba8.len() == shared.size.0 as usize * shared.size.1 as usize * 4,
            "Vulkan output size mismatch"
        );
        let input_shape = Shape::new([1i64, 3, height as i64, width as i64]);
        let input = TensorRef::from_array_view((input_shape, &*self.scratch_f16)).map_err(oerr)?;
        let outputs = self
            .session
            .run_with_options(
                ort::inputs![self.in_name.as_str() => input],
                self.run_options.armed()?,
            )
            .map_err(oerr)?;
        let reference = outputs
            .get(self.out_name.as_str())
            .ok_or_else(|| anyhow!("DirectML reference output missing"))?;
        let (_, values) = reference.try_extract_tensor::<f16>().map_err(oerr)?;
        let plane = shared.size.0 as usize * shared.size.1 as usize;
        anyhow::ensure!(
            values.len() >= plane * 3,
            "DirectML reference shape mismatch"
        );
        let lut = f16_unorm_to_u8_lut();
        let mut maximum_error = 0u8;
        for pixel in 0..plane {
            for channel in 0..3 {
                let expected = lut[values[channel * plane + pixel].to_bits() as usize];
                let actual = image.output_rgba8[pixel * 4 + channel];
                maximum_error = maximum_error.max(actual.abs_diff(expected));
            }
        }
        anyhow::ensure!(
            maximum_error <= 2,
            "Vulkan shared image differs from DirectML reference: max u8 error {maximum_error}"
        );
        let checksum = image
            .output_rgba8
            .iter()
            .fold(0u64, |sum, &v| sum.wrapping_add(v as u64));
        Ok(Some((
            u64::from_le_bytes(shared.luid),
            shared.byte_len,
            ms,
            checksum,
            maximum_error,
        )))
    }

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
        app_dir: &Path,
    ) -> Result<Self> {
        match preference {
            OnnxBackendPreference::DirectML => {
                let neo_image =
                    neoaccel_enabled() && analyze_local_image_model(path).ok().flatten().is_some();
                if neo_image {
                    let amd_target = {
                        let adapters = crate::platform::gpu::enumerate_adapters_quiet();
                        match dml_adapter_id {
                            Some(device_id) => adapters
                                .iter()
                                .find(|adapter| adapter.device_id == device_id)
                                .is_some_and(|adapter| adapter.vendor_id == 0x1002),
                            None => adapters.len() == 1 && adapters[0].vendor_id == 0x1002,
                        }
                    };
                    if amd_target && crate::render::winml_migraphx::available_runtime() {
                        match Self::load_migraphx(path, dml_adapter_id) {
                            Ok(stage) => return Ok(stage),
                            Err(error) => {
                                log::warn!(
                                    "NeoAccel AMD provider probe rejected: model={} provider=MIGraphX reason={error:#}; action=DirectML-FP16",
                                    path.display()
                                );
                            }
                        }
                    } else if amd_target {
                        log::info!(
                            "NeoAccel provider probe: model={} requested=DirectML low_precision=not-active reason=Windows-ML-MIGraphX-runtime-not-prepared fallback=DirectML-FP16",
                            path.display()
                        );
                    } else {
                        log::info!(
                            "NeoAccel provider probe: model={} requested=DirectML provider=DirectML reason=selected-GPU-is-not-unambiguous-AMD",
                            path.display()
                        );
                    }
                } else if neoaccel_enabled() {
                    log::info!(
                        "NeoAccel provider probe skipped: model={} requested=DirectML reason=not-conservative-image-cnn",
                        path.display()
                    );
                }
                Self::load_directml_arc_bridged(path, dml_adapter_id, app_dir)
            }
            OnnxBackendPreference::NeoAMD => Self::load_neoamd(path, dml_adapter_id, app_dir),
            OnnxBackendPreference::NeoRTX => Self::load_neortx(path, dml_adapter_id, app_dir),
            OnnxBackendPreference::NeoArc => Self::load_neoarc(path, dml_adapter_id, app_dir),
            OnnxBackendPreference::TensorRT => {
                let device_id =
                    trt_device_id.ok_or_else(|| anyhow!("TensorRT CUDA device is unavailable"))?;
                let qdq_sidecar = if neoaccel_enabled()
                    && analyze_local_image_model(path).ok().flatten().is_some()
                {
                    neoaccel_tensorrt_qdq_sidecar(path)
                } else {
                    None
                };
                if neoaccel_enabled() && qdq_sidecar.is_none() {
                    log::info!(
                        "NeoAccel explicit-QDQ skipped: model={} reason=no-validated-sidecar-for-this-model fallback=TensorRT-FP16",
                        path.display()
                    );
                }
                let engine_model = qdq_sidecar.as_deref().unwrap_or(path);
                let explicit_qdq = qdq_sidecar.is_some();
                match Self::load_tensorrt(
                    path,
                    engine_model,
                    device_id,
                    dml_adapter_id,
                    cache_root,
                    explicit_qdq,
                ) {
                    Ok(stage) => Ok(stage),
                    Err(qdq_error) if explicit_qdq => {
                        log::warn!(
                            "NeoAccel explicit-QDQ session rejected: model={} reason={qdq_error:#}; action=retry-TensorRT-FP16",
                            path.display()
                        );
                        match Self::load_tensorrt(
                            path,
                            path,
                            device_id,
                            dml_adapter_id,
                            cache_root,
                            false,
                        ) {
                            Ok(mut stage) => {
                                stage.fallback_reason = Some(format!(
                                    "NeoAccel explicit-QDQ INT8 rejected; TensorRT FP16 restored: {qdq_error:#}"
                                ));
                                log::info!(
                                    "NeoAccel explicit-QDQ fallback complete: model={} provider=TensorRT-FP16 phase=session-create",
                                    path.display()
                                );
                                Ok(stage)
                            }
                            Err(fp16_error) => {
                                mark_tensorrt_model_failed(path, &format!("{fp16_error:#}"));
                                log::warn!(
                                    "onnx-backend-fallback: model={} TensorRT FP16 retry failed: {fp16_error:#}; using DirectML",
                                    path.display()
                                );
                                let mut stage = Self::load_directml(path, dml_adapter_id)?;
                                stage.fallback_reason = Some(format!("{fp16_error:#}"));
                                stage.provider_desc.push_str(" fallback");
                                Ok(stage)
                            }
                        }
                    }
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
        Self::load_directml_level(path, adapter_id, None)
    }

    /// `level` overrides the graph optimization (v993: NeoArc loads its
    /// signature-only DirectML session unoptimized; 360-480 ms -> less).
    fn load_directml_level(path: &Path, adapter_id: Option<i32>, level: Option<GraphOptimizationLevel>) -> Result<Self> {
        init_onnx().map_err(|e| anyhow!(e))?;
        // Keep Auto on ORT's high-performance selection. Pinning every model to
        // Arc also changes interpolation sessions and has regressed playback.
        let graph_optimization = level.unwrap_or_else(|| dml_graph_optimization(path));
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

    /// v971: DirectML graph tuning for interpolation models on Intel Arc.
    /// 1. ConvTranspose(4x4,s2,p1) -> Conv3x3 + DepthToSpace (exact weights;
    ///    see `rife_dml_split::dml_convtranspose_rewrite`). Off with
    ///    CHIDESCALER_DML_CONVT=off.
    /// 2. FP32-IO seven-channel RIFE models (Lite50/Lite75) could not enter
    ///    the GPU-resident WGL route, whose exchange buffers are FP16. They
    ///    fell back to the host route: CPU capture map, CPU NCHW packing,
    ///    upload, readback and re-upload every pair. Run an FP16-IO bridge
    ///    instead (one Cast at each end; weights, internal operators and the
    ///    FP32 motion state unchanged). Off with CHIDESCALER_DML_HALF_BRIDGE=off.
    /// Any failure keeps the original model.
    fn load_directml_arc_bridged(path: &Path, adapter_id: Option<i32>, app_dir: &Path) -> Result<Self> {
        if std::env::var("CHIDESCALER_DML_HALF_BRIDGE").ok().as_deref() == Some("off") {
            return Self::load_directml(path, adapter_id);
        }
        let arc_target = {
            let adapters = crate::platform::gpu::enumerate_adapters_quiet();
            let is_arc = |gpu: &&crate::platform::gpu::GpuAdapter| gpu.vendor_id == 0x8086 && gpu.name.contains("Arc");
            match adapter_id {
                Some(id) => adapters.iter().find(|gpu| gpu.device_id == id).is_some_and(|gpu| is_arc(&gpu)),
                None => {
                    // Auto uses ORT's high-performance adapter; only bridge when
                    // that is unambiguously the single Arc discrete GPU.
                    let mut arc = adapters.iter().filter(is_arc);
                    arc.next().is_some() && arc.next().is_none()
                }
            }
        };
        if !arc_target {
            return Self::load_directml(path, adapter_id);
        }
        struct Tuned { session: PathBuf, split: Option<PathBuf>, convt: usize, identity: usize, bridge: bool }
        let tuned = (|| -> Result<Option<Tuned>> {
            let bytes = std::fs::read(path)?;
            // Seven-channel RIFE only. DRBA gained ~1 ms (3.7%) but showed a
            // local 8-bit difference of 55 on the B570, so it stays original.
            // Image models are classified by their Conv structure elsewhere.
            if rife_dml_split::input_channels(&bytes).ok().flatten() != Some(7)
                || bytes.windows(15).any(|w| w == b"frazerion.split")
            {
                return Ok(None);
            }
            let Ok(half) = rife_dml_split::input_is_half(&bytes) else { return Ok(None) };
            let graph_off = std::env::var("CHIDESCALER_DML_CONVT").ok().as_deref() == Some("off");
            let (rewritten, convt) = if graph_off {
                (bytes.clone(), 0)
            } else {
                rife_dml_split::dml_convtranspose_rewrite(&bytes)?
            };
            let (rewritten, identity) = if graph_off {
                (rewritten, 0)
            } else {
                rife_dml_split::dml_identity_rewrite(&rewritten)?
            };
            let bridge = !half
                && std::env::var("CHIDESCALER_DML_HALF_BRIDGE").ok().as_deref() != Some("off");
            if convt == 0 && identity == 0 && !bridge {
                return Ok(None);
            }
            use sha2::Digest;
            let cache = app_dir.join("cache/directml/arc-tuned-v1");
            let key = format!("{:x}", sha2::Sha256::digest(&bytes));
            // Always compare contents: a truncated or edited cache file is
            // never trusted merely because it exists.
            let store = |name: String, data: &[u8]| -> Result<PathBuf> {
                let target = cache.join(name);
                if std::fs::read(&target).ok().as_deref() != Some(data) {
                    std::fs::create_dir_all(&cache)?;
                    let tmp = target.with_extension(format!("{}.tmp", std::process::id()));
                    std::fs::write(&tmp, data)?;
                    if let Err(error) = std::fs::rename(&tmp, &target) {
                        let _ = std::fs::remove_file(&tmp);
                        if std::fs::read(&target).ok().as_deref() != Some(data) {
                            return Err(error.into());
                        }
                    }
                }
                Ok(target)
            };
            let split = if bridge { Some(store(format!("{key}-split.onnx"), &rewritten)?) } else { None };
            let session_bytes = if bridge { rife_dml_split::half_bridge(&rewritten, true)? } else { rewritten };
            let session = store(format!("{key}.onnx"), &session_bytes)?;
            Ok(Some(Tuned { session, split, convt, identity, bridge }))
        })();
        let tuned = match tuned {
            Ok(Some(tuned)) => tuned,
            Ok(None) => return Self::load_directml(path, adapter_id),
            Err(error) => {
                log::warn!("directml-arc-graph: model={} result=skipped reason={error:#}", path.display());
                return Self::load_directml(path, adapter_id);
            }
        };
        match Self::load_directml(&tuned.session, adapter_id) {
            Ok(mut stage)
                if stage.interp != InterpKind::None
                    && (!tuned.bridge || (stage.fp16 && matches!(stage.interp, InterpKind::RifeV1 { channels: 7 }))) =>
            {
                stage.name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                stage.dml_half_bridge_source = tuned.split;
                log::info!(
                    "directml-arc-graph: model={} convtranspose_rewritten={} identity_removed={} io_bridge_fp16={} route={}",
                    stage.name,
                    tuned.convt,
                    tuned.identity,
                    tuned.bridge,
                    if tuned.bridge { "gpu-resident-eligible" } else { "unchanged" }
                );
                Ok(stage)
            }
            Ok(_) => {
                log::warn!("directml-arc-graph: model={} result=unexpected-signature action=original", path.display());
                Self::load_directml(path, adapter_id)
            }
            Err(error) => {
                log::warn!("directml-arc-graph: model={} result=session-failed reason={error:#} action=original", path.display());
                Self::load_directml(path, adapter_id)
            }
        }
    }

    /// v971: boost trunk/head graphs. A bridged FP32 model is split from its
    /// original (layout detection is per original precision) and only the
    /// exchange tensors are bridged. Returns whether the state is FP16.
    fn dml_split_graphs(&self, h: i64, w: i64) -> Result<(Vec<u8>, Vec<u8>, bool)> {
        if let Some(original) = &self.dml_half_bridge_source {
            let bytes = std::fs::read(original)?;
            let (trunk, head) = rife_dml_split::split(&bytes, h, w, false)?;
            Ok((
                rife_dml_split::half_bridge(&trunk, false)?,
                rife_dml_split::half_bridge(&head, true)?,
                false,
            ))
        } else {
            let bytes = std::fs::read(&self.source_path)?;
            let (trunk, head) = rife_dml_split::split(&bytes, h, w, self.fp16)?;
            Ok((trunk, head, self.fp16))
        }
    }

    fn load_neoarc(path: &Path, adapter_id: Option<i32>, app_dir: &Path) -> Result<Self> {
        let load_started = Instant::now();
        // v993: the DirectML session of a NeoArc stage only supplies the model
        // signature (and a degraded fallback), so it is created unoptimized.
        // Any branch that really runs DirectML reloads it with optimization.
        // Measured: Disable did not shorten creation (DirectML init dominates),
        // so it stays opt-in (CHIDESCALER_NEOARC_DML_META=light) and the normal
        // optimized session is kept.
        let light = std::env::var("CHIDESCALER_NEOARC_DML_META").ok().as_deref() == Some("light");
        let mut stage = if light {
            Self::load_directml_level(path, adapter_id, Some(GraphOptimizationLevel::Disable))?
        } else {
            Self::load_directml(path, adapter_id)?
        };
        let dml_session_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let dml_fallback = |reason: String| -> Result<Self> {
            let mut fallback = Self::load_directml(path, adapter_id)?;
            fallback.fallback_reason = Some(reason);
            Ok(fallback)
        };
        let is_image = stage.interp == InterpKind::None && stage.temporal_frames.is_none()
            && stage.session.inputs().len() == 1 && stage.session.outputs().len() == 1
            && parse_channels(&format!("{}", stage.session.inputs()[0].dtype())) == Some(3);
        let is_temporal = stage.interp == InterpKind::None && stage.temporal_frames.is_some()
            && stage.session.inputs().len() == 1 && stage.session.outputs().len() == 1
            && parse_channels(&format!("{}", stage.session.outputs()[0].dtype())) == Some(3);
        let float_rife = !stage.fp16 && stage.dml_rife_boost
            && matches!(stage.interp, InterpKind::RifeV1 { channels: 7 });
        if !(matches!(stage.interp, InterpKind::RifeV1 { channels: 7 } | InterpKind::Drba) || is_image || is_temporal)
            || (!stage.fp16 && !float_rife) {
            let reason = "NeoArc requires FP16 NCHW RGB SISR, seven-channel RIFE fourteen-channel DRBA or multi-frame RGB temporal restoration; DirectML retained".to_string();
            if light { return dml_fallback(reason); }
            stage.fallback_reason = Some(reason);
            return Ok(stage);
        }
        let adapters = crate::platform::gpu::enumerate_adapters_quiet();
        let target = adapter_id
            .and_then(|id| adapters.iter().find(|gpu| gpu.device_id == id))
            .or_else(|| {
                let mut arc = adapters.iter().filter(|gpu|
                    gpu.vendor_id == 0x8086 && gpu.name.contains("Arc"));
                let first = arc.next();
                if arc.next().is_none() { first } else { None }
            });
        if !target.is_some_and(|gpu| gpu.vendor_id == 0x8086 && gpu.name.contains("Arc")) {
            let reason = "NeoArc needs one selected Intel Arc GPU; DirectML retained".to_string();
            if light { return dml_fallback(reason); }
            stage.fallback_reason = Some(reason);
            return Ok(stage);
        }
        // v973: seven-channel RIFE runs the ConvTranspose/identity-tuned graph on
        // OpenVINO too (NeoRIFE v1.6 720p: 11.6 -> 9.7 ms, both stable-FP16).
        // Same IO precision as the original; any failure keeps the original.
        let rife7 = matches!(stage.interp, InterpKind::RifeV1 { channels: 7 });
        let tuned_started = Instant::now();
        let tuned = if rife7 && std::env::var("CHIDESCALER_NEOARC_CONVT").ok().as_deref() != Some("off") {
            match neoarc_tuned_rife_graph_memo(app_dir, path) {
                Ok(tuned) => tuned,
                Err(error) => {
                    log::warn!("neoarc-arc-graph: model={} result=skipped reason={error:#}", stage.name);
                    None
                }
            }
        } else {
            None
        };
        let tuned_ms = tuned_started.elapsed().as_secs_f64() * 1000.0;
        let bridge_started = Instant::now();
        // The OpenVINO input for a RIFE graph: FP16 as is, FP32 through the
        // half bridge. When the original's input was found unstable in plain
        // FP16, the tuned input starts on the stabilized graph as well.
        let neoarc_input = |source: &Path| -> Result<PathBuf> {
            if float_rife {
                // v993: memoized per source identity (see `memo_derived`).
                memo_derived(source, "neoarc-half-bridge", || {
                    let bytes = std::fs::read(source)?;
                    anyhow::ensure!(!rife_dml_split::input_is_half(&bytes)?, "expected FP32 RIFE input");
                    let wrapped = rife_dml_split::half_bridge(&bytes, true)?;
                    Ok(Some((super::neoarc_backend::cache_rife_bridge(app_dir, source, &wrapped)?, 0, 0)))
                })?
                .map(|(output, _, _)| output)
                .ok_or_else(|| anyhow!("half bridge produced no graph"))
            } else {
                Ok(source.to_path_buf())
            }
        };
        let rife_input = if rife7 {
            (|| -> Result<PathBuf> {
                let original = neoarc_input(path)?;
                if float_rife { log::info!("neoarc-rife-half-bridge: model={}", stage.name); }
                let Some((tuned_path, convt, identity)) = tuned.as_ref() else { return Ok(original) };
                let candidate = neoarc_input(tuned_path)?;
                let inherited = super::neoarc_backend::inherit_stable_fp16(app_dir, &original, &candidate)?;
                log::info!(
                    "neoarc-arc-graph: model={} convtranspose_rewritten={} identity_removed={} stable_fp16_inherited={}",
                    stage.name, convt, identity, inherited
                );
                Ok(candidate)
            })()
        } else {
            Ok(path.to_path_buf())
        };
        let bridge_ms = bridge_started.elapsed().as_secs_f64() * 1000.0;
        let session_started = Instant::now();
        let session_result = if is_image { NeoArcSession::load_image(app_dir, path) }
            else if is_temporal { NeoArcSession::load_temporal(app_dir, path, stage.temporal_frames.unwrap()) }
            else if stage.interp == InterpKind::Drba { NeoArcSession::load_drba(app_dir, path) }
            else { rife_input.and_then(|input| NeoArcSession::load(app_dir, &input)) };
        log::info!("neoarc-load-timing: model={} dml_session_ms={:.1} tuned_graph_ms={:.1} bridge_ms={:.1} neoarc_session_ms={:.1}",
            stage.name, dml_session_ms, tuned_ms, bridge_ms, session_started.elapsed().as_secs_f64() * 1000.0);
        match session_result {
            Ok(session) => {
                stage.neoarc_session = Some(session);
                stage.provider_desc = "NeoARC GPU".into();
                stage.fallback_reason = None;
                // Boost graphs are split from the same tuned graph.
                if let Some((tuned_path, _, _)) = tuned { stage.source_path = tuned_path; }
                log::info!("neoarc-session-ready: model={} adapter={} transport=CPU-visible validation=per-shape",
                    stage.name, adapter_id.unwrap_or_default());
            }
            Err(error) if tuned.is_some() => {
                log::warn!("neoarc-arc-graph: model={} result=session-failed reason={error:#} action=original", stage.name);
                match neoarc_input(path).and_then(|input| NeoArcSession::load(app_dir, &input)) {
                    Ok(session) => {
                        stage.neoarc_session = Some(session);
                        stage.provider_desc = "NeoARC GPU".into();
                        stage.fallback_reason = None;
                    }
                    Err(error) => {
                        let reason = format!("NeoArc runtime unavailable; DirectML retained: {error:#}");
                        if light { return dml_fallback(reason); }
                        stage.fallback_reason = Some(reason);
                    }
                }
            }
            Err(error) => {
                let reason = format!("NeoArc runtime unavailable; DirectML retained: {error:#}");
                if light { return dml_fallback(reason); }
                stage.fallback_reason = Some(reason);
            }
        }
        Ok(stage)
    }


    /// Keep the dynamic session for resize recovery. Eligibility comes from
    /// the ONNX graph and tensor signature, independent of the filename.
    fn ensure_dml_local_image_shape(&mut self, width: i32, height: i32) {
        if self.provider != OnnxProvider::DirectML {
            return;
        }
        let Some(symbols) = self.dml_image_symbols.clone() else {
            return;
        };
        if let Some(shape) = self.dml_image_shape {
            if shape != (width, height) {
                self.direct_output = None;
                if let Some((cached_shape, cached)) = self.dml_image_cached_fixed_session.take() {
                    if cached_shape == (width, height) {
                        let old_fixed = std::mem::replace(&mut self.session, cached);
                        if cache_dml_image_shape(shape) {
                            self.dml_image_cached_fixed_session = Some((shape, old_fixed));
                        }
                        self.dml_image_shape = Some(cached_shape);
                        log::info!(
                            "directml-local-image-shape: model={} reused cached {}x{} session",
                            self.name,
                            width,
                            height
                        );
                        return;
                    }
                }
                if let Some(dynamic) = self.dml_image_dynamic_session.take() {
                    let old_fixed = std::mem::replace(&mut self.session, dynamic);
                    if cache_dml_image_shape(shape) {
                        self.dml_image_cached_fixed_session = Some((shape, old_fixed));
                    }
                }
                self.dml_image_shape = None;
                // Restore the dynamic session for this frame. Once the new
                // geometry repeats, it may be specialized again; permanently
                // disabling it would leave Arc on the slower dynamic graph
                // after any ordinary resize.
                self.dml_image_specialization_disabled = false;
                log::info!(
                    "directml-compact-shape: resize to {}x{}; restored dynamic session",
                    width,
                    height
                );
            }
            return;
        }
        if self.dml_image_specialization_disabled
            || std::env::var("CHIDESCALER_DML_IMAGE_SHAPE").ok().as_deref() == Some("off")
            || width < 16
            || height < 16
            // Keep a bounded video-resolution specialization. The original
            // dynamic session remains available for resize recovery.
            || i64::from(width) * i64::from(height) > std::env::var("CHIDESCALER_DML_IMAGE_MAX_PIXELS").ok().and_then(|s|s.parse::<i64>().ok()).filter(|n|*n>0&&*n<=3840*2160).unwrap_or(1920*1080)
        {
            return;
        }
        let verified_shared = self
            .direct_output
            .as_ref()
            .is_some_and(|output| intel_arc_luid(u64::from_le_bytes(output.luid)));
        let verified_normal = self.last_upscale_input_size == Some((width, height))
            && self
                .fallback_dml_adapter
                .is_some_and(intel_arc_device_id);
        // Auto selection must be verified from the session's actual D3D12
        // device, rather than assuming the first/highest-VRAM DXGI adapter.
        let mut verified_auto=false;
        if !verified_shared && !verified_normal && self.fallback_dml_adapter.is_none()
            && self.last_upscale_input_size==Some((width,height)) {
            if self.dml_image_device_luid.is_none(){
                self.dml_image_device_luid=Some(match dml_session_luid(&self.session){
                    Ok(luid)=>{log::info!("directml-session-device: model={} luid={luid:016x} intel_arc={} source=provider-allocation host_pixels=0",self.name,intel_arc_luid(luid));Some(luid)},
                    Err(error)=>{log::debug!("directml-session-device-unavailable: model={} reason={error:#}",self.name);None}
                });
            }
            verified_auto=self.dml_image_device_luid.flatten().is_some_and(intel_arc_luid);
        }
        if !verified_shared && !verified_normal && !verified_auto {return;}
        if i64::from(width)*i64::from(height)>512*512 {
            if self.dml_image_budget_retry.is_some_and(|last|last.elapsed()<Duration::from_secs(2)){return;}
            let luid=self.direct_output.as_ref().map(|o|u64::from_le_bytes(o.luid))
                .or(self.dml_image_device_luid.flatten())
                .or_else(||self.fallback_dml_adapter.and_then(|id|intel_arc_adapters().iter().find(|a|a.0==id).map(|a|a.1)));
            if !luid.is_some_and(|luid|dml_image_shape_budget(luid,width,height)){
                self.dml_image_budget_retry=Some(Instant::now());return;
            }
            self.dml_image_budget_retry=None;
        }
        let graph_optimization = dml_graph_optimization(&self.source_path);
        let directml = match self.fallback_dml_adapter {
            Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
            None => ort::ep::DirectML::default().with_performance_preference(
                ort::ep::directml::PerformancePreference::HighPerformance,
            ),
        };
        let built = (|| -> Result<Session> {
            let mut builder = Session::builder()
                .map_err(oerr)?
                .with_optimization_level(graph_optimization)
                .map_err(oerr)?
                .with_execution_providers([directml.build()])
                .map_err(oerr)?
                .with_memory_pattern(false)
                .map_err(oerr)?;
            for (symbol, value) in [
                (&symbols[0], 1),
                (&symbols[1], i64::from(height)),
                (&symbols[2], i64::from(width)),
            ] {
                if !symbol.is_empty() {
                    builder = builder
                        .with_dimension_override(symbol, value)
                        .map_err(oerr)?;
                }
            }
            builder.commit_from_file(&self.source_path).map_err(oerr)
        })();
        match built {
            Ok(fixed) => {
                self.direct_output = None;
                self.dml_image_dynamic_session = Some(std::mem::replace(&mut self.session, fixed));
                self.dml_image_shape = Some((width, height));
                log::info!(
                    "directml-local-image-shape: model={} fixed {}x{} on Intel Arc",
                    self.name,
                    width,
                    height
                );
            }
            Err(error) => {
                self.dml_image_specialization_disabled = true;
                log::warn!(
                    "directml-local-image-shape: fixed session unavailable; dynamic session retained: {error:#}"
                );
            }
        }
    }

    /// Compile the first playback geometry once so ORT can fold RIFE's shape
    /// arithmetic and DirectML can fuse the resulting fixed-size graph.
    /// A resolution change returns to the dynamic session safely.
    fn ensure_dml_rife_shape(&mut self, width: i32, height: i32) -> Result<()> {
        if self.provider != OnnxProvider::DirectML
            || std::env::var("CHIDESCALER_DML_RIFE_SHAPE").ok().as_deref() == Some("off")
            || self.dml_rife_specialization_disabled
            || self.dml_rife_shape == Some((width, height))
        {
            return Ok(());
        }
        self.dml_host_interp_binding = None;
        let Some(symbols) = self.dml_rife_symbols.clone() else {
            return Ok(());
        };
        if self.dml_rife_cached_session.as_ref().is_some_and(|(size, _)| *size == (width, height)) {
            let (_, cached) = self.dml_rife_cached_session.take().unwrap();
            let previous = std::mem::replace(&mut self.session, cached);
            self.dml_rife_cached_session = self.dml_rife_shape.map(|size| (size, previous));
            self.dml_rife_shape = Some((width, height));
            log::info!("directml-rife-shape: model={} input={}x{} fixed=true result=cache-hit",
                self.name, width, height);
            return Ok(());
        }
        let graph_optimization = dml_graph_optimization(&self.source_path);
        let directml = match self.fallback_dml_adapter {
            Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
            None => ort::ep::DirectML::default().with_performance_preference(
                ort::ep::directml::PerformancePreference::HighPerformance,
            ),
        };
        let mut builder = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(graph_optimization)
            .map_err(oerr)?
            .with_execution_providers([directml.build()])
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?;
        if !symbols[0].is_empty() {
            builder = builder
                .with_dimension_override(&symbols[0], 1)
                .map_err(oerr)?
        }
        builder = builder
            .with_dimension_override(&symbols[1], i64::from(height))
            .map_err(oerr)?
            .with_dimension_override(&symbols[2], i64::from(width))
            .map_err(oerr)?;
        match builder.commit_from_file(&self.source_path).map_err(oerr) {
            Ok(session) => {
                let previous = std::mem::replace(&mut self.session, session);
                self.dml_rife_cached_session = self.dml_rife_shape.map(|size| (size, previous));
                self.dml_rife_shape = Some((width, height));
                log::info!(
                    "directml-rife-shape: model={} input={}x{} fixed=true result=ok",
                    self.name, width, height,
                );
                Ok(())
            }
            Err(error) => {
                self.dml_rife_specialization_disabled = true;
                log::warn!(
                    "directml-rife-shape: model={} input={}x{} fixed=true result=dynamic-fallback reason={error:#}",
                    self.name,
                    width,
                    height
                );
                // The old fixed session cannot serve a new resolution. Rebuild
                // the original dynamic DirectML path before accepting frames.
                if self.dml_rife_shape.is_some() {
                    let directml = match self.fallback_dml_adapter {
                        Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
                        None => ort::ep::DirectML::default().with_performance_preference(
                            ort::ep::directml::PerformancePreference::HighPerformance,
                        ),
                    };
                    let dynamic = Session::builder().map_err(oerr)?
                        .with_optimization_level(graph_optimization).map_err(oerr)?
                        .with_execution_providers([directml.build()]).map_err(oerr)?
                        .with_memory_pattern(false).map_err(oerr)?
                        .commit_from_file(&self.source_path).map_err(oerr)?;
                    self.session = dynamic;
                    self.dml_rife_cached_session = None;
                    self.dml_rife_shape = None;
                }
                Ok(())
            }
        }
    }

    fn load_neoamd(path: &Path, dml_adapter_id: Option<i32>, app_dir: &Path) -> Result<Self> {
        // Resolve the AMD adapter before creating DirectML metadata/fallback so
        // an Auto configuration with one AMD GPU plus another vendor does not
        // accidentally bind the fallback ORT session to a different device.
        // NeoAMD inference itself is still owned entirely by the external pack.
        let adapters = crate::platform::gpu::enumerate_adapters_quiet();
        let target = match dml_adapter_id {
            Some(device_id) => adapters
                .iter()
                .find(|adapter| adapter.device_id == device_id),
            None => {
                let mut amd = adapters
                    .iter()
                    .filter(|adapter| adapter.vendor_id == NEOAMD_VENDOR_AMD);
                let first = amd.next();
                if amd.next().is_some() { None } else { first }
            }
        };

        // A ready DirectML stage is retained as the immediate fail-safe. If a
        // valid AMD target is known, bind that fallback to the same adapter.
        let fallback_adapter = target
            .filter(|adapter| adapter.vendor_id == NEOAMD_VENDOR_AMD)
            .map(|adapter| adapter.device_id)
            .or(dml_adapter_id);
        let mut stage = Self::load_directml(path, fallback_adapter)?;

        let Some(target) = target else {
            stage.fallback_reason = Some(
                "NeoAMD could not resolve one unambiguous AMD compute adapter; DirectML retained"
                    .to_string(),
            );
            return Ok(stage);
        };
        if target.vendor_id != NEOAMD_VENDOR_AMD {
            stage.fallback_reason = Some(format!(
                "NeoAMD requires AMD RDNA4; selected GPU '{}' is not AMD",
                target.name
            ));
            return Ok(stage);
        }
        let backend = match NeoAmdBackend::load_from_app_dir(app_dir) {
            Ok(backend) => backend,
            Err(error) => {
                stage.fallback_reason = Some(format!(
                    "NeoAMD Backend Pack unavailable; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let adapter = match backend.query_adapter(target.luid) {
            Ok(info) => info,
            Err(error) => {
                stage.fallback_reason = Some(format!(
                    "NeoAMD rejected selected GPU; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let architecture = adapter.architecture_string();
        let required_caps = NEOAMD_CAP_DYNAMIC_RESOLUTION
            | NEOAMD_CAP_FP16_NATIVE
            | NEOAMD_CAP_GFX12_WMMA
            | NEOAMD_CAP_STRUCTURAL_ROUTING;
        if adapter.vendor_id != NEOAMD_VENDOR_AMD
            || !matches!(architecture.as_str(), "gfx1200" | "gfx1201")
            || adapter.capabilities & required_caps != required_caps
        {
            stage.fallback_reason = Some(format!(
                "NeoAMD unsupported GPU/backend capabilities (arch='{architecture}', caps=0x{:x}); DirectML retained",
                adapter.capabilities
            ));
            return Ok(stage);
        }
        let session = match backend.create_session(path, target.luid) {
            Ok(session) => session,
            Err(error) => {
                // Unsupported/ambiguous graphs deliberately land here. The
                // pack performs structural routing; Neo does not special-case
                // model names, file hashes or node names.
                stage.fallback_reason = Some(format!(
                    "NeoAMD graph not accepted; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let info = session.info();
        let host_kind = if stage.interp != InterpKind::None {
            NEOAMD_SESSION_INTERPOLATION
        } else if stage.temporal_frames.is_some() {
            NEOAMD_SESSION_TEMPORAL
        } else {
            NEOAMD_SESSION_IMAGE
        };
        if info.kind != host_kind {
            stage.fallback_reason = Some(format!(
                "NeoAMD session contract mismatch (pack kind {}, host kind {}); DirectML retained",
                info.kind, host_kind
            ));
            return Ok(stage);
        }
        if host_kind == NEOAMD_SESSION_TEMPORAL
            && stage.temporal_frames != Some(info.temporal_frames as usize)
        {
            stage.fallback_reason = Some(format!(
                "NeoAMD temporal-frame contract mismatch (pack {}, model {:?}); DirectML retained",
                info.temporal_frames, stage.temporal_frames
            ));
            return Ok(stage);
        }
        stage.provider = OnnxProvider::NeoAMD;
        stage.provider_desc = format!(
            "NeoAMD {}{}",
            architecture,
            backend
                .description()
                .map(|value| format!(" / {value}"))
                .unwrap_or_default()
        );
        stage.fallback_reason = None;
        stage.neoamd_session = Some(session);
        log::info!(
            "neoamd-session-ready: model={} gpu='{}' luid={:016x} arch={} kind={} dynamic_resolution=true engine_cache=false",
            path.display(),
            target.name,
            target.luid,
            architecture,
            info.kind
        );
        Ok(stage)
    }

    fn load_neortx(path: &Path, dml_adapter_id: Option<i32>, app_dir: &Path) -> Result<Self> {
        // Resolve the AMD adapter before creating DirectML metadata/fallback so
        // an Auto configuration with one AMD GPU plus another vendor does not
        // accidentally bind the fallback ORT session to a different device.
        // NeoRTX inference itself is still owned entirely by the external pack.
        let adapters = crate::platform::gpu::enumerate_adapters_quiet();
        let target = match dml_adapter_id {
            Some(device_id) => adapters
                .iter()
                .find(|adapter| adapter.device_id == device_id),
            None => {
                let mut amd = adapters
                    .iter()
                    .filter(|adapter| adapter.vendor_id == 0x10DE);
                let first = amd.next();
                if amd.next().is_some() { None } else { first }
            }
        };

        // A ready DirectML stage is retained as the immediate fail-safe. If a
        // valid AMD target is known, bind that fallback to the same adapter.
        let fallback_adapter = target
            .filter(|adapter| adapter.vendor_id == 0x10DE)
            .map(|adapter| adapter.device_id)
            .or(dml_adapter_id);
        let mut stage = Self::load_directml(path, fallback_adapter)?;

        let Some(target) = target else {
            stage.fallback_reason = Some(
                "NeoRTX could not resolve one unambiguous NVIDIA compute adapter; DirectML retained"
                    .to_string(),
            );
            return Ok(stage);
        };
        if target.vendor_id != 0x10DE {
            stage.fallback_reason = Some(format!(
                "NeoRTX requires NVIDIA Ampere or newer; selected GPU '{}' is not NVIDIA",
                target.name
            ));
            return Ok(stage);
        }
        let backend = match NeoAmdBackend::load_neortx_from_app_dir(app_dir) {
            Ok(backend) => backend,
            Err(error) => {
                stage.fallback_reason = Some(format!(
                    "NeoRTX Backend Pack unavailable; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let adapter = match backend.query_adapter(target.luid) {
            Ok(info) => info,
            Err(error) => {
                stage.fallback_reason = Some(format!(
                    "NeoRTX rejected selected GPU; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let architecture = adapter.architecture_string();
        let required_caps = NEOAMD_CAP_DYNAMIC_RESOLUTION
            | NEOAMD_CAP_FP16_NATIVE
            | NEOAMD_CAP_GFX12_WMMA
            | NEOAMD_CAP_STRUCTURAL_ROUTING;
        if adapter.vendor_id != 0x10DE
            || !matches!(
                architecture.as_str(),
                "sm_80" | "sm_86" | "sm_89" | "sm_120"
            )
            || adapter.capabilities & required_caps != required_caps
        {
            stage.fallback_reason = Some(format!(
                "NeoRTX unsupported GPU/backend capabilities (arch='{architecture}', caps=0x{:x}); DirectML retained",
                adapter.capabilities
            ));
            return Ok(stage);
        }
        let session = match backend.create_session(path, target.luid) {
            Ok(session) => session,
            Err(error) => {
                // Unsupported/ambiguous graphs deliberately land here. The
                // pack performs structural routing; Neo does not special-case
                // model names, file hashes or node names.
                stage.fallback_reason = Some(format!(
                    "NeoRTX graph not accepted; DirectML retained: {error}"
                ));
                return Ok(stage);
            }
        };
        let info = session.info();
        let host_kind = if stage.interp != InterpKind::None {
            NEOAMD_SESSION_INTERPOLATION
        } else if stage.temporal_frames.is_some() {
            NEOAMD_SESSION_TEMPORAL
        } else {
            NEOAMD_SESSION_IMAGE
        };
        if info.kind != host_kind {
            stage.fallback_reason = Some(format!(
                "NeoRTX session contract mismatch (pack kind {}, host kind {}); DirectML retained",
                info.kind, host_kind
            ));
            return Ok(stage);
        }
        if host_kind == NEOAMD_SESSION_TEMPORAL
            && stage.temporal_frames != Some(info.temporal_frames as usize)
        {
            stage.fallback_reason = Some(format!(
                "NeoRTX temporal-frame contract mismatch (pack {}, model {:?}); DirectML retained",
                info.temporal_frames, stage.temporal_frames
            ));
            return Ok(stage);
        }
        stage.provider = OnnxProvider::NeoRTX;
        stage.provider_desc = format!(
            "NeoRTX {}{}",
            architecture,
            backend
                .description()
                .map(|value| format!(" / {value}"))
                .unwrap_or_default()
        );
        stage.fallback_reason = None;
        stage.neoamd_session = Some(session);
        log::info!(
            "neortx-session-ready: model={} gpu='{}' luid={:016x} arch={} kind={} dynamic_resolution=true engine_cache=false",
            path.display(),
            target.name,
            target.luid,
            architecture,
            info.kind
        );
        Ok(stage)
    }

    fn disable_neoamd_after_runtime_failure(&mut self, reason: &str) {
        if !matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            return;
        }
        log::warn!(
            "neoamd-runtime-fallback: model={} reason={} action=DirectML",
            self.source_path.display(),
            reason
        );
        self.neoamd_session = None;
        // A TemporalFix session can reject a runtime geometry that the first
        // NeoAMD production gate does not yet support (v049: only 1280x720).
        // Do not immediately promote that same live texture chain into the
        // DirectML temporal shared-buffer bridge: on stacked ONNX -> Temporal
        // chains the provider transition can otherwise reuse cross-API temporal
        // state created under a different ownership contract and present black.
        // The ordinary CPU-visible DirectML retry is slower but authoritative
        // and remains enabled. A newly built DirectML session starts clean.
        if self.temporal_frames.is_some() {
            self.direct_output_disabled = true;
            log::warn!(
                "neoamd-temporal-fallback-safety: model={} directml_temporal_gpu_path=disabled_for_session fallback=cpu-visible-directml",
                self.name
            );
        }
        self.provider = OnnxProvider::DirectML;
        self.provider_desc = self
            .fallback_dml_adapter
            .map(|id| format!("DirectML device {id}"))
            .unwrap_or_else(|| "DirectML auto/high-performance".to_string());
        self.fallback_reason = Some(format!("NeoAMD runtime fallback: {reason}"));
    }

    fn run_neoamd_u8(
        &mut self,
        w: i32,
        h: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phase: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        // NeoAMD interpolation uses the same run_u8 ABI as image/temporal
        // sessions, but its timing belongs to last_interp_profile. v757 stored
        // every NeoAMD run in last_upscale_profile, so the asynchronous RIFE
        // worker returned profile=None and the GUI fell back to the tiny CPU
        // handoff timing ("Frame interpolation" ~0-1ms) even while HIP was
        // doing ~10ms of real interpolation work. Keep the provider timing in
        // the profile family that the engine consumes for the active session.
        let is_interp = self.interp != InterpKind::None;
        if is_interp {
            self.last_interp_profile = None;
        } else {
            self.last_upscale_profile = None;
        }

        let run_started = Instant::now();
        let result = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .run_u8(w, h, inputs, input_pixel_stride, output_pixel_stride, phase)
            .map_err(anyhow::Error::msg);
        match result {
            Ok((ow, oh, bytes)) => {
                let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
                if is_interp {
                    self.last_interp_profile = Some(InterpProfile {
                        pack_ms: 0.0,
                        run_ms,
                        out_ms: 0.0,
                        // Backend Pack v017 is runtime-W/H safe and does not
                        // expose a host-visible padded shape. The logical input
                        // size is therefore the correct profile geometry.
                        padded_size: (w.max(0) as usize, h.max(0) as usize),
                    });
                    log::debug!(
                        "neoamd-interp-profile: model='{}' run_ms={:.3} total_ms={:.3} input={}x{} output={}x{} phase={:.4}",
                        self.name,
                        run_ms,
                        run_ms,
                        w,
                        h,
                        ow,
                        oh,
                        phase
                    );
                } else {
                    self.last_upscale_profile = Some(UpscaleProfile {
                        pack_ms: 0.0,
                        run_ms,
                        out_ms: 0.0,
                        output_size: (ow.max(0) as usize, oh.max(0) as usize),
                    });
                    self.last_upscale_input_size = Some((w, h));
                }
                Ok((ow, oh, bytes))
            }
            Err(error) => {
                let reason = format!("{error:#}");
                if self.temporal_frames.is_some() {
                    // TemporalFix runtime geometry rejection is a per-W/H
                    // fail-closed decision, not a provider transition. v782
                    // changed the cached stage itself to DirectML here; after
                    // OFF -> ON, reset_backend_runtime_state cleared the safety
                    // latch and resurrected the unstable DML temporal GPU bridge.
                    // Preserve the NeoAMD session/provider and remember only the
                    // rejected geometry. The caller retries authoritative DML
                    // CPU-visible inference for this frame/size.
                    self.neoamd_temporal_rejected_size = Some((w, h));
                    self.direct_output_disabled = true;
                    log::warn!(
                        "neoamd-temporal-runtime-failclosed: model={} size={}x{} reason={} provider=NeoAMD-preserved fallback=DirectML-cpu-visible",
                        self.name,
                        w,
                        h,
                        reason
                    );
                } else {
                    self.disable_neoamd_after_runtime_failure(&reason);
                }
                Err(error)
            }
        }
    }

    fn run_neoamd_many_u8(
        &mut self,
        w: i32,
        h: i32,
        inputs: &[&[u8]],
        input_pixel_stride: usize,
        output_pixel_stride: usize,
        phases: &[f32],
    ) -> Result<Option<Vec<(i32, i32, Vec<u8>)>>> {
        if phases.is_empty() {
            return Ok(Some(Vec::new()));
        }
        self.last_interp_profile = None;
        let run_started = Instant::now();
        let result = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .run_interp_many_u8(
                w,
                h,
                inputs,
                input_pixel_stride,
                output_pixel_stride,
                phases,
            )
            .map_err(anyhow::Error::msg);
        match result {
            Ok(Some(results)) => {
                let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
                self.last_interp_profile = Some(InterpProfile {
                    pack_ms: 0.0,
                    run_ms,
                    out_ms: 0.0,
                    padded_size: (w.max(0) as usize, h.max(0) as usize),
                });
                log::debug!(
                    "neoamd-interp-many-profile: model='{}' outputs={} phases={:?} run_ms={:.3} per_output_ms={:.3} input={}x{}",
                    self.name,
                    results.len(),
                    phases,
                    run_ms,
                    run_ms / results.len().max(1) as f64,
                    w,
                    h
                );

                // v765/v023 native DRBA performance diagnostic. Earlier hosts treated
                // three slow Native intervals as a runtime failure and silently switched
                // the stage to DirectML. That made it impossible to validate the restored
                // v107-style NeoAMD path. Performance is now diagnostic only; only a real
                // runtime/ABI failure may change provider.
                if self.interp == InterpKind::Drba {
                    let pixels = (w.max(1) as f64) * (h.max(1) as f64);
                    let scale = (pixels / (1280.0 * 720.0)).clamp(0.25, 8.0);
                    let per_output_budget_ms = (12.0 * scale).clamp(5.0, 60.0);
                    let total_budget_ms = per_output_budget_ms * results.len().max(1) as f64;
                    if run_ms > total_budget_ms {
                        self.neoamd_drba_slow_streak =
                            self.neoamd_drba_slow_streak.saturating_add(1);
                        log::warn!(
                            "neoamd-drba-performance-guard: model={} streak={} run_ms={:.2} budget_ms={:.2} outputs={} action={}",
                            self.name,
                            self.neoamd_drba_slow_streak,
                            run_ms,
                            total_budget_ms,
                            results.len(),
                            if self.neoamd_drba_slow_streak >= 3 {
                                "native-stays-active"
                            } else {
                                "observe"
                            }
                        );
                        if self.neoamd_drba_slow_streak >= 3 {
                            // v765/v023: performance is diagnostic only. Runtime/ABI
                            // failures still fail safe to DirectML, but a slow Native
                            // DRBA interval must remain on NeoAMD so the optimized
                            // production path can be measured instead of silently
                            // changing providers underneath the user.
                            log::warn!(
                                "neoamd-drba-performance-guard: model={} streak={} action=native-stays-active diagnostic=1",
                                self.name,
                                self.neoamd_drba_slow_streak
                            );
                            self.neoamd_drba_slow_streak = 0;
                        }
                    } else {
                        self.neoamd_drba_slow_streak = 0;
                    }
                } else {
                    self.neoamd_drba_slow_streak = 0;
                }
                Ok(Some(results))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                let reason = format!("{error:#}");
                self.disable_neoamd_after_runtime_failure(&reason);
                Err(error)
            }
        }
    }

    pub(crate) fn supports_neoamd_interp_shared_rgba8(&self) -> bool {
        matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && matches!(&self.interp, InterpKind::RifeV1 { .. } | InterpKind::Drba)
            && !self.neoamd_shared_disabled
            && self
                .neoamd_session
                .as_ref()
                .is_some_and(|session| session.supports_interp_stream_shared_rgba8())
    }

    pub(crate) fn prepare_neoamd_interp_stream_shared_rgba8(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
    ) -> Result<Option<crate::render::neoamd_backend::NeoAmdSharedOutput>> {
        if !self.supports_neoamd_interp_shared_rgba8() {
            return Ok(None);
        }
        self.last_interp_profile = None;
        let started = Instant::now();
        let result = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .prepare_interp_stream_shared_rgba8(w, h, frames)
            .map_err(anyhow::Error::msg);
        match result {
            Ok(Some(shared)) => {
                let previous_key = self.neoamd_shared_key;
                self.neoamd_shared_key = Some(shared.resource_key);
                self.neoamd_shared_input_key = None;
                self.neoamd_shared_input_size = Some((w, h));
                let common_ms = started.elapsed().as_secs_f64() * 1000.0;
                let log_index = NEOAMD_STREAM_PREP_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                if previous_key != Some(shared.resource_key) {
                    log::info!(
                        "neoamd-interp-shared-output-ready: model='{}' size={}x{} key={} bytes={} allocation_bytes={} common_ms={:.3}",
                        self.name,
                        w,
                        h,
                        shared.resource_key,
                        shared.byte_len,
                        shared.allocation_byte_len,
                        common_ms
                    );
                } else if log_index < 8 || log_index % 120 == 0 {
                    log::debug!(
                        "neoamd-interp-shared-prepare: model='{}' input={}x{} frames={} common_ms={:.3} sampled_index={}",
                        self.name,
                        w,
                        h,
                        frames.len(),
                        common_ms,
                        log_index
                    );
                }
                Ok(Some(shared))
            }
            Ok(None) => Ok(None),
            Err(error) => {
                self.neoamd_shared_disabled = true;
                log::warn!(
                    "neoamd-interp-shared-output-disabled: model='{}' reason={error:#} fallback=cpu-visible-cooperative-stream",
                    self.name
                );
                Ok(None)
            }
        }
    }

    /// v978: slot 0 plus up to `want - 1` extra cooperative output slots with
    /// the same geometry. Returns only slot 0 when the Backend Pack has no ring
    /// export or allocation fails, so callers keep the serialized v977 contract.
    pub(crate) fn neoamd_interp_ring_outputs(
        &mut self,
        w: i32,
        h: i32,
        base: crate::render::neoamd_backend::NeoAmdSharedOutput,
        want: usize,
    ) -> Vec<crate::render::neoamd_backend::NeoAmdSharedOutput> {
        let want = want.clamp(1, 8);
        if self.neoamd_interp_ring_base != Some(base.resource_key) {
            // A new slot-0 allocation released every previous ring slot.
            self.neoamd_interp_ring.clear();
            self.neoamd_interp_ring_base = Some(base.resource_key);
        }
        if want > 1 && !self.neoamd_interp_ring_unavailable {
            while self.neoamd_interp_ring.len() + 1 < want {
                let slot = (self.neoamd_interp_ring.len() + 1) as u32;
                let result = match self.neoamd_session.as_mut() {
                    Some(session) => session.prepare_interp_shared_ring_slot(w, h, slot),
                    None => Ok(None),
                };
                match result {
                    Ok(Some(extra)) if extra.width == base.width && extra.height == base.height => {
                        self.neoamd_interp_ring.push(extra);
                    }
                    Ok(Some(_)) => {
                        self.neoamd_interp_ring_unavailable = true;
                        log::warn!("neoamd-interp-ring-disabled: model='{}' reason=geometry-mismatch", self.name);
                        break;
                    }
                    Ok(None) => {
                        self.neoamd_interp_ring_unavailable = true;
                        log::info!("neoamd-interp-ring-unavailable: model='{}' reason=backend-without-ring-export mode=single-buffer", self.name);
                        break;
                    }
                    Err(error) => {
                        self.neoamd_interp_ring_unavailable = true;
                        log::warn!("neoamd-interp-ring-disabled: model='{}' slot={} reason={error}", self.name, slot);
                        break;
                    }
                }
            }
        }
        let mut outputs = Vec::with_capacity(want);
        outputs.push(base);
        outputs.extend(self.neoamd_interp_ring.iter().take(want - 1).copied());
        outputs
    }

    pub(crate) fn run_neoamd_interp_stream_phase_shared_rgba8(
        &mut self,
        w: i32,
        h: i32,
        input_count: usize,
        phase: f32,
        shared: crate::render::neoamd_backend::NeoAmdSharedOutput,
    ) -> Result<PreparedNeoAmdSharedOutput> {
        anyhow::ensure!(
            self.supports_neoamd_interp_shared_rgba8(),
            "NeoAMD shared interpolation route is unavailable"
        );
        let run_started = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .run_interp_stream_phase_shared_rgba8(w, h, input_count, phase, shared.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
        self.last_interp_profile = Some(InterpProfile {
            pack_ms: 0.0,
            run_ms,
            out_ms: 0.0,
            padded_size: (w.max(0) as usize, h.max(0) as usize),
        });
        let handle = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .shared_output_handle(shared.resource_key)
            .map_err(anyhow::Error::msg)?;
        let log_index = NEOAMD_STREAM_PHASE_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
        if log_index < 16 || log_index % 240 == 0 {
            log::debug!(
                "neoamd-interp-shared-phase: model='{}' phase={:.6} run_ms={:.3} input={}x{} output={}x{} key={} sampled_index={}",
                self.name,
                phase,
                run_ms,
                w,
                h,
                shared.width,
                shared.height,
                shared.resource_key,
                log_index
            );
        }
        Ok(PreparedNeoAmdSharedOutput {
            key: shared.resource_key,
            handle,
            byte_len: shared.byte_len,
            heap_byte_len: shared.allocation_byte_len,
            luid: self
                .neoamd_session
                .as_ref()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .adapter_luid()
                .to_le_bytes(),
            size: (shared.width, shared.height),
        })
    }

    pub(crate) fn supports_neoamd_interp_stream(&self) -> bool {
        matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && matches!(&self.interp, InterpKind::RifeV1 { .. } | InterpKind::Drba)
            && self
                .neoamd_session
                .as_ref()
                .is_some_and(|session| session.supports_interp_stream_u8())
    }

    pub(crate) fn supports_neoarc_interp_stream(&self) -> bool {
        self.neoarc_session.is_some()
            && matches!(&self.interp, InterpKind::RifeV1 { .. } | InterpKind::Drba)
    }

    pub(crate) fn prepare_neoamd_interp_stream_rgba8(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
    ) -> Result<bool> {
        if !self.supports_neoamd_interp_stream() {
            return Ok(false);
        }
        self.last_interp_profile = None;
        let started = Instant::now();
        let result = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .prepare_interp_stream_u8(w, h, frames, 4, 3)
            .map_err(anyhow::Error::msg);
        match result {
            Ok(prepared) => {
                if prepared {
                    let log_index = NEOAMD_STREAM_PREP_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                    if log_index < 8 || log_index % 120 == 0 {
                        log::debug!(
                            "neoamd-interp-stream-prepare: model='{}' input={}x{} frames={} common_ms={:.3} sampled_index={}",
                            self.name,
                            w,
                            h,
                            frames.len(),
                            started.elapsed().as_secs_f64() * 1000.0,
                            log_index
                        );
                    }
                }
                Ok(prepared)
            }
            Err(error) => {
                let reason = format!("{error:#}");
                self.disable_neoamd_after_runtime_failure(&reason);
                Err(error)
            }
        }
    }

    pub(crate) fn run_neoamd_interp_stream_phase_rgba8(
        &mut self,
        w: i32,
        h: i32,
        input_count: usize,
        phase: f32,
    ) -> Result<(i32, i32, Vec<u8>)> {
        anyhow::ensure!(
            matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX),
            "NeoAMD interpolation stream provider is no longer active"
        );
        let started = Instant::now();
        let result = self
            .neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?
            .run_interp_stream_phase_u8(w, h, input_count, 4, 3, phase)
            .map_err(anyhow::Error::msg);
        match result {
            Ok((ow, oh, bytes)) => {
                let run_ms = started.elapsed().as_secs_f64() * 1000.0;
                self.last_interp_profile = Some(InterpProfile {
                    pack_ms: 0.0,
                    run_ms,
                    out_ms: 0.0,
                    padded_size: (w.max(0) as usize, h.max(0) as usize),
                });
                let log_index = NEOAMD_STREAM_PHASE_LOG_COUNT.fetch_add(1, Ordering::Relaxed);
                if log_index < 16 || log_index % 240 == 0 {
                    log::debug!(
                        "neoamd-interp-stream-phase: model='{}' phase={:.6} run_ms={:.3} input={}x{} output={}x{} sampled_index={}",
                        self.name,
                        phase,
                        run_ms,
                        w,
                        h,
                        ow,
                        oh,
                        log_index
                    );
                }
                Ok((ow, oh, bytes))
            }
            Err(error) => {
                let reason = format!("{error:#}");
                self.disable_neoamd_after_runtime_failure(&reason);
                Err(error)
            }
        }
    }

    pub(crate) fn is_neoamd_temporal_shared_filter(&self) -> bool {
        matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && self.temporal_frames.is_some()
            && self.should_try_neoamd_gpu_texture()
    }

    pub(crate) fn should_try_neoamd_gpu_texture(&self) -> bool {
        matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && !self.neoamd_shared_disabled
            && self.fp16
            && self.interp == InterpKind::None
            && self.neoamd_session.as_ref().is_some_and(|session| {
                if self.temporal_frames.is_some() {
                    session.supports_temporal_shared_io_fp16()
                } else {
                    session.supports_shared_io_fp16()
                }
            })
    }

    fn process_neoamd_temporal_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<Option<GpuTex>> {
        let (w, h) = (input_texture.w(), input_texture.h());
        if self.neoamd_shared_input_size != Some((w, h)) {
            self.retire_neoamd_shared_output(gc);
        }
        let (adapter_luid, io) = {
            let session = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?;
            let adapter_luid = session.adapter_luid();
            if gc.external_device_luid() != Some(adapter_luid.to_le_bytes()) {
                return Ok(None);
            }
            let io = session
                .prepare_temporal_shared_io_fp16(w, h)
                .map_err(anyhow::Error::msg)?;
            (adapter_luid, io)
        };

        for old_key in [self.neoamd_shared_input_key, self.neoamd_shared_key]
            .into_iter()
            .flatten()
        {
            if old_key != io.input.resource_key && old_key != io.output.resource_key {
                let _ = gc.wait_external_buffer_idle(old_key);
                gc.clear_external_buffer_key(old_key);
            }
        }
        self.neoamd_shared_input_key = Some(io.input.resource_key);
        self.neoamd_shared_key = Some(io.output.resource_key);
        self.neoamd_shared_input_size = Some((w, h));

        let input_imported = gc.has_external_import(io.input.resource_key);
        if !input_imported {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_input_handle(io.input.resource_key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                io.input.resource_key,
                handle,
                io.input.byte_len,
                io.input.allocation_byte_len,
                adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }
        let output_imported = gc.has_external_import(io.output.resource_key);
        if !output_imported {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_output_handle(io.output.resource_key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                io.output.resource_key,
                handle,
                io.output.byte_len,
                io.output.allocation_byte_len,
                adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }

        // The previous OpenGL consumer must release the shared output before HIP
        // overwrites it. rgba_texture_to_external_nchw_f16 finishes the current
        // frame's GL write before the Backend Pack reads the shared input.
        gc.wait_external_buffer_idle(io.output.resource_key)
            .map_err(anyhow::Error::msg)?;
        let pack_started = Instant::now();
        gc.rgba_texture_to_external_nchw_f16(io.input.resource_key, input_texture, w, h)
            .map_err(anyhow::Error::msg)?;
        let pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;

        let run_started = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .run_temporal_shared_io_fp16(w, h, io.input.resource_key, io.output.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;

        let out_started = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(io.output.resource_key, w, h)
            .map_err(anyhow::Error::msg)?;
        let out_ms = out_started.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms,
            output_size: (w as usize, h as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        if should_log_neoamd_profile() {
            log::debug!(
                "neoamd-temporal-profile: model='{}' ingress=shared-fp16-gpu-ring pack_ms={:.3} run_ms={:.3} out_ms={:.3} total_ms={:.3} input={}x{}",
                self.name,
                pack_ms,
                run_ms,
                out_ms,
                pack_ms + run_ms + out_ms,
                w,
                h
            );
        }
        if !input_imported || !output_imported {
            log::info!(
                "NeoAMD TemporalFix GPU-resident IO ready: model='{}' size={}x{} input_key={} output_key={} LUID={:016x} path=OpenGL->D3D12-current-NCHW3->HIP-GPU-TRUE7-ring->D3D12-NCHW3->OpenGL",
                self.name,
                w,
                h,
                io.input.resource_key,
                io.output.resource_key,
                adapter_luid
            );
        }
        Ok(Some(texture))
    }

    fn process_neoamd_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
    ) -> Result<Option<GpuTex>> {
        if !self.should_try_neoamd_gpu_texture() {
            return Ok(None);
        }
        if self.temporal_frames.is_some() {
            return self.process_neoamd_temporal_gpu_texture(gc, input_texture);
        }
        let (w, h) = (input_texture.w(), input_texture.h());

        if self.neoamd_shared_input_size != Some((w, h)) {
            self.retire_neoamd_shared_output(gc);
        }

        let (adapter_luid, io) = {
            let session = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?;
            let adapter_luid = session.adapter_luid();
            if gc.external_device_luid() != Some(adapter_luid.to_le_bytes()) {
                return Ok(None);
            }
            let io = session
                .prepare_shared_io_fp16(w, h, 0.5)
                .map_err(anyhow::Error::msg)?;
            (adapter_luid, io)
        };

        for old_key in [self.neoamd_shared_input_key, self.neoamd_shared_key]
            .into_iter()
            .flatten()
        {
            if old_key != io.input.resource_key && old_key != io.output.resource_key {
                let _ = gc.wait_external_buffer_idle(old_key);
                gc.clear_external_buffer_key(old_key);
            }
        }
        self.neoamd_shared_input_key = Some(io.input.resource_key);
        self.neoamd_shared_key = Some(io.output.resource_key);
        self.neoamd_shared_input_size = Some((w, h));

        let input_imported = gc.has_external_import(io.input.resource_key);
        if !input_imported {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_input_handle(io.input.resource_key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                io.input.resource_key,
                handle,
                io.input.byte_len,
                io.input.allocation_byte_len,
                adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }
        let output_imported = gc.has_external_import(io.output.resource_key);
        if !output_imported {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_output_handle(io.output.resource_key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                io.output.resource_key,
                handle,
                io.output.byte_len,
                io.output.allocation_byte_len,
                adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }

        // Previous frame's OpenGL read must complete before HIP overwrites the
        // output. The input is safe because the preceding HIP run synchronizes
        // before returning and the pack call below glFinish-es its write.
        gc.wait_external_buffer_idle(io.output.resource_key)
            .map_err(anyhow::Error::msg)?;

        let pack_started = Instant::now();
        gc.rgba_texture_to_external_nchw_f16(io.input.resource_key, input_texture, w, h)
            .map_err(anyhow::Error::msg)?;
        let pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;

        let run_started = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .run_shared_io_fp16(w, h, 0.5, io.input.resource_key, io.output.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;

        let out_started = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(io.output.resource_key, io.output.width, io.output.height)
            .map_err(anyhow::Error::msg)?;
        let out_ms = out_started.elapsed().as_secs_f64() * 1000.0;

        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms,
            output_size: (io.output.width as usize, io.output.height as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        if should_log_neoamd_profile() {
            log::debug!(
                "neoamd-profile: model='{}' ingress=shared-fp16 pack_ms={:.3} run_ms={:.3} out_ms={:.3} total_ms={:.3} input={}x{} output={}x{}",
                self.name,
                pack_ms,
                run_ms,
                out_ms,
                pack_ms + run_ms + out_ms,
                w,
                h,
                io.output.width,
                io.output.height
            );
        }
        if !input_imported || !output_imported {
            log::info!(
                "NeoAMD full GPU-resident IO ready: model='{}' input={}x{} output={}x{} input_key={} output_key={} LUID={:016x} path=OpenGL->D3D12-shared-FP16->HIP-WMMA->D3D12-shared-FP16->OpenGL",
                self.name,
                w,
                h,
                io.output.width,
                io.output.height,
                io.input.resource_key,
                io.output.resource_key,
                adapter_luid
            );
        }
        Ok(Some(texture))
    }

    pub(crate) fn supports_neoamd_peer_chain(&self) -> bool {
        matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && !self.neoamd_shared_disabled
            && self.fp16
            && self.interp == InterpKind::None
            && self.neoamd_session.as_ref().is_some_and(|session| {
                session.supports_peer_shared_io_fp16()
                    && if self.temporal_frames.is_some() {
                        session.supports_temporal_shared_io_fp16()
                    } else {
                        session.supports_shared_io_fp16()
                    }
            })
    }

    /// First stage of a consecutive NeoAMD chain. Keep the model output as the
    /// Backend-Pack-owned NCHW FP16 tensor; do not convert it to RGBA/OpenGL.
    pub(crate) fn process_neoamd_peer_chain_rgba8(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(NeoAmdPeerTensor, f64)>> {
        if !self.supports_neoamd_peer_chain() || self.temporal_frames.is_some() {
            return Ok(None);
        }
        let pixels = usize::try_from(w)?
            .checked_mul(usize::try_from(h)?)
            .ok_or_else(|| anyhow!("NeoAMD peer-chain input size overflow"))?;
        anyhow::ensure!(
            rgba.len() >= pixels * 4,
            "NeoAMD peer-chain RGBA input buffer too small"
        );
        let (adapter_luid, shared) = {
            let session = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?;
            let adapter_luid = session.adapter_luid();
            let shared = session
                .prepare_shared_fp16(w, h, 1, 4, 0.5)
                .map_err(anyhow::Error::msg)?;
            (adapter_luid, shared)
        };
        self.neoamd_shared_key = Some(shared.resource_key);
        self.neoamd_shared_input_size = Some((w, h));
        if gc.has_external_import(shared.resource_key) {
            gc.wait_external_buffer_idle(shared.resource_key)
                .map_err(anyhow::Error::msg)?;
        }
        let started = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .run_u8_shared_fp16(w, h, &[rgba], 4, 0.5, shared.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms: 0.0,
            run_ms,
            out_ms: 0.0,
            output_size: (shared.width as usize, shared.height as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        Ok(Some((
            NeoAmdPeerTensor {
                key: shared.resource_key,
                width: shared.width,
                height: shared.height,
                byte_len: shared.byte_len,
                allocation_byte_len: shared.allocation_byte_len,
                adapter_luid,
            },
            run_ms,
        )))
    }

    /// Consume another NeoAMD session's resident NCHW FP16 output directly in
    /// HIP. v056 Backend Pack resolves the resource key to the producer's GPU
    /// pointer, avoiding NCHW->RGBA8->NCHW between stages.
    pub(crate) fn process_neoamd_peer_chain_input(
        &mut self,
        gc: &mut GlContext,
        input: NeoAmdPeerTensor,
    ) -> Result<Option<(NeoAmdPeerTensor, f64)>> {
        if !self.supports_neoamd_peer_chain() {
            return Ok(None);
        }
        let (w, h) = (input.width, input.height);
        let (adapter_luid, output) = {
            let session = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?;
            if session.adapter_luid() != input.adapter_luid {
                return Ok(None);
            }
            let adapter_luid = session.adapter_luid();
            let output = if self.temporal_frames.is_some() {
                session
                    .prepare_temporal_shared_io_fp16(w, h)
                    .map_err(anyhow::Error::msg)?
                    .output
            } else {
                session
                    .prepare_shared_io_fp16(w, h, 0.5)
                    .map_err(anyhow::Error::msg)?
                    .output
            };
            (adapter_luid, output)
        };
        self.neoamd_shared_key = Some(output.resource_key);
        self.neoamd_shared_input_size = Some((w, h));
        if gc.has_external_import(output.resource_key) {
            gc.wait_external_buffer_idle(output.resource_key)
                .map_err(anyhow::Error::msg)?;
        }
        let started = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .run_peer_shared_io_fp16(w, h, input.key, output.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms: 0.0,
            run_ms,
            out_ms: 0.0,
            output_size: (output.width as usize, output.height as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        Ok(Some((
            NeoAmdPeerTensor {
                key: output.resource_key,
                width: output.width,
                height: output.height,
                byte_len: output.byte_len,
                allocation_byte_len: output.allocation_byte_len,
                adapter_luid,
            },
            run_ms,
        )))
    }

    pub(crate) fn neoamd_peer_chain_to_texture(
        &mut self,
        gc: &mut GlContext,
        tensor: NeoAmdPeerTensor,
    ) -> Result<GpuTex> {
        anyhow::ensure!(
            self.neoamd_shared_key == Some(tensor.key),
            "NeoAMD peer-chain owner mismatch"
        );
        if gc.external_device_luid() != Some(tensor.adapter_luid.to_le_bytes()) {
            return Err(anyhow!("NeoAMD peer-chain presentation LUID mismatch"));
        }
        if !gc.has_external_import(tensor.key) {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_output_handle(tensor.key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                tensor.key,
                handle,
                tensor.byte_len,
                tensor.allocation_byte_len,
                tensor.adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }
        let out_started = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(tensor.key, tensor.width, tensor.height)
            .map_err(anyhow::Error::msg)?;
        let out_ms = out_started.elapsed().as_secs_f64() * 1000.0;
        if let Some(profile) = self.last_upscale_profile.as_mut() {
            profile.out_ms = out_ms;
        }
        Ok(texture)
    }

    fn process_neoamd_rgba8_gpu_output(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(i32, i32, GpuTex)>> {
        if !matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            || self.neoamd_shared_disabled
            || !self.fp16
            || self.interp != InterpKind::None
            || self.temporal_frames.is_some()
        {
            return Ok(None);
        }
        let pixels = usize::try_from(w)?
            .checked_mul(usize::try_from(h)?)
            .ok_or_else(|| anyhow!("NeoAMD shared input size overflow"))?;
        anyhow::ensure!(
            rgba.len() >= pixels * 4,
            "NeoAMD RGBA input buffer too small"
        );

        // A geometry change can make the Backend Pack replace its shareable
        // D3D12 resource. Retire the old GL import only after its read fence has
        // completed, before asking HIP to replace the backing resource.
        if self.neoamd_shared_input_size != Some((w, h)) {
            self.retire_neoamd_shared_output(gc);
        }

        let (adapter_luid, shared) = {
            let session = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session is unavailable"))?;
            if !session.supports_shared_fp16()
                || session.info().capabilities & NEOAMD_CAP_D3D12_SHARED_BUFFER == 0
            {
                return Ok(None);
            }
            let adapter_luid = session.adapter_luid();
            if gc.external_device_luid() != Some(adapter_luid.to_le_bytes()) {
                return Ok(None);
            }
            let shared = session
                .prepare_shared_fp16(w, h, 1, 4, 0.5)
                .map_err(anyhow::Error::msg)?;
            (adapter_luid, shared)
        };

        if let Some(old_key) = self.neoamd_shared_key {
            if old_key != shared.resource_key {
                gc.wait_external_buffer_idle(old_key)
                    .map_err(anyhow::Error::msg)?;
                gc.clear_external_buffer_key(old_key);
            }
        }
        self.neoamd_shared_key = Some(shared.resource_key);
        self.neoamd_shared_input_size = Some((w, h));

        let imported = gc.has_external_import(shared.resource_key);
        if !imported {
            let handle = self
                .neoamd_session
                .as_mut()
                .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
                .shared_output_handle(shared.resource_key)
                .map_err(anyhow::Error::msg)?;
            gc.import_external_d3d12_buffer(
                shared.resource_key,
                handle,
                shared.byte_len,
                shared.allocation_byte_len,
                adapter_luid.to_le_bytes(),
            )
            .map_err(anyhow::Error::msg)?;
        }
        // The GL read fence protects the resource from the next HIP overwrite.
        // Keep this timing separate from Backend execution so a live capture
        // slowdown cannot be incorrectly attributed to the HIP kernels.
        let reuse_wait_start = Instant::now();
        gc.wait_external_buffer_idle(shared.resource_key)
            .map_err(anyhow::Error::msg)?;
        let reuse_wait_ms = reuse_wait_start.elapsed().as_secs_f64() * 1000.0;

        let run_start = Instant::now();
        self.neoamd_session
            .as_mut()
            .ok_or_else(|| anyhow!("NeoAMD session disappeared"))?
            .run_u8_shared_fp16(w, h, &[rgba], 4, 0.5, shared.resource_key)
            .map_err(anyhow::Error::msg)?;
        let run_ms = run_start.elapsed().as_secs_f64() * 1000.0;
        let run_route = neoamd_run_route_label(
            self.neoamd_session
                .as_ref()
                .and_then(|session| session.last_run_route()),
        );
        let gpu_ms = self
            .neoamd_session
            .as_ref()
            .and_then(|session| session.last_image_gpu_ms())
            .map(f64::from);
        let copy_gpu_ms = self
            .neoamd_session
            .as_ref()
            .and_then(|session| session.last_image_copy_gpu_ms())
            .map(f64::from);
        let compute_gpu_ms = self
            .neoamd_session
            .as_ref()
            .and_then(|session| session.last_image_compute_gpu_ms())
            .map(f64::from);
        let host_sync_ms = gpu_ms.map(|gpu| (run_ms - gpu).max(0.0));

        let output_start = Instant::now();
        let texture = gc
            .external_nchw_f16_to_rgba8(shared.resource_key, shared.width, shared.height)
            .map_err(anyhow::Error::msg)?;
        let out_ms = output_start.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms: 0.0,
            run_ms,
            out_ms,
            output_size: (shared.width as usize, shared.height as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        // v791: the fused live path used to emit a DEBUG line every frame,
        // unlike the already-sampled shared-FP16 paths. With diagnostic file
        // logging enabled this adds avoidable CPU/file-I/O pressure precisely
        // to the hot NeoAMD lane we are measuring. Keep the same profile
        // content but use the common sampled cadence.
        if should_log_neoamd_profile() {
            log::debug!(
                "neoamd-profile: model='{}' ingress=rgba8-fused route={} run_ms={:.3} gpu_ms={} copy_gpu_ms={} compute_gpu_ms={} host_sync_ms={} reuse_wait_ms={:.3} out_ms={:.3} total_ms={:.3} input={}x{} output={}x{}",
                self.name,
                run_route,
                run_ms,
                gpu_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                copy_gpu_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                compute_gpu_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                host_sync_ms
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "n/a".into()),
                reuse_wait_ms,
                out_ms,
                reuse_wait_ms + run_ms + out_ms,
                w,
                h,
                shared.width,
                shared.height
            );
        }
        if !imported {
            log::info!(
                "NeoAMD GPU-resident output ready: model='{}' input={}x{} output={}x{} bytes={} key={} LUID={:016x} path=HIP->D3D12-shared->OpenGL",
                self.name,
                w,
                h,
                shared.width,
                shared.height,
                shared.byte_len,
                shared.resource_key,
                adapter_luid
            );
        }
        Ok(Some((shared.width, shared.height, texture)))
    }

    fn load_migraphx(path: &Path, fallback_dml_adapter: Option<i32>) -> Result<Self> {
        init_onnx().map_err(|e| anyhow!(e))?;
        log::info!(
            "NeoAccel provider probe: model={} requested=DirectML provider=MIGraphX precision=source-model-FP16 low_precision=calibration-pending cpu_fallback=false",
            path.display()
        );
        let session = crate::render::winml_migraphx::create_session(path)?;
        Self::from_session(
            path,
            session,
            fallback_dml_adapter,
            OnnxProvider::MigraphX,
            "MIGraphX/WindowsML GPU".into(),
            None,
            None,
            false,
        )
    }

    fn load_tensorrt(
        source_path: &Path,
        engine_model_path: &Path,
        device_id: i32,
        fallback_dml_adapter: Option<i32>,
        cache_root: &Path,
        explicit_qdq: bool,
    ) -> Result<Self> {
        Self::load_tensorrt_with_profile(
            source_path,
            engine_model_path,
            device_id,
            fallback_dml_adapter,
            cache_root,
            None,
            explicit_qdq,
        )
    }

    fn load_tensorrt_with_profile(
        source_path: &Path,
        engine_model_path: &Path,
        device_id: i32,
        fallback_dml_adapter: Option<i32>,
        cache_root: &Path,
        profile: Option<(&str, usize, i32, i32)>,
        explicit_qdq: bool,
    ) -> Result<Self> {
        init_onnx().map_err(|e| anyhow!(e))?;
        let builder_level = tensorrt_builder_level();
        let workspace_mb = tensorrt_workspace_mb();
        let profile_variant = profile
            .map(|(_, channels, height, width)| format!("_profile-{channels}x{height}x{width}"))
            .unwrap_or_default();
        let precision_variant = if explicit_qdq { "_qdq-int8-v1" } else { "" };
        let cumsum_workaround = tensorrt_temporalfix_cumsum_workaround(source_path)
            || tensorrt_temporalfix_cumsum_workaround(engine_model_path);
        let accuracy_variant = if cumsum_workaround {
            "_cumsum-cuda-v2-shapes"
        } else {
            ""
        };
        let cache_variant = format!(
            "builder-{builder_level}_workspace-{}mb_fp16-1{precision_variant}{profile_variant}{accuracy_variant}",
            workspace_mb
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".into())
        );
        let cache_dir =
            tensorrt_model_cache_dir(cache_root, engine_model_path, cache_variant.as_bytes());
        migrate_legacy_tensorrt_cache(
            cache_root,
            engine_model_path,
            cache_variant.as_bytes(),
            &cache_dir,
        );
        maintain_tensorrt_cache_root(cache_root, Some(&cache_dir));
        let cache_enabled = match std::fs::create_dir_all(&cache_dir) {
            Ok(()) => {
                touch_tensorrt_cache(&cache_dir);
                true
            }
            Err(error) => {
                log::warn!(
                    "tensorrt-cache: model={} disabled reason={error}",
                    engine_model_path.display()
                );
                false
            }
        };
        let stale_incomplete = cache_enabled && cache_dir.join(TENSORRT_BUILD_MARKER).exists();
        let cache_was_present =
            cache_enabled && !stale_incomplete && Self::has_tensorrt_engine_files(&cache_dir);
        let provider_verification_cached =
            cache_was_present && tensorrt_provider_verification_ready(&cache_dir);
        // A profile-less session is only used to inspect model metadata. The
        // actual runtime session is rebuilt with an exact single-input NCHW
        // profile before its first inference, so do not advertise or mark this
        // provisional metadata session as an engine build.
        if !cache_was_present && profile.is_some() {
            begin_tensorrt_build(source_path);
        }
        if cache_enabled && profile.is_some() {
            prepare_tensorrt_cache_attempt(&cache_dir, cache_was_present)?;
        }
        // Explicit Q/DQ controls INT8 placement inside the graph. Do not enable
        // TensorRT's legacy implicit-INT8 flag here: it requires calibration
        // metadata for non-Q/DQ models and was the reason the v688 probe failed.
        let mut tensorrt = ort::ep::TensorRT::default()
            .with_device_id(device_id)
            .with_fp16(true)
            .with_int8(false)
            .with_detailed_build_log(explicit_qdq)
            .with_force_sequential_engine_build(true)
            .with_builder_optimization_level(builder_level);
        if cumsum_workaround {
            // TensorRT 10.x can mix axes across CumSum layers. TemporalFix uses
            // them to construct sampling grids; corrupted grids produce NaNs
            // and a black right border. Keep convolutions in TensorRT, and run
            // only CumSum with the already-enabled CUDA execution provider.
            tensorrt = tensorrt.with_arbitrary_config("trt_op_types_to_exclude", "CumSum");
            log::info!(
                "tensorrt-cumsum-workaround: model={} excluded=CumSum provider=CUDA cache=cumsum-cuda-v2-shapes",
                engine_model_path.display()
            );
        }
        // CumSum partitioning introduces additional dynamic subgraph inputs.
        // ORT derives all their profiles from runtime tensors; a main-input-only
        // explicit profile would make these otherwise valid partitions fail.
        if let Some((input_name, channels, height, width)) = profile.filter(|_| !cumsum_workaround)
        {
            let shapes = format!("{input_name}:1x{channels}x{height}x{width}");
            tensorrt = tensorrt
                .with_profile_min_shapes(shapes.clone())
                .with_profile_opt_shapes(shapes.clone())
                .with_profile_max_shapes(shapes.clone());
            log::info!(
                "tensorrt-explicit-profile: model={} input={} shape=1x{}x{}x{} min=opt=max policy=runtime-padded-shape",
                engine_model_path.display(),
                input_name,
                channels,
                height,
                width
            );
        }
        if let Some(workspace_mb) = workspace_mb {
            tensorrt = tensorrt
                .with_max_workspace_size(workspace_mb.saturating_mul(1024).saturating_mul(1024));
        }
        log::info!(
            "tensorrt-builder-config: model={} source_model={} level={} workspace_mb={} fp16=true int8=false explicit_qdq={}",
            engine_model_path.display(),
            source_path.display(),
            builder_level,
            workspace_mb
                .map(|value| value.to_string())
                .unwrap_or_else(|| "default".into()),
            explicit_qdq
        );
        if explicit_qdq {
            log::info!(
                "NeoAccel explicit-QDQ: model={} sidecar={} provider=TensorRT precision=FP16+INT8-QDQ quantized_convs=Conv_10,Conv_12,Conv_14,Conv_16,Conv_18,Conv_20,Conv_22,Conv_24 activation=INT8-per-tensor weight=INT8-per-channel high_precision_output=true fallback=TensorRT-FP16",
                source_path.display(),
                engine_model_path.display()
            );
        }
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
                engine_model_path.display(),
                cache_dir.display()
            );
        }
        let cuda = ort::ep::CUDA::default().with_device_id(device_id);
        let profile_key = TENSORRT_PROFILE_KEY.fetch_add(1, Ordering::Relaxed);
        let profile_root = if provider_verification_cached {
            log::info!(
                "tensorrt-cache: provider verification reused model={} path={}",
                engine_model_path.display(),
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
        let staged_path = stage_tensorrt_model(engine_model_path, &cache_dir)?;
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
            source_path,
            session,
            fallback_dml_adapter,
            OnnxProvider::TensorRT,
            if explicit_qdq {
                format!("TensorRT FP16 + explicit-QDQ INT8(8 Conv) + CUDA device {device_id}")
            } else {
                format!("TensorRT FP16 + CUDA device {device_id}")
            },
            cache_enabled.then_some(cache_dir),
            diagnostic_guard,
            profile_prefix.is_some(),
        )
        .map(|mut stage| {
            stage.tensorrt_device_id = Some(device_id);
            stage.tensorrt_engine_verified = provider_verification_cached;
            stage.tensorrt_cache_root = Some(cache_root.to_path_buf());
            stage.tensorrt_profile_shape =
                profile.map(|(_, channels, height, width)| (channels, height, width));
            stage.tensorrt_model_path = Some(engine_model_path.to_path_buf());
            stage.tensorrt_explicit_qdq = explicit_qdq;
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
        let dml_rife_symbols =
            if provider == OnnxProvider::DirectML && matches!(&interp, InterpKind::RifeV1 { .. }) {
                match session.inputs()[0].dtype() {
                    ort::value::ValueType::Tensor {
                        dimension_symbols, ..
                    } if dimension_symbols.len() == 4
                        && !dimension_symbols[2].is_empty()
                        && !dimension_symbols[3].is_empty() =>
                    {
                        Some([
                            dimension_symbols[0].clone(),
                            dimension_symbols[2].clone(),
                            dimension_symbols[3].clone(),
                        ])
                    }
                    _ => None,
                }
            } else {
                None
            };
        let dml_image_symbols = if provider == OnnxProvider::DirectML
            && fp16
            && interp == InterpKind::None
            && session.inputs().len() == 1
            && session.outputs().len() == 1
            && parse_channels(&format!("{}", session.inputs()[0].dtype())) == Some(3)
            && parse_channels(&format!("{}", session.outputs()[0].dtype())) == Some(3)
            && (inspect_local_image_model(path)
                .ok()
                .flatten()
                .is_some_and(|plan| matches!(plan.conv_count, 10 | 18))
                || inspect_spanf3_image_model(path).unwrap_or(false)
                || inspect_rtmosr_image_model(path).unwrap_or(false)
                || std::env::var("NEO_DML_STRUCTURAL_PROBE").ok().as_deref() == Some("1"))
        {
            match session.inputs()[0].dtype() {
                ort::value::ValueType::Tensor {
                    dimension_symbols, ..
                } if dimension_symbols.len() == 4
                    && !dimension_symbols[2].is_empty()
                    && !dimension_symbols[3].is_empty() =>
                {
                    Some([
                        dimension_symbols[0].clone(),
                        dimension_symbols[2].clone(),
                        dimension_symbols[3].clone(),
                    ])
                }
                _ => None,
            }
        } else {
            None
        };
        let temporal_frames = input_channels
            .filter(|channels| *channels >= 9 && *channels % 3 == 0)
            .map(|channels| channels / 3);
        // v687: the v684-v686 temporal reuse experiment is intentionally retired.
        // NeoAccel now controls only TensorRT mixed precision at session-build
        // time; no frame synthesis, motion warp, or temporal detail reuse is
        // allowed in the active render path.
        let neo_accel = None;
        if let Some(frames) = temporal_frames {
            provider_desc.push_str(&format!(" + temporal-{frames}f"));
        }
        // v933: FraZerion exports that assume one source pair per batch (all phases in one call).
        // v934: opt-in only (CHIDESCALER_DML_PAIR_BATCH=on). The presentation clock is causal: the first
        // in-between is due when the pair arrives, so finishing all phases together delays that first
        // output by the whole batch (v933 measured ~76fps at 1080p x5 and ~110fps at 720p x5).
        let dml_pair_batch_capable = provider == OnnxProvider::DirectML
            && fp16
            && std::env::var("CHIDESCALER_DML_PAIR_BATCH").ok().as_deref() == Some("on")
            && session
                .metadata()
                .ok()
                .and_then(|meta| meta.custom("frazerion.pair_batch"))
                .as_deref()
                == Some("1");
        // DirectML GridSample writes at most 2^23 output pixels per call; the model warps full-size
        // images in `frazerion.warp_rows` row bands, which bounds how many phases one call may hold.
        let dml_pair_batch_rows = session
            .metadata()
            .ok()
            .and_then(|meta| meta.custom("frazerion.warp_rows"))
            .and_then(|rows| rows.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        if dml_pair_batch_capable {
            log::info!("frazerion-pair-batch: capable model={name} warp_rows={dml_pair_batch_rows}");
        }
        // v936: channels of the trunk -> head state (8 for V24, 12 with the 1/2-size stage of V25+)
        let dml_split_state_channels = session
            .metadata()
            .ok()
            .and_then(|meta| meta.custom("frazerion.state_channels"))
            .and_then(|c| c.parse::<usize>().ok())
            .filter(|c| (1..=64).contains(c))
            .unwrap_or(8);
        // v935: FraZerion trunk/head sidecars (see DmlSplit). CHIDESCALER_DML_SPLIT=off disables.
        let dml_split_paths = if provider == OnnxProvider::DirectML
            && fp16
            && std::env::var("CHIDESCALER_DML_SPLIT").ok().as_deref() != Some("off")
            && session
                .metadata()
                .ok()
                .and_then(|meta| meta.custom("frazerion.split"))
                .as_deref()
                == Some("1")
        {
            let trunk = PathBuf::from(format!("{}.trunk", path.display()));
            let head = PathBuf::from(format!("{}.head", path.display()));
            if trunk.is_file() && head.is_file() {
                log::info!("frazerion-split: capable model={name} state_channels={dml_split_state_channels}");
                Some((trunk, head))
            } else {
                None
            }
        } else {
            None
        };
        // v937: no sidecar files -> trunk/head embedded in the model itself (metadata frazerion.trunk_b64 /
        // frazerion.head_b64, base64 ONNX). Sidecar files, when present, keep priority (v935 behaviour).
        let dml_split_embedded = if dml_split_paths.is_none()
            && provider == OnnxProvider::DirectML
            && fp16
            && std::env::var("CHIDESCALER_DML_SPLIT").ok().as_deref() != Some("off")
        {
            let meta = session.metadata().ok();
            let split = meta.as_ref().and_then(|m| m.custom("frazerion.split")).as_deref() == Some("1");
            let trunk = meta.as_ref().and_then(|m| m.custom("frazerion.trunk_b64"));
            let head = meta.as_ref().and_then(|m| m.custom("frazerion.head_b64"));
            match (split, trunk, head) {
                (true, Some(t), Some(h)) => match (decode_base64(&t), decode_base64(&h)) {
                    (Some(t), Some(h)) => {
                        log::info!(
                            "frazerion-split: capable model={name} state_channels={dml_split_state_channels} source=embedded trunk={}B head={}B",
                            t.len(),
                            h.len()
                        );
                        Some(std::sync::Arc::new((t, h)))
                    }
                    _ => {
                        log::warn!("frazerion-split: embedded graphs unreadable model={name} action=single-model");
                        None
                    }
                },
                (true, _, _) => {
                    log::warn!("frazerion-split: sidecars missing model={name} action=single-model");
                    None
                }
                _ => None,
            }
        } else {
            None
        };
        let dml_rife_boost = provider == OnnxProvider::DirectML
            && input_channels == Some(7)
            && in_name == "input"
            && out_name == "output"
            && dml_split_paths.is_none()
            && dml_split_embedded.is_none()
            && std::env::var("CHIDESCALER_DML_SPLIT").ok().as_deref() != Some("off");
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
            dml_half_bridge_source: None,
            fallback_dml_adapter,
            session,
            dml_rife_symbols,
            dml_rife_shape: None,
            dml_host_interp_binding: None,
            dml_rife_cached_session: None,
            dml_rife_specialization_disabled: false,
            dml_rife_native_rejected: std::collections::HashSet::new(),
            dml_image_symbols,
            dml_wgl_converter: None,
            dml_wgl_disabled: false,
            dml_drba_sequential_logged: false,
            dml_wgl_last_geometry: None,
            dml_wgl_input_converter: None,
            dml_wgl_input_failed: None,
            dml_image_device_luid: None,
            dml_image_budget_retry: None,
            dml_image_dynamic_session: None,
            dml_image_cached_fixed_session: None,
            dml_image_shape: None,
            dml_image_specialization_disabled: false,
            run_options,
            tensorrt_diagnostic_dir,
            in_name,
            out_name,
            fp16,
            scratch_f32: Vec::new(),
            scratch_f16: Vec::new(),
            neoarc_pair_input: Vec::new(),
            neoarc_pair_reuse: false,
            neoarc_pair_packed: false,
            pack_const_key: None,
            pack_t_cached: None,
            pack_frame_key: None,
            pack_second_ptr: None,
            cross_gpu_rife_full_repack_logged: false,
            last_interp_profile: None,
            interp_gpu_diag: GpuInterpDiagnostics::default(),
            last_upscale_profile: None,
            image_diag_last_log: None,
            image_diag_samples: std::collections::VecDeque::new(),
            image_diag_shape: None,
            last_upscale_input_size: None,
            direct_output: None,
            direct_input: None,
            dml_temporal_bridge: None,
            direct_output_disabled: false,
            neoamd_shared_disabled: false,
            neoamd_shared_key: None,
            neoamd_interp_ring: Vec::new(),
            neoamd_interp_ring_base: None,
            neoamd_interp_ring_unavailable: false,
            neoamd_shared_input_key: None,
            neoamd_shared_input_size: None,
            tensorrt_device_id: None,
            neoamd_session: None,
            neoarc_session: None,
            neoarc_interp_gl: None,
            tensorrt_gpu_bridge: None,
            tensorrt_temporal_bridge: None,
            tensorrt_interp_bridge: None,
            dml_interp_bridge: None,
            dml_pair_batch: None,
            dml_pair_batch_capable,
            dml_pair_batch_rows,
            dml_pair_batch_failed: false,
            dml_pair_batch_logged: false,
            dml_pair_batch_skip_logged: false,
            dml_split: None,
            dml_split_paths,
            dml_split_embedded,
            dml_split_state_channels,
            dml_rife_boost,
            fused_host_boost: Vec::new(),
            fused_host_failed: std::env::var("CHIDESCALER_DML_FUSED_PAIR").as_deref()==Ok("off"),
            dml_host_boost: None,
            dml_split_failed: false,
            dml_split_logged: false,
            dml_wgl_interp_bridge: None,
            dml_wgl_interp_failed: false,
            dml_cpu_interp_shared_outputs: [Vec::new(), Vec::new()],
            dml_cpu_interp_shared_disabled: false,
            neoamd_drba_slow_streak: 0,
            interp_gpu_path_disabled: false,
            tensorrt_gpu_path_disabled: false,
            tensorrt_gpu_unsupported_logged: false,
            tensorrt_shape_seen: Default::default(),
            input_channels,
            neo_accel,
            temporal_frames,
            neoarc_temporal_packed: false,
            dml_temporal_cursor: None,
            dml_temporal_mirror: std::env::var("CHIDESCALER_DML_TEMPORAL_MIRROR").as_deref()!=Ok("off"),
            neoarc_gl_history_active: false,
            temporal_history: VecDeque::new(),
            temporal_history_stride: 0,
            temporal_size: None,
            neoamd_temporal_rejected_size: None,
            provider_desc,
            provider,
            fallback_reason: None,
            tensorrt_cache_dir,
            tensorrt_cache_root: None,
            tensorrt_profile_shape: None,
            tensorrt_engine_verified: false,
            tensorrt_profiling_active,
            tensorrt_model_path: None,
            tensorrt_explicit_qdq: false,
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
            && self.neoarc_session.is_none()
            && self.temporal_frames.is_some()
            && self.interp == InterpKind::None
    }

    pub fn supports_dml_gl_bridge(&self) -> bool {
        self.provider == OnnxProvider::DirectML && self.neoarc_session.is_none()
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
        if self.tensorrt_profile_shape.is_none() {
            log::debug!(
                "tensorrt-shape-cache: model={} input={}x{} action=defer-until-explicit-profile",
                self.name,
                width,
                height
            );
            return;
        }
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
                "neoaccel_explicit_qdq": self.tensorrt_explicit_qdq,
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

    fn replace_migraphx_with_directml(&mut self, reason: &str) -> Result<()> {
        anyhow::ensure!(
            self.provider == OnnxProvider::MigraphX,
            "MIGraphX fallback requested for a non-MIGraphX session"
        );
        let mut replacement = Self::load_directml(&self.source_path, self.fallback_dml_adapter)?;
        log::warn!(
            "NeoAccel AMD provider runtime fallback: model={} provider=MIGraphX reason={} action=DirectML-FP16",
            self.source_path.display(),
            reason
        );
        replacement.provider_desc.push_str(" after MIGraphX probe");
        *self = replacement;
        Ok(())
    }

    fn cleanup_tensorrt_diagnostic_dir(&mut self) {
        if let Some(mut diagnostic) = self.tensorrt_diagnostic_dir.take() {
            diagnostic.cleanup();
        }
    }

    pub fn finalize_provider_after_warmup(&mut self) {
        // Each successful stateless pass has already released its GL images.
        // Keep the validated executable and bindings across the remaining passes.
        // Shape/context changes are handled by prepare_gl_rgb before the next run.
        let warmed_image = self.temporal_frames.is_none() && self.interp == InterpKind::None
            && self.neoarc_session.as_ref().is_some_and(|s| s.gl_rgb_import_stats().is_some());
        if !warmed_image { self.retire_neoarc_gl(); }
        if let Some(session)=self.neoarc_session.as_mut(){session.reset_rife_validation();}
        // Warm-up frames are synthetic provider priming, not live temporal
        // history. Stop/Start begins with an empty TRUE-N window; live provider
        // switching must do the same while retaining warmed sessions/Graphs.
        if self.temporal_frames.is_some() {
            self.temporal_history.clear();
            self.neoarc_temporal_packed = false;
            self.temporal_history_stride = 0;
            self.temporal_size = None;
            if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
                if let Some(session) = self.neoamd_session.as_mut() {
                    if let Err(error) = session.reset_temporal_history() {
                        log::warn!(
                            "neoamd-temporal-warmup-history-reset-failed: model={} reason={} action=continue-cpu-history-cleared",
                            self.name,
                            error
                        );
                    } else {
                        log::debug!(
                            "neoamd-temporal-warmup-history-reset: model={} provider-state=warm history=empty",
                            self.name
                        );
                    }
                }
            }
        }
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
        self.dml_host_interp_binding = None;
        self.retire_neoarc_gl();
        if let Some(session)=self.neoarc_session.as_mut(){session.reset_rife_validation();}
        self.pack_const_key = None;
        self.pack_t_cached = None;
        self.pack_frame_key = None;
        self.pack_second_ptr = None;
        self.cross_gpu_rife_full_repack_logged = false;
        self.last_interp_profile = None;
        self.last_upscale_profile = None;
        self.last_upscale_input_size = None;
        self.temporal_history.clear();
        self.neoarc_temporal_packed = false;
        self.temporal_history_stride = 0;
        self.temporal_size = None;
        // Keep a TemporalFix runtime rejection scoped to its W/H across a
        // filter OFF -> ON cache reuse. Do not resurrect the experimental DML
        // temporal shared bridge merely because the chain generation changed.
        self.direct_output_disabled = self.temporal_frames.is_some();
        self.dml_wgl_disabled=false;
        self.dml_drba_sequential_logged=false;
        if self.dml_wgl_interp_failed{self.interp_gpu_path_disabled=false;self.dml_wgl_interp_failed=false;}
        self.dml_wgl_input_failed=None;
        self.dml_wgl_last_geometry=None;
        // An interop error is a per-generation condition. Explicit backend/model
        // transitions retire the old shared resource first, so allow the next
        // generation to probe GPU-resident NeoAMD again instead of permanently
        // poisoning the cached session.
        self.neoamd_shared_disabled = false;
        for bank in &mut self.dml_cpu_interp_shared_outputs {
            bank.clear();
        }
        self.dml_cpu_interp_shared_disabled = false;
        self.neoamd_drba_slow_streak = 0;
        self.tensorrt_gpu_path_disabled = false;
        self.tensorrt_gpu_unsupported_logged = false;
    }

    pub fn neoarc_precision_for_shape(&self,w:usize,h:usize)->Option<&'static str> {
        self.neoarc_session.as_ref().map(|session|{let(w,h)=if self.interp==InterpKind::Drba{((w+63)/64*64,(h+63)/64*64)}else{(w,h)};session.precision_for_shape(w,h)})
    }
    pub fn last_interp_profile(&self) -> Option<InterpProfile> {
        self.last_interp_profile
    }

    pub fn last_upscale_profile(&self) -> Option<UpscaleProfile> {
        self.last_upscale_profile
    }

    pub(crate) fn log_image_pipeline(&mut self, input_size: (i32, i32), input_bytes: usize,
        output_size: (i32, i32), host_output_bytes: usize, stage_wall_ms: f64) {
        if self.temporal_frames.is_some() || self.interp != InterpKind::None || !crate::logging::diagnostics_enabled(){return;}
        if self.provider==OnnxProvider::DirectML&&self.neoarc_session.is_none(){
            if self.image_diag_shape!=Some(input_size){self.image_diag_samples.clear();self.image_diag_shape=Some(input_size);}
            if let Some(p)=self.last_upscale_profile{
                if self.image_diag_samples.len()==120{self.image_diag_samples.pop_front();}
                self.image_diag_samples.push_back([p.pack_ms,p.run_ms,p.out_ms,stage_wall_ms]);
            }
        }
        if self.image_diag_last_log.is_some_and(|last| last.elapsed() < Duration::from_secs(1)) {
            return;
        }
        let first_sample = self.image_diag_last_log.is_none();
        self.image_diag_last_log = Some(Instant::now());
        if !self.image_diag_samples.is_empty()&&self.provider==OnnxProvider::DirectML&&self.neoarc_session.is_none(){
            let quantile=|column:usize,q:f64|{let mut v:Vec<f64>=self.image_diag_samples.iter().map(|r|r[column]).collect();v.sort_by(f64::total_cmp);v[((v.len()-1)as f64*q).round()as usize]};
            log::info!("directml-image-window: model={} input={}x{} samples={} fixed_shape={} total_p50_ms={:.3} total_p95_ms={:.3} total_max_ms={:.3} pack_p50_ms={:.3} infer_p50_ms={:.3} infer_p95_ms={:.3} output_p50_ms={:.3} timing=cpu-wall-with-transfers frame_omission=false",
                self.name,input_size.0,input_size.1,self.image_diag_samples.len(),self.dml_image_shape==Some(input_size),quantile(3,0.5),quantile(3,0.95),quantile(3,1.),quantile(0,0.5),quantile(1,0.5),quantile(1,0.95),quantile(2,0.5));
        }
        if let Some(profile) = self.last_upscale_profile {
            let accounted = profile.pack_ms + profile.run_ms + profile.out_ms;
            log::info!("sisr-pipeline: model='{}' provider={} input={}x{} output={}x{} first_sample={} pack_ms={:.3} infer_call_ms={:.3} output_convert_ms={:.3} stage_wall_ms={:.3} unattributed_wall_ms={:.3} input_host_bytes={} output_host_bytes={} capture_storage=host end_to_end_gpu_resident=false timing=cpu-wall infer_includes_transfer=true",
                self.name, self.metrics_provider_label(), input_size.0, input_size.1,
                output_size.0, output_size.1, first_sample, profile.pack_ms, profile.run_ms,
                profile.out_ms, stage_wall_ms, (stage_wall_ms - accounted).max(0.0),
                input_bytes, host_output_bytes);
        } else {
            log::info!("sisr-pipeline: model='{}' provider={} input={}x{} output={}x{} first_sample={} profile_valid=false stage_wall_ms={:.3} input_host_bytes={} output_host_bytes={} capture_storage=host end_to_end_gpu_resident=false timing=cpu-wall",
                self.name, self.metrics_provider_label(), input_size.0, input_size.1,
                output_size.0, output_size.1, first_sample, stage_wall_ms, input_bytes, host_output_bytes);
        }
    }

    pub(crate) fn reset_image_profile(&mut self) {
        // FilterChain clears per-frame telemetry before dispatch. Preserve the
        // last completed shape separately: telemetry is not route capability.
        if let (Some(input),Some(profile))=(self.last_upscale_input_size,self.last_upscale_profile){
            self.dml_wgl_last_geometry=Some((input,profile.output_size));
        }
        self.last_upscale_profile=None;
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

    /// v994: a NeoArc GL interpolation bridge will be built on the next pair
    /// (compile + validation, seconds on a first start). The engine keeps the
    /// overlay hidden meanwhile so the live source stays visible instead of
    /// a frozen first frame.
    pub fn neoarc_interp_bridge_pending(&self) -> bool {
        self.neoarc_session.is_some() && self.supports_interp_gpu() && self.neoarc_interp_gl.is_none()
    }

    pub fn supports_interp_gpu(&self) -> bool {
        if self.neoarc_session.is_some() {
            return !self.interp_gpu_path_disabled && matches!(self.interp,InterpKind::RifeV1{channels:7}|InterpKind::Drba)
                && std::env::var("CHIDESCALER_NEOARC_INTERP_GL").ok().as_deref()!=Some("off");
        }
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

    /// Use the existing owned-output cooperative slots on verified Arc DML
    /// CPU fallback sessions. Leave other adapters and shared routes alone.
    pub fn supports_arc_dml_cpu_stream(&self) -> bool {
        if self.provider != OnnxProvider::DirectML || !self.interp_gpu_path_disabled
            || self.neoarc_session.is_some()
            || !matches!(self.interp, InterpKind::RifeV1 { .. }) {
            return false;
        }
        match self.fallback_dml_adapter {
            Some(id) => intel_arc_device_id(id),
            None => {
                static SINGLE_ARC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *SINGLE_ARC.get_or_init(|| {
                    let adapters = crate::platform::gpu::enumerate_adapters_quiet();
                    adapters.len() == 1 && intel_arc_device_id(adapters[0].device_id)
                })
            }
        }
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
        self.prepare_interp_gpu_history(gc, frames, timesteps, generation, capacity_factor, stable_post_handoff, &[])
    }

    pub fn prepare_interp_gpu_history(
        &mut self,
        gc: &mut GlContext,
        frames: &[GpuTex],
        timesteps: &[f32],
        generation: u64,
        capacity_factor: usize,
        stable_post_handoff: bool,
        sequences: &[u64],
    ) -> Result<Option<u64>> {
        if !self.supports_interp_gpu() || timesteps.is_empty() {
            return Ok(None);
        }
        if self.neoarc_session.is_some() {
            match self.prepare_neoarc_interp_gl(gc,frames,timesteps,capacity_factor) {
                Ok(fence)=>{INTERP_GPU_INPUT_FRAMES.fetch_add(1,Ordering::Relaxed);return Ok(Some(fence));}
                Err(error)=>{
                    self.retire_neoarc_interp_gl(gc);self.interp_gpu_path_disabled=true;
                    log::warn!("neoarc-gl-interp-fallback: model={} reason={error:#} action=validated-rgb-worker",self.name);
                    INTERP_CPU_FALLBACK_FRAMES.fetch_add(1,Ordering::Relaxed);
                    if std::env::var("CHIDESCALER_INTERP_GPU").ok().as_deref()==Some("require") {
                        return Err(anyhow!("NeoArc GPU interpolation required but unavailable: {error:#}"));
                    }
                    return Ok(None);
                }
            }
        }

        if self.supports_dml_wgl_interp()&&gc.external_device_luid().is_none()&&crate::render::vulkan_gpu::production_selected_luid().is_none(){
            return match self.prepare_dml_wgl_interp(gc,frames,timesteps,capacity_factor,generation,sequences){
                Ok(fence)=>{INTERP_GPU_INPUT_FRAMES.fetch_add(1,Ordering::Relaxed);Ok(Some(fence))},
                Err(error)=>{self.retire_dml_wgl_interp(gc);self.dml_wgl_interp_failed=true;self.interp_gpu_path_disabled=true;INTERP_CPU_FALLBACK_FRAMES.fetch_add(1,Ordering::Relaxed);log::warn!("directml-wgl-interp-fallback: model={} reason={error:#} action=original-host-path",self.name);Ok(None)}
            };
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
                    self.interp_gpu_path_disabled = true;
                    self.reset_interp_pack_cache();
                    log::info!(
                        "dml-interp-interop-policy: mode=nonresident reason=explicit-safe-fallback action=disable-gpu-path-for-session"
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
            OnnxProvider::MigraphX
            | OnnxProvider::Cuda
            | OnnxProvider::NeoAMD
            | OnnxProvider::NeoRTX => Ok(None),
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
        if self.dml_wgl_interp_bridge.is_some(){return self.run_dml_wgl_interp_slot(index);}
        if let Some(bridge)=self.neoarc_interp_gl.as_mut() {
            let phase=*bridge.phases.get(index).ok_or_else(||anyhow!("NeoArc shared phase missing"))?;
            let output=bridge.outputs.get(index).ok_or_else(||anyhow!("NeoArc shared output missing"))?.1;
            let times=bridge.io.run(&bridge.frames,output.tex.0.get(),phase)?;
            self.last_interp_profile=Some(InterpProfile{pack_ms:times[0],run_ms:times[1],out_ms:times[2],padded_size:(bridge.padded.0 as usize,bridge.padded.1 as usize)});
            log::debug!("neoarc-gl-interp-profile: model={} input={}x{} slot={} t={} pack_ms={:.3} run_ms={:.3} output_ms={:.3} host_input_bytes=0 host_output_bytes=0",self.name,bridge.size.0,bridge.size.1,index,phase,times[0],times[1],times[2]);
            let wall=times.iter().sum();let padded=bridge.padded;let size=bridge.size;
            self.interp_gpu_diag.record(&self.name,"NeoARC",size,padded,[times[0],times[1],times[2],wall]);
            return Ok((wall,padded));
        }
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
                if let Some(bridge) = self.dml_interp_bridge.as_mut() {
                    bridge.slots_done = bridge.slots_done.max(index + 1);
                }
                if let Some(wall_ms) = self.try_dml_pair_batch(index) {
                    let padded = self.dml_interp_bridge.as_ref().map(|b| b.padded_size).unwrap_or((0, 0));
                    return Ok((wall_ms, padded));
                }
                if let Some(wall_ms) = self.try_dml_split(index) {
                    let padded = self.dml_interp_bridge.as_ref().map(|b| b.padded_size).unwrap_or((0, 0));
                    return Ok((wall_ms, padded));
                }
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
            OnnxProvider::MigraphX
            | OnnxProvider::Cuda
            | OnnxProvider::NeoAMD
            | OnnxProvider::NeoRTX => {
                anyhow::bail!("unsupported interpolation backend")
            }
        };
        Ok((started.elapsed().as_secs_f64() * 1000.0, padded))
    }

    /// v933: FraZerion pair batch (see `DmlPairBatch`). Some(wall_ms) when this slot is served by
    /// the batch (slot 0 runs all phases, later slots are already done); None takes the
    /// established per-phase path unchanged.
    fn try_dml_pair_batch(&mut self, index: usize) -> Option<f64> {
        if !self.dml_pair_batch_capable || self.dml_pair_batch_failed || !self.fp16 {
            return None;
        }
        let bridge = self.dml_interp_bridge.as_ref()?;
        if bridge.interop_mode != DmlInterpInteropMode::ConservativeCopy || bridge.prepared_count < 2 {
            return None;
        }
        // each phase warps two full-size images; one GridSample call (a row band) holds <= 2^23 pixels
        let (pw, ph) = bridge.padded_size;
        let band_pixels = 2 * bridge.prepared_count * pw.max(0) as usize * ph.max(0) as usize
            / self.dml_pair_batch_rows;
        if band_pixels > 1 << 23 {
            if !self.dml_pair_batch_skip_logged {
                self.dml_pair_batch_skip_logged = true;
                log::info!(
                    "frazerion-pair-batch: skipped model={} phases={} padded={}x{} reason=gridsample-limit action=per-phase-directml",
                    self.name,
                    bridge.prepared_count,
                    pw,
                    ph
                );
            }
            return None;
        }
        let serial = bridge.prepare_serial;
        if index > 0 {
            let batch = self.dml_pair_batch.as_ref()?;
            return (batch.done_serial == serial && index < batch.done_count).then_some(0.0);
        }
        match self.run_dml_pair_batch() {
            Ok(wall_ms) => Some(wall_ms),
            Err(error) => {
                self.dml_pair_batch_failed = true;
                self.dml_pair_batch = None;
                log::warn!(
                    "frazerion-pair-batch-fallback: model={} reason={error:#} action=per-phase-directml",
                    self.name
                );
                None
            }
        }
    }

    fn build_dml_pair_batch(
        &mut self,
        padded: (i32, i32),
        count: usize,
        channels: usize,
    ) -> Result<DmlPairBatch> {
        let symbols = self
            .dml_rife_symbols
            .clone()
            .ok_or_else(|| anyhow!("model has no dynamic input shape"))?;
        anyhow::ensure!(!symbols[0].is_empty(), "model batch dimension is fixed");
        let (device, luid) = {
            let bridge = self
                .dml_interp_bridge
                .as_ref()
                .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
            let output = &bridge
                .invocations
                .first()
                .ok_or_else(|| anyhow!("DirectML interpolation slots missing"))?
                .output;
            (output.device.clone(), output.luid)
        };
        let directml = match self.fallback_dml_adapter {
            Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
            None => ort::ep::DirectML::default().with_performance_preference(
                ort::ep::directml::PerformancePreference::HighPerformance,
            ),
        };
        let session = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(dml_graph_optimization(&self.source_path))
            .map_err(oerr)?
            .with_execution_providers([directml.build()])
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?
            .with_dimension_override(&symbols[0], count as i64)
            .map_err(oerr)?
            .with_dimension_override(&symbols[1], i64::from(padded.1))
            .map_err(oerr)?
            .with_dimension_override(&symbols[2], i64::from(padded.0))
            .map_err(oerr)?
            .commit_from_file(&self.source_path)
            .map_err(oerr)?;
        let mut input = create_direct_input_channels_typed_state(
            &device,
            luid,
            padded,
            channels * count,
            std::mem::size_of::<f16>(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        input.dims = [count as i64, channels as i64, padded.1 as i64, padded.0 as i64];
        let memory = MemoryInfo::new(
            AllocationDevice::DIRECTML,
            0,
            AllocatorType::Device,
            MemoryType::Default,
        )?;
        let input_value = unsafe {
            TensorRefMut::<f16>::from_raw(memory.clone(), input.allocation, Shape::new(input.dims))?
        };
        let mut binding = session.create_binding()?;
        let input_name = CString::new(self.in_name.as_bytes())?;
        ort_status(unsafe {
            (ort::api().BindInput)(binding.ptr().cast_mut(), input_name.as_ptr(), input_value.ptr())
        })?;
        binding
            .bind_output_to_device(self.out_name.clone(), &memory)
            .map_err(oerr)?;
        let allocator = Allocator::new(&session, memory)?;
        let copy = DmlCopyContext::new(&device)?;
        Ok(DmlPairBatch {
            key: (padded, count),
            session,
            _input_value: input_value,
            _input_name: input_name,
            binding,
            allocator,
            copy,
            input,
            done_serial: 0,
            done_count: 0,
        })
    }

    fn run_dml_pair_batch(&mut self) -> Result<f64> {
        let started = Instant::now();
        let (padded, count, channels) = {
            let bridge = self
                .dml_interp_bridge
                .as_ref()
                .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
            (bridge.padded_size, bridge.prepared_count, bridge.channels)
        };
        if self.dml_pair_batch.as_ref().is_none_or(|batch| batch.key != (padded, count)) {
            self.dml_pair_batch = None;
            let built = self.build_dml_pair_batch(padded, count, channels)?;
            log::info!(
                "frazerion-pair-batch: active model={} phases={} padded={}x{} build_ms={:.0}",
                self.name,
                count,
                padded.0,
                padded.1,
                started.elapsed().as_secs_f64() * 1000.0
            );
            self.dml_pair_batch = Some(built);
        }
        let bridge = self.dml_interp_bridge.as_ref().unwrap();
        let batch = self.dml_pair_batch.as_mut().unwrap();
        // 1. the packed phase inputs -> one batch tensor (GPU copies)
        let slice_in = batch.input.byte_len / count;
        let mut sources = Vec::with_capacity(count);
        let mut regions = Vec::with_capacity(count);
        for (i, invocation) in bridge.invocations.iter().take(count).enumerate() {
            let safe = invocation
                .safe_input
                .as_ref()
                .ok_or_else(|| anyhow!("DirectML conservative input buffer missing"))?;
            anyhow::ensure!(safe.byte_len <= slice_in, "pair batch input slice too small");
            sources.push((safe.resource.clone(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
            regions.push((i, 0u64, 0usize, (i * slice_in) as u64, safe.byte_len as u64));
        }
        let dests = [(batch.input.resource.clone(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS)];
        batch.copy.copy_regions_and_wait(&sources, &dests, &regions)?;
        // 2. one provider call for all phases
        let run_started = Instant::now();
        let mut outputs = batch
            .session
            .run_binding_with_options(&batch.binding, self.run_options.armed()?)
            .map_err(oerr)?;
        batch.binding.synchronize_outputs().map_err(oerr)?;
        let run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
        let provider = outputs
            .remove(self.out_name.as_str())
            .ok_or_else(|| anyhow!("pair batch output missing"))?
            .downcast::<TensorValueType<f16>>()?;
        anyhow::ensure!(
            provider.memory_info().allocation_device().as_str() == "DML",
            "pair batch output silently fell back to CPU"
        );
        let dims: Vec<i64> = provider.shape().iter().copied().collect();
        anyhow::ensure!(
            dims == vec![count as i64, 3, padded.1 as i64, padded.0 as i64],
            "pair batch output shape {dims:?}"
        );
        let api = dml_api()?;
        let mut raw_resource = std::ptr::null_mut();
        ort_status(unsafe {
            (api.GetD3D12ResourceFromAllocation)(
                batch.allocator.ptr().cast_mut(),
                provider.data_ptr().cast_mut(),
                &mut raw_resource,
            )
        })?;
        let provider_resource = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
            .ok_or_else(|| anyhow!("pair batch output resource is null"))?
            .clone();
        // 3. each phase's slice -> that phase's GL-shared output buffer (as the per-phase path)
        let slice_out = 3 * usize::try_from(padded.0)? * usize::try_from(padded.1)? * std::mem::size_of::<f16>();
        let mut dests = Vec::with_capacity(count);
        let mut regions = Vec::with_capacity(count);
        for (i, invocation) in bridge.invocations.iter().take(count).enumerate() {
            anyhow::ensure!(invocation.output.byte_len >= slice_out, "pair batch output slot too small");
            dests.push((invocation.output.resource.clone(), D3D12_RESOURCE_STATE_COMMON));
            regions.push((0usize, (i * slice_out) as u64, i, 0u64, slice_out as u64));
        }
        batch.copy.copy_regions_and_wait(
            &[(provider_resource, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
            &dests,
            &regions,
        )?;
        drop(provider);
        drop(outputs);
        batch.done_serial = bridge.prepare_serial;
        batch.done_count = count;
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.last_interp_profile = Some(InterpProfile {
            pack_ms: 0.0,
            run_ms,
            out_ms: wall_ms - run_ms,
            padded_size: (padded.0 as usize, padded.1 as usize),
        });
        if !self.dml_pair_batch_logged {
            self.dml_pair_batch_logged = true;
            log::info!(
                "frazerion-pair-batch: first pair model={} phases={} run_ms={:.2} wall_ms={:.2}",
                self.name,
                count,
                run_ms,
                wall_ms
            );
        }
        Ok(wall_ms)
    }

    fn try_dml_split(&mut self, index: usize) -> Option<f64> {
        let frazerion = self.fp16
            && (self.dml_split_paths.is_some() || self.dml_split_embedded.is_some());
        if self.dml_split_failed || !(self.dml_rife_boost || frazerion) {
            return None;
        }
        let bridge = self.dml_interp_bridge.as_ref()?;
        if bridge.interop_mode != DmlInterpInteropMode::ConservativeCopy
            || (self.dml_rife_boost && !bridge.rife_boost_for_pair)
        {
            return None;
        }
        match self.run_dml_split(index) {
            Ok(wall_ms) => Some(wall_ms),
            Err(error) => {
                if self.dml_rife_boost && onnx_cancel_requested() {
                    return None;
                }
                self.dml_split_failed = true;
                self.dml_split = None;
                log::warn!(
                    "{}-split-fallback: model={} reason={error:#} action=single-model-directml",
                    self.dml_split_label(),
                    self.name
                );
                None
            }
        }
    }

    fn dml_split_label(&self) -> &'static str {
        if self.dml_rife_boost { "rife-boost" } else { "frazerion" }
    }

    fn dml_sidecar_session(&self, path: &Path, dims: &[(&str, i64)]) -> Result<Session> {
        self.dml_split_session(Ok(path), dims)
    }

    /// v937: one builder for both sources: a sidecar file (v935) or embedded ONNX bytes (one-file models).
    /// Embedded graphs carry no weights of their own: their initializers are external data in the virtual file
    /// "neo_main.bin" = the model file's own bytes (offset/length), so the one-file model stays ~3% bigger
    /// instead of doubling. ORT copies the weights during session creation; the buffer is freed afterwards.
    fn dml_split_session(
        &self,
        source: std::result::Result<&Path, (&[u8], Option<Vec<u8>>)>,
        dims: &[(&str, i64)],
    ) -> Result<Session> {
        let directml = match self.fallback_dml_adapter {
            Some(device_id) => ort::ep::DirectML::default().with_device_id(device_id),
            None => ort::ep::DirectML::default().with_performance_preference(
                ort::ep::directml::PerformancePreference::HighPerformance,
            ),
        };
        let mut builder = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(dml_graph_optimization(&self.source_path))
            .map_err(oerr)?
            .with_execution_providers([directml.build()])
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?;
        for (name, value) in dims {
            builder = builder.with_dimension_override(name, *value).map_err(oerr)?;
        }
        match source {
            Ok(path) => builder.commit_from_file(path).map_err(oerr),
            Err((bytes, main)) => {
                if let Some(main) = main {
                    builder = builder
                        .with_external_initializer_file_in_memory("neo_main.bin", std::borrow::Cow::Owned(main))
                        .map_err(oerr)?;
                }
                builder.commit_from_memory(bytes).map_err(oerr)
            }
        }
    }

    fn build_dml_split(&mut self, padded: (i32, i32)) -> Result<DmlSplit> {
        let (h, w) = (i64::from(padded.1), i64::from(padded.0));
        let trunk_dims = [("height", h), ("width", w)];
        let head_dims = [("batch", 1), ("height", h), ("width", w)];
        let (trunk, head) = if self.dml_rife_boost {
            let (trunk, head, _) = self.dml_split_graphs(h, w)?;
            (
                self.dml_split_session(Err((&trunk, None)), &[])?,
                self.dml_split_session(Err((&head, None)), &[])?,
            )
        } else if let Some((trunk_path, head_path)) = self.dml_split_paths.clone() {
            (
                self.dml_sidecar_session(&trunk_path, &trunk_dims)?,
                self.dml_sidecar_session(&head_path, &head_dims)?,
            )
        } else {
            let embedded = self
                .dml_split_embedded
                .clone()
                .ok_or_else(|| anyhow!("split graphs missing"))?;
            let main = std::fs::read(&self.source_path)
                .map_err(|e| anyhow!("model reread failed {}: {e}", self.source_path.display()))?;
            (
                self.dml_split_session(Err((embedded.0.as_slice(), Some(main.clone()))), &trunk_dims)?,
                self.dml_split_session(Err((embedded.1.as_slice(), Some(main))), &head_dims)?,
            )
        };
        let bridge = self
            .dml_interp_bridge
            .as_ref()
            .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
        let first = &bridge
            .invocations
            .first()
            .ok_or_else(|| anyhow!("DirectML interpolation slots missing"))?
            .output;
        let (device, luid) = (first.device.clone(), first.luid);
        let channels = if self.dml_rife_boost { 5 } else { self.dml_split_state_channels };
        let mut state = create_direct_input_channels_typed_state(
            &device,
            luid,
            padded,
            channels,
            if self.fp16 { std::mem::size_of::<f16>() } else { std::mem::size_of::<f32>() },
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        state.dims = [1, channels as i64, h, w];
        let memory = MemoryInfo::new(AllocationDevice::DIRECTML, 0, AllocatorType::Device, MemoryType::Default)?;
        let tensor = |allocation, dims| -> Result<InterpTensorValue> {
            let shape = Shape::new(dims);
            Ok(unsafe {
                if self.fp16 {
                    InterpTensorValue::F16(TensorRefMut::from_raw(memory.clone(), allocation, shape)?)
                } else {
                    InterpTensorValue::F32(TensorRefMut::from_raw(memory.clone(), allocation, shape)?)
                }
            })
        };
        let state_value = tensor(state.allocation, state.dims)?;
        let names: Vec<CString> = ["input", "state", "output"]
            .iter()
            .map(|n| CString::new(*n))
            .collect::<std::result::Result<_, _>>()?;
        let mut input_values = Vec::new();
        let mut trunk_bindings = Vec::new();
        let mut head_bindings = Vec::new();
        let mut key_ptrs = Vec::new();
        for invocation in &bridge.invocations {
            let safe = invocation
                .safe_input
                .as_ref()
                .ok_or_else(|| anyhow!("DirectML conservative input buffer missing"))?;
            anyhow::ensure!(safe.dims == [1, 7, h, w], "phase input shape {:?}", safe.dims);
            key_ptrs.push(safe.allocation as usize);
            let value = tensor(safe.allocation, safe.dims)?;
            let trunk_binding = trunk.create_binding()?;
            let head_binding = head.create_binding()?;
            ort_status(unsafe { (ort::api().BindInput)(trunk_binding.ptr().cast_mut(), names[0].as_ptr(), value.ptr()) })?;
            ort_status(unsafe { (ort::api().BindOutput)(trunk_binding.ptr().cast_mut(), names[1].as_ptr(), state_value.ptr()) })?;
            ort_status(unsafe { (ort::api().BindInput)(head_binding.ptr().cast_mut(), names[0].as_ptr(), value.ptr()) })?;
            ort_status(unsafe { (ort::api().BindInput)(head_binding.ptr().cast_mut(), names[1].as_ptr(), state_value.ptr()) })?;
            let mut head_binding = head_binding;
            head_binding.bind_output_to_device("output", &memory).map_err(oerr)?;
            input_values.push(value);
            trunk_bindings.push(trunk_binding);
            head_bindings.push(head_binding);
        }
        let head_allocator = Allocator::new(&head, memory)?;
        Ok(DmlSplit {
            key: (padded, key_ptrs),
            trunk,
            head,
            trunk_bindings,
            head_bindings,
            _input_values: input_values,
            _state_value: state_value,
            _names: names,
            state,
            head_allocator,
            trunk_serial: None,
            midpair_trunks: 0,
        })
    }

    /// Build (or reuse) the trunk/head sessions for the current GPU bridge.
    fn ensure_dml_split(&mut self) -> Result<()> {
        let started = Instant::now();
        let (padded, ptrs) = {
            let bridge = self
                .dml_interp_bridge
                .as_ref()
                .ok_or_else(|| anyhow!("DirectML interpolation bridge missing"))?;
            let ptrs = bridge
                .invocations
                .iter()
                .map(|inv| inv.safe_input.as_ref().map(|s| s.allocation as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| anyhow!("DirectML conservative input buffer missing"))?;
            (bridge.padded_size, ptrs)
        };
        if self.dml_split.as_ref().is_none_or(|split| split.key != (padded, ptrs.clone())) {
            self.dml_split = None;
            let built = self.build_dml_split(padded)?;
            log::info!(
                "{}-split: active model={} slots={} padded={}x{} build_ms={:.0}",
                self.dml_split_label(),
                self.name,
                ptrs.len(),
                padded.0,
                padded.1,
                started.elapsed().as_secs_f64() * 1000.0
            );
            self.dml_split = Some(built);
        }
        Ok(())
    }

    fn run_dml_split(&mut self, index: usize) -> Result<f64> {
        let started = Instant::now();
        self.ensure_dml_split()?;
        let padded = self.dml_interp_bridge.as_ref().map(|b| b.padded_size).unwrap_or((0, 0));
        let label = self.dml_split_label();
        let bridge = self.dml_interp_bridge.as_mut().unwrap();
        let split = self.dml_split.as_mut().unwrap();
        anyhow::ensure!(index < split.head_bindings.len(), "split slot {index} missing");
        // 1. the trunk, once per prepared source pair (it ignores t, any phase input will do)
        let mut trunk_ms = 0.0;
        // v949: slot 0 always starts a new source pair, so it always refreshes the
        // shared state. Relying on prepare_serial alone let a pair reuse the
        // previous pair's motion when the next prepare interleaved with the
        // cooperative worker, which presented frames out of time order.
        let serial_changed = split.trunk_serial != Some(bridge.prepare_serial);
        if serial_changed && index != 0 {
            split.midpair_trunks = split.midpair_trunks.saturating_add(1);
            if split.midpair_trunks <= 8 || split.midpair_trunks % 240 == 0 {
                log::warn!(
                    "{label}-split: trunk-midpair model={} slot={} prepare_serial={} count={}",
                    self.name,
                    index + 1,
                    bridge.prepare_serial,
                    split.midpair_trunks
                );
            }
        }
        if index == 0 || serial_changed {
            let t0 = Instant::now();
            split
                .trunk
                .run_binding_with_options(&split.trunk_bindings[index], self.run_options.armed()?)
                .map_err(oerr)?;
            split.trunk_bindings[index].synchronize_outputs().map_err(oerr)?;
            split.trunk_serial = Some(bridge.prepare_serial);
            trunk_ms = t0.elapsed().as_secs_f64() * 1000.0;
        }
        // 2. the head for this phase -> the phase's GL-shared output buffer (as the per-phase path)
        let t1 = Instant::now();
        let mut outputs = split
            .head
            .run_binding_with_options(&split.head_bindings[index], self.run_options.armed()?)
            .map_err(oerr)?;
        split.head_bindings[index].synchronize_outputs().map_err(oerr)?;
        let head_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let provider = outputs
            .remove("output")
            .ok_or_else(|| anyhow!("split head output missing"))?;
        let provider = if self.fp16 {
            provider.downcast::<TensorValueType<f16>>()?.upcast()
        } else {
            provider.downcast::<TensorValueType<f32>>()?.upcast()
        };
        anyhow::ensure!(
            provider.memory_info().allocation_device().as_str() == "DML",
            "split head output silently fell back to CPU"
        );
        let invocation = bridge
            .invocations
            .get_mut(index)
            .ok_or_else(|| anyhow!("DirectML interpolation slot {index} missing"))?;
        let dims: Vec<i64> = provider.shape().iter().copied().collect();
        anyhow::ensure!(dims == invocation.output.dims, "split head output shape {dims:?} != {:?}", invocation.output.dims);
        let api = dml_api()?;
        let mut raw_resource = std::ptr::null_mut();
        ort_status(unsafe {
            (api.GetD3D12ResourceFromAllocation)(
                split.head_allocator.ptr().cast_mut(),
                provider.data_ptr().cast_mut(),
                &mut raw_resource,
            )
        })?;
        let provider_resource = unsafe { ID3D12Resource::from_raw_borrowed(&raw_resource) }
            .ok_or_else(|| anyhow!("split head output resource is null"))?
            .clone();
        let _ = invocation.output.copy.copy_and_wait(
            &provider_resource,
            &invocation.output.resource,
            invocation.output.byte_len,
        )?;
        drop(provider);
        drop(outputs);
        let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.last_interp_profile = Some(InterpProfile {
            pack_ms: 0.0,
            run_ms: trunk_ms + head_ms,
            out_ms: wall_ms - trunk_ms - head_ms,
            padded_size: (padded.0 as usize, padded.1 as usize),
        });
        if !self.dml_split_logged && trunk_ms == 0.0 {
            self.dml_split_logged = true;
            log::info!(
                "{label}-split: first head-only phase model={} slot={} head_ms={:.2} wall_ms={:.2}",
                self.name,
                index + 1,
                head_ms,
                wall_ms
            );
        } else if trunk_ms > 0.0 && !self.dml_split_logged {
            log::info!(
                "{label}-split: trunk model={} slot={} trunk_ms={:.2} head_ms={:.2} wall_ms={:.2}",
                self.name,
                index + 1,
                trunk_ms,
                head_ms,
                wall_ms
            );
        }
        Ok(wall_ms)
    }

    pub(crate) fn prepared_interp_gpu_outputs(
        &self,
        count: usize,
        preserve_float_after_dml: bool,
    ) -> Result<Vec<PreparedInterpGpuOutput>> {
        if let Some(bridge)=self.dml_wgl_interp_bridge.as_ref(){anyhow::ensure!(count<=bridge.slots.len(),"DML WGL output capacity");return Ok(bridge.slots.iter().take(count).map(|slot|PreparedInterpGpuOutput{key:slot.key,texture:None,size:bridge.size,padded:bridge.padded,fp16:true,directml_drba_snapshot_candidate:false,preserve_float:false}).collect());}
        if let Some(bridge)=self.neoarc_interp_gl.as_ref() {
            anyhow::ensure!(count<=bridge.outputs.len(),"NeoArc shared output capacity exceeded");
            return Ok(bridge.outputs.iter().take(count).map(|(_,tex)|PreparedInterpGpuOutput{
                key:0,texture:Some(*tex),size:bridge.size,padded:bridge.padded,fp16:true,
                directml_drba_snapshot_candidate:false,preserve_float:false,
            }).collect());
        }
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
            OnnxProvider::MigraphX
            | OnnxProvider::Cuda
            | OnnxProvider::NeoAMD
            | OnnxProvider::NeoRTX => {
                anyhow::bail!("unsupported interpolation backend")
            }
        };
        anyhow::ensure!(
            keys.len() == count,
            "interpolation output slots are incomplete"
        );
        Ok(keys
            .into_iter()
            .map(|key| PreparedInterpGpuOutput {
                texture: None,
                key,
                size,
                padded,
                fp16: self.fp16,
                directml_drba_snapshot_candidate: matches!(self.provider, OnnxProvider::DirectML)
                    && self.interp == InterpKind::Drba,
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
        if !matches!(self.provider, OnnxProvider::DirectML)||output.key&DML_WGL_INTERP_KEY!=0 {
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
        if output.key&DML_WGL_INTERP_KEY!=0{
            let full=gc.copy_dml_interp_image(output.key,output.padded.0,output.padded.1)?;
            let texture=if output.size==output.padded{full}else{super::scaler::crop_edge(gc,full,output.size.0,output.size.1)?};gc.finish();INTERP_GPU_OUTPUT_FRAMES.fetch_add(1,Ordering::Relaxed);return Ok(texture);
        }
        if let Some(texture)=output.texture {
            // Return a detached GL image; provider slots may be reused by a later pair.
            let result=if output.size==output.padded{super::scaler::detach_identity(gc,texture)}else{super::scaler::crop_edge(gc,texture,output.size.0,output.size.1)};
            if result.is_ok(){
                // Arc CL/GL outputs cross a second ownership boundary when GL
                // detaches the provider slot. Complete that copy before a later
                // provider invocation or post chain can recycle either resource.
                gc.finish();
                INTERP_GPU_OUTPUT_FRAMES.fetch_add(1,Ordering::Relaxed);
            }
            return result;
        }
        let use_drba_local_snapshot = output.directml_drba_snapshot_candidate
            && output.fp16
            && !output.preserve_float
            && gc.external_import_uses_bound_storage_fallback(output.key);
        let texture = if use_drba_local_snapshot {
            gc.external_nchw_f16_to_rgba8_crop_local_snapshot(
                output.key,
                output.size.0,
                output.size.1,
                output.padded.0,
                output.padded.1,
            )
        } else {
            match (output.fp16, output.preserve_float) {
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
            5 => 7,
            _ => factor.saturating_add(2), // experimental x6..x15 only
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
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            match self.run_neoamd_u8(w, h, frames, 3, 3, t) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!(
                        "neoamd-interp-retry-directml: model={} reason={error:#}",
                        self.name
                    );
                }
            }
        }
        // Backend-switch warmup can enter this CPU-visible path before the
        // resident interpolation bridge. Build the same explicit profile here
        // so TensorRT never sees a profile-less first inference.
        self.ensure_tensorrt_interp_profile_for_size((w, h))?;
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
            return result;
        }
        if let Err(error) = &result
            && self.fallback_tensorrt_after_engine_creation_failure(error)?
        {
            return self.process_interp(w, h, frames, t);
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
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            match self.run_neoamd_u8(w, h, frames, 4, 3, t) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!(
                        "neoamd-interp-rgba-retry-directml: model={} reason={error:#}",
                        self.name
                    );
                }
            }
        }
        self.ensure_tensorrt_interp_profile_for_size((w, h))?;
        self.prepare_tensorrt_input_shape(w, h);
        mark_tensorrt_model_started(&self.name);
        let result = if let InterpKind::RifeV1 { channels } = self.interp.clone() {
            self.rife_multi_strided(w, h, frames, channels, t, 4, false)
        } else if self.interp == InterpKind::Drba {
            anyhow::ensure!(frames.len() >= 4, "DRBA needs 4 frames");
            self.drba_strided(w, h, frames, t, 4)
        } else {
            anyhow::bail!("RGBA direct interpolation is only supported for RIFE/DRBA");
        };
        if result.is_ok() {
            self.complete_successful_tensorrt_shape(w, h);
            return result;
        }
        if let Err(error) = &result
            && self.fallback_tensorrt_after_engine_creation_failure(error)?
        {
            return self.process_interp_rgba8(w, h, frames, t);
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
        // NeoArc uses validated sequential phases, including RGB GPU IO. The
        // former opt-in multi-request shortcut bypassed the phase stability gate.
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            match self.run_neoamd_many_u8(w, h, frames, 4, 3, ts) {
                Ok(Some(results)) => return Ok(results),
                Ok(None) => {
                    // Older ABI-1 packs do not expose the optional multi-phase
                    // entry point. Preserve their validated sequential route.
                    return ts
                        .iter()
                        .map(|t| self.process_interp_rgba8(w, h, frames, *t))
                        .collect();
                }
                Err(error) => {
                    log::warn!(
                        "neoamd-interp-many-retry-directml: model={} outputs={} reason={error:#}",
                        self.name,
                        ts.len()
                    );
                    // run_neoamd_many_u8 atomically switched this stage to its
                    // DirectML fallback. Continue below with the established
                    // provider path instead of losing the entire source pair.
                }
            }
        }
        if self.interp==InterpKind::Drba&&self.neoarc_session.is_none()&&ts.len()>1&&self.provider==OnnxProvider::DirectML&&self.dml_wgl_device_eligible(){
            // B570: dynamic batch > 1 produced corrupt later phases. Isolate
            // provider invocations, but pack the four image planes only once
            // within this call. Never reuse source pixels across source pairs.
            if !self.dml_drba_sequential_logged{log::info!("directml-drba-phase-isolation: model={} adapter=B570 image_planes=within-call-reuse provider_batch=1 reason=multi-batch-phase-drift",self.name);self.dml_drba_sequential_logged=true;}
            let mut images=Vec::with_capacity(ts.len());let mut total=InterpProfile::default();
            for(index,&phase)in ts.iter().enumerate(){images.push(self.drba_strided_cached(w,h,frames,phase,4,index>0)?);if let Some(profile)=self.last_interp_profile{total.pack_ms+=profile.pack_ms;total.run_ms+=profile.run_ms;total.out_ms+=profile.out_ms;total.padded_size=profile.padded_size;}}
            self.last_interp_profile=Some(total);return Ok(images);
        }
        if self.interp == InterpKind::Drba && self.neoarc_session.is_none() && ts.len() > 1 {
            self.ensure_tensorrt_interp_profile_for_size((w, h))?;
            self.prepare_tensorrt_input_shape(w, h);
            mark_tensorrt_model_started(&self.name);
            let result = self.drba_batch_strided(w, h, frames, ts, 4);
            if result.is_ok() {
                self.complete_successful_tensorrt_shape(w, h);
                return result;
            }
            if let Err(error) = &result
                && self.fallback_tensorrt_after_engine_creation_failure(error)?
            {
                return self.process_interp_many_rgba8(w, h, frames, ts);
            }
            return result;
        }
        if let InterpKind::RifeV1 { channels } = self.interp.clone()
            && ts.len() > 1
        {
            // The phases share one input pair. Reuse image planes only inside
            // this call, never by pointer identity across asynchronous pairs.
            let mut outputs = Vec::with_capacity(ts.len());
            let mut total = InterpProfile::default();
            let mut complete = true;
            for (index, &t) in ts.iter().enumerate() {
                self.last_interp_profile = None;
                outputs.push(self.rife_multi_strided(w, h, frames, channels, t, 4, index > 0)?);
                if let Some(profile) = self.last_interp_profile {
                    total.pack_ms += profile.pack_ms;
                    total.run_ms += profile.run_ms;
                    total.out_ms += profile.out_ms;
                    total.padded_size = profile.padded_size;
                } else {
                    complete = false;
                }
            }
            self.last_interp_profile = complete.then_some(total);
            return Ok(outputs);
        }
        ts.iter()
            .map(|t| self.process_interp_rgba8(w, h, frames, *t))
            .collect()
    }

    /// Keep the boost decision fixed even on the single-midpoint pairs of x2.5.
    pub(crate) fn wants_dml_host_boost(&self, capacity: usize) -> bool {
        capacity >= 3 && self.dml_rife_boost && !self.dml_split_failed
            && self.provider == OnnxProvider::DirectML && self.neoarc_session.is_none()
            && directml_rife_boost_setting().load(Ordering::Relaxed)
    }

    pub fn process_interp_rgba8_pair(
        &mut self, w: i32, h: i32, frames: &[&[u8]], phases: &[f32], capacity: usize,
    ) -> Result<Vec<(i32, i32, Vec<u8>)>> {
        if !phases.is_empty() && self.wants_dml_host_boost(capacity) {
            match self.run_dml_host_boost(w, h, frames, phases) {
                Ok(output) => return Ok(output),
                Err(error) if onnx_cancel_requested() => return Err(error),
                Err(error) => {
                    self.dml_split_failed = true;
                    self.dml_host_boost = None;
                    log::warn!("directml-host-rife-boost-fallback: model={} reason={error:#}", self.name);
                }
            }
        }
        self.process_interp_many_rgba8(w, h, frames, phases)
    }

    fn run_fused_host_pair(&mut self,w:i32,h:i32,pw:i32,ph:i32,frames:&[&[u8]],phases:&[f32]) -> Result<Vec<(i32,i32,Vec<u8>)>> {
        let key=(pw,ph,phases.len());
        if !self.fused_host_boost.iter().any(|(k,_)| *k==key) {
            let model=std::fs::read(&self.source_path)?;
            let bytes=rife_dml_split::fused_host_pair(&model,ph.into(),pw.into(),phases.len())?;
            let session=self.dml_split_session(Err((&bytes,None)),&[])?;
            // Fractional cadence alternates between two counts at one resolution.
            self.fused_host_boost.retain(|(k,_)| (k.0,k.1)==(pw,ph));
            if self.fused_host_boost.len()>=2 { self.fused_host_boost.remove(0); }
            self.fused_host_boost.push((key,session));
            log::info!("directml-fused-pair-ready: model={} size={}x{} phases={} state_host_bytes=0",self.name,pw,ph,phases.len());
        }
        self.reset_interp_pack_cache();
        self.pack_const_key=None;
        self.pack_t_cached=None;
        let (pw,ph)=(pw as usize,ph as usize);
        let t=Instant::now();
        let input=&mut self.scratch_f32;
        input.resize(7*pw*ph,0.0);
        fill_padded_rgb_planes_f32(input,frames,2,w as usize,h as usize,pw,ph,pw*ph,4);
        let tensor=TensorRef::from_array_view(([1,7,ph,pw],input.as_slice()))?;
        let times=TensorRef::from_array_view(([phases.len()],phases))?;
        let pack_ms=t.elapsed().as_secs_f64()*1000.;
        let session=&mut self.fused_host_boost.iter_mut().find(|(k,_)| *k==key).unwrap().1;
        let t=Instant::now();
        let outputs=session.run_with_options(ort::inputs!["input"=>tensor,"phases"=>times],self.run_options.armed()?).map_err(oerr)?;
        let run_ms=t.elapsed().as_secs_f64()*1000.;
        let t=Instant::now();
        let mut result=Vec::with_capacity(phases.len());
        for i in 0..phases.len() {
            let name=format!("neo_pair_head_{i}/output");
            let rgb=outputs.get(name.as_str()).ok_or_else(||anyhow!("missing fused RIFE output {i}"))?;
            result.push((w,h,interp_output_to_rgb8(rgb,w as usize,h as usize,"DirectML fused boost")?));
        }
        self.last_interp_profile=Some(InterpProfile{pack_ms,run_ms,out_ms:t.elapsed().as_secs_f64()*1000.,padded_size:(pw,ph)});
        Ok(result)
    }

    fn run_dml_host_boost(
        &mut self, w: i32, h: i32, frames: &[&[u8]], phases: &[f32],
    ) -> Result<Vec<(i32, i32, Vec<u8>)>> {
        anyhow::ensure!(w > 0 && h > 0 && frames.len() == 2
            && frames.iter().all(|f| f.len() == w as usize * h as usize * 4), "invalid RIFE pair");
        let (_, _, (pw, ph)) = self.interp_gpu_layout((w, h))?;
        if !self.fp16 && !self.fused_host_failed && (1..=4).contains(&phases.len()) {
            match self.run_fused_host_pair(w,h,pw,ph,frames,phases) {
                Ok(result) => return Ok(result),
                Err(error) if onnx_cancel_requested() => return Err(error),
                Err(error) => {
                    self.fused_host_failed=true;
                    self.fused_host_boost.clear();
                    log::warn!("directml-fused-pair-fallback: model={} reason={error:#}",self.name);
                }
            }
        }
        if self.dml_host_boost.as_ref().is_none_or(|(size, _, _)| *size != (pw, ph)) {
            let (trunk, head, _) = self.dml_split_graphs(ph.into(), pw.into())?;
            let trunk = self.dml_split_session(Err((&trunk, None)), &[])?;
            let head = self.dml_split_session(Err((&head, None)), &[])?;
            self.dml_host_boost = Some(((pw, ph), trunk, head));
            log::info!("directml-host-rife-boost-ready: model={} padded={}x{} fp16={}", self.name, pw, ph, self.fp16);
        }
        let (_, trunk, head) = self.dml_host_boost.as_mut().unwrap();
        let (pw, ph) = (pw as usize, ph as usize);
        let pn = pw * ph;
        let shape = [1, 7, ph, pw];
        let mut profile = InterpProfile { padded_size: (pw, ph), ..Default::default() };
        let mut result = Vec::with_capacity(phases.len());
        macro_rules! run_pair {
            ($scratch:ident, $fill:ident, $phase:expr) => {{
                let started = Instant::now();
                let input = &mut self.$scratch;
                input.resize(7 * pn, Default::default());
                $fill(input, frames, 2, w as usize, h as usize, pw, ph, pn, 4);
                input[6*pn..].fill(($phase)(0.5));
                profile.pack_ms += started.elapsed().as_secs_f64()*1000.0;
                let started = Instant::now();
                let tensor = TensorRef::from_array_view((shape, input.as_slice())).map_err(oerr)?;
                let mut outputs = trunk.run_with_options(ort::inputs!["input" => tensor], self.run_options.armed()?).map_err(oerr)?;
                let state = outputs.remove("state").ok_or_else(|| anyhow!("RIFE boost state missing"))?;
                anyhow::ensure!(state.view().downcast::<ort::value::DynTensorValueType>()?.memory_info().is_cpu_accessible(), "RIFE boost state is not host-accessible");
                profile.run_ms += started.elapsed().as_secs_f64()*1000.0;
                for &phase in phases {
                    let started = Instant::now();
                    input[6*pn..].fill(($phase)(phase));
                    let tensor = TensorRef::from_array_view((shape, input.as_slice())).map_err(oerr)?;
                    profile.pack_ms += started.elapsed().as_secs_f64()*1000.0;
                    let started = Instant::now();
                    let outputs = head.run_with_options(ort::inputs!["input" => tensor, "state" => &state], self.run_options.armed()?).map_err(oerr)?;
                    profile.run_ms += started.elapsed().as_secs_f64()*1000.0;
                    let started = Instant::now();
                    let output = outputs.get("output").ok_or_else(|| anyhow!("RIFE boost output missing"))?;
                    result.push((w, h, interp_output_to_rgb8(output, w as usize, h as usize, "DirectML RIFE boost")?));
                    profile.out_ms += started.elapsed().as_secs_f64()*1000.0;
                }
            }};
        }
        // A previous canonical call's scratch metadata no longer describes these planes.
        self.pack_const_key = None;
        self.pack_t_cached = None;
        self.pack_frame_key = None;
        self.pack_second_ptr = None;
        if self.fp16 {
            run_pair!(scratch_f16, fill_padded_rgb_planes_f16, f16::from_f32);
        } else {
            run_pair!(scratch_f32, fill_padded_rgb_planes_f32, |t| t);
        }
        static HOST_BOOST_PAIRS: AtomicU64 = AtomicU64::new(0);
        let count = HOST_BOOST_PAIRS.fetch_add(1, Ordering::Relaxed);
        if count < 4 || count % 600 == 0 {
            log::info!("directml-host-rife-boost: model={} phases={} trunk_runs=1 run_ms={:.3}", self.name, phases.len(), profile.run_ms);
        }
        self.last_interp_profile = Some(profile);
        Ok(result)
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
            || self.neoarc_session.is_some()
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
            self.ensure_dml_rife_shape(padded.0, padded.1)?;

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
                        false,
                    )?;
                    total_run_ms += run_ms;
                    total_copy_ms += copy_ms;
                    prepared.push(PreparedInterpGpuOutput {
                        texture: None,
                        key: output.key,
                        size: (w, h),
                        padded,
                        fp16: true,
                        directml_drba_snapshot_candidate: false,
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
                        false,
                    )?;
                    total_run_ms += run_ms;
                    total_copy_ms += copy_ms;
                    prepared.push(PreparedInterpGpuOutput {
                        texture: None,
                        key: output.key,
                        size: (w, h),
                        padded,
                        fp16: false,
                        directml_drba_snapshot_candidate: false,
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

    /// v687 compatibility hook. Temporal reconstruction is retired; active
    /// NeoAccel acceleration is selected only while a TensorRT session is built.
    /// This function deliberately never changes the current frame.
    pub fn try_process_rgba8_neo_texture(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<Option<(i32, i32, GpuTex)>> {
        // v687: temporal reconstruction was rejected for image quality.
        // Keep the call site as a compatibility no-op so NeoAccel OFF and ON
        // both use the exact current frame; acceleration is now exclusively
        // selected when the TensorRT session itself is built.
        let _ = (gc, w, h, rgba);
        Ok(None)
    }

    #[allow(dead_code)]
    fn neoaccel_run_full_rgba8_texture(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
    ) -> Result<(i32, i32, GpuTex)> {
        // Preserve the same provider-specific fast routes as the ordinary
        // chain. DirectML prefers raw RGBA -> shared output (no GL input
        // upload); TensorRT prefers its GPU texture bridge.
        if self.provider == OnnxProvider::TensorRT && self.should_try_tensorrt_gpu_texture(w, h) {
            let source = gc.upload_rgba8(w, h, rgba);
            if let Some(texture) = self.process_gpu_texture(gc, source)? {
                return Ok((texture.w(), texture.h(), texture));
            }
        }
        if let Some((ow, oh, texture)) = self.process_rgba8_gpu_output(gc, w, h, rgba) {
            return Ok((ow, oh, texture));
        }
        let (ow, oh, out) = self.process_rgba8_native_output(w, h, rgba)?;
        Ok((ow, oh, gc.upload_rgba8(ow, oh, &out)))
    }

    #[allow(dead_code)]
    fn try_process_rgba8_neo_temporal_inner(
        &mut self,
        gc: &mut GlContext,
        w: i32,
        h: i32,
        rgba: &[u8],
        accel: &mut NeoAccelState,
    ) -> Result<Option<(i32, i32, GpuTex)>> {
        let width = usize::try_from(w)?;
        let height = usize::try_from(h)?;
        let expected = width
            .checked_mul(height)
            .and_then(|pixels| pixels.checked_mul(4))
            .ok_or_else(|| anyhow!("NeoAccel temporal input size overflow"))?;
        anyhow::ensure!(
            rgba.len() >= expected,
            "NeoAccel temporal RGBA input too small"
        );

        if accel.input_size.is_some() && accel.input_size != Some((width, height)) {
            gc.remove_persistent_texture(&accel.temporal_ai_key);
            gc.remove_persistent_texture(&accel.temporal_base_key);
            gc.remove_persistent_texture(&accel.temporal_flow_key);
            accel.reset_for_size(width, height);
        }

        // Only the frame immediately following a real ONNX result may attempt
        // reuse. Motion/scene uncertainty falls through to a real model run.
        if accel.temporal_can_attempt(width, height, expected) {
            let flow_started = Instant::now();
            if let Some(flow) = accel.temporal_flow_plan(width, height, &rgba[..expected]) {
                let flow_ms = flow_started.elapsed().as_secs_f64() * 1000.0;
                let reconstruct_started = Instant::now();
                let previous_ai = gc.persistent_texture_by_key(&accel.temporal_ai_key);
                let previous_base = gc.persistent_texture_by_key(&accel.temporal_base_key);
                if let (Some(previous_ai), Some(previous_base), Some((ow, oh))) =
                    (previous_ai, previous_base, accel.output_size)
                {
                    let flow_tex = gc
                        .update_persistent_rgba8(
                            &accel.temporal_flow_key,
                            i32::try_from(flow.grid_w)?,
                            i32::try_from(flow.grid_h)?,
                            &flow.rgba,
                        )
                        .map_err(anyhow::Error::msg)?;
                    let source = gc.upload_rgba8(w, h, &rgba[..expected]);
                    let current_base = crate::render::scaler::resample(
                        gc,
                        source,
                        i32::try_from(ow)?,
                        i32::try_from(oh)?,
                        crate::render::scaler::Kernel::Spline36,
                    )?;
                    let composed = gc
                        .neoaccel_temporal_detail_compose(
                            current_base,
                            previous_ai,
                            previous_base,
                            flow_tex,
                            w,
                            h,
                        )
                        .map_err(anyhow::Error::msg)?;
                    accel.temporal_commit_reuse();
                    if accel.temporal_reused_frames <= 4 || accel.temporal_reused_frames % 120 == 0
                    {
                        log::info!(
                            "NeoAccel temporal-reuse: model='{}' frame={} input={}x{} output={}x{} flow={}x{} confidence={:.1}% mae={:.2} flow_ms={:.3} reconstruct_ms={:.3} next=full-onnx",
                            self.name,
                            accel.temporal_reused_frames,
                            w,
                            h,
                            ow,
                            oh,
                            flow.grid_w,
                            flow.grid_h,
                            flow.confident_ratio * 100.0,
                            flow.mean_error,
                            flow_ms,
                            reconstruct_started.elapsed().as_secs_f64() * 1000.0
                        );
                    }
                    return Ok(Some((i32::try_from(ow)?, i32::try_from(oh)?, composed)));
                }
            }
        }

        // Exact frame: run the unmodified model and return its original texture
        // untouched. Persistent copies are only references for the next frame.
        let exact_started = Instant::now();
        let (ow, oh, exact) = self.neoaccel_run_full_rgba8_texture(gc, w, h, &rgba[..expected])?;
        let exact_ms = exact_started.elapsed().as_secs_f64() * 1000.0;
        let reference_started = Instant::now();
        let scale_x = ow as f32 / w.max(1) as f32;
        let scale_y = oh as f32 / h.max(1) as f32;
        anyhow::ensure!(
            scale_x.is_finite() && scale_y.is_finite() && ow > 0 && oh > 0,
            "NeoAccel temporal invalid ONNX output geometry"
        );
        let source = gc.upload_rgba8(w, h, &rgba[..expected]);
        let base = crate::render::scaler::resample(
            gc,
            source,
            ow,
            oh,
            crate::render::scaler::Kernel::Spline36,
        )?;
        gc.copy_to_persistent_rgba8(&accel.temporal_ai_key, exact)
            .map_err(anyhow::Error::msg)?;
        gc.copy_to_persistent_rgba8(&accel.temporal_base_key, base)
            .map_err(anyhow::Error::msg)?;
        accel.temporal_commit_full(
            width,
            height,
            &rgba[..expected],
            (usize::try_from(ow)?, usize::try_from(oh)?),
        );
        if accel.temporal_full_frames <= 4 || accel.temporal_full_frames % 120 == 0 {
            log::info!(
                "NeoAccel exact-reference: model='{}' frame={} input={}x{} output={}x{} scale={:.3}x{:.3} onnx_ms={:.3} reference_ms={:.3} next=reuse-candidate",
                self.name,
                accel.temporal_full_frames,
                w,
                h,
                ow,
                oh,
                scale_x,
                scale_y,
                exact_ms,
                reference_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        Ok(Some((ow, oh, exact)))
    }

    #[allow(dead_code)]
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
        if self.neoarc_session.is_some() { return self.process_neoarc_gl_texture(gc,input_texture); }
        match self.provider {
            OnnxProvider::DirectML => {
                self.process_dml_gpu_texture(gc, input_texture, after_interpolation)
            }
            OnnxProvider::TensorRT => self.process_tensorrt_gpu_texture(gc, input_texture),
            OnnxProvider::NeoAMD | OnnxProvider::NeoRTX => {
                match self.process_neoamd_gpu_texture(gc, input_texture) {
                    Ok(output) => Ok(output),
                    Err(error) => {
                        self.retire_neoamd_shared_output(gc);
                        self.neoamd_shared_disabled = true;
                        log::warn!(
                            "NeoAMD full GPU-resident IO disabled for '{}'; CPU-visible NeoAMD path retained: {error:#}",
                            self.name
                        );
                        Ok(None)
                    }
                }
            }
            OnnxProvider::MigraphX | OnnxProvider::Cuda => Ok(None),
        }
    }

    fn process_dml_gpu_texture(
        &mut self,
        gc: &mut GlContext,
        input_texture: GpuTex,
        after_interpolation: bool,
    ) -> Result<Option<GpuTex>> {
        if self.dml_wgl_disabled&&self.last_upscale_input_size!=Some((input_texture.w(),input_texture.h())){self.dml_wgl_disabled=false;}

        let dml_temporal_gpu_opt_in = matches!(
            std::env::var("CHIDESCALER_DML_TEMPORAL_GPU")
                .ok()
                .as_deref(),
            Some("1") | Some("true") | Some("on") | Some("yes")
        );
        if self.temporal_frames.is_some() && self.interp == InterpKind::None {
            // TemporalFix has a 21-channel / seven-frame input contract. It must
            // never fall through into the ordinary one-image DirectML GPU path.
            // v782 only gated the dedicated temporal bridge; when that bridge was
            // disabled for safety the function continued below and treated the
            // temporal model like a 3-channel image model. That invalid route is
            // the primary cause of the intermittent all-black TemporalFix output,
            // especially after OFF -> ON cache reuse.
            if self.fp16
                && dml_temporal_gpu_opt_in
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
            return Ok(None);
        }
        // When Intel cannot import a D3D12 GL buffer, retain the legacy
        // input readback but keep the much larger output on the GPU. This also
        // covers GLSL -> SISR chains rather than just the leading CPU stage.
        if gc.external_device_luid().is_none()&&self.fp16&&self.input_channels==Some(3)
            &&self.interp==InterpKind::None&&!self.dml_wgl_disabled&&input_texture.d()==1
            &&input_texture.key.dtype!=super::gl::Dtype::U32
            &&self.dml_wgl_output_requested(input_texture.w(),input_texture.h())&&self.dml_wgl_device_eligible(){
            if self.dml_wgl_input_failed!=Some((input_texture.w(),input_texture.h()))&&!matches!(std::env::var("CHIDESCALER_DML_WGL_INPUT").ok().as_deref(),Some("off"|"0"|"false"))&&self.direct_output.as_ref().is_some_and(|s|s.input_size==(input_texture.w(),input_texture.h())){
                match self.process_dml_wgl_gpu_input(gc,input_texture){Ok(output)=>return Ok(Some(output)),Err(error)=>{self.dml_wgl_input_converter.take();if !onnx_cancel_requested(){self.dml_wgl_input_failed=Some((input_texture.w(),input_texture.h()));}log::warn!("directml-wgl-input-fallback: model={} reason={error:#}",self.name);}}
            }
            let started=Instant::now();let rgba=gc.download_rgba8(input_texture);
            let read_ms=started.elapsed().as_secs_f64()*1000.;
            match self.process_dml_d3d11_output(gc,input_texture.w(),input_texture.h(),&rgba){
                Ok(Some(output))=>{
                    if let Some(profile)=self.last_upscale_profile.as_mut(){profile.pack_ms+=read_ms;}
                    self.log_image_pipeline((input_texture.w(),input_texture.h()),rgba.len(),(output.w(),output.h()),0,started.elapsed().as_secs_f64()*1000.);
                    return Ok(Some(output));
                },
                Ok(None)=>{},
                Err(error)=>{self.dml_wgl_converter.take();if !onnx_cancel_requested(){self.dml_wgl_disabled=true;log::warn!("directml-wgl-output-fallback: model={} reason={error:#} action=original-texture-path",self.name);}}
            }
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
        self.ensure_tensorrt_single_input_profile(channels, padded, "interpolation-padded")?;
        self.prepare_tensorrt_input_shape(size.0, size.1);
        mark_tensorrt_model_started(&self.name);
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

    fn ensure_tensorrt_single_input_profile(
        &mut self,
        channels: usize,
        size: (i32, i32),
        policy: &str,
    ) -> Result<()> {
        let requested = (channels, size.1, size.0);
        if self.provider != OnnxProvider::TensorRT || self.tensorrt_profile_shape == Some(requested)
        {
            return Ok(());
        }
        anyhow::ensure!(
            self.tensorrt_interp_bridge.is_none()
                && self.tensorrt_gpu_bridge.is_none()
                && self.tensorrt_temporal_bridge.is_none(),
            "TensorRT profile cannot change while a shared GPU bridge is active"
        );
        let device_id = self
            .tensorrt_device_id
            .ok_or_else(|| anyhow!("TensorRT CUDA device unavailable for explicit profile"))?;
        let cache_root = self
            .tensorrt_cache_root
            .clone()
            .ok_or_else(|| anyhow!("TensorRT cache root unavailable for explicit profile"))?;
        let input_name = self.in_name.clone();
        let source_path = self.source_path.clone();
        let fallback_dml_adapter = self.fallback_dml_adapter;
        let expected_interp = self.interp.clone();
        let expected_temporal_frames = self.temporal_frames;
        log::info!(
            "tensorrt-profile-rebuild: model={} input={} shape=1x{}x{}x{} policy={} action=rebuild-before-first-inference",
            self.name,
            input_name,
            channels,
            size.1,
            size.0,
            policy
        );
        let old_cache_dir = self.tensorrt_cache_dir.clone();
        let engine_model_path = self
            .tensorrt_model_path
            .clone()
            .unwrap_or_else(|| source_path.clone());
        let explicit_qdq = self.tensorrt_explicit_qdq;
        let qdq_attempt = Self::load_tensorrt_with_profile(
            &source_path,
            &engine_model_path,
            device_id,
            fallback_dml_adapter,
            &cache_root,
            Some((input_name.as_str(), channels, size.1, size.0)),
            explicit_qdq,
        );
        let replacement = match qdq_attempt {
            Ok(stage) => stage,
            Err(qdq_error) if explicit_qdq => {
                log::warn!(
                    "NeoAccel explicit-QDQ profile rejected: model={} shape=1x{}x{}x{} reason={qdq_error:#}; action=retry-TensorRT-FP16",
                    self.name,
                    channels,
                    size.1,
                    size.0
                );
                let mut stage = Self::load_tensorrt_with_profile(
                    &source_path,
                    &source_path,
                    device_id,
                    fallback_dml_adapter,
                    &cache_root,
                    Some((input_name.as_str(), channels, size.1, size.0)),
                    false,
                )
                .with_context(|| {
                    format!(
                        "TensorRT FP16 explicit-profile retry failed for 1x{channels}x{}x{} after NeoAccel explicit-QDQ rejection: {qdq_error:#}",
                        size.1, size.0
                    )
                })?;
                stage.fallback_reason = Some(format!(
                    "NeoAccel explicit-QDQ INT8 profile rejected; TensorRT FP16 restored: {qdq_error:#}"
                ));
                log::info!(
                    "NeoAccel explicit-QDQ fallback complete: model={} provider=TensorRT-FP16 phase=profile-create",
                    self.name
                );
                stage
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "TensorRT explicit-profile session creation failed for 1x{channels}x{}x{}",
                        size.1, size.0
                    )
                });
            }
        };
        anyhow::ensure!(
            replacement.interp == expected_interp,
            "TensorRT explicit-profile session changed interpolation model classification"
        );
        anyhow::ensure!(
            replacement.temporal_frames == expected_temporal_frames,
            "TensorRT explicit-profile session changed temporal model classification"
        );
        let replacement_cache_dir = replacement.tensorrt_cache_dir.clone();
        *self = replacement;
        // The profile-less session is metadata-only. If its distinct cache key
        // contains no engine, remove it after ORT has released that session.
        if let Some(old_cache_dir) = old_cache_dir.filter(|old| {
            replacement_cache_dir.as_ref() != Some(old) && !Self::has_tensorrt_engine_files(old)
        }) {
            cleanup_incomplete_tensorrt_cache(&old_cache_dir);
        }
        log::info!(
            "tensorrt-profile-rebuild: model={} shape=1x{}x{}x{} policy={} result=ready",
            self.name,
            channels,
            size.1,
            size.0,
            policy
        );
        Ok(())
    }

    fn ensure_tensorrt_interp_profile_for_size(&mut self, size: (i32, i32)) -> Result<()> {
        if self.provider != OnnxProvider::TensorRT {
            return Ok(());
        }
        match self.interp_gpu_layout(size) {
            Ok((channels, _, padded)) => {
                self.ensure_tensorrt_single_input_profile(channels, padded, "interpolation-padded")
            }
            // Legacy two-input interpolation models need a multi-input profile,
            // so retain their previous provider behavior rather than inventing
            // an incomplete single-input profile.
            Err(_) => Ok(()),
        }
    }

    fn is_tensorrt_engine_creation_failure(error: &anyhow::Error) -> bool {
        let message = format!("{error:#}").to_ascii_lowercase();
        message.contains("tensorrt ep failed to create engine from network")
            || message.contains("no engine is found")
            || message.contains("failed to create engine from network")
            || message.contains("tensorrt explicit-profile session creation failed")
    }

    fn fallback_tensorrt_after_engine_creation_failure(
        &mut self,
        error: &anyhow::Error,
    ) -> Result<bool> {
        if self.provider != OnnxProvider::TensorRT {
            return Ok(false);
        }
        // v691 explicit-Q/DQ experiment: any runtime/engine failure from the
        // sidecar must retry the user's original FP16 model with TensorRT first.
        // Preserve the narrower historical engine-build filter for normal FP16
        // sessions so NeoAccel OFF keeps the established behavior.
        if !self.tensorrt_explicit_qdq && !Self::is_tensorrt_engine_creation_failure(error) {
            return Ok(false);
        }
        let reason = format!("{error:#}");
        let source_path = self.source_path.clone();
        let fallback_dml_adapter = self.fallback_dml_adapter;
        let failed_cache = self.tensorrt_cache_dir.clone();
        if self.tensorrt_explicit_qdq {
            let Some(device_id) = self.tensorrt_device_id else {
                return Ok(false);
            };
            let Some(cache_root) = self.tensorrt_cache_root.clone() else {
                return Ok(false);
            };
            let profile = self
                .tensorrt_profile_shape
                .map(|(channels, height, width)| (self.in_name.clone(), channels, height, width));
            log::warn!(
                "NeoAccel explicit-QDQ engine rejected: model={} reason={}; action=retry-TensorRT-FP16",
                self.name,
                reason
            );
            let replacement = Self::load_tensorrt_with_profile(
                &source_path,
                &source_path,
                device_id,
                fallback_dml_adapter,
                &cache_root,
                profile.as_ref().map(|(name, channels, height, width)| {
                    (name.as_str(), *channels, *height, *width)
                }),
                false,
            );
            if let Ok(mut replacement) = replacement {
                replacement.fallback_reason = Some(format!(
                    "NeoAccel explicit-QDQ INT8 rejected; TensorRT FP16 restored: {reason}"
                ));
                *self = replacement;
                if let Some(cache_dir) = failed_cache {
                    cleanup_incomplete_tensorrt_cache(&cache_dir);
                }
                log::info!(
                    "NeoAccel explicit-QDQ fallback complete: model={} provider=TensorRT-FP16",
                    self.name
                );
                return Ok(true);
            }
            log::warn!(
                "NeoAccel explicit-QDQ fallback: TensorRT FP16 retry also failed; continuing final DirectML safety fallback"
            );
        }
        mark_tensorrt_model_failed(&source_path, &reason);
        log::error!(
            "tensorrt-engine-build-failed: model={} profile={:?} reason={}; action=discard-cache-and-final-directml-safety-fallback",
            self.name,
            self.tensorrt_profile_shape,
            reason
        );
        let mut replacement = Self::load_directml(&source_path, fallback_dml_adapter)?;
        replacement.fallback_reason = Some(reason);
        replacement
            .provider_desc
            .push_str(" fallback-after-trt-engine-build-failure");
        *self = replacement;
        if let Some(cache_dir) = failed_cache {
            cleanup_incomplete_tensorrt_cache(&cache_dir);
        }
        Ok(true)
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

    pub(crate) fn supports_dml_wgl_interp(&mut self)->bool{
        self.provider==OnnxProvider::DirectML&&self.neoarc_session.is_none()&&self.fp16&&!self.interp_gpu_path_disabled&&!self.dml_wgl_interp_failed&&matches!(self.interp,InterpKind::RifeV1{channels:7}|InterpKind::Drba)&&self.dml_wgl_device_eligible()&&std::env::var("CHIDESCALER_DML_WGL_INTERP").ok().as_deref()!=Some("off")
    }
    fn retire_dml_wgl_interp(&mut self,gc:&mut GlContext){if let Some(mut bridge)=self.dml_wgl_interp_bridge.take(){gc.finish();bridge.boost.take();for slot in &bridge.slots{gc.clear_dml_interp_image(slot.key);}}}
    fn run_dml_wgl_interp_slot(&mut self,index:usize)->Result<(f64,(i32,i32))>{
        let bridge=self.dml_wgl_interp_bridge.as_mut().ok_or_else(||anyhow!("DML WGL bridge missing"))?;let phase=*bridge.phases.get(index).ok_or_else(||anyhow!("DML WGL phase missing"))?;
        let sources:Vec<_>=bridge.history.iter().map(|(input,_)|input.resource.clone()).collect();let slot=bridge.slots.get_mut(index).ok_or_else(||anyhow!("DML WGL slot missing"))?;let started=Instant::now();
        // v972: a single-model run with the fused graph reads the history directly.
        let fused_run=!(bridge.boost_pair&&!bridge.boost_failed)&&bridge.fused.is_some()&&!bridge.fused_failed;
        if !fused_run{bridge.history[0].1.assemble(&sources,&slot.input.resource,bridge.padded,phase)?;}
        let pack_ms=started.elapsed().as_secs_f64()*1000.;
        let mut fused_ms=None;
        if fused_run{
            let fused=bridge.fused.as_mut().unwrap();
            match Self::run_dml_wgl_fused(fused,&bridge.history,bridge.size,index,phase,&self.run_options){
                Ok(ms)=>fused_ms=Some(ms),
                Err(error)=>{
                    // Rebuild this slot's input the original way and continue.
                    bridge.fused_failed=true;bridge.fused=None;
                    log::warn!("directml-wgl-fused-input-fallback: model={} reason={error:#} action=assemble-pass",self.name);
                    bridge.history[0].1.assemble(&sources,&slot.input.resource,bridge.padded,phase)?;
                }
            }
        }
        // v963: boosted pair -> trunk once (any phase input works, it fixes t=0.5),
        // then this phase's head straight into the slot's shared output. Any
        // failure falls back to the single model for the rest of this bridge.
        let mut boosted=None;
        if bridge.boost_pair&&!bridge.boost_failed{
            if let Some(split)=bridge.boost.as_mut(){
                let run_options=&self.run_options;let trunk_done=bridge.boost_trunk_done;
                let result=(||->Result<(f64,f64)>{
                    let started=Instant::now();let mut trunk_ms=0.0;
                    if !trunk_done{
                        split.trunk.run_binding_with_options(&split.trunk_bindings[index],run_options.armed()?).map_err(oerr)?;
                        // The head session must observe the finished state. ORT does not
                        // promise the trunk's commands are flushed to the shared queue at
                        // Run end, so keep the wait (v949 ordering lesson).
                        split.trunk_bindings[index].synchronize_outputs().map_err(oerr)?;
                        trunk_ms=started.elapsed().as_secs_f64()*1000.;
                    }
                    let head_started=Instant::now();
                    split.head.run_binding_with_options(&split.head_bindings[index],run_options.armed()?).map_err(oerr)?;
                    split.head_bindings[index].synchronize_outputs().map_err(oerr)?;
                    Ok((trunk_ms,head_started.elapsed().as_secs_f64()*1000.))
                })();
                match result{
                    Ok((trunk_ms,head_ms))=>{bridge.boost_trunk_done=true;boosted=Some(trunk_ms+head_ms);
                        static WGL_BOOST_LOGS:AtomicU64=AtomicU64::new(0);let n=WGL_BOOST_LOGS.fetch_add(1,Ordering::Relaxed);
                        if n<4||n%600==0{log::info!("directml-wgl-rife-boost: model={} slot={} phase={} trunk_ms={:.3} head_ms={:.3} count={}",self.name,index+1,phase,trunk_ms,head_ms,n+1);}}
                    Err(error)=>{bridge.boost_failed=true;log::warn!("directml-wgl-rife-boost-fallback: model={} reason={error:#} action=single-model-directml",self.name);}
                }
            }
        }
        let(run_ms,copy_ms)=if let Some(ms)=boosted.or(fused_ms){(ms,0.)}else{
        let memory=MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?;let tensor=unsafe{TensorRefMut::<f16>::from_raw(memory,slot.input.allocation,Shape::new(slot.input.dims))?};
        run_cpu_tensor_to_dml_shared_output(&mut self.session,&self.run_options,&self.in_name,&self.out_name,&tensor.view(),&mut slot.output,true)?};
        let out_start=Instant::now();slot.converter.convert(&slot.output.resource,copy_ms==0.)?;let out_ms=copy_ms+out_start.elapsed().as_secs_f64()*1000.;
        self.last_interp_profile=Some(InterpProfile{pack_ms:pack_ms+if index==0{bridge.pack_ms}else{0.},run_ms,out_ms,padded_size:(bridge.padded.0 as usize,bridge.padded.1 as usize)});
        log::debug!("directml-wgl-interp-slot: model={} index={} phase={} pack_ms={:.3} infer_ms={:.3} output_ms={:.3} host_input_bytes=0 host_output_bytes=0",self.name,index,phase,pack_ms,run_ms,out_ms);
        let wall=started.elapsed().as_secs_f64()*1000.;let padded=bridge.padded;let size=bridge.size;let validated=bridge.validated;
        if validated{if let Some(p)=self.last_interp_profile{self.interp_gpu_diag.record(&self.name,"DirectML-WGL",size,padded,[p.pack_ms,p.run_ms,p.out_ms,wall]);}}
        Ok((wall,padded))
    }
    fn dml_wgl_split_session(
        &self,
        bytes: &[u8],
        device: &windows::core::IUnknown,
        queue: &ID3D12CommandQueue,
    ) -> Result<Session> {
        let mut builder = Session::builder()
            .map_err(oerr)?
            .with_optimization_level(dml_graph_optimization(&self.source_path))
            .map_err(oerr)?
            .with_memory_pattern(false)
            .map_err(oerr)?
            .with_parallel_execution(false)
            .map_err(oerr)?;
        ort_status(unsafe {
            (dml_api()?.SessionOptionsAppendExecutionProvider_DML1)(
                builder.ptr().cast_mut(), device.as_raw(), queue.as_raw(),
            )
        })?;
        builder.commit_from_memory(bytes).map_err(oerr)
    }

    /// v963: build the boost trunk/head for the WGL route's slots (FP16 RIFE only).
    fn build_dml_wgl_split(&self,padded:(i32,i32))->Result<DmlWglSplit>{
        let bridge=self.dml_wgl_interp_bridge.as_ref().ok_or_else(||anyhow!("DML WGL bridge missing"))?;
        let(h,w)=(i64::from(padded.1),i64::from(padded.0));
        let(trunk_bytes,head_bytes,state_half)=self.dml_split_graphs(h,w)?;
        let first=&bridge.slots.first().ok_or_else(||anyhow!("DML WGL slots missing"))?.output;
        let(device,luid)=(first.device.clone(),first.luid);
        // Both sessions bind resources created by this exact D3D12 device.
        // Selecting the same adapter again does not guarantee the same device.
        let(dml_device,queue)=dml_device_and_queue(&device)?;
        let trunk = self.dml_wgl_split_session(&trunk_bytes, &dml_device, &queue)?;
        let head = self.dml_wgl_split_session(&head_bytes, &dml_device, &queue)?;
        // v971: a bridged Lite model keeps its FP32 motion state.
        let state_bytes=if state_half{std::mem::size_of::<f16>()}else{std::mem::size_of::<f32>()};
        let mut state=create_direct_input_channels_typed_state(&device,luid,padded,5,state_bytes,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;
        state.dims=[1,5,h,w];
        let memory=MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?;
        let tensor=|allocation:*mut c_void,dims:Vec<i64>|->Result<InterpTensorValue>{Ok(unsafe{InterpTensorValue::F16(TensorRefMut::from_raw(memory.clone(),allocation,Shape::new(dims))?)})};
        let state_value=if state_half{tensor(state.allocation,state.dims.to_vec())?}
            else{unsafe{InterpTensorValue::F32(TensorRefMut::from_raw(memory.clone(),state.allocation,Shape::new(state.dims.to_vec()))?)}};
        let names:Vec<CString>=["input","state","output"].iter().map(|n|CString::new(*n)).collect::<std::result::Result<_,_>>()?;
        let(mut values,mut trunk_bindings,mut head_bindings)=(Vec::new(),Vec::new(),Vec::new());
        for slot in &bridge.slots{
            anyhow::ensure!(slot.input.dims==[1,7,h,w],"WGL phase input shape {:?}",slot.input.dims);
            anyhow::ensure!(slot.output.dims==vec![1,3,h,w],"WGL phase output shape {:?}",slot.output.dims);
            let input=tensor(slot.input.allocation,slot.input.dims.to_vec())?;
            let output=tensor(slot.output.allocation,slot.output.dims.clone())?;
            let trunk_binding=trunk.create_binding()?;let head_binding=head.create_binding()?;
            ort_status(unsafe{(ort::api().BindInput)(trunk_binding.ptr().cast_mut(),names[0].as_ptr(),input.ptr())})?;
            ort_status(unsafe{(ort::api().BindOutput)(trunk_binding.ptr().cast_mut(),names[1].as_ptr(),state_value.ptr())})?;
            ort_status(unsafe{(ort::api().BindInput)(head_binding.ptr().cast_mut(),names[0].as_ptr(),input.ptr())})?;
            ort_status(unsafe{(ort::api().BindInput)(head_binding.ptr().cast_mut(),names[1].as_ptr(),state_value.ptr())})?;
            ort_status(unsafe{(ort::api().BindOutput)(head_binding.ptr().cast_mut(),names[2].as_ptr(),output.ptr())})?;
            values.push(input);values.push(output);trunk_bindings.push(trunk_binding);head_bindings.push(head_binding);
        }
        Ok(DmlWglSplit{trunk,head,trunk_bindings,head_bindings,_values:values,_state_value:state_value,_names:names,_state:state})
    }
    /// v972: input-fused single model for the WGL route (see `DmlWglFused`).
    fn build_dml_wgl_fused(&self)->Result<DmlWglFused>{
        let bridge=self.dml_wgl_interp_bridge.as_ref().ok_or_else(||anyhow!("DML WGL bridge missing"))?;
        anyhow::ensure!(bridge.history.len()==2,"WGL fused input needs a two-frame history");
        let(w,h)=bridge.size;let(pw,ph)=bridge.padded;
        let bytes=std::fs::read(&self.source_path)?;
        let graph=rife_dml_split::wgl_fused_input(&bytes,h.into(),w.into(),ph.into(),pw.into())?;
        let first=&bridge.slots.first().ok_or_else(||anyhow!("DML WGL slots missing"))?.output;
        let(dml_device,queue)=dml_device_and_queue(&first.device)?;
        let session=self.dml_wgl_split_session(&graph,&dml_device,&queue)?;
        for(input,_)in &bridge.history{
            anyhow::ensure!(input.byte_len>=3*w as usize*h as usize*std::mem::size_of::<f16>(),"WGL history buffer too small");
        }
        let memory=MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?;
        let output_name=CString::new("output")?;
        let(mut values,mut bindings)=(Vec::new(),Vec::new());
        for slot in &bridge.slots{
            anyhow::ensure!(slot.output.dims==vec![1,3,i64::from(ph),i64::from(pw)],"WGL fused output shape {:?}",slot.output.dims);
            let output=unsafe{InterpTensorValue::F16(TensorRefMut::from_raw(memory.clone(),slot.output.allocation,Shape::new(slot.output.dims.clone()))?)};
            let binding=session.create_binding()?;
            ort_status(unsafe{(ort::api().BindOutput)(binding.ptr().cast_mut(),output_name.as_ptr(),output.ptr())})?;
            values.push(output);bindings.push(binding);
        }
        Ok(DmlWglFused{session,bindings,_values:values,names:[CString::new("neo_wgl_a")?,CString::new("neo_wgl_b")?,CString::new("neo_wgl_t")?],_device:dml_device,_queue:queue})
    }

    fn run_dml_wgl_fused(fused:&mut DmlWglFused,history:&[(DmlDirectInput,super::dml_gl_bridge::Converter)],size:(i32,i32),index:usize,phase:f32,run_options:&CancelableRunOptions)->Result<f64>{
        let memory=MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?;
        let dims=[1i64,3,i64::from(size.1),i64::from(size.0)];
        let a=unsafe{InterpTensorValue::F16(TensorRefMut::from_raw(memory.clone(),history[0].0.allocation,Shape::new(dims))?)};
        let b=unsafe{InterpTensorValue::F16(TensorRefMut::from_raw(memory,history[1].0.allocation,Shape::new(dims))?)};
        // Same FP16 bits as the D3D12 assemble pass wrote into the time channel.
        let time=[f16::from_f32(phase.clamp(0.,1.))];
        let t=TensorRef::from_array_view(([1usize,1,1,1],&time[..])).map_err(oerr)?;
        let binding=fused.bindings.get(index).ok_or_else(||anyhow!("WGL fused slot missing"))?;
        for(name,value)in[(&fused.names[0],a.ptr()),(&fused.names[1],b.ptr()),(&fused.names[2],t.ptr())]{
            ort_status(unsafe{(ort::api().BindInput)(binding.ptr().cast_mut(),name.as_ptr(),value)})?;
        }
        let started=Instant::now();
        fused.session.run_binding_with_options(binding,run_options.armed()?).map_err(oerr)?;
        binding.synchronize_outputs().map_err(oerr)?;
        Ok(started.elapsed().as_secs_f64()*1000.)
    }

    fn prepare_dml_wgl_interp(&mut self,gc:&mut GlContext,frames:&[GpuTex],phases:&[f32],capacity:usize,generation:u64,sequences:&[u64])->Result<u64>{
        use windows::Win32::Graphics::OpenGL::{wglGetCurrentContext,wglGetCurrentDC};
        let need=self.interp_frames();let frames=&frames[frames.len().checked_sub(need).ok_or_else(||anyhow!("DML WGL history incomplete"))?..];let size=(frames[0].w(),frames[0].h());anyhow::ensure!(frames.iter().all(|f|(f.w(),f.h())==size),"DML WGL history extent mismatch");
        let(channels,_,padded)=self.interp_gpu_layout(size)?;let count=capacity.saturating_sub(1).max(phases.len());let context=(unsafe{wglGetCurrentContext()}.0 as isize,unsafe{wglGetCurrentDC()}.0 as isize);
        if self.dml_wgl_interp_bridge.as_ref().is_some_and(|b|b.size!=size||b.context!=context||b.slots.len()<count||b.slots.iter().any(|s|!gc.has_dml_interp_image(s.key))){self.retire_dml_wgl_interp(gc);}
        if self.dml_wgl_interp_bridge.is_none(){
            self.ensure_dml_rife_shape(padded.0,padded.1)?;let zero=vec![f16::ZERO;padded.0 as usize*padded.1 as usize*channels];let shape=Shape::new([1,channels as i64,padded.1 as i64,padded.0 as i64]);let tensor=TensorRef::from_array_view((shape,zero.as_slice())).map_err(oerr)?;
            let mut outputs=Vec::new();for _ in 0..count{outputs.push(create_direct_output(&mut self.session,&self.run_options,&self.in_name,&self.out_name,&tensor,padded)?);}
            let device=outputs[0].device.clone();let luid=outputs[0].luid;anyhow::ensure!(verified_b570_luid(u64::from_le_bytes(luid)),"DML WGL selected device changed");
            let mut history=Vec::new();for _ in 0..need{history.push((create_direct_input_channels_typed_state(&device,luid,size,3,2,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?,super::dml_gl_bridge::Converter::new(&device,size.0,size.1)?));}
            let mut slots=Vec::new();for output in outputs{anyhow::ensure!(output.dims==vec![1,3,padded.1 as i64,padded.0 as i64],"DML WGL RGB output required");let input=create_direct_input_channels_typed_state(&device,luid,padded,channels,2,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?;let converter=super::dml_gl_bridge::Converter::new(&device,padded.0,padded.1)?;let key=DML_WGL_INTERP_KEY|DIRECT_OUTPUT_KEY.fetch_add(1,Ordering::Relaxed);slots.push(DmlWglInterpSlot{input,output,converter,key});}
            // Install ownership before registration so partial failure can retire all pins.
            self.dml_wgl_interp_bridge=Some(DmlWglInterpBridge{packed_keys:Vec::new(),history,slots,size,padded,phases:phases.to_vec(),context,validated:false,pack_ms:0.,boost:None,boost_pair:false,boost_trunk_done:false,boost_failed:false,fused:None,fused_failed:false});
            for slot in &self.dml_wgl_interp_bridge.as_ref().unwrap().slots{gc.register_dml_interp_image(slot.key,&slot.converter.input_texture()?,padded.0,padded.1)?;}
        }
        let started = Instant::now();
        let bridge = self.dml_wgl_interp_bridge.as_mut().unwrap();
        let keys: Vec<_> = if sequences.len() == need && frames.iter().all(|f| f.allocation_id != 0) {
            sequences.iter().zip(frames).map(|(&seq, frame)| (generation, seq, frame.allocation_id)).collect()
        } else { Vec::new() };
        // A sliding window retains the actual source frames in engine history.
        // Pool storage alone is insufficient: it may now hold a different capture.
        let retained = if bridge.validated && keys.len() == need && bridge.packed_keys.len() == need
            && keys.windows(2).all(|w| w[0].1 < w[1].1)
            && bridge.packed_keys[1..] == keys[..need-1] {
            bridge.history.rotate_left(1);
            need - 1
        } else { 0 };
        for (frame, (input, converter)) in frames.iter().zip(&mut bridge.history).skip(retained) {
            gc.upload_dml_d3d11(&converter.input_texture()?, *frame)?;
            converter.pack(&input.resource)?;
        }
        bridge.packed_keys = keys;
        bridge.phases = phases.to_vec();
        bridge.pack_ms = started.elapsed().as_secs_f64()*1000.;
        log::debug!("directml-wgl-history: generation={} retained={} uploaded={} pack_ms={:.3}", generation, retained, need-retained, bridge.pack_ms);
        if !self.dml_wgl_interp_bridge.as_ref().unwrap().validated{
            // Real-image canonical guard: repeat several distinct phases, without
            // skipping existing FP16 accuracy requirements. Readback is startup only.
            let pixels:Vec<_>=frames.iter().map(|f|gc.download_rgba8(*f)).collect();let input:Vec<&[u8]>=pixels.iter().map(Vec::as_slice).collect();let requested=phases.to_vec();
            for phase in [0.2f32,0.5,0.8]{for _ in 0..2{
                self.dml_wgl_interp_bridge.as_mut().unwrap().phases=vec![phase];self.run_dml_wgl_interp_slot(0)?;let bridge=self.dml_wgl_interp_bridge.as_ref().unwrap();let full=gc.copy_dml_interp_image(bridge.slots[0].key,padded.0,padded.1)?;let output=if padded==size{full}else{super::scaler::crop_edge(gc,full,size.0,size.1)?};let actual=gc.download_rgb8(output);let(_,_,expected)=self.process_interp_rgba8(size.0,size.1,&input,phase)?;anyhow::ensure!(actual.len()==expected.len(),"DML WGL canonical output extent");let maximum=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)).max().unwrap_or(255);let mean=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)as f64).sum::<f64>()/actual.len().max(1)as f64;anyhow::ensure!(maximum<=4&&mean<0.5,"DML WGL phase drift phase={phase} max={maximum} mean={mean}");log::info!("directml-wgl-interp-validation: model={} phase={} max_byte={} mean_byte={:.6} canonical=pass repeat=true",self.name,phase,maximum,mean);
            }}
            let bridge=self.dml_wgl_interp_bridge.as_mut().unwrap();bridge.validated=true;bridge.phases=requested;log::info!("directml-wgl-interp-ready: model={} input={}x{} padded={}x{} slots={} host_input_bytes=0 host_output_bytes=0 scope=steady-pair validation=phase-repeat-canonical",self.name,size.0,size.1,padded.0,padded.1,bridge.slots.len());
        }
        // v972: input fusion for single-model runs. Built once per bridge after
        // the canonical validation, and kept only if it reproduces the assemble
        // route on the same history and phases (GPU output, padded RGB8).
        let want_fused=channels==7&&std::env::var("CHIDESCALER_DML_WGL_FUSED_INPUT").ok().as_deref()!=Some("off");
        if want_fused&&self.dml_wgl_interp_bridge.as_ref().is_some_and(|b|b.validated&&b.fused.is_none()&&!b.fused_failed){
            let started=Instant::now();
            let(requested,boost_pair)={let b=self.dml_wgl_interp_bridge.as_mut().unwrap();let r=(b.phases.clone(),b.boost_pair);b.boost_pair=false;r};
            let result=(||->Result<(DmlWglFused,u8)>{
                let mut fused=Some(self.build_dml_wgl_fused()?);let mut worst=0u8;
                for phase in [0.2f32,0.5,0.8]{
                    let key={let b=self.dml_wgl_interp_bridge.as_mut().unwrap();b.phases=vec![phase];b.fused=None;b.slots[0].key};
                    self.run_dml_wgl_interp_slot(0)?;
                    let image=gc.copy_dml_interp_image(key,padded.0,padded.1)?;let reference=gc.download_rgb8(image);
                    self.dml_wgl_interp_bridge.as_mut().unwrap().fused=fused.take();
                    self.run_dml_wgl_interp_slot(0)?;
                    let b=self.dml_wgl_interp_bridge.as_mut().unwrap();
                    fused=b.fused.take();
                    anyhow::ensure!(!b.fused_failed&&fused.is_some(),"fused run failed during verification");
                    let image=gc.copy_dml_interp_image(key,padded.0,padded.1)?;let actual=gc.download_rgb8(image);
                    anyhow::ensure!(actual.len()==reference.len(),"fused output extent");
                    worst=worst.max(actual.iter().zip(&reference).map(|(a,b)|a.abs_diff(*b)).max().unwrap_or(255));
                }
                Ok((fused.unwrap(),worst))
            })();
            let bridge=self.dml_wgl_interp_bridge.as_mut().unwrap();bridge.phases=requested;bridge.boost_pair=boost_pair;
            match result{
                Ok((fused,worst)) if worst<=1=>{bridge.fused=Some(fused);log::info!("directml-wgl-fused-input: model={} active=true max_byte_vs_assemble={} build_ms={:.0} assemble_pass=skipped-for-single-model",self.name,worst,started.elapsed().as_secs_f64()*1000.);}
                Ok((_,worst))=>{bridge.fused_failed=true;log::warn!("directml-wgl-fused-input: model={} active=false reason=mismatch max_byte_vs_assemble={worst} action=assemble-pass",self.name);}
                Err(error)=>{bridge.fused=None;bridge.fused_failed=true;log::warn!("directml-wgl-fused-input: model={} active=false reason={error:#} action=assemble-pass",self.name);}
            }
        }
        // v963: the canonical validation above always runs the single model. The
        // boost choice is fixed per prepared pair (x2.5 and up, every pair alike).
        let want=self.dml_rife_boost&&self.fp16&&channels==7&&capacity>=3
            &&directml_rife_boost_setting().load(Ordering::Relaxed)
            &&std::env::var("CHIDESCALER_DML_WGL_BOOST").ok().as_deref()!=Some("off");
        let needs_build=want&&self.dml_wgl_interp_bridge.as_ref().is_some_and(|b|b.boost.is_none()&&!b.boost_failed);
        if needs_build{
            let started=Instant::now();
            match self.build_dml_wgl_split(padded){
                Ok(split)=>{let bridge=self.dml_wgl_interp_bridge.as_mut().unwrap();log::info!("directml-wgl-rife-boost-split: active model={} slots={} padded={}x{} build_ms={:.0}",self.name,bridge.slots.len(),padded.0,padded.1,started.elapsed().as_secs_f64()*1000.);bridge.boost=Some(split);}
                Err(error)=>{if let Some(b)=self.dml_wgl_interp_bridge.as_mut(){b.boost_failed=true;}log::warn!("directml-wgl-rife-boost-fallback: model={} reason={error:#} stage=build action=single-model-directml",self.name);}
            }
        }
        {let bridge=self.dml_wgl_interp_bridge.as_mut().unwrap();bridge.boost_pair=want&&bridge.boost.is_some()&&!bridge.boost_failed;bridge.boost_trunk_done=false;
         DML_RIFE_SPLIT_PAIR.store(bridge.boost_pair,Ordering::Relaxed);}
        gc.submit_commands_fence().map_err(anyhow::Error::msg)
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
        self.ensure_dml_rife_shape(padded.0, padded.1)?;
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
                prepared_count: 0,
                prepare_serial: 0,
                rife_boost_for_pair: false,
                slots_done: 0,
                overlap_count: 0,
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
        // v946: build the RIFE boost split together with the GPU bridge, even
        // while boost is off. Building it on the first boosted pair stalled
        // playback for ~0.3 s when boost was switched on mid-stream, and the
        // presentation stayed visibly wrong afterwards.
        // v949: only while boost is on, so a boost-off stream is exactly the
        // pre-boost path (no extra sessions on the GPU bridge).
        if self.dml_rife_boost
            && !self.dml_split_failed
            && interop_mode == DmlInterpInteropMode::ConservativeCopy
            && capacity_factor >= 3
            && directml_rife_boost_setting().load(Ordering::Relaxed)
        {
            if let Err(error) = self.ensure_dml_split() {
                if !onnx_cancel_requested() {
                    self.dml_split_failed = true;
                    self.dml_split = None;
                    log::warn!(
                        "rife-boost-split-fallback: model={} reason={error:#} action=single-model-directml stage=prebuild",
                        self.name
                    );
                }
            }
        }
        let bridge = self.dml_interp_bridge.as_mut().unwrap();
        // v949: a new pair prepared before every phase of the previous pair ran
        // overwrites inputs that are still pending (suspected cause of frames
        // shown out of time order). Log it so a field repro can confirm.
        if bridge.prepare_serial > 0 && bridge.slots_done < bridge.prepared_count {
            bridge.overlap_count = bridge.overlap_count.saturating_add(1);
            if bridge.overlap_count <= 16 || bridge.overlap_count % 240 == 0 {
                log::warn!(
                    "interp-gpu-prepare-overlap: model={} prepare_serial={} slots_done={} prepared={} boost={} count={}",
                    self.name, bridge.prepare_serial, bridge.slots_done, bridge.prepared_count,
                    bridge.rife_boost_for_pair, bridge.overlap_count
                );
            }
        }
        bridge.slots_done = 0;
        bridge.generation = generation;
        bridge.prepared_count = timesteps.len().min(bridge.invocations.len());
        bridge.prepare_serial = bridge.prepare_serial.wrapping_add(1);
        // Every phase of a prepared pair uses the same setting, even when the
        // GUI changes it while a cooperative GPU worker is presenting phases.
        let rife_boost_for_pair = self.dml_rife_boost
            && !self.dml_split_failed
            && interop_mode == DmlInterpInteropMode::ConservativeCopy
            && capacity_factor >= 3
            && directml_rife_boost_setting().load(Ordering::Relaxed);
        // v947: a pair with one admitted midpoint used the full model while the
        // other pairs used the boost, so consecutive pairs alternated between two
        // different motion estimates (visible pulsing). Every pair of an x3+
        // stream now uses the same path.
        DML_RIFE_SPLIT_PAIR.store(rife_boost_for_pair, Ordering::Relaxed);
        if bridge.rife_boost_for_pair != rife_boost_for_pair {
            log::info!(
                "directml-rife-boost-pair: model={} enabled={} generation={} prepare_serial={} phases={} input={}x{} padded={}x{} split_cached={}",
                self.name, rife_boost_for_pair, generation, bridge.prepare_serial, bridge.prepared_count,
                size.0, size.1, padded.0, padded.1, self.dml_split.is_some()
            );
        }
        bridge.rife_boost_for_pair = rife_boost_for_pair;
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
        self.dml_split = None;
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

    pub(crate) fn can_retain_neoarc_interp(&self)->bool { self.neoarc_session.is_some()&&self.neoarc_interp_gl.is_some() }
    /// v985b: called on the render (GL owner) thread after the prepared pair's
    /// GL fence has signalled and before the job reaches the worker. Moves the
    /// NeoArc endpoints from the GL pixel buffers into CL-owned memory so the
    /// worker thread never acquires GL buffers. Failure disables the buffer
    /// input for later pairs; the current pair keeps a working fallback.
    pub(crate) fn stage_neoarc_interp_inputs(&mut self) {
        let Some(bridge)=self.neoarc_interp_gl.as_mut() else {return};
        if bridge.input_buffers.is_empty()||bridge.buffer_input_failed{return;}
        match bridge.io.stage_pair_buffers(){
            Ok(true)=>{}
            Ok(false)=>{bridge.buffer_input_failed=true;log::warn!("neoarc-gl-buffer-input-disabled: stage=staging reason=unsupported action=gl-image-input-next-pair");}
            Err(error)=>{bridge.buffer_input_failed=true;log::warn!("neoarc-gl-buffer-input-disabled: stage=staging reason={error:#} action=gl-image-input-next-pair");}
        }
    }

    /// Workers and presentation slots have already been drained by the engine.
    /// A multiplier changes phases, not model shape, precision or GL context.
    pub fn reset_interp_gpu_for_factor_change(&mut self, gc: &mut GlContext) {
        if self.neoarc_session.is_some() {
            if let Some(bridge)=self.neoarc_interp_gl.as_mut() {
                bridge.frames.clear();bridge.phases.clear();bridge.io.begin_pair();
                log::info!("neoarc-factor-transition: compiled_bridge=retained pending_frames=cleared validation=retained");
            }
        } else { self.retire_interp_gpu_bridges(gc); }
    }

    pub fn retire_interp_gpu_bridges(&mut self, gc: &mut GlContext) {
        self.retire_dml_wgl_interp(gc);
        self.retire_neoarc_interp_gl(gc);
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
        self.ensure_tensorrt_single_input_profile(frames * 3, size, "temporal-runtime")?;
        self.prepare_tensorrt_input_shape(size.0, size.1);
        mark_tensorrt_model_started(&self.name);
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
        self.complete_successful_tensorrt_shape(size.0, size.1);
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
        if self.dml_wgl_disabled&&self.last_upscale_input_size!=Some((w,h)){self.dml_wgl_disabled=false;}
        if self.neoarc_session.is_some() {
            let input=gc.upload_rgba8(w,h,rgba);
            return match self.process_neoarc_gl_texture(gc,input) {
                Ok(Some(output))=>Some((output.w(),output.h(),output)),
                Ok(None)=>None,
                Err(error)=>{log::warn!("neoarc-gl-ingress-fallback: model={} reason={error:#}",self.name);None}
            };
        }
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            return match self.process_neoamd_rgba8_gpu_output(gc, w, h, rgba) {
                Ok(output) => output,
                Err(error) => {
                    self.retire_neoamd_shared_output(gc);
                    self.neoamd_shared_disabled = true;
                    log::warn!(
                        "NeoAMD GPU-resident output disabled for '{}'; CPU-visible NeoAMD path retained: {error:#}",
                        self.name
                    );
                    None
                }
            };
        }
        // The D3D11/WGL bridge does not require Intel's unsupported
        // GL_EXT_memory_object_win32 buffer import.
        if self.provider==OnnxProvider::DirectML&&self.fp16&&self.interp==InterpKind::None&&self.temporal_frames.is_none()
            &&!self.dml_wgl_disabled&&self.dml_wgl_output_requested(w,h)&&self.dml_wgl_device_eligible() {
            match self.process_dml_d3d11_output(gc,w,h,rgba){
                Ok(Some(texture))=>return Some((texture.w(),texture.h(),texture)),
                Ok(None)=>{},
                Err(error)=>{self.dml_wgl_converter.take();if !onnx_cancel_requested(){self.dml_wgl_disabled=true;log::warn!("directml-wgl-output-fallback: model={} reason={error:#} action=original-output-path",self.name);}}
            }
        }
        if !self.supports_dml_gl_bridge()
            || !self.fp16
            || self.interp != InterpKind::None
            || self.temporal_frames.is_some()
            || self.direct_output_disabled
            || gc.external_device_luid().is_none()
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
        if self.neoarc_session.is_some() || self.provider != OnnxProvider::DirectML
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
        self.ensure_dml_local_image_shape(w, h);
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
        pack_rgbx_nchw_f16(rgba, wu, hu, 4, &mut self.scratch_f16);
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
        let direct_bind =
            self.dml_image_symbols.is_some() && verified_b570_luid(u64::from_le_bytes(state.luid));
        let (run_ms, copy_ms) = run_cpu_tensor_to_dml_shared_output(
            &mut self.session,
            &self.run_options,
            &self.in_name,
            &self.out_name,
            &input,
            state,
            direct_bind,
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

    pub(crate) fn supports_dml_wgl_capture(&mut self)->bool{
        self.provider==OnnxProvider::DirectML&&self.neoarc_session.is_none()&&self.fp16&&self.input_channels==Some(3)&&self.interp==InterpKind::None&&self.temporal_frames.is_none()&&self.dml_image_symbols.is_some()&&!self.dml_wgl_disabled&&self.dml_wgl_input_failed!=self.last_upscale_input_size
            &&self.last_upscale_input_size.is_some_and(|(w,h)|self.dml_wgl_output_requested(w,h))&&self.dml_wgl_device_eligible()
            &&!matches!(std::env::var("CHIDESCALER_DML_WGL_INPUT").ok().as_deref(),Some("off"|"0"|"false"))
    }

    fn dml_wgl_device_eligible(&mut self)->bool{
        if self.fallback_dml_adapter.is_some_and(|id|!intel_arc_device_id(id)){return false;}
        if self.dml_image_device_luid.is_none(){self.dml_image_device_luid=Some(dml_session_luid(&self.session).ok());}
        self.dml_image_device_luid.flatten().is_some_and(verified_b570_luid)
    }

    fn dml_wgl_output_requested(&self,w:i32,h:i32)->bool{
        match std::env::var("CHIDESCALER_DML_WGL_OUTPUT").ok().as_deref(){
            Some("off"|"0"|"false")=>false,
            Some("on"|"1"|"true")=>true,
            _=>self.dml_image_symbols.is_some()&&self.last_upscale_input_size==Some((w,h))&&self.last_upscale_profile.map(|p|p.output_size)
                .or_else(||self.dml_wgl_last_geometry.filter(|(input,_)|*input==(w,h)).map(|(_,output)|output))
                .is_some_and(|(ow,oh)|ow.saturating_mul(oh)>=512*512),
        }
    }

    fn process_dml_wgl_gpu_input(&mut self,gc:&mut GlContext,input:GpuTex)->Result<GpuTex>{
        let size=(input.w(),input.h());let start=Instant::now();
        let state=self.direct_output.as_ref().ok_or_else(||anyhow!("GPU input needs established output"))?;
        anyhow::ensure!(state.input_size==size&&state.dims.len()==4&&state.dims[1]==3,"GPU input geometry changed");
        if self.direct_input.as_ref().is_none_or(|s|s.input_size!=size){self.direct_input=Some(create_direct_input_channels_typed_state(&state.device,state.luid,size,3,2,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)?);}
        if self.dml_wgl_input_converter.as_ref().is_none_or(|c|!c.matches(&state.device,size)){self.dml_wgl_input_converter=Some(super::dml_gl_bridge::Converter::new(&state.device,size.0,size.1)?);}
        let fresh=!self.dml_wgl_input_converter.as_ref().unwrap().validated_current_context();
        let converter=self.dml_wgl_input_converter.as_mut().unwrap();let image=converter.input_texture()?;gc.upload_dml_d3d11(&image,input)?;converter.pack(&self.direct_input.as_ref().unwrap().resource)?;
        let pack_ms=start.elapsed().as_secs_f64()*1000.;let state=self.direct_output.as_mut().unwrap();let gpu_input=self.direct_input.as_ref().unwrap();
        let memory=MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?;
        let tensor=unsafe{TensorRefMut::<f16>::from_raw(memory,gpu_input.allocation,Shape::new(gpu_input.dims))?};
        let(run_ms,copy_ms)=run_cpu_tensor_to_dml_shared_output(&mut self.session,&self.run_options,&self.in_name,&self.out_name,&tensor.view(),state,self.dml_image_symbols.is_some())?;
        let output_size=(i32::try_from(state.dims[3])?,i32::try_from(state.dims[2])?);
        if self.dml_wgl_converter.as_ref().is_none_or(|c|!c.matches(&state.device,output_size)){self.dml_wgl_converter=Some(super::dml_gl_bridge::Converter::new(&state.device,output_size.0,output_size.1)?);}
        let out_start=Instant::now();let image=self.dml_wgl_converter.as_mut().unwrap().convert(&state.resource,copy_ms==0.)?;let(output,_)=gc.copy_dml_d3d11(&image,output_size.0,output_size.1)?;
        let profile=UpscaleProfile{pack_ms,run_ms,out_ms:copy_ms+out_start.elapsed().as_secs_f64()*1000.,output_size:(output_size.0 as usize,output_size.1 as usize)};
        if fresh{
            let rgba=gc.download_rgba8(input);let(ow,oh,expected)=self.process_rgba8_native_output(size.0,size.1,&rgba)?;let actual=gc.download_rgba8(output);
            anyhow::ensure!((ow,oh)==output_size&&actual.len()==expected.len(),"GPU input validation extent mismatch");let maximum=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)).max().unwrap_or(255);let mean=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)as f64).sum::<f64>()/actual.len().max(1)as f64;
            anyhow::ensure!(maximum<=4&&mean<0.5,"GPU input validation drift max={maximum} mean={mean}");self.dml_wgl_input_converter.as_mut().unwrap().mark_validated();log::info!("directml-wgl-input-ready: model={} input={}x{} max_byte={} mean_byte={:.6} steady_input_host_bytes=0 steady_output_host_bytes=0",self.name,size.0,size.1,maximum,mean);
        }
        self.last_upscale_profile=Some(profile);self.last_upscale_input_size=Some(size);self.log_image_pipeline(size,0,output_size,0,start.elapsed().as_secs_f64()*1000.);Ok(output)
    }

    /// Shared GPU output diagnostic and implementation. Input remains CPU RGBA.
    pub fn process_dml_d3d11_output(&mut self,gc:&mut GlContext,w:i32,h:i32,rgba:&[u8])->Result<Option<GpuTex>>{
        if self.provider!=OnnxProvider::DirectML||self.neoarc_session.is_some()||!self.fp16||self.interp!=InterpKind::None||self.temporal_frames.is_some(){return Ok(None);}
        if self.dml_image_device_luid.is_none(){self.dml_image_device_luid=Some(dml_session_luid(&self.session).ok());}
        if !self.dml_image_device_luid.flatten().is_some_and(verified_b570_luid){return Ok(None);}
        let Some((shared,_))=self.process_rgba8_dml_shared_output(w,h,rgba)? else{return Ok(None);};
        let actual_luid=u64::from_le_bytes(shared.luid);self.dml_image_device_luid=Some(Some(actual_luid));
        if !verified_b570_luid(actual_luid){return Ok(None);}
        let state=self.direct_output.as_ref().ok_or_else(||anyhow!("DML output disappeared"))?;
        anyhow::ensure!(state.dims.len()==4&&state.dims[1]==3,"GPU RGB output required");
        let fresh=self.dml_wgl_converter.as_ref().is_none_or(|c|!c.matches(&state.device,shared.size));
        if fresh{self.dml_wgl_converter=Some(super::dml_gl_bridge::Converter::new(&state.device,shared.size.0,shared.size.1)?);}
        let started=Instant::now();let profile=self.last_upscale_profile.ok_or_else(||anyhow!("DML output timings missing"))?;
        // The existing B570 direct binding leaves UAV state; provider-copy
        // leaves COMMON. Its measured copy time records the branch taken.
        let image=self.dml_wgl_converter.as_mut().unwrap().convert(&state.resource,profile.out_ms==0.)?;
        let(output,_new_import)=gc.copy_dml_d3d11(&image,shared.size.0,shared.size.1)?;
        self.last_upscale_profile=Some(UpscaleProfile{out_ms:profile.out_ms+started.elapsed().as_secs_f64()*1000.,..profile});
        if !self.dml_wgl_converter.as_ref().unwrap().validated_current_context(){
            let validation_started=Instant::now();let saved=self.last_upscale_profile;
            let (ow,oh,expected)=self.process_rgba8_native_output(w,h,rgba)?;
            let actual=gc.download_rgba8(output);self.last_upscale_profile=saved;
            anyhow::ensure!((ow,oh)==shared.size&&actual.len()==expected.len(),"DML WGL canonical extent mismatch");
            let maximum=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)).max().unwrap_or(255);
            let mean=actual.iter().zip(&expected).map(|(a,b)|a.abs_diff(*b)as f64).sum::<f64>()/actual.len().max(1)as f64;
            anyhow::ensure!(maximum<=4&&mean<0.5,"DML WGL canonical drift max={maximum} mean={mean}");
            self.dml_wgl_converter.as_mut().unwrap().mark_validated();
            log::info!("directml-wgl-output-validation: model={} canonical=pass max_byte={} mean_byte={:.6} validation_ms={:.3} host_readback=surface-validation-only",self.name,maximum,mean,validation_started.elapsed().as_secs_f64()*1000.);
        }
        if fresh{log::info!("directml-wgl-output-ready: model={} output={}x{} transport=D3D12-compute-D3D11-WGL output_host_bytes=0 input=cpu-rgba",self.name,shared.size.0,shared.size.1);}
        Ok(Some(output))
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
        pack_rgbx_nchw_f16(rgba, wu, hu, 4, &mut self.scratch_f16);
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
        let reuse_wait_start = Instant::now();
        gc.wait_external_buffer_idle(state.key)
            .map_err(|error| anyhow!(error))?;
        let reuse_wait_ms = reuse_wait_start.elapsed().as_secs_f64() * 1000.0;

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
        let out_ms = output_start.elapsed().as_secs_f64() * 1000.0;
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms,
            output_size: (ow as usize, oh as usize),
        });
        self.last_upscale_input_size = Some((w, h));
        if should_log_neoamd_profile() {
            log::debug!(
                "directml-profile: model='{}' ingress=cpu-fp16 pack_ms={:.3} run_ms={:.3} reuse_wait_ms={:.3} out_ms={:.3} total_ms={:.3} input={}x{} output={}x{}",
                self.name,
                pack_ms,
                run_ms,
                reuse_wait_ms,
                out_ms,
                pack_ms + reuse_wait_ms + run_ms + out_ms,
                w,
                h,
                ow,
                oh
            );
        }
        Ok((ow, oh, texture))
    }

    /// Release NeoAccel persistent textures before a chain/session transition.
    /// They intentionally live outside the transient pool, so explicit cleanup
    /// prevents VRAM growth when presets or ONNX models are switched repeatedly.
    pub(crate) fn retire_neoaccel_textures(&mut self, gc: &mut GlContext) {
        if let Some(accel) = self.neo_accel.as_mut() {
            gc.remove_persistent_texture(&accel.texture_key);
            gc.remove_persistent_texture(&accel.temporal_ai_key);
            gc.remove_persistent_texture(&accel.temporal_base_key);
            gc.remove_persistent_texture(&accel.temporal_flow_key);
            accel.temporal_ready = false;
            accel.temporal_skip_allowed = false;
            accel.cached_output.clear();
            accel.previous_input.clear();
            accel.output_size = None;
            accel.input_size = None;
        }
    }

    /// Detach OpenGL from the shared D3D12 allocation before DirectML frees it.
    /// The ordering matters on source resizes: freeing the allocation first can
    /// leave the GL memory object pointing at released driver memory.
    /// Detach OpenGL from the NeoAMD D3D12 shared output before a backend/model
    /// transition can drop or unload the owning Backend Pack session.  v750
    /// kept this import alive while DirectML did the equivalent retirement, so
    /// a later DLL reload could reuse the same process-local resource key and GL
    /// would silently keep sampling the old allocation.
    pub(crate) fn retire_neoamd_shared_output(&mut self, gc: &mut GlContext) {
        let output_key = self.neoamd_shared_key.take();
        let input_key = self.neoamd_shared_input_key.take();
        if output_key.is_none() && input_key.is_none() {
            if let Some(session) = self.neoamd_session.as_mut() {
                session.release_shared_output();
            }
            self.neoamd_shared_input_size = None;
            return;
        }
        // Provider/model transitions are rare. Finish GL before releasing
        // either imported D3D12 surface so HIP/GL never retain a stale view.
        gc.finish();
        if let Some(key) = output_key {
            gc.clear_external_buffer_key(key);
        }
        if let Some(key) = input_key {
            gc.clear_external_buffer_key(key);
        }
        for extra in self.neoamd_interp_ring.drain(..) {
            gc.clear_external_buffer_key(extra.resource_key);
        }
        self.neoamd_interp_ring_base = None;
        if let Some(session) = self.neoamd_session.as_mut() {
            session.release_shared_output();
        }
        self.neoamd_shared_input_size = None;
        log::info!(
            "NeoAMD shared IO retired: model='{}' input_key={:?} output_key={:?} reason=gpu-transition",
            self.name,
            input_key,
            output_key
        );
    }

    pub(crate) fn disable_neoamd_shared_output_for_recovery(
        &mut self,
        gc: &mut GlContext,
        reason: &str,
    ) {
        self.retire_neoamd_shared_output(gc);
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            self.neoamd_shared_disabled = true;
            log::warn!(
                "NeoAMD GPU-resident output disabled for transition recovery: model='{}' reason={} fallback=cpu-visible-neoamd",
                self.name,
                reason
            );
        }
    }

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

    pub(crate) fn retire_neoarc_gl(&mut self) {
        if let Some(session)=self.neoarc_session.as_mut(){session.retire_gl_rgb();}
        if self.neoarc_gl_history_active {
            self.temporal_history.clear();self.temporal_size=None;self.temporal_history_stride=0;self.neoarc_temporal_packed=false;
        }
        self.neoarc_gl_history_active=false;
    }
    fn process_neoarc_gl_texture(&mut self,gc:&mut GlContext,input:GpuTex)->Result<Option<GpuTex>> {
        let stage_started=Instant::now();
        use windows::Win32::Graphics::OpenGL::{wglGetCurrentContext,wglGetCurrentDC};
        if input.d()!=1||input.key.dtype==super::gl::Dtype::U32{return Ok(None);}
        let Some(session)=self.neoarc_session.as_mut()else{return Ok(None);};
        let rc=unsafe{wglGetCurrentContext()}.0 as isize;let dc=unsafe{wglGetCurrentDC()}.0 as isize;
        let dimensions=session.prepare_gl_rgb(usize::try_from(input.w())?,usize::try_from(input.h())?,rc,dc)?;
        if let Some((ow,oh))=dimensions {
            let output=gc.make_tex(i32::try_from(ow)?,i32::try_from(oh)?,4,super::gl::Dtype::U8);
            let wait_started=Instant::now();gc.finish();let gl_input_wait_ms=wait_started.elapsed().as_secs_f64()*1000.0;
            if let Some(times)=session.run_gl_rgb(input.tex.0.get(),output.tex.0.get(),!self.neoarc_gl_history_active,input.allocation_id,output.allocation_id)? {
                self.last_upscale_profile=Some(UpscaleProfile{pack_ms:times[0],run_ms:times[1],out_ms:times[2],output_size:(ow,oh)});
                self.last_upscale_input_size=Some((input.w(),input.h()));
                self.neoarc_gl_history_active=self.temporal_frames.is_some();
                let stage_wall_ms=stage_started.elapsed().as_secs_f64()*1000.0;
                if crate::logging::diagnostics_enabled()&&self.image_diag_last_log.is_none_or(|last|last.elapsed()>=Duration::from_secs(1)){
                    let first=self.image_diag_last_log.is_none();self.image_diag_last_log=Some(Instant::now());
                    let cache=session.gl_rgb_import_stats().unwrap_or((false,0,0,0,0));
                    log::info!("sisr-gpu-pipeline: model={} input={}x{} output={}x{} first_sample={} gl_input_wait_ms={:.3} import_pack_ms={:.3} queued_infer_ms={:.3} convert_release_ms={:.3} stage_wall_ms={:.3} unattributed_wall_ms={:.3} import_cache_enabled={} import_cache_entries={} import_cache_hits={} import_cache_misses={} import_cache_logical_bytes={} stage_host_input_bytes=0 stage_host_output_bytes=0 capture_storage=see-wgc-capture-audit final_handoff=OpenGL",
                        self.name,input.w(),input.h(),ow,oh,first,gl_input_wait_ms,times[0],times[1],times[2],stage_wall_ms,(stage_wall_ms-gl_input_wait_ms-times.iter().sum::<f64>()).max(0.0),cache.0,cache.1,cache.2,cache.3,cache.4);
                }
                log::debug!("neoarc-gl-io-profile: model={} input={}x{} output={}x{} frames={} host_input_bytes=0 host_output_bytes=0 gpu_pack_ms={:.3} run_ms={:.3} gpu_output_ms={:.3} final_handoff=OpenGL capture_storage=see-wgc-capture-audit run_scope=queued-input-plus-infer",
                    self.name,input.w(),input.h(),ow,oh,self.temporal_frames.unwrap_or(1),times[0],times[1],times[2]);
                return Ok(Some(output));
            }
        }
        if self.neoarc_gl_history_active {
            self.temporal_history.clear();self.temporal_size=None;self.temporal_history_stride=0;self.neoarc_temporal_packed=false;
            self.neoarc_gl_history_active=false;
            log::warn!("neoarc-gl-temporal-history: model={} action=reset-for-byte-fallback",self.name);
        }
        Ok(None)
    }

    fn try_neoarc_rgb_u8(&mut self, w:i32,h:i32,pixels:&[u8],stride:usize,output_stride:usize,reset:bool)
        ->Result<Option<(i32,i32,Vec<u8>)>> {
        let Some(session)=self.neoarc_session.as_mut() else {return Ok(None);};
        let (wu,hu)=(usize::try_from(w)?,usize::try_from(h)?);
        let Some((ow,oh,data,times))=session.run_rgb_image(wu,hu,pixels,stride,output_stride,reset)? else {return Ok(None);};
        self.last_upscale_profile=Some(UpscaleProfile{pack_ms:times[0],run_ms:times[1],out_ms:times[2],output_size:(ow,oh)});
        self.last_upscale_input_size=Some((w,h));
        log::debug!("neoarc-rgb-io-profile: model={} input={}x{} output={}x{} frames={} host_upload_bytes={} host_read_bytes={} host_fp16_bytes=0 gpu_pack_ms={:.3} run_ms={:.3} gpu_compact_read_ms={:.3} gpu_resident_end_to_end=false run_scope=queued-input-plus-infer",
            self.name,w,h,ow,oh,self.temporal_frames.unwrap_or(1),wu*hu*stride,data.len(),times[0],times[1],times[2]);
        Ok(Some((i32::try_from(ow)?,i32::try_from(oh)?,data)))
    }

    fn process_neoarc_temporal(&mut self, w: i32, h: i32, pixels: &[u8],
        stride: usize, output_stride: usize, frames: usize) -> Result<(i32, i32, Vec<u8>)> {
        use super::neoarc_backend::ImageOutput;
        use rayon::prelude::*;
        if let Some(result)=self.try_neoarc_rgb_u8(w,h,pixels,stride,output_stride,self.temporal_history.len()==1)? {
            self.neoarc_temporal_packed=false;
            return Ok(result);
        }
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let n = wu.checked_mul(hu).filter(|n| *n > 0).ok_or_else(|| anyhow!("invalid temporal extent"))?;
        let frame_len = n.checked_mul(3).ok_or_else(|| anyhow!("temporal frame overflow"))?;
        let total = frame_len.checked_mul(frames).ok_or_else(|| anyhow!("temporal input overflow"))?;
        let started = Instant::now();
        let rolling = self.neoarc_temporal_packed && self.scratch_f16.len() == total;
        prepare_len_f16(&mut self.scratch_f16, total);
        if rolling {
            self.scratch_f16.copy_within(frame_len..total, 0);
            pack_rgbx_nchw_f16(pixels, wu, hu, stride, &mut self.scratch_f16[total-frame_len..]);
        } else {
            // A GPU IO failure may occur after several live frames. Rebuild
            // from the TRUE-N byte history, not N copies of the newest frame.
            let first=self.temporal_history.front().map(Vec::as_slice).unwrap_or(pixels);
            let history=std::iter::repeat_n(first,frames.saturating_sub(self.temporal_history.len()))
                .chain(self.temporal_history.iter().map(Vec::as_slice));
            for(index,frame) in history.enumerate() {
                if index>0&&frame.as_ptr()==first.as_ptr(){self.scratch_f16.copy_within(..frame_len,index*frame_len);}
                else{pack_rgbx_nchw_f16(frame,wu,hu,stride,&mut self.scratch_f16[index*frame_len..(index+1)*frame_len]);}
            }
        }
        self.neoarc_temporal_packed = true;
        let pack_ms = started.elapsed().as_secs_f64()*1000.0;
        let run_started = Instant::now();
        let mut run_ms = 0.0;
        let mut out_ms = 0.0;
        let result = self.neoarc_session.as_mut().ok_or_else(|| anyhow!("NeoArc temporal session missing"))?
            .run_image_with_output(wu, hu, &self.scratch_f16, |ow, oh, output| {
                run_ms = run_started.elapsed().as_secs_f64()*1000.0;
                let started = Instant::now();
                let len = ow.checked_mul(oh).ok_or_else(|| anyhow!("temporal output overflow"))?;
                let mut data = vec![0u8; len.checked_mul(output_stride).ok_or_else(|| anyhow!("temporal RGB overflow"))?];
                let lut = f16_unorm_to_u8_lut();
                data.par_chunks_mut(output_stride).enumerate().for_each(|(i, pixel)| {
                    for channel in 0..3 { pixel[channel] = match &output {
                        ImageOutput::F16(values) => lut[values[channel*len+i].to_bits() as usize],
                        ImageOutput::F32(values) => unit_to_u8(values[channel*len+i]),
                    }; }
                    if output_stride == 4 { pixel[3] = 255; }
                });
                out_ms = started.elapsed().as_secs_f64()*1000.0;
                Ok((i32::try_from(ow)?, i32::try_from(oh)?, data))
            })?;
        self.last_upscale_profile = Some(UpscaleProfile { pack_ms, run_ms, out_ms,
            output_size: (usize::try_from(result.0)?, usize::try_from(result.1)?) });
        self.last_upscale_input_size = Some((w, h));
        log::debug!("neoarc-temporal-profile: model={} frames={} initialized_frames={} pack=rolling-fp16 new_packed_frames={} pack_ms={:.3} run_ms={:.3} out_ms={:.3} input={}x{} output={}x{} gpu_resident_end_to_end=false",
            self.name, frames, self.temporal_history.len(), if rolling { 1 } else { frames }, pack_ms, run_ms, out_ms, w, h, result.0, result.1);
        Ok(result)
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
        if self.neoarc_gl_history_active { self.retire_neoarc_gl(); }
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
        // Only a completed pack on the preceding call can seed this window.
        let previous_cursor = self.dml_temporal_cursor.take();
        self.ensure_tensorrt_single_input_profile(frames * 3, (w, h), "temporal-runtime")?;
        self.prepare_tensorrt_input_shape(w, h);
        mark_tensorrt_model_started(&self.name);
        if self.temporal_size != Some((w, h)) || self.temporal_history_stride != stride {
            self.temporal_history.clear();
            self.neoarc_temporal_packed = false;
            self.temporal_history_stride = stride;
            self.temporal_size = Some((w, h));
            if self.neoamd_temporal_rejected_size != Some((w, h)) {
                self.neoamd_temporal_rejected_size = None;
            }
        }
        // Reuse the oldest history allocation once the TRUE-N ring is full.
        // NeoAMD accepts RGB/RGBA strides directly, so retain the source bytes
        // instead of allocating and repacking RGBA -> RGB on every frame.
        let mut current = if self.temporal_history.len() >= frames {
            self.temporal_history.pop_front().unwrap_or_default()
        } else {
            Vec::new()
        };
        current.resize(n * stride, 0);
        current.copy_from_slice(&pixels[..n * stride]);
        self.temporal_history.push_back(current);
        if self.neoarc_session.is_some() {
            match self.process_neoarc_temporal(w, h, pixels, stride, output_stride, frames) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!("neoarc-temporal-fallback: model={} reason={error:#} action=DirectML history=retained", self.name);
                    self.neoarc_session = None;
                    self.neoarc_temporal_packed = false;
                    self.provider_desc = "DirectML (NeoARC temporal fallback)".into();
                    self.fallback_reason = Some(format!("NeoArc temporal: {error:#}"));
                }
            }
        }
        // Native NeoAMD used to clone/repack the complete 7-frame window here.
        // The history now retains the live RGB/RGBA stride and reuses its ring
        // buffers; Backend Pack receives stable references without another Host
        // color-layout conversion. Temporarily move the
        // history deque out of `self` so we can borrow its stable frame buffers
        // across the mutable provider call without duplicating pixel data.
        // `run_neoamd_u8` does not own/modify temporal_history; restore it before
        // handling success/fallback. The DirectML retry below still materializes
        // an owned snapshot only when NeoAMD actually fails.
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX)
            && self.neoamd_temporal_rejected_size != Some((w, h))
        {
            let history_store = std::mem::take(&mut self.temporal_history);
            let first_ref = history_store.front().map(Vec::as_slice).unwrap_or(&[]);
            let history_refs: Vec<&[u8]> =
                std::iter::repeat_n(first_ref, frames.saturating_sub(history_store.len()))
                    .chain(history_store.iter().map(Vec::as_slice))
                    .collect();
            let native_result = self.run_neoamd_u8(
                w,
                h,
                &history_refs,
                self.temporal_history_stride,
                output_stride,
                0.5,
            );
            self.temporal_history = history_store;
            match native_result {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!(
                        "neoamd-temporal-retry-directml: model={} reason={error:#}",
                        self.name
                    );
                }
            }
        }
        // DirectML needs a read-only snapshot of the logical TRUE-N window,
        // not ownership of N full RGB frames. v782 cloned the complete history
        // every frame (about 19 MiB at 1280x720 and 25.8 MiB at 1280x960).
        // Borrow the stable VecDeque storage through packing instead; the ORT
        // tensor owns the packed scratch, so these references can be dropped
        // before provider execution.
        let first = self
            .temporal_history
            .front()
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let history: Vec<&[u8]> =
            std::iter::repeat_n(first, frames.saturating_sub(self.temporal_history.len()))
                .chain(self.temporal_history.iter().map(Vec::as_slice))
                .collect();
        let channels = frames * 3;
        let shape = vec![1i64, channels as i64, h as i64, w as i64];
        let pack_start = Instant::now();
        let (outputs, pack_ms, run_ms) = if self.fp16 {
            let plane = 3*n;
            let window = n*channels;
            let mirror = self.dml_temporal_mirror && self.provider == OnnxProvider::DirectML
                && window <= 32*1024*1024 && frames > 1;
            let offset = if mirror {
                let cursor = previous_cursor.filter(|_|self.temporal_history.len()>1
                    && self.scratch_f16.len()==2*window);
                prepare_len_f16(&mut self.scratch_f16,2*window);
                let next = if let Some(cursor)=cursor {
                    let start=cursor*plane;
                    pack_rgbx_nchw_f16(pixels,wu,hu,stride,&mut self.scratch_f16[start..start+plane]);
                    self.scratch_f16.copy_within(start..start+plane,window+start);
                    (cursor+1)%frames
                } else {
                    for (i,frame) in history.iter().enumerate() {
                        let start=i*plane;
                        if i>0 && frame.as_ptr()==history[0].as_ptr() {
                            self.scratch_f16.copy_within(..plane,start);
                        } else {
                            pack_rgbx_nchw_f16(frame,wu,hu,stride,&mut self.scratch_f16[start..start+plane]);
                        }
                    }
                    self.scratch_f16.copy_within(..window,window);
                    0
                };
                self.dml_temporal_cursor=Some(next);
                next*plane
            } else {
            prepare_len_f16(&mut self.scratch_f16, n * channels);
            let lut = u8_to_f16_lut();
            // Pack one complete RGB frame per Rayon task. The old channel-wise
            // loop reread every RGB history frame three times and created 21
            // independent tasks. A seven-frame task layout reads each source
            // pixel once and writes the same NCHW21 tensor/FP16 LUT bits.
            self.scratch_f16
                .par_chunks_mut(n * 3)
                .enumerate()
                .for_each(|(frame_index, dst)| {
                    let frame = history[frame_index];
                    let (r, gb) = dst.split_at_mut(n);
                    let (g, b) = gb.split_at_mut(n);
                    for i in 0..n {
                        let src = i * self.temporal_history_stride;
                        r[i] = lut[frame[src] as usize];
                        g[i] = lut[frame[src + 1] as usize];
                        b[i] = lut[frame[src + 2] as usize];
                    }
                });


                0
            };
            drop(history);
            let input = TensorRef::from_array_view((shape,&self.scratch_f16[offset..offset+window])).map_err(oerr)?;
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
                .par_chunks_mut(n * 3)
                .enumerate()
                .for_each(|(frame_index, dst)| {
                    let frame = history[frame_index];
                    let (r, gb) = dst.split_at_mut(n);
                    let (g, b) = gb.split_at_mut(n);
                    for i in 0..n {
                        let src = i * self.temporal_history_stride;
                        r[i] = frame[src] as f32 * (1.0 / 255.0);
                        g[i] = frame[src + 1] as f32 * (1.0 / 255.0);
                        b[i] = frame[src + 2] as f32 * (1.0 / 255.0);
                    }
                });
            drop(history);
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
        drop(outputs);
        self.last_upscale_profile = Some(UpscaleProfile {
            pack_ms,
            run_ms,
            out_ms: out_start.elapsed().as_secs_f64() * 1000.0,
            output_size: (ow, oh),
        });
        self.last_upscale_input_size = Some((w, h));
        if self.provider == OnnxProvider::DirectML {
            if let Some(profile) = self.last_upscale_profile {
                log::debug!(
                    "dml-temporal-profile: model='{}' pack_ms={:.3} run_ms={:.3} out_ms={:.3} total_ms={:.3} input={}x{} output={}x{}",
                    self.name,
                    profile.pack_ms,
                    profile.run_ms,
                    profile.out_ms,
                    profile.pack_ms + profile.run_ms + profile.out_ms,
                    w,
                    h,
                    profile.output_size.0,
                    profile.output_size.1
                );
            }
        }
        self.complete_successful_tensorrt_shape(w, h);
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
        let result = self.process_u8_strided_once(w, h, pixels, stride, output_stride);
        if let Err(error) = &result {
            if self.provider == OnnxProvider::MigraphX {
                let reason = format!("{error:#}");
                self.replace_migraphx_with_directml(&reason)?;
                return self.process_u8_strided_once(w, h, pixels, stride, output_stride);
            }
            if self.fallback_tensorrt_after_engine_creation_failure(error)? {
                return self.process_u8_strided_once(w, h, pixels, stride, output_stride);
            }
        }
        result
    }

    fn process_neoarc_image(&mut self, w: i32, h: i32, pixels: &[u8], stride: usize,
        output_stride: usize) -> Result<(i32, i32, Vec<u8>)> {
        use super::neoarc_backend::ImageOutput;
        use rayon::prelude::*;
        if let Some(result)=self.try_neoarc_rgb_u8(w,h,pixels,stride,output_stride,false)? {return Ok(result);}
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let n = wu.checked_mul(hu).filter(|n| *n > 0)
            .ok_or_else(|| anyhow!("NeoArc SISR input size invalid"))?;
        anyhow::ensure!(stride >= 3 && pixels.len() >= n.checked_mul(stride)
            .ok_or_else(|| anyhow!("NeoArc SISR input overflow"))?, "NeoArc SISR input too short");
        anyhow::ensure!(matches!(output_stride, 3 | 4), "NeoArc SISR output stride invalid");
        let pack_started = Instant::now();
        prepare_len_f16(&mut self.scratch_f16, n.checked_mul(3)
            .ok_or_else(|| anyhow!("NeoArc SISR plane overflow"))?);
        pack_rgbx_nchw_f16(pixels, wu, hu, stride, &mut self.scratch_f16);
        let pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let mut run_ms = 0.0;
        let mut out_ms = 0.0;
        let result = self.neoarc_session.as_mut().ok_or_else(|| anyhow!("NeoArc SISR session missing"))?
            .run_image_with_output(wu, hu, &self.scratch_f16, |ow, oh, output| {
                run_ms = started.elapsed().as_secs_f64() * 1000.0;
                let out_started = Instant::now();
                let plane = ow.checked_mul(oh).ok_or_else(|| anyhow!("NeoArc SISR output overflow"))?;
                let size = plane.checked_mul(output_stride).ok_or_else(|| anyhow!("NeoArc SISR RGB overflow"))?;
                let (out_w, out_h) = (i32::try_from(ow)?, i32::try_from(oh)?);
                let mut rgb = vec![0u8; size];
                let lut = f16_unorm_to_u8_lut();
                rgb.par_chunks_mut(output_stride).enumerate().for_each(|(i, px)| {
                    for c in 0..3 {
                        px[c] = match &output {
                            ImageOutput::F16(v) => lut[v[c * plane + i].to_bits() as usize],
                            ImageOutput::F32(v) => unit_to_u8(v[c * plane + i]),
                        };
                    }
                    if output_stride == 4 { px[3] = 255; }
                });
                out_ms = out_started.elapsed().as_secs_f64() * 1000.0;
                Ok((out_w, out_h, rgb))
            })?;
        self.last_upscale_profile = Some(UpscaleProfile { pack_ms, run_ms, out_ms,
            output_size: (usize::try_from(result.0)?, usize::try_from(result.1)?) });
        Ok(result)
    }

    fn process_u8_strided_once(
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
        if self.neoarc_session.is_some() && self.interp == InterpKind::None {
            match self.process_neoarc_image(w, h, pixels, stride, output_stride) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!("neoarc-sisr-fallback: model={} reason={error:#} action=DirectML", self.name);
                    self.neoarc_session = None;
                    self.provider_desc = "DirectML (NeoARC SISR fallback)".into();
                    self.fallback_reason = Some(format!("NeoArc SISR: {error:#}"));
                }
            }
        }
        if matches!(self.provider, OnnxProvider::NeoAMD | OnnxProvider::NeoRTX) {
            match self.run_neoamd_u8(w, h, &[pixels], stride, output_stride, 0.5) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    // run_neoamd_u8 has already atomically retired the native
                    // session and restored the pre-created DirectML session.
                    log::warn!(
                        "neoamd-image-retry-directml: model={} reason={error:#}",
                        self.name
                    );
                }
            }
        }
        self.ensure_dml_local_image_shape(w, h);
        self.ensure_tensorrt_single_input_profile(3, (w, h), "image-runtime")?;
        self.prepare_tensorrt_input_shape(w, h);
        mark_tensorrt_model_started(&self.name);
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
            pack_rgbx_nchw_f16(pixels, w as usize, h as usize, stride, x);
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

#[allow(dead_code)]
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

const DML_WGL_INTERP_KEY:u64=1u64<<63;
struct DmlWglInterpSlot{input:DmlDirectInput,output:DmlDirectOutput,converter:super::dml_gl_bridge::Converter,key:u64}
impl Drop for DmlWglInterpSlot{fn drop(&mut self){self.output.binding.clear();}}
struct DmlWglInterpBridge{packed_keys:Vec<(u64,u64,u64)>,history:Vec<(DmlDirectInput,super::dml_gl_bridge::Converter)>,slots:Vec<DmlWglInterpSlot>,size:(i32,i32),padded:(i32,i32),phases:Vec<f32>,context:(isize,isize),validated:bool,pack_ms:f64,
    // v963: RIFE boost on the B570/Arc WGL route (trunk once per pair, head per phase).
    // The split points into the slots' allocations, so Drop releases it first.
    boost:Option<DmlWglSplit>,boost_pair:bool,boost_trunk_done:bool,boost_failed:bool,
    // v972: single-model runs read the history buffers directly (input fusion).
    fused:Option<DmlWglFused>,fused_failed:bool}
unsafe impl Send for DmlWglInterpBridge{}
impl Drop for DmlWglInterpBridge{fn drop(&mut self){self.boost.take();self.fused.take();}}

/// v972: the single model with the 7-channel input assembled inside the graph
/// (`rife_dml_split::wgl_fused_input`). It binds the planar RGB history
/// buffers and a one-element phase directly, so the per-slot D3D12 assemble
/// pass and its CPU fence wait disappear. Outputs go to the slots' buffers.
struct DmlWglFused{session:Session,bindings:Vec<IoBinding>,_values:Vec<InterpTensorValue>,names:[CString;3],_device:windows::core::IUnknown,_queue:ID3D12CommandQueue}

/// DirectML device and a direct queue on an existing D3D12 device, so a session
/// can bind resources that this device created.
fn dml_device_and_queue(device:&ID3D12Device)->Result<(windows::core::IUnknown,ID3D12CommandQueue)>{
    use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    use windows::core::{GUID, HRESULT, IUnknown, PCSTR, w};
    type CreateDmlDevice = unsafe extern "system" fn(*mut c_void, u32, *const GUID, *mut *mut c_void) -> HRESULT;
    let module = unsafe { GetModuleHandleW(w!("DirectML.dll"))? };
    let entry = unsafe { GetProcAddress(module, PCSTR(c"DMLCreateDevice".as_ptr().cast())) }
        .ok_or_else(|| anyhow!("DMLCreateDevice unavailable"))?;
    let create: CreateDmlDevice = unsafe { std::mem::transmute(entry) };
    // IDMLDevice IID from DirectML.h.
    let iid = GUID::from_u128(0x6dbd6437_96fd_423f_a98c_ae5e7c2a573f);
    let mut raw = std::ptr::null_mut();
    unsafe { create(device.as_raw(), 0, &iid, &mut raw).ok()?; }
    anyhow::ensure!(!raw.is_null(), "DMLCreateDevice returned null");
    let dml_device = unsafe { IUnknown::from_raw(raw) };
    let queue = unsafe { device.CreateCommandQueue::<ID3D12CommandQueue>(
        &D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() },
    )? };
    Ok((dml_device, queue))
}

/// v963: trunk/head sessions bound to the WGL slots' own D3D12 input/output
/// allocations. The head writes straight into the slot's shared output, so the
/// existing converter/GL import path is unchanged.
struct DmlWglSplit{
    trunk:Session,head:Session,trunk_bindings:Vec<IoBinding>,head_bindings:Vec<IoBinding>,
    _values:Vec<InterpTensorValue>,_state_value:InterpTensorValue,_names:Vec<CString>,_state:DmlDirectInput,
}
impl Drop for DmlWglSplit{fn drop(&mut self){for binding in self.trunk_bindings.iter_mut().chain(self.head_bindings.iter_mut()){binding.clear();}}}
unsafe impl Send for DmlWglSplit{}

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
    /// v933: phases packed by the latest prepare and a serial identifying that prepare
    /// (the FraZerion pair batch runs all of them in one provider call).
    prepared_count: usize,
    prepare_serial: u64,
    rife_boost_for_pair: bool,
    /// v949 diagnostics: slots executed for the current prepared pair.
    slots_done: usize,
    overlap_count: u64,
}

/// v933 FraZerion pair batch: the in-betweens of one source pair (same frames, different t)
/// in ONE DirectML call of a batch-K session. The model's shared trunk then runs once per pair
/// instead of once per phase, and the per-phase provider sync disappears. Only models exported
/// with `frazerion.pair_batch=1` (they assume every batch item is the same pair) use it; any
/// failure returns to the per-phase path for the session.
struct DmlPairBatch {
    key: ((i32, i32), usize),
    session: Session,
    _input_value: TensorRefMut<'static, f16>,
    _input_name: CString,
    binding: IoBinding,
    allocator: Allocator,
    copy: DmlCopyContext,
    input: DmlDirectInput,
    done_serial: u64,
    done_count: usize,
}

impl Drop for DmlPairBatch {
    fn drop(&mut self) {
        // the binding holds an OrtValue that points into `input`
        self.binding.clear();
    }
}

unsafe impl Send for DmlPairBatch {}

/// v935 FraZerion split: the model's shared trunk (frames only, no t) runs ONCE per source pair and
/// every phase runs only the per-t head. Sidecar files next to the model (`<model>.onnx.trunk`,
/// `<model>.onnx.head`, enabled by `frazerion.split=1` in the model) hold the two halves; the trunk
/// result is one full-size 8-channel tensor that every head call reads. The first in-between is not
/// delayed (unlike the v933 pair batch): phase 1 = trunk + head, phases 2.. = head only. The single
/// model file stays the fallback for every other path and for any failure here.
struct DmlSplit {
    /// padded size + the phase input allocations the bindings point into
    key: ((i32, i32), Vec<usize>),
    trunk: Session,
    head: Session,
    trunk_bindings: Vec<IoBinding>,
    head_bindings: Vec<IoBinding>,
    _input_values: Vec<InterpTensorValue>,
    _state_value: InterpTensorValue,
    _names: Vec<CString>,
    state: DmlDirectInput,
    head_allocator: Allocator,
    trunk_serial: Option<u64>,
    midpair_trunks: u64,
}

impl Drop for DmlSplit {
    fn drop(&mut self) {
        // the bindings hold OrtValues that point into `state` and the bridge's phase inputs
        for binding in self.trunk_bindings.iter_mut().chain(self.head_bindings.iter_mut()) {
            binding.clear();
        }
    }
}

unsafe impl Send for DmlSplit {}

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

    /// v933: region copies between buffers that each appear once, in the given resting states
    /// (returned to them afterwards). Regions: (source index, source offset, dest index, dest
    /// offset, bytes).
    fn copy_regions_and_wait(
        &mut self,
        sources: &[(ID3D12Resource, D3D12_RESOURCE_STATES)],
        dests: &[(ID3D12Resource, D3D12_RESOURCE_STATES)],
        regions: &[(usize, u64, usize, u64, u64)],
    ) -> Result<(f64, f64)> {
        let copy_start = Instant::now();
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(
                &self.allocator,
                None::<&windows::Win32::Graphics::Direct3D12::ID3D12PipelineState>,
            )?;
            for (resource, state) in sources {
                Self::transition(&self.list, resource, *state, D3D12_RESOURCE_STATE_COPY_SOURCE);
            }
            for (resource, state) in dests {
                Self::transition(&self.list, resource, *state, D3D12_RESOURCE_STATE_COPY_DEST);
            }
            for &(si, src_offset, di, dst_offset, bytes) in regions {
                self.list.CopyBufferRegion(&dests[di].0, dst_offset, &sources[si].0, src_offset, bytes);
            }
            for (resource, state) in sources {
                Self::transition(&self.list, resource, D3D12_RESOURCE_STATE_COPY_SOURCE, *state);
            }
            for (resource, state) in dests {
                Self::transition(&self.list, resource, D3D12_RESOURCE_STATE_COPY_DEST, *state);
            }
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
        .and_then(|n|n.checked_add(3)).map(|n|n&!3)
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
        .and_then(|bytes|bytes.checked_add(3))
        .map(|bytes|bytes&!3)
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

// A tiny provider allocation identifies the actual Auto-selected adapter.
// This never maps or copies frame pixels and all resources are RAII-owned.
fn dml_session_luid(session:&Session)->Result<u64>{
    let allocator=Allocator::new(session,MemoryInfo::new(AllocationDevice::DIRECTML,0,AllocatorType::Device,MemoryType::Default)?)?;
    let mut allocation=allocator.alloc::<u8>(65536)?;let mut resource=std::ptr::null_mut();
    ort_status(unsafe{(dml_api()?.GetD3D12ResourceFromAllocation)(allocator.ptr().cast_mut(),allocation.as_mut_ptr(),&mut resource)})?;
    let resource=unsafe{ID3D12Resource::from_raw_borrowed(&resource)}.ok_or_else(||anyhow!("null DirectML device probe resource"))?;
    let mut device:Option<ID3D12Device>=None;unsafe{resource.GetDevice(&mut device)?};
    let device=device.ok_or_else(||anyhow!("DirectML device probe missing device"))?;
    Ok(u64::from_le_bytes(luid_bytes(unsafe{device.GetAdapterLuid()})))
}
fn dml_image_shape_budget(luid:u64,w:i32,h:i32)->bool{
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1,IDXGIFactory1,IDXGIAdapter3,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,DXGI_QUERY_VIDEO_MEMORY_INFO};
    let result=(||->windows::core::Result<bool>{unsafe{
        let factory:IDXGIFactory1=CreateDXGIFactory1()?;let mut index=0;
        while let Ok(adapter)=factory.EnumAdapters1(index){index+=1;let desc=adapter.GetDesc1()?;
            if u64::from_le_bytes(luid_bytes(desc.AdapterLuid))!=luid{continue;}
            let adapter:IDXGIAdapter3=adapter.cast()?;let mut info=DXGI_QUERY_VIDEO_MEMORY_INFO::default();adapter.QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&mut info)?;
            let reserve=(w as u64)*(h as u64)*64*2*4+64*1024*1024;
            let allowed=info.Budget>0&&info.CurrentUsage.saturating_add(reserve)<info.Budget/100*85;
            log::info!("directml-image-shape-budget: luid={luid:016x} input={w}x{h} usage_bytes={} budget_bytes={} reserve_bytes={} allowed={allowed}",info.CurrentUsage,info.Budget,reserve);
            return Ok(allowed);
        }Ok(false)
    }})();result.unwrap_or(false)
}

fn verified_b570_luid(luid: u64) -> bool {
    static B570_LUIDS: std::sync::OnceLock<Vec<u64>> = std::sync::OnceLock::new();
    B570_LUIDS
        .get_or_init(|| {
            crate::platform::gpu::enumerate_adapters_quiet()
                .into_iter()
                .filter(|adapter| adapter.vendor_id == 0x8086 && adapter.name.contains("B570"))
                .map(|adapter| adapter.luid)
                .collect()
        })
        .contains(&luid)
}

fn intel_arc_adapters() -> &'static [(i32, u64)] {
    static ARC_ADAPTERS: std::sync::OnceLock<Vec<(i32, u64)>> = std::sync::OnceLock::new();
    ARC_ADAPTERS
        .get_or_init(|| {
            crate::platform::gpu::enumerate_adapters_quiet()
                .into_iter()
                .filter(|adapter| {
                    adapter.vendor_id == 0x8086
                        && adapter.name.to_ascii_lowercase().contains("arc")
                })
                .map(|adapter| (adapter.device_id, adapter.luid))
                .collect()
        })
}

fn intel_arc_luid(luid: u64) -> bool {
    intel_arc_adapters().iter().any(|adapter| adapter.1 == luid)
}

fn intel_arc_device_id(device_id: i32) -> bool {
    intel_arc_adapters()
        .iter()
        .any(|adapter| adapter.0 == device_id)
}

fn dml_graph_optimization(path: &Path) -> GraphOptimizationLevel {
    match std::env::var("CHIDESCALER_DML_GRAPH_OPT").ok().as_deref() {
        Some("disable") => GraphOptimizationLevel::Disable,
        Some("basic") => GraphOptimizationLevel::Level1,
        Some("extended") => GraphOptimizationLevel::Level2,
        Some("all") => GraphOptimizationLevel::All,
        _ => {
            let standard_upscaler = inspect_local_image_model(path)
                .ok()
                .flatten()
                .is_some_and(|plan| matches!(plan.conv_count, 10 | 18))
                || inspect_spanf3_image_model(path).unwrap_or(false)
                || inspect_rtmosr_image_model(path).unwrap_or(false);
            if standard_upscaler {
                GraphOptimizationLevel::Level2
            } else {
                GraphOptimizationLevel::All
            }
        }
    }
}

fn cache_dml_image_shape(shape: (i32, i32)) -> bool {
    shape.0 > 0
        && shape.1 > 0
        && shape.0 as i64 * shape.1 as i64 <= 256 * 256
        && std::env::var("CHIDESCALER_DML_IMAGE_SHAPE_CACHE")
            .ok()
            .as_deref()
            != Some("off")
}

/// Write a derived graph under `cache`, trusting an existing file only when
/// its contents match byte for byte.
fn store_derived_graph(cache: &Path, name: &str, data: &[u8]) -> Result<PathBuf> {
    let target = cache.join(name);
    if std::fs::read(&target).ok().as_deref() != Some(data) {
        std::fs::create_dir_all(cache)?;
        let tmp = target.with_extension(format!("{}.tmp", std::process::id()));
        std::fs::write(&tmp, data)?;
        if let Err(error) = std::fs::rename(&tmp, &target) {
            let _ = std::fs::remove_file(&tmp);
            if std::fs::read(&target).ok().as_deref() != Some(data) {
                return Err(error.into());
            }
        }
    }
    Ok(target)
}

/// v973: ConvTranspose/identity-tuned copy of a seven-channel RIFE graph for
/// OpenVINO, with the original's IO precision. None when nothing changes.
/// v993: in-process memo of derived NeoArc graph files, keyed by the source
/// file identity (path, length, mtime) and a kind tag. Every model or geometry
/// switch rebuilds the stage; re-deriving + hashing the same ONNX each time
/// cost ~0.3 s. An entry is used only while its output file still exists.
type DerivedKey = (PathBuf, u64, std::time::SystemTime, &'static str);
fn derived_memo() -> &'static std::sync::Mutex<HashMap<DerivedKey, (PathBuf, usize, usize)>> {
    static MEMO: std::sync::OnceLock<std::sync::Mutex<HashMap<DerivedKey, (PathBuf, usize, usize)>>> = std::sync::OnceLock::new();
    MEMO.get_or_init(Default::default)
}
fn derived_key(path: &Path, kind: &'static str) -> Option<DerivedKey> {
    let meta = std::fs::metadata(path).ok()?;
    Some((path.to_path_buf(), meta.len(), meta.modified().ok()?, kind))
}
fn memo_derived(path: &Path, kind: &'static str, build: impl FnOnce() -> Result<Option<(PathBuf, usize, usize)>>) -> Result<Option<(PathBuf, usize, usize)>> {
    let disabled = std::env::var("CHIDESCALER_NEOARC_DERIVED_MEMO").ok().as_deref() == Some("off");
    let key = if disabled { None } else { derived_key(path, kind) };
    if let Some(key) = key.as_ref() {
        if let Some(hit) = derived_memo().lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned() {
            if hit.0.is_file() { return Ok(Some(hit)); }
        }
    }
    let built = build()?;
    if let (Some(key), Some(value)) = (key, built.as_ref()) {
        derived_memo().lock().unwrap_or_else(|e| e.into_inner()).insert(key, value.clone());
    }
    Ok(built)
}
fn neoarc_tuned_rife_graph_memo(app_dir: &Path, path: &Path) -> Result<Option<(PathBuf, usize, usize)>> {
    memo_derived(path, "neoarc-tuned", || neoarc_tuned_rife_graph(app_dir, path))
}

fn neoarc_tuned_rife_graph(app_dir: &Path, path: &Path) -> Result<Option<(PathBuf, usize, usize)>> {
    let bytes = std::fs::read(path)?;
    if rife_dml_split::input_channels(&bytes)? != Some(7) || bytes.windows(15).any(|w| w == b"frazerion.split") {
        return Ok(None);
    }
    let (rewritten, convt) = rife_dml_split::dml_convtranspose_rewrite(&bytes)?;
    let (rewritten, identity) = rife_dml_split::dml_identity_rewrite(&rewritten)?;
    if convt == 0 && identity == 0 {
        return Ok(None);
    }
    use sha2::Digest;
    let name = format!("{:x}.onnx", sha2::Sha256::digest(&bytes));
    let tuned = store_derived_graph(&app_dir.join("cache/neoarc/arc-tuned-v1"), &name, &rewritten)?;
    Ok(Some((tuned, convt, identity)))
}

fn run_cpu_tensor_to_dml_shared_output<T: PrimitiveTensorElementType + std::fmt::Debug>(
    session: &mut Session,
    run_options: &CancelableRunOptions,
    input_name: &str,
    output_name: &str,
    input: &TensorRef<'_, T>,
    output: &mut DmlDirectOutput,
    verified_direct_bind: bool,
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
    let direct_bind = match std::env::var("CHIDESCALER_DML_SHARED_DIRECT_BIND")
        .ok()
        .as_deref()
    {
        Some("0" | "off" | "false") => false,
        Some("1" | "on" | "true") => true,
        _ => verified_direct_bind,
    };
    if direct_bind {
        let mut submitted = false;
        let direct = (|| -> Result<(f64, f64)> {
            let value = unsafe {
                TensorRefMut::<T>::from_raw(
                    memory.clone(),
                    output.allocation,
                    Shape::new(output.dims.iter().copied()),
                )?
            };
            let output_c = CString::new(output_name.as_bytes())?;
            ort_status(unsafe {
                (ort::api().BindOutput)(
                    output.binding.ptr().cast_mut(),
                    output_c.as_ptr(),
                    value.ptr(),
                )
            })?;
            static DIAGNOSTIC_RUNS:AtomicU64=AtomicU64::new(0);
            let sample=DIAGNOSTIC_RUNS.fetch_add(1,Ordering::Relaxed)%120==0;
            let cpu_before=if sample{current_thread_cpu_100ns()}else{None};
            let started = Instant::now();
            submitted = true;
            let result = session
                .run_binding_with_options(&output.binding, run_options.armed()?)
                .map_err(oerr)?;
            let run_call_ms=started.elapsed().as_secs_f64()*1000.;let sync_started=Instant::now();
            output.binding.synchronize_outputs().map_err(oerr)?;
            let sync_ms=sync_started.elapsed().as_secs_f64()*1000.;
            let measured_ms=started.elapsed().as_secs_f64()*1000.;
            if sample{let cpu_ms=cpu_before.zip(current_thread_cpu_100ns()).map(|(a,b)|b.saturating_sub(a)as f64/10000.).unwrap_or(-1.);log::info!("directml-binding-timing: input={}x{} run_call_ms={:.3} completion_sync_ms={:.3} thread_cpu_ms={:.3} timing=host-wall-includes-gpu-queue-and-os-scheduling pure_gpu_kernel_time=false",output.input_size.0,output.input_size.1,run_call_ms,sync_ms,cpu_ms);}
            drop(result);
            Ok((measured_ms, 0.0))
        })();
        output.binding.clear();
        match direct {
            Ok(timings) => return Ok(timings),
            Err(error) => {
                if submitted {
                    return Err(error);
                }
                log::warn!(
                    "DirectML shared direct bind failed; retrying provider-copy path: {error:#}"
                );
                ort_status(unsafe {
                    (ort::api().BindInput)(
                        output.binding.ptr().cast_mut(),
                        input_c.as_ptr(),
                        input.ptr(),
                    )
                })?;
            }
        }
    }
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
    gc: &GlContext,
    _stable_post_handoff: bool,
) -> Option<DmlInterpInteropMode> {
    // An OpenGL external-memory bridge cannot be constructed without an
    // imported device LUID. Intel Windows OpenGL lacks the Win32 import API;
    // bypass its allocation attempt before the first interpolation frame.
    if gc.external_device_luid().is_none() {
        return None;
    }
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
        self.rife_multi_strided(w, h, frames, channels, t, 3, false)
    }

    pub(crate) fn set_neoarc_pair_reuse(&mut self, enabled: bool) {
        self.neoarc_pair_reuse = enabled;
        self.neoarc_pair_packed = false;
        if let Some(session) = self.neoarc_session.as_mut() {
            session.set_pair_reuse(enabled);
        }
    }

    fn rife_neoarc_strided(
        &mut self, w: i32, h: i32, frames: &[&[u8]],
        channels: usize, t: f32, stride: usize,
    ) -> Result<(i32, i32, Vec<u8>)> {
        anyhow::ensure!(((frames.len() == 2 && channels == 7) || (frames.len() == 4 && channels == 14)) && stride >= 3,
            "NeoArc requires two RIFE or four DRBA RGB/RGBA frames");
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        anyhow::ensure!(wu > 0 && hu > 0, "NeoArc empty frame");
        // The bundled dynamic RIFE graphs accept the native geometry. The
        // runtime checks output dimensions and repeat validity per shape.
        let (pw, ph) = if channels == 14 { (wu.div_ceil(64) * 64, hu.div_ceil(64) * 64) } else { (wu, hu) };
        if channels==7 {
            if let Some((rgb,times))=self.neoarc_session.as_mut().ok_or_else(||anyhow!("NeoArc missing"))?.run_rife_rgb(wu,hu,frames,stride,t)? {
                self.neoarc_pair_packed=false;
                self.last_interp_profile=Some(InterpProfile{pack_ms:times[0],run_ms:times[1],out_ms:times[2],padded_size:(pw,ph)});
                log::debug!("neoarc-rife-rgb-profile: input={}x{} host_upload_bytes={} host_read_bytes={} host_fp16_bytes=0 pack_ms={:.3} run_ms={:.3} out_ms={:.3} capture_gpu_resident=false output_gpu_resident=false",w,h,wu*hu*stride*2,rgb.len(),times[0],times[1],times[2]);
                return Ok((w,h,rgb));
            }
        }
        let frame_count = frames.len();
        let timestep_channel = frame_count * 3;
        let plane = pw.checked_mul(ph).ok_or_else(|| anyhow!("NeoArc plane overflow"))?;
        let pack_started = Instant::now();
        let input = if self.neoarc_pair_reuse {
            prepare_len_f16(&mut self.neoarc_pair_input, channels * plane);
            if !self.neoarc_pair_packed {
                fill_padded_rgb_planes_f16(&mut self.neoarc_pair_input, frames, frame_count,
                    wu, hu, pw, ph, plane, stride);
                self.neoarc_pair_packed = true;
            }
            self.neoarc_pair_input.as_mut_slice()
        } else {
            prepare_len_f16(&mut self.scratch_f16, channels * plane);
            let input = self.scratch_f16.as_mut_slice();
            fill_padded_rgb_planes_f16(input, frames, frame_count, wu, hu, pw, ph, plane, stride);
            input
        };
        input[timestep_channel * plane..(timestep_channel + 1) * plane].fill(f16::from_f32(t.clamp(0.0, 1.0)));
        if channels == 14 { input[13 * plane..14 * plane].fill(f16::ONE); }
        let pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;
        let precision_before=self.neoarc_session.as_ref().unwrap().precision_for_shape(wu,hu);
        let run_started = Instant::now();
        let mut run_ms = 0.0;
        let mut out_ms = 0.0;
        let mut rgb8 = self.neoarc_session.as_mut()
            .ok_or_else(|| anyhow!("NeoArc session unavailable"))?
            .run_with_output(pw, ph, input, |output| {
                run_ms = run_started.elapsed().as_secs_f64() * 1000.0;
                let out_started = Instant::now();
                let result = nchw_rgb_f16_to_rgb8(output, pw, ph, wu, hu, "NeoArc RIFE");
                out_ms = out_started.elapsed().as_secs_f64() * 1000.0;
                result
            })?;
        if channels==7 {
            let validation_started=Instant::now();
            self.neoarc_session.as_mut().unwrap().prepare_rife_rgb(wu,hu,input,frames,stride);
            if precision_before!="f32"&&self.neoarc_session.as_ref().unwrap().precision_for_shape(wu,hu)=="f32" {
                // A rejected optimization must not return its old FP16 frame after
                // selecting a safer per-shape precision. Recompute the real phase.
                rgb8=self.neoarc_session.as_mut().unwrap().run_with_output(pw,ph,input,|output|nchw_rgb_f16_to_rgb8(output,pw,ph,wu,hu,"NeoArc RIFE f32 recovery"))?;
            }
            run_ms+=validation_started.elapsed().as_secs_f64()*1000.0;
            // Keep the validated legacy pair cache when falling back. The GPU
            // fast path invalidates it only after actually processing a frame.
        }
        self.last_interp_profile = Some(InterpProfile {
            pack_ms, run_ms, out_ms,
            padded_size: (pw, ph),
        });
        Ok((w, h, rgb8))
    }

    fn rife_neoarc_many_rgba8(
        &mut self, w: i32, h: i32, frames: &[&[u8]], ts: &[f32],
    ) -> Result<Vec<(i32, i32, Vec<u8>)>> {
        anyhow::ensure!(frames.len() == 2 && !ts.is_empty() && ts.len() <= 4,
            "NeoArc async phases require two RGBA frames and at most four outputs");
        let (wu, hu) = (usize::try_from(w)?, usize::try_from(h)?);
        let (pw, ph) = (wu, hu);
        let plane = pw.checked_mul(ph).ok_or_else(|| anyhow!("NeoArc plane overflow"))?;
        let pack_started = Instant::now();
        prepare_len_f16(&mut self.scratch_f16, 7 * plane);
        let input = self.scratch_f16.as_mut_slice();
        fill_padded_rgb_planes_f16(input, frames, 2, wu, hu, pw, ph, plane, 4);
        let pack_ms = pack_started.elapsed().as_secs_f64() * 1000.0;
        let run_started = Instant::now();
        let mut out_ms = 0.0;
        let results = self.neoarc_session.as_mut()
            .ok_or_else(|| anyhow!("NeoArc session unavailable"))?
            .run_many_async_with_output(pw, ph, input, ts, |output| {
                let out_started = Instant::now();
                let rgb8 = nchw_rgb_f16_to_rgb8(output, pw, ph, wu, hu, "NeoArc RIFE")?;
                out_ms += out_started.elapsed().as_secs_f64() * 1000.0;
                Ok((w, h, rgb8))
            })?;
        let run_ms = (run_started.elapsed().as_secs_f64() * 1000.0 - out_ms).max(0.0);
        self.last_interp_profile = Some(InterpProfile {
            pack_ms, run_ms, out_ms,
            padded_size: (pw, ph),
        });
        Ok(results)
    }

    fn rife_multi_strided(
        &mut self, w: i32, h: i32, frames: &[&[u8]], channels: usize,
        t: f32, stride: usize, reuse_same_pair_planes: bool,
    ) -> Result<(i32, i32, Vec<u8>)> {
        let native = self.fp16 && channels == 7 && self.supports_arc_dml_cpu_stream()
            && !self.dml_rife_native_rejected.contains(&(w, h));
        let result = self.rife_multi_strided_once(w, h, frames, channels, t, stride,
            reuse_same_pair_planes, native);
        if let Err(error) = &result
            && native && !onnx_cancel_requested() {
            log::warn!("directml-rife-native-shape: model={} size={}x{} result=rejected reason={error:#} retry=legacy-padding", self.name, w, h);
            self.dml_rife_native_rejected.insert((w, h));
            self.reset_interp_pack_cache();
            return self.rife_multi_strided_once(w, h, frames, channels, t, stride, false, false);
        }
        result
    }

    fn rife_multi_strided_once(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        channels: usize,
        t: f32,
        stride: usize,
        reuse_same_pair_planes: bool,
        native_geometry: bool,
    ) -> Result<(i32, i32, Vec<u8>)> {
        // The runtime validates previously unstable shapes before accepting
        // their first output. Runtime failure explicitly retires NeoArc below.
        if self.neoarc_session.is_some() {
            match self.rife_neoarc_strided(w, h, frames, channels, t, stride) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!("neoarc-runtime-fallback: model={} reason={error:#} action=DirectML", self.name);
                    self.neoarc_session = None;
                    self.provider_desc = "DirectML (NeoARC fallback)".into();
                    self.fallback_reason = Some(format!("NeoArc runtime failed: {error:#}"));
                    self.reset_interp_pack_cache();
                }
            }
        }
        let nf = frames.len();
        anyhow::ensure!(stride >= 3, "frame stride too small");
        anyhow::ensure!(channels >= 3 * nf + 1, "model wants more frames");
        let (w, h) = (w as usize, h as usize);
        anyhow::ensure!(w > 0 && h > 0, "empty frame");
        let (pw, ph) = if native_geometry { (w, h) } else {
            (w.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE,
             h.div_ceil(RIFE_PAD_MULTIPLE) * RIFE_PAD_MULTIPLE)
        };
        self.ensure_dml_rife_shape(i32::try_from(pw)?, i32::try_from(ph)?)?;
        let pn = pw * ph;
        use rayon::prelude::*;
        let aux = 3 * nf;
        let shape = vec![1i64, channels as i64, ph as i64, pw as i64];
        let pack_t0 = Instant::now();
        let const_key = (pw, ph, channels, aux);
        let frame_key = (w, h, pw, ph, stride);
        if self.pack_frame_key != Some(frame_key) {
            log::info!("rife-geometry: model={} provider={} image={}x{} tensor={}x{} padding_pixels={} native_arc_dml={}",
                self.name, self.metrics_provider_label(), w, h, pw, ph,
                pw * ph - w * h, native_geometry);
        }
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
            if reuse_same_pair_planes {
                // Image planes were packed by the preceding phase of this call.
            } else if reuse_prev {
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
            if channels==7 && self.provider==OnnxProvider::DirectML
                && self.neoarc_session.is_none()
                && std::env::var("CHIDESCALER_DML_HOST_OUTPUT_REUSE").ok().as_deref()==Some("on") {
                let run_t0=Instant::now();
                if self.dml_host_interp_binding.as_ref().is_none_or(|(size,_)|*size!=(pw,ph)) {
                    let mut binding=self.session.create_binding().map_err(oerr)?;
                    let output=Tensor::from_array(([1usize,3,ph,pw],vec![f16::ZERO;3*pn])).map_err(oerr)?;
                    binding.bind_output(self.out_name.as_str(),output).map_err(oerr)?;
                    self.dml_host_interp_binding=Some(((pw,ph),binding));
                }
                let (_,binding)=self.dml_host_interp_binding.as_mut().unwrap();
                binding.bind_input(self.in_name.as_str(),&tin).map_err(oerr)?;
                let result=(||->Result<_>{
                    let outputs=self.session.run_binding_with_options(binding,self.run_options.armed()?).map_err(oerr)?;
                    binding.synchronize_outputs().map_err(oerr)?;
                    let run_ms=run_t0.elapsed().as_secs_f64()*1000.0;
                    let out_started=Instant::now();
                    let output=outputs.get(self.out_name.as_str()).ok_or_else(||anyhow!("missing reused DirectML output"))?;
                    let rgb=interp_output_to_rgb8(output,w,h,"reused DirectML output")?;
                    Ok((rgb,run_ms,out_started.elapsed().as_secs_f64()*1000.0))
                })();
                binding.clear_inputs();
                let(rgb,run_ms,out_ms)=result?;
                self.last_interp_profile=Some(InterpProfile{pack_ms,run_ms,out_ms,padded_size:(pw,ph)});
                self.pack_const_key=Some(const_key);self.pack_t_cached=Some(t.clamp(0.0,1.0));
                self.pack_frame_key=Some(frame_key);self.pack_second_ptr=frames.get(1).map(|frame|frame.as_ptr()as usize);
                return Ok((w as i32,h as i32,rgb));
            }
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
            if reuse_same_pair_planes {
                // Image planes were packed by the preceding phase of this call.
            } else if reuse_prev {
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
        self.drba_strided_cached(w,h,frames,t,stride,false)
    }

    fn drba_strided_cached(
        &mut self,
        w: i32,
        h: i32,
        frames: &[&[u8]],
        t: f32,
        stride: usize,
        reuse_planes: bool,
    ) -> Result<(i32, i32, Vec<u8>)> {
        if self.neoarc_session.is_some() {
            match self.rife_neoarc_strided(w, h, frames, 14, t, stride) {
                Ok(result) => return Ok(result),
                Err(error) => {
                    log::warn!("neoarc-runtime-fallback: model={} reason={error:#} action=DirectML", self.name);
                    self.neoarc_session = None;
                    self.provider_desc = "DirectML (NeoARC fallback)".into();
                    self.fallback_reason = Some(format!("NeoArc DRBA runtime failed: {error:#}"));
                    self.reset_interp_pack_cache();
                }
            }
        }
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
            if !reuse_planes{fill_padded_rgb_planes_f16(x, frames, 4, w, h, pw, ph, pn, stride);}
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
            if !reuse_planes{fill_padded_rgb_planes_f32(x, frames, 4, w, h, pw, ph, pn, stride);}
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
    // A row stays inside one channel. Parallel tiles expose more than three
    // tasks at HD sizes and keep edge replication outside the pixel loop.
    let tile = pw * 32;
    x[base * pn..(base + 3) * pn].par_chunks_mut(pn).enumerate().for_each(|(c, plane)| {
        plane.par_chunks_mut(tile).enumerate().for_each(|(block, rows)| {
            for (r, row) in rows.chunks_exact_mut(pw).enumerate() {
                let y = (block * 32 + r).min(h - 1);
                let source = &src[y*w*stride..(y+1)*w*stride];
                let (pixels, edge) = row.split_at_mut(w);
                if stride == 4 {
                    for (dst, rgba) in pixels.iter_mut().zip(source.chunks_exact(4)) {
                        *dst = f32::from(rgba[c]) * (1.0 / 255.0);
                    }
                } else {
                    for (dst, pixel) in pixels.iter_mut().zip(source.chunks_exact(stride)) {
                        *dst = f32::from(pixel[c]) * (1.0 / 255.0);
                    }
                }
                edge.fill(pixels[w-1]);
            }
        });
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

/// Pack one RGB or RGBA row into all three NCHW planes while the source row is hot
/// in cache. The previous three-plane loop scanned the same RGBA input three
/// times and exposed only three tasks to Rayon at any resolution.
fn pack_rgbx_nchw_f16(src: &[u8], w: usize, h: usize, stride: usize, dst: &mut [f16]) {
    use rayon::prelude::*;
    let n = w * h;
    debug_assert!(stride >= 3 && src.len() >= n * stride && dst.len() >= n * 3);
    let (red, rest) = dst.split_at_mut(n);
    let (green, blue) = rest.split_at_mut(n);
    let lut = u8_to_f16_lut();
    if n < 64 * 1024 {
        for (i, pixel) in src[..n * stride].chunks_exact(stride).enumerate() {
            red[i] = lut[pixel[0] as usize];
            green[i] = lut[pixel[1] as usize];
            blue[i] = lut[pixel[2] as usize];
        }
        return;
    }
    // Group rows so a 256-pixel frame creates tens of Rayon jobs rather
    // than hundreds; each worker still writes disjoint channel ranges.
    let chunk_pixels = w * 16;
    red.par_chunks_mut(chunk_pixels)
        .zip(green.par_chunks_mut(chunk_pixels))
        .zip(blue.par_chunks_mut(chunk_pixels))
        .enumerate()
        .for_each(|(chunk, ((red, green), blue))| {
            let start = chunk * chunk_pixels * stride;
            let pixels = &src[start..start + red.len() * stride];
            for (i, pixel) in pixels.chunks_exact(stride).enumerate() {
                red[i] = lut[pixel[0] as usize];
                green[i] = lut[pixel[1] as usize];
                blue[i] = lut[pixel[2] as usize];
            }
        });
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
    debug_assert_eq!(pn, pw * ph);
    // Read each source row once for all channels, including repeated edge
    // rows used by RIFE padding. Three separate channel scans waste memory
    // bandwidth on the CPU-visible DirectML fallback path.
    let planes = &mut x[base * pn..(base + 3) * pn];
    let (red, rest) = planes.split_at_mut(pn);
    let (green, blue) = rest.split_at_mut(pn);
    red.par_chunks_mut(pw)
        .zip(green.par_chunks_mut(pw))
        .zip(blue.par_chunks_mut(pw))
        .enumerate()
        .for_each(|(y, ((red, green), blue))| {
            let sy = y.min(h - 1);
            let src_row = &src[sy * w * stride..(sy * w + w) * stride];
            for (i, px) in src_row.chunks_exact(stride).enumerate() {
                red[i] = lut[px[0] as usize];
                green[i] = lut[px[1] as usize];
                blue[i] = lut[px[2] as usize];
            }
            let (last_r, last_g, last_b) = (red[w - 1], green[w - 1], blue[w - 1]);
            red[w..].fill(last_r);
            green[w..].fill(last_g);
            blue[w..].fill(last_b);
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
    if want_w <= ow && want_h <= oh {
        rgb8.par_chunks_mut(3 * want_w)
            .enumerate()
            .for_each(|(y, row)| {
                let src = y * ow;
                for (x, px) in row.chunks_exact_mut(3).enumerate() {
                    let i = src + x;
                    px[0] = lut[v[i].to_bits() as usize];
                    px[1] = lut[v[on + i].to_bits() as usize];
                    px[2] = lut[v[2 * on + i].to_bits() as usize];
                }
            });
        return Ok(rgb8);
    }
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

const NEOACCEL_QDQ_SOURCE_NAME: &str = "2x_AnimeJaNai_SD_V1beta34_Compact_93k-fp16.onnx";
const NEOACCEL_QDQ_SIDECAR_NAME: &str = "2x_AnimeJaNai_SD_V1beta34_Compact_93k-neo-int8qdq.onnx";
const NEOACCEL_QDQ_SOURCE_SHA256: &str =
    "2806934124e583a74936c492f2863c772aad4562bd428ac166ba79198c941528";
const NEOACCEL_QDQ_SIDECAR_SHA256: &str =
    "03e77016b6d85e27825e20327a097bd289f768e0408ce8ac6500639733bc3c5f";

fn file_sha256_hex(path: &Path) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let digest = Sha256::digest(&bytes);
    Some(
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
    )
}

fn neoaccel_tensorrt_qdq_sidecar(source_path: &Path) -> Option<PathBuf> {
    if source_path.file_name().and_then(|name| name.to_str()) != Some(NEOACCEL_QDQ_SOURCE_NAME) {
        return None;
    }
    let source_hash = file_sha256_hex(source_path)?;
    if source_hash != NEOACCEL_QDQ_SOURCE_SHA256 {
        log::warn!(
            "NeoAccel explicit-QDQ skipped: model={} reason=source-hash-mismatch expected={} actual={}",
            source_path.display(),
            NEOACCEL_QDQ_SOURCE_SHA256,
            source_hash
        );
        return None;
    }
    let models_dir = source_path
        .ancestors()
        .find(|candidate| candidate.file_name().and_then(|name| name.to_str()) == Some("models"))?;
    let app_root = models_dir.parent().unwrap_or_else(|| Path::new(""));
    let sidecar_path = app_root
        .join("backends")
        .join("NeoAccel")
        .join(NEOACCEL_QDQ_SIDECAR_NAME);
    let Some(sidecar_hash) = file_sha256_hex(&sidecar_path) else {
        log::warn!(
            "NeoAccel explicit-QDQ skipped: model={} reason=sidecar-missing path={}",
            source_path.display(),
            sidecar_path.display()
        );
        return None;
    };
    if sidecar_hash != NEOACCEL_QDQ_SIDECAR_SHA256 {
        log::warn!(
            "NeoAccel explicit-QDQ skipped: model={} reason=sidecar-hash-mismatch path={} expected={} actual={}",
            source_path.display(),
            sidecar_path.display(),
            NEOACCEL_QDQ_SIDECAR_SHA256,
            sidecar_hash
        );
        return None;
    }
    Some(sidecar_path)
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

fn tensorrt_temporalfix_cumsum_workaround(model: &Path) -> bool {
    model.file_name().is_some_and(|name| {
        name.to_string_lossy()
            .to_ascii_lowercase()
            .contains("temporalfix")
    })
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
    // The in-progress marker is the authoritative completion contract. Engine,
    // timing, or cache files may already be non-zero when a build is cancelled
    // or the process exits, so their mere presence must never convert an
    // unfinished build into a reusable cache.
    std::fs::create_dir_all(cache_dir)?;
    let marker = cache_dir.join(TENSORRT_BUILD_MARKER);
    let stale_incomplete = marker.exists();
    if stale_incomplete {
        log::warn!(
            "tensorrt-cache: stale incomplete build detected path={}; removing incomplete cache even if engine artifacts exist",
            cache_dir.display()
        );
        cleanup_incomplete_tensorrt_cache(cache_dir);
        std::fs::create_dir_all(cache_dir)?;
    }
    if stale_incomplete || !cache_was_present {
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

fn temporalfix_shape_hints(bytes: &[u8]) -> Option<&'static [u8]> {
    match format!("{:x}", Sha256::digest(bytes)).as_str() {
        "90deed0d4b672c918e112a760107455a9a1689db22b5b45311b23953ab3dd35f" => Some(include_bytes!(
            "../../assets/onnx-shape-hints/temporalfix-s1-cumsum.bin"
        )),
        "9d6c7f6f3d63ebe5b4db0d715f9760c949d7dc1fe8142fce48ccfedcca62a477" => Some(include_bytes!(
            "../../assets/onnx-shape-hints/temporalfix-s2-cumsum.bin"
        )),
        _ => None,
    }
}
fn onnx_wire_varint(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*offset)
            .ok_or_else(|| anyhow!("truncated ONNX protobuf"))?;
        *offset += 1;
        anyhow::ensure!(shift != 63 || byte <= 1, "ONNX varint overflow");
        value |= u64::from(byte & 127) << shift;
        if byte & 128 == 0 {
            return Ok(value);
        }
    }
    Err(anyhow!("invalid ONNX protobuf varint"))
}
// Preserve every operator and weight byte while adding inferred value_info.
fn append_onnx_graph_metadata(bytes: &[u8], hints: &[u8]) -> Result<Vec<u8>> {
    let mut offset = 0;
    let mut graph = None;
    while offset < bytes.len() {
        let start = offset;
        let key = onnx_wire_varint(bytes, &mut offset)?;
        anyhow::ensure!(key >> 3 != 0, "invalid ONNX field");
        match key & 7 {
            0 => {
                onnx_wire_varint(bytes, &mut offset)?;
            }
            1 => {
                offset = offset
                    .checked_add(8)
                    .ok_or_else(|| anyhow!("ONNX offset overflow"))?;
            }
            2 => {
                let len = usize::try_from(onnx_wire_varint(bytes, &mut offset)?)?;
                let end = offset
                    .checked_add(len)
                    .ok_or_else(|| anyhow!("ONNX length overflow"))?;
                if key >> 3 == 7 {
                    anyhow::ensure!(graph.is_none(), "duplicate ONNX graph");
                    graph = Some((start, offset, end));
                }
                offset = end;
            }
            5 => {
                offset = offset
                    .checked_add(4)
                    .ok_or_else(|| anyhow!("ONNX offset overflow"))?;
            }
            _ => return Err(anyhow!("unsupported ONNX wire type")),
        }
        anyhow::ensure!(offset <= bytes.len(), "truncated ONNX field");
    }
    let (start, payload, end) = graph.ok_or_else(|| anyhow!("ONNX graph missing"))?;
    let mut out = Vec::with_capacity(bytes.len() + hints.len() + 10);
    out.extend_from_slice(&bytes[..start]);
    out.push(0x3a);
    let mut len = (end - payload)
        .checked_add(hints.len())
        .ok_or_else(|| anyhow!("ONNX graph length overflow"))?;
    while len >= 128 {
        out.push((len as u8 & 127) | 128);
        len >>= 7;
    }
    out.push(len as u8);
    out.extend_from_slice(&bytes[payload..end]);
    out.extend_from_slice(hints);
    out.extend_from_slice(&bytes[end..]);
    Ok(out)
}

fn stage_tensorrt_model(model: &Path, cache_dir: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(cache_dir)?;
    let staged = cache_dir.join("model.onnx");
    let patched = if tensorrt_temporalfix_cumsum_workaround(model) {
        let bytes = std::fs::read(model)?;
        if let Some(hints) = temporalfix_shape_hints(&bytes) {
            log::info!(
                "tensorrt-shape-hints: model={} metadata_bytes={}",
                model.display(),
                hints.len()
            );
            Some(append_onnx_graph_metadata(&bytes, hints)?)
        } else {
            None
        }
    } else {
        None
    };
    let source_len = patched
        .as_ref()
        .map(|bytes| bytes.len() as u64)
        .unwrap_or(std::fs::metadata(model)?.len());
    if std::fs::metadata(&staged).is_ok_and(|metadata| metadata.len() == source_len) {
        log::info!(
            "tensorrt-model-stage: source={} staged={} reused=true",
            model.display(),
            staged.display()
        );
        return Ok(staged);
    }
    let temporary = cache_dir.join("model.onnx.tmp");
    if let Some(bytes) = patched {
        std::fs::write(&temporary, bytes)?;
    } else {
        std::fs::copy(model, &temporary)?;
    }
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

/// v937: standard base64 (RFC 4648, '=' padding, whitespace ignored) for graphs embedded in model metadata.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    fn value(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some(u32::from(c - b'A')),
            b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
            b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for &c in text.as_bytes() {
        if c == b'=' {
            break;
        }
        if c.is_ascii_whitespace() {
            continue;
        }
        acc = (acc << 6) | value(c)?;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    (!out.is_empty()).then_some(out)
}

#[cfg(test)]
mod v937_base64_tests {
    #[test]
    fn decodes_padded_and_unpadded() {
        assert_eq!(super::decode_base64("TmVv").unwrap(), b"Neo");
        assert_eq!(super::decode_base64("TmVvUklGRQ==").unwrap(), b"NeoRIFE");
        assert_eq!(super::decode_base64("TmVvUklGRTI=").unwrap(), b"NeoRIFE2");
        assert!(super::decode_base64("Tm!v").is_none());
    }
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
    #[test]
    fn padded_rife_fp16_pack_matches_original_channel_scan() {
        use super::*;
        for (w, h, pw, ph, stride, base) in [(3, 2, 8, 4, 3, 0), (17, 7, 32, 16, 4, 3)] {
            let pn = pw * ph;
            let src: Vec<u8> = (0..w * h * stride)
                .map(|i| ((i * 97 + 13) % 256) as u8)
                .collect();
            let mut actual = vec![f16::ZERO; (base + 3) * pn];
            fill_padded_rgb_frame_f16(&mut actual, &src, base, w, h, pw, ph, pn, stride);
            assert!(actual[..base * pn].iter().all(|v| *v == f16::ZERO));
            let lut = u8_to_f16_lut();
            for c in 0..3 {
                for y in 0..ph {
                    for x in 0..pw {
                        let sy = y.min(h - 1);
                        let sx = x.min(w - 1);
                        assert_eq!(
                            actual[(base + c) * pn + y * pw + x],
                            lut[src[(sy * w + sx) * stride + c] as usize]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn rgbx_nchw_pack_matches_original_channel_scan() {
        use super::*;
        for (w, h, stride) in [(1, 1, 3), (7, 3, 4), (1920, 2, 3), (1024, 128, 4)] {
            let n = w * h;
            let rgba: Vec<u8> = (0..n * stride)
                .map(|i| ((i * 73 + 251) % 256) as u8)
                .collect();
            let mut actual = vec![f16::ZERO; n * 3];
            pack_rgbx_nchw_f16(&rgba, w, h, stride, &mut actual);
            let lut = u8_to_f16_lut();
            for channel in 0..3 {
                for i in 0..n {
                    assert_eq!(
                        actual[channel * n + i],
                        lut[rgba[stride * i + channel] as usize]
                    );
                }
            }
        }
    }

    #[test]
    fn temporalfix_cumsum_workaround_is_scoped_to_temporalfix_models() {
        assert!(super::tensorrt_temporalfix_cumsum_workaround(
            std::path::Path::new("models/TemporalFix/temporalfix_s2_v1_op18_fp16.onnx")
        ));
        assert!(super::tensorrt_temporalfix_cumsum_workaround(
            std::path::Path::new("TemporalFix_S1.ONNX")
        ));
        assert!(!super::tensorrt_temporalfix_cumsum_workaround(
            std::path::Path::new("models/NeoRIFE_fp16.onnx")
        ));
    }
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
    fn high_factor_slots_grow_only_above_legacy_x5() {
        assert_eq!(OnnxStage::interp_slot_count(2), 3);
        assert_eq!(OnnxStage::interp_slot_count(3), 5);
        assert_eq!(OnnxStage::interp_slot_count(4), 6);
        assert_eq!(OnnxStage::interp_slot_count(5), 7);
        for factor in 6..=15 {
            assert!(OnnxStage::interp_slot_count(factor) >= factor);
        }
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

#[cfg(test)]
mod temporalfix_shape_metadata_tests {
    use super::*;
    #[test]
    fn shipped_temporalfix_models_stage_with_complete_shape_metadata() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/TemporalFix");
        let cache =
            std::env::temp_dir().join(format!("Neo-v859-shape-test-{}", std::process::id()));
        for filename in [
            "temporalfix_s1_v1.1_op18_fp16.onnx",
            "temporalfix_s2_v1_op18_fp16.onnx",
        ] {
            let model = root.join(filename);
            let original = std::fs::read(&model).unwrap();
            let hints = temporalfix_shape_hints(&original).expect("shipped model fingerprint");
            let expected = append_onnx_graph_metadata(&original, hints).unwrap();
            let target = stage_tensorrt_model(&model, &cache.join(filename)).unwrap();
            assert_eq!(std::fs::read(&target).unwrap(), expected);
            assert_eq!(
                stage_tensorrt_model(&model, &cache.join(filename)).unwrap(),
                target
            );
            assert_eq!(std::fs::read(&model).unwrap(), original);
            println!("STAGED_ONNX {}", target.display());
        }
    }
    #[test]
    fn shape_metadata_preserves_graph_and_other_fields() {
        // Model ir_version, graph with an opaque payload, producer_name.
        let original = [0x08, 0x09, 0x3a, 0x03, 0x08, 0x07, 0x01, 0x12, 0x01, b'x'];
        let patched = append_onnx_graph_metadata(&original, &[0x6a, 0x01, 0x42]).unwrap();
        assert_eq!(
            patched,
            [
                0x08, 0x09, 0x3a, 0x06, 0x08, 0x07, 0x01, 0x6a, 0x01, 0x42, 0x12, 0x01, b'x'
            ]
        );
        assert_eq!(
            append_onnx_graph_metadata(&original, &[]).unwrap(),
            original
        );
        assert!(temporalfix_shape_hints(&original).is_none());
    }
    #[test]
    fn shape_metadata_rejects_truncated_and_ambiguous_graphs() {
        for invalid in [
            &[0x3a, 0x04, 0x01][..],
            &[0x80][..],
            &[0x3a, 0, 0x3a, 0][..],
            &[0x08, 0x01][..],
        ] {
            assert!(append_onnx_graph_metadata(invalid, &[1]).is_err());
        }
    }
}

impl OnnxStage {
    fn retire_neoarc_interp_gl(&mut self,gc:&mut GlContext) {
        if let Some(mut bridge)=self.neoarc_interp_gl.take() {
            gc.finish();let keys:Vec<_>=bridge.outputs.iter().map(|(key,_)|key.clone()).collect();
            // CL imports of these buffers are freed when the bridge IO is parked
            // or dropped below; the GL names are deleted only after that.
            let buffers=std::mem::take(&mut bridge.input_buffers);
            if let Some(program)=bridge.endpoint_program.take(){unsafe{use glow::HasContext;gc.gl.delete_program(program);}}
            if let Some(session)=self.neoarc_session.as_ref(){session.park_gl_interp(bridge.padded.0 as usize,bridge.padded.1 as usize,bridge.context.0,bridge.context.1,bridge.io);}
            else{drop(bridge);}
            for key in keys{gc.remove_persistent_texture(&key);}
            for buffer in buffers{unsafe{use glow::HasContext;gc.gl.delete_buffer(buffer);}}
        }
    }
    fn prepare_neoarc_interp_gl(&mut self,gc:&mut GlContext,frames:&[GpuTex],phases:&[f32],capacity:usize)->Result<u64> {
        use windows::Win32::Graphics::OpenGL::{wglGetCurrentContext,wglGetCurrentDC};
        let required=self.interp_frames();let channels=if required==4{14}else{7};
        let frames=&frames[frames.len().checked_sub(required).ok_or_else(||anyhow!("NeoArc frame history incomplete"))?..];
        let size=(frames[0].w(),frames[0].h());
        anyhow::ensure!(frames.iter().all(|t|(t.w(),t.h())==size),"NeoArc frame geometry differs");
        let padded=if required==4{((size.0+63)/64*64,(size.1+63)/64*64)}else{size};
        let context=(unsafe{wglGetCurrentContext()}.0 as isize,unsafe{wglGetCurrentDC()}.0 as isize);
        let count=capacity.saturating_sub(1).max(phases.len());
        let want_boost = channels == 7 && capacity >= 3 && directml_rife_boost_setting().load(Ordering::Relaxed)
            && std::env::var("CHIDESCALER_NEOARC_RIFE_BOOST").ok().as_deref() != Some("off");
        if self.neoarc_interp_gl.as_ref().is_some_and(|b|b.size!=size||b.context!=context || (want_boost && !b.boost_attempted)) {
            self.retire_neoarc_interp_gl(gc);
        }
        if self.neoarc_interp_gl.is_none() {
            // v963: RIFE boost graphs from the original ONNX (model IO precision kept).
            let boost_graphs=if want_boost {
                let built=(||->Result<(Vec<u8>,Vec<u8>,bool)>{
                    let bytes=std::fs::read(&self.source_path)?;let fp16=rife_dml_split::input_is_half(&bytes)?;
                    let (trunk, head) = rife_dml_split::split(&bytes,i64::from(padded.1),i64::from(padded.0),fp16)?;
                    // v983: plain Lite final stage -> NeoArc may run the head as one kernel.
                    let native=rife_dml_split::native_head_compatible(&bytes,fp16);
                    if fp16 { Ok((trunk, head, native)) }
                    else { Ok((rife_dml_split::half_bridge(&trunk, false)?, rife_dml_split::half_bridge(&head, true)?, native)) }
                })();
                match built{Ok(v)=>Some(v),Err(error)=>{log::warn!("neoarc-rife-boost-fallback: model={} reason={error:#} stage=split action=original-gpu-graph",self.name);None}}
            }else{None};
            let session=self.neoarc_session.as_mut().ok_or_else(||anyhow!("NeoArc session missing"))?;
            match boost_graphs{
                Some((trunk,head,native))=>if let Err(error)=session.set_rife_boost_graphs(padded.0 as usize,padded.1 as usize,&trunk,&head,native){
                    session.clear_rife_boost();log::warn!("neoarc-rife-boost-fallback: reason={error:#} stage=write action=original-gpu-graph");},
                None=>session.clear_rife_boost(),
            }
            let mut outputs=Vec::new();let id=DIRECT_OUTPUT_KEY.fetch_add(1,Ordering::Relaxed);
            for index in 0..count {
                let key=format!("neoarc-interp-{id}-{index}");
                let tex=gc.persistent_texture(&key,padded.0,padded.1,1,4,super::gl::Dtype::U8,false,glow::CLAMP_TO_EDGE,&[]);
                outputs.push((key,tex));
            }
            // One-time canonical validation only. Steady-state input/output never maps pixels.
            let n=padded.0 as usize*padded.1 as usize;let mut packed=vec![f16::ONE;channels*n];
            for(index,frame)in frames.iter().enumerate() {
                let pixels=gc.download_rgb8(*frame);
                for channel in 0..3 {for y in 0..padded.1 as usize{for x in 0..padded.0 as usize {
                    let src=(y.min(size.1 as usize-1)*size.0 as usize+x.min(size.0 as usize-1))*3+channel;
                    packed[(index*3+channel)*n+y*padded.0 as usize+x]=f16::from_f32(pixels[src]as f32*(1.0/255.0));
                }}}
            }
            gc.finish();let ids:Vec<_>=frames.iter().map(|t|t.tex.0.get()).collect();
            let io=session.create_gl_interp(padded.0 as usize,padded.1 as usize,context.0,context.1,&packed,&ids,outputs[0].1.tex.0.get());
            let io=match io {Ok(io)=>io,Err(error)=>{for(key,_)in outputs{gc.remove_persistent_texture(&key);}return Err(error)}};
            self.neoarc_interp_gl=Some(NeoArcInterpGl{io,boost_attempted:want_boost,capacity,size,padded,context,frames:ids,phases:phases.to_vec(),outputs,input_buffers:Vec::new(),buffer_input_failed:false,endpoint_program:None,endpoint_compute_failed:false});
        }
        let bridge=self.neoarc_interp_gl.as_mut().unwrap();
        // Growing phase capacity does not invalidate compiled tensors or the
        // completed phase-repeat validation. Old slots keep their ownership.
        // Fractional cadence can request 3 then 2 then 3 phases while the
        // user's capacity stays x3. Keep those bounded slots warm; repeatedly
        // destroying/re-importing the third GL/CL surface caused avoidable
        // allocation/synchronization pressure. Explicit capacity reductions
        // still release excess surfaces.
        if bridge.outputs.len()>count&&capacity<bridge.capacity {
            for(key,_)in bridge.outputs.drain(count..){gc.remove_persistent_texture(&key);}
        }
        bridge.capacity=capacity;
        if bridge.outputs.len()<count {
            let id=DIRECT_OUTPUT_KEY.fetch_add(1,Ordering::Relaxed);
            for index in bridge.outputs.len()..count {
                let key=format!("neoarc-interp-{id}-{index}");
                let tex=gc.persistent_texture(&key,padded.0,padded.1,1,4,super::gl::Dtype::U8,false,glow::CLAMP_TO_EDGE,&[]);
                bridge.outputs.push((key,tex));
            }
        }
        bridge.frames=frames.iter().map(|t|t.tex.0.get()).collect();bridge.phases=phases.to_vec();
        // A prepare call owns one immutable source pair/window. Never reuse
        // image planes across prepares, texture IDs, source frames or resizes.
        let output_ids:Vec<_>=bridge.outputs.iter().take(phases.len()).map(|(_,t)|(t.tex.0.get(),t.allocation_id)).collect();
        let input_ids:Vec<_>=frames.iter().map(|t|(t.tex.0.get(),t.allocation_id)).collect();
        // v963: boost choice fixed per pair (x2.5 and up, every pair alike).
        let boost=want_boost;
        bridge.io.set_boost(boost);DML_RIFE_SPLIT_PAIR.store(boost&&bridge.io.has_boost(),Ordering::Relaxed);
        // v985: copy the two endpoints into GL pixel buffers (GPU copy, ordered
        // before the fence below) and let OpenCL read those buffers.
        let buffered=channels==7&&!bridge.buffer_input_failed&&bridge.io.supports_buffer_input()
            &&std::env::var("CHIDESCALER_NEOARC_GL_INPUT").ok().as_deref().map(str::trim)!=Some("image")
            &&match neoarc_copy_endpoints_to_buffers(gc,bridge,frames){
                Ok(())=>true,
                Err(error)=>{bridge.buffer_input_failed=true;log::warn!("neoarc-gl-buffer-input-disabled: stage=copy reason={error:#} action=gl-image-input");false}
            };
        let prepared=if buffered{
            let names:Vec<u32>=bridge.input_buffers.iter().map(|b|b.0.get()).collect();
            match bridge.io.prepare_pair_buffers(&names,&output_ids){
                Ok(())=>Ok(()),
                Err(error)=>{bridge.buffer_input_failed=true;log::warn!("neoarc-gl-buffer-input-disabled: stage=import reason={error:#} action=gl-image-input");bridge.io.prepare_pair(&input_ids,&output_ids)}
            }
        }else{bridge.io.prepare_pair(&input_ids,&output_ids)};
        prepared?;
        gc.submit_commands_fence().map_err(anyhow::Error::msg)
    }
}


/// v985: GPU copy of the two RIFE endpoint textures into persistent GL pixel
/// buffers as RGBA FP16 (exact values of RGBA16F sources; RGBA8 sources become
/// half(k/255), which the CL quantization maps back to k). Commands are only
/// queued; the caller's fence orders them before the OpenCL acquire.
fn neoarc_copy_endpoints_to_buffers(gc:&mut GlContext,bridge:&mut NeoArcInterpGl,frames:&[GpuTex])->Result<()>{
    use glow::HasContext;
    anyhow::ensure!(frames.len()==2,"NeoArc buffer input needs two endpoints");
    let bytes=bridge.size.0 as usize*bridge.size.1 as usize*8;
    anyhow::ensure!(frames.iter().all(|t|(t.w(),t.h())==bridge.size),"NeoArc buffer input geometry differs");
    let gl=gc.gl.clone();
    unsafe{
        while bridge.input_buffers.len()<2{
            let buffer=gl.create_buffer().map_err(|e|anyhow!("create GL input buffer: {e}"))?;
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER,Some(buffer));
            gl.buffer_data_size(glow::PIXEL_PACK_BUFFER,i32::try_from(bytes)?,glow::STREAM_COPY);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER,None);
            bridge.input_buffers.push(buffer);
        }
        // Drop stale error flags from unrelated earlier GL work.
        for _ in 0..8{if gl.get_error()==glow::NO_ERROR{break;}}
        // v992: write the endpoints with a compute shader (texelFetch ->
        // packHalf2x16) instead of glGetTexImage into a pixel-pack buffer. The
        // pack path took ~9 ms to complete on Arc (the driver's external
        // readback of a GL texture); a compute pass reads it like any shader.
        // Same RGBA FP16 bytes per pixel. Falls back to the pack path.
        let use_compute=!bridge.endpoint_compute_failed
            &&std::env::var("CHIDESCALER_NEOARC_ENDPOINT_COPY").ok().as_deref()!=Some("pack");
        let program=if use_compute{
            if bridge.endpoint_program.is_none(){
                match neoarc_endpoint_compute_program(&gl){
                    Ok(program)=>bridge.endpoint_program=Some(program),
                    Err(error)=>{bridge.endpoint_compute_failed=true;log::warn!("neoarc-endpoint-compute-disabled: reason={error:#} action=pixel-pack-copy");}
                }
            }
            bridge.endpoint_program
        }else{None};
        if let Some(program)=program{
            gl.use_program(Some(program));
            let size=gl.get_uniform_location(program,"u_size");let src=gl.get_uniform_location(program,"u_src");
            gl.uniform_2_i32(size.as_ref(),bridge.size.0,bridge.size.1);gl.uniform_1_i32(src.as_ref(),0);
            gl.active_texture(glow::TEXTURE0);
            for(frame,buffer)in frames.iter().zip(bridge.input_buffers.iter()){
                gl.bind_texture(glow::TEXTURE_2D,Some(frame.tex));
                gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER,0,Some(*buffer));
                gl.dispatch_compute((bridge.size.0 as u32).div_ceil(16),(bridge.size.1 as u32).div_ceil(16),1);
            }
            gl.memory_barrier(glow::ALL_BARRIER_BITS);
            gl.bind_buffer_base(glow::SHADER_STORAGE_BUFFER,0,None);
            gl.bind_texture(glow::TEXTURE_2D,None);
            gl.use_program(None);
        }else{
            gl.pixel_store_i32(glow::PACK_ALIGNMENT,4);
            for(frame,buffer)in frames.iter().zip(bridge.input_buffers.iter()){
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER,Some(*buffer));
                gl.bind_texture(glow::TEXTURE_2D,Some(frame.tex));
                gl.get_tex_image(glow::TEXTURE_2D,0,glow::RGBA,glow::HALF_FLOAT,glow::PixelPackData::BufferOffset(0));
            }
            gl.bind_texture(glow::TEXTURE_2D,None);
            gl.bind_buffer(glow::PIXEL_PACK_BUFFER,None);
        }
        let error=gl.get_error();
        anyhow::ensure!(error==glow::NO_ERROR,"GL endpoint copy error 0x{error:04x}");
    }
    Ok(())
}

/// v992: compute program that stores one RGBA texture as packed FP16 RGBA
/// (two uints per pixel, little-endian r,g then b,a) into SSBO binding 0.
fn neoarc_endpoint_compute_program(gl:&glow::Context)->Result<glow::Program>{
    use glow::HasContext;
    const SOURCE:&str="#version 430\nlayout(local_size_x=16,local_size_y=16) in;\nuniform sampler2D u_src;\nuniform ivec2 u_size;\nlayout(std430,binding=0) writeonly buffer Endpoint{uvec2 px[];};\nvoid main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(p.x>=u_size.x||p.y>=u_size.y)return;vec4 v=texelFetch(u_src,p,0);px[p.y*u_size.x+p.x]=uvec2(packHalf2x16(v.rg),packHalf2x16(v.ba));}\n";
    unsafe{
        let shader=gl.create_shader(glow::COMPUTE_SHADER).map_err(|e|anyhow!("create compute shader: {e}"))?;
        gl.shader_source(shader,SOURCE);gl.compile_shader(shader);
        if !gl.get_shader_compile_status(shader){let log=gl.get_shader_info_log(shader);gl.delete_shader(shader);anyhow::bail!("compile endpoint compute: {log}");}
        let program=gl.create_program().map_err(|e|anyhow!("create compute program: {e}"))?;
        gl.attach_shader(program,shader);gl.link_program(program);gl.detach_shader(program,shader);gl.delete_shader(shader);
        if !gl.get_program_link_status(program){let log=gl.get_program_info_log(program);gl.delete_program(program);anyhow::bail!("link endpoint compute: {log}");}
        Ok(program)
    }
}

fn current_thread_cpu_100ns()->Option<u64>{
    use windows::Win32::{Foundation::FILETIME,System::Threading::{GetCurrentThread,GetThreadTimes}};
    let(mut creation,mut exit,mut kernel,mut user)=(FILETIME::default(),FILETIME::default(),FILETIME::default(),FILETIME::default());
    unsafe{GetThreadTimes(GetCurrentThread(),&mut creation,&mut exit,&mut kernel,&mut user).ok()?;}
    let ticks=|v:FILETIME|((v.dwHighDateTime as u64)<<32)|v.dwLowDateTime as u64;Some(ticks(kernel).saturating_add(ticks(user)))
}
#[derive(Default)]
struct GpuInterpDiagnostics{key:Option<(&'static str,(i32,i32),(i32,i32))>,rows:Vec<[f64;4]>}
impl GpuInterpDiagnostics{
 fn record(&mut self,model:&str,backend:&'static str,input:(i32,i32),tensor:(i32,i32),row:[f64;4]){
  if !row.iter().all(|v|v.is_finite()&&*v>=0.){return;}
  let key=(backend,input,tensor);if self.key!=Some(key){self.key=Some(key);self.rows.clear();}
  self.rows.push(row);if self.rows.len()<120{return;}
  let q=|col:usize,p:f64|gpu_interp_quantile(&self.rows,col,p);
  log::info!("interp-gpu-stage-window: model={} backend={} input={}x{} tensor={}x{} samples={} pack_p50_ms={:.3} pack_p95_ms={:.3} infer_p50_ms={:.3} infer_p95_ms={:.3} infer_max_ms={:.3} output_p50_ms={:.3} output_p95_ms={:.3} slot_wall_p50_ms={:.3} slot_wall_p95_ms={:.3} slot_wall_max_ms={:.3} slots_over_20ms={} timing=completed-host-wall-not-pure-gpu pack_includes_owner_input=true slot_wall_excludes_owner_prepare=true display_statistics_unchanged=true",model,backend,input.0,input.1,tensor.0,tensor.1,self.rows.len(),q(0,0.5),q(0,0.95),q(1,0.5),q(1,0.95),q(1,1.),q(2,0.5),q(2,0.95),q(3,0.5),q(3,0.95),q(3,1.),self.rows.iter().filter(|r|r[3]>20.).count());
  self.rows.clear();
 }
}
fn gpu_interp_quantile(rows:&[[f64;4]],column:usize,p:f64)->f64{let mut values:Vec<_>=rows.iter().map(|r|r[column]).collect();values.sort_by(f64::total_cmp);let i=((values.len()-1)as f64*p).ceil()as usize;values[i]}
#[cfg(test)]
mod gpu_interp_diagnostic_tests{
 use super::*;
 #[test]fn quantiles_keep_real_tail(){let rows=[[0.,7.,0.,8.],[0.,8.,0.,9.],[0.,30.,0.,31.]];assert_eq!(gpu_interp_quantile(&rows,1,0.5),8.);assert_eq!(gpu_interp_quantile(&rows,1,0.95),30.);}
 #[test]fn changing_shape_or_backend_does_not_mix_windows(){let mut d=GpuInterpDiagnostics::default();d.record("m","DirectML-WGL",(960,540),(1024,640),[1.;4]);d.record("m","DirectML-WGL",(1280,720),(1280,768),[2.;4]);assert_eq!(d.rows,vec![[2.;4]]);d.record("m","NeoARC",(1280,720),(1280,720),[3.;4]);assert_eq!(d.rows,vec![[3.;4]]);}
}


impl OnnxStage{
 /// Headless diagnostic of the production detach/worker handshake. Deadlines
 /// are simulated; this does not operate a player or claim monitor FPS.
 pub fn diagnostic_paced_gpu_pair(&mut self,gc:&mut GlContext,frames:&[GpuTex],phases:&[f32],factor:usize,period:Duration,early:bool)->Result<(Vec<GpuTex>,Vec<[f64;4]>)>{
  self.diagnostic_paced_gpu_pair_with_post(gc,frames,phases,factor,period,early,None,(frames[0].w(),frames[0].h()))
 }
 pub fn diagnostic_paced_gpu_pair_with_post(&mut self,gc:&mut GlContext,frames:&[GpuTex],phases:&[f32],factor:usize,period:Duration,early:bool,mut post:Option<&mut super::chain::FilterChain>,out_size:(i32,i32))->Result<(Vec<GpuTex>,Vec<[f64;4]>)>{
  let fence=self.prepare_interp_gpu_textures_with_factor(gc,frames,phases,1,factor,true)?.ok_or_else(||anyhow!("GPU path unavailable"))?;
  while !gc.poll_commands_fence(fence).map_err(anyhow::Error::msg)?{std::thread::yield_now();}
  self.stage_neoarc_interp_inputs();
  let finish_post_first=post.is_some()&&(std::env::var("CHIDESCALER_INTEL_INTERP_POST_READY").ok().as_deref()==Some("on")||(factor==3&&self.dml_wgl_interp_bridge.is_some()&&std::env::var("CHIDESCALER_DML_INTERP_POST_PRIORITY").ok().as_deref()!=Some("off")));
  let slots=self.prepared_interp_gpu_outputs(phases.len(),false)?;let count=phases.len();let start=Instant::now();
  let mut tracer=if std::env::var("CHIDESCALER_INTERP_GPU_TIMING_BENCH").ok().as_deref()==Some("on"){Some(super::interp_timing::AsyncInterpTiming::for_probe(gc.gl.clone()))}else{None};
  std::thread::scope(|scope|->Result<_>{
   let(ready_tx,ready_rx)=std::sync::mpsc::channel();let(permit_tx,permit_rx)=std::sync::mpsc::channel();
   let worker=scope.spawn(move||->Result<()>{for index in 0..count{let(run_ms,_)=self.run_prepared_interp_gpu_slot(index)?;ready_tx.send((index,run_ms,start.elapsed().as_secs_f64()*1000.))?;if index+1<count{permit_rx.recv()?;}}Ok(())});
   let result=(||->Result<_>{let mut images=Vec::new();let mut timings=Vec::new();for index in 0..count{
    let(done,run_ms,ready_ms)=ready_rx.recv()?;anyhow::ensure!(done==index,"diagnostic slot order");let mut output=Self::finish_prepared_interp_gpu_output(gc,slots[index])?;
    let deadline=start+period.mul_f64((index+1)as f64);
    if !early{let prepare=deadline.checked_sub(Duration::from_micros(500)).unwrap_or(deadline);if let Some(wait)=prepare.checked_duration_since(Instant::now()){std::thread::sleep(wait);}}
    let sample=tracer.as_mut().and_then(|tracer|{tracer.poll();tracer.begin(index as u64,0)});
    if let Some(chain)=post.as_deref_mut(){output=chain.diagnostic_interp_gpu_suffix(gc,output,out_size)?;}
    if let Some(id)=sample{tracer.as_mut().unwrap().end(id,true);}
    unsafe{use glow::HasContext;gc.gl.flush();}
    if early&&finish_post_first{gc.finish();}
    if index+1<count{permit_tx.send(())?;}
    gc.finish();let filtered_ready=start.elapsed().as_secs_f64()*1000.;
    if let Some(wait)=deadline.checked_duration_since(Instant::now()){std::thread::sleep(wait);}
    if let Some(id)=sample{tracer.as_mut().unwrap().observe_swap(id);tracer.as_mut().unwrap().poll();}
    timings.push([run_ms,ready_ms,Instant::now().saturating_duration_since(deadline).as_secs_f64()*1000.,filtered_ready]);images.push(output);
   }Ok((images,timings))})();drop(permit_tx);drop(ready_rx);
   let work=worker.join().map_err(|_|anyhow!("diagnostic worker panic"))?;result.and_then(|r|work.map(|_|r))
  })
 }
}

impl OnnxStage{
 pub fn diagnostic_interp_output_storage_ids(&self)->Vec<(u64,u32)>{
  if let Some(bridge)=self.neoarc_interp_gl.as_ref(){return bridge.outputs.iter().map(|(_,t)|(t.allocation_id,t.tex.0.get())).collect();}
  if let Some(bridge)=self.dml_wgl_interp_bridge.as_ref(){return bridge.slots.iter().map(|s|(s.key,0)).collect();}
  Vec::new()
 }
}
