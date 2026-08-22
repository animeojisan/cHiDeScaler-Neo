//! Resident render engine thread.
//!
//! The GL context, overlay window and stage factory are created once at app
//! startup and remain alive until shutdown. Start/stop only reconfigures
//! capture, the filter chain, and overlay visibility.

use crate::capture::wgc::{FrameBuf, WgcSource};
use crate::core::config::{HdrSdrMode, OnnxBackendPreference, ScaleMode, StageKind, StageSpec};
use crate::core::metrics::Metrics;
use crate::overlay::window::OverlayWindow;
use crate::platform::win32;
use crate::render::chain::{FilterChain, StageFactory};
use crate::render::gl::{GlContext, GpuTex};
use crate::render::onnx_stage::PreparedInterpGpuOutput;
use anyhow::Result;
use glow::HasContext;
use half::f16;
use rayon::prelude::*;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Receiver, Sender, SyncSender, TrySendError, channel, sync_channel};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

// Motion interpolation needs adjacent frames, but a deep FIFO silently turns
// into A/V drift because audio belongs to the source application and cannot be
// delayed with our overlay. Keep a very short queue that can absorb WGC's 24p
// compositor bursts; sustained overload is handled by dropping stale frames.
// NeoFlow consumes one adjacent pair at a time. Keeping two frames after every
// dequeue made that queue a permanent 80ms A/V offset once presentation was
// paced exactly to the source cadence. Keep only the newest pending frame;
// sequence-gap detection below safely skips interpolation if WGC really bursts.
const NEOFLOW_QUEUE_MAX: usize = 1;
const ONNX_INTERP_QUEUE_MAX: usize = 3;
const SMOOTH_PACING_QUEUE_MAX: usize = 1;
const LOAD_REDUCTION_QUEUE_MAX: usize = 2;
// Smooth pacing does not need adjacent source pairs. Keep only the newest
// pending frame so compositor bursts cannot create a hidden full-frame backlog.
const NEOFLOW_FULL_DELAY_FRAMES: f64 = 5.0;
const NEOFLOW_HARD_SKIP_DELAY_FRAMES: f64 = 5.25;
const IDLE_KEEPALIVE_INTERVAL: Duration = Duration::from_secs(1);
// A capture-size change resizes the foreign HWND on the GUI thread before the
// render command can be dequeued. During that tiny hand-off window a normal
// PIP/mpv window can already cover the monitor and look exactly like a source
// fullscreen toggle. Give the geometry command time to arrive before deciding
// that monitor coverage is application-owned.
const FULLSCREEN_ENTER_GEOMETRY_GRACE: Duration = Duration::from_millis(400);
// Temporarily frozen: capture the cadence exactly as WGC/the source app
// presents it, including browser-video fullscreen. The detector remains
// dormant for a possible later opt-in design.
const AUTO_CONTENT_CADENCE_ENABLED: bool = false;
/// A half-second at 30fps (quarter-second at 60fps) proves a genuinely held
/// picture without weakening any of the motion/zoom/fade comparisons.
const STRICT_STATIC_HOLD_FRAMES: u16 = 15;

// DirectML RIFE becomes both impractically expensive and, on the reproduced
// NVIDIA path, can return corrupted high-resolution midpoints when a preceding
// upscale chain raises its input to ~1920p.  Match the proven mpv/VapourSynth
// practice: cap only DirectML RIFE input height, preserve aspect ratio, and let
// the ordinary post-chain/final display scaler handle the capped result.
// TensorRT/CUDA and non-RIFE interpolation are intentionally untouched.
const DIRECTML_RIFE_MAX_HEIGHT: i32 = 1440;

// v348b-v348e: low-end responsiveness guard for filter chains containing GLSL.
// The GUI owns the existing PDH sampler; the render library receives only its
// latest coarse percentage via this atomic handoff. v348c added GUI-priority
// admission after overload is proven; v348d also covers ordinary ONNX+GLSL.
// v348e extends protection to interpolation routes only when post-interpolation
// GLSL exists, without changing healthy interpolation scheduling. Keep
// resource_monitor binary-local exactly as in v347.
const GLSL_GUARD_SOFT_GPU_PERCENT: f32 = 86.0;
const GLSL_GUARD_HARD_GPU_PERCENT: f32 = 94.0;
const GLSL_GUARD_RELEASE_GPU_PERCENT: f32 = 82.0;
// v348h: once a processed frame proves that saturation is also missing the
// presentation deadline, do not leave the GUI sluggish for another half
// second before arming protection. Mild/transient misses still use a short
// confirmation hold; severe misses have a one-frame fast path below.
const GLSL_GUARD_SOFT_HOLD: Duration = Duration::from_millis(220);
const GLSL_GUARD_HARD_HOLD: Duration = Duration::from_millis(90);
const GLSL_GUARD_RELEASE_HOLD: Duration = Duration::from_millis(1200);
// v348o: a live filter/preset replacement performs resource teardown, texture-pool
// trimming, provider/shared-buffer reconnects and first-use shader/ONNX warmup.
// Those transition frames are not representative of steady-state load. Ignore
// overload admission for roughly one second after each successful chain apply,
// then judge only the settled frames. A truly catastrophic frame that itself
// takes longer than the settle window is still judged as soon as it returns.
const GLSL_GUARD_CHAIN_SETTLE: Duration = Duration::from_millis(1000);
// After the one-second settle window, keep the one-frame emergency shortcut
// disabled briefly so steady overload must survive the ordinary 90/220ms
// confirmation holds. This avoids latching on one last cache/compositor spike
// exactly at the settle boundary while still reacting at about 1.1-1.2s.
const GLSL_GUARD_CHAIN_CONFIRM: Duration = Duration::from_millis(300);
// v348p: capture-resolution changes alter the workload enough that the old
// guard state/history is no longer evidence for the new geometry. Keep the
// warning visible while the resized provider/GPU clocks settle, but evaluate
// the new geometry exactly like a fresh capture session: no old frame-hold, no
// geometry-independent fast-reconfirm, and the ordinary guard thresholds.
const GLSL_GUARD_RESOLUTION_RECHECK_SETTLE: Duration = Duration::from_millis(1000);
const GLSL_GUARD_RESOLUTION_RECHECK_TIMEOUT: Duration = Duration::from_secs(8);
// Once the existing overload detector has *proven* the current GLSL
// workload is beyond the machine, give the user a readable warning interval
// and then use the ordinary Stop route. While that grace period is active,
// non-interpolation GLSL chains hold the last complete filtered frame instead
// of continuing to saturate the GPU. This does not alter the overload
// thresholds or healthy-frame scheduling.
const GLSL_OVERLOAD_AUTO_STOP_GRACE: Duration = Duration::from_secs(6);
// GUI GPU usage is still sampled for the ordinary non-interpolation GLSL
// admission guard. Interpolation chains never omit selected post filters.
static GUI_GPU_PERCENT_X10: AtomicU32 = AtomicU32::new(u32::MAX);

// v528: the native panel host is positioned by the render thread, while lurk
// restore is prepared atomically by the GUI thread. During the few milliseconds
// before Cmd::SetPanelState(chip=false) reaches the render FIFO, the render
// thread may still hold panel_chipped=true and can otherwise shrink the host
// back to stale lurk-chip geometry after the GUI has already restored/drawn
// the full bar.
// This guard is deliberately narrower than making all panel state lock-free:
// it only suppresses that stale chip geometry until the queued final state is
// consumed by the render thread.
static PANEL_LURK_RESTORE_GEOMETRY_GUARD: AtomicBool = AtomicBool::new(false);

pub fn begin_panel_lurk_restore_geometry_guard() {
    PANEL_LURK_RESTORE_GEOMETRY_GUARD.store(true, Ordering::Release);
}

fn panel_lurk_restore_geometry_guard_active() -> bool {
    PANEL_LURK_RESTORE_GEOMETRY_GUARD.load(Ordering::Acquire)
}

/// Publish the GUI's already-sampled GPU percentage to the render engine.
/// This is intentionally a scalar handoff: no PDH object crosses threads or
/// crate/module boundaries, and the v347 ResourceMonitor lifetime is unchanged.
pub fn set_gui_gpu_percent(value: Option<f32>) {
    let raw = value
        .filter(|value| value.is_finite())
        .map(|value| (value.clamp(0.0, 100.0) * 10.0).round() as u32)
        .unwrap_or(u32::MAX);
    GUI_GPU_PERCENT_X10.store(raw, Ordering::Relaxed);
}

fn latest_gui_gpu_percent() -> Option<f32> {
    let raw = GUI_GPU_PERCENT_X10.load(Ordering::Relaxed);
    (raw != u32::MAX).then_some(raw as f32 / 10.0)
}

// Safety fallback for an unexpected Rgba16F frame. The normal RGBA8
// HDR option deliberately requests Rgba8 and uses the lightweight byte-domain
// highlight knee below, so these Low400 constants are not on the hot path.
const SCRGB_REFERENCE_WHITE_NITS: f32 = 80.0;
const HDR_ASSUMED_PEAK_NITS: f32 = 400.0;
const HDR_OUTPUT_PAPER_WHITE: f32 = 0.66;
const HDR_SHOULDER_STRENGTH: f32 = 2.20;
const HDR_NEUTRAL_WHITE_TARGET: f32 = 0.93;
const HDR_NEUTRAL_WHITE_STRENGTH: f32 = 0.90;
const HDR_NEUTRAL_RECOVERY_FADE_START_NITS: f32 = 240.0;
const HDR_NEUTRAL_RECOVERY_FADE_END_NITS: f32 = 360.0;
const HDR_OUTPUT_LIMIT: f32 = 0.999_5;

/// 12-bit peak lookup over 0..16 scRGB (0..1280 nits). Linear interpolation
/// between entries preserves very small same-colour brightness differences.
const HDR_TONEMAP_LUT_STEPS_PER_SCRGB: usize = 4096;
const HDR_TONEMAP_LUT_MAX_SCRGB: usize = 16;
const HDR_TONEMAP_LUT_LEN: usize = HDR_TONEMAP_LUT_STEPS_PER_SCRGB * HDR_TONEMAP_LUT_MAX_SCRGB + 1;
const SRGB_ENCODE_LUT_LEN: usize = 65_536;

// Fast HDR-option path: capture exactly like HDR OFF (RGBA8) and apply only a
// gentle, hue-preserving soft knee to the top end. This cannot reconstruct
// values already clipped by DWM, but it removes the harsh 255-white appearance
// without the expensive Rgba16F -> CPU tone-mapping pass.
const SDR_HIGHLIGHT_KNEE_U8: u8 = 232;
const SDR_HIGHLIGHT_CEILING_U8: u8 = 248;
const SDR_HIGHLIGHT_MAX_CHROMA_U8: u8 = 48;

fn hdr_shoulder_exact(source_peak: f32) -> f32 {
    let source_peak_nits = source_peak.max(0.0) * SCRGB_REFERENCE_WHITE_NITS;
    if source_peak_nits <= SCRGB_REFERENCE_WHITE_NITS {
        (source_peak_nits / SCRGB_REFERENCE_WHITE_NITS) * HDR_OUTPUT_PAPER_WHITE
    } else {
        let shoulder_range = (HDR_ASSUMED_PEAK_NITS - SCRGB_REFERENCE_WHITE_NITS).max(1.0);
        let shoulder_position = (source_peak_nits - SCRGB_REFERENCE_WHITE_NITS) / shoulder_range;
        HDR_OUTPUT_PAPER_WHITE
            + (1.0 - HDR_OUTPUT_PAPER_WHITE)
                * (1.0 - (-HDR_SHOULDER_STRENGTH * shoulder_position).exp())
    }
    .clamp(0.0, HDR_OUTPUT_LIMIT)
}

fn hdr_tonemap_lut() -> &'static [f32] {
    static LUT: OnceLock<Box<[f32]>> = OnceLock::new();
    LUT.get_or_init(|| {
        (0..HDR_TONEMAP_LUT_LEN)
            .map(|index| hdr_shoulder_exact(index as f32 / HDR_TONEMAP_LUT_STEPS_PER_SCRGB as f32))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    })
    .as_ref()
}

#[inline]
fn hdr_shoulder_lut(source_peak: f32) -> f32 {
    let lut = hdr_tonemap_lut();
    let scaled = source_peak.max(0.0) * HDR_TONEMAP_LUT_STEPS_PER_SCRGB as f32;
    let lower = (scaled as usize).min(HDR_TONEMAP_LUT_LEN - 1);
    let upper = (lower + 1).min(HDR_TONEMAP_LUT_LEN - 1);
    let frac = if lower == upper {
        0.0
    } else {
        scaled - lower as f32
    };
    lut[lower] + (lut[upper] - lut[lower]) * frac
}

fn hdr_tonemap_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let available = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(4);
        // Four workers are enough for 1080p conversion after the LUT rewrite,
        // while preventing the HDR stage from occupying every logical core.
        let workers = available.clamp(1, 4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|index| format!("neo-hdr-tonemap-{index}"))
            .build()
            .expect("HDR tonemap thread pool")
    })
}

#[derive(Debug)]
pub enum Cmd {
    Start {
        hwnd: isize,
        specs: Vec<StageSpec>,
        mode: ScaleMode,
        ratio: f32,
        fps_cap: Option<u32>,
        hide_source: bool,
        client_only: bool,
        hdr: bool,
        hdr_sdr_mode: HdrSdrMode,
        gpu_adapter: Option<i32>,
        source_restore_rect: Option<(i32, i32, i32, i32)>,
        source_was_maximized: bool,
        /// ((requested w/h), (aspect-safe applied w/h)); performed only after
        /// the first filtered frame covers the source.
        deferred_capture_resolution: Option<((u32, u32), (u32, u32))>,
        capture_canvas: Option<(u32, u32)>,
    },
    Stop,
    /// Wake the render thread after an out-of-band Stop request. Unlike Stop,
    /// this token has no teardown side effects of its own, so it can safely
    /// remain queued after the priority lane has already done the work.
    WakeForStop,
    /// Save the most recently presented, fully filtered frame.
    SaveScreenshot(std::path::PathBuf),
    ApplyChain(Vec<StageSpec>),
    SwitchOnnxBackend {
        backend: OnnxBackendPreference,
        trt_device_id: Option<i32>,
        cache_root: std::path::PathBuf,
        specs: Vec<StageSpec>,
    },
    SetMode {
        mode: ScaleMode,
        ratio: f32,
    },
    /// Transactional live capture-resolution change. The GUI has already
    /// resized the source client area; this command synchronizes WGC queues,
    /// display geometry, interpolation history and shape-specific ONNX state
    /// before the first frame at the new dimensions is processed.
    SetCaptureGeometry {
        requested: (u32, u32),
        applied: (u32, u32),
    },
    /// Return a running session to native WGC geometry (Capture size = Auto).
    /// This also clears any capture-resolution reservation that was suspended
    /// while the source owned a real fullscreen presentation.
    ClearCaptureGeometry,
    /// Screen rects (GUI window / control panel) where the cursor must not
    /// engage, so those windows stay clickable above the overlay.
    SetNoEngage(Vec<(i32, i32, i32, i32, isize)>),
    /// Cursor options: auto-hide seconds (0 = off) and natural-speed fix.
    SetInputOpts {
        autohide_secs: f32,
        speed_fix: bool,
    },
    /// Frame-interpolation output multiplier (2 = double fps).
    SetInterpFactor(u32),
    /// Final downscale kernel name (spline36/lanczos3/bicubic/bilinear/nearest).
    SetDownscaler(String),
    /// Control panel window + its two sizes (physical px). The ENGINE owns
    /// the panel's position: it repositions it every tick in lockstep with
    /// the overlay (the GUI-side 100ms follow visibly detached).
    SetPanel {
        hwnd: isize,
        bar: (i32, i32),
        chip: (i32, i32),
    },
    /// Main GUI z-order policy. When the user enables "keep GUI on top", the
    /// control panel must not fight it for the topmost slot.
    SetGuiPriority {
        hwnd: isize,
        topmost: bool,
    },
    /// visible + lurking-chip state (drawing happens in the GUI viewport)
    SetPanelState {
        visible: bool,
        chip: bool,
    },
    /// sync presentation to the monitor refresh rate (wglSwapInterval)
    SetVsync(bool),
    /// Stable source-cadence presentation with a short bounded frame queue.
    SetSmoothPacing(bool),
    /// Skip expensive upscaling for perceptually identical source frames.
    SetDuplicateFrameReduction(bool),
    Shutdown,
}

#[derive(Default, Clone, Debug)]
pub struct Status {
    /// A Start command has been accepted, but the filter chain / first
    /// inference is still being prepared and has not been published yet.
    pub starting: bool,
    pub running: bool,
    /// Stop was requested from the GUI. Visual/input recovery is already
    /// complete, while provider/session cleanup may still be unwinding.
    pub stopping: bool,
    pub target_title: String,
    pub last_error: Option<String>,
    pub warning: Option<String>,
    /// Live monitor-covering fullscreen state of the selected source. This may
    /// change after Start when an application uses its own fullscreen button.
    pub source_live_fullscreen: bool,
    /// GUI-side live resize intent published before resize_client_area(). This
    /// closes the small cross-thread window where the foreign HWND already
    /// covers the monitor but SetCaptureGeometry has not been dequeued yet.
    pub capture_resolution_resize_intent: Option<(u32, u32)>,
    /// Incremented once whenever an active capture-resolution reservation is
    /// suspended because the source entered real fullscreen.
    pub capture_resolution_fullscreen_notice_seq: u64,
    /// Incremented after a fullscreen source has stably returned to windowed
    /// geometry so the GUI can reapply the latest capture-size preference.
    pub capture_resolution_reapply_seq: u64,
    /// A GLSL-containing chain has already been proven overloaded and heavy
    /// filter work is temporarily reduced while the real cursor owns Neo's GUI.
    /// This is the live protection state only; it may turn off when the pointer
    /// leaves the GUI.
    pub glsl_interactive_pause: bool,
    /// v348h/v348k: user-facing overload notice is latched for the CURRENT
    /// filter chain. Keep it visible across cursor/GUI ownership changes. It is
    /// cleared by an actual chain change/Stop, or after a live capture-resolution
    /// change is measured stably healthy and a full-chain verification frame passes.
    pub glsl_overload_notice_latched: bool,
    /// Deadline armed only after the existing overload detector has
    /// latched the user-facing warning. The GUI dispatches the same ordinary
    /// Cmd::Stop route when this expires; None means no automatic Stop pending.
    pub glsl_overload_auto_stop_deadline: Option<Instant>,
    pub chain_errors: Vec<String>,
    pub presented: u64,
    pub overlay_rect: (i32, i32, i32, i32),
    /// Visible video area inside the overlay, in physical desktop pixels.
    pub content_rect: (i32, i32, i32, i32),
    pub overlay_hwnd: isize,
    /// (hwnd, was_layered) of a visually-hidden source — insurance so the GUI
    /// can restore it even if the engine thread died.
    pub hidden_src: Option<(isize, bool)>,
    /// Full source recovery record kept outside Session so a render-thread
    /// error/panic can restore a hidden or moved PIP even after stack unwind.
    /// (hwnd, original window rect, was_maximized, was_topmost)
    pub source_recovery: Option<(isize, Option<(i32, i32, i32, i32)>, bool, bool)>,
    pub onnx_backend: OnnxBackendPreference,
    pub onnx_backend_switching: bool,
    pub onnx_backend_error: Option<String>,
    pub onnx_backend_revision: u64,
    pub onnx_tensorrt_stages: usize,
    pub onnx_cuda_stages: usize,
    pub onnx_directml_fallbacks: usize,
}

/// Compute the control panel's physical desktop position. Both the GUI
/// viewport creation and the engine follow loop use this exact function.
pub fn panel_target_position(
    mode: ScaleMode,
    overlay: (i32, i32, i32, i32),
    content: (i32, i32, i32, i32),
    panel: (i32, i32),
) -> (i32, i32) {
    let (ox, oy, _ow, _oh) = overlay;
    let (cx, cy, cw, _ch) = content;
    let (pw, ph) = panel;
    match mode {
        ScaleMode::Fixed => {
            let above = oy - ph;
            if above >= 0 { (ox, above) } else { (ox, oy) }
        }
        ScaleMode::Auto => {
            let x = if pw <= cw { cx + (cw - pw) / 2 } else { cx };
            (x, cy + 4)
        }
    }
}

#[derive(Default)]
struct PendingNoEngage {
    latest: Option<Vec<(i32, i32, i32, i32, isize)>>,
    last_published: Vec<(i32, i32, i32, i32, isize)>,
    overwritten: u64,
}

impl PendingNoEngage {
    fn publish(&mut self, rects: Vec<(i32, i32, i32, i32, isize)>) {
        // GUI repaint cadence must not become render-thread work. If the real
        // HWND geometry is unchanged, keep it entirely on the GUI side.
        if self.last_published == rects {
            return;
        }
        self.last_published = rects.clone();
        if self.latest.replace(rects).is_some() {
            self.overwritten = self.overwritten.saturating_add(1);
        }
    }

    fn take_latest(&mut self) -> Option<(Vec<(i32, i32, i32, i32, isize)>, u64)> {
        let rects = self.latest.take()?;
        let overwritten = std::mem::take(&mut self.overwritten);
        Some((rects, overwritten))
    }
}

pub struct EngineHandle {
    tx: Sender<Cmd>,
    pending_chain: Arc<Mutex<Option<Vec<StageSpec>>>>,
    // GUI geometry is state, not an ordered command stream. Language, DPI,
    // mode changes and dragging can publish many rectangles while a heavy GPU
    // frame owns the render thread; retaining only the newest measurement
    // prevents old GUI positions from replaying later and confusing cursor
    // ownership/z-order.
    pending_no_engage: Arc<Mutex<PendingNoEngage>>,
    stop_requested: Arc<AtomicBool>,
    pub metrics: Metrics,
    pub status: Arc<Mutex<Status>>,
    thread: Option<std::thread::JoinHandle<()>>,
    done_rx: Receiver<()>,
}

#[derive(Clone)]
pub struct EngineStopHandle {
    tx: Sender<Cmd>,
    pending_chain: Arc<Mutex<Option<Vec<StageSpec>>>>,
    pending_no_engage: Arc<Mutex<PendingNoEngage>>,
    stop_requested: Arc<AtomicBool>,
    status: Arc<Mutex<Status>>,
}

impl EngineStopHandle {
    pub fn request_stop(&self, reason: &str) {
        let active = {
            let state = self
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state.starting || state.running
        };
        if !active {
            return;
        }
        *self.pending_chain.lock().unwrap() = None;
        self.pending_no_engage.lock().unwrap().publish(Vec::new());
        self.stop_requested.store(true, Ordering::Release);
        let _ = crate::render::onnx_stage::request_onnx_cancel();
        let _ = crate::render::onnx_stage::request_tensorrt_cancel();
        recover_visible_capture_state_now(&self.status, reason);
        let _ = self.tx.send(Cmd::WakeForStop);
    }
}

struct EngineThreadDone(Sender<()>);
impl Drop for EngineThreadDone {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

fn emergency_recover_engine_state(
    status: &Arc<Mutex<Status>>,
    reason: &str,
    message: Option<String>,
) {
    crate::input::emergency_release_all();

    let (hidden, recovery) = {
        // A panic may have happened while Status was locked. Recovery must
        // still run with the poisoned inner value rather than panicking again.
        let mut state = status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.starting = false;
        state.running = false;
        state.stopping = false;
        state.glsl_interactive_pause = false;
        state.glsl_overload_notice_latched = false;
        state.glsl_overload_auto_stop_deadline = None;
        state.overlay_hwnd = 0;
        state.content_rect = (0, 0, 0, 0);
        state.overlay_rect = (0, 0, 0, 0);
        if let Some(message) = message {
            state.last_error = Some(message);
        }
        (state.hidden_src.take(), state.source_recovery.take())
    };

    if let Some((hwnd, was_layered)) = hidden {
        if win32::is_window_valid(hwnd) {
            win32::show_window_visual(hwnd, was_layered);
        }
    }
    if let Some((hwnd, restore_rect, was_maximized, was_topmost)) = recovery {
        if win32::is_window_valid(hwnd) {
            if let Some((hidden_hwnd, was_layered)) = hidden {
                if hidden_hwnd == hwnd {
                    win32::show_window_visual(hwnd, was_layered);
                }
            }
            if let Some(rect) = restore_rect {
                let ok = win32::restore_window_rect(hwnd, rect, was_maximized);
                log::warn!(
                    "engine-failure-source-geometry-restored: reason={reason} hwnd={hwnd:#x} rect={rect:?} maximized={was_maximized} ok={ok}"
                );
            }
            if !was_topmost {
                win32::set_topmost(hwnd, false);
            }
        }
    }
    crate::input::emergency_release_all();
    log::error!("engine-failure-failsafe-complete: reason={reason}");
}

/// Make Stop feel instantaneous without pretending the provider has already
/// finished unwinding. This is intentionally limited to user-visible state:
/// the engine thread still owns the ONNX session and releases it normally.
fn recover_visible_capture_state_now(status: &Arc<Mutex<Status>>, reason: &str) {
    crate::input::emergency_release_all();

    let (overlay_hwnd, hidden, recovery) = {
        let mut state = status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.starting = false;
        state.running = false;
        state.stopping = true;
        state.glsl_interactive_pause = false;
        state.glsl_overload_notice_latched = false;
        state.glsl_overload_auto_stop_deadline = None;
        let overlay_hwnd = state.overlay_hwnd;
        state.overlay_hwnd = 0;
        state.content_rect = (0, 0, 0, 0);
        state.overlay_rect = (0, 0, 0, 0);
        (
            overlay_hwnd,
            state.hidden_src.take(),
            state.source_recovery.take(),
        )
    };

    // Break USER32 helper-owner chains before the overlay is hidden. Otherwise
    // hiding the overlay can implicitly hide an owned panel/cursor and poison
    // the next Start, especially when GUI-topmost is OFF.
    detach_overlay_owned_helpers(overlay_hwnd);

    // The overlay HWND is owned by the render thread, but changing the alpha
    // of a top-level Win32 window is thread-safe and avoids waiting behind a
    // long DirectML/TensorRT call before the enlarged image disappears.
    if overlay_hwnd != 0 && win32::is_window_valid(overlay_hwnd) {
        win32::set_window_alpha(overlay_hwnd, 0);
    }

    if let Some((hwnd, was_layered)) = hidden {
        if win32::is_window_valid(hwnd) {
            win32::show_window_visual(hwnd, was_layered);
        }
    }
    if let Some((hwnd, restore_rect, was_maximized, was_topmost)) = recovery {
        if win32::is_window_valid(hwnd) {
            if let Some(rect) = restore_rect {
                let ok = win32::restore_window_rect(hwnd, rect, was_maximized);
                log::info!(
                    "stop-immediate-source-geometry-restored: reason={reason} hwnd={hwnd:#x} rect={rect:?} maximized={was_maximized} ok={ok}"
                );
            }
            if !was_topmost {
                win32::set_topmost(hwnd, false);
            }
        }
    }

    crate::input::emergency_release_all();
    log::info!("stop-immediate-visible-recovery-complete: reason={reason}");
}

impl EngineHandle {
    pub fn stop_handle(&self) -> EngineStopHandle {
        EngineStopHandle {
            tx: self.tx.clone(),
            pending_chain: self.pending_chain.clone(),
            pending_no_engage: self.pending_no_engage.clone(),
            stop_requested: self.stop_requested.clone(),
            status: self.status.clone(),
        }
    }
    pub fn spawn(base_dir: std::path::PathBuf) -> Self {
        let cache_root = base_dir.join("cache").join("TensorRT");
        Self::spawn_with_backend(base_dir, OnnxBackendPreference::DirectML, None, cache_root)
    }

    pub fn spawn_with_backend(
        base_dir: std::path::PathBuf,
        backend: OnnxBackendPreference,
        trt_device_id: Option<i32>,
        trt_cache_root: std::path::PathBuf,
    ) -> Self {
        PANEL_LURK_RESTORE_GEOMETRY_GUARD.store(false, Ordering::Release);
        let (tx, rx) = channel();
        let (done_tx, done_rx) = channel();
        let pending_chain = Arc::new(Mutex::new(None));
        let pending_no_engage = Arc::new(Mutex::new(PendingNoEngage::default()));
        let stop_requested = Arc::new(AtomicBool::new(false));
        let metrics = Metrics::default();
        let status = Arc::new(Mutex::new(Status {
            onnx_backend: backend,
            ..Status::default()
        }));
        let m2 = metrics.clone();
        let s2 = status.clone();
        let pending_chain2 = pending_chain.clone();
        let pending_no_engage2 = pending_no_engage.clone();
        let stop_requested2 = stop_requested.clone();
        // The render thread also needs a sender for the native floating-panel
        // Stop path. That path is consumed out-of-band before the GUI can turn
        // it into Cmd::Stop, so it must enqueue the same WakeForStop boundary
        // token itself or Status::stopping would remain latched forever.
        let control_tx = tx.clone();
        let thread = std::thread::Builder::new()
            .name("render-engine".into())
            .spawn(move || {
                let _done = EngineThreadDone(done_tx);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    engine_main(
                        rx,
                        control_tx,
                        pending_chain2,
                        pending_no_engage2,
                        stop_requested2,
                        m2,
                        s2.clone(),
                        base_dir,
                        backend,
                        trt_device_id,
                        trt_cache_root,
                    )
                }));
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        let message = format!("{error:#}");
                        log::error!("engine died: {message}");
                        emergency_recover_engine_state(&s2, "render-thread-error", Some(message));
                    }
                    Err(_) => {
                        log::error!("engine panicked");
                        emergency_recover_engine_state(
                            &s2,
                            "render-thread-panic",
                            Some(
                                "描画エンジンで予期しないエラーが発生したため、安全停止しました"
                                    .into(),
                            ),
                        );
                    }
                }
            })
            .expect("spawn engine");
        Self {
            tx,
            pending_chain,
            pending_no_engage,
            stop_requested,
            metrics,
            status,
            thread: Some(thread),
            done_rx,
        }
    }

    pub fn send(&self, cmd: Cmd) {
        match cmd {
            Cmd::Stop => {
                PANEL_LURK_RESTORE_GEOMETRY_GUARD.store(false, Ordering::Release);
                // Do not queue cancellation behind the render thread: a heavy
                // DirectML/TensorRT Session::Run may currently own that thread.
                // RunOptions::terminate is cooperative and returns immediately,
                // allowing the blocked call to unwind and read Cmd::Stop.
                *self.pending_chain.lock().unwrap() = None;
                self.pending_no_engage.lock().unwrap().publish(Vec::new());
                self.stop_requested.store(true, Ordering::Release);
                let _ = crate::render::onnx_stage::request_onnx_cancel();
                let _ = crate::render::onnx_stage::request_tensorrt_cancel();
                recover_visible_capture_state_now(&self.status, "user-stop");
                if self.tx.send(Cmd::WakeForStop).is_err() {
                    self.status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .stopping = false;
                }
            }
            Cmd::Shutdown => {
                PANEL_LURK_RESTORE_GEOMETRY_GUARD.store(false, Ordering::Release);
                *self.pending_chain.lock().unwrap() = None;
                self.pending_no_engage.lock().unwrap().publish(Vec::new());
                self.stop_requested.store(true, Ordering::Release);
                let _ = crate::render::onnx_stage::request_onnx_cancel();
                let _ = crate::render::onnx_stage::request_tensorrt_cancel();
                recover_visible_capture_state_now(&self.status, "application-shutdown");
                let _ = self.tx.send(Cmd::Shutdown);
            }
            start @ Cmd::Start {
                hwnd,
                source_restore_rect,
                source_was_maximized,
                ..
            } => {
                {
                    let mut state = self
                        .status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    if state.stopping {
                        log::info!("capture-start-ignored: stop cleanup is still in progress");
                        return;
                    }
                    state.starting = true;
                    state.last_error = None;
                    let fullscreen_origin = source_restore_rect
                        .map(|rect| win32::is_rect_monitor_fullscreen(hwnd, rect))
                        .unwrap_or_else(|| win32::is_monitor_fullscreen(hwnd));
                    state.source_live_fullscreen = fullscreen_origin;
                    state.source_recovery = Some((
                        hwnd,
                        source_restore_rect,
                        source_was_maximized,
                        // Fullscreen pixel-lock sessions deliberately never
                        // promote the foreign source HWND, so recovery must
                        // not issue a matching NOTOPMOST mutation either.
                        win32::is_topmost(hwnd) || fullscreen_origin,
                    ));
                }
                self.stop_requested.store(false, Ordering::Release);
                crate::render::onnx_stage::begin_tensorrt_start_request();
                if self.tx.send(start).is_err() {
                    let mut state = self
                        .status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    state.starting = false;
                    state.source_recovery = None;
                }
            }
            Cmd::ApplyChain(specs) => {
                // Live editing can produce several changes in one GUI gesture.
                // Keep the newest complete chain instead of queueing stale rebuilds
                // behind frame interpolation work.
                *self.pending_chain.lock().unwrap() = Some(specs);
            }
            Cmd::SetNoEngage(rects) => {
                // GUI geometry is latest-only state. Never put position/size
                // samples into the render command FIFO: under a multi-second
                // GLSL/ONNX frame they would replay later (Full/Mini/language
                // sizes and drag positions included), causing false engage,
                // cursor flicker and repeated foreground promotion.
                self.pending_no_engage.lock().unwrap().publish(rects);
            }
            switch @ Cmd::SwitchOnnxBackend { .. } => {
                // The switch carries the newest complete specs. Discard a
                // queued edit so it cannot rebuild the old backend afterward.
                *self.pending_chain.lock().unwrap() = None;
                let _ = self.tx.send(switch);
            }
            other => {
                let _ = self.tx.send(other);
            }
        }
    }

    pub fn shutdown(&mut self) {
        *self.pending_chain.lock().unwrap() = None;
        self.pending_no_engage.lock().unwrap().publish(Vec::new());
        self.stop_requested.store(true, Ordering::Release);
        let _ = crate::render::onnx_stage::request_onnx_cancel();
        let _ = crate::render::onnx_stage::request_tensorrt_cancel();
        recover_visible_capture_state_now(&self.status, "application-drop");
        let _ = self.tx.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            if self.done_rx.recv_timeout(Duration::from_secs(2)).is_ok() {
                let _ = t.join();
            } else {
                log::error!("render-engine-shutdown-timeout: elapsed_ms=2000 action=detach");
                emergency_recover_engine_state(
                    &self.status,
                    "shutdown-timeout",
                    Some(
                        "描画エンジンの終了がタイムアウトしたため、入力とソースを強制復元しました"
                            .into(),
                    ),
                );
                drop(t);
            }
        }
    }

    /// Application-window close path. Once cursor/source/overlay recovery has
    /// completed synchronously, the OS process is about to terminate and there
    /// is no user-visible benefit in waiting for provider/cache/thread destructors.
    /// Waiting here caused the root eframe viewport to remain visible for the
    /// outer 2 s render-engine timeout even with no active filters.
    pub fn shutdown_for_app_exit(&mut self) {
        *self.pending_chain.lock().unwrap() = None;
        self.pending_no_engage.lock().unwrap().publish(Vec::new());
        self.stop_requested.store(true, Ordering::Release);
        let _ = crate::render::onnx_stage::request_onnx_cancel();
        let _ = crate::render::onnx_stage::request_tensorrt_cancel();
        recover_visible_capture_state_now(&self.status, "application-exit");
        let _ = self.tx.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            // Dropping JoinHandle detaches it. Process teardown will reclaim the
            // thread, WGL objects and provider state. All native input/source
            // state that can escape the process was restored above first.
            drop(t);
            log::info!("render-engine app-exit detached after visible recovery");
        }
    }
}

fn enforce_gui_priority(
    gui_hwnd: isize,
    gui_topmost: bool,
    panel_hwnd: isize,
    overlay_hwnd: isize,
) {
    // Exact v235 contract: GUI priority logic has no authority at all when the
    // GUI is not TOPMOST. This hard split prevents later GUI-topmost fixes from
    // leaking SetWindowPos/DWM side effects into the historically stable OFF
    // path. The ordinary panel block below retains its own v235-style repair.
    if !gui_topmost
        || gui_hwnd == 0
        || !win32::is_window_valid(gui_hwnd)
        || win32::is_minimized(gui_hwnd)
    {
        return;
    }

    // GUI-topmost ON keeps the modern stable hierarchy.
    win32::normalize_neo_topmost_stack(gui_hwnd, true, panel_hwnd, overlay_hwnd);
    crate::input::keep_cursor_sprite_on_top();
}

fn reposition_overlay_for_gui_mode(
    overlay: &mut OverlayWindow,
    x: i32,
    y: i32,
    w: i32,
    h: i32,
    gui_topmost: bool,
) {
    if gui_topmost {
        overlay.reposition(x, y, w, h);
    } else {
        overlay.reposition_v235_topmost(x, y, w, h);
    }
}

fn detach_overlay_owned_helpers(overlay_hwnd: isize) {
    crate::input::detach_cursor_sprite_owner();
    let panel_hwnd = win32::panel_gdi_host_hwnd();
    if panel_hwnd != 0
        && win32::is_window_valid(panel_hwnd)
        && (overlay_hwnd == 0 || win32::window_owner(panel_hwnd) == overlay_hwnd)
    {
        win32::set_panel_overlay_owner(panel_hwnd, overlay_hwnd, false);
    }
}

/// cur -> prev -> prev2; returns the texture that fell off (to recycle).
fn rotate_interp_history(s: &mut Session, cur_tex: GpuTex) -> Option<GpuTex> {
    let old_prev = s.prev_tex.replace(cur_tex);
    match old_prev {
        Some(op) => s.prev2_tex.replace(op),
        None => None,
    }
}

fn sdr_highlight_scale_lut() -> &'static [u32; 256] {
    static LUT: OnceLock<[u32; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|peak| {
            if peak == 0 || peak <= SDR_HIGHLIGHT_KNEE_U8 as usize {
                1u32 << 16
            } else {
                let input_span = (u16::from(u8::MAX) - u16::from(SDR_HIGHLIGHT_KNEE_U8)).max(1);
                let output_span =
                    u16::from(SDR_HIGHLIGHT_CEILING_U8) - u16::from(SDR_HIGHLIGHT_KNEE_U8);
                let above = peak as u16 - u16::from(SDR_HIGHLIGHT_KNEE_U8);
                let mapped = u16::from(SDR_HIGHLIGHT_KNEE_U8)
                    + (above * output_span + input_span / 2) / input_span;
                (u32::from(mapped) << 16) / peak as u32
            }
        })
    })
}

/// Apply a stable, allocation-free highlight soft knee to an ordinary RGBA8
/// frame. Only bright, nearly neutral pixels are eligible. All RGB channels
/// use one scale derived from maxRGB, so colour ratios remain unchanged.
/// Saturated red, magenta, purple and skin colours are left byte-for-byte
/// untouched. Alpha is untouched.
fn protect_sdr_highlights_in_place(frame: &mut FrameBuf) -> bool {
    if frame.hdr || frame.w <= 0 || frame.h <= 0 {
        return false;
    }
    let expected = match (frame.w as usize)
        .checked_mul(frame.h as usize)
        .and_then(|pixels| pixels.checked_mul(4))
    {
        Some(expected) => expected,
        None => return false,
    };
    if frame.data.len() != expected {
        return false;
    }

    let scales = sdr_highlight_scale_lut();
    for pixel in frame.data.chunks_exact_mut(4) {
        let peak = pixel[0].max(pixel[1]).max(pixel[2]);
        let floor = pixel[0].min(pixel[1]).min(pixel[2]);
        if peak <= SDR_HIGHLIGHT_KNEE_U8 || peak.saturating_sub(floor) > SDR_HIGHLIGHT_MAX_CHROMA_U8
        {
            continue;
        }
        let scale = scales[peak as usize];
        pixel[0] = ((u32::from(pixel[0]) * scale + 32_768) >> 16).min(255) as u8;
        pixel[1] = ((u32::from(pixel[1]) * scale + 32_768) >> 16).min(255) as u8;
        pixel[2] = ((u32::from(pixel[2]) * scale + 32_768) >> 16).min(255) as u8;
    }
    true
}

fn sanitize_hdr_channel(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if edge1 <= edge0 {
        return if value >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Raise only nearly neutral reference-white graphics (for example white
/// subtitles and captions) toward clean SDR white. The mask fades out before
/// high-energy HDR highlights, so lamps, skin reflections and cloud detail
/// continue to use the conservative Low400 shoulder.
fn recover_neutral_reference_white(rgb: [f32; 3], source_peak: f32, mapped_peak: f32) -> f32 {
    if source_peak <= 1e-6 || mapped_peak >= HDR_NEUTRAL_WHITE_TARGET {
        return mapped_peak;
    }
    let source_min = rgb[0].min(rgb[1]).min(rgb[2]);
    let relative_chroma = ((source_peak - source_min) / source_peak).max(0.0);
    let neutral_mask = 1.0 - smoothstep(0.025, 0.10, relative_chroma);
    let reference_white_entry = smoothstep(0.72, 1.0, source_peak);
    let source_peak_nits = source_peak * SCRGB_REFERENCE_WHITE_NITS;
    let high_hdr_exit = 1.0
        - smoothstep(
            HDR_NEUTRAL_RECOVERY_FADE_START_NITS,
            HDR_NEUTRAL_RECOVERY_FADE_END_NITS,
            source_peak_nits,
        );
    let weight =
        (neutral_mask * reference_white_entry * high_hdr_exit * HDR_NEUTRAL_WHITE_STRENGTH)
            .clamp(0.0, 1.0);
    mapped_peak + (HDR_NEUTRAL_WHITE_TARGET - mapped_peak).max(0.0) * weight
}

/// Convert one linear scRGB pixel to linear SDR/Rec.709 using the former
/// Low400 maxRGB curve. All channels receive the same scale, retaining hue and
/// colour ratios while the early shoulder protects lamps, faces and gradients.
/// `_mode` is retained only for settings-file compatibility.
fn tonemap_scrgb_to_sdr_linear(rgb: [f32; 3], _mode: HdrSdrMode) -> [f32; 3] {
    let mut c = [
        sanitize_hdr_channel(rgb[0]),
        sanitize_hdr_channel(rgb[1]),
        sanitize_hdr_channel(rgb[2]),
    ];
    let source_peak = c[0].max(c[1]).max(c[2]);
    if source_peak <= 1e-6 {
        return [0.0; 3];
    }

    let shoulder_mapped_peak = hdr_shoulder_lut(source_peak);
    let mapped_peak = recover_neutral_reference_white(c, source_peak, shoulder_mapped_peak)
        .clamp(0.0, HDR_OUTPUT_LIMIT);
    let scale = mapped_peak / source_peak;
    for channel in &mut c {
        *channel *= scale;
    }

    [
        c[0].clamp(0.0, HDR_OUTPUT_LIMIT),
        c[1].clamp(0.0, HDR_OUTPUT_LIMIT),
        c[2].clamp(0.0, HDR_OUTPUT_LIMIT),
    ]
}

fn srgb_encode_lut16() -> &'static [u16; SRGB_ENCODE_LUT_LEN] {
    static LUT: OnceLock<[u16; SRGB_ENCODE_LUT_LEN]> = OnceLock::new();
    LUT.get_or_init(|| {
        std::array::from_fn(|index| {
            let x = index as f32 / (SRGB_ENCODE_LUT_LEN - 1) as f32;
            let encoded = if x <= 0.003_130_8 {
                12.92 * x
            } else {
                1.055 * x.powf(1.0 / 2.4) - 0.055
            };
            (encoded * 65_535.0 + 0.5).clamp(0.0, 65_535.0) as u16
        })
    })
}

#[inline]
fn ordered_dither_threshold(pixel_index: usize) -> u16 {
    const BAYER8: [u8; 64] = [
        0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38,
        60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57,
        25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21,
    ];
    (BAYER8[pixel_index & 63] as u16 * 257 + 31) / 63
}

#[inline]
fn srgb_encode_u8_dithered(
    linear: f32,
    lut: &[u16; SRGB_ENCODE_LUT_LEN],
    pixel_index: usize,
) -> u8 {
    let index = (linear.clamp(0.0, 1.0) * (SRGB_ENCODE_LUT_LEN - 1) as f32 + 0.5) as usize;
    let encoded = lut[index.min(SRGB_ENCODE_LUT_LEN - 1)] as u32;
    let base = encoded / 257;
    let remainder = encoded % 257;
    let threshold = ordered_dither_threshold(pixel_index) as u32;
    (base + u32::from(remainder > threshold)).min(255) as u8
}

fn hdr_half_pixel_slice_to_sdr_u8(
    pixel: &[u8],
    srgb_lut: &[u16; SRGB_ENCODE_LUT_LEN],
    mode: HdrSdrMode,
    pixel_index: usize,
) -> [u8; 4] {
    debug_assert!(pixel.len() >= 8);
    let half_at = |offset: usize| {
        f16::from_bits(u16::from_le_bytes([pixel[offset], pixel[offset + 1]])).to_f32()
    };
    let rgb = tonemap_scrgb_to_sdr_linear([half_at(0), half_at(2), half_at(4)], mode);
    let alpha = sanitize_hdr_channel(half_at(6)).clamp(0.0, 1.0);
    [
        srgb_encode_u8_dithered(rgb[0], srgb_lut, pixel_index),
        srgb_encode_u8_dithered(rgb[1], srgb_lut, pixel_index),
        srgb_encode_u8_dithered(rgb[2], srgb_lut, pixel_index),
        (alpha * 255.0 + 0.5) as u8,
    ]
}

fn hdr_half_pixel_to_sdr_u8(
    data: &[u8],
    byte_offset: usize,
    srgb_lut: &[u16; SRGB_ENCODE_LUT_LEN],
    mode: HdrSdrMode,
    pixel_index: usize,
) -> Option<[u8; 4]> {
    let pixel = data.get(byte_offset..byte_offset.checked_add(8)?)?;
    Some(hdr_half_pixel_slice_to_sdr_u8(
        pixel,
        srgb_lut,
        mode,
        pixel_index,
    ))
}

/// Legacy safety fallback: WGC Rgba16F -> ordinary RGBA8 SDR. The normal
/// The current option path requests Rgba8 and does not call this conversion.
fn tonemap_hdr_frame_to_sdr(
    frame: &mut FrameBuf,
    output: &mut Vec<u8>,
    mode: HdrSdrMode,
) -> Option<Vec<u8>> {
    if !frame.hdr {
        return Some(Vec::new());
    }
    let pixels = (frame.w.max(0) as usize).checked_mul(frame.h.max(0) as usize)?;
    let expected_hdr_bytes = pixels.checked_mul(8)?;
    let expected_sdr_bytes = pixels.checked_mul(4)?;
    if pixels == 0 || frame.data.len() != expected_hdr_bytes {
        return None;
    }
    output.resize(expected_sdr_bytes, 0);
    let source = &frame.data;
    let srgb_lut = srgb_encode_lut16();
    hdr_tonemap_pool().install(|| {
        source
            .par_chunks(8 * 4096)
            .zip(output.par_chunks_mut(4 * 4096))
            .enumerate()
            .for_each(|(chunk_index, (source_chunk, output_chunk))| {
                let base_pixel = chunk_index * 4096;
                for (local_index, (source_pixel, output_pixel)) in source_chunk
                    .chunks_exact(8)
                    .zip(output_chunk.chunks_exact_mut(4))
                    .enumerate()
                {
                    output_pixel.copy_from_slice(&hdr_half_pixel_slice_to_sdr_u8(
                        source_pixel,
                        srgb_lut,
                        mode,
                        base_pixel + local_index,
                    ));
                }
            });
    });
    let raw_hdr = std::mem::take(&mut frame.data);
    frame.data = std::mem::take(output);
    frame.hdr = false;
    Some(raw_hdr)
}

#[derive(Clone)]
struct FrameSignature {
    size: (i32, i32),
    luma: Vec<u8>,
}

/// Conservative perceptual signature for duplicate animation frames. Four
/// samples per 64x36 cell suppress codec grain, while the strict maximum
/// delta guard prevents a small moving object or subtitle from being missed.
fn frame_signature(frame: &FrameBuf) -> Option<FrameSignature> {
    if frame.w <= 0 || frame.h <= 0 {
        return None;
    }
    let bytes_per_pixel = if frame.hdr { 8usize } else { 4usize };
    let expected = (frame.w as usize)
        .checked_mul(frame.h as usize)?
        .checked_mul(bytes_per_pixel)?;
    if frame.data.len() < expected {
        return None;
    }
    const GW: usize = 64;
    const GH: usize = 36;
    let (w, h) = (frame.w as usize, frame.h as usize);
    let srgb_lut = srgb_encode_lut16();
    let sampled_luma = |x: usize, y: usize| -> Option<u32> {
        let pixel = y.checked_mul(w)?.checked_add(x)?;
        if frame.hdr {
            let rgba = hdr_half_pixel_to_sdr_u8(
                &frame.data,
                pixel.checked_mul(8)?,
                srgb_lut,
                HdrSdrMode::Low400,
                pixel,
            )?;
            Some((rgba[0] as u32 * 54 + rgba[1] as u32 * 183 + rgba[2] as u32 * 19) >> 8)
        } else {
            let p = pixel.checked_mul(4)?;
            Some(
                (frame.data[p] as u32 * 54
                    + frame.data[p + 1] as u32 * 183
                    + frame.data[p + 2] as u32 * 19)
                    >> 8,
            )
        }
    };
    let mut luma = Vec::with_capacity(GW * GH);
    for gy in 0..GH {
        for gx in 0..GW {
            let x0 = gx * w / GW;
            let x1 = ((gx + 1) * w / GW).saturating_sub(1).min(w - 1);
            let y0 = gy * h / GH;
            let y1 = ((gy + 1) * h / GH).saturating_sub(1).min(h - 1);
            let mut sum = 0u32;
            for (x, y) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                sum += sampled_luma(x, y)?;
            }
            luma.push(((sum + 2) / 4) as u8);
        }
    }
    Some(FrameSignature {
        size: (frame.w, frame.h),
        luma,
    })
}

/// Sampled one-line comb detector used only while a NeoDeint interlace shader
/// is active. Chromium can expose transitional woven pictures between stable
/// progressive frames. Spatial bobbing cannot remove a second image already
/// present inside each field, so a short look-ahead selects the least-combed
/// complete picture before the GLSL reconstruction pass.
fn frame_comb_fraction(frame: &FrameBuf) -> f32 {
    if frame.hdr || frame.w < 16 || frame.h < 16 || frame.data.len() < 4 {
        return 0.0;
    }
    let w = frame.w as usize;
    let h = frame.h as usize;
    let stride = w * 4;
    if frame.data.len() < stride * h {
        return 0.0;
    }
    let y0 = (h / 32).max(2);
    let y1 = h.saturating_sub((h / 16).max(3));
    let mut combed = 0usize;
    let mut tested = 0usize;
    let luma = |i: usize| -> i32 {
        (54 * frame.data[i] as i32 + 183 * frame.data[i + 1] as i32 + 19 * frame.data[i + 2] as i32)
            >> 8
    };
    for y in (y0..y1).step_by(2) {
        for x in (4..w.saturating_sub(4)).step_by(4) {
            let i = y * stride + x * 4;
            let u = luma(i - stride);
            let c = luma(i);
            let d = luma(i + stride);
            let alternating = (c - u).abs().min((c - d).abs()) - (u - d).abs() / 4;
            combed += usize::from(alternating > 18);
            tested += 1;
        }
    }
    if tested == 0 {
        0.0
    } else {
        combed as f32 / tested as f32
    }
}

const NEODEINT_PROGRESSIVE_BYPASS_MAX: f32 = 0.020;

fn update_neodeint_scene_latch(history: &mut u8, hold: &mut u8, combed: bool) -> bool {
    *history = ((*history << 1) | u8::from(combed)) & 0x0f;
    if history.count_ones() >= 2 {
        *hold = 4;
    } else if *hold > 0 {
        *hold -= 1;
    }
    !combed && *hold == 0
}

fn signatures_match(a: &FrameSignature, b: &FrameSignature) -> bool {
    signatures_luma_close(a, b) && !signature_has_coherent_motion(a, b)
}

fn signatures_luma_close(a: &FrameSignature, b: &FrameSignature) -> bool {
    if a.size != b.size || a.luma.len() != b.luma.len() {
        return false;
    }
    let mut sum = 0u32;
    let mut max_delta = 0u8;
    for (&x, &y) in a.luma.iter().zip(&b.luma) {
        let d = x.abs_diff(y);
        sum += d as u32;
        max_delta = max_delta.max(d);
        if max_delta > 2 {
            return false;
        }
    }
    sum * 5 <= a.luma.len() as u32
}

/// A slow camera zoom can change each 64x36 signature sample by only one
/// luma step and therefore resemble codec shimmer.  Codec noise is spatially
/// incoherent; zoom change follows the image's radial derivative around the
/// frame centre.  Reject that coherent component before reusing the previous
/// filtered output, so gradual zooms retain their original cadence.
fn signature_has_coherent_motion(a: &FrameSignature, b: &FrameSignature) -> bool {
    const GW: usize = 64;
    const GH: usize = 36;
    if a.luma.len() != GW * GH {
        return false;
    }
    let mut dot = 0i64;
    let mut dot_x = 0i64;
    let mut dot_y = 0i64;
    let mut temporal_energy = 0i64;
    let mut radial_energy = 0i64;
    let mut gradient_x_energy = 0i64;
    let mut gradient_y_energy = 0i64;
    let mut changed = 0u32;
    let mut brighter = 0u32;
    let mut darker = 0u32;
    for y in 1..GH - 1 {
        for x in 1..GW - 1 {
            let i = y * GW + x;
            let dt = b.luma[i] as i32 - a.luma[i] as i32;
            if dt == 0 {
                continue;
            }
            if dt > 0 {
                brighter += 1;
            } else {
                darker += 1;
            }
            let gx = a.luma[i + 1] as i32 - a.luma[i - 1] as i32;
            let gy = a.luma[i + GW] as i32 - a.luma[i - GW] as i32;
            // Twice-centred coordinates avoid floating point in the hot path.
            let radial =
                gx * (x as i32 * 2 - (GW - 1) as i32) + gy * (y as i32 * 2 - (GH - 1) as i32);
            if radial == 0 {
                continue;
            }
            changed += 1;
            dot += dt as i64 * radial as i64;
            dot_x += dt as i64 * gx as i64;
            dot_y += dt as i64 * gy as i64;
            temporal_energy += (dt * dt) as i64;
            radial_energy += radial as i64 * radial as i64;
            gradient_x_energy += (gx * gx) as i64;
            gradient_y_energy += (gy * gy) as i64;
        }
    }
    // At least a small, frame-wide set of samples must agree. Squared
    // correlation makes zoom-in and zoom-out symmetric. A broad change may
    // use a lower threshold because random codec shimmer cancels across that
    // many cells; a sparse change remains deliberately strict.
    let correlation_percent = if changed >= 96 { 12 } else { 30 };
    let correlated = |d: i64, spatial_energy: i64| {
        spatial_energy > 0
            && d.saturating_mul(d).saturating_mul(100)
                >= temporal_energy
                    .saturating_mul(spatial_energy)
                    .saturating_mul(correlation_percent)
    };
    // A fade changes samples predominantly in one direction. Quantisation can
    // make a very slow fade sparse, so use sample count rather than magnitude.
    let fade = changed >= 16 && brighter.max(darker) * 100 >= changed * 78;
    // Brightness differences caused by translation correlate with the image
    // gradient. Testing X/Y separately is a compact Lucas-Kanade style motion
    // test and catches pans as well as moving foreground/background layers.
    let translation = changed >= 24
        && (correlated(dot_x, gradient_x_energy) || correlated(dot_y, gradient_y_energy));
    let zoom = changed >= 24 && correlated(dot, radial_energy);
    fade || translation || zoom
}

/// Detect low-amplitude motion that is ambiguous in either adjacent pair but
/// continues through prev -> current -> next. Vertical/horizontal scrolling,
/// slow zoom and fades retain the same temporal sign over many samples;
/// decoder shimmer tends to reverse or move randomly between deliveries.
fn signature_has_temporal_continuity(
    previous: &FrameSignature,
    current: &FrameSignature,
    next: &FrameSignature,
) -> bool {
    if previous.size != current.size
        || current.size != next.size
        || previous.luma.len() != current.luma.len()
        || current.luma.len() != next.luma.len()
    {
        return true;
    }
    let mut overlap = 0u32;
    let mut same_direction = 0u32;
    let mut temporal_dot = 0i64;
    let mut energy0 = 0i64;
    let mut energy1 = 0i64;
    for ((&p, &c), &n) in previous.luma.iter().zip(&current.luma).zip(&next.luma) {
        let d0 = c as i32 - p as i32;
        let d1 = n as i32 - c as i32;
        if d0 != 0 && d1 != 0 {
            overlap += 1;
            if d0.signum() == d1.signum() {
                same_direction += 1;
            }
        }
        temporal_dot += d0 as i64 * d1 as i64;
        energy0 += (d0 * d0) as i64;
        energy1 += (d1 * d1) as i64;
    }
    let sign_continuity =
        overlap >= 10 && same_direction.saturating_mul(100) >= overlap.saturating_mul(62);
    let correlation = energy0 > 0
        && energy1 > 0
        && temporal_dot > 0
        && temporal_dot
            .saturating_mul(temporal_dot)
            .saturating_mul(100)
            >= energy0.saturating_mul(energy1).saturating_mul(8);
    // The two-frame span magnifies sub-LSB-per-frame camera movement and is
    // particularly effective for slow credits and vertical background pans.
    let strong_reversal = overlap >= 8 && (overlap - same_direction) * 100 >= overlap * 70;
    let accumulated_motion = !strong_reversal && signature_has_coherent_motion(previous, next);
    sign_continuity || correlation || accumulated_motion
}

/// Strict comparison used when load reduction is disabled. WGC compositor
/// repeats are normally identical; this tiny allowance is only for format
/// conversion noise and must not merge neighboring video pictures.
fn compositor_signatures_match(a: &FrameSignature, b: &FrameSignature) -> bool {
    if a.size != b.size || a.luma.len() != b.luma.len() {
        return false;
    }
    let mut sum = 0u32;
    for (&x, &y) in a.luma.iter().zip(&b.luma) {
        let d = x.abs_diff(y);
        if d > 1 {
            return false;
        }
        sum += d as u32;
    }
    sum * 20 <= a.luma.len() as u32
}

/// Choose the native input coordinate space that corresponds to the WGC frame.
///
/// Client-only capture intentionally maps to the client rect. Full-window WGC
/// can use either DWM extended-frame bounds or the Win32 outer rect depending
/// on window type; choose the candidate whose dimensions best match the actual
/// captured/display aspect (allowing the existing right/bottom even-pad row).
fn source_input_reference_rect(
    hwnd: isize,
    client_only: bool,
    frame_size: (i32, i32),
) -> Option<((i32, i32, i32, i32), bool, &'static str)> {
    if client_only {
        return win32::client_rect_on_screen(hwnd).map(|r| (r, false, "client"));
    }

    let (fw, fh) = frame_size;
    let mut best: Option<(i64, (i32, i32, i32, i32), &'static str)> = None;
    for (kind, candidate) in [
        ("dwm", win32::extended_frame_bounds(hwnd)),
        ("outer", win32::window_rect(hwnd)),
    ] {
        let Some(rect @ (_, _, rw, rh)) = candidate else {
            continue;
        };
        if rw <= 0 || rh <= 0 {
            continue;
        }
        let score = (rw as i64 - fw as i64).abs() + (rh as i64 - fh as i64).abs();
        if best.is_none_or(|(best_score, _, _)| score < best_score) {
            best = Some((score, rect, kind));
        }
    }

    if let Some((_, rect, kind)) = best {
        return Some((rect, true, kind));
    }

    // Defensive fallback for unusual windows where DWM/outer geometry cannot
    // be queried. This preserves operability rather than retaining stale input.
    win32::client_rect_on_screen(hwnd).map(|r| (r, false, "client-fallback"))
}

/// Position changes do not alter the WGC surface.  Keep the fullscreen/resize
/// watchdog strictly size-based so ordinary title-bar dragging can never arm
/// a capture restart.
fn rect_size_changed(previous: (i32, i32, i32, i32), next: (i32, i32, i32, i32)) -> bool {
    previous.2 != next.2 || previous.3 != next.3
}

/// Compare a delivered WGC frame with the native rectangle that actually
/// corresponds to that capture mode.  A one-pixel right/bottom even pad is
/// expected and must not be treated as stale geometry.
fn frame_size_mismatches_reference(frame: (i32, i32), reference: (i32, i32, i32, i32)) -> bool {
    (frame.0 - reference.2).abs() > 2 || (frame.1 - reference.3).abs() > 2
}

fn should_enter_live_fullscreen(
    source_monitor_fullscreen: bool,
    live_monitor_fullscreen: bool,
    capture_canvas: Option<(i32, i32)>,
    monitor_size: (i32, i32),
    source_presentation_signature: Option<(u32, u32, bool)>,
    live_presentation_signature: Option<(u32, u32, bool)>,
    capture_resize_intent: Option<(u32, u32)>,
    recent_geometry_change: bool,
) -> bool {
    if !live_monitor_fullscreen {
        return false;
    }
    if source_monitor_fullscreen {
        return true;
    }

    // Fullscreen buttons/F11 commonly remove normal window chrome or change
    // maximized presentation state. That is source-owned even when the user's
    // requested capture canvas happens to equal the monitor resolution.
    let presentation_changed = source_presentation_signature
        .zip(live_presentation_signature)
        .is_some_and(|(origin, live)| origin != live);
    if presentation_changed {
        return true;
    }

    // resize_client_area() runs before SetCaptureGeometry reaches this thread.
    // The GUI publishes the requested dimensions synchronously before touching
    // the foreign HWND, so even a very fast WGC resize cannot be mistaken for
    // an application-owned fullscreen transition.
    let resize_intent_targets_monitor = capture_resize_intent.is_some_and(|target| {
        (target.0 as i32 - monitor_size.0).abs() <= 2
            && (target.1 as i32 - monitor_size.1).abs() <= 2
    });
    if resize_intent_targets_monitor {
        return false;
    }

    // Keep a short geometry-only fallback as a second line of defence for
    // unusual source-window transitions that race the GUI status hand-off. A
    // genuine source transition with no matching capture-size command is
    // reconsidered after the grace period.
    if recent_geometry_change {
        return false;
    }

    // The important v471 distinction: a windowed-origin PIP/mpv window that
    // Neo intentionally resized to the monitor's exact client canvas remains a
    // normal resizable source. Pixel dimensions alone are not fullscreen state.
    // If the application later changes its presentation style, the branch above
    // still detects the real fullscreen transition.
    let neo_owned_monitor_canvas = capture_canvas.is_some_and(|target| {
        (target.0 - monitor_size.0).abs() <= 2 && (target.1 - monitor_size.1).abs() <= 2
    });
    !neo_owned_monitor_canvas
}

struct Session {
    hwnd: isize,
    /// Immutable session-origin monitor coverage. Fullscreen source hiding must
    /// happen only after the opaque overlay has been committed, otherwise the
    /// desktop is exposed between the two DWM updates.
    source_monitor_fullscreen: bool,
    /// Live fullscreen state. Unlike source_monitor_fullscreen this can change
    /// after Start.
    source_live_fullscreen: bool,
    /// Fullscreen-relevant source chrome/maximize state sampled at session
    /// start. Geometry alone is insufficient because Neo may intentionally
    /// resize a normal PIP/mpv window to exactly 1920x1080 on a 1080p monitor.
    source_presentation_signature: Option<(u32, u32, bool)>,
    /// Keep the explicit capture canvas reserved, but do not force it onto WGC
    /// while the source owns a real fullscreen presentation.
    capture_canvas_suspended_for_fullscreen: bool,
    /// After fullscreen exit, keep native WGC authoritative until the GUI
    /// reapplies the user's latest capture-size preference.
    capture_resolution_reapply_pending: bool,
    fullscreen_exit_candidate_since: Option<Instant>,
    browser_fullscreen_cadence: bool,
    source: WgcSource,
    chain: FilterChain,
    mode: ScaleMode,
    ratio: f32,
    fps_cap: Option<u32>,
    capture_client_only: bool,
    capture_hdr: bool,
    hdr_sdr_mode: HdrSdrMode,
    /// Logged after the first successful lightweight RGBA8 highlight pass.
    hdr_sdr_preprocess_logged: bool,
    /// Avoid flooding the log if a malformed fallback frame is received.
    hdr_sdr_preprocess_error_logged: bool,
    /// Cached frames can be reprocessed after a filter change. Do not apply the
    /// in-place soft knee twice to the same WGC sequence.
    hdr_highlight_protected_seq: Option<u64>,
    capture_canvas: Option<(i32, i32)>,
    source_restart_since: Option<Instant>,
    source_restart_last: Instant,
    last_tex: Option<GpuTex>,
    last_present: Instant,
    // Start of the most recent SwapBuffers call. Interpolation pacing must
    // anchor to submission time, not completion time, because a healthy DWM
    // path may block inside SwapBuffers for several milliseconds.
    last_present_started: Instant,
    frame: FrameBuf,
    /// A live chain replacement must process the cached source even when a
    /// static PIP does not emit another WGC frame.
    chain_reprocess_pending: bool,
    in_size: (i32, i32),
    /// Display aspect captured when magnification starts. Processing follows
    /// later source resolutions, but the picture must not become anamorphic.
    display_aspect: (i32, i32),
    pending_resize_size: Option<(i32, i32)>,
    /// A real source-size transition invalidates shape-specialized ONNX/EP
    /// execution state. Rebuild ONNX stages after the exact new WGC geometry
    /// is committed, while preserving resident GLSL programs and pools.
    onnx_geometry_rebuild_pending: bool,
    /// windowed mode: overlay placed once, then the user may move it freely
    placed: bool,
    /// Mode/ratio changes replace the overlay geometry. Keep mapped input
    /// disabled briefly so stale rectangles can never clip the real cursor.
    input_reenable_after: Option<Instant>,
    source_input_geometry_missing_logged: bool,
    hid_source: Option<bool>, // Some(was_layered) if we hid the source
    /// Delay hiding the source until a valid filtered frame has been rendered.
    /// This prevents a black flash while a newly-selected ONNX filter warms up.
    hide_source_pending: bool,
    src_was_topmost: bool,
    /// Immutable outer-window rectangle captured before any Neo-initiated
    /// capture-resolution resize. Never rebase this during the session.
    source_restore_rect: Option<(i32, i32, i32, i32)>,
    /// Maximized state paired with the immutable session-origin rectangle.
    source_was_maximized: bool,
    deferred_capture_resolution: Option<((u32, u32), (u32, u32))>,
    /// The source window has been resized and the first exact-size WGC frame
    /// must be filtered before the overlay is revealed.
    capture_resolution_applied: bool,
    capture_resolution_wait_started: Option<Instant>,
    /// Last non-invasive repaint request while waiting for Chromium PIP to
    /// deliver the first real frame at the requested client size.
    capture_resolution_repaint_last: Option<Instant>,
    /// Chromium can ignore paint invalidation for a transparent static PIP.
    /// A one-time hidden client-size round trip forces compositor relayout.
    capture_resolution_nudge_done: bool,
    /// A static Chromium PIP may not repaint while visually hidden. In that
    /// case reveal a freshly filtered native-size cache (never resampled to
    /// the requested capture size) while continuing to wait for the real frame.
    capture_resolution_native_fallback: bool,
    /// Suppress duplicate diagnostics while ONNX waits for the real requested
    /// WGC geometry. Reset after the target frame arrives or the request ends.
    onnx_geometry_deferred_logged: bool,
    /// frame interpolation: recent captured frames (RGB, or RGBA for interpolation models)
    hist: std::collections::VecDeque<HistFrame>,
    gpu_interp_hist: std::collections::VecDeque<GpuInterpFrame>,
    interp_generation: u64,
    /// NeoFlow: previous frame kept GPU-resident
    prev_tex: Option<GpuTex>,
    /// Frame before prev_tex, used by NeoFlow's broad-transition protection.
    prev2_tex: Option<GpuTex>,
    prev_tex_seq: u64,
    prev_tex_source_time_100ns: Option<i64>,
    last_neoflow_log: Instant,
    monitor_refresh_hz: Option<f64>,
    monitor_rect: (i32, i32, i32, i32),
    /// windowed mode: overlay tracks source-window movement by delta
    last_src_pos: Option<(i32, i32)>,
    /// A real source title-bar/button drag is in progress. While held, the
    /// hidden source must be allowed to move naturally; repeatedly rebasing it
    /// fights the native move loop and eventually corrupts the input mapping.
    source_drag_active: bool,
    /// Chromium PIP/custom-frame windows can move from a drag that begins
    /// entirely in the client area, so input.rs never sees a native caption
    /// drag. Track that separately and rebase the hidden source exactly once
    /// after button-up; otherwise its client rect can remain beyond the
    /// desktop while ClipCursor still uses the full source rect, making the
    /// lower/right part of the magnified view physically unreachable.
    source_client_drag_active: bool,
    /// Recovery latch for an unowned client-only move.  A provider/input
    /// transition can clear SOURCE_CLIENT_DRAG_OWNER_HWND while Chromium is
    /// still inside the same physical LEFT move loop.  In that case the hidden
    /// source may continue moving off-desktop even though the visual follower
    /// is correctly suppressed.  Rebase it once after LEFT is released so the
    /// existing v423-v430 cursor reachability guarantees are not bypassed.
    source_client_unowned_rebase_pending: bool,
    /// Visual authority for a client-only window drag. The source HWND may hit
    /// Chromium/Windows off-screen constraints (observed x=-284 for a 640px
    /// PIP) while the user's raw WH_MOUSE_LL coordinate keeps travelling. Once
    /// a native client move proves that this gesture is a window drag, keep the
    /// overlay anchored to raw LEFT-down -> current movement instead of the
    /// source HWND's potentially clamped per-frame delta.
    source_client_drag_raw_origin: Option<(i32, i32)>,
    source_client_drag_overlay_origin: Option<(i32, i32)>,
    arrival_interval: f64,
    last_arrival: Instant,
    /// set once the 3s frame-starvation watchdog released the cursor
    starve_released: bool,
    /// last time the source window's client geometry changed (arms the watchdog)
    last_geom_change: Option<Instant>,
    /// FPS cap: next accept deadline of the target-rate decimator
    /// (wall-clock fallback when the frame has no WGC timestamp)
    next_cap_deadline: Option<Instant>,
    /// FPS cap: next accept deadline in the source-timestamp domain (100ns)
    next_cap_deadline_src: Option<i64>,
    cap_rate_gate: CapRateGate,
    /// Contiguous sequence seen by every filter after both pre-chain gates:
    /// FPS cap first, duplicate reduction second. WGC/cap/reuse gaps are input
    /// details and must not look like discontinuities to temporal filters.
    cap_filter_seq: u64,
    smooth_pacing: bool,
    source_is_elevated: bool,
    duplicate_frame_reduction: bool,
    duplicate_signature: Option<FrameSignature>,
    duplicate_skipped: u64,
    /// Stop duplicate reuse briefly after accumulated coherent movement is
    /// found. This prevents a slow zoom from being emitted in coarse steps.
    duplicate_motion_guard: u8,
    duplicate_lookahead_wait: Option<Instant>,
    strict_static_streak: u16,
    strict_static_last_present: Instant,
    strict_static_present_skipped: u64,
    /// Four-frame comb history and short scene latch. Telecined progressive
    /// capture often alternates a woven and a clean-looking frame; toggling
    /// NeoDeint on every frame leaves half the cadence unrepaired.
    neodeint_comb_history: u8,
    neodeint_scene_hold: u8,
    /// Smooth mode extracts the cadence of changing picture content from
    /// compositor-driven WGC updates (e.g. Chrome 24p video delivered at 30Hz).
    smooth_content_signature: Option<FrameSignature>,
    smooth_content_duplicates: u64,
    smooth_content_seq: u64,
    smooth_content_unique_since_duplicate: u8,
    smooth_content_last_candidate_seq: Option<u64>,
    smooth_content_last_candidate_time: Option<i64>,
    smooth_content_pattern_hits: u8,
    smooth_content_candidate_gaps: std::collections::VecDeque<u64>,
    smooth_content_24p_detected: bool,
    cadence: CadenceEstimator,
    smooth_pacer: SmoothPacer,
    /// Exact source-cadence presentation slot for a non-interpolated frame.
    paced_present_deadline: Option<Instant>,
    flow_output_cadence: FlowOutputCadence,
    present_cadence: PresentCadence,
    cap_diag: CapDiag,
    /// pipelined RIFE: inference worker + the pair awaiting presentation
    interp_worker: Option<InterpWorker>,
    interp_pending: Option<PendingInterp>,
    gpu_interp_worker: Option<GpuInterpWorker>,
    /// GL input packing has been issued; the render loop polls its fence and
    /// submits the provider job only after the shared buffers are ready.
    gpu_interp_pack_pending: Option<PendingGpuPack>,
    gpu_interp_pending: Option<PendingGpuInterp>,
    /// Results received during a short deadline-driven wait are staged here
    /// and consumed by the ordinary validation/import path on the next drain.
    gpu_interp_result_stash: std::collections::VecDeque<GpuInterpResult>,
    /// x4/x5 scheduling handshake: release the next provider timestep only
    /// after the current output's GL post-chain has been submitted. This
    /// prioritizes the frame that is about to be presented without waiting
    /// through SwapBuffers before the next unique RIFE/DRBA invocation starts.
    gpu_interp_permit_after_post_submit: Option<(u64, u64)>,
    gpu_interp_pair_id: u64,
    gpu_interp_active_logged: bool,
    /// Cursor mapping stays released while a newly-selected interpolation
    /// provider performs its first real inference. This avoids treating a
    /// legitimate cold TensorRT build as a render-heartbeat failure.
    provider_transition_input_suspended: bool,
    provider_transition_input_suspended_since: Option<Instant>,
    /// diagnostics: previous pipelined-tick start (tick-to-tick gap)
    interp_last_tick: Option<Instant>,
    /// diagnostics: end of the previous pipelined tick (wait+take cost)
    interp_tail: Option<Instant>,
    /// Smooth-mode output clock. Unlike WGC arrival time, this advances by an
    /// exact source-period/factor step (20.833 ms for 24p x2).
    interp_present_deadline: Option<Instant>,
    /// A cold DirectML post-stage can be 2-3 ms slower until its command
    /// path and GPU clocks have been exercised. At 48fps that crosses the
    /// 20.83ms slot and leaves the capture queue permanently behind.
    interp_post_warm: bool,
    /// Suppress per-frame file logging for the correctness-first pre-chain
    /// readback route. Log only when its geometry or stage count changes.
    pre_chain_route_log_key: Option<(i32, i32, i32, i32, usize)>,
    /// DirectML RIFE has a fixed safety/performance input-height cap. Keep the
    /// diagnostic one-shot per source/limited geometry and chain position.
    dml_rife_limit_log_key: Option<(i32, i32, i32, i32, usize)>,
    /// source client rect as of the previous tick (geometry-change detection)
    last_client_rect: Option<(i32, i32, i32, i32)>,
    last_process_ms: f64,
    /// Pure filter/resample work before SwapBuffers. GPU interpolation pacing
    /// must use this instead of last_process_ms: the latter also contains any
    /// compositor wait and feeding that wait back into the next lead time can
    /// phase-shift x4/x5 onto the following refresh slot.
    last_compute_ms: f64,
    /// Diagnostic-only CPU-side timing of the render path. These values do not
    /// change scheduling or presentation; they only make diagnostic logs capable of
    /// separating user-chain submission, the internal display resampler,
    /// OUTPUT/SCALED post passes, pacing wait and the final present call.
    last_upload_submit_ms: f64,
    last_chain_submit_ms: f64,
    last_resample_submit_ms: f64,
    last_post_submit_ms: f64,
    last_pacer_wait_ms: f64,
    last_present_call_ms: f64,
    /// Time spent inside the most recent SwapBuffers/present call. A blocking
    /// present gives us a trustworthy DWM/vblank phase sample; a non-blocking
    /// present leaves the manual deadline as the authority.
    last_present_block_ms: f64,
    /// Diagnostic-only compositor pressure accumulator. It correlates long
    /// overlay SwapBuffers waits with the GUI-side `panel-frame-diag` timeline
    /// without changing any presentation or pacing decision.
    compositor_pressure_window_started: Instant,
    compositor_pressure_samples: u32,
    compositor_pressure_max_ms: f64,
    compositor_pressure_sum_ms: f64,
    last_filter_retry_log: Option<Instant>,
    metric_seq: u64,
    last_metrics_log: Instant,
    phase_diag_samples: u8,
    glsl_guard: GlslResponsivenessGuard,
    /// GLSL-containing chains that were already proven
    /// overloaded during this capture session. This survives chain edits so
    /// re-selecting the same heavy chain
    /// can be recognized after a single verification frame instead of another
    /// long saturation window. It is discarded on Stop because Session is.
    glsl_overload_history: std::collections::VecDeque<String>,
    glsl_overload_hint: bool,
    /// v348o: chain/preset replacement settle window. Guard observations made
    /// before this instant are discarded because provider/shader/cache setup
    /// can make the first replacement frames much slower than steady state.
    glsl_chain_settle_until: Option<Instant>,
    glsl_chain_immediate_until: Option<Instant>,
    /// v348k: a live capture-resolution change may make the current chain
    /// healthy without changing any filters. While the user remains in the
    /// GUI, admit guarded validation frames and clear the latched warning only
    /// after the new geometry is stably healthy and one final full-chain frame
    /// also passes.
    glsl_resolution_recheck_active: bool,
    glsl_resolution_recheck_started: Option<Instant>,
    glsl_resolution_recheck_healthy_since: Option<Instant>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GlslGuardMode {
    Off,
    Soft,
    Hard,
}

struct GlslResponsivenessGuard {
    mode: GlslGuardMode,
    overload_since: Option<Instant>,
    hard_overload_since: Option<Instant>,
    healthy_since: Option<Instant>,
    admission_phase: u8,
    last_log: Instant,
    skipped_since_log: u32,
}

impl Default for GlslResponsivenessGuard {
    fn default() -> Self {
        Self {
            mode: GlslGuardMode::Off,
            overload_since: None,
            hard_overload_since: None,
            healthy_since: None,
            admission_phase: 0,
            last_log: Instant::now() - Duration::from_secs(5),
            skipped_since_log: 0,
        }
    }
}

impl GlslResponsivenessGuard {
    fn set_mode(&mut self, next: GlslGuardMode, gpu: f32, frame_ms: f64, budget_ms: f64) {
        if self.mode == next {
            return;
        }
        self.mode = next;
        self.admission_phase = 0;
        log::info!(
            "glsl-responsiveness-guard: mode={:?} gpu={:.1}% frame_ms={:.2} budget_ms={:.2} action={}",
            next,
            gpu,
            frame_ms,
            budget_ms,
            match next {
                GlslGuardMode::Off => "normal-all-frames",
                GlslGuardMode::Soft => "process-1-of-2-hold-last-complete-frame",
                GlslGuardMode::Hard => "process-1-of-3-hold-last-complete-frame",
            }
        );
        self.last_log = Instant::now();
    }

    fn reset(&mut self) {
        self.mode = GlslGuardMode::Off;
        self.overload_since = None;
        self.hard_overload_since = None;
        self.healthy_since = None;
        self.admission_phase = 0;
        self.skipped_since_log = 0;
    }

    fn observe_processed(
        &mut self,
        eligible: bool,
        gpu: Option<f32>,
        frame_ms: f64,
        present_ms: f64,
        budget_ms: f64,
        allow_immediate: bool,
    ) {
        if !eligible {
            self.reset();
            return;
        }
        let Some(gpu) = gpu else {
            return;
        };
        let budget_ms = budget_ms.clamp(4.0, 1000.0);
        let soft_deadline_miss = frame_ms >= budget_ms * 1.75 || present_ms >= budget_ms * 1.25;
        let hard_deadline_miss = frame_ms >= budget_ms * 2.25 || present_ms >= budget_ms * 1.50;
        let soft_overload = gpu >= GLSL_GUARD_SOFT_GPU_PERCENT && soft_deadline_miss;
        let hard_overload = gpu >= GLSL_GUARD_HARD_GPU_PERCENT && hard_deadline_miss;

        // v348h fast path: a single *severe* saturated frame is enough evidence.
        // This intentionally requires BOTH GPU saturation and a large deadline
        // miss so shader compilation/CPU stalls on otherwise healthy hardware do
        // not latch the low-end guard. It cannot interrupt a GLSL frame already
        // executing, but it removes the old extra 250-500ms confirmation delay
        // immediately after that first bad frame returns.
        let immediate_hard = gpu >= GLSL_GUARD_HARD_GPU_PERCENT
            && ((frame_ms >= 120.0 && frame_ms >= budget_ms * 3.0)
                || (present_ms >= 80.0 && present_ms >= budget_ms * 2.0));
        let immediate_soft = gpu >= GLSL_GUARD_SOFT_GPU_PERCENT
            && ((frame_ms >= 90.0 && frame_ms >= budget_ms * 2.0)
                || (present_ms >= 60.0 && present_ms >= budget_ms * 1.5));
        if allow_immediate && immediate_hard {
            self.overload_since = Some(Instant::now());
            self.hard_overload_since = Some(Instant::now());
            self.healthy_since = None;
            self.set_mode(GlslGuardMode::Hard, gpu, frame_ms, budget_ms);
            return;
        }
        if allow_immediate && immediate_soft {
            self.overload_since = Some(Instant::now());
            self.hard_overload_since = None;
            self.healthy_since = None;
            self.set_mode(GlslGuardMode::Soft, gpu, frame_ms, budget_ms);
            return;
        }

        if hard_overload {
            let since = self.hard_overload_since.get_or_insert_with(Instant::now);
            self.overload_since.get_or_insert_with(Instant::now);
            self.healthy_since = None;
            if since.elapsed() >= GLSL_GUARD_HARD_HOLD {
                self.set_mode(GlslGuardMode::Hard, gpu, frame_ms, budget_ms);
            }
            return;
        }
        self.hard_overload_since = None;

        if soft_overload {
            let since = self.overload_since.get_or_insert_with(Instant::now);
            self.healthy_since = None;
            if since.elapsed() >= GLSL_GUARD_SOFT_HOLD {
                self.set_mode(GlslGuardMode::Soft, gpu, frame_ms, budget_ms);
            }
            return;
        }

        self.overload_since = None;
        if self.mode == GlslGuardMode::Off {
            self.healthy_since = None;
            return;
        }

        let healthy = gpu <= GLSL_GUARD_RELEASE_GPU_PERCENT
            && frame_ms <= budget_ms * 1.35
            && present_ms <= budget_ms * 1.10;
        if healthy {
            let since = self.healthy_since.get_or_insert_with(Instant::now);
            if since.elapsed() >= GLSL_GUARD_RELEASE_HOLD {
                self.set_mode(GlslGuardMode::Off, gpu, frame_ms, budget_ms);
                self.healthy_since = None;
            }
        } else {
            self.healthy_since = None;
        }
    }

    fn should_skip_frame(&mut self) -> bool {
        match self.mode {
            GlslGuardMode::Off => false,
            GlslGuardMode::Soft => {
                self.admission_phase = (self.admission_phase + 1) % 2;
                self.admission_phase != 0
            }
            GlslGuardMode::Hard => {
                self.admission_phase = (self.admission_phase + 1) % 3;
                self.admission_phase != 0
            }
        }
    }

    fn note_skipped(&mut self, gpu: Option<f32>) {
        self.skipped_since_log = self.skipped_since_log.saturating_add(1);
        if self.last_log.elapsed() >= Duration::from_secs(1) {
            log::info!(
                "glsl-responsiveness-frame-hold: mode={:?} skipped={} gpu={:.1}% action=keep-last-complete-frame",
                self.mode,
                self.skipped_since_log,
                gpu.unwrap_or(-1.0)
            );
            self.skipped_since_log = 0;
            self.last_log = Instant::now();
        }
    }
}

fn interp_post_glsl_start(s: &Session) -> Option<usize> {
    let interp_index = s.chain.interp_index()?;
    let start = (interp_index + 1).min(s.chain.stage_count());
    let post = s.chain.stages.iter().skip(start).collect::<Vec<_>>();
    let first_glsl = post
        .iter()
        .position(|stage| stage.kind() == StageKind::Glsl)?;
    // Bypass is dependency-safe only when GLSL forms the trailing suffix. A
    // later ONNX stage could depend on a GLSL resize/transform, so do not alter
    // such a route automatically. ONNX-before-GLSL remains safe and is kept.
    post.iter()
        .skip(first_glsl)
        .all(|stage| stage.kind() == StageKind::Glsl)
        .then_some(start)
}

fn glsl_guard_eligible(s: &Session) -> bool {
    // v348e: ordinary GLSL and ONNX+GLSL chains stay protected exactly as in
    // v348d. Interpolation routes become eligible only when GLSL exists AFTER
    // the interpolation stage. Pre-interpolation GLSL is deliberately not
    // bypassed because changing the interpolator input would alter semantics.
    if !s.chain.has_glsl() {
        return false;
    }
    if !s.chain.has_interp() {
        return true;
    }
    interp_post_glsl_start(s).is_some()
}

fn glsl_chain_key(chain: &FilterChain) -> String {
    chain
        .stages
        .iter()
        .map(|stage| stage.name())
        .collect::<Vec<_>>()
        .join("\u{1f}")
}

fn remember_glsl_overload(s: &mut Session) {
    if !glsl_guard_eligible(s) {
        return;
    }
    let key = glsl_chain_key(&s.chain);
    if key.is_empty() || s.glsl_overload_history.iter().any(|known| known == &key) {
        return;
    }
    const MAX_HISTORY: usize = 8;
    if s.glsl_overload_history.len() >= MAX_HISTORY {
        s.glsl_overload_history.pop_front();
    }
    log::info!(
        "glsl-responsiveness-history: action=remember-overloaded-chain stages={:?}",
        s.chain
            .stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
    );
    s.glsl_overload_history.push_back(key);
}

fn glsl_guard_budget_ms(s: &Session) -> f64 {
    s.cadence
        .period_s()
        .map(|seconds| seconds * 1000.0)
        .filter(|ms| ms.is_finite() && *ms > 0.0)
        .or_else(|| {
            s.fps_cap
                .filter(|fps| *fps > 0)
                .map(|fps| 1000.0 / fps as f64)
        })
        .or_else(|| {
            s.monitor_refresh_hz
                .filter(|hz| hz.is_finite() && *hz > 1.0)
                .map(|hz| 1000.0 / hz)
        })
        .unwrap_or(1000.0 / 60.0)
}

fn update_glsl_guard_after_processed(s: &mut Session, status: &Arc<Mutex<Status>>) {
    let eligible = glsl_guard_eligible(s);
    let gpu = latest_gui_gpu_percent();
    let frame_ms = s.last_process_ms.max(0.0);
    let present_ms = s.last_present_call_ms.max(s.last_present_block_ms).max(0.0);
    let budget_ms = glsl_guard_budget_ms(s);
    let before = s.glsl_guard.mode;

    // v348o: chain replacement is a transient workload, not steady-state load.
    // Discard observations during the settle window instead of teaching the
    // overload history from shader/provider/cache warmup. This is intentionally
    // time-based: if one catastrophic processed frame itself takes >1s, this
    // check runs after the deadline and the frame is judged immediately.
    let now = Instant::now();
    if eligible {
        if let Some(until) = s.glsl_chain_settle_until {
            if now < until {
                s.glsl_guard.reset();
                return;
            }
            s.glsl_chain_settle_until = None;
            // Start post-transition confirmation from a clean sample boundary.
            s.glsl_guard.reset();
            log::info!(
                "glsl-responsiveness-settle: phase=complete action=begin-steady-state-evaluation"
            );
        }
    } else {
        s.glsl_chain_settle_until = None;
        s.glsl_chain_immediate_until = None;
    }

    // v348p: while a latched warning is being re-evaluated after a capture
    // resolution change, do not let the *old* Soft/Hard admission state keep
    // skipping frames.  Run the full new geometry for one second, then feed
    // those settled frames into the ordinary fresh-session guard below.
    if s.glsl_resolution_recheck_active {
        let started = s
            .glsl_resolution_recheck_started
            .get_or_insert_with(Instant::now);
        if started.elapsed() < GLSL_GUARD_RESOLUTION_RECHECK_SETTLE {
            s.glsl_guard.reset();
            return;
        }
    }

    let allow_immediate = match s.glsl_chain_immediate_until {
        Some(until) if now < until => false,
        Some(_) => {
            s.glsl_chain_immediate_until = None;
            true
        }
        None => true,
    };

    // If the user re-selects a chain that was already proven overloaded during
    // this capture session, require one *settled and GPU-saturated* verification
    // frame. v348n accepted frame time alone here, so a cache/provider transition
    // spike could poison the history and later fast-reconfirm a healthy preset.
    if eligible && allow_immediate && s.glsl_overload_hint && before == GlslGuardMode::Off {
        let budget = budget_ms.clamp(4.0, 1000.0);
        let gpu_value = gpu.unwrap_or(-1.0);
        let severe = gpu_value >= GLSL_GUARD_SOFT_GPU_PERCENT
            && frame_ms >= 80.0
            && frame_ms >= budget * 2.0;
        if severe {
            let next = if gpu_value >= GLSL_GUARD_HARD_GPU_PERCENT
                && (frame_ms >= budget * 2.25 || present_ms >= budget * 1.50)
            {
                GlslGuardMode::Hard
            } else {
                GlslGuardMode::Soft
            };
            s.glsl_guard.set_mode(next, gpu_value, frame_ms, budget);
            s.glsl_overload_hint = false;
            remember_glsl_overload(s);
            {
                let mut state = status.lock().unwrap();
                if !state.glsl_overload_notice_latched {
                    state.glsl_overload_notice_latched = true;
                    log::info!("glsl-overload-notice: latched=true reason=fast-reconfirm");
                }
                if state.glsl_overload_auto_stop_deadline.is_none() {
                    state.glsl_overload_auto_stop_deadline =
                        Some(Instant::now() + GLSL_OVERLOAD_AUTO_STOP_GRACE);
                    log::info!(
                        "glsl-overload-auto-stop: armed grace_ms={} reason=fast-reconfirm action=warn-then-normal-stop",
                        GLSL_OVERLOAD_AUTO_STOP_GRACE.as_millis()
                    );
                }
            }
            log::info!(
                "glsl-responsiveness-history: action=fast-reconfirm mode={next:?} gpu={gpu_value:.1}% frame_ms={frame_ms:.2} budget_ms={budget:.2}"
            );
            return;
        }
        if frame_ms <= budget * 1.35 && present_ms <= budget * 1.10 {
            s.glsl_overload_hint = false;
            log::info!(
                "glsl-responsiveness-history: action=clear-hint reason=current-load-healthy frame_ms={frame_ms:.2} budget_ms={budget:.2}"
            );
        }
    }

    s.glsl_guard.observe_processed(
        eligible,
        gpu,
        frame_ms,
        present_ms,
        budget_ms,
        allow_immediate,
    );
    if s.glsl_guard.mode != GlslGuardMode::Off {
        let mut state = status.lock().unwrap();
        if !state.glsl_overload_notice_latched {
            state.glsl_overload_notice_latched = true;
            log::info!(
                "glsl-overload-notice: latched=true mode={:?} reason=overload-proven",
                s.glsl_guard.mode
            );
        }
        if state.glsl_overload_auto_stop_deadline.is_none() {
            state.glsl_overload_auto_stop_deadline =
                Some(Instant::now() + GLSL_OVERLOAD_AUTO_STOP_GRACE);
            log::info!(
                "glsl-overload-auto-stop: armed grace_ms={} mode={:?} reason=overload-proven action=warn-then-normal-stop",
                GLSL_OVERLOAD_AUTO_STOP_GRACE.as_millis(),
                s.glsl_guard.mode
            );
        }
    }
    if s.glsl_guard.mode != GlslGuardMode::Off && s.glsl_guard.mode != before {
        remember_glsl_overload(s);
        s.glsl_overload_hint = false;
    }
    update_glsl_resolution_recheck(s, status, gpu, frame_ms, present_ms, budget_ms);
}

fn clear_glsl_resolution_recheck(s: &mut Session) {
    s.glsl_resolution_recheck_active = false;
    s.glsl_resolution_recheck_started = None;
    s.glsl_resolution_recheck_healthy_since = None;
}

fn update_glsl_resolution_recheck(
    s: &mut Session,
    status: &Arc<Mutex<Status>>,
    gpu: Option<f32>,
    frame_ms: f64,
    present_ms: f64,
    budget_ms: f64,
) {
    if !s.glsl_resolution_recheck_active {
        return;
    }

    let notice_still_latched = status
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .glsl_overload_notice_latched;
    if !notice_still_latched {
        clear_glsl_resolution_recheck(s);
        return;
    }

    let started = s
        .glsl_resolution_recheck_started
        .get_or_insert_with(Instant::now);
    let elapsed = started.elapsed();
    let timed_out = elapsed >= GLSL_GUARD_RESOLUTION_RECHECK_TIMEOUT;

    // The caller suppresses guard admission during this window.  Keep this
    // branch as an extra safety net in case the update ordering changes later.
    if elapsed < GLSL_GUARD_RESOLUTION_RECHECK_SETTLE {
        s.glsl_guard.reset();
        s.glsl_resolution_recheck_healthy_since = None;
        return;
    }

    let Some(gpu) = gpu else {
        if timed_out {
            log::info!(
                "glsl-overload-notice: resolution-recheck-kept reason=gpu-sample-unavailable"
            );
            clear_glsl_resolution_recheck(s);
        }
        return;
    };

    let budget = budget_ms.clamp(4.0, 1000.0);
    // IMPORTANT: use the *same* overload definition as a fresh session.  The
    // previous v348k/o recovery path required gpu<=82% plus a much tighter
    // deadline even though a fresh capture is allowed to run at high GPU load
    // when it is not missing the normal guard deadline.  That mismatch is why
    // Stop -> Start could be accepted while a live resolution reduction kept
    // the warning latched forever.
    let soft_deadline_miss = frame_ms >= budget * 1.75 || present_ms >= budget * 1.25;
    let overloaded_now = gpu >= GLSL_GUARD_SOFT_GPU_PERCENT && soft_deadline_miss;

    // If the ordinary guard itself reconfirms Soft/Hard on the new geometry,
    // the warning is genuinely still required.  No special recovery threshold
    // is allowed to overrule that decision.
    if s.glsl_guard.mode != GlslGuardMode::Off {
        log::info!(
            "glsl-overload-notice: resolution-recheck-kept reason=normal-guard-reconfirmed mode={:?} gpu={gpu:.1}% frame_ms={frame_ms:.2} budget_ms={budget:.2}",
            s.glsl_guard.mode
        );
        clear_glsl_resolution_recheck(s);
        return;
    }

    if overloaded_now {
        s.glsl_resolution_recheck_healthy_since = None;
    } else {
        let since = s
            .glsl_resolution_recheck_healthy_since
            .get_or_insert_with(Instant::now);
        if since.elapsed() >= GLSL_GUARD_RELEASE_HOLD {
            let mut state = status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.glsl_overload_notice_latched {
                state.glsl_overload_notice_latched = false;
                state.glsl_overload_auto_stop_deadline = None;
                log::info!(
                    "glsl-overload-notice: latched=false reason=capture-resolution-fresh-equivalent gpu={gpu:.1}% frame_ms={frame_ms:.2} budget_ms={budget:.2}"
                );
            }
            drop(state);
            clear_glsl_resolution_recheck(s);
            return;
        }
    }

    if timed_out {
        log::info!(
            "glsl-overload-notice: resolution-recheck-kept reason=new-geometry-not-stably-accepted gpu={gpu:.1}% frame_ms={frame_ms:.2} budget_ms={budget:.2}"
        );
        clear_glsl_resolution_recheck(s);
    }
}

/// FPS-cap diagnostics separate upstream delivery limits from local decimation:
/// source pacing / DWM / GPU clocks) from intentional decimation so diagnostics
/// can tell which side is responsible for a capture-fps reading.
struct CapDiag {
    last_log: Instant,
    delivered0: u64,
    taken: u32,
    accepted: u32,
    passthrough: u32,
    /// EWMA of the SOURCE-timestamp delta between consecutive deliveries (ms)
    src_dt_ms: f64,
    /// min/max delta inside the log window (exposes cadence patterns like
    /// 3:2 pulldown that an average hides)
    dt_min_ms: f64,
    dt_max_ms: f64,
    prev_ts: Option<i64>,
}

impl CapDiag {
    fn new() -> Self {
        Self {
            last_log: Instant::now(),
            delivered0: 0,
            taken: 0,
            accepted: 0,
            passthrough: 0,
            src_dt_ms: 0.0,
            dt_min_ms: f64::INFINITY,
            dt_max_ms: 0.0,
            prev_ts: None,
        }
    }
}

/// Deadline-accumulator decimation in an arbitrary monotonic time domain.
/// Accepts the frame at `t` when the deadline has been reached, then advances
/// the deadline by EXACTLY `interval` — so the accepted rate locks onto the
/// target no matter how the input grid beats against it. Resyncs instead of
/// bursting after a gap (static content, window occlusion).
///
/// The quarter-interval TOLERANCE is essential: a 24p video composited at
/// 60Hz arrives as a 33.3ms/50ms (3:2 pulldown) grid whose short step sits
/// EXACTLY on the cap-30 interval — without tolerance, timestamp jitter
/// rejected those frames and a 24fps source under a HIGHER cap played at
/// below the requested rate. A cap ≥ the source rate must pass every frame; a
/// true 60→30 halving is unaffected (its early frames are half an interval
/// early, far beyond the tolerance).
fn cap_accept(deadline: &mut Option<i64>, t: i64, interval: i64) -> bool {
    let tol = interval / 4;
    match *deadline {
        Some(d) if t < d - tol => false,
        Some(d) => {
            let next = d + interval;
            *deadline = Some(if next + tol <= t { t + interval } else { next });
            true
        }
        None => {
            *deadline = Some(t + interval);
            true
        }
    }
}

#[derive(Default)]
struct CapRateGate {
    prev_t: Option<i64>,
    intervals: std::collections::VecDeque<i64>,
}

impl CapRateGate {
    /// Returns true when the measured source cadence is already at or below
    /// the requested cap. In that case decimation must be bypassed completely:
    /// e.g. a 24p source under a 30fps cap remains 24p, including irregular
    /// DWM timestamps such as 16.7/50/66.7ms.
    fn source_is_within_cap(&mut self, t: i64, cap_interval: i64) -> bool {
        if let Some(prev) = self.prev_t {
            let dt = t - prev;
            if dt > 0 && dt < 5 * 10_000_000 {
                if dt > cap_interval.saturating_mul(10) {
                    self.intervals.clear();
                } else {
                    self.intervals.push_back(dt);
                    while self.intervals.len() > 24 {
                        self.intervals.pop_front();
                    }
                }
            }
        }
        self.prev_t = Some(t);

        // Preserve an unknown low-rate source during warm-up, but engage
        // immediately once one measured interval proves the source is far above the cap
        // (for example 60fps -> 15fps). This prevents the heavy chain from
        // processing a burst of native-rate startup frames.
        if self.intervals.len() < 8 {
            if !self.intervals.is_empty() {
                let mean =
                    self.intervals.iter().copied().sum::<i64>() / self.intervals.len() as i64;
                if mean.saturating_mul(5) < cap_interval.saturating_mul(2) {
                    return false;
                }
            }
            return true;
        }
        let mut samples: Vec<i64> = self.intervals.iter().copied().collect();
        samples.sort_unstable();
        let trim = (samples.len() / 8).max(1);
        let kept = &samples[trim..samples.len() - trim];
        let mean = kept.iter().map(|v| *v as i128).sum::<i128>() / kept.len() as i128;
        // Two percent of headroom handles 29.97/30 and timestamp quantization
        // without treating a genuinely faster source as cap-compliant.
        mean * 100 >= cap_interval as i128 * 98
    }
}

#[derive(Default)]
struct CadenceEstimator {
    prev_t: Option<i64>,
    prev_seq: Option<u64>,
    intervals: std::collections::VecDeque<i64>,
    /// Intervals are between visually distinct pictures. Their multimodal
    /// distribution (33/66ms for 24p in a 30Hz compositor) is meaningful and
    /// must not be trimmed like delivery jitter.
    content_aware: bool,
    forced_period_s: Option<f64>,
}

impl CadenceEstimator {
    #[allow(dead_code)] // retained for cadence diagnostics and focused tests
    fn observe(&mut self, t: Option<i64>) {
        let next_seq = self.prev_seq.unwrap_or(0).saturating_add(1);
        self.observe_frame(t, next_seq);
    }

    fn observe_frame(&mut self, t: Option<i64>, seq: u64) {
        let Some(t) = t else { return };
        if let Some(prev) = self.prev_t {
            let dt = t - prev;
            if dt <= 0 || dt >= 5_000_000 {
                self.intervals.clear();
            } else {
                let steps = seq
                    .saturating_sub(self.prev_seq.unwrap_or(seq.saturating_sub(1)))
                    .max(1);
                // v515 regression fix: restore the v454/v235 rolling cadence
                // estimator. A short burst of compositor-rate callbacks must
                // not replace an already converged video cadence. At heavy
                // 1080p interpolation load that eager regime switch could
                // reinterpret transient delivery timing as source timing and
                // destabilize interpolation phases until a seek/restart.
                // Genuine rate changes still converge naturally through the
                // rolling 120-sample history.
                self.intervals.push_back(dt / steps as i64);
                while self.intervals.len() > 120 {
                    self.intervals.pop_front();
                }
            }
        }
        self.prev_t = Some(t);
        self.prev_seq = Some(seq);
    }

    fn observe_content(&mut self, t: Option<i64>) {
        let next_seq = self.prev_seq.unwrap_or(0).saturating_add(1);
        self.observe_frame(t, next_seq);
        self.content_aware = true;
    }

    fn period_s(&self) -> Option<f64> {
        if let Some(period) = self.forced_period_s {
            return Some(period);
        }
        if self.intervals.len() < 8 {
            return None;
        }
        let mut samples: Vec<i64> = self.intervals.iter().copied().collect();
        samples.sort_unstable();
        let trim = if self.content_aware {
            0
        } else {
            (samples.len() / 8).max(1)
        };
        let kept = &samples[trim..samples.len() - trim];
        let mean =
            kept.iter().map(|v| *v as i128).sum::<i128>() as f64 / kept.len() as f64 / 10_000_000.0;
        if !mean.is_finite() || !(1.0 / 240.0..=0.2).contains(&mean) {
            return None;
        }
        let mean_fps = 1.0 / mean;
        if (28.8..32.5).contains(&mean_fps) {
            // Do not turn a variable-rate game whose *average* happens to be
            // near 30fps into fixed 30p. Lock only when at least 75% of recent
            // intervals form a real 30p cluster (roughly 26-37fps per sample).
            let clustered = kept
                .iter()
                .filter(|sample| (270_000..=385_000).contains(*sample))
                .count();
            if clustered * 4 >= kept.len() * 3 {
                return Some(1.0 / 30.0);
            }
            return Some(mean);
        }
        Some(snap_video_period_with_history(mean, self.intervals.len()))
    }

    /// True when recent timestamps have no dominant frame interval. Smooth
    /// pacing must then follow each arrival instead of imposing an averaged
    /// fixed clock. Stable 24p carried by an irregular compositor grid is
    /// exempt because its long-term cadence is intentionally reconstructed.
    fn variable_rate(&self) -> bool {
        if self.forced_period_s.is_some() || self.content_aware || self.intervals.len() < 16 {
            return false;
        }
        let samples: Vec<i64> = self.intervals.iter().copied().collect();
        let mean = samples.iter().map(|v| *v as f64).sum::<f64>() / samples.len() as f64;
        if !mean.is_finite() || mean <= 0.0 {
            return false;
        }
        let mean_fps = 10_000_000.0 / mean;
        if (21.5..24.75).contains(&mean_fps) {
            return false;
        }
        const PERIODS_100NS: &[i64] = &[
            41_708,  // 240
            69_444,  // 144
            83_417,  // 119.88
            100_000, // 100
            111_111, // 90
            166_667, // 60
            200_000, // 50
            333_333, // 30
            400_000, // 25
            416_667, // 24
        ];
        let dominant = PERIODS_100NS
            .iter()
            .map(|period| {
                samples
                    .iter()
                    .filter(|sample| {
                        ((**sample - *period).unsigned_abs() as f64 / *period as f64) <= 0.12
                    })
                    .count()
            })
            .max()
            .unwrap_or(0);
        dominant * 3 < samples.len() * 2
    }

    fn reset(&mut self) {
        self.prev_t = None;
        self.prev_seq = None;
        self.intervals.clear();
        self.content_aware = false;
        self.forced_period_s = None;
    }

    fn force_period(&mut self, period_s: f64) {
        self.forced_period_s = Some(period_s);
    }
}

fn normalized_frame_span_s(
    prev_time_100ns: Option<i64>,
    cur_time_100ns: Option<i64>,
    prev_seq: u64,
    cur_seq: u64,
) -> Option<f64> {
    let (Some(prev), Some(cur)) = (prev_time_100ns, cur_time_100ns) else {
        return None;
    };
    if cur <= prev {
        return None;
    }
    let steps = cur_seq.saturating_sub(prev_seq).max(1);
    Some((cur - prev) as f64 / 10_000_000.0 / steps as f64)
}

fn neoflow_source_period_s(
    stable_cadence: Option<f64>,
    pair_period: Option<f64>,
    arrival_interval: f64,
) -> Option<f64> {
    // WGC timestamps for 24p on a desktop compositor commonly alternate
    // between short and long intervals. Once the cadence estimator has enough
    // samples, it is more trustworthy than one pair. Preferring the pair made
    // NeoFlow briefly classify 24p as 60p and disable x2 on a 60Hz monitor.
    stable_cadence
        .or(pair_period)
        .or_else(|| (arrival_interval > 0.0).then_some(arrival_interval))
}

#[allow(dead_code)] // compatibility helper for cadence regression tests
fn snap_video_period(measured: f64) -> f64 {
    snap_video_period_with_history(measured, 0)
}

fn snap_video_period_with_history(measured: f64, _sample_count: usize) -> f64 {
    let measured_fps = 1.0 / measured;
    // 24p composed on a 60Hz desktop commonly measures below 23fps when a
    // A long ONNX pass can overlap WGC delivery; preserve nonblocking capture even
    // though the player source was fixed 24p). Keep that known cadence locked
    // to 24 instead of letting the pacer drift to a self-reinforcing 44-46ms.
    if (21.5..24.75).contains(&measured_fps) {
        // Keep the exact 24 Hz output clock. Switching an already locked
        // session to 24000/1001 after 120 samples moves the deadline phase and
        // can turn one late Chromium delivery into an 83 ms present hole.
        return 1.0 / 24.0;
    }
    const VIDEO_RATES: &[f64] = &[
        23.976, 24.0, 25.0, 29.97, 30.0, 47.952, 48.0, 50.0, 59.94, 60.0, 90.0, 100.0, 119.88,
        120.0, 144.0, 165.0, 240.0,
    ];
    let nearest = VIDEO_RATES.iter().copied().min_by(|a, b| {
        (measured_fps - *a)
            .abs()
            .total_cmp(&(measured_fps - *b).abs())
    });
    match nearest {
        Some(fps) if ((measured_fps - fps) / fps).abs() <= 0.035 => 1.0 / fps,
        _ => measured,
    }
}

fn refresh_limited_output_ratio(
    requested: u32,
    source_period_s: Option<f64>,
    refresh_hz: Option<f64>,
) -> f64 {
    let (Some(period), Some(refresh)) = (source_period_s, refresh_hz) else {
        return requested as f64;
    };
    if !period.is_finite()
        || !refresh.is_finite()
        || !(1.0 / 240.0..=0.2).contains(&period)
        || !(20.0..=1000.0).contains(&refresh)
    {
        return requested as f64;
    }
    // The GUI multiplier is a processing contract, not a request to fill the
    // monitor. x2 must remain exactly two outputs per source interval (24p ->
    // 48fps). Expanding x2 to 2.5x made the same RIFE chain alternate between
    // 60fps and a 42-50fps overload state depending on downstream cost.
    // Refresh-limited callers such as NeoFlow may still use 2.5x on a
    // 60Hz monitor. ONNX x3 reaches this helper only through its explicit
    // onnx_x3_60hz_mode() branch; every other refresh stays strict integer.
    let display_ratio = refresh * period;
    display_ratio.clamp(1.0, requested as f64)
}

/// Return true only for a monitor that is genuinely operating in the 60 Hz
/// family. Windows commonly reports 59.94 or 60.00 Hz, so keep a narrow
/// tolerance around 60 while deliberately excluding 59/61/75/120 Hz modes.
///
/// This predicate is the hard boundary for ONNX x3's 60fps adaptation. No
/// fractional x3 state is allowed to leak into any other refresh rate.
fn onnx_x3_60hz_mode(refresh_hz: Option<f64>) -> bool {
    refresh_hz.is_some_and(|refresh| {
        refresh.is_finite() && (refresh - 60.0).abs() <= 0.75
    })
}

/// ONNX x3 has two explicitly separated contracts.
///
/// * 59.94/60.00 Hz monitor: keep the established refresh-limited path so a
///   24p source targets 60fps (2.5x), 25p targets 60fps (2.4x), and 30p
///   targets 60fps (2.0x).
/// * Every other monitor refresh: strict integer x3, always 1/3, 2/3,
///   endpoint. The 60 Hz fractional accumulator is never consulted.
///
/// The explicit refresh predicate prevents startup/source-cadence noise from
/// accidentally selecting the 60 Hz path on 75/120/144/165/240 Hz displays.
fn onnx_output_ratio(
    requested: u32,
    source_period_s: Option<f64>,
    refresh_hz: Option<f64>,
) -> f64 {
    if requested == 3 {
        if onnx_x3_60hz_mode(refresh_hz) {
            refresh_limited_output_ratio(requested, source_period_s, refresh_hz)
        } else {
            3.0
        }
    } else {
        refresh_limited_output_ratio(requested, source_period_s, refresh_hz)
    }
}

#[derive(Default)]
struct FlowOutputCadence {
    next_phase: Option<f64>,
    phase_step: f64,
    next_present: Option<Instant>,
    present_period_s: f64,
    /// Actual back-buffer submission phase. Startup cadence estimation can
    /// move from a temporary 22-30fps estimate to the locked 24fps clock;
    /// rebasing from this timestamp keeps that transition on the compositor
    /// phase already proven to work instead of anchoring to an arbitrary loop
    /// iteration.
    last_present_started: Option<Instant>,
}

/// Multi-slot interpolation releases the next provider timestep after the
/// current output's GL work has been submitted. Integer x3/x4/x5 share the
/// same cooperative scheduling rule; x3's direct-output detach is synchronized
/// before this release so provider work does not race the copy.
fn gpu_interp_slot_policy(factor: u32, timestep_count: usize) -> (bool, bool) {
    if factor == 3 && timestep_count >= 2 {
        (true, true)
    } else if factor >= 4 && timestep_count >= 3 {
        (true, true)
    } else {
        (false, false)
    }
}

impl FlowOutputCadence {
    fn phases(&mut self, output_ratio: f64) -> Vec<f32> {
        let ratio = output_ratio.max(1.0);
        let step = 1.0 / ratio;
        if self.next_phase.is_none()
            || self.phase_step <= 0.0
            || ((step - self.phase_step) / self.phase_step).abs() > 0.02
        {
            // Start with the larger half of a fractional cadence. At 2.5x
            // this yields 3,2,3,2... outputs; starting 2,3 would leave a long
            // first gap because the next source frame has not arrived yet.
            let count = ratio.ceil().max(1.0);
            self.next_phase = Some((1.0 - (count - 1.0) * step).max(step * 0.25));
            self.phase_step = step;
            self.next_present = None;
        }

        let mut phase = self.next_phase.unwrap_or(step);
        let mut out = Vec::with_capacity(ratio.ceil() as usize);
        while phase <= 1.0 + 1e-6 {
            out.push(phase.min(1.0) as f32);
            phase += step;
        }
        self.next_phase = Some((phase - 1.0).max(step * 0.05));
        out
    }

    fn wait_for_present(
        &mut self,
        overlay: &mut OverlayWindow,
        period_s: f64,
        vsync_on: bool,
        smooth: bool,
    ) {
        self.wait_for_present_with_lead(overlay, period_s, vsync_on, smooth, 0.0);
    }

    /// Reserve one output slot on an absolute interpolation clock.
    ///
    /// Two details are intentionally handled here rather than in SwapBuffers:
    ///
    /// 1. During startup the cadence estimator may briefly report a temporary
    ///    source period before locking to 24/30fps.  Keeping deadlines created
    ///    from that temporary period was enough to make identical 4K x2
    ///    sessions start either near 48fps or around 44-46fps.  A material
    ///    period change therefore rebases from the *actual previous submit*
    ///    phase, not from an arbitrary render-loop timestamp.
    /// 2. If a compositor/provider stall misses one or more complete output
    ///    slots, advance along the existing phase grid instead of setting the
    ///    clock to `now`.  Re-anchoring to `now` permanently changed the DWM
    ///    phase and could turn a transient x4 miss into a sustained 89-94fps
    ///    state.  Phase-preserving advancement recovers the target clock
    ///    without presenting a burst of every missed slot.
    fn reserve_present_deadline(&mut self, now: Instant, period_s: f64) -> Instant {
        let period = Duration::from_secs_f64(period_s);
        let period_changed = self.present_period_s.is_finite()
            && self.present_period_s > 0.0
            && ((period_s - self.present_period_s).abs() / self.present_period_s.max(period_s))
                > 0.01;

        if period_changed {
            let old_period_ms = self.present_period_s * 1000.0;
            let anchor = self
                .last_present_started
                .and_then(|started| started.checked_add(period))
                .unwrap_or(now);
            self.next_present = Some(anchor);
            log::debug!(
                "interp-cadence-period-rebase: old_ms={:.3} new_ms={:.3} anchor=last-submit-phase",
                old_period_ms,
                period_s * 1000.0
            );
        }

        // Causal interpolation receives endpoint B one source period after A,
        // so the first in-between is already due when the pair becomes
        // available. Present that first real model output immediately; only
        // subsequent outputs wait for their 1/factor slots.
        let mut deadline = self.next_present.unwrap_or(now);
        if now > deadline + period {
            let late_s = now.duration_since(deadline).as_secs_f64();
            let elapsed_slots = (late_s / period_s).floor().max(1.0);
            deadline = deadline + period.mul_f64(elapsed_slots);
            log::debug!(
                "interp-cadence-phase-recover: late_ms={:.3} skipped_clock_slots={} period_ms={:.3}",
                late_s * 1000.0,
                elapsed_slots as u64,
                period_s * 1000.0
            );
        }

        self.next_present = Some(deadline + period);
        self.present_period_s = period_s;
        deadline
    }

    /// Start post-processing slightly before the presentation slot. GPU
    /// interpolation can then compute later, unique timesteps on its worker
    /// while the render thread prepares and displays the earliest ready slot.
    fn wait_for_present_with_lead(
        &mut self,
        overlay: &mut OverlayWindow,
        period_s: f64,
        vsync_on: bool,
        smooth: bool,
        lead_s: f64,
    ) {
        if !smooth || vsync_on || !period_s.is_finite() || period_s <= 0.0 {
            return;
        }
        let now = Instant::now();
        let deadline = self.reserve_present_deadline(now, period_s);
        let lead = Duration::from_secs_f64(lead_s.clamp(0.0, period_s * 0.90));
        let prepare_at = deadline.checked_sub(lead).unwrap_or(deadline);
        wait_until_with_pump(overlay, prepare_at);
        // reserve_present_deadline already advanced next_present by one exact
        // period.  Never anchor it to the post-SwapBuffers completion time.
        self.present_period_s = period_s;
    }

    /// Keep the interpolation presentation clock phase-locked without
    /// feeding normal SwapBuffers/DWM wait time back into the next interval.
    ///
    /// `present_started_at` is the instant the back buffer was submitted. A
    /// healthy compositor may then block inside SwapBuffers for 2-8 ms, but
    /// that wait is *inside* the current presentation and must not be added to
    /// the next x2/x3/x4/x5 slot. The old code anchored to the completion
    /// timestamp and therefore turned, for example, 20.833 ms x2 slots into
    /// roughly 23 ms slots on AMD.
    ///
    /// A genuinely late submission advances to a future slot on the same
    /// absolute phase grid. This avoids both burst catch-up and a permanent
    /// cadence shift caused by a random compositor return timestamp.
    fn observe_blocking_present(
        &mut self,
        present_started_at: Instant,
        _present_completed_at: Instant,
        period_s: f64,
        present_block_s: f64,
    ) -> bool {
        if !period_s.is_finite() || period_s <= 0.0 {
            return false;
        }
        self.last_present_started = Some(present_started_at);
        if !present_block_s.is_finite() || present_block_s < 0.000_5 {
            return false;
        }
        let period = Duration::from_secs_f64(period_s);
        let Some(next) = self.next_present else {
            self.next_present = Some(present_started_at + period);
            self.present_period_s = period_s;
            return true;
        };
        let scheduled = next.checked_sub(period).unwrap_or(next);
        // Compare the *submission* to the scheduled slot. SwapBuffers return
        // time is deliberately ignored here; otherwise a stable compositor
        // block is accumulated once per output and lower multipliers can run
        // slower than higher multipliers.
        let late = present_started_at.saturating_duration_since(scheduled);
        // Normal dispatch noise must not move the cadence clock on every
        // output. If a genuinely large stall spans multiple slots, keep the
        // original phase grid and advance `next_present` to its first future
        // slot. Anchoring to `present_started_at + period` made the new phase
        // depend on a random DWM/driver return time and could leave x2/x4 in a
        // persistently slower lock state until resolution change or restart.
        let recovery_threshold = period.mul_f64(2.0);
        if present_started_at >= scheduled && late > recovery_threshold {
            let late_to_next_s = present_started_at
                .saturating_duration_since(next)
                .as_secs_f64();
            let advance_slots = if present_started_at < next {
                0.0
            } else {
                (late_to_next_s / period_s).floor() + 1.0
            };
            if advance_slots > 0.0 {
                self.next_present = Some(next + period.mul_f64(advance_slots));
            }
            self.present_period_s = period_s;
            return true;
        }
        false
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Build the output phases for every pipelined ONNX interpolator.
///
/// x2 is a strict processing contract: one midpoint followed by the real
/// endpoint. A fractional/stale cadence accumulator must never turn it into
/// two synthetic frames. That doubled RIFE inference after a preset-loaded
/// pre-ONNX stage, while toggling the same stage rebuilt a clean 1-mid route.
/// x3 uses fractional cadence only behind the explicit 59.94/60.00Hz mode;
/// on every other refresh it is a strict integer grid. x4-x5 keep the existing
/// exact-grid / refresh-limited behavior.
fn onnx_interpolation_phases(
    factor: u32,
    output_ratio: f64,
    x3_60hz_mode: bool,
    cadence: &mut FlowOutputCadence,
) -> Vec<f32> {
    let factor = factor.clamp(2, 5);
    if factor == 3 {
        if x3_60hz_mode {
            // 60 Hz is the one intentional exception: use the refresh-limited
            // cadence so 24p -> 60fps can alternate 3,2,3,2... outputs. This
            // state is reachable only behind onnx_x3_60hz_mode().
            return cadence.phases(output_ratio);
        }
        // Every non-60Hz monitor is strict integer x3. Clear any previous
        // fractional phase state (for example after moving the source window
        // from a 60Hz monitor to a 120Hz monitor) before producing the fixed
        // 1/3, 2/3, endpoint timeline.
        cadence.next_phase = None;
        cadence.phase_step = 0.0;
        return vec![1.0 / 3.0, 2.0 / 3.0, 1.0];
    }
    // When the monitor can display the complete integer multiplier, use an
    // exact fixed phase grid. Carrying a fractional cadence accumulator into
    // 24p/120Hz x5 produced shifted phases such as 0.026/0.276/... and could
    // omit the real endpoint for several pairs. Besides uneven motion, that
    // wasted provider work and prevented the presentation queue from settling.
    if output_ratio >= factor as f64 - 0.02 {
        // Clear only the fractional phase accumulator. The presentation clock
        // must remain continuous across source pairs or every pair would begin
        // with an immediate burst.
        cadence.next_phase = None;
        cadence.phase_step = 0.0;
        return (1..=factor)
            .map(|index| index as f32 / factor as f32)
            .collect();
    }
    if factor == 2 && output_ratio >= 1.5 {
        vec![0.5, 1.0]
    } else {
        let mut phases = cadence.phases(output_ratio);
        // Never spend a fifth model invocation on an x5 startup sample. Until
        // cadence locks, a fractional estimate can temporarily produce five
        // synthetic phases with no real endpoint. They are all real model
        // frames, but the fifth exceeds the requested x5 generation budget
        // and only steals GPU time from the following source pair.
        if factor >= 4
            && phases.last().is_some_and(|phase| *phase < 1.0 - 1e-5)
            && phases.len() > factor.saturating_sub(1) as usize
        {
            phases.truncate(factor.saturating_sub(1) as usize);
        }
        phases
    }
}

struct SmoothPacer {
    next_present: Option<Instant>,
    process_samples_s: std::collections::VecDeque<f64>,
    compositor_paced: bool,
    blocking_streak: u8,
    nonblocking_streak: u8,
    long_interval_streak: u8,
    last_present_block_s: f64,
}

impl Default for SmoothPacer {
    fn default() -> Self {
        Self {
            next_present: None,
            process_samples_s: std::collections::VecDeque::with_capacity(32),
            compositor_paced: false,
            blocking_streak: 0,
            nonblocking_streak: 0,
            long_interval_streak: 0,
            last_present_block_s: 0.0,
        }
    }
}

impl SmoothPacer {
    fn processing_deadline(
        &mut self,
        now: Instant,
        period_s: Option<f64>,
        process_budget_s: f64,
        _monitor_refresh_hz: Option<f64>,
    ) -> Option<(Instant, Instant)> {
        let period_s = period_s?;
        if self.compositor_paced {
            return None;
        }
        // Keep the exact source clock. Present/DWM maps that
        // clock onto the actual monitor refresh phase: 24p naturally becomes
        // 2/3 holds at 60Hz and five holds at 120Hz. Generating 33/50ms waits
        // ourselves is mathematically equivalent in count but not phase-locked
        // to vblank, and measured almost twice the presentation jitter.
        let period = Duration::from_secs_f64(period_s);
        let mut present = self.next_present.unwrap_or(now + period);
        // Never "catch up" a missed slot by keeping the following deadline
        // on the old phase. That produces the perceptually bad long+short
        // pair seen with 30p WGC input (for example 48 ms then 18 ms), even
        // though the one-second average still reports 30 fps. Re-anchor at
        // the late frame; the next output remains a full period away.
        if now > present {
            present = now;
        }
        self.next_present = Some(present + period);
        let mut samples: Vec<f64> = self.process_samples_s.iter().copied().collect();
        samples.sort_by(f64::total_cmp);
        let measured_budget = if samples.is_empty() {
            process_budget_s
        } else {
            samples[((samples.len() - 1) * 9) / 10]
        };
        let budget_s = (measured_budget + 0.0015).clamp(0.0, period_s * 0.9);
        let processing = present
            .checked_sub(Duration::from_secs_f64(budget_s))
            .unwrap_or(now);
        Some((processing, present))
    }

    fn observe_process(&mut self, process_s: f64) {
        if process_s.is_finite() && process_s > 0.0 {
            self.process_samples_s.push_back(process_s);
            while self.process_samples_s.len() > 30 {
                self.process_samples_s.pop_front();
            }
        }
    }

    /// Diagnostic only. A short SwapBuffers block also occurs on healthy
    /// Chromium/DWM paths and must not by itself disable manual pacing.
    #[allow(dead_code)] // diagnostic hook retained for compositor pacing tests
    fn observe_present_block(&mut self, elapsed_s: f64, period_s: Option<f64>) -> Option<bool> {
        self.last_present_block_s = elapsed_s.max(0.0);
        let period_s = period_s.filter(|p| p.is_finite() && *p > 0.0)?;
        let definitely_paced = elapsed_s >= period_s * 0.72;
        let likely_paced = elapsed_s >= (period_s * 0.25).clamp(0.004, 0.008);

        if likely_paced {
            self.blocking_streak = self.blocking_streak.saturating_add(1);
            self.nonblocking_streak = 0;
        } else if elapsed_s <= 0.0025 {
            self.nonblocking_streak = self.nonblocking_streak.saturating_add(1);
            self.blocking_streak = 0;
        } else {
            self.blocking_streak = 0;
            self.nonblocking_streak = 0;
        }

        if !self.compositor_paced && (definitely_paced || self.blocking_streak >= 2) {
            self.compositor_paced = true;
            self.next_present = None;
            return Some(true);
        }
        None
    }

    fn observe_effective_present_interval(
        &mut self,
        elapsed_s: f64,
        period_s: Option<f64>,
    ) -> Option<bool> {
        if self.compositor_paced {
            return None;
        }
        let period_s = period_s.filter(|p| p.is_finite() && *p > 0.0)?;
        // A fullscreen transition and normal 24p-on-60Hz cadence can briefly
        // produce 1.6x intervals. Only a sustained near-2x interval proves
        // that DWM is really halving our output (the elevated 30 -> 15fps
        // failure), so do not permanently surrender pacing on a transient.
        if elapsed_s >= period_s * 1.85 {
            self.long_interval_streak = self.long_interval_streak.saturating_add(1);
        } else {
            self.long_interval_streak = 0;
        }
        if self.long_interval_streak >= 3 {
            self.compositor_paced = true;
            self.next_present = None;
            return Some(true);
        }
        None
    }

    fn reanchor_after_missed_present(
        &mut self,
        now: Instant,
        deadline: Instant,
        period_s: Option<f64>,
    ) {
        let Some(period_s) = period_s.filter(|p| p.is_finite() && *p > 0.0) else {
            return;
        };
        if now > deadline + Duration::from_millis(1) {
            self.next_present = Some(now + Duration::from_secs_f64(period_s));
        }
    }

    /// The no-new-WGC-frame safety path can re-present the last filtered
    /// texture after a genuinely missed source slot. That present is outside
    /// the normal paced processing path, so the old manual deadline must not
    /// survive it; otherwise the next real source frame can be presented only
    /// a few milliseconds later as a catch-up burst.
    fn reanchor_after_gap_fill_present(&mut self, now: Instant, period_s: Option<f64>) {
        if self.compositor_paced {
            return;
        }
        let Some(period_s) = period_s.filter(|p| p.is_finite() && *p > 0.0) else {
            return;
        };
        self.next_present = Some(now + Duration::from_secs_f64(period_s));
    }

    fn reset(&mut self) {
        self.next_present = None;
        self.compositor_paced = false;
        self.blocking_streak = 0;
        self.nonblocking_streak = 0;
        self.long_interval_streak = 0;
        self.last_present_block_s = 0.0;
    }
}

fn wait_until_with_pump(overlay: &mut OverlayWindow, deadline: Instant) {
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        overlay.win.pump_messages();
        let left = deadline.saturating_duration_since(Instant::now());
        if left > Duration::from_millis(2) {
            std::thread::sleep(
                left.saturating_sub(Duration::from_micros(600))
                    .min(Duration::from_millis(1)),
            );
        } else if left > Duration::from_micros(350) {
            std::thread::yield_now();
        } else {
            std::hint::spin_loop();
        }
    }
}

#[derive(Default)]
struct PresentCadence {
    last: Option<Instant>,
    intervals_ms: std::collections::VecDeque<f64>,
}

impl PresentCadence {
    fn record(&mut self, now: Instant) {
        if let Some(last) = self.last {
            let ms = now.duration_since(last).as_secs_f64() * 1000.0;
            if (0.1..500.0).contains(&ms) {
                self.intervals_ms.push_back(ms);
                while self.intervals_ms.len() > 120 {
                    self.intervals_ms.pop_front();
                }
            }
        }
        self.last = Some(now);
    }

    fn summary(&self) -> (f64, f64, f64, f64) {
        if self.intervals_ms.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let n = self.intervals_ms.len() as f64;
        let mean = self.intervals_ms.iter().sum::<f64>() / n;
        let variance = self
            .intervals_ms
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / n;
        let min = self
            .intervals_ms
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let max = self.intervals_ms.iter().copied().fold(0.0, f64::max);
        (mean, variance.sqrt(), min, max)
    }
}

fn pace_processing_start(s: &mut Session, overlay: &mut OverlayWindow) -> Instant {
    if s.smooth_pacing && !s.cadence.variable_rate() {
        if let Some((processing_deadline, present_deadline)) = s.smooth_pacer.processing_deadline(
            Instant::now(),
            s.cadence.period_s(),
            s.last_process_ms / 1000.0,
            s.monitor_refresh_hz,
        ) {
            s.paced_present_deadline = Some(present_deadline);
            wait_until_with_pump(overlay, processing_deadline);
        } else {
            s.paced_present_deadline = None;
        }
    } else {
        s.paced_present_deadline = None;
    }
    Instant::now()
}

#[derive(Clone)]
struct HistFrame {
    seq: u64,
    received_at: Option<Instant>,
    source_time_100ns: Option<i64>,
    w: i32,
    h: i32,
    /// Arc so the interp worker thread can borrow frames without an 8MB copy
    data: Arc<Vec<u8>>,
}

#[derive(Clone, Copy)]
struct GpuInterpFrame {
    tex: GpuTex,
    seq: u64,
    received_at: Option<Instant>,
    source_time_100ns: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GpuInterpContinuity {
    Continuous,
    /// WGC delivered the same video picture again.  Preserve interpolation
    /// history and do not manufacture another output pair from it.
    Duplicate,
    Broken,
}

fn classify_gpu_interp_continuity(
    previous_seq: u64,
    previous_received_at: Option<Instant>,
    previous_source_time_100ns: Option<i64>,
    current: &FrameBuf,
    cadence_period: Option<f64>,
    arrival_interval: f64,
) -> GpuInterpContinuity {
    if current.seq <= previous_seq {
        return GpuInterpContinuity::Broken;
    }
    let baseline = cadence_period
        .filter(|period| period.is_finite() && *period > 0.0 && *period <= 0.2)
        .or_else(|| {
            (arrival_interval.is_finite() && arrival_interval > 0.0 && arrival_interval <= 0.2)
                .then_some(arrival_interval)
        })
        .unwrap_or(1.0 / 24.0);
    // WGC sequence numbers count compositor deliveries, not unique video
    // pictures. take_next_queued may legitimately skip many sequence values
    // after a resize or while x4/x5 output is being paced. Time continuity is
    // the reliable signal; a raw seq gap must not reset history every frame.
    let max_gap_s = (baseline * 4.5).clamp(0.12, 0.5);
    if let (Some(a), Some(b)) = (previous_source_time_100ns, current.source_time_100ns) {
        if b < a {
            return GpuInterpContinuity::Broken;
        }
        if b == a {
            return GpuInterpContinuity::Duplicate;
        }
        return if (b - a) as f64 / 10_000_000.0 > max_gap_s {
            GpuInterpContinuity::Broken
        } else {
            GpuInterpContinuity::Continuous
        };
    }
    if let (Some(a), Some(b)) = (previous_received_at, current.received_at) {
        let gap = b.saturating_duration_since(a).as_secs_f64();
        return if gap > max_gap_s {
            GpuInterpContinuity::Broken
        } else {
            GpuInterpContinuity::Continuous
        };
    }
    GpuInterpContinuity::Continuous
}

impl HistFrame {
    fn from_frame(frame: &FrameBuf, data: Arc<Vec<u8>>) -> Self {
        Self {
            seq: frame.seq,
            received_at: frame.received_at,
            source_time_100ns: frame.source_time_100ns,
            w: frame.w,
            h: frame.h,
            data,
        }
    }

    fn from_processed(frame: &FrameBuf, w: i32, h: i32, data: Arc<Vec<u8>>) -> Self {
        Self {
            seq: frame.seq,
            received_at: frame.received_at,
            source_time_100ns: frame.source_time_100ns,
            w,
            h,
            data,
        }
    }

    fn timing(&self, fallback_start: Instant) -> FrameTiming {
        FrameTiming {
            received_at: self.received_at,
            source_time_100ns: self.source_time_100ns,
            fallback_start,
        }
    }
}

/// Pipelined ONNX frame interpolation (RIFE): the ~30-40ms DML inference for
/// pair (N-1, N) runs on this worker while the engine presents the PREVIOUS
/// pair — so it hides inside the source period instead of blocking it
/// (1080/24p→48fps needs everything inside 41.7ms; the sync path measured
/// ~45ms and wobbled at 44fps). Presentation lags one source pair (~80ms at
/// 24p) — the user explicitly allows latency for interpolation filters.
struct InterpWorker {
    job_tx: Option<Sender<InterpJob>>,
    res_rx: Receiver<InterpResult>,
    /// identity of the OnnxStage this worker drives (rebuild on chain swap)
    stage_ptr: usize,
    thread: Option<std::thread::JoinHandle<()>>,
    done_rx: Receiver<()>,
}

struct InterpWorkerDone(Sender<()>);
impl Drop for InterpWorkerDone {
    fn drop(&mut self) {
        let _ = self.0.send(());
    }
}

struct InterpJob {
    w: i32,
    h: i32,
    frames: Vec<Arc<Vec<u8>>>,
    ts: Vec<f32>,
    rgba: bool,
}

struct InterpResult {
    idx: usize,
    r: Result<(i32, i32, Vec<u8>)>,
    /// (pack_ms, run_ms, out_ms)
    profile: Option<(f64, f64, f64)>,
}

impl InterpWorker {
    fn spawn(stage: Arc<Mutex<crate::render::onnx_stage::OnnxStage>>, stage_ptr: usize) -> Self {
        stage.lock().unwrap().reset_interp_pack_cache();
        let (job_tx, job_rx) = channel::<InterpJob>();
        let (res_tx, res_rx) = channel::<InterpResult>();
        let (done_tx, done_rx) = channel::<()>();
        let thread = std::thread::Builder::new()
            .name("interp-worker".into())
            .spawn(move || {
                let _done = InterpWorkerDone(done_tx);
                while let Ok(job) = job_rx.recv() {
                    let frames: Vec<&[u8]> =
                        job.frames.iter().map(|frame| frame.as_slice()).collect();
                    if job.rgba {
                        let mut st = stage.lock().unwrap();
                        let results = st.process_interp_many_rgba8(job.w, job.h, &frames, &job.ts);
                        let profile = st
                            .last_interp_profile()
                            .map(|p| (p.pack_ms, p.run_ms, p.out_ms));
                        drop(st);
                        match results {
                            Ok(results) => {
                                for (idx, r) in results.into_iter().map(Ok).enumerate() {
                                    if res_tx.send(InterpResult { idx, r, profile }).is_err() {
                                        return;
                                    }
                                }
                            }
                            Err(error) => {
                                let message = format!("{error:#}");
                                for idx in 0..job.ts.len() {
                                    let r = Err(anyhow::anyhow!(message.clone()));
                                    if res_tx.send(InterpResult { idx, r, profile }).is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                    } else {
                        for (idx, t) in job.ts.iter().enumerate() {
                            let mut st = stage.lock().unwrap();
                            let r = st.process_interp(job.w, job.h, &frames, *t);
                            let profile = st
                                .last_interp_profile()
                                .map(|p| (p.pack_ms, p.run_ms, p.out_ms));
                            drop(st);
                            if res_tx.send(InterpResult { idx, r, profile }).is_err() {
                                return;
                            }
                        }
                    }
                }
            })
            .expect("spawn interp worker");
        Self {
            job_tx: Some(job_tx),
            res_rx,
            stage_ptr,
            thread: Some(thread),
            done_rx,
        }
    }
}

impl Drop for InterpWorker {
    fn drop(&mut self) {
        // Closing the channel and joining prevents an old interpolation model
        // from consuming GPU time after a RIFE -> DRBA or stop/start switch.
        self.job_tx.take();
        if let Some(thread) = self.thread.take() {
            let started = Instant::now();
            if self
                .done_rx
                .recv_timeout(GPU_INTERP_WORKER_SHUTDOWN_TIMEOUT)
                .is_ok()
            {
                let _ = thread.join();
                log::info!(
                    "interp-worker-stopped: join_ms={:.2}",
                    started.elapsed().as_secs_f64() * 1000.0
                );
            } else {
                log::error!(
                    "interp-worker-stop-timeout: elapsed_ms={} action=detach-and-quarantine",
                    GPU_INTERP_WORKER_SHUTDOWN_TIMEOUT.as_millis()
                );
                drop(thread);
            }
        }
    }
}

struct GpuInterpWorker {
    job_tx: Option<SyncSender<GpuInterpJob>>,
    permit_tx: Option<Sender<GpuInterpPermit>>,
    result_rx: Receiver<GpuInterpResult>,
    done_rx: Receiver<()>,
    thread: Option<std::thread::JoinHandle<()>>,
    /// Identity of the exact ONNX stage owned by this worker. A backend or
    /// interpolation-model switch must never reuse a worker from the old stage.
    stage_ptr: usize,
}

struct GpuInterpJob {
    generation: u64,
    pair_id: u64,
    count: usize,
    cooperative_slots: bool,
}

#[derive(Clone, Copy)]
struct GpuInterpPermit {
    generation: u64,
    pair_id: u64,
}

struct GpuInterpResult {
    generation: u64,
    pair_id: u64,
    index: usize,
    run_ms: f64,
    result: std::result::Result<(), String>,
}

struct PendingGpuInterp {
    generation: u64,
    pair_id: u64,
    /// User-selected interpolation factor.  Keep this explicit so narrow
    /// factor-specific safety workarounds never leak into x2/x4/x5.
    factor: u32,
    cooperative_slots: bool,
    /// x4/x5 release the next unique inference after the current output's
    /// post-GLSL commands are submitted but before SwapBuffers. x3 leaves this
    /// false and releases only after present to avoid DirectML/GL contention.
    permit_next_after_post_submit: bool,
    stage: Arc<Mutex<crate::render::onnx_stage::OnnxStage>>,
    /// Exact metric identity from the executable FilterChain, including the
    /// original file extension, duplicate suffix and active provider.  Do not
    /// rebuild this from OnnxStage::name: interpolation stages store a model
    /// stem there (for example `rife_v4.22_lite_fp16`), while the chain row is
    /// `rife_v4.22_lite_fp16.onnx [TensorRT]`.  Mixing those identities leaves
    /// RIFE unmatched and appends it after the image filters after a live drag.
    metric_name: String,
    real_tex: GpuTex,
    history_keep: Vec<GpuTex>,
    timesteps: Vec<f32>,
    output_slots: Vec<PreparedInterpGpuOutput>,
    ready_outputs: Vec<Option<GpuTex>>,
    next_output_index: usize,
    completed_outputs: usize,
    run_ms_total: f64,
    present_real: bool,
    post_chain_start: usize,
    frame: FrameBuf,
    start_source_time_100ns: Option<i64>,
    output_period: f64,
    out_size: (i32, i32),
    /// Set when the provider job is actually submitted. Keep the current
    /// filtered frame on screen until this exact provider result is ready;
    /// never substitute an unfiltered/live frame while editing a preset.
    submitted_at: Instant,
    /// One-shot diagnostic proving whether the render thread is still cycling
    /// while a provider invocation is unusually slow.
    slow_wait_logged: bool,
}

struct PendingGpuPack {
    fence: u64,
    started: Instant,
    job: GpuInterpJob,
    pending: PendingGpuInterp,
}

// TensorRT can spend several seconds inside a cold provider call. Keep the
// worker teardown timeout generous, but never hide that delay by substituting
// a different (unfiltered) live frame.
const GPU_INTERP_WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(15);

impl GpuInterpWorker {
    fn spawn(stage: Arc<Mutex<crate::render::onnx_stage::OnnxStage>>) -> Self {
        let stage_ptr = Arc::as_ptr(&stage) as usize;
        let (job_tx, job_rx) = sync_channel::<GpuInterpJob>(2);
        // x3/x4/x5 cooperatively schedule one provider invocation per output.
        let (permit_tx, permit_rx) = channel::<GpuInterpPermit>();
        // Bound one source pair of genuinely different model outputs. x2 has
        // one midpoint; x3/x4/x5 use cooperative permits when more than one
        // unique midpoint is required. The queue never duplicates or
        // re-presents an interpolation result.
        let (result_tx, result_rx) = sync_channel::<GpuInterpResult>(5);
        let (done_tx, done_rx) = channel();
        let thread = std::thread::Builder::new()
            .name("interp-gpu-worker".into())
            .spawn(move || {
                let _done = InterpWorkerDone(done_tx);
                while let Ok(job) = job_rx.recv() {
                    let pair_started = Instant::now();
                    for index in 0..job.count {
                        let lock_started = Instant::now();
                        let result = stage
                            .lock()
                            .map_err(|_| "stage mutex poisoned".to_string())
                            .and_then(|mut stage| {
                                let lock_wait_ms =
                                    lock_started.elapsed().as_secs_f64() * 1000.0;
                                let provider_started = Instant::now();
                                let result = stage
                                    .run_prepared_interp_gpu_slot(index)
                                    .map(|(run_ms, _)| run_ms)
                                    .map_err(|error| format!("{error:#}"));
                                let provider_wall_ms =
                                    provider_started.elapsed().as_secs_f64() * 1000.0;
                                if lock_wait_ms >= 25.0 || provider_wall_ms >= 50.0 {
                                    log::warn!(
                                        "interp-gpu-worker-slow: pair={} generation={} slot={}/{} stage_lock_wait_ms={:.2} provider_wall_ms={:.2}",
                                        job.pair_id,
                                        job.generation,
                                        index + 1,
                                        job.count,
                                        lock_wait_ms,
                                        provider_wall_ms
                                    );
                                }
                                result
                            });
                        let (run_ms, status) = match result {
                            Ok(run_ms) => (run_ms, Ok(())),
                            Err(error) => (0.0, Err(error)),
                        };
                        if job.pair_id < 3 || status.is_err() {
                            log::info!(
                                "interp-gpu-worker-slot: pair={} generation={} slot={}/{} slot_ms={:.2} pair_elapsed_ms={:.2} ok={}",
                                job.pair_id,
                                job.generation,
                                index + 1,
                                job.count,
                                run_ms,
                                pair_started.elapsed().as_secs_f64() * 1000.0,
                                status.is_ok()
                            );
                        }
                        let failed = status.is_err();
                        if result_tx
                            .send(GpuInterpResult {
                                generation: job.generation,
                                pair_id: job.pair_id,
                                index,
                                run_ms,
                                result: status,
                            })
                            .is_err()
                        {
                            return;
                        }
                        if failed {
                            break;
                        }
                        if job.cooperative_slots && index + 1 < job.count {
                            // The render thread releases the next integer-factor
                            // slot after this frame's GL work has been submitted
                            // but before SwapBuffers. x3 follows the same simple
                            // cooperative rule as x4/x5; its direct 1:1 detach is
                            // completed explicitly before this point.
                            loop {
                                match permit_rx.recv() {
                                    Ok(permit)
                                        if permit.generation == job.generation
                                            && permit.pair_id == job.pair_id =>
                                    {
                                        break;
                                    }
                                    Ok(_) => continue,
                                    Err(_) => return,
                                }
                            }
                        }
                    }
                }
            })
            .expect("spawn GPU interpolation worker");
        log::info!(
            "interp-gpu-worker-ready: thread=interp-gpu-worker result_queue=one-pair capacity=5"
        );
        Self {
            job_tx: Some(job_tx),
            permit_tx: Some(permit_tx),
            result_rx,
            done_rx,
            thread: Some(thread),
            stage_ptr,
        }
    }

    fn matches_stage(&self, stage: &Arc<Mutex<crate::render::onnx_stage::OnnxStage>>) -> bool {
        self.stage_ptr == Arc::as_ptr(stage) as usize
    }

    fn submit(&self, job: GpuInterpJob) -> std::result::Result<(), TrySendError<GpuInterpJob>> {
        self.job_tx
            .as_ref()
            .expect("GPU worker sender")
            .try_send(job)
    }

    fn permit_next_slot(&self, generation: u64, pair_id: u64) {
        if let Some(tx) = self.permit_tx.as_ref() {
            let _ = tx.send(GpuInterpPermit {
                generation,
                pair_id,
            });
        }
    }
}

impl GpuInterpWorker {
    fn shutdown(&mut self) -> bool {
        self.job_tx.take();
        // A cooperative x3/x4/x5 worker may be sleeping between unique slots.
        // Closing this channel wakes it immediately; Stop never waits for a
        // presentation deadline merely to release the provider thread.
        self.permit_tx.take();
        if let Some(thread) = self.thread.take() {
            let shutdown_started = Instant::now();
            let deadline = Instant::now() + GPU_INTERP_WORKER_SHUTDOWN_TIMEOUT;
            loop {
                // A streamed worker may be waiting for the render thread to
                // accept its latest unique output slot. Drain it during
                // teardown so geometry/backend changes cannot deadlock the
                // rendezvous channel.
                // recv_timeout also accepts a worker blocked behind the bounded
                // result queue; a pure try_recv loop can miss that hand-off
                // repeatedly on some schedulers during rapid geometry changes.
                let _ = self.result_rx.recv_timeout(Duration::from_millis(1));
                while self.result_rx.try_recv().is_ok() {}
                if self.done_rx.try_recv().is_ok() {
                    let _ = thread.join();
                    log::info!(
                        "interp-gpu-worker-stopped: join_ms={:.2}",
                        shutdown_started.elapsed().as_secs_f64() * 1000.0
                    );
                    return true;
                }
                if Instant::now() >= deadline {
                    log::error!(
                        "interp-gpu-worker-stop-timeout: elapsed_ms={} action=detach-and-quarantine",
                        GPU_INTERP_WORKER_SHUTDOWN_TIMEOUT.as_millis()
                    );
                    drop(thread);
                    return false;
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        } else {
            true
        }
    }
}

impl Drop for GpuInterpWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// The source pair whose in-betweens are cooking on the worker: its END frame
/// is presented AFTER the mids arrive (next tick), giving the 1-pair delay.
struct PendingInterp {
    /// Exact ONNX model name plus the provider selected by the live stage.
    interp_name: String,
    real: HistFrame,
    count: usize,
    rgba: bool,
    /// Whether the display-grid cadence includes the real endpoint for this
    /// source interval. 24p -> 60Hz x3 alternates 3 and 2 outputs; the
    /// two-output interval deliberately omits the endpoint.
    present_real: bool,
    mid_ts: Vec<f32>,
    output_period_s: Option<f64>,
    start_source_time_100ns: Option<i64>,
    /// CONTENT interval of the pair (source timestamps), for presentation
    /// pacing. Using arrival-EWMA here caused a death spiral: our own pacing
    /// sleep inflated arrivals, which inflated the sleep (66ms/tick field
    /// log). Source timestamps are immune to our loop behaviour.
    pair_dt: Option<f64>,
    /// First GUI-chain stage after the interpolation model. Stages before the
    /// model were already executed once on each real endpoint.
    chain_start_index: usize,
}

fn interp_pair_is_contiguous(
    pair_dt: Option<f64>,
    cadence_period: Option<f64>,
    arrival_interval: f64,
) -> bool {
    let Some(pair_dt) = pair_dt.filter(|dt| dt.is_finite() && *dt >= 0.0) else {
        return true;
    };
    let baseline = cadence_period
        .filter(|dt| dt.is_finite() && *dt > 0.0 && *dt <= 0.2)
        .or_else(|| {
            (arrival_interval.is_finite() && arrival_interval > 0.0 && arrival_interval <= 0.2)
                .then_some(arrival_interval)
        });
    let limit = baseline
        .map(|period| (period * 4.0).clamp(0.1, 0.2))
        .unwrap_or(0.2);
    pair_dt <= limit
}

#[derive(Clone, Copy)]
struct FrameTiming {
    received_at: Option<Instant>,
    source_time_100ns: Option<i64>,
    fallback_start: Instant,
}

impl FrameTiming {
    fn from_frame(frame: &FrameBuf, fallback_start: Instant) -> Self {
        Self {
            received_at: frame.received_at,
            source_time_100ns: frame.source_time_100ns,
            fallback_start,
        }
    }

    fn from_interpolated(
        frame: &FrameBuf,
        prev_source_time_100ns: Option<i64>,
        t: f32,
        fallback_start: Instant,
    ) -> Self {
        let source_time_100ns = match (prev_source_time_100ns, frame.source_time_100ns) {
            (Some(a), Some(b)) if b >= a => {
                Some(a + ((b - a) as f64 * t.clamp(0.0, 1.0) as f64).round() as i64)
            }
            _ => frame.source_time_100ns,
        };
        Self {
            received_at: frame.received_at,
            source_time_100ns,
            fallback_start,
        }
    }
}

#[derive(Clone, Copy)]
struct PresentTime {
    instant: Instant,
    qpc_time_100ns: Option<i64>,
}

impl PresentTime {
    fn now() -> Self {
        Self {
            instant: Instant::now(),
            qpc_time_100ns: crate::platform::win32::qpc_time_100ns(),
        }
    }
}

fn apply_chain_update(
    s: &mut Session,
    gc: &mut GlContext,
    factory: &mut StageFactory,
    specs: &[StageSpec],
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
) {
    let old_interp = s.chain.interp_key();
    let old_interp_pre = s
        .chain
        .interpolation_plan()
        .map(|(pre, _interp, _post)| pre);
    let (mut chain, errs) = FilterChain::from_specs(factory, specs);
    // A warm TensorRT Session may be reused from StageFactory. Per-capture
    // temporal/packing state must never cross the session boundary even though
    // the expensive ORT/TensorRT execution session remains resident.
    chain.reset_backend_runtime_state();
    let requested = specs.iter().filter(|spec| spec.enabled).count();
    if requested > 0 && chain.stages.is_empty() {
        log::error!(
            "chain-apply rejected: all {requested} enabled filters failed; keeping previous chain: {}",
            errs.join(" | ")
        );
        status.lock().unwrap().chain_errors = errs;
        return;
    }
    if s.interp_worker.is_some() {
        log::info!(
            "chain-apply: stopping previous interpolation worker after replacement validation"
        );
        s.interp_pending = None;
        s.interp_worker = None;
        s.interp_post_warm = false;
    }
    if let Some(pack) = s.gpu_interp_pack_pending.take() {
        gc.cancel_commands_fence(pack.fence);
    }
    if let Some(mut pending) = s.gpu_interp_pending.take() {
        recycle_pending_gpu_outputs(gc, &mut pending);
    }
    if let Some(mut worker) = s.gpu_interp_worker.take() {
        let _ = worker.shutdown();
    }
    while let Some(frame) = s.gpu_interp_hist.pop_front() {
        gc.recycle(frame.tex);
    }
    s.gpu_interp_result_stash.clear();
    s.gpu_interp_permit_after_post_submit = None;
    s.interp_generation = s.interp_generation.saturating_add(1);
    s.gpu_interp_active_logged = false;
    let new_interp = chain.interp_key();
    let new_interp_pre = chain
        .interpolation_plan()
        .map(|(pre, _interp, _post)| pre);
    let directml_interp_pre_route_changed = old_interp.is_some()
        && old_interp.as_ref() == new_interp.as_ref()
        && chain.interp_provider() == Some(crate::render::onnx_stage::OnnxProvider::DirectML)
        && old_interp_pre != new_interp_pre;
    if old_interp.is_some() || new_interp.is_some() {
        // Interpolators keep different history and pacing state. Carrying a
        // RIFE queue/deadline across a chain reorder is also invalid even when
        // the interpolation model itself is unchanged: its cached endpoint
        // may have passed through what is now a post stage (1280x960), while
        // the replacement plan expects the native 640x480 endpoint. That
        // startup-order leak was the reproducible 42-vs-48 fps split.
        s.hist.clear();
        s.interp_pending = None;
        s.interp_worker = None;
        if let Some(texture) = s.prev_tex.take() {
            gc.recycle(texture);
        }
        if let Some(texture) = s.prev2_tex.take() {
            gc.recycle(texture);
        }
        s.prev_tex_seq = 0;
        s.prev_tex_source_time_100ns = None;
        s.smooth_pacer.reset();
        s.paced_present_deadline = None;
        s.flow_output_cadence.reset();
        s.present_cadence = PresentCadence::default();
        s.source.set_queue_enabled(false);
        log::info!(
            "chain-apply: interpolation state reset for route replacement old={old_interp:?} new={new_interp:?}"
        );
    }
    let enabled = chain.stages.len();
    // A DirectML first-stage output can be an OpenGL view of shared D3D12
    // memory. Detaching that memory while the overlay still holds last_tex
    // made the retained image sample the replacement allocation with the old
    // dimensions (the brief giant crop seen when switching ONNX on static
    // PIP). Preserve it as an ordinary owned GL texture before teardown.
    let transition_snapshot = s.last_tex.map(|texture| {
        (
            texture,
            texture.w(),
            texture.h(),
            gc.download_rgba8(texture),
        )
    });
    s.chain.prepare_gpu_transition(gc);
    if directml_interp_pre_route_changed {
        // The candidate chain was intentionally validated before touching the
        // running chain, so its DirectML interpolator may still share the warm
        // StageFactory session with the old route. Retire the old bridge first,
        // then replace only that interpolation session. This mirrors the
        // Stop/Start recovery path without disturbing TensorRT or unrelated
        // ONNX stages.
        log::info!(
            "directml-interp-session-rebuild: reason=pre-chain-route-changed old_pre={old_interp_pre:?} new_pre={new_interp_pre:?}"
        );
        if let Err(error) = chain.rebuild_directml_interpolation_session(factory) {
            // The already-validated candidate remains usable if recreation
            // fails; its bridge was retired above and can be rebuilt normally.
            // Prefer a recoverable warning over rejecting the whole live edit.
            log::warn!(
                "directml-interp-session-rebuild-failed: {error:#}; action=continue-with-validated-session"
            );
        }
    }
    gc.clear_gpu_timers();
    gc.clear_temporal_shader_storage();
    if let Some((old, width, height, rgba)) = transition_snapshot {
        gc.recycle(old);
        s.last_tex = Some(gc.upload_rgba8(width, height, &rgba));
    }
    let previous_glsl_key = glsl_chain_key(&s.chain);
    let next_glsl_key = glsl_chain_key(&chain);
    let filter_chain_changed = previous_glsl_key != next_glsl_key;
    let next_glsl_overload_hint = !next_glsl_key.is_empty()
        && s.glsl_overload_history
            .iter()
            .any(|known| known == &next_glsl_key);
    s.chain = chain;
    // A newly-selected chain gets a clean live admission state. A bounded
    // same-session history survives only as a one-frame re-check hint; it never
    // directly throttles a replacement chain.
    s.glsl_guard.reset();
    s.glsl_overload_hint = next_glsl_overload_hint;
    s.glsl_chain_settle_until = None;
    s.glsl_chain_immediate_until = None;
    s.glsl_resolution_recheck_active = false;
    s.glsl_resolution_recheck_started = None;
    s.glsl_resolution_recheck_healthy_since = None;
    {
        let mut state = status.lock().unwrap();
        state.glsl_interactive_pause = false;
        if filter_chain_changed {
            if state.glsl_overload_notice_latched {
                state.glsl_overload_notice_latched = false;
                log::info!("glsl-overload-notice: latched=false reason=filter-chain-changed");
            }
            if state.glsl_overload_auto_stop_deadline.take().is_some() {
                log::info!(
                    "glsl-overload-auto-stop: cancelled reason=filter-chain-changed"
                );
            }
        }
    }
    if next_glsl_overload_hint {
        log::info!(
            "glsl-responsiveness-history: action=arm-fast-recheck stages={:?}",
            s.chain
                .stages
                .iter()
                .map(|stage| stage.name())
                .collect::<Vec<_>>()
        );
    }
    // Chain edits can leave large shader working sets in the free-texture pool.
    // Keep the same bounded warm cache used by geometry transitions instead of
    // waiting for a later source resize to reclaim gigabytes of stale textures.
    // This touches only recycled/free textures and never runs on the frame path.
    gc.trim_transient_pool(2);
    if glsl_guard_eligible(s) {
        let guard_arm = Instant::now();
        s.glsl_chain_settle_until = Some(guard_arm + GLSL_GUARD_CHAIN_SETTLE);
        s.glsl_chain_immediate_until =
            Some(guard_arm + GLSL_GUARD_CHAIN_SETTLE + GLSL_GUARD_CHAIN_CONFIRM);
        log::info!(
            "glsl-responsiveness-settle: phase=armed duration_ms={} confirm_ms={} stages={:?}",
            GLSL_GUARD_CHAIN_SETTLE.as_millis(),
            GLSL_GUARD_CHAIN_CONFIRM.as_millis(),
            s.chain
                .stages
                .iter()
                .map(|stage| stage.name())
                .collect::<Vec<_>>()
        );
    }
    let usage = s.chain.onnx_backend_usage();
    s.chain_reprocess_pending = !s.frame.data.is_empty();
    // A filter-chain edit changes the meaning of the cached output even when
    // the source pixels are byte-identical. Never allow duplicate/cadence
    // reuse from the previous chain to win over the forced reprocess.
    s.duplicate_signature = None;
    s.smooth_content_signature = None;
    s.smooth_content_seq = 0;
    s.smooth_content_unique_since_duplicate = 4;
    s.smooth_content_last_candidate_seq = None;
    s.smooth_content_last_candidate_time = None;
    s.smooth_content_pattern_hits = 0;
    s.smooth_content_candidate_gaps.clear();
    s.smooth_content_24p_detected = false;
    s.source.set_queue_enabled(s.chain.has_interp());
    metrics.reset();
    metrics.set_stage_order(s.chain.metric_stage_order());
    {
        let mut state = status.lock().unwrap();
        state.chain_errors = errs;
        state.onnx_tensorrt_stages = usage.tensorrt;
        state.onnx_cuda_stages = usage.cuda;
        state.onnx_directml_fallbacks = usage.directml_fallback;
    }
    log::info!(
        "chain-apply: requested={} enabled={} interp={:?} stages={:?}",
        requested,
        enabled,
        new_interp,
        s.chain
            .stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
    );
    if let Some((pre, interp, post)) = s.chain.interpolation_plan() {
        log::info!(
            "Interpolation chain plan: pre={pre:?} interp={interp:?} post={post:?} effective_order=pre->interp->post verified=true"
        );
    }
}

fn reset_gpu_interp_for_backend_switch(
    s: &mut Session,
    gc: &mut GlContext,
) -> std::result::Result<(), String> {
    // A backend switch is a hard ownership boundary. No job created by the old
    // ONNX stage may survive until its bridge is retired, otherwise the worker
    // can wake after commit and execute against a bridge that has already been
    // detached while a TensorRT interpolation bridge was still required.
    if s.gpu_interp_pack_pending.is_some() || s.gpu_interp_pending.is_some() {
        return Err("a GPU interpolation pair is still in flight".to_string());
    }
    if let Some(mut worker) = s.gpu_interp_worker.take() {
        if !worker.shutdown() {
            return Err(
                "the previous GPU interpolation worker did not stop within 15 seconds; backend switch was cancelled"
                    .to_string(),
            );
        }
    }
    while let Some(frame) = s.gpu_interp_hist.pop_front() {
        gc.recycle(frame.tex);
    }
    s.gpu_interp_result_stash.clear();
    s.gpu_interp_permit_after_post_submit = None;
    s.interp_generation = s.interp_generation.saturating_add(1);
    s.gpu_interp_pair_id = 0;
    s.gpu_interp_active_logged = false;
    log::info!(
        "interp-gpu-backend-transition-reset: generation={} worker=stopped pending=0 history=0",
        s.interp_generation
    );
    Ok(())
}

fn reset_gpu_interp_for_geometry_transition(s: &mut Session, gc: &mut GlContext, reason: &str) {
    if let Some(pack) = s.gpu_interp_pack_pending.take() {
        gc.cancel_commands_fence(pack.fence);
    }
    if let Some(mut worker) = s.gpu_interp_worker.take() {
        let _ = worker.shutdown();
    }
    if let Some(mut pending) = s.gpu_interp_pending.take() {
        recycle_pending_gpu_outputs(gc, &mut pending);
    }
    while let Some(frame) = s.gpu_interp_hist.pop_front() {
        gc.recycle(frame.tex);
    }
    s.gpu_interp_result_stash.clear();
    s.gpu_interp_permit_after_post_submit = None;
    s.interp_generation = s.interp_generation.saturating_add(1);
    s.gpu_interp_pair_id = 0;
    s.gpu_interp_active_logged = false;
    s.flow_output_cadence.reset();
    s.interp_present_deadline = None;
    s.chain.prepare_gpu_transition(gc);
    log::info!(
        "interp-gpu-geometry-transition-reset: reason={} generation={} worker=stopped pending=0 history=0",
        reason,
        s.interp_generation
    );
}

fn switch_onnx_backend(
    session: &mut Option<Session>,
    gc: &mut GlContext,
    factory: &mut StageFactory,
    backend: OnnxBackendPreference,
    trt_device_id: Option<i32>,
    cache_root: std::path::PathBuf,
    specs: &[StageSpec],
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
) -> bool {
    let previous = factory.onnx_preference();
    log::info!("onnx-backend-switch-begin: {previous:?} -> {backend:?}");
    {
        let mut state = status.lock().unwrap();
        state.onnx_backend_switching = true;
        state.onnx_backend_error = None;
    }

    let mut candidate_factory =
        factory.fork_for_backend(backend, trt_device_id, cache_root.clone());
    let Some(s) = session.as_mut() else {
        *factory = candidate_factory;
        let mut state = status.lock().unwrap();
        state.onnx_backend = backend;
        state.onnx_backend_switching = false;
        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
        state.onnx_tensorrt_stages = 0;
        state.onnx_cuda_stages = 0;
        state.onnx_directml_fallbacks = 0;
        log::info!("onnx-backend-switch-commit: active={backend:?} idle=true fallback_count=0");
        crate::render::onnx_stage::finish_tensorrt_build_progress();
        return true;
    };

    let (mut candidate_chain, errors) = FilterChain::from_specs(&mut candidate_factory, specs);
    let requested = specs.iter().filter(|spec| spec.enabled).count();
    log::info!(
        "onnx-backend-switch-candidate: requested={} enabled={} errors={} stages={:?}",
        requested,
        candidate_chain.stages.len(),
        errors.len(),
        candidate_chain
            .stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
    );
    if (requested > 0 && candidate_chain.stages.is_empty()) || !errors.is_empty() {
        let reason = if errors.is_empty() {
            "all enabled filters failed while preparing the candidate chain".to_string()
        } else {
            errors.join(" | ")
        };
        log::error!(
            "onnx-backend-switch-rollback: active={previous:?} requested={backend:?} reason={reason}"
        );
        let mut state = status.lock().unwrap();
        state.onnx_backend_switching = false;
        state.onnx_backend_error = Some(reason);
        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
        return false;
    }

    let preserve = s.last_tex.into_iter().collect::<Vec<_>>();
    let warmup_result = if s.frame.data.is_empty() || s.frame.w <= 0 || s.frame.h <= 0 {
        log::info!("onnx-backend-warmup: skipped=no-current-frame backend={backend:?}");
        Ok(candidate_chain.onnx_backend_usage())
    } else {
        let rgba = if s.frame.hdr {
            let tone_mapped = upload_frame_timed(gc, &s.frame, &mut s.last_upload_submit_ms);
            let rgba = gc.download_rgba8(tone_mapped);
            gc.release_frame(&preserve);
            rgba
        } else {
            s.frame.data.clone()
        };
        let out_size = s
            .last_tex
            .map(|texture| (texture.w(), texture.h()))
            .filter(|(w, h)| *w > 0 && *h > 0)
            .unwrap_or((s.frame.w, s.frame.h));
        let started = Instant::now();
        let result =
            candidate_chain.warmup_candidate(gc, s.frame.w, s.frame.h, &rgba, out_size, &preserve);
        match &result {
            Ok(usage) => log::info!(
                "onnx-backend-warmup: backend={backend:?} size={}x{} result=ok elapsed_ms={:.2} tensorrt={} cuda={} directml={} directml_fallback={}",
                s.frame.w,
                s.frame.h,
                started.elapsed().as_secs_f64() * 1000.0,
                usage.tensorrt,
                usage.cuda,
                usage.directml,
                usage.directml_fallback
            ),
            Err(error) => log::error!(
                "onnx-backend-warmup: backend={backend:?} size={}x{} result=failed elapsed_ms={:.2} error={error:#}",
                s.frame.w,
                s.frame.h,
                started.elapsed().as_secs_f64() * 1000.0
            ),
        }
        result
    };

    let usage = match warmup_result {
        Ok(usage) => usage,
        Err(error) => {
            candidate_chain.prepare_gpu_transition(gc);
            let reason = format!("{error:#}");
            let cancelled = crate::render::onnx_stage::onnx_cancel_requested()
                || crate::render::onnx_stage::tensorrt_cancel_requested();
            let mut state = status.lock().unwrap();
            state.onnx_backend_switching = false;
            state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
            if cancelled {
                state.onnx_backend_error = None;
                log::info!(
                    "onnx-backend-switch-cancelled-by-stop: active={previous:?} requested={backend:?} reason={reason}"
                );
            } else {
                state.onnx_backend_error = Some(reason.clone());
                log::error!(
                    "onnx-backend-switch-rollback: active={previous:?} requested={backend:?} reason={reason}"
                );
            }
            return false;
        }
    };

    // Stop and drain the old provider before retiring any of its shared GPU
    // bridges. Backend switches are deferred while a pair is in flight, so the
    // steady-state worker should join immediately here.
    if let Err(reason) = reset_gpu_interp_for_backend_switch(s, gc) {
        candidate_chain.prepare_gpu_transition(gc);
        log::error!(
            "onnx-backend-switch-rollback: active={previous:?} requested={backend:?} reason={reason}"
        );
        let mut state = status.lock().unwrap();
        state.onnx_backend_switching = false;
        state.onnx_backend_error = Some(reason);
        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
        return false;
    }

    // Warmup must not become temporal history for the first live frame.
    candidate_chain.prepare_gpu_transition(gc);
    candidate_chain.reset_backend_runtime_state();

    let transition_snapshot = s.last_tex.map(|texture| {
        (
            texture,
            texture.w(),
            texture.h(),
            gc.download_rgba8(texture),
        )
    });
    s.interp_pending = None;
    s.interp_worker = None;
    s.interp_post_warm = false;
    s.hist.clear();
    if let Some(texture) = s.prev_tex.take() {
        gc.recycle(texture);
    }
    if let Some(texture) = s.prev2_tex.take() {
        gc.recycle(texture);
    }
    s.prev_tex_seq = 0;
    s.prev_tex_source_time_100ns = None;
    s.interp_last_tick = None;
    s.interp_tail = None;
    s.interp_present_deadline = None;
    s.smooth_pacer.reset();
    s.paced_present_deadline = None;
    s.flow_output_cadence.reset();
    s.present_cadence = PresentCadence::default();

    s.chain.prepare_gpu_transition(gc);
    gc.clear_gpu_timers();
    gc.clear_temporal_shader_storage();
    if let Some((old, width, height, rgba)) = transition_snapshot {
        gc.recycle(old);
        s.last_tex = Some(gc.upload_rgba8(width, height, &rgba));
    }
    s.chain = candidate_chain;
    *factory = candidate_factory;
    s.chain_reprocess_pending = !s.frame.data.is_empty();
    s.smooth_content_signature = None;
    s.smooth_content_duplicates = 0;
    s.smooth_content_seq = 0;
    s.smooth_content_unique_since_duplicate = 4;
    s.smooth_content_last_candidate_seq = None;
    s.smooth_content_last_candidate_time = None;
    s.smooth_content_pattern_hits = 0;
    s.smooth_content_candidate_gaps.clear();
    s.smooth_content_24p_detected = false;
    s.source.set_queue_enabled(s.chain.has_interp());
    metrics.reset();
    metrics.set_stage_order(s.chain.metric_stage_order());

    {
        let mut state = status.lock().unwrap();
        state.chain_errors.clear();
        state.onnx_backend = backend;
        state.onnx_backend_switching = false;
        state.onnx_backend_error = None;
        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
        state.onnx_tensorrt_stages = usage.tensorrt;
        state.onnx_cuda_stages = usage.cuda;
        state.onnx_directml_fallbacks = usage.directml_fallback;
    }
    log::info!(
        "onnx-backend-switch-commit: active={backend:?} tensorrt={} cuda={} directml={} fallback_count={}",
        usage.tensorrt,
        usage.cuda,
        usage.directml,
        usage.directml_fallback
    );
    // TensorRT builds lazily on first inference. Chain commit alone is not a
    // completion signal; each stage closes its own progress after inference.
    true
}

fn recycle_pending_gpu_outputs(gc: &mut GlContext, pending: &mut PendingGpuInterp) {
    for output in &mut pending.ready_outputs {
        if let Some(texture) = output.take() {
            gc.recycle(texture);
        }
    }
}

/// Wait only inside the time that would otherwise be spent waiting for the
/// next presentation slot. This prevents a completed provider output from
/// sitting behind another full render-engine tick (geometry, input and WGC
/// polling), while messages remain responsive in sub-millisecond slices.
fn wait_for_gpu_interp_result(
    overlay: &mut OverlayWindow,
    worker: &GpuInterpWorker,
    max_wait: Duration,
) -> Option<GpuInterpResult> {
    if max_wait.is_zero() {
        return None;
    }
    let deadline = Instant::now() + max_wait;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return None;
        }
        let slice = deadline
            .saturating_duration_since(now)
            .min(Duration::from_micros(500));
        match worker.result_rx.recv_timeout(slice) {
            Ok(result) => return Some(result),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                overlay.win.pump_messages();
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return None,
        }
    }
}

/// Lead only by the work that must complete before presenting the texture.
/// `last_process_ms` also includes a potentially blocking SwapBuffers and used
/// to feed 8-12 ms of compositor wait back into an 8.33 ms x5 slot. That can
/// force every output onto the next refresh even though RIFE/DRBA inference
/// itself has already completed.
fn gpu_interp_present_lead_s(
    last_compute_ms: f64,
    last_present_block_ms: f64,
    output_period_s: f64,
) -> f64 {
    if !output_period_s.is_finite() || output_period_s <= 0.0 {
        return 0.0;
    }
    let compute_s = (last_compute_ms / 1000.0).max(0.0);
    // GLSL is submitted asynchronously, so its real GPU duration is commonly
    // paid inside SwapBuffers rather than `last_compute_ms`. Subtract the
    // normal lightweight compositor floor and start the next output early by
    // the remaining GPU tail. Without this, adding even one post-RIFE shader
    // waits until the nominal slot to submit work and then misses that slot by
    // another 5-10 ms on AMD.
    let gpu_tail_s = ((last_present_block_ms - 1.25).max(0.0) / 1000.0).min(output_period_s * 0.75);
    // A small submit margin gives DWM time to accept the already-filtered back
    // buffer before the target vblank. Keep it bounded so we never turn x5
    // into busy waiting for most of the 8.33 ms interval.
    let submit_margin_s = if last_present_block_ms >= 0.5 {
        0.001_25
    } else {
        0.000_50
    };
    (compute_s + gpu_tail_s + submit_margin_s).clamp(0.0, output_period_s * 0.90)
}

fn provider_uses_gpu_x3_mid_detach(
    provider: crate::render::onnx_stage::OnnxProvider,
) -> bool {
    matches!(
        provider,
        crate::render::onnx_stage::OnnxProvider::DirectML
            | crate::render::onnx_stage::OnnxProvider::TensorRT
    )
}

fn should_detach_gpu_x3_midpoint(
    overlay: &OverlayWindow,
    s: &Session,
    pending: &PendingGpuInterp,
    mid: GpuTex,
) -> bool {
    // Keep this workaround narrower than the observed failure: x3, one
    // interpolation ONNX stage only.  Any additional GLSL/ONNX stage already
    // establishes a separate output texture/lifetime and must stay untouched.
    if pending.factor != 3
        || s.chain.stage_count() != 1
        || pending.post_chain_start != s.chain.stage_count()
    {
        return false;
    }
    if !provider_uses_gpu_x3_mid_detach(pending.stage.lock().unwrap().provider) {
        return false;
    }

    // Mirror process_and_present_from's final viewport calculation.  The bug
    // condition is specifically the no-post, no-geometric-resample route: a
    // generated GPU-resident interpolation midpoint already has exactly the
    // final content size.
    let desired_overlay = overlay_geometry(s, overlay);
    let (vw, vh) = (desired_overlay.2, desired_overlay.3);
    let display_aspect = if s.display_aspect.0 > 0 && s.display_aspect.1 > 0 {
        s.display_aspect
    } else {
        (mid.w(), mid.h())
    };
    let (dw, dh) = fit_aspect_inside(display_aspect, (vw, vh));
    mid.w() == dw && mid.h() == dh
}

#[allow(clippy::too_many_arguments)]
fn drain_gpu_interp_stream(
    gc: &mut GlContext,
    overlay: &mut OverlayWindow,
    s: &mut Session,
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
    vsync_on: bool,
    downscaler: crate::render::scaler::Kernel,
) -> bool {
    let mut completed = s.gpu_interp_result_stash.drain(..).collect::<Vec<_>>();
    if completed.is_empty() {
        if let Some(worker) = s.gpu_interp_worker.as_ref() {
            // Import one output at a time. The deeper channel keeps provider
            // execution running, while leaving later completed slots queued
            // prevents four GL conversions from delaying the first midpoint.
            if let Ok(result) = worker.result_rx.try_recv() {
                completed.push(result);
            }
        }
    }
    let Some(mut pending) = s.gpu_interp_pending.take() else {
        for stale in completed {
            log::info!(
                "interp-gpu-result-stale: pair={} generation={} slot={}",
                stale.pair_id,
                stale.generation,
                stale.index
            );
        }
        return false;
    };

    let mut progressed = false;
    let mut failure: Option<String> = None;
    for done in completed {
        if done.generation != pending.generation || done.pair_id != pending.pair_id {
            log::info!(
                "interp-gpu-result-stale: pair={} generation={} slot={}",
                done.pair_id,
                done.generation,
                done.index
            );
            continue;
        }
        progressed = true;
        if let Err(error) = done.result {
            failure = Some(error);
            break;
        }
        pending.completed_outputs = pending.completed_outputs.saturating_add(1);
        pending.run_ms_total += done.run_ms;
        let Some(slot) = pending.output_slots.get(done.index).copied() else {
            failure = Some(format!(
                "interpolation output slot {} is missing",
                done.index
            ));
            break;
        };
        match crate::render::onnx_stage::OnnxStage::finish_prepared_interp_gpu_output(gc, slot) {
            Ok(texture) => {
                if let Some(existing) = pending.ready_outputs[done.index].replace(texture) {
                    gc.recycle(existing);
                }
            }
            Err(error) => {
                failure = Some(format!("{error:#}"));
                break;
            }
        }
    }

    if let Some(reason) = failure {
        let cancelled = crate::render::onnx_stage::onnx_cancel_requested();
        if cancelled {
            log::info!(
                "interp-gpu-worker-cancelled: pair={} generation={} reason={}",
                pending.pair_id,
                pending.generation,
                reason
            );
        } else {
            log::error!(
                "interp-gpu-worker-error: pair={} generation={} reason={}",
                pending.pair_id,
                pending.generation,
                reason
            );
        }
        if let Some(mut worker) = s.gpu_interp_worker.take() {
            let _ = worker.shutdown();
        }
        if !cancelled {
            pending
                .stage
                .lock()
                .unwrap()
                .disable_interp_gpu_path(&reason);
        }
        recycle_pending_gpu_outputs(gc, &mut pending);
        while let Some(old) = s.gpu_interp_hist.pop_front() {
            gc.recycle(old.tex);
        }
        return true;
    }

    if pending.next_output_index < pending.timesteps.len() {
        let index = pending.next_output_index;
        if let Some(mut mid) = pending.ready_outputs[index].take() {
            let lead_s = gpu_interp_present_lead_s(
                s.last_compute_ms,
                s.last_present_block_ms,
                pending.output_period,
            );
            s.flow_output_cadence.wait_for_present_with_lead(
                overlay,
                pending.output_period,
                vsync_on,
                s.smooth_pacing,
                lead_s,
            );

            if should_detach_gpu_x3_midpoint(overlay, s, &pending, mid) {
                match crate::render::scaler::detach_identity(gc, mid) {
                    Ok(detached) => {
                        if pending.pair_id < 3 || pending.pair_id % 120 == 0 {
                            log::info!(
                                "interp-gpu-x3-mid-detach: pair={} generation={} slot={}/{} size={}x{} path=gpu-identity-f16 reason=glsl-parity-direct-1to1-no-post",
                                pending.pair_id,
                                pending.generation,
                                index + 1,
                                pending.timesteps.len(),
                                mid.w(),
                                mid.h()
                            );
                        }
                        mid = detached;
                        // Keep the same completion boundary used by v522, but
                        // the detach target is now F16 to mirror a real GLSL
                        // render pass. Complete this tiny handoff before the next
                        // provider invocation so DML/TensorRT cannot contend with
                        // the copy. Final back-buffer draw/SwapBuffers remains
                        // overlapped.
                        gc.finish();
                    }
                    Err(error) => {
                        // Without a detached texture, keep this pair on the
                        // conservative post-present permit path so a failed
                        // safety copy can never reintroduce the old output
                        // lifetime race.
                        pending.permit_next_after_post_submit = false;
                        if pending.pair_id < 3 || pending.pair_id % 120 == 0 {
                            log::warn!(
                                "interp-gpu-x3-mid-detach-fallback: pair={} generation={} slot={}/{} reason={error:#}",
                                pending.pair_id,
                                pending.generation,
                                index + 1,
                                pending.timesteps.len()
                            );
                        }
                    }
                }
            }
            let now = Instant::now();
            let timing = FrameTiming::from_interpolated(
                &pending.frame,
                pending.start_source_time_100ns,
                pending.timesteps[index],
                now,
            );
            let mut keep = pending.history_keep.clone();
            keep.extend(pending.ready_outputs.iter().flatten().copied());
            if pending.permit_next_after_post_submit
                && pending.cooperative_slots
                && index + 1 < pending.timesteps.len()
            {
                // Do not let the next DirectML/TensorRT timestep jump ahead of
                // this frame's GL post-chain. process_and_present_from releases
                // the permit immediately after all GL work for this output has
                // been submitted, before any compositor wait/SwapBuffers.
                s.gpu_interp_permit_after_post_submit = Some((pending.generation, pending.pair_id));
            }
            let presented_before = status.lock().unwrap().presented;
            process_and_present_from(
                gc,
                overlay,
                s,
                metrics,
                status,
                mid,
                pending.out_size,
                metrics.detailed_enabled() && pending.frame.seq % 30 == 0,
                now,
                0,
                Some(timing),
                &keep,
                downscaler,
                pending.post_chain_start,
            );
            let last_present = s.last_present;
            let last_present_block_s = s.last_present_block_ms / 1000.0;
            let phase_corrected = s.smooth_pacing
                && !vsync_on
                && s.flow_output_cadence.observe_blocking_present(
                    s.last_present_started,
                    last_present,
                    pending.output_period,
                    last_present_block_s,
                );
            if phase_corrected && (pending.pair_id < 3 || pending.pair_id % 120 == 0) {
                log::info!(
                    "interp-present-phase-lock: pair={} slot={}/{} period_ms={:.3} compute_ms={:.2} present_block_ms={:.2}",
                    pending.pair_id,
                    index + 1,
                    pending.timesteps.len(),
                    pending.output_period * 1000.0,
                    s.last_compute_ms,
                    s.last_present_block_ms
                );
            }
            let presented_after = status.lock().unwrap().presented;
            if !s.gpu_interp_active_logged && presented_after > presented_before {
                let stage = pending.stage.lock().unwrap();
                log::info!(
                    "interp-gpu-path-active: backend={} kind={:?} worker=cooperative-unique first_presented_pair={}",
                    stage.provider_desc,
                    stage.interp,
                    pending.pair_id
                );
                s.gpu_interp_active_logged = true;
            }
            pending.next_output_index += 1;
            // Multi-slot jobs normally release from process_and_present_from
            // after post-chain submission, overlapping only compositor wait.
            if pending.cooperative_slots
                && !pending.permit_next_after_post_submit
                && pending.next_output_index < pending.timesteps.len()
            {
                if let Some(worker) = s.gpu_interp_worker.as_ref() {
                    worker.permit_next_slot(pending.generation, pending.pair_id);
                }
            }
            s.gpu_interp_pending = Some(pending);
            return true;
        }
    }

    let pair_complete = pending.completed_outputs >= pending.timesteps.len()
        && pending.next_output_index >= pending.timesteps.len();
    if !pair_complete {
        let waiting_for_provider = pending.completed_outputs < pending.timesteps.len()
            && pending
                .ready_outputs
                .get(pending.next_output_index)
                .is_none_or(|slot| slot.is_none());
        // Keep the provider wait inside one output slot, but do not cut the
        // common ~10.5-11 ms 720p RIFE x2/x3 result off at an arbitrary 10 ms
        // boundary. Missing that result by <1 ms forced another full engine
        // iteration and made startup phase depend on loop/driver timing. The
        // wait still pumps Win32 messages every 0.5 ms and remains capped well
        // below a 60 Hz frame.
        let wait_budget =
            Duration::from_secs_f64((pending.output_period * 0.95).clamp(0.0005, 0.012));
        s.gpu_interp_pending = Some(pending);

        // If the next unique slot is still cooking, wait only inside its
        // presentation budget while pumping Win32 messages.
        if waiting_for_provider {
            if let Some(worker) = s.gpu_interp_worker.as_ref() {
                if let Some(result) = wait_for_gpu_interp_result(overlay, worker, wait_budget) {
                    s.gpu_interp_result_stash.push_back(result);
                    return true;
                }
            }
            // One bounded wait per drain is enough; the outer loop will retry
            // without multiplying the wait budget through recursion.
            return progressed;
        }
        if progressed {
            return true;
        }
        return false;
    }

    if pending.pair_id < 3 || pending.pair_id % 120 == 0 {
        // Diagnostics only: keep the proven v349e timing path unchanged.
        // These values let us compare x4/x5 cadence without feeding any
        // historical/driver/vendor-specific estimate back into scheduling.
        let raw_lead_ms = gpu_interp_present_lead_s(
            s.last_compute_ms,
            s.last_present_block_ms,
            pending.output_period,
        ) * 1000.0;
        log::info!(
            "interp-gpu-stream-pair: pair={} generation={} unique_mids={} phases={:?} inference_ms={:.2} wall_ms={:.2} real_endpoint={} target_fps={:.2} raw_present_block_ms={:.2} raw_lead_ms={:.2}",
            pending.pair_id,
            pending.generation,
            pending.timesteps.len(),
            pending.timesteps,
            pending.run_ms_total,
            pending.submitted_at.elapsed().as_secs_f64() * 1000.0,
            pending.present_real,
            1.0 / pending.output_period.max(1e-6),
            s.last_present_block_ms,
            raw_lead_ms
        );
    }

    if metrics.detailed_enabled() {
        metrics.probe(&pending.metric_name, "onnx", pending.run_ms_total);
    }

    if pending.present_real {
        let lead_s = gpu_interp_present_lead_s(
            s.last_compute_ms,
            s.last_present_block_ms,
            pending.output_period,
        );
        s.flow_output_cadence.wait_for_present_with_lead(
            overlay,
            pending.output_period,
            vsync_on,
            s.smooth_pacing,
            lead_s,
        );
        let now = Instant::now();
        let delivered = s.source.delivered();
        let captures = delivered.saturating_sub(s.metric_seq) as u32;
        s.metric_seq = delivered;
        process_and_present_from(
            gc,
            overlay,
            s,
            metrics,
            status,
            pending.real_tex,
            pending.out_size,
            metrics.detailed_enabled() && pending.frame.seq % 30 == 0,
            now,
            captures,
            Some(FrameTiming::from_frame(&pending.frame, now)),
            &pending.history_keep,
            downscaler,
            pending.post_chain_start,
        );
        let last_present = s.last_present;
        let last_present_block_s = s.last_present_block_ms / 1000.0;
        let _ = s.smooth_pacing
            && !vsync_on
            && s.flow_output_cadence.observe_blocking_present(
                s.last_present_started,
                last_present,
                pending.output_period,
                last_present_block_s,
            );
    }

    if crate::logging::diagnostics_enabled() {
        let residency = crate::render::onnx_stage::interp_residency_snapshot();
        if residency.gpu_output_frames > 0
            && residency.gpu_output_frames % 300 < pending.timesteps.len() as u64
        {
            log::debug!(
                "interp-gpu-residency: input_frames={} output_frames={} cpu_readbacks={} cpu_uploads={} cpu_pack_frames={} cpu_output_conversions={} cpu_fallback_frames={} violations={} result={}",
                residency.gpu_input_frames,
                residency.gpu_output_frames,
                residency.cpu_frame_readbacks,
                residency.cpu_frame_uploads,
                residency.cpu_pack_frames,
                residency.cpu_output_conversions,
                residency.cpu_fallback_frames,
                residency.violations,
                if residency.cpu_frame_readbacks == 0
                    && residency.cpu_frame_uploads == 0
                    && residency.cpu_pack_frames == 0
                    && residency.cpu_output_conversions == 0
                {
                    "full"
                } else {
                    "violation"
                }
            );
        }
    }
    true
}

fn apply_interp_factor_now(
    s: Option<&mut Session>,
    gc: &mut GlContext,
    current: &mut u32,
    requested: u32,
) {
    let next = requested.clamp(2, 5);
    let previous = *current;
    *current = next;
    if let Some(s) = s {
        let factor_changed = previous != next;

        // A live x4 <-> x5 transition used to keep the same cooperative GPU
        // worker (and, when its capacity was already large enough, the same
        // DirectML interpolation bridge).  The old permit/result scheduling
        // state could then survive the multiplier change: 24p x5 had enough
        // raw GPU headroom for 120fps, yet the presentation clock recovered a
        // missed 8.333ms slot every few outputs and settled around 111-116fps
        // until Stop/Start created a fresh worker.  Multiplier changes are a
        // natural interpolation ownership boundary, so rebuild only that
        // sub-pipeline while leaving WGC, the filter chain, and source cadence
        // alive.  This mirrors the clean part of Stop/Start without flashing
        // or dropping GPU residency for the rest of the application.
        if factor_changed {
            if let Some(mut worker) = s.gpu_interp_worker.take() {
                let _ = worker.shutdown();
            }
            if let Some(crate::render::chain::InterpHandle::Onnx { stage, .. }) =
                s.chain.interp_stage()
            {
                gc.finish();
                let mut stage = stage.lock().unwrap_or_else(|poisoned| {
                    log::error!("onnx-stage-lock-poisoned: action=recover-for-factor-transition");
                    poisoned.into_inner()
                });
                stage.retire_interp_gpu_bridges(gc);
                stage.reset_interp_pack_cache();
            }
            s.gpu_interp_pair_id = 0;
            // Old x4 present/GL-tail timings are not valid preparation hints
            // for the shorter x5 slot (and vice versa).  Starting those hints
            // clean is what Stop/Start already does.
            s.last_compute_ms = 0.0;
            s.last_present_block_ms = 0.0;
            s.last_present_call_ms = 0.0;
            s.last_pacer_wait_ms = 0.0;
            s.smooth_pacer.reset();
            s.paced_present_deadline = None;
            log::info!(
                "interp-gpu-factor-transition-reset: previous=x{} next=x{} worker=stopped bridge=retired pair_id=0 timing=reset",
                previous,
                next
            );
        }

        s.hist.clear();
        while let Some(frame) = s.gpu_interp_hist.pop_front() {
            gc.recycle(frame.tex);
        }
        s.gpu_interp_result_stash.clear();
        s.gpu_interp_permit_after_post_submit = None;
        s.interp_generation = s.interp_generation.saturating_add(1);
        s.flow_output_cadence.reset();
        s.interp_present_deadline = None;
        s.present_cadence = PresentCadence::default();
        s.gpu_interp_active_logged = false;
    }
    log::info!(
        "frame-interpolation factor applied: requested=x{} effective=x{} previous=x{} history=reset",
        requested,
        next,
        previous
    );
}

fn drag_source_reference_rect(
    hwnd: isize,
    reference_kind: &'static str,
) -> Option<(i32, i32, i32, i32)> {
    // Keep one coordinate family for the whole gesture. Falling from DWM to
    // client coordinates mid-drag changes the non-client offset and appears as
    // a fake window move, which is enough to shift the visible grab point.
    match reference_kind {
        "dwm" => win32::extended_frame_bounds(hwnd),
        "outer" => win32::window_rect(hwnd),
        "client" | "client-fallback" => win32::client_rect_on_screen(hwnd),
        _ => None,
    }
}

fn drag_overlay_target(
    source_start: (i32, i32),
    source_now: (i32, i32),
    overlay_start: (i32, i32, i32, i32),
    monitor_rect: (i32, i32, i32, i32),
) -> (i32, i32) {
    let (mx, my, mw, mh) = monitor_rect;
    let dx = source_now.0.saturating_sub(source_start.0);
    let dy = source_now.1.saturating_sub(source_start.1);
    (
        clamp_axis_to_bounds(overlay_start.0.saturating_add(dx), overlay_start.2, mx, mw),
        clamp_axis_to_bounds(overlay_start.1.saturating_add(dy), overlay_start.3, my, mh),
    )
}

struct DragFollower {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl DragFollower {
    #[allow(clippy::too_many_arguments)]
    fn start(
        source_hwnd: isize,
        overlay_hwnd: isize,
        panel_hwnd: isize,
        source_start: (i32, i32, i32, i32),
        source_reference_kind: &'static str,
        overlay_start: (i32, i32, i32, i32),
        panel_start: Option<(i32, i32, i32, i32)>,
        monitor_rect: (i32, i32, i32, i32),
        drag_epoch: u64,
    ) -> Option<Self> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = std::thread::Builder::new()
            .name("neo-window-drag-follow".into())
            .spawn(move || {
                let mut last_source = (source_start.0, source_start.1);
                let mut last_source_size = (source_start.2, source_start.3);
                let mut last_overlay = (overlay_start.0, overlay_start.1);
                let mut samples: u64 = 0;
                let mut moves: u64 = 0;
                let mut geometry_changes: u64 = 0;
                let started = Instant::now();
                let mut last_diag = Instant::now() - Duration::from_secs(1);

                while !thread_stop.load(Ordering::Acquire)
                    && crate::input::source_caption_drag_active()
                    && crate::input::source_caption_drag_epoch() == drag_epoch
                {
                    if let Some((sx, sy, sw, sh)) =
                        drag_source_reference_rect(source_hwnd, source_reference_kind)
                    {
                        samples = samples.saturating_add(1);
                        if (sw, sh) != last_source_size {
                            geometry_changes = geometry_changes.saturating_add(1);
                            last_source_size = (sw, sh);
                        }
                        last_source = (sx, sy);

                        // Derive every overlay position from the drag's START
                        // geometry. Never integrate the previous delta. This is
                        // idempotent and, critically, does not accumulate a
                        // clamp history when the hidden source travels farther
                        // than the visible overlay can move at a monitor edge.
                        let (nx, ny) = drag_overlay_target(
                            (source_start.0, source_start.1),
                            (sx, sy),
                            overlay_start,
                            monitor_rect,
                        );
                        // Move + readback + anchored sprite publication is one
                        // generation-scoped transaction. BUTTON-UP takes the
                        // same guard, so this closure can never move the overlay
                        // after the drag contract has already ended.
                        let previous_overlay = last_overlay;
                        let mut observed_overlay = previous_overlay;
                        let committed = crate::input::commit_source_caption_drag_overlay_update(
                            drag_epoch,
                            || {
                                let actual_before = win32::window_rect(overlay_hwnd)
                                    .map(|r| (r.0, r.1))
                                    .unwrap_or(previous_overlay);
                                if (nx, ny) != actual_before {
                                    win32::set_window_rect(
                                        overlay_hwnd,
                                        nx,
                                        ny,
                                        overlay_start.2,
                                        overlay_start.3,
                                    );
                                }
                                let actual_overlay = win32::window_rect(overlay_hwnd)
                                    .map(|r| (r.0, r.1))
                                    .unwrap_or((nx, ny));
                                observed_overlay = actual_overlay;
                                if actual_overlay == previous_overlay {
                                    return None;
                                }

                                if panel_hwnd != 0 {
                                    if let Some((px, py, pw, ph)) = panel_start {
                                        let pnx = px.saturating_add(
                                            actual_overlay.0 - overlay_start.0,
                                        );
                                        let pny = py.saturating_add(
                                            actual_overlay.1 - overlay_start.1,
                                        );
                                        win32::set_window_rect(panel_hwnd, pnx, pny, pw, ph);
                                    }
                                }
                                Some(actual_overlay)
                            },
                        );
                        if committed && observed_overlay != previous_overlay {
                            last_overlay = observed_overlay;
                            moves = moves.saturating_add(1);
                        } else if !committed
                            && (!crate::input::source_caption_drag_active()
                                || crate::input::source_caption_drag_epoch() != drag_epoch)
                        {
                            break;
                        }

                        if last_diag.elapsed() >= Duration::from_millis(250) {
                            log::debug!(
                                "source-drag-follower-sample: epoch={} reference={} source=({sx},{sy}) source_size={}x{} overlay=({},{}) samples={} moves={} geometry_changes={}",
                                drag_epoch,
                                source_reference_kind,
                                sw,
                                sh,
                                last_overlay.0,
                                last_overlay.1,
                                samples,
                                moves,
                                geometry_changes
                            );
                            last_diag = Instant::now();
                        }
                    }
                    std::thread::sleep(Duration::from_millis(8));
                }
                log::info!(
                    "source-drag-follower-stop: epoch={} reference={} samples={} moves={} geometry_changes={} elapsed_ms={} final_source=({},{}) final_overlay=({},{}) active={} current_epoch={}",
                    drag_epoch,
                    source_reference_kind,
                    samples,
                    moves,
                    geometry_changes,
                    started.elapsed().as_millis(),
                    last_source.0,
                    last_source.1,
                    last_overlay.0,
                    last_overlay.1,
                    crate::input::source_caption_drag_active(),
                    crate::input::source_caption_drag_epoch()
                );
            })
            .ok()?;
        Some(Self {
            stop,
            handle: Some(handle),
        })
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }

    fn finished(&self) -> bool {
        self.handle
            .as_ref()
            .is_none_or(|handle| handle.is_finished())
    }
}

fn stop_drag_follower(slot: &mut Option<DragFollower>) {
    if let Some(mut follower) = slot.take() {
        follower.stop_and_join();
    }
}

fn resume_provider_transition_after_interpolated_present(
    input: &crate::input::InputSystem,
    s: &mut Session,
    route: &'static str,
) {
    if !s.provider_transition_input_suspended {
        return;
    }
    input.set_transition_suspended(false);
    s.provider_transition_input_suspended = false;
    s.provider_transition_input_suspended_since = None;
    s.input_reenable_after = Some(Instant::now() + Duration::from_millis(120));
    log::info!(
        "provider-transition-ready: first-interpolated-frame-presented route={} input=resume-after-120ms",
        route
    );
}

fn engine_main(
    rx: Receiver<Cmd>,
    control_tx: Sender<Cmd>,
    pending_chain: Arc<Mutex<Option<Vec<StageSpec>>>>,
    pending_no_engage: Arc<Mutex<PendingNoEngage>>,
    stop_requested: Arc<AtomicBool>,
    metrics: Metrics,
    status: Arc<Mutex<Status>>,
    base_dir: std::path::PathBuf,
    initial_onnx_backend: OnnxBackendPreference,
    initial_trt_device_id: Option<i32>,
    initial_trt_cache_root: std::path::PathBuf,
) -> Result<()> {
    // ---- resident resources (created exactly once) ----
    let mut overlay = OverlayWindow::new(64, 64)?;
    let gl = unsafe { glow::Context::from_loader_function(|s| overlay.win.loader(s)) };
    let mut gc = GlContext::new(Rc::new(gl));
    match gc.init_external_memory(|symbol| overlay.win.loader(symbol)) {
        Ok(()) => log::info!(
            "DirectML/OpenGL GPU bridge available: GL LUID={:02x?}",
            gc.external_device_luid().unwrap_or_default()
        ),
        Err(error) => {
            log::info!("DirectML/OpenGL GPU bridge unavailable; using stable CPU transfer: {error}")
        }
    }
    unsafe {
        log::info!(
            "OpenGL renderer: vendor='{}' renderer='{}' version='{}'",
            gc.gl.get_parameter_string(glow::VENDOR),
            gc.gl.get_parameter_string(glow::RENDERER),
            gc.gl.get_parameter_string(glow::VERSION)
        );
    }
    let mut factory = StageFactory::new(base_dir);
    factory.set_onnx_backend(
        initial_onnx_backend,
        initial_trt_device_id,
        initial_trt_cache_root,
    );
    let mut session: Option<Session> = None;
    let mut drag_follower: Option<DragFollower> = None;
    let mut drag_session_epoch: Option<u64> = None;
    let mut input = crate::input::InputSystem::start();
    let mut no_engage_rects: Vec<(i32, i32, i32, i32, isize)> = Vec::new();
    let mut input_autohide_secs: f32 = 3.0;
    let mut input_speed_fix: bool = true;
    let mut interp_factor: u32 = 2;
    let mut downscaler = crate::render::scaler::Kernel::Spline36;
    let mut panel_hwnd: isize = 0;
    let mut panel_bar = (126, 30);
    let mut panel_chip = (30, 24);
    let mut panel_visible = true;
    // vsync doubles as frame pacing: with a blocking SwapBuffers the manual
    // mid-frame pacing sleep must be OFF, otherwise both throttles stack and
    // the presentation cadence can stutter.
    let mut vsync_on = false;
    // Separate from VSync: this mode intentionally uses a short bounded queue.
    let mut smooth_pacing_on = false;
    let mut duplicate_frame_reduction_on = false;
    let mut panel_chipped = false;
    let mut last_panel_raise_check = Instant::now() - Duration::from_secs(1);
    let mut last_helper_compositor_diag = Instant::now() - Duration::from_secs(3);
    let mut last_source_window_poll = Instant::now() - Duration::from_millis(250);
    let mut last_source_visible_windows: Vec<isize> = Vec::new();
    let mut gui_priority_hwnd: isize = 0;
    let mut gui_priority_topmost = false;
    let mut deferred_backend_switch: Option<(
        OnnxBackendPreference,
        Option<i32>,
        std::path::PathBuf,
        Vec<StageSpec>,
    )> = None;
    let mut deferred_interp_factor: Option<u32> = None;
    log::info!("render engine ready (persistent GL context)");

    loop {
        if crate::input::take_panel_stop_action() {
            log::info!("panel-priority-direct-dispatch: action=stop");
            *pending_chain.lock().unwrap() = None;
            pending_no_engage.lock().unwrap().publish(Vec::new());
            stop_requested.store(true, Ordering::Release);
            let _ = crate::render::onnx_stage::request_onnx_cancel();
            let _ = crate::render::onnx_stage::request_tensorrt_cancel();
            recover_visible_capture_state_now(&status, "control-panel-priority");

            // Native panel Stop bypasses EngineHandle::send(Cmd::Stop), so no
            // WakeForStop token would otherwise exist. Enqueue it *after* the
            // direct click is observed: every older queued command remains on
            // the pre-Stop side of the boundary, while the WakeForStop arm
            // clears Status::stopping only after those stale commands drain.
            if control_tx.send(Cmd::WakeForStop).is_err() {
                status
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .stopping = false;
                log::warn!("panel-stop-boundary-enqueue-failed: stopping latch cleared");
            } else {
                log::debug!("panel-stop-boundary-enqueued");
            }
        }
        // Stop has an out-of-band priority lane. A normal mpsc Stop command
        // can sit behind geometry/panel/backend messages; after a cancelled
        // inference unwinds, visible/session teardown must run before any of
        // those stale commands or a coalesced chain rebuild.
        if stop_requested.swap(false, Ordering::AcqRel) {
            deferred_backend_switch = None;
            deferred_interp_factor = None;
            *pending_chain.lock().unwrap() = None;
            input.set_transition_suspended(false);
            input.release();
            stop_drag_follower(&mut drag_follower);
            drag_session_epoch = None;
            stop_session(&mut session, &mut overlay, &mut gc, &status);
            factory.retain_tensorrt_sessions_for_capture_restart();
            // Keep Start locked until the WakeForStop token reaches the head
            // of the queue. Every command before that token predates Stop and
            // must not become live again if the user clicks Start quickly.
            status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .stopping = true;
            log::info!("stop-priority-lane-complete");
        }
        // Geometry is sampled on the GUI thread from the actual Win32 window
        // rect. Consume only the newest sample before input.configure(). This
        // is intentionally independent of language, DPI, UI mode and stats-row
        // height: no assumed GUI dimensions exist here.
        // Cursor/UI geometry is advisory state. Never block the video thread
        // behind a GUI-side mutex; if the GUI happens to be publishing now,
        // consume it on the next loop instead.
        let latest_no_engage = pending_no_engage
            .try_lock()
            .ok()
            .and_then(|mut pending| pending.take_latest());
        if let Some((rects, overwritten)) = latest_no_engage {
            if no_engage_rects != rects {
                log::info!(
                    "no-engage latest-applied: coalesced={} rects={:?}",
                    overwritten,
                    rects
                );
            } else if overwritten > 0 {
                log::debug!(
                    "no-engage latest-coalesced: coalesced={} geometry=unchanged",
                    overwritten
                );
            }
            no_engage_rects = rects;
        }
        if let Some(specs) = pending_chain.lock().unwrap().take() {
            if deferred_backend_switch.take().is_some() {
                let mut state = status.lock().unwrap();
                state.onnx_backend_switching = false;
                state.onnx_backend_error = None;
                state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
                log::info!("onnx-backend-switch-cancelled: reason=coalesced-chain-replaced");
            }
            if let Some(s) = session.as_mut() {
                input.set_transition_suspended(true);
                s.provider_transition_input_suspended = true;
                s.provider_transition_input_suspended_since = Some(Instant::now());
                apply_chain_update(s, &mut gc, &mut factory, &specs, &metrics, &status);
                let wait_for_first_interp = matches!(
                    s.chain.interp_stage(),
                    Some(crate::render::chain::InterpHandle::Onnx { .. })
                );
                if !wait_for_first_interp {
                    input.set_transition_suspended(false);
                    s.provider_transition_input_suspended = false;
                    s.provider_transition_input_suspended_since = None;
                }
            }
        }
        // -- commands --
        let cmd = if session.is_some() {
            rx.try_recv().ok()
        } else {
            // idle: block until something arrives (zero CPU)
            rx.recv_timeout(Duration::from_millis(250)).ok()
        };
        if let Some(cmd) = cmd {
            match cmd {
                Cmd::Shutdown => {
                    // The render loop returns immediately from this arm, so clearing the
                    // deferred request here had no observable effect and only produced an
                    // unused-assignment warning. Session teardown discards it with the loop.
                    let shutdown_t0 = Instant::now();
                    log::debug!("render-engine shutdown begin");
                    input.set_transition_suspended(false);
                    stop_session(&mut session, &mut overlay, &mut gc, &status);
                    factory.clear_onnx_cache();
                    let input_t0 = Instant::now();
                    input.stop();
                    log::debug!(
                        "render-engine shutdown complete: total_ms={:.1} input_stop_ms={:.1}",
                        shutdown_t0.elapsed().as_secs_f64() * 1000.0,
                        input_t0.elapsed().as_secs_f64() * 1000.0
                    );
                    return Ok(());
                }
                Cmd::Stop => {
                    deferred_backend_switch = None;
                    deferred_interp_factor = None;
                    // Release native cursor/button ownership immediately. ONNX
                    // cache/session teardown can take noticeable time, and the
                    // GUI must remain movable while that cleanup finishes.
                    input.set_transition_suspended(false);
                    input.release();
                    stop_session(&mut session, &mut overlay, &mut gc, &status);
                    factory.retain_tensorrt_sessions_for_capture_restart();
                }
                Cmd::WakeForStop => {
                    // Idle engines wake inside recv_timeout before the loop's
                    // priority check. Perform the same teardown here only when
                    // the atomic request has not already been consumed.
                    if stop_requested.swap(false, Ordering::AcqRel) {
                        deferred_backend_switch = None;
                        deferred_interp_factor = None;
                        *pending_chain.lock().unwrap() = None;
                        input.set_transition_suspended(false);
                        input.release();
                        stop_session(&mut session, &mut overlay, &mut gc, &status);
                        factory.retain_tensorrt_sessions_for_capture_restart();
                    }
                    status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .stopping = false;
                    // The panel Stop can arrive without an egui frame. Wake the
                    // real GUI HWND at the exact idle boundary so the main
                    // Start/Stop control repaints immediately instead of
                    // waiting for the 250 ms idle heartbeat.
                    crate::platform::win32::wake_main_gui_for_panel_action(false);
                    // All commands queued before Stop have now been drained and
                    // the old session has been released. Clear the cooperative
                    // cancellation epoch so an idle TensorRT checkbox switch is
                    // accepted without requiring another capture Start first.
                    crate::render::onnx_stage::finish_onnx_stop_request();
                    log::info!("stop-queue-boundary-complete");
                }
                Cmd::SaveScreenshot(path) => {
                    if let Some(s) = session.as_ref() {
                        if let Some(tex) = s.last_tex {
                            // Readback already matches PNG's top-left pixel order.
                            // Any row/column correction mirrors the saved image.
                            let rgba = gc.download_rgba8(tex);
                            save_screenshot_async(path, tex.w() as u32, tex.h() as u32, rgba);
                        } else {
                            log::warn!(
                                "screenshot skipped: no filtered frame has been presented yet"
                            );
                        }
                    } else {
                        log::warn!("screenshot skipped: capture is not running");
                    }
                }
                Cmd::Start {
                    hwnd,
                    specs,
                    mode,
                    ratio,
                    fps_cap,
                    hide_source,
                    client_only,
                    hdr,
                    hdr_sdr_mode,
                    gpu_adapter,
                    source_restore_rect,
                    source_was_maximized,
                    deferred_capture_resolution,
                    capture_canvas,
                } => {
                    deferred_backend_switch = None;
                    deferred_interp_factor = None;
                    input.set_transition_suspended(false);
                    stop_session(&mut session, &mut overlay, &mut gc, &status);
                    if crate::render::onnx_stage::tensorrt_cancel_requested()
                        || crate::render::onnx_stage::onnx_cancel_requested()
                    {
                        log::info!(
                            "capture-start-cancelled-before-provider-prepare: stop request already pending"
                        );
                        continue;
                    }
                    {
                        let mut state = status
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        state.starting = true;
                        state.stopping = false;
                        let fullscreen_origin = source_restore_rect
                            .map(|rect| win32::is_rect_monitor_fullscreen(hwnd, rect))
                            .unwrap_or_else(|| win32::is_monitor_fullscreen(hwnd));
                        state.source_recovery = Some((
                            hwnd,
                            source_restore_rect,
                            source_was_maximized,
                            // Fullscreen pixel-lock sessions deliberately never
                            // promote the foreign source HWND, so recovery must
                            // not issue a matching NOTOPMOST mutation either.
                            win32::is_topmost(hwnd) || fullscreen_origin,
                        ));
                    }
                    factory.set_gpu_adapter(gpu_adapter);
                    // Keep a previously warmed TensorRT ORT session across a normal
                    // Stop -> Start. DirectML remains recreate-on-start, matching
                    // its existing shape-specialization safety behavior.
                    factory.retain_tensorrt_sessions_for_capture_restart();
                    match start_session(
                        hwnd,
                        &specs,
                        mode,
                        ratio,
                        fps_cap,
                        hide_source,
                        client_only,
                        hdr,
                        hdr_sdr_mode,
                        source_restore_rect,
                        source_was_maximized,
                        deferred_capture_resolution,
                        capture_canvas,
                        smooth_pacing_on,
                        duplicate_frame_reduction_on,
                        &mut factory,
                    ) {
                        Ok((mut s, errs, warning)) => {
                            if crate::render::onnx_stage::tensorrt_cancel_requested()
                                || crate::render::onnx_stage::onnx_cancel_requested()
                            {
                                log::info!(
                                    "capture-start-cancelled: provider preparation completed after stop request; session will not be published"
                                );
                                session = Some(s);
                                stop_session(&mut session, &mut overlay, &mut gc, &status);
                                continue;
                            }
                            let usage = s.chain.onnx_backend_usage();
                            if matches!(
                                s.chain.interp_stage(),
                                Some(crate::render::chain::InterpHandle::Onnx { .. })
                            ) {
                                input.set_transition_suspended(true);
                                s.provider_transition_input_suspended = true;
                                s.provider_transition_input_suspended_since = Some(Instant::now());
                            }
                            let rect = overlay_geometry(&s, &overlay);
                            reposition_overlay_for_gui_mode(
                                &mut overlay, rect.0, rect.1, rect.2, rect.3, gui_priority_topmost,
                            );
                            log::info!("overlay reveal deferred until first valid filtered frame");
                            enforce_gui_priority(
                                gui_priority_hwnd,
                                gui_priority_topmost,
                                panel_hwnd,
                                overlay.hwnd().0 as isize,
                            );
                            let source_rect = if client_only {
                                win32::client_rect_on_screen(hwnd)
                            } else {
                                win32::window_rect(hwnd)
                            };
                            let initial_content = source_rect
                                .map(|(_, _, w, h)| {
                                    crate::input::content_rect(
                                        crate::input::Rect {
                                            x: rect.0,
                                            y: rect.1,
                                            w: rect.2,
                                            h: rect.3,
                                        },
                                        w,
                                        h,
                                    )
                                })
                                .unwrap_or(crate::input::Rect {
                                    x: rect.0,
                                    y: rect.1,
                                    w: rect.2,
                                    h: rect.3,
                                });
                            let mut g = status.lock().unwrap();
                            // Publish complete geometry before `running`.
                            // The GUI creates its panel as soon as running is
                            // true, so a later geometry write visibly flashes
                            // the panel at Windows' default position.
                            g.overlay_rect = rect;
                            g.content_rect = (
                                initial_content.x,
                                initial_content.y,
                                initial_content.w,
                                initial_content.h,
                            );
                            g.starting = false;
                            g.running = true;
                            g.stopping = false;
                            g.target_title = win32::window_title(hwnd);
                            g.chain_errors = errs;
                            g.last_error = None;
                            g.warning = warning;
                            g.source_live_fullscreen = s.source_live_fullscreen;
                            g.presented = 0;
                            g.overlay_hwnd = overlay.hwnd().0 as isize;
                            g.hidden_src = s.hid_source.map(|wl| (s.hwnd, wl));
                            g.source_recovery = Some((
                                s.hwnd,
                                s.source_restore_rect,
                                s.source_was_maximized,
                                // `true` here means recovery must leave the
                                // z-order alone. Fullscreen sessions were not
                                // promoted, even when the source originally
                                // was not TOPMOST.
                                s.src_was_topmost || s.source_monitor_fullscreen,
                            ));
                            g.onnx_tensorrt_stages = usage.tensorrt;
                            g.onnx_cuda_stages = usage.cuda;
                            g.onnx_directml_fallbacks = usage.directml_fallback;
                            metrics.reset();
                            metrics.set_stage_order(s.chain.metric_stage_order());
                            metrics.set_monitor(
                                (s.monitor_rect.2, s.monitor_rect.3),
                                s.monitor_refresh_hz,
                            );
                            session = Some(s);
                        }
                        Err(e) => {
                            let cancelled = crate::render::onnx_stage::tensorrt_cancel_requested()
                                || crate::render::onnx_stage::onnx_cancel_requested();
                            {
                                let mut state = status
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                                state.starting = false;
                                state.running = false;
                                if !cancelled {
                                    state.stopping = false;
                                }
                            }
                            if let Some(rect) = source_restore_rect {
                                let restored =
                                    win32::restore_window_rect(hwnd, rect, source_was_maximized);
                                log::warn!(
                                    "capture start failed; source geometry rollback: hwnd={hwnd:#x} rect={rect:?} maximized={source_was_maximized} ok={restored}"
                                );
                            }
                            let message = format!("{e:#}");
                            if cancelled {
                                log::info!(
                                    "capture start cancelled by Stop: hwnd={hwnd:#x}: {message}"
                                );
                            } else {
                                log::error!("capture start failed: hwnd={hwnd:#x}: {message}");
                                status.lock().unwrap().last_error = Some(message);
                            }
                        }
                    }
                }
                Cmd::ApplyChain(specs) => {
                    if crate::render::onnx_stage::onnx_cancel_requested() {
                        log::info!("filter-chain-update-skipped: Stop is pending");
                        input.set_transition_suspended(false);
                        continue;
                    }
                    if deferred_backend_switch.take().is_some() {
                        let mut state = status.lock().unwrap();
                        state.onnx_backend_switching = false;
                        state.onnx_backend_error = None;
                        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
                        log::info!("onnx-backend-switch-cancelled: reason=filter-chain-replaced");
                    }
                    if let Some(s) = session.as_mut() {
                        input.set_transition_suspended(true);
                        s.provider_transition_input_suspended = true;
                        s.provider_transition_input_suspended_since = Some(Instant::now());
                        apply_chain_update(s, &mut gc, &mut factory, &specs, &metrics, &status);
                        let wait_for_first_interp = matches!(
                            s.chain.interp_stage(),
                            Some(crate::render::chain::InterpHandle::Onnx { .. })
                        );
                        if !wait_for_first_interp {
                            input.set_transition_suspended(false);
                            s.provider_transition_input_suspended = false;
                            s.provider_transition_input_suspended_since = None;
                        }
                    } else {
                        input.set_transition_suspended(false);
                    }
                }
                Cmd::SwitchOnnxBackend {
                    backend,
                    trt_device_id,
                    cache_root,
                    specs,
                } => {
                    if crate::render::onnx_stage::onnx_cancel_requested() {
                        log::info!("onnx-backend-switch-skipped: Stop is pending");
                        input.set_transition_suspended(false);
                        let mut state = status.lock().unwrap();
                        state.onnx_backend_switching = false;
                        state.onnx_backend_error = None;
                        state.onnx_backend_revision = state.onnx_backend_revision.saturating_add(1);
                        continue;
                    }
                    input.set_transition_suspended(true);
                    if let Some(s) = session.as_mut() {
                        s.provider_transition_input_suspended = true;
                        s.provider_transition_input_suspended_since = Some(Instant::now());
                    }
                    let gpu_pair_in_flight = session.as_ref().is_some_and(|s| {
                        s.gpu_interp_pack_pending.is_some() || s.gpu_interp_pending.is_some()
                    });
                    if gpu_pair_in_flight {
                        let previous = factory.onnx_preference();
                        deferred_backend_switch = Some((backend, trt_device_id, cache_root, specs));
                        let mut state = status.lock().unwrap();
                        state.onnx_backend_switching = true;
                        state.onnx_backend_error = None;
                        log::info!(
                            "onnx-backend-switch-deferred: {previous:?} -> {backend:?} reason=gpu-pair-in-flight"
                        );
                    } else {
                        let committed = switch_onnx_backend(
                            &mut session,
                            &mut gc,
                            &mut factory,
                            backend,
                            trt_device_id,
                            cache_root,
                            &specs,
                            &metrics,
                            &status,
                        );
                        let wait_for_first_interp = committed
                            && session.as_ref().is_some_and(|s| {
                                matches!(
                                    s.chain.interp_stage(),
                                    Some(crate::render::chain::InterpHandle::Onnx { .. })
                                )
                            });
                        if !wait_for_first_interp {
                            input.set_transition_suspended(false);
                            if let Some(s) = session.as_mut() {
                                s.provider_transition_input_suspended = false;
                                s.provider_transition_input_suspended_since = None;
                            }
                        }
                    }
                }
                Cmd::SetMode { mode, ratio } => {
                    if let Some(s) = session.as_mut() {
                        let previous_mode = s.mode;
                        let previous_ratio = s.ratio;
                        let mode_changed = previous_mode != mode;
                        let ratio_changed = (previous_ratio - ratio).abs() > 0.0005;
                        if mode_changed || ratio_changed {
                            // A geometry transition must be transactional from the input
                            // layer's point of view. Releasing only on Auto/Fixed changes
                            // left Fixed-mode ratio edits mapped to the old overlay rect.
                            input.release();
                            s.paced_present_deadline = None;
                            // Display-only geometry does not invalidate source
                            // timestamps or RIFE history. Resetting the output
                            // cadence here made ratio edits lose 96/120fps until
                            // a full stop/start rebuilt the clock.
                            s.input_reenable_after =
                                Some(Instant::now() + Duration::from_millis(120));
                            if mode_changed {
                                s.placed = false;
                            }
                            s.source_drag_active = false;
                            s.source_client_drag_active = false;
                            s.source_client_unowned_rebase_pending = false;
                            s.source_client_drag_raw_origin = None;
                            s.source_client_drag_overlay_origin = None;
                            if mode == ScaleMode::Fixed {
                                s.last_src_pos = keep_source_window_reachable(s.hwnd);
                            } else {
                                s.last_src_pos =
                                    win32::client_rect_on_screen(s.hwnd).map(|(x, y, _, _)| (x, y));
                            }
                            log::info!(
                                "display-geometry-transition: {:?}/{:.2} -> {:?}/{:.2} input=released reenable_ms=120 overlay_replaced={} source_client={:?} interp_cadence=preserved",
                                previous_mode,
                                previous_ratio,
                                mode,
                                ratio,
                                mode_changed,
                                s.last_src_pos
                            );
                        }
                        s.mode = mode;
                        s.ratio = ratio;
                    }
                }
                Cmd::SetCaptureGeometry { requested, applied } => {
                    status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .capture_resolution_resize_intent = None;
                    if let Some(s) = session.as_mut() {
                        let normalized_applied = normalize_capture_client_size(applied);
                        if normalized_applied != applied {
                            log::info!(
                                "capture-geometry-even-normalize: requested={}x{} applied={}x{} normalized={}x{} reason=wgc-even-frame-contract",
                                requested.0,
                                requested.1,
                                applied.0,
                                applied.1,
                                normalized_applied.0,
                                normalized_applied.1
                            );
                        }
                        let applied = normalized_applied;
                        let target = (applied.0 as i32, applied.1 as i32);
                        s.capture_canvas_suspended_for_fullscreen = false;
                        s.capture_resolution_reapply_pending = false;
                        s.fullscreen_exit_candidate_since = None;
                        let previous = s.in_size;
                        let overload_notice_latched = status
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner())
                            .glsl_overload_notice_latched;

                        // The same filter chain can have completely different
                        // admission behavior at a different capture size.  A
                        // user capture-resolution command therefore starts a
                        // new workload context even when the WGC auto-follow
                        // reached the target before this command was dequeued.
                        if !s.glsl_overload_history.is_empty() || s.glsl_overload_hint {
                            log::info!(
                                "glsl-responsiveness-history: action=clear reason=capture-resolution-context-changed entries={}",
                                s.glsl_overload_history.len()
                            );
                        }
                        s.glsl_overload_history.clear();
                        s.glsl_overload_hint = false;

                        if overload_notice_latched && glsl_guard_eligible(s) {
                            s.glsl_resolution_recheck_active = true;
                            s.glsl_resolution_recheck_started = Some(Instant::now());
                            s.glsl_resolution_recheck_healthy_since = None;

                            // Workload history is geometry-specific.  Carrying a
                            // 1920x1080/1440x810 overload hint into 1280x720 (or
                            // lower) can fast-reconfirm a warning that a clean
                            // Stop -> Start would never raise.  Treat the new
                            // capture size as a fresh admission context.
                            s.glsl_guard.reset();
                            s.glsl_chain_settle_until = None;
                            s.glsl_chain_immediate_until = None;

                            // Geometry changes already reset temporal state.
                            // Interpolation never suppresses selected post filters,
                            // so there is no GUI-dependent interpolation pause to clear.
                            {
                                let mut state = status
                                    .lock()
                                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                                state.glsl_interactive_pause = false;
                                if state.glsl_overload_auto_stop_deadline.take().is_some() {
                                    log::info!(
                                        "glsl-overload-auto-stop: cancelled reason=capture-resolution-recheck"
                                    );
                                }
                            }
                            log::info!(
                                "glsl-overload-notice: resolution-recheck-armed current={}x{} requested={}x{} applied={}x{} settle_ms={} policy=fresh-session-equivalent history=cleared",
                                previous.0,
                                previous.1,
                                requested.0,
                                requested.1,
                                applied.0,
                                applied.1,
                                GLSL_GUARD_RESOLUTION_RECHECK_SETTLE.as_millis()
                            );
                        }
                        // Chromium can deliver the exact new WGC size before
                        // the GUI thread's SetCaptureGeometry command reaches
                        // this loop. If automatic detection already committed
                        // and rebuilt this same shape, a second reset only
                        // destroys a warm provider and causes another blackout.
                        if previous == target
                            && s.pending_resize_size.is_none()
                            && !s.onnx_geometry_rebuild_pending
                        {
                            s.capture_canvas = Some(target);
                            s.display_aspect = target;
                            s.deferred_capture_resolution = None;
                            s.capture_resolution_applied = false;
                            s.capture_resolution_wait_started = None;
                            s.capture_resolution_repaint_last = None;
                            s.capture_resolution_nudge_done = false;
                            s.capture_resolution_native_fallback = false;
                            s.onnx_geometry_deferred_logged = false;
                            log::info!(
                                "capture-geometry-transition-deduplicated: current={}x{} requested={}x{} applied={}x{} action=metadata-only onnx_rebuild=false",
                                previous.0,
                                previous.1,
                                requested.0,
                                requested.1,
                                applied.0,
                                applied.1
                            );
                            continue;
                        }
                        input.release();
                        // Keep the last complete magnified frame visible while
                        // the hidden source HWND changes size.  Hiding the
                        // overlay here exposed the desktop because the source is
                        // already visually hidden during capture.  Wrong-size
                        // WGC transition frames are discarded below, so the
                        // currently presented framebuffer is a safe transition
                        // shield until the first fully filtered exact-size frame
                        // replaces it atomically.
                        let transition_shield_visible =
                            overlay.is_visible() && s.last_tex.is_some();
                        s.deferred_capture_resolution = Some((requested, applied));
                        s.capture_resolution_applied = true;
                        s.capture_resolution_wait_started = Some(Instant::now());
                        s.capture_resolution_repaint_last = None;
                        s.capture_resolution_nudge_done = false;
                        s.capture_resolution_native_fallback = false;
                        s.onnx_geometry_deferred_logged = false;
                        s.capture_canvas = Some(target);
                        // Preserve the currently presented geometry while the
                        // old complete frame is acting as the transition
                        // shield. The first exact target-size WGC frame commits
                        // display_aspect below; Fixed mode then resizes on the
                        // following tick without ever exposing the desktop.
                        s.pending_resize_size = Some(target);
                        s.input_reenable_after = Some(Instant::now() + Duration::from_millis(120));
                        s.last_geom_change = Some(Instant::now());
                        s.starve_released = false;
                        s.paced_present_deadline = None;
                        s.interp_present_deadline = None;
                        reset_gpu_interp_for_geometry_transition(
                            s,
                            &mut gc,
                            "capture-resolution-command",
                        );
                        s.interp_pending = None;
                        s.interp_worker = None;
                        s.interp_last_tick = None;
                        s.interp_tail = None;
                        s.hist.clear();
                        if let Some(texture) = s.prev_tex.take() {
                            gc.recycle(texture);
                        }
                        if let Some(texture) = s.prev2_tex.take() {
                            gc.recycle(texture);
                        }
                        s.prev_tex_seq = 0;
                        s.prev_tex_source_time_100ns = None;
                        s.cadence.reset();
                        s.smooth_pacer.reset();
                        s.flow_output_cadence.reset();
                        s.present_cadence = PresentCadence::default();
                        s.source.set_queue_enabled(false);
                        s.source.set_queue_enabled(s.chain.has_interp());
                        s.onnx_geometry_rebuild_pending = s.chain.has_onnx();
                        metrics.reset();
                        win32::request_window_repaint(s.hwnd);
                        log::info!(
                            "capture-geometry-transition-armed: previous={}x{} requested={}x{} applied={}x{} overlay_shield={} input=released queue=flushed onnx_rebuild={} display_aspect={}x{}",
                            previous.0,
                            previous.1,
                            requested.0,
                            requested.1,
                            applied.0,
                            applied.1,
                            transition_shield_visible,
                            s.onnx_geometry_rebuild_pending,
                            s.display_aspect.0,
                            s.display_aspect.1
                        );
                    }
                }
                Cmd::ClearCaptureGeometry => {
                    status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .capture_resolution_resize_intent = None;
                    if let Some(s) = session.as_mut() {
                        input.release();
                        s.capture_canvas = None;
                        s.deferred_capture_resolution = None;
                        s.capture_resolution_applied = false;
                        s.capture_resolution_wait_started = None;
                        s.capture_resolution_repaint_last = None;
                        s.capture_resolution_nudge_done = false;
                        s.capture_resolution_native_fallback = false;
                        s.capture_canvas_suspended_for_fullscreen = false;
                        s.capture_resolution_reapply_pending = false;
                        s.fullscreen_exit_candidate_since = None;
                        if s.in_size.0 > 0 && s.in_size.1 > 0 {
                            s.display_aspect = s.in_size;
                        }
                        s.pending_resize_size = None;
                        s.input_reenable_after = Some(Instant::now() + Duration::from_millis(120));
                        log::info!(
                            "capture-resolution cleared: action=native-wgc-follow current={}x{}",
                            s.in_size.0,
                            s.in_size.1
                        );
                    }
                }
                Cmd::SetNoEngage(r) => {
                    // Compatibility fallback for any internal direct sender.
                    // EngineHandle::send coalesces this command out-of-band, so
                    // normal GUI updates never enter this FIFO.
                    if no_engage_rects != r {
                        log::warn!("no-engage legacy-queued update applied: {:?}", r);
                    }
                    no_engage_rects = r;
                }
                Cmd::SetInputOpts {
                    autohide_secs,
                    speed_fix,
                } => {
                    input_autohide_secs = autohide_secs;
                    input_speed_fix = speed_fix;
                }
                Cmd::SetInterpFactor(f) => {
                    let in_flight = session.as_ref().is_some_and(|s| {
                        s.gpu_interp_pack_pending.is_some()
                            || s.gpu_interp_pending.is_some()
                            || s.interp_pending.is_some()
                    });
                    if in_flight {
                        deferred_interp_factor = Some(f.clamp(2, 5));
                        log::info!(
                            "frame-interpolation factor deferred: requested=x{} reason=pair-in-flight",
                            f.clamp(2, 5)
                        );
                    } else {
                        apply_interp_factor_now(session.as_mut(), &mut gc, &mut interp_factor, f);
                    }
                }
                Cmd::SetDownscaler(name) => {
                    downscaler = crate::render::scaler::Kernel::from_name(&name)
                }
                Cmd::SetPanel { hwnd, bar, chip } => {
                    panel_hwnd = hwnd;
                    panel_bar = bar;
                    panel_chip = chip;
                    if panel_hwnd != 0 && panel_visible {
                        // The egui thread owns visibility and layered alpha.
                        // Cross-thread ShowWindow here blocked AMD presentation
                        // and made the entire magnified view flash.
                        enforce_gui_priority(
                            gui_priority_hwnd,
                            gui_priority_topmost,
                            panel_hwnd,
                            overlay.hwnd().0 as isize,
                        );
                    }
                }
                Cmd::SetGuiPriority { hwnd, topmost } => {
                    gui_priority_hwnd = hwnd;
                    gui_priority_topmost = topmost;
                    enforce_gui_priority(
                        gui_priority_hwnd,
                        gui_priority_topmost,
                        panel_hwnd,
                        overlay.hwnd().0 as isize,
                    );
                }
                Cmd::SetPanelState { visible, chip } => {
                    panel_visible = visible;
                    panel_chipped = chip;
                    // Keep the v528 restore guard through any older queued
                    // chip=true state. It may have been sent before the GUI
                    // began the atomic restore and must not re-shrink the host.
                    // Release the guard only when this thread actually consumes
                    // the final bar state (chip=false), or when the panel hides.
                    if !chip || !visible {
                        PANEL_LURK_RESTORE_GEOMETRY_GUARD.store(false, Ordering::Release);
                    }
                    // Visibility is still owned by the GUI/settings path, but
                    // helper ownership must be updated immediately. A hidden
                    // panel cannot remain the cursor sprite's owner or Windows
                    // may hide the cursor with it.
                    enforce_gui_priority(
                        gui_priority_hwnd,
                        gui_priority_topmost,
                        panel_hwnd,
                        overlay.hwnd().0 as isize,
                    );
                }
                Cmd::SetVsync(on) => {
                    vsync_on = on;
                    overlay.win.set_swap_interval(if on { 1 } else { 0 });
                }
                Cmd::SetSmoothPacing(on) => {
                    smooth_pacing_on = on;
                    if let Some(s) = session.as_mut() {
                        s.smooth_pacing = on;
                        s.smooth_pacer.reset();
                        s.paced_present_deadline = None;
                        s.flow_output_cadence.reset();
                        s.present_cadence = PresentCadence::default();
                        s.interp_present_deadline = None;
                        s.smooth_content_signature = None;
                        s.smooth_content_duplicates = 0;
                        s.smooth_content_seq = 0;
                        s.smooth_content_unique_since_duplicate = 4;
                        s.smooth_content_last_candidate_seq = None;
                        s.smooth_content_last_candidate_time = None;
                        s.smooth_content_pattern_hits = 0;
                        s.smooth_content_candidate_gaps.clear();
                        s.smooth_content_24p_detected = false;
                        s.source.set_queue_enabled(s.chain.has_interp());
                    }
                    log::info!("smooth-pacing: enabled={on}");
                }
                Cmd::SetDuplicateFrameReduction(on) => {
                    duplicate_frame_reduction_on = on;
                    if let Some(s) = session.as_mut() {
                        s.duplicate_frame_reduction = on;
                        s.duplicate_signature = None;
                        s.duplicate_skipped = 0;
                        s.duplicate_motion_guard = 0;
                        s.duplicate_lookahead_wait = None;
                        s.strict_static_streak = 0;
                        s.strict_static_last_present = Instant::now();
                        s.strict_static_present_skipped = 0;
                    }
                    log::info!("duplicate-frame reduction: enabled={on}");
                }
            }
        }

        let deferred_switch_ready = deferred_backend_switch.is_some()
            && session.as_ref().is_none_or(|s| {
                s.gpu_interp_pack_pending.is_none() && s.gpu_interp_pending.is_none()
            });
        if deferred_switch_ready {
            let (backend, trt_device_id, cache_root, specs) = deferred_backend_switch
                .take()
                .expect("deferred backend switch");
            log::info!("onnx-backend-switch-resume: requested={backend:?} state=gpu-idle");
            let committed = switch_onnx_backend(
                &mut session,
                &mut gc,
                &mut factory,
                backend,
                trt_device_id,
                cache_root,
                &specs,
                &metrics,
                &status,
            );
            let wait_for_first_interp = committed
                && session.as_ref().is_some_and(|s| {
                    matches!(
                        s.chain.interp_stage(),
                        Some(crate::render::chain::InterpHandle::Onnx { .. })
                    )
                });
            if !wait_for_first_interp {
                input.set_transition_suspended(false);
                if let Some(s) = session.as_mut() {
                    s.provider_transition_input_suspended = false;
                    s.provider_transition_input_suspended_since = None;
                }
            }
        }
        let deferred_factor_ready = deferred_interp_factor.is_some()
            && session.as_ref().is_none_or(|s| {
                s.gpu_interp_pack_pending.is_none()
                    && s.gpu_interp_pending.is_none()
                    && s.interp_pending.is_none()
            });
        if deferred_factor_ready {
            let factor = deferred_interp_factor
                .take()
                .expect("deferred interpolation factor");
            log::info!(
                "frame-interpolation factor resume: requested=x{} state=pair-idle",
                factor
            );
            apply_interp_factor_now(session.as_mut(), &mut gc, &mut interp_factor, factor);
        }

        let phase_tick_started = Instant::now();
        // Drive the REAL system-cursor hide/show on THIS (overlay-owning) thread.
        // MagShowSystemCursor is a silent no-op on the LL-hook thread, so the hook
        // only records intent (request_cursor_hidden) and we apply it here — the
        // the cursor-routing design architecture. Runs every tick; internally rate-limited.
        crate::input::pump_cursor_visibility();
        crate::input::pump_cursor_engage_commit();
        let phase_cursor_finished = Instant::now();

        // -- per-tick work --
        let Some(s) = session.as_mut() else {
            stop_drag_follower(&mut drag_follower);
            drag_session_epoch = None;
            // WATCHDOG: no session may ever leave a (stale-frame, possibly
            // fullscreen) overlay on screen. The user hit exactly this with a
            // PIP source: capture died without the stop path, the overlay kept
            // showing its last frame fullscreen and the mouse felt dead.
            if overlay.is_visible() {
                log::warn!("overlay watchdog: session gone but overlay visible — force hiding");
                detach_overlay_owned_helpers(overlay.hwnd().0 as isize);
                overlay.hide_for_stop();
                input.release();
            }
            overlay.win.pump_messages();
            continue;
        };

        // A fullscreen/orientation toggle can retire only the WGC session while
        // preserving the source HWND. Reconnect in place so the overlay, filter
        // chain, cursor mapping, and saved restore geometry survive the change.
        if !win32::is_window_valid(s.hwnd) || win32::is_minimized(s.hwnd) {
            log::warn!(
                "capture source unavailable: hwnd={:#x} valid={} minimized={}; stopping session",
                s.hwnd,
                win32::is_window_valid(s.hwnd),
                win32::is_minimized(s.hwnd)
            );
            stop_session(&mut session, &mut overlay, &mut gc, &status);
            input.release();
            continue;
        }
        if !s.source.alive() {
            let now = Instant::now();
            let reconnect_started = *s.source_restart_since.get_or_insert_with(|| {
                log::warn!(
                    "WGC session ended while source HWND remains valid: hwnd={:#x}; beginning in-place reconnect",
                    s.hwnd
                );
                input.release();
                now
            });
            if now.duration_since(s.source_restart_last) >= Duration::from_millis(250) {
                s.source_restart_last = now;
                let pixel_exact_raw_hint = if s.source_monitor_fullscreen
                    && s.capture_canvas.is_none()
                {
                    s.source_restore_rect
                        .or_else(|| win32::window_rect(s.hwnd))
                        .and_then(|(_, _, w, h)| (w > 0 && h > 0).then_some((w as u32, h as u32)))
                } else {
                    None
                };
                match WgcSource::start_fmt_with_pixel_lock(
                    s.hwnd,
                    s.fps_cap,
                    s.capture_client_only,
                    false,
                    pixel_exact_raw_hint,
                ) {
                    Ok(new_source) => {
                        s.source.stop();
                        s.source = new_source;
                        s.source.set_queue_enabled(s.chain.has_interp());
                        s.metric_seq = 0;
                        s.next_cap_deadline = None;
                        s.next_cap_deadline_src = None;
                        s.cap_rate_gate = CapRateGate::default();
                        s.cap_filter_seq = 0;
                        s.hdr_highlight_protected_seq = None;
                        s.last_arrival = now;
                        s.starve_released = false;
                        s.source_restart_since = None;
                        if s.in_size != (0, 0) {
                            let stable_size = s.in_size;
                            reset_for_source_resize(&mut gc, s, &metrics, stable_size, stable_size);
                        }
                        log::info!(
                            "WGC in-place reconnect succeeded: hwnd={:#x} elapsed_ms={:.1}; filter chain and overlay preserved",
                            s.hwnd,
                            now.duration_since(reconnect_started).as_secs_f64() * 1000.0
                        );
                    }
                    Err(error) => {
                        log::debug!(
                            "WGC in-place reconnect pending: hwnd={:#x} error={error:#}",
                            s.hwnd
                        );
                    }
                }
            }
            if s.source_restart_since.is_some()
                && now.duration_since(reconnect_started) >= Duration::from_secs(5)
            {
                log::warn!(
                    "WGC in-place reconnect timed out after geometry change: hwnd={:#x}; stopping session",
                    s.hwnd
                );
                stop_session(&mut session, &mut overlay, &mut gc, &status);
                input.release();
                status.lock().unwrap().warning = Some(
                    "画面モードの変更後に映像を再取得できなかったため、拡大を安全に停止しました。"
                        .into(),
                );
            } else {
                overlay.win.pump_messages();
                std::thread::sleep(Duration::from_millis(10));
            }
            continue;
        } else {
            s.source_restart_since = None;
        }

        // FRAME-STARVATION watchdog: a source that stops delivering frames
        // (e.g. a PIP window the player switched to fullscreen — WGC keeps the
        // session "alive" but sends nothing) must never trap the user behind a
        // frozen overlay with a confined cursor. 3s starved → free the cursor
        // and warn; 10s starved → stop the session entirely.
        // Poll the source client rect every tick: an mpv double-click
        // fullscreen toggle (or a PIP promotion) can kill the capture BEFORE a
        // differently-sized frame ever arrives, so frame-based detection alone
        // never arms the watchdog and the overlay freezes on its last frame —
        // the "mystery fullscreen ghost" the user hit twice.
        let pointer_gesture_active = win32::any_mouse_button_down();
        if let Some(rect) = win32::client_rect_on_screen(s.hwnd) {
            if let Some(prev) = s.last_client_rect {
                // v371: moving a window changes x/y every tick but does NOT
                // change the WGC surface geometry. v370 armed the fullscreen
                // watchdog on any rect inequality, then compared a full-window
                // WGC frame against CLIENT dimensions. That guaranteed a false
                // stale-surface restart during ordinary title-bar dragging.
                let size_changed = rect_size_changed(prev, rect);
                if size_changed {
                    s.last_geom_change = Some(Instant::now());
                    log::debug!(
                        "source-geometry-watchdog-armed: hwnd={:#x} old=({},{} {}x{}) new=({},{} {}x{}) pointer_down={}",
                        s.hwnd,
                        prev.0,
                        prev.1,
                        prev.2,
                        prev.3,
                        rect.0,
                        rect.1,
                        rect.2,
                        rect.3,
                        pointer_gesture_active
                    );
                }
            }
            s.last_client_rect = Some(rect);
        }

        let stale_reference = if !s.frame.data.is_empty() {
            source_input_reference_rect(s.hwnd, s.capture_client_only, (s.frame.w, s.frame.h))
        } else {
            None
        };
        let stale_surface_mismatch = stale_reference.is_some_and(|(rect, _, _)| {
            frame_size_mismatches_reference((s.frame.w, s.frame.h), rect)
        });

        // If WGC has already converged to the new native size, the geometry
        // transition is complete.  Disarm immediately; otherwise a static
        // source can trip the unrelated 3-second starvation release later.
        if s.last_geom_change.is_some() && !stale_surface_mismatch {
            if let Some((rect, _, reference)) = stale_reference {
                log::debug!(
                    "source-geometry-watchdog-satisfied: hwnd={:#x} frame={}x{} expected={}x{} reference={} action=disarm",
                    s.hwnd,
                    s.frame.w,
                    s.frame.h,
                    rect.2,
                    rect.3,
                    reference
                );
            }
            s.last_geom_change = None;
            s.starve_released = false;
        }
        if s.last_geom_change
            .is_some_and(|changed| changed.elapsed() >= Duration::from_millis(250))
            && (s.capture_canvas.is_none()
                || s.capture_canvas_suspended_for_fullscreen
                || s.capture_resolution_reapply_pending)
            && s.deferred_capture_resolution.is_none()
            && stale_surface_mismatch
            && !pointer_gesture_active
        {
            let (rect, _, reference) = stale_reference.expect("mismatch requires reference rect");
            log::warn!(
                "source geometry changed but WGC retained stale surface: hwnd={:#x} frame={}x{} expected={}x{} reference={} client_only={} pointer_down=false; restarting WGC in place",
                s.hwnd,
                s.frame.w,
                s.frame.h,
                rect.2,
                rect.3,
                reference,
                s.capture_client_only
            );
            s.display_aspect = (rect.2, rect.3);
            s.last_geom_change = None;
            s.source.stop();
            overlay.win.pump_messages();
            continue;
        } else if stale_surface_mismatch && pointer_gesture_active {
            // Never tear down WGC or release source mouse capture in the middle
            // of a real drag/resize gesture. The size-change watchdog remains
            // armed and is reconsidered after button-up.
            if s.last_geom_change
                .is_some_and(|changed| changed.elapsed() >= Duration::from_millis(250))
            {
                let (rect, _, reference) =
                    stale_reference.expect("mismatch requires reference rect");
                log::debug!(
                    "source-geometry-restart-deferred: hwnd={:#x} frame={}x{} expected={}x{} reference={} reason=pointer-gesture-active",
                    s.hwnd,
                    s.frame.w,
                    s.frame.h,
                    rect.2,
                    rect.3,
                    reference
                );
            }
        }

        // Watchdog is armed ONLY right after a source geometry change (the PIP
        // fullscreen-transition failure mode). WGC delivers frames only when
        // content CHANGES, so "no frames" while reading a static page is
        // normal and must not trigger automatic stop.
        let geom_changed_recently = s
            .last_geom_change
            .map(|t| t.elapsed() < Duration::from_secs(12))
            .unwrap_or(false);
        if geom_changed_recently {
            // ONNX/TensorRT compilation can block the render thread while WGC
            // continues filling its latest-frame slot. Those pending frames
            // prove the source is alive, even if the last processed timestamp
            // is old. Refresh the watchdog instead of stopping a healthy or
            // static source immediately after a long first engine build.
            if s.source.delivered() > s.frame.seq {
                s.last_arrival = Instant::now();
                s.starve_released = false;
            }
            let starved = s.last_arrival.elapsed();
            if starved > Duration::from_secs(10) {
                if overlay.is_visible() && s.last_tex.is_some() {
                    // WGC is change-driven: a valid, already-presented static
                    // frame can legitimately remain silent forever. Keep the
                    // capture active instead of treating inactivity as death.
                    log::info!(
                        "frame starvation >10s with valid visible frame: treating source as static and keeping capture active"
                    );
                    s.last_geom_change = None;
                    s.starve_released = false;
                } else {
                    log::warn!(
                        "frame starvation >10s after geometry change: stopping dead session"
                    );
                    stop_session(&mut session, &mut overlay, &mut gc, &status);
                    input.release();
                    let mut g = status.lock().unwrap();
                    g.warning = Some("ソース変形後にフレームが届かないため停止しました".into());
                    continue;
                }
            } else if starved > Duration::from_secs(3)
                && !s.starve_released
                && !pointer_gesture_active
            {
                log::warn!("frame starvation >3s after geometry change: releasing cursor");
                input.release();
                s.starve_released = true;
            }
        }
        let phase_watchdog_finished = Instant::now();

        // keep the source directly BELOW the overlay in the topmost band:
        // clicks must always land on it, but the GUI/panel float above
        let overlay_hwnd = overlay.hwnd().0 as isize;
        if !win32::window_is_above(overlay_hwnd, s.hwnd) {
            win32::place_below(s.hwnd, overlay_hwnd);
        }
        let phase_zorder_finished = Instant::now();

        let current_monitor_rect = win32::monitor_rect_of(s.hwnd);
        if current_monitor_rect != s.monitor_rect {
            s.monitor_rect = current_monitor_rect;
            s.monitor_refresh_hz = win32::monitor_refresh_hz(s.hwnd);
            s.smooth_pacer.reset();
            s.paced_present_deadline = None;
            s.flow_output_cadence.reset();
            metrics.set_monitor(
                (current_monitor_rect.2, current_monitor_rect.3),
                s.monitor_refresh_hz,
            );
            log::info!(
                "capture monitor changed: {}x{} @ {:.2}Hz",
                current_monitor_rect.2,
                current_monitor_rect.3,
                s.monitor_refresh_hz.unwrap_or(0.0)
            );
        }

        // Windowed source-titlebar drag: input.rs classifies the physical
        // caption gesture and owns the fixed visual grab anchor. The follower
        // only mirrors the hidden source HWND position into the overlay. Its
        // coordinate family and generation are frozen for the whole gesture.
        if s.mode == ScaleMode::Fixed && s.placed {
            if let Some(client_rect @ (cx, cy, _cw, _ch)) = win32::client_rect_on_screen(s.hwnd) {
                let caption_drag = crate::input::source_caption_drag_active();
                let current_drag_epoch = if caption_drag {
                    Some(crate::input::source_caption_drag_epoch())
                } else {
                    None
                };

                // A release/re-press can happen faster than one render loop.
                // If we missed the brief inactive state, the epoch still makes
                // the generation change explicit: retire the old follower now
                // and allow the new generation to start in this same tick.
                if caption_drag && s.source_drag_active && drag_session_epoch != current_drag_epoch
                {
                    log::info!(
                        "source-drag-generation-rollover: previous={:?} current={:?} action=restart-follower",
                        drag_session_epoch,
                        current_drag_epoch
                    );
                    stop_drag_follower(&mut drag_follower);
                    s.source_drag_active = false;
                    drag_session_epoch = None;
                }

                if !caption_drag && s.source_drag_active {
                    s.source_drag_active = false;
                    stop_drag_follower(&mut drag_follower);
                    drag_session_epoch = None;
                    let before = win32::window_rect(s.hwnd);
                    let rebased = keep_source_window_reachable(s.hwnd);
                    let after_client = win32::client_rect_on_screen(s.hwnd)
                        .map(|(x, y, _, _)| (x, y))
                        .or(rebased)
                        .unwrap_or((cx, cy));
                    s.last_src_pos = Some(after_client);
                    log::info!(
                        "source-drag-session-end: owner=anchored-high-rate-source-follower rebase_once=true before={before:?} client_after={after_client:?}"
                    );
                } else {
                    let drag_reference =
                        if caption_drag {
                            source_input_reference_rect(s.hwnd, false, s.display_aspect)
                                .unwrap_or((client_rect, true, "client-fallback"))
                        } else {
                            (client_rect, false, "client")
                        };
                    let ((sx, sy, sw, sh), _window_frame, reference_kind) = drag_reference;

                    if caption_drag && !s.source_drag_active {
                        s.source_drag_active = true;
                        let overlay_start = overlay.current_rect();
                        let panel_start = if panel_hwnd != 0 && panel_visible {
                            win32::window_rect(panel_hwnd)
                        } else {
                            None
                        };
                        let drag_epoch = crate::input::source_caption_drag_epoch();
                        drag_session_epoch = Some(drag_epoch);
                        drag_follower = DragFollower::start(
                            s.hwnd,
                            overlay.hwnd().0 as isize,
                            if panel_visible { panel_hwnd } else { 0 },
                            (sx, sy, sw, sh),
                            reference_kind,
                            overlay_start,
                            panel_start,
                            s.monitor_rect,
                            drag_epoch,
                        );
                        // The first fallback delta must use the same coordinate
                        // family as the follower, not a previous client-space
                        // snapshot.
                        s.last_src_pos = Some((sx, sy));
                        log::info!(
                            "source-drag-session-begin: epoch={} source=({sx},{sy}) size={}x{} reference={} owner={} caption=true poll_ms=8 anchor=fixed",
                            drag_epoch,
                            sw,
                            sh,
                            reference_kind,
                            if drag_follower.is_some() {
                                "anchored-high-rate-source-follower"
                            } else {
                                "engine-follow-fallback"
                            }
                        );
                    }

                    if drag_follower
                        .as_ref()
                        .is_some_and(|follower| follower.finished())
                    {
                        stop_drag_follower(&mut drag_follower);
                    }

                    if let Some((lx, ly)) = s.last_src_pos {
                        let (dx, dy) = (sx - lx, sy - ly);
                        if dx != 0 || dy != 0 {
                            if caption_drag {
                                if drag_follower.is_none() {
                                    // Thread creation/follower loss only:
                                    // preserve operability with the v377 engine
                                    // follower. The visual cursor remains locked
                                    // by input.rs even on this fallback path.
                                    let drag_epoch = crate::input::source_caption_drag_epoch();
                                    let committed =
                                        crate::input::commit_source_caption_drag_overlay_update(
                                            drag_epoch,
                                            || {
                                                let before = overlay.current_rect();
                                                move_windowed_overlay_by_source_delta(
                                                    &mut overlay,
                                                    dx,
                                                    dy,
                                                    gui_priority_topmost,
                                                );
                                                let after = overlay.current_rect();
                                                if (after.0, after.1) != (before.0, before.1) {
                                                    Some((after.0, after.1))
                                                } else {
                                                    None
                                                }
                                            },
                                        );
                                    if committed {
                                        log::debug!(
                                            "source-drag-follow: delta=({dx},{dy}) owner=engine-follow-fallback anchor=fixed epoch={drag_epoch}"
                                        );
                                    }
                                } else if dx.abs() > 256 || dy.abs() > 256 {
                                    log::debug!(
                                        "source-drag-observed-large-delta: delta=({dx},{dy}) owner=anchored-high-rate-source-follower"
                                    );
                                }
                            } else if s.capture_client_only && win32::left_mouse_button_down() {
                                // GetAsyncKeyState alone is not proof that this source owns
                                // the gesture. A LEFT hold can have started on Neo's GUI or
                                // desktop, and failed/off-desktop cursor handoffs can release
                                // source ownership while the physical button is still down.
                                let source_owned = crate::input::source_client_drag_owns(s.hwnd);
                                if !source_owned {
                                    // Always refresh last_src_pos below. This intentionally
                                    // drops the unowned delta instead of banking it and
                                    // applying a large catch-up move if ownership returns.
                                    if s.source_client_drag_active
                                        || dx.abs() >= 64
                                        || dy.abs() >= 64
                                    {
                                        log::debug!(
                                            "source-client-drag-follow-suppressed: delta=({dx},{dy}) physical_left=true source_owned=false hwnd={:#x} baseline=refresh",
                                            s.hwnd
                                        );
                                    }
                                    s.source_client_drag_raw_origin = None;
                                    s.source_client_drag_overlay_origin = None;
                                    // Do not fight the native move loop while LEFT is down,
                                    // but remember if this unowned movement has made the
                                    // hidden client unreachable.  The old v423 behavior
                                    // rebased after an owned drag; provider/input transitions
                                    // can now clear ownership before Chromium finishes moving.
                                    if client_rect_rebase_delta((sx, sy, sw, sh), s.monitor_rect)
                                        != (0, 0)
                                    {
                                        if !s.source_client_unowned_rebase_pending {
                                            log::warn!(
                                                "source-client-unowned-offscreen: hwnd={:#x} client=({sx},{sy},{sw}x{sh}) physical_left=true source_owned=false action=rebase-on-release",
                                                s.hwnd
                                            );
                                        }
                                        s.source_client_unowned_rebase_pending = true;
                                    }
                                } else {
                                    // A native client-area position change proves that this
                                    // source-owned LEFT gesture is actually moving the custom
                                    // frame (Chromium PIP etc.), rather than clicking a control.
                                    // From this point onward the source HWND is NOT a valid
                                    // visual movement authority: Windows/Chromium can stop it
                                    // at an off-screen retention boundary while raw mouse
                                    // coordinates continue. Arm one absolute raw-coordinate
                                    // anchor and let the follower below own visual movement.
                                    if !s.source_client_drag_active {
                                        s.source_client_drag_active = true;
                                        let before = overlay.current_rect();
                                        if let Some((raw_origin, raw_current)) =
                                            crate::input::source_client_drag_raw_span(s.hwnd)
                                        {
                                            s.source_client_drag_raw_origin = Some(raw_origin);
                                            s.source_client_drag_overlay_origin =
                                                Some((before.0, before.1));
                                            log::info!(
                                                "source-client-drag-session-begin: hwnd={:#x} owner=input-source-left-hold visual_owner=raw-hook-anchor raw_origin=({},{}) raw_current=({},{}) source_delta=({dx},{dy}) clamp=none",
                                                s.hwnd,
                                                raw_origin.0,
                                                raw_origin.1,
                                                raw_current.0,
                                                raw_current.1
                                            );
                                        } else {
                                            // Extremely narrow fallback (e.g. an ownership
                                            // transition raced this geometry sample). Preserve
                                            // one-frame operability with the already-vetted
                                            // native delta, but do not create a permanent
                                            // source-delta follower. A later raw sample will
                                            // re-anchor without applying banked movement.
                                            let nx = before.0.saturating_add(dx);
                                            let ny = before.1.saturating_add(dy);
                                            if (nx, ny) != (before.0, before.1) {
                                                reposition_overlay_for_gui_mode(
                                                    &mut overlay, nx, ny, before.2, before.3, gui_priority_topmost,
                                                );
                                            }
                                            log::warn!(
                                                "source-client-drag-session-begin: hwnd={:#x} raw_anchor=unavailable fallback=native-delta-once delta=({dx},{dy})",
                                                s.hwnd
                                            );
                                        }
                                    }
                                    // Do not mirror dx/dy here. The absolute raw-hook follower
                                    // below runs even when source geometry does not change,
                                    // which is exactly the case at Chromium's off-screen wall.
                                }
                            } else {
                                log::info!(
                                    "source moved without caption drag: delta=({dx},{dy}); keep overlay independent"
                                );
                            }
                        }
                    }
                    s.last_src_pos = Some((sx, sy));

                    // v426: once a client-only native move has identified a
                    // window-drag gesture, the visible overlay follows the raw
                    // WH_MOUSE_LL down->current span, not source HWND deltas.
                    // This follower runs every engine tick even when sx/sy are
                    // unchanged, so an OS-clamped hidden PIP cannot create a
                    // visible movement wall. Absolute-from-origin positioning
                    // also coalesces interpolation/render stalls without
                    // accumulating delayed per-frame deltas or overshoot.
                    if !caption_drag
                        && s.capture_client_only
                        && s.source_client_drag_active
                        && win32::left_mouse_button_down()
                    {
                        let source_owned = crate::input::source_client_drag_owns(s.hwnd);
                        if source_owned {
                            if let Some((published_origin, raw_current)) =
                                crate::input::source_client_drag_raw_span(s.hwnd)
                            {
                                // If the raw generation changed unexpectedly,
                                // re-anchor at the current overlay instead of
                                // mixing two gestures and creating a jump.
                                if s.source_client_drag_raw_origin != Some(published_origin)
                                    || s.source_client_drag_overlay_origin.is_none()
                                {
                                    let cur = overlay.current_rect();
                                    let raw_dx = raw_current.0.saturating_sub(published_origin.0);
                                    let raw_dy = raw_current.1.saturating_sub(published_origin.1);
                                    // Preserve the current visual position while adopting the
                                    // newly published generation.  Back-solving the overlay
                                    // origin prevents the next tick from replaying movement
                                    // that already happened before this refresh.
                                    let rebased_overlay_origin = (
                                        cur.0.saturating_sub(raw_dx),
                                        cur.1.saturating_sub(raw_dy),
                                    );
                                    s.source_client_drag_raw_origin = Some(published_origin);
                                    s.source_client_drag_overlay_origin =
                                        Some(rebased_overlay_origin);
                                    log::debug!(
                                        "source-client-drag-raw-anchor-refresh: hwnd={:#x} published_origin=({},{}) current=({},{}) overlay=({},{}) rebased_overlay_origin=({},{}) reason=generation-or-anchor-mismatch",
                                        s.hwnd,
                                        published_origin.0,
                                        published_origin.1,
                                        raw_current.0,
                                        raw_current.1,
                                        cur.0,
                                        cur.1,
                                        rebased_overlay_origin.0,
                                        rebased_overlay_origin.1
                                    );
                                }
                                if let (Some(raw_origin), Some(overlay_origin)) = (
                                    s.source_client_drag_raw_origin,
                                    s.source_client_drag_overlay_origin,
                                ) {
                                    let (nx, ny) = client_drag_overlay_position_from_raw(
                                        overlay_origin,
                                        raw_origin,
                                        raw_current,
                                    );
                                    let before = overlay.current_rect();
                                    if (nx, ny) != (before.0, before.1) {
                                        reposition_overlay_for_gui_mode(
                                            &mut overlay, nx, ny, before.2, before.3, gui_priority_topmost,
                                        );
                                        let after = overlay.current_rect();
                                        log::debug!(
                                            "source-client-drag-follow: raw_delta=({},{}) input=client left_down=true source_owned=true overlay=({},{}) -> ({},{}) owner=raw-hook-anchor native_source=({sx},{sy}) clamp=none",
                                            raw_current.0.saturating_sub(raw_origin.0),
                                            raw_current.1.saturating_sub(raw_origin.1),
                                            before.0,
                                            before.1,
                                            after.0,
                                            after.1
                                        );
                                    }
                                }
                            }
                        } else {
                            // Ownership cannot return during the same physical
                            // LEFT hold. Drop visual anchors immediately; the
                            // button-up path still rebases the hidden source.
                            s.source_client_drag_raw_origin = None;
                            s.source_client_drag_overlay_origin = None;
                        }
                    }

                    // Client-only PIP/custom-frame drags are not native caption
                    // drags, so the caption-drag end path above never rebases
                    // them. Once the physical button is released, restore the
                    // hidden source to a fully reachable monitor position while
                    // deliberately leaving the visible overlay where the user
                    // dropped it. This preserves unrestricted overlay movement
                    // and prevents ClipCursor/SetCursorPos from targeting source
                    // pixels beyond the desktop (the observed lower/right wall).
                    if !caption_drag
                        && (s.source_client_drag_active || s.source_client_unowned_rebase_pending)
                        && !win32::left_mouse_button_down()
                    {
                        let recovery_reason = if s.source_client_drag_active {
                            "owned-client-drag-end"
                        } else {
                            "unowned-offscreen-move-end"
                        };
                        let before = win32::window_rect(s.hwnd);
                        let rebased = keep_source_window_reachable(s.hwnd);
                        let after_client = win32::client_rect_on_screen(s.hwnd)
                            .map(|(x, y, _, _)| (x, y))
                            .or(rebased)
                            .unwrap_or((sx, sy));
                        s.last_src_pos = Some(after_client);
                        s.source_client_drag_active = false;
                        s.source_client_unowned_rebase_pending = false;
                        s.source_client_drag_raw_origin = None;
                        s.source_client_drag_overlay_origin = None;
                        log::info!(
                            "source-client-drag-session-end: rebase_once=true reason={recovery_reason} before={before:?} client_after={after_client:?} overlay_preserved=true"
                        );
                    }
                }
            }
        } else {
            s.source_drag_active = false;
            s.source_client_drag_active = false;
            s.source_client_unowned_rebase_pending = false;
            s.source_client_drag_raw_origin = None;
            s.source_client_drag_overlay_origin = None;
            stop_drag_follower(&mut drag_follower);
            drag_session_epoch = None;
            if let Some((sx, sy, _, _)) = win32::client_rect_on_screen(s.hwnd) {
                s.last_src_pos = Some((sx, sy));
            }
        }

        // follow source geometry (windowed mode: user may have moved the
        // overlay -> keep its position, only track size)
        let rect = overlay_geometry(s, &overlay);
        let cur = overlay.current_rect();
        if rect != cur {
            reposition_overlay_for_gui_mode(
                &mut overlay, rect.0, rect.1, rect.2, rect.3, gui_priority_topmost,
            );
            enforce_gui_priority(
                gui_priority_hwnd,
                gui_priority_topmost,
                panel_hwnd,
                overlay.hwnd().0 as isize,
            );
        }
        // A hard Stop physically removes the persistent WGL HWND from DWM to
        // guarantee that no frozen frame survives. Reinsert it here at alpha=0
        // as soon as the next session has restored the correct geometry. This
        // gives DWM the same pre-existing transparent overlay tree that v235
        // used, instead of introducing a fullscreen surface only at reveal.
        if !overlay.is_visible() {
            overlay.prepare_hidden_for_reveal();
        }
        let rect = overlay.current_rect();
        s.placed = true;
        status.lock().unwrap().overlay_rect = rect;
        let phase_geometry_finished = Instant::now();

        // ---- control panel: reposition EVERY tick, glued to the overlay
        // (old-tool behaviour: windowed = above the view's left edge,
        // fullscreen = fixed top-centre INSIDE the content so the confined
        // cursor can always reach it) ----
        let content_now = {
            let (fw, fh) = if s.display_aspect.0 > 0 {
                s.display_aspect
            } else {
                (rect.2, rect.3)
            };
            crate::input::content_rect(
                crate::input::Rect {
                    x: rect.0,
                    y: rect.1,
                    w: rect.2,
                    h: rect.3,
                },
                fw,
                fh,
            )
        };
        status.lock().unwrap().content_rect =
            (content_now.x, content_now.y, content_now.w, content_now.h);
        let mut panel_no_engage: Option<(i32, i32, i32, i32)> = None;
        if panel_hwnd != 0 && panel_visible {
            let stale_chip_suppressed =
                panel_chipped && panel_lurk_restore_geometry_guard_active();
            let geometry_chipped = panel_chipped && !stale_chip_suppressed;
            let (pw, ph) = if geometry_chipped { panel_chip } else { panel_bar };
            let (px, py) = panel_target_position(
                s.mode,
                rect,
                (content_now.x, content_now.y, content_now.w, content_now.h),
                (pw, ph),
            );
            let target_panel_rect = (px, py, pw, ph);
            let current_panel_rect = win32::window_rect(panel_hwnd);
            let panel_moved = current_panel_rect != Some(target_panel_rect);
            let panel_rect = if panel_moved {
                win32::set_window_rect(panel_hwnd, px, py, pw, ph);
                target_panel_rect
            } else {
                current_panel_rect.unwrap_or(target_panel_rect)
            };
            if panel_moved || last_panel_raise_check.elapsed() >= Duration::from_millis(200) {
                let overlay_hwnd = overlay.hwnd().0 as isize;
                if gui_priority_topmost {
                    // Modern GUI-topmost path: maintain one deterministic stack
                    // to avoid GUI/panel flashing while either window moves.
                    enforce_gui_priority(
                        gui_priority_hwnd,
                        true,
                        panel_hwnd,
                        overlay_hwnd,
                    );
                } else {
                    // Exact v235 GUI-topmost-OFF panel contract: GUI priority is
                    // completely inactive; only repair the panel if USER32 says
                    // it actually fell below the overlay.
                    if !win32::window_is_above(panel_hwnd, overlay_hwnd) {
                        win32::raise_topmost(panel_hwnd);
                        crate::input::keep_cursor_sprite_on_top();
                    }
                }
                last_panel_raise_check = Instant::now();
            }
            panel_no_engage = Some(panel_rect);
        }

        if crate::logging::diagnostics_enabled()
            && last_source_window_poll.elapsed() >= Duration::from_millis(250)
        {
            last_source_window_poll = Instant::now();
            let overlay_hwnd = overlay.hwnd().0 as isize;
            let cursor_hwnd = crate::input::cursor_sprite_hwnd();
            let source_windows = win32::visible_top_level_windows_for_pid(
                win32::window_pid(s.hwnd),
                12,
            );
            if source_windows != last_source_visible_windows {
                let had_source_popup = last_source_visible_windows.len() > 1;
                let has_source_popup = source_windows.len() > 1;
                win32::log_helper_compositor_snapshot(
                    "source-window-set-change",
                    gui_priority_hwnd,
                    panel_hwnd,
                    overlay_hwnd,
                    cursor_hwnd,
                    s.hwnd,
                );
                // Desktop-DC readback can block AMD/DWM for hundreds of ms.
                // Sample pixels only when the source crosses the important
                // no-popup <-> popup boundary (mpv #32768 appears/disappears),
                // not for startup or every submenu mutation.
                if had_source_popup != has_source_popup {
                    win32::log_helper_physical_visibility(
                        if has_source_popup {
                            "source-popup-enter"
                        } else {
                            "source-popup-exit"
                        },
                        gui_priority_topmost,
                        panel_hwnd,
                        overlay_hwnd,
                        cursor_hwnd,
                    );
                }
                last_source_visible_windows = source_windows;
                last_helper_compositor_diag = Instant::now();
            } else if last_helper_compositor_diag.elapsed() >= Duration::from_secs(2) {
                win32::log_helper_compositor_snapshot(
                    "periodic-running",
                    gui_priority_hwnd,
                    panel_hwnd,
                    overlay_hwnd,
                    cursor_hwnd,
                    s.hwnd,
                );
                last_helper_compositor_diag = Instant::now();
            }
        }
        // Desktop-DC visibility sampling is deliberately popup-boundary only.
        // On the RX 9060 XT it can block for ~0.5 s and itself perturb DWM.
        let phase_panel_finished = Instant::now();

        // Native windows that are actually above the magnified overlay are
        // represented as ordinary no-engage zones below. This reuses the
        // mature input handoff path and keeps cursor ownership in one state
        // machine. Do not run a second engine-side Task Manager owner latch:
        // point sampling and the real handoff could race, pulling the cursor
        // away from a source scrollbar after the virtual pointer had moved.
        let overlay_hwnd = overlay.hwnd().0 as isize;

        let mut phase_input_started = Instant::now();
        let mut phase_input_finished = phase_input_started;
        // Cursor engage geometry must use the SAME coordinate space as the
        // visible WGC frame. Previously this always used client_rect even when
        // title-bar removal was OFF, so a full 1900x1072 frame was mapped into
        // a 1896x1011 client rect: title-bar clicks landed in the client area,
        // source-window dragging failed, and edge mapping was vertically skewed.
        let frame_hint = if s.display_aspect.0 > 0 {
            s.display_aspect
        } else {
            s.in_size
        };
        if let Some((src, window_frame_input, input_reference)) =
            source_input_reference_rect(s.hwnd, s.capture_client_only, frame_hint)
        {
            let (fw, fh) = if s.display_aspect.0 > 0 {
                s.display_aspect
            } else {
                (src.2, src.3)
            };
            let content = crate::input::content_rect(
                crate::input::Rect {
                    x: rect.0,
                    y: rect.1,
                    w: rect.2,
                    h: rect.3,
                },
                fw,
                fh,
            );
            let mut no_engage: Vec<crate::input::NoEngageRect> = no_engage_rects
                .iter()
                .map(|&(px, py, pw, ph, hwnd)| {
                    crate::input::NoEngageRect::new(crate::input::Rect {
                        x: px,
                        y: py,
                        w: pw,
                        h: ph,
                    })
                    .with_hwnd(hwnd)
                })
                .collect();
            if let Some((px, py, pw, ph)) = panel_no_engage {
                no_engage.push(
                    crate::input::panel_no_engage_rect(crate::input::Rect {
                        x: px,
                        y: py,
                        w: pw,
                        h: ph,
                    })
                    .with_hwnd(panel_hwnd),
                );
            }
            for (x, y, w, h, hwnd) in
                win32::external_window_rects_above_overlay(overlay_hwnd, s.hwnd)
            {
                no_engage.push(
                    crate::input::NoEngageRect::new(crate::input::Rect { x, y, w, h })
                        .with_hwnd(hwnd),
                );
            }
            phase_input_started = Instant::now();
            let capture_geometry_ready = s.pending_resize_size.is_none()
                && !(s.capture_resolution_applied && s.deferred_capture_resolution.is_some());
            let input_geometry_ready = capture_geometry_ready
                && s.input_reenable_after
                    .is_none_or(|deadline| Instant::now() >= deadline);
            if input_geometry_ready {
                s.input_reenable_after = None;
            }
            s.source_input_geometry_missing_logged = false;
            input.configure(
                input_geometry_ready && overlay.is_visible(),
                matches!(s.mode, ScaleMode::Auto),
                window_frame_input,
                input_reference,
                crate::input::Rect {
                    x: rect.0,
                    y: rect.1,
                    w: rect.2,
                    h: rect.3,
                },
                overlay_hwnd,
                content,
                crate::input::Rect {
                    x: src.0,
                    y: src.1,
                    w: src.2,
                    h: src.3,
                },
                s.hwnd,
                no_engage,
                input_autohide_secs,
                // Cursor-speed compensation: slow the OS pointer by 1/zoom while engaged
                // so the sprite tracks the hand 1:1 (seamless edge crossing).
                input_speed_fix,
            );
            phase_input_finished = Instant::now();
        } else {
            // Never retain the previous source clip when a PIP changes
            // owner/size or is destroyed between two Win32 geometry queries.
            input.release();
            s.input_reenable_after = Some(Instant::now() + Duration::from_millis(120));
            if !s.source_input_geometry_missing_logged {
                log::warn!(
                    "source-input-geometry-unavailable: hwnd={:#x}; cursor mapping released",
                    s.hwnd
                );
                s.source_input_geometry_missing_logged = true;
            }
        }

        // GUI responsiveness must never split, flush, finish or sleep
        // the normal GPU filter path. Low-spec relief is GUI-thread-only.

        // Advance the GL -> external-compute ownership handoff without ever
        // waiting on the render thread. Blocking here before the first job
        // reached the worker, which froze video, input and Stop/Shutdown.
        if let Some((fence, elapsed)) = s
            .gpu_interp_pack_pending
            .as_ref()
            .map(|pack| (pack.fence, pack.started.elapsed()))
        {
            match gc.poll_commands_fence(fence) {
                Ok(true) => {
                    let pack = s.gpu_interp_pack_pending.take().expect("GPU pack pending");
                    if pack.pending.pair_id < 3 {
                        log::info!(
                            "interp-gpu-input-pack-complete: pair={} generation={} elapsed_ms={:.2}",
                            pack.pending.pair_id,
                            pack.pending.generation,
                            elapsed.as_secs_f64() * 1000.0
                        );
                    }
                    if s.gpu_interp_worker.is_none() {
                        s.gpu_interp_worker =
                            Some(GpuInterpWorker::spawn(pack.pending.stage.clone()));
                    }
                    let cooperative_slots = pack.job.cooperative_slots;
                    match s
                        .gpu_interp_worker
                        .as_ref()
                        .expect("GPU worker")
                        .submit(pack.job)
                    {
                        Ok(()) => {
                            let mut pending = pack.pending;
                            pending.submitted_at = Instant::now();
                            if pending.pair_id < 3 {
                                log::info!(
                                    "interp-gpu-job-submitted: pair={} generation={} outputs={} scheduling={}",
                                    pending.pair_id,
                                    pending.generation,
                                    pending.timesteps.len(),
                                    if pending.permit_next_after_post_submit {
                                        "cooperative-after-post-submit"
                                    } else if cooperative_slots {
                                        "cooperative-slots"
                                    } else {
                                        "streamed"
                                    }
                                );
                            }
                            s.gpu_interp_pending = Some(pending);
                        }
                        Err(error) => {
                            let reason = match error {
                                TrySendError::Full(_) => "GPU interpolation worker queue full",
                                TrySendError::Disconnected(_) => {
                                    "GPU interpolation worker disconnected"
                                }
                            };
                            log::error!("interp-gpu-submit-error: reason={reason}");
                            pack.pending
                                .stage
                                .lock()
                                .unwrap()
                                .disable_interp_gpu_path(reason);
                            while let Some(old) = s.gpu_interp_hist.pop_front() {
                                gc.recycle(old.tex);
                            }
                        }
                    }
                }
                Ok(false) if elapsed >= Duration::from_millis(250) => {
                    gc.cancel_commands_fence(fence);
                    let pack = s.gpu_interp_pack_pending.take().expect("GPU pack pending");
                    let reason = "OpenGL interpolation input handoff timed out after 250ms";
                    log::error!(
                        "interp-gpu-timeout: pair={} generation={} state=gl-input-pack elapsed_ms={:.2} action=poison-and-continue",
                        pack.pending.pair_id,
                        pack.pending.generation,
                        elapsed.as_secs_f64() * 1000.0
                    );
                    pack.pending
                        .stage
                        .lock()
                        .unwrap()
                        .disable_interp_gpu_path(reason);
                    while let Some(old) = s.gpu_interp_hist.pop_front() {
                        gc.recycle(old.tex);
                    }
                }
                Ok(false) => {}
                Err(error) => {
                    let pack = s.gpu_interp_pack_pending.take().expect("GPU pack pending");
                    log::error!(
                        "interp-gpu-fence-error: pair={} generation={} reason={}",
                        pack.pending.pair_id,
                        pack.pending.generation,
                        error
                    );
                    pack.pending
                        .stage
                        .lock()
                        .unwrap()
                        .disable_interp_gpu_path(&error);
                    while let Some(old) = s.gpu_interp_hist.pop_front() {
                        gc.recycle(old.tex);
                    }
                }
            }
        }

        // Stream each genuinely interpolated timestep as soon as that unique
        // output slot is ready. This overlaps later RIFE inference with the
        // presentation/post-chain work of earlier slots instead of serialising
        // the whole x4/x5 batch before the first displayed midpoint.
        let gpu_path_was_active = s.gpu_interp_active_logged;
        let gpu_stream_progressed = drain_gpu_interp_stream(
            &mut gc,
            &mut overlay,
            s,
            &metrics,
            &status,
            vsync_on,
            downscaler,
        );
        if s.provider_transition_input_suspended
            && !gpu_path_was_active
            && s.gpu_interp_active_logged
        {
            resume_provider_transition_after_interpolated_present(&input, s, "gpu-resident");
        }
        // Absolute safety cap: never leave input intentionally suspended if a
        // provider fails to produce a frame. The mapping remains released, so
        // resuming only permits a later healthy configure tick to re-engage.
        if s.provider_transition_input_suspended
            && s.provider_transition_input_suspended_since
                .is_some_and(|since| since.elapsed() > Duration::from_secs(60))
        {
            log::warn!(
                "provider-transition-timeout: no presented frame within 60s; input suspension released"
            );
            input.set_transition_suspended(false);
            s.provider_transition_input_suspended = false;
            s.provider_transition_input_suspended_since = None;
            s.input_reenable_after = Some(Instant::now() + Duration::from_millis(250));
        }
        if gpu_stream_progressed {
            overlay.win.pump_messages();
            continue;
        }

        // While a GPU interpolation pair is in flight, keep the last fully
        // filtered image on screen. Never dequeue/present an unfiltered live
        // replacement just because a TensorRT invocation is slow: preset
        // editing must show exactly the selected filter state. The bounded
        // result poll above keeps Win32/Stop responsive.
        if s.gpu_interp_pack_pending.is_some() || s.gpu_interp_pending.is_some() {
            if let Some(pending) = s.gpu_interp_pending.as_mut() {
                let elapsed = pending.submitted_at.elapsed();
                if !pending.slow_wait_logged && elapsed >= Duration::from_millis(250) {
                    pending.slow_wait_logged = true;
                    log::warn!(
                        "interp-gpu-pending-slow: pair={} generation={} elapsed_ms={:.2} render_thread_alive=true policy=hold-last-filtered-frame",
                        pending.pair_id,
                        pending.generation,
                        elapsed.as_secs_f64() * 1000.0
                    );
                }
            }
            overlay.win.pump_messages();
            std::thread::yield_now();
            continue;
        }

        let flow_active = matches!(
            s.chain.interp_stage(),
            Some(crate::render::chain::InterpHandle::Flow { .. })
        );
        // Do not force the no-drop queue onto ONNX interpolation;
        // (RIFE) made fps and latency WORSE in the field — RIFE ran fine on
        // take_latest. The queue stays NeoFlow-only as originally designed;
        // the DRBA-specific fps deficit is tracked separately.
        let interp_active = s.chain.has_interp();
        let neodeint_active = s.chain.has_neodeint();
        // A one-frame queue is useful only when interpolation needs adjacent
        // source frames.  With plain upscaling its max depth of one is
        // semantically identical to newest-frame sampling, but transfers
        // ownership of a full RGBA allocation on every callback.  On the
        // 9060 XT test system that raised callback time from 2.39 to 3.55 ms
        // and reduced capture from 57.08 to 55.21 fps.  Smooth presentation
        // is now handled independently by the gap-pacing path below.
        let load_reduction_lookahead =
            s.duplicate_frame_reduction && s.smooth_pacing && !interp_active;
        // NeoDeint must react to the current frame. Its previous one-frame
        // look-ahead delayed activation and preferentially selected the
        // clean-looking half of an alternating woven cadence.
        let temporal_lookahead = load_reduction_lookahead;
        let queued_capture = interp_active || temporal_lookahead;
        s.source.set_queue_enabled(queued_capture);
        let phase_capture_started = Instant::now();

        // A source may enter its own fullscreen presentation after capture has
        // already started. Capture size is a windowed-source policy; forcing
        // the old client canvas onto a monitor-covering WGC surface produces a
        // zoom/crop-looking view. Let WGC's live fullscreen geometry own the
        // session while keeping the user's requested size reserved for the
        // return to windowed mode.
        let live_monitor_fullscreen = win32::is_monitor_fullscreen(s.hwnd);
        if !s.source_live_fullscreen && live_monitor_fullscreen {
            let monitor_size = (s.monitor_rect.2, s.monitor_rect.3);
            let live_presentation_signature = win32::fullscreen_presentation_signature(s.hwnd);
            let recent_geometry_change = s
                .last_geom_change
                .is_some_and(|changed| changed.elapsed() < FULLSCREEN_ENTER_GEOMETRY_GRACE);
            let capture_resize_intent = status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .capture_resolution_resize_intent;
            let source_owned_fullscreen = should_enter_live_fullscreen(
                s.source_monitor_fullscreen,
                live_monitor_fullscreen,
                s.capture_canvas,
                monitor_size,
                s.source_presentation_signature,
                live_presentation_signature,
                capture_resize_intent,
                recent_geometry_change,
            );
            // Do not equate monitor-sized pixels with fullscreen state. Neo can
            // intentionally make a normal PIP/mpv client 1920x1080 on a 1080p
            // display. Only source-owned presentation evidence may suspend the
            // capture-resolution option.
            if source_owned_fullscreen {
                s.source_live_fullscreen = true;
                s.fullscreen_exit_candidate_since = None;
                let suspended_target = s.capture_canvas;
                if suspended_target.is_some() {
                    s.capture_canvas_suspended_for_fullscreen = true;
                    s.capture_resolution_reapply_pending = false;
                    s.deferred_capture_resolution = None;
                    s.capture_resolution_applied = false;
                    s.capture_resolution_wait_started = None;
                    s.pending_resize_size = None;
                    input.release();
                }
                {
                    let mut st = status
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    st.source_live_fullscreen = true;
                    if suspended_target.is_some() {
                        st.capture_resolution_fullscreen_notice_seq = st
                            .capture_resolution_fullscreen_notice_seq
                            .saturating_add(1);
                    }
                }
                if let Some(target) = suspended_target {
                    log::info!(
                        "source-fullscreen-live-enter: hwnd={:#x} capture_canvas={}x{} monitor={}x{} action=suspend-capture-resolution follow=wgc-native",
                        s.hwnd,
                        target.0,
                        target.1,
                        monitor_size.0,
                        monitor_size.1
                    );
                } else {
                    log::info!(
                        "source-fullscreen-live-enter: hwnd={:#x} capture_canvas=auto monitor={}x{} action=follow-wgc-native",
                        s.hwnd,
                        monitor_size.0,
                        monitor_size.1
                    );
                }
            }
        } else if s.source_live_fullscreen {
            if live_monitor_fullscreen {
                s.fullscreen_exit_candidate_since = None;
            } else {
                let exit_since = s
                    .fullscreen_exit_candidate_since
                    .get_or_insert_with(Instant::now);
                if exit_since.elapsed() >= Duration::from_millis(250) {
                    s.source_live_fullscreen = false;
                    s.fullscreen_exit_candidate_since = None;
                    s.capture_resolution_reapply_pending = s.capture_canvas.is_some();
                    {
                        let mut st = status
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        st.source_live_fullscreen = false;
                        st.capture_resolution_reapply_seq =
                            st.capture_resolution_reapply_seq.saturating_add(1);
                    }
                    log::info!(
                        "source-fullscreen-live-exit: hwnd={:#x} action=windowed-stable capture_resolution_reapply_pending={}",
                        s.hwnd,
                        s.capture_resolution_reapply_pending
                    );
                }
            }
        }

        // In smooth, non-interpolated playback, wake before the next display
        // slot.  A fixed 20 ms wait limited the recovery path itself to 50 Hz
        // whenever WGC omitted one 16.7 ms notification.
        // Smooth presentation already performs the long wait at its output
        // deadlines. A second 20 ms WGC wait made 30p interpolation miss a
        // just-arrived frame and alternate between roughly 33/42 ms input
        // gaps, accumulating and dropping queued frames.
        let capture_wait = if s.smooth_pacing && s.last_tex.is_some() {
            Duration::from_millis(2)
        } else {
            Duration::from_millis(20)
        };
        let mut got_new = s.source.wait_new(capture_wait);
        let lookahead_ready = !temporal_lookahead
            || s.source.queued_len() >= 2
            || s.duplicate_lookahead_wait
                .is_some_and(|started| started.elapsed() >= Duration::from_millis(40));
        if temporal_lookahead && !lookahead_ready {
            s.duplicate_lookahead_wait.get_or_insert_with(Instant::now);
            // Do not spin while last_seq intentionally remains behind.
            std::thread::sleep(Duration::from_millis(1));
        }
        let mut took_frame = if queued_capture && lookahead_ready {
            let max_queue = if flow_active {
                NEOFLOW_QUEUE_MAX
            } else if interp_active {
                ONNX_INTERP_QUEUE_MAX
            } else if load_reduction_lookahead {
                LOAD_REDUCTION_QUEUE_MAX
            } else if neodeint_active {
                6
            } else {
                SMOOTH_PACING_QUEUE_MAX
            };
            s.source.take_next_queued(&mut s.frame, max_queue)
        } else if queued_capture {
            false
        } else {
            s.source.take_latest(&mut s.frame)
        };
        if took_frame && s.capture_canvas_suspended_for_fullscreen {
            let monitor_size = (s.monitor_rect.2, s.monitor_rect.3);
            let frame_is_fullscreen =
                (s.frame.w - monitor_size.0).abs() <= 2 && (s.frame.h - monitor_size.1).abs() <= 2;
            if !frame_is_fullscreen {
                log::debug!(
                    "fullscreen-transition frame ignored: observed={}x{} monitor={}x{} reason=await-native-wgc-fullscreen",
                    s.frame.w,
                    s.frame.h,
                    monitor_size.0,
                    monitor_size.1
                );
                overlay.win.pump_messages();
                continue;
            }
        }
        if took_frame && s.capture_resolution_reapply_pending {
            // Keep the last complete fullscreen frame as a short transition
            // shield. The GUI will immediately reissue the latest capture-size
            // preference (or ClearCaptureGeometry for Auto), avoiding a brief
            // window-native frame followed by a second resize/rebuild.
            overlay.win.pump_messages();
            continue;
        }
        if took_frame
            && !s.capture_canvas_suspended_for_fullscreen
            && !s.capture_resolution_reapply_pending
            && let Some(target) = s.capture_canvas
            && should_apply_capture_canvas(
                (s.frame.w, s.frame.h),
                target,
                s.deferred_capture_resolution.is_some(),
            )
        {
            fit_frame_canvas_dot_by_dot(&mut s.frame, target);
        }
        // Apply the user FPS cap before the optional highlight-protection pass.
        // This prevents rejected frames from paying even the lightweight byte
        // processing cost. Geometry-establishing frames bypass the cap so initial capture,
        // explicit capture-size changes, and live source resizes cannot stall
        // on a static image that emits only one repaint.
        let cap_geometry_transition = took_frame
            && (s.in_size == (0, 0)
                || (s.frame.w, s.frame.h) != s.in_size
                || s.pending_resize_size.is_some()
                || s.deferred_capture_resolution.is_some_and(|(_, applied)| {
                    (s.frame.w, s.frame.h) != (applied.0 as i32, applied.1 as i32)
                }));
        // FPS cap uses target-rate decimation (for example, 20 on a 60fps
        // source must pin capture/present to a steady 20fps; 45 → 45fps).
        // The old WGC MinimumUpdateInterval gate beats against the source's
        // frame grid and rounds UP (30 setting → ~24fps measured). Here the
        // deadline advances by exactly 1/fps per accepted frame, so the
        // average locks onto the target regardless of the source cadence;
        // frames before the deadline are consumed but not processed (the
        // newest one at each deadline wins).
        let mut cap_skip = false;
        let mut cap_passthrough = false;
        if got_new && took_frame && !cap_geometry_transition {
            if let Some(cap) = s.fps_cap.filter(|c| *c > 0) {
                let now = Instant::now();
                // Decimate in the SOURCE-TIMESTAMP domain (WGC frame
                // time), not wall-clock arrival. WGC delivery wobbles under
                // load when capture delivery falls slightly below the source rate, and
                // wall-clock picking then samples the content unevenly =
                // visible judder. Picking by the frame's own timestamp keeps
                // the ACCEPTED frames evenly spaced on the video's clock —
                // the spacing the eye judges — regardless of delivery jitter.
                cap_skip = if let Some(ts) = s.frame.source_time_100ns {
                    let interval = 10_000_000i64 / cap.max(1) as i64;
                    if s.cap_rate_gate.source_is_within_cap(ts, interval) {
                        cap_passthrough = true;
                        s.next_cap_deadline_src = Some(ts + interval);
                        false
                    } else {
                        !cap_accept(&mut s.next_cap_deadline_src, ts, interval)
                    }
                } else {
                    let interval = Duration::from_secs_f64(1.0 / cap as f64);
                    let mut skip = false;
                    match s.next_cap_deadline {
                        Some(deadline) if now < deadline => skip = true,
                        Some(deadline) => {
                            // catch-up guard: resync instead of bursting
                            let next = deadline + interval;
                            s.next_cap_deadline =
                                Some(if next < now { now + interval } else { next });
                        }
                        None => s.next_cap_deadline = Some(now + interval),
                    }
                    skip
                };
                if cap_skip {
                    // frames ARE arriving: keep the starvation watchdog quiet
                    s.last_arrival = now;
                    s.starve_released = false;
                    // Re-present the LAST output (no chain work, ~sub-ms):
                    // keeps our present cadence at the CAPTURE rate. When the
                    // overlay only presented at the cap rate, the whole
                    // GPU/DWM pipeline slowed and the SOURCE dropped to
                    // ~55fps (fps-cap diag: src_frame_dt 17-20ms). The cap's
                    // purpose is to cut CHAIN work, not display cadence.
                    // Duplicate presents are not counted in the presentation statistic.
                    // With interpolation active, its exact capped output
                    // clock owns presentation. Re-presenting rejected native
                    // frames here steals those slots (15fps x3 collapsed to
                    // roughly 30fps in the field). Plain filtering keeps the
                    // lightweight duplicate present so WGC/DWM remains live.
                    if !interp_active && let Some(t) = s.last_tex {
                        let _ = overlay.present(&mut gc, t);
                        s.last_present = Instant::now();
                    }
                }
                // diag: delivered vs taken vs accepted + source frame pacing
                let d = &mut s.cap_diag;
                d.taken += 1;
                if !cap_skip {
                    d.accepted += 1;
                }
                if cap_passthrough {
                    d.passthrough += 1;
                }
                if let Some(ts) = s.frame.source_time_100ns {
                    if let Some(prev) = d.prev_ts {
                        let dt_ms = (ts - prev) as f64 / 10_000.0;
                        if dt_ms > 0.0 && dt_ms < 500.0 {
                            d.src_dt_ms = if d.src_dt_ms == 0.0 {
                                dt_ms
                            } else {
                                d.src_dt_ms * 0.9 + dt_ms * 0.1
                            };
                            d.dt_min_ms = d.dt_min_ms.min(dt_ms);
                            d.dt_max_ms = d.dt_max_ms.max(dt_ms);
                        }
                    }
                    d.prev_ts = Some(ts);
                }
                let dt = now.duration_since(d.last_log).as_secs_f64();
                if dt >= 2.0 {
                    let delivered = s.source.delivered();
                    log::info!(
                        "fps-cap diag: wgc_delivered={:.1}/s taken={:.1}/s accepted={:.1}/s rejected={:.1}/s passthrough={:.1}/s src_frame_dt={:.2}ms (min {:.1} max {:.1})",
                        delivered.saturating_sub(d.delivered0) as f64 / dt,
                        d.taken as f64 / dt,
                        d.accepted as f64 / dt,
                        (d.taken.saturating_sub(d.accepted)) as f64 / dt,
                        d.passthrough as f64 / dt,
                        d.src_dt_ms,
                        if d.dt_min_ms.is_finite() {
                            d.dt_min_ms
                        } else {
                            0.0
                        },
                        d.dt_max_ms
                    );
                    d.last_log = now;
                    d.delivered0 = delivered;
                    d.taken = 0;
                    d.accepted = 0;
                    d.passthrough = 0;
                    d.dt_min_ms = f64::INFINITY;
                    d.dt_max_ms = 0.0;
                }
            }
        }
        // A settings/filter change may request reprocessing while WGC is
        // static. This is a *semantic* refresh, not a duplicate source frame:
        // the same source pixels must pass through the newly selected chain.
        // Keep an explicit flag so cadence/duplicate suppression cannot reuse
        // the old chain's last_tex and defer the visible effect until a later
        // mouse/hover repaint from the source window.
        let mut chain_reprocess_forced = false;
        if !took_frame && s.chain_reprocess_pending && !s.frame.data.is_empty() {
            got_new = true;
            took_frame = true;
            chain_reprocess_forced = true;
            log::info!("chain-apply: reprocessing cached static source frame");
        }
        if took_frame && !cap_skip && s.capture_hdr {
            let current_seq = s.frame.seq;
            if s.hdr_highlight_protected_seq != Some(current_seq) {
                let source_size = (s.frame.w, s.frame.h);
                let source_bytes = s.frame.data.len();
                let source_was_hdr = s.frame.hdr;
                let processed = if source_was_hdr {
                    // Safety fallback only. The normal option path requests RGBA8.
                    let mut sdr_buffer = s.source.take_recycled_data_buffer();
                    let converted =
                        tonemap_hdr_frame_to_sdr(&mut s.frame, &mut sdr_buffer, s.hdr_sdr_mode);
                    if let Some(raw_hdr) = converted {
                        s.source.recycle_data_buffer(raw_hdr);
                        true
                    } else {
                        s.source.recycle_data_buffer(sdr_buffer);
                        false
                    }
                } else {
                    protect_sdr_highlights_in_place(&mut s.frame)
                };

                if processed {
                    s.hdr_highlight_protected_seq = Some(current_seq);
                    if !s.hdr_sdr_preprocess_logged {
                        log::info!(
                            "hdr-highlight-protection: input=RGBA8/sRGB output=RGBA8/sRGB size={}x{} knee={} ceiling={} neutral_max_chroma={} colour=maxrgb-ratio-preserving dither=none allocation=in-place cpu_path=u8-fixed-lut order=wgc->fps-cap->highlight-protection->duplicate-reduction->glsl/onnx/interpolation",
                            source_size.0,
                            source_size.1,
                            SDR_HIGHLIGHT_KNEE_U8,
                            SDR_HIGHLIGHT_CEILING_U8,
                            SDR_HIGHLIGHT_MAX_CHROMA_U8
                        );
                        s.hdr_sdr_preprocess_logged = true;
                    }
                } else if !s.hdr_sdr_preprocess_error_logged {
                    let bytes_per_pixel = if source_was_hdr { 8 } else { 4 };
                    log::error!(
                        "hdr-highlight-protection failed: size={}x{} bytes={} expected={} format={}",
                        source_size.0,
                        source_size.1,
                        source_bytes,
                        source_size.0.max(0) as usize
                            * source_size.1.max(0) as usize
                            * bytes_per_pixel,
                        if source_was_hdr { "Rgba16F" } else { "Rgba8" }
                    );
                    s.hdr_sdr_preprocess_error_logged = true;
                }
            }
        }
        let phase_capture_finished = Instant::now();
        if took_frame {
            s.duplicate_lookahead_wait = None;
        }
        if took_frame && !cap_skip && neodeint_active {
            let best_comb = frame_comb_fraction(&s.frame);
            // A clean progressive frame must remain bit-for-bit untouched by
            // NeoDeint. Once two of the latest four frames are woven, however,
            // hold reconstruction across the clean-looking half of that
            // cadence. Four consecutive clean frames release the latch.
            let combed = best_comb > NEODEINT_PROGRESSIVE_BYPASS_MAX;
            let bypass = update_neodeint_scene_latch(
                &mut s.neodeint_comb_history,
                &mut s.neodeint_scene_hold,
                combed,
            );
            if bypass != s.chain.neodeint_bypass() {
                log::info!(
                    "NeoDeint progressive bypass: enabled={} comb={:.3}% history={:04b} hold={}",
                    bypass,
                    best_comb * 100.0,
                    s.neodeint_comb_history,
                    s.neodeint_scene_hold
                );
            }
            s.chain.set_neodeint_bypass(bypass);
        } else if !neodeint_active {
            s.chain.set_neodeint_bypass(false);
            s.neodeint_comb_history = 0;
            s.neodeint_scene_hold = 0;
        }
        let lookahead_signature = if took_frame && !cap_skip && load_reduction_lookahead {
            s.source.inspect_next_queued(frame_signature).flatten()
        } else {
            None
        };
        if !took_frame
            && s.capture_resolution_applied
            && s.capture_resolution_wait_started.is_some()
            && s.capture_resolution_repaint_last
                .is_none_or(|last| last.elapsed() >= Duration::from_millis(120))
        {
            // Never stretch the cached pre-resize image into the new capture
            // dimensions. Its browser viewport is still the old one, so
            // Anime4K makes that temporary mismatch especially visible as a
            // slight zoom until mouse motion causes Chromium to repaint.
            win32::request_window_repaint(s.hwnd);
            s.capture_resolution_repaint_last = Some(Instant::now());
        }
        if !took_frame
            && s.capture_resolution_applied
            && !s.capture_resolution_nudge_done
            && s.capture_resolution_wait_started
                .is_some_and(|started| started.elapsed() >= Duration::from_millis(240))
            && let Some((_, applied)) = s.deferred_capture_resolution
        {
            // Chrome's GPU-composited PIP may suppress WM_PAINT while it is
            // visually hidden. A tiny even-sized client round trip invalidates
            // its compositor surface without exposing an intermediate frame.
            // This changes window geometry only; captured pixels are never
            // resampled by cHiDeScaler-Neo.
            let nudge = (applied.0.saturating_sub(2), applied.1.saturating_sub(2));
            let nudged = win32::resize_client_area(s.hwnd, nudge.0, nudge.1);
            let restored = win32::resize_client_area(s.hwnd, applied.0, applied.1);
            win32::request_window_repaint(s.hwnd);
            s.capture_resolution_nudge_done = true;
            s.capture_resolution_repaint_last = Some(Instant::now());
            log::info!(
                "capture-resolution static compositor nudge: {}x{} -> {}x{} hidden round-trip nudged={} restored={}",
                nudge.0,
                nudge.1,
                applied.0,
                applied.1,
                nudged,
                restored
            );
        }
        // A requested capture canvas is authoritative for ONNX. Never let a
        // transient native-size frame become the first DirectML run or a
        // TensorRT shape-build request: both providers keep shape-dependent
        // state and the former ordering could halve steady-state throughput.
        if got_new
            && took_frame
            && s.chain.has_onnx()
            && let Some((requested, applied)) = s.deferred_capture_resolution
        {
            let observed = (s.frame.w, s.frame.h);
            let target = (applied.0 as i32, applied.1 as i32);
            if observed != target {
                if !s.onnx_geometry_deferred_logged {
                    log::info!(
                        "onnx-inference-deferred: observed={}x{} target={}x{} requested={}x{} reason=capture-resolution-transition",
                        observed.0,
                        observed.1,
                        target.0,
                        target.1,
                        requested.0,
                        requested.1
                    );
                    s.onnx_geometry_deferred_logged = true;
                }
                if !s.capture_resolution_applied {
                    let hidden = if s.hide_source_pending {
                        if let Some(was_layered) = win32::hide_window_visual(s.hwnd) {
                            s.hide_source_pending = false;
                            s.hid_source = Some(was_layered);
                            status.lock().unwrap().hidden_src = Some((s.hwnd, was_layered));
                            true
                        } else {
                            false
                        }
                    } else {
                        true
                    };
                    if hidden && win32::resize_client_area(s.hwnd, applied.0, applied.1) {
                        s.capture_resolution_applied = true;
                        s.capture_resolution_wait_started = Some(Instant::now());
                        s.capture_resolution_repaint_last = None;
                        s.capture_resolution_nudge_done = false;
                        s.capture_resolution_native_fallback = false;
                        win32::request_window_repaint(s.hwnd);
                        log::info!(
                            "capture-resolution applied before first ONNX inference: hwnd={:#x} client={}x{}",
                            s.hwnd,
                            applied.0,
                            applied.1
                        );
                    }
                }
                overlay.win.pump_messages();
                continue;
            }
            s.onnx_geometry_deferred_logged = false;
        } else if s.deferred_capture_resolution.is_none() {
            s.onnx_geometry_deferred_logged = false;
        }
        if !took_frame
            && s.capture_resolution_applied
            && !s.capture_resolution_native_fallback
            && !s.chain.has_onnx()
            && s.capture_resolution_wait_started
                .is_some_and(|started| started.elapsed() >= Duration::from_millis(1200))
            && !s.frame.data.is_empty()
        {
            // Re-filter the exact source pixels captured before Chromium's
            // client resize. Do not mutate frame.w/h and do not resample its
            // pixels: output geometry remains derived from the true source
            // aspect, avoiding the temporary zoom of the former fallback.
            got_new = true;
            took_frame = true;
            s.capture_resolution_native_fallback = true;
            log::info!(
                "capture-resolution static native fallback: reprocessing cached {}x{} frame without resize while awaiting real target frame",
                s.frame.w,
                s.frame.h
            );
        }
        // Chrome PIP can switch 4:3 -> 16:9 -> 4:3 while retaining the same
        // HWND. Keep the old display geometry until the new size is observed
        // twice; otherwise Windows stretches the old framebuffer into the new
        // window shape, visibly swapping the 4:3/16:9 aspect during transition.
        let mut resize_committed = false;
        if got_new && took_frame {
            let new_in_size = (s.frame.w, s.frame.h);
            match s.pending_resize_size {
                Some(pending) if new_in_size == pending => {
                    s.in_size = new_in_size;
                    s.pending_resize_size = None;
                    resize_committed = true;
                    // Once WGC confirms the new geometry twice, it is the
                    // authoritative content aspect even when the session began
                    // with an explicit capture canvas. Keeping the old canvas
                    // here produced the zoom/crop-looking output after a live
                    // 1280x720 -> 960x540 (or 16:9 -> 4:3) change.
                    s.display_aspect = new_in_size;
                    if s.capture_canvas.is_some()
                        && !s.capture_canvas_suspended_for_fullscreen
                        && !s.capture_resolution_reapply_pending
                    {
                        s.capture_canvas = Some(new_in_size);
                    }
                    s.last_geom_change = None;
                    s.starve_released = false;
                    if s.deferred_capture_resolution.is_none() {
                        // Never overwrite the session-origin restore snapshot.
                        // A GUI SetWindowPos can reach Chromium/WGC before the
                        // matching SetCaptureGeometry command reaches this
                        // render loop. In that race the automatic resize path
                        // used to "adopt" the newly requested capture size as
                        // the restore target, so Stop first restored the true
                        // origin via Status::source_recovery and cleanup then
                        // resized it again to the most recent capture size.
                        // Processing/display geometry may follow the new WGC
                        // shape, but Stop always returns to the pre-capture
                        // position, size and maximized state.
                        log::info!(
                            "source geometry transition observed without changing session restore origin: hwnd={:#x} live_rect={:?} origin_rect={:?} origin_maximized={} display_aspect={}x{}",
                            s.hwnd,
                            win32::window_rect(s.hwnd),
                            s.source_restore_rect,
                            s.source_was_maximized,
                            s.display_aspect.0,
                            s.display_aspect.1
                        );
                    }
                    log::info!(
                        "source frame size stable after resize: {}x{}; aspect/display commit armed with filter chain preserved",
                        new_in_size.0,
                        new_in_size.1
                    );
                }
                Some(pending) if s.capture_resolution_applied && new_in_size != pending => {
                    // Chromium/DWM may emit old-size or intermediate-size
                    // frames after SetWindowPos succeeds. They are not a new
                    // authoritative geometry. Keep the transition shield and
                    // wait for the exact requested WGC frame instead of
                    // rebuilding ONNX for a transient shape or revealing it.
                    log::debug!(
                        "capture-resolution transitional frame ignored: observed={}x{} current={}x{} target={}x{}",
                        new_in_size.0,
                        new_in_size.1,
                        s.in_size.0,
                        s.in_size.1,
                        pending.0,
                        pending.1
                    );
                    overlay.win.pump_messages();
                    continue;
                }
                Some(_) if new_in_size == s.in_size => {
                    s.pending_resize_size = None;
                    resize_committed = true;
                    log::info!(
                        "source resize transition reverted to {}x{}; rebuilding filters without changing display aspect",
                        new_in_size.0,
                        new_in_size.1
                    );
                }
                Some(_) | None if s.in_size != (0, 0) && new_in_size != s.in_size => {
                    let old_size = s.in_size;
                    reset_for_source_resize(&mut gc, s, &metrics, old_size, new_in_size);
                    let now = Instant::now();
                    s.last_geom_change = Some(now);
                    s.last_arrival = now;
                    s.starve_released = false;
                    let expected_capture_frame = s.capture_resolution_applied
                        && s.deferred_capture_resolution.is_some_and(|(_, applied)| {
                            new_in_size == (applied.0 as i32, applied.1 as i32)
                        });
                    if expected_capture_frame {
                        // A user-requested resize is deterministic. The first
                        // exact target-size WGC frame is already the correct
                        // static image; requiring a duplicate frame deadlocks
                        // until mouse activity makes the source repaint.
                        s.in_size = new_in_size;
                        s.display_aspect = new_in_size;
                        if s.capture_canvas.is_some() {
                            s.capture_canvas = Some(new_in_size);
                        }
                        s.pending_resize_size = None;
                        resize_committed = true;
                        log::info!(
                            "capture-resolution first exact frame accepted: {}x{}; filtering behind the transition shield",
                            new_in_size.0,
                            new_in_size.1
                        );
                    } else {
                        overlay.win.pump_messages();
                        continue;
                    }
                }
                _ => {
                    // The Win32 client rect is only a startup estimate. Chrome
                    // commonly reports a rect including compositor margins
                    // (for example 1059x689) while WGC's exact client pixels
                    // are 1044x682. Keeping the estimate made a filter-free
                    // 1.0x session resample every frame and visibly blur it.
                    // Commit the first real WGC geometry immediately. An
                    // explicit capture canvas remains authoritative because
                    // the source was resized to that size before capture.
                    if s.in_size == (0, 0) {
                        s.in_size = new_in_size;
                        s.display_aspect = initial_display_aspect(s.capture_canvas, new_in_size);
                        resize_committed = true;
                        log::info!(
                            "initial WGC geometry committed: frame={}x{} display={}x{} capture_canvas={:?} pixel_exact={}",
                            new_in_size.0,
                            new_in_size.1,
                            s.display_aspect.0,
                            s.display_aspect.1,
                            s.capture_canvas,
                            new_in_size == s.display_aspect
                        );
                    }
                }
            }
        }
        if resize_committed && s.onnx_geometry_rebuild_pending {
            match s.chain.rebuild_onnx_sessions_for_geometry(&mut factory) {
                Ok(count) => {
                    s.onnx_geometry_rebuild_pending = false;
                    let usage = s.chain.onnx_backend_usage();
                    {
                        let mut state = status.lock().unwrap();
                        state.onnx_tensorrt_stages = usage.tensorrt;
                        state.onnx_cuda_stages = usage.cuda;
                        state.onnx_directml_fallbacks = usage.directml_fallback;
                    }
                    log::info!(
                        "onnx-geometry-session-rebuilt: input={}x{} stages={} gl_cache=preserved reason=shape-specific-runtime-reset",
                        s.in_size.0,
                        s.in_size.1,
                        count
                    );
                }
                Err(error) => {
                    s.onnx_geometry_rebuild_pending = false;
                    let message = format!("解像度変更後のONNX再初期化に失敗しました: {error:#}");
                    log::error!("onnx-geometry-session-rebuild-failed: {error:#}");
                    status.lock().unwrap().last_error = Some(message);
                }
            }
        }
        let mut smooth_content_duplicate = false;
        let browser_fullscreen_cadence = AUTO_CONTENT_CADENCE_ENABLED
            && ((win32::is_browser_window(s.hwnd) && win32::is_monitor_fullscreen(s.hwnd))
                || neodeint_active);
        if browser_fullscreen_cadence != s.browser_fullscreen_cadence {
            s.browser_fullscreen_cadence = browser_fullscreen_cadence;
            s.smooth_content_signature = None;
            s.smooth_content_last_candidate_seq = None;
            s.smooth_content_last_candidate_time = None;
            s.smooth_content_pattern_hits = 0;
            s.smooth_content_candidate_gaps.clear();
            s.smooth_content_24p_detected = false;
            s.cadence.reset();
            s.smooth_pacer.reset();
            s.paced_present_deadline = None;
            log::info!(
                "browser fullscreen content-cadence detection: enabled={}",
                browser_fullscreen_cadence
            );
        }
        if got_new
            && took_frame
            && !cap_skip
            && !chain_reprocess_forced
            && (s.smooth_pacing || neodeint_active)
            && (s.duplicate_frame_reduction || s.browser_fullscreen_cadence)
            && !resize_committed
            && let Some(signature) = frame_signature(&s.frame)
        {
            let input_seq = s.frame.seq;
            let signature_matches = if s.duplicate_frame_reduction {
                let forward_motion = lookahead_signature
                    .as_ref()
                    .is_some_and(|next| signature_has_coherent_motion(&signature, next));
                let temporal_motion = s
                    .smooth_content_signature
                    .as_ref()
                    .zip(lookahead_signature.as_ref())
                    .is_some_and(|(previous, next)| {
                        signature_has_temporal_continuity(previous, &signature, next)
                    });
                if forward_motion || temporal_motion {
                    // The current picture begins/continues a coherent change.
                    // Reject it before duplicate reuse; look-ahead is useful
                    // precisely for the first low-amplitude fade/zoom step.
                    s.duplicate_motion_guard = 12;
                    false
                } else if s.duplicate_motion_guard > 0 {
                    s.duplicate_motion_guard -= 1;
                    false
                } else if let Some(previous) = s.smooth_content_signature.as_ref() {
                    let adjacent_close = signatures_luma_close(previous, &signature);
                    let adjacent_match =
                        adjacent_close && !signature_has_coherent_motion(previous, &signature);
                    // The immediate comparison alone can "walk" along a slow
                    // zoom forever. The last actually processed frame is a
                    // fixed anchor; divergence from it proves real movement
                    // even while every adjacent pair remains near-identical.
                    let anchor_match = s
                        .duplicate_signature
                        .as_ref()
                        .is_none_or(|anchor| signatures_match(anchor, &signature));
                    if adjacent_close && (!adjacent_match || !anchor_match) {
                        // Keep a half-second 24p window fully processed. This
                        // avoids alternating reuse/process steps through the
                        // remainder of a zoom or camera push.
                        s.duplicate_motion_guard = 12;
                        false
                    } else {
                        adjacent_match
                    }
                } else {
                    false
                }
            } else {
                s.smooth_content_signature
                    .as_ref()
                    .is_some_and(|previous| compositor_signatures_match(previous, &signature))
            };
            if !s.duplicate_frame_reduction && signature_matches {
                let candidate_time = s.frame.source_time_100ns;
                if let (Some(previous_candidate), Some(previous_time), Some(now_time)) = (
                    s.smooth_content_last_candidate_seq,
                    s.smooth_content_last_candidate_time,
                    candidate_time,
                ) {
                    let gap = input_seq.saturating_sub(previous_candidate);
                    let gap_ms = (now_time - previous_time) as f64 / 10_000.0;
                    if gap == 5 && (150.0..=185.0).contains(&gap_ms) {
                        s.smooth_content_pattern_hits =
                            s.smooth_content_pattern_hits.saturating_add(1);
                    } else {
                        s.smooth_content_pattern_hits = 0;
                    }
                    if (1..=2).contains(&gap) && (8.0..=40.0).contains(&gap_ms) {
                        s.smooth_content_candidate_gaps.push_back(gap);
                        while s.smooth_content_candidate_gaps.len() > 10 {
                            s.smooth_content_candidate_gaps.pop_front();
                        }
                    } else {
                        s.smooth_content_candidate_gaps.clear();
                    }
                    let sixty_hz_24p = s.smooth_content_candidate_gaps.len() >= 8
                        && s.smooth_content_candidate_gaps.contains(&1)
                        && s.smooth_content_candidate_gaps.contains(&2)
                        && {
                            let mean = s.smooth_content_candidate_gaps.iter().copied().sum::<u64>()
                                as f64
                                / s.smooth_content_candidate_gaps.len() as f64;
                            (1.3..=1.7).contains(&mean)
                        };
                    if !s.smooth_content_24p_detected
                        && (s.smooth_content_pattern_hits >= 3 || sixty_hz_24p)
                    {
                        s.smooth_content_24p_detected = true;
                        s.cadence.force_period(1.0 / 24.0);
                        log::info!(
                            "generic contained-24p cadence locked: duplicate_gap={} duplicate_period_ms={:.2} confirmations={} surface={} source_period_ms=41.67",
                            gap,
                            gap_ms,
                            s.smooth_content_pattern_hits,
                            if sixty_hz_24p { "60Hz" } else { "30Hz" }
                        );
                    }
                }
                s.smooth_content_last_candidate_seq = Some(input_seq);
                s.smooth_content_last_candidate_time = candidate_time;
            } else if !s.duplicate_frame_reduction
                && s.smooth_content_24p_detected
                && let (Some(last), Some(now)) = (
                    s.smooth_content_last_candidate_time,
                    s.frame.source_time_100ns,
                )
                && now - last > 5_000_000
            {
                s.smooth_content_24p_detected = false;
                s.smooth_content_pattern_hits = 0;
                s.smooth_content_candidate_gaps.clear();
                s.smooth_content_last_candidate_seq = None;
                s.smooth_content_last_candidate_time = None;
                s.cadence.reset();
                s.smooth_pacer.reset();
                s.paced_present_deadline = None;
                log::info!("generic contained-24p cadence unlocked: pattern absent for 500ms");
            }
            // Load reduction may remove every matching picture. With it off,
            // omission begins only after a repeated ~5-frame 3:2 pattern has
            // proved that this is 24p contained in a 30Hz surface.
            smooth_content_duplicate =
                signature_matches && (s.duplicate_frame_reduction || s.smooth_content_24p_detected);
            // Compare against the immediately preceding WGC delivery even
            // when that delivery is omitted from presentation.
            s.smooth_content_signature = Some(signature);
            if smooth_content_duplicate {
                // WGC timestamps describe Chrome/DWM composition updates.
                // A fullscreen 24p YouTube surface is commonly emitted at
                // 30Hz with one repeated picture in each five-frame group.
                // Re-present the cached filtered result for that source slot:
                // this keeps 120/240Hz surfaces and the displayed FPS honest
                // without running the expensive filter chain again.
                let now = Instant::now();
                s.last_arrival = now;
                s.starve_released = false;
                if s.duplicate_frame_reduction {
                    s.strict_static_streak = s.strict_static_streak.saturating_add(1);
                }
                let strict_static_hold = s.duplicate_frame_reduction
                    && s.strict_static_streak >= STRICT_STATIC_HOLD_FRAMES;
                if s.strict_static_streak == STRICT_STATIC_HOLD_FRAMES {
                    // A retained DWM surface needs no per-frame SwapBuffers.
                    // Re-anchor before motion resumes so the old deadline
                    // cannot delay the first changed frame.
                    s.smooth_pacer.reset();
                    s.paced_present_deadline = None;
                    log::info!(
                        "duplicate-frame reduction: strict static hold entered after {} matching frames",
                        STRICT_STATIC_HOLD_FRAMES
                    );
                }
                let keepalive_due =
                    s.strict_static_last_present.elapsed() >= IDLE_KEEPALIVE_INTERVAL;
                let should_present = !strict_static_hold || keepalive_due;
                if should_present && let Some(tex) = s.last_tex {
                    if overlay.present(&mut gc, tex).is_ok() {
                        s.last_present = now;
                        s.strict_static_last_present = now;
                        s.present_cadence.record(now);
                        status.lock().unwrap().presented += 1;
                        let delivered = s.source.delivered();
                        let captures = delivered.saturating_sub(s.metric_seq) as u32;
                        s.metric_seq = delivered;
                        metrics.frame(
                            true,
                            captures,
                            None,
                            Some(0.0),
                            s.in_size,
                            (tex.w(), tex.h()),
                        );
                    }
                } else if strict_static_hold {
                    let delivered = s.source.delivered();
                    let captures = delivered.saturating_sub(s.metric_seq) as u32;
                    s.metric_seq = delivered;
                    metrics.frame(false, captures, None, Some(0.0), s.in_size, (0, 0));
                    s.strict_static_present_skipped += 1;
                    if s.strict_static_present_skipped % 120 == 1 {
                        log::info!(
                            "duplicate-frame reduction: strict static present suppressed count={}",
                            s.strict_static_present_skipped
                        );
                    }
                }
                s.smooth_content_duplicates += 1;
                s.smooth_content_unique_since_duplicate = 0;
                if s.smooth_content_duplicates % 120 == 1 {
                    log::info!(
                        "smooth content-cadence extraction: compositor duplicate omitted count={}",
                        s.smooth_content_duplicates
                    );
                }
            } else {
                if s.strict_static_streak >= STRICT_STATIC_HOLD_FRAMES {
                    s.smooth_pacer.reset();
                    s.paced_present_deadline = None;
                    log::info!("duplicate-frame reduction: strict static hold exited on change");
                }
                s.strict_static_streak = 0;
                s.smooth_content_unique_since_duplicate =
                    s.smooth_content_unique_since_duplicate.saturating_add(1);
                s.smooth_content_seq = s.smooth_content_seq.saturating_add(1);
                // Downstream interpolation must see contiguous *content*
                // sequence IDs. Otherwise the omitted compositor duplicate
                // makes a 24p pair look like a multi-frame capture gap.
                s.frame.seq = s.smooth_content_seq;
            }
        } else if got_new && took_frame && !cap_skip && (resize_committed || !s.smooth_pacing) {
            s.strict_static_streak = 0;
            s.smooth_content_signature = None;
            s.smooth_content_seq = 0;
            s.smooth_content_unique_since_duplicate = 4;
            s.smooth_content_last_candidate_seq = None;
            s.smooth_content_last_candidate_time = None;
            s.smooth_content_pattern_hits = 0;
            s.smooth_content_candidate_gaps.clear();
            s.smooth_content_24p_detected = false;
        }
        let mut duplicate_skip = false;
        if got_new
            && took_frame
            && !cap_skip
            && !chain_reprocess_forced
            && !interp_active
            && !resize_committed
            && !smooth_content_duplicate
            && s.duplicate_frame_reduction
            && s.last_tex.is_some()
            && let Some(signature) = frame_signature(&s.frame)
        {
            duplicate_skip = s
                .duplicate_signature
                .as_ref()
                .is_some_and(|previous| signatures_match(previous, &signature));
            if duplicate_skip {
                // Preserve the selected capture timing. Smooth mode consumes
                // its bounded queue; low-latency mode uses the newest arrival.
                // Only the expensive chain is skipped.
                let now = Instant::now();
                s.last_arrival = now;
                s.starve_released = false;
                if let Some(tex) = s.last_tex {
                    let _ = overlay.present(&mut gc, tex);
                    s.last_present = now;
                    s.present_cadence.record(now);
                }
                s.duplicate_skipped += 1;
                if s.duplicate_skipped % 120 == 1 {
                    log::info!(
                        "duplicate-frame reduction: reused filtered output count={}",
                        s.duplicate_skipped
                    );
                }
            } else {
                s.duplicate_signature = Some(signature);
            }
        } else if got_new && took_frame && !cap_skip && !interp_active {
            // Geometry/HDR/mode transitions must establish a fresh baseline.
            s.duplicate_signature = frame_signature(&s.frame);
        }
        if got_new && took_frame && !cap_skip && !duplicate_skip && !smooth_content_duplicate {
            // The only frames that reach the chain have passed both gates in this
            // exact order: FPS cap -> duplicate reduction -> GLSL/ONNX/interpolation.
            // Assign a contiguous sequence only now, after duplicate reuse, so
            // temporal filters never see intentional cap/reduction gaps.
            if s.fps_cap.is_some_and(|cap| cap > 0) {
                s.cap_filter_seq = s.cap_filter_seq.saturating_add(1);
                s.frame.seq = s.cap_filter_seq;
            }
            if s.smooth_pacing && s.browser_fullscreen_cadence {
                // Consecutive accepted pictures are the content cadence. Do
                // not divide their timestamp span by skipped WGC deliveries.
                s.cadence.observe_content(s.frame.source_time_100ns);
            } else {
                s.cadence
                    .observe_frame(s.frame.source_time_100ns, s.frame.seq);
            }
            metrics.set_source_fps(s.cadence.period_s().map(|period| 1.0 / period));
        }
        if got_new && took_frame && !cap_skip && !duplicate_skip && !smooth_content_duplicate {
            // render target size: at least 2x the source so mpv shaders with
            // WHEN "OUTPUT/MAIN > 1.2" activate even at small display ratios
            // (the presented image is then supersampled down = better quality)
            let t0 = Instant::now();
            let mut frame_paced = false;
            let new_in_size = (s.frame.w, s.frame.h);
            // arrival-interval EWMA (drives interpolation pacing)
            let dt = t0.duration_since(s.last_arrival).as_secs_f64();
            if dt < 0.5 {
                s.arrival_interval = if s.arrival_interval == 0.0 {
                    dt
                } else {
                    s.arrival_interval * 0.8 + dt * 0.2
                };
            }
            s.last_arrival = t0;
            s.in_size = new_in_size;
            // virtual OUTPUT = 4x the source: upscalers gated by
            // "WHEN OUTPUT/MAIN > x" fire fully even stacked (2x+2x = 4x);
            // the result is fitted to the window with the downscale kernel
            let out_size = virtual_shader_output_size((s.frame.w, s.frame.h));
            // Per-stage GLSL timing calls glFinish before and after every
            // stage. Sampling preserves useful GUI statistics without
            // serializing the GPU pipeline on every video frame.
            let stats = metrics.detailed_enabled() && s.frame.seq % 30 == 0;

            // ---- frame interpolation: synthesize (factor-1) in-between
            // frames, pacing each at k/factor of the arrival interval ----
            if let Some(crate::render::chain::InterpHandle::Flow {
                name: interp_name,
                external_source,
            }) = s.chain.interp_stage()
            {
                // ---- built-in interpolation ----
                // The source arrival already paces this branch. Waiting for a
                // full source period here and then another half-period between
                // the generated midpoint and real frame starved a 24p source:
                // NeoFlow itself took <1ms, yet its queue overflowed and output
                // settled around 36-44fps. Start immediately, then pace only
                // the midpoint -> real-frame interval below.
                // Respect a leading ONNX image filter before NeoFlow. Running
                // that stage once on each captured endpoint is both the stated
                // chain order and far cheaper than re-running it on every
                // interpolated output. The existing direct RGBA path also
                // avoids an unnecessary GL readback/upload round-trip.
                let mut flow_chain_start = 0usize;
                let cur_tex = if !s.frame.hdr && s.chain.interp_index().is_some_and(|i| i > 0) {
                    match s.chain.process_first_onnx_rgba8(
                        &mut gc,
                        s.frame.w,
                        s.frame.h,
                        &s.frame.data,
                    ) {
                        Ok(Some((tex, name, ms))) => {
                            flow_chain_start = 1;
                            if stats {
                                metrics.probe(&name, "onnx", ms);
                            }
                            tex
                        }
                        Ok(None) | Err(_) => {
                            upload_frame_timed(&mut gc, &s.frame, &mut s.last_upload_submit_ms)
                        }
                    }
                } else {
                    upload_frame_timed(&mut gc, &s.frame, &mut s.last_upload_submit_ms)
                };
                let requested_factor = interp_factor.max(2);
                let pair_period_s = normalized_frame_span_s(
                    s.prev_tex_source_time_100ns,
                    s.frame.source_time_100ns,
                    s.prev_tex_seq,
                    s.frame.seq,
                );
                let source_period_s = neoflow_source_period_s(
                    s.cadence.period_s(),
                    pair_period_s,
                    s.arrival_interval,
                );
                let delay_frames = if s.arrival_interval > 0.0 {
                    (s.last_process_ms / 1000.0) / s.arrival_interval
                } else {
                    0.0
                };
                let delay_limited_factor = if delay_frames < NEOFLOW_FULL_DELAY_FRAMES {
                    requested_factor
                } else if delay_frames < NEOFLOW_HARD_SKIP_DELAY_FRAMES {
                    2
                } else {
                    1
                };
                let output_ratio = refresh_limited_output_ratio(
                    delay_limited_factor,
                    source_period_s,
                    s.monitor_refresh_hz,
                );
                let fixed_midpoint_external = external_source.as_deref().is_some_and(|source| {
                    source.contains("NeoFlow GameDIS")
                        || source.contains("NeoFlow GameMesh")
                        || source.contains("NeoFlow HybridCadence")
                        || source.contains("NeoFlow CausalStable")
                });
                let output_phases = if output_ratio >= 1.5 && fixed_midpoint_external {
                    // These experimental hosts implement the certified x2
                    // midpoint only. Do not feed them the fractional cadence
                    // accumulator: after a static/paused WGC source it may
                    // resume at e.g. [0.025, 0.525], which incorrectly asks
                    // for two synthetic frames and disables the filter.
                    vec![0.5, 1.0]
                } else if output_ratio >= 1.5 {
                    s.flow_output_cadence.phases(output_ratio)
                } else {
                    Vec::new()
                };
                let present_period_s = source_period_s
                    .map(|period| period / output_ratio.max(1.0))
                    .unwrap_or_else(|| s.arrival_interval / output_ratio.max(1.0));
                let present_real = output_phases
                    .last()
                    .is_some_and(|phase| *phase >= 1.0 - 1e-5);
                let effective_factor = output_phases.len() as u32;
                let mut skipped_reason = "no-prev";
                let mut interp_total_ms = 0.0;
                let cur_seq = s.frame.seq;
                let prev_seq = s.prev_tex_seq;
                let delivered = s.source.delivered();
                let mut captures_pending = delivered.saturating_sub(s.metric_seq) as u32;
                s.metric_seq = delivered;
                let seq_gap = if prev_seq > 0 {
                    cur_seq.saturating_sub(prev_seq)
                } else {
                    0
                };
                if let Some(prev_tex) = s.prev_tex {
                    if prev_tex.key.w != cur_tex.key.w || prev_tex.key.h != cur_tex.key.h {
                        skipped_reason = "size-change";
                    } else if delay_frames >= NEOFLOW_HARD_SKIP_DELAY_FRAMES {
                        skipped_reason = "delay>=1.95";
                    } else if seq_gap > NEOFLOW_QUEUE_MAX as u64 {
                        skipped_reason = "seq-gap";
                    } else if output_phases.is_empty() {
                        skipped_reason = if output_ratio < 1.5 {
                            "refresh-limit"
                        } else {
                            "factor<2"
                        };
                    } else {
                        skipped_reason = "none";
                        let phases: Vec<f32> = output_phases
                            .iter()
                            .copied()
                            .filter(|phase| *phase < 1.0 - 1e-5)
                            .collect();
                        let analysis_t0 = Instant::now();
                        if seq_gap > 1 || captures_pending > 1 {
                            gc.invalidate_external_neoflow_history(true);
                        }
                        let interpolation = if let Some(source) = external_source.as_deref() {
                            crate::render::flow::interpolate_many_external(
                                &mut gc, prev_tex, cur_tex, &phases, source,
                            )
                        } else {
                            crate::render::flow::interpolate_many(
                                &mut gc, prev_tex, cur_tex, &phases,
                            )
                        };
                        match interpolation {
                            Ok(mids) => {
                                interp_total_ms = analysis_t0.elapsed().as_secs_f64() * 1000.0;
                                for (index, (&t, &mid)) in
                                    phases.iter().zip(mids.iter()).enumerate()
                                {
                                    s.flow_output_cadence.wait_for_present(
                                        &mut overlay,
                                        present_period_s,
                                        vsync_on,
                                        s.smooth_pacing,
                                    );
                                    let mid_t0 = Instant::now();
                                    let mid_timing = FrameTiming::from_interpolated(
                                        &s.frame,
                                        s.prev_tex_source_time_100ns,
                                        t,
                                        mid_t0,
                                    );
                                    let mut keep = Vec::with_capacity(2 + mids.len() - index - 1);
                                    keep.push(prev_tex);
                                    keep.push(cur_tex);
                                    keep.extend_from_slice(&mids[index + 1..]);
                                    let captures = std::mem::take(&mut captures_pending);
                                    process_and_present_from(
                                        &mut gc,
                                        &mut overlay,
                                        s,
                                        &metrics,
                                        &status,
                                        mid,
                                        out_size,
                                        stats,
                                        mid_t0,
                                        captures,
                                        Some(mid_timing),
                                        &keep,
                                        downscaler,
                                        flow_chain_start,
                                    );
                                }
                            }
                            Err(e) => {
                                gc.invalidate_external_neoflow_history(true);
                                skipped_reason = "interpolate-error";
                                let msg = format!("{interp_name}: {e:#}");
                                log::error!("frame interpolation failed: {msg}");
                                let mut g = status.lock().unwrap();
                                if !g.chain_errors.contains(&msg) {
                                    g.chain_errors.push(msg);
                                }
                            }
                        }
                    }
                }
                if stats && interp_total_ms > 0.0 {
                    metrics.probe(&interp_name, "gpu", interp_total_ms);
                }
                if s.last_neoflow_log.elapsed() >= Duration::from_secs(1) {
                    let source_fps = if s.arrival_interval > 0.0 {
                        1.0 / s.arrival_interval
                    } else {
                        0.0
                    };
                    log::info!(
                        "neoflow-stats: source_fps={:.2} arrival_ms={:.2} interp_factor={} scheduled_outputs={} output_ratio={:.3} refresh_hz={:.2} present_period_ms={:.2} endpoint={} delay_frames={:.2} last_process_ms={:.2} interpolate_ms={:.2} skipped_reason={} dropped_source_frames={} prev_seq={} cur_seq={} seq_gap={}",
                        source_fps,
                        s.arrival_interval * 1000.0,
                        requested_factor,
                        effective_factor,
                        output_ratio,
                        s.monitor_refresh_hz.unwrap_or(0.0),
                        present_period_s * 1000.0,
                        present_real,
                        delay_frames,
                        s.last_process_ms,
                        interp_total_ms,
                        skipped_reason,
                        s.source.queued_dropped(),
                        prev_seq,
                        cur_seq,
                        seq_gap
                    );
                    s.last_neoflow_log = Instant::now();
                }
                // A fractional display cadence may intentionally omit this
                // exact endpoint. For 24p -> 60Hz the output counts are
                // 3,2,3,2...; every generated frame still uses cur_tex and it
                // becomes the next motion-analysis endpoint below.
                let should_present_real = present_real || skipped_reason != "none";
                if should_present_real {
                    s.flow_output_cadence.wait_for_present(
                        &mut overlay,
                        present_period_s,
                        vsync_on,
                        s.smooth_pacing,
                    );
                    let captures = std::mem::take(&mut captures_pending);
                    let real_t0 = Instant::now();
                    process_and_present_from(
                        &mut gc,
                        &mut overlay,
                        s,
                        &metrics,
                        &status,
                        cur_tex,
                        out_size,
                        stats,
                        real_t0,
                        captures,
                        Some(FrameTiming::from_frame(&s.frame, real_t0)),
                        &[cur_tex],
                        downscaler,
                        flow_chain_start,
                    );
                }
                if let Some(old) = rotate_interp_history(s, cur_tex) {
                    gc.recycle(old);
                }
                s.prev_tex_seq = cur_seq;
                s.prev_tex_source_time_100ns = s.frame.source_time_100ns;
                overlay.win.pump_messages();
                continue;
            }
            if let Some(crate::render::chain::InterpHandle::Onnx {
                name: interp_name,
                stage: ist,
            }) = s.chain.interp_stage()
            {
                // The provider worker owns the stage mutex while inference is
                // running. Check in-flight state before touching the stage so
                // the render/input thread never waits on that mutex.
                if s.gpu_interp_pack_pending.is_some() {
                    // GL packing normally completes within the same refresh;
                    // keep pumping UI/Stop/Shutdown instead of waiting here.
                    overlay.win.pump_messages();
                    continue;
                }
                if s.frame.hdr {
                    // ONNX interp works on 8-bit RGB; HDR capture skips it
                    s.hist.clear();
                    // dropping the worker discards any queued stale results
                    s.interp_worker = None;
                    s.interp_pending = None;
                }
                let (need, delayed, rgba_direct, dml_rife) = {
                    let ist = ist.lock().unwrap();
                    (
                        ist.interp_frames(),
                        ist.interp_delayed(),
                        matches!(
                            &ist.interp,
                            crate::render::onnx_stage::InterpKind::RifeV1 { .. }
                                | crate::render::onnx_stage::InterpKind::Drba
                        ),
                        ist.provider == crate::render::onnx_stage::OnnxProvider::DirectML
                            && matches!(
                                &ist.interp,
                                crate::render::onnx_stage::InterpKind::RifeV1 { .. }
                            ),
                    )
                };
                let interp_index = s.chain.interp_index().unwrap_or(0);
                let post_chain_start = (interp_index + 1).min(s.chain.stage_count());
                let gpu_capable = !s.frame.hdr && ist.lock().unwrap().supports_interp_gpu();
                if gpu_capable {
                    let input = upload_frame_timed(&mut gc, &s.frame, &mut s.last_upload_submit_ms);
                    let mut pre_probe = |name: &str, kind: StageKind, ms: f64| {
                        metrics.probe(
                            name,
                            if kind == StageKind::Glsl {
                                "glsl"
                            } else {
                                "onnx"
                            },
                            ms,
                        );
                    };
                    let cur_tex = if interp_index > 0 {
                        match s.chain.process_range(
                            &mut gc,
                            input,
                            out_size,
                            0,
                            interp_index,
                            if stats { Some(&mut pre_probe) } else { None },
                        ) {
                            Ok(texture) => texture,
                            Err(error) => {
                                let message = format!("Pre-interpolation GPU chain: {error:#}");
                                if crate::render::onnx_stage::onnx_cancel_requested() {
                                    // A user Stop deliberately terminates the
                                    // in-flight ONNX run. Do not retain that
                                    // expected cancellation as a yellow filter
                                    // warning or a provider failure.
                                    log::info!(
                                        "pre-interpolation-chain-cancelled-by-stop: {error:#}"
                                    );
                                } else {
                                    log::error!("{message}");
                                    status.lock().unwrap().chain_errors.push(message);
                                }
                                input
                            }
                        }
                    } else {
                        input
                    };
                    let cur_tex = if dml_rife {
                        if let Some((limited_w, limited_h)) =
                            directml_rife_limited_size(cur_tex.w(), cur_tex.h())
                        {
                            let source_size = (cur_tex.w(), cur_tex.h());
                            match crate::render::scaler::resample(
                                &mut gc,
                                cur_tex,
                                limited_w,
                                limited_h,
                                crate::render::scaler::Kernel::Spline36,
                            ) {
                                Ok(limited) => {
                                    log_directml_rife_height_limit(
                                        s,
                                        source_size,
                                        (limited_w, limited_h),
                                        interp_index,
                                        "gpu-resident",
                                    );
                                    limited
                                }
                                Err(error) => {
                                    let message = format!(
                                        "DirectML RIFE height-limit resample failed: {error:#}"
                                    );
                                    log::error!("{message}");
                                    let mut state = status.lock().unwrap();
                                    if !state.chain_errors.contains(&message) {
                                        state.chain_errors.push(message);
                                    }
                                    drop(state);
                                    // Never feed an over-limit frame to DirectML RIFE after
                                    // the safety cap itself failed. Preserve the selected
                                    // post-chain and present the real pre-RIFE frame instead.
                                    let now = Instant::now();
                                    process_and_present_from(
                                        &mut gc,
                                        &mut overlay,
                                        s,
                                        &metrics,
                                        &status,
                                        cur_tex,
                                        out_size,
                                        stats,
                                        now,
                                        0,
                                        Some(FrameTiming::from_frame(&s.frame, now)),
                                        &[],
                                        downscaler,
                                        post_chain_start,
                                    );
                                    overlay.win.pump_messages();
                                    continue;
                                }
                            }
                        } else {
                            cur_tex
                        }
                    } else {
                        cur_tex
                    };
                    let dimensions_changed = s.gpu_interp_hist.front().is_some_and(|frame| {
                        (frame.tex.w(), frame.tex.h()) != (cur_tex.w(), cur_tex.h())
                    });
                    let continuity = s
                        .gpu_interp_hist
                        .back()
                        .map(|previous| {
                            classify_gpu_interp_continuity(
                                previous.seq,
                                previous.received_at,
                                previous.source_time_100ns,
                                &s.frame,
                                s.cadence.period_s(),
                                s.arrival_interval,
                            )
                        })
                        .unwrap_or(GpuInterpContinuity::Continuous);
                    if !dimensions_changed && continuity == GpuInterpContinuity::Duplicate {
                        // Never turn a compositor duplicate into a synthetic
                        // interpolation pair.  Keeping the previous endpoint is
                        // also what lets resize/ratio transitions recover without
                        // a stop/start cycle.
                        gc.recycle(cur_tex);
                        if s.frame.seq < 4 || s.frame.seq % 300 == 0 {
                            log::info!(
                                "interp-gpu-source-duplicate-skipped: seq={} history={} action=preserve",
                                s.frame.seq,
                                s.gpu_interp_hist.len()
                            );
                        }
                        overlay.win.pump_messages();
                        continue;
                    }
                    let continuity_broken = continuity == GpuInterpContinuity::Broken;
                    if dimensions_changed || continuity_broken {
                        while let Some(old) = s.gpu_interp_hist.pop_front() {
                            gc.recycle(old.tex);
                        }
                        s.interp_generation = s.interp_generation.saturating_add(1);
                        log::info!(
                            "interp-gpu-history-reset: generation={} dimensions_changed={} continuity_broken={}",
                            s.interp_generation,
                            dimensions_changed,
                            continuity_broken
                        );
                    }
                    s.gpu_interp_hist.push_back(GpuInterpFrame {
                        tex: cur_tex,
                        seq: s.frame.seq,
                        received_at: s.frame.received_at,
                        source_time_100ns: s.frame.source_time_100ns,
                    });
                    while s.gpu_interp_hist.len() > need {
                        if let Some(old) = s.gpu_interp_hist.pop_front() {
                            gc.recycle(old.tex);
                        }
                    }
                    let factor = interp_factor.clamp(2, 5);
                    let mut history = Vec::with_capacity(need);
                    history.extend(s.gpu_interp_hist.iter().map(|frame| frame.tex));
                    // DRBA is a delayed four-frame model.  Its interpolation
                    // interval is the middle pair [N-2, N-1], so duplicating
                    // startup frames or presenting N as its endpoint breaks
                    // temporal order and looks like severe judder.  RIFE is
                    // the ordinary two-frame [N-1, N] case.
                    let history_ready = if delayed {
                        s.gpu_interp_hist.len() >= need
                    } else {
                        s.gpu_interp_hist.len() >= 2
                    };
                    let source_period = s
                        .cadence
                        .period_s()
                        .or_else(|| (s.arrival_interval > 0.0).then_some(s.arrival_interval))
                        .unwrap_or(1.0 / 30.0);
                    let output_ratio = onnx_output_ratio(
                        factor,
                        Some(source_period),
                        s.monitor_refresh_hz,
                    );
                    let phases =
                        onnx_interpolation_phases(
                            factor,
                            output_ratio,
                            onnx_x3_60hz_mode(s.monitor_refresh_hz),
                            &mut s.flow_output_cadence,
                        );
                    let present_real = phases.last().is_some_and(|phase| *phase >= 1.0 - 1e-5);
                    let timesteps = phases
                        .into_iter()
                        .filter(|phase| *phase < 1.0 - 1e-5)
                        .collect::<Vec<_>>();
                    let output_period = source_period / output_ratio.max(1.0);
                    let history_keep = s.gpu_interp_hist.iter().map(|f| f.tex).collect::<Vec<_>>();
                    if !history_ready {
                        let now = Instant::now();
                        process_and_present_from(
                            &mut gc,
                            &mut overlay,
                            s,
                            &metrics,
                            &status,
                            cur_tex,
                            out_size,
                            stats,
                            now,
                            0,
                            Some(FrameTiming::from_frame(&s.frame, now)),
                            &history_keep,
                            downscaler,
                            post_chain_start,
                        );
                        overlay.win.pump_messages();
                        continue;
                    }
                    if timesteps.is_empty() {
                        let now = Instant::now();
                        process_and_present_from(
                            &mut gc,
                            &mut overlay,
                            s,
                            &metrics,
                            &status,
                            cur_tex,
                            out_size,
                            stats,
                            now,
                            0,
                            Some(FrameTiming::from_frame(&s.frame, now)),
                            &history_keep,
                            downscaler,
                            post_chain_start,
                        );
                        overlay.win.pump_messages();
                        continue;
                    }
                    let worker_matches_stage = s
                        .gpu_interp_worker
                        .as_ref()
                        .is_some_and(|worker| worker.matches_stage(&ist));
                    if !worker_matches_stage {
                        if let Some(mut old_worker) = s.gpu_interp_worker.take() {
                            let _ = old_worker.shutdown();
                            log::info!(
                                "interp-gpu-worker-replaced: reason=stage-or-backend-changed"
                            );
                        }
                        s.gpu_interp_worker = Some(GpuInterpWorker::spawn(ist.clone()));
                        let stage = ist.lock().unwrap();
                        log::info!(
                            "interp-execution-path: mode=gpu-resident backend={} kind={:?} handoff=nonblocking",
                            stage.provider_desc,
                            stage.interp
                        );
                    }
                    let pair_id = s.gpu_interp_pair_id;
                    s.gpu_interp_pair_id = s.gpu_interp_pair_id.saturating_add(1);
                    let endpoint = if delayed {
                        s.gpu_interp_hist
                            .iter()
                            .rev()
                            .nth(1)
                            .copied()
                            .expect("complete DRBA history")
                    } else {
                        s.gpu_interp_hist
                            .back()
                            .copied()
                            .expect("complete RIFE history")
                    };
                    let mut endpoint_frame = s.frame.clone();
                    endpoint_frame.seq = endpoint.seq;
                    endpoint_frame.received_at = endpoint.received_at;
                    endpoint_frame.source_time_100ns = endpoint.source_time_100ns;
                    let start_source_time_100ns = if delayed {
                        s.gpu_interp_hist
                            .iter()
                            .rev()
                            .nth(2)
                            .and_then(|frame| frame.source_time_100ns)
                    } else {
                        s.gpu_interp_hist
                            .iter()
                            .rev()
                            .nth(1)
                            .and_then(|frame| frame.source_time_100ns)
                    };
                    let (prepared, output_slots) = {
                        let mut stage = ist.lock().unwrap();
                        let post_glsl = s.chain.has_glsl_in_range(post_chain_start);
                        // Keep real frames and generated frames on the same
                        // texture contract before post-GLSL processing. The
                        // former and TensorRT output already cross an RGBA8
                        // boundary; preserving only DirectML's raw FP16 result
                        // exposed sub-8-bit RIFE noise which Anime4K amplified
                        // into a visible cell/dot pattern every other frame.
                        // Conversion remains GPU-resident and adds no CPU copy.
                        let preserve_float_after_dml = false;
                        let prepared = stage.prepare_interp_gpu_textures_with_factor(
                            &mut gc,
                            &history,
                            &timesteps,
                            s.interp_generation,
                            factor as usize,
                            post_glsl,
                        )?;
                        let slots = if prepared.is_some() {
                            stage.prepared_interp_gpu_outputs(
                                timesteps.len(),
                                preserve_float_after_dml,
                            )?
                        } else {
                            Vec::new()
                        };
                        if pair_id < 3
                            && post_glsl
                            && matches!(
                                stage.provider,
                                crate::render::onnx_stage::OnnxProvider::DirectML
                            )
                        {
                            log::info!(
                                "interp-post-handoff: backend=DirectML input=NCHW-FP output=RGBA8 post_glsl=true quantize_before_post=true parity=real-frame+TensorRT"
                            );
                        }
                        (prepared, slots)
                    };
                    let ready_outputs = vec![None; timesteps.len()];
                    let (cooperative_slots, permit_next_after_post_submit) =
                        gpu_interp_slot_policy(factor, timesteps.len());
                    let pending = PendingGpuInterp {
                        generation: s.interp_generation,
                        pair_id,
                        factor,
                        cooperative_slots,
                        permit_next_after_post_submit,
                        stage: ist.clone(),
                        metric_name: interp_name.clone(),
                        real_tex: endpoint.tex,
                        history_keep,
                        timesteps,
                        output_slots,
                        ready_outputs,
                        next_output_index: 0,
                        completed_outputs: 0,
                        run_ms_total: 0.0,
                        present_real,
                        post_chain_start,
                        frame: endpoint_frame,
                        start_source_time_100ns,
                        output_period,
                        out_size,
                        submitted_at: Instant::now(),
                        slow_wait_logged: false,
                    };
                    match prepared {
                        Some(fence) => {
                            if pair_id < 3 {
                                log::info!(
                                    "interp-gpu-input-pack-issued: pair={} generation={} fence={}",
                                    pair_id,
                                    s.interp_generation,
                                    fence
                                );
                            }
                            let job = GpuInterpJob {
                                generation: s.interp_generation,
                                pair_id,
                                count: pending.timesteps.len(),
                                cooperative_slots: pending.cooperative_slots,
                            };
                            s.gpu_interp_pack_pending = Some(PendingGpuPack {
                                fence,
                                started: Instant::now(),
                                job,
                                pending,
                            });
                            overlay.win.pump_messages();
                            continue;
                        }
                        None => {}
                    }
                    // The stage disabled its GPU path for this session. Return
                    // the temporary GL history before entering CPU fallback.
                    while let Some(old) = s.gpu_interp_hist.pop_front() {
                        gc.recycle(old.tex);
                    }
                } else if matches!(
                    std::env::var("CHIDESCALER_INTERP_GPU").ok().as_deref(),
                    Some("require")
                ) && !s.frame.hdr
                {
                    anyhow::bail!(
                        "interpolation GPU residency required but unavailable: backend={} model={} reason=unsupported model descriptor",
                        ist.lock().unwrap().provider_desc,
                        interp_name
                    );
                }
                let conv_t0 = Instant::now();
                let (mut interp_w, mut interp_h, mut cur_data) = if s.frame.hdr {
                    (s.frame.w, s.frame.h, Vec::new())
                } else if interp_index > 0 {
                    let input = upload_frame_timed(&mut gc, &s.frame, &mut s.last_upload_submit_ms);
                    let m = metrics.clone();
                    let mut pre_probe = |name: &str, kind: StageKind, ms: f64| {
                        m.probe(
                            name,
                            if kind == StageKind::Glsl {
                                "glsl"
                            } else {
                                "onnx"
                            },
                            ms,
                        );
                    };
                    match s.chain.process_range(
                        &mut gc,
                        input,
                        out_size,
                        0,
                        interp_index,
                        if stats { Some(&mut pre_probe) } else { None },
                    ) {
                        Ok(texture) => {
                            let data = if rgba_direct {
                                gc.download_rgba8(texture)
                            } else {
                                gc.download_rgb8(texture)
                            };
                            let route_key =
                                (s.frame.w, s.frame.h, texture.w(), texture.h(), interp_index);
                            if s.pre_chain_route_log_key != Some(route_key) {
                                log::info!(
                                    "pre-chain readback path: source={}x{} rife_input={}x{} stages={}",
                                    s.frame.w,
                                    s.frame.h,
                                    texture.w(),
                                    texture.h(),
                                    interp_index
                                );
                                s.pre_chain_route_log_key = Some(route_key);
                            }
                            (texture.w(), texture.h(), data)
                        }
                        Err(error) => {
                            let message = format!("Pre-interpolation chain: {error:#}");
                            if crate::render::onnx_stage::onnx_cancel_requested() {
                                log::info!(
                                    "pre-interpolation-cpu-chain-cancelled-by-stop: {error:#}"
                                );
                            } else {
                                log::error!("{message}");
                                let mut state = status.lock().unwrap();
                                if !state.chain_errors.contains(&message) {
                                    state.chain_errors.push(message);
                                }
                            }
                            if rgba_direct {
                                (s.frame.w, s.frame.h, s.frame.data.clone())
                            } else {
                                (s.frame.w, s.frame.h, rgba_to_rgb(&s.frame.data))
                            }
                        }
                    }
                } else if rgba_direct {
                    (s.frame.w, s.frame.h, s.frame.data.clone())
                } else {
                    (s.frame.w, s.frame.h, rgba_to_rgb(&s.frame.data))
                };
                if dml_rife && !s.frame.hdr {
                    if let Some((limited_w, limited_h)) =
                        directml_rife_limited_size(interp_w, interp_h)
                    {
                        // RifeV1/DRBA use the RGBA-direct engine path. For the
                        // DirectML RIFE cap keep the real endpoints and generated
                        // midpoints in the same limited geometry so cadence never
                        // alternates between full-size and capped frames.
                        if rgba_direct {
                            if let Some(limited) = resize_rgba8_bilinear(
                                &cur_data,
                                interp_w,
                                interp_h,
                                limited_w,
                                limited_h,
                            ) {
                                let source_size = (interp_w, interp_h);
                                interp_w = limited_w;
                                interp_h = limited_h;
                                cur_data = limited;
                                log_directml_rife_height_limit(
                                    s,
                                    source_size,
                                    (limited_w, limited_h),
                                    interp_index,
                                    "cpu-visible",
                                );
                            } else {
                                let message = format!(
                                    "DirectML RIFE CPU height-limit resize failed: {}x{} -> {}x{}",
                                    interp_w, interp_h, limited_w, limited_h
                                );
                                log::error!("{message}");
                                let mut state = status.lock().unwrap();
                                if !state.chain_errors.contains(&message) {
                                    state.chain_errors.push(message);
                                }
                            }
                        }
                    }
                }
                let cur_frame: Arc<Vec<u8>> = Arc::new(cur_data);
                if stats && !s.frame.hdr {
                    if rgba_direct {
                        metrics.probe(
                            "interp frame copy",
                            "cpu",
                            conv_t0.elapsed().as_secs_f64() * 1000.0,
                        );
                    } else {
                        metrics.probe(
                            "interp RGBA->RGB",
                            "cpu",
                            conv_t0.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
                let dims_ok = s.hist.iter().all(|f| f.w == interp_w && f.h == interp_h);
                if !dims_ok {
                    s.hist.clear();
                    s.interp_worker = None;
                    s.interp_pending = None;
                }
                let factor = interp_factor.max(2);
                // ---- PIPELINED path: RIFE and 4-frame delayed DRBA ----
                // Inference for pair (N-1, N) runs on the worker while THIS
                // tick presents the previous pair's mids + real frame, hiding
                // the ~30-40ms DML run inside the source period (48fps pin).
                if ((!delayed && need == 2) || (delayed && need == 4)) && !s.frame.hdr {
                    let pair_dt = s.hist.back().and_then(|prev| {
                        normalized_frame_span_s(
                            prev.source_time_100ns,
                            s.frame.source_time_100ns,
                            prev.seq,
                            s.frame.seq,
                        )
                        .or_else(|| {
                            match (prev.received_at, s.frame.received_at) {
                                (Some(a), Some(b)) if b >= a => {
                                    Some(b.duration_since(a).as_secs_f64())
                                }
                                _ => None,
                            }
                        })
                    });
                    if s.hist.back().is_some()
                        && !interp_pair_is_contiguous(
                            pair_dt,
                            s.cadence.period_s(),
                            s.arrival_interval,
                        )
                    {
                        log::info!(
                            "interp discontinuity: pair_dt_ms={:.1}; discard stale mids and present latest real frame",
                            pair_dt.unwrap_or(0.0) * 1000.0
                        );
                        s.interp_worker = None;
                        s.interp_pending = None;
                        s.hist.clear();
                        s.interp_last_tick = None;
                        s.interp_present_deadline = None;
                        s.flow_output_cadence.reset();
                        s.arrival_interval = 0.0;
                        s.cadence.reset();
                        s.smooth_pacer.reset();
                        s.paced_present_deadline = None;
                        let delivered = s.source.delivered();
                        let captures = delivered.saturating_sub(s.metric_seq) as u32;
                        s.metric_seq = delivered;
                        let rt = Instant::now();
                        let tex = if rgba_direct {
                            gc.upload_rgba8(interp_w, interp_h, &cur_frame)
                        } else {
                            gc.upload_rgb8(interp_w, interp_h, &cur_frame)
                        };
                        process_and_present_from(
                            &mut gc,
                            &mut overlay,
                            s,
                            &metrics,
                            &status,
                            tex,
                            out_size,
                            stats,
                            rt,
                            captures,
                            Some(FrameTiming::from_frame(&s.frame, rt)),
                            &[],
                            downscaler,
                            post_chain_start,
                        );
                        s.hist.push_back(HistFrame::from_processed(
                            &s.frame, interp_w, interp_h, cur_frame,
                        ));
                        overlay.win.pump_messages();
                        s.interp_tail = Some(Instant::now());
                        continue;
                    }
                    // 1) (re)create the worker if the stage instance changed
                    let stage_ptr = Arc::as_ptr(&ist) as usize;
                    if !s.interp_post_warm && s.chain.has_onnx_in_range(post_chain_start) {
                        let warm_t0 = Instant::now();
                        let mut warm_ok = true;
                        // Two hidden passes were sufficient to move the
                        // reproduced AnimeJaNai path from ~22.5 ms cold to
                        // ~20 ms steady-state. Actual displayed frames still
                        // execute the GUI chain in its declared order.
                        for _ in 0..2 {
                            let tex = if rgba_direct {
                                gc.upload_rgba8(interp_w, interp_h, &cur_frame)
                            } else {
                                gc.upload_rgb8(interp_w, interp_h, &cur_frame)
                            };
                            if let Err(error) =
                                s.chain
                                    .process_from(&mut gc, tex, out_size, post_chain_start, None)
                            {
                                warm_ok = false;
                                log::warn!(
                                    "interpolation post-chain warm-up skipped after error: {error:#}"
                                );
                                break;
                            }
                            gc.release_frame(&[]);
                        }
                        s.interp_post_warm = true;
                        s.interp_present_deadline = None;
                        s.flow_output_cadence.reset();
                        log::info!(
                            "interpolation post-chain warm-up: passes={} elapsed_ms={:.2}",
                            if warm_ok { 2 } else { 0 },
                            warm_t0.elapsed().as_secs_f64() * 1000.0
                        );
                    } else if !s.interp_post_warm {
                        s.interp_post_warm = true;
                    }
                    if s.interp_worker.as_ref().map(|w| w.stage_ptr) != Some(stage_ptr) {
                        s.interp_worker = Some(InterpWorker::spawn(ist.clone(), stage_ptr));
                    }
                    // 2) submit the new pair FIRST, so the ~27ms DML run
                    //    starts NOW and finishes well before the source's
                    //    next DWM composition. Submitting after the paced
                    //    presentation made the inference overlap that
                    //    composition on the shared GPU — WGC delivery slipped
                    //    by the inference time every frame and the take rate
                    //    fell to ~15/s (measured wait_take_ms≈38).
                    let job_inputs = if delayed {
                        (s.hist.len() + 1 >= need).then(|| {
                            let mut frames: Vec<Arc<Vec<u8>>> =
                                s.hist.iter().map(|frame| frame.data.clone()).collect();
                            frames.push(cur_frame.clone());
                            let real = s.hist.back().cloned().expect("DRBA history");
                            (frames, real)
                        })
                    } else {
                        s.hist.back().map(|prev| {
                            (
                                vec![prev.data.clone(), cur_frame.clone()],
                                HistFrame::from_processed(
                                    &s.frame,
                                    interp_w,
                                    interp_h,
                                    cur_frame.clone(),
                                ),
                            )
                        })
                    };
                    let (submitted, next_pending) = if let Some((frames, real)) = job_inputs {
                        let source_period_s =
                            s.cadence.period_s().or(pair_dt).or_else(|| {
                                (s.arrival_interval > 0.0).then_some(s.arrival_interval)
                            });
                        let output_ratio = onnx_output_ratio(
                            factor,
                            source_period_s,
                            s.monitor_refresh_hz,
                        );
                        let phases = onnx_interpolation_phases(
                            factor,
                            output_ratio,
                            onnx_x3_60hz_mode(s.monitor_refresh_hz),
                            &mut s.flow_output_cadence,
                        );
                        let present_real = phases.last().is_some_and(|phase| *phase >= 1.0 - 1e-5);
                        let ts: Vec<f32> = phases
                            .into_iter()
                            .filter(|phase| *phase < 1.0 - 1e-5)
                            .collect();
                        let start_source_time_100ns = if delayed {
                            s.hist
                                .iter()
                                .rev()
                                .nth(1)
                                .and_then(|frame| frame.source_time_100ns)
                        } else {
                            s.hist.back().and_then(|frame| frame.source_time_100ns)
                        };
                        let output_period_s =
                            source_period_s.map(|period| period / output_ratio.max(1.0));
                        let job = InterpJob {
                            w: interp_w,
                            h: interp_h,
                            frames,
                            ts: ts.clone(),
                            rgba: rgba_direct,
                        };
                        let submitted = s
                            .interp_worker
                            .as_ref()
                            .and_then(|worker| worker.job_tx.as_ref())
                            .is_some_and(|tx| tx.send(job).is_ok());
                        (
                            submitted,
                            Some(PendingInterp {
                                interp_name: interp_name.clone(),
                                real,
                                count: ts.len(),
                                rgba: rgba_direct,
                                present_real,
                                mid_ts: ts,
                                output_period_s,
                                start_source_time_100ns,
                                pair_dt,
                                chain_start_index: post_chain_start,
                            }),
                        )
                    } else {
                        (false, None)
                    };
                    // The WGC frame arrival is already the source cadence. A
                    // second full-period wait here starves x3/x4 and can also
                    // pull x2 below its target. Pace only the mids inside the
                    // interval in drain_pending_interp().
                    let paced_tick_t0 = Instant::now();
                    // 3) present the PREVIOUS pair (its inference finished
                    //    during the last period; the worker is FIFO so the
                    //    results read here belong to that pair, not the job
                    //    just submitted)
                    drain_pending_interp(
                        &mut gc,
                        &mut overlay,
                        &input,
                        s,
                        &metrics,
                        &status,
                        out_size,
                        stats,
                        vsync_on,
                        paced_tick_t0,
                        downscaler,
                    );
                    s.interp_pending = submitted.then_some(next_pending).flatten();
                    if s.interp_pending.is_none() {
                        // very first frame: nothing to interpolate against —
                        // present it directly so the view appears instantly
                        let delivered = s.source.delivered();
                        let captures = delivered.saturating_sub(s.metric_seq) as u32;
                        s.metric_seq = delivered;
                        let rt = Instant::now();
                        let tex = if rgba_direct {
                            gc.upload_rgba8(interp_w, interp_h, &cur_frame)
                        } else {
                            gc.upload_rgb8(interp_w, interp_h, &cur_frame)
                        };
                        process_and_present_from(
                            &mut gc,
                            &mut overlay,
                            s,
                            &metrics,
                            &status,
                            tex,
                            out_size,
                            stats,
                            rt,
                            captures,
                            Some(FrameTiming::from_frame(&s.frame, rt)),
                            &[],
                            downscaler,
                            post_chain_start,
                        );
                    }
                    s.hist.push_back(HistFrame::from_processed(
                        &s.frame, interp_w, interp_h, cur_frame,
                    ));
                    while s.hist.len() > need.saturating_sub(1).max(1) {
                        s.hist.pop_front();
                    }
                    overlay.win.pump_messages();
                    s.interp_tail = Some(Instant::now());
                    continue;
                }
                // ---- legacy sync path (delayed DRBA / multi-frame models) ----
                // if the chain already takes longer than a frame interval,
                // synthesizing extra frames only makes it worse — skip them
                let behind = s.arrival_interval > 0.0
                    && s.last_process_ms / 1000.0 > s.arrival_interval * 0.9;
                let can_interpolate = s.hist.len() + 1 >= need && !behind;
                let will_present_delayed = delayed && s.hist.len() >= 3;
                if s.last_neoflow_log.elapsed() >= Duration::from_secs(1) {
                    log::info!(
                        "onnx-interp-state: name={} factor={} need={} hist={} delayed={} behind={} can_interpolate={} arrival_ms={:.2} last_process_ms={:.2}",
                        interp_name,
                        factor,
                        need,
                        s.hist.len(),
                        delayed,
                        behind,
                        can_interpolate,
                        s.arrival_interval * 1000.0,
                        s.last_process_ms
                    );
                    s.last_neoflow_log = Instant::now();
                }
                let t0 = if can_interpolate || will_present_delayed {
                    frame_paced = true;
                    Instant::now()
                } else {
                    Instant::now()
                };
                if can_interpolate {
                    let mut interp_total_ms = 0.0;
                    if delayed && factor > 2 {
                        let interp_t0 = Instant::now();
                        let ts: Vec<f32> = (1..factor).map(|k| k as f32 / factor as f32).collect();
                        let results = {
                            let mut frames: Vec<&[u8]> =
                                s.hist.iter().map(|f| f.data.as_slice()).collect();
                            frames.push(&cur_frame);
                            let mut ist = ist.lock().unwrap();
                            let results =
                                ist.process_interp_many_rgba8(s.frame.w, s.frame.h, &frames, &ts);
                            if stats {
                                if let Some(p) = ist.last_interp_profile() {
                                    metrics.probe("Frame interpolation pack", "cpu", p.pack_ms);
                                    metrics.probe("Frame interpolation run", "onnx", p.run_ms);
                                    metrics.probe("Frame interpolation output", "cpu", p.out_ms);
                                }
                            }
                            results
                        };
                        interp_total_ms = interp_t0.elapsed().as_secs_f64() * 1000.0;
                        match results {
                            Ok(results) => {
                                for (index, (mw, mh, mid)) in results.into_iter().enumerate() {
                                    let k = index + 1;
                                    let mid_t0 = Instant::now();
                                    let tex = gc.upload_rgb8(mw, mh, &mid);
                                    process_and_present(
                                        &mut gc,
                                        &mut overlay,
                                        s,
                                        &metrics,
                                        &status,
                                        tex,
                                        out_size,
                                        stats,
                                        mid_t0,
                                        0,
                                        Some(FrameTiming::from_frame(&s.frame, mid_t0)),
                                        &[],
                                        downscaler,
                                    );
                                    let pacing_period = if s.smooth_pacing {
                                        s.cadence.period_s().unwrap_or(s.arrival_interval)
                                    } else {
                                        s.arrival_interval
                                    };
                                    let target =
                                        (pacing_period * k as f64 / factor as f64).min(0.08);
                                    let spent = t0.elapsed().as_secs_f64();
                                    if !vsync_on && target > spent {
                                        wait_until_with_pump(
                                            &mut overlay,
                                            Instant::now()
                                                + Duration::from_secs_f64(target - spent),
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                let msg = format!("Frame interpolation: {e:#}");
                                if crate::render::onnx_stage::onnx_cancel_requested() {
                                    log::info!("frame-interpolation-cancelled-by-stop: {e:#}");
                                } else {
                                    let mut g = status.lock().unwrap();
                                    if !g.chain_errors.contains(&msg) {
                                        g.chain_errors.push(msg);
                                    }
                                }
                            }
                        }
                    } else {
                        for k in 1..factor {
                            let t = k as f32 / factor as f32;
                            // scope the hist borrows so process_and_present can
                            // take &mut s afterwards
                            let interp_t0 = Instant::now();
                            let r = {
                                let mut frames: Vec<&[u8]> =
                                    s.hist.iter().map(|f| f.data.as_slice()).collect();
                                frames.push(&cur_frame);
                                let mut ist = ist.lock().unwrap();
                                let r = if rgba_direct {
                                    ist.process_interp_rgba8(s.frame.w, s.frame.h, &frames, t)
                                } else {
                                    ist.process_interp(s.frame.w, s.frame.h, &frames, t)
                                };
                                if stats {
                                    if let Some(p) = ist.last_interp_profile() {
                                        metrics.probe("Frame interpolation pack", "cpu", p.pack_ms);
                                        metrics.probe("Frame interpolation run", "onnx", p.run_ms);
                                        metrics.probe(
                                            "Frame interpolation output",
                                            "cpu",
                                            p.out_ms,
                                        );
                                    }
                                }
                                r
                            };
                            if stats {
                                interp_total_ms += interp_t0.elapsed().as_secs_f64() * 1000.0;
                            }
                            match r {
                                Ok((mw, mh, mid)) => {
                                    let interp_timing =
                                        FrameTiming::from_frame(&s.frame, interp_t0);
                                    let upload_t0 = Instant::now();
                                    let tex = gc.upload_rgb8(mw, mh, &mid);
                                    if stats {
                                        metrics.probe(
                                            "interp RGB upload",
                                            "gpu",
                                            upload_t0.elapsed().as_secs_f64() * 1000.0,
                                        );
                                    }
                                    process_and_present(
                                        &mut gc,
                                        &mut overlay,
                                        s,
                                        &metrics,
                                        &status,
                                        tex,
                                        out_size,
                                        stats,
                                        interp_t0,
                                        0, // synthetic frame
                                        Some(interp_timing),
                                        &[],
                                        downscaler,
                                    );
                                    // pace so mids are evenly spread over the interval
                                    let pacing_period = if s.smooth_pacing {
                                        s.cadence.period_s().unwrap_or(s.arrival_interval)
                                    } else {
                                        s.arrival_interval
                                    };
                                    let target =
                                        (pacing_period * k as f64 / factor as f64).min(0.08);
                                    let spent = t0.elapsed().as_secs_f64();
                                    if !vsync_on && target > spent {
                                        wait_until_with_pump(
                                            &mut overlay,
                                            Instant::now()
                                                + Duration::from_secs_f64(target - spent),
                                        );
                                    }
                                }
                                Err(e) => {
                                    let msg = format!("補間: {e:#}");
                                    if crate::render::onnx_stage::onnx_cancel_requested() {
                                        log::info!("interpolation-cancelled-by-stop: {e:#}");
                                    } else {
                                        let mut g = status.lock().unwrap();
                                        if !g.chain_errors.contains(&msg) {
                                            g.chain_errors.push(msg);
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    if stats && interp_total_ms > 0.0 {
                        metrics.probe(&interp_name, "onnx", interp_total_ms);
                    }
                }
                // DRBA runs one frame behind: after the in-betweens, present
                // the DELAYED real frame (hist[-1] = f(n+1)), not the newest
                if delayed && s.hist.len() >= 3 {
                    let real_t0 = Instant::now();
                    let delayed_frame = s.hist.back().cloned().unwrap();
                    let tex = if rgba_direct {
                        gc.upload_rgba8(delayed_frame.w, delayed_frame.h, &delayed_frame.data)
                    } else {
                        gc.upload_rgb8(delayed_frame.w, delayed_frame.h, &delayed_frame.data)
                    };
                    let delivered = s.source.delivered();
                    let captures = delivered.saturating_sub(s.metric_seq) as u32;
                    s.metric_seq = delivered;
                    process_and_present(
                        &mut gc,
                        &mut overlay,
                        s,
                        &metrics,
                        &status,
                        tex,
                        out_size,
                        stats,
                        real_t0,
                        captures,
                        Some(delayed_frame.timing(real_t0)),
                        &[],
                        downscaler,
                    );
                    s.hist.push_back(HistFrame::from_frame(&s.frame, cur_frame));
                    while s.hist.len() > 3 {
                        s.hist.pop_front();
                    }
                    overlay.win.pump_messages();
                    continue;
                }
                s.hist.push_back(HistFrame::from_frame(&s.frame, cur_frame));
                while s.hist.len() > need.saturating_sub(1).max(1) {
                    s.hist.pop_front();
                }
            } else {
                // interpolation stage removed from the chain: drop history
                // and the pipeline (worker drop discards queued results)
                if !s.hist.is_empty() {
                    s.hist.clear();
                }
                if s.interp_worker.is_some() || s.interp_pending.is_some() {
                    s.interp_worker = None;
                    s.interp_pending = None;
                }
            }

            // capture fps = frames DELIVERED by WGC (incl. dropped), not consumed
            let delivered = s.source.delivered();
            let captures = delivered.saturating_sub(s.metric_seq) as u32;
            s.metric_seq = delivered;
            let real_t0 = if frame_paced {
                Instant::now()
            } else {
                pace_processing_start(s, &mut overlay)
            };
            if crate::logging::diagnostics_enabled() && captures > 1 && s.phase_diag_samples < 12 {
                log::debug!(
                    "frame-phase diag: captures={} seq={} since_present_ms={:.2} setup_ms={:.2} cursor_ms={:.2} watchdog_ms={:.2} zorder_ms={:.2} geometry_ms={:.2} panel_ms={:.2} foreground_ms={:.2} input_ms={:.2} post_input_ms={:.2} capture_wait_take_ms={:.2} pre_process_ms={:.2}",
                    captures,
                    s.frame.seq,
                    s.last_present.elapsed().as_secs_f64() * 1000.0,
                    phase_capture_started
                        .duration_since(phase_tick_started)
                        .as_secs_f64()
                        * 1000.0,
                    phase_cursor_finished
                        .duration_since(phase_tick_started)
                        .as_secs_f64()
                        * 1000.0,
                    phase_watchdog_finished
                        .duration_since(phase_cursor_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_zorder_finished
                        .duration_since(phase_watchdog_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_geometry_finished
                        .duration_since(phase_zorder_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_panel_finished
                        .duration_since(phase_geometry_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_input_started
                        .duration_since(phase_panel_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_input_finished
                        .duration_since(phase_input_started)
                        .as_secs_f64()
                        * 1000.0,
                    phase_capture_started
                        .duration_since(phase_input_finished)
                        .as_secs_f64()
                        * 1000.0,
                    phase_capture_finished
                        .duration_since(phase_capture_started)
                        .as_secs_f64()
                        * 1000.0,
                    real_t0.duration_since(phase_capture_finished).as_secs_f64() * 1000.0
                );
                s.phase_diag_samples += 1;
            }
            let direct_onnx = if s.frame.hdr {
                Ok(None)
            } else {
                s.chain
                    .process_first_onnx_rgba8(&mut gc, s.frame.w, s.frame.h, &s.frame.data)
            };
            match direct_onnx {
                Ok(Some((input, name, ms))) => {
                    if stats {
                        metrics.probe(&name, "onnx", ms);
                    }
                    process_and_present_from(
                        &mut gc,
                        &mut overlay,
                        s,
                        &metrics,
                        &status,
                        input,
                        out_size,
                        stats,
                        real_t0,
                        captures,
                        Some(FrameTiming::from_frame(&s.frame, real_t0)),
                        &[],
                        downscaler,
                        1,
                    );
                }
                Ok(None) | Err(_) => {
                    let input = upload_frame_timed(&mut gc, &s.frame, &mut s.last_upload_submit_ms);
                    process_and_present(
                        &mut gc,
                        &mut overlay,
                        s,
                        &metrics,
                        &status,
                        input,
                        out_size,
                        stats,
                        real_t0,
                        captures,
                        Some(FrameTiming::from_frame(&s.frame, real_t0)),
                        &[],
                        downscaler,
                    );
                }
            }
        } else if s.interp_pending.is_some()
            && s.last_arrival.elapsed().as_secs_f64() > (s.arrival_interval * 2.5).max(0.15)
        {
            // The source REALLY stopped delivering (video paused / static
            // content): present the last pair so the stream ends on the
            // newest frame. The quiet-time gate is essential — wait_new's
            // 20ms timeout fires INSIDE every normal frame gap (41.7ms at
            // 24p), and draining here each gap serialized the pipeline down
            // well below the delivered rate when pacing is constrained.
            let out_size = virtual_shader_output_size((s.frame.w, s.frame.h));
            let stats = metrics.detailed_enabled() && s.frame.seq % 30 == 0;
            drain_pending_interp(
                &mut gc,
                &mut overlay,
                &input,
                s,
                &metrics,
                &status,
                out_size,
                stats,
                vsync_on,
                Instant::now(),
                downscaler,
            );
        } else if s.smooth_pacing
            && !interp_active
            && s.last_tex.is_some()
            && s.last_arrival.elapsed() < Duration::from_millis(250)
        {
            // WGC occasionally exposes a 33.3 ms source-timestamp gap even
            // though the callback completes in under 1 ms.  Keep presentation
            // on the source cadence by reusing the already filtered texture
            // for the missing display slot.  Capture metrics remain honest:
            // this is a presentation repeat, not a fabricated captured frame.
            let source_period_s = s
                .cadence
                .period_s()
                .unwrap_or(s.arrival_interval)
                .clamp(1.0 / 240.0, 0.1);
            let period = Duration::from_secs_f64(source_period_s);
            // A 24p source must not receive an extra 48fps swap in every
            // normal inter-frame gap. Repeat only after a genuinely missed
            // source slot; DWM retains the completed frame before then.
            if s.last_arrival.elapsed() >= period.mul_f64(1.5) && s.last_present.elapsed() >= period
            {
                if let Some(t) = s.last_tex
                    && overlay.present(&mut gc, t).is_ok()
                {
                    let now = Instant::now();
                    s.last_present = now;
                    s.present_cadence.record(now);
                    metrics.present_only();
                    s.smooth_pacer
                        .reanchor_after_gap_fill_present(now, Some(source_period_s));
                    s.paced_present_deadline = None;
                    if crate::logging::diagnostics_enabled() {
                        log::debug!(
                            "smooth-pacing gap-fill reanchor: cadence_ms={:.2} next_gap_ms={:.2}",
                            source_period_s * 1000.0,
                            source_period_s * 1000.0
                        );
                    }
                }
            }
        } else if s.last_present.elapsed() > IDLE_KEEPALIVE_INTERVAL {
            // Idle keepalive: re-present the completed frame at ~1fps. DWM
            // normally retains the surface; this low-rate safety refresh avoids
            // needless 5fps swaps while WGC correctly reports a static 0fps.
            if let Some(t) = s.last_tex {
                let _ = overlay.present(&mut gc, t);
            }
            s.last_present = Instant::now();
            metrics.frame(true, 0, None, None, s.in_size, (0, 0));
        }
        overlay.win.pump_messages();
    }
}

fn start_session(
    hwnd: isize,
    specs: &[StageSpec],
    mode: ScaleMode,
    ratio: f32,
    fps_cap: Option<u32>,
    hide_source: bool,
    client_only: bool,
    hdr: bool,
    hdr_sdr_mode: HdrSdrMode,
    // Immutable outer-window rectangle captured before any Neo-initiated
    // capture-resolution resize. Never rebase this during the session.
    source_restore_rect: Option<(i32, i32, i32, i32)>,
    // Maximized state paired with the immutable session-origin rectangle.
    source_was_maximized: bool,
    deferred_capture_resolution: Option<((u32, u32), (u32, u32))>,
    capture_canvas: Option<(u32, u32)>,
    smooth_pacing: bool,
    duplicate_frame_reduction: bool,
    factory: &mut StageFactory,
) -> Result<(Session, Vec<String>, Option<String>)> {
    // Defensive invariant: WGC edge-pads odd dimensions before the frame
    // reaches the engine. Keep the source client, transition target and input
    // canvas on that same even geometry so an odd PIP aspect-fit can never
    // leave the overlay waiting for an impossible frame size.
    let deferred_capture_resolution = deferred_capture_resolution.map(|(requested, applied)| {
        let normalized = normalize_capture_client_size(applied);
        if normalized != applied {
            log::info!(
                "capture-start-even-normalize: requested={}x{} applied={}x{} normalized={}x{}",
                requested.0,
                requested.1,
                applied.0,
                applied.1,
                normalized.0,
                normalized.1
            );
        }
        (requested, normalized)
    });
    let capture_canvas = deferred_capture_resolution
        .map(|(_, applied)| applied)
        .or_else(|| capture_canvas.map(normalize_capture_client_size));
    let monitor_rect = win32::monitor_rect_of(hwnd);
    let monitor_refresh_hz = win32::monitor_refresh_hz(hwnd);
    // A monitor-covering source is already fullscreen presentation content.
    // Cropping it again through Win32/DWM client coordinates can introduce a
    // 1px x/y origin shift on geometry-sensitive fullscreen applications.
    // Preserve the WGC frame exactly and let the capture layer only edge-pad
    // an odd final row/column. Explicit capture canvases belong to the
    // window/PIP path and keep the existing client-area semantics.
    let source_monitor_fullscreen = source_restore_rect
        .map(|rect| win32::is_rect_monitor_fullscreen(hwnd, rect))
        .unwrap_or_else(|| win32::is_monitor_fullscreen(hwnd));
    let effective_client_only =
        client_only && !(source_monitor_fullscreen && capture_canvas.is_none());
    if client_only != effective_client_only {
        log::info!(
            "fullscreen-pixel-exact: hwnd={hwnd:#x} client_crop=bypassed origin=(0,0) edge_pad=right/bottom-only reason=monitor-cover-source"
        );
    }
    let display_aspect = capture_canvas
        .map(|(w, h)| (w as i32, h as i32))
        .unwrap_or_else(|| {
            if effective_client_only {
                win32::client_rect_on_screen(hwnd)
            } else {
                win32::window_rect(hwnd)
            }
            .map(|(_, _, w, h)| (w, h))
            .unwrap_or((0, 0))
        });
    log::info!(
        "capture monitor refresh: {:.2}Hz",
        monitor_refresh_hz.unwrap_or(0.0)
    );
    log::info!(
        "capture-path-diag: source_pid={} monitor=({},{} {}x{}) refresh_hz={:.2} scale_mode={:?} ratio={:.3} hide_source={} client_only={} fps_cap={:?} smooth={} duplicate_reduction={}",
        win32::window_pid(hwnd),
        monitor_rect.0,
        monitor_rect.1,
        monitor_rect.2,
        monitor_rect.3,
        monitor_refresh_hz.unwrap_or(0.0),
        mode,
        ratio,
        hide_source,
        effective_client_only,
        fps_cap,
        smooth_pacing,
        duplicate_frame_reduction,
    );
    // UIPI: an elevated source ignores our confined clicks — refuse instead
    // of confusing the user with a half-working session
    let source_is_elevated = matches!(win32::process_elevated(win32::window_pid(hwnd)), Some(true));
    if source_is_elevated && !win32::own_process_elevated() {
        anyhow::bail!(
            "このウィンドウは管理者権限で動作しているため拡大できません。GUIの「管理者として再起動」をオンにしてから再度お試しください"
        );
    }
    // The user-facing HDR option now reuses the same fast RGBA8 WGC path as
    // HDR OFF. Only a small post-capture highlight knee is enabled below.
    let pixel_exact_raw_hint = if source_monitor_fullscreen && capture_canvas.is_none() {
        source_restore_rect
            .or_else(|| win32::window_rect(hwnd))
            .and_then(|(_, _, w, h)| (w > 0 && h > 0).then_some((w as u32, h as u32)))
    } else {
        None
    };
    let source = WgcSource::start_fmt_with_pixel_lock(
        hwnd,
        fps_cap,
        effective_client_only,
        false,
        pixel_exact_raw_hint,
    )?;
    if let Some((w, h)) = pixel_exact_raw_hint {
        log::info!(
            "fullscreen-pixel-exact-raw-lock: hwnd={hwnd:#x} raw_hint={}x{} tolerance=1px normalize=right/bottom-only material-resize=passthrough",
            w,
            h
        );
    }
    if let Some(cap) = fps_cap.filter(|cap| *cap > 0) {
        log::info!(
            "fps-cap pipeline: wgc_request=native target={} order={} applies_to=glsl/onnx/interpolation contiguous_post_reduction_seq=true",
            cap,
            if hdr {
                "fps-cap->highlight-protection->duplicate-reduction->filter-chain"
            } else {
                "fps-cap->duplicate-reduction->filter-chain"
            }
        );
    }
    let (chain, errs) = FilterChain::from_specs(factory, specs);
    let requested = specs.iter().filter(|spec| spec.enabled).count();
    if requested > 0 && chain.stages.is_empty() {
        anyhow::bail!(
            "有効なフィルターを読み込めませんでした: {}",
            errs.join(" | ")
        );
    }
    log::info!(
        "filter-chain-ready: requested={} enabled={} errors={} stages={:?}",
        requested,
        chain.stages.len(),
        errs.len(),
        chain
            .stages
            .iter()
            .map(|stage| stage.name())
            .collect::<Vec<_>>()
    );
    if let Some((pre, interp, post)) = chain.interpolation_plan() {
        log::info!(
            "Interpolation chain plan: pre={pre:?} interp={interp:?} post={post:?} effective_order=pre->interp->post verified=true"
        );
    }
    source.set_queue_enabled(chain.has_interp());
    let warning: Option<String> = None;
    log::info!(
        "source-fullscreen-origin: hwnd={hwnd:#x} monitor_cover={source_monitor_fullscreen} client_only_requested={client_only} client_only_effective={effective_client_only}"
    );
    // Windowed/PIP sources are promoted just under Neo's overlay so a
    // click-through lands on the selected source instead of an unrelated
    // normal window. Do NOT promote a monitor-covering fullscreen source.
    // Some applications/compositors can react to HWND_TOPMOST with a subtle
    // outer-rect or presentation-layout change (for example a one-pixel size
    // transition). That can look like a 1-2 px zoom even with an empty filter
    // chain. The overlay is itself TOPMOST and the fullscreen source already
    // covers the monitor, so source promotion is redundant.
    let src_was_topmost = win32::is_topmost(hwnd);
    log::info!(
        "source-geometry-saved: hwnd={hwnd:#x} rect={source_restore_rect:?} maximized={source_was_maximized}"
    );
    if source_monitor_fullscreen {
        log::info!(
            "source-zorder-pixel-lock: hwnd={hwnd:#x} fullscreen=true topmost_promotion=skipped original_topmost={src_was_topmost} rect={:?}",
            win32::window_rect(hwnd)
        );
    } else {
        win32::set_topmost(hwnd, true);
        log::debug!(
            "source-zorder: hwnd={hwnd:#x} fullscreen=false topmost_promotion=applied original_topmost={src_was_topmost}"
        );
    }
    let source_presentation_signature = win32::fullscreen_presentation_signature(hwnd);
    log::debug!(
        "source-presentation-origin: hwnd={hwnd:#x} signature={source_presentation_signature:?} monitor_cover={source_monitor_fullscreen}"
    );
    let (hid_source, hide_source_pending) = if hide_source {
        log::info!("source {hwnd:#x} visual hide deferred until first valid filtered frame");
        (None, true)
    } else {
        (None, false)
    };
    Ok((
        Session {
            hwnd,
            source_monitor_fullscreen,
            source_live_fullscreen: source_monitor_fullscreen,
            source_presentation_signature,
            capture_canvas_suspended_for_fullscreen: false,
            capture_resolution_reapply_pending: false,
            fullscreen_exit_candidate_since: None,
            browser_fullscreen_cadence: AUTO_CONTENT_CADENCE_ENABLED
                && win32::is_browser_window(hwnd)
                && win32::is_monitor_fullscreen(hwnd),
            source,
            chain,
            mode,
            ratio,
            fps_cap,
            capture_client_only: effective_client_only,
            capture_hdr: hdr,
            hdr_sdr_mode,
            hdr_sdr_preprocess_logged: false,
            hdr_sdr_preprocess_error_logged: false,
            hdr_highlight_protected_seq: None,
            capture_canvas: capture_canvas.map(|(w, h)| (w as i32, h as i32)),
            source_restart_since: None,
            source_restart_last: Instant::now() - Duration::from_secs(1),
            last_tex: None,
            last_present: Instant::now(),
            last_present_started: Instant::now(),
            frame: FrameBuf::default(),
            chain_reprocess_pending: false,
            in_size: (0, 0),
            display_aspect,
            pending_resize_size: None,
            onnx_geometry_rebuild_pending: false,
            placed: false,
            input_reenable_after: None,
            source_input_geometry_missing_logged: false,
            hid_source,
            hide_source_pending,
            src_was_topmost,
            source_restore_rect,
            source_was_maximized,
            deferred_capture_resolution,
            capture_resolution_applied: false,
            capture_resolution_wait_started: None,
            capture_resolution_repaint_last: None,
            capture_resolution_nudge_done: false,
            capture_resolution_native_fallback: false,
            onnx_geometry_deferred_logged: false,
            hist: std::collections::VecDeque::new(),
            gpu_interp_hist: std::collections::VecDeque::new(),
            interp_generation: 1,
            prev_tex: None,
            prev2_tex: None,
            prev_tex_seq: 0,
            prev_tex_source_time_100ns: None,
            last_neoflow_log: Instant::now() - Duration::from_secs(2),
            monitor_refresh_hz,
            monitor_rect,
            last_src_pos: None,
            source_drag_active: false,
            source_client_drag_active: false,
            source_client_unowned_rebase_pending: false,
            source_client_drag_raw_origin: None,
            source_client_drag_overlay_origin: None,
            arrival_interval: 0.0,
            last_arrival: Instant::now(),
            starve_released: false,
            last_geom_change: None,
            last_client_rect: None,
            next_cap_deadline: None,
            next_cap_deadline_src: None,
            cap_rate_gate: CapRateGate::default(),
            cap_filter_seq: 0,
            smooth_pacing,
            source_is_elevated,
            duplicate_frame_reduction,
            duplicate_signature: None,
            duplicate_skipped: 0,
            duplicate_motion_guard: 0,
            duplicate_lookahead_wait: None,
            strict_static_streak: 0,
            strict_static_last_present: Instant::now(),
            strict_static_present_skipped: 0,
            neodeint_comb_history: 0,
            neodeint_scene_hold: 0,
            smooth_content_signature: None,
            smooth_content_duplicates: 0,
            smooth_content_seq: 0,
            smooth_content_unique_since_duplicate: 4,
            smooth_content_last_candidate_seq: None,
            smooth_content_last_candidate_time: None,
            smooth_content_pattern_hits: 0,
            smooth_content_candidate_gaps: std::collections::VecDeque::new(),
            smooth_content_24p_detected: false,
            cadence: CadenceEstimator::default(),
            smooth_pacer: SmoothPacer::default(),
            paced_present_deadline: None,
            flow_output_cadence: FlowOutputCadence::default(),
            present_cadence: PresentCadence::default(),
            cap_diag: CapDiag::new(),
            interp_worker: None,
            interp_pending: None,
            gpu_interp_worker: None,
            gpu_interp_pack_pending: None,
            gpu_interp_pending: None,
            gpu_interp_result_stash: std::collections::VecDeque::new(),
            gpu_interp_permit_after_post_submit: None,
            gpu_interp_pair_id: 0,
            gpu_interp_active_logged: false,
            provider_transition_input_suspended: false,
            provider_transition_input_suspended_since: None,
            interp_last_tick: None,
            interp_tail: None,
            interp_present_deadline: None,
            interp_post_warm: false,
            pre_chain_route_log_key: None,
            dml_rife_limit_log_key: None,
            last_process_ms: 0.0,
            last_compute_ms: 0.0,
            last_upload_submit_ms: 0.0,
            last_chain_submit_ms: 0.0,
            last_resample_submit_ms: 0.0,
            last_post_submit_ms: 0.0,
            last_pacer_wait_ms: 0.0,
            last_present_call_ms: 0.0,
            last_present_block_ms: 0.0,
            compositor_pressure_window_started: Instant::now(),
            compositor_pressure_samples: 0,
            compositor_pressure_max_ms: 0.0,
            compositor_pressure_sum_ms: 0.0,
            last_filter_retry_log: None,
            metric_seq: 0,
            last_metrics_log: Instant::now(),
            phase_diag_samples: 0,
            glsl_guard: GlslResponsivenessGuard::default(),
            glsl_overload_history: std::collections::VecDeque::new(),
            glsl_overload_hint: false,
            glsl_chain_settle_until: None,
            glsl_chain_immediate_until: None,
            glsl_resolution_recheck_active: false,
            glsl_resolution_recheck_started: None,
            glsl_resolution_recheck_healthy_since: None,
        },
        errs,
        warning,
    ))
}

/// mpv-compatible shaders use OUTPUT only as the requested final-size reference
/// for WHEN/WIDTH/HEIGHT expressions. Keep that virtual reference at 4x so two
/// x2 upscalers can run in sequence, while the actual presented resolution is
/// independently fitted to the overlay/monitor in process_and_present_from.
fn virtual_shader_output_size(frame_size: (i32, i32)) -> (i32, i32) {
    (
        frame_size.0.max(1).saturating_mul(4),
        frame_size.1.max(1).saturating_mul(4),
    )
}

fn fit_fixed_overlay_size(sw: i32, sh: i32, ratio: f32, mw: i32, mh: i32) -> (i32, i32) {
    let desired_w = (sw as f32 * ratio).max(64.0);
    let desired_h = (sh as f32 * ratio).max(64.0);
    let fit = (mw as f32 / desired_w)
        .min(mh as f32 / desired_h)
        .min(1.0)
        .max(0.0);
    let w = (desired_w * fit).round().clamp(64.0, mw as f32) as i32;
    let h = (desired_h * fit).round().clamp(64.0, mh as f32) as i32;
    (w, h)
}

fn fit_aspect_inside(aspect: (i32, i32), bounds: (i32, i32)) -> (i32, i32) {
    let (aw, ah) = (aspect.0.max(1) as f64, aspect.1.max(1) as f64);
    let fit = (bounds.0.max(1) as f64 / aw).min(bounds.1.max(1) as f64 / ah);
    (
        ((aw * fit).round() as i32).max(1),
        ((ah * fit).round() as i32).max(1),
    )
}

fn fixed_overlay_aspect_basis(frame_size: (i32, i32), client_size: (i32, i32)) -> (i32, i32) {
    if frame_size.0 > 0 && frame_size.1 > 0 {
        frame_size
    } else {
        client_size
    }
}

fn initial_display_aspect(
    capture_canvas: Option<(i32, i32)>,
    first_wgc_frame: (i32, i32),
) -> (i32, i32) {
    capture_canvas.unwrap_or(first_wgc_frame)
}

fn clamp_axis_to_bounds(pos: i32, size: i32, bounds_pos: i32, bounds_size: i32) -> i32 {
    if size >= bounds_size {
        bounds_pos
    } else {
        pos.clamp(bounds_pos, bounds_pos + bounds_size - size)
    }
}

fn client_rect_rebase_delta(
    client: (i32, i32, i32, i32),
    bounds: (i32, i32, i32, i32),
) -> (i32, i32) {
    let (cx, cy, cw, ch) = client;
    let (bx, by, bw, bh) = bounds;
    let safe_x = clamp_axis_to_bounds(cx, cw, bx, bw);
    let safe_y = clamp_axis_to_bounds(cy, ch, by, bh);
    (safe_x.saturating_sub(cx), safe_y.saturating_sub(cy))
}

/// Keep the hidden source CLIENT fully reachable by the hardware cursor.
///
/// v426 still clamped the outer HWND rect. Chromium/PIP can report a client
/// origin below that outer origin (31 px in the 2026-08-15 regression log).
/// Consequently an outer rect ending at y=1080 could leave the actual input
/// client ending at y=1111. ClipCursor/SetCursorPos then hit the physical
/// desktop edge before the virtual cursor reached the magnified-window edge,
/// which felt like a sticky boundary that needed an extra push to escape.
///
/// Translate the OUTER window by the amount required to put the CLIENT rect
/// inside the monitor, then re-read the client geometry. Windows may adjust
/// non-client geometry while moving Chromium custom-frame windows, so verify
/// convergence for a few synchronous passes instead of assuming one SetWindowPos
/// produces the requested client origin. The visible Neo overlay is untouched.
fn keep_source_window_reachable(hwnd: isize) -> Option<(i32, i32)> {
    if win32::is_monitor_fullscreen(hwnd) {
        return win32::client_rect_on_screen(hwnd).map(|(x, y, _, _)| (x, y));
    }

    let monitor = win32::monitor_rect_of(hwnd);
    let mut moved = false;
    let mut attempts = 0usize;

    for attempt in 0..3 {
        attempts = attempt + 1;
        let client_before = win32::client_rect_on_screen(hwnd)?;
        let (dx, dy) = client_rect_rebase_delta(client_before, monitor);
        if dx == 0 && dy == 0 {
            if moved {
                log::info!(
                    "source-window-client-rebase-verified: hwnd={:#x} client=({}, {}, {}x{}) monitor=({}, {}, {}x{}) attempts={}",
                    hwnd,
                    client_before.0,
                    client_before.1,
                    client_before.2,
                    client_before.3,
                    monitor.0,
                    monitor.1,
                    monitor.2,
                    monitor.3,
                    attempts
                );
            }
            return Some((client_before.0, client_before.1));
        }

        let (wx, wy, ww, wh) = win32::window_rect(hwnd)?;
        let nx = wx.saturating_add(dx);
        let ny = wy.saturating_add(dy);
        win32::set_window_rect(hwnd, nx, ny, ww, wh);
        moved = true;
        log::info!(
            "source-window-rebased: hwnd={:#x} outer_from=({}, {}) outer_to=({}, {}) size={}x{} client_before=({}, {}, {}x{}) client_shift=({},{}) monitor=({}, {}, {}x{}) basis=client",
            hwnd,
            wx,
            wy,
            nx,
            ny,
            ww,
            wh,
            client_before.0,
            client_before.1,
            client_before.2,
            client_before.3,
            dx,
            dy,
            monitor.0,
            monitor.1,
            monitor.2,
            monitor.3
        );
    }

    let client_after = win32::client_rect_on_screen(hwnd)?;
    let residual = client_rect_rebase_delta(client_after, monitor);
    if residual != (0, 0) {
        log::warn!(
            "source-window-client-rebase-incomplete: hwnd={:#x} client=({}, {}, {}x{}) residual=({},{}) monitor=({}, {}, {}x{}) attempts={}",
            hwnd,
            client_after.0,
            client_after.1,
            client_after.2,
            client_after.3,
            residual.0,
            residual.1,
            monitor.0,
            monitor.1,
            monitor.2,
            monitor.3,
            attempts
        );
    }
    Some((client_after.0, client_after.1))
}

fn client_drag_overlay_position_from_raw(
    overlay_origin: (i32, i32),
    raw_origin: (i32, i32),
    raw_current: (i32, i32),
) -> (i32, i32) {
    (
        overlay_origin
            .0
            .saturating_add(raw_current.0.saturating_sub(raw_origin.0)),
        overlay_origin
            .1
            .saturating_add(raw_current.1.saturating_sub(raw_origin.1)),
    )
}

fn move_windowed_overlay_by_source_delta(
    overlay: &mut OverlayWindow,
    dx: i32,
    dy: i32,
    gui_topmost: bool,
) {
    let (x, y, w, h) = overlay.current_rect();
    let (mx, my, mw, mh) = win32::monitor_rect_of(overlay.hwnd().0 as isize);
    let nx = clamp_axis_to_bounds(x.saturating_add(dx), w, mx, mw);
    let ny = clamp_axis_to_bounds(y.saturating_add(dy), h, my, mh);
    if (nx, ny) != (x, y) {
        reposition_overlay_for_gui_mode(overlay, nx, ny, w, h, gui_topmost);
    }
}

fn overlay_geometry(s: &Session, overlay: &OverlayWindow) -> (i32, i32, i32, i32) {
    match s.mode {
        ScaleMode::Auto => win32::monitor_rect_of(s.hwnd),
        ScaleMode::Fixed => {
            let (sx, sy, sw, sh) = win32::client_rect_on_screen(s.hwnd)
                .unwrap_or_else(|| win32::monitor_rect_of(s.hwnd));
            let (mx, my, mw, mh) = win32::monitor_rect_of(s.hwnd);
            // The Win32 client rect often changes one WGC frame before/after
            // the pixels do. Size the display from the committed frame instead
            // so old pixels are never stretched into the next aspect ratio.
            let (aspect_w, aspect_h) = fixed_overlay_aspect_basis(s.display_aspect, (sw, sh));
            let (w, h) = fit_fixed_overlay_size(aspect_w, aspect_h, s.ratio, mw, mh);
            if s.placed {
                // keep the user's position; only track size changes
                let (cx, cy, _, _) = overlay.current_rect();
                (cx, cy, w, h)
            } else {
                // first placement: centered over the source, inside the monitor
                let cx = sx + sw / 2 - w / 2;
                let cy = sy + sh / 2 - h / 2;
                (cx.clamp(mx, mx + mw - w), cy.clamp(my, my + mh - h), w, h)
            }
        }
    }
}

fn stop_session(
    session: &mut Option<Session>,
    overlay: &mut OverlayWindow,
    gc: &mut GlContext,
    status: &Arc<Mutex<Status>>,
) {
    crate::render::onnx_stage::finish_tensorrt_build_progress();
    let mut gpu_resources_safe_to_clear = true;
    if let Some(mut s) = session.take() {
        s.source.stop();
        if let Some(pack) = s.gpu_interp_pack_pending.take() {
            gc.cancel_commands_fence(pack.fence);
        }
        // Textures referenced by pending/history are pool-managed and are
        // released by clear_pool below. Recycling one alias here can invalidate
        // another live history handle.
        s.gpu_interp_pending.take();
        // Stop inference before detaching GL views. The bounded worker Drop
        // keeps Stop/Shutdown responsive even inside a stuck provider call.
        let worker_clean = s
            .gpu_interp_worker
            .take()
            .map(|mut worker| worker.shutdown())
            .unwrap_or(true);
        if worker_clean {
            s.chain.prepare_gpu_transition(gc)
        } else {
            gpu_resources_safe_to_clear = false;
            log::error!(
                "interp-gpu-resources-quarantined: reason=worker-timeout action=preserve-gl-resources"
            );
        }
        // prev_tex is pool-managed; clear_pool below frees everything
        // ALWAYS restore the source's visual state (never leave an invisible
        // window behind)
        if let Some(was_layered) = s.hid_source {
            win32::show_window_visual(s.hwnd, was_layered);
        }
        if !s.src_was_topmost && !s.source_monitor_fullscreen {
            win32::set_topmost(s.hwnd, false);
        }
        if let Some(rect) = s.source_restore_rect {
            let before = win32::window_rect(s.hwnd);
            let already_current =
                before == Some(rect) && win32::is_maximized(s.hwnd) == s.source_was_maximized;
            let ok =
                already_current || win32::restore_window_rect(s.hwnd, rect, s.source_was_maximized);
            let after = win32::window_rect(s.hwnd);
            log::info!(
                "source-geometry-restored: hwnd={:#x} requested={rect:?} before={before:?} after={after:?} maximized={} already_current={} ok={ok}",
                s.hwnd,
                s.source_was_maximized,
                already_current
            );
        }
    }
    detach_overlay_owned_helpers(overlay.hwnd().0 as isize);
    overlay.hide_for_stop();
    if gpu_resources_safe_to_clear {
        gc.clear_pool();
    } else {
        log::error!(
            "GL pool cleanup skipped because a detached GPU worker may still own shared resources"
        );
    }
    let cuda = crate::render::cuda_interop::cuda_shared_stats();
    let dml = crate::render::onnx_stage::dml_shared_stats();
    let gl_active = gc.external_import_active_count();
    let gl_active_bytes = gc.external_import_active_bytes();
    let gl_retired = gc.external_import_retired_count();
    let resource_result = if gpu_resources_safe_to_clear
        && gl_active == 0
        && cuda.active_buffers == 0
        && cuda.release_failures == 0
        && dml.active_allocations == 0
        && dml.release_failures == 0
    {
        "clean"
    } else {
        "retained"
    };
    log::info!(
        "gpu-resource-retirement: result={} gl_active={} gl_active_mb={:.1} gl_quarantine={} cuda_active_buffers={} cuda_active_mb={:.1} cuda_created={} cuda_freed={} cuda_release_failures={} dml_active_allocations={} dml_active_mb={:.1} dml_created={} dml_freed={} dml_release_failures={}",
        resource_result,
        gl_active,
        gl_active_bytes as f64 / (1024.0 * 1024.0),
        gl_retired,
        cuda.active_buffers,
        cuda.active_bytes as f64 / (1024.0 * 1024.0),
        cuda.total_created,
        cuda.total_freed,
        cuda.release_failures,
        dml.active_allocations,
        dml.active_bytes as f64 / (1024.0 * 1024.0),
        dml.total_created,
        dml.total_freed,
        dml.release_failures
    );
    let mut g = status.lock().unwrap();
    // Expected RunOptions termination must never survive Stop as a user-facing
    // yellow warning, even if another cancellation-aware path reported it just
    // before the priority lane took ownership.
    g.chain_errors
        .retain(|message| !message.contains("ONNX inference cancelled by Stop request"));
    if g.last_error
        .as_deref()
        .is_some_and(|message| message.contains("ONNX inference cancelled by Stop request"))
    {
        g.last_error = None;
    }
    if g.onnx_backend_error
        .as_deref()
        .is_some_and(|message| message.contains("ONNX inference cancelled by Stop request"))
    {
        g.onnx_backend_error = None;
    }
    g.starting = false;
    g.running = false;
    g.stopping = false;
    g.glsl_interactive_pause = false;
    g.glsl_overload_notice_latched = false;
    g.glsl_overload_auto_stop_deadline = None;
    g.hidden_src = None;
    g.source_recovery = None;
    g.warning = None;
    g.source_live_fullscreen = false;
    g.capture_resolution_resize_intent = None;
    g.onnx_tensorrt_stages = 0;
    g.onnx_cuda_stages = 0;
    g.onnx_directml_fallbacks = 0;
}

fn save_screenshot_async(path: std::path::PathBuf, w: u32, h: u32, rgba: Vec<u8>) {
    let _ = std::thread::Builder::new()
        .name("screenshot-png".into())
        .spawn(move || {
            let result = (|| -> anyhow::Result<()> {
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let file = std::fs::File::create(&path)?;
                let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), w, h);
                encoder.set_color(png::ColorType::Rgba);
                encoder.set_depth(png::BitDepth::Eight);
                let mut writer = encoder.write_header()?;
                writer.write_image_data(&rgba)?;
                Ok(())
            })();
            match result {
                Ok(()) => log::info!("screenshot saved: {} ({}x{})", path.display(), w, h),
                Err(error) => log::error!("screenshot failed: {}: {error:#}", path.display()),
            }
        });
}

/// Return the aspect-preserving DirectML RIFE safety size. The limit is
/// deliberately based on logical RIFE input height (matching the mpv script),
/// not ONNX's later 128-pixel padding. Width is rounded to the nearest even
/// pixel so downstream video/image stages never receive a needless odd width.
fn directml_rife_limited_size(w: i32, h: i32) -> Option<(i32, i32)> {
    if w <= 0 || h <= DIRECTML_RIFE_MAX_HEIGHT {
        return None;
    }
    let scaled = (w as f64 * DIRECTML_RIFE_MAX_HEIGHT as f64 / h as f64).round() as i32;
    let limited_w = ((scaled.max(2) + 1) / 2) * 2;
    Some((limited_w, DIRECTML_RIFE_MAX_HEIGHT))
}

fn log_directml_rife_height_limit(
    s: &mut Session,
    source: (i32, i32),
    limited: (i32, i32),
    interp_index: usize,
    route: &str,
) {
    let key = (source.0, source.1, limited.0, limited.1, interp_index);
    if s.dml_rife_limit_log_key != Some(key) {
        log::info!(
            "directml-rife-height-limit: source={}x{} limited={}x{} max_height={} interp_index={} position={} route={} scaler=spline36 policy=always-for-directml-rife",
            source.0,
            source.1,
            limited.0,
            limited.1,
            DIRECTML_RIFE_MAX_HEIGHT,
            interp_index,
            if interp_index == 0 { "first" } else { "after-prechain" },
            route
        );
        s.dml_rife_limit_log_key = Some(key);
    }
}

/// Bilinear CPU fallback for the diagnostic/non-resident interpolation route.
/// The production DirectML RIFE path uses the GPU Spline36 limiter above; this
/// keeps the same 1440p contract even when GPU interpolation is explicitly
/// disabled or unavailable.
fn resize_rgba8_bilinear(
    src: &[u8],
    sw: i32,
    sh: i32,
    dw: i32,
    dh: i32,
) -> Option<Vec<u8>> {
    if sw <= 0
        || sh <= 0
        || dw <= 0
        || dh <= 0
        || src.len() != sw as usize * sh as usize * 4
    {
        return None;
    }
    if (sw, sh) == (dw, dh) {
        return Some(src.to_vec());
    }

    let dst_stride = dw as usize * 4;
    let mut dst = vec![0u8; dst_stride * dh as usize];
    dst.par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy = (((dy as f64 + 0.5) * sh as f64 / dh as f64) - 0.5)
                .clamp(0.0, sh as f64 - 1.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(sh as usize - 1);
            let fy = sy - y0 as f64;
            for dx in 0..dw as usize {
                let sx = (((dx as f64 + 0.5) * sw as f64 / dw as f64) - 0.5)
                    .clamp(0.0, sw as f64 - 1.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(sw as usize - 1);
                let fx = sx - x0 as f64;
                for channel in 0..4 {
                    let at = |x: usize, y: usize| {
                        src[(y * sw as usize + x) * 4 + channel] as f64
                    };
                    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                    let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                    row[dx * 4 + channel] =
                        (top * (1.0 - fy) + bottom * fy).round() as u8;
                }
            }
        });
    Some(dst)
}

/// Diagnostic wrapper for the capture-buffer -> GL texture submission.
/// The measurement is CPU-side wall time around the existing upload path and
/// does not add synchronization or change residency/pacing behavior.
fn upload_frame_timed(gc: &mut GlContext, frame: &FrameBuf, last_ms: &mut f64) -> GpuTex {
    let started = Instant::now();
    let tex = upload_frame(gc, frame);
    *last_ms = started.elapsed().as_secs_f64() * 1000.0;
    tex
}

/// Upload a captured frame. HDR normally became RGBA8 in the pre-chain stage;
/// the fp16 branch is retained only as a malformed-frame safety fallback.
fn upload_frame(gc: &mut GlContext, frame: &FrameBuf) -> GpuTex {
    if frame.hdr {
        let t = gc.upload_rgba16f(frame.w, frame.h, &frame.data);
        crate::render::scaler::tonemap_hdr(gc, t).unwrap_or(t)
    } else {
        gc.upload_rgba8(frame.w, frame.h, &frame.data)
    }
}

/// WGC is change-driven and some static Chromium surfaces do not emit a frame
/// after their window is resized. In that one startup case, scale the cached
/// RGBA8 still image to the explicitly requested capture size so the normal
/// filter chain can run and the overlay never remains transparent indefinitely.
#[allow(dead_code)] // retained for static-source recovery experiments
fn resize_static_rgba8_frame(frame: &mut FrameBuf, target: (i32, i32)) -> bool {
    use rayon::prelude::*;

    let (sw, sh) = (frame.w, frame.h);
    let (dw, dh) = target;
    if frame.hdr
        || sw <= 0
        || sh <= 0
        || dw <= 0
        || dh <= 0
        || frame.data.len() != (sw as usize) * (sh as usize) * 4
    {
        return false;
    }
    if (sw, sh) == (dw, dh) {
        return true;
    }

    let src = &frame.data;
    let dst_stride = dw as usize * 4;
    let mut dst = vec![0u8; dst_stride * dh as usize];
    dst.par_chunks_mut(dst_stride)
        .enumerate()
        .for_each(|(dy, row)| {
            let sy =
                (((dy as f64 + 0.5) * sh as f64 / dh as f64) - 0.5).clamp(0.0, sh as f64 - 1.0);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(sh as usize - 1);
            let fy = sy - y0 as f64;
            for dx in 0..dw as usize {
                let sx =
                    (((dx as f64 + 0.5) * sw as f64 / dw as f64) - 0.5).clamp(0.0, sw as f64 - 1.0);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(sw as usize - 1);
                let fx = sx - x0 as f64;
                for channel in 0..4 {
                    let at = |x: usize, y: usize| src[(y * sw as usize + x) * 4 + channel] as f64;
                    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                    let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                    row[dx * 4 + channel] = (top * (1.0 - fy) + bottom * fy).round() as u8;
                }
            }
        });

    frame.w = dw;
    frame.h = dh;
    frame.data = dst;
    frame.seq = frame.seq.wrapping_add(1);
    frame.received_at = Some(Instant::now());
    frame.source_time_100ns = None;
    true
}

/// Center-crop or black-pad a captured frame without resampling a single
/// source pixel. Padding is deliberately limited to one pixel per axis: it is
/// only for odd/even edge reconciliation, never for filling a real resolution
/// mismatch with a large border.
fn fit_frame_canvas_dot_by_dot(frame: &mut FrameBuf, target: (i32, i32)) -> bool {
    let (sw, sh) = (frame.w, frame.h);
    let (dw, dh) = target;
    let bytes_per_pixel = if frame.hdr { 8usize } else { 4usize };
    if sw <= 0
        || sh <= 0
        || dw <= 0
        || dh <= 0
        || frame.data.len() != sw as usize * sh as usize * bytes_per_pixel
        || dw - sw > 1
        || dh - sh > 1
    {
        return false;
    }
    if (sw, sh) == (dw, dh) {
        return true;
    }

    let copy_w = sw.min(dw);
    let copy_h = sh.min(dh);
    let src_x = ((sw - copy_w) / 2) as usize;
    let src_y = ((sh - copy_h) / 2) as usize;
    let dst_x = ((dw - copy_w) / 2) as usize;
    let dst_y = ((dh - copy_h) / 2) as usize;
    let row_bytes = copy_w as usize * bytes_per_pixel;
    let mut dst = vec![0u8; dw as usize * dh as usize * bytes_per_pixel];
    if !frame.hdr {
        for pixel in dst.chunks_exact_mut(4) {
            pixel[3] = 255;
        }
    }
    for row in 0..copy_h as usize {
        let src_start = ((src_y + row) * sw as usize + src_x) * bytes_per_pixel;
        let dst_start = ((dst_y + row) * dw as usize + dst_x) * bytes_per_pixel;
        dst[dst_start..dst_start + row_bytes]
            .copy_from_slice(&frame.data[src_start..src_start + row_bytes]);
    }
    frame.w = dw;
    frame.h = dh;
    frame.data = dst;
    true
}

fn normalize_capture_client_size(size: (u32, u32)) -> (u32, u32) {
    let even = |value: u32, min_value: u32, max_value: u32| {
        let bounded = value.clamp(min_value, max_value);
        if bounded & 1 == 0 {
            bounded
        } else if bounded < max_value {
            bounded + 1
        } else {
            bounded.saturating_sub(1).max(min_value)
        }
    };
    (even(size.0, 160, 7680), even(size.1, 120, 4320))
}

/// Do not let a crop/pad fallback disguise the old source size while an
/// explicit client resize is pending. The reveal state machine must first see
/// a real WGC frame produced at the requested client dimensions.
fn should_apply_capture_canvas(
    native_size: (i32, i32),
    target: (i32, i32),
    client_resize_pending: bool,
) -> bool {
    native_size != target && !client_resize_pending
}

fn reset_for_source_resize(
    gc: &mut GlContext,
    s: &mut Session,
    metrics: &Metrics,
    old_size: (i32, i32),
    new_size: (i32, i32),
) {
    log::info!(
        "source frame size changed: {}x{} -> {}x{}; preserving filter chain and preparing automatic follow",
        old_size.0,
        old_size.1,
        new_size.0,
        new_size.1
    );
    // A queued interpolation job still owns frames and possibly a DirectML
    // dispatch for the old dimensions. Join it before retiring shared GPU
    // output, then discard queued transitional captures and resume from WGC's
    // newest frame at the new aspect ratio.
    reset_gpu_interp_for_geometry_transition(s, gc, "source-resolution-change");
    s.interp_pending = None;
    s.interp_worker = None;
    s.interp_last_tick = None;
    s.interp_tail = None;
    s.interp_present_deadline = None;
    s.source.set_queue_enabled(false);
    s.source.set_queue_enabled(s.chain.has_interp());
    log::info!(
        "source resize transition ready: stale interpolation/capture queues cleared; waiting for stable {}x{} frame gl_cache=preserved onnx_rebuild={}",
        new_size.0,
        new_size.1,
        s.chain.has_onnx()
    );
    s.hist.clear();
    if let Some(texture) = s.prev_tex.take() {
        gc.recycle(texture);
    }
    if let Some(texture) = s.prev2_tex.take() {
        gc.recycle(texture);
    }
    if let Some(texture) = s.last_tex.take() {
        gc.recycle(texture);
    }
    s.prev_tex_seq = 0;
    s.prev_tex_source_time_100ns = None;
    s.arrival_interval = 0.0;
    s.cadence.reset();
    s.smooth_content_signature = None;
    s.smooth_content_duplicates = 0;
    s.smooth_content_seq = 0;
    s.smooth_content_unique_since_duplicate = 4;
    s.smooth_content_last_candidate_seq = None;
    s.smooth_content_last_candidate_time = None;
    s.smooth_content_pattern_hits = 0;
    s.smooth_content_candidate_gaps.clear();
    s.smooth_content_24p_detected = false;
    s.smooth_pacer.reset();
    s.paced_present_deadline = None;
    s.flow_output_cadence.reset();
    s.present_cadence = PresentCadence::default();
    s.last_process_ms = 0.0;
    s.last_compute_ms = 0.0;
    s.last_present_block_ms = 0.0;
    s.pending_resize_size = Some(new_size);
    s.onnx_geometry_rebuild_pending = s.chain.has_onnx();
    metrics.reset();
    // Preserve reusable shader textures and a bounded set of old/new frame
    // sizes. This makes repeated capture-resolution tests recover in-session
    // instead of requiring Stop/Start to rebuild the entire GL working set.
    gc.trim_transient_pool(2);
}

fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
    use rayon::prelude::*;
    let n = rgba.len() / 4;
    let mut rgb = vec![0u8; n * 3];
    rgb.par_chunks_mut(3 * 8192)
        .zip(rgba.par_chunks(4 * 8192))
        .for_each(|(o, i)| {
            for (op, ip) in o.chunks_mut(3).zip(i.chunks(4)) {
                op[0] = ip[0];
                op[1] = ip[1];
                op[2] = ip[2];
            }
        });
    rgb
}

fn frame_present_latency_ms(timing: FrameTiming, present_time: PresentTime) -> f64 {
    if let (Some(source_time), Some(now_time)) =
        (timing.source_time_100ns, present_time.qpc_time_100ns)
    {
        if now_time >= source_time {
            return (now_time - source_time) as f64 / 10_000.0;
        }
    }
    let start = timing.received_at.unwrap_or(timing.fallback_start);
    present_time.instant.duration_since(start).as_secs_f64() * 1000.0
}

/// Run the chain on `input`, present, and do pool bookkeeping. Chain errors
/// keep the last valid filtered frame visible while the complete chain retries.
#[allow(clippy::too_many_arguments)]
/// Present the pending interpolated pair: the mids (paced across the source
/// interval, in t order) then the delayed REAL end frame. Worker errors and
/// timeouts skip the affected mids but the real frame is always presented.
#[allow(clippy::too_many_arguments)]
fn drain_pending_interp(
    gc: &mut GlContext,
    overlay: &mut OverlayWindow,
    input: &crate::input::InputSystem,
    s: &mut Session,
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
    out_size: (i32, i32),
    stats: bool,
    vsync_on: bool,
    tick_t0: Instant,
    downscaler: crate::render::scaler::Kernel,
) {
    let Some(pending) = s.interp_pending.take() else {
        return;
    };
    let drain_t0 = Instant::now();
    let gap_ms = s
        .interp_last_tick
        .map(|t| tick_t0.duration_since(t).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    s.interp_last_tick = Some(tick_t0);
    let pre_ms = drain_t0.duration_since(tick_t0).as_secs_f64() * 1000.0;
    let wait_take_ms = s
        .interp_tail
        .map(|t| tick_t0.duration_since(t).as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let mut got: Vec<Option<(i32, i32, Vec<u8>)>> = (0..pending.count).map(|_| None).collect();
    let mut fail: Option<String> = None;
    let mut interp_run_ms = 0.0f64;
    if let Some(worker) = s.interp_worker.as_ref() {
        for _ in 0..pending.count {
            match worker.res_rx.recv_timeout(Duration::from_millis(400)) {
                Ok(res) => {
                    if stats {
                        if let Some((p, r, o)) = res.profile {
                            metrics.probe("Frame interpolation pack", "cpu", p);
                            metrics.probe("Frame interpolation run", "onnx", r);
                            metrics.probe("Frame interpolation output", "cpu", o);
                            interp_run_ms = interp_run_ms.max(r);
                        }
                    }
                    match res.r {
                        Ok(v) => got[res.idx.min(pending.count - 1)] = Some(v),
                        Err(e) => fail = Some(format!("{e:#}")),
                    }
                }
                Err(_) => {
                    fail = Some("補間ワーカーの応答がありません（タイムアウト）".into());
                    break;
                }
            }
        }
    }
    if stats && interp_run_ms > 0.0 {
        metrics.probe(&pending.interp_name, "onnx", interp_run_ms);
    }
    if let Some(msg) = &fail {
        let m = format!("補間: {msg}");
        let mut g = status.lock().unwrap();
        if !g.chain_errors.contains(&m) {
            g.chain_errors.push(m);
        }
    }
    let n = pending.count + usize::from(pending.present_real);
    let mids_ok = got.iter().filter(|g| g.is_some()).count();
    let recv_ms = drain_t0.elapsed().as_secs_f64() * 1000.0;
    let content_dt = if s.smooth_pacing {
        s.cadence
            .period_s()
            .or(pending.pair_dt)
            .unwrap_or(s.arrival_interval)
    } else {
        pending.pair_dt.unwrap_or(s.arrival_interval)
    }
    .clamp(1.0 / 240.0, 0.1);
    let output_step = Duration::from_secs_f64(
        pending
            .output_period_s
            .unwrap_or_else(|| content_dt / n.max(1) as f64)
            .clamp(1.0 / 1000.0, 0.1),
    );
    let mut output_deadline = if s.smooth_pacing {
        let now = Instant::now();
        match s.interp_present_deadline {
            Some(deadline)
                if now.saturating_duration_since(deadline) <= output_step.mul_f64(1.5) =>
            {
                deadline
            }
            _ => now,
        }
    } else {
        tick_t0
    };
    let mids_t0 = Instant::now();
    let mut deadline_miss_count = 0u32;
    for (index, value) in got.into_iter().enumerate() {
        let Some((mw, mh, mid)) = value else {
            continue;
        };
        if s.smooth_pacing {
            // Begin the expensive post-chain before its absolute presentation
            // slot. Waiting until the slot and only then running a 20 ms ONNX
            // stage produced alternating late/bunched presents on 60 Hz.
            let predicted = Duration::from_secs_f64(
                (s.last_process_ms / 1000.0).clamp(0.0, output_step.as_secs_f64()),
            );
            let prepare_at = output_deadline
                .checked_sub(predicted)
                .unwrap_or(output_deadline);
            wait_until_with_pump(overlay, prepare_at);
        }
        let t0 = Instant::now();
        let tex = gc.upload_rgb8(mw, mh, &mid);
        let transition_presented_before = if s.provider_transition_input_suspended {
            Some(status.lock().unwrap().presented)
        } else {
            None
        };
        process_and_present_from(
            gc,
            overlay,
            s,
            metrics,
            status,
            tex,
            out_size,
            stats,
            t0,
            0, // synthetic frame
            Some(FrameTiming {
                received_at: pending.real.received_at,
                source_time_100ns: match (
                    pending.start_source_time_100ns,
                    pending.real.source_time_100ns,
                    pending.mid_ts.get(index),
                ) {
                    (Some(a), Some(b), Some(t)) if b >= a => {
                        Some(a + ((b - a) as f64 * *t as f64).round() as i64)
                    }
                    _ => pending.real.source_time_100ns,
                },
                fallback_start: t0,
            }),
            &[],
            downscaler,
            pending.chain_start_index,
        );
        if let Some(presented_before) = transition_presented_before {
            let presented_after = status.lock().unwrap().presented;
            if presented_after > presented_before {
                resume_provider_transition_after_interpolated_present(input, s, "cpu-worker");
            }
        }
        if s.smooth_pacing {
            let scheduled_next = output_deadline + output_step;
            let now = Instant::now();
            // Never catch up by presenting the next output immediately after
            // a missed slot. Resynchronise one full slot ahead instead.
            output_deadline = if now >= scheduled_next {
                deadline_miss_count += 1;
                now + output_step
            } else {
                scheduled_next
            };
        }
        if !s.smooth_pacing && !vsync_on {
            // Low-latency mode retains arrival-relative pacing.
            let target = (content_dt * (index + 1) as f64 / n as f64).min(0.025);
            let deadline = tick_t0 + Duration::from_secs_f64(target);
            wait_until_with_pump(overlay, deadline);
        }
    }
    let mids_ms = mids_t0.elapsed().as_secs_f64() * 1000.0;
    let real_t0 = Instant::now();
    if pending.present_real {
        if s.smooth_pacing {
            let predicted = Duration::from_secs_f64(
                (s.last_process_ms / 1000.0).clamp(0.0, output_step.as_secs_f64()),
            );
            let prepare_at = output_deadline
                .checked_sub(predicted)
                .unwrap_or(output_deadline);
            wait_until_with_pump(overlay, prepare_at);
        }
        let t0 = Instant::now();
        let tex = if pending.rgba {
            gc.upload_rgba8(pending.real.w, pending.real.h, &pending.real.data)
        } else {
            gc.upload_rgb8(pending.real.w, pending.real.h, &pending.real.data)
        };
        let delivered = s.source.delivered();
        let captures = delivered.saturating_sub(s.metric_seq) as u32;
        s.metric_seq = delivered;
        process_and_present_from(
            gc,
            overlay,
            s,
            metrics,
            status,
            tex,
            out_size,
            stats,
            t0,
            captures,
            Some(pending.real.timing(t0)),
            &[],
            downscaler,
            pending.chain_start_index,
        );
        if s.smooth_pacing {
            let scheduled_next = output_deadline + output_step;
            let now = Instant::now();
            output_deadline = if now >= scheduled_next {
                deadline_miss_count += 1;
                now + output_step
            } else {
                scheduled_next
            };
        }
    }
    if s.smooth_pacing {
        s.interp_present_deadline = Some(output_deadline);
    } else {
        s.interp_present_deadline = None;
    }
    if s.last_neoflow_log.elapsed() >= Duration::from_secs(1) {
        let source_fps = 1.0 / content_dt.max(1.0 / 240.0);
        let output_fps = 1.0 / output_step.as_secs_f64().max(1.0 / 1000.0);
        let pre_stage_count = pending.chain_start_index.saturating_sub(1);
        let post_stage_count = s
            .chain
            .stage_count()
            .saturating_sub(pending.chain_start_index);
        log::info!(
            "interp-pipeline: rife_input={}x{} mids={}/{} fail={:?} smooth={} cadence_ms={:.2} arrival_ms={:.1} pair_dt_ms={:.1} present_period_ms={:.2} gap_ms={:.1} wait_take_ms={:.1} pre_ms={:.1} recv_ms={:.1} mids_ms={:.1} real_ms={:.1} pre_stage_count={} post_stage_count={} pre_invocations_per_s={:.2} interp_invocations_per_s={:.2} post_outputs_per_s={:.2} deadline_miss_count={} dropped_mid_count={} repeated_present_count=0",
            pending.real.w,
            pending.real.h,
            mids_ok,
            pending.count,
            fail.as_deref(),
            s.smooth_pacing,
            s.cadence.period_s().unwrap_or(0.0) * 1000.0,
            s.arrival_interval * 1000.0,
            pending.pair_dt.unwrap_or(0.0) * 1000.0,
            output_step.as_secs_f64() * 1000.0,
            gap_ms,
            wait_take_ms,
            pre_ms,
            recv_ms,
            mids_ms,
            real_t0.elapsed().as_secs_f64() * 1000.0,
            pre_stage_count,
            post_stage_count,
            if pre_stage_count > 0 { source_fps } else { 0.0 },
            source_fps,
            if post_stage_count > 0 {
                output_fps
            } else {
                0.0
            },
            deadline_miss_count,
            pending.count.saturating_sub(mids_ok),
        );
        s.last_neoflow_log = Instant::now();
    }
}

fn process_and_present(
    gc: &mut GlContext,
    overlay: &mut OverlayWindow,
    s: &mut Session,
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
    input: GpuTex,
    out_size: (i32, i32),
    stats: bool,
    t0: Instant,
    captures: u32,
    timing: Option<FrameTiming>,
    keep_extra: &[GpuTex],
    downscaler: crate::render::scaler::Kernel,
) {
    process_and_present_from(
        gc, overlay, s, metrics, status, input, out_size, stats, t0, captures, timing, keep_extra,
        downscaler, 0,
    );
}

fn release_gpu_interp_permit_after_post_submit(s: &mut Session) {
    let Some((generation, pair_id)) = s.gpu_interp_permit_after_post_submit.take() else {
        return;
    };
    if let Some(worker) = s.gpu_interp_worker.as_ref() {
        worker.permit_next_slot(generation, pair_id);
        if pair_id < 3 || pair_id % 120 == 0 {
            log::debug!(
                "interp-gpu-slot-release: pair={} generation={} point=after-post-submit",
                pair_id,
                generation
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_and_present_from(
    gc: &mut GlContext,
    overlay: &mut OverlayWindow,
    s: &mut Session,
    metrics: &Metrics,
    status: &Arc<Mutex<Status>>,
    input: GpuTex,
    out_size: (i32, i32),
    stats: bool,
    t0: Instant,
    captures: u32,
    timing: Option<FrameTiming>,
    keep_extra: &[GpuTex],
    downscaler: crate::render::scaler::Kernel,
    chain_start_index: usize,
) {
    // `process_from()` clears chain_reprocess_pending after a successful chain
    // run. Remember whether this present came from an explicit filter edit so
    // we can synchronously commit that one frame to DWM below.
    let chain_reprocess_commit = s.chain_reprocess_pending;

    // Selected filters are part of the requested image semantics. Never omit
    // any post-interpolation stage. The low-spec admission guard is therefore
    // limited to NON-interpolation chains, where it may hold the previous
    // already-filtered frame but never substitute an unfiltered one.
    let guard_eligible = glsl_guard_eligible(s);
    let guard_can_hold = guard_eligible
        && !s.chain.has_interp()
        && overlay.is_visible()
        && s.last_tex.is_some()
        && !s.chain_reprocess_pending
        && !s.capture_resolution_applied
        && s.deferred_capture_resolution.is_none()
        && s.pending_resize_size.is_none();

    let interactive_pause = guard_can_hold
        && s.glsl_guard.mode != GlslGuardMode::Off
        && !s.glsl_resolution_recheck_active
        && crate::input::main_gui_has_native_cursor_ownership();
    let auto_stop_hold = guard_can_hold
        && !s.glsl_resolution_recheck_active
        && status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .glsl_overload_auto_stop_deadline
            .is_some();

    {
        let mut state = status.lock().unwrap();
        if state.glsl_interactive_pause != interactive_pause {
            state.glsl_interactive_pause = interactive_pause;
            log::info!(
                "glsl-responsiveness-gui-priority: active={} mode={:?} action={}",
                interactive_pause,
                s.glsl_guard.mode,
                if interactive_pause {
                    "hold-last-complete-frame-for-main-gui"
                } else {
                    "resume-guarded-rendering"
                }
            );
        }
    }
    if auto_stop_hold {
        // The overload decision is already final for this workload.
        // Preserve the last complete filtered frame during the short warning
        // grace instead of continuing to consume the very GPU time needed by
        // cursor/DWM/GUI recovery. The actual shutdown is still dispatched by
        // the GUI through the ordinary Cmd::Stop path.
        release_gpu_interp_permit_after_post_submit(s);
        let mut keep = Vec::with_capacity(1 + keep_extra.len());
        if let Some(last_good) = s.last_tex {
            keep.push(last_good);
        }
        keep.extend_from_slice(keep_extra);
        gc.release_frame(&keep);
        return;
    }

    if interactive_pause {
        release_gpu_interp_permit_after_post_submit(s);
        let mut keep = Vec::with_capacity(1 + keep_extra.len());
        if let Some(last_good) = s.last_tex {
            keep.push(last_good);
        }
        keep.extend_from_slice(keep_extra);
        gc.release_frame(&keep);
        return;
    }

    if guard_can_hold && s.glsl_guard.should_skip_frame() {
        release_gpu_interp_permit_after_post_submit(s);
        let gpu = latest_gui_gpu_percent();
        s.glsl_guard.note_skipped(gpu);
        let mut keep = Vec::with_capacity(1 + keep_extra.len());
        if let Some(last_good) = s.last_tex {
            keep.push(last_good);
        }
        keep.extend_from_slice(keep_extra);
        gc.release_frame(&keep);
        return;
    }

    let m = metrics.clone();
    let mut probe_fn = |name: &str, kind: StageKind, ms: f64| {
        m.probe(
            name,
            if kind == StageKind::Glsl {
                "glsl"
            } else {
                "onnx"
            },
            ms,
        );
    };
    let chain_input_size = (input.w(), input.h());
    let chain_started = Instant::now();
    let result = s.chain.process_from(
        gc,
        input,
        out_size,
        chain_start_index,
        if stats { Some(&mut probe_fn) } else { None },
    );
    s.last_chain_submit_ms = chain_started.elapsed().as_secs_f64() * 1000.0;
    match result {
        Ok(chain_tex) => {
            s.chain_reprocess_pending = false;
            if stats && s.chain.has_interp() && chain_start_index > 0 {
                let post_stages = s
                    .chain
                    .stages
                    .iter()
                    .skip(chain_start_index)
                    .map(|stage| stage.name().to_string())
                    .collect::<Vec<_>>();
                log::debug!(
                    "interp-post-chain-verified: start={} total={} input={}x{} output={}x{} stages={:?}",
                    chain_start_index,
                    s.chain.stage_count(),
                    chain_input_size.0,
                    chain_input_size.1,
                    chain_tex.w(),
                    chain_tex.h(),
                    post_stages
                );
            }
            // fit into the window with the selected kernel (Spline36 default),
            // then run OUTPUT/SCALED-hook shaders (sharpeners) at display size
            // Compute against the committed frame geometry. During an aspect
            // transition the actual overlay is deliberately kept at the old
            // size until this new filtered frame is ready to present.
            let desired_overlay = overlay_geometry(s, overlay);
            let (vw, vh) = (desired_overlay.2, desired_overlay.3);
            let display_aspect = if s.display_aspect.0 > 0 && s.display_aspect.1 > 0 {
                s.display_aspect
            } else {
                (chain_tex.w(), chain_tex.h())
            };
            let (dw, dh) = fit_aspect_inside(display_aspect, (vw, vh));
            let resample_started = Instant::now();
            let mut final_tex = crate::render::scaler::resample(gc, chain_tex, dw, dh, downscaler)
                .unwrap_or(chain_tex);
            s.last_resample_submit_ms = resample_started.elapsed().as_secs_f64() * 1000.0;
            let post_started = Instant::now();
            for (metric_label, shader) in s.chain.post_shaders_with_metric_labels() {
                // OUTPUT/SCALED-hook shaders execute outside FilterChain::process_range,
                // after the final display-size resample. They therefore need their own
                // asynchronous GL_TIME_ELAPSED query; otherwise the effect is applied
                // correctly but the per-stage statistics row never receives a sample.
                let timer = stats
                    .then(|| gc.begin_gpu_timer(&metric_label))
                    .flatten();
                let post_result =
                    crate::render::glsl_engine::GlslEngine::run_post(gc, &shader, final_tex);
                if let Some(query) = timer {
                    gc.end_gpu_timer(query, metric_label.clone());
                }
                match post_result {
                    Ok(t) => final_tex = t,
                    Err(e) => {
                        let msg = format!("{e:#}");
                        let mut g = status.lock().unwrap();
                        if !g.chain_errors.contains(&msg) {
                            log::error!(
                                "GLSL post-stage failed: shader={} path={} error={}",
                                shader.name(),
                                shader.path,
                                msg
                            );
                            g.chain_errors.push(msg);
                        }
                    }
                }
            }
            s.last_post_submit_ms = post_started.elapsed().as_secs_f64() * 1000.0;
            // x4/x5: the current output's conversion + GLSL post-chain is now
            // ordered on the GL queue. Release the next unique provider slot
            // here, not after import (which lets RIFE preempt the GLSL chain)
            // and not after SwapBuffers (which needlessly serializes DML with
            // compositor wait). All interpolation/output buffers are per-slot.
            release_gpu_interp_permit_after_post_submit(s);
            if overlay.size() != (desired_overlay.2, desired_overlay.3) {
                let old_size = overlay.size();
                let gui_hwnd = win32::main_gui_hwnd();
                let gui_topmost = gui_hwnd != 0
                    && win32::is_window_valid(gui_hwnd)
                    && win32::is_topmost(gui_hwnd);
                reposition_overlay_for_gui_mode(
                    overlay,
                    desired_overlay.0,
                    desired_overlay.1,
                    desired_overlay.2,
                    desired_overlay.3,
                    gui_topmost,
                );
                status.lock().unwrap().overlay_rect = desired_overlay;
                log::info!(
                    "source aspect display committed atomically: overlay={}x{} -> {}x{} frame={}x{}",
                    old_size.0,
                    old_size.1,
                    desired_overlay.2,
                    desired_overlay.3,
                    s.in_size.0,
                    s.in_size.1
                );
            }
            let compute_elapsed = t0.elapsed();
            s.last_compute_ms = compute_elapsed.as_secs_f64() * 1000.0;

            let pacer_wait_started = Instant::now();
            if let Some(deadline) = s.paced_present_deadline.take() {
                wait_until_with_pump(overlay, deadline);
                s.smooth_pacer.reanchor_after_missed_present(
                    Instant::now(),
                    deadline,
                    s.cadence.period_s(),
                );
            }
            s.last_pacer_wait_ms = pacer_wait_started.elapsed().as_secs_f64() * 1000.0;
            let present_call_started = Instant::now();
            s.last_present_started = present_call_started;
            let present_result = overlay.present_with_swap_timing(gc, final_tex);
            s.last_present_call_ms = present_call_started.elapsed().as_secs_f64() * 1000.0;
            let present_time = PresentTime::now();
            s.last_present_block_ms = present_result
                .as_ref()
                .map(|elapsed| elapsed.as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            if crate::logging::diagnostics_enabled() {
                // A panel blink cannot be observed from this render thread, but
                // long SwapBuffers waits are the compositor-pressure side of the
                // same timeline. Aggregate instead of logging every frame so a
                // heavy preset does not create a multi-megabyte log flood.
                if s.last_present_block_ms >= 12.0 {
                    s.compositor_pressure_samples = s.compositor_pressure_samples.saturating_add(1);
                    s.compositor_pressure_max_ms =
                        s.compositor_pressure_max_ms.max(s.last_present_block_ms);
                    s.compositor_pressure_sum_ms += s.last_present_block_ms;
                }
                if s.compositor_pressure_window_started.elapsed() >= Duration::from_millis(500) {
                    if s.compositor_pressure_samples > 0 {
                        let avg_ms = s.compositor_pressure_sum_ms
                            / s.compositor_pressure_samples.max(1) as f64;
                        log::debug!(
                            "overlay-compositor-pressure: window_ms={:.0} samples={} avg_swap_ms={:.2} max_swap_ms={:.2} last_swap_ms={:.2} present_call_ms={:.2} pacer_wait_ms={:.2} compute_ms={:.2} smooth={} glsl={} interp={}",
                            s.compositor_pressure_window_started.elapsed().as_secs_f64() * 1000.0,
                            s.compositor_pressure_samples,
                            avg_ms,
                            s.compositor_pressure_max_ms,
                            s.last_present_block_ms,
                            s.last_present_call_ms,
                            s.last_pacer_wait_ms,
                            s.last_compute_ms,
                            s.smooth_pacing,
                            s.chain.has_glsl(),
                            s.chain.has_interp()
                        );
                    }
                    s.compositor_pressure_window_started = Instant::now();
                    s.compositor_pressure_samples = 0;
                    s.compositor_pressure_max_ms = 0.0;
                    s.compositor_pressure_sum_ms = 0.0;
                }
            }
            let effective_present_interval_s = s
                .present_cadence
                .last
                .map(|last| present_time.instant.duration_since(last).as_secs_f64());
            // OpenGL stages are submitted asynchronously. Their GPU time can
            // therefore appear in SwapBuffers instead of `compute_elapsed`.
            // Feed that deferred tail into the next-frame preparation budget
            // as well as the interpolation path, so an ordinary shader chain
            // starts early enough without changing its presentation cadence.
            let deferred_gpu_tail_s = s
                .cadence
                .period_s()
                .map(|period_s| {
                    ((s.last_present_block_ms - 1.25).max(0.0) / 1000.0).min(period_s * 0.75)
                })
                .unwrap_or(0.0);
            s.smooth_pacer
                .observe_process(compute_elapsed.as_secs_f64() + deferred_gpu_tail_s);
            // A normal Chromium/DWM path can block SwapBuffers for 8-12ms
            // without losing output cadence. Never disable manual pacing from
            // that signal alone.
            if s.source_is_elevated
                && s.cadence.period_s().is_some_and(|period| period >= 0.025)
                && let Some(interval_s) = effective_present_interval_s
                && let Some(compositor_paced) = s
                    .smooth_pacer
                    .observe_effective_present_interval(interval_s, s.cadence.period_s())
            {
                log::debug!(
                    "smooth-pacing effective cadence handoff: compositor_paced={} interval_ms={:.2} cadence_ms={:.2} monitor_hz={:.2}",
                    compositor_paced,
                    interval_s * 1000.0,
                    s.cadence.period_s().unwrap_or(0.0) * 1000.0,
                    s.monitor_refresh_hz.unwrap_or(0.0)
                );
                s.paced_present_deadline = None;
            }
            if let Err(e) = present_result {
                status.lock().unwrap().last_error = Some(format!("{e:#}"));
            } else {
                if chain_reprocess_commit && overlay.is_visible() {
                    // WGC is change-driven, so a paused source may not deliver
                    // another frame after a filter toggle. Force DWM to consume
                    // this newly filtered SwapBuffers result now rather than
                    // relying on a later mouse/hover repaint to make it visible.
                    overlay.flush_compositor();
                    log::info!(
                        "chain-reprocess-present-commit: action=dwm-flush filtered_frame_visible=true"
                    );
                }
                s.present_cadence.record(present_time.instant);
                if overlay.is_visible()
                    && s.capture_resolution_applied
                    && let Some((requested, applied)) = s.deferred_capture_resolution
                    && (s.frame.w, s.frame.h) == (applied.0 as i32, applied.1 as i32)
                {
                    s.deferred_capture_resolution = None;
                    s.capture_resolution_applied = false;
                    s.capture_resolution_wait_started = None;
                    s.last_client_rect = win32::client_rect_on_screen(s.hwnd);
                    s.last_geom_change = None;
                    s.starve_released = false;
                    log::info!(
                        "capture-resolution filtered target replaced transition shield: hwnd={:#x} requested={}x{} frame={}x{}",
                        s.hwnd,
                        requested.0,
                        requested.1,
                        s.frame.w,
                        s.frame.h
                    );
                }
                if !overlay.is_visible() {
                    let mut reveal_ready = true;
                    if let Some((requested, applied)) = s.deferred_capture_resolution {
                        let target = (applied.0 as i32, applied.1 as i32);
                        if (s.frame.w, s.frame.h) != target && !s.capture_resolution_applied {
                            // The source window must be visually hidden before
                            // changing its client size. Otherwise a static PIP
                            // itself expands from (for example) 366x206 to
                            // 1280x720 on the desktop and looks like a giant
                            // zoomed frame even though the overlay is hidden.
                            if s.hide_source_pending {
                                if let Some(was_layered) = win32::hide_window_visual(s.hwnd) {
                                    s.hide_source_pending = false;
                                    s.hid_source = Some(was_layered);
                                    status.lock().unwrap().hidden_src = Some((s.hwnd, was_layered));
                                    log::info!(
                                        "source {:#x} visually hidden before deferred capture-resolution resize (was_layered={was_layered})",
                                        s.hwnd
                                    );
                                } else {
                                    reveal_ready = false;
                                    status.lock().unwrap().last_error = Some(
                                        "ソース画面を安全に隠せなかったため、キャプチャ解像度の変更を中止しました"
                                            .to_string(),
                                    );
                                    log::error!(
                                        "source {:#x} visual hide failed; deferred resize cancelled",
                                        s.hwnd
                                    );
                                    s.deferred_capture_resolution = None;
                                }
                            }
                            // Keep the overlay hidden while the source changes size.
                            // A source-sized transition texture appeared as a giant
                            // zoomed frame when a static browser/PIP did not repaint.
                            if reveal_ready
                                && win32::resize_client_area(s.hwnd, applied.0, applied.1)
                            {
                                s.capture_resolution_applied = true;
                                s.capture_resolution_wait_started = Some(Instant::now());
                                s.capture_resolution_repaint_last = None;
                                s.capture_resolution_nudge_done = false;
                                s.capture_resolution_native_fallback = false;
                                win32::request_window_repaint(s.hwnd);
                                reveal_ready = false;
                                log::info!(
                                    "capture-resolution applied before overlay reveal: hwnd={:#x} requested={}x{} client={}x{}; waiting for first exact filtered frame",
                                    s.hwnd,
                                    requested.0,
                                    requested.1,
                                    applied.0,
                                    applied.1
                                );
                            } else if reveal_ready {
                                s.deferred_capture_resolution = None;
                                s.capture_resolution_wait_started = None;
                                let actual = win32::client_rect_on_screen(s.hwnd)
                                    .map(|(_, _, w, h)| format!("{w}x{h}"))
                                    .unwrap_or_else(|| "unavailable".to_string());
                                log::warn!(
                                    "deferred capture-resolution rejected before reveal: hwnd={:#x} requested={}x{} applied={}x{} actual={actual}; revealing native filtered frame",
                                    s.hwnd,
                                    requested.0,
                                    requested.1,
                                    applied.0,
                                    applied.1
                                );
                            }
                        }
                    }
                    if reveal_ready && s.hide_source_pending && !s.source_monitor_fullscreen {
                        if let Some(was_layered) = win32::hide_window_visual(s.hwnd) {
                            s.hide_source_pending = false;
                            s.hid_source = Some(was_layered);
                            status.lock().unwrap().hidden_src = Some((s.hwnd, was_layered));
                            log::info!(
                                "source {:#x} visually hidden after first valid filtered frame (was_layered={was_layered})",
                                s.hwnd
                            );
                        } else {
                            reveal_ready = false;
                            let message = "ソース画面を非表示にできなかったため、二重表示を防ぐため拡大画面を表示しませんでした";
                            status.lock().unwrap().last_error = Some(message.to_string());
                            log::error!(
                                "source {:#x} visual hide failed; overlay kept hidden",
                                s.hwnd
                            );
                        }
                    }
                    if reveal_ready
                        && let Some((requested, applied)) = s.deferred_capture_resolution
                    {
                        let target = (applied.0 as i32, applied.1 as i32);
                        if (s.frame.w, s.frame.h) == target {
                            s.deferred_capture_resolution = None;
                            s.capture_resolution_applied = false;
                            s.capture_resolution_wait_started = None;
                            s.capture_resolution_repaint_last = None;
                            s.capture_resolution_nudge_done = false;
                            s.capture_resolution_native_fallback = false;
                            s.last_client_rect = win32::client_rect_on_screen(s.hwnd);
                            s.last_geom_change = None;
                            s.starve_released = false;
                            log::info!(
                                "capture-resolution filtered frame ready before overlay reveal: hwnd={:#x} requested={}x{} frame={}x{}",
                                s.hwnd,
                                requested.0,
                                requested.1,
                                s.frame.w,
                                s.frame.h
                            );
                        } else if s.capture_resolution_applied
                            && s.capture_resolution_native_fallback
                        {
                            log::info!(
                                "capture-resolution static native frame ready before overlay reveal: frame={}x{} target={}x{}; awaiting real target repaint in background",
                                s.frame.w,
                                s.frame.h,
                                target.0,
                                target.1
                            );
                        } else if !s.capture_resolution_applied {
                            reveal_ready = false;
                        } else {
                            reveal_ready = false;
                        }
                    }
                    if reveal_ready {
                        reveal_ready = crate::input::InputSystem::prepare_overlay_reveal();
                    }
                    if reveal_ready && s.provider_transition_input_suspended {
                        // v421: provider-transition suspension is only a guard while
                        // the new provider/session has not produced a usable frame.
                        // Reaching this point proves the complete filter chain already
                        // produced the first target-sized filtered frame and cursor
                        // handoff preparation succeeded. Waiting specifically for a
                        // later synthetic/interpolated midpoint left DirectML RIFE
                        // sessions suspended indefinitely on routes that present the
                        // first real filtered frame before any midpoint diagnostic.
                        crate::input::InputSystem::resume_transition_suspended();
                        s.provider_transition_input_suspended = false;
                        s.provider_transition_input_suspended_since = None;
                        s.input_reenable_after = Some(Instant::now() + Duration::from_millis(120));
                        log::info!(
                            "provider-transition-ready: first-filtered-frame-ready-before-overlay-reveal input=resume-after-120ms"
                        );
                    }
                    if reveal_ready {
                        // The persistent WGL HWND is double-buffered. On AMD,
                        // a very fast GLSL chain can finish and reveal before
                        // DWM has retired the other buffer from the previous
                        // size/session, exposing one composition of a giant
                        // crop. ONNX cold start only hid this race by taking
                        // longer. Fill the second buffer with the same complete
                        // filtered frame while alpha is still zero, then show.
                        if let Err(error) = overlay.present(gc, final_tex) {
                            reveal_ready = false;
                            status.lock().unwrap().last_error =
                                Some(format!("transition surface prime failed: {error:#}"));
                        } else {
                            log::info!(
                                "overlay transition surface double-primed before first reveal"
                            );
                        }
                    }
                    if reveal_ready {
                        if crate::logging::diagnostics_enabled() {
                            win32::log_helper_compositor_snapshot(
                                "reveal-before-show",
                                win32::main_gui_hwnd(),
                                win32::panel_gdi_host_hwnd(),
                                overlay.hwnd().0 as isize,
                                crate::input::cursor_sprite_hwnd(),
                                s.hwnd,
                            );
                        }
                        overlay.show();
                        let overlay_hwnd = overlay.hwnd().0 as isize;
                        let gui_priority_hwnd = win32::main_gui_hwnd();
                        let gui_priority_topmost = gui_priority_hwnd != 0
                            && win32::is_window_valid(gui_priority_hwnd)
                            && win32::is_topmost(gui_priority_hwnd);
                        let panel_hwnd = win32::panel_gdi_host_hwnd();
                        let panel_visible = panel_hwnd != 0
                            && win32::is_window_valid(panel_hwnd)
                            && win32::is_window_visible(panel_hwnd);

                        if gui_priority_topmost {
                            // Modern GUI-topmost fix, intentionally isolated to
                            // ON only. Reassert GUI > panel > overlay and commit
                            // that hierarchy before continuing.
                            enforce_gui_priority(
                                gui_priority_hwnd,
                                true,
                                if panel_visible { panel_hwnd } else { 0 },
                                overlay_hwnd,
                            );
                            overlay.flush_compositor();
                        } else {
                            // v235 path: when GUI topmost is OFF, show() is the
                            // complete reveal operation. Do not run the newer
                            // stack normalizer or an extra DwmFlush afterwards.
                            log::info!(
                                "overlay-reveal-path: gui_topmost=false mode=v235-show-only"
                            );
                        }

                        if crate::logging::diagnostics_enabled() {
                            win32::log_helper_compositor_snapshot(
                                if gui_priority_topmost {
                                    "reveal-after-topmost-restack"
                                } else {
                                    "reveal-after-v235-off-path"
                                },
                                gui_priority_hwnd,
                                panel_hwnd,
                                overlay_hwnd,
                                crate::input::cursor_sprite_hwnd(),
                                s.hwnd,
                            );
                            let panel_above = panel_hwnd == 0
                                || !panel_visible
                                || win32::window_is_above(panel_hwnd, overlay_hwnd);
                            log::info!(
                                "overlay-reveal-zorder-postcondition: gui_topmost={} panel_visible={} panel={:#x} overlay={:#x} panel_owner={:#x} cursor_owner={:#x} panel_above_overlay={}",
                                gui_priority_topmost,
                                panel_visible,
                                panel_hwnd,
                                overlay_hwnd,
                                win32::window_owner(panel_hwnd),
                                crate::input::cursor_sprite_owner(),
                                panel_above
                            );
                        }
                        log::info!(
                            "overlay shown after atomic cursor handoff and first target-sized filtered frame"
                        );
                        // v348t fullscreen handoff: OverlayWindow::show() does
                        // DwmFlush after alpha=255, so the complete filtered
                        // frame is already composited before the fullscreen
                        // source becomes transparent. This removes the desktop
                        // flash caused by the old source-hide -> overlay-show
                        // ordering. Windowed/deferred-resize behavior is left
                        // unchanged.
                        if s.source_monitor_fullscreen && s.hide_source_pending {
                            if let Some(was_layered) = win32::hide_window_visual(s.hwnd) {
                                s.hide_source_pending = false;
                                s.hid_source = Some(was_layered);
                                status.lock().unwrap().hidden_src = Some((s.hwnd, was_layered));
                                log::info!(
                                    "source {:#x} visually hidden after opaque fullscreen overlay commit (was_layered={was_layered})",
                                    s.hwnd
                                );
                            } else {
                                // The opaque overlay already covers the source,
                                // so hiding failure is not allowed to expose the
                                // desktop or abort a valid fullscreen capture.
                                log::warn!(
                                    "source {:#x} fullscreen visual hide failed after overlay commit; overlay remains visible",
                                    s.hwnd
                                );
                            }
                        }
                        // The revealed frame can still be the transition
                        // shield while TensorRT compiles asynchronously.
                    }
                }
            }
            if let Some(prev) = s.last_tex.take() {
                gc.recycle(prev);
            }
            let mut keep = vec![final_tex];
            keep.extend_from_slice(keep_extra);
            gc.release_frame(&keep);
            s.last_tex = Some(final_tex);
            s.last_present = Instant::now();
            s.last_filter_retry_log = None;
            status.lock().unwrap().presented += 1;
            let elapsed_ms = present_time.instant.duration_since(t0).as_secs_f64() * 1000.0;
            s.last_process_ms = elapsed_ms;
            update_glsl_guard_after_processed(s, status);
            let present_latency_ms =
                timing.map(|frame_timing| frame_present_latency_ms(frame_timing, present_time));
            metrics.set_internal_size((chain_tex.w(), chain_tex.h()));
            metrics.frame(
                true,
                captures,
                present_latency_ms,
                Some(elapsed_ms),
                s.in_size,
                (final_tex.w(), final_tex.h()),
            );
            if stats {
                log_metrics_if_due(
                    s,
                    metrics,
                    out_size,
                    (chain_tex.w(), chain_tex.h()),
                    (final_tex.w(), final_tex.h()),
                );
            }
        }
        Err(e) => {
            release_gpu_interp_permit_after_post_submit(s);
            s.paced_present_deadline = None;
            let msg = format!("{e:#}");
            if crate::render::onnx_stage::onnx_cancel_requested() {
                log::info!("filter-chain inference cancelled for Stop: {msg}");
                gc.release_frame(keep_extra);
                s.last_process_ms = t0.elapsed().as_secs_f64() * 1000.0;
                return;
            }
            {
                let mut g = status.lock().unwrap();
                let family = msg.split(" Status Message:").next().unwrap_or(&msg);
                if !g
                    .chain_errors
                    .iter()
                    .any(|existing| existing.starts_with(family))
                {
                    log::error!("filter chain processing failed: {msg}");
                    g.chain_errors.push(msg);
                }
            }
            // A transient stage failure must not replace a valid filtered
            // image with the raw source. Hold the last good frame and retry
            // the complete chain on the next capture.
            if let Some(last_good) = s.last_tex {
                let _ = overlay.present(gc, last_good);
                let mut keep = vec![last_good];
                keep.extend_from_slice(keep_extra);
                gc.release_frame(&keep);
                if s.last_filter_retry_log
                    .is_none_or(|last| last.elapsed() >= Duration::from_secs(2))
                {
                    log::warn!("filter-chain retry: holding last valid filtered frame");
                    s.last_filter_retry_log = Some(Instant::now());
                }
            } else {
                gc.release_frame(keep_extra);
                if s.last_filter_retry_log
                    .is_none_or(|last| last.elapsed() >= Duration::from_secs(2))
                {
                    log::warn!("filter-chain retry: waiting for first valid filtered frame");
                    s.last_filter_retry_log = Some(Instant::now());
                }
            }
            s.last_process_ms = t0.elapsed().as_secs_f64() * 1000.0;
            update_glsl_guard_after_processed(s, status);
        }
    }
}

fn log_metrics_if_due(
    s: &mut Session,
    metrics: &Metrics,
    virtual_output: (i32, i32),
    internal: (i32, i32),
    displayed: (i32, i32),
) {
    if s.last_metrics_log.elapsed() < Duration::from_secs(1) {
        return;
    }
    if !crate::logging::diagnostics_enabled() {
        s.last_metrics_log = Instant::now();
        return;
    }
    let snap = metrics.snapshot();
    log::debug!(
        "stats-resolution: input={}x{} virtual_output={}x{} internal={}x{} displayed={}x{}",
        s.in_size.0,
        s.in_size.1,
        virtual_output.0,
        virtual_output.1,
        internal.0,
        internal.1,
        displayed.0,
        displayed.1
    );
    let (present_ms, present_jitter_ms, present_min_ms, present_max_ms) =
        s.present_cadence.summary();
    log::debug!(
        "stats-snapshot: source_locked_fps={:.3} capture_dequeue_fps={:.1} present_fps={:.1} total_ms={:.2} delay_frames={} in={}x{} out={}x{} last_process_ms={:.2} compute_ms={:.2} path_ms(upload={:.2},chain={:.2},resample={:.2},post={:.2},pacer_wait={:.2},present_call={:.2},swap={:.2}) smooth={} cadence_ms={:.3} present_interval_ms={:.2} jitter_ms={:.2} present_min_ms={:.2} present_max_ms={:.2} queue_depth={} queue_dropped={}",
        snap.source_fps,
        snap.capture_fps,
        snap.present_fps,
        snap.total_ms,
        snap.lag_frames,
        snap.in_size.0,
        snap.in_size.1,
        snap.out_size.0,
        snap.out_size.1,
        s.last_process_ms,
        s.last_compute_ms,
        s.last_upload_submit_ms,
        s.last_chain_submit_ms,
        s.last_resample_submit_ms,
        s.last_post_submit_ms,
        s.last_pacer_wait_ms,
        s.last_present_call_ms,
        s.last_present_block_ms,
        s.smooth_pacing,
        s.cadence.period_s().unwrap_or(0.0) * 1000.0,
        present_ms,
        present_jitter_ms,
        present_min_ms,
        present_max_ms,
        s.source.queued_len(),
        s.source.queued_dropped()
    );
    for (name, st) in snap.display_stages_in_order(&s.chain.metric_stage_order()) {
        log::debug!(
            "stats-stage: ms={:.2} kind={} name={}",
            st.ms,
            st.kind,
            name
        );
    }
    s.last_metrics_log = Instant::now();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::wgc::FrameBuf;
    use half::f16;
    use std::time::{Duration, Instant};

    #[test]
    fn directml_rife_height_limit_matches_mpv_1440_policy() {
        assert_eq!(directml_rife_limited_size(3416, 1920), Some((2562, 1440)));
        assert_eq!(directml_rife_limited_size(2560, 1920), Some((1920, 1440)));
        assert_eq!(directml_rife_limited_size(3840, 2160), Some((2560, 1440)));
        assert_eq!(directml_rife_limited_size(2560, 1440), None);
        assert_eq!(directml_rife_limited_size(1920, 1080), None);
    }

    #[test]
    fn geometry_watchdog_ignores_position_only_window_moves() {
        let original = (100, 200, 1896, 1011);
        let moved = (800, 600, 1896, 1011);
        let resized = (800, 600, 1900, 1014);
        assert!(!rect_size_changed(original, moved));
        assert!(rect_size_changed(original, resized));
    }

    #[test]
    fn monitor_sized_capture_canvas_is_not_source_fullscreen_by_pixels_alone() {
        let windowed_sig = Some((1, 2, false));
        assert!(!should_enter_live_fullscreen(
            false,
            true,
            Some((1920, 1080)),
            (1920, 1080),
            windowed_sig,
            windowed_sig,
            None,
            false,
        ));
    }

    #[test]
    fn real_fullscreen_presentation_change_still_wins_at_monitor_sized_canvas() {
        assert!(should_enter_live_fullscreen(
            false,
            true,
            Some((1920, 1080)),
            (1920, 1080),
            Some((1, 2, false)),
            Some((0, 0, false)),
            None,
            false,
        ));
    }

    #[test]
    fn capture_resize_handoff_cannot_false_trigger_fullscreen() {
        assert!(!should_enter_live_fullscreen(
            false,
            true,
            None,
            (1920, 1080),
            Some((1, 2, false)),
            Some((1, 2, false)),
            Some((1920, 1080)),
            false,
        ));
        // Without a matching Neo capture canvas or live resize intent, the same stable monitor cover
        // is source-owned once the short resize hand-off has elapsed.
        assert!(should_enter_live_fullscreen(
            false,
            true,
            None,
            (1920, 1080),
            Some((1, 2, false)),
            Some((1, 2, false)),
            None,
            false,
        ));
    }

    #[test]
    fn drag_overlay_target_is_absolute_and_clamp_history_independent() {
        let source_start = (100, 100);
        let overlay_start = (300, 200, 800, 450);
        let monitor = (0, 0, 1920, 1080);

        // Far beyond the right edge clamps the visible overlay.
        assert_eq!(
            drag_overlay_target(source_start, (2000, 100), overlay_start, monitor),
            (1120, 200)
        );
        // A small reversal while the hidden source is still far beyond the
        // clamp must remain clamped; an incremental-delta follower would creep.
        assert_eq!(
            drag_overlay_target(source_start, (1900, 100), overlay_start, monitor),
            (1120, 200)
        );
        // Returning to an in-range absolute displacement lands at the exact
        // deterministic position, independent of the path taken before it.
        assert_eq!(
            drag_overlay_target(source_start, (500, 250), overlay_start, monitor),
            (700, 350)
        );
    }

    #[test]
    fn client_drag_raw_anchor_keeps_moving_after_hidden_source_hits_os_wall() {
        // The v425 failure log shows Chromium holding a 640px PIP at x=-284
        // while the user's drag continued. Visual placement must therefore
        // depend only on the raw LEFT-down span, never on source HWND deltas.
        let overlay_origin = (808, 156);
        let raw_down = (1110, 420);
        assert_eq!(
            client_drag_overlay_position_from_raw(overlay_origin, raw_down, (600, 420)),
            (298, 156)
        );
        // Continue another 500px left even if the native source x is still
        // frozen at -284. There is deliberately no monitor clamp here.
        assert_eq!(
            client_drag_overlay_position_from_raw(overlay_origin, raw_down, (100, 420)),
            (-202, 156)
        );
    }

    #[test]
    fn client_drag_raw_anchor_coalesces_stalled_samples_without_incremental_debt() {
        let overlay_origin = (500, 300);
        let raw_down = (900, 500);
        // A render/interpolation stall may skip every intermediate mouse sample.
        // Absolute-from-origin placement must land directly at the latest hand
        // position rather than replaying/banking delayed source deltas.
        assert_eq!(
            client_drag_overlay_position_from_raw(overlay_origin, raw_down, (1250, 760)),
            (850, 560)
        );
        // Reversing is deterministic and history-independent.
        assert_eq!(
            client_drag_overlay_position_from_raw(overlay_origin, raw_down, (820, 460)),
            (420, 260)
        );
    }

    #[test]
    fn full_window_wgc_reference_allows_even_pad_without_false_restart() {
        let dwm = (1696, 447, 1900, 1071);
        assert!(!frame_size_mismatches_reference((1900, 1072), dwm));
        assert!(frame_size_mismatches_reference((2100, 1200), dwm));
    }

    #[test]
    fn no_engage_geometry_keeps_only_latest_language_mode_or_drag_sample() {
        let mut pending = PendingNoEngage::default();
        // Simulate three measurements arriving while the render thread is busy:
        // a compact language/layout, an expanded translation, then the latest
        // dragged position. Only the newest real Win32 rectangle may survive.
        pending.publish(vec![(40, 50, 746, 499, 0x1111)]);
        pending.publish(vec![(40, 50, 996, 899, 0x1111)]);
        pending.publish(vec![(420, 180, 996, 899, 0x1111)]);
        let (latest, overwritten) = pending.take_latest().expect("latest geometry");
        assert_eq!(latest, vec![(420, 180, 996, 899, 0x1111)]);
        assert_eq!(overwritten, 2);
        assert!(pending.take_latest().is_none());
    }

    #[test]
    fn neoflow_queue_cannot_accumulate_several_audio_frames() {
        assert!(NEOFLOW_QUEUE_MAX <= 3);
    }

    #[test]
    fn static_capture_fallback_builds_the_requested_frame_without_new_wgc_input() {
        let mut frame = FrameBuf {
            w: 2,
            h: 1,
            data: vec![255, 0, 0, 255, 0, 0, 255, 255],
            hdr: false,
            seq: 7,
            received_at: None,
            source_time_100ns: Some(123),
        };
        assert!(resize_static_rgba8_frame(&mut frame, (4, 2)));
        assert_eq!((frame.w, frame.h), (4, 2));
        assert_eq!(frame.data.len(), 4 * 2 * 4);
        assert_eq!(&frame.data[..4], &[255, 0, 0, 255]);
        assert_eq!(&frame.data[frame.data.len() - 4..], &[0, 0, 255, 255]);
        assert_eq!(frame.seq, 8);
        assert!(frame.received_at.is_some());
        assert_eq!(frame.source_time_100ns, None);
    }

    #[test]
    fn explicit_capture_canvas_crops_and_pads_without_resampling() {
        let mut crop = FrameBuf {
            w: 4,
            h: 1,
            data: vec![1, 0, 0, 255, 2, 0, 0, 255, 3, 0, 0, 255, 4, 0, 0, 255],
            hdr: false,
            ..FrameBuf::default()
        };
        assert!(fit_frame_canvas_dot_by_dot(&mut crop, (2, 1)));
        assert_eq!(crop.data, vec![2, 0, 0, 255, 3, 0, 0, 255]);

        let mut pad = crop.clone();
        assert!(fit_frame_canvas_dot_by_dot(&mut pad, (3, 2)));
        assert_eq!((pad.w, pad.h), (3, 2));
        assert_eq!(&pad.data[..12], &[2, 0, 0, 255, 3, 0, 0, 255, 0, 0, 0, 255]);

        let mut reject_large_pad = crop.clone();
        assert!(!fit_frame_canvas_dot_by_dot(&mut reject_large_pad, (4, 3)));
        assert_eq!((reject_large_pad.w, reject_large_pad.h), (2, 1));
    }

    #[test]
    fn pending_client_resize_cannot_be_faked_by_cropping_the_old_large_frame() {
        assert!(!should_apply_capture_canvas((1036, 776), (640, 480), true));
        assert!(should_apply_capture_canvas((1036, 776), (640, 480), false));
        assert!(!should_apply_capture_canvas((640, 480), (640, 480), false));
    }

    #[test]
    fn onnx_x2_always_requests_one_midpoint_even_after_fractional_cadence() {
        let mut cadence = FlowOutputCadence::default();
        assert_eq!(cadence.phases(2.5).len(), 3);
        assert_eq!(
            onnx_interpolation_phases(2, 2.0, false, &mut cadence),
            vec![0.5, 1.0]
        );
        assert_eq!(
            onnx_interpolation_phases(2, 2.0, false, &mut cadence),
            vec![0.5, 1.0]
        );
    }

    #[test]
    fn x4_and_x5_generate_all_intermediate_phases_on_120hz() {
        let mut cadence = FlowOutputCadence::default();
        assert_eq!(
            onnx_interpolation_phases(4, 4.0, false, &mut cadence),
            vec![0.25, 0.5, 0.75, 1.0]
        );
        cadence.reset();
        assert_eq!(
            onnx_interpolation_phases(5, 5.0, false, &mut cadence),
            vec![0.2, 0.4, 0.6, 0.8, 1.0]
        );
        // 24000/1001 sources on a nominal 120 Hz display may report a ratio
        // a few thousandths below five. They still require the exact x5 grid.
        assert_eq!(
            onnx_interpolation_phases(5, 4.995, false, &mut cadence),
            vec![0.2, 0.4, 0.6, 0.8, 1.0]
        );
    }

    #[test]
    fn x5_startup_fractional_cadence_never_runs_a_fifth_synthetic_inference() {
        let mut cadence = FlowOutputCadence::default();
        // Covers a transient cadence overshoot during interpolation startup.
        // The old accumulator emitted five synthetic timesteps and no endpoint.
        let phases = onnx_interpolation_phases(5, 4.14, false, &mut cadence);
        assert!(
            phases.len() <= 4,
            "x5 may generate at most four synthetic mids"
        );
        assert!(phases.iter().all(|phase| *phase < 1.0 - 1e-5));
        for pair in phases.windows(2) {
            assert!(pair[0] < pair[1], "every generated midpoint must be unique");
        }
    }

    #[test]
    fn x5_rife_and_drba_contract_is_four_unique_model_frames_plus_real_endpoint() {
        // RIFE and DRBA share onnx_interpolation_phases before the stage kind
        // is dispatched, so this is the common x5 generation contract: four
        // genuinely different model timesteps, never duplicated presents.
        for _kind in ["RIFE", "DRBA"] {
            let mut cadence = FlowOutputCadence::default();
            let phases = onnx_interpolation_phases(5, 5.0, false, &mut cadence);
            let mids = phases
                .iter()
                .copied()
                .filter(|phase| *phase < 1.0 - 1e-5)
                .collect::<Vec<_>>();
            assert_eq!(mids, vec![0.2, 0.4, 0.6, 0.8]);
            assert_eq!(phases.last().copied(), Some(1.0));
        }
    }

    #[test]
    fn gpu_interp_lead_accounts_for_deferred_shader_gpu_tail() {
        let period = 1.0 / 120.0;
        // GLSL submission itself can take ~2 ms while its deferred GPU work
        // is paid by the following SwapBuffers. The preparation lead must
        // include that tail, but remain below one complete output slot.
        let lead = gpu_interp_present_lead_s(2.0, 8.0, period);
        assert!(lead > period * 0.85, "lead={lead}");
        assert!(lead <= period * 0.90 + f64::EPSILON);
    }

    #[test]
    fn pacing_guard_swap_block_does_not_extend_next_interp_slot() {
        for fps in [48.0, 72.0, 96.0, 120.0] {
            for block_ms in [1u64, 3, 8] {
                let period_s = 1.0 / fps;
                let period = Duration::from_secs_f64(period_s);
                let base = Instant::now();
                let mut cadence = FlowOutputCadence::default();
                cadence.next_present = Some(base + period);

                // Submit exactly on the scheduled slot, then model healthy
                // NVIDIA/AMD/DWM block durations. Completion is late,
                // submission is not, so the next slot must not move.
                let started = base;
                let completed = started + Duration::from_millis(block_ms);
                assert!(!cadence.observe_blocking_present(
                    started,
                    completed,
                    period_s,
                    block_ms as f64 / 1000.0,
                ));
                assert_eq!(cadence.next_present, Some(base + period));
            }
        }
    }

    #[test]
    fn pacing_guard_irrecoverable_lateness_preserves_original_phase_grid() {
        let period_s = 1.0 / 72.0;
        let period = Duration::from_secs_f64(period_s);
        let base = Instant::now();
        let mut cadence = FlowOutputCadence::default();
        cadence.next_present = Some(base + period);
        cadence.present_period_s = period_s;

        // Model a genuinely irrecoverable delay of more than two slots. The
        // old path re-anchored to `started + period`, making a random driver
        // return time the new permanent cadence phase. The recovered deadline
        // must instead stay on the original base + N*period grid.
        let started = base + period.mul_f64(2.5);
        let completed = started + Duration::from_millis(4);
        assert!(cadence.observe_blocking_present(started, completed, period_s, 0.004,));
        let recovered = cadence.next_present.unwrap();
        let recovered_slots = recovered.duration_since(base).as_secs_f64() / period_s;
        assert!(
            (recovered_slots - 3.0).abs() < 0.000_1,
            "slots={recovered_slots}"
        );

        // The next on-grid submission must remain stable even if SwapBuffers
        // blocks again; block duration must never accumulate into the clock.
        let next_started = base + period.mul_f64(3.0);
        cadence.next_present = Some(base + period.mul_f64(4.0));
        let next_completed = next_started + Duration::from_millis(4);
        assert!(!cadence.observe_blocking_present(next_started, next_completed, period_s, 0.004,));
        let stable = cadence.next_present.unwrap();
        let stable_slots = stable.duration_since(base).as_secs_f64() / period_s;
        assert!((stable_slots - 4.0).abs() < 0.000_1, "slots={stable_slots}");
    }

    #[test]
    fn interp_cadence_period_lock_rebases_from_last_submit_not_loop_time() {
        let base = Instant::now();
        let mut cadence = FlowOutputCadence::default();
        let transient_period_s = 1.0 / 44.0;
        let locked_period_s = 1.0 / 48.0;
        cadence.present_period_s = transient_period_s;
        cadence.last_present_started = Some(base);
        cadence.next_present = Some(base + Duration::from_secs_f64(transient_period_s));

        // The cadence estimator locks after startup while the render loop is at
        // an unrelated timestamp. The new 48fps phase must be derived from the
        // actual previous back-buffer submission.
        let loop_now = base + Duration::from_millis(7);
        let deadline = cadence.reserve_present_deadline(loop_now, locked_period_s);
        let deadline_slots = deadline.duration_since(base).as_secs_f64() / locked_period_s;
        let next_slots = cadence
            .next_present
            .unwrap()
            .duration_since(base)
            .as_secs_f64()
            / locked_period_s;
        assert!(
            (deadline_slots - 1.0).abs() < 0.000_1,
            "slots={deadline_slots}"
        );
        assert!((next_slots - 2.0).abs() < 0.000_1, "slots={next_slots}");
    }

    #[test]
    fn interp_cadence_missed_slots_advance_without_random_reanchor_or_burst() {
        for fps in [48.0, 72.0, 96.0, 120.0] {
            let period_s = 1.0 / fps;
            let period = Duration::from_secs_f64(period_s);
            let base = Instant::now();
            let mut cadence = FlowOutputCadence::default();
            cadence.present_period_s = period_s;
            cadence.next_present = Some(base);

            // Arrive 2.4 slots late. Reserve the latest slot on the original
            // phase grid, leaving the following deadline in the future instead
            // of anchoring the whole clock to this delayed loop iteration.
            let now = base + period.mul_f64(2.4);
            let deadline = cadence.reserve_present_deadline(now, period_s);
            let deadline_slots = deadline.duration_since(base).as_secs_f64() / period_s;
            let next = cadence.next_present.unwrap();
            let next_slots = next.duration_since(base).as_secs_f64() / period_s;
            assert!(
                (deadline_slots - 2.0).abs() < 0.000_1,
                "fps={fps} slots={deadline_slots}"
            );
            assert!(
                (next_slots - 3.0).abs() < 0.000_1,
                "fps={fps} slots={next_slots}"
            );
            assert!(next > now);
        }
    }

    #[test]
    fn pacing_guard_integer_multi_output_slot_policy_is_uniform() {
        assert_eq!(gpu_interp_slot_policy(2, 1), (false, false));
        assert_eq!(gpu_interp_slot_policy(3, 1), (false, false));
        assert_eq!(gpu_interp_slot_policy(3, 2), (true, true));
        assert_eq!(gpu_interp_slot_policy(4, 3), (true, true));
        assert_eq!(gpu_interp_slot_policy(5, 4), (true, true));
    }

    #[test]
    fn gpu_x3_mid_detach_covers_directml_and_tensorrt_only() {
        use crate::render::onnx_stage::OnnxProvider;
        assert!(provider_uses_gpu_x3_mid_detach(OnnxProvider::DirectML));
        assert!(provider_uses_gpu_x3_mid_detach(OnnxProvider::TensorRT));
        assert!(!provider_uses_gpu_x3_mid_detach(OnnxProvider::Cuda));
    }

    #[test]
    fn pacing_guard_x2_to_x5_keep_unique_model_phases_and_expected_periods() {
        for (factor, expected) in [
            (2u32, vec![0.5f32, 1.0f32]),
            (3u32, vec![1.0f32 / 3.0, 2.0f32 / 3.0, 1.0f32]),
            (4u32, vec![0.25f32, 0.5f32, 0.75f32, 1.0f32]),
            (5u32, vec![0.2f32, 0.4f32, 0.6f32, 0.8f32, 1.0f32]),
        ] {
            let mut cadence = FlowOutputCadence::default();
            let phases = onnx_interpolation_phases(factor, factor as f64, false, &mut cadence);
            assert_eq!(phases, expected);
            for pair in phases.windows(2) {
                assert!(pair[0] < pair[1], "model phases must be unique");
            }
            let output_period = (1.0 / 24.0) / factor as f64;
            let expected_fps = 1.0 / output_period;
            assert!((expected_fps - 24.0 * factor as f64).abs() < 1e-9);
        }
    }

    #[test]
    fn interpolation_sequence_gap_with_contiguous_timestamp_preserves_history() {
        let now = Instant::now();
        let current = FrameBuf {
            seq: 108,
            received_at: Some(now + Duration::from_millis(42)),
            source_time_100ns: Some(10_416_667),
            ..FrameBuf::default()
        };
        assert_eq!(
            classify_gpu_interp_continuity(
                100,
                Some(now),
                Some(10_000_000),
                &current,
                Some(1.0 / 24.0),
                1.0 / 24.0,
            ),
            GpuInterpContinuity::Continuous
        );
    }

    #[test]
    fn repeated_source_timestamp_is_skipped_not_treated_as_a_break() {
        let now = Instant::now();
        let current = FrameBuf {
            seq: 101,
            received_at: Some(now + Duration::from_millis(16)),
            source_time_100ns: Some(10_000_000),
            ..FrameBuf::default()
        };
        assert_eq!(
            classify_gpu_interp_continuity(
                100,
                Some(now),
                Some(10_000_000),
                &current,
                Some(1.0 / 24.0),
                1.0 / 24.0,
            ),
            GpuInterpContinuity::Duplicate
        );
    }

    #[test]
    fn true_timestamp_discontinuity_rebuilds_interpolation_history() {
        let now = Instant::now();
        let current = FrameBuf {
            seq: 101,
            received_at: Some(now + Duration::from_millis(250)),
            source_time_100ns: Some(13_000_000),
            ..FrameBuf::default()
        };
        assert_eq!(
            classify_gpu_interp_continuity(
                100,
                Some(now),
                Some(10_000_000),
                &current,
                Some(1.0 / 24.0),
                1.0 / 24.0,
            ),
            GpuInterpContinuity::Broken
        );
    }

    #[test]
    fn windowed_rebase_clamps_source_and_overlay_to_monitor_edges() {
        assert_eq!(clamp_axis_to_bounds(1613, 640, 0, 1920), 1280);
        assert_eq!(clamp_axis_to_bounds(-120, 640, 0, 1920), 0);
        assert_eq!(clamp_axis_to_bounds(896, 1024, 0, 1920), 896);
        // Regression for the v422 log: the final input clip was
        // (1532, 667, 640, 480), outside a 1920x1080 monitor. Both axes must
        // be rebased before the next ClipCursor ownership cycle.
        assert_eq!(clamp_axis_to_bounds(1532, 640, 0, 1920), 1280);
        assert_eq!(clamp_axis_to_bounds(667, 480, 0, 1080), 600);

        // v426 regression: the outer PIP had already been moved to y=360,
        // but its actual input client was still y=391..1110. Rebase decisions
        // must therefore use the client rect and request another -31 px move.
        assert_eq!(
            client_rect_rebase_delta((618, 391, 1280, 720), (0, 0, 1920, 1080)),
            (0, -31)
        );
        assert_eq!(
            client_rect_rebase_delta((618, 360, 1280, 720), (0, 0, 1920, 1080)),
            (0, 0)
        );
    }

    #[test]
    fn neoflow_uses_fractional_24p_to_60hz_cadence_without_starving_capture() {
        let ratio = refresh_limited_output_ratio(3, Some(1.0 / 24.0), Some(60.0));
        assert!((ratio - 2.5).abs() < 1e-9);
        assert_eq!(
            refresh_limited_output_ratio(3, Some(1.0 / 24.0), Some(120.0)),
            3.0
        );
        assert_eq!(
            refresh_limited_output_ratio(2, Some(1.0 / 24.0), Some(60.0)),
            2.0
        );
        assert!((refresh_limited_output_ratio(5, Some(1.0 / 24.0), Some(60.0)) - 2.5).abs() < 1e-9);
        assert_eq!(
            refresh_limited_output_ratio(5, Some(1.0 / 24.0), Some(120.0)),
            5.0
        );

        let mut cadence = FlowOutputCadence::default();
        let first = cadence.phases(ratio);
        let second = cadence.phases(ratio);
        assert_eq!(first, vec![0.2, 0.6, 1.0]);
        assert_eq!(second, vec![0.4, 0.8]);
        cadence.reset();
        let counts: Vec<usize> = (0..6).map(|_| cadence.phases(ratio).len()).collect();
        assert_eq!(counts, vec![3, 2, 3, 2, 3, 2]);
        assert_eq!(counts.iter().sum::<usize>(), 15);
    }

    #[test]
    fn neoflow_keeps_stable_24p_when_one_dwm_pair_looks_like_60p() {
        let period = neoflow_source_period_s(Some(1.0 / 24.0), Some(1.0 / 60.0), 1.0 / 24.0);
        assert_eq!(period, Some(1.0 / 24.0));
        assert_eq!(refresh_limited_output_ratio(2, period, Some(60.0)), 2.0);
    }

    #[test]
    fn interpolated_timing_uses_the_content_midpoint() {
        let mut frame = FrameBuf::default();
        frame.source_time_100ns = Some(1_416_667);
        let timing = FrameTiming::from_interpolated(&frame, Some(1_000_000), 0.5, Instant::now());
        assert_eq!(timing.source_time_100ns, Some(1_208_334));
    }

    const F60: i64 = 166_667; // 60fps source grid in 100ns units

    #[test]
    fn static_keepalive_is_one_hz() {
        assert_eq!(IDLE_KEEPALIVE_INTERVAL, Duration::from_secs(1));
    }

    #[test]
    fn interpolation_accepts_normal_and_coalesced_video_pairs() {
        assert!(interp_pair_is_contiguous(
            Some(1.0 / 24.0),
            Some(1.0 / 24.0),
            1.0 / 24.0
        ));
        assert!(interp_pair_is_contiguous(
            Some(2.0 / 24.0),
            Some(1.0 / 24.0),
            1.0 / 24.0
        ));
    }

    #[test]
    fn interpolation_rejects_static_or_tab_switch_gaps() {
        assert!(!interp_pair_is_contiguous(
            Some(0.783),
            Some(1.0 / 24.0),
            0.113
        ));
        assert!(!interp_pair_is_contiguous(Some(4.7), None, 4.7));
    }

    #[test]
    fn screenshot_png_writer_preserves_asymmetric_pixel_order() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("neo-screenshot-test-{stamp}.png"));
        let rgba = vec![
            255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
        ];
        let expected = rgba.clone();
        save_screenshot_async(path.clone(), 2, 2, rgba);
        let start = Instant::now();
        let mut decoded = None;
        while decoded.is_none() && start.elapsed() < Duration::from_secs(3) {
            decoded = (|| {
                let file = std::fs::File::open(&path).ok()?;
                let decoder = png::Decoder::new(std::io::BufReader::new(file));
                let mut reader = decoder.read_info().ok()?;
                let mut pixels = vec![0; reader.output_buffer_size()?];
                let info = reader.next_frame(&mut pixels).ok()?;
                (info.width == 2 && info.height == 2 && info.color_type == png::ColorType::Rgba)
                    .then(|| pixels[..info.buffer_size()].to_vec())
            })();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(decoded.as_deref(), Some(expected.as_slice()));
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn cadence_snap_keeps_loaded_24p_at_24fps() {
        assert!((snap_video_period(1.0 / 22.2) - 1.0 / 24.0).abs() < 1e-9);
        assert!((snap_video_period(1.0 / 24.6) - 1.0 / 24.0).abs() < 1e-9);
    }

    #[test]
    fn long_timestamp_history_keeps_exact_24_clock() {
        let film = snap_video_period_with_history(1001.0 / 24_000.0, 120);
        let exact = snap_video_period_with_history(1.0 / 24.0, 120);
        assert!((1.0 / film - 24.0).abs() < 1e-9);
        assert!((1.0 / exact - 24.0).abs() < 1e-9);
    }

    #[test]
    fn cadence_estimator_absorbs_stable_chromium_30p_callback_wobble() {
        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        cadence.observe(Some(t));
        for i in 0..80 {
            t += match i % 5 {
                0 => 312_500, // 32fps compositor update
                1 => 322_581, // 31fps compositor update
                _ => 333_333,
            };
            cadence.observe(Some(t));
        }
        assert!((cadence.period_s().unwrap() - 1.0 / 30.0).abs() < 1e-9);
    }

    #[test]
    fn cadence_estimator_does_not_lock_variable_game_near_30fps() {
        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        cadence.observe(Some(t));
        for i in 0..80 {
            t += match i % 4 {
                0 => 200_000, // 50fps
                1 => 250_000, // 40fps
                2 => 400_000, // 25fps
                _ => 500_000, // 20fps
            };
            cadence.observe(Some(t));
        }
        assert!(cadence.variable_rate());
    }

    fn run_cap(timestamps: &[i64], cap: u32) -> Vec<i64> {
        let interval = 10_000_000i64 / cap as i64;
        let mut deadline = None;
        timestamps
            .iter()
            .copied()
            .filter(|t| cap_accept(&mut deadline, *t, interval))
            .collect()
    }

    fn run_rate_aware_cap(timestamps: &[i64], cap: u32) -> Vec<i64> {
        let interval = 10_000_000i64 / cap as i64;
        let mut deadline = None;
        let mut gate = CapRateGate::default();
        timestamps
            .iter()
            .copied()
            .filter(|t| {
                if gate.source_is_within_cap(*t, interval) {
                    deadline = Some(*t + interval);
                    true
                } else {
                    cap_accept(&mut deadline, *t, interval)
                }
            })
            .collect()
    }

    #[test]
    fn onnx_x3_60hz_mode_is_narrow_and_does_not_match_other_refresh_rates() {
        assert!(onnx_x3_60hz_mode(Some(59.94)));
        assert!(onnx_x3_60hz_mode(Some(60.0)));
        assert!(!onnx_x3_60hz_mode(Some(59.0)));
        assert!(!onnx_x3_60hz_mode(Some(61.0)));
        assert!(!onnx_x3_60hz_mode(Some(75.0)));
        assert!(!onnx_x3_60hz_mode(Some(120.0)));
        assert!(!onnx_x3_60hz_mode(Some(144.0)));
        assert!(!onnx_x3_60hz_mode(Some(165.0)));
        assert!(!onnx_x3_60hz_mode(None));
    }

    #[test]
    fn onnx_x3_60hz_keeps_refresh_limited_60fps_cadence() {
        let period_24p = Some(1.0 / 24.0);
        let ratio = onnx_output_ratio(3, period_24p, Some(60.0));
        assert!((ratio - 2.5).abs() < 1e-9);
        assert!((onnx_output_ratio(3, Some(1.0 / 25.0), Some(60.0)) - 2.4).abs() < 1e-9);
        assert!((onnx_output_ratio(3, Some(1.0 / 30.0), Some(60.0)) - 2.0).abs() < 1e-9);
        let mut cadence = FlowOutputCadence::default();
        assert_eq!(
            onnx_interpolation_phases(3, ratio, true, &mut cadence),
            vec![0.2, 0.6, 1.0]
        );
        assert_eq!(
            onnx_interpolation_phases(3, ratio, true, &mut cadence),
            vec![0.4, 0.8]
        );
    }

    #[test]
    fn onnx_x3_non_60hz_is_strict_integer_and_clears_60hz_fractional_state() {
        let period = Some(1.0 / 24.0);
        assert_eq!(onnx_output_ratio(3, period, Some(120.0)), 3.0);
        assert_eq!(onnx_output_ratio(3, period, Some(75.0)), 3.0);
        let mut cadence = FlowOutputCadence::default();
        cadence.next_phase = Some(0.4);
        cadence.phase_step = 0.4;
        assert_eq!(
            onnx_interpolation_phases(3, 3.0, false, &mut cadence),
            vec![1.0 / 3.0, 2.0 / 3.0, 1.0]
        );
        assert!(cadence.next_phase.is_none());
        assert_eq!(cadence.phase_step, 0.0);
    }

    #[test]
    fn x3_uses_same_cooperative_slot_release_class_as_x4_x5() {
        assert_eq!(gpu_interp_slot_policy(3, 2), (true, true));
        assert_eq!(gpu_interp_slot_policy(4, 3), (true, true));
        assert_eq!(gpu_interp_slot_policy(5, 4), (true, true));
    }

    #[test]
    fn capped_15fps_x3_targets_45fps_on_a_120hz_monitor() {
        let capped_period = Some(1.0 / 15.0);
        assert_eq!(
            refresh_limited_output_ratio(3, capped_period, Some(120.0)),
            3.0
        );
        let mut cadence = FlowOutputCadence::default();
        let phases = onnx_interpolation_phases(3, 3.0, false, &mut cadence);
        assert_eq!(phases, vec![1.0 / 3.0, 2.0 / 3.0, 1.0]);
        assert_eq!(phases.len() * 15, 45);
    }

    #[test]
    fn cap_gate_engages_quickly_for_60_to_15_reduction() {
        let mut gate = CapRateGate::default();
        let interval = 10_000_000i64 / 15;
        assert!(gate.source_is_within_cap(0, interval));
        assert!(!gate.source_is_within_cap(F60, interval));
        assert!(!gate.source_is_within_cap(2 * F60, interval));
    }

    #[test]
    fn fps_cap_then_highlight_protection_then_duplicate_reduction_then_filters_is_the_chain_order()
    {
        let source = include_str!("engine.rs");
        assert!(source.contains(
            "order=wgc->fps-cap->highlight-protection->duplicate-reduction->glsl/onnx/interpolation"
        ));
        let cap = source
            .find("// fps上限 = TARGET-rate decimation")
            .expect("FPS cap gate");
        let highlight = source
            .find("if took_frame && !cap_skip && s.capture_hdr")
            .expect("highlight-protection gate");
        assert!(
            cap < highlight,
            "FPS cap must run before highlight protection"
        );
        assert!(source[cap..highlight].contains("!cap_geometry_transition"));
        let smooth_duplicate = source[cap..]
            .find("let mut smooth_content_duplicate = false;")
            .map(|offset| cap + offset)
            .expect("smooth duplicate gate");
        let fallback_duplicate = source[smooth_duplicate..]
            .find("let mut duplicate_skip = false;")
            .map(|offset| smooth_duplicate + offset)
            .expect("fallback duplicate gate");
        let post_reduction_sequence = source[fallback_duplicate..]
            .find("exact order: FPS cap -> duplicate reduction -> GLSL/ONNX/interpolation")
            .map(|offset| fallback_duplicate + offset)
            .expect("post-reduction contiguous sequence");
        let filter_chain = source[post_reduction_sequence..]
            .find("// render target size: at least 2x the source")
            .map(|offset| post_reduction_sequence + offset)
            .expect("filter-chain entry");
        assert!(cap < highlight);
        assert!(highlight < smooth_duplicate);
        assert!(smooth_duplicate < fallback_duplicate);
        assert!(fallback_duplicate < post_reduction_sequence);
        assert!(post_reduction_sequence < filter_chain);
        assert!(source[smooth_duplicate..fallback_duplicate].contains("&& !cap_skip"));
        assert!(source.contains("s.frame.seq = s.cap_filter_seq"));
    }

    #[test]
    fn cap_30_on_clean_60fps_grid_accepts_every_second_frame_evenly() {
        let ts: Vec<i64> = (0..120).map(|i| i * F60).collect();
        let accepted = run_cap(&ts, 30);
        // 2 seconds of 60fps input -> ~60 accepted at 30fps
        assert!((59..=61).contains(&accepted.len()), "{}", accepted.len());
        for w in accepted.windows(2) {
            let dt = w[1] - w[0];
            assert_eq!(dt, 2 * F60, "uneven content spacing: {dt}");
        }
    }

    #[test]
    fn cap_holds_target_rate_when_delivery_wobbles() {
        // the field case: 60fps content but ~1 of every 12 deliveries
        // coalesced away (55/s measured). Content spacing of ACCEPTED frames
        // must stay on the source grid and never fall below the interval.
        let ts: Vec<i64> = (0..240).filter(|i| i % 12 != 7).map(|i| i * F60).collect();
        let accepted = run_cap(&ts, 30);
        let span_s = (accepted.last().unwrap() - accepted[0]) as f64 / 1e7;
        let rate = (accepted.len() - 1) as f64 / span_s;
        assert!((rate - 30.0).abs() < 1.0, "rate {rate}");
        for w in accepted.windows(2) {
            let dt = w[1] - w[0];
            assert!(dt >= 2 * F60, "burst pair: {dt}");
            assert!(dt <= 3 * F60, "long gap: {dt}");
        }
    }

    #[test]
    fn cap_45_on_60fps_converges_to_45() {
        let ts: Vec<i64> = (0..600).map(|i| i * F60).collect();
        let accepted = run_cap(&ts, 45);
        let span_s = (accepted.last().unwrap() - accepted[0]) as f64 / 1e7;
        let rate = (accepted.len() - 1) as f64 / span_s;
        assert!((rate - 45.0).abs() < 0.5, "rate {rate}");
    }

    #[test]
    fn cap_above_source_rate_passes_every_frame() {
        // user spec: cap 30 on a 24fps source must play at 24fps. A 24p video
        // composited at 60Hz arrives on a 33.3/50ms (3:2 pulldown) grid whose
        // short step sits exactly ON the cap interval — with timestamp jitter
        // this rejected ~40% of frames in the field (12-17fps observed).
        let vbl: i64 = 166_667; // 60Hz vblank in 100ns
        for jitter in [0i64, -8_000, 8_000, -25_000] {
            let mut ts = Vec::new();
            let mut t = 0i64;
            for i in 0..240 {
                t += if i % 2 == 0 { 2 * vbl } else { 3 * vbl }; // 3:2 cadence
                ts.push(t + if i % 2 == 0 { jitter } else { 0 });
            }
            let accepted = run_cap(&ts, 30);
            assert_eq!(
                accepted.len(),
                ts.len(),
                "cap 30 dropped frames of a 24fps source (jitter {jitter})"
            );
        }
        // plain uniform 24fps grid too
        let ts: Vec<i64> = (1..=240).map(|i| i * 416_667).collect();
        assert_eq!(run_cap(&ts, 30).len(), ts.len());
    }

    #[test]
    fn rate_aware_cap_passes_irregular_24p_dwm_timestamps() {
        // Capture diagnostics can contain individual 16.7ms steps even though the stream
        // averages 24fps. A plain per-frame deadline mistakes those short
        // timestamp steps for a >30fps source and drops valid content.
        let vblank = 166_667i64;
        let mut t = 0i64;
        let ts: Vec<i64> = (0..240)
            .map(|i| {
                t += match i % 4 {
                    0 => vblank,
                    1 => 4 * vblank,
                    2 => 2 * vblank,
                    _ => 3 * vblank,
                };
                t
            })
            .collect();
        assert_eq!(run_rate_aware_cap(&ts, 30), ts);
    }

    #[test]
    fn rate_aware_cap_still_limits_60fps_to_30fps() {
        let ts: Vec<i64> = (0..600).map(|i| i * F60).collect();
        let accepted = run_rate_aware_cap(&ts, 30);
        // Warm-up deliberately passes eight frames; the steady-state cadence
        // after that must be every second source frame.
        let tail: Vec<i64> = accepted.into_iter().filter(|t| *t >= 20 * F60).collect();
        for pair in tail.windows(2) {
            assert_eq!(pair[1] - pair[0], 2 * F60);
        }
    }

    #[test]
    fn cadence_estimator_recovers_24p_from_irregular_dwm_steps() {
        let vblank = 166_667i64;
        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        cadence.observe(Some(t));
        for i in 0..48 {
            t += match i % 4 {
                0 => vblank,
                1 => 4 * vblank,
                2 => 2 * vblank,
                _ => 3 * vblank,
            };
            cadence.observe(Some(t));
        }
        let period_ms = cadence.period_s().unwrap() * 1000.0;
        assert!(
            (period_ms - 41.667).abs() < 0.05,
            "estimated period was {period_ms:.3}ms"
        );
    }

    #[test]
    fn cadence_estimator_switches_quickly_between_60p_and_30p() {
        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        cadence.observe(Some(t));
        for _ in 0..40 {
            t += 166_667;
            cadence.observe(Some(t));
        }
        assert!((cadence.period_s().unwrap() - 1.0 / 60.0).abs() < 0.000_01);

        for _ in 0..6 {
            t += 333_333;
            cadence.observe(Some(t));
        }
        assert!((cadence.period_s().unwrap() - 1.0 / 30.0).abs() < 0.000_01);

        for _ in 0..6 {
            t += 166_667;
            cadence.observe(Some(t));
        }
        assert!((cadence.period_s().unwrap() - 1.0 / 60.0).abs() < 0.000_01);
    }

    #[test]
    fn smooth_clock_preserves_exact_30p_and_60p_periods() {
        // Smooth mode is cadence-generic: it must not contain a 24p-only
        // clock. Verify both common drama/video rates are retained exactly,
        // independent of a 60/120Hz presentation surface.
        for (step_100ns, expected_fps) in [(333_333i64, 30.0), (166_667i64, 60.0)] {
            let mut cadence = CadenceEstimator::default();
            let mut t = 0i64;
            cadence.observe(Some(t));
            for _ in 0..120 {
                t += step_100ns;
                cadence.observe(Some(t));
            }
            let period = cadence.period_s().unwrap();
            assert!(
                (1.0 / period - expected_fps).abs() < 0.001,
                "{expected_fps}p was estimated as {:.6}fps",
                1.0 / period
            );
            let mut pacer = SmoothPacer::default();
            let base = Instant::now();
            let (_, first_present) = pacer
                .processing_deadline(base, Some(period), 0.001, Some(60.0))
                .unwrap();
            let (_, second_present) = pacer
                .processing_deadline(base, Some(period), 0.001, Some(120.0))
                .unwrap();
            assert!(
                (second_present.duration_since(first_present).as_secs_f64() - period).abs()
                    < 0.000_001
            );
        }
    }

    #[test]
    fn cadence_estimator_ignores_isolated_missed_60p_slots() {
        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        cadence.observe(Some(t));
        for i in 0..80 {
            t += if i % 17 == 0 { 333_334 } else { 166_667 };
            cadence.observe(Some(t));
        }
        assert!((cadence.period_s().unwrap() - 1.0 / 60.0).abs() < 0.000_01);
    }

    #[test]
    fn dropped_wgc_sequence_does_not_turn_24fps_into_12fps() {
        let period = normalized_frame_span_s(Some(1_000_000), Some(1_833_334), 10, 12)
            .expect("valid timestamp span");
        assert!((period - 1.0 / 24.0).abs() < 0.000_001, "{period}");

        let mut cadence = CadenceEstimator::default();
        let mut t = 0i64;
        for seq in (0..20u64).step_by(2) {
            cadence.observe_frame(Some(t), seq);
            t += 833_334;
        }
        let measured = cadence.period_s().expect("warmed cadence");
        assert!((measured - 1.0 / 24.0).abs() < 0.000_001, "{measured}");
    }

    #[test]
    fn cadence_snap_removes_small_24p_estimation_wobble() {
        let mut snapped = Vec::new();
        for measured_ms in [40.74, 41.18, 41.67, 42.59] {
            let snapped_ms = snap_video_period(measured_ms / 1000.0) * 1000.0;
            assert!((41.66..=41.72).contains(&snapped_ms));
            snapped.push(snapped_ms);
        }
        let spread = snapped.iter().copied().fold(0.0, f64::max)
            - snapped.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(spread < 0.05, "snapped period spread was {spread:.4}ms");
        let nonstandard = snap_video_period(1.0 / 26.0);
        assert!((nonstandard - 1.0 / 26.0).abs() < f64::EPSILON);
    }

    fn synthetic_zoom_signature(scale: f64) -> super::FrameSignature {
        let mut luma = Vec::with_capacity(64 * 36);
        for y in 0..36 {
            for x in 0..64 {
                let sx = (x as f64 - 31.5) * scale;
                let sy = (y as f64 - 17.5) * scale;
                let v = 128.0
                    + 13.0 * (sx * 0.31).sin()
                    + 11.0 * (sy * 0.43).cos()
                    + 7.0 * ((sx + sy) * 0.19).sin();
                luma.push(v.round().clamp(0.0, 255.0) as u8);
            }
        }
        super::FrameSignature {
            size: (640, 480),
            luma,
        }
    }

    #[test]
    fn load_reduction_does_not_merge_slow_global_zoom() {
        let a = synthetic_zoom_signature(1.0);
        let b = synthetic_zoom_signature(1.006);
        assert!(signature_has_coherent_motion(&a, &b));
        assert!(!signatures_match(&a, &b));
    }

    #[test]
    fn load_reduction_still_absorbs_sparse_codec_shimmer() {
        let a = synthetic_zoom_signature(1.0);
        let mut b = a.clone();
        for i in (7..b.luma.len()).step_by(29) {
            b.luma[i] = if (i / 29) % 2 == 0 {
                b.luma[i].saturating_add(1)
            } else {
                b.luma[i].saturating_sub(1)
            };
        }
        assert!(!signature_has_coherent_motion(&a, &b));
        assert!(signatures_match(&a, &b));
    }

    #[test]
    fn load_reduction_never_merges_quantized_fade_step() {
        let a = synthetic_zoom_signature(1.0);
        let mut b = a.clone();
        // A very slow fade may advance only part of the 8-bit samples on a
        // given frame, but every changed sample moves in the same direction.
        for i in (3..b.luma.len()).step_by(8) {
            b.luma[i] = b.luma[i].saturating_sub(1);
        }
        assert!(signature_has_coherent_motion(&a, &b));
        assert!(!signatures_match(&a, &b));
    }

    #[test]
    fn load_reduction_never_merges_subpixel_pan_step() {
        let a = synthetic_zoom_signature(1.0);
        let mut b = a.clone();
        for y in 0..36 {
            for x in 1..63 {
                let i = y * 64 + x;
                // One-quarter-cell horizontal translation, quantised back to
                // the same 8-bit signature representation.
                let v = a.luma[i] as f32 * 0.75 + a.luma[i - 1] as f32 * 0.25;
                b.luma[i] = v.round() as u8;
            }
        }
        assert!(signature_has_coherent_motion(&a, &b));
        assert!(!signatures_match(&a, &b));
    }

    #[test]
    fn load_reduction_detects_three_frame_vertical_scroll_continuity() {
        let previous = synthetic_zoom_signature(1.0);
        let mut current = previous.clone();
        let mut next = previous.clone();
        // Quantised slow upward scroll: only alternating samples advance in
        // each individual frame, while the two-frame direction is continuous.
        for y in 1..35 {
            for x in 0..64 {
                let i = y * 64 + x;
                let above = (y - 1) * 64 + x;
                if (x + y) % 2 == 0 {
                    current.luma[i] =
                        ((previous.luma[i] as u16 * 3 + previous.luma[above] as u16) / 4) as u8;
                }
                next.luma[i] = ((previous.luma[i] as u16 + previous.luma[above] as u16) / 2) as u8;
            }
        }
        assert!(signature_has_temporal_continuity(
            &previous, &current, &next
        ));
    }

    #[test]
    fn load_reduction_does_not_call_random_three_frame_noise_motion() {
        let previous = synthetic_zoom_signature(1.0);
        let mut current = previous.clone();
        let mut next = previous.clone();
        for i in (11..current.luma.len()).step_by(37) {
            current.luma[i] = current.luma[i].saturating_add(1);
            next.luma[i] = next.luma[i].saturating_sub(1);
        }
        assert!(!signature_has_temporal_continuity(
            &previous, &current, &next
        ));
    }

    #[test]
    fn content_duplicates_recover_24p_from_a_30hz_compositor() {
        let mut cadence = CadenceEstimator::default();
        let mut previous = None;
        let mut content_id = 0u8;
        let mut accepted = 0usize;
        for compositor_frame in 0..150i64 {
            // Four changing video pictures followed by one repeated picture:
            // 120 unique pictures over five seconds = 24fps content in a
            // 30Hz Chrome/DWM composition stream.
            if compositor_frame % 5 != 4 {
                content_id = content_id.wrapping_add(17);
            }
            let frame = FrameBuf {
                w: 64,
                h: 36,
                data: vec![content_id; 64 * 36 * 4],
                ..Default::default()
            };
            let signature = frame_signature(&frame).unwrap();
            let duplicate = previous
                .as_ref()
                .is_some_and(|old| signatures_match(old, &signature));
            if !duplicate {
                previous = Some(signature);
                cadence.observe_content(Some(compositor_frame * 333_333));
                accepted += 1;
            }
        }
        assert_eq!(accepted, 120);
        let fps = 1.0 / cadence.period_s().unwrap();
        assert!((fps - 24.0).abs() < 0.05, "content fps={fps}");
        assert_eq!(
            refresh_limited_output_ratio(2, cadence.period_s(), Some(120.0)),
            2.0
        );
        assert_eq!(
            refresh_limited_output_ratio(3, cadence.period_s(), Some(120.0)),
            3.0
        );
    }

    #[test]
    fn similarity_cannot_cascade_below_24fps_on_a_30hz_surface() {
        let mut unique_since_duplicate = 4u8;
        let mut accepted = 0usize;
        for _ in 0..150 {
            let omit = unique_since_duplicate >= 4;
            if omit {
                unique_since_duplicate = 0;
            } else {
                unique_since_duplicate = unique_since_duplicate.saturating_add(1);
                accepted += 1;
            }
        }
        assert_eq!(accepted, 120);
    }

    #[test]
    fn smooth_pacer_accumulates_deadlines_without_drift() {
        let mut pacer = SmoothPacer::default();
        let now = Instant::now();
        let period = 1.0 / 24.0;
        let (a, present_a) = pacer
            .processing_deadline(now, Some(period), 0.0, None)
            .unwrap();
        let (b, present_b) = pacer
            .processing_deadline(now, Some(period), 0.0, None)
            .unwrap();
        let spacing = b.duration_since(a).as_secs_f64();
        assert!((spacing - period).abs() < 0.000_001);
        assert!((present_b.duration_since(present_a).as_secs_f64() - period).abs() < 0.000_001);
        assert!(a < present_a);
    }

    #[test]
    fn smooth_pacer_gap_fill_present_forces_a_full_next_period() {
        let period = 1.0 / 24.0;
        let base = Instant::now();

        // Without the gap-fill re-anchor the saved target is already in the
        // past, so the next real frame is allowed to present immediately.
        let mut stale = SmoothPacer::default();
        let (_, first_present) = stale
            .processing_deadline(base, Some(period), 0.0, Some(120.0))
            .unwrap();
        let gap_fill_complete = first_present + Duration::from_secs_f64(period * 1.55);
        let (_, stale_next_present) = stale
            .processing_deadline(gap_fill_complete, Some(period), 0.0, Some(120.0))
            .unwrap();
        assert_eq!(stale_next_present, gap_fill_complete);

        // The real gap-fill path now resets that obsolete phase. The next
        // frame therefore remains one complete source period away.
        let mut fixed = SmoothPacer::default();
        let _ = fixed
            .processing_deadline(base, Some(period), 0.0, Some(120.0))
            .unwrap();
        fixed.reanchor_after_gap_fill_present(gap_fill_complete, Some(period));
        let (_, fixed_next_present) = fixed
            .processing_deadline(gap_fill_complete, Some(period), 0.0, Some(120.0))
            .unwrap();
        assert!(
            (fixed_next_present
                .duration_since(gap_fill_complete)
                .as_secs_f64()
                - period)
                .abs()
                < 0.000_001
        );
    }

    #[test]
    fn smooth_pacer_preserves_exact_24p_clock_for_dwm_at_any_refresh() {
        let mut pacer = SmoothPacer::default();
        let mut now = Instant::now();
        let mut intervals = Vec::new();
        for _ in 0..8 {
            let (_, present) = pacer
                .processing_deadline(now, Some(1.0 / 24.0), 0.0, Some(60.0))
                .unwrap();
            let interval = present.duration_since(now).as_secs_f64();
            intervals.push(interval);
            now = present;
        }
        assert!(
            intervals
                .iter()
                .all(|interval| (*interval - 1.0 / 24.0).abs() < 0.000_001)
        );
        assert!((intervals.iter().sum::<f64>() - 8.0 / 24.0).abs() < 0.000_001);
    }

    #[test]
    fn smooth_pacer_hands_timing_to_a_blocking_compositor_at_any_refresh_rate() {
        for source_fps in [24.0, 30.0, 50.0, 60.0] {
            let period = 1.0 / source_fps;
            let mut pacer = SmoothPacer::default();
            assert!(
                pacer
                    .processing_deadline(Instant::now(), Some(period), 0.002, None)
                    .is_some()
            );
            assert_eq!(
                pacer.observe_present_block(period * 0.9, Some(period)),
                Some(true)
            );
            assert!(
                pacer
                    .processing_deadline(Instant::now(), Some(period), 0.002, None)
                    .is_none()
            );
        }
    }

    #[test]
    fn smooth_pacer_hands_off_when_effective_output_falls_to_half_rate() {
        for source_fps in [24.0, 30.0, 50.0, 60.0] {
            let period = 1.0 / source_fps;
            let mut pacer = SmoothPacer::default();
            assert_eq!(
                pacer.observe_effective_present_interval(period * 2.0, Some(period)),
                None
            );
            assert_eq!(
                pacer.observe_effective_present_interval(period * 2.0, Some(period)),
                None
            );
            assert_eq!(
                pacer.observe_effective_present_interval(period * 2.0, Some(period)),
                Some(true)
            );
            assert!(
                pacer
                    .processing_deadline(Instant::now(), Some(period), 0.002, None)
                    .is_none()
            );
        }
    }

    #[test]
    fn smooth_pacer_does_not_handoff_on_24p_sixty_hz_transition_intervals() {
        let period = 1.0 / 24.0;
        let mut pacer = SmoothPacer::default();
        for multiplier in [1.6, 1.1, 1.7, 1.6, 1.0, 1.7] {
            assert_eq!(
                pacer.observe_effective_present_interval(period * multiplier, Some(period)),
                None
            );
        }
        assert!(
            pacer
                .processing_deadline(Instant::now(), Some(period), 0.002, Some(60.0))
                .is_some()
        );
    }

    #[test]
    fn smooth_pacer_handoff_persists_until_session_reset() {
        let period = 1.0 / 30.0;
        let mut pacer = SmoothPacer::default();
        assert_eq!(
            pacer.observe_present_block(period, Some(period)),
            Some(true)
        );
        for _ in 0..8 {
            assert_eq!(pacer.observe_present_block(0.001, Some(period)), None);
        }
        assert!(
            pacer
                .processing_deadline(Instant::now(), Some(period), 0.002, None)
                .is_none()
        );
        pacer.reset();
        assert!(
            pacer
                .processing_deadline(Instant::now(), Some(period), 0.002, None)
                .is_some()
        );
    }

    #[test]
    fn cap_tolerance_does_not_break_true_halving() {
        // 60fps -> cap 30 must still drop every second frame (early frames
        // are half an interval before the deadline, beyond the tolerance)
        let ts: Vec<i64> = (0..240).map(|i| i * F60).collect();
        let accepted = run_cap(&ts, 30);
        let span_s = (accepted.last().unwrap() - accepted[0]) as f64 / 1e7;
        let rate = (accepted.len() - 1) as f64 / span_s;
        assert!((rate - 30.0).abs() < 1.0, "rate {rate}");
    }

    #[test]
    fn cap_resyncs_after_a_static_pause_without_bursting() {
        // 1s of frames, 5s pause (static content), then more frames: the
        // deadline must resync, not burst-accept the backlog.
        let mut ts: Vec<i64> = (0..60).map(|i| i * F60).collect();
        ts.extend((0..60).map(|i| 6 * 10_000_000 + i * F60));
        let accepted = run_cap(&ts, 30);
        for w in accepted.windows(2) {
            assert!(w[1] - w[0] >= 2 * F60, "burst after pause: {}", w[1] - w[0]);
        }
    }

    #[test]
    fn sdr_highlight_protection_leaves_normal_range_untouched() {
        let mut frame = FrameBuf {
            w: 2,
            h: 1,
            data: vec![220, 120, 40, 255, SDR_HIGHLIGHT_KNEE_U8, 200, 180, 127],
            hdr: false,
            ..FrameBuf::default()
        };
        let before = frame.data.clone();
        assert!(protect_sdr_highlights_in_place(&mut frame));
        assert_eq!(frame.data, before);
    }

    #[test]
    fn sdr_highlight_protection_softens_only_the_top_end() {
        let mut frame = FrameBuf {
            w: 2,
            h: 1,
            data: vec![255, 255, 255, 255, 240, 240, 240, 64],
            hdr: false,
            ..FrameBuf::default()
        };
        assert!(protect_sdr_highlights_in_place(&mut frame));
        assert_eq!(
            &frame.data[..4],
            &[
                SDR_HIGHLIGHT_CEILING_U8,
                SDR_HIGHLIGHT_CEILING_U8,
                SDR_HIGHLIGHT_CEILING_U8,
                255
            ]
        );
        assert!(frame.data[4] < 240 && frame.data[4] > SDR_HIGHLIGHT_KNEE_U8);
        assert_eq!(frame.data[4], frame.data[5]);
        assert_eq!(frame.data[5], frame.data[6]);
        assert_eq!(frame.data[7], 64);
    }

    #[test]
    fn sdr_highlight_protection_preserves_saturated_colours_and_pale_ratios() {
        let mut frame = FrameBuf {
            w: 3,
            h: 1,
            data: vec![255, 0, 0, 255, 255, 180, 220, 255, 255, 230, 240, 255],
            hdr: false,
            ..FrameBuf::default()
        };
        let saturated_before = frame.data[..8].to_vec();
        assert!(protect_sdr_highlights_in_place(&mut frame));
        assert_eq!(&frame.data[..8], saturated_before.as_slice());
        assert_eq!(frame.data[8], SDR_HIGHLIGHT_CEILING_U8);
        assert!(frame.data[9] < 230 && frame.data[9] > 220);
        assert!(frame.data[10] < 240 && frame.data[10] > frame.data[9]);
    }

    #[test]
    fn hdr_sdr_fixed_low400_curve_preserves_detail_without_hard_clipping() {
        for mode in [HdrSdrMode::Low400, HdrSdrMode::Mid600, HdrSdrMode::High1000] {
            let mid = tonemap_scrgb_to_sdr_linear([0.18; 3], mode);
            let paper = tonemap_scrgb_to_sdr_linear([1.0; 3], mode);
            let highlight = tonemap_scrgb_to_sdr_linear(
                [HDR_ASSUMED_PEAK_NITS / SCRGB_REFERENCE_WHITE_NITS; 3],
                mode,
            );
            let extreme = tonemap_scrgb_to_sdr_linear([25.0; 3], mode);
            assert!(
                mid[0] > 0.0 && mid[0] < paper[0],
                "mode={mode:?} mid={mid:?}"
            );
            assert!(
                paper[0] < highlight[0],
                "mode={mode:?} paper={paper:?} highlight={highlight:?}"
            );
            assert!(highlight[0] < 1.0, "mode={mode:?} highlight={highlight:?}");
            assert!(extreme[0] < 1.0, "mode={mode:?} extreme={extreme:?}");
        }
    }

    #[test]
    fn hdr_sdr_legacy_profile_values_produce_identical_low400_output() {
        let rgb = [2.0, 0.7, 0.15];
        let low = tonemap_scrgb_to_sdr_linear(rgb, HdrSdrMode::Low400);
        let mid = tonemap_scrgb_to_sdr_linear(rgb, HdrSdrMode::Mid600);
        let high = tonemap_scrgb_to_sdr_linear(rgb, HdrSdrMode::High1000);
        assert_eq!(low, mid);
        assert_eq!(mid, high);
    }

    #[test]
    fn hdr_sdr_preserves_same_colour_order_through_interpolated_lut() {
        let a = tonemap_scrgb_to_sdr_linear([0.42, 0.18, 0.07], HdrSdrMode::Low400);
        let b = tonemap_scrgb_to_sdr_linear([0.44, 0.19, 0.074], HdrSdrMode::Low400);
        assert!(
            b[0] > a[0] && b[1] > a[1] && b[2] > a[2],
            "same-colour detail collapsed: a={a:?} b={b:?}"
        );
    }

    #[test]
    fn hdr_sdr_preserves_orange_highlight_colour_and_gradient() {
        let dim = tonemap_scrgb_to_sdr_linear([2.0, 0.5, 0.0625], HdrSdrMode::Low400);
        let bright = tonemap_scrgb_to_sdr_linear([8.0, 2.0, 0.25], HdrSdrMode::Low400);
        for mapped in [dim, bright] {
            assert!(
                mapped[0] > mapped[1] && mapped[1] > mapped[2],
                "mapped={mapped:?}"
            );
            assert!(
                mapped[1] / mapped[0] < 0.30,
                "orange became too neutral: {mapped:?}"
            );
            assert!(mapped[0] < 1.0);
        }
        assert!(
            bright[0] > dim[0],
            "orange gradient collapsed: dim={dim:?} bright={bright:?}"
        );
    }

    #[test]
    fn hdr_sdr_restores_neutral_caption_white_without_hard_clipping() {
        let reference_white = tonemap_scrgb_to_sdr_linear([1.0; 3], HdrSdrMode::Low400);
        let caption_white = tonemap_scrgb_to_sdr_linear([2.5; 3], HdrSdrMode::Low400);
        assert!(
            reference_white[0] >= 0.89 && reference_white[0] < 0.94,
            "reference white={reference_white:?}"
        );
        assert!(
            caption_white[0] > reference_white[0] && caption_white[0] < 0.94,
            "caption white={caption_white:?}"
        );
    }

    #[test]
    fn hdr_sdr_white_recovery_does_not_lift_skin_or_coloured_lights() {
        for rgb in [[2.0_f32, 0.5, 0.0625], [2.5_f32, 1.8, 1.5]] {
            let source_peak = rgb[0].max(rgb[1]).max(rgb[2]);
            let shoulder_only = hdr_shoulder_exact(source_peak);
            let recovered = recover_neutral_reference_white(rgb, source_peak, shoulder_only);
            assert!(
                (recovered - shoulder_only).abs() < 1e-6,
                "coloured highlight was lifted: rgb={rgb:?} shoulder={shoulder_only} recovered={recovered}"
            );
        }
    }

    #[test]
    fn hdr_sdr_neutral_dither_does_not_create_colour_bias() {
        let lut = srgb_encode_lut16();
        for pixel in [0usize, 7, 31, 63] {
            let r = srgb_encode_u8_dithered(0.5, lut, pixel);
            let g = srgb_encode_u8_dithered(0.5, lut, pixel);
            let b = srgb_encode_u8_dithered(0.5, lut, pixel);
            assert_eq!([r, g, b], [r, r, r]);
        }
    }

    #[test]
    fn hdr_frame_is_rgba8_before_downstream_gates() {
        let mut data = Vec::new();
        for channel in [1.0, 0.5, 0.0, 1.0] {
            data.extend_from_slice(&f16::from_f32(channel).to_bits().to_le_bytes());
        }
        let mut frame = FrameBuf {
            w: 1,
            h: 1,
            data,
            hdr: true,
            ..FrameBuf::default()
        };
        assert!(frame_signature(&frame).is_some());
        let mut output = Vec::new();
        assert!(tonemap_hdr_frame_to_sdr(&mut frame, &mut output, HdrSdrMode::Low400,).is_some());
        assert!(!frame.hdr);
        assert_eq!(frame.data.len(), 4);
        assert!(frame.data[0] > frame.data[1]);
        assert_eq!(frame.data[2], 0);
        assert_eq!(frame.data[3], 255);
        assert!(frame_signature(&frame).is_some());
    }

    #[test]
    fn virtual_four_x_is_only_the_shader_reference() {
        assert_eq!(virtual_shader_output_size((640, 480)), (2560, 1920));
        // The displayed output remains independently fitted to the monitor.
        assert_eq!(fit_aspect_inside((640, 480), (2560, 1440)), (1920, 1440));
    }

    #[test]
    fn fixed_overlay_fit_preserves_aspect_when_clamped_to_monitor() {
        let (w, h) = fit_fixed_overlay_size(1280, 720, 2.0, 1920, 1080);
        assert_eq!((w, h), (1920, 1080));
    }

    #[test]
    fn fixed_overlay_fit_does_not_clamp_width_and_height_independently() {
        let (w, h) = fit_fixed_overlay_size(1200, 500, 2.0, 1920, 1080);
        assert_eq!(w, 1920);
        assert_eq!(h, 800);
    }

    #[test]
    fn fixed_overlay_aspect_follows_committed_frame_not_early_client_resize() {
        assert_eq!(
            fixed_overlay_aspect_basis((960, 720), (1280, 720)),
            (960, 720)
        );
        assert_eq!(
            fixed_overlay_aspect_basis((1280, 720), (960, 720)),
            (1280, 720)
        );
        assert_eq!(fixed_overlay_aspect_basis((0, 0), (854, 480)), (854, 480));
    }

    #[test]
    fn initial_wgc_frame_replaces_inexact_win32_client_estimate() {
        assert_eq!(initial_display_aspect(None, (1044, 682)), (1044, 682));
    }

    #[test]
    fn requested_capture_canvas_remains_the_display_basis() {
        assert_eq!(
            initial_display_aspect(Some((640, 480)), (640, 480)),
            (640, 480)
        );
    }

    #[test]
    fn capture_client_size_matches_wgc_even_frame_contract() {
        assert_eq!(normalize_capture_client_size((640, 359)), (640, 360));
        assert_eq!(normalize_capture_client_size((959, 719)), (960, 720));
        assert_eq!(normalize_capture_client_size((640, 360)), (640, 360));
    }

    #[test]
    fn display_aspect_stays_four_three_inside_widescreen_bounds() {
        assert_eq!(fit_aspect_inside((640, 480), (1920, 1080)), (1440, 1080));
    }

    #[test]
    fn fullscreen_panel_starts_top_center_of_visible_content() {
        assert_eq!(
            panel_target_position(
                ScaleMode::Auto,
                (0, 0, 1920, 1080),
                (160, 0, 1600, 1080),
                (236, 30),
            ),
            (842, 4)
        );
    }

    #[test]
    fn windowed_panel_prefers_above_and_falls_inside_near_desktop_top() {
        assert_eq!(
            panel_target_position(
                ScaleMode::Fixed,
                (300, 200, 800, 450),
                (300, 200, 800, 450),
                (236, 30),
            ),
            (300, 170)
        );
        assert_eq!(
            panel_target_position(
                ScaleMode::Fixed,
                (300, 10, 800, 450),
                (300, 10, 800, 450),
                (236, 30),
            ),
            (300, 10)
        );
    }

    #[test]
    fn neodeint_comb_detector_separates_clean_and_woven_rows() {
        let make = |woven: bool| {
            let (w, h) = (128usize, 96usize);
            let mut data = vec![0u8; w * h * 4];
            for y in 0..h {
                let v = if woven && y % 2 != 0 { 220 } else { 32 };
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    data[i..i + 4].copy_from_slice(&[v, v, v, 255]);
                }
            }
            FrameBuf {
                w: w as i32,
                h: h as i32,
                data,
                ..FrameBuf::default()
            }
        };
        assert!(frame_comb_fraction(&make(false)) < 0.001);
        assert!(frame_comb_fraction(&make(true)) > 0.90);
    }

    #[test]
    fn neodeint_latches_an_alternating_woven_cadence_and_releases_on_clean_run() {
        let mut history = 0u8;
        let mut hold = 0u8;
        let bypass: Vec<bool> = [true, false, true, false, true]
            .into_iter()
            .map(|combed| update_neodeint_scene_latch(&mut history, &mut hold, combed))
            .collect();
        assert_eq!(bypass, vec![false, true, false, false, false]);
        for _ in 0..4 {
            update_neodeint_scene_latch(&mut history, &mut hold, false);
        }
        assert!(update_neodeint_scene_latch(&mut history, &mut hold, false));
    }
}
