//! Seamless cursor mapping: operate the (hidden) source window through the
//! magnified overlay.
//!
//! Cursor-routing invariants:
//! - The overlay remains click-through while the real cursor is confined to the
//!   source client rect and a mapped sprite cursor is drawn over the output.
//! - Hardware movement entering the visible content engages source control.
//! - Pushing against a source edge disengages and returns the cursor to the
//!   matching content edge.
//! - Injected low-level mouse events are ignored to prevent feedback loops from
//!   internal SetCursorPos calls.
//! - The floating panel and GUI can receive redirected input without moving the
//!   confined real cursor away from the source.
//! - Every stop path releases cursor confinement.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicIsize, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use windows::Win32::Foundation::{
    CloseHandle, GetLastError, HANDLE, HWND, LPARAM, LRESULT, POINT, RECT, WAIT_OBJECT_0, WPARAM,
};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::Threading::{
    CreateEventW, INFINITE, OpenProcess, PROCESS_SYNCHRONIZE, SetEvent,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_MOUSE, MOD_ALT, MOD_CONTROL, MOD_NOREPEAT, MOD_SHIFT,
    MOUSE_EVENT_FLAGS, MOUSEEVENTF_LEFTDOWN, MOUSEEVENTF_LEFTUP, MOUSEEVENTF_MIDDLEDOWN,
    MOUSEEVENTF_MIDDLEUP, MOUSEEVENTF_RIGHTDOWN, MOUSEEVENTF_RIGHTUP, RegisterHotKey, SendInput,
    UnregisterHotKey, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON, mouse_event,
};
use windows::Win32::UI::Magnification::{MagInitialize, MagShowSystemCursor, MagUninitialize};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::PCWSTR;

const ENTER_MARGIN_PX: i32 = 8;
// Native UI routing is exact. Invisible click margins and post-hover grace
// can make the GUI/panel keep ownership after the visible pointer has
// already crossed the boundary.
const UI_CLICK_MARGIN_PX: i32 = 0;
const EXIT_PLACE_MARGIN_PX: i32 = 6;
const EXIT_WINDOW_MARGIN_PX: i32 = 6;
/// After a windowed edge exit, refuse to re-engage until the cursor either
/// clearly leaves the content region or deliberately returns far enough inside
/// the source. This hysteresis prevents a small post-exit drift from immediately
/// re-engaging and moving the cursor back into the magnified view.
const POST_ESCAPE_SRC_MARGIN_PX: i32 = ENTER_MARGIN_PX;
/// Hysteresis dead-band in source space. Re-engagement requires the mapped
/// source position to move clearly inside the source while edge escape still
/// fires at the boundary, creating a stable gap between the two transitions.
const ENGAGE_SRC_MARGIN_PX: i32 = 4;
const REENTER_COOLDOWN_MS: u64 = 200;
/// Accumulated outward travel required before a windowed edge releases the
/// cursor. A brief touch remains clipped at the edge; sustained outward motion
/// crosses the threshold. Fullscreen mode never exits through this path.
const EXIT_TRAVEL_PX: f64 = 1.0;
pub const PANEL_ACTION_STOP: u32 = 1 << 0;
pub const PANEL_ACTION_COLLAPSE: u32 = 1 << 1;
pub const PANEL_ACTION_EXPAND: u32 = 1 << 2;
pub const PANEL_ACTION_SCREENSHOT: u32 = 1 << 3;
pub const PANEL_ACTION_GUI_TOPMOST: u32 = 1 << 4;
static PANEL_ACTIONS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static PANEL_GUI_ACTION_SEQ: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Lock-free snapshot of the currently visible floating panel. The LL mouse
/// hook must never lose a panel click merely because the main cursor state
/// mutex is momentarily busy on another thread.
static ACTIVE_PANEL_HWND: AtomicIsize = AtomicIsize::new(0);
/// Matching button-up for a lock-free direct panel DOWN is swallowed here,
/// independently of the main State mutex.
static PANEL_DIRECT_SWALLOW_UP: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// v348u: lock-free main-window Start/Stop routing. Both directions use the
/// same physical pointer-DOWN edge so the control has symmetric latency and
/// cannot lose Start to egui focus/ownership settling after a fast Stop.
pub const MAIN_ACTION_STOP: u32 = 1 << 0;
pub const MAIN_ACTION_START: u32 = 1 << 1;
pub const MAIN_CONTROL_DISABLED: u32 = 0;
pub const MAIN_CONTROL_START: u32 = 1;
pub const MAIN_CONTROL_STOP: u32 = 2;
static MAIN_ACTIONS: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
static GUI_MINIMIZE_CURSOR: AtomicI64 = AtomicI64::new(0);
static GUI_MINIMIZE_CURSOR_AT_MS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_MAIN_GUI_HWND: AtomicIsize = AtomicIsize::new(0);
static MAIN_CONTROL_MODE: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(MAIN_CONTROL_DISABLED);
static MAIN_DIRECT_ACTIVE: AtomicBool = AtomicBool::new(false);
static MAIN_CONTROL_LEFT: AtomicI32 = AtomicI32::new(0);
static MAIN_CONTROL_TOP: AtomicI32 = AtomicI32::new(0);
static MAIN_CONTROL_RIGHT: AtomicI32 = AtomicI32::new(0);
static MAIN_CONTROL_BOTTOM: AtomicI32 = AtomicI32::new(0);
/// Engagement is committed synchronously inside the hook. A hardware move that
/// was queued before the cursor warp can still arrive with stale screen-space
/// coordinates, so edge-exit detection is briefly suppressed after engagement.
const ENGAGE_GRACE_MS: u64 = 60;
/// Post-engage settle: an event is FRESH when its hook pt agrees with the
/// OS-clipped cursor (GetCursorPos) within this radius — after the engage
/// teleport the real cursor sits at the target, so stale pre-teleport events
/// (screen coords from the queue backlog) diverge by hundreds of px while
/// genuine moves diverge by at most one event's delta. We cannot wait for the
/// injected SetCursorPos event itself: the hook ignores LLMHF_INJECTED, so it
/// never arrives; waiting for it would keep the swallow active until timeout.
const TELEPORT_SETTLE_PX: i32 = 64;
/// Hard cap on the settle swallow. Under heavy GPU load (4K FSRCNNX ~190ms
/// frames) the stale backlog outlived the 60ms engage grace; a fresh event
/// always ends the swallow early, so this only bounds pathological cases.
const TELEPORT_SETTLE_TIMEOUT_MS: u64 = 250;
// Even after the first post-warp event agrees with GetCursorPos, older
// screen-space hook moves may still drain behind it under a saturated GPU.
// Keep a short commit-relative firewall so those late events can never be
// interpreted as source-space and fling the virtual cursor to a screen corner.
const POST_COMMIT_STALE_GUARD_MS: u64 = 250;
// v378: capture-wide sprite ownership means edge exit no longer waits for a native-cursor reveal.
// Keep only one mouse-sample-sized defer so the triggering WH_MOUSE_LL event has returned before
// SetCursorPos moves the hidden native cursor into desktop coordinates.
const EDGE_TRANSFER_DEFER_MS: u64 = 8;
/// A cursor reveal must never wait forever. A bad/off-desktop target used to
/// leave Magnification hiding the native cursor until capture was stopped.
const CURSOR_REVEAL_MAX_ATTEMPTS: u8 = 8;
/// Capture-wide sprite mode must not depend on the render/ONNX thread to
/// complete an edge handoff. A cold TensorRT shape can stall that thread for
/// seconds, while the mouse hook remains responsive. Use a one-shot timer on
/// the hook-owned sprite window after the triggering WH_MOUSE_LL callback has
/// returned, then move the hidden real cursor from source space to desktop
/// space there.
const CURSOR_EDGE_TRANSFER_TIMER_ID: usize = 2;
static DRAG_GEOMETRY_DIAG_NEXT_MS: AtomicU64 = AtomicU64::new(0);
const POST_COMMIT_RAW_DIVERGENCE_PX: i32 = 96;
/// After an edge handoff the LL-hook queue can still deliver one or more
/// pre-warp source/content-space points even though GetCursorPos already sits
/// at the verified desktop target.  Quarantine only those large-divergence
/// events for a short bounded window; normal desktop motion takes over as soon
/// as the two coordinate streams converge again.
const POST_EDGE_RAW_GUARD_MS: u64 = 250;
const POST_EDGE_RAW_DIVERGENCE_PX: i32 = 96;
const POST_EDGE_RAW_CATCHUP_PX: i32 = 24;

fn post_edge_raw_is_stale(dx: i32, dy: i32, domain_disagrees: bool) -> bool {
    dx > POST_EDGE_RAW_DIVERGENCE_PX
        || dy > POST_EDGE_RAW_DIVERGENCE_PX
        || (domain_disagrees && (dx > POST_EDGE_RAW_CATCHUP_PX || dy > POST_EDGE_RAW_CATCHUP_PX))
}

const OSC_SHORT_MS: u64 = 250;
const OSC_MAX: u32 = 6;
/// Kept public for source compatibility with older diagnostic binaries.
/// The panel hit/no-engage surface is exact: the visible panel
/// rectangle is the only panel surface that may block or own the cursor.
pub const PANEL_NO_ENGAGE_HALO_PX: i32 = 0;
const BTN_LEFT: u8 = 0x01;
const BTN_RIGHT: u8 = 0x02;
const BTN_MIDDLE: u8 = 0x04;

/// HWND of the source that owns the current client-surface LEFT-button
/// gesture.  This is deliberately stricter than GetAsyncKeyState(VK_LBUTTON):
/// a physical hold may have started on Neo's GUI/desktop, or source ownership
/// may already have been released after a failed/off-desktop handoff.  The
/// engine may mirror native source movement into the visible overlay ONLY while
/// this token still names the active source.
static SOURCE_CLIENT_DRAG_OWNER_HWND: AtomicIsize = AtomicIsize::new(0);
/// Raw WH_MOUSE_LL screen coordinate at the exact LEFT-down that armed the
/// client-only source gesture, plus the latest raw move from that same owner.
/// Unlike GetCursorPos these raw coordinates keep travelling past ClipCursor
/// and even past the physical monitor edge, so they remain a usable visual
/// drag authority after Windows/Chromium stops moving the hidden source HWND.
static SOURCE_CLIENT_DRAG_RAW_ORIGIN: AtomicI64 = AtomicI64::new(0);

static SOURCE_CLIENT_DRAG_RAW_CURRENT: AtomicI64 = AtomicI64::new(0);

// v643: safe deferred routing for the one case where a physical source-space
// button edge must not reach another top-level window that covers the hidden
// source coordinate.
//
// IMPORTANT SAFETY CONTRACT:
// - WH_MOUSE_LL does ONLY atomic reads/writes here.
// - No WindowFromPoint, PostMessage, SetCursorPos, cursor visibility, Z-order,
//   logging, or State mutex work is performed by this guard.
// - Win32 delivery to the source is drained later from configure(), on the
//   normal engine/Magnification owner thread.
//
// v642 violated this contract by doing window queries/message delivery/sprite
// manipulation directly inside WH_MOUSE_LL and could destabilize DWM.  Keep
// this path deliberately small and bounded.
const DEFERRED_SOURCE_EVENT_CAP: usize = 16;
const DEFERRED_OCCLUSION_CAP: usize = 8;
static DEFERRED_SOURCE_ENGAGED: AtomicBool = AtomicBool::new(false);
static DEFERRED_SOURCE_CLIENT_ONLY: AtomicBool = AtomicBool::new(false);
static DEFERRED_SOURCE_HWND: AtomicIsize = AtomicIsize::new(0);
// Seqlock-published geometry used by the atomic-only LL-hook gate.  The hook
// maps the visible Neo sprite to its intended source-screen target without
// touching State or calling Win32.
static DEFERRED_GEOMETRY_SEQ: AtomicU64 = AtomicU64::new(0);
static DEFERRED_CONTENT_XY: AtomicI64 = AtomicI64::new(0);
static DEFERRED_CONTENT_WH: AtomicI64 = AtomicI64::new(0);
static DEFERRED_SOURCE_XY: AtomicI64 = AtomicI64::new(0);
static DEFERRED_SOURCE_WH: AtomicI64 = AtomicI64::new(0);
static DEFERRED_SOURCE_LAST_TARGET: AtomicI64 = AtomicI64::new(0);
static DEFERRED_OCCLUSION_COUNT: std::sync::atomic::AtomicU32 =
    std::sync::atomic::AtomicU32::new(0);
static DEFERRED_OCCLUSION_X: [AtomicI32; DEFERRED_OCCLUSION_CAP] =
    [const { AtomicI32::new(0) }; DEFERRED_OCCLUSION_CAP];
static DEFERRED_OCCLUSION_Y: [AtomicI32; DEFERRED_OCCLUSION_CAP] =
    [const { AtomicI32::new(0) }; DEFERRED_OCCLUSION_CAP];
static DEFERRED_OCCLUSION_W: [AtomicI32; DEFERRED_OCCLUSION_CAP] =
    [const { AtomicI32::new(0) }; DEFERRED_OCCLUSION_CAP];
static DEFERRED_OCCLUSION_H: [AtomicI32; DEFERRED_OCCLUSION_CAP] =
    [const { AtomicI32::new(0) }; DEFERRED_OCCLUSION_CAP];

static DEFERRED_EVENT_WRITE: AtomicU64 = AtomicU64::new(0);
static DEFERRED_EVENT_READ: AtomicU64 = AtomicU64::new(0);
static DEFERRED_EVENT_MSG: [std::sync::atomic::AtomicU32; DEFERRED_SOURCE_EVENT_CAP] =
    [const { std::sync::atomic::AtomicU32::new(0) }; DEFERRED_SOURCE_EVENT_CAP];
static DEFERRED_EVENT_POS: [AtomicI64; DEFERRED_SOURCE_EVENT_CAP] =
    [const { AtomicI64::new(0) }; DEFERRED_SOURCE_EVENT_CAP];
static DEFERRED_EVENT_VISUAL: [AtomicI64; DEFERRED_SOURCE_EVENT_CAP] =
    [const { AtomicI64::new(0) }; DEFERRED_SOURCE_EVENT_CAP];
static DEFERRED_SOURCE_HELD_BITS: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
static DEFERRED_SOURCE_OVERFLOW: AtomicBool = AtomicBool::new(false);
static DEFERRED_SOURCE_MOVE_POS: AtomicI64 = AtomicI64::new(0);
static DEFERRED_SOURCE_MOVE_SEQ: AtomicU64 = AtomicU64::new(0);
static DEFERRED_SOURCE_MOVE_DRAINED_SEQ: AtomicU64 = AtomicU64::new(0);
static DEFERRED_SOURCE_QUEUED_COUNT: AtomicU64 = AtomicU64::new(0);

/// Lock-free provenance check used by the render/geometry thread.  A client
/// drag is valid only when its LEFT-down began while Neo actually owned source
/// input; merely observing the physical button held is not sufficient.
pub fn source_client_drag_owns(hwnd: isize) -> bool {
    hwnd != 0 && SOURCE_CLIENT_DRAG_OWNER_HWND.load(Ordering::Acquire) == hwnd
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

impl Rect {
    pub fn contains(&self, px: i32, py: i32) -> bool {
        px >= self.x && px < self.x + self.w && py >= self.y && py < self.y + self.h
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NoEngageRect {
    pub rect: Rect,
    pub land: Rect,
    /// Top-level window this zone belongs to (control panel / GUI). When set,
    /// clicks over this zone are handed to that window instead of leaking
    /// through to the source.
    pub hwnd: isize,
    /// Explicit role flag. Older builds inferred "panel" from rect != land,
    /// which coupled panel identity to an invisible 24px halo and made that
    /// halo leak into ownership/clamping logic.
    pub panel: bool,
}

impl NoEngageRect {
    pub fn new(rect: Rect) -> Self {
        Self {
            rect,
            land: rect,
            hwnd: 0,
            panel: false,
        }
    }

    pub fn with_land(rect: Rect, land: Rect) -> Self {
        Self {
            rect,
            land,
            hwnd: 0,
            panel: false,
        }
    }

    pub fn with_hwnd(mut self, hwnd: isize) -> Self {
        self.hwnd = hwnd;
        self
    }

    pub fn as_panel(mut self) -> Self {
        self.panel = true;
        self
    }

    pub fn contains(&self, px: i32, py: i32) -> bool {
        self.rect.contains(px, py)
    }

    pub fn landing_point(&self, px: i32, py: i32) -> (i32, i32) {
        clamp_point_inside(self.land, px, py)
    }
}

fn clamp_point_inside(rect: Rect, px: i32, py: i32) -> (i32, i32) {
    if rect.w <= 0 || rect.h <= 0 {
        return (rect.x, rect.y);
    }
    (
        px.clamp(rect.x, rect.x + rect.w - 1),
        py.clamp(rect.y, rect.y + rect.h - 1),
    )
}

pub fn panel_no_engage_rect(panel: Rect) -> NoEngageRect {
    // Exact visible panel only. No invisible halo is allowed to block source
    // ownership or clamp virtual motion; panel identity is explicit now.
    NoEngageRect::new(panel).as_panel()
}

fn is_panel_hit(hit: NoEngageRect) -> bool {
    hit.panel
}

fn ui_landing_point(hit: NoEngageRect, px: i32, py: i32) -> (i32, i32) {
    // Ownership is exact: never magnetize a visible cursor event to an inset or
    // edge of GUI/panel geometry. The click/hover point is the point the user
    // actually sees. HWND identity, not cached geometry, established ownership.
    let _ = hit;
    (px, py)
}

/// Resolve a tracked role from the HWND Win32 already chose. Vector order is
/// deliberately irrelevant; this helper exists so overlap priority can be
/// stress-tested without any native calls.
fn tracked_ui_for_top_hwnd(no_engage: &[NoEngageRect], top_hwnd: isize) -> Option<NoEngageRect> {
    if top_hwnd == 0 {
        return None;
    }
    no_engage.iter().copied().find(|r| r.hwnd == top_hwnd)
}

/// Authoritative cursor ownership hit-test.
///
/// The single source of truth is Win32's top-level window at the VISIBLE screen
/// point. Geometry/list order is never allowed to decide between overlapping
/// GUI/panel/native windows. This matters because the main GUI, floating panel
/// and transparent overlay are all topmost siblings and their rectangles can
/// overlap while the GUI is moved or the panel expands/collapses.
///
/// IMPORTANT: this function is read-only. It must not raise/reorder a window;
/// changing Z-order while deciding Z-order makes ownership self-modifying and
/// was one of the sources of enter/exit oscillation.
fn native_hit_is_covered_by_active_overlay(hwnd: isize, x: i32, y: i32) -> bool {
    if hwnd == 0 {
        return false;
    }
    let overlay = ACTIVE_OVERLAY_HWND.load(Ordering::Acquire);
    if overlay == 0
        || overlay == hwnd
        || !crate::platform::win32::is_window_valid(overlay)
        || !crate::platform::win32::is_window_visible(overlay)
        || !crate::platform::win32::point_is_inside_window(overlay, x, y)
    {
        return false;
    }
    !crate::platform::win32::window_is_above(hwnd, overlay)
}

fn hit_ui_for_cursor_ownership(no_engage: &[NoEngageRect], x: i32, y: i32) -> Option<NoEngageRect> {
    // Unit/synthetic geometry has no live HWND at all. Resolve that case before
    // WindowFromPoint so desktop/test-runner windows cannot accidentally become
    // the authority for fake coordinates. Production Neo zones have live HWNDs.
    let has_live_tracked_hwnd = no_engage
        .iter()
        .any(|r| r.hwnd != 0 && crate::platform::win32::is_window_valid(r.hwnd));
    if !has_live_tracked_hwnd {
        return no_engage.iter().copied().find(|r| r.land.contains(x, y));
    }

    // In production, Win32 Z-order at the visible point is the ONLY authority.
    // Rectangle order, previous owner, hover history and panel role must never
    // overrule the actual top-level window under this point.
    let top = crate::platform::win32::direct_top_level_window_at_point(x, y);
    if top == 0 {
        return None;
    }
    // The magnified overlay is intentionally WS_EX_TRANSPARENT, so the Win32
    // hit test can report a GUI/source/external window that is physically
    // underneath the visible magnified image.  Never let such a covered HWND
    // become native cursor owner.  A GUI that is genuinely raised above the
    // overlay still passes this test and remains fully interactive.
    if native_hit_is_covered_by_active_overlay(top, x, y) {
        if COVERED_NATIVE_HIT_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed) % 128 == 0 {
            log::debug!(
                "covered-native-hit ignored: hwnd={top:#x} overlay={:#x} point=({x},{y}) reason=overlay-visual-priority",
                ACTIVE_OVERLAY_HWND.load(Ordering::Acquire)
            );
        }
        return None;
    }
    let tracked = tracked_ui_for_top_hwnd(no_engage, top);

    // A GUI/panel HWND can briefly outrun the latest coalesced geometry sample
    // during resize/mode/language changes. If Win32 says the top interactive
    // window is one of Neo's own real windows, keep native ownership instead of
    // dropping to source ownership for one event. Transparent overlay/sprite
    // helpers are filtered by direct_top_level_window_at_point().
    if tracked.is_none() && !crate::platform::win32::is_own_window(top) {
        return None;
    }

    let fallback_land = tracked.map(|r| r.land).unwrap_or(Rect { x, y, w: 1, h: 1 });
    let live = crate::platform::win32::window_rect(top)
        .map(|(rx, ry, rw, rh)| Rect {
            x: rx,
            y: ry,
            w: rw,
            h: rh,
        })
        .filter(|r| r.w > 0 && r.h > 0)
        .unwrap_or(fallback_land);
    Some(match tracked {
        Some(r) if r.panel => panel_no_engage_rect(live).with_hwnd(top),
        _ => NoEngageRect::new(live).with_hwnd(top),
    })
}

/// While source ownership is active, the real cursor physically lives in the
/// source rectangle. If the topmost Neo GUI overlaps that rectangle, Windows
/// would otherwise hover/click the GUI at the hidden source point even though
/// the visible virtual cursor is elsewhere. Keep Neo's main GUI click-through
/// during source ownership and use its live geometry for the visible pointer.
fn own_main_gui_at_visible_point(
    no_engage: &[NoEngageRect],
    x: i32,
    y: i32,
) -> Option<NoEngageRect> {
    no_engage.iter().copied().find_map(|zone| {
        if zone.panel
            || zone.hwnd == 0
            || !crate::platform::win32::is_own_window(zone.hwnd)
            || !crate::platform::win32::is_window_valid(zone.hwnd)
            || crate::platform::win32::is_minimized(zone.hwnd)
        {
            return None;
        }
        let live = crate::platform::win32::window_rect(zone.hwnd)
            .map(|(rx, ry, rw, rh)| Rect {
                x: rx,
                y: ry,
                w: rw,
                h: rh,
            })
            .filter(|r| r.w > 0 && r.h > 0)
            .unwrap_or(zone.land);
        live.contains(x, y)
            .then(|| NoEngageRect::new(live).with_hwnd(zone.hwnd))
    })
}

fn own_main_gui_at_visible_point_for_ownership(
    no_engage: &[NoEngageRect],
    x: i32,
    y: i32,
) -> Option<NoEngageRect> {
    own_main_gui_at_visible_point(no_engage, x, y)
        .filter(|gui| !native_hit_is_covered_by_active_overlay(gui.hwnd, x, y))
}

fn hit_ui_for_visible_cursor(g: &State, x: i32, y: i32) -> Option<NoEngageRect> {
    if g.engaged {
        if let Some(gui) = own_main_gui_at_visible_point_for_ownership(&g.no_engage, x, y) {
            return Some(gui);
        }
    }
    hit_ui_for_cursor_ownership(&g.no_engage, x, y)
}

fn route_clock_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_millis() as u64
}

pub fn record_gui_minimize_cursor() {
    let pos = read_cursor_pos_or((i32::MIN, i32::MIN), "gui native minimize");
    if pos.0 == i32::MIN || pos.1 == i32::MIN {
        return;
    }
    GUI_MINIMIZE_CURSOR.store(pack_drag_pair(pos.0, pos.1), Ordering::Release);
    GUI_MINIMIZE_CURSOR_AT_MS.store(route_clock_ms(), Ordering::Release);
    log::info!(
        "gui-native-minimize-cursor-saved: pos=({},{})",
        pos.0,
        pos.1
    );
}

/// Fail-visible handoff for native root-GUI minimization during capture.
///
/// The real system cursor is intentionally Magnification-hidden while the
/// capture sprite owns presentation. Minimizing the root GUI removes its native
/// UI ownership immediately; waiting for the next mouse move to rebuild source
/// ownership creates a cursor-less gap. Preserve the visible desktop coordinate
/// and hand presentation back to the capture sprite synchronously. The explicit
/// idle auto-hide option remains authoritative: when it intentionally hid the
/// sprite, minimization does not override that user setting.
pub fn handle_main_gui_minimize_begin() {
    record_gui_minimize_cursor();
    end_native_gui_caption_drag("native-minimize");
    NATIVE_GUI_OWNER.store(0, Ordering::Release);

    if !capture_sprite_active() {
        return;
    }

    let saved = unpack_drag_pair(GUI_MINIMIZE_CURSOR.load(Ordering::Acquire));
    if saved.0 == i32::MIN || saved.1 == i32::MIN {
        return;
    }

    let mut idle_hidden = false;
    if let Ok(mut g) = state().try_lock() {
        idle_hidden = g.hidden_by_idle;
        g.native_ui_hold_bits = 0;
        g.ui_hold_bits = 0;
        g.last_ui_hover = None;
        g.ui_hover_active = false;
        g.pending_engage = None;
        g.virt = (saved.0 as f64, saved.1 as f64);
        // A title-bar click is current user activity. Do not manufacture an
        // idle timeout, but preserve an already-committed idle-hide state.
        if !idle_hidden {
            g.last_move = Some(std::time::Instant::now());
        }
    }

    // During active capture the native cursor stays hidden by design. Keep a
    // visible Neo sprite at the exact pre-minimize point until normal source or
    // panel ownership takes over. If idle auto-hide intentionally hid it, keep
    // the sprite hidden and only maintain the native-hide contract.
    request_cursor_hidden(true);
    if idle_hidden {
        sprite_hide();
        log::info!(
            "gui-minimize-cursor-handoff: pos=({},{}) mode=idle-autohide-preserved",
            saved.0,
            saved.1
        );
    } else {
        sprite_move_now(saved.0, saved.1, true);
        keep_cursor_sprite_on_top();
        log::info!(
            "gui-minimize-cursor-handoff: pos=({},{}) mode=capture-sprite-fail-visible",
            saved.0,
            saved.1
        );
    }
}

fn publish_main_gui_passthrough(hwnd: isize, passthrough: bool, reason: &str) {
    let previous = MAIN_GUI_MOUSE_PASSTHROUGH.swap(passthrough, Ordering::AcqRel);
    let previous_hwnd = MAIN_GUI_HWND.swap(hwnd, Ordering::AcqRel);
    let changed = previous != passthrough || previous_hwnd != hwnd;
    let native_mismatch = crate::platform::win32::window_input_passthrough(hwnd) != passthrough;
    if changed || native_mismatch {
        MAIN_GUI_ROUTE_REPAIR_UNTIL_MS.store(route_clock_ms() + 250, Ordering::Release);
        crate::platform::win32::set_window_input_passthrough(hwnd, passthrough);
        log::info!(
            "gui-input-route: hwnd={hwnd:#x} passthrough={passthrough} reason={reason} changed={changed} native_mismatch={native_mismatch}"
        );
    }
}

fn set_own_main_gui_passthrough(g: &State, passthrough: bool, reason: &str) {
    for zone in &g.no_engage {
        if !zone.panel
            && zone.hwnd != 0
            && crate::platform::win32::is_own_window(zone.hwnd)
            && crate::platform::win32::is_window_valid(zone.hwnd)
        {
            publish_main_gui_passthrough(zone.hwnd, passthrough, reason);
        }
    }
}

/// Source ownership only needs the main GUI to be click-through when the HIDDEN
/// native cursor's mapped source point physically falls underneath that GUI.
/// v377 toggled the whole GUI on every overlay enter/exit, forcing
/// SWP_FRAMECHANGED across the LL-hook path (~50-65ms in user logs) even when
/// the source cursor was nowhere near the GUI.  Keep normal GUI hit testing
/// intact and shield lazily only for the rare physical overlap case.
fn shield_main_gui_for_source_target(
    g: &State,
    source_x: i32,
    source_y: i32,
    visible_x: i32,
    visible_y: i32,
) {
    let Some(gui) = own_main_gui_at_visible_point(&g.no_engage, source_x, source_y) else {
        return;
    };
    if own_main_gui_at_visible_point(&g.no_engage, visible_x, visible_y)
        .is_some_and(|visible_gui| visible_gui.hwnd == gui.hwnd)
    {
        return;
    }
    if !crate::platform::win32::window_input_passthrough(gui.hwnd) {
        publish_main_gui_passthrough(gui.hwnd, true, "source-target-overlap-shield");
        log::debug!(
            "source-target GUI shield: hwnd={:#x} source=({source_x},{source_y}) visible=({visible_x},{visible_y})",
            gui.hwnd
        );
    }
}

fn repair_main_gui_passthrough_if_needed() {
    let hwnd = MAIN_GUI_HWND.load(Ordering::Acquire);
    if hwnd == 0 || !crate::platform::win32::is_window_valid(hwnd) {
        return;
    }
    let wanted = MAIN_GUI_MOUSE_PASSTHROUGH.load(Ordering::Acquire);
    // Passthrough=true is a short-lived source-ownership state, so its race
    // repair remains bounded. Interactive=false is the safe/idle state: never
    // time-limit that repair. If a stale queued viewport command makes the GUI
    // click-through after Stop, the next physical move OR button edge must be
    // able to recover it even several seconds later.
    if wanted && route_clock_ms() > MAIN_GUI_ROUTE_REPAIR_UNTIL_MS.load(Ordering::Acquire) {
        return;
    }
    if crate::platform::win32::window_input_passthrough(hwnd) != wanted {
        crate::platform::win32::set_window_input_passthrough(hwnd, wanted);
        log::warn!("gui-input-route race repaired: hwnd={hwnd:#x} passthrough={wanted}");
    }
}

fn contains_for_engage(rect: Rect, px: i32, py: i32) -> bool {
    if rect.w <= ENTER_MARGIN_PX * 2 + 1 || rect.h <= ENTER_MARGIN_PX * 2 + 1 {
        return rect.contains(px, py);
    }
    px >= rect.x + ENTER_MARGIN_PX
        && px < rect.x + rect.w - ENTER_MARGIN_PX
        && py >= rect.y + ENTER_MARGIN_PX
        && py < rect.y + rect.h - ENTER_MARGIN_PX
}

#[cfg(test)]
fn exit_point_for_virtual(content: Rect, vx: f64, vy: f64) -> (i32, i32) {
    let left = content.x;
    let right = content.x + content.w - 1;
    let top = content.y;
    let bottom = content.y + content.h - 1;
    let x = if vx < left as f64 {
        content.x - EXIT_PLACE_MARGIN_PX
    } else if vx > right as f64 {
        content.x + content.w + EXIT_PLACE_MARGIN_PX
    } else {
        (vx.round() as i32).clamp(left, right)
    };
    let y = if vy < top as f64 {
        content.y - EXIT_PLACE_MARGIN_PX
    } else if vy > bottom as f64 {
        content.y + content.h + EXIT_PLACE_MARGIN_PX
    } else {
        (vy.round() as i32).clamp(top, bottom)
    };
    (x, y)
}

fn reenter_cooldown(now: std::time::Instant, oscillations: u32) -> std::time::Instant {
    let mult = 1 + oscillations.min(OSC_MAX) as u64;
    now + std::time::Duration::from_millis(REENTER_COOLDOWN_MS * mult)
}

fn map_content_to_source(content: Rect, src: Rect, vx: f64, vy: f64) -> (i32, i32) {
    if content.w <= 1 || content.h <= 1 || src.w <= 1 || src.h <= 1 {
        return (src.x, src.y);
    }
    let fx = ((vx - content.x as f64 + 0.5) / content.w as f64).clamp(0.0, 1.0);
    let fy = ((vy - content.y as f64 + 0.5) / content.h as f64).clamp(0.0, 1.0);
    let tx = (src.x as f64 + fx * src.w as f64 - 0.5).round() as i32;
    let ty = (src.y as f64 + fy * src.h as f64 - 0.5).round() as i32;
    (
        tx.clamp(src.x, src.x + src.w - 1),
        ty.clamp(src.y, src.y + src.h - 1),
    )
}

fn map_source_to_content(content: Rect, src: Rect, sx: f64, sy: f64) -> (f64, f64) {
    if content.w <= 1 || content.h <= 1 || src.w <= 1 || src.h <= 1 {
        return (content.x as f64, content.y as f64);
    }
    let fx = ((sx - src.x as f64 + 0.5) / src.w as f64).clamp(0.0, 1.0);
    let fy = ((sy - src.y as f64 + 0.5) / src.h as f64).clamp(0.0, 1.0);
    (
        content.x as f64 + fx * content.w as f64 - 0.5,
        content.y as f64 + fy * content.h as f64 - 0.5,
    )
}

/// Like `map_source_to_content` but WITHOUT clamping the fraction to [0,1], so a
/// source position OUTSIDE the source rect (a pre-clip hook pt past the edge)
/// maps to a content position OUTSIDE the content rect. Used for the SPRITE
/// position so that, as the hand pushes past the source edge, the visible
/// cursor keeps travelling toward/past the content edge instead of being pulled
/// back to the clamped source-edge mapping.
fn map_source_to_content_unclamped(content: Rect, src: Rect, sx: f64, sy: f64) -> (f64, f64) {
    if content.w <= 1 || content.h <= 1 || src.w <= 1 || src.h <= 1 {
        return (content.x as f64, content.y as f64);
    }
    let fx = (sx - src.x as f64 + 0.5) / src.w as f64;
    let fy = (sy - src.y as f64 + 0.5) / src.h as f64;
    (
        content.x as f64 + fx * content.w as f64 - 0.5,
        content.y as f64 + fy * content.h as f64 - 0.5,
    )
}

fn content_per_source_px(content: Rect, src: Rect) -> (f64, f64) {
    if content.w <= 1 || content.h <= 1 || src.w <= 1 || src.h <= 1 {
        return (1.0, 1.0);
    }
    (
        content.w as f64 / src.w as f64,
        content.h as f64 / src.h as f64,
    )
}

fn clamp_virtual_point(content: Rect, vx: f64, vy: f64, margin: f64) -> (f64, f64) {
    (
        vx.clamp(
            content.x as f64 - margin,
            (content.x + content.w) as f64 + margin,
        ),
        vy.clamp(
            content.y as f64 - margin,
            (content.y + content.h) as f64 + margin,
        ),
    )
}

fn remap_virtual_between_content(
    old_content: Rect,
    new_content: Rect,
    vx: f64,
    vy: f64,
) -> (f64, f64) {
    let margin = EXIT_PLACE_MARGIN_PX as f64;
    if old_content.w <= 1 || old_content.h <= 1 || new_content.w <= 1 || new_content.h <= 1 {
        let dx = (new_content.x - old_content.x) as f64;
        let dy = (new_content.y - old_content.y) as f64;
        return clamp_virtual_point(new_content, vx + dx, vy + dy, margin);
    }
    let fx = (vx - old_content.x as f64) / old_content.w as f64;
    let fy = (vy - old_content.y as f64) / old_content.h as f64;
    clamp_virtual_point(
        new_content,
        new_content.x as f64 + fx * new_content.w as f64,
        new_content.y as f64 + fy * new_content.h as f64,
        margin,
    )
}

fn edge_out_amount(src: Rect, px: i32, py: i32) -> i32 {
    let right = src.x + src.w;
    let bottom = src.y + src.h;
    [src.x - px, px - right, src.y - py, py - bottom, 0]
        .into_iter()
        .max()
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeSide {
    Left,
    Right,
    Top,
    Bottom,
}

fn edge_out_side(src: Rect, px: i32, py: i32) -> Option<EdgeSide> {
    let right = src.x + src.w;
    let bottom = src.y + src.h;
    let candidates = [
        (EdgeSide::Left, src.x - px),
        (EdgeSide::Right, px - right),
        (EdgeSide::Top, src.y - py),
        (EdgeSide::Bottom, py - bottom),
    ];
    candidates
        .into_iter()
        .filter(|(_, out)| *out > 0)
        .max_by_key(|(_, out)| *out)
        .map(|(side, _)| side)
}

fn fullscreen_ui_target_for_edge(g: &State, side: EdgeSide) -> Option<Rect> {
    let c = g.content;
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    // Ask Win32 which REAL top-level window owns the adjacent screen point.
    // This preserves access to a GUI in letterbox space while preventing list
    // order or a stale/expanded panel rectangle from inventing an edge target.
    let sample = match side {
        EdgeSide::Left => (c.x - 1, vy),
        EdgeSide::Right => (c.x + c.w, vy),
        EdgeSide::Top => (vx, c.y - 1),
        EdgeSide::Bottom => (vx, c.y + c.h),
    };
    hit_ui_for_cursor_ownership(&g.no_engage, sample.0, sample.1).map(|h| h.land)
}

fn extend_fullscreen_virtual_into_ui(g: &mut State, side: EdgeSide, out: i32) -> bool {
    let Some(target) = fullscreen_ui_target_for_edge(g, side) else {
        return false;
    };
    let c = g.content;
    let (scale_x, scale_y) = content_per_source_px(c, g.src);
    let step = match side {
        EdgeSide::Left | EdgeSide::Right => scale_x,
        EdgeSide::Top | EdgeSide::Bottom => scale_y,
    }
    .max(1.0);
    g.edge_out_accum += out as f64 * step;
    let travel = g.edge_out_accum;
    let right = (c.x + c.w - 1) as f64;
    let bottom = (c.y + c.h - 1) as f64;
    g.virt = match side {
        EdgeSide::Left => (
            (c.x as f64 - travel).clamp(target.x as f64, c.x as f64),
            g.virt.1,
        ),
        EdgeSide::Right => (
            (right + travel).clamp(right, (target.x + target.w - 1) as f64),
            g.virt.1,
        ),
        EdgeSide::Top => (
            g.virt.0,
            (c.y as f64 - travel).clamp(target.y as f64, c.y as f64),
        ),
        EdgeSide::Bottom => (
            g.virt.0,
            (bottom + travel).clamp(bottom, (target.y + target.h - 1) as f64),
        ),
    };
    true
}

fn virtual_over_ui(g: &State) -> bool {
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    hit_ui_for_visible_cursor(g, vx, vy).is_some()
}

fn note_ui_hover(g: &mut State, _now: std::time::Instant) -> Option<UiHover> {
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    let hit = hit_ui_for_visible_cursor(g, vx, vy)?;
    let hover = UiHover {
        hit,
        pos: ui_landing_point(hit, vx, vy),
    };
    g.last_ui_hover = Some(hover);
    Some(hover)
}

fn ui_target_at(
    g: &mut State,
    x: i32,
    y: i32,
    margin: i32,
    _now: std::time::Instant,
) -> Option<UiHover> {
    debug_assert_eq!(margin, 0, "native UI routing must be exact");
    let hit = hit_ui_for_cursor_ownership(&g.no_engage, x, y)?;
    let hover = UiHover {
        hit,
        pos: ui_landing_point(hit, x, y),
    };
    g.last_ui_hover = Some(hover);
    Some(hover)
}

fn invalidate_stale_ui_hover_after_no_engage_change(
    g: &mut State,
) -> Option<(NoEngageRect, Option<NoEngageRect>)> {
    let last = g.last_ui_hover?;
    if g.no_engage.iter().any(|r| *r == last.hit) {
        return None;
    }
    let replacement = g
        .no_engage
        .iter()
        .find(|r| r.hwnd != 0 && r.hwnd == last.hit.hwnd)
        .copied();
    g.last_ui_hover = None;
    g.last_ui_post = None;
    g.ui_hover_active = false;
    Some((last.hit, replacement))
}

/// Refresh Neo's ordinary GUI exclusion rectangles directly from Win32 on the
/// low-level mouse path. The GUI thread publishes geometry latest-only, but a
/// very heavy render frame can still delay the render thread from consuming
/// that latest sample for a few hundred milliseconds. During a native title-bar
/// drag that delay is enough for the old rect to say "content", causing an
/// engage warp that Windows then interprets as part of the window drag.
///
/// This is intentionally dimension/language/DPI agnostic: the HWND is the
/// identity and GetWindowRect is the source of truth. Main GUI and control
/// panel are refreshed identically; panel identity is explicit, not inferred
/// from an expanded geometry halo.
fn refresh_live_own_no_engage_geometry(g: &mut State, now: std::time::Instant) -> bool {
    let mut changed = false;
    for hit in &mut g.no_engage {
        if hit.hwnd == 0 || !crate::platform::win32::is_own_window(hit.hwnd) {
            continue;
        }
        unsafe {
            let mut wr = RECT::default();
            if GetWindowRect(HWND(hit.hwnd as *mut _), &mut wr).is_ok() {
                let live = Rect {
                    x: wr.left,
                    y: wr.top,
                    w: wr.right - wr.left,
                    h: wr.bottom - wr.top,
                };
                if live.w > 0 && live.h > 0 && (hit.rect != live || hit.land != live) {
                    hit.rect = live;
                    hit.land = live;
                    changed = true;
                }
            }
        }
    }
    if changed {
        g.no_engage_moved_at = Some(now);
        // A native drag owns its gesture until button-up. Do not churn hover
        // state merely because the owned window rectangle moves underneath it.
        if g.native_ui_hold_bits == 0 {
            let _ = invalidate_stale_ui_hover_after_no_engage_change(g);
        }
    }
    changed
}

fn ui_click_target_from_virtual(g: &mut State, now: std::time::Instant) -> Option<UiHover> {
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    // Clicks follow the window currently under the visible cursor. Never use a
    // previous-frame hover or an invisible margin after the cursor has left.
    ui_target_at(g, vx, vy, UI_CLICK_MARGIN_PX, now)
}

fn ui_click_target_for_event(
    g: &mut State,
    px: i32,
    py: i32,
    now: std::time::Instant,
) -> Option<UiHover> {
    if g.engaged {
        // While engaged the real cursor is hidden/clipped in SOURCE coordinates.
        // Reinterpreting that hidden raw coordinate as a screen-space UI hit was
        // another source of phantom GUI/panel ownership.
        ui_click_target_from_virtual(g, now)
    } else {
        ui_target_at(g, px, py, UI_CLICK_MARGIN_PX, now)
    }
}

fn post_mouse_move_to_hwnd(hwnd: isize, sx: i32, sy: i32) {
    if hwnd == 0 {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let mut pt = POINT { x: sx, y: sy };
        let _ = ScreenToClient(h, &mut pt);
        let lp = (((pt.y as u32 & 0xFFFF) << 16) | (pt.x as u32 & 0xFFFF)) as isize;
        let _ = PostMessageW(Some(h), WM_MOUSEMOVE, WPARAM(0), LPARAM(lp));
    }
}

fn queue_panel_action(hover: UiHover, bit: u8) {
    if bit != BTN_LEFT || !is_panel_hit(hover.hit) {
        return;
    }
    let land = hover.hit.land;
    let rel_x = hover.pos.0 - land.x;
    let action = panel_action_for_relative_x(land.w, land.h, rel_x);
    let Some(action) = action else { return };
    if action == PANEL_ACTION_GUI_TOPMOST {
        PANEL_GUI_ACTION_SEQ.fetch_add(1, Ordering::Release);
    }
    PANEL_ACTIONS.fetch_or(action, Ordering::Release);
    // Same logical action as Ctrl+Alt+G: wake the GUI event loop only. Do not
    // minimize, restore, or otherwise alter GUI visibility from the panel.
    crate::platform::win32::wake_main_gui_for_panel_action(false);
    log::info!(
        "panel direct action queued: action={} hwnd={:#x} pos=({},{}) land={:?}",
        match action {
            PANEL_ACTION_STOP => "stop",
            PANEL_ACTION_COLLAPSE => "collapse",
            PANEL_ACTION_EXPAND => "expand",
            PANEL_ACTION_SCREENSHOT => "screenshot",
            PANEL_ACTION_GUI_TOPMOST => "gui-topmost",
            _ => "unknown",
        },
        hover.hit.hwnd,
        hover.pos.0,
        hover.pos.1,
        land
    );
}

/// Hit-test the fixed [stop][fps][camera][GUI][collapse] panel layout. Boundaries
/// sit in the middle of each 3pt visual gap, so direct hook actions and the
/// painted controls cannot disagree near an edge.
pub fn panel_action_for_relative_x(width: i32, height: i32, relative_x: i32) -> Option<u32> {
    if width <= 0 || height <= 0 || relative_x < 0 || relative_x >= width {
        return None;
    }
    // Lurk geometry must be recognized independently of DPI. The old <=60px
    // width heuristic only worked while the transparent chip was 34pt wide;
    // after widening the hit area to 68pt it can exceed 60 physical pixels even
    // at ordinary Windows scaling. The lurk target is compact (~2.8:1), while
    // the full control bar is wide (~9:1), so classify by aspect ratio instead.
    if width <= height.saturating_mul(4) {
        return Some(PANEL_ACTION_EXPAND);
    }
    const LAYOUT_UNITS: i32 = 270;
    const STOP_END: i32 = 84;
    const FPS_END: i32 = 155;
    const CAMERA_END: i32 = 192;
    const GUI_END: i32 = 229;
    let scaled = relative_x.saturating_mul(LAYOUT_UNITS) / width;
    if scaled < STOP_END {
        Some(PANEL_ACTION_STOP)
    } else if scaled < FPS_END {
        None
    } else if scaled < CAMERA_END {
        Some(PANEL_ACTION_SCREENSHOT)
    } else if scaled < GUI_END {
        Some(PANEL_ACTION_GUI_TOPMOST)
    } else {
        Some(PANEL_ACTION_COLLAPSE)
    }
}

pub fn take_panel_actions() -> u32 {
    PANEL_ACTIONS.swap(0, Ordering::AcqRel)
}

pub fn panel_gui_action_seq() -> u32 {
    PANEL_GUI_ACTION_SEQ.load(Ordering::Acquire)
}

/// Consume only the emergency Stop bit. The render thread uses this while the
/// root GUI is minimized and its normal egui action drain is suspended.
pub fn take_panel_stop_action() -> bool {
    PANEL_ACTIONS.fetch_and(!PANEL_ACTION_STOP, Ordering::AcqRel) & PANEL_ACTION_STOP != 0
}

/// Publish the physical client-space rectangle and current action of the main
/// capture control. The LL hook combines this client rectangle with the live
/// window origin on every click, so moving the GUI cannot leave stale hit-test
/// coordinates behind.
pub fn set_main_control_surface(hwnd: isize, rect: Option<(i32, i32, i32, i32)>, mode: u32) {
    if mode == MAIN_CONTROL_DISABLED || hwnd == 0 {
        MAIN_CONTROL_MODE.store(MAIN_CONTROL_DISABLED, Ordering::Release);
        MAIN_DIRECT_ACTIVE.store(false, Ordering::Release);
        MAIN_ACTIONS.store(0, Ordering::Release);
        ACTIVE_MAIN_GUI_HWND.store(hwnd, Ordering::Release);
        return;
    }
    let Some((x, y, w, h)) = rect else {
        MAIN_CONTROL_MODE.store(MAIN_CONTROL_DISABLED, Ordering::Release);
        MAIN_DIRECT_ACTIVE.store(false, Ordering::Release);
        MAIN_ACTIONS.store(0, Ordering::Release);
        return;
    };
    if w <= 0 || h <= 0 {
        MAIN_CONTROL_MODE.store(MAIN_CONTROL_DISABLED, Ordering::Release);
        MAIN_DIRECT_ACTIVE.store(false, Ordering::Release);
        MAIN_ACTIONS.store(0, Ordering::Release);
        return;
    }
    // A few physical pixels cover rounding between egui points, DPI scaling
    // and Win32 client coordinates. The control is isolated at the left edge
    // of its row, so this does not overlap a neighbouring action.
    const PAD: i32 = 4;
    MAIN_CONTROL_LEFT.store(x - PAD, Ordering::Relaxed);
    MAIN_CONTROL_TOP.store(y - PAD, Ordering::Relaxed);
    MAIN_CONTROL_RIGHT.store(x + w + PAD, Ordering::Relaxed);
    MAIN_CONTROL_BOTTOM.store(y + h + PAD, Ordering::Relaxed);
    ACTIVE_MAIN_GUI_HWND.store(hwnd, Ordering::Release);
    MAIN_CONTROL_MODE.store(mode, Ordering::Release);
}

pub fn take_main_actions() -> u32 {
    MAIN_ACTIONS.swap(0, Ordering::AcqRel)
}

pub fn main_control_direct_pressed() -> bool {
    MAIN_DIRECT_ACTIVE.load(Ordering::Acquire)
}

fn try_main_control_lockfree(msg: u32, px: i32, py: i32) -> bool {
    // The matching UP is deliberately still delivered to winit/egui so its
    // pointer state stays balanced. GUI release/click is presentation-only;
    // the actual Start/Stop action is committed on this physical DOWN.
    if msg == WM_LBUTTONUP && MAIN_DIRECT_ACTIVE.swap(false, Ordering::AcqRel) {
        return true;
    }
    if msg != WM_LBUTTONDOWN {
        return false;
    }
    let mode = MAIN_CONTROL_MODE.load(Ordering::Acquire);
    if mode == MAIN_CONTROL_DISABLED {
        return false;
    }
    let hwnd = ACTIVE_MAIN_GUI_HWND.load(Ordering::Acquire);
    if hwnd == 0
        || !crate::platform::win32::is_window_valid(hwnd)
        || !crate::platform::win32::is_window_visible(hwnd)
        || crate::platform::win32::window_input_passthrough(hwnd)
    {
        return false;
    }
    let Some((cx, cy, _, _)) = crate::platform::win32::client_rect_on_screen(hwnd) else {
        return false;
    };
    let left = cx + MAIN_CONTROL_LEFT.load(Ordering::Relaxed);
    let top = cy + MAIN_CONTROL_TOP.load(Ordering::Relaxed);
    let right = cx + MAIN_CONTROL_RIGHT.load(Ordering::Relaxed);
    let bottom = cy + MAIN_CONTROL_BOTTOM.load(Ordering::Relaxed);
    if px < left || px >= right || py < top || py >= bottom {
        return false;
    }
    if crate::platform::win32::direct_top_level_window_at_point(px, py) != hwnd {
        return false;
    }
    let (action, name) = match mode {
        MAIN_CONTROL_START => (MAIN_ACTION_START, "start"),
        MAIN_CONTROL_STOP => (MAIN_ACTION_STOP, "stop"),
        _ => return false,
    };
    MAIN_ACTIONS.fetch_or(action, Ordering::Release);
    MAIN_DIRECT_ACTIVE.store(true, Ordering::Release);
    log::info!(
        "main-control-direct: action={name} hwnd={:#x} pos=({px},{py}) rect=({left},{top})-({right},{bottom}) lockfree=true",
        hwnd
    );
    true
}

fn direct_panel_point(px: i32, py: i32) -> (i32, i32, &'static str) {
    // When the mapped sprite owns the cursor, its coalesced target is the
    // visible screen-space pointer. Otherwise the hook's hardware point is
    // already the visible native cursor position.
    if SPRITE_SHOW.load(Ordering::Acquire) && CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
        let packed = SPRITE_TARGET.load(Ordering::Acquire);
        (packed as i32, (packed >> 32) as i32, "sprite")
    } else {
        (px, py, "native")
    }
}

/// First-chance, lock-free floating-panel click routing. This runs before any
/// State::try_lock() path. It fixes the long-standing intermittent case where
/// the first click only changed ownership/focus and the second click performed
/// the action, or where a brief state-mutex collision let the first DOWN fall
/// through. The panel still has to be the actual top interactive window at the
/// visible point, so a main GUI genuinely above it remains authoritative.
fn try_panel_direct_lockfree(msg: u32, px: i32, py: i32) -> bool {
    let Some((bit, down)) = button_transition(msg) else {
        return false;
    };
    if bit != BTN_LEFT {
        return false;
    }

    if !down {
        if PANEL_DIRECT_SWALLOW_UP.load(Ordering::Acquire) & bit != 0 {
            PANEL_DIRECT_SWALLOW_UP.fetch_and(!bit, Ordering::AcqRel);
            return true;
        }
        return false;
    }

    let hwnd = ACTIVE_PANEL_HWND.load(Ordering::Acquire);
    if hwnd == 0
        || !crate::platform::win32::is_window_valid(hwnd)
        || !crate::platform::win32::is_window_visible(hwnd)
        || crate::platform::win32::window_input_passthrough(hwnd)
    {
        return false;
    }

    let (vx, vy, owner) = direct_panel_point(px, py);
    let Some((rx, ry, rw, rh)) = crate::platform::win32::window_rect(hwnd) else {
        return false;
    };
    if rw <= 0 || rh <= 0 || vx < rx || vx >= rx + rw || vy < ry || vy >= ry + rh {
        return false;
    }
    if crate::platform::win32::direct_top_level_window_at_point(vx, vy) != hwnd {
        return false;
    }

    if let Some(action) = panel_action_for_relative_x(rw, rh, vx - rx) {
        if action == PANEL_ACTION_STOP {
            let gui = crate::platform::win32::main_gui_hwnd();
            if gui != 0 && crate::platform::win32::is_minimized(gui) {
                // Return to this panel click after emergency release, not to
                // the old title-bar point saved when minimization began.
                record_gui_minimize_cursor();
            }
        }
        if action == PANEL_ACTION_GUI_TOPMOST {
            PANEL_GUI_ACTION_SEQ.fetch_add(1, Ordering::Release);
        }
        PANEL_ACTIONS.fetch_or(action, Ordering::Release);
        // Keep the GUI button identical to Ctrl+Alt+G. It only requests the
        // existing TOPMOST toggle and never changes minimize/restore state.
        crate::platform::win32::wake_main_gui_for_panel_action(false);
        PANEL_DIRECT_SWALLOW_UP.fetch_or(bit, Ordering::AcqRel);
        log::debug!(
            "panel first-click direct: action={} hwnd={:#x} pos=({vx},{vy}) owner={} lockfree=true",
            match action {
                PANEL_ACTION_STOP => "stop",
                PANEL_ACTION_COLLAPSE => "collapse",
                PANEL_ACTION_EXPAND => "expand",
                PANEL_ACTION_SCREENSHOT => "screenshot",
                PANEL_ACTION_GUI_TOPMOST => "gui-topmost",
                _ => "unknown",
            },
            hwnd,
            owner
        );
        return true;
    }
    false
}

/// Returns true when this move handed ownership to a native GUI and moved the
/// real cursor there. The hook must consume that triggering source-space event,
/// or Windows will apply its old source coordinate after the handoff.
fn post_ui_hover_from_state(g: &mut State, now: std::time::Instant) -> bool {
    if let Some(hover) = note_ui_hover(g, now) {
        if hover.hit.hwnd != 0 {
            if is_panel_hit(hover.hit) {
                // v370: treat the floating panel as an ordinary native-owned
                // window, exactly like the main GUI. The old sprite-only hover
                // path kept the real cursor clipped in source space and relied
                // on synthetic WM_MOUSEMOVE/click redirection; near the panel's
                // 66px boundary that ownership could be lost before the user
                // could settle on a control. Release source ownership at the
                // first exact panel pixel and let normal Win32 hover/capture
                // semantics own the gesture until the pointer leaves.
                if handoff_to_gui(g, hover, now) {
                    log::info!(
                        "panel entry handoff to hwnd={:#x} at ({},{}) owner=native",
                        hover.hit.hwnd,
                        hover.pos.0,
                        hover.pos.1
                    );
                    return true;
                }
                log::warn!(
                    "panel entry deferred after unverified warp: hwnd={:#x} at ({},{})",
                    hover.hit.hwnd,
                    hover.pos.0,
                    hover.pos.1
                );
            } else {
                // The GUI is our own window; the moment
                // the virtual cursor enters it, switch to native input ownership
                // (real cursor at the sprite position, visually represented by
                // the proven Neo sprite during active capture). Inside the GUI
                // everything follows ordinary Windows hit-test/capture rules;
                // leaving it is classified immediately by the live HWND edge.
                if handoff_to_gui(g, hover, now) {
                    log::info!(
                        "{} entry handoff to hwnd={:#x} at ({},{})",
                        if crate::platform::win32::is_own_window(hover.hit.hwnd) {
                            "gui"
                        } else {
                            "external-native-window"
                        },
                        hover.hit.hwnd,
                        hover.pos.0,
                        hover.pos.1
                    );
                    return true;
                }
                log::warn!(
                    "native UI entry deferred after unverified warp: hwnd={:#x} at ({},{})",
                    hover.hit.hwnd,
                    hover.pos.0,
                    hover.pos.1
                );
            }
        } else {
            post_mouse_move_to_hwnd(hover.hit.hwnd, hover.pos.0, hover.pos.1);
            // UI hover is never an idle-hidden state. A topmost GUI can be
            // raised above the cursor sprite after we enter it, so keep the
            // sprite visible and reassert its z-order while the virtual cursor
            // is over UI.
            g.hidden_by_idle = false;
            g.ui_hover_active = true;
            sprite_move(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
            keep_cursor_sprite_on_top();
        }
    } else {
        g.ui_hover_active = false;
    }
    false
}

fn fullscreen_ui_virtual_move(
    g: &mut State,
    old_virt: (f64, f64),
    _raw: (i32, i32),
    raw_delta: (i32, i32),
) -> bool {
    if !g.fullscreen || g.buttons_down != 0 {
        return false;
    }

    let ox = old_virt.0.round() as i32;
    let oy = old_virt.1.round() as i32;
    let Some(old_hit) = hit_ui_for_cursor_ownership(&g.no_engage, ox, oy) else {
        return false;
    };

    // The floating panel lives inside fullscreen content. It must behave like a
    // transparent waypoint for cursor motion: normal source->content mapping
    // already moves the sprite through it smoothly. Never switch to a special
    // panel coordinate mode and never clamp into the panel rectangle.
    if is_panel_hit(old_hit) {
        return false;
    }

    // A non-panel native window can briefly remain in virtual mode during an
    // edge/letterbox handoff. Continue the visible cursor by physical delta but
    // never clamp it back into the old window. The next exact WindowFromPoint
    // result decides ownership; leaving the window is therefore immediate.
    let (scale_x, scale_y) = content_per_source_px(g.content, g.src);
    let next = (
        old_virt.0 + raw_delta.0 as f64 * scale_x.max(1.0),
        old_virt.1 + raw_delta.1 as f64 * scale_y.max(1.0),
    );
    g.virt = next;
    hit_ui_for_cursor_ownership(&g.no_engage, next.0.round() as i32, next.1.round() as i32)
        .is_some()
}

fn should_hide_cursor_for_idle(g: &State, now: std::time::Instant) -> bool {
    if !g.engaged || g.hidden_by_idle || g.autohide_secs <= 0.0 || virtual_over_ui(g) {
        return false;
    }
    g.last_move
        .map(|t| now.duration_since(t).as_secs_f32() > g.autohide_secs)
        .unwrap_or(false)
}

fn log_fullscreen_ui_edge_miss(
    g: &State,
    side: EdgeSide,
    px: i32,
    py: i32,
    old_virt: (f64, f64),
    mapped: (f64, f64),
) {
    static COUNT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    if g.no_engage.is_empty() || COUNT.fetch_add(1, Ordering::Relaxed) >= 8 {
        return;
    }
    log::info!(
        "fullscreen-ui edge miss: side={side:?} raw=({px},{py}) old_virt=({:.1},{:.1}) mapped=({:.1},{:.1}) content={:?} no_engage={:?}",
        old_virt.0,
        old_virt.1,
        mapped.0,
        mapped.1,
        g.content,
        g.no_engage
    );
}

fn on_or_past_source_edge(src: Rect, px: i32, py: i32) -> bool {
    px <= src.x || px >= src.x + src.w - 1 || py <= src.y || py >= src.y + src.h - 1
}

/// Max px the exit placement may sit outside the content — clamps a stale/huge
/// pre-clip sprite position so a bad event can't fling the cursor across the
/// screen on exit.
#[cfg(test)]
const EXIT_OVERSHOOT_MAX_PX: i32 = 300;

/// Place the exit cursor exactly where the sprite (raw-driven `virt`) already
/// is, when that is OUTSIDE the content — so revealing the system cursor there
/// is seamless (no jump between sprite and cursor). None if still inside.
#[cfg(test)]
fn exit_place_from_virt(c: Rect, virt: (f64, f64)) -> Option<(i32, i32)> {
    let x = virt.0.round() as i32;
    let y = virt.1.round() as i32;
    if x >= c.x && x < c.x + c.w && y >= c.y && y < c.y + c.h {
        return None;
    }
    let m = EXIT_OVERSHOOT_MAX_PX;
    Some((
        x.clamp(c.x - m, c.x + c.w + m),
        y.clamp(c.y - m, c.y + c.h + m),
    ))
}

#[cfg(test)]
fn exit_point_for_source_edge(content: Rect, src: Rect, px: i32, py: i32) -> (i32, i32) {
    let (sx, sy) = content_per_source_px(content, src);
    let mut vx = content.x as f64 + (px - src.x) as f64 * sx;
    let mut vy = content.y as f64 + (py - src.y) as f64 * sy;
    if px < src.x {
        vx = (content.x - EXIT_PLACE_MARGIN_PX) as f64;
    } else if px >= src.x + src.w {
        vx = (content.x + content.w + EXIT_PLACE_MARGIN_PX) as f64;
    }
    if py < src.y {
        vy = (content.y - EXIT_PLACE_MARGIN_PX) as f64;
    } else if py >= src.y + src.h {
        vy = (content.y + content.h + EXIT_PLACE_MARGIN_PX) as f64;
    }
    (
        (vx.round() as i32).clamp(
            content.x - EXIT_PLACE_MARGIN_PX,
            content.x + content.w + EXIT_PLACE_MARGIN_PX,
        ),
        (vy.round() as i32).clamp(
            content.y - EXIT_PLACE_MARGIN_PX,
            content.y + content.h + EXIT_PLACE_MARGIN_PX,
        ),
    )
}

fn escape_rect(g: &State) -> Rect {
    if !g.fullscreen && g.overlay.w > 0 && g.overlay.h > 0 {
        g.overlay
    } else {
        g.content
    }
}

fn exit_point_for_escape(g: &State, px: i32, py: i32) -> (i32, i32) {
    let r = escape_rect(g);
    let c = g.content;
    let s = g.src;
    let virt = g.virt;
    let margin = if g.fullscreen {
        EXIT_PLACE_MARGIN_PX
    } else if g.window_frame_input {
        // Full-frame input and visible WGC coordinates are now identical, so
        // the native cursor can cross at the same one-pixel boundary as an
        // ordinary Windows window.
        1
    } else {
        EXIT_WINDOW_MARGIN_PX
    };
    let left = px < s.x || virt.0 < c.x as f64;
    let right = px >= s.x + s.w || virt.0 >= (c.x + c.w) as f64;
    let top = py < s.y || virt.1 < c.y as f64;
    let bottom = py >= s.y + s.h || virt.1 >= (c.y + c.h) as f64;
    let x = if left {
        r.x - margin
    } else if right {
        r.x + r.w + margin
    } else {
        (virt.0.round() as i32).clamp(r.x, r.x + r.w - 1)
    };
    let y = if top {
        r.y - margin
    } else if bottom {
        r.y + r.h + margin
    } else {
        (virt.1.round() as i32).clamp(r.y, r.y + r.h - 1)
    };
    (x, y)
}

fn clip_rect(src: Rect) -> RECT {
    RECT {
        left: src.x,
        top: src.y,
        right: src.x + src.w,
        bottom: src.y + src.h,
    }
}

/// Restore process-wide cursor/clip/speed state without injecting any mouse
/// button event. This is the only recovery routine that may run at startup.
/// Injecting an unconditional RIGHTUP here can be interpreted by Explorer as
/// a completed right-click and opened its context menu before Neo appeared.
pub fn startup_recover_input_state() {
    start_input_failsafe_worker();
    // PID values are reusable. Never inherit a stale recovery snapshot from an
    // older crashed process that happened to own the same numeric PID.
    clear_janitor_source_recovery();
    INPUT_FAILSAFE_LATCHED.store(false, Ordering::Release);
    CAPTURE_SESSION_ACTIVE.store(false, Ordering::Release);
    SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    // ask the render-engine thread (Mag owner) to reveal the cursor if it is
    // still alive. Independent fail-safe/janitor recovery covers the case where
    // that owner thread is unavailable, so startup never relies on an implicit
    // OS cursor-visibility reset.
    WANT_CURSOR_HIDDEN.store(false, Ordering::Relaxed);
    CURSOR_HIDE_APPLIED.store(false, Ordering::Release);
    DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
    MAG_REINIT_REQUESTED.store(true, Ordering::Release);
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
    ACTIVE_OVERLAY_HWND.store(0, Ordering::Release);
    cancel_cursor_reveal();
    sprite_hide();
    restore_configured_system_cursors();
    unsafe {
        let _ = ClipCursor(None);
    }
    restore_mouse_speed();
    heal_leftover_mouse_speed();
}

/// Stop/panic recovery. Button release is allowed only on an active teardown;
/// startup paths must call startup_recover_input_state instead.
///
/// IMPORTANT: unlike startup recovery, teardown must not publish
/// CURSOR_HIDE_APPLIED=false or hide Neo's sprite before the Magnification
/// owner thread has actually completed MagShowSystemCursor(true). TensorRT
/// teardown can keep that owner busy for >1 s; clearing the visual contract
/// first creates a visible cursor-less gap immediately after Stop.
pub fn emergency_release_all() {
    SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    // Disable capture ownership lock-free first so the render/Mag thread cannot
    // reassert WANT_CURSOR_HIDDEN=true while teardown is trying to reveal the
    // native cursor. Keep the sprite itself alive until native reveal commits.
    CAPTURE_SPRITE_ACTIVE.store(false, Ordering::Release);
    cancel_source_caption_drag_contract();
    end_native_gui_caption_drag("emergency-release");
    WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
    ACTIVE_OVERLAY_HWND.store(0, Ordering::Release);
    cancel_cursor_reveal();
    unsafe {
        let _ = ClipCursor(None);
    }
    restore_mouse_speed();
    heal_leftover_mouse_speed();
    restore_configured_system_cursors();

    // The GUI can currently be click-through while source ownership is armed.
    // Restoring it must not depend on acquiring the input-state mutex: Stop can
    // race a high-rate LL-hook move which briefly owns that lock. Previously a
    // single failed try_lock left WS_EX_TRANSPARENT set after capture stopped,
    // making the GUI visible but neither clickable nor draggable.
    let gui_hwnd = MAIN_GUI_HWND.load(Ordering::Acquire);
    if gui_hwnd != 0 && crate::platform::win32::is_window_valid(gui_hwnd) {
        publish_main_gui_passthrough(gui_hwnd, false, "emergency-release");
        crate::platform::win32::set_window_input_passthrough(gui_hwnd, false);
    }

    // Release only state that Neo itself is tracking. Never synthesize a
    // process-wide UP merely because the physical button happens to be held.
    for attempt in 1..=8 {
        if let Ok(mut g) = state().try_lock() {
            let fallback = (g.virt.0.round() as i32, g.virt.1.round() as i32);
            let minimize_at = GUI_MINIMIZE_CURSOR_AT_MS.load(Ordering::Acquire);
            let minimize_cursor = unpack_drag_pair(GUI_MINIMIZE_CURSOR.load(Ordering::Acquire));
            let gui_is_minimized = gui_hwnd != 0
                && crate::platform::win32::is_minimized(gui_hwnd)
                && minimize_at != 0
                && route_clock_ms().saturating_sub(minimize_at) <= 60_000;
            // While engaged, GetCursorPos is intentionally in hidden-source
            // space, so the visible sprite coordinate is authoritative. Once
            // disengaged (for example the Stop button on Neo's GUI), the real
            // cursor is already at the visible desktop point and is preferable.
            let visible = if gui_is_minimized {
                minimize_cursor
            } else if g.engaged || g.cursor_hidden {
                fallback
            } else {
                read_cursor_pos_or(fallback, "emergency release visible bridge")
            };
            g.virt = (visible.0 as f64, visible.1 as f64);
            GUI_MINIMIZE_CURSOR_AT_MS.store(0, Ordering::Release);
            g.active = false;
            clear_capture_sprite_contract(&mut g);
            release_locked(&mut g);
            g.buttons_down = 0;
            g.native_ui_hold_bits = 0;

            if CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
                // Fail-visible bridge: keep the Neo sprite following the user
                // until the Mag-owner thread actually reveals native. Do not
                // lie by clearing CURSOR_HIDE_APPLIED here; only
                // pump_cursor_visibility() may publish that transition.
                DESKTOP_REVEAL_BRIDGE_ACTIVE.store(true, Ordering::Release);
                SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
                sprite_move_now(visible.0, visible.1, true);
                request_cursor_hidden(false);
                log::info!(
                    "emergency-cursor-reveal-bridge: armed=true pos=({},{}) native_hidden=true",
                    visible.0,
                    visible.1
                );
            } else {
                DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
                sprite_hide();
            }
            log::info!("emergency-input-state-release: result=complete attempt={attempt}");
            return;
        }
        if attempt < 8 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    // Even if the state mutex is poisoned/busy, never trade an input recovery
    // failure for an invisible cursor. The hook can continue moving this
    // bridge lock-free until the native reveal is confirmed.
    if CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
        DESKTOP_REVEAL_BRIDGE_ACTIVE.store(true, Ordering::Release);
        SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
        let fallback = unpack_drag_pair(SPRITE_TARGET.load(Ordering::Acquire));
        let visible = read_cursor_pos_or(fallback, "emergency release lock-timeout bridge");
        sprite_move(visible.0, visible.1, true);
        request_cursor_hidden(false);
    }
    log::error!(
        "emergency-input-state-release: result=state-lock-timeout gui_passthrough_recovered=true cursor_bridge={}",
        DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire)
    );
}

#[derive(Clone, Copy, Debug)]
struct PendingEngage {
    origin: (i32, i32),
    target: (i32, i32),
    sprite: (i32, i32),
    src: Rect,
    requested_at: std::time::Instant,
}

#[derive(Clone, Copy, Debug)]
struct NativeGuiSettle {
    target: (i32, i32),
    source_origin: (i32, i32),
    until: std::time::Instant,
    reasserted: bool,
}

#[derive(Default)]
struct State {
    active: bool,
    /// fullscreen (Auto) mode: the magnified view IS the whole screen, so an
    /// edge push must NOT slip the cursor out (there is nowhere to go and the
    /// control panel lives at the top edge). Matches the cursor-routing design, where the
    /// fullscreen/"clip" policy keeps the cursor confined and only the optional
    /// peek reveals the desktop. Windowed (Fixed) mode still exits on a push.
    fullscreen: bool,
    /// Windowed full-frame capture maps input against the same DWM/outer
    /// coordinate space as the visible WGC frame. In this mode boundaries
    /// should behave like an ordinary window: no hidden re-entry band/timer.
    window_frame_input: bool,
    /// Exact native rectangle family used for the current source mapping.
    /// Change-only diagnostics make WGC/input coordinate mismatches obvious
    /// without flooding the log on every render tick.
    input_reference_kind: &'static str,
    overlay: Rect,
    content: Rect, // on-screen rect where the source image is displayed
    src: Rect,     // source client rect on screen
    src_hwnd: isize,
    no_engage: Vec<NoEngageRect>,
    engaged: bool,
    /// GUI -> magnified-content handoff waiting for the Magnification owner
    /// thread to confirm that the native cursor is hidden. Until committed we
    /// consume movement without warping the visible cursor into source space.
    pending_engage: Option<PendingEngage>,
    /// Updated by the render engine on every geometry/configuration tick.
    /// Mouse-hook events use it as a last-resort escape if rendering stalls.
    last_configure_at: Option<std::time::Instant>,
    /// Provider/session transitions may intentionally pause the render heartbeat
    /// for many seconds (notably a cold TensorRT engine build). During that
    /// window input ownership is released and the hook must not report a stale
    /// mapping failure or re-engage against old geometry.
    transition_suspended: bool,
    cursor_hidden: bool,
    /// virtual cursor in CONTENT (screen) space — the user's hand moves this
    /// at natural 1:1 speed; the real cursor is teleported to the mapped
    /// source position (sub-pixel accurate)
    virt: (f64, f64),
    /// last source position we injected (hardware pt = this + raw delta)
    last_set: (i32, i32),
    /// last hardware pt seen (stale queued events continue from this base)
    last_hw: (i32, i32),
    /// re-engage cooldown after a disengage (prevents instant suck-back)
    cooldown_until: Option<std::time::Instant>,
    /// last hardware-move time (cursor auto-hide)
    last_move: Option<std::time::Instant>,
    hidden_by_idle: bool,
    /// auto-hide seconds (0 = disabled)
    autohide_secs: f32,
    /// when true, slow the OS pointer by 1/zoom while engaged so the sprite
    /// tracks the hand 1:1 (matches the disengaged speed → seamless edge
    /// crossing. Driven by the cursor-speed compensation option.
    adjust_speed: bool,
    /// Mouse buttons currently held according to the hardware hook. While a
    /// button is held, movement is a source drag and must not trigger edge
    /// escape.
    buttons_down: u8,
    /// Physical button gestures that STARTED on Neo's native main GUI while
    /// source ownership was disengaged. These are ordinary Windows move/resize/
    /// control gestures, not the synthetic engaged->GUI `ui_hold_bits` path.
    /// While set, source engage is forbidden so SetCursorPos can never warp a
    /// live native window drag.
    native_ui_hold_bits: u8,
    /// Buttons whose hardware UP must be swallowed because their DOWN was
    /// redirected to the panel as a synthetic full click (bit per button).
    swallow_up: u8,
    /// Physical source-button gestures that had to be forwarded directly to
    /// the source HWND because the hidden source point was occluded by another
    /// top-most window (or the in-hook cursor warp could not be verified).
    /// While a bit is set, matching move/up messages are also routed directly
    /// so an occluding Task Manager/browser window can never steal half of the
    /// gesture.
    source_direct_bits: u8,
    /// Coalesces WM_MOUSEMOVE only while Neo owns a directly-forwarded source
    /// button gesture. Ordinary hover/movement always stays on the established
    /// native v636 path; this field must never make occlusion alone change cursor
    /// ownership.
    last_source_direct_post: Option<(isize, (i32, i32), u8, std::time::Instant)>,
    /// Buttons currently handed to a real GUI/control-panel window. While set,
    /// the magnifier must not re-engage: the user is dragging/clicking UI.
    ui_hold_bits: u8,
    /// Short post-release guard so the engine has a tick to refresh the moving
    /// Last physical/redirected button edge. The 50ms watchdog uses this to
    /// distinguish a genuinely held button from a lost UP event.
    last_button_event: Option<std::time::Instant>,
    /// last time the no-engage rect set CHANGED (GUI window being dragged)
    no_engage_moved_at: Option<std::time::Instant>,
    edge_out_accum: f64,
    last_engage_at: Option<std::time::Instant>,
    /// Engage teleports the real cursor with SetCursorPos; LL-hook events
    /// queued BEFORE that teleport still carry pre-teleport SCREEN coordinates
    /// but arrive AFTER, where they'd be mapped as SOURCE coordinates and
    /// fling the sprite (often into a topmost GUI => instant handoff bounce).
    /// Until the injected teleport event itself (pt == this target) is seen,
    /// swallow moves entirely — don't let them touch `virt`.
    expect_teleport: Option<(i32, i32)>,
    /// Commit-relative stale-event firewall. A saturated render thread can let
    /// the SetCursorPos-consistent event overtake older screen-space moves; the
    /// latter must remain quarantined briefly even after expect_teleport clears.
    teleport_guard_until: Option<std::time::Instant>,
    /// Post-edge screen-space settle.  The verified desktop warp can overtake
    /// older WH_MOUSE_LL events that were queued while the hardware cursor was
    /// still inside source space.  During this short window a wildly divergent
    /// raw point must not reclaim coordinate authority from GetCursorPos.
    edge_release_settle_until: Option<std::time::Instant>,
    oscillations: u32,
    /// Set after a windowed edge exit: suppress re-engage while the cursor still
    /// lingers in the edge band, so it cannot immediately snap back in.
    must_leave_content: bool,
    /// Last GUI/control-panel hover from the virtual cursor. Used as a short
    /// click grace so a one-frame edge clamp cannot send an intended GUI click
    /// through to the source.
    last_ui_hover: Option<UiHover>,
    /// Coalesce synthetic hover messages from high-polling mice. The action
    /// routing remains synchronous, while visual hover updates are capped at
    /// 125Hz so the panel cannot flood its viewport message queue.
    last_ui_post: Option<(isize, (i32, i32), std::time::Instant)>,
    ui_hover_active: bool,
    /// Native main-GUI ownership is independent from hover/cache state.
    /// Geometry refreshes may invalidate `last_ui_hover` while the physical
    /// cursor is still inside the same HWND; they must not retrigger entry.
    native_gui_owner_hwnd: isize,
    /// Last synchronous sprite update while Neo's native GUI owns the pointer.
    /// Normal GUI hover uses a bounded low-latency path: direct compositor
    /// updates at most every few milliseconds, with the existing coalesced
    /// message path handling intermediate/high-polling samples.
    last_gui_sprite_sync_at: Option<std::time::Instant>,
    /// Rate-limited desktop cursor ownership heartbeat. This remains
    /// diagnostic-only; it records both native/sprite state when the user is
    /// outside every Neo-owned surface so a reported invisible cursor can be
    /// matched to the exact internal visual-owner state.
    last_desktop_cursor_diag_at: Option<std::time::Instant>,
    /// Quarantines source-coordinate mouse moves already queued when ownership
    /// is handed to the real GUI cursor.
    native_gui_settle: Option<NativeGuiSettle>,
}

#[derive(Clone, Copy, Debug)]
struct UiHover {
    hit: NoEngageRect,
    pos: (i32, i32),
}

static STATE: OnceLock<Arc<Mutex<State>>> = OnceLock::new();
static SPRITE_HWND: OnceLock<isize> = OnceLock::new();
/// Native GUI ownership is published independently from the render cadence.
/// Native GUI ownership is published independently from render cadence. During
/// an active capture v382 keeps the real cursor Magnification-hidden and uses
/// the Neo sprite at the same screen coordinate; outside capture/provider
/// guards this owner still participates in ordinary native cursor restoration.
static NATIVE_GUI_OWNER: AtomicIsize = AtomicIsize::new(0);
/// The currently visible magnified overlay.  It is WS_EX_TRANSPARENT for the
/// native mouse, so WindowFromPoint can see a GUI that is actually BEHIND the
/// magnified image.  Ownership must follow visual Z-order instead: a window
/// covered by this overlay cannot steal the virtual/source cursor merely
/// because the overlay is click-through.
static ACTIVE_OVERLAY_HWND: AtomicIsize = AtomicIsize::new(0);
/// During active capture Neo's sprite is the sole visual cursor on overlay,
/// GUI/panel and desktop. The real Windows cursor remains at the native input
/// coordinate but stays Magnification-hidden until Stop/provider guard.
static CAPTURE_SPRITE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// Classification only: a physical left-button gesture that began on the
/// mapped source title bar. Unlike v376 this NEVER changes clip, mouse speed,
/// or moves the overlay from the LL hook; engine.rs alone follows source HWND.
///
/// v379 also treats each drag as a generation-scoped transaction. The visible
/// sprite is anchored to the exact grab offset inside the overlay for the whole
/// gesture, while the hidden native cursor remains the sole Windows input
/// authority. This prevents render/input geometry publication skew from making
/// the cursor creep across the title bar during aggressive moves.
static SOURCE_CAPTION_DRAG_ACTIVE: AtomicBool = AtomicBool::new(false);
static SOURCE_CAPTION_DRAG_EPOCH: AtomicU64 = AtomicU64::new(0);
static SOURCE_CAPTION_DRAG_ANCHOR_VALID: AtomicBool = AtomicBool::new(false);
/// Packed signed `(y << 32) | x` pairs. A single atomic load/store keeps X/Y
/// from coming from different follower samples during fast diagonal drags.
static SOURCE_CAPTION_DRAG_ANCHOR: AtomicI64 = AtomicI64::new(0);
static SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN: AtomicI64 = AtomicI64::new(0);
/// Generation tag for OVERLAY_ORIGIN. The origin and epoch are published as a
/// tiny seqlock-like pair: readers only use the origin when this tag matches
/// the currently active drag. This prevents a follower from an old drag from
/// overwriting the new drag's visual anchor during a fast release/re-press.
static SOURCE_CAPTION_DRAG_OVERLAY_EPOCH: AtomicU64 = AtomicU64::new(0);
/// Serializes the very small caption-drag geometry commit. The follower holds
/// this only while it moves the overlay, reads the committed HWND position and
/// publishes the matching sprite position. BUTTON-UP takes the same guard before
/// closing the generation, so no old follower can move the overlay after the
/// drag contract has ended.
static SOURCE_CAPTION_DRAG_WRITE_LOCK: AtomicBool = AtomicBool::new(false);
static SOURCE_CAPTION_DRAG_VISUAL_COMMIT_COUNT: AtomicU64 = AtomicU64::new(0);
/// Counts generic sprite writes rejected while the source-caption follower owns
/// the visual cursor. Any non-zero value is diagnostic evidence that another
/// clock tried to overwrite the fixed drag anchor; v382 blocks the write at
/// the common sprite API boundary instead of relying on every caller to guard.
static SOURCE_CAPTION_DRAG_SUPPRESSED_SPRITE_WRITES: AtomicU64 = AtomicU64::new(0);

/// Diagnostic anchor for a native Neo GUI title-bar drag. Windows continues to
/// own the actual move gesture, while v382 keeps the Neo sprite at the raw
/// WH_MOUSE_LL screen coordinate. The live-window anchor is retained only so
/// logs can quantify DWM/window lag without feeding it back into cursor position.
static NATIVE_GUI_CAPTION_DRAG_ACTIVE: AtomicBool = AtomicBool::new(false);
static NATIVE_GUI_CAPTION_DRAG_HWND: AtomicIsize = AtomicIsize::new(0);
static NATIVE_GUI_CAPTION_DRAG_ANCHOR: AtomicI64 = AtomicI64::new(0);
static NATIVE_GUI_CAPTION_DRAG_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);

struct SourceCaptionDragWriterGuard;

impl Drop for SourceCaptionDragWriterGuard {
    fn drop(&mut self) {
        SOURCE_CAPTION_DRAG_WRITE_LOCK.store(false, Ordering::Release);
    }
}

fn lock_source_caption_drag_writer() -> SourceCaptionDragWriterGuard {
    let mut spins = 0u32;
    loop {
        if SOURCE_CAPTION_DRAG_WRITE_LOCK
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
        {
            return SourceCaptionDragWriterGuard;
        }
        if spins < 64 {
            std::hint::spin_loop();
            spins += 1;
        } else {
            std::thread::yield_now();
        }
    }
}

fn try_lock_source_caption_drag_writer() -> Option<SourceCaptionDragWriterGuard> {
    SOURCE_CAPTION_DRAG_WRITE_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .ok()
        .map(|_| SourceCaptionDragWriterGuard)
}
static COVERED_NATIVE_HIT_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);
static MAIN_GUI_MOUSE_PASSTHROUGH: AtomicBool = AtomicBool::new(false);
static MAIN_GUI_HWND: AtomicIsize = AtomicIsize::new(0);
static MAIN_GUI_ROUTE_REPAIR_UNTIL_MS: AtomicU64 = AtomicU64::new(0);
static LAST_CURSOR_CONTRACT_OWNER: AtomicIsize = AtomicIsize::new(isize::MIN);
static DEBUG_LAST_EDGE_PLACE_X: AtomicI32 = AtomicI32::new(i32::MIN);
static DEBUG_LAST_EDGE_PLACE_Y: AtomicI32 = AtomicI32::new(i32::MIN);
static GHOST_ROUTE_SAMPLE_COUNT: AtomicU64 = AtomicU64::new(0);

fn state() -> &'static Arc<Mutex<State>> {
    STATE.get_or_init(|| Arc::new(Mutex::new(State::default())))
}

pub fn main_gui_mouse_passthrough() -> bool {
    MAIN_GUI_MOUSE_PASSTHROUGH.load(Ordering::Acquire)
}

/// Lock-free read used by the low-end GLSL responsiveness guard. True only
/// when the native cursor is currently owned by Neo's MAIN GUI window; the
/// floating panel and overlay do not trigger the interactive render pause.
pub fn main_gui_has_native_cursor_ownership() -> bool {
    let main = MAIN_GUI_HWND.load(Ordering::Acquire);
    main != 0 && NATIVE_GUI_OWNER.load(Ordering::Acquire) == main
}

pub fn source_caption_drag_active() -> bool {
    SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
}

pub fn source_caption_drag_epoch() -> u64 {
    SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire)
}

fn pack_drag_pair(x: i32, y: i32) -> i64 {
    (((y as u32) as i64) << 32) | ((x as u32) as i64)
}

fn unpack_drag_pair(packed: i64) -> (i32, i32) {
    (packed as i32, (packed >> 32) as i32)
}

/// Exact raw screen-space span for the currently owned client-only LEFT drag.
/// The owner is read before and after the packed coordinates so a release or
/// new gesture cannot mix samples from two generations.  The engine uses this
/// absolute span (not per-frame source HWND deltas) to move the visible overlay.
pub fn source_client_drag_raw_span(hwnd: isize) -> Option<((i32, i32), (i32, i32))> {
    if hwnd == 0 {
        return None;
    }
    let owner_before = SOURCE_CLIENT_DRAG_OWNER_HWND.load(Ordering::Acquire);
    if owner_before != hwnd {
        return None;
    }
    let origin = unpack_drag_pair(SOURCE_CLIENT_DRAG_RAW_ORIGIN.load(Ordering::Acquire));
    let current = unpack_drag_pair(SOURCE_CLIENT_DRAG_RAW_CURRENT.load(Ordering::Acquire));
    let owner_after = SOURCE_CLIENT_DRAG_OWNER_HWND.load(Ordering::Acquire);
    if owner_after == hwnd {
        Some((origin, current))
    } else {
        None
    }
}

fn source_caption_drag_visual_position() -> Option<(i32, i32)> {
    if !SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        || !SOURCE_CAPTION_DRAG_ANCHOR_VALID.load(Ordering::Acquire)
    {
        return None;
    }
    let epoch = SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire);
    if SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.load(Ordering::Acquire) != epoch {
        return None;
    }
    let (ox, oy) = unpack_drag_pair(SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN.load(Ordering::Acquire));
    // Re-check the generation after the coordinate load. A stale follower can
    // race a fast release/re-press, but it can never become visual authority
    // for a different drag generation.
    if SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.load(Ordering::Acquire) != epoch
        || SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire) != epoch
        || !SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
    {
        return None;
    }
    let (ax, ay) = unpack_drag_pair(SOURCE_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire));
    Some((ox.saturating_add(ax), oy.saturating_add(ay)))
}

/// Execute one high-rate overlay move as an epoch-scoped drag transaction.
/// The closure performs the Win32 move and returns the position Windows really
/// committed. Normal BUTTON-UP uses the same writer guard, therefore after it
/// returns no follower from this generation can perform a late window move.
/// Emergency stop/provider cancellation invalidates the epoch lock-free to avoid
/// any render-thread/cross-thread SetWindowPos dependency cycle.
pub fn commit_source_caption_drag_overlay_update<F>(epoch: u64, update: F) -> bool
where
    F: FnOnce() -> Option<(i32, i32)>,
{
    let Some(writer) = try_lock_source_caption_drag_writer() else {
        return false;
    };
    if !SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        || SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire) != epoch
        || !SOURCE_CAPTION_DRAG_ANCHOR_VALID.load(Ordering::Acquire)
    {
        return false;
    }
    let Some((x, y)) = update() else {
        return true;
    };
    // Stop/provider/geometry cancellation is intentionally lock-free so the
    // render thread can never deadlock against this follower's cross-thread
    // SetWindowPos. If such a cancellation happened while the closure ran, do
    // not publish its now-stale geometry or cursor target. Normal BUTTON-UP is
    // stronger: it takes this writer guard and therefore excludes the move.
    if !SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        || SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire) != epoch
        || !SOURCE_CAPTION_DRAG_ANCHOR_VALID.load(Ordering::Acquire)
    {
        return false;
    }
    SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN.store(pack_drag_pair(x, y), Ordering::Release);
    SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.store(epoch, Ordering::Release);
    let (ax, ay) = unpack_drag_pair(SOURCE_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire));
    let sprite = (x.saturating_add(ax), y.saturating_add(ay));
    set_sprite_caption_anchor_target(sprite.0, sprite.1);
    // Do not call SetWindowPos on the sprite while holding the writer guard.
    // The sprite HWND belongs to the LL-hook thread; BUTTON-UP on that thread
    // also takes this guard. A cross-thread SetWindowPos here while locked can
    // therefore form a lock/message-pump cycle. Publish geometry atomically,
    // release the guard, then perform the immediate visual commit.
    drop(writer);
    sprite_move_source_caption_writer_now(sprite.0, sprite.1, true);
    let visual_commit = SOURCE_CAPTION_DRAG_VISUAL_COMMIT_COUNT.fetch_add(1, Ordering::Relaxed);
    if visual_commit % 32 == 0 {
        log::debug!(
            "source-caption-drag-visual-commit: epoch={} overlay=({x},{y}) sprite=({},{}) writer=source-follower-direct",
            epoch,
            sprite.0,
            sprite.1
        );
    }
    // If BUTTON-UP/Stop changed the generation during that cross-thread window
    // call, immediately converge to the newest published target. This closes
    // the only late-follower race without making the input hook wait on a
    // render/follower-owned HWND operation.
    if !SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        || SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire) != epoch
    {
        // Do not let a late follower rewrite SPRITE_TARGET after BUTTON-UP.
        // Restore the HWND from the newest target published by the release path.
        sprite_restore_window_if_requested_now();
    }
    true
}

fn cancel_source_caption_drag_contract() {
    // Emergency/transition cancellation must never wait on the follower: that
    // follower can be inside a cross-thread SetWindowPos to the render-owned
    // overlay HWND. Invalidate the generation first; commit() re-checks it after
    // the Win32 move and refuses stale publication. Normal BUTTON-UP uses the
    // stronger finish_source_caption_drag_contract() writer transaction.
    let was_active = SOURCE_CAPTION_DRAG_ACTIVE.swap(false, Ordering::AcqRel);
    SOURCE_CAPTION_DRAG_ANCHOR_VALID.store(false, Ordering::Release);
    if was_active {
        SOURCE_CAPTION_DRAG_EPOCH.fetch_add(1, Ordering::AcqRel);
    }
}

/// Close a caption drag as one transaction, returning the generation, final
/// visual point and immutable grab anchor for diagnostics. The overlay origin
/// is sampled while the writer guard excludes the follower; after ACTIVE is
/// cleared, the follower can no longer move this generation's overlay.
fn finish_source_caption_drag_contract(g: &mut State) -> (u64, Option<(i32, i32)>, (i32, i32)) {
    let _writer = lock_source_caption_drag_writer();
    let epoch = SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire);
    let anchor = unpack_drag_pair(SOURCE_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire));
    let valid = SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        && SOURCE_CAPTION_DRAG_ANCHOR_VALID.load(Ordering::Acquire)
        && SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.load(Ordering::Acquire) == epoch;

    let overlay_hwnd = ACTIVE_OVERLAY_HWND.load(Ordering::Acquire);
    let origin = if overlay_hwnd != 0 {
        crate::platform::win32::window_rect(overlay_hwnd).map(|r| (r.0, r.1))
    } else {
        None
    }
    .unwrap_or_else(|| {
        unpack_drag_pair(SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN.load(Ordering::Acquire))
    });

    SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN.store(pack_drag_pair(origin.0, origin.1), Ordering::Release);
    SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.store(epoch, Ordering::Release);
    let visual = valid.then(|| {
        (
            origin.0.saturating_add(anchor.0),
            origin.1.saturating_add(anchor.1),
        )
    });
    if let Some((vx, vy)) = visual {
        g.content.x = origin.0;
        g.content.y = origin.1;
        g.virt = (vx as f64, vy as f64);
        set_sprite_caption_anchor_target(vx, vy);
    }

    let was_active = SOURCE_CAPTION_DRAG_ACTIVE.swap(false, Ordering::AcqRel);
    SOURCE_CAPTION_DRAG_ANCHOR_VALID.store(false, Ordering::Release);
    if was_active {
        SOURCE_CAPTION_DRAG_EPOCH.fetch_add(1, Ordering::AcqRel);
    }
    drop(_writer);
    if let Some((vx, vy)) = visual {
        // BUTTON-UP runs on the LL-hook/sprite owner thread. Commit the final
        // fixed-anchor point synchronously before ordinary mapping resumes; do
        // not leave the release position to a later coalesced WM_APP message.
        sprite_move_now(vx, vy, true);
        log::debug!(
            "source-caption-drag-release-visual-commit: epoch={epoch} sprite=({vx},{vy}) writer=button-up-final"
        );
    }

    // Re-sample the same full-window coordinate family used for input. This
    // makes the first post-drag move start from one coherent source/content
    // snapshot instead of waiting for a later render configure tick.
    if g.src_hwnd != 0 && g.window_frame_input {
        let live = match g.input_reference_kind {
            "dwm" => crate::platform::win32::extended_frame_bounds(g.src_hwnd),
            "outer" => crate::platform::win32::window_rect(g.src_hwnd),
            _ => crate::platform::win32::client_rect_on_screen(g.src_hwnd),
        };
        if let Some((x, y, w, h)) = live.filter(|r| r.2 > 0 && r.3 > 0) {
            g.src = Rect { x, y, w, h };
            let clip = clip_rect(g.src);
            unsafe {
                let _ = ClipCursor(Some(&clip));
            }
        }
    }
    g.expect_teleport = None;
    g.teleport_guard_until = None;
    (epoch, visual, anchor)
}

fn capture_sprite_active() -> bool {
    CAPTURE_SPRITE_ACTIVE.load(Ordering::Acquire) && !tensorrt_build_native_cursor_guard()
}

fn clear_capture_sprite_contract(_g: &mut State) {
    CAPTURE_SPRITE_ACTIVE.store(false, Ordering::Release);
    cancel_source_caption_drag_contract();
    end_native_gui_caption_drag("capture-contract-clear");
}

fn point_is_window_titlebar(hwnd: isize, px: i32, py: i32) -> bool {
    let Some((wx, wy, ww, wh)) = crate::platform::win32::window_rect(hwnd) else {
        return false;
    };
    let Some((cx, cy, _, _)) = crate::platform::win32::client_rect_on_screen(hwnd) else {
        return false;
    };
    if cy <= wy || ww <= 16 || wh <= 16 {
        return false;
    }
    let inset = ((cx - wx).abs().max(6)).min((ww / 8).max(6));
    px >= wx + inset && px < wx + ww - inset && py >= wy + 4 && py < cy
}

fn point_is_source_titlebar(hwnd: isize, px: i32, py: i32) -> bool {
    // Keep the conservative resize-border exclusion for the HIDDEN SOURCE.
    // Misclassifying a resize as a source-caption move would move the overlay.
    point_is_window_titlebar(hwnd, px, py)
}

fn gui_caption_band(
    hwnd: isize,
    px: i32,
    py: i32,
) -> Option<((i32, i32, i32, i32), (i32, i32, i32, i32), bool)> {
    let window = crate::platform::win32::window_rect(hwnd)?;
    let client = crate::platform::win32::client_rect_on_screen(hwnd)?;
    let (wx, wy, ww, wh) = window;
    let (_cx, cy, _cw, _ch) = client;
    if cy <= wy || ww <= 0 || wh <= 0 {
        return Some((window, client, false));
    }
    // GUI caption classification is diagnostic/gesture tracking only in v382;
    // visual ownership no longer changes based on this result. Cover the full
    // horizontal non-client caption band so the left icon/system-menu side does
    // not silently take a different cursor path. The SOURCE classifier above
    // remains conservative because it has geometry side effects.
    let system_buttons = crate::platform::win32::caption_system_button_cluster_width();
    let button_cluster_left = wx.saturating_add(ww).saturating_sub(system_buttons);
    let inside = px >= wx && px < button_cluster_left && py >= wy && py < cy;
    Some((window, client, inside))
}

fn begin_native_gui_caption_drag(hwnd: isize, px: i32, py: i32) -> bool {
    if hwnd == 0 {
        return false;
    }
    let Some((window, client, is_caption)) = gui_caption_band(hwnd, px, py) else {
        log::warn!(
            "native-gui-caption-classify: hwnd={hwnd:#x} cursor=({px},{py}) result=unavailable visual_owner=neo-sprite"
        );
        return false;
    };
    log::info!(
        "native-gui-caption-classify: hwnd={hwnd:#x} cursor=({px},{py}) window={window:?} client={client:?} result={} visual_owner=neo-sprite",
        if is_caption { "caption" } else { "non-caption" }
    );
    if !is_caption {
        return false;
    }
    let (wx, wy, _, _) = window;
    let anchor = (px.saturating_sub(wx), py.saturating_sub(wy));
    NATIVE_GUI_CAPTION_DRAG_ANCHOR.store(pack_drag_pair(anchor.0, anchor.1), Ordering::Release);
    NATIVE_GUI_CAPTION_DRAG_HWND.store(hwnd, Ordering::Release);
    NATIVE_GUI_CAPTION_DRAG_SAMPLE_COUNT.store(0, Ordering::Release);
    NATIVE_GUI_CAPTION_DRAG_ACTIVE.store(true, Ordering::Release);
    // v382 keeps capture-wide Neo sprite ownership even for a native GUI
    // caption drag. The real cursor stays at the exact Win32 input coordinate
    // for hit-testing/WM_NCLBUTTON semantics but remains Magnification-hidden.
    // This removes the unprovable native-reveal transition that could leave
    // both native and sprite visually absent on some title-bar grab positions.
    if external_native_cursor_owner(hwnd) {
        request_cursor_hidden(false);
    } else {
        request_cursor_hidden(true);
        sprite_move_now(px, py, true);
    }
    log::info!(
        "native-gui-caption-drag-begin: hwnd={hwnd:#x} cursor=({px},{py}) window=({wx},{wy}) anchor=({},{}) visual_owner={} raw-authority=true",
        anchor.0,
        anchor.1,
        if external_native_cursor_owner(hwnd) {
            "native"
        } else {
            "neo-sprite"
        }
    );
    true
}

fn native_gui_caption_drag_visual(hwnd: isize) -> Option<(i32, i32)> {
    if !NATIVE_GUI_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        || NATIVE_GUI_CAPTION_DRAG_HWND.load(Ordering::Acquire) != hwnd
    {
        return None;
    }
    let (wx, wy, _, _) = crate::platform::win32::window_rect(hwnd)?;
    let (ax, ay) = unpack_drag_pair(NATIVE_GUI_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire));
    Some((wx.saturating_add(ax), wy.saturating_add(ay)))
}

fn end_native_gui_caption_drag(reason: &str) {
    if NATIVE_GUI_CAPTION_DRAG_ACTIVE.swap(false, Ordering::AcqRel) {
        let hwnd = NATIVE_GUI_CAPTION_DRAG_HWND.swap(0, Ordering::AcqRel);
        let (ax, ay) = unpack_drag_pair(NATIVE_GUI_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire));
        log::info!(
            "native-gui-caption-drag-end: hwnd={hwnd:#x} anchor=({ax},{ay}) reason={reason}"
        );
    } else {
        NATIVE_GUI_CAPTION_DRAG_HWND.store(0, Ordering::Release);
    }
}

// ---------------- sprite cursor (layered arrow window) ----------------

/// Sprite box at 96dpi with the default cursor size. The REAL system cursor
/// scales with monitor DPI and the accessibility cursor-size setting; a fixed
/// A fixed 24px sprite can look undersized beside a 4K/150% system cursor (e.g.
/// on high-DPI displays, so the sprite box is computed
/// once at creation from DPI × CursorBaseSize.
const SPRITE_BASE_SIZE: i32 = 24;
static SPRITE_SIZE_PX: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(SPRITE_BASE_SIZE);

/// Match the system cursor scale: monitor/system DPI × the Windows 11
/// accessibility cursor size (HKCU\Control Panel\Cursors\CursorBaseSize,
/// 32 = default slider position).
fn compute_sprite_size() -> i32 {
    let dpi = unsafe { windows::Win32::UI::HiDpi::GetDpiForSystem() } as f32;
    let scale = dpi / 96.0;
    ((SPRITE_BASE_SIZE as f32 * scale).round() as i32).clamp(SPRITE_BASE_SIZE, 192)
}

fn arrow_pixels(size: i32) -> Vec<u32> {
    // classic arrow polygon in a 24x24 box (premultiplied BGRA), scaled to
    // `size`
    let poly: &[(f32, f32)] = &[
        (1.0, 1.0),
        (1.0, 17.0),
        (5.2, 13.4),
        (8.2, 20.4),
        (10.8, 19.3),
        (7.9, 12.5),
        (13.4, 12.2),
    ];
    let s = SPRITE_BASE_SIZE as f32 / size as f32;
    let inside = |x: f32, y: f32| -> bool {
        let (x, y) = (x * s, y * s);
        let mut c = false;
        let n = poly.len();
        for i in 0..n {
            let (x1, y1) = poly[i];
            let (x2, y2) = poly[(i + 1) % n];
            if ((y1 > y) != (y2 > y)) && (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1) {
                c = !c;
            }
        }
        c
    };
    // outline thickness scales with the sprite (1px at 24, ~2px at 48 …)
    let edge_r = (1.0 / s).round().max(1.0) as i32;
    let mut px = vec![0u32; (size * size) as usize];
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let v = if inside(fx, fy) {
                0xFFFFFFFFu32 // white fill
            } else {
                // black outline: any neighbour inside?
                let mut edge = false;
                'o: for dy in -edge_r..=edge_r {
                    for dx in -edge_r..=edge_r {
                        if inside(fx + dx as f32, fy + dy as f32) {
                            edge = true;
                            break 'o;
                        }
                    }
                }
                if edge { 0xFF000000u32 } else { 0 }
            };
            px[(y * size + x) as usize] = v;
        }
    }
    px
}

/// Coalesced sprite move (the cursor-routing design parity). The LL hook must return fast,
/// so instead of calling SetWindowPos (a z-order op) on EVERY mouse event —
/// which snags the whole system at high polling rates — the hook only stores
/// the target and posts ONE message; the hook thread's own message loop then
/// does the actual SetWindowPos, collapsing a burst of moves into a single
/// update. `SPRITE_TARGET` packs y<<32 | x (as u32s); `SPRITE_SHOW` the vis.
const WM_APP_SPRITE_MOVE: u32 = WM_APP + 1;
static SPRITE_TARGET: std::sync::atomic::AtomicI64 = std::sync::atomic::AtomicI64::new(0);
/// Desired sprite visibility. Actual visibility is additionally gated by
/// CURSOR_HIDE_APPLIED so the native cursor and sprite are never shown together.
static SPRITE_SHOW: AtomicBool = AtomicBool::new(false);
static SPRITE_MOVE_PENDING: AtomicBool = AtomicBool::new(false);
/// v348g: closes the last sprite/native race during a hidden -> visible
/// Magnification ownership transfer. While this gate is set, *every* sprite
/// show path (LL-hook move, coalesced WM_APP move, z-order reassert) is denied.
/// v348f only hid the layered window synchronously, so a mouse event arriving
/// in the few milliseconds before MagShowSystemCursor(true) completed could
/// re-show the white sprite at a slightly stale coordinate.
static SPRITE_NATIVE_REVEAL_BLOCK: AtomicBool = AtomicBool::new(false);
/// GUI/panel -> desktop visual bridge. While the native cursor is still hidden
/// by the Magnification owner thread, keep Neo's sprite at the CURRENT desktop
/// mouse point so there is never a cursor-less gap during the asynchronous reveal.
static DESKTOP_REVEAL_BRIDGE_ACTIVE: AtomicBool = AtomicBool::new(false);
/// TensorRT can lazily build a shape-specific engine while the GUI remains
/// interactive. During that epoch the Windows cursor must remain the sole
/// visual owner; otherwise moving onto Neo's GUI can re-enter sprite ownership
/// and strand the sprite at the last render-thread update for many seconds.
static TENSORRT_BUILD_NATIVE_CURSOR_GUARD: AtomicBool = AtomicBool::new(false);

fn tensorrt_build_native_cursor_guard() -> bool {
    TENSORRT_BUILD_NATIVE_CURSOR_GUARD.load(Ordering::Acquire)
}

/// While capture is active and Neo's own native GUI/panel owns the pointer,
/// keep native hit-testing at the real GUI coordinate but use Neo's sprite as
/// the visual cursor. This avoids relying on the Magnification cursor reveal
/// being visually committed even when Win32 reports a valid visible HCURSOR.
/// Read-only diagnostic/runtime signal for the eframe event loop.
/// True only while a physical left-button gesture that began on Neo's native
/// caption is actively moving the main GUI. This does not alter cursor/input
/// ownership; consumers may use it to avoid unnecessary GUI surface redraws.
pub fn native_gui_caption_drag_active() -> bool {
    NATIVE_GUI_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
}

fn native_gui_caption_drag_active_for_owner(owner: isize) -> bool {
    owner != 0
        && NATIVE_GUI_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire)
        && NATIVE_GUI_CAPTION_DRAG_HWND.load(Ordering::Acquire) == owner
}

fn external_native_cursor_owner(owner: isize) -> bool {
    owner != 0 && !crate::platform::win32::is_own_window(owner)
}

fn active_native_gui_sprite_mode(owner: isize) -> bool {
    owner != 0
        && !external_native_cursor_owner(owner)
        && ACTIVE_OVERLAY_HWND.load(Ordering::Acquire) != 0
        && !tensorrt_build_native_cursor_guard()
}

fn sprite_visibility_allowed(requested: bool, real_cursor_hidden: bool) -> bool {
    requested
        && real_cursor_hidden
        && !SPRITE_NATIVE_REVEAL_BLOCK.load(Ordering::Acquire)
        && !tensorrt_build_native_cursor_guard()
}

/// Timer that RE-ASSERTS the system-cursor hide while engaged (the cursor-routing design's
/// TIMER_CURSOR). A cached "already hidden" flag is not enough: the Magnification
/// hide can be reset by the compositor / capture / another app, and then the
/// REAL cursor — which sits at the SOURCE position, offset ~100px from the
/// sprite and appears as a second cursor or visible jump.
/// It is intermittent, matching the "sometimes smooth, sometimes returns"
/// report. MagShowSystemCursor(false) is cheap and idempotent, so we simply
/// re-assert it ~every 50ms on the hook thread (which owns the Mag API).
const CURSOR_REASSERT_TIMER_ID: usize = 1;

unsafe fn set_sprite_window_pos(h: HWND, x: i32, y: i32, show: bool) {
    unsafe {
        // Keep the logical cursor target untouched, but never place the
        // layered sprite so far beyond a physical monitor edge that its arrow
        // becomes completely invisible. This is especially important at the
        // bottom/right edge: the arrow shape starts one pixel inside its
        // top-left transparent border, so a 24px sprite positioned at y=1079
        // on a 1080p monitor has no visible arrow pixels at all.
        let (x, y) = sprite_window_visual_position((x, y));
        let flags = SWP_NOACTIVATE
            | SWP_NOSIZE
            | SWP_NOSENDCHANGING
            | SWP_NOOWNERZORDER
            | if show { SWP_SHOWWINDOW } else { SWP_HIDEWINDOW };
        let _ = SetWindowPos(h, Some(HWND_TOPMOST), x, y, 0, 0, flags);
    }
}

unsafe fn raise_sprite_window(h: HWND) {
    unsafe {
        // Only use the heavy two-step resort when another topmost window was
        // explicitly raised. Doing this on every mouse move makes topmost
        // siblings fight and can flicker; the cursor-routing design moves the sprite with a
        // plain HWND_TOPMOST call and reserves forced resorting for priorities.
        let flags = SWP_NOMOVE
            | SWP_NOSIZE
            | SWP_NOACTIVATE
            | SWP_NOSENDCHANGING
            | SWP_NOOWNERZORDER
            | SWP_SHOWWINDOW;
        let _ = SetWindowPos(h, Some(HWND_TOPMOST), 0, 0, 0, 0, flags);
        let _ = SetWindowPos(h, Some(HWND_TOP), 0, 0, 0, 0, flags);
    }
}

unsafe extern "system" fn sprite_proc(h: HWND, m: u32, w: WPARAM, l: LPARAM) -> LRESULT {
    unsafe {
        if m == WM_APP_SPRITE_MOVE {
            SPRITE_MOVE_PENDING.store(false, Ordering::Release);
            let packed = SPRITE_TARGET.load(Ordering::Acquire);
            let x = packed as i32;
            let y = (packed >> 32) as i32;
            let show = sprite_visibility_allowed(
                SPRITE_SHOW.load(Ordering::Acquire),
                CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
            );
            set_sprite_window_pos(h, x, y, show);
            return LRESULT(0);
        }
        if m == WM_TIMER && w.0 == CURSOR_EDGE_TRANSFER_TIMER_ID {
            let _ = KillTimer(Some(h), CURSOR_EDGE_TRANSFER_TIMER_ID);
            complete_capture_sprite_cursor_transfer_on_hook_thread();
            return LRESULT(0);
        }
        if m == WM_TIMER && w.0 == CURSOR_REASSERT_TIMER_ID {
            // Fallback in case the one-shot edge timer was coalesced/lost.
            // This remains independent of the render/ONNX thread.
            if capture_sprite_active() && cursor_reveal_pending() {
                complete_capture_sprite_cursor_transfer_on_hook_thread();
            }
            // The hide itself is driven by the render-engine thread; here we only
            // keep the desired-visibility flag in sync with the engaged state, so
            // a missed engage/disengage edge cannot strand the real cursor.
            let now = std::time::Instant::now();
            let Ok(mut g) = state().try_lock() else {
                // A busy engine/configure tick means "state temporarily
                // unknown", not "show the native cursor". Treating lock
                // contention as false toggled native↔sprite every 250ms while
                // the pointer was stationary over the topmost GUI.
                return LRESULT(0);
            };
            let stale_buttons = reconcile_stale_buttons(&mut g, now);
            // One visual owner for the whole active capture. Native stays
            // hidden on overlay, GUI/panel and desktop; Stop/provider guard
            // explicitly restores it.
            let state_wants_hidden = !tensorrt_build_native_cursor_guard()
                && g.active
                && !external_native_cursor_owner(g.native_gui_owner_hwnd);
            drop(g);
            if stale_buttons != 0 {
                // begin_ui_hold injects the DOWN that gives the native GUI full
                // drag ownership. If its physical UP was lost during a focus
                // handoff, Windows itself remains in a button-down state until
                // Stop. Send the matching UP here so GUI/source input recovers
                // without ending magnification.
                for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
                    if stale_buttons & bit != 0 {
                        inject_mouse_button(bit, false);
                    }
                }
                log::warn!("mouse-button-watchdog released stale bits={stale_buttons:#04x}");
            }
            let want_hidden =
                cursor_reassert_want_hidden(state_wants_hidden, cursor_reveal_pending());
            request_cursor_hidden(want_hidden);
            if state_wants_hidden
                && CURSOR_HIDE_APPLIED.load(Ordering::Acquire)
                && SPRITE_SHOW.load(Ordering::Acquire)
                && !SPRITE_NATIVE_REVEAL_BLOCK.load(Ordering::Acquire)
            {
                let visible = SPRITE_HWND
                    .get()
                    .is_some_and(|&hwnd| IsWindowVisible(HWND(hwnd as *mut _)).as_bool());
                if !visible {
                    sprite_restore_window_if_requested_now();
                    let packed = SPRITE_TARGET.load(Ordering::Acquire);
                    log::warn!(
                        "cursor-sprite-watchdog-restored: target=({},{}) hide_applied=true",
                        packed as i32,
                        (packed >> 32) as i32
                    );
                }
            }
            return LRESULT(0);
        }
        DefWindowProcW(h, m, w, l)
    }
}

unsafe fn create_sprite_window() -> Option<HWND> {
    unsafe {
        let instance = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok()?;
        let class: Vec<u16> = "NeoCursorSprite\0".encode_utf16().collect();
        let wc = WNDCLASSW {
            lpfnWndProc: Some(sprite_proc),
            hInstance: instance.into(),
            lpszClassName: windows::core::PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wc);
        let sprite_px = compute_sprite_size();
        SPRITE_SIZE_PX.store(sprite_px, Ordering::Relaxed);
        log::info!("cursor sprite size: {sprite_px}px (dpi/cursor-size scaled)");
        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
            windows::core::PCWSTR(class.as_ptr()),
            windows::core::PCWSTR(class.as_ptr()),
            WS_POPUP,
            0,
            0,
            sprite_px,
            sprite_px,
            None,
            None,
            Some(instance.into()),
            None,
        )
        .ok()?;

        // paint the ARGB arrow via UpdateLayeredWindow
        let screen_dc = GetDC(None);
        let mem_dc = CreateCompatibleDC(Some(screen_dc));
        let bi = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: sprite_px,
                biHeight: -sprite_px, // top-down
                biPlanes: 1,
                biBitCount: 32,
                biCompression: 0, // BI_RGB
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits: *mut std::ffi::c_void = std::ptr::null_mut();
        let dib = CreateDIBSection(Some(mem_dc), &bi, DIB_RGB_COLORS, &mut bits, None, 0).ok()?;
        let old = SelectObject(mem_dc, dib.into());
        let pixels = arrow_pixels(sprite_px);
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), bits as *mut u32, pixels.len());

        let mut pos = POINT { x: 0, y: 0 };
        let size = windows::Win32::Foundation::SIZE {
            cx: sprite_px,
            cy: sprite_px,
        };
        let src_pos = POINT { x: 0, y: 0 };
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER as u8,
            SourceConstantAlpha: 255,
            AlphaFormat: AC_SRC_ALPHA as u8,
            ..Default::default()
        };
        let _ = UpdateLayeredWindow(
            hwnd,
            Some(screen_dc),
            Some(&mut pos),
            Some(&size),
            Some(mem_dc),
            Some(&src_pos),
            windows::Win32::Foundation::COLORREF(0),
            Some(&blend),
            ULW_ALPHA,
        );
        SelectObject(mem_dc, old);
        let _ = DeleteObject(dib.into());
        let _ = DeleteDC(mem_dc);
        ReleaseDC(None, screen_dc);
        // 50ms timer to re-assert the system-cursor hide while engaged
        // (the cursor-routing design's TIMER_CURSOR — prevents the real cursor bleeding
        // through at the offset source position).
        let _ = SetTimer(Some(hwnd), CURSOR_REASSERT_TIMER_ID, 50, None);
        Some(hwnd)
    }
}

fn post_sprite_update() {
    if !SPRITE_MOVE_PENDING.swap(true, Ordering::AcqRel) {
        if let Some(&h) = SPRITE_HWND.get() {
            unsafe {
                let _ = PostMessageW(
                    Some(HWND(h as *mut _)),
                    WM_APP_SPRITE_MOVE,
                    WPARAM(0),
                    LPARAM(0),
                );
            }
        }
    }
}

fn note_suppressed_source_drag_sprite_write(kind: &str, x: i32, y: i32, show: bool) {
    let count = SOURCE_CAPTION_DRAG_SUPPRESSED_SPRITE_WRITES.fetch_add(1, Ordering::Relaxed) + 1;
    if count <= 4 || count % 32 == 0 {
        let expected = source_caption_drag_visual_position();
        log::warn!(
            "source-caption-drag-generic-sprite-write-suppressed: kind={kind} requested=({x},{y}) show={show} expected={expected:?} count={count}"
        );
    }
}

fn sprite_move_unchecked(x: i32, y: i32, show: bool) {
    SPRITE_TARGET.store(pack_drag_pair(x, y), Ordering::Release);
    SPRITE_SHOW.store(show, Ordering::Release);
    post_sprite_update();
}

fn sprite_move_now_unchecked(x: i32, y: i32, show: bool) {
    SPRITE_TARGET.store(pack_drag_pair(x, y), Ordering::Release);
    SPRITE_SHOW.store(show, Ordering::Release);
    if let Some(&h) = SPRITE_HWND.get() {
        unsafe {
            set_sprite_window_pos(
                HWND(h as *mut _),
                x,
                y,
                sprite_visibility_allowed(show, CURSOR_HIDE_APPLIED.load(Ordering::Acquire)),
            );
        }
    }
}

fn sprite_move(x: i32, y: i32, show: bool) {
    // v382 hard single-writer boundary. While a source-caption drag is active,
    // ONLY commit_source_caption_drag_overlay_update() may publish/move the
    // sprite. This blocks configure(), UI hover, LL-hook and any future caller
    // that forgets a local guard.
    if source_caption_drag_active() {
        note_suppressed_source_drag_sprite_write("coalesced", x, y, show);
        return;
    }
    sprite_move_unchecked(x, y, show);
}

fn sprite_move_now(x: i32, y: i32, show: bool) {
    if source_caption_drag_active() {
        note_suppressed_source_drag_sprite_write("immediate", x, y, show);
        return;
    }
    sprite_move_now_unchecked(x, y, show);
}

/// Privileged source-caption visual commit. The writer transaction already
/// published SPRITE_TARGET atomically with the overlay origin. This function
/// therefore moves only the HWND and deliberately does NOT rewrite the target
/// after releasing the writer lock. If BUTTON-UP wins that race, its newer
/// target remains authoritative and the stale follower can be converged below.
fn sprite_move_source_caption_writer_now(x: i32, y: i32, show: bool) {
    let Some(&h) = SPRITE_HWND.get() else {
        return;
    };
    unsafe {
        set_sprite_window_pos(
            HWND(h as *mut _),
            x,
            y,
            sprite_visibility_allowed(show, CURSOR_HIDE_APPLIED.load(Ordering::Acquire)),
        );
    }
}

/// Ordered caption-drag target publication. Geometry transactions update this
/// target while holding the drag writer guard. The follower may move the sprite
/// HWND after releasing the guard, but it never rewrites this atomic afterward;
/// BUTTON-UP can therefore publish a newer final target without being clobbered.
fn set_sprite_caption_anchor_target(x: i32, y: i32) {
    SPRITE_TARGET.store(pack_drag_pair(x, y), Ordering::Release);
    SPRITE_SHOW.store(true, Ordering::Release);
}

/// Keep the Neo cursor sprite physically above the magnified overlay.
///
/// On the AMD/DWM reproduction, WS_EX_TOPMOST plus SetWindowPos could report a
/// visible/topmost sprite while the WGL overlay was still the surface actually
/// scanned out above it. While GUI-topmost is OFF, use a USER32 owner chain as
/// the authoritative ordering contract:
///   cursor -> visible panel -> overlay
/// or, when the panel is hidden:
///   cursor -> overlay
/// When GUI-topmost is ON the owner is removed and the established sibling
/// stack is retained. This function never changes the requested cursor
/// visibility, so configured idle auto-hide remains authoritative.
pub fn enforce_cursor_overlay_priority(
    _gui_topmost: bool,
    _panel_hwnd: isize,
    _overlay_hwnd: isize,
) {
    // v235-compatible cursor model: the sprite is an independent TOPMOST
    // helper window. Never make it an owned popup of the panel/overlay: hiding
    // or reordering an owner can otherwise take the cursor with it.  Re-raise
    // it after the panel so the stable order is cursor > panel > overlay.
    if let Some(&sprite_hwnd) = SPRITE_HWND.get() {
        if crate::platform::win32::window_owner(sprite_hwnd) != 0 {
            crate::platform::win32::set_owned_popup_owner(sprite_hwnd, 0);
        }
    }
    keep_cursor_sprite_on_top();
}

/// Stop/failure boundary: never leave the cursor sprite owned by an overlay or
/// panel that is about to be hidden. The sprite itself is still hidden/revealed
/// by the existing cursor ownership state machine.
pub fn detach_cursor_sprite_owner() {
    let Some(&sprite_hwnd) = SPRITE_HWND.get() else {
        return;
    };
    crate::platform::win32::set_owned_popup_owner(sprite_hwnd, 0);
}

pub fn cursor_sprite_owner() -> isize {
    SPRITE_HWND
        .get()
        .map(|&hwnd| crate::platform::win32::window_owner(hwnd))
        .unwrap_or(0)
}

pub fn cursor_sprite_hwnd() -> isize {
    SPRITE_HWND.get().copied().unwrap_or(0)
}

/// Sample the desktop pixels under the layered cursor sprite and compare them
/// with Neo's own opaque black/white arrow template. This gives diagnostics a
/// direct "API visible, but pixels not found" signal instead of trusting only
/// IsWindowVisible/TOPMOST. The result is still a capture-side proxy; hardware
/// overlay/MPO scanout can be outside some desktop-readback APIs, so logs keep
/// the raw match ratio as well as the classification.
///
/// Returns (requested, api_visible, matched, sampled, average_rgb_delta).
pub fn cursor_sprite_screen_probe() -> (bool, bool, usize, usize, u32) {
    let requested = sprite_visibility_allowed(
        SPRITE_SHOW.load(Ordering::Acquire),
        CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
    );
    let Some(&raw_hwnd) = SPRITE_HWND.get() else {
        return (requested, false, 0, 0, 0);
    };
    let hwnd = HWND(raw_hwnd as *mut _);
    let api_visible = unsafe { IsWindowVisible(hwnd).as_bool() };
    if !requested || !api_visible {
        return (requested, api_visible, 0, 0, 0);
    }
    let Some((sx, sy, sw, sh)) = crate::platform::win32::window_rect(raw_hwnd) else {
        return (requested, api_visible, 0, 0, 0);
    };
    let size = SPRITE_SIZE_PX
        .load(Ordering::Relaxed)
        .max(1)
        .min(sw.max(sh).max(1));
    let expected = arrow_pixels(size);
    let mut opaque = Vec::new();
    for (idx, pixel) in expected.iter().copied().enumerate() {
        if (pixel >> 24) != 0 {
            opaque.push((idx, pixel));
        }
    }
    if opaque.is_empty() {
        return (requested, api_visible, 0, 0, 0);
    }
    let stride = (opaque.len() / 16).max(1);
    unsafe {
        let screen_dc = GetDC(None);
        if screen_dc.is_invalid() {
            return (requested, api_visible, 0, 0, 0);
        }
        let mut matched = 0usize;
        let mut sampled = 0usize;
        let mut delta_sum = 0u64;
        for (sample_idx, (idx, pixel)) in opaque.into_iter().enumerate() {
            if sample_idx % stride != 0 || sampled >= 20 {
                continue;
            }
            let x = (idx as i32) % size;
            let y = (idx as i32) / size;
            if x >= sw || y >= sh {
                continue;
            }
            let actual = GetPixel(screen_dc, sx + x, sy + y).0;
            if actual == 0xffff_ffff {
                continue;
            }
            // arrow_pixels is opaque black/white. COLORREF stores 0x00BBGGRR;
            // black and white are byte-order invariant, so the low 24 bits are
            // directly comparable.
            let expected_rgb = pixel & 0x00ff_ffff;
            let ar = actual & 0xff;
            let ag = (actual >> 8) & 0xff;
            let ab = (actual >> 16) & 0xff;
            let er = expected_rgb & 0xff;
            let eg = (expected_rgb >> 8) & 0xff;
            let eb = (expected_rgb >> 16) & 0xff;
            let delta = ar.abs_diff(er) + ag.abs_diff(eg) + ab.abs_diff(eb);
            if delta <= 48 {
                matched += 1;
            }
            sampled += 1;
            delta_sum += delta as u64;
        }
        let _ = ReleaseDC(None, screen_dc);
        let avg = if sampled == 0 {
            0
        } else {
            (delta_sum / sampled as u64) as u32
        };
        (requested, api_visible, matched, sampled, avg)
    }
}

pub fn keep_cursor_sprite_on_top() {
    if !sprite_visibility_allowed(
        SPRITE_SHOW.load(Ordering::Acquire),
        CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
    ) {
        return;
    }
    let Some(&h) = SPRITE_HWND.get() else {
        return;
    };
    let packed = SPRITE_TARGET.load(Ordering::Acquire);
    let x = packed as i32;
    let y = (packed >> 32) as i32;
    unsafe {
        let hwnd = HWND(h as *mut _);
        set_sprite_window_pos(hwnd, x, y, true);
        raise_sprite_window(hwnd);
    }
}

fn sprite_hide_window_only() {
    // Physically hide the sprite without changing the desired visibility.
    // This is used during sprite -> native cursor ownership transfer: the
    // sprite must leave the compositor BEFORE MagShowSystemCursor(true), but
    // if that native reveal fails we still need to restore the bridge sprite.
    if let Some(&h) = SPRITE_HWND.get() {
        unsafe {
            let _ = ShowWindow(HWND(h as *mut _), SW_HIDE);
        }
    }
}

fn sprite_restore_window_if_requested_now() {
    if !sprite_visibility_allowed(
        SPRITE_SHOW.load(Ordering::Acquire),
        CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
    ) {
        return;
    }
    let Some(&h) = SPRITE_HWND.get() else {
        return;
    };
    let packed = SPRITE_TARGET.load(Ordering::Acquire);
    let x = packed as i32;
    let y = (packed >> 32) as i32;
    unsafe {
        set_sprite_window_pos(HWND(h as *mut _), x, y, true);
    }
}

fn sprite_hide() {
    // hide immediately (disengage) and make sure a pending move cannot re-show it
    SPRITE_SHOW.store(false, Ordering::Release);
    sprite_hide_window_only();
}

// ---------------- low-level mouse hook ----------------

const LLMHF_INJECTED: u32 = 0x1;

fn button_transition(msg: u32) -> Option<(u8, bool)> {
    match msg {
        WM_LBUTTONDOWN => Some((BTN_LEFT, true)),
        WM_LBUTTONUP => Some((BTN_LEFT, false)),
        WM_RBUTTONDOWN => Some((BTN_RIGHT, true)),
        WM_RBUTTONUP => Some((BTN_RIGHT, false)),
        WM_MBUTTONDOWN => Some((BTN_MIDDLE, true)),
        WM_MBUTTONUP => Some((BTN_MIDDLE, false)),
        _ => None,
    }
}

fn publish_deferred_source_engaged(committed: bool) {
    DEFERRED_SOURCE_ENGAGED.store(committed, Ordering::Release);
}

fn clear_deferred_source_queue() {
    let w = DEFERRED_EVENT_WRITE.load(Ordering::Acquire);
    DEFERRED_EVENT_READ.store(w, Ordering::Release);
    DEFERRED_SOURCE_HELD_BITS.store(0, Ordering::Release);
    DEFERRED_SOURCE_LAST_TARGET.store(0, Ordering::Release);
    DEFERRED_SOURCE_OVERFLOW.store(false, Ordering::Release);
    let move_seq = DEFERRED_SOURCE_MOVE_SEQ.load(Ordering::Acquire);
    DEFERRED_SOURCE_MOVE_DRAINED_SEQ.store(move_seq, Ordering::Release);
}

fn publish_deferred_source_geometry(
    active: bool,
    client_only: bool,
    src_hwnd: isize,
    content: Rect,
    src: Rect,
    no_engage: &[NoEngageRect],
) {
    DEFERRED_SOURCE_CLIENT_ONLY.store(active && client_only, Ordering::Release);
    DEFERRED_SOURCE_HWND.store(if active { src_hwnd } else { 0 }, Ordering::Release);

    // Odd sequence = writer in progress, even = coherent snapshot.  There is
    // only one configure/geometry publisher, but the LL hook may sample at any
    // instruction boundary.
    DEFERRED_GEOMETRY_SEQ.fetch_add(1, Ordering::AcqRel);
    let (content_xy, content_wh, source_xy, source_wh) = if active {
        (
            pack_drag_pair(content.x, content.y),
            pack_drag_pair(content.w, content.h),
            pack_drag_pair(src.x, src.y),
            pack_drag_pair(src.w, src.h),
        )
    } else {
        (0, 0, 0, 0)
    };
    DEFERRED_CONTENT_XY.store(content_xy, Ordering::Relaxed);
    DEFERRED_CONTENT_WH.store(content_wh, Ordering::Relaxed);
    DEFERRED_SOURCE_XY.store(source_xy, Ordering::Relaxed);
    DEFERRED_SOURCE_WH.store(source_wh, Ordering::Relaxed);

    let mut n = 0usize;
    if active {
        for hit in no_engage {
            if n >= DEFERRED_OCCLUSION_CAP {
                break;
            }
            if hit.hwnd == 0
                || hit.panel
                || crate::platform::win32::is_own_window(hit.hwnd)
                || hit.land.w <= 0
                || hit.land.h <= 0
            {
                continue;
            }
            DEFERRED_OCCLUSION_X[n].store(hit.land.x, Ordering::Relaxed);
            DEFERRED_OCCLUSION_Y[n].store(hit.land.y, Ordering::Relaxed);
            DEFERRED_OCCLUSION_W[n].store(hit.land.w, Ordering::Relaxed);
            DEFERRED_OCCLUSION_H[n].store(hit.land.h, Ordering::Relaxed);
            n += 1;
        }
    }
    DEFERRED_OCCLUSION_COUNT.store(n as u32, Ordering::Relaxed);
    DEFERRED_GEOMETRY_SEQ.fetch_add(1, Ordering::Release);
    if !active || src_hwnd == 0 {
        publish_deferred_source_engaged(false);
    }
}

fn deferred_source_target_from_visual_lockfree(visual_x: i32, visual_y: i32) -> Option<(i32, i32)> {
    // Bounded seqlock read: never wait/spin in WH_MOUSE_LL.  A concurrent
    // geometry update merely lets this one edge use the ordinary safe fallback.
    for _ in 0..2 {
        let seq_before = DEFERRED_GEOMETRY_SEQ.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            continue;
        }
        let (cx, cy) = unpack_drag_pair(DEFERRED_CONTENT_XY.load(Ordering::Relaxed));
        let (cw, ch) = unpack_drag_pair(DEFERRED_CONTENT_WH.load(Ordering::Relaxed));
        let (sx, sy) = unpack_drag_pair(DEFERRED_SOURCE_XY.load(Ordering::Relaxed));
        let (sw, sh) = unpack_drag_pair(DEFERRED_SOURCE_WH.load(Ordering::Relaxed));
        let seq_after = DEFERRED_GEOMETRY_SEQ.load(Ordering::Acquire);
        if seq_before != seq_after || seq_after & 1 != 0 {
            continue;
        }
        if cw <= 1 || ch <= 1 || sw <= 1 || sh <= 1 {
            return None;
        }
        // Same pixel-centre mapping as map_content_to_source(), kept local so
        // the hook performs arithmetic + atomics only.
        let fx = ((visual_x as f64 - cx as f64 + 0.5) / cw as f64).clamp(0.0, 1.0);
        let fy = ((visual_y as f64 - cy as f64 + 0.5) / ch as f64).clamp(0.0, 1.0);
        let tx = (sx as f64 + fx * sw as f64 - 0.5).round() as i32;
        let ty = (sy as f64 + fy * sh as f64 - 0.5).round() as i32;
        return Some((
            tx.clamp(sx, sx.saturating_add(sw).saturating_sub(1)),
            ty.clamp(sy, sy.saturating_add(sh).saturating_sub(1)),
        ));
    }
    None
}

fn deferred_source_occluded_target_from_visual_lockfree(
    visual_x: i32,
    visual_y: i32,
) -> Option<(i32, i32)> {
    // Read geometry + external-window rectangles under one bounded seqlock
    // snapshot.  This avoids classifying a target with a rectangle list from a
    // different configure tick while keeping WH_MOUSE_LL strictly atomic-only.
    for _ in 0..2 {
        let seq_before = DEFERRED_GEOMETRY_SEQ.load(Ordering::Acquire);
        if seq_before & 1 != 0 {
            continue;
        }
        let (cx, cy) = unpack_drag_pair(DEFERRED_CONTENT_XY.load(Ordering::Relaxed));
        let (cw, ch) = unpack_drag_pair(DEFERRED_CONTENT_WH.load(Ordering::Relaxed));
        let (sx, sy) = unpack_drag_pair(DEFERRED_SOURCE_XY.load(Ordering::Relaxed));
        let (sw, sh) = unpack_drag_pair(DEFERRED_SOURCE_WH.load(Ordering::Relaxed));
        if cw <= 1 || ch <= 1 || sw <= 1 || sh <= 1 {
            return None;
        }
        let fx = ((visual_x as f64 - cx as f64 + 0.5) / cw as f64).clamp(0.0, 1.0);
        let fy = ((visual_y as f64 - cy as f64 + 0.5) / ch as f64).clamp(0.0, 1.0);
        let target_x = ((sx as f64 + fx * sw as f64 - 0.5).round() as i32)
            .clamp(sx, sx.saturating_add(sw).saturating_sub(1));
        let target_y = ((sy as f64 + fy * sh as f64 - 0.5).round() as i32)
            .clamp(sy, sy.saturating_add(sh).saturating_sub(1));

        let count =
            (DEFERRED_OCCLUSION_COUNT.load(Ordering::Relaxed) as usize).min(DEFERRED_OCCLUSION_CAP);
        let mut occluded = false;
        for i in 0..count {
            let x = DEFERRED_OCCLUSION_X[i].load(Ordering::Relaxed);
            let y = DEFERRED_OCCLUSION_Y[i].load(Ordering::Relaxed);
            let w = DEFERRED_OCCLUSION_W[i].load(Ordering::Relaxed);
            let h = DEFERRED_OCCLUSION_H[i].load(Ordering::Relaxed);
            if w <= 0 || h <= 0 {
                continue;
            }
            let target_inside = target_x >= x
                && target_x < x.saturating_add(w)
                && target_y >= y
                && target_y < y.saturating_add(h);
            if !target_inside {
                continue;
            }
            let visual_inside = visual_x >= x
                && visual_x < x.saturating_add(w)
                && visual_y >= y
                && visual_y < y.saturating_add(h);
            if !visual_inside {
                occluded = true;
                break;
            }
        }
        let seq_after = DEFERRED_GEOMETRY_SEQ.load(Ordering::Acquire);
        if seq_before != seq_after || seq_after & 1 != 0 {
            continue;
        }
        return occluded.then_some((target_x, target_y));
    }
    None
}

fn enqueue_deferred_source_event(msg: u32, px: i32, py: i32, vx: i32, vy: i32) -> bool {
    let write = DEFERRED_EVENT_WRITE.load(Ordering::Relaxed);
    let read = DEFERRED_EVENT_READ.load(Ordering::Acquire);
    if write.saturating_sub(read) >= DEFERRED_SOURCE_EVENT_CAP as u64 {
        DEFERRED_SOURCE_OVERFLOW.store(true, Ordering::Release);
        return false;
    }
    let slot = (write as usize) % DEFERRED_SOURCE_EVENT_CAP;
    DEFERRED_EVENT_POS[slot].store(pack_drag_pair(px, py), Ordering::Relaxed);
    DEFERRED_EVENT_VISUAL[slot].store(pack_drag_pair(vx, vy), Ordering::Relaxed);
    DEFERRED_EVENT_MSG[slot].store(msg, Ordering::Relaxed);
    DEFERRED_EVENT_WRITE.store(write.wrapping_add(1), Ordering::Release);
    DEFERRED_SOURCE_QUEUED_COUNT.fetch_add(1, Ordering::Relaxed);
    true
}

/// LL-hook safety gate for source-space button edges. This function is
/// intentionally atomic-only; it must stay safe even when the main State mutex
/// is busy or the render thread is stalled.
fn queue_occluded_source_button_edge_lockfree(msg: u32, _px: i32, _py: i32) -> bool {
    let Some((bit, down)) = button_transition(msg) else {
        return false;
    };
    let held = DEFERRED_SOURCE_HELD_BITS.load(Ordering::Acquire);
    let visual = unpack_drag_pair(SPRITE_TARGET.load(Ordering::Acquire));

    // Once a DOWN was swallowed by this gate, its matching UP belongs to the
    // same deferred source gesture regardless of any ownership transition in
    // between.  Re-map the CURRENT visual point; if geometry is being published
    // at this exact instant, fall back to the last coherent target from the
    // gesture instead of leaking the UP elsewhere.
    if !down && held & bit != 0 {
        let target = deferred_source_target_from_visual_lockfree(visual.0, visual.1)
            .unwrap_or_else(|| {
                unpack_drag_pair(DEFERRED_SOURCE_LAST_TARGET.load(Ordering::Acquire))
            });
        let queued = enqueue_deferred_source_event(msg, target.0, target.1, visual.0, visual.1);
        DEFERRED_SOURCE_HELD_BITS.fetch_and(!bit, Ordering::AcqRel);
        if !queued {
            DEFERRED_SOURCE_OVERFLOW.store(true, Ordering::Release);
        }
        return true;
    }

    if !down
        || !DEFERRED_SOURCE_ENGAGED.load(Ordering::Acquire)
        || !DEFERRED_SOURCE_CLIENT_ONLY.load(Ordering::Acquire)
        || DEFERRED_SOURCE_HWND.load(Ordering::Acquire) == 0
        || NATIVE_GUI_OWNER.load(Ordering::Acquire) != 0
    {
        return false;
    }

    let Some(target) = deferred_source_occluded_target_from_visual_lockfree(visual.0, visual.1)
    else {
        return false;
    };

    DEFERRED_SOURCE_LAST_TARGET.store(pack_drag_pair(target.0, target.1), Ordering::Release);
    let queued = enqueue_deferred_source_event(msg, target.0, target.1, visual.0, visual.1);
    DEFERRED_SOURCE_HELD_BITS.fetch_or(bit, Ordering::AcqRel);
    if !queued {
        DEFERRED_SOURCE_OVERFLOW.store(true, Ordering::Release);
    }
    // Safety-first: even queue overflow must not leak this physical DOWN to the
    // covering window. The engine-side overflow recovery releases bookkeeping.
    true
}

fn note_deferred_source_drag_move_lockfree(_px: i32, _py: i32) {
    if DEFERRED_SOURCE_HELD_BITS.load(Ordering::Acquire) == 0 {
        return;
    }
    let visual = unpack_drag_pair(SPRITE_TARGET.load(Ordering::Acquire));
    let Some(target) = deferred_source_target_from_visual_lockfree(visual.0, visual.1) else {
        return;
    };
    DEFERRED_SOURCE_LAST_TARGET.store(pack_drag_pair(target.0, target.1), Ordering::Release);
    DEFERRED_SOURCE_MOVE_POS.store(pack_drag_pair(target.0, target.1), Ordering::Relaxed);
    DEFERRED_SOURCE_MOVE_SEQ.fetch_add(1, Ordering::Release);
}

fn physical_button_bits() -> u8 {
    let mut bits = 0;
    unsafe {
        if GetAsyncKeyState(VK_LBUTTON.0 as i32) < 0 {
            bits |= BTN_LEFT;
        }
        if GetAsyncKeyState(VK_RBUTTON.0 as i32) < 0 {
            bits |= BTN_RIGHT;
        }
        if GetAsyncKeyState(VK_MBUTTON.0 as i32) < 0 {
            bits |= BTN_MIDDLE;
        }
    }
    bits
}

const STALE_BUTTON_WATCHDOG_MS: u64 = 180;
/// During the rare directly-forwarded source drag, cap posted WM_MOUSEMOVE
/// traffic so a 1000/8000Hz mouse cannot flood the source queue. Occlusion by
/// itself never enables synthetic movement; ordinary cursor motion remains on
/// the stable native path used before v637.
const SOURCE_DIRECT_MOVE_MIN_MS: u64 = 4;

fn reconcile_stale_buttons(g: &mut State, now: std::time::Instant) -> u8 {
    reconcile_stale_buttons_with_physical(g, now, physical_button_bits())
}

fn reconcile_stale_buttons_with_physical(
    g: &mut State,
    now: std::time::Instant,
    physical: u8,
) -> u8 {
    // A directly-forwarded source DOWN is intentionally swallowed in the LL hook.
    // On Windows that can leave GetAsyncKeyState() reporting the button as UP
    // even though the matching physical LL-hook UP has not arrived yet. Never let
    // the generic watchdog synthesize an early UP for those owned gestures; the
    // matching hook UP (or release_locked() on Stop/transition) closes them.
    let direct_owned = g.source_direct_bits;
    let tracked = (g.buttons_down & !direct_owned) | g.ui_hold_bits | g.native_ui_hold_bits;
    if tracked == 0
        || g.last_button_event.is_some_and(|edge| {
            now.saturating_duration_since(edge)
                < std::time::Duration::from_millis(STALE_BUTTON_WATCHDOG_MS)
        })
    {
        return 0;
    }
    let stale = tracked & !physical;
    if stale != 0 {
        g.buttons_down &= physical;
        if stale & BTN_LEFT != 0 {
            SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
        }
        g.ui_hold_bits &= physical;
        g.native_ui_hold_bits &= physical;
        g.swallow_up &= physical;
        if stale & BTN_LEFT != 0 && SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire) {
            let (epoch, visual, anchor) = finish_source_caption_drag_contract(g);
            log::warn!(
                "source-caption-drag stale-button recovery: physical_left=false action=end-classification epoch={} visual={visual:?} anchor=({},{})",
                epoch,
                anchor.0,
                anchor.1
            );
        }
        if stale & BTN_LEFT != 0 && NATIVE_GUI_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire) {
            end_native_gui_caption_drag("stale-button-watchdog");
            if g.active {
                if external_native_cursor_owner(g.native_gui_owner_hwnd) {
                    request_cursor_hidden(false);
                } else {
                    request_cursor_hidden(true);
                    sprite_move(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
                }
            }
        }
        g.last_button_event = Some(now);
    }
    stale
}

fn track_button_state(msg: u32, px: i32, py: i32) {
    let Some((bit, down)) = button_transition(msg) else {
        return;
    };
    // Reset LEFT-drag provenance before taking the state lock. Even if the LL
    // hook loses this edge to transient mutex contention, a stale source owner
    // must never survive into a later GUI/desktop hold. A successful source
    // DOWN re-arms the exact HWND below after classification.
    if bit == BTN_LEFT {
        SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    }
    // A click can be the first physical input after Stop. Repair the native
    // route before CallNextHookEx delivers this very button edge so the user
    // does not need an extra mouse move to unstick the GUI.
    repair_main_gui_passthrough_if_needed();
    let Ok(mut g) = state().try_lock() else {
        return;
    };
    let now = std::time::Instant::now();
    if bit == BTN_RIGHT && crate::logging::diagnostics_enabled() {
        log::info!(
            "physical-right-button: edge={} raw=({},{}) active={} engaged={} src={:#x} native_gui={:#x} sprite_hwnd={:#x}",
            if down { "down" } else { "up" },
            px,
            py,
            g.active,
            g.engaged,
            g.src_hwnd,
            MAIN_GUI_HWND.load(Ordering::Acquire),
            cursor_sprite_hwnd(),
        );
    }
    // v421: a provider/session transition means Neo cursor mapping is explicitly
    // suspended. Do not start native-GUI caption/sprite ownership or mutate
    // Neo button bookkeeping during this epoch. The low-level hook will still
    // pass the physical edge to Windows/eframe, so presets, mode buttons and
    // the title bar remain ordinary native GUI controls while the provider is
    // preparing. set_transition_suspended(true) already cleared any previous
    // Neo-held button/cursor state before entering this branch.
    if g.transition_suspended {
        return;
    }
    if down {
        g.buttons_down |= bit;
        if bit == BTN_LEFT {
            let client_source_owner = if g.active
                && g.engaged
                && !g.window_frame_input
                && g.pending_engage.is_none()
                && g.src_hwnd != 0
            {
                g.src_hwnd
            } else {
                0
            };
            if client_source_owner != 0 {
                let raw = pack_drag_pair(px, py);
                // Publish the generation's coordinates first and ownership
                // last. Readers that observe this HWND are therefore guaranteed
                // to see the matching raw origin/current pair.
                SOURCE_CLIENT_DRAG_RAW_ORIGIN.store(raw, Ordering::Release);
                SOURCE_CLIENT_DRAG_RAW_CURRENT.store(raw, Ordering::Release);
                SOURCE_CLIENT_DRAG_OWNER_HWND.store(client_source_owner, Ordering::Release);
                log::debug!(
                    "source-client-drag-owner-armed: hwnd={:#x} source_pt=({px},{py}) raw_anchor=armed",
                    client_source_owner
                );
            } else {
                SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
            }
        }
        if bit == BTN_LEFT
            && g.active
            && g.engaged
            && g.window_frame_input
            && g.src_hwnd != 0
            && point_is_source_titlebar(g.src_hwnd, px, py)
        {
            if SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire) {
                log::warn!(
                    "source-caption-drag duplicate-down ignored: hwnd={:#x} source_pt=({px},{py}) epoch={}",
                    g.src_hwnd,
                    SOURCE_CAPTION_DRAG_EPOCH.load(Ordering::Acquire)
                );
            } else {
                let visual = if SPRITE_SHOW.load(Ordering::Acquire)
                    && CURSOR_HIDE_APPLIED.load(Ordering::Acquire)
                {
                    unpack_drag_pair(SPRITE_TARGET.load(Ordering::Acquire))
                } else {
                    (g.virt.0.round() as i32, g.virt.1.round() as i32)
                };
                let overlay_hwnd = ACTIVE_OVERLAY_HWND.load(Ordering::Acquire);
                let (ox, oy) = if overlay_hwnd != 0 {
                    crate::platform::win32::window_rect(overlay_hwnd)
                        .map(|r| (r.0, r.1))
                        .unwrap_or((g.content.x, g.content.y))
                } else {
                    (g.content.x, g.content.y)
                };
                let _writer = lock_source_caption_drag_writer();
                let epoch = SOURCE_CAPTION_DRAG_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
                SOURCE_CAPTION_DRAG_ANCHOR.store(
                    pack_drag_pair(visual.0.saturating_sub(ox), visual.1.saturating_sub(oy)),
                    Ordering::Release,
                );
                SOURCE_CAPTION_DRAG_OVERLAY_ORIGIN.store(pack_drag_pair(ox, oy), Ordering::Release);
                SOURCE_CAPTION_DRAG_OVERLAY_EPOCH.store(epoch, Ordering::Release);
                SOURCE_CAPTION_DRAG_ANCHOR_VALID.store(true, Ordering::Release);
                // A physical button edge on the mapped source title bar is
                // authoritative current input. Any pre-drag teleport quarantine
                // belongs to the previous ownership transition and must not
                // survive into this native caption gesture.
                g.expect_teleport = None;
                g.teleport_guard_until = None;
                g.last_hw = (px, py);
                SOURCE_CAPTION_DRAG_VISUAL_COMMIT_COUNT.store(0, Ordering::Release);
                SOURCE_CAPTION_DRAG_SUPPRESSED_SPRITE_WRITES.store(0, Ordering::Release);
                SOURCE_CAPTION_DRAG_ACTIVE.store(true, Ordering::Release);
                drop(_writer);
                log::info!(
                    "source-caption-drag-begin: hwnd={:#x} source_pt=({px},{py}) visual=({},{}) anchor=({},{}) overlay=({ox},{oy}) epoch={} owner=anchored-engine-follow",
                    g.src_hwnd,
                    visual.0,
                    visual.1,
                    visual.0.saturating_sub(ox),
                    visual.1.saturating_sub(oy),
                    epoch
                );
            }
        }
        // Re-sample the real HWND before classifying the gesture. The render
        // thread can be hundreds of milliseconds behind under a heavy GLSL
        // chain, but native window dragging must never depend on that cadence.
        refresh_live_own_no_engage_geometry(&mut g, now);
        g.native_ui_hold_bits &= !bit;
        let native_gui_hit = if !g.engaged {
            hit_ui_for_cursor_ownership(&g.no_engage, px, py)
        } else {
            None
        };
        if let Some(hit) = native_gui_hit {
            g.native_ui_hold_bits |= bit;
            g.native_gui_owner_hwnd = hit.hwnd;
            NATIVE_GUI_OWNER.store(hit.hwnd, Ordering::Release);
            if bit == BTN_LEFT {
                NATIVE_GUI_CAPTION_DRAG_SAMPLE_COUNT.store(0, Ordering::Release);
                end_native_gui_caption_drag("new-left-down-reset");
                let _ = begin_native_gui_caption_drag(hit.hwnd, px, py);
            }
            if g.active {
                if external_native_cursor_owner(hit.hwnd) {
                    request_cursor_hidden(false);
                } else {
                    // v382 behavior remains unchanged for Neo GUI/panel.
                    request_cursor_hidden(true);
                    sprite_move(px, py, true);
                    keep_cursor_sprite_on_top();
                }
            }
            log::info!(
                "native-gui-gesture-begin: bit={bit:#04x} hwnd={:#x} pos=({px},{py}) rect={:?}",
                hit.hwnd,
                hit.land
            );
        }
    } else {
        g.buttons_down &= !bit;
        if bit == BTN_LEFT && SOURCE_CAPTION_DRAG_ACTIVE.load(Ordering::Acquire) {
            let (epoch, visual, anchor) = finish_source_caption_drag_contract(&mut g);
            let visual = visual.unwrap_or((g.virt.0.round() as i32, g.virt.1.round() as i32));
            let actual = read_cursor_pos_or((px, py), "caption drag release");
            g.last_hw = (px, py);
            g.last_set = actual;
            let suppressed = SOURCE_CAPTION_DRAG_SUPPRESSED_SPRITE_WRITES.load(Ordering::Acquire);
            log::info!(
                "source-caption-drag-end: hwnd={:#x} source_pt=({px},{py}) actual=({},{}) visual=({},{}) anchor=({},{}) epoch={} owner=anchored-engine-follow suppressed_generic_sprite_writes={suppressed}",
                g.src_hwnd,
                actual.0,
                actual.1,
                visual.0,
                visual.1,
                anchor.0,
                anchor.1,
                epoch
            );
        }
        let was_native_ui = g.native_ui_hold_bits & bit != 0;
        g.native_ui_hold_bits &= !bit;
        if was_native_ui && g.native_ui_hold_bits == 0 {
            if bit == BTN_LEFT {
                end_native_gui_caption_drag("button-up");
            }
            if g.active && g.native_gui_owner_hwnd != 0 {
                if external_native_cursor_owner(g.native_gui_owner_hwnd) {
                    request_cursor_hidden(false);
                } else {
                    // v382 behavior remains unchanged for Neo GUI/panel.
                    request_cursor_hidden(true);
                    sprite_move_now(px, py, true);
                }
            }
            // The next move re-samples the live HWND rectangle. A time gate here
            // makes ownership crossing-speed dependent.
            log::info!(
                "native-gui-gesture-end: bit={bit:#04x} pos=({px},{py}) reengage_grace_ms=0 exact_owner=true"
            );
        }
    }
    g.last_button_event = Some(now);
}

fn inject_mouse_button(bit: u8, down: bool) {
    let flag = match (bit, down) {
        (BTN_LEFT, true) => MOUSEEVENTF_LEFTDOWN,
        (BTN_LEFT, false) => MOUSEEVENTF_LEFTUP,
        (BTN_RIGHT, true) => MOUSEEVENTF_RIGHTDOWN,
        (BTN_RIGHT, false) => MOUSEEVENTF_RIGHTUP,
        (BTN_MIDDLE, true) => MOUSEEVENTF_MIDDLEDOWN,
        (BTN_MIDDLE, false) => MOUSEEVENTF_MIDDLEUP,
        _ => return,
    };
    let mut input = INPUT {
        r#type: INPUT_MOUSE,
        ..Default::default()
    };
    input.Anonymous.mi.dwFlags = flag;
    unsafe {
        SendInput(&[input], std::mem::size_of::<INPUT>() as i32);
    }
}

/// the cursor-routing design panel model: while engaged the real cursor is confined to the
/// (small/hidden) source and can never physically reach the control panel or
/// top-most GUI. Instead the VIRTUAL cursor travels over the panel, and a
/// press there is turned into a synthetic full click AT the panel's on-screen
/// position, then the real cursor is restored to the source and re-clipped.
/// The matching hardware button-up is swallowed too. Returns true if the event
/// was consumed (the hook must then swallow it by returning non-zero).
///
/// This NEVER disengages, so it cannot oscillate or fling the cursor — the
/// failure mode caused by a stale no-engage-zone handoff.
/// Native hold: send only the DOWN at the visual UI point, then let the
/// user's real drag/move/UP continue natively on that GUI.
fn begin_ui_hold(g: &mut State, hover: UiHover, bit: u8) -> bool {
    let click = hover.pos;
    let Some(actual) = release_windows_to_native_at(g, click, "native UI click handoff", true)
    else {
        // The visible cursor was over native UI, so never let the physical DOWN
        // fall through to the hidden source when the ownership warp could not
        // be verified. Swallow the matching UP and retry on the next gesture.
        g.swallow_up |= bit;
        return false;
    };
    g.engaged = false;
    publish_deferred_source_engaged(false);
    g.edge_out_accum = 0.0;
    g.last_engage_at = None;
    g.teleport_guard_until = None;
    g.must_leave_content = false;
    g.ui_hold_bits |= bit;
    g.last_ui_hover = Some(hover);
    if hover.hit.hwnd != 0
        && !is_panel_hit(hover.hit)
        && crate::platform::win32::is_own_window(hover.hit.hwnd)
    {
        g.native_gui_owner_hwnd = hover.hit.hwnd;
        NATIVE_GUI_OWNER.store(hover.hit.hwnd, Ordering::Release);
    }
    g.virt = (actual.0 as f64, actual.1 as f64);
    g.last_set = actual;
    g.last_hw = actual;
    if external_native_cursor_owner(hover.hit.hwnd) {
        request_cursor_hidden(false);
    }
    inject_mouse_button(bit, true);
    true
}

fn handoff_to_gui(g: &mut State, hover: UiHover, _now: std::time::Instant) -> bool {
    let pos = hover.pos;
    let source_origin = g.last_hw;
    let native_own_gui = !is_panel_hit(hover.hit)
        && hover.hit.hwnd != 0
        && crate::platform::win32::is_own_window(hover.hit.hwnd);
    if native_own_gui {
        // Restore normal hit-testing before placing the native cursor here.
        publish_main_gui_passthrough(hover.hit.hwnd, false, "native-gui-handoff");
    }
    // Crossing from sprite ownership to native UI is a read-only ownership
    // transfer. The dedicated GUI/panel z-order manager decides topmost policy;
    // the input classifier must never mutate Z-order while deciding ownership.
    let Some(actual) = release_windows_to_native_at(g, pos, "native GUI hover handoff", true)
    else {
        if native_own_gui {
            publish_main_gui_passthrough(hover.hit.hwnd, true, "native-gui-handoff-failed");
        }
        g.ui_hover_active = true;
        g.hidden_by_idle = false;
        g.last_ui_hover = Some(hover);
        sprite_move_now(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
        keep_cursor_sprite_on_top();
        return false;
    };
    g.engaged = false;
    publish_deferred_source_engaged(false);
    g.edge_out_accum = 0.0;
    g.last_engage_at = None;
    g.teleport_guard_until = None;
    g.must_leave_content = false;
    g.ui_hold_bits = 0;
    g.ui_hover_active = true;
    g.hidden_by_idle = false;
    // No time-based ownership grace here. While the real cursor is inside the
    // exact live GUI rectangle plan_engage() rejects source ownership; the first
    // pixel outside can re-engage immediately. This removes the old fuzzy
    // 120ms/20px boundary while remaining stable.
    g.native_gui_owner_hwnd = hover.hit.hwnd;
    NATIVE_GUI_OWNER.store(hover.hit.hwnd, Ordering::Release);
    g.last_ui_hover = Some(UiHover {
        hit: hover.hit,
        pos: actual,
    });
    g.virt = (actual.0 as f64, actual.1 as f64);
    g.last_set = actual;
    g.last_hw = actual;
    g.native_gui_settle = Some(NativeGuiSettle {
        target: actual,
        source_origin,
        until: std::time::Instant::now() + std::time::Duration::from_millis(45),
        reasserted: false,
    });
    if external_native_cursor_owner(hover.hit.hwnd) {
        request_cursor_hidden(false);
    } else {
        // Neo GUI/panel keeps the established capture-wide sprite behavior.
        keep_cursor_sprite_on_top();
    }
    log::debug!(
        "native-ui-ownership-enter: hwnd={:#x} boundary=exact zorder=win32 pos=({},{})",
        hover.hit.hwnd,
        actual.0,
        actual.1
    );
    true
}

/// When the mapper is disengaged, native UI ownership is decided by the
/// CURRENT visible screen point and WindowFromPoint/GA_ROOT. No timer, geometry
/// hysteresis, list-order priority, or input-side Z-order mutation participates.
fn hold_native_gui_ownership_if_inside(
    g: &mut State,
    px: i32,
    py: i32,
    now: std::time::Instant,
) -> bool {
    // A physical gesture that STARTED on a native window follows normal Win32
    // capture semantics until button-up. This is the only ownership latch: it
    // is gesture-based, not a hidden spatial/time band, so ordinary hover can
    // always pass straight through GUI/panel boundaries.
    if g.native_ui_hold_bits != 0 && g.native_gui_owner_hwnd != 0 {
        let pos = (px, py);
        g.virt = (px as f64, py as f64);
        g.last_set = pos;
        g.last_hw = pos;
        if g.active {
            let external_native_owner = external_native_cursor_owner(g.native_gui_owner_hwnd);
            if external_native_owner {
                request_cursor_hidden(false);
            } else {
                // v382 behavior remains unchanged for Neo GUI/panel.
                request_cursor_hidden(true);
                sprite_move_now(px, py, true);
            }
            if !external_native_owner && g.native_ui_hold_bits & BTN_LEFT != 0 {
                let sample = NATIVE_GUI_CAPTION_DRAG_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed);
                if sample % 16 == 0 {
                    let caption_tracked =
                        native_gui_caption_drag_active_for_owner(g.native_gui_owner_hwnd);
                    let anchored =
                        native_gui_caption_drag_visual(g.native_gui_owner_hwnd).unwrap_or((px, py));
                    let (ax, ay) = if caption_tracked {
                        unpack_drag_pair(NATIVE_GUI_CAPTION_DRAG_ANCHOR.load(Ordering::Acquire))
                    } else {
                        (i32::MIN, i32::MIN)
                    };
                    let target = unpack_drag_pair(SPRITE_TARGET.load(Ordering::Acquire));
                    let visible = SPRITE_HWND.get().is_some_and(|&hwnd| unsafe {
                        IsWindowVisible(HWND(hwnd as *mut _)).as_bool()
                    });
                    log::debug!(
                        "native-gui-left-hold-sample: hwnd={:#x} raw=({px},{py}) caption_tracked={} sprite_target=({},{}) sprite_visible={} hide_applied={} want_hidden={} window_anchor=({},{}) anchor=({ax},{ay}) raw_anchor_delta=({},{}) visual_owner=neo-sprite",
                        g.native_gui_owner_hwnd,
                        caption_tracked,
                        target.0,
                        target.1,
                        visible,
                        CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
                        WANT_CURSOR_HIDDEN.load(Ordering::Acquire),
                        anchored.0,
                        anchored.1,
                        px - anchored.0,
                        py - anchored.1
                    );
                }
            }
        }
        return true;
    }

    // The main GUI is intentionally hit-test transparent while the visible
    // cursor is outside it. Therefore WindowFromPoint cannot be used to enter
    // it again; its exact live HWND rectangle is authoritative for this one
    // owned window, and the first pixel inside restores native hit-testing.
    let hit = own_main_gui_at_visible_point_for_ownership(&g.no_engage, px, py)
        .or_else(|| hit_ui_for_cursor_ownership(&g.no_engage, px, py));
    let new_owner = hit.map_or(0, |h| h.hwnd);
    let old_owner = g.native_gui_owner_hwnd;

    if new_owner == 0 {
        let content_hit = g.content.contains(px, py);
        let hide_applied = CURSOR_HIDE_APPLIED.load(Ordering::Acquire);

        // Publish ownership exit immediately. The render/Mag thread reads this
        // atomic without taking the input mutex, so native reveal can proceed
        // in parallel even if the following GUI style mutation is slow.
        g.native_gui_owner_hwnd = 0;
        g.last_gui_sprite_sync_at = None;
        NATIVE_GUI_OWNER.store(0, Ordering::Release);

        // Active capture keeps a single visual cursor owner on every surface.
        // The real cursor remains at this exact desktop/content coordinate for
        // normal Windows hit-testing, but stays Magnification-hidden; the Neo
        // sprite is therefore deterministic even if Windows' native visual
        // cursor state cannot be queried reliably.
        if g.active && capture_sprite_active() {
            g.virt = (px as f64, py as f64);
            g.last_set = (px, py);
            g.last_hw = (px, py);
            DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
            request_cursor_hidden(true);
            sprite_move_now(px, py, true);
            keep_cursor_sprite_on_top();

            if old_owner != 0 {
                log::debug!(
                    "native-ui-ownership-exit: hwnd={:#x} boundary=exact pos=({px},{py})",
                    old_owner
                );
                if crate::platform::win32::is_own_window(old_owner) {
                    // Do NOT flip the whole GUI to WS_EX_TRANSPARENT merely
                    // because the visible cursor left its rectangle. That
                    // FRAMECHANGED path costs tens of milliseconds and is the
                    // hitch felt when GUI and overlay overlap. Keep the GUI
                    // interactive; apply click-through lazily only if the
                    // *hidden mapped source cursor* actually lands under it.
                    if MAIN_GUI_MOUSE_PASSTHROUGH.load(Ordering::Acquire) {
                        log::debug!(
                            "native-ui-exit keeps existing overlap shield: hwnd={:#x} desktop_owner=sprite",
                            old_owner
                        );
                    }
                }
            }

            if !content_hit
                && g.last_desktop_cursor_diag_at.map_or(true, |last| {
                    now.saturating_duration_since(last) >= std::time::Duration::from_millis(100)
                })
            {
                g.last_desktop_cursor_diag_at = Some(now);
                let sprite_requested = SPRITE_SHOW.load(Ordering::Acquire);
                let sprite_blocked = SPRITE_NATIVE_REVEAL_BLOCK.load(Ordering::Acquire);
                let sprite_window_visible = SPRITE_HWND
                    .get()
                    .is_some_and(|&h| unsafe { IsWindowVisible(HWND(h as *mut _)).as_bool() });
                let sprite_rect = SPRITE_HWND
                    .get()
                    .and_then(|&h| crate::platform::win32::window_rect(h));
                let top = unsafe { WindowFromPoint(POINT { x: px, y: py }) };
                log::info!(
                    "desktop-cursor-heartbeat: pos=({px},{py}) visual_owner=sprite want_hidden={} hide_applied={} sprite_requested={} sprite_blocked={} sprite_window_visible={} sprite_rect={:?} top_hwnd={:#x}",
                    WANT_CURSOR_HIDDEN.load(Ordering::Acquire),
                    CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
                    sprite_requested,
                    sprite_blocked,
                    sprite_window_visible,
                    sprite_rect,
                    top.0 as isize
                );
            }
            g.ui_hover_active = false;
            g.last_ui_hover = None;
            g.last_ui_post = None;
            return false;
        }

        // IMPORTANT: own-main-GUI passthrough mutation can take tens of
        // milliseconds on some systems. Start the desktop visual bridge and
        // publish native reveal BEFORE that potentially blocking Win32 route.
        // Otherwise the cursor has already crossed the GUI boundary but both
        // the native reveal request and the sprite update wait behind the GUI
        // style transition, producing the periodic 40-70 ms invisible gap.
        if g.active && old_owner != 0 && !content_hit && hide_applied {
            let bridge_started = !DESKTOP_REVEAL_BRIDGE_ACTIVE.swap(true, Ordering::AcqRel);
            sprite_move_now(px, py, true);
            keep_cursor_sprite_on_top();
            request_cursor_hidden(false);
            if bridge_started {
                log::info!(
                    "desktop-reveal-bridge-start: from_hwnd={:#x} pos=({px},{py}) hide_applied=true phase=pre-gui-route",
                    old_owner
                );
            }
        }

        if old_owner != 0 {
            log::debug!(
                "native-ui-ownership-exit: hwnd={:#x} boundary=exact pos=({px},{py})",
                old_owner
            );
            if crate::platform::win32::is_own_window(old_owner) {
                let route_started = std::time::Instant::now();
                set_own_main_gui_passthrough(g, true, "exact-gui-exit");
                let route_ms = route_started.elapsed().as_millis();
                if route_ms >= 8 {
                    log::info!(
                        "native-ui-exit-route-latency: hwnd={:#x} elapsed_ms={} desktop_bridge={}",
                        old_owner,
                        route_ms,
                        DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire)
                    );
                }
            }
        }
        if g.active {
            // The Mag thread may have completed native reveal while the GUI
            // passthrough call above was blocked. Re-read APPLIED here; never
            // resurrect the sprite bridge after native has already become the
            // visual owner.
            let hide_applied_now = CURSOR_HIDE_APPLIED.load(Ordering::Acquire);
            // After the first exit sample, keep the bridge sprite following
            // every desktop LL event until the Mag owner reports successful
            // native reveal. No hidden margin or time latch is introduced.
            if !content_hit && hide_applied_now {
                let bridge_started = !DESKTOP_REVEAL_BRIDGE_ACTIVE.swap(true, Ordering::AcqRel);
                sprite_move_now(px, py, true);
                keep_cursor_sprite_on_top();
                if bridge_started {
                    log::info!(
                        "desktop-reveal-bridge-start: from_hwnd={:#x} pos=({px},{py}) hide_applied=true phase=desktop-follow",
                        old_owner
                    );
                }
            } else {
                DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
            }
            log::info!(
                "native-ui-exit-visibility: hwnd={:#x} actual=({px},{py}) content_hit={} hide_applied={} desktop_bridge={} action=request-native",
                old_owner,
                content_hit,
                hide_applied_now,
                DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire)
            );
            request_cursor_hidden(false);

            if !content_hit
                && !hide_applied_now
                && g.last_desktop_cursor_diag_at.map_or(true, |last| {
                    now.saturating_duration_since(last) >= std::time::Duration::from_millis(250)
                })
            {
                g.last_desktop_cursor_diag_at = Some(now);
                let sprite_requested = SPRITE_SHOW.load(Ordering::Acquire);
                let sprite_blocked = SPRITE_NATIVE_REVEAL_BLOCK.load(Ordering::Acquire);
                let sprite_window_visible = SPRITE_HWND
                    .get()
                    .is_some_and(|&h| unsafe { IsWindowVisible(HWND(h as *mut _)).as_bool() });
                let top = unsafe { WindowFromPoint(POINT { x: px, y: py }) };
                log::info!(
                    "desktop-cursor-heartbeat: pos=({px},{py}) want_hidden={} hide_applied={} mag_showcursor_diag={} sprite_requested={} sprite_blocked={} sprite_window_visible={} bridge={} top_hwnd={:#x}",
                    WANT_CURSOR_HIDDEN.load(Ordering::Acquire),
                    CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
                    system_cursor_showing(),
                    sprite_requested,
                    sprite_blocked,
                    sprite_window_visible,
                    DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire),
                    top.0 as isize
                );
            }
        }
        g.ui_hover_active = false;
        g.last_ui_hover = None;
        g.last_ui_post = None;
        return false;
    }

    let hit = hit.expect("nonzero native UI owner requires a hit");
    if old_owner != new_owner {
        log::debug!(
            "native-ui-ownership-enter: hwnd={:#x} previous={:#x} boundary=exact zorder=win32 pos=({px},{py})",
            new_owner,
            old_owner
        );
    }

    g.native_gui_owner_hwnd = new_owner;
    NATIVE_GUI_OWNER.store(new_owner, Ordering::Release);
    DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
    if crate::platform::win32::is_own_window(new_owner) && !is_panel_hit(hit) {
        publish_main_gui_passthrough(new_owner, false, "native-gui-owner-enter");
    }
    let pos = (px, py);
    g.ui_hover_active = true;
    g.last_ui_hover = Some(UiHover { hit, pos });
    g.virt = (px as f64, py as f64);
    g.last_set = pos;
    g.last_hw = pos;
    if g.active {
        if external_native_cursor_owner(new_owner) {
            request_cursor_hidden(false);
            if old_owner != new_owner {
                log::debug!(
                    "external-native cursor ownership entered: hwnd={:#x} pos=({px},{py})",
                    new_owner
                );
            }
        } else {
            // Existing Neo GUI/panel capture-wide sprite path.
            request_cursor_hidden(true);
            let force_sync = old_owner != new_owner;
            let sync_due = g
                .last_gui_sprite_sync_at
                .is_none_or(|last| now.duration_since(last) >= std::time::Duration::from_millis(4));
            if force_sync || sync_due {
                sprite_move_now(px, py, true);
                g.last_gui_sprite_sync_at = Some(now);
            } else {
                sprite_move(px, py, true);
            }
            if old_owner != new_owner {
                keep_cursor_sprite_on_top();
                log::debug!(
                    "native-gui sprite ownership entered: hwnd={:#x} pos=({px},{py})",
                    new_owner
                );
            }
        }
    } else if CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
        if old_owner == new_owner {
            // Idle/non-capture fallback retains the proven native-cursor path.
            if SPRITE_SHOW.swap(false, Ordering::AcqRel) {
                sprite_hide_window_only();
                log::debug!(
                    "native-gui stable ownership cancelled stale bridge sprite: hwnd={:#x} pos=({px},{py})",
                    new_owner
                );
            }
        } else {
            sprite_move(px, py, true);
            keep_cursor_sprite_on_top();
        }
    } else if old_owner == new_owner && SPRITE_SHOW.swap(false, Ordering::AcqRel) {
        sprite_hide_window_only();
        log::debug!(
            "native-gui stable ownership drained queued sprite after native reveal: hwnd={:#x} pos=({px},{py})",
            new_owner
        );
    }
    true
}

fn click_panel_once(g: &mut State, hover: UiHover, bit: u8, _now: std::time::Instant) {
    g.edge_out_accum = 0.0;
    g.last_engage_at = None;
    g.must_leave_content = false;
    g.swallow_up |= bit;
    g.ui_hold_bits = 0;
    g.ui_hover_active = true;
    g.hidden_by_idle = false;
    g.last_ui_hover = Some(hover);

    // v348j: panel buttons have exactly one authoritative action path.  Older
    // builds both posted a synthetic native button sequence to the egui panel
    // AND queued the low-level direct action.  Depending on whether cursor
    // ownership had already handed off to the native panel, the first physical
    // click could therefore become a focus/ownership click while the action was
    // only observed on the second gesture.  Keep hover messages for visual
    // feedback, but execute panel controls only through PANEL_ACTIONS.
    queue_panel_action(hover, bit);
    if g.cursor_hidden {
        request_cursor_hidden(true);
    }
    sprite_move(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
    keep_cursor_sprite_on_top();
}

fn consume_button_during_pending_engage(msg: u32) -> bool {
    let Some((bit, down)) = button_transition(msg) else {
        return false;
    };
    let Ok(mut g) = state().try_lock() else {
        return false;
    };
    if g.pending_engage.is_none() {
        return false;
    }
    if down {
        g.swallow_up |= bit;
    } else {
        g.swallow_up &= !bit;
    }
    g.last_button_event = Some(std::time::Instant::now());
    log::info!("cursor handoff consumed button edge: bit={bit:#04x} down={down}");
    true
}

fn pending_engage_active() -> bool {
    state().try_lock().is_ok_and(|g| g.pending_engage.is_some())
}

fn try_panel_redirect(msg: u32, px: i32, py: i32) -> bool {
    let Some((bit, down)) = button_transition(msg) else {
        return false;
    };
    let st = state();
    let mut g = match st.try_lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if !g.active {
        return false;
    }
    g.last_button_event = Some(std::time::Instant::now());
    if !down {
        if g.ui_hold_bits & bit != 0 {
            g.ui_hold_bits &= !bit;
            if g.ui_hold_bits == 0 {}
            return false;
        }
        // swallow the up that pairs with a redirected down
        if g.swallow_up & bit != 0 {
            g.swallow_up &= !bit;
            return true;
        }
        return false;
    }
    if !g.engaged {
        // v348j: the floating panel must not require a first click merely to
        // activate/focus its native viewport.  When the real Windows cursor is
        // already over the visible topmost panel, route the very first LEFT
        // button-down through the same direct action queue used while engaged.
        // Swallow the matching UP so egui cannot also treat this physical click
        // as a second, duplicate action.  Other buttons keep normal Win32
        // semantics.
        if bit == BTN_LEFT {
            refresh_live_own_no_engage_geometry(&mut g, std::time::Instant::now());
            if let Some(hit) =
                hit_ui_for_cursor_ownership(&g.no_engage, px, py).filter(|hit| is_panel_hit(*hit))
            {
                let hover = UiHover { hit, pos: (px, py) };
                g.swallow_up |= bit;
                g.ui_hover_active = true;
                g.last_ui_hover = Some(hover);
                queue_panel_action(hover, bit);
                log::debug!(
                    "panel first-click direct: hwnd={:#x} pos=({px},{py}) owner=native",
                    hit.hwnd
                );
                return true;
            }
        }
        return false;
    }
    // Is the VISIBLE virtual cursor exactly over the current topmost panel /
    // GUI? No halo, no click margin and no previous-hover grace are allowed.
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    let now = std::time::Instant::now();
    let Some(hover) = ui_click_target_for_event(&mut g, px, py, now) else {
        return false;
    };
    let hit = hover.hit;
    let click = hover.pos;
    let s = g.src;
    let back = g.last_set;
    if hit.hwnd != 0 {
        if is_panel_hit(hit) {
            click_panel_once(&mut g, hover, bit, now);
            drop(g);
            log::debug!(
                "panel click handoff to hwnd={:#x} at ({},{}) virt=({vx},{vy}) raw=({px},{py})",
                hit.hwnd,
                click.0,
                click.1
            );
            return true;
        } else {
            let handoff_ok = begin_ui_hold(&mut g, hover, bit);
            drop(g);
            if handoff_ok {
                log::info!(
                    "ui hold handoff to hwnd={:#x} at ({},{}) virt=({vx},{vy}) raw=({px},{py})",
                    hit.hwnd,
                    click.0,
                    click.1
                );
            } else {
                log::warn!(
                    "ui hold click swallowed after unverified native warp: hwnd={:#x} at ({},{}) virt=({vx},{vy}) raw=({px},{py})",
                    hit.hwnd,
                    click.0,
                    click.1
                );
            }
            return true;
        }
    } else {
        g.swallow_up |= bit;
        g.last_hw = back;
        g.last_set = back;
        drop(g);
        let (clicked, _) = warp_unclipped_verified(click, "synthetic panel click handoff");
        if clicked {
            inject_mouse_button(bit, true);
            inject_mouse_button(bit, false);
        }
        let (restored, actual) = warp_then_clip_source(back, s);
        log::info!(
            "panel click redirected (inject) target=({},{}) clicked={} source_restored={} actual=({},{})",
            click.0,
            click.1,
            clicked,
            restored,
            actual.0,
            actual.1
        );
        return true;
    }
}

unsafe extern "system" fn mouse_proc(code: i32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    unsafe {
        if code >= 0 {
            if INPUT_FAILSAFE_LATCHED.load(Ordering::Acquire) {
                return CallNextHookEx(None, code, wp, lp);
            }
            let info = &*(lp.0 as *const MSLLHOOKSTRUCT);
            // Hardware events only. The native desktop regression harness can
            // opt in to one private SendInput tag so it exercises this exact
            // hook path; ordinary injected input and our SetCursorPos feedback
            // remain ignored.
            let is_injected = info.flags & LLMHF_INJECTED != 0;
            if is_injected
                && desktop_test_input_enabled()
                && !DESKTOP_TEST_INJECTED_DIAG.swap(true, Ordering::Relaxed)
            {
                log::info!(
                    "native desktop injected DIAG: flags={:?} extra={:#x} expected={:#x}",
                    info.flags,
                    info.dwExtraInfo,
                    DESKTOP_TEST_INPUT_TAG
                );
            }
            let test_mode = desktop_test_input_enabled();
            let native_test_move =
                is_injected && info.dwExtraInfo == DESKTOP_TEST_INPUT_TAG && test_mode;
            if native_test_move && !DESKTOP_TEST_INPUT_SEEN.swap(true, Ordering::Relaxed) {
                log::info!("native desktop test input entered WH_MOUSE_LL production path");
            }
            if test_mode && !native_test_move && wp.0 as u32 == WM_MOUSEMOVE {
                return LRESULT(1);
            }
            // Keep the native desktop regression deterministic: while its
            // explicit opt-in is active, physical mouse jitter must not mix
            // with the tagged path being measured.
            if native_test_move || (!test_mode && info.flags & LLMHF_INJECTED == 0) {
                let msg = wp.0 as u32;
                match msg {
                    WM_MOUSEMOVE => {
                        note_deferred_source_drag_move_lockfree(info.pt.x, info.pt.y);
                        // SetCursorPos during engagement moves the real cursor into the
                        // source rect. Do not let the original hardware move continue
                        // afterwards and overwrite that new position with its old point.
                        let consumed = on_hardware_move(info.pt.x, info.pt.y);
                        if native_test_move {
                            let n = DESKTOP_TEST_MOVE_COUNT.fetch_add(1, Ordering::Relaxed) + 1;
                            if n % 32 == 0 {
                                let virt = virtual_cursor_pos();
                                log::info!(
                                    "native-desktop-cursor-sample: n={n} raw=({},{}) virtual={virt:?} owner={:#x} hidden={} consumed={consumed}",
                                    info.pt.x,
                                    info.pt.y,
                                    NATIVE_GUI_OWNER.load(Ordering::Acquire),
                                    CURSOR_HIDE_APPLIED.load(Ordering::Acquire)
                                );
                            }
                        }
                        if consumed {
                            return LRESULT(1);
                        }
                    }
                    WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN | WM_LBUTTONUP
                    | WM_RBUTTONUP | WM_MBUTTONUP => {
                        // Floating-panel buttons are latency-critical controls.
                        // Resolve them before any mutex-backed cursor state so
                        // a first click cannot be lost to focus/ownership or a
                        // transient try_lock collision.
                        if try_panel_direct_lockfree(msg, info.pt.x, info.pt.y) {
                            return LRESULT(1);
                        }
                        // The main capture control uses the same lock-free
                        // physical-DOWN routing for Start and Stop. Passing the
                        // edge onward keeps winit/egui pointer state balanced;
                        // the atomic action is consumed at the start of that frame.
                        if try_main_control_lockfree(msg, info.pt.x, info.pt.y) {
                            // This is unquestionably a native main-GUI control
                            // gesture. Bypass source/pending-engage routing but
                            // still deliver the edge to winit/egui so its event
                            // loop wakes immediately and consumes the action.
                            return CallNextHookEx(None, code, wp, lp);
                        }
                        if tensorrt_build_native_cursor_guard() {
                            // The TensorRT progress popup and the rest of Neo's
                            // GUI stay fully interactive, but source cursor
                            // routing is disabled for the whole build epoch.
                            return CallNextHookEx(None, code, wp, lp);
                        }
                        // A GUI -> overlay handoff may be waiting a few
                        // milliseconds for the owner-thread cursor hide. Never
                        // let a click warp or land at source coordinates during
                        // that atomic transition.
                        if consume_button_during_pending_engage(msg) {
                            return LRESULT(1);
                        }
                        // a press aimed (visually) at the panel is redirected and
                        // swallowed so it never reaches the source underneath
                        if try_panel_redirect(msg, info.pt.x, info.pt.y) {
                            return LRESULT(1);
                        }
                        // v643: when only the hidden mapped source cursor is
                        // covered by an external window, swallow this edge
                        // without taking State and defer ALL Win32/source work
                        // to the normal engine thread. This atomic-only guard
                        // replaces v642's unsafe in-hook window/message route.
                        if queue_occluded_source_button_edge_lockfree(msg, info.pt.x, info.pt.y) {
                            return LRESULT(1);
                        }
                        track_button_state(msg, info.pt.x, info.pt.y);
                        let source_edge_consumed =
                            align_engaged_cursor_for_input(msg, info.pt.x, info.pt.y);
                        if matches!(msg, WM_LBUTTONDOWN | WM_RBUTTONDOWN | WM_MBUTTONDOWN) {
                            diagnose_click(info.pt.x, info.pt.y);
                        }
                        if source_edge_consumed {
                            return LRESULT(1);
                        }
                    }
                    WM_MOUSEWHEEL | WM_MOUSEHWHEEL => {
                        if tensorrt_build_native_cursor_guard() {
                            return CallNextHookEx(None, code, wp, lp);
                        }
                        if pending_engage_active() {
                            return LRESULT(1);
                        }
                        if align_engaged_cursor_for_input(msg, info.pt.x, info.pt.y) {
                            return LRESULT(1);
                        }
                    }
                    _ => {}
                }
            }
        }
        CallNextHookEx(None, code, wp, lp)
    }
}

fn mouse_button_flag(msg: u32) -> Option<MOUSE_EVENT_FLAGS> {
    match msg {
        WM_LBUTTONDOWN => Some(MOUSEEVENTF_LEFTDOWN),
        WM_LBUTTONUP => Some(MOUSEEVENTF_LEFTUP),
        WM_RBUTTONDOWN => Some(MOUSEEVENTF_RIGHTDOWN),
        WM_RBUTTONUP => Some(MOUSEEVENTF_RIGHTUP),
        WM_MBUTTONDOWN => Some(MOUSEEVENTF_MIDDLEDOWN),
        WM_MBUTTONUP => Some(MOUSEEVENTF_MIDDLEUP),
        _ => None,
    }
}

fn post_mouse_button(msg: u32, tx: i32, ty: i32, src_hwnd: isize) -> bool {
    unsafe {
        let screen = POINT { x: tx, y: ty };
        let mut hit = WindowFromPoint(screen);
        if hit.0.is_null() {
            hit = HWND(src_hwnd as *mut _);
        }
        let mut client = screen;
        let _ = ScreenToClient(hit, &mut client);
        let down_wparam = match msg {
            WM_LBUTTONDOWN => 0x0001usize,
            WM_RBUTTONDOWN => 0x0002usize,
            WM_MBUTTONDOWN => 0x0010usize,
            _ => 0usize,
        };
        let lp = ((client.y as u32 & 0xFFFF) << 16) | (client.x as u32 & 0xFFFF);
        PostMessageW(Some(hit), msg, WPARAM(down_wparam), LPARAM(lp as isize)).is_ok()
    }
}

/// Post directly to the known source root instead of WindowFromPoint. This is
/// only a fallback for an engaged source gesture whose physical source point is
/// covered by another top-most window. Normal unoccluded input still follows
/// the native hardware path, preserving browser/raw-input compatibility.
fn post_mouse_button_to_source(msg: u32, tx: i32, ty: i32, src_hwnd: isize) -> bool {
    if src_hwnd == 0 {
        return false;
    }
    unsafe {
        let target = HWND(src_hwnd as *mut _);
        let mut client = POINT { x: tx, y: ty };
        let _ = ScreenToClient(target, &mut client);
        let wparam = match msg {
            WM_LBUTTONDOWN => 0x0001usize,
            WM_RBUTTONDOWN => 0x0002usize,
            WM_MBUTTONDOWN => 0x0010usize,
            _ => 0usize,
        };
        let lp = ((client.y as u32 & 0xFFFF) << 16) | (client.x as u32 & 0xFFFF);
        PostMessageW(Some(target), msg, WPARAM(wparam), LPARAM(lp as isize)).is_ok()
    }
}

fn post_mouse_move_to_source(tx: i32, ty: i32, src_hwnd: isize, buttons: u8) -> bool {
    if src_hwnd == 0 {
        return false;
    }
    unsafe {
        let target = HWND(src_hwnd as *mut _);
        let mut client = POINT { x: tx, y: ty };
        let _ = ScreenToClient(target, &mut client);
        let mut wparam = 0usize;
        if buttons & BTN_LEFT != 0 {
            wparam |= 0x0001;
        }
        if buttons & BTN_RIGHT != 0 {
            wparam |= 0x0002;
        }
        if buttons & BTN_MIDDLE != 0 {
            wparam |= 0x0010;
        }
        let lp = ((client.y as u32 & 0xFFFF) << 16) | (client.x as u32 & 0xFFFF);
        PostMessageW(
            Some(target),
            WM_MOUSEMOVE,
            WPARAM(wparam),
            LPARAM(lp as isize),
        )
        .is_ok()
    }
}

fn clamp_deferred_source_point(g: &State, px: i32, py: i32) -> (i32, i32) {
    if g.src.w <= 1 || g.src.h <= 1 {
        return g.last_set;
    }
    (
        px.clamp(g.src.x, g.src.x.saturating_add(g.src.w).saturating_sub(1)),
        py.clamp(g.src.y, g.src.y.saturating_add(g.src.h).saturating_sub(1)),
    )
}

fn drain_deferred_source_input(g: &mut State) {
    let src_hwnd = DEFERRED_SOURCE_HWND.load(Ordering::Acquire);
    if DEFERRED_SOURCE_OVERFLOW.swap(false, Ordering::AcqRel) {
        // A full queue means the hook deliberately swallowed at least one edge
        // rather than leaking it into the covering window. Recover source-side
        // bookkeeping here and keep the capture usable; do not escalate into
        // cursor/DWM manipulation from the hook.
        let held = g.source_direct_bits | DEFERRED_SOURCE_HELD_BITS.swap(0, Ordering::AcqRel);
        if held != 0 && g.src_hwnd != 0 {
            let target = if g.src.w > 1 && g.src.h > 1 && g.content.w > 1 && g.content.h > 1 {
                map_content_to_source(g.content, g.src, g.virt.0, g.virt.1)
            } else {
                g.last_set
            };
            for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
                if held & bit == 0 {
                    continue;
                }
                let msg = match bit {
                    BTN_LEFT => WM_LBUTTONUP,
                    BTN_RIGHT => WM_RBUTTONUP,
                    BTN_MIDDLE => WM_MBUTTONUP,
                    _ => continue,
                };
                let _ = post_mouse_button_to_source(msg, target.0, target.1, g.src_hwnd);
            }
        }
        g.buttons_down &= !held;
        g.source_direct_bits &= !held;
        g.last_source_direct_post = None;
        SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
        let write = DEFERRED_EVENT_WRITE.load(Ordering::Acquire);
        DEFERRED_EVENT_READ.store(write, Ordering::Release);
        log::error!(
            "deferred-source-input overflow recovered: swallowed_edge=true held={held:#04x} action=release-and-drop"
        );
    }

    let mut read = DEFERRED_EVENT_READ.load(Ordering::Relaxed);
    let write = DEFERRED_EVENT_WRITE.load(Ordering::Acquire);
    while read != write {
        let slot = (read as usize) % DEFERRED_SOURCE_EVENT_CAP;
        let msg = DEFERRED_EVENT_MSG[slot].load(Ordering::Relaxed);
        let (px, py) = unpack_drag_pair(DEFERRED_EVENT_POS[slot].load(Ordering::Relaxed));
        let (vx, vy) = unpack_drag_pair(DEFERRED_EVENT_VISUAL[slot].load(Ordering::Relaxed));
        read = read.wrapping_add(1);
        DEFERRED_EVENT_READ.store(read, Ordering::Release);

        let Some((bit, down)) = button_transition(msg) else {
            continue;
        };
        if src_hwnd == 0 || g.src_hwnd == 0 || g.src_hwnd != src_hwnd || !g.active {
            g.buttons_down &= !bit;
            g.source_direct_bits &= !bit;
            if bit == BTN_LEFT {
                SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
            }
            continue;
        }

        let now = std::time::Instant::now();
        g.last_button_event = Some(now);
        g.last_move = Some(now);
        g.hidden_by_idle = false;
        // The queued position is the intended source-screen target mapped
        // lock-free from the visible Neo cursor. Clamp only to the committed
        // source client rect in case geometry changed before this drain.
        let (tx, ty) = clamp_deferred_source_point(g, px, py);

        if down {
            let move_buttons = g.buttons_down & !bit;
            let move_forwarded = post_mouse_move_to_source(tx, ty, src_hwnd, move_buttons);
            let forwarded = post_mouse_button_to_source(msg, tx, ty, src_hwnd);
            g.buttons_down |= bit;
            g.source_direct_bits |= bit;

            // Deferred delivery is virtual: no SetCursorPos is performed here.
            // Keep hardware bookkeeping anchored to the REAL hidden cursor so
            // the next LL move cannot turn the source target delta into a huge
            // synthetic cursor jump.
            let physical = read_cursor_pos_or(g.last_hw, "deferred source down physical preserve");
            g.last_set = physical;
            g.last_hw = physical;

            // Do not arm the native source-window drag provenance for a
            // synthetic client gesture (seek-bar/button drag). That owner is
            // consumed by overlay-follow code and expects raw hardware screen
            // coordinates, while this route intentionally carries mapped
            // source targets.
            if bit == BTN_LEFT {
                SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
            }
            request_cursor_hidden(true);
            sprite_move_now(vx, vy, true);
            keep_cursor_sprite_on_top();
            log::warn!(
                "source-button-deferred-occlusion: edge=down msg={msg:#x} source=({tx},{ty}) visual=({vx},{vy}) physical=({},{}) hwnd={src_hwnd:#x} move_forwarded={move_forwarded} forwarded={forwarded} hook_win32_calls=0 swallowed=true",
                physical.0,
                physical.1
            );
        } else {
            let move_forwarded = post_mouse_move_to_source(tx, ty, src_hwnd, g.source_direct_bits);
            let forwarded = post_mouse_button_to_source(msg, tx, ty, src_hwnd);
            g.buttons_down &= !bit;
            g.source_direct_bits &= !bit;
            if g.source_direct_bits == 0 {
                g.last_source_direct_post = None;
            }
            if bit == BTN_LEFT {
                SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
            }
            let physical = read_cursor_pos_or(g.last_hw, "deferred source up physical preserve");
            g.last_set = physical;
            g.last_hw = physical;
            request_cursor_hidden(true);
            sprite_move_now(vx, vy, true);
            keep_cursor_sprite_on_top();
            log::info!(
                "source-button-deferred-occlusion: edge=up msg={msg:#x} source=({tx},{ty}) visual=({vx},{vy}) physical=({},{}) hwnd={src_hwnd:#x} move_forwarded={move_forwarded} forwarded={forwarded} hook_win32_calls=0 swallowed=true",
                physical.0,
                physical.1
            );
        }
    }

    // While a deferred DOWN is held, preserve source drag hover even if the
    // mutex-backed move path misses a high-rate sample. Latest-only is enough;
    // source_direct_bits continues to own ordinary mapped move forwarding too.
    if g.source_direct_bits != 0 && g.src_hwnd == src_hwnd && src_hwnd != 0 {
        let seq = DEFERRED_SOURCE_MOVE_SEQ.load(Ordering::Acquire);
        let drained = DEFERRED_SOURCE_MOVE_DRAINED_SEQ.load(Ordering::Relaxed);
        if seq != drained {
            let (px, py) = unpack_drag_pair(DEFERRED_SOURCE_MOVE_POS.load(Ordering::Acquire));
            let (tx, ty) = clamp_deferred_source_point(g, px, py);
            let _ = post_mouse_move_to_source(tx, ty, src_hwnd, g.source_direct_bits);
            DEFERRED_SOURCE_MOVE_DRAINED_SEQ.store(seq, Ordering::Release);
        }
    }
}

/// A directly-forwarded source gesture is still logically owned by the
/// magnified source, even though the hidden physical cursor cannot be placed on
/// the mapped point because another top-most window covers it. Keep the active
/// capture's sprite/native-hide contract authoritative for the whole gesture;
/// otherwise an unrelated foreground window (Task Manager is the common case)
/// can become visual cursor owner between the synthetic DOWN and UP.
fn maintain_direct_source_cursor_contract(g: &mut State) {
    g.native_gui_owner_hwnd = 0;
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
    g.native_gui_settle = None;
    g.ui_hover_active = false;
    g.last_ui_hover = None;
    g.last_ui_post = None;
    g.hidden_by_idle = false;
    g.cursor_hidden = true;
    DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
    request_cursor_hidden(true);
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    sprite_move_now(vx, vy, true);
    keep_cursor_sprite_on_top();
}

fn source_root_owns_screen_point(src_hwnd: isize, x: i32, y: i32) -> bool {
    if src_hwnd == 0 {
        return false;
    }
    unsafe {
        let hit = WindowFromPoint(POINT { x, y });
        if hit.0.is_null() {
            return false;
        }
        let root = GetAncestor(hit, GA_ROOT).0 as isize;
        if root == src_hwnd {
            return true;
        }
        // Menus/tooltips/owned popups can have their own top-level HWND even
        // though they are still part of the source interaction. Treat the
        // source's root-owner and same-process popup family as native source
        // ownership so the occlusion fallback never steals input from a real
        // source menu (mpv's #32768 context menu is one concrete example).
        let root_owner = GetAncestor(hit, GA_ROOTOWNER).0 as isize;
        if root_owner == src_hwnd {
            return true;
        }
        let mut src_pid = 0u32;
        let mut hit_pid = 0u32;
        let _ = GetWindowThreadProcessId(HWND(src_hwnd as *mut _), Some(&mut src_pid));
        let _ = GetWindowThreadProcessId(hit, Some(&mut hit_pid));
        src_pid != 0 && hit_pid == src_pid
    }
}

fn send_mouse_button(flag: MOUSE_EVENT_FLAGS, msg: u32, tx: i32, ty: i32, src_hwnd: isize) -> bool {
    let mut input = INPUT {
        r#type: INPUT_MOUSE,
        ..Default::default()
    };
    input.Anonymous.mi.dwFlags = flag;
    let sent = unsafe { SendInput(&[input], std::mem::size_of::<INPUT>() as i32) };
    if sent == 1 {
        return true;
    }
    let err = unsafe { GetLastError() };
    log::warn!("SendInput mouse button failed: sent={sent} error={err:?}; posting mouse message");
    if post_mouse_button(msg, tx, ty, src_hwnd) {
        return true;
    }
    log::warn!("PostMessage mouse button failed; using mouse_event as final fallback");
    unsafe {
        mouse_event(flag, 0, 0, 0, 0);
    }
    true
}

fn forward_engaged_button(msg: u32) -> bool {
    let Some(flag) = mouse_button_flag(msg) else {
        return false;
    };
    let st = state();
    let mut g = match st.try_lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if !g.active || !g.engaged {
        return false;
    }
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    if !g.content.contains(vx, vy) {
        let now = std::time::Instant::now();
        if let Some(actual) =
            release_windows_to_native_at(&mut g, (vx, vy), "button edge native handoff", false)
        {
            disengage_state(&mut g, now);
            g.cooldown_until = Some(reenter_cooldown(now, g.oscillations));
            log::info!(
                "button pass-through after verified edge release at ({},{})",
                actual.0,
                actual.1
            );
            return false;
        }
        log::warn!(
            "button edge release swallowed because native position was not verified: target=({vx},{vy})"
        );
        return true;
    }
    let s = g.src;
    if s.w <= 1 || s.h <= 1 {
        return false;
    }
    let (tx, ty) = map_content_to_source(g.content, s, g.virt.0, g.virt.1);
    let (aligned, actual) = warp_then_clip_source((tx, ty), s);
    if !aligned {
        log::warn!(
            "forward button source alignment rejected: virtual=({vx},{vy}) target=({tx},{ty}) actual=({},{})",
            actual.0,
            actual.1
        );
        return true;
    }
    g.last_set = actual;
    g.last_hw = actual;
    log::info!("forward button {msg:#x}: virtual=({vx},{vy}) -> source=({tx},{ty})");
    let src_hwnd = g.src_hwnd;
    drop(g);
    send_mouse_button(flag, msg, tx, ty, src_hwnd)
}

/// Align a button/wheel edge to the mapped source. Returns true when the
/// physical event must be swallowed because Neo forwarded it directly to the
/// source (or safely dropped it) instead of allowing it to hit an unrelated
/// desktop/top-most window.
fn align_engaged_cursor_for_input(msg: u32, _px: i32, _py: i32) -> bool {
    let st = state();
    let mut g = match st.try_lock() {
        Ok(g) => g,
        Err(_) => return false,
    };

    let transition = button_transition(msg);
    // A directly-forwarded DOWN owns its matching UP even if source ownership
    // changed between the two edges. This prevents a half gesture from leaking
    // to the desktop while the source remains logically pressed.
    if let Some((bit, down)) = transition {
        if !down && g.source_direct_bits & bit != 0 {
            let src_hwnd = g.src_hwnd;
            let (tx, ty) = if g.src.w > 1 && g.src.h > 1 && g.content.w > 1 && g.content.h > 1 {
                map_content_to_source(g.content, g.src, g.virt.0, g.virt.1)
            } else {
                g.last_set
            };
            // Keep the direct gesture visually/source-owned until the matching
            // UP is delivered. Do not attempt to steal foreground from an
            // unrelated top-most window; Windows can legitimately reject that
            // request. The synthetic source transaction is self-contained.
            maintain_direct_source_cursor_contract(&mut g);
            let now = std::time::Instant::now();
            g.last_move = Some(now);
            g.hidden_by_idle = false;
            let move_forwarded = post_mouse_move_to_source(tx, ty, src_hwnd, g.source_direct_bits);
            let forwarded = post_mouse_button_to_source(msg, tx, ty, src_hwnd);
            g.source_direct_bits &= !bit;
            if g.source_direct_bits == 0 {
                g.last_source_direct_post = None;
            }
            SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
            maintain_direct_source_cursor_contract(&mut g);
            log::info!(
                "source-button-direct-forward: edge=up msg={msg:#x} source=({tx},{ty}) hwnd={src_hwnd:#x} move_forwarded={move_forwarded} activation=skipped forwarded={forwarded} reason=matching-direct-down"
            );
            return true;
        }
    }

    if !g.active || !g.engaged || g.pending_engage.is_some() {
        return false;
    }
    let s = g.src;
    if s.w <= 1 || s.h <= 1 {
        return false;
    }
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    if !g.content.contains(vx, vy) {
        let now = std::time::Instant::now();
        if release_windows_to_native_at(&mut g, (vx, vy), "input edge native handoff", false)
            .is_some()
        {
            disengage_state(&mut g, now);
            g.cooldown_until = Some(reenter_cooldown(now, g.oscillations));
        }
        return false;
    }

    // Keep the visible cursor authoritative. First ask who owns the mapped
    // source point BEFORE touching the physical cursor. If Task Manager or
    // another top-most window covers that point, SetCursorPos cannot make the
    // physical click safely land on the source and repeated in-hook warp attempts
    // visibly destabilize the cursor. In that case we skip warping completely
    // and use the direct source route below. For an unoccluded point, retain the
    // established v636 warp-then-clip path.
    let (tx, ty) = map_content_to_source(g.content, s, g.virt.0, g.virt.1);
    let source_point_unoccluded = source_root_owns_screen_point(g.src_hwnd, tx, ty);
    let (aligned, actual) = if source_point_unoccluded {
        warp_then_clip_source((tx, ty), s)
    } else {
        (
            false,
            read_cursor_pos_or(g.last_set, "engaged input occlusion bypass"),
        )
    };
    if aligned {
        g.last_set = actual;
        g.last_hw = actual;
    }

    // Wheel input keeps the established native path when alignment succeeds.
    // If it fails we simply swallow this one edge; posting synthetic wheel
    // messages here would change high-resolution wheel semantics.
    if transition.is_none() {
        if !aligned {
            log::warn!(
                "engaged wheel alignment failed: target=({tx},{ty}) actual=({},{}) action=swallow",
                actual.0,
                actual.1
            );
            return true;
        }
        return false;
    }

    let (bit, down) = transition.unwrap();
    let source_owns_point = aligned && source_point_unoccluded;
    if source_owns_point {
        return false;
    }

    // v643 safety rollback: occluded client-source button delivery is handled
    // by the atomic-only hook gate + engine-thread deferred queue above.
    // If that gate did not classify this edge (for example because the external
    // geometry snapshot changed between engine ticks), do NOT resurrect the
    // v640/v641 in-hook realign/PostMessage experiment. Swallow this edge here;
    // losing one click is safer than touching another process/DWM from the
    // low-level hook.
    let src_hwnd = g.src_hwnd;

    // Never allow an engaged source click with failed/occluded alignment to
    // fall through to Program Manager or an unrelated top-most window.
    if down {
        g.swallow_up |= bit;
    }
    log::warn!(
        "engaged source button swallowed: edge={} msg={msg:#x} target=({tx},{ty}) actual=({},{}) aligned={aligned} source_owns_point={source_owns_point} window_frame_input={} hwnd={src_hwnd:#x}",
        if down { "down" } else { "up" },
        actual.0,
        actual.1,
        g.window_frame_input
    );
    true
}

/// Log exactly where a click will land while engaged — the key diagnostic for
/// cases where clicks would otherwise fail to reach the source.
fn diagnose_click(px: i32, py: i32) {
    let st = state();
    let Ok(g) = st.try_lock() else { return };
    if !g.active {
        return;
    }
    unsafe {
        let mut point = POINT { x: px, y: py };
        if g.engaged {
            let _ = GetCursorPos(&mut point);
        }
        let px = point.x;
        let py = point.y;
        let hit = WindowFromPoint(point);
        let root = GetAncestor(hit, GA_ROOT);
        let src = g.src_hwnd;
        let mut title = [0u16; 128];
        let n = GetWindowTextW(root, &mut title);
        let title = String::from_utf16_lossy(&title[..n.max(0) as usize]);
        log::info!(
            "click at ({px},{py}) engaged={} -> hwnd={:?} root={:?} (src={src:#x}) match={} title='{title}'",
            g.engaged,
            hit.0,
            root.0,
            root.0 as isize == src
        );
    }
}

/// Show/hide the REAL system cursor via the Magnification API — WITH the
/// the cursor-routing design recovery. The Mag context is thread-affine and can go stale
/// (e.g. across start/stop sessions), and then MagShowSystemCursor silently
/// FAILS: the real cursor stays visible and, confined to the source rect,
/// appears "stuck inside" the magnified view at the source-edge position —
/// exactly the stale-cursor return path. On failure we drop the stale context and
/// re-initialise on THIS thread, then retry. MUST be called on the hook thread
/// (which owns the Mag context).
fn set_system_cursor_visible(visible: bool) -> bool {
    unsafe {
        let ok = MagShowSystemCursor(visible).as_bool();
        if !ok {
            // Never repair a thread-affine Mag context here: this helper is
            // also reached from the LL hook. Ask the owner thread to do it.
            MAG_REINIT_REQUESTED.store(true, Ordering::Release);
        }
        if !ok && !visible {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                log::warn!("MagShowSystemCursor(false) failed; owner-thread re-init requested");
            }
        }
        ok
    }
}

/// Diagnostic only: ordinary ShowCursor/CURSOR_SHOWING state.
/// Magnification's MagShowSystemCursor(false) does not change this flag, so
/// `true` is normal even while Magnification is correctly hiding native.
/// Never use this helper to decide native-vs-sprite ownership.
fn system_cursor_showing() -> bool {
    unsafe {
        let mut ci = CURSORINFO {
            cbSize: std::mem::size_of::<CURSORINFO>() as u32,
            ..Default::default()
        };
        if GetCursorInfo(&mut ci).is_ok() {
            ci.flags.0 & CURSOR_SHOWING.0 != 0
        } else {
            true
        }
    }
}

// ---------------- cross-thread real-cursor hide (the 80 architecture) ----------
//
// Cursor pullback root cause:
// `MagShowSystemCursor(false).ok=true cursor_still_showing=true`): the
// Magnification API is a SILENT NO-OP when called from the WH_MOUSE_LL hook
// thread. It reports success but never hides the cursor, so the real cursor —
// confined to the source rect, which sits ~100-200px INSIDE the magnified view —
// bleeds through and looks like the cursor "returned" to the source edge.
//
// the cursor-routing design never hits this because it runs the whole Magnification API on
// its OVERLAY thread (the one that owns a shown top-level window and pumps its
// messages), with the LL-hook on a separate arithmetic-only thread. So we do the
// same: the hook thread only RECORDS the desired visibility here; the
// render-engine thread (which owns the overlay window + pumps it) APPLIES it via
// pump_cursor_visibility(). Do not assume process exit will repair every global
// cursor/clip state. v583 keeps a separate post-exit/explicit-emergency janitor,
// but no automatic worker is allowed to mutate ordinary capture state. Do not add a
// SetSystemCursor fallback here because it permanently replaces global cursor
// resources rather than simply changing visibility.
static WANT_CURSOR_HIDDEN: AtomicBool = AtomicBool::new(false);
/// True only after the Magnification owner thread successfully applied hide.
/// The sprite is gated by this flag, preventing a native+sprite double cursor.
static CURSOR_HIDE_APPLIED: AtomicBool = AtomicBool::new(false);
/// Cross-thread request; consumed only by the render-engine/Mag owner thread.
static MAG_REINIT_REQUESTED: AtomicBool = AtomicBool::new(true);
// Manual-only lock-free input rescue. Normal cursor ownership belongs to the
// render/Magnification thread. This latch is not armed by ordinary Start/Stop;
// it is reserved for exceptional direct recovery paths.
static INPUT_FAILSAFE_LATCHED: AtomicBool = AtomicBool::new(false);
static CAPTURE_SESSION_ACTIVE: AtomicBool = AtomicBool::new(false);
static CURSOR_OWNER_HEARTBEAT_MS: AtomicU64 = AtomicU64::new(0);
static HOOK_THREAD_HEARTBEAT_MS: AtomicU64 = AtomicU64::new(0);
static HOOK_HEARTBEAT_ENABLED: AtomicBool = AtomicBool::new(false);
const INPUT_HOOK_HEARTBEAT_TIMER_ID: usize = 0x4348_5346;
const INPUT_JANITOR_HOTKEY_ID: i32 = 0x4348;
const INPUT_JANITOR_REASSERT_MS: u64 = 4_000;
const INPUT_JANITOR_QUIT_GRACE_MS: u64 = 3_000;
static JANITOR_QUIT_EVENT_HANDLE: OnceLock<isize> = OnceLock::new();

fn start_input_failsafe_worker() {
    // v583: intentionally dormant. Automatic watchdog recovery must never
    // mutate cursor/input/source state during ordinary Neo operation. The
    // separate janitor process is the last-resort boundary after application
    // exit/quit-timeout, and Ctrl+Alt+Shift+Q remains an explicit user rescue.
}

pub fn capture_session_active() -> bool {
    CAPTURE_SESSION_ACTIVE.load(Ordering::Acquire)
}

#[derive(Clone, Copy, Debug)]
struct JanitorSourceRecovery {
    hwnd: isize,
    pid: u32,
    rect: Option<(i32, i32, i32, i32)>,
    placement: Option<crate::platform::win32::WindowPlacementSnapshot>,
    was_maximized: bool,
    was_topmost: bool,
    was_layered: bool,
    corner_preference: Option<i32>,
}

fn janitor_recovery_path(parent_pid: u32) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("cHiDeScaler-Neo-recovery-{parent_pid}.txt"))
}

fn parse_i32_list<const N: usize>(value: &str) -> Option<[i32; N]> {
    let parsed = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<i32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    parsed.try_into().ok()
}

fn read_janitor_source_recovery(parent_pid: u32) -> Option<JanitorSourceRecovery> {
    let text = std::fs::read_to_string(janitor_recovery_path(parent_pid)).ok()?;
    let mut values = std::collections::HashMap::<&str, &str>::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        values.insert(key.trim(), value.trim());
    }
    if values.get("version").copied() != Some("1") {
        return None;
    }
    let hwnd = values.get("hwnd")?.parse::<i64>().ok()? as isize;
    let pid = values.get("pid")?.parse::<u32>().ok()?;
    let rect = values
        .get("rect")
        .and_then(|value| parse_i32_list::<4>(value))
        .map(|v| (v[0], v[1], v[2], v[3]));
    let placement = values.get("placement").and_then(|value| {
        let parts = value.split(',').map(str::trim).collect::<Vec<_>>();
        if parts.len() != 10 {
            return None;
        }
        let flags = parts[0].parse::<u32>().ok()?;
        let show_cmd = parts[1].parse::<u32>().ok()?;
        let nums = parts[2..]
            .iter()
            .map(|part| part.parse::<i32>())
            .collect::<Result<Vec<_>, _>>()
            .ok()?;
        Some(
            crate::platform::win32::WindowPlacementSnapshot::from_janitor_raw_parts(
                flags,
                show_cmd,
                (nums[0], nums[1]),
                (nums[2], nums[3]),
                (nums[4], nums[5], nums[6], nums[7]),
            ),
        )
    });
    Some(JanitorSourceRecovery {
        hwnd,
        pid,
        rect,
        placement,
        was_maximized: values.get("maximized").copied() == Some("1"),
        was_topmost: values.get("topmost").copied() == Some("1"),
        was_layered: values.get("layered").copied() == Some("1"),
        corner_preference: values.get("corner").and_then(|value| {
            if *value == "none" {
                None
            } else {
                value.parse::<i32>().ok()
            }
        }),
    })
}

/// Publish only the source state Neo may temporarily mutate. The record lives
/// in the user's temp directory and is consumed solely by the isolated janitor
/// if Neo exits/crashes while a capture session still owns the source.
pub fn publish_janitor_source_recovery(
    hwnd: isize,
    pid: u32,
    rect: Option<(i32, i32, i32, i32)>,
    placement: Option<crate::platform::win32::WindowPlacementSnapshot>,
    was_maximized: bool,
    was_topmost: bool,
    was_layered: bool,
    corner_preference: Option<i32>,
) {
    if hwnd == 0 || pid == 0 || !crate::platform::win32::window_matches_pid(hwnd, pid) {
        return;
    }
    let rect_text = rect
        .map(|r| format!("{},{},{},{}", r.0, r.1, r.2, r.3))
        .unwrap_or_default();
    let placement_text = placement
        .map(|snapshot| {
            let (flags, show_cmd, min, max, normal) = snapshot.janitor_raw_parts();
            format!(
                "{flags},{show_cmd},{},{},{},{},{},{},{},{}",
                min.0, min.1, max.0, max.1, normal.0, normal.1, normal.2, normal.3
            )
        })
        .unwrap_or_default();
    let corner_text = corner_preference
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_string());
    let text = format!(
        "version=1\nhwnd={}\npid={}\nrect={}\nplacement={}\nmaximized={}\ntopmost={}\nlayered={}\ncorner={}\n",
        hwnd as i64,
        pid,
        rect_text,
        placement_text,
        u8::from(was_maximized),
        u8::from(was_topmost),
        u8::from(was_layered),
        corner_text,
    );
    let path = janitor_recovery_path(std::process::id());
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::rename(&tmp, &path);
    }
}

/// A completed ordinary Stop no longer needs post-exit source recovery. Clear
/// the record only after the engine has reached a confirmed idle state.
pub fn clear_janitor_source_recovery() {
    let _ = std::fs::remove_file(janitor_recovery_path(std::process::id()));
}

fn janitor_restore_source(parent_pid: u32) {
    let Some(recovery) = read_janitor_source_recovery(parent_pid) else {
        return;
    };
    if !crate::platform::win32::window_matches_pid(recovery.hwnd, recovery.pid) {
        return;
    }
    // Mirror the proven v578 order: visual opacity/style first, then DWM corner
    // preference, z-order, and finally the immutable start geometry.
    crate::platform::win32::show_window_visual(recovery.hwnd, recovery.was_layered);
    if let Some(preference) = recovery.corner_preference {
        let _ = crate::platform::win32::restore_window_corner_preference_checked(
            recovery.hwnd,
            recovery.pid,
            preference,
        );
    }
    if crate::platform::win32::window_matches_pid(recovery.hwnd, recovery.pid) {
        crate::platform::win32::set_topmost(recovery.hwnd, recovery.was_topmost);
    }
    if crate::platform::win32::window_matches_pid(recovery.hwnd, recovery.pid) {
        let _ = crate::platform::win32::restore_window_origin(
            recovery.hwnd,
            recovery.rect,
            recovery.was_maximized,
            recovery.placement,
        );
    }
}

fn janitor_quit_event_name(parent_pid: u32) -> Vec<u16> {
    format!("Local\\cHiDeScalerNeoQuit-{parent_pid}")
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect()
}

fn create_janitor_quit_event(parent_pid: u32) -> Option<HANDLE> {
    let name = janitor_quit_event_name(parent_pid);
    unsafe { CreateEventW(None, false, false, PCWSTR(name.as_ptr())).ok() }
}

/// Start the janitor's existing stable-exit grace clock. This does not touch
/// cursor, ClipCursor, source windows, providers, or Neo's input state.
pub fn notify_cursor_janitor_quit_requested(reason: &str) {
    let raw = if let Some(raw) = JANITOR_QUIT_EVENT_HANDLE.get().copied() {
        raw
    } else {
        let Some(handle) = create_janitor_quit_event(std::process::id()) else {
            log::warn!("cursor-janitor-quit-signal-unavailable: reason={reason}");
            return;
        };
        let raw = handle.0 as isize;
        let _ = JANITOR_QUIT_EVENT_HANDLE.set(raw);
        raw
    };
    let result = unsafe { SetEvent(HANDLE(raw as *mut _)) };
    if result.is_ok() {
        log::info!(
            "cursor-janitor-quit-grace-armed: reason={reason} grace_ms={INPUT_JANITOR_QUIT_GRACE_MS}"
        );
    } else {
        log::warn!("cursor-janitor-quit-signal-failed: reason={reason} error={result:?}");
    }
}

fn janitor_restore_cursor(mag_initialized: &mut bool) {
    unsafe {
        let _ = ClipCursor(None);
        if !*mag_initialized {
            *mag_initialized = MagInitialize().as_bool();
        }
        // Reassert more than once because a dying parent can still have one
        // queued hide transition while its threads are unwinding.
        for _ in 0..4 {
            let _ = MagShowSystemCursor(true);
            std::thread::sleep(std::time::Duration::from_millis(12));
        }
    }
}

/// One-shot isolated recovery helper retained for explicit diagnostics. It is
/// not launched by ordinary capture or by an automatic watchdog in v583.
pub fn run_cursor_rescue_once() {
    let mut mag_initialized = unsafe { MagInitialize() }.as_bool();
    janitor_restore_cursor(&mut mag_initialized);
    if mag_initialized {
        unsafe {
            let _ = MagUninitialize();
        }
    }
}

/// Run the no-GUI companion mode. It never participates in ordinary capture
/// Start/Stop. It wakes only when Neo exits, when Ctrl+Alt+Q / GUI-X arms the
/// post-quit grace (the normal 2 s shutdown window plus margin), or when the user explicitly presses the independent
/// Ctrl+Alt+Shift+Q emergency rescue key.
fn record_production_janitor_breadcrumb(line: &str) {
    let enabled = std::env::var("NEO_VULKAN_PRODUCTION_1PASS")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if !enabled {
        return;
    }
    let Ok(path) = std::env::var("NEO_VULKAN_PROBE_RESULT") else {
        return;
    };
    let path = path.trim();
    if path.is_empty() {
        return;
    }
    use std::io::Write as _;
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "{line}");
    }
}

pub fn run_cursor_janitor(parent_pid: u32) {
    record_production_janitor_breadcrumb(&format!(
        "vulkan-production-glsl: role=cursor-janitor phase=start pid={} parent_pid={parent_pid}",
        std::process::id()
    ));
    // The janitor never initializes Neo's GUI/GPU/capture stack. It waits on
    // the parent process plus one named quit-grace event and a manual emergency
    // hotkey message queue. Ordinary capture Stop never signals this process.
    let Ok(parent) = (unsafe { OpenProcess(PROCESS_SYNCHRONIZE, false, parent_pid) }) else {
        let mut mag_initialized = unsafe { MagInitialize() }.as_bool();
        janitor_restore_cursor(&mut mag_initialized);
        janitor_restore_source(parent_pid);
        let _ = std::fs::remove_file(janitor_recovery_path(parent_pid));
        if mag_initialized {
            unsafe {
                let _ = MagUninitialize();
            }
        }
        record_production_janitor_breadcrumb(&format!(
            "vulkan-production-glsl: role=cursor-janitor phase=exit-complete pid={} parent_pid={parent_pid} reason=parent-open-failed",
            std::process::id()
        ));
        return;
    };

    let quit_event = create_janitor_quit_event(parent_pid);
    let quit_event_present = quit_event.is_some();
    let mut handles = vec![parent];
    if let Some(event) = quit_event {
        handles.push(event);
    }

    let mut rescue_until = 0u64;
    let mut quit_deadline = 0u64;
    let mut mag_initialized = false;
    unsafe {
        let _ = RegisterHotKey(
            None,
            INPUT_JANITOR_HOTKEY_ID,
            MOD_CONTROL | MOD_ALT | MOD_SHIFT | MOD_NOREPEAT,
            b'Q' as u32,
        );
    }

    loop {
        let now = route_clock_ms();
        if quit_deadline != 0 && now >= quit_deadline {
            // Ctrl+Alt+Q / GUI-X had its normal shutdown grace but the Neo
            // process is still alive. Recover only external/native state; do
            // not terminate Neo or the source process.
            janitor_restore_cursor(&mut mag_initialized);
            janitor_restore_source(parent_pid);
            // One-shot post-grace recovery: once the pre-capture source state
            // has been restored, forget it so a later delayed parent exit cannot
            // rewind legitimate user changes made after recovery.
            let _ = std::fs::remove_file(janitor_recovery_path(parent_pid));
            quit_deadline = 0;
            // v622 production-test safety: after GUI close was explicitly
            // requested and this helper has already restored every external
            // state it owns, keeping the janitor resident provides no further
            // recovery value if the parent itself is stuck. Exit the helper so
            // Task Manager never shows an orphaned Neo test process. Ordinary
            // product launches keep the established parent-lifetime contract.
            if std::env::var("NEO_VULKAN_PRODUCTION_1PASS")
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(false)
            {
                record_production_janitor_breadcrumb(&format!(
                    "vulkan-production-glsl: role=cursor-janitor phase=quit-grace-expired pid={} parent_pid={parent_pid} action=exit-after-recovery",
                    std::process::id()
                ));
                break;
            }
        }

        let rescue_active = rescue_until != 0 && now < rescue_until;
        let mut timeout = if rescue_active { 100 } else { INFINITE };
        if quit_deadline != 0 {
            let remaining = quit_deadline.saturating_sub(now).max(1);
            timeout = timeout.min(remaining.min(u32::MAX as u64) as u32);
        }

        let wait = unsafe {
            MsgWaitForMultipleObjectsEx(
                Some(handles.as_slice()),
                timeout,
                QS_ALLINPUT,
                MWMO_INPUTAVAILABLE,
            )
        };
        if wait == WAIT_OBJECT_0 {
            record_production_janitor_breadcrumb(&format!(
                "vulkan-production-glsl: role=cursor-janitor phase=parent-terminated pid={} parent_pid={parent_pid}",
                std::process::id()
            ));
            break; // parent terminated: final one-shot cleanup below
        }

        if quit_event_present && wait.0 == WAIT_OBJECT_0.0 + 1 {
            // Do not recover immediately. Preserve the established 2 s stable shutdown opportunity plus
            // the janitor-only safety margin first.
            quit_deadline = route_clock_ms().saturating_add(INPUT_JANITOR_QUIT_GRACE_MS);
            continue;
        }

        let message_index = WAIT_OBJECT_0.0 + handles.len() as u32;
        if wait.0 == message_index {
            unsafe {
                let mut msg = MSG::default();
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    if msg.message == WM_HOTKEY && msg.wParam.0 as i32 == INPUT_JANITOR_HOTKEY_ID {
                        rescue_until = route_clock_ms().saturating_add(INPUT_JANITOR_REASSERT_MS);
                        // Explicit emergency key: the user asked for the last
                        // resort now, so cursor + captured source state may be
                        // restored even while the parent remains alive.
                        janitor_restore_cursor(&mut mag_initialized);
                        janitor_restore_source(parent_pid);
                    }
                }
            }
        }
        if rescue_until != 0 && route_clock_ms() < rescue_until {
            janitor_restore_cursor(&mut mag_initialized);
        }
    }

    // Normal close, crash and Task-Manager termination all converge here.
    // Reapply only Neo-owned external state once, then forget the snapshot.
    janitor_restore_cursor(&mut mag_initialized);
    janitor_restore_source(parent_pid);
    let _ = std::fs::remove_file(janitor_recovery_path(parent_pid));
    unsafe {
        let _ = UnregisterHotKey(None, INPUT_JANITOR_HOTKEY_ID);
        for handle in handles {
            let _ = CloseHandle(handle);
        }
        if mag_initialized {
            let _ = MagUninitialize();
        }
    }
    record_production_janitor_breadcrumb(&format!(
        "vulkan-production-glsl: role=cursor-janitor phase=exit-complete pid={} parent_pid={parent_pid}",
        std::process::id()
    ));
}

/// Spawn the companion as the same portable executable. No extra binary or
/// installed service is required; the child exits as soon as this process does.
pub fn spawn_cursor_janitor() {
    let Ok(exe) = std::env::current_exe() else {
        log::warn!("cursor-janitor-spawn-skipped: current_exe unavailable");
        return;
    };
    let parent_pid = std::process::id();
    // Create the named auto-reset event before launching the helper so a very
    // early Ctrl+Alt+Q / WM_CLOSE cannot race janitor initialization.
    if let Some(event) = create_janitor_quit_event(parent_pid) {
        let raw = event.0 as isize;
        if JANITOR_QUIT_EVENT_HANDLE.set(raw).is_err() {
            unsafe {
                let _ = CloseHandle(event);
            }
        }
    } else {
        log::warn!("cursor-janitor-quit-event-create-failed");
    }
    let arg = format!("--cursor-janitor={parent_pid}");
    match std::process::Command::new(exe).arg(arg).spawn() {
        Ok(child) => {
            log::info!("cursor-janitor-started: pid={}", child.id());
            record_production_janitor_breadcrumb(&format!(
                "vulkan-production-glsl: role=main phase=janitor-spawned pid={} janitor_pid={}",
                std::process::id(),
                child.id()
            ));
        }
        Err(error) => log::warn!("cursor-janitor-spawn-failed: {error}"),
    }
}

/// Clamp a point to the inclusive pixel bounds of a monitor rectangle.
/// Kept pure so the exact 1920x1080 regression can be unit-tested without Win32.
fn clamp_point_to_monitor_rect(pos: (i32, i32), monitor: Rect) -> (i32, i32) {
    if monitor.w <= 0 || monitor.h <= 0 {
        return pos;
    }
    let right = monitor.x.saturating_add(monitor.w.saturating_sub(1));
    let bottom = monitor.y.saturating_add(monitor.h.saturating_sub(1));
    (
        pos.0.clamp(monitor.x, right),
        pos.1.clamp(monitor.y, bottom),
    )
}

/// Return a SetCursorPos-reachable desktop pixel. If `pos` is already on any
/// monitor it is preserved exactly; if it lies beyond an outer screen edge or
/// in a monitor gap, Windows' nearest monitor is used and the point is clamped
/// to that monitor's real rcMonitor (not the work area).
fn reachable_desktop_cursor_point(pos: (i32, i32)) -> (i32, i32) {
    unsafe {
        let monitor = MonitorFromPoint(POINT { x: pos.0, y: pos.1 }, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(monitor, &mut info).as_bool() {
            let r = info.rcMonitor;
            return clamp_point_to_monitor_rect(
                pos,
                Rect {
                    x: r.left,
                    y: r.top,
                    w: r.right.saturating_sub(r.left),
                    h: r.bottom.saturating_sub(r.top),
                },
            );
        }
    }
    pos
}

/// Pure visual clamp for the layered sprite window. `target` is the logical
/// cursor point; only the HWND top-left is adjusted. Keep the ENTIRE sprite
/// inside the nearest physical monitor so an edge/corner exit can never make
/// the Neo cursor look absent. Input geometry is deliberately untouched.
fn clamp_sprite_origin_to_monitor(target: (i32, i32), monitor: Rect, sprite_px: i32) -> (i32, i32) {
    if monitor.w <= 0 || monitor.h <= 0 {
        return target;
    }
    let sprite_px = sprite_px.max(1);
    let right_exclusive = monitor.x.saturating_add(monitor.w);
    let bottom_exclusive = monitor.y.saturating_add(monitor.h);
    let max_x = right_exclusive.saturating_sub(sprite_px).max(monitor.x);
    let max_y = bottom_exclusive.saturating_sub(sprite_px).max(monitor.y);
    // One rectangle rule for all four edges/corners: clamp the sprite WINDOW,
    // not the logical/hardware cursor. Left/top naturally remain at monitor.x/y;
    // right/bottom pull the HWND inward by its full size. SetCursorPos,
    // ClipCursor and all content/source mapping still use the exact target.
    (
        target.0.clamp(monitor.x, max_x),
        target.1.clamp(monitor.y, max_y),
    )
}

/// Physical top-left used only for the layered cursor sprite window. The
/// logical cursor target remains exact in SPRITE_TARGET and in all input
/// mapping. At every monitor edge keep the full arrow HWND visible while the
/// logical/hardware point remains exactly at that edge.
fn sprite_window_visual_position(pos: (i32, i32)) -> (i32, i32) {
    unsafe {
        let monitor = MonitorFromPoint(POINT { x: pos.0, y: pos.1 }, MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if !GetMonitorInfoW(monitor, &mut info).as_bool() {
            return pos;
        }
        let r = info.rcMonitor;
        clamp_sprite_origin_to_monitor(
            pos,
            Rect {
                x: r.left,
                y: r.top,
                w: r.right.saturating_sub(r.left),
                h: r.bottom.saturating_sub(r.top),
            },
            SPRITE_SIZE_PX.load(Ordering::Relaxed),
        )
    }
}

#[derive(Clone, Copy)]
struct PendingCursorReveal {
    at: std::time::Instant,
    pos: (i32, i32),
    attempts: u8,
}

static PENDING_CURSOR_REVEAL: OnceLock<Mutex<Option<PendingCursorReveal>>> = OnceLock::new();

fn cursor_reassert_want_hidden(state_wants_hidden: bool, reveal_pending: bool) -> bool {
    state_wants_hidden || reveal_pending
}

/// Called from the HOOK thread: record whether the real cursor should be hidden.
/// Applied asynchronously (within ~1 frame) by the render-engine thread.
pub fn request_cursor_hidden(hidden: bool) {
    if !hidden {
        cancel_cursor_reveal();
        // Keep CURSOR_HIDE_APPLIED true until the Magnification owner thread
        // actually shows the native cursor. This lets the sprite bridge the
        // GUI-entry handoff with no cursor-less frame.
    }
    let previous = WANT_CURSOR_HIDDEN.swap(hidden, Ordering::AcqRel);
    if hidden && !previous && !CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
        // The native cursor is currently visible, so keep the sprite gated off
        // until the owner confirms the hide. If APPLIED is already true (rapid
        // GUI<->content reversal), native never became visible; preserve the
        // sprite bridge instead of creating a blank boundary frame.
        post_sprite_update();
    }
}

fn pending_cursor_reveal() -> &'static Mutex<Option<PendingCursorReveal>> {
    PENDING_CURSOR_REVEAL.get_or_init(|| Mutex::new(None))
}

fn defer_cursor_reveal(pos: (i32, i32)) {
    let safe_pos = reachable_desktop_cursor_point(pos);
    if safe_pos != pos {
        log::info!(
            "cursor-reveal target clamped to reachable desktop: requested=({},{}) safe=({},{})",
            pos.0,
            pos.1,
            safe_pos.0,
            safe_pos.1
        );
    }
    let reveal_at =
        std::time::Instant::now() + std::time::Duration::from_millis(EDGE_TRANSFER_DEFER_MS);
    *pending_cursor_reveal().lock().unwrap() = Some(PendingCursorReveal {
        at: reveal_at,
        pos: safe_pos,
        attempts: 0,
    });
    request_cursor_hidden(true);

    // Capture-wide sprite mode never needs MagShowSystemCursor(true) for an
    // edge exit; it only needs the hidden real cursor moved out of source
    // space. Do that on the hook-owned sprite window timer so a cold
    // TensorRT/ONNX render-thread stall cannot freeze this transaction.
    if capture_sprite_active() {
        if let Some(&h) = SPRITE_HWND.get() {
            unsafe {
                let _ = SetTimer(
                    Some(HWND(h as *mut _)),
                    CURSOR_EDGE_TRANSFER_TIMER_ID,
                    EDGE_TRANSFER_DEFER_MS as u32,
                    None,
                );
            }
        }
    }
}

fn cancel_cursor_reveal() {
    *pending_cursor_reveal().lock().unwrap() = None;
}

fn cursor_reveal_pending() -> bool {
    pending_cursor_reveal().lock().unwrap().is_some()
}

/// Complete a capture-wide sprite edge transfer on the mouse-hook thread,
/// outside the WH_MOUSE_LL callback itself. This path deliberately does not
/// call Magnification APIs; the native cursor remains hidden and Neo's sprite
/// stays the visual owner. The render/ONNX thread may be blocked for seconds
/// during the first TensorRT inference without affecting cursor escape.
fn complete_capture_sprite_cursor_transfer_on_hook_thread() {
    if !capture_sprite_active() {
        return;
    }

    let now = std::time::Instant::now();
    let pending = {
        let guard = pending_cursor_reveal().lock().unwrap();
        *guard
    };
    let Some(p) = pending else {
        return;
    };
    if now < p.at {
        if let Some(&h) = SPRITE_HWND.get() {
            unsafe {
                let _ = SetTimer(
                    Some(HWND(h as *mut _)),
                    CURSOR_EDGE_TRANSFER_TIMER_ID,
                    EDGE_TRANSFER_DEFER_MS as u32,
                    None,
                );
            }
        }
        return;
    }

    let (reached, actual) =
        warp_unclipped_verified(p.pos, "capture sprite hook-timer edge transfer");
    if reached {
        *pending_cursor_reveal().lock().unwrap() = None;
        WANT_CURSOR_HIDDEN.store(true, Ordering::Release);
        sprite_move_now(actual.0, actual.1, true);
        log::debug!(
            "cursor-desktop-transfer verified: target=({},{}) actual=({},{}) visual_owner=sprite executor=hook-timer",
            p.pos.0,
            p.pos.1,
            actual.0,
            actual.1
        );
        return;
    }

    let mut exhausted = false;
    let mut attempts = 0u8;
    {
        let mut guard = pending_cursor_reveal().lock().unwrap();
        if let Some(mut current) = *guard {
            current.attempts = current.attempts.saturating_add(1);
            attempts = current.attempts;
            if current.attempts >= CURSOR_REVEAL_MAX_ATTEMPTS {
                *guard = None;
                exhausted = true;
            } else {
                current.at = std::time::Instant::now()
                    + std::time::Duration::from_millis(EDGE_TRANSFER_DEFER_MS);
                *guard = Some(current);
            }
        }
    }

    let safe_actual = reachable_desktop_cursor_point(actual);
    sprite_move_now(safe_actual.0, safe_actual.1, true);
    WANT_CURSOR_HIDDEN.store(true, Ordering::Release);
    if exhausted {
        log::error!(
            "cursor-edge-transfer fail-safe released pending state after {} attempts: target=({},{}) actual=({},{}) safe_actual=({},{}) executor=hook-timer",
            attempts,
            p.pos.0,
            p.pos.1,
            actual.0,
            actual.1,
            safe_actual.0,
            safe_actual.1
        );
    } else if let Some(&h) = SPRITE_HWND.get() {
        unsafe {
            let _ = SetTimer(
                Some(HWND(h as *mut _)),
                CURSOR_EDGE_TRANSFER_TIMER_ID,
                EDGE_TRANSFER_DEFER_MS as u32,
                None,
            );
        }
        log::warn!(
            "cursor-edge-transfer deferred: attempt={}/{} target=({},{}) actual=({},{}) executor=hook-timer",
            attempts,
            CURSOR_REVEAL_MAX_ATTEMPTS,
            p.pos.0,
            p.pos.1,
            actual.0,
            actual.1
        );
    }
}

fn restore_configured_system_cursors() {
    unsafe {
        let _ = SystemParametersInfoW(
            SPI_SETCURSORS,
            0,
            None,
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
    }
}

thread_local! {
    static MAG_ON_THIS_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static LAST_CURSOR_ASSERT: std::cell::Cell<Option<std::time::Instant>> = const { std::cell::Cell::new(None) };
    static LAST_CURSOR_WANT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Drive the real system-cursor hide/show to match request_cursor_hidden().
/// MUST be called repeatedly from the thread that owns the overlay window and
/// pumps its messages (the render-engine thread) — see the module note above for
/// why the hook thread cannot do this. Idempotent + cheap; rate-limited so a
/// high frame rate does not spam MagShowSystemCursor. Re-asserts on a ~40ms
/// heartbeat because a game/other app can reset the hide on a cursor-shape change.
pub fn pump_cursor_visibility() {
    CURSOR_OWNER_HEARTBEAT_MS.store(route_clock_ms(), Ordering::Release);
    // A latched rescue always wins over ordinary cursor ownership until a new
    // capture session explicitly rearms it. This prevents a recovered render
    // thread from immediately hiding the cursor again after the watchdog fired.
    if INPUT_FAILSAFE_LATCHED.load(Ordering::Acquire) {
        WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
    }
    // Initialize and repair Magnification only on its owning engine thread.
    let reinit_requested = MAG_REINIT_REQUESTED.swap(false, Ordering::AcqRel);
    MAG_ON_THIS_THREAD.with(|f| {
        if reinit_requested && f.get() {
            unsafe {
                let _ = MagUninitialize();
            }
            f.set(false);
        }
        if !f.get() {
            let ok = unsafe { MagInitialize() }.as_bool();
            log::info!("render-engine thread: MagInitialize={ok} (owns the real-cursor hide)");
            f.set(ok);
            if !ok {
                MAG_REINIT_REQUESTED.store(true, Ordering::Release);
            }
        }
    });
    if reinit_requested {
        LAST_CURSOR_ASSERT.with(|c| c.set(None));
    }
    let native_gui_owner = NATIVE_GUI_OWNER.load(Ordering::Acquire);
    let capture_sprite_mode = capture_sprite_active();
    let external_native_owner = external_native_cursor_owner(native_gui_owner);
    let mut want_hidden = WANT_CURSOR_HIDDEN.load(Ordering::Relaxed);
    if INPUT_FAILSAFE_LATCHED.load(Ordering::Acquire) {
        want_hidden = false;
        WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
    } else if capture_sprite_mode && !external_native_owner {
        want_hidden = true;
        WANT_CURSOR_HIDDEN.store(true, Ordering::Release);
    }
    // While capture is active, native GUI ownership deliberately keeps the
    // system cursor hidden and uses Neo's sprite at the real GUI point. Native
    // hit-testing remains unchanged; outside capture, retain the normal native
    // cursor path.
    let gui_sprite_mode = active_native_gui_sprite_mode(native_gui_owner);
    if native_gui_owner != 0 {
        want_hidden = gui_sprite_mode;
        WANT_CURSOR_HIDDEN.store(gui_sprite_mode, Ordering::Release);
    }
    let mut reveal_due = false;
    let mut reveal_target = None;
    {
        let pending = pending_cursor_reveal().lock().unwrap();
        if let Some(p) = *pending {
            if !capture_sprite_mode && std::time::Instant::now() >= p.at {
                reveal_target = Some(p.pos);
            }
            // Capture-wide sprite edge transfers are completed by the
            // hook-window timer. Native-reveal cases still use this
            // Magnification-owner thread.
            want_hidden = true;
        }
    }
    if let Some(pos) = reveal_target {
        let (reached, actual) = warp_unclipped_verified(pos, "deferred cursor reveal");
        if reached {
            *pending_cursor_reveal().lock().unwrap() = None;
            if capture_sprite_mode {
                reveal_due = false;
                want_hidden = true;
                WANT_CURSOR_HIDDEN.store(true, Ordering::Relaxed);
                sprite_move_now(actual.0, actual.1, true);
                log::debug!(
                    "cursor-desktop-transfer verified: target=({},{}) actual=({},{}) visual_owner=sprite stable-deferred",
                    pos.0,
                    pos.1,
                    actual.0,
                    actual.1
                );
            } else {
                reveal_due = true;
                want_hidden = false;
                WANT_CURSOR_HIDDEN.store(false, Ordering::Relaxed);
                log::debug!(
                    "cursor-reveal verified: target=({},{}) actual=({},{})",
                    pos.0,
                    pos.1,
                    actual.0,
                    actual.1
                );
            }
        } else {
            let mut exhausted = false;
            let mut attempts = 0u8;
            {
                let mut pending = pending_cursor_reveal().lock().unwrap();
                if let Some(mut p) = *pending {
                    p.attempts = p.attempts.saturating_add(1);
                    attempts = p.attempts;
                    if p.attempts >= CURSOR_REVEAL_MAX_ATTEMPTS {
                        *pending = None;
                        exhausted = true;
                    } else {
                        p.at = std::time::Instant::now()
                            + std::time::Duration::from_millis(EDGE_TRANSFER_DEFER_MS);
                        *pending = Some(p);
                    }
                }
            }
            if exhausted {
                // Fail visible instead of holding the desktop cursor hidden
                // forever. During active capture the Neo sprite remains the
                // visual owner at the best OS-reported reachable point. Outside
                // capture, allow the native cursor to be shown there.
                let safe_actual = reachable_desktop_cursor_point(actual);
                sprite_move_now(safe_actual.0, safe_actual.1, true);
                if capture_sprite_mode {
                    want_hidden = true;
                    WANT_CURSOR_HIDDEN.store(true, Ordering::Relaxed);
                } else {
                    reveal_due = true;
                    want_hidden = false;
                    WANT_CURSOR_HIDDEN.store(false, Ordering::Relaxed);
                }
                log::error!(
                    "cursor-reveal fail-safe released pending state after {} attempts: target=({},{}) actual=({},{}) safe_actual=({},{}) capture_sprite={}",
                    attempts,
                    pos.0,
                    pos.1,
                    actual.0,
                    actual.1,
                    safe_actual.0,
                    safe_actual.1,
                    capture_sprite_mode
                );
            } else {
                want_hidden = true;
                WANT_CURSOR_HIDDEN.store(true, Ordering::Relaxed);
                log::warn!(
                    "cursor-reveal deferred: attempt={}/{} target=({},{}) actual=({},{})",
                    attempts,
                    CURSOR_REVEAL_MAX_ATTEMPTS,
                    pos.0,
                    pos.1,
                    actual.0,
                    actual.1
                );
            }
        }
    }
    // Re-check ownership after deferred-reveal processing. GUI ownership is
    // final authority, but during active capture that authority is now the
    // deterministic GUI-sprite contract rather than native visual reveal.
    if native_gui_owner != 0 {
        *pending_cursor_reveal().lock().unwrap() = None;
        reveal_due = false;
        want_hidden = gui_sprite_mode;
        WANT_CURSOR_HIDDEN.store(gui_sprite_mode, Ordering::Release);
    }
    if capture_sprite_mode && !external_native_owner {
        want_hidden = true;
        WANT_CURSOR_HIDDEN.store(true, Ordering::Release);
    }
    // v382: GUI caption drag is no longer a native-reveal exception. Active
    // capture keeps Neo sprite ownership independent of where the title bar was grabbed.
    if tensorrt_build_native_cursor_guard() {
        // Final authority during a build epoch: stale timer/GUI requests must
        // never re-hide native even if they were queued before the guard rose.
        want_hidden = false;
        WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
    }
    let changed = LAST_CURSOR_WANT.with(|c| c.replace(want_hidden)) != want_hidden;
    let due = LAST_CURSOR_ASSERT.with(|c| match c.get() {
        Some(t) => t.elapsed() >= std::time::Duration::from_millis(40),
        None => true,
    });
    if !changed && !due && !reveal_due {
        return;
    }
    LAST_CURSOR_ASSERT.with(|c| c.set(Some(std::time::Instant::now())));

    // Native reveal must never overlap the white Neo cursor sprite. v348f
    // physically hid the sprite before MagShowSystemCursor(true), but the LL
    // mouse hook could still arrive in that tiny interval and issue another
    // sprite_move(..., true). That explains the remaining one-frame white
    // cursor, often slightly offset because the async sprite position was one
    // mouse sample behind the native cursor. v348g makes the reveal a real
    // transaction: all sprite show paths are atomically blocked until native
    // reveal either succeeds or fails.
    let native_reveal_transition = !want_hidden && CURSOR_HIDE_APPLIED.load(Ordering::Acquire);
    let desktop_bridge_active =
        native_reveal_transition && DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire);
    let bridge_sprite_for_native_reveal =
        native_reveal_transition && SPRITE_SHOW.load(Ordering::Acquire);
    if native_reveal_transition && !desktop_bridge_active {
        SPRITE_NATIVE_REVEAL_BLOCK.store(true, Ordering::Release);
        // Ordinary native reveal keeps the strict no-overlap transaction.
        // Desktop exit is the fail-visible bridge; GUI caption drag no longer
        // enters this native-reveal path in v382.
        sprite_hide_window_only();
    }

    let reveal_started = std::time::Instant::now();
    let ok = set_system_cursor_visible(!want_hidden);
    // GetCursorInfo/CURSOR_SHOWING tracks the ordinary ShowCursor display
    // state and does NOT reflect MagShowSystemCursor(false). Keep it only as
    // a diagnostic signal; never use it to decide whether Magnification hide
    // succeeded (v372 did that and could hide both native and Neo sprite).
    let refcount_visible = system_cursor_showing();
    if changed || reveal_due || !ok {
        let actual = read_cursor_pos_or((i32::MIN, i32::MIN), "cursor visibility safe apply");
        log::info!(
            "cursor-visibility-safe: request_hidden={} mag_ok={} showcursor_refcount_visible={} diagnostic_only=true hide_applied_before={} owner={:#x} actual=({},{})",
            want_hidden,
            ok,
            refcount_visible,
            CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
            native_gui_owner,
            actual.0,
            actual.1
        );
    }
    if want_hidden {
        // Defensive reset: a hidden-cursor request is outside the native-reveal
        // transaction, so the sprite may become visible after Mag hide succeeds.
        SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
        if ok {
            if !CURSOR_HIDE_APPLIED.swap(true, Ordering::AcqRel) {
                // The initial engage move recorded the sprite position but kept
                // its window hidden. Reveal it only after native hide succeeds.
                post_sprite_update();
                log::debug!("cursor handoff committed: native hidden -> sprite visible");
            }
        } else {
            CURSOR_HIDE_APPLIED.store(false, Ordering::Release);
            post_sprite_update();
        }
    } else {
        if ok {
            // Once native reveal has succeeded, close the bridge atomically:
            // block any queued sprite re-show, publish native ownership, then
            // synchronously remove the bridge window.
            if desktop_bridge_active {
                SPRITE_NATIVE_REVEAL_BLOCK.store(true, Ordering::Release);
            }
            let was_hidden = CURSOR_HIDE_APPLIED.swap(false, Ordering::AcqRel);
            let desktop_bridge = DESKTOP_REVEAL_BRIDGE_ACTIVE.swap(false, Ordering::AcqRel);
            if desktop_bridge {
                let actual =
                    read_cursor_pos_or((i32::MIN, i32::MIN), "desktop reveal bridge complete");
                log::info!(
                    "desktop-reveal-bridge-complete: native_visible=true actual=({},{}) bridge_ms={}",
                    actual.0,
                    actual.1,
                    reveal_started.elapsed().as_millis()
                );
            }
            if was_hidden {
                SPRITE_SHOW.store(false, Ordering::Release);
                sprite_hide_window_only();
            }
            if native_reveal_transition {
                SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
            }
        } else {
            // If show failed, native is still presumed hidden. Re-open sprite
            // visibility first, then restore the bridge immediately and retry on
            // the next owner-thread heartbeat. This avoids trading the flash for
            // a cursor-less frame.
            if native_reveal_transition {
                SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
            }
            if bridge_sprite_for_native_reveal {
                sprite_restore_window_if_requested_now();
            }
            log::warn!("native cursor reveal deferred: MagShowSystemCursor(true) failed");
        }
        if reveal_due && ok {
            sprite_hide();
            let showing = system_cursor_showing();
            unsafe {
                let mut pt = POINT::default();
                let _ = GetCursorPos(&mut pt);
                log::info!(
                    "cursor-reveal shown: MagShowSystemCursor(true).ok={ok} actual=({},{}) cursor_showing={showing}",
                    pt.x,
                    pt.y
                );
            }
        }
    }
    let previous_owner = LAST_CURSOR_CONTRACT_OWNER.swap(native_gui_owner, Ordering::AcqRel);
    if previous_owner != native_gui_owner {
        let actual = read_cursor_pos_or((i32::MIN, i32::MIN), "cursor ownership contract");
        let top = if actual.0 == i32::MIN {
            0
        } else {
            crate::platform::win32::direct_top_level_window_at_point(actual.0, actual.1)
        };
        log::debug!(
            "cursor-ownership-contract: native_gui={:#x} previous={:#x} want_hidden={} mag_ok={} hide_applied={} actual=({},{}) top_hwnd={:#x}",
            native_gui_owner,
            previous_owner,
            want_hidden,
            ok,
            CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
            actual.0,
            actual.1,
            top
        );
    }
}

/// Move the hidden real cursor to its mapped source point BEFORE applying the
/// source clip. ClipCursor immediately clamps an out-of-rect cursor to the
/// nearest corner; applying it first occasionally left the cursor parked at the
/// source top-left when SetCursorPos lost the race, producing the visible
/// "jump upward". Warp-first then clip is atomic from the user's perspective
/// because the native cursor is already hidden at this point.
fn warp_unclipped_verified(target: (i32, i32), reason: &str) -> (bool, (i32, i32)) {
    unsafe {
        let mut actual = read_cursor_pos_or(target, reason);
        for attempt in 1..=4 {
            // ClipCursor can remain effective for a scheduling turn after an
            // ownership transition. Clear it before EVERY attempt, then trust
            // GetCursorPos rather than SetCursorPos's return value alone.
            let unclipped = ClipCursor(None).is_ok();
            let moved = SetCursorPos(target.0, target.1).is_ok();
            actual = read_cursor_pos_or(actual, reason);
            let reached = (actual.0 - target.0).abs() <= 1 && (actual.1 - target.1).abs() <= 1;
            if unclipped && moved && reached {
                if attempt > 1 {
                    log::info!(
                        "cursor warp recovered: reason={reason} attempt={attempt} target=({},{}) actual=({},{})",
                        target.0,
                        target.1,
                        actual.0,
                        actual.1
                    );
                }
                return (true, actual);
            }
            if attempt < 4 {
                std::thread::yield_now();
            }
        }
        log::warn!(
            "cursor warp verification failed: reason={reason} target=({},{}) actual=({},{})",
            target.0,
            target.1,
            actual.0,
            actual.1
        );
        (false, actual)
    }
}

fn warp_then_clip_source(target: (i32, i32), src: Rect) -> (bool, (i32, i32)) {
    unsafe {
        let (warped, actual) = warp_unclipped_verified(target, "source handoff warp verify");
        if !warped {
            return (false, actual);
        }

        let clip = clip_rect(src);
        if ClipCursor(Some(&clip)).is_err() {
            let _ = ClipCursor(None);
            return (false, actual);
        }
        let after_clip = read_cursor_pos_or(actual, "source handoff clip verify");
        let reached = (after_clip.0 - target.0).abs() <= 1 && (after_clip.1 - target.1).abs() <= 1;
        if !reached {
            // Never leave a rejected corner clamp active. Keeping the native
            // cursor hidden and unclipped is safer than exposing a source-corner
            // position as a visible top-left jump.
            let _ = ClipCursor(None);
        }
        (reached, after_clip)
    }
}

/// Commit a normal GUI -> magnified-content cursor handoff after the
/// Magnification owner thread has successfully hidden the native cursor.
/// Called immediately after pump_cursor_visibility() on the render thread.
pub fn pump_cursor_engage_commit() {
    if !CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
        return;
    }
    let Ok(mut g) = state().try_lock() else {
        return;
    };
    let Some(pending) = g.pending_engage else {
        return;
    };
    if !g.active || !g.engaged || !g.cursor_hidden || g.src != pending.src {
        log::info!(
            "cursor handoff cancelled before commit: active={} engaged={} hidden={} geometry_match={}",
            g.active,
            g.engaged,
            g.cursor_hidden,
            g.src == pending.src
        );
        g.pending_engage = None;
        g.engaged = false;
        publish_deferred_source_engaged(false);
        g.expect_teleport = None;
        g.teleport_guard_until = None;
        g.cursor_hidden = false;
        set_own_main_gui_passthrough(&g, false, "engage-cancelled");
        request_cursor_hidden(false);
        sprite_hide();
        pump_cursor_visibility();
        return;
    }

    let (warp_applied, warp_actual) = warp_then_clip_source(pending.target, pending.src);

    if !warp_applied {
        log::warn!(
            "cursor handoff commit warp rejected: requested=({},{}) actual=({},{}) src={:?}",
            pending.target.0,
            pending.target.1,
            warp_actual.0,
            warp_actual.1,
            pending.src
        );
        let (origin_restored, origin_actual) =
            warp_unclipped_verified(pending.origin, "engage abort native restore");
        g.pending_engage = None;
        g.engaged = false;
        publish_deferred_source_engaged(false);
        g.expect_teleport = None;
        g.teleport_guard_until = None;
        g.cursor_hidden = false;
        set_own_main_gui_passthrough(&g, false, "engage-warp-rejected");
        if origin_restored {
            sprite_move_now(origin_actual.0, origin_actual.1, true);
            request_cursor_hidden(false);
        } else {
            // Keep the sprite at the intended visible origin and let the Mag
            // owner retry the native placement before revealing it.
            sprite_move_now(pending.origin.0, pending.origin.1, true);
            defer_cursor_reveal(pending.origin);
        }
        restore_mouse_speed();
        g.cooldown_until =
            Some(std::time::Instant::now() + std::time::Duration::from_millis(REENTER_COOLDOWN_MS));
        pump_cursor_visibility();
        return;
    }

    if g.adjust_speed {
        let zoom = (g.content.w as f64 / pending.src.w.max(1) as f64)
            .max(g.content.h as f64 / pending.src.h.max(1) as f64);
        slow_mouse_for_zoom(zoom);
    }
    g.pending_engage = None;
    let commit_now = std::time::Instant::now();
    mark_engage_committed(&mut g, pending.target, commit_now);
    sprite_move_now(pending.sprite.0, pending.sprite.1, true);
    log::info!(
        "cursor handoff committed atomically: native-hidden target-source=({},{}) sprite=({},{}) arm_to_commit_ms={} stale_guard_ms={}",
        pending.target.0,
        pending.target.1,
        pending.sprite.0,
        pending.sprite.1,
        commit_now.duration_since(pending.requested_at).as_millis(),
        POST_COMMIT_STALE_GUARD_MS
    );
}

// ---------------- pointer-speed matching (the cursor-routing design parity) ----------------
//
// While engaged the real cursor is confined to the (smaller) source, so a raw
// hand movement of H px moves the SPRITE H*zoom px on the magnified view — the
// sprite is `zoom`× faster than the hand. When the cursor slips out at an edge
// it reverts to 1:1 desktop speed. That abrupt speed change at the border is
// an edge-return jerk. The cursor-routing design compensates by slowing the OS
// pointer speed by 1/zoom while engaged, so the sprite tracks the hand 1:1 in
// BOTH states — the edge crossing is then perfectly continuous. We save the
// user's speed the first time we slow it and always restore it (also from
// emergency_release_all), using fwinini=0 so nothing is broadcast or persisted.

/// Windows mouse-speed slider (1..=20) → pointer multiplier.
const SPEED_MULT: [f64; 20] = [
    0.03125, 0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25,
    2.5, 2.75, 3.0, 3.25, 3.5,
];
/// The user's pointer speed while we have it slowed (0 = we are not slowing it).
static SAVED_MOUSE_SPEED: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

fn get_mouse_speed() -> i32 {
    let mut val: i32 = 10;
    unsafe {
        let _ = SystemParametersInfoW(
            SPI_GETMOUSESPEED,
            0,
            Some(&mut val as *mut i32 as *mut core::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
    }
    val.clamp(1, 20)
}

fn set_mouse_speed(speed: i32) {
    unsafe {
        // for SPI_SETMOUSESPEED the value IS passed in the pvparam slot
        let _ = SystemParametersInfoW(
            SPI_SETMOUSESPEED,
            0,
            Some(speed.clamp(1, 20) as usize as *mut core::ffi::c_void),
            SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
        );
    }
}

/// slider whose multiplier best matches current/zoom (the cursor-routing design).
fn slow_speed_for_zoom(current: i32, zoom: f64) -> i32 {
    let cur_mult = SPEED_MULT[(current.clamp(1, 20) - 1) as usize];
    let target = cur_mult / zoom.max(1e-6);
    let mut best = 0usize;
    let mut best_diff = f64::INFINITY;
    for (i, &m) in SPEED_MULT.iter().enumerate() {
        let d = (m - target).abs();
        if d < best_diff {
            best_diff = d;
            best = i;
        }
    }
    (best + 1) as i32
}

/// Crash-safety: the original speed is also written here while slowed, so a
/// hard kill (Task Manager) mid-engagement can be healed on the next start
/// (in-process restore paths cover every clean exit).
fn speed_backup_path() -> std::path::PathBuf {
    crate::core::config::app_dir()
        .join("cache")
        .join("chidescaler_neo_ptr_speed.bak")
}

/// v343f and earlier stored the crash-recovery marker in %TEMP%.
/// Never write there again, but heal/remove an old marker once so upgrades
/// cannot leave a previously slowed pointer behind.
fn legacy_speed_backup_path() -> std::path::PathBuf {
    std::env::temp_dir().join("chidescaler_neo_ptr_speed.bak")
}

/// Slow the pointer by 1/zoom so the sprite tracks the hand 1:1. No-op if the
/// magnification is negligible or we are already slowing it.
fn slow_mouse_for_zoom(zoom: f64) {
    if zoom <= 1.05 {
        return;
    }
    if SAVED_MOUSE_SPEED.load(Ordering::Acquire) != 0 {
        return; // already slowed
    }
    let cur = get_mouse_speed();
    let slowed = slow_speed_for_zoom(cur, zoom);
    if slowed != cur {
        let backup = speed_backup_path();
        if let Some(parent) = backup.parent() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                log::warn!(
                    "pointer-speed adjustment skipped: portable backup directory unavailable path={} error={error}",
                    parent.display()
                );
                return;
            }
        }
        if let Err(error) = std::fs::write(&backup, cur.to_string()) {
            log::warn!(
                "pointer-speed adjustment skipped: portable backup unavailable path={} error={error}",
                backup.display()
            );
            return;
        }
        SAVED_MOUSE_SPEED.store(cur, Ordering::Release);
        set_mouse_speed(slowed);
    }
}

/// Restore the user's pointer speed if we slowed it.
fn restore_mouse_speed() {
    let saved = SAVED_MOUSE_SPEED.swap(0, Ordering::AcqRel);
    if saved != 0 {
        set_mouse_speed(saved);
        let _ = std::fs::remove_file(speed_backup_path());
    }
}

/// If a previous run was hard-killed while it had the pointer slowed, restore
/// the user's speed from the backup file. Called once at hook startup.
fn heal_leftover_mouse_speed() {
    for p in [speed_backup_path(), legacy_speed_backup_path()] {
        if let Ok(s) = std::fs::read_to_string(&p) {
            if let Ok(v) = s.trim().parse::<i32>() {
                if (1..=20).contains(&v) {
                    set_mouse_speed(v);
                    log::warn!(
                        "restored pointer speed {v} left slowed by a previous run path={}",
                        p.display()
                    );
                }
            }
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// What the applier should do after planning an engaged move.
#[derive(Debug, PartialEq)]
enum MovePlan {
    /// stay engaged; redraw the sprite at the virtual cursor
    Stay,
    /// disengage and place the REAL cursor at `place` (edge escape)
    Escape {
        place: (i32, i32),
        sprite: (i32, i32),
    },
}

fn mark_engage_committed(g: &mut State, target: (i32, i32), now: std::time::Instant) {
    publish_deferred_source_engaged(
        g.active && !g.window_frame_input && g.src_hwnd != 0 && g.pending_engage.is_none(),
    );
    g.native_gui_owner_hwnd = 0;
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
    // The engage is not real until the owner thread has hidden the native
    // cursor and SetCursorPos has actually reached the source target. Start all
    // stale-event timing from THIS moment, not from the earlier LL-hook arm.
    g.last_engage_at = Some(now);
    g.last_set = target;
    g.last_hw = target;
    g.expect_teleport = Some(target);
    g.teleport_guard_until =
        Some(now + std::time::Duration::from_millis(POST_COMMIT_STALE_GUARD_MS));
    g.edge_release_settle_until = None;
}

fn suppress_post_commit_stale_move(
    g: &mut State,
    px: i32,
    py: i32,
    actual: (i32, i32),
    now: std::time::Instant,
) -> bool {
    let Some(until) = g.teleport_guard_until else {
        return false;
    };
    if now >= until {
        g.teleport_guard_until = None;
        return false;
    }
    // Source dragging must remain immediate. Outside a drag, a raw hook point
    // hundreds of pixels away from the clipped OS cursor immediately after a
    // commit is an old pre-warp screen-space event, not genuine source motion.
    if g.buttons_down != 0 {
        return false;
    }
    let dx = (px - actual.0).abs();
    let dy = (py - actual.1).abs();
    if dx <= POST_COMMIT_RAW_DIVERGENCE_PX && dy <= POST_COMMIT_RAW_DIVERGENCE_PX {
        return false;
    }
    g.last_hw = actual;
    log::info!(
        "post-commit stale cursor event suppressed: raw=({px},{py}) actual=({},{}) divergence=({dx},{dy}) guard_remaining_ms={}",
        actual.0,
        actual.1,
        until.saturating_duration_since(now).as_millis()
    );
    true
}

/// Pure planner for a hardware move while engaged (stable legacy behavior
/// model). `actual` is the OS-clipped cursor position (GetCursorPos) = the
/// TRUTH for where the confined cursor sits; `px,py` is the raw (pre-clip)
/// hook position, which is what reveals an outward push past the source edge.
///
/// Crucially this NEVER disengages over the control panel / GUI: the panel is
/// reached with the virtual cursor and clicks are redirected in the hook. Only
/// a genuine edge push leaves the view. No Windows calls — unit-testable.
fn plan_engaged(
    g: &mut State,
    px: i32,
    py: i32,
    now: std::time::Instant,
    actual: (i32, i32),
) -> MovePlan {
    // Stale pre-teleport strays (see `expect_teleport`): swallow every move
    // until the injected engage teleport shows up in the hook stream, so a
    // queued screen-coordinate event can never be mapped as a source
    // coordinate and yank the sprite during post-engage handoff.
    // bounce in the 4K GUI-topmost logs). Time-capped by the engage grace so
    // a lost injection can't freeze the cursor.
    if let Some(expected) = g.expect_teleport {
        // A GUI-side stale event can agree perfectly with GetCursorPos after
        // Windows overwrites our SetCursorPos, while both are hundreds of
        // pixels away from the requested SOURCE target. Treating raw==actual
        // as acknowledgement caused that screen coordinate to be remapped as
        // source space and produced the observed jump toward the upper-left.
        let raw_at_target = (px - expected.0).abs() <= TELEPORT_SETTLE_PX
            && (py - expected.1).abs() <= TELEPORT_SETTLE_PX;
        let actual_at_target = (actual.0 - expected.0).abs() <= TELEPORT_SETTLE_PX
            && (actual.1 - expected.1).abs() <= TELEPORT_SETTLE_PX;
        let fresh = raw_at_target && actual_at_target;
        let expired = g.last_engage_at.map_or(true, |t| {
            now.duration_since(t) >= std::time::Duration::from_millis(TELEPORT_SETTLE_TIMEOUT_MS)
        });
        if fresh || expired {
            g.expect_teleport = None;
            if fresh {
                // The first verified event is also real user motion. Base its
                // delta on the warp target and continue through the normal
                // planner; discarding it caused a one-event boundary snag and
                // could leave a fast reversal one pixel short of the GUI.
                g.last_hw = expected;
            } else {
                // Never resume source-space mapping from an unacknowledged
                // teleport. `actual` may still be an old GUI/screen coordinate;
                // rebasing to it can cause a jump toward the desktop's
                // upper-left. Abort to the cursor's current VISIBLE position.
                let visible = (g.virt.0.round() as i32, g.virt.1.round() as i32);
                log::warn!(
                    "cursor teleport settle expired: aborting at visible=({},{}) expected=({},{}) stale_raw=({px},{py}) stale_actual=({},{})",
                    visible.0,
                    visible.1,
                    expected.0,
                    expected.1,
                    actual.0,
                    actual.1
                );
                disengage_state(g, now);
                g.cooldown_until = Some(reenter_cooldown(now, g.oscillations));
                return MovePlan::Escape {
                    place: visible,
                    sprite: visible,
                };
            }
        } else {
            if (px - actual.0).abs() <= TELEPORT_SETTLE_PX
                && (py - actual.1).abs() <= TELEPORT_SETTLE_PX
            {
                log::warn!(
                    "cursor teleport false-ack suppressed: expected=({},{}) raw=({px},{py}) actual=({},{})",
                    expected.0,
                    expected.1,
                    actual.0,
                    actual.1
                );
            }
            return MovePlan::Stay;
        }
    }
    if suppress_post_commit_stale_move(g, px, py, actual, now) {
        return MovePlan::Stay;
    }
    let s = g.src;
    let c = g.content;
    let old_virt = g.virt;
    // CLICK delivery / source sync uses the ACTUAL clipped cursor (confined to
    // the source). The SPRITE, however, follows the RAW pre-clip hook pt mapped
    // WITHOUT clamping — so as the hand pushes past the source edge the visible
    // cursor keeps moving toward/past the content edge, instead of snapping back
    // to the clamped source-edge mapping (map_source_to_content(clipped)) which
    // creates a visible return. Interior events have raw == clipped.
    let ax = actual.0.clamp(s.x, s.x + s.w - 1);
    let ay = actual.1.clamp(s.y, s.y + s.h - 1);
    g.last_set = (ax, ay);
    let mapped = map_source_to_content_unclamped(c, s, px as f64, py as f64);
    let caption_drag_visual = if g.buttons_down & BTN_LEFT != 0 && source_caption_drag_active() {
        source_caption_drag_visual_position()
    } else {
        None
    };
    if let Some((vx, vy)) = caption_drag_visual {
        // Native Windows dragging already keeps the hidden real cursor at a
        // fixed grab offset inside the source caption. Re-mapping that cursor
        // through independently published source/content rectangles introduces
        // a second geometry clock and is exactly what made the visible sprite
        // creep across the title bar. During a caption drag the visual cursor
        // has one authority only: overlay_origin + grab_anchor.
        g.virt = (vx as f64, vy as f64);
    } else {
        g.virt = mapped;
    }
    let prev_hw = g.last_hw;
    g.last_hw = (px, py);

    if caption_drag_visual.is_some() {
        g.edge_out_accum = 0.0;
        return MovePlan::Stay;
    }

    if fullscreen_ui_virtual_move(g, old_virt, (px, py), (px - prev_hw.0, py - prev_hw.1)) {
        g.edge_out_accum = 0.0;
        return MovePlan::Stay;
    }

    // dragging the source with a held button: never escape — the whole gesture
    // belongs to the source window (e.g. dragging its own title bar toward a
    // screen edge must not yank the cursor out).
    if g.buttons_down != 0 {
        g.edge_out_accum = 0.0;
        return MovePlan::Stay;
    }

    // post-engage grace: swallow the stale queued pre-teleport event(s) so they
    // cannot instantly fling the view out (see ENGAGE_GRACE_MS).
    if let Some(t) = g.last_engage_at {
        if now.duration_since(t) < std::time::Duration::from_millis(ENGAGE_GRACE_MS) {
            g.edge_out_accum = 0.0;
            return MovePlan::Stay;
        }
    }

    // v370: the windowed floating panel sits immediately outside the content
    // edge. EXIT_TRAVEL_PX is intentionally one physical pixel for ordinary
    // window-like escape, but that also meant the first pixel toward the panel
    // escaped to desktop before `post_ui_hover_from_state()` could hand native
    // ownership to the panel. Give ONLY the panel's exact visible rectangle
    // priority over edge-release; there is no halo, timer or hidden margin.
    let vx = g.virt.0.round() as i32;
    let vy = g.virt.1.round() as i32;
    // A collapsed panel is a deliberately alpha=0 rediscovery region. A fully
    // transparent layered HWND can be omitted by WindowFromPoint, so the normal
    // native ownership classifier cannot be the wake source for this one state.
    // Publish the exact virtual-cursor point to the Win32 lurk-hover bridge; it
    // wakes the root only on an outside->inside transition, never per mouse event.
    crate::platform::win32::update_panel_lurk_virtual_hover(vx, vy);
    if let Some(hit) = ui_target_at(g, vx, vy, 0, now).filter(|h| is_panel_hit(h.hit)) {
        g.last_ui_hover = Some(hit);
        g.ui_hover_active = true;
        g.edge_out_accum = 0.0;
        return MovePlan::Stay;
    }

    // Edge push. The LL hook reports PRE-clip coordinates, so a push beyond the
    // source border is observable as `out > 0`. We accumulate ONLY the ACTIVE
    // outward travel (how much FURTHER out the hand moved this event), NOT the
    // absolute overshoot and NOT a time-held gate. That distinction is what the
    // user feels: a deliberate push (fast or slow) moves the cursor outward and
    // accumulates → exits cleanly; merely brushing/resting AT the edge adds no
    // travel → never triggers an unintended escape that then "returns".
    let out = edge_out_amount(s, px, py);
    if out > 0 {
        // FULLSCREEN: never slip out at an edge — the view is the whole screen
        // and the panel sits at the top edge, so exiting there is what made the
        // panel unreachable and flung the cursor away. However, if a top-most
        // GUI no-engage rect lives in the letterbox area, let only the virtual
        // cursor continue into that GUI so it remains operable.
        if g.fullscreen {
            if let Some(side) = edge_out_side(s, px, py) {
                if !extend_fullscreen_virtual_into_ui(g, side, out) {
                    log_fullscreen_ui_edge_miss(g, side, px, py, old_virt, mapped);
                    g.edge_out_accum = 0.0;
                    g.virt = clamp_virtual_point(c, g.virt.0, g.virt.1, 0.0);
                }
            }
            return MovePlan::Stay;
        }
        // Accumulate the (per-event, roughly constant while pushing) outward
        // overshoot scaled to content px. NO time gate: merely resting/brushing
        // against the edge adds little and never crosses the threshold, so it
        // cannot trigger an unintended escape that then "returns". A deliberate
        // push — a firm slow push or a fast flick — accumulates quickly and
        // exits. `out` is essentially the hand's per-event outward speed, so a
        // faster push exits in fewer events (matches the user's observation).
        let (scale_x, scale_y) = content_per_source_px(c, s);
        g.edge_out_accum += out as f64 * scale_x.max(scale_y).max(1.0);
        if g.edge_out_accum >= EXIT_TRAVEL_PX {
            // Place the real cursor WHERE THE SPRITE IS (g.virt, driven by raw,
            // already outside the content) then reveal it — so there is no jump
            // between the sprite and the revealed system cursor. Fall back to a
            // just-outside point if the sprite is somehow still inside.
            let place = exit_point_for_escape(g, px, py);
            let sprite = place;
            disengage_state(g, now);
            g.cooldown_until = if g.window_frame_input {
                None
            } else {
                Some(reenter_cooldown(now, g.oscillations))
            };
            return MovePlan::Escape { place, sprite };
        }
        return MovePlan::Stay;
    }
    g.edge_out_accum = 0.0;
    MovePlan::Stay
}

/// What the applier should do after a hardware move. Pure — no Windows calls,
/// so the whole engage/disengage state machine is fully unit-testable via the
/// Sim harness in the tests.
#[derive(Debug, PartialEq)]
enum MoveOutcome {
    /// stay in the current state; optionally redraw the sprite at (x,y)
    Stay { sprite: Option<(i32, i32)> },
    /// engage: clip to source + place the real cursor at the source point
    Engage { tx: i32, ty: i32, vx: f64, vy: f64 },
    /// disengage: release the clip + place the real cursor at `place`
    Disengage {
        place: (i32, i32),
        sprite: (i32, i32),
    },
}

/// Core engage/disengage decision for a hardware move. `actual` is the
/// OS-clipped cursor (GetCursorPos); `px,py` is the raw pre-clip hook pt.
fn handle_move(
    g: &mut State,
    px: i32,
    py: i32,
    now: std::time::Instant,
    actual: (i32, i32),
) -> MoveOutcome {
    if !g.active {
        return MoveOutcome::Stay { sprite: None };
    }
    if g.engaged {
        match plan_engaged(g, px, py, now, actual) {
            MovePlan::Stay => {
                let sprite = if g.hidden_by_idle {
                    None
                } else {
                    Some((g.virt.0.round() as i32, g.virt.1.round() as i32))
                };
                MoveOutcome::Stay { sprite }
            }
            MovePlan::Escape { place, sprite } => {
                // With full-frame window coordinates the exit/re-entry boundary
                // is exact and needs no hidden hysteresis. Keep the legacy
                // guard only for client-space mapping.
                g.must_leave_content = !g.window_frame_input;
                MoveOutcome::Disengage { place, sprite }
            }
        }
    } else if let Some(e) = plan_engage(g, actual.0, actual.1, now) {
        // Guard against a small bounce immediately after an edge exit: only a
        // DELIBERATE return re-engages — the ENGAGE point (mapped source pos)
        // must sit well inside every source edge. A shallow drift back near the
        // edge stays disengaged (no teleport-back). Evaluated ONLY here, at the
        // instant of an actual engage, and `must_leave` is cleared ONLY when we
        // truly engage — so a brief overshoot during the re-enter cooldown can
        // no longer silently disarm the guard.
        if g.must_leave_content {
            let s = g.src;
            let m = POST_ESCAPE_SRC_MARGIN_PX;
            let deliberate = s.w > 2 * m
                && s.h > 2 * m
                && e.tx >= s.x + m
                && e.tx < s.x + s.w - m
                && e.ty >= s.y + m
                && e.ty < s.y + s.h - m;
            if !deliberate {
                return MoveOutcome::Stay { sprite: None };
            }
            log::info!(
                "post-edge reengage accepted: raw=({px},{py}) actual=({},{}) mapped_source=({},{}) content={:?} src={:?}",
                actual.0,
                actual.1,
                e.tx,
                e.ty,
                g.content,
                g.src
            );
        }
        log::info!(
            "source-reentry-authority: raw=({px},{py}) actual=({},{}) mapped_source=({},{}) window_frame={} action=engage",
            actual.0,
            actual.1,
            e.tx,
            e.ty,
            g.window_frame_input
        );
        g.must_leave_content = false;
        g.engaged = true;
        g.virt = (e.vx, e.vy);
        g.last_set = (e.tx, e.ty);
        // New ownership starts from the verified visible desktop point. Raw
        // hook coordinates may still belong to the pre-warp source domain.
        g.last_hw = actual;
        g.edge_out_accum = 0.0;
        g.last_engage_at = Some(now);
        g.expect_teleport = Some((e.tx, e.ty));
        g.teleport_guard_until = None;
        g.hidden_by_idle = false;
        MoveOutcome::Engage {
            tx: e.tx,
            ty: e.ty,
            vx: e.vx,
            vy: e.vy,
        }
    } else {
        MoveOutcome::Stay { sprite: None }
    }
}

fn read_cursor_pos_or(fallback: (i32, i32), context: &str) -> (i32, i32) {
    unsafe {
        let mut pt = POINT::default();
        match GetCursorPos(&mut pt) {
            Ok(()) => (pt.x, pt.y),
            Err(err) => {
                // POINT::default() is (0,0). Never let an API failure inject
                // that sentinel into cursor state, because subsequent mapping
                // can make the visible cursor appear to jump to monitor origin.
                log::warn!(
                    "GetCursorPos failed during {context}: {err:?}; preserving fallback=({},{})",
                    fallback.0,
                    fallback.1
                );
                fallback
            }
        }
    }
}

/// Returns true only when this move engaged the mapper and successfully warped
/// the real cursor. The low-level hook must consume that one triggering event.
fn on_hardware_move(px: i32, py: i32) -> bool {
    // Stop/native-reveal fail-visible bridge. This update is deliberately
    // lock-free/coalesced: TensorRT teardown may hold the render thread and the
    // shared input mutex must never be allowed to freeze the only visible
    // cursor while Windows' real cursor is still Magnification-hidden.
    if DESKTOP_REVEAL_BRIDGE_ACTIVE.load(Ordering::Acquire)
        && CURSOR_HIDE_APPLIED.load(Ordering::Acquire)
    {
        sprite_move(px, py, true);
    }
    // Publish owned client-drag raw movement before touching the shared input
    // mutex. WH_MOUSE_LL must never block; under heavy interpolation/render
    // load try_lock can fail, but visual dragging still needs the newest raw
    // pointer position. The owner token is published only for a validated
    // source LEFT gesture and cleared lock-free on release/recovery.
    if SOURCE_CLIENT_DRAG_OWNER_HWND.load(Ordering::Acquire) != 0 {
        SOURCE_CLIENT_DRAG_RAW_CURRENT.store(pack_drag_pair(px, py), Ordering::Release);
    }
    let st = state();
    let mut g = match st.try_lock() {
        Ok(g) => g,
        Err(_) => return false, // never block the LL hook
    };
    let now = std::time::Instant::now();
    repair_main_gui_passthrough_if_needed();
    g.last_move = Some(now);
    if tensorrt_build_native_cursor_guard() {
        // Build-progress is modal only for cursor routing. Keep Windows at the
        // real desktop point and never re-engage source mapping until the engine
        // reports ready; GUI hover/click continues through normal Win32 input.
        clear_capture_sprite_contract(&mut g);
        if g.engaged || g.cursor_hidden || g.pending_engage.is_some() {
            release_locked(&mut g);
        }
        g.active = false;
        g.buttons_down = 0;
        g.native_ui_hold_bits = 0;
        return false;
    }
    let stale_source_mapping = g.engaged || g.pending_engage.is_some();
    if !g.transition_suspended
        && stale_source_mapping
        && g.last_configure_at
            .is_none_or(|last| now.duration_since(last) > std::time::Duration::from_millis(1500))
    {
        // A stalled render/provider thread must never strand the user inside a
        // source ClipCursor.  However, active capture intentionally uses the
        // Neo sprite as the visual cursor on *all* surfaces.  v429 cleared that
        // sprite contract first and only then requested native reveal; if the
        // first TensorRT job blocked the render/Magnification owner for several
        // seconds, both cursors were invisible for that entire interval.
        //
        // Recover only the stale SOURCE mapping here. Preserve capture/overlay
        // and sprite ownership, unclip synchronously on the LL-hook thread, put
        // the hidden real cursor back at the visible virtual point, and keep the
        // sprite alive.  No render-thread acknowledgement is required.
        let held = g.ui_hold_bits;
        for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
            if held & bit != 0 {
                inject_mouse_button(bit, false);
            }
        }
        if held != 0 {
            log::warn!("input-heartbeat-timeout: released stale GUI hold bits={held:#04x}");
        }
        SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
        cancel_source_caption_drag_contract();
        end_native_gui_caption_drag("input-heartbeat-timeout");
        if g.src_hwnd != 0 {
            unsafe {
                let _ = PostMessageW(
                    Some(HWND(g.src_hwnd as *mut _)),
                    WM_CANCELMODE,
                    WPARAM(0),
                    LPARAM(0),
                );
                let _ = ClipCursor(None);
            }
        } else {
            unsafe {
                let _ = ClipCursor(None);
            }
        }
        restore_mouse_speed();

        let logical = (g.virt.0.round() as i32, g.virt.1.round() as i32);
        let visible = reachable_desktop_cursor_point(logical);
        let (reached, actual) = warp_unclipped_verified(visible, "heartbeat stale mapping release");
        let visual = if reached { actual } else { visible };

        disengage_state(&mut g, now);
        g.pending_engage = None;
        g.must_leave_content = true;
        g.cooldown_until = Some(now + std::time::Duration::from_millis(500));
        // Start a fresh watchdog epoch so a still-cold provider does not fire
        // the same recovery on every subsequent desktop mouse sample.
        g.last_configure_at = Some(now);
        g.buttons_down = 0;
        g.native_ui_hold_bits = 0;
        g.virt = (visual.0 as f64, visual.1 as f64);
        g.last_set = visual;
        g.last_hw = visual;

        request_cursor_hidden(true);
        sprite_move_now(visual.0, visual.1, true);
        keep_cursor_sprite_on_top();
        log::warn!(
            "input-heartbeat-timeout: stale source mapping released fail-visible=true capture_preserved={} sprite_preserved={} logical=({},{}) visual=({},{}) warp_verified={}",
            g.active,
            capture_sprite_active(),
            logical.0,
            logical.1,
            visual.0,
            visual.1,
            reached
        );
        return false;
    }
    // A low-level button-up can be lost while a synthetic UI handoff changes
    // focus. Never let that leave the mapper permanently in "dragging" mode,
    // which makes both the GUI and magnified window feel immovable.
    let stale_buttons = reconcile_stale_buttons(&mut g, now);
    for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
        if stale_buttons & bit != 0 {
            inject_mouse_button(bit, false);
        }
    }
    if stale_buttons != 0 {
        log::warn!("mouse-move released stale GUI hold bits={stale_buttons:#04x}");
    }
    if g.hidden_by_idle && g.engaged {
        g.hidden_by_idle = false;
        let vx = g.virt.0 as i32;
        let vy = g.virt.1 as i32;
        sprite_move(vx, vy, true);
    }
    if !g.active {
        if g.engaged {
            release_locked(&mut g);
        }
        return false;
    }
    // The stable edge-release path intentionally completes warp+native reveal
    // on the Magnification owner thread after the triggering LL event returns.
    // Until that transaction finishes, never re-arm source mapping.
    if !g.engaged && cursor_reveal_pending() {
        let pending_actual = read_cursor_pos_or((px, py), "edge reveal pending move");
        log::debug!(
            "edge-release reveal pending: reengage blocked raw=({px},{py}) actual=({},{})",
            pending_actual.0,
            pending_actual.1
        );
        return false;
    }
    if let Some(pending) = g.pending_engage {
        if now.duration_since(pending.requested_at) <= std::time::Duration::from_millis(250) {
            // Keep the native cursor at its GUI-side position while the owner
            // thread applies MagShowSystemCursor(false). Consuming these few
            // events prevents a 120/144Hz display from exposing the cursor at
            // the hidden source position for one refresh.
            request_cursor_hidden(true);
            return true;
        }
        log::warn!(
            "cursor handoff timed out before native hide: origin=({},{}) target=({},{}) age_ms={}",
            pending.origin.0,
            pending.origin.1,
            pending.target.0,
            pending.target.1,
            now.duration_since(pending.requested_at).as_millis()
        );
        g.pending_engage = None;
        g.engaged = false;
        publish_deferred_source_engaged(false);
        g.expect_teleport = None;
        g.teleport_guard_until = None;
        g.cursor_hidden = false;
        // The native cursor was never warped on this path. Rebase bookkeeping
        // to GetCursorPos, not the timeout-triggering LL sample: the latter can
        // still be a pre-warp/source-space coordinate.
        let recovered = read_cursor_pos_or((px, py), "pending engage timeout recovery");
        g.last_set = recovered;
        g.last_hw = recovered;
        g.cooldown_until = Some(now + std::time::Duration::from_millis(REENTER_COOLDOWN_MS));
        request_cursor_hidden(false);
        sprite_hide();
        log::info!(
            "cursor handoff timeout recovered: raw=({px},{py}) actual=({},{}) cooldown_ms={REENTER_COOLDOWN_MS}",
            recovered.0,
            recovered.1
        );
        return false;
    }
    if g.ui_hold_bits != 0 {
        return false;
    }
    if let Some(mut settle) = g.native_gui_settle {
        let dx = px - settle.target.0;
        let dy = py - settle.target.1;
        let near_target = dx.abs() <= 24 && dy.abs() <= 24;

        // A stale post-handoff event should still resemble the OLD source-space
        // coordinate. v361 treated every point farther than 24px from the GUI
        // entry point as stale, which incorrectly caught ordinary continued
        // mouse motion inside the GUI (especially across its top edge) and
        // re-warped the hidden native cursor back toward the entry point.
        //
        // Compare the raw event against both coordinate domains instead:
        // quarantine only when it is genuinely closer to the old source point
        // than to the verified GUI entry point. Ambiguous/GUI-side motion is
        // released immediately and handled by the normal exact-GUI path.
        let source_dx = px - settle.source_origin.0;
        let source_dy = py - settle.source_origin.1;
        let target_dist2 = i64::from(dx) * i64::from(dx) + i64::from(dy) * i64::from(dy);
        let source_dist2 = i64::from(source_dx) * i64::from(source_dx)
            + i64::from(source_dy) * i64::from(source_dy);
        let source_like = source_dist2 < target_dist2;

        if near_target {
            g.native_gui_settle = None;
        } else if now <= settle.until && source_like {
            if !settle.reasserted {
                let (_, actual) =
                    warp_unclipped_verified(settle.target, "native GUI stale-source quarantine");
                settle.target = actual;
                settle.reasserted = true;
                log::warn!(
                    "native-gui stale-source move quarantined: target=({},{}) source_origin=({},{}) stale=({px},{py})",
                    settle.target.0,
                    settle.target.1,
                    settle.source_origin.0,
                    settle.source_origin.1
                );
            }
            g.native_gui_settle = Some(settle);
            return true;
        } else if now <= settle.until {
            // Normal GUI-side continuation: do not fight the user's motion
            // with another SetCursorPos. Clearing the settle guard here keeps
            // the v350g/v361 GUI-sprite ownership contract while removing the
            // visible top-edge "snap back" caused by the over-broad quarantine.
            g.native_gui_settle = None;
        } else if source_like {
            let (_, actual) =
                warp_unclipped_verified(settle.target, "native GUI settle timeout recovery");
            g.native_gui_settle = None;
            g.last_set = actual;
            g.last_hw = actual;
            log::warn!(
                "native-gui stale-source settle timeout recovered: target=({},{}) stale=({px},{py})",
                actual.0,
                actual.1
            );
            return true;
        } else {
            g.native_gui_settle = None;
        }
    }
    // Bypass render-thread geometry latency while Neo's GUI is moving/resizing.
    // This samples the actual HWND on the LL hook path, so language/DPI/mode
    // changes and native title-bar dragging cannot leave a stale exclusion rect
    // long enough to arm a source engage.
    refresh_live_own_no_engage_geometry(&mut g, now);

    // v372: WH_MOUSE_LL gives the CURRENT hardware event in
    // MSLLHOOKSTRUCT.pt. GetCursorPos inside the callback can still describe the
    // PREVIOUS event because Windows has not yet applied this one. v371 treated
    // that one-event-behind value as ground truth, which made a GUI/edge exit
    // look as if the pointer were still inside magnified content and immediately
    // re-armed source ownership. Use the hook point for ownership decisions;
    // sample GetCursorPos only as a diagnostic comparison.
    let mut disengaged_event = (px, py);
    if !g.engaged {
        let observed = read_cursor_pos_or(disengaged_event, "disengaged event diagnostic");
        let dx = (px - observed.0).abs();
        let dy = (py - observed.1).abs();
        let event_in_content = g.content.contains(px, py);
        let observed_in_content = g.content.contains(observed.0, observed.1);
        let event_in_ui = hit_ui_for_cursor_ownership(&g.no_engage, px, py).is_some();
        let observed_in_ui =
            hit_ui_for_cursor_ownership(&g.no_engage, observed.0, observed.1).is_some();

        let mut authority = "use-event";
        if let Some(until) = g.edge_release_settle_until {
            if now >= until {
                g.edge_release_settle_until = None;
            } else {
                let caught_up = dx <= POST_EDGE_RAW_CATCHUP_PX && dy <= POST_EDGE_RAW_CATCHUP_PX;
                let domain_disagrees =
                    event_in_content != observed_in_content || event_in_ui != observed_in_ui;
                let raw_is_stale = post_edge_raw_is_stale(dx, dy, domain_disagrees);
                if raw_is_stale {
                    // The verified edge warp has already moved the hardware
                    // cursor to desktop space; this hook point belongs to the
                    // pre-warp queue.  Use GetCursorPos for this sample only
                    // instead of letting stale content/source coordinates
                    // immediately suck the cursor back into the overlay.
                    disengaged_event = observed;
                    g.last_hw = observed;
                    authority = "use-getcursor-post-edge";
                    log::info!(
                        "post-edge stale raw cursor event suppressed: raw=({px},{py}) actual=({},{}) divergence=({dx},{dy}) event_content={} actual_content={} event_ui={} actual_ui={} guard_remaining_ms={}",
                        observed.0,
                        observed.1,
                        event_in_content,
                        observed_in_content,
                        event_in_ui,
                        observed_in_ui,
                        until.saturating_duration_since(now).as_millis()
                    );
                } else if caught_up {
                    g.edge_release_settle_until = None;
                }
            }
        }

        if dx > 8
            || dy > 8
            || event_in_content != observed_in_content
            || event_in_ui != observed_in_ui
        {
            log::info!(
                "disengaged-coordinate-authority: event=({px},{py}) getcursor=({},{}) divergence=({dx},{dy}) event_content={} getcursor_content={} event_ui={} getcursor_ui={} action={authority}",
                observed.0,
                observed.1,
                event_in_content,
                observed_in_content,
                event_in_ui,
                observed_in_ui
            );
        }
    }

    if !g.engaged
        && hold_native_gui_ownership_if_inside(&mut g, disengaged_event.0, disengaged_event.1, now)
    {
        return false;
    }

    // TRUTH while engaged = OS-clipped source cursor. While disengaged = this
    // hardware event's screen point, not one-event-behind GetCursorPos.
    let actual = if g.engaged {
        let s = g.src;
        if s.w <= 1 || s.h <= 1 {
            return false;
        }
        if g.cursor_hidden && on_or_past_source_edge(s, px, py) {
            request_cursor_hidden(true);
        }
        read_cursor_pos_or(g.last_set, "engaged hardware move")
    } else {
        disengaged_event
    };
    // Do not reissue SetCursorPos while stale pre-handoff moves are draining.
    // Repeated warps erase genuine reversal input at a GUI boundary and are the
    // direct cause of the visible snag/jump. `plan_engaged` owns the bounded
    // stale-event quarantine and clears it on the verified target event.
    if g.engaged {
        let vx = g.virt.0.round() as i32;
        let vy = g.virt.1.round() as i32;
        let physical_gui = own_main_gui_at_visible_point(&g.no_engage, actual.0, actual.1);
        let visible_gui = own_main_gui_at_visible_point(&g.no_engage, vx, vy);
        if let Some(gui) = physical_gui.filter(|_| visible_gui.is_none()) {
            let mut shielded = crate::platform::win32::window_input_passthrough(gui.hwnd);
            if !shielded {
                publish_main_gui_passthrough(gui.hwnd, true, "gui-ghost-route-repair");
                shielded = crate::platform::win32::window_input_passthrough(gui.hwnd);
                log::error!(
                    "gui-ghost-route repaired: hwnd={:#x} physical=({},{}) virtual=({vx},{vy}) rect={:?} passthrough_after={shielded}",
                    gui.hwnd,
                    actual.0,
                    actual.1,
                    gui.land
                );
            } else if GHOST_ROUTE_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed) % 128 == 0 {
                log::info!(
                    "gui-ghost-route shielded: hwnd={:#x} physical=({},{}) virtual=({vx},{vy}) rect={:?} passthrough=true",
                    gui.hwnd,
                    actual.0,
                    actual.1,
                    gui.land
                );
            }
        }
    }
    match handle_move(&mut g, px, py, now, actual) {
        MoveOutcome::Stay { sprite } => {
            // Re-assert the system-cursor hide on every engaged move (not a
            // cached flag): the Magnification hide can be reset by the
            // compositor / capture / another app, and then the REAL cursor —
            // sitting at the offset source position — bleeds through as a second
            // cursor or a bidirectional resize cursor. Cheap and idempotent.
            if g.engaged && g.cursor_hidden {
                request_cursor_hidden(true);
            }
            if let Some((sx, sy)) = sprite {
                if g.buttons_down & BTN_LEFT != 0 && source_caption_drag_active() {
                    // v382: source-caption drag has exactly ONE visual writer:
                    // commit_source_caption_drag_overlay_update(). The common
                    // sprite API also hard-blocks any missed generic writer, so
                    // a mouse event from another clock cannot overwrite it.
                    let _ = (sx, sy);
                } else {
                    sprite_move(sx, sy, true);
                }
            }
            // v639 regression rollback: v638 mirrored ordinary WM_MOUSEMOVE to
            // the source whenever the mapped source point was occluded. That made
            // occlusion itself alter the input route and raced the long-standing
            // native-GUI ownership/ghost-route repair logic. The v638 log showed
            // `source-direct-motion` immediately followed by `gui-ghost-route
            // repaired` and a native GUI handoff. Restore the stable pre-v637
            // movement path: synthetic movement is allowed ONLY while Neo owns a
            // directly-forwarded button gesture. Never swallow the physical MOVE.
            if g.source_direct_bits != 0 && g.engaged && !g.window_frame_input && g.src_hwnd != 0 {
                let (tx, ty) = map_content_to_source(g.content, g.src, g.virt.0, g.virt.1);
                let buttons = g.buttons_down | g.source_direct_bits;
                let should_post =
                    g.last_source_direct_post
                        .map_or(true, |(hwnd, pos, old_buttons, at)| {
                            hwnd != g.src_hwnd
                                || (pos != (tx, ty)
                                    && now.saturating_duration_since(at)
                                        >= std::time::Duration::from_millis(
                                            SOURCE_DIRECT_MOVE_MIN_MS,
                                        ))
                                || old_buttons != buttons
                        });
                if should_post {
                    let forwarded = post_mouse_move_to_source(tx, ty, g.src_hwnd, buttons);
                    if forwarded {
                        g.last_source_direct_post = Some((g.src_hwnd, (tx, ty), buttons, now));
                    }
                    if GHOST_ROUTE_SAMPLE_COUNT.fetch_add(1, Ordering::Relaxed) % 128 == 0 {
                        log::debug!(
                            "source-direct-drag-move: source=({tx},{ty}) hwnd={:#x} buttons={buttons:#04x} forwarded={forwarded}",
                            g.src_hwnd
                        );
                    }
                }
            } else if g.source_direct_bits == 0 {
                g.last_source_direct_post = None;
            }
            let gui_handoff_applied = if g.engaged && !g.hidden_by_idle && g.buttons_down == 0 {
                post_ui_hover_from_state(&mut g, now)
            } else {
                false
            };
            if gui_handoff_applied {
                log::debug!("gui handoff trigger consumed: raw=({px},{py})");
            }
            gui_handoff_applied
        }
        MoveOutcome::Engage { tx, ty, vx, vy } => {
            // MoveOutcome::Engage is only produced from the disengaged state.
            // Preserve the actual visible cursor as the handoff origin so an
            // old LL-hook sample can never become the pending-engage origin.
            apply_engage_windows(&mut g, actual.0, actual.1, tx, ty, vx, vy, now, false)
        }
        MoveOutcome::Disengage { place, sprite: _ } => {
            set_own_main_gui_passthrough(&g, false, "edge-release");
            let requested_place = place;
            let place = reachable_desktop_cursor_point(requested_place);
            let sprite = place;
            if place != requested_place {
                log::info!(
                    "edge-release target clamped to monitor: requested=({},{}) safe=({},{})",
                    requested_place.0,
                    requested_place.1,
                    place.0,
                    place.1
                );
            }
            // Keep the native cursor hidden while we move it out of source
            // space. The old path waited ~45ms for the render-thread reveal
            // pump before even attempting this warp. During that gap the LL
            // hook could still receive source-space coordinates and later
            // mistake them for a fresh screen-space re-entry.
            if g.cursor_hidden {
                request_cursor_hidden(true);
            }
            sprite_move_now(sprite.0, sprite.1, true);
            log::debug!(
                "edge-release detail: raw=({px},{py}) actual=({},{}) sprite=({},{}) place=({},{}) content={:?} src={:?}",
                actual.0,
                actual.1,
                sprite.0,
                sprite.1,
                place.0,
                place.1,
                g.content,
                g.src
            );
            DEBUG_LAST_EDGE_PLACE_X.store(place.0, Ordering::Release);
            DEBUG_LAST_EDGE_PLACE_Y.store(place.1, Ordering::Release);

            // Keep the stable deferred edge transaction: never SetCursorPos
            // from inside this WH_MOUSE_LL callback and then pass the same
            // hardware move onward.  With capture-wide sprite ownership we no
            // longer need the old 45ms native-reveal grace. After one short
            // mouse-sample defer, capture-wide sprite mode completes the
            // verified desktop transfer on the hook-owned sprite timer; native
            // reveal cases remain on the Magnification owner thread. A cold ONNX/TensorRT
            // render-thread stall cannot hold the cursor transaction open.
            // Pending state still blocks stale re-entry until the warp verifies.
            unsafe {
                let _ = ClipCursor(None);
            }
            defer_cursor_reveal(place);
            g.edge_release_settle_until =
                Some(now + std::time::Duration::from_millis(POST_EDGE_RAW_GUARD_MS));
            let transfer_mode = if capture_sprite_active() {
                "hook-timer-deferred"
            } else {
                "owner-thread-deferred"
            };
            log::info!(
                "edge-release transaction armed: event=({px},{py}) target=({},{}) mode={transfer_mode} defer_ms={EDGE_TRANSFER_DEFER_MS}",
                place.0,
                place.1
            );
            restore_mouse_speed();
            g.cursor_hidden = false;
            g.hidden_by_idle = false;
            log::debug!("disengage(edge push): cursor=({},{})", place.0, place.1);
            false
        }
    }
}

/// Apply the Windows side of an engage. Normal mouse-hook engages can only
/// request the thread-affine Magnification hide asynchronously. The initial
/// overlay reveal runs on the Mag owner thread, so it can commit that hide
/// before moving the native cursor into the offset source window.
#[allow(clippy::too_many_arguments)]
fn apply_engage_windows(
    g: &mut State,
    px: i32,
    py: i32,
    tx: i32,
    ty: i32,
    vx: f64,
    vy: f64,
    now: std::time::Instant,
    commit_hide_before_warp: bool,
) -> bool {
    let s = g.src;
    DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
    // Do not make the entire GUI click-through merely because source ownership
    // starts.  Only shield it if the mapped hidden-source point is physically
    // underneath the GUI; the engaged-move ghost-route guard maintains that
    // shield if the hidden cursor later crosses beneath it.
    shield_main_gui_for_source_target(g, tx, ty, px, py);
    if !g.cursor_hidden {
        request_cursor_hidden(true);
        g.cursor_hidden = true;
    }
    let capture_sprite_fast_path =
        capture_sprite_active() && CURSOR_HIDE_APPLIED.load(Ordering::Acquire);
    if commit_hide_before_warp {
        pump_cursor_visibility();
        if !CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
            log::warn!("pre-reveal cursor hide not committed; keeping overlay hidden");
            clear_capture_sprite_contract(g);
            release_locked(g);
            g.active = false;
            return false;
        }
    } else if !capture_sprite_fast_path {
        // Outside the capture-wide sprite contract, preserve the conservative
        // owner-thread hide-before-warp transaction.
        g.pending_engage = Some(PendingEngage {
            origin: (px, py),
            target: (tx, ty),
            sprite: (vx.round() as i32, vy.round() as i32),
            src: s,
            requested_at: now,
        });
        sprite_move(vx.round() as i32, vy.round() as i32, true);
        log::debug!(
            "cursor handoff armed: native-visible position=({px},{py}) target-source=({tx},{ty})"
        );
        return true;
    } else {
        // v376+ keeps the native cursor hidden for the whole active capture.
        // There is nothing left to wait for here: consume this LL event, warp
        // the already-hidden native cursor synchronously into source space and
        // keep the sprite at the visible point.  This removes the ~60-80ms
        // render-thread handoff that made overlay re-entry feel heavy.
        log::debug!("capture-sprite fast engage: visible=({px},{py}) target-source=({tx},{ty})");
    }
    let (warp_applied, warp_actual) = warp_then_clip_source((tx, ty), s);
    if !warp_applied {
        log::warn!(
            "engage warp rejected: requested=({tx},{ty}) actual=({},{}) src={:?} raw=({px},{py})",
            warp_actual.0,
            warp_actual.1,
            s
        );
        let (origin_restored, _) =
            warp_unclipped_verified((px, py), "synchronous engage abort native restore");
        release_locked(g);
        if !origin_restored {
            sprite_move_now(px, py, true);
            defer_cursor_reveal((px, py));
        }
        g.must_leave_content = true;
        g.cooldown_until = Some(now + std::time::Duration::from_millis(REENTER_COOLDOWN_MS));
        return false;
    }
    // The synchronous pre-reveal path also becomes real only after this warp
    // succeeds. Use the same commit-relative stale-event guard as the normal
    // asynchronous handoff so there is no unprotected first-reveal window.
    mark_engage_committed(g, (tx, ty), std::time::Instant::now());
    if g.adjust_speed {
        let zoom =
            (g.content.w as f64 / s.w.max(1) as f64).max(g.content.h as f64 / s.h.max(1) as f64);
        slow_mouse_for_zoom(zoom);
    }
    sprite_move(vx.round() as i32, vy.round() as i32, true);
    log::info!(
        "engage: ({px},{py}) -> source({tx},{ty}) clip={:?} trigger_consumed={warp_applied} pre_reveal={commit_hide_before_warp}",
        (s.x, s.y, s.w, s.h),
    );
    warp_applied
}

/// Result of the pure engage planner.
#[derive(Debug, PartialEq)]
struct EngagePlan {
    tx: i32,
    ty: i32,
    vx: f64,
    vy: f64,
}

/// Pure planner deciding whether a disengaged hardware move should engage.
/// No Windows calls — unit-testable.
fn plan_engage(g: &State, px: i32, py: i32, now: std::time::Instant) -> Option<EngagePlan> {
    plan_engage_impl(g, px, py, now, true)
}

fn plan_engage_for_overlay_reveal(
    g: &State,
    px: i32,
    py: i32,
    now: std::time::Instant,
) -> Option<EngagePlan> {
    // Startup panel placement updates the no-engage geometry moments before
    // the first filtered frame. Keep the real UI rectangles, but do not let the
    // normal 300ms drag guard delay this one atomic cursor handoff.
    plan_engage_impl(g, px, py, now, false)
}

fn plan_engage_impl(
    g: &State,
    px: i32,
    py: i32,
    now: std::time::Instant,
    honor_recent_no_engage_motion: bool,
) -> Option<EngagePlan> {
    // Never start source ownership in the middle of a Neo-native GUI gesture.
    // `native_ui_hold_bits` marks only a physical button press that STARTED on
    // Neo's real GUI. This is deliberately narrower than `buttons_down`: a
    // normal press/drag aimed at magnified content keeps the historic source
    // input behaviour. Warping during a native title-bar drag would make
    // Windows interpret source coordinates as the next window-move point.
    if g.native_ui_hold_bits != 0 || g.ui_hold_bits != 0 {
        return None;
    }
    if let Some(t) = g.cooldown_until {
        if now < t {
            return None;
        }
    }
    // Fullscreen ownership cannot escape through the display edge (see the
    // engaged edge path below), so an entry dead-band there only creates a
    // non-interactive strip. This was visible with 4:3 Auto: MPC-HC's seek bar
    // at the bottom could not be the first source interaction, but worked after
    // a click higher in the image had already engaged source ownership. Keep
    // the proven dead-band only for windowed client-space re-entry. Window-frame
    // input already uses an exact boundary as before.
    let inside_visible_content = if g.fullscreen || g.window_frame_input {
        g.content.contains(px, py)
    } else {
        contains_for_engage(g.content, px, py)
    };
    if !inside_visible_content {
        return None;
    }
    // Do not block engage from cached rectangles. A GUI can move many pixels
    // between render-thread geometry publications; using that stale rectangle
    // here created invisible no-engage islands near the GUI boundary.
    // The current topmost Win32 window at the actual pointer point is authority.
    if hit_ui_for_cursor_ownership(&g.no_engage, px, py).is_some() {
        return None;
    }
    let c = g.content;
    let s = g.src;
    if s.w <= 1 || s.h <= 1 || c.w <= 1 || c.h <= 1 {
        return None;
    }
    let (tx, ty) = map_content_to_source(c, s, px as f64, py as f64);
    // hysteresis dead-band in SOURCE space (see ENGAGE_SRC_MARGIN_PX): don't
    // engage while the mapped source position sits within the band of a source
    // edge, so the engage boundary is well inside the escape boundary and the
    // windowed edge cannot oscillate / pull the cursor back. Skipped for a
    // source too small to hold the band.
    let m = ENGAGE_SRC_MARGIN_PX;
    if !g.fullscreen
        && !g.window_frame_input
        && s.w > 2 * m
        && s.h > 2 * m
        && (tx < s.x + m || tx >= s.x + s.w - m || ty < s.y + m || ty >= s.y + s.h - m)
    {
        return None;
    }
    let (vx, vy) = map_source_to_content(c, s, tx as f64, ty as f64);
    // If the SPRITE would materialise inside a GUI zone, engaging is pointless:
    // the very next move hands straight back to the GUI — the log showed
    // engage↔handoff pairs at 10ms period while skirting the GUI boundary.
    // Stay a free cursor until the mapped position is clearly outside.
    if hit_ui_for_cursor_ownership(&g.no_engage, vx.round() as i32, vy.round() as i32).is_some() {
        return None;
    }
    // Engage is blocked for the lifetime of a physical GUI gesture
    // and refreshes the HWND rect directly in the LL hook. A second 300ms
    // movement timer only made ownership feel random at different crossing
    // speeds, so normal GUI ownership is now purely geometric.
    let _ = honor_recent_no_engage_motion;
    Some(EngagePlan { tx, ty, vx, vy })
}

/// Windows-side release ONLY (no state fields): clear the clip, restore the
/// system cursor, hide the sprite. State transitions live in `disengage_state`
/// so the move planner can stay pure and unit-testable.
fn release_windows(g: &mut State) {
    SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    set_own_main_gui_passthrough(g, false, "source-ownership-release");
    g.pending_engage = None;
    g.native_gui_settle = None;
    let visible = (g.virt.0.round() as i32, g.virt.1.round() as i32);
    let native_is_hidden =
        g.cursor_hidden || CURSOR_HIDE_APPLIED.load(Ordering::Acquire) || cursor_reveal_pending();

    let keep_capture_sprite_owner = g.active && capture_sprite_active();

    if keep_capture_sprite_owner {
        let (reached, actual) = warp_unclipped_verified(visible, "capture sprite release");
        if reached {
            sprite_move_now(actual.0, actual.1, true);
        } else {
            sprite_move_now(visible.0, visible.1, true);
            defer_cursor_reveal(visible);
        }
        request_cursor_hidden(true);
        g.cursor_hidden = false;
        restore_mouse_speed();
        return;
    }

    if native_is_hidden {
        let (reached, actual) = warp_unclipped_verified(visible, "generic native release");
        if reached {
            // Bridge with the sprite until the Mag owner confirms native show.
            // This flag keeps pump_cursor_visibility() from hiding the sprite
            // one step before MagShowSystemCursor(true) actually succeeds.
            DESKTOP_REVEAL_BRIDGE_ACTIVE.store(true, Ordering::Release);
            SPRITE_NATIVE_REVEAL_BLOCK.store(false, Ordering::Release);
            sprite_move_now(actual.0, actual.1, true);
            request_cursor_hidden(false);
        } else {
            // Never reveal the source-space cursor merely because a Stop,
            // geometry transition or watchdog asked for release. Retry the
            // visible screen point on the Mag-owner thread first.
            sprite_move_now(visible.0, visible.1, true);
            defer_cursor_reveal(visible);
        }
        g.cursor_hidden = false;
    } else {
        unsafe {
            let _ = ClipCursor(None);
        }
        DESKTOP_REVEAL_BRIDGE_ACTIVE.store(false, Ordering::Release);
        sprite_hide();
    }
    restore_mouse_speed();
}

/// Transfer ownership from the magnified sprite to a native window without a
/// one-frame cursor flash at the hidden source position. Move the still-hidden
/// real cursor first, then request native visibility. This ordering makes GUI
/// entry deterministic even when the render thread is between visibility pumps.
fn release_windows_to_native_at(
    g: &mut State,
    pos: (i32, i32),
    reason: &str,
    keep_native_hidden: bool,
) -> Option<(i32, i32)> {
    // State flags can lag the Magnification owner by one pump. Base the reveal
    // decision on BOTH logical and actually-applied hide state so a rapid
    // GUI/content reversal cannot leave the verified native cursor hidden or
    // hide the sprite prematurely.
    let native_was_hidden =
        g.cursor_hidden || CURSOR_HIDE_APPLIED.load(Ordering::Acquire) || cursor_reveal_pending();
    // Keep the last known source point so a failed native handoff can be rolled
    // back while the real cursor is still hidden. Ownership is not committed
    // until the requested native point is verified by GetCursorPos.
    let source_origin = read_cursor_pos_or(g.last_set, "native handoff source origin");
    let (reached, actual) = warp_unclipped_verified(pos, reason);
    if !reached {
        let mut recovered = false;
        let mut recovery_actual = source_origin;
        if g.engaged && g.src.w > 1 && g.src.h > 1 {
            (recovered, recovery_actual) = warp_then_clip_source(source_origin, g.src);
        }
        if recovered {
            g.last_set = recovery_actual;
            g.last_hw = recovery_actual;
        } else {
            unsafe {
                // Never reveal a cursor at an unverified location. If recovery
                // also failed, leave it hidden/unclipped and retry ownership on
                // the next real move instead of exposing a source corner.
                let _ = ClipCursor(None);
            }
        }
        request_cursor_hidden(true);
        g.cursor_hidden = true;
        sprite_move_now(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
        log::warn!(
            "native cursor ownership transfer aborted: reason={reason} requested=({},{}) actual=({},{}) source_recovered={} recovery=({},{})",
            pos.0,
            pos.1,
            actual.0,
            actual.1,
            recovered,
            recovery_actual.0,
            recovery_actual.1
        );
        return None;
    }

    g.pending_engage = None;
    let capture_sprite_owner = g.active && capture_sprite_active();
    if (keep_native_hidden || capture_sprite_owner) && g.active {
        // Active capture uses one visual owner on every surface. Preserve the
        // real cursor at the verified native coordinate for hit-testing while
        // Neo's sprite remains visible; only Stop/provider guards reveal native.
        unsafe {
            let _ = ClipCursor(None);
        }
        sprite_move_now(actual.0, actual.1, true);
        request_cursor_hidden(true);
        g.cursor_hidden = false;
    } else if native_was_hidden {
        // Ordinary edge/desktop release still bridges until native show.
        sprite_move_now(actual.0, actual.1, true);
        request_cursor_hidden(false);
        g.cursor_hidden = false;
    } else {
        sprite_hide();
    }
    restore_mouse_speed();
    Some(actual)
}

/// Pure state transition for a disengage (no Windows calls).
fn disengage_state(g: &mut State, now: std::time::Instant) {
    publish_deferred_source_engaged(false);
    SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    mark_disengage(g, now);
    g.engaged = false;
    g.edge_out_accum = 0.0;
    g.last_engage_at = None;
    g.teleport_guard_until = None;
    g.edge_release_settle_until = None;
    g.swallow_up = 0;
    g.source_direct_bits = 0;
    g.last_source_direct_post = None;
    g.ui_hold_bits = 0;
    g.native_ui_hold_bits = 0;
    g.last_button_event = None;
    g.last_ui_hover = None;
    g.last_ui_post = None;
    g.native_gui_owner_hwnd = 0;
    g.native_gui_settle = None;
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
}

fn release_locked(g: &mut State) {
    publish_deferred_source_engaged(false);
    clear_deferred_source_queue();
    SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
    // begin_ui_hold injects a DOWN at the native GUI so dragging behaves like
    // an ordinary window. A focus transition can occasionally lose the
    // matching physical UP. Close any outstanding synthetic hold before
    // clearing the bookkeeping, or the GUI can remain immovable after Stop.
    let held = g.ui_hold_bits;
    for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
        if held & bit != 0 {
            inject_mouse_button(bit, false);
        }
    }
    if held != 0 {
        log::warn!("release cleared outstanding GUI mouse hold bits={held:#04x}");
    }
    if g.src_hwnd != 0 {
        if g.source_direct_bits != 0 {
            let (tx, ty) = if g.src.w > 1 && g.src.h > 1 && g.content.w > 1 && g.content.h > 1 {
                map_content_to_source(g.content, g.src, g.virt.0, g.virt.1)
            } else {
                g.last_set
            };
            for bit in [BTN_LEFT, BTN_RIGHT, BTN_MIDDLE] {
                if g.source_direct_bits & bit != 0 {
                    let msg = match bit {
                        BTN_LEFT => WM_LBUTTONUP,
                        BTN_RIGHT => WM_RBUTTONUP,
                        BTN_MIDDLE => WM_MBUTTONUP,
                        _ => continue,
                    };
                    let _ = post_mouse_button_to_source(msg, tx, ty, g.src_hwnd);
                }
            }
            log::info!(
                "release closed direct source hold bits={:#04x} hwnd={:#x}",
                g.source_direct_bits,
                g.src_hwnd
            );
        }
        unsafe {
            // Tell the source to abandon any native move/resize/drag modal
            // loop entered while Neo owned the mapped cursor. This is needed
            // in addition to button-up: mpv can retain window capture after a
            // synthetic/native handoff even though GetAsyncKeyState is clear.
            let _ = PostMessageW(
                Some(HWND(g.src_hwnd as *mut _)),
                WM_CANCELMODE,
                WPARAM(0),
                LPARAM(0),
            );
        }
        log::info!(
            "source input capture cancelled on release: hwnd={:#x}",
            g.src_hwnd
        );
    }
    release_windows(g);
    g.engaged = false;
    g.edge_out_accum = 0.0;
    g.last_engage_at = None;
    g.teleport_guard_until = None;
    g.edge_release_settle_until = None;
    g.swallow_up = 0;
    g.source_direct_bits = 0;
    g.last_source_direct_post = None;
    g.ui_hold_bits = 0;
    g.native_ui_hold_bits = 0;
    g.last_button_event = None;
    g.must_leave_content = false;
    g.last_ui_hover = None;
    g.last_ui_post = None;
    g.native_gui_owner_hwnd = 0;
    NATIVE_GUI_OWNER.store(0, Ordering::Release);
    ACTIVE_OVERLAY_HWND.store(0, Ordering::Release);
    g.src_hwnd = 0;
}

fn mark_disengage(g: &mut State, now: std::time::Instant) {
    if let Some(start) = g.last_engage_at {
        if now.duration_since(start) < std::time::Duration::from_millis(OSC_SHORT_MS) {
            g.oscillations = (g.oscillations + 1).min(OSC_MAX);
        } else {
            g.oscillations = 0;
        }
    }
    g.last_engage_at = None;
}

// ---------------- public interface ----------------

pub struct InputSystem {
    thread_id: u32,
    handle: Option<std::thread::JoinHandle<()>>,
    done_rx: std::sync::mpsc::Receiver<()>,
}

static HOOK_RUNNING: AtomicBool = AtomicBool::new(false);

/// Marker accepted only by the opt-in native desktop regression harness.
/// Normal injected input remains ignored by the low-level hook.
pub const DESKTOP_TEST_INPUT_TAG: usize = 0x4554_5354;
static DESKTOP_TEST_INPUT_ENABLED: OnceLock<bool> = OnceLock::new();
static DESKTOP_TEST_INPUT_SEEN: AtomicBool = AtomicBool::new(false);
static DESKTOP_TEST_INJECTED_DIAG: AtomicBool = AtomicBool::new(false);
static DESKTOP_TEST_MOVE_COUNT: AtomicU64 = AtomicU64::new(0);

fn desktop_test_input_enabled() -> bool {
    *DESKTOP_TEST_INPUT_ENABLED.get_or_init(|| {
        std::env::var_os("CHIDESCALER_DESKTOP_TEST_INPUT").is_some_and(|value| value == "1")
    })
}

impl InputSystem {
    pub fn start() -> Self {
        // Startup recovery must never inject mouse-button UP events.
        startup_recover_input_state();
        start_input_failsafe_worker();
        let (tx, rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("mouse-hook".into())
            .spawn(move || unsafe {
                struct HookDone(Option<std::sync::mpsc::Sender<()>>);
                impl Drop for HookDone {
                    fn drop(&mut self) {
                        if let Some(tx) = self.0.take() {
                            let _ = tx.send(());
                        }
                    }
                }
                let _done = HookDone(Some(done_tx));
                // NOTE: the Magnification API (real-cursor hide) is deliberately
                // NOT initialised here — it is a silent no-op on the LL-hook
                // thread. It lives on the render-engine thread instead
                // (pump_cursor_visibility). See WANT_CURSOR_HIDDEN.
                if let Some(h) = create_sprite_window() {
                    let _ = SPRITE_HWND.set(h.0 as isize);
                }
                // NOTE: hMod must be a real module handle; a truncated handle
                // gives SetWindowsHookEx error 126.
                let hmod = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok();
                let hook = SetWindowsHookExW(
                    WH_MOUSE_LL,
                    Some(mouse_proc),
                    hmod.map(|m| windows::Win32::Foundation::HINSTANCE(m.0)),
                    0,
                );
                let tid = windows::Win32::System::Threading::GetCurrentThreadId();
                let _ = tx.send(tid);
                HOOK_RUNNING.store(true, Ordering::Release);
                HOOK_THREAD_HEARTBEAT_MS.store(route_clock_ms(), Ordering::Release);
                let heartbeat_timer = SetTimer(None, INPUT_HOOK_HEARTBEAT_TIMER_ID, 250, None);
                HOOK_HEARTBEAT_ENABLED.store(heartbeat_timer != 0, Ordering::Release);
                if heartbeat_timer == 0 {
                    log::warn!(
                        "mouse-hook heartbeat timer unavailable; hook-stall watchdog disabled"
                    );
                }
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    if msg.message == WM_TIMER && msg.wParam.0 == heartbeat_timer {
                        HOOK_THREAD_HEARTBEAT_MS.store(route_clock_ms(), Ordering::Release);
                        continue;
                    }
                    DispatchMessageW(&msg);
                }
                HOOK_HEARTBEAT_ENABLED.store(false, Ordering::Release);
                if heartbeat_timer != 0 {
                    let _ = KillTimer(None, heartbeat_timer);
                }
                if let Ok(h) = hook {
                    let _ = UnhookWindowsHookEx(h);
                }
                // final safety: never leave a clip, hidden cursor, or stuck
                // button behind if the hook thread is asked to stop.
                emergency_release_all();
                HOOK_RUNNING.store(false, Ordering::Release);
            })
            .expect("mouse hook thread");
        let thread_id = rx.recv().unwrap_or(0);
        Self {
            thread_id,
            handle: Some(handle),
            done_rx,
        }
    }

    /// Force Windows/native cursor ownership while a TensorRT engine is being
    /// created. This is separate from provider-transition suspension because
    /// lazy shape builds also occur later after capture-resolution changes.
    pub fn set_tensorrt_build_cursor_guard(active: bool) {
        let previous = TENSORRT_BUILD_NATIVE_CURSOR_GUARD.swap(active, Ordering::AcqRel);
        if previous == active {
            return;
        }

        if active {
            WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
            // Prevent both the currently visible sprite and an already queued
            // WM_APP move from resurfacing during the native-only build epoch.
            SPRITE_SHOW.store(false, Ordering::Release);
            sprite_hide_window_only();
            if let Ok(mut g) = state().try_lock() {
                clear_capture_sprite_contract(&mut g);
                if g.engaged || g.cursor_hidden || g.pending_engage.is_some() {
                    release_locked(&mut g);
                }
                g.active = false;
                g.buttons_down = 0;
                g.native_ui_hold_bits = 0;
                g.last_gui_sprite_sync_at = None;
            }
            log::info!("TensorRT build cursor guard: native-only");
        } else {
            // Do not force an immediate native->sprite handoff. The next normal
            // configure/mouse edge re-enters the proven capture cursor contract.
            LAST_CURSOR_ASSERT.with(|last| last.set(None));
            log::info!("TensorRT build cursor guard: released");
        }
    }

    /// Resume provider-transition input mapping from render paths where the
    /// local name `input` may refer to a GPU texture rather than InputSystem.
    /// This is intentionally the exact lightweight `false` half of
    /// set_transition_suspended(): no cursor ownership is created here.
    pub(crate) fn resume_transition_suspended() {
        let mut g = state().lock().unwrap();
        g.last_configure_at = Some(std::time::Instant::now());
        if !g.transition_suspended {
            return;
        }
        g.transition_suspended = false;
        log::info!("input mapping resumed after provider transition");
    }

    /// Explicitly suspend cursor mapping during a provider/session transition.
    /// This is distinct from the emergency heartbeat path: a cold TensorRT
    /// build is expected work, so release ownership immediately and wait for
    /// the first successfully presented frame before allowing re-engagement.
    pub fn set_transition_suspended(&self, suspended: bool) {
        let mut g = state().lock().unwrap();
        if g.transition_suspended == suspended {
            g.last_configure_at = Some(std::time::Instant::now());
            return;
        }
        g.transition_suspended = suspended;
        g.last_configure_at = Some(std::time::Instant::now());
        if suspended {
            clear_capture_sprite_contract(&mut g);
            let mapping_was_live = g.active
                || g.engaged
                || g.pending_engage.is_some()
                || g.src_hwnd != 0
                || g.cursor_hidden;
            if mapping_was_live {
                release_locked(&mut g);
            } else {
                // At initial capture startup there is no source mapping to
                // release yet. `virt` is still its default (0,0); routing that
                // through release_locked() teleports the user's native cursor
                // to monitor origin before the first filtered frame. Preserve
                // the actual desktop point instead.
                let actual = read_cursor_pos_or(
                    (g.virt.0.round() as i32, g.virt.1.round() as i32),
                    "provider-transition inactive preserve",
                );
                unsafe {
                    let _ = ClipCursor(None);
                }
                g.virt = (actual.0 as f64, actual.1 as f64);
                g.last_set = actual;
                g.last_hw = actual;
                g.source_direct_bits = 0;
                request_cursor_hidden(false);
                sprite_hide();
                log::debug!(
                    "provider-transition inactive cursor preserved: actual=({},{})",
                    actual.0,
                    actual.1
                );
            }
            g.active = false;
            g.src_hwnd = 0;
            g.buttons_down = 0;
            g.native_ui_hold_bits = 0;
            g.source_direct_bits = 0;
            g.last_gui_sprite_sync_at = None;
            log::info!("input mapping suspended during provider transition");
            drop(g);

            // Provider switches, especially a cold TensorRT shape/engine build,
            // can block this render/Magnification-owner thread for tens of
            // seconds immediately after this call. release_locked() only
            // REQUESTS native-cursor reveal; if we enter ORT/TensorRT before
            // pump_cursor_visibility() runs, the GUI sprite stops updating while
            // the real cursor remains hidden and appears frozen over the build
            // popup. Commit the reveal synchronously before any blocking build.
            WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
            NATIVE_GUI_OWNER.store(0, Ordering::Release);
            ACTIVE_OVERLAY_HWND.store(0, Ordering::Release);
            LAST_CURSOR_ASSERT.with(|last| last.set(None));
            pump_cursor_visibility();
            if CURSOR_HIDE_APPLIED.load(Ordering::Acquire) {
                // One immediate retry is cheap and protects against a transient
                // Magnification ownership/compositor miss. Never spin here.
                LAST_CURSOR_ASSERT.with(|last| last.set(None));
                pump_cursor_visibility();
            }
            log::debug!(
                "provider-transition cursor release committed: native_hidden={}",
                CURSOR_HIDE_APPLIED.load(Ordering::Acquire)
            );
            return;
        }

        log::info!("input mapping resumed after provider transition");
    }

    /// Update the geometry/activity for the engage logic (engine tick).
    pub fn configure(
        &self,
        active: bool,
        fullscreen: bool,
        window_frame_input: bool,
        input_reference_kind: &'static str,
        overlay: Rect,
        overlay_hwnd: isize,
        content: Rect,
        src: Rect,
        src_hwnd: isize,
        no_engage: Vec<NoEngageRect>,
        autohide_secs: f32,
        adjust_speed: bool,
    ) {
        // Publish the visible panel HWND outside the main state lock so the LL
        // mouse hook always has a lock-free first-click route. The engine only
        // includes a panel zone while that panel is logically visible.
        let active_panel = if active {
            no_engage
                .iter()
                .find(|hit| hit.panel && hit.hwnd != 0)
                .map_or(0, |hit| hit.hwnd)
        } else {
            0
        };
        ACTIVE_PANEL_HWND.store(active_panel, Ordering::Release);
        publish_deferred_source_geometry(
            active,
            !window_frame_input,
            src_hwnd,
            content,
            src,
            &no_engage,
        );

        let was_session_active = CAPTURE_SESSION_ACTIVE.load(Ordering::Acquire);
        let mut g = state().lock().unwrap();
        // Do not publish inactive until Windows ownership has actually been
        // released below. If this configure call stalls on teardown, the
        // watchdog must remain armed rather than assuming the cursor is safe.
        if active {
            CAPTURE_SESSION_ACTIVE.store(true, Ordering::Release);
        }
        if active && !was_session_active {
            // Only a genuine inactive -> active capture-session edge may rearm
            // cursor ownership after a fail-safe latch. A stalled session that
            // resumes still sees was_session_active=true and remains pass-through.
            INPUT_FAILSAFE_LATCHED.store(false, Ordering::Release);
            CURSOR_OWNER_HEARTBEAT_MS.store(route_clock_ms(), Ordering::Release);
            HOOK_THREAD_HEARTBEAT_MS.store(route_clock_ms(), Ordering::Release);
            log::info!("input-failsafe-rearmed: reason=new-capture-session");
        }
        if INPUT_FAILSAFE_LATCHED.load(Ordering::Acquire)
            && (g.engaged || g.cursor_hidden || g.pending_engage.is_some())
        {
            release_locked(&mut g);
            g.buttons_down = 0;
            g.native_ui_hold_bits = 0;
        }
        g.last_configure_at = Some(std::time::Instant::now());
        g.fullscreen = fullscreen;
        let caption_drag_diag = g.buttons_down != 0 && source_caption_drag_active();
        let drag_diag_due = if caption_drag_diag {
            let now_ms = route_clock_ms();
            let next_ms = DRAG_GEOMETRY_DIAG_NEXT_MS.load(Ordering::Relaxed);
            if now_ms >= next_ms {
                DRAG_GEOMETRY_DIAG_NEXT_MS.store(now_ms + 100, Ordering::Relaxed);
                true
            } else {
                false
            }
        } else {
            true
        };
        let input_space_changed = g.window_frame_input != window_frame_input
            || g.input_reference_kind != input_reference_kind
            || g.src != src
            || g.content != content;
        if input_space_changed && drag_diag_due {
            log::info!(
                "input-coordinate-space: kind={} reference={} src={:?} content={:?} fullscreen={} drag_sampled={}",
                if window_frame_input {
                    "window-frame"
                } else {
                    "client"
                },
                input_reference_kind,
                src,
                content,
                fullscreen,
                caption_drag_diag
            );
        }
        g.window_frame_input = window_frame_input;
        g.input_reference_kind = input_reference_kind;
        g.overlay = overlay;
        ACTIVE_OVERLAY_HWND.store(if active { overlay_hwnd } else { 0 }, Ordering::Release);
        // During a caption drag the high-rate follower (or its explicit
        // engine fallback) is the ONLY overlay-origin publisher. The render/WGC
        // configure path is a different clock and must never overwrite the
        // fixed visual grab anchor with an asynchronously sampled content rect.
        g.adjust_speed = adjust_speed;
        let was_engaged = g.engaged;
        let had_source_ownership = g.src_hwnd != 0;
        // Source/overlay geometry can move while engaged. Preserve the visible
        // virtual cursor across the new content rect, then move the hidden real
        // cursor to the matching source point. Rebuilding the sprite from
        // GetCursorPos() would use the clipped source-edge cursor and produce
        // the inward "source edge" pull-back.
        if g.pending_engage.is_some() && (src != g.src || content != g.content) {
            log::info!(
                "cursor handoff cancelled by geometry change: old_src={:?} new_src={:?}",
                g.src,
                src
            );
            release_windows(&mut g);
            g.engaged = false;
            publish_deferred_source_engaged(false);
            g.expect_teleport = None;
            g.teleport_guard_until = None;
        }
        if g.engaged && g.pending_engage.is_none() && (src != g.src || content != g.content) {
            // A caption drag has an explicit visual grab anchor. Geometry
            // publication may arrive from the WGC/render loop at a different
            // instant than the 8ms source follower, so never derive the sprite
            // from those two snapshots while the button is held.
            let anchored_drag_visual = if caption_drag_diag {
                source_caption_drag_visual_position()
            } else {
                None
            };
            let (vx, vy) = if g.buttons_down == 0 {
                remap_virtual_between_content(g.content, content, g.virt.0, g.virt.1)
            } else if let Some((vx, vy)) = anchored_drag_visual {
                if drag_diag_due {
                    log::debug!(
                        "drag-geometry-anchor-preserved: virt=({vx},{vy}) old_content={:?} new_content={:?} old_src={:?} new_src={:?} sampled_100ms=true",
                        g.content,
                        content,
                        g.src,
                        src
                    );
                }
                (vx as f64, vy as f64)
            } else {
                if drag_diag_due {
                    log::debug!(
                        "drag-geometry-remap-bypassed: virt=({:.1},{:.1}) old_content={:?} new_content={:?} old_src={:?} new_src={:?} sampled_100ms={}",
                        g.virt.0,
                        g.virt.1,
                        g.content,
                        content,
                        g.src,
                        src,
                        caption_drag_diag
                    );
                }
                g.virt
            };
            g.virt = (vx, vy);
            let (tx, ty) = map_content_to_source(content, src, vx, vy);
            if g.buttons_down == 0 {
                let (remapped, actual) = warp_then_clip_source((tx, ty), src);
                if remapped {
                    g.last_set = actual;
                    g.last_hw = actual;
                } else {
                    log::warn!(
                        "geometry-change cursor remap rejected: target=({tx},{ty}) actual=({},{}) src={:?}",
                        actual.0,
                        actual.1,
                        src
                    );
                }
            } else {
                let actual = read_cursor_pos_or(g.last_set, "geometry change with button held");
                g.last_set = actual;
                g.last_hw = actual;
                let clip = clip_rect(src);
                unsafe {
                    let _ = ClipCursor(Some(&clip));
                }
            }
            if !g.hidden_by_idle && !caption_drag_diag {
                sprite_move(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
            }
        }
        let effective_active = active
            && !g.transition_suspended
            && !tensorrt_build_native_cursor_guard()
            && !INPUT_FAILSAFE_LATCHED.load(Ordering::Acquire);
        g.active = effective_active;
        CAPTURE_SPRITE_ACTIVE.store(effective_active, Ordering::Release);
        if !effective_active {
            cancel_source_caption_drag_contract();
        }
        g.content = content;
        g.src = src;
        g.src_hwnd = src_hwnd;
        let no_engage_changed = g.no_engage != no_engage;
        if no_engage_changed {
            g.no_engage_moved_at = Some(std::time::Instant::now());
        }
        g.no_engage = no_engage;
        drain_deferred_source_input(&mut g);
        if no_engage_changed {
            // Full/Basic/Mini changes keep the same GUI HWND while replacing
            // its rectangle. `native_gui_owner_hwnd` is deliberately not
            // invalidated here; physical ownership is independently re-tested
            // from the live HWND rectangle on the next hook event. A hover
            // remembered from the old, larger layout
            // must not keep blocking source re-engage over space that the GUI
            // no longer occupies. Preserve the grace only when the exact hit
            // rectangle is still part of the current no-engage set.
            if let Some((old_hit, replacement)) =
                invalidate_stale_ui_hover_after_no_engage_change(&mut g)
            {
                log::info!(
                    "stale GUI hover invalidated after no-engage geometry change: hwnd={:#x} old={:?} replacement={:?}",
                    old_hit.hwnd,
                    old_hit.land,
                    replacement.map(|r| r.land)
                );
            }
        }
        if effective_active
            && !g.engaged
            && g.last_ui_hover
                .filter(|h| h.hit.hwnd != 0 && !is_panel_hit(h.hit))
                .is_some()
        {
            unsafe {
                let _ = ClipCursor(None);
            }
        }
        g.autohide_secs = autohide_secs;
        if !effective_active
            && (was_engaged
                || had_source_ownership
                || g.cursor_hidden
                || WANT_CURSOR_HIDDEN.load(Ordering::Acquire))
        {
            // Geometry can disappear while the logical engaged flag has
            // already been cleared (PIP replacement, render-thread error,
            // ratio transition). Always release the actual Windows ownership
            // state as well, not only the high-level engaged state.
            release_locked(&mut g);
            g.buttons_down = 0;
            g.native_ui_hold_bits = 0;
            g.oscillations = 0;
        }
        if !effective_active {
            g.src_hwnd = 0;
            publish_deferred_source_geometry(
                false,
                false,
                0,
                Rect::default(),
                Rect::default(),
                &[],
            );
            clear_deferred_source_queue();
        } else if g.engaged && g.pending_engage.is_none() {
            publish_deferred_source_engaged(!g.window_frame_input && g.src_hwnd != 0);
        }
        // The topmost GUI rect can arrive one GUI tick after the first filtered
        // overlay frame. If the stationary virtual cursor is already inside it,
        // hand ownership back immediately instead of waiting for mouse movement.
        if no_engage_changed && g.engaged && !g.hidden_by_idle && g.buttons_down == 0 {
            let now = std::time::Instant::now();
            if let Some(hover) = note_ui_hover(&mut g, now) {
                if hover.hit.hwnd != 0 && !is_panel_hit(hover.hit) {
                    if handoff_to_gui(&mut g, hover, now) {
                        // configure() is called by the Magnification owner thread,
                        // so commit the native-cursor reveal in this same tick.
                        pump_cursor_visibility();
                        log::info!(
                            "native-window geometry handoff committed without mouse movement: hwnd={:#x} at ({},{})",
                            hover.hit.hwnd,
                            hover.pos.0,
                            hover.pos.1
                        );
                    } else {
                        log::warn!(
                            "native-window geometry handoff deferred after unverified warp: hwnd={:#x} at ({},{})",
                            hover.hit.hwnd,
                            hover.pos.0,
                            hover.pos.1
                        );
                    }
                }
            }
        }
        // cursor auto-hide (engine tick drives the timeout)
        let over_ui = g.engaged && virtual_over_ui(&g);
        if over_ui {
            g.hidden_by_idle = false;
            sprite_move(g.virt.0.round() as i32, g.virt.1.round() as i32, true);
            keep_cursor_sprite_on_top();
        } else {
            g.ui_hover_active = false;
        }
        if should_hide_cursor_for_idle(&g, std::time::Instant::now()) {
            g.hidden_by_idle = true;
            sprite_hide();
        }
        if !active {
            CAPTURE_SESSION_ACTIVE.store(false, Ordering::Release);
        }
    }

    /// Complete the first cursor handoff before the filtered overlay is shown.
    /// Called only by the render-engine thread, which owns Magnification.
    pub fn prepare_overlay_reveal() -> bool {
        let mut g = state().lock().unwrap();
        g.active = true;
        CAPTURE_SPRITE_ACTIVE.store(true, Ordering::Release);
        if g.engaged {
            if g.cursor_hidden {
                request_cursor_hidden(true);
                pump_cursor_visibility();
                return CURSOR_HIDE_APPLIED.load(Ordering::Acquire);
            }
            return true;
        }

        let (px, py) = unsafe {
            let mut pt = POINT::default();
            if GetCursorPos(&mut pt).is_err() {
                return true;
            }
            (pt.x, pt.y)
        };
        let native_hit = unsafe { WindowFromPoint(POINT { x: px, y: py }) };
        if !native_hit.0.is_null() && crate::platform::win32::is_own_window(native_hit.0 as isize) {
            // In particular, "bring GUI to front" leaves the stationary cursor
            // on our normal GUI. Keep Windows' native cursor from the outset;
            // engaging here would create a sprite until the delayed GUI rect
            // reaches the engine.
            sprite_move_now(px, py, true);
            request_cursor_hidden(true);
            pump_cursor_visibility();
            log::info!(
                "pre-reveal cursor uses capture sprite over own window: hwnd={:?} at ({px},{py})",
                native_hit.0
            );
            return true;
        }
        let now = std::time::Instant::now();
        g.last_move = Some(now);
        let Some(e) = plan_engage_for_overlay_reveal(&g, px, py, now) else {
            sprite_move_now(px, py, true);
            request_cursor_hidden(true);
            pump_cursor_visibility();
            return true;
        };

        g.must_leave_content = false;
        g.engaged = true;
        g.virt = (e.vx, e.vy);
        g.last_set = (e.tx, e.ty);
        g.last_hw = (px, py);
        g.edge_out_accum = 0.0;
        g.last_engage_at = Some(now);
        g.expect_teleport = Some((e.tx, e.ty));
        g.hidden_by_idle = false;
        apply_engage_windows(&mut g, px, py, e.tx, e.ty, e.vx, e.vy, now, true)
    }

    /// Force-release the clip (stop paths).
    pub fn release(&self) {
        CAPTURE_SPRITE_ACTIVE.store(false, Ordering::Release);
        cancel_source_caption_drag_contract();
        let mut g = state().lock().unwrap();
        release_locked(&mut g);
        g.active = false;
        g.buttons_down = 0;
        g.native_ui_hold_bits = 0;
        g.oscillations = 0;
        drop(g);
        // A completed Stop is the only point that rearms the next capture.
        // Emergency release itself intentionally keeps this true while a wedged
        // session may still be alive, preventing that same session from silently
        // reclaiming the cursor after the fail-safe fired.
        CAPTURE_SESSION_ACTIVE.store(false, Ordering::Release);
    }

    pub fn stop(&mut self) {
        // Application close must never wait indefinitely for the WH_MOUSE_LL
        // thread *or* for the normal input-state mutex. Ordinary capture Stop
        // still uses release(), preserving the stable v578-style route. App
        // shutdown is different: revoke ownership lock-free and ask the hook
        // thread to unwind with a bounded wait. The full visible/input recovery
        // has already been requested by the application-close path.
        CAPTURE_SESSION_ACTIVE.store(false, Ordering::Release);
        CAPTURE_SPRITE_ACTIVE.store(false, Ordering::Release);
        SOURCE_CLIENT_DRAG_OWNER_HWND.store(0, Ordering::Release);
        cancel_source_caption_drag_contract();
        // Cmd::Shutdown is reached only after the application-close visible
        // recovery has already requested cursor/input release. Do not run the
        // full emergency recovery a second time from InputSystem::stop(): that
        // path touches foreign/system cursor presentation and was observed to
        // consume the entire 2.5 s outer shutdown budget. The hook shutdown
        // itself must be strictly bounded and lock-free.
        WANT_CURSOR_HIDDEN.store(false, Ordering::Release);
        NATIVE_GUI_OWNER.store(0, Ordering::Release);
        ACTIVE_OVERLAY_HWND.store(0, Ordering::Release);
        unsafe {
            let _ = ClipCursor(None);
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
        if let Some(h) = self.handle.take() {
            match self
                .done_rx
                .recv_timeout(std::time::Duration::from_millis(50))
            {
                Ok(()) => {
                    let _ = h.join();
                    log::debug!("mouse-hook shutdown complete");
                }
                Err(_) => {
                    // Dropping a JoinHandle detaches the thread. The visible
                    // recovery path has already released input best-effort, and
                    // the external janitor is the final cursor/clip safety net if
                    // this process exits before the hook thread unwinds. Never
                    // hold the GUI close for it.
                    log::warn!("mouse-hook shutdown deferred: elapsed_ms=50 action=detach");
                    drop(h);
                }
            }
        }
    }
}

impl Drop for InputSystem {
    fn drop(&mut self) {
        self.stop();
    }
}

/// TEST ONLY: run the engage logic as if a hardware move happened at (x,y).
pub fn debug_force_move(x: i32, y: i32) {
    let _ = on_hardware_move(x, y);
}

/// TEST ONLY: run a move and report whether the real hook would consume it.
pub fn debug_force_move_consumed(x: i32, y: i32) -> bool {
    on_hardware_move(x, y)
}

/// TEST ONLY: last edge-release cursor placement requested by the input layer.
pub fn debug_last_edge_release() -> Option<(i32, i32)> {
    let x = DEBUG_LAST_EDGE_PLACE_X.load(Ordering::Acquire);
    let y = DEBUG_LAST_EDGE_PLACE_Y.load(Ordering::Acquire);
    if x == i32::MIN || y == i32::MIN {
        None
    } else {
        Some((x, y))
    }
}

/// TEST ONLY: clear the recorded edge-release placement.
pub fn debug_clear_last_edge_release() {
    DEBUG_LAST_EDGE_PLACE_X.store(i32::MIN, Ordering::Release);
    DEBUG_LAST_EDGE_PLACE_Y.store(i32::MIN, Ordering::Release);
}

/// TEST ONLY: route a hardware mouse button while engaged.
pub fn debug_force_button(msg: u32) -> bool {
    forward_engaged_button(msg)
}

/// The current VIRTUAL cursor position (content/screen space) while engaged, so
/// the GUI can detect the sprite hovering its collapsed control-panel chip and
/// auto-expand it — the real cursor is confined to the source and never lands
/// on the chip, so a plain GetCursorPos hit-test would miss it (the cursor-routing design's
/// `_check_panel_hover`). None when not engaged.
pub fn virtual_cursor_pos() -> Option<(i32, i32)> {
    let g = state().try_lock().ok()?;
    if g.active && g.engaged {
        Some((g.virt.0.round() as i32, g.virt.1.round() as i32))
    } else {
        None
    }
}

/// Lock-free visible virtual-cursor sample for the floating control panel.
///
/// The normal `virtual_cursor_pos()` intentionally reads the full input state,
/// but its `try_lock()` can transiently fail while the low-level hook is
/// processing a mouse packet.  A transient `None` must not be interpreted as
/// "the pointer left the panel" because that makes a custom-painted hover
/// button blink.  The sprite mover already publishes the latest visible point
/// atomically, so the panel can consume that exact display coordinate without
/// touching the input mutex.
pub fn panel_virtual_cursor_pos_lockfree() -> Option<(i32, i32)> {
    if !sprite_visibility_allowed(
        SPRITE_SHOW.load(Ordering::Acquire),
        CURSOR_HIDE_APPLIED.load(Ordering::Acquire),
    ) {
        return None;
    }
    let packed = SPRITE_TARGET.load(Ordering::Acquire);
    Some((packed as i32, (packed >> 32) as i32))
}

/// Compute the on-screen content rect (letterbox area inside the overlay
/// where the source image actually appears) — must match OverlayWindow::present.
pub fn content_rect(overlay: Rect, frame_w: i32, frame_h: i32) -> Rect {
    if frame_w <= 0 || frame_h <= 0 || overlay.w <= 0 || overlay.h <= 0 {
        return overlay;
    }
    let scale = (overlay.w as f64 / frame_w as f64).min(overlay.h as f64 / frame_h as f64);
    let vw = (frame_w as f64 * scale).round() as i32;
    let vh = (frame_h as f64 * scale).round() as i32;
    Rect {
        x: overlay.x + (overlay.w - vw) / 2,
        y: overlay.y + (overlay.h - vh) / 2,
        w: vw.max(1),
        h: vh.max(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabling_main_control_clears_queued_and_pressed_state() {
        MAIN_ACTIONS.store(MAIN_ACTION_STOP, Ordering::Release);
        MAIN_DIRECT_ACTIVE.store(true, Ordering::Release);

        set_main_control_surface(0x1234, None, MAIN_CONTROL_DISABLED);

        assert_eq!(take_main_actions(), 0);
        assert!(!main_control_direct_pressed());
        assert_eq!(
            MAIN_CONTROL_MODE.load(Ordering::Acquire),
            MAIN_CONTROL_DISABLED
        );
    }

    #[test]
    fn letterbox_content() {
        // 16:9 frame in a square overlay -> horizontal bars
        let c = content_rect(
            Rect {
                x: 100,
                y: 100,
                w: 400,
                h: 400,
            },
            1920,
            1080,
        );
        assert_eq!(c.w, 400);
        assert_eq!(c.h, 225);
        assert_eq!(c.x, 100);
        assert_eq!(c.y, 100 + (400 - 225) / 2);
    }

    #[test]
    fn roundtrip_mapping_error_subpixel() {
        // content -> source -> content must round-trip within 1px
        let c = Rect {
            x: 0,
            y: 20,
            w: 1440,
            h: 960,
        };
        let s = Rect {
            x: 300,
            y: 200,
            w: 480,
            h: 320,
        };
        for (px, py) in [(10, 30), (700, 500), (1430, 970)] {
            let (tx, ty) = map_content_to_source(c, s, px as f64, py as f64);
            let bx = (c.x as f64 + ((tx - s.x) as f64 + 0.5) * c.w as f64 / s.w as f64 - 0.5)
                .round() as i32;
            let by = (c.y as f64 + ((ty - s.y) as f64 + 0.5) * c.h as f64 / s.h as f64 - 0.5)
                .round() as i32;
            assert!(
                (bx - px).abs() <= 1 && (by - py).abs() <= 1,
                "err too big: {px},{py} -> {bx},{by}"
            );
        }
    }

    #[test]
    fn content_source_mapping_hits_edges() {
        let c = Rect {
            x: 100,
            y: 50,
            w: 3840,
            h: 2160,
        };
        let s = Rect {
            x: 841,
            y: 122,
            w: 1040,
            h: 785,
        };
        assert_eq!(
            map_content_to_source(c, s, c.x as f64, c.y as f64),
            (s.x, s.y)
        );
        assert_eq!(
            map_content_to_source(c, s, (c.x + c.w - 1) as f64, (c.y + c.h - 1) as f64),
            (s.x + s.w - 1, s.y + s.h - 1)
        );
    }

    #[test]
    fn source_delta_spans_scaled_content() {
        let c = Rect {
            x: 200,
            y: 100,
            w: 1920,
            h: 1080,
        };
        let s = Rect {
            x: 20,
            y: 30,
            w: 640,
            h: 360,
        };
        let (sx, sy) = content_per_source_px(c, s);
        assert!((sx - 3.0).abs() < f64::EPSILON);
        assert!((sy - 3.0).abs() < f64::EPSILON);

        let mut vx = c.x as f64;
        for _ in 0..(s.w - 1) {
            vx += sx;
        }
        assert!(
            vx >= (c.x + c.w - 4) as f64,
            "virtual cursor did not reach the magnified edge: {vx}"
        );
    }

    #[test]
    fn source_to_content_roundtrip_is_within_one_pixel() {
        let c = Rect {
            x: 100,
            y: 50,
            w: 1920,
            h: 1080,
        };
        let s = Rect {
            x: 878,
            y: 190,
            w: 1040,
            h: 785,
        };
        for (sx, sy) in [(s.x, s.y), (1200, 500), (s.x + s.w - 1, s.y + s.h - 1)] {
            let (vx, vy) = map_source_to_content(c, s, sx as f64, sy as f64);
            let (back_x, back_y) = map_content_to_source(c, s, vx, vy);
            assert!(
                (back_x - sx).abs() <= 1 && (back_y - sy).abs() <= 1,
                "{sx},{sy} -> {vx:.2},{vy:.2} -> {back_x},{back_y}"
            );
        }
    }

    #[test]
    fn engage_requires_inner_margin_to_prevent_edge_resuck() {
        let c = Rect {
            x: 100,
            y: 50,
            w: 800,
            h: 450,
        };
        assert!(!contains_for_engage(c, 100, 200));
        assert!(!contains_for_engage(c, 107, 200));
        assert!(contains_for_engage(c, 108, 200));
        assert!(!contains_for_engage(c, 899, 200));
        assert!(!contains_for_engage(c, 892, 200));
        assert!(contains_for_engage(c, 891, 200));
    }

    #[test]
    fn panel_no_engage_is_exact_visible_rect() {
        let panel = Rect {
            x: 862,
            y: 4,
            w: 126,
            h: 33,
        };
        let hit = panel_no_engage_rect(panel);
        assert_eq!(hit.rect, panel);
        assert_eq!(hit.land, panel);
        assert!(hit.panel);
        assert!(hit.contains(panel.x, panel.y));
        assert!(hit.contains(panel.x + panel.w - 1, panel.y + panel.h - 1));
        assert!(!hit.contains(panel.x - 1, panel.y + 10));
        assert!(!hit.contains(panel.x + panel.w, panel.y + 10));
        assert!(!hit.contains(panel.x + 10, panel.y - 1));
        assert!(!hit.contains(panel.x + 10, panel.y + panel.h));
    }

    #[test]
    fn panel_identity_is_explicit_not_geometry_shape() {
        let panel = Rect {
            x: 900,
            y: 4,
            w: 126,
            h: 30,
        };
        let gui = Rect {
            x: 50,
            y: 60,
            w: 600,
            h: 420,
        };
        assert!(is_panel_hit(panel_no_engage_rect(panel).with_hwnd(0x2222)));
        assert!(is_panel_hit(panel_no_engage_rect(panel)));
        assert!(!is_panel_hit(NoEngageRect::new(gui).with_hwnd(0x1111)));
    }

    #[test]
    fn panel_hover_and_click_point_is_never_magnetized() {
        let panel = Rect {
            x: 890,
            y: 4,
            w: 139,
            h: 33,
        };
        let hit = panel_no_engage_rect(panel).with_hwnd(0x2222);
        for point in [
            (panel.x, panel.y),
            (panel.x + 50, panel.y + 1),
            (panel.x + panel.w - 1, panel.y + panel.h - 1),
        ] {
            assert_eq!(ui_landing_point(hit, point.0, point.1), point);
        }
    }

    #[test]
    fn panel_action_regions_match_the_painted_layout() {
        let width = 270;
        assert_eq!(
            panel_action_for_relative_x(width, 30, 42),
            Some(PANEL_ACTION_STOP)
        );
        assert_eq!(panel_action_for_relative_x(width, 30, 120), None);
        assert_eq!(
            panel_action_for_relative_x(width, 30, 174),
            Some(PANEL_ACTION_SCREENSHOT)
        );
        assert_eq!(
            panel_action_for_relative_x(width, 30, 210),
            Some(PANEL_ACTION_GUI_TOPMOST)
        );
        assert_eq!(
            panel_action_for_relative_x(width, 30, 250),
            Some(PANEL_ACTION_COLLAPSE)
        );
        assert_eq!(panel_action_for_relative_x(width, 30, -1), None);
        assert_eq!(panel_action_for_relative_x(width, 30, width), None);
    }

    #[test]
    fn panel_lurk_hit_area_is_dpi_independent() {
        // 68x24pt lurk geometry at representative physical scales. Every pixel
        // across the transparent target must restore the full panel.
        for (w, h) in [(68, 24), (75, 27), (136, 48), (204, 72)] {
            assert_eq!(
                panel_action_for_relative_x(w, h, 0),
                Some(PANEL_ACTION_EXPAND)
            );
            assert_eq!(
                panel_action_for_relative_x(w, h, w - 1),
                Some(PANEL_ACTION_EXPAND)
            );
        }

        // Full-bar geometry must never be mistaken for a lurk target, even at
        // small or high DPI scales.
        assert_eq!(
            panel_action_for_relative_x(135, 15, 10),
            Some(PANEL_ACTION_STOP)
        );
        assert_eq!(panel_action_for_relative_x(540, 60, 240), None);
    }

    #[test]
    fn slow_speed_halves_multiplier_at_2x() {
        // slider 10 = 1.0×; at zoom 2 we want ~0.5× → slider 6 (0.5). And the
        // slowed speed is always <= the original so the sprite never overshoots.
        assert_eq!(slow_speed_for_zoom(10, 2.0), 6);
        for cur in 1..=20 {
            for &z in &[1.2_f64, 1.5, 2.0, 3.0, 4.0] {
                let s = slow_speed_for_zoom(cur, z);
                assert!(s >= 1 && s <= 20);
                assert!(
                    SPEED_MULT[(s - 1) as usize] <= SPEED_MULT[(cur - 1) as usize] + 1e-9,
                    "slowed faster than original: cur={cur} z={z} -> {s}"
                );
            }
        }
    }

    #[test]
    fn offscreen_cursor_reveal_target_clamps_to_last_reachable_pixel() {
        // 2026-08-15 regression: a 1024x768 window at y=312 ends exactly at
        // the 1080p desktop edge. EXIT_WINDOW_MARGIN_PX produced y=1086, so
        // SetCursorPos was permanently clamped by Windows to y=1079 and the
        // deferred reveal loop hid the cursor until Stop.
        let monitor = Rect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        assert_eq!(
            clamp_point_to_monitor_rect((1426, 1086), monitor),
            (1426, 1079)
        );
        assert_eq!(
            clamp_point_to_monitor_rect((1926, 500), monitor),
            (1919, 500)
        );
        assert_eq!(clamp_point_to_monitor_rect((-6, 500), monitor), (0, 500));
    }

    #[test]
    fn sprite_stays_fully_visible_on_all_four_monitor_edges_and_corners() {
        // v429 left only ~8px visible at bottom/right. That can still look like
        // a vanished cursor, especially at the lower-left escape tested by the
        // user. Clamp the sprite HWND as a whole while leaving the logical point
        // untouched.
        let monitor = Rect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        assert_eq!(
            clamp_sprite_origin_to_monitor((1205, 1079), monitor, 24),
            (1205, 1056)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((1919, 500), monitor, 24),
            (1896, 500)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((0, 500), monitor, 24),
            (0, 500)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((500, 0), monitor, 24),
            (500, 0)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((0, 1079), monitor, 24),
            (0, 1056)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((1919, 1079), monitor, 24),
            (1896, 1056)
        );
        assert_eq!(
            clamp_sprite_origin_to_monitor((100, 500), monitor, 24),
            (100, 500)
        );
    }

    #[test]
    fn post_edge_raw_guard_rejects_pre_warp_backlog_but_not_normal_motion() {
        // v428 log: raw=(602,806), GetCursorPos=(378,804) immediately after
        // a verified left-edge transfer. That 224px disagreement is stale.
        assert!(post_edge_raw_is_stale(224, 2, true));
        // One-event-behind GetCursorPos during ordinary fast motion remains
        // below the hard divergence threshold and in the same coordinate domain.
        assert!(!post_edge_raw_is_stale(58, 20, false));
        // A domain disagreement near a content/UI boundary needs only the
        // smaller catch-up threshold to stay quarantined.
        assert!(post_edge_raw_is_stale(31, 5, true));
        assert!(!post_edge_raw_is_stale(20, 5, true));
    }

    #[test]
    fn exit_point_is_placed_outside_content_with_margin() {
        let c = Rect {
            x: 100,
            y: 50,
            w: 800,
            h: 450,
        };
        assert_eq!(
            exit_point_for_virtual(c, 99.0, 200.0),
            (c.x - EXIT_PLACE_MARGIN_PX, 200)
        );
        assert_eq!(
            exit_point_for_virtual(c, 900.0, 200.0),
            (c.x + c.w + EXIT_PLACE_MARGIN_PX, 200)
        );
        assert_eq!(
            exit_point_for_virtual(c, 300.0, 49.0),
            (300, c.y - EXIT_PLACE_MARGIN_PX)
        );
        assert_eq!(
            exit_point_for_virtual(c, 300.0, 500.0),
            (300, c.y + c.h + EXIT_PLACE_MARGIN_PX)
        );
    }

    #[test]
    fn edge_push_exit_uses_source_overshoot() {
        let c = Rect {
            x: 100,
            y: 50,
            w: 1000,
            h: 500,
        };
        let s = Rect {
            x: 10,
            y: 20,
            w: 28,
            h: 14,
        };
        assert_eq!(edge_out_amount(s, s.x + s.w - 1, s.y + 3), 0);
        assert_eq!(edge_out_amount(s, s.x + s.w + 2, s.y + 3), 2);
        assert_eq!(
            exit_point_for_source_edge(c, s, s.x + s.w + 2, s.y + 3).0,
            c.x + c.w + EXIT_PLACE_MARGIN_PX
        );
        assert_eq!(
            exit_point_for_source_edge(c, s, s.x - 2, s.y + 3).0,
            c.x - EXIT_PLACE_MARGIN_PX
        );
    }

    // ---- engaged-move planner scenario suite ----------------------------
    // Exercise the FULL engaged-move state machine (plan_engaged) with no
    // Windows calls, covering the patterns a user actually produces.

    fn engaged_state(content: Rect, src: Rect, panels: &[Rect]) -> State {
        State {
            active: true,
            engaged: true,
            overlay: content,
            content,
            src,
            src_hwnd: 0x1234,
            no_engage: panels.iter().map(|p| panel_no_engage_rect(*p)).collect(),
            last_hw: (src.x, src.y),
            last_set: (src.x, src.y),
            virt: (content.x as f64, content.y as f64),
            last_engage_at: Some(std::time::Instant::now()),
            adjust_speed: false,
            ..Default::default()
        }
    }

    // fullscreen: content = whole monitor, source = small hidden window
    fn fs() -> (Rect, Rect) {
        (
            Rect {
                x: 0,
                y: 0,
                w: 1920,
                h: 1080,
            },
            Rect {
                x: 795,
                y: 38,
                w: 973,
                h: 548,
            },
        )
    }
    // windowed: content = source * ~1.5
    fn win() -> (Rect, Rect) {
        (
            Rect {
                x: 200,
                y: 100,
                w: 1836,
                h: 1033,
            },
            Rect {
                x: 544,
                y: 38,
                w: 1224,
                h: 689,
            },
        )
    }
    // Windowed offset case: content (mag≈1.2) is centred over a
    // smaller source, so it extends ~97px LEFT of the source on screen and the
    // content-left edge maps right onto the source-left edge (the oscillation
    // geometry). content.x(724) -> source.x(821).
    fn win_offset() -> (Rect, Rect) {
        (
            Rect {
                x: 724,
                y: 99,
                w: 1159,
                h: 817,
            },
            Rect {
                x: 821,
                y: 167,
                w: 965,
                h: 681,
            },
        )
    }
    // HIGH magnification (3x) windowed geometry — the doc says the bug is most
    // visible at high zoom. content is 3x the source, concentric.
    fn win_hi() -> (Rect, Rect) {
        (
            Rect {
                x: 100,
                y: 100,
                w: 1200,
                h: 900,
            },
            Rect {
                x: 500,
                y: 250,
                w: 400,
                h: 300,
            },
        )
    }

    #[test]
    fn sprite_follows_raw_outward_at_edge_not_pulled_back() {
        // Regression case: while engaged, the ACTUAL
        // cursor is clamped at the source edge, but RAW input keeps moving
        // outward. The visible sprite (g.virt) MUST keep travelling toward/past
        // the content edge, never snap back to map(clamped source-edge). Checked
        // on all four edges, and (separately) DURING the post-engage grace.
        let (c, s) = win_hi();
        let raw_of = |ei: usize, k: i32| -> (i32, i32) {
            match ei {
                0 => (s.x - k, s.y + s.h / 2),           // left
                1 => (s.x + s.w - 1 + k, s.y + s.h / 2), // right
                2 => (s.x + s.w / 2, s.y - k),           // top
                _ => (s.x + s.w / 2, s.y + s.h - 1 + k), // bottom
            }
        };
        for ei in 0..4 {
            let mut g = engaged_state(c, s, &[]);
            let now = std::time::Instant::now();
            // actual pinned at the corresponding source edge
            let a = raw_of(ei, 0);
            let actual = (a.0.clamp(s.x, s.x + s.w - 1), a.1.clamp(s.y, s.y + s.h - 1));
            let mut prev = None::<(f64, f64)>;
            for k in 1..=40 {
                let raw = raw_of(ei, k * 2);
                let t = now + std::time::Duration::from_millis(200 + k as u64 * 8);
                let _ = plan_engaged(&mut g, raw.0, raw.1, t, actual);
                let (vx, vy) = g.virt;
                // the sprite must be OUTSIDE the content on the pushed axis
                let outside = match ei {
                    0 => vx <= c.x as f64 + 1.0,
                    1 => vx >= (c.x + c.w) as f64 - 1.0,
                    2 => vy <= c.y as f64 + 1.0,
                    _ => vy >= (c.y + c.h) as f64 - 1.0,
                };
                assert!(
                    outside,
                    "edge {ei}: sprite pulled back inside: virt=({vx},{vy})"
                );
                // and it must keep moving further OUT (never back toward inside)
                if let Some((pvx, pvy)) = prev {
                    let progressing = match ei {
                        0 => vx <= pvx + 0.001,
                        1 => vx >= pvx - 0.001,
                        2 => vy <= pvy + 0.001,
                        _ => vy >= pvy - 0.001,
                    };
                    assert!(
                        progressing,
                        "edge {ei}: sprite returned inward: {vx},{vy} after {pvx},{pvy}"
                    );
                }
                prev = Some((vx, vy));
                if !g.engaged {
                    break; // exited — the sprite was already outside, good
                }
            }
        }
    }

    #[test]
    fn sprite_follows_raw_outward_even_during_grace() {
        // The doc: even DURING the post-engage grace, the sprite must not be
        // sucked back to the source edge.
        let (c, s) = win_hi();
        let mut g = engaged_state(c, s, &[]);
        let engage_t = std::time::Instant::now();
        g.last_engage_at = Some(engage_t);
        let actual = (s.x, s.y + s.h / 2); // clamped left edge
        let mut prev = f64::INFINITY;
        for k in 1..=6 {
            let raw = (s.x - k * 3, s.y + s.h / 2);
            // WITHIN grace (< ENGAGE_GRACE_MS)
            let t = engage_t + std::time::Duration::from_millis(k as u64 * 5);
            let plan = plan_engaged(&mut g, raw.0, raw.1, t, actual);
            assert_eq!(plan, MovePlan::Stay, "should not escape during grace");
            assert!(
                g.virt.0 <= c.x as f64 + 1.0,
                "grace: sprite not outside: {}",
                g.virt.0
            );
            assert!(
                g.virt.0 < prev,
                "grace: sprite pulled back inward: {} !< {}",
                g.virt.0,
                prev
            );
            prev = g.virt.0;
        }
    }

    #[test]
    fn sprite_matches_clamped_mapping_in_interior() {
        // Sanity: away from the edges (raw == clamped actual), the raw-driven
        // sprite equals the old clamped mapping — no behaviour change interior.
        let (c, s) = win_hi();
        let mut g = engaged_state(c, s, &[]);
        let now = std::time::Instant::now();
        for (rx, ry) in [
            (s.x + 100, s.y + 100),
            (s.x + 200, s.y + 150),
            (s.x + 350, s.y + 250),
        ] {
            let _ = plan_engaged(&mut g, rx, ry, now, (rx, ry));
            let expect = map_source_to_content(c, s, rx as f64, ry as f64);
            assert!((g.virt.0 - expect.0).abs() < 0.01 && (g.virt.1 - expect.1).abs() < 0.01);
        }
    }

    #[test]
    fn engage_dead_band_sits_inside_source_edge() {
        // The core fix: a content position whose mapped SOURCE position is near
        // the source edge must NOT engage (else the next tiny push re-escapes).
        let (c, s) = win_offset();
        let mut g = engaged_state(c, s, &[]);
        g.engaged = false;
        let now = std::time::Instant::now();
        // content pos mapping to just inside the source-left edge -> blocked
        let (nx, ny) = map_source_to_content(c, s, (s.x + 2) as f64, (s.y + s.h / 2) as f64);
        assert!(
            plan_engage(&g, nx.round() as i32, ny.round() as i32, now).is_none(),
            "engaged right at the source edge (would oscillate)"
        );
        // content pos mapping WELL inside the source -> allowed
        let (dx, dy) = map_source_to_content(
            c,
            s,
            (s.x + ENGAGE_SRC_MARGIN_PX + 12) as f64,
            (s.y + s.h / 2) as f64,
        );
        assert!(
            plan_engage(&g, dx.round() as i32, dy.round() as i32, now).is_some(),
            "failed to engage well inside the source"
        );
    }

    #[test]
    fn cursor_timer_preserves_deferred_reveal_after_edge_exit() {
        assert!(
            cursor_reassert_want_hidden(false, true),
            "timer must not reveal the real cursor while an edge-exit reveal is pending"
        );
        assert!(
            cursor_reassert_want_hidden(true, false),
            "timer must keep hiding while still engaged"
        );
        assert!(
            !cursor_reassert_want_hidden(false, false),
            "timer may reveal only after both engagement and pending reveal are clear"
        );
    }

    #[test]
    fn sprite_waits_until_native_cursor_hide_is_committed() {
        assert!(!sprite_visibility_allowed(true, false));
        assert!(sprite_visibility_allowed(true, true));
        assert!(!sprite_visibility_allowed(false, true));
    }

    #[test]
    fn post_engage_grace_swallows_stale_edge_event() {
        // Regression case: engage near the source-left, then a queued
        // pre-teleport hook event arrives with a raw px far left of the source.
        // Within the post-engage grace it must NOT escape.
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        let engage_t = std::time::Instant::now();
        g.last_engage_at = Some(engage_t);
        let actual = (s.x, s.y + 200);
        // event 20ms after engage (inside the grace window)
        let plan = plan_engaged(
            &mut g,
            120,
            240,
            engage_t + std::time::Duration::from_millis(20),
            actual,
        );
        assert_eq!(
            plan,
            MovePlan::Stay,
            "stale event inside grace must not escape"
        );
        assert!(g.engaged);
    }

    #[test]
    fn many_normal_moves_never_escape() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        let now = std::time::Instant::now();
        for i in 0..=200 {
            let ax = s.x + (i * (s.w - 1) / 200);
            let ay = s.y + s.h / 2;
            let plan = plan_engaged(&mut g, ax, ay, now, (ax, ay));
            assert_eq!(plan, MovePlan::Stay, "normal move {i} escaped");
            assert!(g.engaged);
        }
    }

    #[test]
    fn sustained_edge_push_eventually_escapes_each_edge() {
        for edge in 0..4 {
            let (c, s) = win(); // windowed: edge push exits (fullscreen does not)
            let mut g = engaged_state(c, s, &[]);
            let t0 = std::time::Instant::now();
            let mut escaped = false;
            for k in 0..40 {
                let (ax, ay, px, py) = match edge {
                    0 => (s.x, s.y + 100, s.x - 5, s.y + 100),
                    1 => (s.x + s.w - 1, s.y + 100, s.x + s.w + 5, s.y + 100),
                    2 => (s.x + 100, s.y, s.x + 100, s.y - 5),
                    _ => (s.x + 100, s.y + s.h - 1, s.x + 100, s.y + s.h + 5),
                };
                let now = t0 + std::time::Duration::from_millis(k * 20);
                if let MovePlan::Escape { place, .. } = plan_engaged(&mut g, px, py, now, (ax, ay))
                {
                    match edge {
                        0 => assert!(place.0 < c.x),
                        1 => assert!(place.0 > c.x + c.w - 1),
                        2 => assert!(place.1 < c.y),
                        _ => assert!(place.1 > c.y + c.h - 1),
                    }
                    assert!(!g.engaged);
                    escaped = true;
                    break;
                }
            }
            assert!(escaped, "edge {edge} never escaped under sustained push");
        }
    }

    #[test]
    fn fullscreen_edge_push_never_exits() {
        // The core fullscreen fix: pushing HARD against every edge for a long
        // time must keep the cursor confined (the view is the whole screen and
        // the panel lives at the top edge). Covers the edge-transition case where the
        // cursor being flung to the left/top and the panel being unreachable.
        for edge in 0..4 {
            let (c, s) = fs();
            let mut g = engaged_state(c, s, &[]);
            g.fullscreen = true;
            let t0 = std::time::Instant::now();
            for k in 0..100 {
                let (ax, ay, px, py) = match edge {
                    0 => (s.x, s.y + 100, s.x - 60, s.y + 100),
                    1 => (s.x + s.w - 1, s.y + 100, s.x + s.w + 60, s.y + 100),
                    2 => (s.x + 100, s.y, s.x + 100, s.y - 60),
                    _ => (s.x + 100, s.y + s.h - 1, s.x + 100, s.y + s.h + 60),
                };
                let now = t0 + std::time::Duration::from_millis(k * 20);
                let plan = plan_engaged(&mut g, px, py, now, (ax, ay));
                assert_eq!(
                    plan,
                    MovePlan::Stay,
                    "fullscreen edge {edge} exited at k={k}"
                );
                assert!(g.engaged, "fullscreen edge {edge} disengaged");
            }
        }
    }

    #[test]
    fn a_single_edge_spike_does_not_escape() {
        let (c, s) = win();
        let mut g = engaged_state(c, s, &[]);
        let plan = plan_engaged(
            &mut g,
            s.x + s.w + 40,
            s.y + 100,
            std::time::Instant::now(),
            (s.x + s.w - 1, s.y + 100),
        );
        assert_eq!(plan, MovePlan::Stay);
        assert!(g.engaged);
    }

    #[test]
    fn dragging_source_never_escapes_across_the_edge() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        g.buttons_down = BTN_LEFT;
        let t0 = std::time::Instant::now();
        for k in 0..50 {
            let now = t0 + std::time::Duration::from_millis(k * 20);
            let plan = plan_engaged(
                &mut g,
                s.x + s.w + 30,
                s.y + 100,
                now,
                (s.x + s.w - 1, s.y + 100),
            );
            assert_eq!(plan, MovePlan::Stay, "drag escaped at k={k}");
            assert!(g.engaged);
        }
    }

    #[test]
    fn dragging_source_ignores_panel_zone() {
        // a drag whose virtual cursor passes over the panel must NOT hand off
        // (the drag belongs to the source).
        let (c, s) = fs();
        let panel = Rect {
            x: 897,
            y: 4,
            w: 126,
            h: 30,
        };
        let mut g = engaged_state(c, s, &[panel]);
        g.buttons_down = BTN_LEFT;
        let cx = panel.x + panel.w / 2;
        let cy = panel.y + panel.h / 2;
        let (tx, ty) = map_content_to_source(c, s, cx as f64, cy as f64);
        let plan = plan_engaged(&mut g, tx, ty, std::time::Instant::now(), (tx, ty));
        assert_eq!(plan, MovePlan::Stay);
        assert!(g.engaged);
    }

    #[test]
    fn panel_zone_stays_engaged_never_disengages() {
        // the cursor-routing design model: the virtual cursor moving over the panel must NOT
        // disengage (that was the fling/oscillation bug). The panel is reached
        // with the sprite and clicks are redirected in the hook instead.
        let (c, s) = fs();
        let panel = Rect {
            x: 897,
            y: 4,
            w: 126,
            h: 30,
        };
        let mut g = engaged_state(c, s, &[panel]);
        // sweep the virtual cursor all across the panel (map its centre + corners
        // back to source and feed those as clipped positions)
        for (fxp, fyp) in [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5), (0.5, 0.1), (0.5, 0.9)] {
            let vx = panel.x as f64 + fxp * panel.w as f64;
            let vy = panel.y as f64 + fyp * panel.h as f64;
            let (tx, ty) = map_content_to_source(c, s, vx, vy);
            let plan = plan_engaged(&mut g, tx, ty, std::time::Instant::now(), (tx, ty));
            assert_eq!(
                plan,
                MovePlan::Stay,
                "panel sweep disengaged at ({vx},{vy})"
            );
            assert!(g.engaged, "panel sweep disengaged");
            // the virtual cursor is reported over the panel (for hover-expand)
            let (rvx, rvy) = (g.virt.0.round() as i32, g.virt.1.round() as i32);
            assert!(
                panel.contains(rvx, rvy),
                "virtual cursor not over panel: ({rvx},{rvy})"
            );
        }
    }

    #[test]
    fn windowed_panel_first_pixel_beats_edge_release() {
        // The floating panel is glued immediately above the windowed content.
        // With a one-pixel escape threshold, the first outward source event must
        // remain engaged long enough for post_ui_hover_from_state() to perform
        // the native panel handoff instead of escaping to desktop first.
        let (c, s) = win();
        let panel = Rect {
            x: c.x + 120,
            y: c.y - 40,
            w: 300,
            h: 40,
        };
        let mut g = engaged_state(c, s, &[panel]);
        g.last_engage_at = None;
        let target_x = panel.x + panel.w / 2;
        let (raw_x, _) = map_content_to_source(c, s, target_x as f64, c.y as f64);
        let raw_y = s.y - 1;
        let actual = (raw_x, s.y);
        let plan = plan_engaged(&mut g, raw_x, raw_y, std::time::Instant::now(), actual);
        assert_eq!(plan, MovePlan::Stay);
        assert!(g.engaged);
        let hover = g
            .last_ui_hover
            .expect("panel first pixel should be recorded");
        assert!(is_panel_hit(hover.hit));
        assert!(panel.contains(hover.pos.0, hover.pos.1));
    }

    #[test]
    fn windowed_moves_stay_engaged_across_interior() {
        let (c, s) = win();
        let mut g = engaged_state(c, s, &[]);
        let now = std::time::Instant::now();
        for i in 0..=100 {
            let ax = s.x + (i * (s.w - 1) / 100);
            let ay = s.y + (i * (s.h - 1) / 100);
            let plan = plan_engaged(&mut g, ax, ay, now, (ax, ay));
            assert_eq!(plan, MovePlan::Stay, "windowed interior move {i} escaped");
        }
    }

    #[test]
    fn tiny_source_interior_moves_do_not_escape() {
        let c = Rect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        let s = Rect {
            x: 900,
            y: 500,
            w: 40,
            h: 24,
        };
        let mut g = engaged_state(c, s, &[]);
        let now = std::time::Instant::now();
        for ax in s.x + 1..s.x + s.w - 1 {
            let plan = plan_engaged(&mut g, ax, s.y + 12, now, (ax, s.y + 12));
            assert_eq!(plan, MovePlan::Stay, "tiny-source move at {ax} escaped");
        }
    }

    #[test]
    fn engage_planner_respects_cooldown_and_margins() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        g.engaged = false;
        let now = std::time::Instant::now();
        g.cooldown_until = Some(now + std::time::Duration::from_millis(500));
        assert!(plan_engage(&g, c.x + c.w / 2, c.y + c.h / 2, now).is_none());
        let later = now + std::time::Duration::from_millis(600);
        assert!(plan_engage(&g, c.x + c.w / 2, c.y + c.h / 2, later).is_some());
        let panel = Rect {
            x: 897,
            y: 4,
            w: 126,
            h: 30,
        };
        g.no_engage = vec![panel_no_engage_rect(panel)];
        assert!(plan_engage(&g, panel.x + 10, panel.y + 10, later).is_none());
    }

    #[test]
    fn overlay_reveal_and_normal_engage_share_exact_ui_ownership_rules() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        g.engaged = false;
        let now = std::time::Instant::now();
        let centre = (c.x + c.w / 2, c.y + c.h / 2);
        assert!(plan_engage(&g, centre.0, centre.1, now).is_some());
        assert!(plan_engage_for_overlay_reveal(&g, centre.0, centre.1, now).is_some());

        let panel = Rect {
            x: centre.0 - 30,
            y: centre.1 - 20,
            w: 60,
            h: 40,
        };
        g.no_engage = vec![panel_no_engage_rect(panel)];
        assert!(plan_engage(&g, centre.0, centre.1, now).is_none());
        assert!(plan_engage_for_overlay_reveal(&g, centre.0, centre.1, now).is_none());
    }

    #[test]
    fn gui_handoff_releases_exactly_at_the_visible_boundary() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        g.engaged = false;
        let gui = Rect {
            x: 100,
            y: 100,
            w: 800,
            h: 700,
        };
        let hit = NoEngageRect::new(gui).with_hwnd(0x1234);
        g.no_engage = vec![hit];
        let now = std::time::Instant::now() + std::time::Duration::from_secs(1);
        assert!(plan_engage(&g, 899, 400, now).is_none());
        assert!(
            plan_engage(&g, 900, 400, now).is_some(),
            "one pixel outside the live GUI must be immediately eligible; no sticky band"
        );
        assert!(plan_engage(&g, 940, 400, now).is_some());
    }

    #[test]
    fn gui_layout_shrink_drops_stale_hover_reengage_boundary() {
        let (c, s) = fs();
        let mut g = engaged_state(c, s, &[]);
        g.engaged = false;
        let full = NoEngageRect::new(Rect {
            x: 15,
            y: 77,
            w: 996,
            h: 1009,
        })
        .with_hwnd(0x1234);
        let mini = NoEngageRect::new(Rect {
            x: 15,
            y: 77,
            w: 776,
            h: 139,
        })
        .with_hwnd(0x1234);
        g.no_engage = vec![mini];
        g.last_ui_hover = Some(UiHover {
            hit: mini,
            pos: (400, 120),
        });
        assert!(
            invalidate_stale_ui_hover_after_no_engage_change(&mut g).is_none(),
            "an exact current GUI hit may remain as diagnostic hover cache"
        );
        assert_eq!(g.last_ui_hover.map(|hover| hover.hit), Some(mini));

        g.last_ui_hover = Some(UiHover {
            hit: full,
            pos: (900, 500),
        });
        let invalidated = invalidate_stale_ui_hover_after_no_engage_change(&mut g)
            .expect("old Full hover must be invalidated when the same GUI shrinks to Mini");
        assert_eq!(invalidated.0, full);
        assert_eq!(invalidated.1, Some(mini));

        let later = std::time::Instant::now() + std::time::Duration::from_secs(1);
        assert!(
            plan_engage(&g, 900, 500, later).is_some(),
            "space vacated by the old Full GUI must immediately become eligible"
        );
        assert!(
            plan_engage(&g, 400, 120, later).is_none(),
            "the current Mini GUI rectangle must remain protected"
        );
    }

    #[test]
    fn escape_then_reengage_full_cycle() {
        // push out the right edge, then move back into the interior (after the
        // cooldown) and re-engage: the round trip must be clean. Windowed only —
        // fullscreen never exits at an edge.
        let (c, s) = win();
        let mut g = engaged_state(c, s, &[]);
        let t0 = std::time::Instant::now();
        let mut escaped_at = None;
        for k in 0..40 {
            let now = t0 + std::time::Duration::from_millis(k * 20);
            if let MovePlan::Escape { .. } = plan_engaged(
                &mut g,
                s.x + s.w + 5,
                s.y + 100,
                now,
                (s.x + s.w - 1, s.y + 100),
            ) {
                escaped_at = Some(now);
                break;
            }
        }
        let escaped_at = escaped_at.expect("did not escape");
        assert!(!g.engaged);
        // during cooldown: no re-engage
        assert!(plan_engage(&g, c.x + c.w / 2, c.y + c.h / 2, escaped_at).is_none());
        // after cooldown: re-engages in the interior
        let after = escaped_at + std::time::Duration::from_millis(3000);
        assert!(plan_engage(&g, c.x + c.w / 2, c.y + c.h / 2, after).is_some());
    }

    #[test]
    fn move_core_reentry_uses_current_event_point() {
        // The move core follows the point supplied by the LL-hook path. v372
        // makes that supplied point the current MSLLHOOKSTRUCT.pt event; a
        // one-event-behind GetCursorPos sample is diagnostic only.
        let (c, s) = win_offset();
        let now = std::time::Instant::now();
        let mut g = State {
            active: true,
            engaged: false,
            window_frame_input: true,
            overlay: c,
            content: c,
            src: s,
            last_hw: (c.x - 1, c.y + c.h / 2),
            last_set: (c.x - 1, c.y + c.h / 2),
            virt: ((c.x - 1) as f64, (c.y + c.h / 2) as f64),
            ..Default::default()
        };

        // If the supplied current event is outside, no engage.
        let stale_raw = (c.x + 200, c.y + 100);
        let actual_outside = (c.x - 1, c.y + c.h / 2);
        let out = handle_move(&mut g, stale_raw.0, stale_raw.1, now, actual_outside);
        assert!(matches!(out, MoveOutcome::Stay { .. }));
        assert!(!g.engaged, "outside current event re-engaged from desktop");

        // As soon as the supplied current event crosses one pixel into content,
        // re-entry remains immediate: no time latch or hidden spatial band.
        let actual_inside = (c.x + 1, c.y + c.h / 2);
        let out = handle_move(
            &mut g,
            stale_raw.0,
            stale_raw.1,
            now + std::time::Duration::from_millis(8),
            actual_inside,
        );
        assert!(matches!(out, MoveOutcome::Engage { .. }));
        assert!(
            g.engaged,
            "one-pixel current-event re-entry did not engage immediately"
        );
    }

    // ---- full-system Sim: models the OS cursor + clip + teleports so we can
    // reproduce real user gestures (panel approach, edge exit, GUI overlap,
    // window drag, button-held drag) and assert no oscillation / fling-off. ---

    struct Sim {
        g: State,
        cursor: (i32, i32), // the OS cursor (== GetCursorPos)
        t: std::time::Instant,
        transitions: u32, // engage<->disengage flips
    }

    impl Sim {
        fn new(content: Rect, src: Rect, panels: &[NoEngageRect], start: (i32, i32)) -> Self {
            let g = State {
                active: true,
                engaged: false,
                overlay: content,
                content,
                src,
                src_hwnd: 0x1234,
                no_engage: panels.to_vec(),
                last_hw: start,
                last_set: start,
                virt: (start.0 as f64, start.1 as f64),
                adjust_speed: false,
                ..Default::default()
            };
            Sim {
                g,
                cursor: start,
                t: std::time::Instant::now(),
                transitions: 0,
            }
        }

        fn set_geometry(&mut self, content: Rect, src: Rect, panels: &[NoEngageRect]) {
            // mirror the engine's configure() geometry-change re-sync while engaged
            if self.g.engaged && (src != self.g.src || content != self.g.content) {
                let (vx, vy) = remap_virtual_between_content(
                    self.g.content,
                    content,
                    self.g.virt.0,
                    self.g.virt.1,
                );
                self.g.virt = (vx, vy);
                let (tx, ty) = map_content_to_source(content, src, vx, vy);
                self.cursor = (tx, ty);
                self.g.last_set = (tx, ty);
                self.g.last_hw = (tx, ty);
            }
            self.g.overlay = content;
            self.g.content = content;
            self.g.src = src;
            self.g.no_engage = panels.to_vec();
        }

        fn apply(&mut self, raw: (i32, i32), actual: (i32, i32)) {
            let was = self.g.engaged;
            match handle_move(&mut self.g, raw.0, raw.1, self.t, actual) {
                MoveOutcome::Stay { .. } => {}
                MoveOutcome::Engage { tx, ty, .. } => self.cursor = (tx, ty),
                MoveOutcome::Disengage { place, .. } => self.cursor = place,
            }
            if was != self.g.engaged {
                self.transitions += 1;
            }
        }

        /// Move the physical hand by (dx,dy). Hook sees the pre-clip raw pt; the
        /// OS clamps the actual cursor to the clip when engaged; teleports move
        /// the OS cursor.
        fn hand(&mut self, dx: i32, dy: i32, dt_ms: u64) {
            self.t += std::time::Duration::from_millis(dt_ms.max(1));
            let raw = (self.cursor.0 + dx, self.cursor.1 + dy);
            let actual = if self.g.engaged {
                let s = self.g.src;
                (
                    raw.0.clamp(s.x, s.x + s.w - 1),
                    raw.1.clamp(s.y, s.y + s.h - 1),
                )
            } else {
                raw
            };
            self.cursor = actual;
            self.apply(raw, actual);
        }

        /// press/release a mouse button (updates the drag flag like the hook).
        fn button(&mut self, bit: u8, down: bool) {
            if down {
                self.g.buttons_down |= bit;
            } else {
                self.g.buttons_down &= !bit;
            }
        }
    }

    fn panels(rects: &[Rect]) -> Vec<NoEngageRect> {
        rects.iter().map(|r| panel_no_engage_rect(*r)).collect()
    }

    #[test]
    fn fullscreen_panel_reachable_without_fling_or_oscillation() {
        // Fullscreen regression case: panel top-centre INSIDE content.
        // the cursor-routing design model: the sprite reaches the panel while the real cursor
        // stays confined; NO disengage, NO teleport => no fling / oscillation.
        let (c, s) = fs();
        let panel = Rect {
            x: 897,
            y: 4,
            w: 126,
            h: 30,
        };
        let mut sim = Sim::new(c, s, &panels(&[panel]), (c.x + c.w / 2, c.y + c.h / 2));
        sim.g.fullscreen = true;
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged, "did not engage from interior");
        let base = sim.transitions;
        // drive the virtual cursor onto the panel centre
        let pcx = panel.x + panel.w / 2;
        let pcy = panel.y + panel.h / 2;
        let (tx, ty) = map_content_to_source(c, s, pcx as f64, pcy as f64);
        for _ in 0..600 {
            let dx = (tx - sim.cursor.0).clamp(-5, 5);
            let dy = (ty - sim.cursor.1).clamp(-5, 5);
            sim.hand(dx, dy, 16);
            if (sim.cursor.0 - tx).abs() <= 1 && (sim.cursor.1 - ty).abs() <= 1 {
                break;
            }
        }
        assert!(
            sim.g.engaged,
            "panel approach flung the cursor out (disengaged)"
        );
        let (vx, vy) = (sim.g.virt.0.round() as i32, sim.g.virt.1.round() as i32);
        assert!(
            panel.contains(vx, vy),
            "virtual cursor not over panel: ({vx},{vy})"
        );
        // jitter all over the panel: still engaged, zero transitions
        for _ in 0..60 {
            sim.hand(1, 0, 16);
            sim.hand(-1, 1, 16);
            sim.hand(0, -1, 16);
            assert!(sim.g.engaged, "panel jitter disengaged (fling)");
        }
        assert_eq!(sim.transitions, base, "panel hover oscillated");
    }

    #[test]
    fn fullscreen_gui_in_left_letterbox_is_reachable() {
        // Fullscreen with side letterbox: the content does not cover the whole
        // monitor. A top-most GUI can sit in that black bar, so pushing past
        // the content edge should extend only the VIRTUAL cursor into the GUI.
        let content = Rect {
            x: 320,
            y: 0,
            w: 1280,
            h: 1080,
        };
        let src = Rect {
            x: 100,
            y: 100,
            w: 640,
            h: 540,
        };
        let gui = Rect {
            x: 24,
            y: 470,
            w: 260,
            h: 140,
        };
        let zones = vec![NoEngageRect::new(gui).with_hwnd(0x1111)];
        let mut sim = Sim::new(
            content,
            src,
            &zones,
            (content.x + content.w / 2, content.y + content.h / 2),
        );
        sim.g.fullscreen = true;
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);

        let target_y = gui.y + gui.h / 2;
        let (edge_tx, edge_ty) =
            map_content_to_source(content, src, content.x as f64 + 1.0, target_y as f64);
        for _ in 0..500 {
            let dx = (edge_tx - sim.cursor.0).signum() * 8;
            let dy = (edge_ty - sim.cursor.1).signum() * 8;
            sim.hand(dx, dy, 16);
            if (sim.cursor.0 - edge_tx).abs() <= 4 && (sim.cursor.1 - edge_ty).abs() <= 4 {
                break;
            }
        }

        for _ in 0..80 {
            sim.hand(-8, 0, 16);
        }
        assert!(sim.g.engaged, "fullscreen GUI reach must not disengage");
        let (vx, vy) = (sim.g.virt.0.round() as i32, sim.g.virt.1.round() as i32);
        assert!(
            gui.contains(vx, vy),
            "virtual cursor did not reach GUI in letterbox: ({vx},{vy}) gui={gui:?}"
        );
        let hit = hit_ui_for_cursor_ownership(&sim.g.no_engage, vx, vy).unwrap();
        assert_eq!(hit.hwnd, 0x1111);

        for _ in 0..20 {
            sim.hand(0, 1, 16);
            sim.hand(0, -1, 16);
            sim.hand(-1, 0, 16);
            let (vx, vy) = (sim.g.virt.0.round() as i32, sim.g.virt.1.round() as i32);
            assert!(
                gui.contains(vx, vy),
                "letterbox GUI cursor flickered back to content edge: ({vx},{vy})"
            );
        }

        let leftmost = sim.g.virt.0;
        for _ in 0..20 {
            sim.hand(-1, 0, 16);
        }
        assert!(
            sim.g.virt.0 <= leftmost + 0.5,
            "leftward push was pulled right: before={} after={}",
            leftmost,
            sim.g.virt.0
        );
    }

    #[test]
    fn fullscreen_gui_overlapping_left_edge_does_not_snap_to_content_edge() {
        // Real GUI windows often straddle the black bar and the magnified
        // content edge. That must still count as a GUI target; otherwise the
        // virtual cursor is clamped back to the content edge and feels pulled
        // right when the user moves left over the GUI.
        let content = Rect {
            x: 320,
            y: 0,
            w: 1280,
            h: 1080,
        };
        let src = Rect {
            x: 100,
            y: 100,
            w: 640,
            h: 540,
        };
        let gui = Rect {
            x: 300,
            y: 420,
            w: 420,
            h: 240,
        };
        let zones = vec![NoEngageRect::new(gui).with_hwnd(0x2222)];
        let mut sim = Sim::new(
            content,
            src,
            &zones,
            (content.x + content.w / 2, content.y + content.h / 2),
        );
        sim.g.fullscreen = true;
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);

        let target_y = gui.y + gui.h / 2;
        let (edge_tx, edge_ty) =
            map_content_to_source(content, src, content.x as f64 + 1.0, target_y as f64);
        for _ in 0..500 {
            let dx = (edge_tx - sim.cursor.0).signum() * 8;
            let dy = (edge_ty - sim.cursor.1).signum() * 8;
            sim.hand(dx, dy, 16);
            if (sim.cursor.0 - edge_tx).abs() <= 4 && (sim.cursor.1 - edge_ty).abs() <= 4 {
                break;
            }
        }

        for _ in 0..16 {
            sim.hand(-1, 0, 16);
        }
        assert!(sim.g.engaged);
        let (vx, vy) = (sim.g.virt.0.round() as i32, sim.g.virt.1.round() as i32);
        assert!(
            vx < content.x && gui.contains(vx, vy),
            "overlapping GUI cursor snapped to content edge: ({vx},{vy}) content={content:?} gui={gui:?}"
        );

        let leftmost = sim.g.virt.0;
        for _ in 0..24 {
            sim.hand(-1, 0, 16);
            assert!(
                sim.g.virt.0 <= content.x as f64,
                "cursor returned to magnified edge while still over GUI: {}",
                sim.g.virt.0
            );
        }
        assert!(
            sim.g.virt.0 <= leftmost + 0.5,
            "leftward GUI motion was pulled right: before={} after={}",
            leftmost,
            sim.g.virt.0
        );
    }

    #[test]
    fn fullscreen_gui_raw_screen_coordinate_is_kept_as_ui_cursor() {
        // Regression case: while the virtual cursor is over the topmost GUI,
        // a move arrived as raw=(480,188). That is a screen/GUI coordinate, not
        // a meaningful source coordinate. Treating it as source-space mapped it
        // to the content edge and made the cursor vanish/jump. If raw itself is
        // inside a GUI no-engage rect, it must stay a GUI cursor.
        let content = Rect {
            x: 341,
            y: 0,
            w: 1238,
            h: 1080,
        };
        let src = Rect {
            x: 987,
            y: 161,
            w: 879,
            h: 767,
        };
        let gui = Rect {
            x: 78,
            y: 78,
            w: 896,
            h: 859,
        };
        let mut g = engaged_state(content, src, &[]);
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        g.virt = (491.9, 188.9);
        g.last_hw = (1440, 390);
        g.last_engage_at = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));

        let plan = plan_engaged(&mut g, 480, 188, std::time::Instant::now(), (src.x, 188));
        assert_eq!(plan, MovePlan::Stay);
        let (vx, vy) = (g.virt.0.round() as i32, g.virt.1.round() as i32);
        assert!(
            gui.contains(vx, vy),
            "raw GUI coordinate was remapped to content/source edge: ({vx},{vy})"
        );
        assert!((vx - 480).abs() <= 1 && (vy - 188).abs() <= 1);
    }

    #[test]
    fn fullscreen_engaged_move_ignores_gui_rect_under_source_interior() {
        // Fullscreen + GUI-topmost regression case: moving the cursor in the
        // magnified view sometimes "pulls" it. Log: engage raw (1018,436), then a
        // ui-hover handoff at (563,572) which is NOT where the sprite was.
        // Cause: the GUI's SCREEN rect overlaps the SOURCE window's screen rect;
        // while engaged, the raw confined cursor is a SOURCE-space coordinate, so
        // it passing under the GUI rect is coincidental — yet it snapped `virt`
        // to the raw position (a ~470px sprite yank) and handed off/disengaged.
        let content = Rect {
            x: 0,
            y: 0,
            w: 1707,
            h: 1067,
        };
        let src = Rect {
            x: 661,
            y: 63,
            w: 1280,
            h: 720,
        };
        // GUI screen rect overlapping the source rect area (x 661..987 shared)
        let gui = Rect {
            x: 91,
            y: 35,
            w: 896,
            h: 859,
        };
        let mut g = engaged_state(content, src, &[]);
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        g.last_engage_at = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));

        // ordinary interior move: raw INSIDE the source, coincidentally under
        // the GUI screen rect; the sprite (mapped) is NOT over the GUI
        let raw = (700, 400);
        let mapped = map_source_to_content_unclamped(content, src, raw.0 as f64, raw.1 as f64);
        assert!(
            !gui.contains(mapped.0.round() as i32, mapped.1.round() as i32),
            "test geometry: mapped position must be outside the GUI"
        );
        g.virt = mapped;
        g.last_hw = (704, 402);

        let plan = plan_engaged(&mut g, raw.0, raw.1, std::time::Instant::now(), raw);
        assert_eq!(plan, MovePlan::Stay);
        assert!(
            (g.virt.0 - mapped.0).abs() < 1.0 && (g.virt.1 - mapped.1).abs() < 1.0,
            "sprite snapped unexpectedly from mapped {:?} to raw-ish {:?}",
            mapped,
            g.virt
        );
        assert!(
            !virtual_over_ui(&g),
            "interior move misread as GUI hover would hand off / disengage"
        );

        // Sweep across the source∩GUI overlap band: virt may legitimately switch
        // to the delta-integrated UI-cursor hold while the MAPPED sprite is over
        // the visible GUI window, but it must stay CONTINUOUS — the bug was a
        // teleport to the raw screen point (hundreds of px in one event).
        let (sx, sy) = content_per_source_px(content, src);
        let step_limit = 8.0 * sx.max(sy).max(1.0) + 2.0; // one 8px-hand event, scaled
        let mut prev = g.virt;
        for x in (700..980).step_by(8) {
            let _ = plan_engaged(&mut g, x, 400, std::time::Instant::now(), (x, 400));
            let jump = ((g.virt.0 - prev.0).powi(2) + (g.virt.1 - prev.1).powi(2)).sqrt();
            assert!(
                jump <= step_limit,
                "sprite teleported at raw=({x},400): prev={prev:?} -> virt={:?} (jump {jump:.1}px > {step_limit:.1})",
                g.virt
            );
            prev = g.virt;
        }
    }

    // Representative 4K fullscreen geometry: content (462,0,2915,2160),
    // source window at (1759,162,2074,1537), near-maximized topmost GUI at
    // (286,233,1786,1711) overlapping the source's screen rect in x 1759..2072.
    fn fs_4k() -> (Rect, Rect, Rect) {
        (
            Rect {
                x: 462,
                y: 0,
                w: 2915,
                h: 2160,
            },
            Rect {
                x: 1759,
                y: 162,
                w: 2074,
                h: 1537,
            },
            Rect {
                x: 286,
                y: 233,
                w: 1786,
                h: 1711,
            },
        )
    }

    #[test]
    fn fullscreen_source_edge_over_gui_screen_rect_gets_no_phantom_snap() {
        // Pushing past the source BOTTOM edge
        // in the strip below the GUI produced pre-clip raw points like
        // (1800,1706) — a SOURCE-space coordinate that coincidentally lies
        // inside the GUI's SCREEN rect — and the raw-over-UI branch snapped
        // the sprite into the GUI and cause an immediate handoff bounce.
        let (content, src, gui) = fs_4k();
        let mut g = engaged_state(content, src, &[]);
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        g.last_engage_at = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        // sprite in the bottom strip BELOW the GUI, hand pushing down
        g.virt = (512.0, 2147.0);
        g.last_hw = (1795, 1690);
        let plan = plan_engaged(
            &mut g,
            1800,
            1706, // past source bottom (1698); inside GUI screen rect
            std::time::Instant::now(),
            (1800, src.y + src.h - 1),
        );
        assert_eq!(plan, MovePlan::Stay);
        assert!(
            !virtual_over_ui(&g),
            "sprite snapped into the GUI from a source-space edge point: virt={:?}",
            g.virt
        );
    }

    #[test]
    fn fullscreen_letterbox_gui_raw_snap_still_works() {
        // The raw-over-UI branch exists for GUIs living OUTSIDE the content
        // (letterbox): there screen == virtual space and the pre-clip raw
        // point over the GUI is unambiguous. That path must keep working.
        let (content, src, _) = fs_4k();
        let gui = Rect {
            x: 3500,
            y: 600,
            w: 340,
            h: 300,
        }; // right letterbox: fully outside content (content right edge 3376)
        let mut g = engaged_state(content, src, &[]);
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        g.last_engage_at = Some(std::time::Instant::now() - std::time::Duration::from_secs(1));
        g.virt = (3300.0, 700.0);
        g.last_hw = (3830, 700);
        let plan = plan_engaged(
            &mut g,
            3833, // at/past the source right edge (3832), outside the content
            700,
            std::time::Instant::now(),
            (src.x + src.w - 1, 700),
        );
        assert_eq!(plan, MovePlan::Stay);
        assert!(
            virtual_over_ui(&g),
            "letterbox GUI must still catch the raw cursor: virt={:?}",
            g.virt
        );
    }

    #[test]
    fn stale_pre_teleport_events_cannot_yank_the_sprite_into_the_gui() {
        // Engage teleports
        // the real cursor to the mapped source point; a queued PRE-teleport
        // hook event (screen coords, e.g. (2316,891) right of the GUI) then
        // arrives and is mapped as a SOURCE position — virt flew to
        // (1245,1024), INSIDE the GUI, and handed off instantly.
        let (content, src, gui) = fs_4k();
        let mut g = engaged_state(content, src, &[]);
        g.engaged = false;
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        let t0 = std::time::Instant::now();
        // engage right of the GUI, over the magnified content
        let out = handle_move(&mut g, 2288, 887, t0, (2288, 887));
        let (tx, ty) = match out {
            MoveOutcome::Engage { tx, ty, .. } => (tx, ty),
            other => panic!("expected engage, got {other:?}"),
        };
        let virt0 = g.virt;
        assert!(!virtual_over_ui(&g), "engage sprite starts outside the GUI");

        // stale pre-teleport screen-coordinate events: their hook pt is far
        // from the OS-clipped cursor (GetCursorPos == teleport target). The
        // hook never sees the injected SetCursorPos move (LLMHF_INJECTED is
        // ignored), so the swallow must key off that divergence — even long
        // after the engage grace; long GPU stalls can make
        // the stale backlog arrive 60-70ms late and yank the sprite).
        for (ms, sx, sy) in [(8, 2316, 891), (40, 2330, 902), (75, 2350, 915)] {
            handle_move(
                &mut g,
                sx,
                sy,
                t0 + std::time::Duration::from_millis(ms),
                (tx, ty),
            );
            assert_eq!(
                g.virt, virt0,
                "stale screen-coord event at +{ms}ms moved the sprite (would bounce into the GUI)"
            );
            assert!(!virtual_over_ui(&g));
        }

        // first FRESH event (pt agrees with the clipped cursor) ends the settle
        handle_move(
            &mut g,
            tx + 3,
            ty + 2,
            t0 + std::time::Duration::from_millis(90),
            (tx, ty),
        );
        assert!(
            g.expect_teleport.is_none(),
            "fresh event must clear the settle"
        );
        assert_eq!(g.virt, virt0);

        // subsequent real moves track normally again
        handle_move(
            &mut g,
            tx + 11,
            ty + 2,
            t0 + std::time::Duration::from_millis(110),
            (tx + 11, ty + 2),
        );
        assert!(
            (g.virt.0 - virt0.0).abs() > 0.5,
            "sprite must track real moves after the settle"
        );
    }

    #[test]
    fn teleport_settle_times_out_instead_of_freezing() {
        // Safety valve: if GetCursorPos somehow never agrees with the hook pt
        // (e.g. another program fighting over the cursor), the settle must
        // expire on its own timeout instead of eating moves forever.
        let (content, src, gui) = fs_4k();
        let mut g = engaged_state(content, src, &[]);
        g.engaged = false;
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        let t0 = std::time::Instant::now();
        let out = handle_move(&mut g, 2288, 887, t0, (2288, 887));
        let (tx, ty) = match out {
            MoveOutcome::Engage { tx, ty, .. } => (tx, ty),
            other => panic!("expected engage, got {other:?}"),
        };
        let virt0 = g.virt;
        // The event that expires the settle is still untrusted. Abort the
        // engage transaction at the current visible position and leave the
        // mapper disengaged; otherwise later moves remain trapped in source
        // coordinates even though the native cursor has already been revealed.
        let late = t0 + std::time::Duration::from_millis(TELEPORT_SETTLE_TIMEOUT_MS + 20);
        let expired = handle_move(&mut g, tx + 200, ty, late, (tx, ty));
        assert!(matches!(expired, MoveOutcome::Disengage { .. }));
        assert!(g.expect_teleport.is_none());
        assert_eq!(g.virt, virt0);
        assert!(!g.engaged, "settle expiry must release source ownership");
    }

    #[test]
    fn gui_screen_event_cannot_false_ack_source_teleport() {
        // After GUI exit, an old GUI-side hook coordinate and
        // GetCursorPos agreed with each other, but neither had reached the
        // requested source coordinate. The old raw==actual check accepted it,
        // interpreted the screen point as source space, and jumped the sprite
        // hundreds of pixels toward the upper-left.
        let (content, src, gui) = fs_4k();
        let mut g = engaged_state(content, src, &[]);
        g.engaged = false;
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        let t0 = std::time::Instant::now();
        let out = handle_move(&mut g, 2288, 887, t0, (2288, 887));
        let target = match out {
            MoveOutcome::Engage { tx, ty, .. } => (tx, ty),
            other => panic!("expected engage, got {other:?}"),
        };
        let virt0 = g.virt;
        let stale_gui_point = (target.0 - 135, target.1 - 125);
        handle_move(
            &mut g,
            stale_gui_point.0,
            stale_gui_point.1,
            t0 + std::time::Duration::from_millis(16),
            stale_gui_point,
        );
        assert_eq!(g.expect_teleport, Some(target));
        assert_eq!(g.virt, virt0, "false acknowledgement moved the sprite");
    }

    #[test]
    fn commit_rebases_teleport_guard_and_blocks_late_screen_space_event() {
        let (content, src, gui) = fs_4k();
        let mut g = engaged_state(content, src, &[]);
        g.engaged = false;
        g.fullscreen = true;
        g.no_engage = vec![NoEngageRect::new(gui).with_hwnd(0x5555)];
        let t0 = std::time::Instant::now();
        let out = handle_move(&mut g, 2288, 887, t0, (2288, 887));
        let target = match out {
            MoveOutcome::Engage { tx, ty, .. } => (tx, ty),
            other => panic!("expected engage, got {other:?}"),
        };
        let virt0 = g.virt;

        // Simulate a saturated render thread: native hide/warp commits well
        // after the old arm-relative 250ms settle would already have expired.
        let committed = t0 + std::time::Duration::from_millis(380);
        mark_engage_committed(&mut g, target, committed);
        assert_eq!(g.last_hw, target);

        // First matching event can clear expect_teleport...
        let fresh = committed + std::time::Duration::from_millis(8);
        handle_move(&mut g, target.0, target.1, fresh, target);
        assert!(g.expect_teleport.is_none());
        assert_eq!(g.virt, virt0);

        // ...but an older screen-space move that drains AFTER it must still be
        // quarantined by the commit-relative firewall so it cannot map far
        // outside the source and clamp the sprite to (0,0).
        let stale = committed + std::time::Duration::from_millis(90);
        handle_move(&mut g, 1048, 625, stale, target);
        assert_eq!(
            g.virt, virt0,
            "late pre-warp event moved the virtual cursor"
        );
        assert_eq!(g.last_hw, target);
    }

    #[test]
    fn idle_autohide_does_not_hide_cursor_over_gui() {
        let now = std::time::Instant::now();
        let content = Rect {
            x: 320,
            y: 0,
            w: 1280,
            h: 1080,
        };
        let gui = Rect {
            x: 300,
            y: 420,
            w: 420,
            h: 240,
        };
        let mut g = State {
            active: true,
            engaged: true,
            content,
            src: Rect {
                x: 100,
                y: 100,
                w: 640,
                h: 540,
            },
            no_engage: vec![NoEngageRect::new(gui).with_hwnd(0x3333)],
            virt: ((gui.x + 40) as f64, (gui.y + 40) as f64),
            last_move: Some(now - std::time::Duration::from_secs(2)),
            autohide_secs: 0.5,
            ..Default::default()
        };
        assert!(virtual_over_ui(&g));
        assert!(!should_hide_cursor_for_idle(&g, now));

        g.virt = (
            (content.x + content.w / 2) as f64,
            (content.y + content.h / 2) as f64,
        );
        assert!(!virtual_over_ui(&g));
        assert!(should_hide_cursor_for_idle(&g, now));
    }

    #[test]
    fn fullscreen_can_engage_exact_bottom_edge_without_prior_source_click() {
        let now = std::time::Instant::now();
        let content = Rect {
            x: 240,
            y: 0,
            w: 1440,
            h: 1080,
        };
        let src = Rect {
            x: 1316,
            y: 117,
            w: 571,
            h: 344,
        };
        let mut g = State {
            active: true,
            fullscreen: true,
            content,
            src,
            src_hwnd: 0x1234,
            ..Default::default()
        };
        let plan = plan_engage(
            &g,
            content.x + content.w / 2,
            content.y + content.h - 1,
            now,
        )
        .expect("fullscreen bottom edge must be immediately engageable");
        assert_eq!(plan.ty, src.y + src.h - 1);

        // The same exact edge remains guarded in ordinary windowed client mode.
        g.fullscreen = false;
        assert!(
            plan_engage(
                &g,
                content.x + content.w / 2,
                content.y + content.h - 1,
                now
            )
            .is_none()
        );
    }

    #[test]
    fn windowed_edge_exit_is_smooth_single_transition() {
        let (c, s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        let start_flips = sim.transitions;
        let mut exited = false;
        for _ in 0..80 {
            sim.hand(30, 0, 16);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited, "never exited the right edge");
        assert_eq!(
            sim.transitions - start_flips,
            1,
            "exit was not a single clean transition"
        );
        assert!(
            sim.cursor.0 >= c.x + c.w - 1,
            "cursor did not leave right: {:?}",
            sim.cursor
        );
        for _ in 0..20 {
            sim.hand(20, 0, 16);
            assert!(!sim.g.engaged, "cursor snapped back into the magnifier");
        }
    }

    #[test]
    fn windowed_edge_exit_places_cursor_outside_overlay_window() {
        let (c, s) = win();
        let overlay = Rect {
            x: c.x - 32,
            y: c.y - 24,
            w: c.w + 64,
            h: c.h + 48,
        };
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.g.overlay = overlay;
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        for _ in 0..120 {
            sim.hand(-20, 0, 16);
            if !sim.g.engaged {
                break;
            }
        }
        assert!(!sim.g.engaged, "never exited");
        assert!(
            sim.cursor.0 <= overlay.x - EXIT_WINDOW_MARGIN_PX,
            "cursor landed on/in the overlay border: cursor={:?} overlay={:?}",
            sim.cursor,
            overlay
        );
    }

    #[test]
    fn post_top_exit_shallow_return_does_not_reengage() {
        let (c, s) = win_offset();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        for _ in 0..200 {
            sim.hand(0, -8, 16);
            if !sim.g.engaged {
                break;
            }
        }
        assert!(!sim.g.engaged, "never exited top");
        sim.t += std::time::Duration::from_millis(350);

        let shallow_y = c.y + ENTER_MARGIN_PX + 1;
        for _ in 0..80 {
            if sim.cursor.1 >= shallow_y {
                break;
            }
            let dy = (shallow_y - sim.cursor.1).clamp(1, 4);
            sim.hand(0, dy, 16);
            assert!(
                !sim.g.engaged,
                "shallow top-edge return re-engaged and pulled back"
            );
        }
    }

    #[test]
    fn windowed_gentle_push_exits_without_pullback() {
        // A gentle push at
        // the edge must leave within a couple of events (no pinning wall) and,
        // once out, continued outward motion must never snap back inside.
        let (c, s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        // ease over to just INSIDE the right edge (stop before pushing past)
        for _ in 0..600 {
            if sim.cursor.0 >= s.x + s.w - 8 {
                break;
            }
            sim.hand(4, 0, 16);
            assert!(sim.g.engaged, "approach exited prematurely");
        }
        let base = sim.transitions;
        // gentle-but-deliberate outward pushes: exits within a handful of small
        // events (no time gate; it takes a few px of real outward travel).
        let mut exited_after = None;
        for k in 0..14 {
            sim.hand(3, 0, 16);
            if !sim.g.engaged {
                exited_after = Some(k);
                break;
            }
        }
        let k = exited_after.expect("gentle push never exited");
        assert!(k <= 10, "gentle exit took too many events: {k}");
        assert_eq!(
            sim.transitions - base,
            1,
            "exit was not a single clean transition"
        );
        assert!(
            sim.cursor.0 >= c.x + c.w - 1,
            "did not land outside: {:?}",
            sim.cursor
        );
        // continued gentle outward motion must NOT pull back in
        for _ in 0..25 {
            sim.hand(3, 0, 16);
            assert!(!sim.g.engaged, "cursor got pulled back into the view");
        }
    }

    #[test]
    fn windowed_exit_then_edge_wobble_does_not_reengage() {
        // After exiting an edge, a small
        // in/out wobble near that edge must NOT re-engage (which would teleport
        // the cursor back into the offset source region).
        let (c, s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        let mut exited = false;
        for _ in 0..200 {
            sim.hand(20, 0, 16);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited, "never exited the right edge");
        let base = sim.transitions;
        // wobble right at the edge (a few px in and out): never re-engage
        for k in 0..50 {
            sim.hand(-5, 0, 16); // drift back toward the view
            sim.hand(5, 0, 16); // and back out
            assert!(
                !sim.g.engaged,
                "edge wobble re-engaged (pull-back) at k={k}"
            );
        }
        assert_eq!(sim.transitions, base, "edge wobble oscillated");
        // a DELIBERATE deep return re-engages normally
        let mut reengaged = false;
        for _ in 0..200 {
            sim.hand(-30, 0, 16);
            if sim.g.engaged {
                reengaged = true;
                break;
            }
        }
        assert!(reengaged, "could not re-engage on a deliberate deep return");
    }

    #[test]
    fn windowed_offset_left_edge_does_not_oscillate() {
        // Offset-window regression case: cursor remains correct near
        // the content-left edge. Without the source-space dead-band this
        // engages/escapes repeatedly (each re-engage teleporting the cursor back
        // into the source. With it, the edge is stable.
        let (c, s) = win_offset();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        // creep toward the source-left edge and out
        let mut exited = false;
        for _ in 0..400 {
            sim.hand(-6, 0, 16);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited, "never exited the left edge");
        let base = sim.transitions;
        // Drift just to the edge band. It must not re-engage yet.
        for _ in 0..200 {
            let (tx, _) = map_content_to_source(c, s, sim.cursor.0 as f64, sim.cursor.1 as f64);
            if tx >= s.x + 3 {
                break;
            }
            sim.hand(3, 0, 16);
            assert!(
                !sim.g.engaged,
                "re-engaged while drifting back near the edge"
            );
        }
        // Linger just outside the edge with tiny wobbles: still no re-engage.
        for k in 0..80 {
            sim.hand(2, 0, 16);
            sim.hand(-2, 0, 16);
            assert!(!sim.g.engaged, "post-exit wobble re-engaged at k={k}");
        }
        assert_eq!(sim.transitions, base, "post-exit edge zone oscillated");
        // A deliberate push inward DOES re-engage (normal use preserved).
        let mut reengaged = false;
        for _ in 0..200 {
            sim.hand(12, 0, 16);
            if sim.g.engaged {
                reengaged = true;
                break;
            }
        }
        assert!(reengaged, "could not re-engage on a deliberate deep return");
    }

    #[test]
    fn post_exit_cooldown_blocks_fast_return_then_reenters_normally() {
        // Regression case: after an exit, the cursor must not dart deep-right (past the
        // deliberate threshold) and back to a shallow spot WITHIN the re-enter
        // cooldown. The old guard cleared `must_leave` on the deep overshoot even
        // though the cooldown blocked the engage, so the shallow return then
        // re-engaged (source ~50px inside). It must stay disengaged now.
        let (c, s) = win_offset();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        let mut exited = false;
        for _ in 0..400 {
            sim.hand(-6, 0, 8);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited, "never exited");
        let base = sim.transitions;
        // dart deep-right (past POST_ESCAPE), fast (small dt = within cooldown)
        let (deep_cx, _) = map_source_to_content(
            c,
            s,
            (s.x + POST_ESCAPE_SRC_MARGIN_PX + 20) as f64,
            (s.y + s.h / 2) as f64,
        );
        let deep_cx = deep_cx.round() as i32;
        for _ in 0..300 {
            if sim.cursor.0 >= deep_cx {
                break;
            }
            sim.hand(8, 0, 3);
            assert!(
                !sim.g.engaged,
                "engaged during the deep dart (cooldown should block)"
            );
        }
        // back to a shallow spot (~source +48), still fast/within cooldown
        let (shallow_cx, _) =
            map_source_to_content(c, s, (s.x + 48) as f64, (s.y + s.h / 2) as f64);
        let shallow_cx = shallow_cx.round() as i32;
        for _ in 0..300 {
            if sim.cursor.0 <= shallow_cx {
                break;
            }
            sim.hand(-8, 0, 3);
            assert!(
                !sim.g.engaged,
                "engaged on the way back (cooldown should block)"
            );
        }
        // Let the cooldown expire; a normal deliberate deep return should
        // re-engage once.
        sim.t += std::time::Duration::from_millis(300);
        let (deep_return_cx, _) = map_source_to_content(
            c,
            s,
            (s.x + POST_ESCAPE_SRC_MARGIN_PX + 20) as f64,
            (s.y + s.h / 2) as f64,
        );
        let deep_return_cx = deep_return_cx.round() as i32;
        for _ in 0..200 {
            if sim.cursor.0 < deep_return_cx {
                sim.hand(8, 0, 16);
            } else {
                sim.hand(1, 0, 16);
            }
            if sim.g.engaged {
                break;
            }
        }
        assert!(sim.g.engaged, "could not re-enter after cooldown");
        assert_eq!(sim.transitions - base, 1, "cooldown return oscillated");
    }

    #[test]
    fn all_four_edges_exit_smoothly() {
        for (dx, dy) in [(30, 0), (-30, 0), (0, 30), (0, -30)] {
            let (c, s) = win();
            let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
            sim.hand(0, 0, 16);
            let base = sim.transitions;
            let mut exited = false;
            for _ in 0..80 {
                sim.hand(dx, dy, 16);
                if !sim.g.engaged {
                    exited = true;
                    break;
                }
            }
            assert!(exited, "edge ({dx},{dy}) never exited");
            assert_eq!(
                sim.transitions - base,
                1,
                "edge ({dx},{dy}) not a single transition"
            );
        }
    }

    #[test]
    fn gui_entry_hands_off_and_exit_reengages_promptly() {
        // The GUI is our own window; sprite entry hands the
        // real system cursor over (impure layer); the PURE contracts are:
        // (1) virtual_over_ui fires when the mapped sprite enters the GUI
        //     (this is the trigger the hook uses for the handoff), and
        // (2) after the handoff state, plan_engage refuses re-engagement while
        //     the cursor stays inside the GUI, then re-engages promptly
        //     once it leaves toward the content.
        let content = Rect {
            x: 218,
            y: 0,
            w: 1484,
            h: 1080,
        };
        let src = Rect {
            x: 812,
            y: 161,
            w: 1054,
            h: 767,
        };
        let gui = Rect {
            x: 44,
            y: 169,
            w: 896,
            h: 859,
        };
        let zones = vec![NoEngageRect::new(gui).with_hwnd(0x1b0758)];
        let mut sim = Sim::new(content, src, &zones, (1600, 540));
        sim.g.fullscreen = true;
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);

        // (1) sweep left until the mapped sprite is over the GUI zone
        let mut entered = false;
        for _ in 0..40 {
            sim.hand(-25, 0, 16);
            if virtual_over_ui(&sim.g) {
                entered = true;
                break;
            }
        }
        assert!(entered, "sprite never entered the GUI zone");

        // simulate the handoff exactly as handoff_to_gui sets the state
        let inside = (gui.x + gui.w / 2, gui.y + gui.h / 2);
        release_locked(&mut sim.g);
        sim.g.virt = (inside.0 as f64, inside.1 as f64);
        sim.g.last_set = inside;
        sim.g.last_hw = inside;
        sim.cursor = inside;

        // (2a) roaming INSIDE the GUI never re-engages (no_engage zone)
        for _ in 0..30 {
            sim.hand(7, 3, 16);
            assert!(!sim.g.engaged, "re-engaged while inside the GUI");
            sim.hand(-7, -3, 16);
        }

        // (2b) leaving toward the magnified content re-engages promptly
        let mut steps_to_reengage = 0;
        for _ in 0..80 {
            sim.hand(30, 0, 16);
            steps_to_reengage += 1;
            if sim.g.engaged {
                break;
            }
        }
        assert!(sim.g.engaged, "never re-engaged after leaving the GUI");
        assert!(
            steps_to_reengage <= 40,
            "re-engagement too sluggish: {steps_to_reengage} steps"
        );
    }

    #[test]
    fn gui_zone_stays_engaged_no_churn() {
        // windowed GUI inside the content: the sprite passes over it while the
        // real cursor stays confined. Clicks are redirected in the hook; the
        // move planner must simply STAY engaged with zero oscillation.
        let (c, s) = win();
        let gui = Rect {
            x: c.x + 40,
            y: c.y + 40,
            w: 300,
            h: 200,
        };
        let mut sim = Sim::new(c, s, &panels(&[gui]), (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        let base = sim.transitions;
        let gcx = gui.x + gui.w / 2;
        let gcy = gui.y + gui.h / 2;
        let (tx, ty) = map_content_to_source(c, s, gcx as f64, gcy as f64);
        for _ in 0..600 {
            let dx = (tx - sim.cursor.0).signum() * 4;
            let dy = (ty - sim.cursor.1).signum() * 4;
            sim.hand(dx, dy, 16);
            if (sim.cursor.0 - tx).abs() <= 2 && (sim.cursor.1 - ty).abs() <= 2 {
                break;
            }
        }
        assert!(sim.g.engaged, "GUI approach disengaged");
        let (vx, vy) = (sim.g.virt.0.round() as i32, sim.g.virt.1.round() as i32);
        assert!(
            gui.contains(vx, vy),
            "virtual cursor not over GUI: ({vx},{vy})"
        );
        for _ in 0..40 {
            sim.hand(2, 2, 16);
            sim.hand(-2, -2, 16);
            assert!(sim.g.engaged, "GUI hover disengaged");
        }
        assert_eq!(sim.transitions, base, "GUI hover oscillated");
    }

    #[test]
    fn authoritative_top_hwnd_wins_overlap_independent_of_zone_order() {
        let gui = NoEngageRect::new(Rect {
            x: 100,
            y: 100,
            w: 320,
            h: 200,
        })
        .with_hwnd(0x1111);
        let panel = panel_no_engage_rect(Rect {
            x: 160,
            y: 120,
            w: 180,
            h: 40,
        })
        .with_hwnd(0x2222);

        let zones = vec![gui, panel];
        assert_eq!(tracked_ui_for_top_hwnd(&zones, 0x2222), Some(panel));
        assert_eq!(tracked_ui_for_top_hwnd(&zones, 0x1111), Some(gui));

        let reversed = vec![panel, gui];
        assert_eq!(tracked_ui_for_top_hwnd(&reversed, 0x2222), Some(panel));
        assert_eq!(tracked_ui_for_top_hwnd(&reversed, 0x1111), Some(gui));
        assert_eq!(tracked_ui_for_top_hwnd(&reversed, 0x3333), None);
    }

    #[test]
    fn ui_click_has_no_margin_or_previous_hover_grace() {
        let now = std::time::Instant::now();
        let gui = Rect {
            x: 100,
            y: 100,
            w: 320,
            h: 200,
        };
        let mut g = State {
            active: true,
            engaged: true,
            content: Rect {
                x: 0,
                y: 0,
                w: 1920,
                h: 1080,
            },
            src: Rect {
                x: 600,
                y: 200,
                w: 640,
                h: 360,
            },
            no_engage: vec![NoEngageRect::new(gui).with_hwnd(0x4444)],
            virt: ((gui.x + 40) as f64, (gui.y + 40) as f64),
            ..Default::default()
        };

        let hover = note_ui_hover(&mut g, now).expect("hover over GUI");
        assert_eq!(hover.hit.hwnd, 0x4444);

        // The very next visible point outside the window is source/other-window
        // territory. No invisible click margin and no previous-hover latch.
        g.virt = ((gui.x - 1) as f64, (gui.y + 40) as f64);
        assert!(ui_click_target_from_virtual(&mut g, now).is_none());

        g.virt = ((gui.x + gui.w) as f64, (gui.y + 40) as f64);
        assert!(ui_click_target_from_virtual(&mut g, now).is_none());
    }

    #[test]
    fn engaged_click_ignores_hidden_real_source_coordinate() {
        let now = std::time::Instant::now();
        let gui = Rect {
            x: 37,
            y: 33,
            w: 896,
            h: 859,
        };
        let mut g = State {
            active: true,
            engaged: true,
            fullscreen: true,
            content: Rect {
                x: 341,
                y: 0,
                w: 1238,
                h: 1080,
            },
            src: Rect {
                x: 1228,
                y: 160,
                w: 1468,
                h: 825,
            },
            no_engage: vec![NoEngageRect::new(gui).with_hwnd(0x7777)],
            // Visible cursor is NOT over GUI; raw event coordinates below are.
            virt: (1500.0, 900.0),
            ..Default::default()
        };

        assert!(
            ui_click_target_for_event(&mut g, 192, 149, now).is_none(),
            "hidden source-space raw coordinates must never steal visible UI ownership"
        );
    }

    #[test]
    fn native_gui_hold_blocks_engage_without_blocking_content_button_hold() {
        let now = std::time::Instant::now();
        let content = Rect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        let src = Rect {
            x: 640,
            y: 260,
            w: 640,
            h: 360,
        };
        let mut g = State {
            active: true,
            engaged: false,
            content,
            src,
            buttons_down: BTN_LEFT,
            native_ui_hold_bits: BTN_LEFT,
            ..Default::default()
        };

        assert_eq!(
            plan_engage(&g, 900, 500, now),
            None,
            "a button gesture that started on Neo's native GUI must never arm source ownership"
        );

        // Keep the physical button held but mark it as content-originated. This
        // preserves direct source click/drag behavior.
        g.native_ui_hold_bits = 0;
        assert!(
            plan_engage(&g, 900, 500, now + std::time::Duration::from_millis(1)).is_some(),
            "a content-originated held button must not be globally blocked"
        );
    }

    #[test]
    fn ui_hold_blocks_only_while_the_physical_gesture_is_active() {
        let now = std::time::Instant::now();
        let content = Rect {
            x: 0,
            y: 0,
            w: 1920,
            h: 1080,
        };
        let src = Rect {
            x: 640,
            y: 260,
            w: 640,
            h: 360,
        };
        let mut g = State {
            active: true,
            engaged: false,
            content,
            src,
            ui_hold_bits: BTN_LEFT,
            ..Default::default()
        };

        assert_eq!(plan_engage(&g, 900, 500, now), None);
        g.ui_hold_bits = 0;
        assert!(
            plan_engage(&g, 900, 500, now + std::time::Duration::from_millis(1)).is_some(),
            "button-up must remove UI ownership immediately; no time-based sticky grace"
        );
    }

    #[test]
    fn stale_ui_button_is_released_by_watchdog_without_stopping_capture() {
        let now = std::time::Instant::now();
        let mut g = State {
            active: true,
            buttons_down: BTN_LEFT,
            ui_hold_bits: BTN_LEFT,
            swallow_up: BTN_LEFT,
            last_button_event: Some(
                now - std::time::Duration::from_millis(STALE_BUTTON_WATCHDOG_MS + 1),
            ),
            ..Default::default()
        };
        let stale = reconcile_stale_buttons_with_physical(&mut g, now, 0);
        assert_eq!(stale, BTN_LEFT);
        assert_eq!(g.buttons_down, 0);
        assert_eq!(g.ui_hold_bits, 0);
        assert_eq!(g.swallow_up, 0);
    }

    #[test]
    fn direct_source_hold_is_not_released_by_async_key_watchdog() {
        let now = std::time::Instant::now();
        let mut g = State {
            active: true,
            buttons_down: BTN_LEFT,
            source_direct_bits: BTN_LEFT,
            last_button_event: Some(
                now - std::time::Duration::from_millis(STALE_BUTTON_WATCHDOG_MS + 50),
            ),
            ..Default::default()
        };
        // A swallowed LL-hook DOWN may be invisible to GetAsyncKeyState. The
        // matching LL-hook UP, not this watchdog, owns the direct gesture.
        let stale = reconcile_stale_buttons_with_physical(&mut g, now, 0);
        assert_eq!(stale, 0);
        assert_eq!(g.buttons_down, BTN_LEFT);
        assert_eq!(g.source_direct_bits, BTN_LEFT);
    }

    #[test]
    fn panel_over_gui_overlap_stays_engaged_no_churn() {
        // panel and GUI both no-engage AND overlapping: sweeping the overlap and
        // either side must never disengage / oscillate (stays confined; clicks
        // are redirected in the hook).
        let (c, s) = win();
        let gui = Rect {
            x: c.x + 60,
            y: c.y + 60,
            w: 400,
            h: 260,
        };
        let panel = Rect {
            x: c.x + 200,
            y: c.y + 40,
            w: 220,
            h: 40,
        }; // overlaps gui top
        let mut sim = Sim::new(c, s, &panels(&[gui, panel]), (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        let base = sim.transitions;
        // drive into the overlap region (panel∩gui)
        let ox = panel.x + panel.w / 2;
        let oy = gui.y + 10;
        let (tx, ty) = map_content_to_source(c, s, ox as f64, oy as f64);
        for _ in 0..600 {
            let dx = (tx - sim.cursor.0).signum() * 4;
            let dy = (ty - sim.cursor.1).signum() * 4;
            sim.hand(dx, dy, 16);
            if (sim.cursor.0 - tx).abs() <= 2 && (sim.cursor.1 - ty).abs() <= 2 {
                break;
            }
        }
        assert!(sim.g.engaged, "overlap approach disengaged");
        // sweep across the whole overlap + both edges; never disengage
        for k in 0..80 {
            let d = if k % 2 == 0 { 3 } else { -3 };
            sim.hand(d, d, 16);
            assert!(sim.g.engaged, "overlap sweep disengaged at k={k}");
        }
        assert_eq!(sim.transitions, base, "overlap hover oscillated");
    }

    #[test]
    fn window_move_keeps_engagement_and_tracks() {
        // windowed: the source window is dragged (its own title bar), so the
        // engine feeds new geometry every tick. Engagement must survive and the
        // cursor keep tracking — no fling, no stray disengage.
        let (mut c, mut s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        for step in 0..60 {
            // move both source and content by the same delta (window drag)
            let d = if step < 30 { 6 } else { -6 };
            s = Rect {
                x: s.x + d,
                y: s.y + d / 2,
                ..s
            };
            c = Rect {
                x: c.x + d,
                y: c.y + d / 2,
                ..c
            };
            sim.set_geometry(c, s, &[]);
            // a small in-source hand move each tick
            sim.hand(2, 1, 16);
            assert!(sim.g.engaged, "window move disengaged at step {step}");
            // cursor stays confined to the (moved) source
            assert!(
                sim.cursor.0 >= s.x && sim.cursor.0 < s.x + s.w,
                "cursor left source during move: {:?} src={s:?}",
                sim.cursor
            );
        }
    }

    #[test]
    fn button_held_drag_survives_edge_and_geometry_move() {
        // hold left button and drag hard against an edge WHILE the window moves:
        // a source drag must never disengage.
        let (mut c, mut s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        sim.button(BTN_LEFT, true);
        for step in 0..80 {
            sim.hand(28, 4, 16); // shove toward+past the right edge
            if step % 5 == 0 {
                s = Rect { x: s.x + 3, ..s };
                c = Rect { x: c.x + 3, ..c };
                sim.set_geometry(c, s, &[]);
            }
            assert!(sim.g.engaged, "held-button drag disengaged at step {step}");
        }
        sim.button(BTN_LEFT, false);
        // after releasing at the edge, a further push should now be allowed to
        // exit (no longer a drag).
        let mut exited = false;
        for _ in 0..80 {
            sim.hand(28, 0, 16);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited, "could not exit after releasing the drag");
    }

    #[test]
    fn rapid_edge_tapping_does_not_oscillate() {
        // repeatedly tap the edge (in-out-in) fast: must not thrash engagement.
        let (c, s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        let base = sim.transitions;
        for _ in 0..60 {
            sim.hand(40, 0, 8); // toward edge
            sim.hand(-40, 0, 8); // back in
        }
        // at most a handful of transitions across 60 taps (each brief tap must
        // NOT trigger the sustained-push exit)
        assert!(
            sim.transitions - base <= 2,
            "rapid tapping oscillated: {} transitions",
            sim.transitions - base
        );
        assert!(sim.g.engaged, "ended disengaged after returning inside");
    }

    #[test]
    fn full_workflow_engage_wiggle_exit_reengage() {
        let (c, s) = win();
        let mut sim = Sim::new(c, s, &[], (c.x + c.w / 2, c.y + c.h / 2));
        sim.hand(0, 0, 16);
        assert!(sim.g.engaged);
        for _ in 0..50 {
            sim.hand(5, 3, 16);
            sim.hand(-4, -2, 16);
            assert!(sim.g.engaged, "interior wiggle disengaged");
        }
        let mut exited = false;
        for _ in 0..80 {
            sim.hand(0, 30, 16);
            if !sim.g.engaged {
                exited = true;
                break;
            }
        }
        assert!(exited);
        assert!(sim.cursor.1 >= c.y + c.h - 1);
        sim.t += std::time::Duration::from_millis(1200);
        let mut reengaged = false;
        for _ in 0..200 {
            sim.hand(0, -20, 16);
            if sim.g.engaged {
                reengaged = true;
                break;
            }
        }
        assert!(reengaged, "could not re-engage after coming back");
    }

    #[test]
    fn short_engage_disengage_extends_reenter_cooldown() {
        let now = std::time::Instant::now();
        let mut st = State {
            last_engage_at: Some(now - std::time::Duration::from_millis(OSC_SHORT_MS / 2)),
            ..Default::default()
        };
        mark_disengage(&mut st, now);
        assert_eq!(st.oscillations, 1);
        let cooldown = reenter_cooldown(now, st.oscillations);
        assert!(
            cooldown.duration_since(now)
                >= std::time::Duration::from_millis(REENTER_COOLDOWN_MS * 2)
        );

        st.last_engage_at = Some(now - std::time::Duration::from_millis(OSC_SHORT_MS + 50));
        mark_disengage(&mut st, now);
        assert_eq!(st.oscillations, 0);
    }
    #[test]
    fn exact_native_gui_ownership_has_no_hover_deadband() {
        let gui = NoEngageRect::new(Rect {
            x: 100,
            y: 100,
            w: 300,
            h: 200,
        })
        .with_hwnd(0x5555);
        let zones = vec![gui];
        assert!(hit_ui_for_cursor_ownership(&zones, 100, 100).is_some());
        assert!(hit_ui_for_cursor_ownership(&zones, 399, 299).is_some());
        assert!(hit_ui_for_cursor_ownership(&zones, 99, 150).is_none());
        assert!(hit_ui_for_cursor_ownership(&zones, 400, 150).is_none());
    }

    #[test]
    fn panel_ownership_is_exact_visible_rect_with_no_halo() {
        let panel = panel_no_engage_rect(Rect {
            x: 500,
            y: 10,
            w: 300,
            h: 33,
        })
        .with_hwnd(0x2222);
        let zones = vec![panel];
        assert_eq!(panel.rect, panel.land);
        assert_eq!(PANEL_NO_ENGAGE_HALO_PX, 0);
        assert!(hit_ui_for_cursor_ownership(&zones, 500, 10).is_some());
        assert!(hit_ui_for_cursor_ownership(&zones, 799, 42).is_some());
        assert!(hit_ui_for_cursor_ownership(&zones, 499, 20).is_none());
        assert!(hit_ui_for_cursor_ownership(&zones, 800, 20).is_none());
        assert!(hit_ui_for_cursor_ownership(&zones, 600, 9).is_none());
        assert!(hit_ui_for_cursor_ownership(&zones, 600, 43).is_none());
    }

    #[test]
    fn repeated_panel_crossings_never_create_a_spatial_latch() {
        let panel = panel_no_engage_rect(Rect {
            x: 500,
            y: 100,
            w: 300,
            h: 60,
        })
        .with_hwnd(0x2222);
        let zones = vec![panel];

        // 4,000 left<->right traversals. Ownership must be a pure function of
        // the current point; no previous owner or crossing speed may alter it.
        for pass in 0..4000 {
            if pass % 2 == 0 {
                for x in 450..850 {
                    assert_eq!(
                        hit_ui_for_cursor_ownership(&zones, x, 130).is_some(),
                        (500..800).contains(&x),
                        "left->right pass={pass} x={x}"
                    );
                }
            } else {
                for x in (450..850).rev() {
                    assert_eq!(
                        hit_ui_for_cursor_ownership(&zones, x, 130).is_some(),
                        (500..800).contains(&x),
                        "right->left pass={pass} x={x}"
                    );
                }
            }
        }
    }

    #[test]
    fn repeated_gui_crossings_never_create_a_spatial_latch() {
        let gui = NoEngageRect::new(Rect {
            x: 200,
            y: 200,
            w: 600,
            h: 500,
        })
        .with_hwnd(0x3333);
        let zones = vec![gui];
        for pass in 0..3000 {
            let y = 150 + (pass % 600) as i32;
            for x in (100..900).step_by(7) {
                let inside = (200..800).contains(&x) && (200..700).contains(&y);
                assert_eq!(
                    hit_ui_for_cursor_ownership(&zones, x, y).is_some(),
                    inside,
                    "pass={pass} point=({x},{y})"
                );
            }
        }
    }

    #[test]
    fn authoritative_overlap_selection_survives_repeated_z_order_changes() {
        let gui = NoEngageRect::new(Rect {
            x: 100,
            y: 100,
            w: 700,
            h: 500,
        })
        .with_hwnd(0x1111);
        let panel = panel_no_engage_rect(Rect {
            x: 300,
            y: 150,
            w: 300,
            h: 80,
        })
        .with_hwnd(0x2222);
        let zones = vec![gui, panel];
        for i in 0..100_000 {
            let top = if i & 1 == 0 { 0x1111 } else { 0x2222 };
            let expected = if top == 0x1111 { gui } else { panel };
            assert_eq!(tracked_ui_for_top_hwnd(&zones, top), Some(expected));
        }
    }

    #[test]
    fn v579_fullscreen_button_hook_reentry_is_not_present() {
        let source = include_str!("input.rs");
        let production_input_route = source
            .split("fn diagnose_click")
            .nth(1)
            .expect("input diagnostic boundary")
            .split("fn set_system_cursor_visible")
            .next()
            .expect("cursor visibility boundary");
        assert!(!production_input_route.contains("try_fullscreen_first_button_engage"));
        assert!(!production_input_route.contains("fullscreen first-button engage-forward"));
    }

    #[test]
    fn ordinary_release_does_not_arm_janitor_or_manual_failsafe() {
        let source = include_str!("input.rs");
        let start = source
            .find("pub fn emergency_release_all()")
            .expect("release");
        let tail = &source[start..];
        let end = tail.find("struct PendingEngage").expect("boundary");
        let body = &tail[..end];
        assert!(!body.contains("request_emergency_input_release"));
        assert!(!body.contains("spawn_cursor_janitor"));
        assert!(!body.contains("run_cursor_rescue_once"));
    }

    #[test]
    fn automatic_failsafe_worker_is_dormant_during_normal_operation() {
        let source = include_str!("input.rs");
        let start = source
            .find("fn start_input_failsafe_worker")
            .expect("failsafe worker");
        let tail = &source[start..];
        let end = tail
            .find("pub fn capture_session_active")
            .expect("capture session boundary");
        let body = &tail[..end];
        assert!(!body.contains("std::thread::Builder"));
        assert!(!body.contains("MagInitialize"));
        assert!(!body.contains("request_emergency_input_release"));
    }

    #[test]
    fn janitor_waits_for_parent_or_quit_event_and_preserves_normal_stop_isolation() {
        let source = include_str!("input.rs");
        let start = source.find("pub fn run_cursor_janitor").expect("janitor");
        let tail = &source[start..];
        let end = tail
            .find("pub fn spawn_cursor_janitor")
            .expect("janitor spawn");
        let body = &tail[..end];
        assert_eq!(body.matches("OpenProcess(").count(), 1);
        assert!(body.contains("MsgWaitForMultipleObjectsEx"));
        assert!(body.contains("INPUT_JANITOR_QUIT_GRACE_MS"));
        assert!(body.contains("janitor_restore_source(parent_pid)"));
        assert!(!body.contains("TerminateProcess"));
        assert!(!body.contains("request_emergency_input_release"));
    }

    #[test]
    fn moving_gui_geometry_cannot_break_active_native_drag_ownership() {
        let now = std::time::Instant::now();
        let mut g = State {
            active: true,
            engaged: false,
            content: Rect {
                x: 0,
                y: 0,
                w: 1920,
                h: 1080,
            },
            src: Rect {
                x: 640,
                y: 260,
                w: 640,
                h: 360,
            },
            native_ui_hold_bits: BTN_LEFT,
            native_gui_owner_hwnd: 0x1111,
            ..Default::default()
        };

        // Model a title-bar drag with thousands of geometry publications. The
        // physical gesture latch, not a moving rectangle, prevents source warp.
        for i in 0..20_000 {
            g.no_engage = vec![
                NoEngageRect::new(Rect {
                    x: (i % 1200) as i32 - 200,
                    y: (i % 700) as i32 - 100,
                    w: 900,
                    h: 700,
                })
                .with_hwnd(0x1111),
            ];
            assert_eq!(
                plan_engage(
                    &g,
                    900,
                    500,
                    now + std::time::Duration::from_micros(i as u64)
                ),
                None,
                "native drag must retain ownership at iteration {i}"
            );
        }
    }
}
