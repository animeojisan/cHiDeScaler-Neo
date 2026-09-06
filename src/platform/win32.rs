//! Win32 ctypes-style leaf helpers (window queries, monitor geometry).

use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicBool, AtomicIsize, AtomicU8, AtomicU32, Ordering},
};
use windows::Win32::Foundation::{
    COLORREF, CloseHandle, HWND, LPARAM, LRESULT, POINT, RECT, WPARAM,
};
use windows::Win32::Graphics::Dwm::{
    DWM_WINDOW_CORNER_PREFERENCE, DWMWA_CAPTION_COLOR, DWMWA_EXTENDED_FRAME_BOUNDS,
    DWMWA_TEXT_COLOR, DWMWA_USE_IMMERSIVE_DARK_MODE, DWMWA_WINDOW_CORNER_PREFERENCE,
    DWMWCP_DONOTROUND, DwmFlush, DwmGetWindowAttribute, DwmSetWindowAttribute,
};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, ClientToScreen, CreateFontIndirectW, CreateSolidBrush, DEFAULT_GUI_FONT, DEVMODEW,
    DT_CALCRECT, DT_CENTER, DT_NOPREFIX, DT_SINGLELINE, DT_VCENTER, DeleteObject, DrawTextW,
    ENUM_CURRENT_SETTINGS, EndPaint, EnumDisplaySettingsW, FillRect, GetDC, GetMonitorInfoW,
    GetPixel, GetStockObject, LOGFONTW, MONITOR_DEFAULTTONEAREST, MONITORINFO, MONITORINFOEXW,
    MonitorFromWindow, PAINTSTRUCT, RDW_ALLCHILDREN, RDW_FRAME, RDW_INVALIDATE, RDW_UPDATENOW,
    RedrawWindow, ReleaseDC, SelectObject, SetBkMode, SetTextColor, TRANSPARENT,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON,
};
use windows::Win32::UI::WindowsAndMessaging::*;

#[derive(Clone, Debug, Default, PartialEq)]
struct PanelGdiMirrorSnapshot {
    parent: isize,
    width: i32,
    height: i32,
    visible: bool,
    chip: bool,
    stop_text: String,
    fps_tenths: i32,
    hover_slot: u8,
    screenshot_feedback: bool,
}

#[derive(Debug, Default)]
struct PanelGdiMirrorState {
    hwnd: isize,
    snapshot: PanelGdiMirrorSnapshot,
}

static PANEL_GDI_MIRROR: OnceLock<Mutex<PanelGdiMirrorState>> = OnceLock::new();
static PANEL_GDI_HOST: OnceLock<Mutex<isize>> = OnceLock::new();
// 0 = foreground/no background wake, 1 = minimized, 2 = hidden.
// Native panel actions can originate while eframe is event-starved in the
// background. Preserve the state that existed before the temporary wake so
// main.rs can restore it after the action has actually committed.
static PANEL_ACTION_BACKGROUND_WAKE: AtomicU8 = AtomicU8::new(0);
// The physical HWND briefly becomes visible/non-iconic while a background
// command is pumped. Preserve the user's intended presentation separately.
// 0 = foreground, 1 = minimized, 2 = tray-hidden.
static MAIN_GUI_BACKGROUND_INTENT: AtomicU8 = AtomicU8::new(0);
static MAIN_GUI_BACKGROUND_WAKE_ACTIVE: AtomicBool = AtomicBool::new(false);
// Number of native wake_background_gui() calls that have not returned yet.
// RedrawWindow(RDW_UPDATENOW) can synchronously wake/re-enter the root event
// loop before the caller has completed its Win32/DWM wake sequence. A GUI
// restore must not release the temporary cloak until every such wake call has
// returned, otherwise DWM can commit the older cloak one frame later.
static MAIN_GUI_BACKGROUND_WAKE_IN_FLIGHT: AtomicU32 = AtomicU32::new(0);
// Native Stop can complete while the root GUI is minimized and event-starved.
// Keep a process-wide visual latch so a stale running snapshot in the next GUI
// pass cannot accidentally reveal the retained panel host again.
static PANEL_CAPTURE_STOP_QUIESCED: AtomicBool = AtomicBool::new(false);
// The collapsed operation panel is intentionally a fully transparent layered
// HWND. Windows can omit alpha=0 windows from WindowFromPoint, so virtual-cursor
// rediscovery needs one explicit geometric wake bridge. These atomics are only
// presentation/wake state; panel actions and input ownership remain unchanged.
static PANEL_LURK_HOVER_ACTIVE: AtomicBool = AtomicBool::new(false);
static PANEL_LURK_VIRTUAL_HOVER_INSIDE: AtomicBool = AtomicBool::new(false);
// Explicit tray/quit shutdown posts WM_CLOSE to the real root HWND.
// Block any late background wake while that normal native close path runs.
static MAIN_GUI_EXPLICIT_EXITING: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug, Default, PartialEq)]
struct OverloadNoticeGdiSnapshot {
    x: i32,
    y: i32,
    width: i32,
    height: i32,
    font_px: i32,
    text: String,
    visible: bool,
}

#[derive(Debug, Default)]
struct OverloadNoticeGdiState {
    hwnd: isize,
    snapshot: OverloadNoticeGdiSnapshot,
}

static OVERLOAD_NOTICE_GDI: OnceLock<Mutex<OverloadNoticeGdiState>> = OnceLock::new();

#[derive(Debug, Default)]
struct GuiTransitionSnapshotState {
    hwnd: isize,
    bitmap: isize,
    width: i32,
    height: i32,
}

static GUI_TRANSITION_SNAPSHOT: OnceLock<Mutex<GuiTransitionSnapshotState>> = OnceLock::new();

// Top-level Neo windows temporarily excluded from monitor capture while a
// structural display-region fallback is active. The list exists only so Stop
// can restore WDA_NONE deterministically even when individual helper HWNDs were
// created after capture startup.
static CAPTURE_EXCLUDED_WINDOWS: OnceLock<Mutex<Vec<isize>>> = OnceLock::new();

// Keep the native WNDPROC handoff installed for the lifetime of the eframe
// window. Unlike the former Ctrl+Alt+G route, every message (including
// SC_MINIMIZE) is forwarded unchanged so Windows owns minimize/restore.
static MAIN_GUI_SUBCLASS_HWND: AtomicIsize = AtomicIsize::new(0);
static MAIN_GUI_OLD_WNDPROC: AtomicIsize = AtomicIsize::new(0);
static SINGLE_INSTANCE_HANDLE: AtomicIsize = AtomicIsize::new(0);

unsafe extern "system" fn main_gui_caption_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // Make application exit feel immediate without weakening shutdown safety.
    // Hide only Neo's own root GUI as soon as Windows delivers WM_CLOSE, then
    // forward the message unchanged so eframe/on_exit can continue the normal
    // cursor/source/geometry/provider cleanup in the background. Never wait on
    // DWM, the render thread, or any foreign HWND from this native callback.
    if msg == WM_CLOSE {
        // Arm only the janitor's post-quit grace clock. No cursor/source/input
        // state is changed here; the ordinary close path gets its full stable
        // shutdown opportunity first.
        crate::input::notify_cursor_janitor_quit_requested("main-gui-close");
        let _ = unsafe { ShowWindow(hwnd, SW_HIDE) };
        log::info!(
            "main-gui-close-visual-hide: hwnd={:#x} cleanup=continue",
            hwnd.0 as isize
        );
    }

    // Native minimization is a GUI-only presentation transition. The floating
    // panel has its own lifetime and must never be shown/hidden from this WNDPROC.
    // Only preserve the cursor fail-visible contract here.
    if msg == WM_SYSCOMMAND && (wparam.0 & 0xfff0) == SC_MINIMIZE as usize {
        MAIN_GUI_BACKGROUND_INTENT.store(1, Ordering::Release);
        crate::input::handle_main_gui_minimize_begin();
    }
    if msg == WM_SIZE && wparam.0 == SIZE_MINIMIZED as usize {
        MAIN_GUI_BACKGROUND_INTENT.store(1, Ordering::Release);
        // WM_SYSCOMMAND is not guaranteed for every shell/taskbar route.
        crate::input::handle_main_gui_minimize_begin();
    } else if msg == WM_SIZE
        && wparam.0 != SIZE_MINIMIZED as usize
        && !MAIN_GUI_BACKGROUND_WAKE_ACTIVE.load(Ordering::Acquire)
    {
        MAIN_GUI_BACKGROUND_INTENT.store(0, Ordering::Release);
    }
    let old = MAIN_GUI_OLD_WNDPROC.load(Ordering::Acquire);
    if old != 0 {
        let old_proc: WNDPROC = unsafe { std::mem::transmute(old) };
        unsafe { CallWindowProcW(old_proc, hwnd, msg, wparam, lparam) }
    } else {
        unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
    }
}

pub fn install_main_gui_native_minimize(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) || MAIN_GUI_SUBCLASS_HWND.load(Ordering::Acquire) == hwnd
    {
        return;
    }
    unsafe {
        let old = SetWindowLongPtrW(
            HWND(hwnd as *mut _),
            GWL_WNDPROC,
            main_gui_caption_wndproc as *const () as usize as isize,
        );
        if old != 0 {
            MAIN_GUI_OLD_WNDPROC.store(old, Ordering::Release);
            MAIN_GUI_SUBCLASS_HWND.store(hwnd, Ordering::Release);
            log::info!("main-gui-caption-minimize-route: hwnd={hwnd:#x} action=native-minimize");
        } else {
            log::warn!("main-gui-caption-minimize-route: hwnd={hwnd:#x} install=failed");
        }
    }
}

pub fn main_gui_hwnd() -> isize {
    MAIN_GUI_SUBCLASS_HWND.load(Ordering::Acquire)
}

pub fn main_gui_background_intent() -> u8 {
    MAIN_GUI_BACKGROUND_INTENT.load(Ordering::Acquire)
}

pub fn finish_background_gui_wake() {
    MAIN_GUI_BACKGROUND_WAKE_ACTIVE.store(false, Ordering::Release);
}

pub fn background_gui_wake_in_flight() -> bool {
    MAIN_GUI_BACKGROUND_WAKE_IN_FLIGHT.load(Ordering::Acquire) != 0
}

/// Wake the root GUI as soon as a native panel action is queued.
///
/// Wake the root event loop for a native panel interaction.
/// Foreground roots use a cheap WM_NULL. Minimized/hidden roots temporarily
/// enter the same non-activating background pump used by global hotkeys; App
/// restores the exact prior state after the queued interaction has committed.
pub fn wake_main_gui_for_panel_action(_restore_if_minimized: bool) {
    if MAIN_GUI_EXPLICIT_EXITING.load(Ordering::Acquire) {
        return;
    }
    let hwnd = main_gui_hwnd();
    if hwnd == 0 || !is_window_valid(hwnd) {
        return;
    }
    // WM_NULL wakes an ordinary visible root, but a minimized/hidden eframe
    // root may stay event-starved indefinitely. Use the same non-activating
    // background wake as global hotkeys and remember the exact original state
    // so App can restore it after the queued panel action is consumed.
    let intent = main_gui_background_intent();
    let background = if intent != 0 {
        intent
    } else if is_minimized(hwnd) {
        1
    } else if !is_window_visible(hwnd) {
        2
    } else {
        0
    };
    if background != 0 {
        let _ = PANEL_ACTION_BACKGROUND_WAKE.compare_exchange(
            0,
            background,
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        let _ = wake_background_gui(hwnd);
    } else {
        unsafe {
            let h = HWND(hwnd as *mut _);
            // A click can arrive a few milliseconds after a background root
            // was restored. WM_NULL alone does not always schedule another
            // eframe pass at that boundary, leaving the queued action dormant
            // until the user's next click. Force one ordinary paint without
            // changing visibility, activation, ownership, or Z-order.
            let _ = RedrawWindow(Some(h), None, None, RDW_INVALIDATE | RDW_UPDATENOW);
            let _ = PostMessageW(Some(h), WM_PAINT, WPARAM(0), LPARAM(0));
            let _ = PostMessageW(Some(h), WM_NULL, WPARAM(0), LPARAM(0));
        }
    }
}

/// Take the original root-window state recorded by a native panel action.
/// Returns 0 for foreground/no wake, 1 for minimized and 2 for hidden.
pub fn take_panel_action_background_wake() -> u8 {
    PANEL_ACTION_BACKGROUND_WAKE.swap(0, Ordering::AcqRel)
}

/// Immediately make the retained native operation panel non-visible when a
/// capture Stop is requested. The HWND is intentionally retained for reuse on
/// the next Start, but visual/input ownership must end with the capture even if
/// the minimized root GUI does not receive another eframe pass for a while.
pub fn quiesce_panel_for_capture_stop() -> isize {
    PANEL_CAPTURE_STOP_QUIESCED.store(true, Ordering::Release);
    set_panel_lurk_hover_active(false);
    let hwnd = panel_gdi_host_hwnd();
    hide_panel_gdi_mirror();
    if hwnd != 0 && is_panel_gdi_host(hwnd) {
        set_window_alpha(hwnd, 0);
        set_window_input_passthrough(hwnd, true);
        set_panel_gdi_host_visible(hwnd, false);
    }
    hwnd
}

pub fn panel_capture_stop_quiesced() -> bool {
    PANEL_CAPTURE_STOP_QUIESCED.load(Ordering::Acquire)
}

/// Publish whether the native panel is currently the fully transparent lurk
/// chip. Reset the outside/inside edge detector only when that semantic state
/// changes, so ordinary mouse motion inside the chip never floods the GUI pump.
pub fn set_panel_lurk_hover_active(active: bool) {
    let previous = PANEL_LURK_HOVER_ACTIVE.swap(active, Ordering::AcqRel);
    if previous != active {
        PANEL_LURK_VIRTUAL_HOVER_INSIDE.store(false, Ordering::Release);
    }
}

/// Wake an event-starved root when Neo's visible *virtual* cursor enters the
/// exact live rectangle of the alpha=0 lurk chip. This is intentionally a
/// geometric exception only for the transparent rediscovery state; the visible
/// bar continues to use normal Win32 top-level ownership and hit-testing.
pub fn update_panel_lurk_virtual_hover(x: i32, y: i32) {
    if !PANEL_LURK_HOVER_ACTIVE.load(Ordering::Acquire)
        || MAIN_GUI_EXPLICIT_EXITING.load(Ordering::Acquire)
    {
        PANEL_LURK_VIRTUAL_HOVER_INSIDE.store(false, Ordering::Release);
        return;
    }
    let hwnd = panel_gdi_host_hwnd();
    let inside = hwnd != 0
        && is_window_valid(hwnd)
        && is_own_window(hwnd)
        && is_window_visible(hwnd)
        && window_rect(hwnd).is_some_and(|(rx, ry, rw, rh)| {
            rw > 0 && rh > 0 && x >= rx && x < rx + rw && y >= ry && y < ry + rh
        });
    let was_inside = PANEL_LURK_VIRTUAL_HOVER_INSIDE.swap(inside, Ordering::AcqRel);
    if inside && !was_inside {
        if crate::logging::diagnostics_enabled() {
            log::debug!(
                "panel-lurk-virtual-hover-enter: hwnd={hwnd:#x} point=({x},{y}) action=wake-root"
            );
        }
        wake_main_gui_for_panel_action(false);
    }
}

pub fn clear_panel_capture_stop_quiesce() {
    PANEL_CAPTURE_STOP_QUIESCED.store(false, Ordering::Release);
}

fn gui_transition_snapshot_state() -> &'static Mutex<GuiTransitionSnapshotState> {
    GUI_TRANSITION_SNAPSHOT.get_or_init(|| Mutex::new(GuiTransitionSnapshotState::default()))
}

// Keep the tiny transition snapshot helper independent of the windows crate's
// generated GDI signatures. These are stable Win32 ABI functions and are used
// only for a short desktop-to-memory BitBlt around GUI mode changes.
type RawWinHandle = *mut core::ffi::c_void;

#[link(name = "user32")]
unsafe extern "system" {
    #[link_name = "GetDC"]
    fn raw_get_dc(hwnd: RawWinHandle) -> RawWinHandle;
    #[link_name = "ReleaseDC"]
    fn raw_release_dc(hwnd: RawWinHandle, hdc: RawWinHandle) -> i32;
    #[link_name = "SetWindowRgn"]
    fn raw_set_window_rgn(hwnd: RawWinHandle, region: RawWinHandle, redraw: i32) -> i32;
}

#[link(name = "gdi32")]
unsafe extern "system" {
    #[link_name = "CreateCompatibleDC"]
    fn raw_create_compatible_dc(hdc: RawWinHandle) -> RawWinHandle;
    #[link_name = "DeleteDC"]
    fn raw_delete_dc(hdc: RawWinHandle) -> i32;
    #[link_name = "CreateCompatibleBitmap"]
    fn raw_create_compatible_bitmap(hdc: RawWinHandle, width: i32, height: i32) -> RawWinHandle;
    #[link_name = "SelectObject"]
    fn raw_select_object(hdc: RawWinHandle, object: RawWinHandle) -> RawWinHandle;
    #[link_name = "DeleteObject"]
    fn raw_delete_object(object: RawWinHandle) -> i32;
    #[link_name = "CreateRoundRectRgn"]
    fn raw_create_round_rect_rgn(
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
        ellipse_width: i32,
        ellipse_height: i32,
    ) -> RawWinHandle;
    #[link_name = "BitBlt"]
    fn raw_bit_blt(
        dst: RawWinHandle,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        src: RawWinHandle,
        src_x: i32,
        src_y: i32,
        rop: u32,
    ) -> i32;
}

const RAW_SRCCOPY: u32 = 0x00CC_0020;

fn panel_gdi_host_state() -> &'static Mutex<isize> {
    PANEL_GDI_HOST.get_or_init(|| Mutex::new(0))
}

fn overload_notice_gdi_state() -> &'static Mutex<OverloadNoticeGdiState> {
    OVERLOAD_NOTICE_GDI.get_or_init(|| Mutex::new(OverloadNoticeGdiState::default()))
}

/// Return the current native GDI control-panel host HWND.
///
/// The render path uses this at the overlay reveal boundary so z-order
/// normalization observes the actual live helper window instead of relying on
/// engine-loop locals that are out of scope in frame-processing helpers.
pub fn panel_gdi_host_hwnd() -> isize {
    *panel_gdi_host_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn panel_gdi_mirror_state() -> &'static Mutex<PanelGdiMirrorState> {
    PANEL_GDI_MIRROR.get_or_init(|| Mutex::new(PanelGdiMirrorState::default()))
}

fn panel_gdi_scale(value: f32, width: i32) -> i32 {
    ((value * width.max(1) as f32 / 270.0).round() as i32).max(1)
}

fn panel_gdi_vscale(value: f32, height: i32) -> i32 {
    ((value * height.max(1) as f32 / 30.0).round() as i32).max(1)
}

fn panel_gdi_rect_from_points(
    width: i32,
    height: i32,
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
) -> RECT {
    let px = |v: f32| ((v * width.max(1) as f32 / 270.0).round() as i32).clamp(0, width);
    let py = |v: f32| ((v * height.max(1) as f32 / 30.0).round() as i32).clamp(0, height);
    RECT {
        left: (px(left) - 2).max(0),
        top: (py(top) - 2).max(0),
        right: (px(right) + 2).min(width.max(1)),
        bottom: (py(bottom) + 2).min(height.max(1)),
    }
}

fn panel_gdi_union_rect(a: RECT, b: RECT) -> RECT {
    RECT {
        left: a.left.min(b.left),
        top: a.top.min(b.top),
        right: a.right.max(b.right),
        bottom: a.bottom.max(b.bottom),
    }
}

fn panel_gdi_slot_dirty_rect(slot: u8, width: i32, height: i32) -> Option<RECT> {
    let (left, right) = match slot {
        1 => (3.0, 85.0),
        2 => (159.0, 193.0),
        3 => (196.0, 230.0),
        4 => (233.0, 267.0),
        _ => return None,
    };
    Some(panel_gdi_rect_from_points(
        width, height, left, 2.0, right, 28.0,
    ))
}

fn panel_gdi_visual_dirty_rect(
    old: &PanelGdiMirrorSnapshot,
    new: &PanelGdiMirrorSnapshot,
) -> Option<RECT> {
    if old.width != new.width
        || old.height != new.height
        || old.visible != new.visible
        || old.chip != new.chip
    {
        return None;
    }

    let mut dirty: Option<RECT> = None;
    let mut add = |rect: RECT| {
        dirty = Some(match dirty {
            Some(current) => panel_gdi_union_rect(current, rect),
            None => rect,
        });
    };

    if old.stop_text != new.stop_text {
        if let Some(rect) = panel_gdi_slot_dirty_rect(1, new.width, new.height) {
            add(rect);
        }
    }
    if old.fps_tenths != new.fps_tenths {
        add(panel_gdi_rect_from_points(
            new.width, new.height, 88.0, 2.0, 156.0, 28.0,
        ));
    }
    if old.hover_slot != new.hover_slot {
        if let Some(rect) = panel_gdi_slot_dirty_rect(old.hover_slot, new.width, new.height) {
            add(rect);
        }
        if let Some(rect) = panel_gdi_slot_dirty_rect(new.hover_slot, new.width, new.height) {
            add(rect);
        }
    }
    if old.screenshot_feedback != new.screenshot_feedback {
        if let Some(rect) = panel_gdi_slot_dirty_rect(2, new.width, new.height) {
            add(rect);
        }
    }
    dirty
}

/// Publish only a small changed part of the already double-buffered panel.
///
/// v508 fixed panel flicker by drawing a complete frame off-screen before one
/// BitBlt. Keep that contract, but do not invalidate/replace all 297x33 pixels
/// for a periodic FPS-number update or a single hover-face change. On AMD the
/// fullscreen child repaint could momentarily disturb the independent layered
/// cursor even though the cursor HWND itself never hid. A region BitBlt keeps
/// the exact same GDI pixels while leaving unrelated pixels under the cursor
/// untouched.
unsafe fn publish_panel_gdi_region(
    child: HWND,
    snapshot: &PanelGdiMirrorSnapshot,
    rect: RECT,
) -> bool {
    if rect.right <= rect.left || rect.bottom <= rect.top {
        return true;
    }
    unsafe {
        let hdc = raw_get_dc(child.0);
        if hdc.is_null() {
            return false;
        }
        let mem = raw_create_compatible_dc(hdc);
        let bitmap = if !mem.is_null() {
            raw_create_compatible_bitmap(hdc, snapshot.width.max(1), snapshot.height.max(1))
        } else {
            core::ptr::null_mut()
        };
        let mut published = false;
        if !mem.is_null() && !bitmap.is_null() {
            let old = raw_select_object(mem, bitmap);
            if !old.is_null() {
                paint_panel_gdi_mirror(windows::Win32::Graphics::Gdi::HDC(mem), snapshot);
                published = raw_bit_blt(
                    hdc,
                    rect.left,
                    rect.top,
                    (rect.right - rect.left).max(1),
                    (rect.bottom - rect.top).max(1),
                    mem,
                    rect.left,
                    rect.top,
                    RAW_SRCCOPY,
                ) != 0;
                let _ = raw_select_object(mem, old);
            }
        }
        if !bitmap.is_null() {
            let _ = raw_delete_object(bitmap);
        }
        if !mem.is_null() {
            let _ = raw_delete_dc(mem);
        }
        let _ = raw_release_dc(child.0, hdc);
        published
    }
}

unsafe fn panel_gdi_fill(hdc: windows::Win32::Graphics::Gdi::HDC, rect: RECT, color: COLORREF) {
    unsafe {
        if rect.right <= rect.left || rect.bottom <= rect.top {
            return;
        }
        let brush = CreateSolidBrush(color);
        let _ = FillRect(hdc, &rect, brush);
        let _ = DeleteObject(brush.into());
    }
}

/// Rasterize a tiny rounded rectangle with FillRect only. Keeping this helper
/// brush-only avoids introducing another swap/present path while reproducing
/// egui's 3 px panel-button corner radius closely at the 30 px bar scale.
unsafe fn panel_gdi_round_fill(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    rect: RECT,
    radius: i32,
    color: COLORREF,
) {
    unsafe {
        let width = rect.right - rect.left;
        let height = rect.bottom - rect.top;
        if width <= 0 || height <= 0 {
            return;
        }
        let r = radius.max(0).min(width / 2).min(height / 2);
        if r <= 1 {
            panel_gdi_fill(hdc, rect, color);
            return;
        }
        let rf = r as f32;
        for row in 0..height {
            let edge = row.min(height - 1 - row);
            let inset = if edge >= r {
                0
            } else {
                let y = rf - edge as f32 - 0.5;
                (rf - (rf * rf - y * y).max(0.0).sqrt()).ceil() as i32
            }
            .min(r);
            panel_gdi_fill(
                hdc,
                RECT {
                    left: rect.left + inset,
                    top: rect.top + row,
                    right: rect.right - inset,
                    bottom: rect.top + row + 1,
                },
                color,
            );
        }
    }
}

unsafe fn panel_gdi_round_frame(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    rect: RECT,
    radius: i32,
    stroke: i32,
    color: COLORREF,
    interior: COLORREF,
) {
    unsafe {
        let stroke = stroke.max(1);
        panel_gdi_round_fill(hdc, rect, radius, color);
        let inner = RECT {
            left: rect.left + stroke,
            top: rect.top + stroke,
            right: rect.right - stroke,
            bottom: rect.bottom - stroke,
        };
        if inner.right > inner.left && inner.bottom > inner.top {
            panel_gdi_round_fill(hdc, inner, (radius - stroke).max(1), interior);
        }
    }
}

unsafe fn panel_gdi_circle_fill(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    cx: i32,
    cy: i32,
    radius: i32,
    color: COLORREF,
) {
    unsafe {
        let r = radius.max(1);
        let rf = r as f32;
        for dy in -r..=r {
            let yf = dy as f32;
            let half = (rf * rf - yf * yf).max(0.0).sqrt().round() as i32;
            panel_gdi_fill(
                hdc,
                RECT {
                    left: cx - half,
                    top: cy + dy,
                    right: cx + half + 1,
                    bottom: cy + dy + 1,
                },
                color,
            );
        }
    }
}

unsafe fn panel_gdi_circle_frame(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    cx: i32,
    cy: i32,
    radius: i32,
    stroke: i32,
    color: COLORREF,
    interior: COLORREF,
) {
    unsafe {
        let r = radius.max(1);
        panel_gdi_circle_fill(hdc, cx, cy, r, color);
        let inner = (r - stroke.max(1)).max(0);
        if inner > 0 {
            panel_gdi_circle_fill(hdc, cx, cy, inner, interior);
        }
    }
}

fn panel_gdi_font_face(text: &str) -> (&'static str, i32) {
    let cjk = text.chars().any(|ch| {
        matches!(
            ch as u32,
            0x3040..=0x30ff | 0x31f0..=0x31ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff
        )
    });
    if cjk {
        // The egui Japanese family is YuGothM.ttc (medium weight).
        ("Yu Gothic UI", 500)
    } else {
        ("Segoe UI", 400)
    }
}

unsafe fn panel_gdi_text(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    mut rect: RECT,
    text: &str,
    color: COLORREF,
    pixel_height: i32,
    face: &str,
    weight: i32,
) {
    unsafe {
        let mut wide: Vec<u16> = text.encode_utf16().collect();
        if wide.is_empty() {
            return;
        }

        // Match the egui panel's explicit 12.0/11.5 px fonts instead of the
        // DPI-dependent DEFAULT_GUI_FONT, which made v449 text visibly too big.
        let mut lf = LOGFONTW::default();
        lf.lfHeight = -pixel_height.max(1);
        lf.lfWeight = weight;
        lf.lfCharSet = windows::Win32::Graphics::Gdi::FONT_CHARSET(1); // DEFAULT_CHARSET
        lf.lfQuality = windows::Win32::Graphics::Gdi::FONT_QUALITY(5); // CLEARTYPE_QUALITY
        for (dst, src) in lf.lfFaceName.iter_mut().take(31).zip(face.encode_utf16()) {
            *dst = src;
        }
        let font = CreateFontIndirectW(&lf);
        let font_ok = !font.0.is_null();
        let old_font = if font_ok {
            SelectObject(hdc, font.into())
        } else {
            SelectObject(hdc, GetStockObject(DEFAULT_GUI_FONT))
        };
        let _ = SetBkMode(hdc, TRANSPARENT);
        let _ = SetTextColor(hdc, color);
        let _ = DrawTextW(
            hdc,
            &mut wide,
            &mut rect,
            DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX,
        );
        let _ = SelectObject(hdc, old_font);
        if font_ok {
            let _ = DeleteObject(font.into());
        }
    }
}

unsafe fn paint_panel_gdi_mirror(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    snapshot: &PanelGdiMirrorSnapshot,
) {
    unsafe {
        let w = snapshot.width.max(1);
        let h = snapshot.height.max(1);
        let sx = |v: f32| panel_gdi_scale(v, w);
        let sy = |v: f32| panel_gdi_vscale(v, h);
        let px = |v: f32| ((v * w as f32 / 270.0).round() as i32).clamp(0, w);
        let py = |v: f32| ((v * h as f32 / 30.0).round() as i32).clamp(0, h);

        const BG: COLORREF = COLORREF(0x002B2B2B);
        const BUTTON: COLORREF = COLORREF(0x003C3C3C);
        const HOVER: COLORREF = COLORREF(0x00505050);
        // COLORREF is 0x00BBGGRR. PANEL_ACTIVE is rgb(0x2f,0x7d,0x4a).
        const ACTIVE: COLORREF = COLORREF(0x004A7D2F);
        const FG: COLORREF = COLORREF(0x00F2F2F2);
        const GRIP: COLORREF = COLORREF(0x001F1F1F);

        if snapshot.chip {
            panel_gdi_fill(
                hdc,
                RECT {
                    left: 0,
                    top: 0,
                    right: w,
                    bottom: h,
                },
                BG,
            );
            let inset = panel_gdi_scale(1.0, w).max(1);
            panel_gdi_round_fill(
                hdc,
                RECT {
                    left: inset,
                    top: inset,
                    right: (w - inset).max(inset + 1),
                    bottom: (h - inset).max(inset + 1),
                },
                panel_gdi_scale(2.0, w).max(2),
                GRIP,
            );
            return;
        }

        panel_gdi_fill(
            hdc,
            RECT {
                left: 0,
                top: 0,
                right: w,
                bottom: h,
            },
            BG,
        );

        // egui CentralPanel has 2 pt inner margin, then horizontal_centered
        // centers 264 pt of controls in the remaining 266 pt -> 3 pt left/right.
        let y0 = py(3.0);
        let y1 = py(27.0).max(y0 + 1);
        let corner = sx(3.0).max(2);
        let mut x_pts = 3.0_f32;

        macro_rules! panel_button {
            ($width_pts:expr, $slot:expr, $active:expr) => {{
                let rect = RECT {
                    left: px(x_pts),
                    top: y0,
                    right: px(x_pts + $width_pts),
                    bottom: y1,
                };
                let color = if $active {
                    ACTIVE
                } else if snapshot.hover_slot == $slot {
                    HOVER
                } else {
                    BUTTON
                };
                panel_gdi_round_fill(hdc, rect, corner, color);
                x_pts += $width_pts + 3.0;
                (rect, color)
            }};
        }

        let (stop, _stop_fill) = panel_button!(82.0, 1, false);
        let stop_cx = (stop.left + stop.right) / 2;
        let stop_cy = (stop.top + stop.bottom) / 2;
        let stop_square = sx(6.0).max(2);
        let square_cx = stop_cx - sx(22.0);
        panel_gdi_round_fill(
            hdc,
            RECT {
                left: square_cx - stop_square / 2,
                top: stop_cy - stop_square / 2,
                right: square_cx + (stop_square + 1) / 2,
                bottom: stop_cy + (stop_square + 1) / 2,
            },
            1,
            FG,
        );
        let stop_target_x = stop_cx + sx(5.0);
        let (stop_face, stop_weight) = panel_gdi_font_face(&snapshot.stop_text);
        panel_gdi_text(
            hdc,
            RECT {
                left: stop_target_x - sx(31.0),
                top: stop.top,
                right: stop_target_x + sx(31.0),
                bottom: stop.bottom,
            },
            &snapshot.stop_text,
            FG,
            sy(12.0),
            stop_face,
            stop_weight,
        );

        let fps_rect = RECT {
            left: px(x_pts),
            top: y0 - sy(1.0),
            right: px(x_pts + 68.0),
            bottom: y1 - sy(1.0),
        };
        let fps = format!("{:.1} fps", snapshot.fps_tenths as f32 / 10.0);
        panel_gdi_text(hdc, fps_rect, &fps, FG, sy(11.5), "Segoe UI", 400);
        x_pts += 68.0 + 3.0;

        let (camera, camera_fill) = panel_button!(34.0, 2, snapshot.screenshot_feedback);
        let cx = (camera.left + camera.right) / 2;
        let cy = (camera.top + camera.bottom) / 2;
        let body_w = sx(16.0);
        let body_h = sy(10.0);
        let body_cy = cy + sy(1.5);
        let body = RECT {
            left: cx - body_w / 2,
            top: body_cy - body_h / 2,
            right: cx + (body_w + 1) / 2,
            bottom: body_cy + (body_h + 1) / 2,
        };
        panel_gdi_round_frame(hdc, body, sx(2.0).max(2), 1, FG, camera_fill);
        let bump_w = sx(6.0);
        let bump_h = sy(3.0);
        let bump_cx = cx - sx(3.0);
        let bump_cy = cy - sy(5.0);
        panel_gdi_round_fill(
            hdc,
            RECT {
                left: bump_cx - bump_w / 2,
                top: bump_cy - bump_h / 2,
                right: bump_cx + (bump_w + 1) / 2,
                bottom: bump_cy + (bump_h + 1) / 2,
            },
            1,
            FG,
        );
        panel_gdi_circle_frame(hdc, cx, body_cy, sx(2.7).max(2), 1, FG, camera_fill);

        let (gui, gui_fill) = panel_button!(34.0, 3, false);
        let gx = (gui.left + gui.right) / 2;
        let gy = (gui.top + gui.bottom) / 2;
        let gw = sx(16.0);
        let gh = sy(12.0);
        let window = RECT {
            left: gx - gw / 2,
            top: gy - gh / 2,
            right: gx + (gw + 1) / 2,
            bottom: gy + (gh + 1) / 2,
        };
        panel_gdi_round_frame(hdc, window, sx(2.0).max(2), 1, FG, gui_fill);
        let chrome_y = window.top + sy(3.5);
        panel_gdi_fill(
            hdc,
            RECT {
                left: window.left + sx(1.5),
                top: chrome_y,
                right: window.right - sx(1.5),
                bottom: chrome_y + 1,
            },
            FG,
        );
        for dot_x in [3.0_f32, 5.3, 7.6] {
            panel_gdi_circle_fill(hdc, window.left + sx(dot_x), window.top + sy(2.0), 1, FG);
        }

        // Final button: same 34x24 pt rounded face as egui, without advancing x.
        let collapse = RECT {
            left: px(x_pts),
            top: y0,
            right: px(x_pts + 34.0),
            bottom: y1,
        };
        let collapse_fill = if snapshot.hover_slot == 4 {
            HOVER
        } else {
            BUTTON
        };
        panel_gdi_round_fill(hdc, collapse, corner, collapse_fill);
        let ccx = (collapse.left + collapse.right) / 2;
        let ccy = (collapse.top + collapse.bottom) / 2;
        let half_line = sx(6.0);
        let line_h = sy(1.6).max(1);
        panel_gdi_fill(
            hdc,
            RECT {
                left: ccx - half_line,
                top: ccy - line_h / 2,
                right: ccx + half_line + 1,
                bottom: ccy + (line_h + 1) / 2,
            },
            FG,
        );
    }
}

unsafe extern "system" fn gui_transition_snapshot_wndproc(
    hwnd: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_ERASEBKGND => LRESULT(1),
        WM_PAINT => unsafe {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            let snapshot = gui_transition_snapshot_state()
                .lock()
                .ok()
                .and_then(|state| {
                    (state.hwnd == hwnd.0 as isize && state.bitmap != 0).then_some((
                        state.bitmap,
                        state.width,
                        state.height,
                    ))
                });
            if let Some((bitmap, width, height)) = snapshot {
                let mem = raw_create_compatible_dc(hdc.0);
                if !mem.is_null() {
                    let old = raw_select_object(mem, bitmap as RawWinHandle);
                    let _ = raw_bit_blt(hdc.0, 0, 0, width, height, mem, 0, 0, RAW_SRCCOPY);
                    if !old.is_null() {
                        let _ = raw_select_object(mem, old);
                    }
                    let _ = raw_delete_dc(mem);
                }
            }
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        },
        WM_DESTROY => {
            if let Ok(mut state) = gui_transition_snapshot_state().try_lock() {
                if state.hwnd == hwnd.0 as isize {
                    state.hwnd = 0;
                }
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

unsafe fn create_gui_transition_snapshot_window(
    owner_hwnd: isize,
    x: i32,
    y: i32,
    width: i32,
    height: i32,
) -> Option<HWND> {
    unsafe {
        let instance = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok()?;
        let class: Vec<u16> = "NeoGuiTransitionSnapshot\0".encode_utf16().collect();
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(gui_transition_snapshot_wndproc),
            hInstance: instance.into(),
            lpszClassName: windows::core::PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wc); // zero when already registered is fine.
        CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            windows::core::PCWSTR(class.as_ptr()),
            windows::core::PCWSTR(class.as_ptr()),
            WS_POPUP,
            x,
            y,
            width.max(1),
            height.max(1),
            Some(HWND(owner_hwnd as *mut _)),
            None,
            Some(instance.into()),
            None,
        )
        .ok()
    }
}

/// Cover the root GUI's entire old/new footprint with a snapshot of the screen
/// as it looked immediately before the WGPU resize. The capture rectangle is
/// the union of the current outer window and the expected target outer window,
/// so growing and shrinking mode changes both remain visually frozen until the
/// final target layout has been presented behind this helper.
pub fn show_gui_transition_snapshot(
    gui_hwnd: isize,
    target_inner_width_pts: f32,
    target_inner_height_pts: f32,
    pixels_per_point: f32,
) -> bool {
    hide_gui_transition_snapshot();
    if gui_hwnd == 0 || !is_window_valid(gui_hwnd) || !is_own_window(gui_hwnd) {
        return false;
    }
    let Some((outer_x, outer_y, outer_w, outer_h)) = window_rect(gui_hwnd) else {
        return false;
    };
    let Some((client_x, client_y, client_w, client_h)) = client_rect_on_screen(gui_hwnd) else {
        return false;
    };
    let scale = pixels_per_point.max(0.1);
    let target_client_w = (target_inner_width_pts * scale).round().max(1.0) as i32;
    let target_client_h = (target_inner_height_pts * scale).round().max(1.0) as i32;
    let frame_left = client_x - outer_x;
    let frame_top = client_y - outer_y;
    let frame_right = (outer_w - frame_left - client_w).max(0);
    let frame_bottom = (outer_h - frame_top - client_h).max(0);
    let target_outer_w = (target_client_w + frame_left + frame_right).max(1);
    let target_outer_h = (target_client_h + frame_top + frame_bottom).max(1);
    let capture_w = outer_w.max(target_outer_w).max(1);
    let capture_h = outer_h.max(target_outer_h).max(1);

    let desktop = unsafe { raw_get_dc(core::ptr::null_mut()) };
    if desktop.is_null() {
        return false;
    }
    let mem = unsafe { raw_create_compatible_dc(desktop) };
    if mem.is_null() {
        unsafe {
            let _ = raw_release_dc(core::ptr::null_mut(), desktop);
        }
        return false;
    }
    let bitmap = unsafe { raw_create_compatible_bitmap(desktop, capture_w, capture_h) };
    if bitmap.is_null() {
        unsafe {
            let _ = raw_delete_dc(mem);
            let _ = raw_release_dc(core::ptr::null_mut(), desktop);
        }
        return false;
    }
    let old = unsafe { raw_select_object(mem, bitmap) };
    if old.is_null() {
        unsafe {
            let _ = raw_delete_object(bitmap);
            let _ = raw_delete_dc(mem);
            let _ = raw_release_dc(core::ptr::null_mut(), desktop);
        }
        return false;
    }
    let copied = unsafe {
        raw_bit_blt(
            mem,
            0,
            0,
            capture_w,
            capture_h,
            desktop,
            outer_x,
            outer_y,
            RAW_SRCCOPY,
        ) != 0
    };
    unsafe {
        if !old.is_null() {
            let _ = raw_select_object(mem, old);
        }
        let _ = raw_delete_dc(mem);
        let _ = raw_release_dc(core::ptr::null_mut(), desktop);
    }
    if !copied {
        unsafe {
            let _ = raw_delete_object(bitmap);
        }
        return false;
    }

    let Some(hwnd) = (unsafe {
        create_gui_transition_snapshot_window(gui_hwnd, outer_x, outer_y, capture_w, capture_h)
    }) else {
        unsafe {
            let _ = raw_delete_object(bitmap);
        }
        return false;
    };

    {
        let Ok(mut state) = gui_transition_snapshot_state().lock() else {
            unsafe {
                let _ = DestroyWindow(hwnd);
                let _ = raw_delete_object(bitmap);
            }
            return false;
        };
        state.hwnd = hwnd.0 as isize;
        state.bitmap = bitmap as isize;
        state.width = capture_w;
        state.height = capture_h;
    }

    unsafe {
        // Prime the hidden helper's own GDI surface before it is shown, so DWM
        // never sees an empty/background frame from this temporary window.
        let window_dc = raw_get_dc(hwnd.0);
        if !window_dc.is_null() {
            let paint_dc = raw_create_compatible_dc(window_dc);
            if !paint_dc.is_null() {
                let paint_old = raw_select_object(paint_dc, bitmap);
                if !paint_old.is_null() {
                    let _ = raw_bit_blt(
                        window_dc,
                        0,
                        0,
                        capture_w,
                        capture_h,
                        paint_dc,
                        0,
                        0,
                        RAW_SRCCOPY,
                    );
                    let _ = raw_select_object(paint_dc, paint_old);
                }
                let _ = raw_delete_dc(paint_dc);
            }
            let _ = raw_release_dc(hwnd.0, window_dc);
        }
        // Show without activation. The helper is click-through and is removed
        // after two complete target-layout frames.
        let _ = SetWindowPos(
            hwnd,
            Some(HWND_TOPMOST),
            outer_x,
            outer_y,
            capture_w,
            capture_h,
            SWP_NOACTIVATE | SWP_SHOWWINDOW,
        );
        let _ = RedrawWindow(
            Some(hwnd),
            None,
            None,
            RDW_INVALIDATE | RDW_UPDATENOW | RDW_FRAME,
        );
        // The egui viewport resize is applied as soon as the current update
        // returns.  Merely showing and synchronously painting this helper does
        // not mean DWM has composed it yet: on the next desktop frame the root
        // WGPU surface could therefore be seen briefly with the old layout
        // squeezed into the new size.  Wait for the helper to reach DWM before
        // allowing that queued resize to proceed.  This runs only on explicit
        // GUI mode changes and never touches the video presentation path.
        let _ = DwmFlush();
    }
    log::debug!(
        "gui-transition-snapshot: shown hwnd={:#x} rect=({},{} {}x{}) current_outer={}x{} target_outer={}x{} scale={:.3}",
        hwnd.0 as isize,
        outer_x,
        outer_y,
        capture_w,
        capture_h,
        outer_w,
        outer_h,
        target_outer_w,
        target_outer_h,
        scale
    );
    true
}

pub fn keep_gui_transition_snapshot_topmost() {
    let hwnd = gui_transition_snapshot_state()
        .lock()
        .ok()
        .map(|state| state.hwnd)
        .unwrap_or(0);
    if hwnd == 0 || !is_window_valid(hwnd) {
        return;
    }
    unsafe {
        let _ = SetWindowPos(
            HWND(hwnd as *mut _),
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

/// Wait for one DWM composition boundary while the transition snapshot is
/// still visible. This is intentionally used only for explicit GUI mode
/// changes; it must never be called from the video rendering/pacing path.
pub fn sync_gui_transition_with_dwm() {
    unsafe {
        let _ = DwmFlush();
    }
}

/// Wait for one DWM composition boundary after the floating panel and its
/// compositor keep-alive backing surface have both reached their final state.
/// This is GUI-only transition work; it is never called from video pacing.
pub fn sync_panel_composition_with_dwm() {
    unsafe {
        let _ = DwmFlush();
    }
}

pub fn hide_gui_transition_snapshot() {
    let (hwnd, bitmap) = {
        let Ok(mut state) = gui_transition_snapshot_state().lock() else {
            return;
        };
        let hwnd = state.hwnd;
        let bitmap = state.bitmap;
        *state = GuiTransitionSnapshotState::default();
        (hwnd, bitmap)
    };
    unsafe {
        if hwnd != 0 && is_window_valid(hwnd) {
            let _ = ShowWindow(HWND(hwnd as *mut _), SW_HIDE);
            let _ = DestroyWindow(HWND(hwnd as *mut _));
        }
        if bitmap != 0 {
            let _ = raw_delete_object(bitmap as RawWinHandle);
        }
    }
    if hwnd != 0 || bitmap != 0 {
        log::debug!(
            "gui-transition-snapshot: hidden hwnd={:#x} bitmap={:#x}",
            hwnd,
            bitmap
        );
    }
}

unsafe extern "system" fn panel_gdi_mirror_wndproc(
    hwnd: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_ERASEBKGND => LRESULT(1),
        WM_PAINT => unsafe {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            let snapshot = panel_gdi_mirror_state().lock().ok().and_then(|state| {
                (state.hwnd == hwnd.0 as isize && state.snapshot.visible)
                    .then(|| state.snapshot.clone())
            });
            if let Some(snapshot) = snapshot {
                // v508: compose the complete panel off-screen, then publish it
                // with one BitBlt.  v454-v507 painted background/buttons/text
                // directly into the visible child DC; periodic FPS updates can
                // therefore expose an intermediate paint state on some DWM/AMD
                // compositions.  This changes only the GDI paint transaction:
                // window lifetime, z-order, WGPU keep-alive, input and cursor
                // ownership remain untouched.
                let mem = raw_create_compatible_dc(hdc.0);
                let bitmap = if !mem.is_null() {
                    raw_create_compatible_bitmap(
                        hdc.0,
                        snapshot.width.max(1),
                        snapshot.height.max(1),
                    )
                } else {
                    core::ptr::null_mut()
                };
                let mut published = false;
                if !mem.is_null() && !bitmap.is_null() {
                    let old = raw_select_object(mem, bitmap);
                    if !old.is_null() {
                        paint_panel_gdi_mirror(windows::Win32::Graphics::Gdi::HDC(mem), &snapshot);
                        published = raw_bit_blt(
                            hdc.0,
                            0,
                            0,
                            snapshot.width.max(1),
                            snapshot.height.max(1),
                            mem,
                            0,
                            0,
                            RAW_SRCCOPY,
                        ) != 0;
                        let _ = raw_select_object(mem, old);
                    }
                }
                if !bitmap.is_null() {
                    let _ = raw_delete_object(bitmap);
                }
                if !mem.is_null() {
                    let _ = raw_delete_dc(mem);
                }
                if !published {
                    // Allocation/BitBlt failure must never blank the panel.
                    // Preserve the proven legacy direct-paint fallback.
                    paint_panel_gdi_mirror(hdc, &snapshot);
                }
            }
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        },
        WM_DESTROY => {
            // DestroyWindow can synchronously re-enter this wndproc while the
            // updater owns the mirror-state mutex. Never block here.
            if let Ok(mut state) = panel_gdi_mirror_state().try_lock() {
                if state.hwnd == hwnd.0 as isize {
                    state.hwnd = 0;
                    state.snapshot.visible = false;
                }
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

unsafe extern "system" fn panel_gdi_host_wndproc(
    hwnd: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
) -> LRESULT {
    match msg {
        // The low-level input hook owns all panel buttons. Keep the host a real
        // hit-test target so direct_top_level_window_at_point() can verify that
        // the click still belongs to this panel, but never activate/focus it.
        WM_NCHITTEST => LRESULT(HTCLIENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_MOUSEMOVE => {
            // A minimized/tray-hidden eframe root has no regular repaint
            // cadence. Wake it when the pointer reaches the transparent lurk
            // hit area so the ordinary hover-expand state machine can run.
            if main_gui_background_intent() != 0 {
                let lurking = panel_gdi_mirror_state()
                    .try_lock()
                    .is_ok_and(|state| state.snapshot.chip);
                if lurking {
                    wake_main_gui_for_panel_action(false);
                }
            }
            unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
        }
        WM_ERASEBKGND => LRESULT(1),
        WM_PAINT => unsafe {
            // The child GDI mirror covers the full client area. Validate the
            // host paint region without drawing a second surface underneath it.
            let mut ps = PAINTSTRUCT::default();
            let _ = BeginPaint(hwnd, &mut ps);
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        },
        WM_DESTROY => {
            if let Ok(mut host) = panel_gdi_host_state().try_lock() {
                if *host == hwnd.0 as isize {
                    *host = 0;
                }
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

/// Apply only the visible outer shape of the native control panel.
///
/// This changes neither z-order nor ownership. The region is updated only
/// when the host is created/resized, so the v508 repaint path remains intact.
/// SetWindowRgn takes ownership of the HRGN on success.
fn apply_panel_gdi_host_round_region(hwnd: isize, width: i32, height: i32) {
    if hwnd == 0 || width <= 0 || height <= 0 {
        return;
    }
    let radius = 5.min(width / 2).min(height / 2).max(1);
    let diameter = (radius * 2).max(2);
    unsafe {
        let region = raw_create_round_rect_rgn(
            0,
            0,
            width.saturating_add(1),
            height.saturating_add(1),
            diameter,
            diameter,
        );
        if region.is_null() {
            log::debug!(
                "panel-round-region: create failed hwnd={hwnd:#x} size={}x{}",
                width,
                height
            );
            return;
        }
        if raw_set_window_rgn(hwnd as RawWinHandle, region, 0) == 0 {
            let _ = raw_delete_object(region);
            log::debug!(
                "panel-round-region: apply failed hwnd={hwnd:#x} size={}x{}",
                width,
                height
            );
        }
    }
}

fn overload_gdi_font_face(text: &str) -> (&'static str, i32) {
    let hangul = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x1100..=0x11ff | 0x3130..=0x318f | 0xac00..=0xd7af));
    if hangul {
        return ("Malgun Gothic", 500);
    }
    let cjk = text.chars().any(|ch| {
        matches!(
            ch as u32,
            0x3040..=0x30ff | 0x31f0..=0x31ff | 0x3400..=0x4dbf | 0x4e00..=0x9fff
        )
    });
    if cjk {
        ("Yu Gothic UI", 500)
    } else {
        ("Segoe UI", 500)
    }
}

unsafe fn overload_gdi_make_font(
    text: &str,
    pixel_height: i32,
) -> windows::Win32::Graphics::Gdi::HFONT {
    unsafe {
        let (face, weight) = overload_gdi_font_face(text);
        let mut lf = LOGFONTW::default();
        lf.lfHeight = -pixel_height.max(1);
        lf.lfWeight = weight;
        lf.lfCharSet = windows::Win32::Graphics::Gdi::FONT_CHARSET(1);
        lf.lfQuality = windows::Win32::Graphics::Gdi::FONT_QUALITY(5);
        for (dst, src) in lf.lfFaceName.iter_mut().take(31).zip(face.encode_utf16()) {
            *dst = src;
        }
        CreateFontIndirectW(&lf)
    }
}

fn overload_gdi_measure_text(text: &str, pixel_height: i32) -> i32 {
    if text.is_empty() {
        return 0;
    }
    unsafe {
        let hdc = GetDC(None);
        if hdc.0.is_null() {
            return (text.chars().count() as i32 * pixel_height.max(1) / 2).max(1);
        }
        let font = overload_gdi_make_font(text, pixel_height);
        let font_ok = !font.0.is_null();
        let old_font = if font_ok {
            SelectObject(hdc, font.into())
        } else {
            SelectObject(hdc, GetStockObject(DEFAULT_GUI_FONT))
        };
        let mut wide: Vec<u16> = text.encode_utf16().collect();
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: 32767,
            bottom: 32767,
        };
        let _ = DrawTextW(
            hdc,
            &mut wide,
            &mut rect,
            DT_CALCRECT | DT_SINGLELINE | DT_NOPREFIX,
        );
        let _ = SelectObject(hdc, old_font);
        if font_ok {
            let _ = DeleteObject(font.into());
        }
        let _ = ReleaseDC(None, hdc);
        (rect.right - rect.left).max(1)
    }
}

unsafe fn paint_overload_notice_gdi(
    hdc: windows::Win32::Graphics::Gdi::HDC,
    snapshot: &OverloadNoticeGdiSnapshot,
) {
    unsafe {
        const BG: COLORREF = COLORREF(0x00122732); // rgb(50,39,18)
        const BORDER: COLORREF = COLORREF(0x0042A4D6); // rgb(214,164,66)
        const FG: COLORREF = COLORREF(0x00F2F2F2);

        let w = snapshot.width.max(1);
        let h = snapshot.height.max(1);
        // Always initialize every pixel. The helper is layered/click-through,
        // and leaving the rounded-corner pixels unpainted could expose stale
        // GDI backing-store data precisely on the low-spec path we are trying
        // to keep artifact-free.
        panel_gdi_fill(
            hdc,
            RECT {
                left: 0,
                top: 0,
                right: w,
                bottom: h,
            },
            BG,
        );
        let radius = (h / 6).max(3);
        panel_gdi_round_frame(
            hdc,
            RECT {
                left: 0,
                top: 0,
                right: w,
                bottom: h,
            },
            radius,
            1,
            BORDER,
            BG,
        );

        let icon_h = ((h as f32) * 0.36).round() as i32;
        let icon_h = icon_h.clamp(10, (h - 8).max(10));
        let icon_w = ((icon_h as f32) * 1.08).round() as i32;
        let gap = ((h as f32) * 0.16).round() as i32;
        let text_w = overload_gdi_measure_text(&snapshot.text, snapshot.font_px);
        let group_w = icon_w + gap + text_w;
        let group_left = ((w - group_w) / 2).max(4);
        let cx = group_left + icon_w / 2;
        let cy = h / 2;

        // Font-independent warning triangle. Both outer and inner triangles are
        // scanline-filled, so the icon's physical bounds are centered exactly
        // in the native warning window in every language/font.
        let half_h = icon_h / 2;
        let half_w = icon_w / 2;
        for dy in -half_h..=half_h {
            let t = (dy + half_h) as f32 / (icon_h.max(1) as f32);
            let span = (half_w as f32 * t).round() as i32;
            panel_gdi_fill(
                hdc,
                RECT {
                    left: cx - span,
                    top: cy + dy,
                    right: cx + span + 1,
                    bottom: cy + dy + 1,
                },
                FG,
            );
        }
        let inner_h = (icon_h - 4).max(4);
        let inner_w = (icon_w - 5).max(4);
        let inner_half_h = inner_h / 2;
        let inner_half_w = inner_w / 2;
        for dy in -inner_half_h..=inner_half_h {
            let t = (dy + inner_half_h) as f32 / (inner_h.max(1) as f32);
            let span = (inner_half_w as f32 * t).round() as i32;
            panel_gdi_fill(
                hdc,
                RECT {
                    left: cx - span,
                    top: cy + dy,
                    right: cx + span + 1,
                    bottom: cy + dy + 1,
                },
                BG,
            );
        }
        let mark_h = (icon_h / 3).max(3);
        panel_gdi_fill(
            hdc,
            RECT {
                left: cx,
                top: cy - mark_h / 2,
                right: cx + 1,
                bottom: cy + mark_h / 2 + 1,
            },
            FG,
        );
        panel_gdi_circle_fill(hdc, cx, cy + icon_h / 4, 1, FG);

        let font = overload_gdi_make_font(&snapshot.text, snapshot.font_px);
        let font_ok = !font.0.is_null();
        let old_font = if font_ok {
            SelectObject(hdc, font.into())
        } else {
            SelectObject(hdc, GetStockObject(DEFAULT_GUI_FONT))
        };
        let _ = SetBkMode(hdc, TRANSPARENT);
        let _ = SetTextColor(hdc, FG);
        let mut wide: Vec<u16> = snapshot.text.encode_utf16().collect();
        let text_left = group_left + icon_w + gap;
        let mut text_rect = RECT {
            left: text_left,
            top: 0,
            right: (text_left + text_w + 2).min(w - 4),
            bottom: h,
        };
        let _ = DrawTextW(
            hdc,
            &mut wide,
            &mut text_rect,
            DT_CENTER | DT_VCENTER | DT_SINGLELINE | DT_NOPREFIX,
        );
        let _ = SelectObject(hdc, old_font);
        if font_ok {
            let _ = DeleteObject(font.into());
        }
    }
}

unsafe extern "system" fn overload_notice_gdi_wndproc(
    hwnd: HWND,
    msg: u32,
    wp: WPARAM,
    lp: LPARAM,
) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize),
        WM_MOUSEACTIVATE => LRESULT(MA_NOACTIVATE as isize),
        WM_ERASEBKGND => LRESULT(1),
        WM_PAINT => unsafe {
            let snapshot = overload_notice_gdi_state()
                .lock()
                .ok()
                .map(|state| state.snapshot.clone())
                .unwrap_or_default();
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            if snapshot.visible {
                paint_overload_notice_gdi(hdc, &snapshot);
            }
            let _ = EndPaint(hwnd, &ps);
            LRESULT(0)
        },
        WM_DESTROY => {
            if let Ok(mut state) = overload_notice_gdi_state().try_lock() {
                if state.hwnd == hwnd.0 as isize {
                    state.hwnd = 0;
                    state.snapshot.visible = false;
                }
            }
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

unsafe fn create_overload_notice_gdi_host() -> Option<HWND> {
    unsafe {
        let instance = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok()?;
        let class: Vec<u16> = "NeoOverloadNoticeGdi\0".encode_utf16().collect();
        let title: Vec<u16> = "overload-warning\0".encode_utf16().collect();
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(overload_notice_gdi_wndproc),
            hInstance: instance.into(),
            lpszClassName: windows::core::PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wc);
        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            windows::core::PCWSTR(class.as_ptr()),
            windows::core::PCWSTR(title.as_ptr()),
            WS_POPUP,
            -32000,
            -32000,
            1,
            1,
            None,
            None,
            Some(instance.into()),
            None,
        )
        .ok()?;
        let _ = SetLayeredWindowAttributes(hwnd, COLORREF(0), 255, LWA_ALPHA);
        Some(hwnd)
    }
}

/// Show the overload-stop notice on the magnified video surface without
/// creating an eframe/WGPU viewport. The host is a tiny cached native GDI
/// window, click-through and no-activate. Its geometry is derived from the live
/// content rect, so fullscreen and windowed magnification share exactly the
/// same path and it follows overlay moves/resizes without touching rendering.
pub fn show_overload_notice_gdi(
    overlay_hwnd: isize,
    content_rect: (i32, i32, i32, i32),
    text: &str,
) {
    if overlay_hwnd == 0 || !is_window_valid(overlay_hwnd) || !is_own_window(overlay_hwnd) {
        hide_overload_notice_gdi();
        return;
    }
    let (mut cx, mut cy, mut cw, mut ch) = content_rect;
    if cw <= 0 || ch <= 0 {
        let Some((x, y, w, h)) = window_rect(overlay_hwnd) else {
            hide_overload_notice_gdi();
            return;
        };
        cx = x;
        cy = y;
        cw = w;
        ch = h;
    }
    if cw <= 8 || ch <= 8 {
        hide_overload_notice_gdi();
        return;
    }

    let dpi = unsafe { windows::Win32::UI::HiDpi::GetDpiForWindow(HWND(overlay_hwnd as *mut _)) };
    let scale = ((dpi.max(72) as f32) / 96.0).clamp(0.75, 2.5);
    let height = ((38.0 * scale).round() as i32).clamp(28, ch.saturating_sub(4).max(28));
    let pad_x = ((14.0 * scale).round() as i32).max(8);
    let icon_w = ((14.0 * scale).round() as i32).max(10);
    let gap = ((6.0 * scale).round() as i32).max(4);
    let max_width = (cw - ((12.0 * scale).round() as i32).max(8)).max(32);
    let base_font = ((15.0 * scale).round() as i32).max(10);
    let min_font = ((9.0 * scale).round() as i32).max(8);
    let mut font_px = base_font;
    let mut text_w = overload_gdi_measure_text(text, font_px);
    let available_text = (max_width - pad_x * 2 - icon_w - gap).max(16);
    if text_w > available_text {
        let fitted =
            ((font_px as f32) * (available_text as f32 / text_w.max(1) as f32)).floor() as i32;
        font_px = fitted.clamp(min_font, base_font);
        text_w = overload_gdi_measure_text(text, font_px);
    }
    let width = (pad_x * 2 + icon_w + gap + text_w)
        .min(max_width)
        .max((96.0 * scale).round() as i32)
        .min(cw.max(1));
    let x = cx + ((cw - width) / 2).max(0);
    // The native control panel/keepalive helper occupies the very top-center
    // of the overlay. Keep the warning below that stable helper band so it is
    // readable without changing any existing z-order contract. In a very small
    // windowed overlay, fall back to vertical centering rather than clipping.
    let desired_top = ((50.0 * scale).round() as i32).max(10);
    let max_top = (ch - height).max(0);
    let top_inset = if max_top >= desired_top {
        desired_top
    } else {
        max_top / 2
    };
    let y = cy + top_inset;

    let mut state = match overload_notice_gdi_state().lock() {
        Ok(state) => state,
        Err(_) => return,
    };
    if state.hwnd == 0 || !is_window_valid(state.hwnd) || !is_own_window(state.hwnd) {
        let Some(hwnd) = (unsafe { create_overload_notice_gdi_host() }) else {
            log::warn!("overload-notice-gdi: create failed");
            state.hwnd = 0;
            return;
        };
        state.hwnd = hwnd.0 as isize;
        log::info!(
            "overload-notice-gdi: created hwnd={:#x} backend=native-gdi-no-wgpu",
            state.hwnd
        );
    }

    let next = OverloadNoticeGdiSnapshot {
        x,
        y,
        width,
        height,
        font_px,
        text: text.to_owned(),
        visible: true,
    };
    let geometry_changed = state.snapshot.x != next.x
        || state.snapshot.y != next.y
        || state.snapshot.width != next.width
        || state.snapshot.height != next.height;
    let visual_changed = state.snapshot.text != next.text
        || state.snapshot.font_px != next.font_px
        || !state.snapshot.visible;
    let was_visible = state.snapshot.visible && is_window_visible(state.hwnd);
    state.snapshot = next;
    let hwnd = state.hwnd;
    drop(state);

    unsafe {
        let h = HWND(hwnd as *mut _);
        if geometry_changed || !was_visible {
            let _ = SetWindowPos(
                h,
                None,
                x,
                y,
                width,
                height,
                SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOOWNERZORDER | SWP_NOZORDER,
            );
        }
        if !was_visible {
            let _ = ShowWindow(h, SW_SHOWNOACTIVATE);
        }
        // Keep the notice above the magnified overlay but do not raise it to
        // the top of the TOPMOST band. This preserves Task Manager/other
        // external topmost windows and Neo's cursor/panel ordering.
        if !was_visible || !window_is_above(hwnd, overlay_hwnd) {
            let above = GetWindow(HWND(overlay_hwnd as *mut _), GW_HWNDPREV).unwrap_or_default();
            if !above.0.is_null() && above.0 as isize != hwnd {
                let _ = SetWindowPos(
                    h,
                    Some(above),
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE
                        | SWP_NOSIZE
                        | SWP_NOACTIVATE
                        | SWP_NOSENDCHANGING
                        | SWP_NOOWNERZORDER,
                );
            } else {
                let _ = SetWindowPos(
                    h,
                    Some(HWND_TOPMOST),
                    0,
                    0,
                    0,
                    0,
                    SWP_NOMOVE
                        | SWP_NOSIZE
                        | SWP_NOACTIVATE
                        | SWP_NOSENDCHANGING
                        | SWP_NOOWNERZORDER,
                );
            }
        }
        if visual_changed || geometry_changed || !was_visible {
            let _ = RedrawWindow(Some(h), None, None, RDW_INVALIDATE | RDW_UPDATENOW);
        }
    }
    if !was_visible {
        log::info!(
            "overload-notice-gdi: shown hwnd={hwnd:#x} rect=({x},{y} {width}x{height}) content=({cx},{cy} {cw}x{ch}) clickthrough=true"
        );
    } else if geometry_changed {
        log::debug!(
            "overload-notice-gdi: reposition rect=({x},{y} {width}x{height}) content=({cx},{cy} {cw}x{ch})"
        );
    }
}

pub fn hide_overload_notice_gdi() {
    let Ok(mut state) = overload_notice_gdi_state().lock() else {
        return;
    };
    if state.hwnd != 0 && is_window_valid(state.hwnd) && state.snapshot.visible {
        unsafe {
            let _ = ShowWindow(HWND(state.hwnd as *mut _), SW_HIDE);
        }
        log::info!("overload-notice-gdi: hidden hwnd={:#x}", state.hwnd);
    }
    state.snapshot.visible = false;
}

unsafe fn create_panel_gdi_host(width: i32, height: i32) -> Option<HWND> {
    unsafe {
        let instance = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok()?;
        let class: Vec<u16> = "NeoPanelGdiHost\0".encode_utf16().collect();
        let title: Vec<u16> = "panel\0".encode_utf16().collect();
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(panel_gdi_host_wndproc),
            hInstance: instance.into(),
            lpszClassName: windows::core::PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wc); // zero when already registered is fine.
        CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            windows::core::PCWSTR(class.as_ptr()),
            windows::core::PCWSTR(title.as_ptr()),
            WS_POPUP | WS_CLIPCHILDREN,
            0,
            0,
            width.max(1),
            height.max(1),
            None,
            None,
            Some(instance.into()),
            None,
        )
        .ok()
    }
}

/// Create (or recover) the floating control panel as a native GDI host.
///
/// v448 already made the visible panel pixels a cached GDI child and input.rs
/// already owns every panel action through the low-level Win32 hook. Keeping an
/// eframe/WGPU child viewport solely as that GDI child's parent therefore adds
/// an otherwise unrelated swapchain Present to every root-GUI repaint. This
/// host preserves the same HWND/geometry/z-order/input contract without any
/// WGPU surface, so main-GUI hover and mode switches cannot force a panel GPU
/// Present.
pub fn ensure_panel_gdi_host(width: i32, height: i32) -> Option<isize> {
    let mut host = panel_gdi_host_state().lock().ok()?;
    if *host != 0 && is_window_valid(*host) && is_own_window(*host) {
        return Some(*host);
    }
    *host = 0;
    let hwnd = unsafe { create_panel_gdi_host(width, height)? };
    *host = hwnd.0 as isize;
    apply_panel_gdi_host_round_region(*host, width.max(1), height.max(1));
    log::info!(
        "panel-gdi-host: created hwnd={:#x} size={}x{} backend=native-gdi-no-wgpu",
        *host,
        width.max(1),
        height.max(1)
    );
    Some(*host)
}

pub fn set_panel_gdi_host_visible(hwnd: isize, visible: bool) {
    if !is_panel_gdi_host(hwnd) {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let _ = ShowWindow(h, if visible { SW_SHOWNOACTIVATE } else { SW_HIDE });
    }
}

pub fn is_panel_gdi_host(hwnd: isize) -> bool {
    if hwnd == 0 {
        return false;
    }
    panel_gdi_host_state()
        .lock()
        .is_ok_and(|host| *host == hwnd && is_window_valid(hwnd) && is_own_window(hwnd))
}

unsafe fn create_panel_gdi_mirror(parent: HWND, width: i32, height: i32) -> Option<HWND> {
    unsafe {
        let instance = windows::Win32::System::LibraryLoader::GetModuleHandleW(None).ok()?;
        let class: Vec<u16> = "NeoPanelGdiMirror\0".encode_utf16().collect();
        let wc = WNDCLASSW {
            style: CS_HREDRAW | CS_VREDRAW,
            lpfnWndProc: Some(panel_gdi_mirror_wndproc),
            hInstance: instance.into(),
            lpszClassName: windows::core::PCWSTR(class.as_ptr()),
            ..Default::default()
        };
        RegisterClassW(&wc); // zero when already registered is fine.
        CreateWindowExW(
            WS_EX_NOPARENTNOTIFY,
            windows::core::PCWSTR(class.as_ptr()),
            windows::core::PCWSTR(class.as_ptr()),
            WS_CHILD | WS_CLIPSIBLINGS,
            0,
            0,
            width.max(1),
            height.max(1),
            Some(parent),
            None,
            Some(instance.into()),
            None,
        )
        .ok()
    }
}

/// Draw the floating panel into a cached CPU/GDI child surface. v465 uses a
/// native GDI top-level host instead of an eframe/WGPU panel viewport, so this
/// child is the only panel drawing surface. It paints both the ordinary bar and
/// the tiny lurk chip without creating a second GPU swapchain.
///
/// The child returns HTTRANSPARENT so the low-level Win32 input path continues
/// to target the native panel host for exact top-level ownership checks.
pub fn update_panel_gdi_mirror(
    parent_hwnd: isize,
    width: i32,
    height: i32,
    visible: bool,
    chip: bool,
    stop_text: &str,
    present_fps: f64,
    hover_slot: u8,
    screenshot_feedback: bool,
) {
    if parent_hwnd == 0 || !is_window_valid(parent_hwnd) || !is_own_window(parent_hwnd) {
        return;
    }

    let snapshot = PanelGdiMirrorSnapshot {
        parent: parent_hwnd,
        width: width.max(1),
        height: height.max(1),
        visible,
        chip,
        stop_text: stop_text.to_owned(),
        fps_tenths: (present_fps * 10.0).round() as i32,
        hover_slot,
        screenshot_feedback,
    };

    let mut created = false;
    let (child_hwnd, visual_changed, geometry_changed, visibility_changed, dirty_rect) = {
        let Ok(mut state) = panel_gdi_mirror_state().lock() else {
            return;
        };
        let current_valid =
            state.hwnd != 0 && is_window_valid(state.hwnd) && state.snapshot.parent == parent_hwnd;
        if !current_valid {
            if state.hwnd != 0 && is_window_valid(state.hwnd) {
                unsafe {
                    let _ = DestroyWindow(HWND(state.hwnd as *mut _));
                }
            }
            let Some(child) = (unsafe {
                create_panel_gdi_mirror(
                    HWND(parent_hwnd as *mut _),
                    snapshot.width,
                    snapshot.height,
                )
            }) else {
                log::warn!(
                    "panel-gdi-mirror: create failed parent={parent_hwnd:#x} size={}x{}",
                    snapshot.width,
                    snapshot.height
                );
                state.hwnd = 0;
                state.snapshot = snapshot;
                return;
            };
            state.hwnd = child.0 as isize;
            created = true;
        }

        let old = state.snapshot.clone();
        let geometry_changed =
            created || old.width != snapshot.width || old.height != snapshot.height;
        let visibility_changed = created || old.visible != snapshot.visible;
        let visual_changed = created
            || old.chip != snapshot.chip
            || old.stop_text != snapshot.stop_text
            || old.fps_tenths != snapshot.fps_tenths
            || old.hover_slot != snapshot.hover_slot
            || old.screenshot_feedback != snapshot.screenshot_feedback;
        let dirty_rect = if visual_changed && !geometry_changed && !visibility_changed {
            panel_gdi_visual_dirty_rect(&old, &snapshot)
        } else {
            None
        };
        state.snapshot = snapshot.clone();
        (
            state.hwnd,
            visual_changed,
            geometry_changed,
            visibility_changed,
            dirty_rect,
        )
    };

    if child_hwnd == 0 {
        return;
    }
    unsafe {
        let child = HWND(child_hwnd as *mut _);

        // Geometry must be committed even while the mirror is hidden. During a
        // lurk->bar restore the parent is deliberately alpha=0 until the full
        // 297x33 visual stack is ready. v526 only resized this child when it was
        // shown, so its cached snapshot could already say 297x33 while the real
        // child HWND still had the previous lurk geometry. Pre-size it
        // invisibly instead.
        if geometry_changed {
            if visible {
                let _ = SetWindowPos(
                    child,
                    Some(HWND_TOP),
                    0,
                    0,
                    snapshot.width,
                    snapshot.height,
                    SWP_NOACTIVATE | SWP_SHOWWINDOW,
                );
            } else {
                let _ = SetWindowPos(
                    child,
                    None,
                    0,
                    0,
                    snapshot.width,
                    snapshot.height,
                    SWP_NOACTIVATE | SWP_NOZORDER,
                );
            }
        } else if visible && visibility_changed {
            let _ = SetWindowPos(
                child,
                Some(HWND_TOP),
                0,
                0,
                snapshot.width,
                snapshot.height,
                SWP_NOACTIVATE | SWP_SHOWWINDOW,
            );
        }

        let full_rect = RECT {
            left: 0,
            top: 0,
            right: snapshot.width.max(1),
            bottom: snapshot.height.max(1),
        };

        if visible {
            // Ordinary FPS/hover changes keep the small-region BitBlt path.
            // Geometry/visibility transitions are different: publish one
            // complete double-buffered frame before allowing the child to be
            // observed, rather than relying on USER32's invalid-region timing.
            if visual_changed || geometry_changed || visibility_changed {
                let partial_published = if visual_changed
                    && !geometry_changed
                    && !visibility_changed
                    && dirty_rect.is_some()
                {
                    publish_panel_gdi_region(child, &snapshot, dirty_rect.unwrap())
                } else {
                    false
                };
                let full_published =
                    if !partial_published && (geometry_changed || visibility_changed) {
                        publish_panel_gdi_region(child, &snapshot, full_rect)
                    } else {
                        false
                    };
                if !partial_published && !full_published {
                    let _ = RedrawWindow(Some(child), None, None, RDW_INVALIDATE | RDW_UPDATENOW);
                }
            }
        } else {
            if visibility_changed {
                let _ = ShowWindow(child, SW_HIDE);
            }
            if geometry_changed {
                // Prime the entire hidden child now. When the parent/child are
                // atomically revealed on the next pass, every pixel already
                // exists and no pointer-driven invalidation can progressively
                // expose the bar.
                let _ = publish_panel_gdi_region(child, &snapshot, full_rect);
            }
        }
    }

    if created {
        log::info!(
            "panel-gdi-mirror: created child={child_hwnd:#x} parent={parent_hwnd:#x} size={}x{}",
            snapshot.width,
            snapshot.height
        );
    } else if (visual_changed || geometry_changed || visibility_changed)
        && crate::logging::diagnostics_enabled()
    {
        log::debug!(
            "panel-gdi-mirror-state: child={child_hwnd:#x} parent={parent_hwnd:#x} visible={} chip={} size={}x{} fps={:.1} hover_slot={} screenshot_feedback={} visual_changed={} geometry_changed={} visibility_changed={}",
            visible,
            chip,
            snapshot.width,
            snapshot.height,
            snapshot.fps_tenths as f32 / 10.0,
            hover_slot,
            screenshot_feedback,
            visual_changed,
            geometry_changed,
            visibility_changed
        );
    }
}

/// Re-publish the complete cached panel frame after the native host has become
/// visible. Hidden pre-paint remains the primary anti-flicker path; this is a
/// one-shot reveal guard for rare USER32/DWM cases where only part of the child
/// surface is exposed on the first visible composition.
///
/// Ordinary FPS/hover updates continue to use `update_panel_gdi_mirror` and its
/// small dirty-region BitBlt path.
pub fn republish_panel_gdi_mirror_full(parent_hwnd: isize) -> bool {
    if parent_hwnd == 0 || !is_window_valid(parent_hwnd) || !is_own_window(parent_hwnd) {
        return false;
    }

    let (child_hwnd, snapshot) = {
        let Ok(state) = panel_gdi_mirror_state().lock() else {
            return false;
        };
        if state.hwnd == 0
            || state.snapshot.parent != parent_hwnd
            || !state.snapshot.visible
            || !is_window_valid(state.hwnd)
        {
            return false;
        }
        (state.hwnd, state.snapshot.clone())
    };

    let full_rect = RECT {
        left: 0,
        top: 0,
        right: snapshot.width.max(1),
        bottom: snapshot.height.max(1),
    };
    let published = unsafe {
        let child = HWND(child_hwnd as *mut _);
        let ok = publish_panel_gdi_region(child, &snapshot, full_rect);
        if !ok {
            let _ = RedrawWindow(Some(child), None, None, RDW_INVALIDATE | RDW_UPDATENOW);
        }
        ok
    };

    if crate::logging::diagnostics_enabled() {
        log::debug!(
            "panel-gdi-reveal-republish: child={child_hwnd:#x} parent={parent_hwnd:#x} size={}x{} result={}",
            snapshot.width,
            snapshot.height,
            if published {
                "bitblt"
            } else {
                "redraw-fallback"
            }
        );
    }
    true
}

pub fn hide_panel_gdi_mirror() {
    set_panel_lurk_hover_active(false);
    let child_hwnd = panel_gdi_mirror_state()
        .lock()
        .ok()
        .map(|mut state| {
            state.snapshot.visible = false;
            state.hwnd
        })
        .unwrap_or(0);
    if child_hwnd != 0 && is_window_valid(child_hwnd) {
        unsafe {
            let _ = ShowWindow(HWND(child_hwnd as *mut _), SW_HIDE);
        }
    }
}

pub fn panel_gdi_mirror_status(parent_hwnd: isize) -> (bool, bool) {
    let Ok(state) = panel_gdi_mirror_state().lock() else {
        return (false, false);
    };
    if state.hwnd == 0 || state.snapshot.parent != parent_hwnd || !is_window_valid(state.hwnd) {
        return (false, false);
    }
    (true, is_window_visible(state.hwnd))
}

/// True once the cached GDI child has been created and its completed pixels are
/// logically staged for display. Unlike `IsWindowVisible`, this deliberately
/// remains true while the parent host is still physically hidden for an atomic
/// reveal; a hidden parent makes Windows report the child as not visible even
/// though its WS_VISIBLE state and paint are already committed.
pub fn panel_gdi_mirror_ready(parent_hwnd: isize) -> bool {
    let Ok(state) = panel_gdi_mirror_state().lock() else {
        return false;
    };
    state.hwnd != 0
        && state.snapshot.parent == parent_hwnd
        && state.snapshot.visible
        && is_window_valid(state.hwnd)
}

/// Give only the egui/Win32 event-loop thread a small CPU scheduling boost.
/// This does not change process priority and does not alter capture/render/GPU
/// provider scheduling policy.
pub fn promote_current_thread_for_gui() {
    unsafe {
        match SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) {
            Ok(()) => log::info!("gui-thread-priority: above-normal"),
            Err(error) => log::debug!(
                "gui-thread-priority: above-normal unavailable; continuing at normal priority: {error}"
            ),
        }
    }
}

// Hybrid-GPU selection is requested only through the executable exports
// NvOptimusEnablement and AmdPowerXpressRequestHighPerformance in main.rs.
// This portable app creates no persistent OS GPU preference.

#[derive(Clone, Copy, Debug)]
struct SourceCornerOverride {
    hwnd: isize,
    pid: u32,
    original: i32,
}

fn source_corner_override_slot() -> &'static Mutex<Option<SourceCornerOverride>> {
    static SLOT: OnceLock<Mutex<Option<SourceCornerOverride>>> = OnceLock::new();
    SLOT.get_or_init(|| Mutex::new(None))
}

pub fn window_corner_preference(hwnd: isize) -> Option<i32> {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return None;
    }
    let hwnd = HWND(hwnd as *mut _);
    let mut preference = DWM_WINDOW_CORNER_PREFERENCE(0);
    let result = unsafe {
        DwmGetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            (&mut preference as *mut DWM_WINDOW_CORNER_PREFERENCE).cast(),
            size_of::<DWM_WINDOW_CORNER_PREFERENCE>() as u32,
        )
    };
    result.ok().map(|_| preference.0)
}

fn set_window_corner_preference(hwnd: isize, preference: i32) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return false;
    }
    let hwnd = HWND(hwnd as *mut _);
    let preference = DWM_WINDOW_CORNER_PREFERENCE(preference);
    unsafe {
        DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            (&preference as *const DWM_WINDOW_CORNER_PREFERENCE).cast(),
            size_of::<DWM_WINDOW_CORNER_PREFERENCE>() as u32,
        )
        .is_ok()
    }
}

/// Restore a captured DWM corner preference only if the HWND still belongs to
/// the original source process. Used by the isolated post-exit janitor.
pub fn restore_window_corner_preference_checked(
    hwnd: isize,
    expected_pid: u32,
    preference: i32,
) -> bool {
    window_matches_pid(hwnd, expected_pid) && set_window_corner_preference(hwnd, preference)
}

/// Temporarily request square DWM corners for one verified capture target.
/// Unsupported Windows versions and windows that manage their own shape are
/// treated as a no-op. Only a change that is read back successfully is owned.
pub fn suppress_source_rounded_corners(hwnd: isize, expected_pid: u32) -> bool {
    if !window_matches_pid(hwnd, expected_pid) {
        return false;
    }

    // A single Neo process owns at most one capture session. Give a stale
    // retryable cleanup record one last chance before considering a new target.
    let pending = *source_corner_override_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(previous) = pending {
        let _ = restore_source_rounded_corners(previous.hwnd, previous.pid);
        if source_corner_override_slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_some()
        {
            log::warn!(
                "source-corners-suppress-skipped: hwnd={hwnd:#x} pid={expected_pid} reason=previous-override-pending"
            );
            return false;
        }
    }

    let Some(original) = window_corner_preference(hwnd) else {
        log::debug!(
            "source-corners-suppress: hwnd={hwnd:#x} pid={expected_pid} result=unsupported-or-unavailable"
        );
        return false;
    };

    if original == DWMWCP_DONOTROUND.0 {
        log::debug!(
            "source-corners-suppress: hwnd={hwnd:#x} pid={expected_pid} result=already-square"
        );
        return false;
    }

    if !window_matches_pid(hwnd, expected_pid) {
        return false;
    }
    if !set_window_corner_preference(hwnd, DWMWCP_DONOTROUND.0) {
        log::debug!("source-corners-suppress: hwnd={hwnd:#x} pid={expected_pid} result=set-failed");
        return false;
    }

    if !window_matches_pid(hwnd, expected_pid) {
        return false;
    }
    if window_corner_preference(hwnd) != Some(DWMWCP_DONOTROUND.0) {
        // We cannot prove ownership of the visible state, so immediately put
        // back the exact value sampled before the change and keep capturing.
        let _ = set_window_corner_preference(hwnd, original);
        log::debug!(
            "source-corners-suppress: hwnd={hwnd:#x} pid={expected_pid} result=verify-failed"
        );
        return false;
    }

    *source_corner_override_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(SourceCornerOverride {
        hwnd,
        pid: expected_pid,
        original,
    });
    log::info!(
        "source-corners-suppress: hwnd={hwnd:#x} pid={expected_pid} original={original} active=true"
    );
    true
}

/// Restore only a corner preference that this Neo process actually changed.
/// If the source changed the preference while capture was active, leave that
/// newer value alone. A failed restore remains registered so a later cleanup
/// path can retry without touching any unrelated HWND.
pub fn restore_source_rounded_corners(hwnd: isize, expected_pid: u32) -> bool {
    let owned = {
        let mut slot = source_corner_override_slot()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match *slot {
            Some(value) if value.hwnd == hwnd && value.pid == expected_pid => slot.take().unwrap(),
            Some(_) => return false,
            None => return true,
        }
    };

    if !window_matches_pid(hwnd, expected_pid) {
        log::warn!(
            "source-corners-restore-skipped: hwnd={hwnd:#x} expected_pid={expected_pid} current_pid={} reason=identity-changed",
            window_pid(hwnd)
        );
        return false;
    }

    for attempt in 1..=3 {
        if !window_matches_pid(hwnd, expected_pid) {
            log::warn!(
                "source-corners-restore-skipped: hwnd={hwnd:#x} expected_pid={expected_pid} current_pid={} reason=identity-changed",
                window_pid(hwnd)
            );
            return false;
        }

        let Some(current) = window_corner_preference(hwnd) else {
            if attempt < 3 {
                std::thread::sleep(std::time::Duration::from_millis(1));
                continue;
            }
            *source_corner_override_slot()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(owned);
            log::debug!(
                "source-corners-restore-deferred: hwnd={hwnd:#x} pid={expected_pid} reason=readback-unavailable"
            );
            return false;
        };

        if current != DWMWCP_DONOTROUND.0 {
            log::info!(
                "source-corners-restore-skipped: hwnd={hwnd:#x} pid={expected_pid} current={current} reason=source-changed-preference"
            );
            return true;
        }

        if set_window_corner_preference(hwnd, owned.original)
            && window_corner_preference(hwnd) == Some(owned.original)
        {
            log::info!(
                "source-corners-restored: hwnd={hwnd:#x} pid={expected_pid} preference={} verified=true attempt={attempt}",
                owned.original
            );
            return true;
        }

        if attempt < 3 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    *source_corner_override_slot()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(owned);
    log::warn!(
        "source-corners-restore-deferred: hwnd={hwnd:#x} pid={expected_pid} original={} reason=verify-failed",
        owned.original
    );
    false
}

/// Use a neutral dark-gray native caption without replacing the standard
/// Windows resize, minimize, maximize, or close behaviour.
pub fn set_dark_title_bar(hwnd: isize) {
    if hwnd == 0 {
        return;
    }
    let hwnd = HWND(hwnd as *mut _);
    let dark: i32 = 1;
    // DWM colors are COLORREF (0x00bbggrr).
    let caption: u32 = 0x002f_2f2f;
    let text: u32 = 0x00f2_f2f2;
    unsafe {
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_USE_IMMERSIVE_DARK_MODE,
            (&dark as *const i32).cast(),
            size_of::<i32>() as u32,
        );
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_CAPTION_COLOR,
            (&caption as *const u32).cast(),
            size_of::<u32>() as u32,
        );
        let _ = DwmSetWindowAttribute(
            hwnd,
            DWMWA_TEXT_COLOR,
            (&text as *const u32).cast(),
            size_of::<u32>() as u32,
        );
    }
}

#[inline]
fn caption_style_with_disabled_maximize(style: u32) -> u32 {
    // Keep WS_SYSMENU and WS_MINIMIZEBOX so Close and Minimize remain normal
    // Windows caption buttons. Clearing only WS_MAXIMIZEBOX leaves the square
    // Maximize button present but disabled/greyed, matching Neo's older UI.
    style & !WS_MAXIMIZEBOX.0
}

/// Keep the standard Close and Minimize glyphs, while disabling only Maximize.
pub fn disable_native_maximize_button(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let style = GetWindowLongW(h, GWL_STYLE) as u32;
        let next = caption_style_with_disabled_maximize(style);
        if next != style {
            SetWindowLongW(h, GWL_STYLE, next as i32);
            let _ = SetWindowPos(
                h,
                None,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            );
            log::info!(
                "main-gui-caption-buttons: hwnd={hwnd:#x} close=enabled minimize=enabled maximize=disabled native_titlebar=preserved"
            );
        }
    }
}

/// Return a stray maximize request (Win+Up, Snap, title-bar double click, or
/// an old saved state) to the normal placement. Returns true only when a
/// maximized state was actually corrected.
pub fn restore_if_maximized(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_maximized(hwnd) {
        return false;
    }
    unsafe {
        let _ = ShowWindow(HWND(hwnd as *mut _), SW_RESTORE);
    }
    true
}

/// Acquire the process-wide single-instance lock. Returns `true` if THIS is the
/// only/first instance; `false` if another cHiDeScaler-Neo is already running.
/// The named mutex handle is intentionally leaked so it lives for the whole
/// process lifetime (released by the OS on exit). Call once, very early.
pub fn acquire_single_instance() -> bool {
    use windows::Win32::Foundation::{ERROR_ALREADY_EXISTS, GetLastError};
    use windows::Win32::System::Threading::CreateMutexW;
    // Global\ so it is shared across sessions/elevation; a fixed unique name.
    let name: Vec<u16> = "Global\\cHiDeScaler-Neo-singleton-8f2a1c\0"
        .encode_utf16()
        .collect();
    unsafe {
        let handle = CreateMutexW(None, true, windows::core::PCWSTR(name.as_ptr()));
        let already = GetLastError() == ERROR_ALREADY_EXISTS;
        if let Ok(h) = handle {
            if already {
                let _ = CloseHandle(h);
            } else {
                SINGLE_INSTANCE_HANDLE.store(h.0 as isize, Ordering::Release);
            }
        }
        !already
    }
}

/// Release the named single-instance mutex immediately before a deliberate
/// same-EXE GPU-selection relaunch. Ordinary exits keep the historical
/// process-lifetime ownership; this narrow handoff lets the replacement process
/// start while the old, already-shut-down process waits only long enough to
/// restore the user's temporary Windows GPU preference.
pub fn release_single_instance() {
    let raw = SINGLE_INSTANCE_HANDLE.swap(0, Ordering::AcqRel);
    if raw == 0 {
        return;
    }
    unsafe {
        let _ = CloseHandle(windows::Win32::Foundation::HANDLE(raw as *mut _));
    }
    log::info!("single-instance: released for controlled relaunch");
}

/// Find the main window of ANOTHER instance (by exact title, any process that
/// is not us) and bring it to the foreground, so a second launch focuses the
/// running one instead of silently doing nothing.
pub fn activate_other_instance(title: &str) {
    struct Ctx {
        pid: u32,
        title: Vec<u16>,
        found: isize,
    }
    unsafe extern "system" fn cb(
        hwnd: HWND,
        lp: windows::Win32::Foundation::LPARAM,
    ) -> windows::core::BOOL {
        unsafe {
            let ctx = &mut *(lp.0 as *mut Ctx);
            let mut pid = 0u32;
            let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid != ctx.pid {
                let mut buf = [0u16; 64];
                let n = GetWindowTextW(hwnd, &mut buf);
                // PREFIX match: titles carry a build tag ("cHiDeScaler-Neo
                // build tag so which build is running is visible at a glance;
                // an old instance with a different tag must still be found
                let got = &buf[..n.max(0) as usize];
                if got.len() >= ctx.title.len() && got[..ctx.title.len()] == ctx.title[..] {
                    ctx.found = hwnd.0 as isize;
                    return false.into();
                }
            }
            true.into()
        }
    }
    let mut ctx = Ctx {
        pid: unsafe { GetCurrentProcessId() },
        title: title.encode_utf16().collect(),
        found: 0,
    };
    unsafe {
        let _ = EnumWindows(
            Some(cb),
            windows::Win32::Foundation::LPARAM(&mut ctx as *mut _ as isize),
        );
        if ctx.found != 0 {
            let h = HWND(ctx.found as *mut _);
            let _ = ShowWindow(h, SW_RESTORE);
            let _ = SetForegroundWindow(h);
        }
    }
}

/// Find a top-level window of THIS process by exact title.
pub fn find_own_window(title: &str) -> Option<isize> {
    struct Ctx {
        pid: u32,
        title: Vec<u16>,
        found: isize,
    }
    unsafe extern "system" fn cb(
        hwnd: HWND,
        lp: windows::Win32::Foundation::LPARAM,
    ) -> windows::core::BOOL {
        unsafe {
            let ctx = &mut *(lp.0 as *mut Ctx);
            let mut pid = 0u32;
            let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid == ctx.pid {
                let mut buf = [0u16; 64];
                let n = GetWindowTextW(hwnd, &mut buf);
                if buf[..n.max(0) as usize] == ctx.title[..] {
                    ctx.found = hwnd.0 as isize;
                    return false.into();
                }
            }
            true.into()
        }
    }
    let mut ctx = Ctx {
        pid: unsafe { GetCurrentProcessId() },
        title: title.encode_utf16().collect(),
        found: 0,
    };
    unsafe {
        let _ = EnumWindows(
            Some(cb),
            windows::Win32::Foundation::LPARAM(&mut ctx as *mut _ as isize),
        );
    }
    (ctx.found != 0).then_some(ctx.found)
}

/// Whole-window alpha (adds WS_EX_LAYERED as needed). 255 = opaque.
pub fn set_window_alpha(hwnd: isize, alpha: u8) {
    // HWND values are recycled globally. A cached child-viewport handle can
    // become another application's window after the viewport is destroyed.
    // Never mutate layered attributes unless ownership is still ours at the
    // final Win32 boundary.
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window alpha mutation rejected: hwnd={hwnd:#x} alpha={alpha} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let ex = GetWindowLongW(h, GWL_EXSTYLE) as u32;
        if ex & WS_EX_LAYERED.0 == 0 {
            SetWindowLongW(h, GWL_EXSTYLE, (ex | WS_EX_LAYERED.0) as i32);
        }
        let _ = SetLayeredWindowAttributes(
            h,
            windows::Win32::Foundation::COLORREF(0),
            alpha,
            LWA_ALPHA,
        );
    }
}

/// Present a fully opaque UI window through the normal DWM redirection path.
///
/// The floating control panel must stay physically visible while capture is
/// active (hiding its child viewport can stall presentation on some AMD
/// drivers), but it does not need WS_EX_LAYERED while it is fully opaque.
/// Keeping an opaque panel layered can make DWM briefly expose the fast-moving
/// overlay underneath when the GPU compositor is busy. Removing only the
/// layered bit avoids that extra composition path without changing visibility,
/// z-order, activation, or input routing. Transparent/lurk states continue to
/// use `set_window_alpha`, which re-adds WS_EX_LAYERED on demand.
pub fn set_window_opaque_unlayered(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window opaque mutation rejected: hwnd={hwnd:#x} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let ex = GetWindowLongW(h, GWL_EXSTYLE) as u32;
        if ex & WS_EX_LAYERED.0 != 0 {
            // Restore full opacity before leaving layered mode so there is no
            // transparent transition frame. Do not hide/show the HWND.
            let _ = SetLayeredWindowAttributes(
                h,
                windows::Win32::Foundation::COLORREF(0),
                255,
                LWA_ALPHA,
            );
            SetWindowLongW(h, GWL_EXSTYLE, (ex & !WS_EX_LAYERED.0) as i32);
        }
    }
}

/// Keep an invisible helper window from intercepting mouse input without
/// hiding its OpenGL surface. Hiding an eframe child viewport can make AMD's
/// shared WGL context block while it keeps swapping the hidden surface.
pub fn set_window_input_passthrough(hwnd: isize, passthrough: bool) {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window input mutation rejected: hwnd={hwnd:#x} passthrough={passthrough} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let ex = GetWindowLongW(h, GWL_EXSTYLE) as u32;
        let next = if passthrough {
            ex | WS_EX_TRANSPARENT.0
        } else {
            ex & !WS_EX_TRANSPARENT.0
        };
        if next != ex {
            SetWindowLongW(h, GWL_EXSTYLE, next as i32);
        }
        // SetWindowLong updates the stored style, but Windows may keep the
        // old non-client/hit-test cache until a frame change is committed.
        // More importantly, Stop can race an already-queued winit/egui
        // MousePassthrough command: at the instant Stop checks the style it can
        // already be interactive, then the stale command lands a moment later.
        // An explicit interactive restore therefore flushes the native hit-test
        // cache even when WS_EX_TRANSPARENT was already clear. This is a
        // transition-only path, never a render/frame-path cost.
        if next != ex || !passthrough {
            let _ = SetWindowPos(
                h,
                None,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            );
        }
    }
}

/// Read WS_EX_LAYERED from a verified foreign/source window.
/// Unlike `is_window_layered`, this is intentionally not limited to Neo-owned HWNDs.
pub fn source_window_layered(hwnd: isize, expected_pid: u32) -> Option<bool> {
    if !window_matches_pid(hwnd, expected_pid) {
        return None;
    }
    Some(unsafe { GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32 & WS_EX_LAYERED.0 != 0 })
}

pub fn is_window_layered(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    unsafe { GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32 & WS_EX_LAYERED.0 != 0 }
}

pub fn window_input_passthrough(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    unsafe { GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32 & WS_EX_TRANSPARENT.0 != 0 }
}

/// Move+resize without activation or z change.
pub fn set_window_rect(hwnd: isize, x: i32, y: i32, w: i32, h: i32) {
    let panel_size_changed = if is_panel_gdi_host(hwnd) {
        match window_rect(hwnd) {
            Some((_, _, old_w, old_h)) => old_w != w || old_h != h,
            None => true,
        }
    } else {
        false
    };
    unsafe {
        let _ = SetWindowPos(
            HWND(hwnd as *mut _),
            None,
            x,
            y,
            w,
            h,
            SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOZORDER,
        );
    }
    if panel_size_changed {
        apply_panel_gdi_host_round_region(hwnd, w.max(1), h.max(1));
    }
}

pub fn is_maximized(hwnd: isize) -> bool {
    unsafe { IsZoomed(HWND(hwnd as *mut _)).as_bool() }
}

/// Stable subset of Win32 presentation/chrome state used to distinguish an
/// application-owned fullscreen transition from a monitor-sized window that
/// Neo created with SetWindowPos. Z-order bits such as WS_EX_TOPMOST are
/// deliberately excluded because Neo temporarily changes those itself.
pub fn fullscreen_presentation_signature(hwnd: isize) -> Option<(u32, u32, bool)> {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return None;
    }
    unsafe {
        let style = GetWindowLongW(HWND(hwnd as *mut _), GWL_STYLE) as u32;
        let ex_style = GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32;
        let style_mask = WS_CAPTION.0
            | WS_THICKFRAME.0
            | WS_SYSMENU.0
            | WS_MINIMIZEBOX.0
            | WS_MAXIMIZEBOX.0
            | WS_MAXIMIZE.0;
        let ex_style_mask = WS_EX_DLGMODALFRAME.0
            | WS_EX_CLIENTEDGE.0
            | WS_EX_WINDOWEDGE.0
            | WS_EX_TOOLWINDOW.0
            | WS_EX_APPWINDOW.0;
        Some((
            style & style_mask,
            ex_style & ex_style_mask,
            IsZoomed(HWND(hwnd as *mut _)).as_bool(),
        ))
    }
}

/// True when a source window's outer rectangle covers its monitor.
/// Such monitor-covering sources are treated as geometry-sensitive fullscreen
/// surfaces: Neo must avoid source-side resize/Z-order mutations that could
/// change the application's own layout or presentation geometry.
pub fn is_monitor_fullscreen(hwnd: isize) -> bool {
    let Some(window) = window_rect(hwnd) else {
        return false;
    };
    is_rect_monitor_fullscreen(hwnd, window)
}

/// Test a previously snapshotted outer-window rectangle against the monitor
/// that owns `hwnd`. Capture-resolution changes can themselves make the live
/// window monitor-sized, so callers that need session-origin semantics must
/// compare the immutable start rectangle rather than the current rectangle.
pub fn is_rect_monitor_fullscreen(hwnd: isize, rect: (i32, i32, i32, i32)) -> bool {
    rect_covers_monitor(rect, monitor_rect_of(hwnd), 2)
}

fn rect_covers_monitor(
    window: (i32, i32, i32, i32),
    monitor: (i32, i32, i32, i32),
    tolerance: i32,
) -> bool {
    (window.0 - monitor.0).abs() <= tolerance
        && (window.1 - monitor.1).abs() <= tolerance
        && (window.2 - monitor.2).abs() <= tolerance
        && (window.3 - monitor.3).abs() <= tolerance
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindowPlacementSnapshot {
    flags: u32,
    show_cmd: u32,
    min_position: (i32, i32),
    max_position: (i32, i32),
    /// WINDOWPLACEMENT::rcNormalPosition stored as (left, top, right, bottom).
    normal_rect: (i32, i32, i32, i32),
}

impl WindowPlacementSnapshot {
    pub fn normal_rect_xywh(self) -> (i32, i32, i32, i32) {
        (
            self.normal_rect.0,
            self.normal_rect.1,
            self.normal_rect.2 - self.normal_rect.0,
            self.normal_rect.3 - self.normal_rect.1,
        )
    }

    pub fn janitor_raw_parts(self) -> (u32, u32, (i32, i32), (i32, i32), (i32, i32, i32, i32)) {
        (
            self.flags,
            self.show_cmd,
            self.min_position,
            self.max_position,
            self.normal_rect,
        )
    }

    pub fn from_janitor_raw_parts(
        flags: u32,
        show_cmd: u32,
        min_position: (i32, i32),
        max_position: (i32, i32),
        normal_rect: (i32, i32, i32, i32),
    ) -> Self {
        Self {
            flags,
            show_cmd,
            min_position,
            max_position,
            normal_rect,
        }
    }
}

/// Snapshot the full Win32 placement record, including the hidden normal
/// position retained while a window is maximized. GetWindowRect alone cannot
/// preserve the size to which the application should return after the user
/// later leaves the maximized state.
pub fn window_placement_snapshot(hwnd: isize) -> Option<WindowPlacementSnapshot> {
    if !is_window_valid(hwnd) {
        return None;
    }
    unsafe {
        let mut placement = WINDOWPLACEMENT::default();
        placement.length = std::mem::size_of::<WINDOWPLACEMENT>() as u32;
        if GetWindowPlacement(HWND(hwnd as *mut _), &mut placement).is_err() {
            return None;
        }
        Some(WindowPlacementSnapshot {
            flags: placement.flags.0,
            show_cmd: placement.showCmd,
            min_position: (placement.ptMinPosition.x, placement.ptMinPosition.y),
            max_position: (placement.ptMaxPosition.x, placement.ptMaxPosition.y),
            normal_rect: (
                placement.rcNormalPosition.left,
                placement.rcNormalPosition.top,
                placement.rcNormalPosition.right,
                placement.rcNormalPosition.bottom,
            ),
        })
    }
}

/// Restore the exact Win32 placement record captured at Start. This is the
/// authoritative path for a source that was already maximized: it restores
/// rcNormalPosition without temporarily converting the maximized outer frame
/// into the application's future normal-window bounds.
pub fn restore_window_placement_snapshot(hwnd: isize, snapshot: WindowPlacementSnapshot) -> bool {
    if !is_window_valid(hwnd) {
        return false;
    }
    let placement = WINDOWPLACEMENT {
        length: std::mem::size_of::<WINDOWPLACEMENT>() as u32,
        flags: WINDOWPLACEMENT_FLAGS(snapshot.flags),
        showCmd: snapshot.show_cmd,
        ptMinPosition: POINT {
            x: snapshot.min_position.0,
            y: snapshot.min_position.1,
        },
        ptMaxPosition: POINT {
            x: snapshot.max_position.0,
            y: snapshot.max_position.1,
        },
        rcNormalPosition: RECT {
            left: snapshot.normal_rect.0,
            top: snapshot.normal_rect.1,
            right: snapshot.normal_rect.2,
            bottom: snapshot.normal_rect.3,
        },
    };
    let set_ok = unsafe { SetWindowPlacement(HWND(hwnd as *mut _), &placement).is_ok() };
    if !set_ok {
        return false;
    }
    window_placement_snapshot(hwnd).is_some_and(|after| after.normal_rect == snapshot.normal_rect)
}

/// Restore the immutable session origin. Maximized windows require their full
/// WINDOWPLACEMENT snapshot so their pre-maximize restore size is preserved.
pub fn restore_window_origin(
    hwnd: isize,
    rect: Option<(i32, i32, i32, i32)>,
    was_maximized: bool,
    placement: Option<WindowPlacementSnapshot>,
) -> bool {
    if was_maximized {
        if let Some(snapshot) = placement {
            return restore_window_placement_snapshot(hwnd, snapshot);
        }
    }
    rect.is_some_and(|rect| restore_window_rect(hwnd, rect, was_maximized))
}

/// Restore an exact outer-window rectangle. This remains the normal-window
/// path. Maximized session origins should use restore_window_origin() so the
/// hidden pre-maximize WINDOWPLACEMENT is not overwritten.
pub fn restore_window_rect(hwnd: isize, rect: (i32, i32, i32, i32), was_maximized: bool) -> bool {
    if !is_window_valid(hwnd) || rect.2 <= 0 || rect.3 <= 0 {
        return false;
    }
    // If placement metadata was unavailable, at least avoid damaging an
    // already-correct maximized window by restoring it to its own maximized
    // outer rectangle and thereby overwriting its hidden normal bounds.
    if was_maximized && is_maximized(hwnd) && window_rect(hwnd) == Some(rect) {
        return true;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if IsIconic(h).as_bool() || IsZoomed(h).as_bool() {
            let _ = ShowWindow(h, SW_RESTORE);
        }
    }
    set_window_rect(hwnd, rect.0, rect.1, rect.2, rect.3);
    if was_maximized {
        unsafe {
            let _ = ShowWindow(HWND(hwnd as *mut _), SW_MAXIMIZE);
        }
    }
    window_rect(hwnd) == Some(rect) && is_maximized(hwnd) == was_maximized
}

pub fn any_mouse_button_down() -> bool {
    unsafe {
        [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON]
            .into_iter()
            .any(|vk| GetAsyncKeyState(vk.0 as i32) < 0)
    }
}

/// True only while the physical left mouse button is held.
///
/// Kept separate from `any_mouse_button_down()` because client-surface window
/// dragging (Chromium PIP/custom frames) must never be inferred from a right
/// click or middle-button gesture.
pub fn left_mouse_button_down() -> bool {
    unsafe { GetAsyncKeyState(VK_LBUTTON.0 as i32) < 0 }
}

fn fit_axis_to_bounds(pos: i32, size: i32, bounds_pos: i32, bounds_size: i32) -> i32 {
    if size >= bounds_size {
        bounds_pos
    } else {
        pos.clamp(bounds_pos, bounds_pos + bounds_size - size)
    }
}

/// Resize the client/content area of a top-level source window. The outer
/// frame is adjusted from the current window style so a requested 1280x720
/// means the captured client area, not the whole decorated rectangle.
pub fn resize_client_area(hwnd: isize, client_w: u32, client_h: u32) -> bool {
    let client_w = client_w.clamp(160, 7680) as i32;
    let client_h = client_h.clamp(120, 4320) as i32;
    let monitor = monitor_rect_of(hwnd);
    // Cursor mapping confines the real cursor to this client rect. If the
    // client itself cannot fit on the source monitor, part of that coordinate
    // space is physically unreachable and Windows clamps SetCursorPos at the
    // desktop edge. Reject that geometry instead of mapping the clamped point
    // back into the topmost GUI.
    if client_w > monitor.2 || client_h > monitor.3 {
        return false;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if !IsWindow(Some(h)).as_bool() {
            return false;
        }
        let Some((x, y, _, _)) = window_rect(hwnd) else {
            return false;
        };
        let Some((client_x, client_y, _, _)) = client_rect_on_screen(hwnd) else {
            return false;
        };
        let style = GetWindowLongW(h, GWL_STYLE) as u32;
        let ex_style = GetWindowLongW(h, GWL_EXSTYLE) as u32;
        let has_menu = !GetMenu(h).0.is_null();
        let mut r = RECT {
            left: 0,
            top: 0,
            right: client_w,
            bottom: client_h,
        };
        if AdjustWindowRectEx(
            &mut r,
            WINDOW_STYLE(style),
            has_menu,
            WINDOW_EX_STYLE(ex_style),
        )
        .is_err()
        {
            return false;
        }
        let outer_w = (r.right - r.left).max(client_w);
        let outer_h = (r.bottom - r.top).max(client_h);
        let desired_client_x = fit_axis_to_bounds(client_x, client_w, monitor.0, monitor.2);
        let desired_client_y = fit_axis_to_bounds(client_y, client_h, monitor.1, monitor.3);
        let outer_x = x + desired_client_x - client_x;
        let outer_y = y + desired_client_y - client_y;
        if SetWindowPos(
            h,
            None,
            outer_x,
            outer_y,
            outer_w,
            outer_h,
            SWP_NOACTIVATE | SWP_NOZORDER,
        )
        .is_err()
        {
            return false;
        }
    }

    // Chromium PIP and other custom-frame windows can report a normal Win32
    // style while drawing their frame entirely inside the client area. In that
    // case AdjustWindowRectEx overestimates the required outer size. Correct
    // from the size Windows actually produced. This also detects application
    // imposed maximum sizes instead of silently capturing the wrong geometry.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(1000);
    let stable_duration = std::time::Duration::from_millis(160);
    let mut exact_since = None;
    let mut last_requested = window_rect(hwnd);
    loop {
        let Some((client_x, client_y, actual_w, actual_h)) = client_rect_on_screen(hwnd) else {
            return false;
        };
        let now = std::time::Instant::now();
        let desired_client_x = fit_axis_to_bounds(client_x, actual_w, monitor.0, monitor.2);
        let desired_client_y = fit_axis_to_bounds(client_y, actual_h, monitor.1, monitor.3);
        let position_exact = client_x == desired_client_x && client_y == desired_client_y;
        if actual_w == client_w && actual_h == client_h && position_exact {
            let since = exact_since.get_or_insert(now);
            if now.saturating_duration_since(*since) >= stable_duration {
                return true;
            }
        } else {
            exact_since = None;
            let Some((x, y, outer_w, outer_h)) = window_rect(hwnd) else {
                return false;
            };
            let corrected_w = (outer_w + client_w - actual_w).clamp(160, 7680);
            let corrected_h = (outer_h + client_h - actual_h).clamp(120, 4320);
            let corrected_x = x + desired_client_x - client_x;
            let corrected_y = y + desired_client_y - client_y;
            let corrected = (corrected_x, corrected_y, corrected_w, corrected_h);
            if last_requested != Some(corrected) {
                set_window_rect(hwnd, corrected_x, corrected_y, corrected_w, corrected_h);
                last_requested = Some(corrected);
            }
        }
        if now >= deadline {
            return false;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}

/// Ask a static/composited source (notably Chromium PIP) to produce a frame
/// after its client geometry changed. This is deliberately non-activating and
/// does not synthesize mouse input.
pub fn request_window_repaint(hwnd: isize) {
    if hwnd == 0 {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if IsWindow(Some(h)).as_bool() {
            let _ = RedrawWindow(
                Some(h),
                None,
                None,
                RDW_INVALIDATE | RDW_FRAME | RDW_ALLCHILDREN | RDW_UPDATENOW,
            );
        }
    }
}

pub fn preferred_ui_language_tags() -> Vec<String> {
    unsafe extern "system" {
        fn GetUserPreferredUILanguages(
            flags: u32,
            count: *mut u32,
            buffer: *mut u16,
            buffer_len: *mut u32,
        ) -> i32;
        fn GetUserDefaultUILanguage() -> u16;
    }
    const MUI_LANGUAGE_NAME: u32 = 0x8;
    let mut count = 0u32;
    let mut len = 0u32;
    unsafe {
        let _ = GetUserPreferredUILanguages(
            MUI_LANGUAGE_NAME,
            &mut count,
            std::ptr::null_mut(),
            &mut len,
        );
    }
    if len > 1 {
        let mut buffer = vec![0u16; len as usize];
        let ok = unsafe {
            GetUserPreferredUILanguages(
                MUI_LANGUAGE_NAME,
                &mut count,
                buffer.as_mut_ptr(),
                &mut len,
            )
        } != 0;
        if ok {
            let tags: Vec<String> = buffer
                .split(|u| *u == 0)
                .filter(|part| !part.is_empty())
                .map(String::from_utf16_lossy)
                .collect();
            if !tags.is_empty() {
                return tags;
            }
        }
    }
    // Last-resort mapping when the preferred-language API is unavailable.
    let primary = unsafe { GetUserDefaultUILanguage() } & 0x03ff;
    let tag = match primary {
        0x0011 => "ja-JP",
        0x0009 => "en-US",
        0x0004 => "zh-CN",
        0x0012 => "ko-KR",
        0x0016 => "pt-BR",
        0x000a => "es",
        0x000c => "fr-FR",
        0x0007 => "de-DE",
        _ => "en-US",
    };
    vec![tag.to_owned()]
}

pub fn system_ui_language_is_japanese() -> bool {
    preferred_ui_language_tags()
        .first()
        .is_some_and(|tag| tag.to_ascii_lowercase().starts_with("ja"))
}

pub fn cursor_pos() -> (i32, i32) {
    unsafe {
        let mut p = POINT::default();
        let _ = GetCursorPos(&mut p);
        (p.x, p.y)
    }
}

/// Authoritative interactive top-level native window at an arbitrary SCREEN
/// point. WindowFromPoint is used first; if it lands on a WS_EX_TRANSPARENT
/// helper walks downward in real Z-order until the first visible
/// interactive window containing the point. The function is read-only.
pub fn direct_top_level_window_at_point(x: i32, y: i32) -> isize {
    unsafe {
        let point = POINT { x, y };
        let hit = WindowFromPoint(point);
        let mut scan_after = None;
        if !hit.0.is_null() {
            let root = GetAncestor(hit, GA_ROOT);
            let root = if root.0.is_null() { hit } else { root };
            let ex = GetWindowLongW(root, GWL_EXSTYLE) as u32;
            if ex & WS_EX_TRANSPARENT.0 == 0 {
                return root.0 as isize;
            }
            scan_after = Some(root);
        }

        // WindowFromPoint can still report a topmost WS_EX_TRANSPARENT helper
        // (overlay/cursor sprite) even though it must not own the pointer. Walk
        // downward from that exact helper rather than rescanning windows above
        // it, and choose the first visible interactive top-level window that
        // physically contains the point. No Z-order mutation occurs here.
        let mut hwnd = match scan_after {
            Some(root) => GetWindow(root, GW_HWNDNEXT).unwrap_or_default(),
            None => GetTopWindow(None).unwrap_or_default(),
        };
        let mut guard = 0usize;
        while !hwnd.0.is_null() && guard < 4096 {
            if IsWindowVisible(hwnd).as_bool() && !IsIconic(hwnd).as_bool() {
                let ex = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
                if ex & WS_EX_TRANSPARENT.0 == 0 {
                    let mut rect = RECT::default();
                    if GetWindowRect(hwnd, &mut rect).is_ok()
                        && x >= rect.left
                        && x < rect.right
                        && y >= rect.top
                        && y < rect.bottom
                    {
                        return hwnd.0 as isize;
                    }
                }
            }
            hwnd = GetWindow(hwnd, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }
        0
    }
}

pub fn point_is_inside_window(hwnd: isize, x: i32, y: i32) -> bool {
    if hwnd == 0 {
        return false;
    }
    unsafe {
        let mut rect = RECT::default();
        GetWindowRect(HWND(hwnd as *mut _), &mut rect).is_ok()
            && x >= rect.left
            && x < rect.right
            && y >= rect.top
            && y < rect.bottom
    }
}

/// Top-level native window visually under a screen-space point. Interactive
/// Neo GUI/panel windows remain authoritative, while Neo's transparent overlay
/// and cursor sprite are skipped so a topmost external window such as Task
/// Manager can receive the native cursor before it becomes foreground.
pub fn external_top_level_window_at_point(x: i32, y: i32) -> isize {
    unsafe {
        let own_pid = GetCurrentProcessId();
        let point = POINT { x, y };
        let hit = WindowFromPoint(point);
        if !hit.0.is_null() {
            let root = GetAncestor(hit, GA_ROOT);
            let root = if root.0.is_null() { hit } else { root };
            let mut pid = 0u32;
            let _ = GetWindowThreadProcessId(root, Some(&mut pid));
            if pid != own_pid {
                return root.0 as isize;
            }
            // A real Neo GUI or control-panel window must keep normal native
            // ownership. Only click-through helper windows are skipped.
            let ex = GetWindowLongW(root, GWL_EXSTYLE) as u32;
            if ex & WS_EX_TRANSPARENT.0 == 0 {
                return root.0 as isize;
            }
        }

        // WindowFromPoint may still report Neo's transparent topmost helper.
        // Walk top-level z-order and find the first external rectangle under
        // the point; engine-side z-order/PID checks reject windows behind the
        // overlay and source-owned popups.
        let mut hwnd = GetTopWindow(None).unwrap_or_default();
        let mut guard = 0usize;
        while !hwnd.0.is_null() && guard < 4096 {
            let mut pid = 0u32;
            let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
            if pid != own_pid && IsWindowVisible(hwnd).as_bool() && !IsIconic(hwnd).as_bool() {
                let mut rect = RECT::default();
                if GetWindowRect(hwnd, &mut rect).is_ok()
                    && x >= rect.left
                    && x < rect.right
                    && y >= rect.top
                    && y < rect.bottom
                {
                    return hwnd.0 as isize;
                }
            }
            hwnd = GetWindow(hwnd, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }
        0
    }
}

/// Visible native windows that are genuinely above Neo's magnified overlay.
///
/// The previous cursor handoff searched behind Neo's transparent helper at a
/// single sampled cursor point. The virtual cursor could move between that
/// sample and the actual handoff, assigning ownership to Task Manager even
/// though the pointer was already back over the magnified source. Instead,
/// expose stable screen rectangles for the existing input system's no-engage
/// logic. Only windows encountered before the overlay in real Z-order are
/// returned, so windows hidden behind the magnified image can never pull the
/// cursor away.
pub fn external_window_rects_above_overlay(
    overlay_hwnd: isize,
    source_hwnd: isize,
) -> Vec<(i32, i32, i32, i32, isize)> {
    if overlay_hwnd == 0 {
        return Vec::new();
    }

    unsafe {
        let own_pid = GetCurrentProcessId();
        let source_pid = window_pid(source_hwnd);
        let mut out = Vec::new();
        let mut hwnd = GetTopWindow(None).unwrap_or_default();
        let mut guard = 0usize;

        while !hwnd.0.is_null() && guard < 4096 {
            let raw = hwnd.0 as isize;
            if raw == overlay_hwnd {
                break;
            }

            let mut pid = 0u32;
            let _ = GetWindowThreadProcessId(hwnd, Some(&mut pid));
            let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
            let interactive = ex_style & WS_EX_TRANSPARENT.0 == 0;
            if pid != own_pid
                && pid != source_pid
                && interactive
                && IsWindowVisible(hwnd).as_bool()
                && !IsIconic(hwnd).as_bool()
                && !is_cloaked(raw)
                && !is_system_window(raw)
            {
                let mut rect = RECT::default();
                if GetWindowRect(hwnd, &mut rect).is_ok() {
                    let width = rect.right - rect.left;
                    let height = rect.bottom - rect.top;
                    if width > 1 && height > 1 {
                        out.push((rect.left, rect.top, width, height, raw));
                    }
                }
            }

            hwnd = GetWindow(hwnd, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }

        out
    }
}

/// Repair the global z-order boundary of Neo's magnified overlay.
///
/// `WS_EX_TOPMOST` is a style bit, not a sufficient proof that the HWND is
/// currently above every ordinary foreign window in USER32's live z-list. A
/// hidden/re-shown overlay or another top-level z-order transaction can leave
/// the style intact while an ordinary window is physically ahead of it. The
/// owned GUI/panel normalizer cannot detect that state because it intentionally
/// compares only Neo siblings.
///
/// This helper is deliberately conditional: it scans the real top-level z-list
/// and mutates nothing unless a visible, overlapping, non-TOPMOST foreign
/// window is actually above the overlay. When repair is needed, the overlay is
/// recommitted into the TOPMOST band and then placed immediately below the
/// lowest meaningful TOPMOST window that was already above it. Thus genuine
/// always-on-top windows remain above Neo, while ordinary applications cannot
/// cover the magnified image. `source_hwnd` is excluded from the preserved
/// anchor set because the selected source is intentionally kept below Neo.
pub fn repair_overlay_zorder_boundary(overlay_hwnd: isize, source_hwnd: isize) -> bool {
    if overlay_hwnd == 0
        || !is_window_valid(overlay_hwnd)
        || !is_own_window(overlay_hwnd)
        || !is_window_visible(overlay_hwnd)
    {
        return false;
    }

    let Some((ox, oy, ow, oh)) = window_rect(overlay_hwnd) else {
        return false;
    };
    if ow <= 1 || oh <= 1 {
        return false;
    }
    let oright = ox.saturating_add(ow);
    let obottom = oy.saturating_add(oh);

    unsafe {
        let own_pid = GetCurrentProcessId();
        let mut lowest_topmost_anchor = 0isize;
        let mut illegal_ordinary = 0isize;
        let mut illegal_pid = 0u32;
        let mut found_overlay = false;
        let mut hwnd = GetTopWindow(None).unwrap_or_default();
        let mut guard = 0usize;

        while !hwnd.0.is_null() && guard < 4096 {
            let raw = hwnd.0 as isize;
            if raw == overlay_hwnd {
                found_overlay = true;
                break;
            }

            if IsWindowVisible(hwnd).as_bool() && !IsIconic(hwnd).as_bool() && !is_cloaked(raw) {
                let mut rect = RECT::default();
                if GetWindowRect(hwnd, &mut rect).is_ok() {
                    let width = rect.right - rect.left;
                    let height = rect.bottom - rect.top;
                    if width > 1 && height > 1 {
                        if is_topmost(raw) {
                            // Preserve every meaningful TOPMOST window already
                            // above Neo, including Neo's own GUI/panel/cursor
                            // and external always-on-top tools. The selected
                            // source is the sole exception: its contract is to
                            // remain immediately below the magnified overlay.
                            if raw != source_hwnd {
                                lowest_topmost_anchor = raw;
                            }
                        } else if illegal_ordinary == 0 {
                            let pid = window_pid(raw);
                            let ex_style = GetWindowLongW(hwnd, GWL_EXSTYLE) as u32;
                            let interactive = ex_style & WS_EX_TRANSPARENT.0 == 0;
                            let overlaps_overlay = rect.left < oright
                                && rect.right > ox
                                && rect.top < obottom
                                && rect.bottom > oy;
                            if pid != own_pid
                                && interactive
                                && overlaps_overlay
                                && !is_system_window(raw)
                            {
                                illegal_ordinary = raw;
                                illegal_pid = pid;
                            }
                        }
                    }
                }
            }

            hwnd = GetWindow(hwnd, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }

        if !found_overlay || illegal_ordinary == 0 {
            return false;
        }

        let flags =
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOOWNERZORDER;

        // First force USER32 to recommit the HWND into the actual TOPMOST band.
        // Do not trust the already-set WS_EX_TOPMOST style as proof of position.
        let first_ok = SetWindowPos(
            HWND(overlay_hwnd as *mut _),
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            flags,
        )
        .is_ok();

        // Keep all meaningful TOPMOST windows which were already ahead of Neo
        // ahead of it. No DwmFlush is issued between the two operations so DWM
        // can consume the repair as one final stack rather than expose an
        // intermediate overlay-at-absolute-front frame.
        let anchor_ok = if lowest_topmost_anchor != 0
            && lowest_topmost_anchor != overlay_hwnd
            && is_window_valid(lowest_topmost_anchor)
        {
            SetWindowPos(
                HWND(overlay_hwnd as *mut _),
                Some(HWND(lowest_topmost_anchor as *mut _)),
                0,
                0,
                0,
                0,
                flags,
            )
            .is_ok()
        } else {
            true
        };

        let boundary_ok = window_is_above(overlay_hwnd, illegal_ordinary);
        if boundary_ok {
            log::info!(
                "overlay-zorder-boundary-repair: overlay={overlay_hwnd:#x} illegal={illegal_ordinary:#x} illegal_pid={illegal_pid} anchor={lowest_topmost_anchor:#x} action=topmost-recommit+preserve-anchor first_ok={first_ok} anchor_ok={anchor_ok}"
            );
        } else {
            log::warn!(
                "overlay-zorder-boundary-repair-incomplete: overlay={overlay_hwnd:#x} illegal={illegal_ordinary:#x} illegal_pid={illegal_pid} anchor={lowest_topmost_anchor:#x} first_ok={first_ok} anchor_ok={anchor_ok}"
            );
        }
        true
    }
}

/// Current QPC time converted to 100ns ticks, matching WinRT SystemRelativeTime.
pub fn qpc_time_100ns() -> Option<i64> {
    let mut counter = 0i64;
    let mut frequency = 0i64;
    unsafe {
        QueryPerformanceCounter(&mut counter).ok()?;
        QueryPerformanceFrequency(&mut frequency).ok()?;
    }
    if frequency <= 0 {
        return None;
    }
    Some(((counter as i128 * 10_000_000i128) / frequency as i128) as i64)
}

pub fn foreground_window() -> isize {
    unsafe { GetForegroundWindow().0 as isize }
}

pub fn activate_window(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if IsIconic(h).as_bool() {
            let _ = ShowWindow(h, SW_RESTORE);
        }
        let _ = SetForegroundWindow(h);
    }
}

/// Route an explicit application Exit through the exact same native close path
/// as clicking the root window's caption X. The root WNDPROC receives WM_CLOSE,
/// performs its immediate visual hide, then forwards the message unchanged to
/// eframe so the normal close_requested -> on_exit cleanup sequence owns teardown.
pub fn request_main_gui_close(hwnd: isize) -> bool {
    MAIN_GUI_EXPLICIT_EXITING.store(true, Ordering::Release);
    MAIN_GUI_BACKGROUND_WAKE_ACTIVE.store(false, Ordering::Release);
    PANEL_ACTION_BACKGROUND_WAKE.store(0, Ordering::Release);
    set_panel_lurk_hover_active(false);
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    unsafe { PostMessageW(Some(HWND(hwnd as *mut _)), WM_CLOSE, WPARAM(0), LPARAM(0)).is_ok() }
}

pub fn main_gui_explicit_exiting() -> bool {
    MAIN_GUI_EXPLICIT_EXITING.load(Ordering::Acquire)
}

pub fn hide_own_window(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    MAIN_GUI_BACKGROUND_INTENT.store(2, Ordering::Release);
    unsafe { ShowWindow(HWND(hwnd as *mut _), SW_HIDE).as_bool() }
}

pub fn restore_own_window(hwnd: isize) -> bool {
    if MAIN_GUI_EXPLICIT_EXITING.load(Ordering::Acquire) {
        return false;
    }
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    MAIN_GUI_BACKGROUND_WAKE_ACTIVE.store(false, Ordering::Release);
    MAIN_GUI_BACKGROUND_INTENT.store(0, Ordering::Release);
    unsafe {
        let window = HWND(hwnd as *mut _);
        let _ = ShowWindow(
            window,
            if IsIconic(window).as_bool() {
                SW_RESTORE
            } else {
                SW_SHOW
            },
        );
        let _ = SetForegroundWindow(window);
    }
    true
}

pub fn minimize_own_window(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    MAIN_GUI_BACKGROUND_INTENT.store(1, Ordering::Release);
    unsafe { ShowWindow(HWND(hwnd as *mut _), SW_MINIMIZE).as_bool() }
}

/// Let eframe drain a global-hotkey command while its root window is hidden or
/// minimized, without presenting that GUI to the user. The caller restores the
/// previous background state immediately after dispatch on the GUI thread.
pub fn wake_background_gui(hwnd: isize) -> bool {
    if MAIN_GUI_EXPLICIT_EXITING.load(Ordering::Acquire) {
        return false;
    }
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    MAIN_GUI_BACKGROUND_WAKE_ACTIVE.store(true, Ordering::Release);
    MAIN_GUI_BACKGROUND_WAKE_IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
    unsafe {
        let window = HWND(hwnd as *mut _);
        // Cloak both before and after ShowWindow. Some Windows/DWM paths clear
        // a pre-existing cloak while restoring an iconic window; the second
        // commit keeps the temporary event-pump root genuinely invisible.
        let _ = set_window_cloaked(hwnd, true);
        let _ = ShowWindow(window, SW_SHOWNOACTIVATE);
        let _ = set_window_cloaked(hwnd, true);
        let _ = RedrawWindow(Some(window), None, None, RDW_INVALIDATE | RDW_UPDATENOW);
        let _ = PostMessageW(Some(window), WM_PAINT, WPARAM(0), LPARAM(0));
    }
    MAIN_GUI_BACKGROUND_WAKE_IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
    true
}

pub fn window_title(hwnd: isize) -> String {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let mut buf = [0u16; 512];
        let n = GetWindowTextW(h, &mut buf);
        String::from_utf16_lossy(&buf[..n.max(0) as usize])
    }
}

/// Width of the native minimize/maximize/close cluster in physical pixels.
pub fn caption_system_button_cluster_width() -> i32 {
    unsafe {
        let single = windows::Win32::UI::WindowsAndMessaging::GetSystemMetrics(
            windows::Win32::UI::WindowsAndMessaging::SM_CXSIZE,
        )
        .max(1);
        single.saturating_mul(3)
    }
}

/// HWND values can be recycled. Pair them with the process selected at Start
/// before mutating any foreign window state.
pub fn window_matches_pid(hwnd: isize, expected_pid: u32) -> bool {
    expected_pid != 0 && is_window_valid(hwnd) && window_pid(hwnd) == expected_pid
}

pub fn window_pid(hwnd: isize) -> u32 {
    unsafe {
        let mut pid = 0u32;
        let _ = GetWindowThreadProcessId(HWND(hwnd as *mut _), Some(&mut pid));
        pid
    }
}

pub fn is_browser_window(hwnd: isize) -> bool {
    use windows::Win32::System::Threading::{
        OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
        QueryFullProcessImageNameW,
    };
    unsafe {
        let Ok(process) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, window_pid(hwnd))
        else {
            return false;
        };
        let mut path = [0u16; 1024];
        let mut len = path.len() as u32;
        let ok = QueryFullProcessImageNameW(
            process,
            PROCESS_NAME_WIN32,
            windows::core::PWSTR(path.as_mut_ptr()),
            &mut len,
        )
        .is_ok();
        let _ = windows::Win32::Foundation::CloseHandle(process);
        if !ok {
            return false;
        }
        let name = String::from_utf16_lossy(&path[..len as usize]).to_ascii_lowercase();
        [
            "chrome.exe",
            "msedge.exe",
            "firefox.exe",
            "brave.exe",
            "opera.exe",
            "vivaldi.exe",
        ]
        .iter()
        .any(|exe| name.ends_with(exe))
    }
}

pub fn is_own_window(hwnd: isize) -> bool {
    window_pid(hwnd) == unsafe { GetCurrentProcessId() }
}

pub fn is_window_valid(hwnd: isize) -> bool {
    unsafe { IsWindow(Some(HWND(hwnd as *mut _))).as_bool() }
}

pub fn is_window_visible(hwnd: isize) -> bool {
    unsafe { IsWindowVisible(HWND(hwnd as *mut _)).as_bool() }
}

pub fn is_minimized(hwnd: isize) -> bool {
    unsafe { IsIconic(HWND(hwnd as *mut _)).as_bool() }
}

/// Client rect of `hwnd` in screen coordinates (content without title bar).
pub fn client_rect_on_screen(hwnd: isize) -> Option<(i32, i32, i32, i32)> {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let mut r = RECT::default();
        if GetClientRect(h, &mut r).is_err() {
            return None;
        }
        let mut tl = windows::Win32::Foundation::POINT {
            x: r.left,
            y: r.top,
        };
        if !ClientToScreen(h, &mut tl).as_bool() {
            return None;
        }
        let w = r.right - r.left;
        let hgt = r.bottom - r.top;
        if w <= 0 || hgt <= 0 {
            return None;
        }
        Some((tl.x, tl.y, w, hgt))
    }
}

/// DWM extended frame bounds (true visible rect, no drop-shadow slack).
pub fn extended_frame_bounds(hwnd: isize) -> Option<(i32, i32, i32, i32)> {
    unsafe {
        let mut r = RECT::default();
        if DwmGetWindowAttribute(
            HWND(hwnd as *mut _),
            DWMWA_EXTENDED_FRAME_BOUNDS,
            &mut r as *mut _ as *mut _,
            std::mem::size_of::<RECT>() as u32,
        )
        .is_ok()
        {
            Some((r.left, r.top, r.right - r.left, r.bottom - r.top))
        } else {
            None
        }
    }
}

/// Visually hide a window (alpha=1 layered) while WGC still captures it at
/// full brightness and it still receives input. Returns the previous layered
/// state on success so it can be restored exactly.
pub fn hide_window_visual(hwnd: isize) -> Option<bool> {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let ex = GetWindowLongW(h, GWL_EXSTYLE) as u32;
        let was_layered = ex & WS_EX_LAYERED.0 != 0;
        if !was_layered {
            SetWindowLongW(h, GWL_EXSTYLE, (ex | WS_EX_LAYERED.0) as i32);
        }
        if SetLayeredWindowAttributes(h, windows::Win32::Foundation::COLORREF(0), 1, LWA_ALPHA)
            .is_err()
        {
            if !was_layered {
                SetWindowLongW(h, GWL_EXSTYLE, ex as i32);
            }
            None
        } else {
            Some(was_layered)
        }
    }
}

pub fn show_window_visual(hwnd: isize, was_layered: bool) {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let _ =
            SetLayeredWindowAttributes(h, windows::Win32::Foundation::COLORREF(0), 255, LWA_ALPHA);
        if !was_layered {
            let ex = GetWindowLongW(h, GWL_EXSTYLE) as u32;
            SetWindowLongW(h, GWL_EXSTYLE, (ex & !WS_EX_LAYERED.0) as i32);
        }
    }
}

pub fn is_topmost(hwnd: isize) -> bool {
    unsafe { (GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32) & WS_EX_TOPMOST.0 != 0 }
}

/// Whether the process owning `pid` runs elevated (UIPI blocks our clicks if
/// the source is elevated and we are not).
pub fn process_elevated(pid: u32) -> Option<bool> {
    use windows::Win32::Security::{
        GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation,
    };
    use windows::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};
    unsafe {
        let proc = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut token = windows::Win32::Foundation::HANDLE::default();
        let r = windows::Win32::System::Threading::OpenProcessToken(proc, TOKEN_QUERY, &mut token);
        let _ = windows::Win32::Foundation::CloseHandle(proc);
        r.ok()?;
        let mut elev = TOKEN_ELEVATION::default();
        let mut ret = 0u32;
        let r = GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elev as *mut _ as *mut _),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut ret,
        );
        let _ = windows::Win32::Foundation::CloseHandle(token);
        r.ok()?;
        Some(elev.TokenIsElevated != 0)
    }
}

pub fn own_process_elevated() -> bool {
    process_elevated(unsafe { GetCurrentProcessId() }).unwrap_or(false)
}

pub fn offset_window(hwnd: isize, dx: i32, dy: i32) {
    if let Some((x, y, _, _)) = window_rect(hwnd) {
        unsafe {
            // NOZORDER: raising the overlay here would push it above the
            // control panel / GUI and steal their clicks
            let _ = SetWindowPos(
                HWND(hwnd as *mut _),
                None,
                x + dx,
                y + dy,
                0,
                0,
                SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOZORDER,
            );
        }
    }
}

/// True when `a` is above `b` in the current Z-order.
pub fn window_is_above(a: isize, b: isize) -> bool {
    if a == 0 || b == 0 || a == b {
        return true;
    }
    unsafe {
        let mut h = GetWindow(HWND(a as *mut _), GW_HWNDNEXT).unwrap_or_default();
        let mut guard = 0;
        while !h.0.is_null() && guard < 4000 {
            if h.0 as isize == b {
                return true;
            }
            h = GetWindow(h, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }
    }
    false
}

/// Raise a top-level helper window without activating it.
pub fn raise_topmost(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window raise-topmost rejected: hwnd={hwnd:#x} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    unsafe {
        let flags =
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOOWNERZORDER;
        // Match the cursor-routing design: HWND_TOPMOST guarantees the topmost style, but
        // if the window is already topmost Windows may not reorder it above a
        // sibling topmost window. HWND_TOP then performs the actual z-order lift
        // while preserving WS_EX_TOPMOST.
        let _ = SetWindowPos(HWND(hwnd as *mut _), Some(HWND_TOPMOST), 0, 0, 0, 0, flags);
        let _ = SetWindowPos(HWND(hwnd as *mut _), Some(HWND_TOP), 0, 0, 0, 0, flags);
    }
}

/// Recommit the fullscreen WGL overlay into USER32's TOPMOST band, then
/// immediately place the visible panel above it.  This is intentionally used
/// only by the GUI-topmost-OFF helper path.  On the RX 9060 XT reproduction,
/// opening an mpv popup makes the missing panel/cursor appear immediately,
/// which proves that a top-level z-order/compositor transaction can repair the
/// presentation even though GetWindow already reports the logical order as
/// correct.  Recommitting the overlay first and helpers second reproduces that
/// transaction without requiring an external popup.
pub fn recommit_overlay_below_helpers(panel_hwnd: isize, overlay_hwnd: isize) {
    if overlay_hwnd == 0 || !is_window_valid(overlay_hwnd) || !is_own_window(overlay_hwnd) {
        return;
    }
    unsafe {
        let flags =
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOOWNERZORDER;
        let _ = SetWindowPos(
            HWND(overlay_hwnd as *mut _),
            Some(HWND_TOPMOST),
            0,
            0,
            0,
            0,
            flags,
        );
    }
    if panel_hwnd != 0
        && is_window_valid(panel_hwnd)
        && is_own_window(panel_hwnd)
        && is_window_visible(panel_hwnd)
    {
        raise_topmost(panel_hwnd);
        request_window_repaint(panel_hwnd);
    }
}

/// Return visible top-level HWNDs belonging to `pid` in current top-to-bottom
/// z-order. Used by the compositor diagnostics to detect source-owned popup
/// creation/removal (for example mpv's right-click menu).
pub fn visible_top_level_windows_for_pid(pid: u32, limit: usize) -> Vec<isize> {
    if pid == 0 || limit == 0 {
        return Vec::new();
    }
    unsafe {
        let mut out = Vec::new();
        let mut hwnd = GetTopWindow(None).unwrap_or_default();
        let mut guard = 0usize;
        while !hwnd.0.is_null() && guard < 4096 && out.len() < limit {
            let raw = hwnd.0 as isize;
            if window_pid(raw) == pid
                && IsWindowVisible(hwnd).as_bool()
                && !IsIconic(hwnd).as_bool()
            {
                out.push(raw);
            }
            hwnd = GetWindow(hwnd, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }
        out
    }
}

fn z_prev(hwnd: isize) -> isize {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return 0;
    }
    unsafe {
        GetWindow(HWND(hwnd as *mut _), GW_HWNDPREV)
            .unwrap_or_default()
            .0 as isize
    }
}

fn z_next(hwnd: isize) -> isize {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return 0;
    }
    unsafe {
        GetWindow(HWND(hwnd as *mut _), GW_HWNDNEXT)
            .unwrap_or_default()
            .0 as isize
    }
}

fn compact_window_diag(hwnd: isize) -> String {
    if hwnd == 0 {
        return "0".to_string();
    }
    let valid = is_window_valid(hwnd);
    if !valid {
        return format!("{hwnd:#x}:invalid");
    }
    let ex = unsafe { GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32 };
    format!(
        "{:#x}[pid={} cls='{}' title='{}' vis={} min={} top={} cloak={} owner={:#x} prev={:#x} next={:#x} ex={:#010x} rect={:?}]",
        hwnd,
        window_pid(hwnd),
        window_class(hwnd),
        window_title(hwnd)
            .replace('\n', " ")
            .chars()
            .take(80)
            .collect::<String>(),
        is_window_visible(hwnd),
        is_minimized(hwnd),
        is_topmost(hwnd),
        is_cloaked(hwnd),
        window_owner(hwnd),
        z_prev(hwnd),
        z_next(hwnd),
        ex,
        window_rect(hwnd),
    )
}

fn rgb_abs_delta(a: u32, b: u32) -> u32 {
    let ar = a & 0xff;
    let ag = (a >> 8) & 0xff;
    let ab = (a >> 16) & 0xff;
    let br = b & 0xff;
    let bg = (b >> 8) & 0xff;
    let bb = (b >> 16) & 0xff;
    ar.abs_diff(br) + ag.abs_diff(bg) + ab.abs_diff(bb)
}

/// Compare the pixels painted by the native panel child with the pixels that
/// the desktop DC reports at the panel's screen rectangle. This is deliberately
/// independent from IsWindowVisible/TOPMOST: the RX 9060 XT failure reports
/// those API states as correct even while the user cannot see the helper.
///
/// The desktop DC is still a diagnostic proxy (hardware overlay/MPO scanout is
/// not guaranteed to be observable through every capture API), so callers must
/// log this as a screen-sample result rather than an infallible DWM truth.
pub fn panel_screen_visibility_probe(panel_hwnd: isize) -> Option<(usize, usize, u32)> {
    if panel_hwnd == 0 || !is_window_valid(panel_hwnd) || !is_window_visible(panel_hwnd) {
        return None;
    }
    let (child_hwnd, snapshot) = {
        let state = panel_gdi_mirror_state().lock().ok()?;
        if state.hwnd == 0
            || !is_window_valid(state.hwnd)
            || state.snapshot.parent != panel_hwnd
            || !state.snapshot.visible
        {
            return None;
        }
        (state.hwnd, state.snapshot.clone())
    };
    let (sx, sy, sw, sh) = window_rect(panel_hwnd)?;
    let w = snapshot.width.min(sw).max(1);
    let h = snapshot.height.min(sh).max(1);
    unsafe {
        let screen_dc = GetDC(None);
        let child_dc = GetDC(Some(HWND(child_hwnd as *mut _)));
        if screen_dc.is_invalid() || child_dc.is_invalid() {
            if !screen_dc.is_invalid() {
                let _ = ReleaseDC(None, screen_dc);
            }
            if !child_dc.is_invalid() {
                let _ = ReleaseDC(Some(HWND(child_hwnd as *mut _)), child_dc);
            }
            return None;
        }

        // Spread samples over the complete 297x33 bar. Comparing against the
        // mirror's own pixels makes the probe valid for localized text, hover,
        // screenshot feedback, chip mode and DPI-scaled panel geometry.
        let xs = [2, w / 4, w / 2, w * 3 / 4, w - 3];
        let ys = [2, h / 2, h - 3];
        let mut matched = 0usize;
        let mut sampled = 0usize;
        let mut delta_sum = 0u64;
        for &x0 in &xs {
            for &y0 in &ys {
                let x = x0.clamp(0, w - 1);
                let y = y0.clamp(0, h - 1);
                let expected = GetPixel(child_dc, x, y).0;
                let actual = GetPixel(screen_dc, sx + x, sy + y).0;
                if expected == 0xffff_ffff || actual == 0xffff_ffff {
                    continue;
                }
                let delta = rgb_abs_delta(expected, actual);
                // 24 total RGB levels tolerates small desktop color-management
                // or GDI rounding differences while still rejecting video pixels.
                if delta <= 24 {
                    matched += 1;
                }
                sampled += 1;
                delta_sum += delta as u64;
            }
        }
        let _ = ReleaseDC(Some(HWND(child_hwnd as *mut _)), child_dc);
        let _ = ReleaseDC(None, screen_dc);
        if sampled == 0 {
            None
        } else {
            Some((matched, sampled, (delta_sum / sampled as u64) as u32))
        }
    }
}

fn visibility_probe_state(matched: usize, sampled: usize) -> &'static str {
    if sampled == 0 {
        "unknown"
    } else if matched * 100 >= sampled * 70 {
        "visible"
    } else if matched * 100 <= sampled * 40 {
        "missing"
    } else {
        "uncertain"
    }
}

/// Log physical-visibility evidence for the two helper surfaces. A WARN with
/// `helper-visibility-mismatch` is the requested explicit record that USER32
/// says a helper is visible while screen sampling cannot find its pixels.
pub fn log_helper_physical_visibility(
    tag: &str,
    gui_topmost: bool,
    panel_hwnd: isize,
    overlay_hwnd: isize,
    cursor_hwnd: isize,
) {
    let panel_api_visible =
        panel_hwnd != 0 && is_window_valid(panel_hwnd) && is_window_visible(panel_hwnd);
    let panel_probe = panel_screen_visibility_probe(panel_hwnd);
    let panel_state = panel_probe
        .map(|(m, n, _)| visibility_probe_state(m, n))
        .unwrap_or("unknown");
    let (cursor_requested, cursor_api_visible, cursor_matched, cursor_sampled, cursor_avg_delta) =
        crate::input::cursor_sprite_screen_probe();
    let cursor_state = if !cursor_requested {
        "not-requested"
    } else if cursor_sampled == 0 {
        "unknown"
    } else {
        visibility_probe_state(cursor_matched, cursor_sampled)
    };
    let panel_ratio = panel_probe
        .map(|(m, n, _)| format!("{m}/{n}"))
        .unwrap_or_else(|| "n/a".to_owned());
    let panel_avg_delta = panel_probe.map(|(_, _, d)| d).unwrap_or(0);
    let mismatch = (panel_api_visible && panel_state == "missing")
        || (cursor_requested && cursor_api_visible && cursor_state == "missing");
    let msg = format!(
        "tag={tag} probe=desktop-dc gui_topmost={gui_topmost} overlay={overlay_hwnd:#x} panel={panel_hwnd:#x} panel_api_visible={panel_api_visible} panel_screen={panel_state} panel_match={panel_ratio} panel_avg_rgb_delta={panel_avg_delta} cursor={cursor_hwnd:#x} cursor_requested={cursor_requested} cursor_api_visible={cursor_api_visible} cursor_screen={cursor_state} cursor_match={cursor_matched}/{cursor_sampled} cursor_avg_rgb_delta={cursor_avg_delta}"
    );
    if mismatch {
        log::warn!("helper-visibility-mismatch: {msg}");
    } else {
        log::info!("helper-visibility-probe: {msg}");
    }
}

/// High-value DWM/USER32 snapshot for the GUI-topmost-OFF regression. This is
/// deliberately verbose but rate-limited by the engine caller. It records the
/// logical sibling order plus visible source-owned popups so a menu-triggered
/// compositor repair can be compared with the broken state from one log file.
pub fn log_helper_compositor_snapshot(
    tag: &str,
    gui_hwnd: isize,
    panel_hwnd: isize,
    overlay_hwnd: isize,
    cursor_hwnd: isize,
    source_hwnd: isize,
) {
    if !crate::logging::diagnostics_enabled() {
        return;
    }
    let foreground = foreground_window();
    log::info!(
        "helper-compositor-snapshot: tag={} fg={:#x} gui={} panel={} overlay={} cursor={} source={}",
        tag,
        foreground,
        compact_window_diag(gui_hwnd),
        compact_window_diag(panel_hwnd),
        compact_window_diag(overlay_hwnd),
        compact_window_diag(cursor_hwnd),
        compact_window_diag(source_hwnd),
    );

    let source_pid = window_pid(source_hwnd);
    let source_windows = visible_top_level_windows_for_pid(source_pid, 12);
    let source_desc = source_windows
        .iter()
        .map(|&h| compact_window_diag(h))
        .collect::<Vec<_>>()
        .join(" | ");
    log::info!(
        "helper-compositor-source-windows: tag={} source_pid={} count={} windows={}",
        tag,
        source_pid,
        source_windows.len(),
        source_desc
    );

    unsafe {
        let mut top = GetTopWindow(None).unwrap_or_default();
        let mut rows = Vec::new();
        let mut guard = 0usize;
        while !top.0.is_null() && guard < 4096 && rows.len() < 18 {
            let raw = top.0 as isize;
            if IsWindowVisible(top).as_bool() && !IsIconic(top).as_bool() {
                let pid = window_pid(raw);
                if pid == GetCurrentProcessId() || pid == source_pid || is_topmost(raw) {
                    rows.push(compact_window_diag(raw));
                }
            }
            top = GetWindow(top, GW_HWNDNEXT).unwrap_or_default();
            guard += 1;
        }
        log::info!(
            "helper-compositor-zlist: tag={} rows={}",
            tag,
            rows.join(" | ")
        );
    }
}

/// Return the owner HWND of a top-level popup, or 0 when unowned.
pub fn window_owner(hwnd: isize) -> isize {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return 0;
    }
    unsafe {
        GetWindow(HWND(hwnd as *mut _), GW_OWNER)
            .unwrap_or_default()
            .0 as isize
    }
}

/// True when a foreign/source window already owns WS_EX_LAYERED presentation.
/// Neo must never apply its alpha-hide trick on top of an application-owned
/// layered window: WPF/per-pixel-alpha applications can lose their own backing
/// composition state even after alpha is restored to 255.
pub fn is_layered_window(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return false;
    }
    unsafe { (GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32) & WS_EX_LAYERED.0 != 0 }
}

/// Visible same-process top-level secondary/owned windows whose presentation is
/// geometrically part of `hwnd`. Windows Graphics Capture can include these via
/// SetIncludeSecondaryWindows(true). This is intentionally structural rather
/// than app-name/class-name based so WPF/EVR, Qt helper surfaces and similar
/// multi-HWND render hosts can use the same safe path.
pub fn wgc_visible_secondary_windows(hwnd: isize) -> Vec<isize> {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return Vec::new();
    }
    let pid = window_pid(hwnd);
    let Some((tx, ty, tw, th)) = window_rect(hwnd) else {
        return Vec::new();
    };
    if pid == 0 || tw <= 0 || th <= 0 {
        return Vec::new();
    }
    let t_right = tx.saturating_add(tw);
    let t_bottom = ty.saturating_add(th);
    let target_area = i64::from(tw).saturating_mul(i64::from(th)).max(1);

    struct Ctx {
        pid: u32,
        root: isize,
        tx: i32,
        ty: i32,
        tr: i32,
        tb: i32,
        target_area: i64,
        rows: Vec<(i64, isize)>,
    }
    unsafe extern "system" fn callback(candidate: HWND, lp: LPARAM) -> windows::core::BOOL {
        unsafe {
            let ctx = &mut *(lp.0 as *mut Ctx);
            let raw = candidate.0 as isize;
            if raw == ctx.root
                || !IsWindowVisible(candidate).as_bool()
                || IsIconic(candidate).as_bool()
            {
                return true.into();
            }
            let mut candidate_pid = 0u32;
            GetWindowThreadProcessId(candidate, Some(&mut candidate_pid));
            if candidate_pid != ctx.pid {
                return true.into();
            }

            // Secondary windows are top-level owned/helper windows. Walk the
            // owner chain instead of depending on application or class names.
            let mut owned_by_root = false;
            let mut owner = GetWindow(candidate, GW_OWNER).unwrap_or_default();
            let mut guard = 0usize;
            while !owner.0.is_null() && guard < 16 {
                if owner.0 as isize == ctx.root {
                    owned_by_root = true;
                    break;
                }
                owner = GetWindow(owner, GW_OWNER).unwrap_or_default();
                guard += 1;
            }
            if !owned_by_root && GetAncestor(candidate, GA_ROOTOWNER).0 as isize != ctx.root {
                return true.into();
            }

            let Some((cx, cy, cw, ch)) = window_rect(raw) else {
                return true.into();
            };
            if cw <= 0 || ch <= 0 {
                return true.into();
            }
            let cr = cx.saturating_add(cw);
            let cb = cy.saturating_add(ch);
            let ix0 = ctx.tx.max(cx);
            let iy0 = ctx.ty.max(cy);
            let ix1 = ctx.tr.min(cr);
            let iy1 = ctx.tb.min(cb);
            if ix1 <= ix0 || iy1 <= iy0 {
                return true.into();
            }
            let intersection = i64::from(ix1 - ix0).saturating_mul(i64::from(iy1 - iy0));
            let candidate_area = i64::from(cw).saturating_mul(i64::from(ch)).max(1);
            // Ignore tiny tooltips/menus. A persistent render helper normally
            // occupies a meaningful fraction of its host or is mostly inside it.
            let meaningful = intersection.saturating_mul(8) >= candidate_area.saturating_mul(5)
                && (intersection.saturating_mul(20) >= ctx.target_area
                    || candidate_area.saturating_mul(20) >= ctx.target_area);
            if meaningful {
                ctx.rows.push((candidate_area, raw));
            }
            true.into()
        }
    }

    let mut ctx = Ctx {
        pid,
        root: hwnd,
        tx,
        ty,
        tr: t_right,
        tb: t_bottom,
        target_area,
        rows: Vec::new(),
    };
    unsafe {
        let _ = EnumWindows(Some(callback), LPARAM(&mut ctx as *mut _ as isize));
    }
    ctx.rows.sort_by_key(|row| std::cmp::Reverse(row.0));
    ctx.rows.into_iter().map(|(_, hwnd)| hwnd).collect()
}

pub fn wgc_has_visible_secondary_windows(hwnd: isize) -> bool {
    !wgc_visible_secondary_windows(hwnd).is_empty()
}

/// Resolve a meaningful owned/helper presentation surface back to its root
/// owner before the GUI adopts it as the next capture target. This is purely
/// structural: no executable name, title, class name, or application ID is
/// consulted. Ordinary owned dialogs/popups remain selectable unless they are
/// already part of the root's large in-window presentation set.
pub fn normalize_capture_target(hwnd: isize) -> isize {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return hwnd;
    }
    let pid = window_pid(hwnd);
    if pid == 0 {
        return hwnd;
    }
    let root_owner = unsafe { GetAncestor(HWND(hwnd as *mut _), GA_ROOTOWNER).0 as isize };
    if root_owner == 0
        || root_owner == hwnd
        || !is_window_valid(root_owner)
        || window_pid(root_owner) != pid
    {
        return hwnd;
    }
    let Some(region_hwnd) = wgc_display_region_candidate(root_owner) else {
        return hwnd;
    };
    let Some(region_rect) = window_rect(region_hwnd) else {
        return hwnd;
    };
    let same_presentation_rect = window_rect(hwnd).is_some_and(|rect| {
        (rect.0 - region_rect.0).abs() <= 2
            && (rect.1 - region_rect.1).abs() <= 2
            && (rect.2 - region_rect.2).abs() <= 2
            && (rect.3 - region_rect.3).abs() <= 2
    });
    if same_presentation_rect && wgc_visible_secondary_windows(root_owner).contains(&hwnd) {
        log::debug!(
            "capture-target-normalized-secondary: selected={:#x} root={:#x} reason=paired-owned-presentation-region",
            hwnd,
            root_owner
        );
        root_owner
    } else {
        hwnd
    }
}

/// Return a conservative display-region fallback candidate for a multi-HWND
/// presentation host. The signature is deliberately narrow: the root itself
/// already owns layered presentation, and two meaningful owned surfaces occupy
/// essentially the same rectangle, with exactly one of them non-layered. This
/// is characteristic of a compositor/helper pair without tying the behavior to
/// any product, executable, title, class string, or vendor.
pub fn wgc_display_region_candidate(hwnd: isize) -> Option<isize> {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_layered_window(hwnd) {
        return None;
    }
    let rows = wgc_visible_secondary_windows(hwnd);
    if rows.len() < 2 {
        return None;
    }
    for &candidate in &rows {
        if is_layered_window(candidate) {
            continue;
        }
        let Some((cx, cy, cw, ch)) = window_rect(candidate) else {
            continue;
        };
        if cw <= 0 || ch <= 0 {
            continue;
        }
        for &partner in &rows {
            if partner == candidate || !is_layered_window(partner) {
                continue;
            }
            let Some((px, py, pw, ph)) = window_rect(partner) else {
                continue;
            };
            let nearly_same_rect = (cx - px).abs() <= 2
                && (cy - py).abs() <= 2
                && (cw - pw).abs() <= 2
                && (ch - ph).abs() <= 2;
            if nearly_same_rect {
                return Some(candidate);
            }
        }
    }
    None
}

/// Exclude one Neo-owned top-level window from public capture APIs while it is
/// still shown locally. Windows 10 2004+ removes WDA_EXCLUDEFROMCAPTURE windows
/// entirely from supported captures, which prevents a monitor-region fallback
/// from recursively capturing Neo's own fullscreen overlay.
pub fn set_own_window_capture_excluded(hwnd: isize, excluded: bool) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    let affinity = if excluded {
        WDA_EXCLUDEFROMCAPTURE
    } else {
        WDA_NONE
    };
    let ok = unsafe { SetWindowDisplayAffinity(HWND(hwnd as *mut _), affinity).is_ok() };
    if !ok {
        log::warn!(
            "capture-exclusion-change-failed: hwnd={:#x} excluded={}",
            hwnd,
            excluded
        );
        return false;
    }
    let registry = CAPTURE_EXCLUDED_WINDOWS.get_or_init(|| Mutex::new(Vec::new()));
    let mut rows = registry.lock().unwrap();
    if excluded {
        if !rows.contains(&hwnd) {
            rows.push(hwnd);
        }
    } else {
        rows.retain(|h| *h != hwnd);
    }
    true
}

/// Restore WDA_NONE on every Neo helper that was excluded for a monitor-region
/// capture. Safe to call unconditionally at Stop and before a new session.
pub fn clear_own_window_capture_exclusions() {
    let Some(registry) = CAPTURE_EXCLUDED_WINDOWS.get() else {
        return;
    };
    let rows = {
        let mut guard = registry.lock().unwrap();
        std::mem::take(&mut *guard)
    };
    for hwnd in rows {
        if hwnd != 0 && is_window_valid(hwnd) && is_own_window(hwnd) {
            unsafe {
                let _ = SetWindowDisplayAffinity(HWND(hwnd as *mut _), WDA_NONE);
            }
        }
    }
}

/// Candidate host windows for WGC when the exact selected HWND cannot be
/// converted to a GraphicsCaptureItem. This deliberately stays generic: child
/// render surfaces, owned video popups, and helper presentation windows may be
/// uncapturable even though their same-process top-level host is capturable.
///
/// Ordering is conservative: direct ancestors/owners first, then visible
/// same-process top-level windows that geometrically contain/overlap the
/// selected surface. The selected HWND itself is never returned here.
pub fn wgc_fallback_host_candidates(hwnd: isize) -> Vec<isize> {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return Vec::new();
    }
    let pid = window_pid(hwnd);
    let target_rect = window_rect(hwnd);
    let mut out = Vec::<isize>::new();
    let mut push_unique = |candidate: isize| {
        if candidate != 0
            && candidate != hwnd
            && is_window_valid(candidate)
            && window_pid(candidate) == pid
            && !out.contains(&candidate)
        {
            out.push(candidate);
        }
    };

    unsafe {
        let h = HWND(hwnd as *mut _);
        push_unique(GetAncestor(h, GA_ROOT).0 as isize);
        push_unique(GetAncestor(h, GA_ROOTOWNER).0 as isize);
        let mut owner = GetWindow(h, GW_OWNER).unwrap_or_default();
        let mut guard = 0usize;
        while !owner.0.is_null() && guard < 16 {
            push_unique(owner.0 as isize);
            push_unique(GetAncestor(owner, GA_ROOT).0 as isize);
            owner = GetWindow(owner, GW_OWNER).unwrap_or_default();
            guard += 1;
        }
    }

    struct EnumCtx {
        pid: u32,
        selected: isize,
        target_rect: Option<(i32, i32, i32, i32)>,
        rows: Vec<(u8, i64, isize)>,
    }
    unsafe extern "system" fn callback(hwnd: HWND, lp: LPARAM) -> windows::core::BOOL {
        unsafe {
            let ctx = &mut *(lp.0 as *mut EnumCtx);
            let raw = hwnd.0 as isize;
            if raw == ctx.selected || !IsWindowVisible(hwnd).as_bool() || IsIconic(hwnd).as_bool() {
                return true.into();
            }
            let mut candidate_pid = 0u32;
            GetWindowThreadProcessId(hwnd, Some(&mut candidate_pid));
            if candidate_pid != ctx.pid {
                return true.into();
            }
            let Some((cx, cy, cw, ch)) = window_rect(raw) else {
                return true.into();
            };
            if cw <= 0 || ch <= 0 {
                return true.into();
            }
            let Some((tx, ty, tw, th)) = ctx.target_rect else {
                return true.into();
            };
            if tw <= 0 || th <= 0 {
                return true.into();
            }
            let t_right = tx.saturating_add(tw);
            let t_bottom = ty.saturating_add(th);
            let c_right = cx.saturating_add(cw);
            let c_bottom = cy.saturating_add(ch);
            let center_x = tx.saturating_add(tw / 2);
            let center_y = ty.saturating_add(th / 2);
            let contains_center =
                center_x >= cx && center_x < c_right && center_y >= cy && center_y < c_bottom;
            let fully_contains = tx >= cx && ty >= cy && t_right <= c_right && t_bottom <= c_bottom;
            let ix0 = tx.max(cx);
            let iy0 = ty.max(cy);
            let ix1 = t_right.min(c_right);
            let iy1 = t_bottom.min(c_bottom);
            let intersection_area = if ix1 > ix0 && iy1 > iy0 {
                i64::from(ix1 - ix0).saturating_mul(i64::from(iy1 - iy0))
            } else {
                0
            };
            let target_area = i64::from(tw).saturating_mul(i64::from(th)).max(1);
            let area = i64::from(cw).saturating_mul(i64::from(ch));
            // Same-process windows can include tooltips/menus over the selected
            // surface. Never prefer one of those tiny overlaps as a capture
            // host. A useful host contains the whole target, or at minimum its
            // center with comparable area / a strong geometric overlap.
            let rank = if fully_contains {
                0
            } else if contains_center && area >= target_area {
                1
            } else if intersection_area.saturating_mul(10) >= target_area.saturating_mul(8) {
                2
            } else {
                return true.into();
            };
            ctx.rows.push((rank, area, raw));
            true.into()
        }
    }

    if pid != 0 && target_rect.is_some() {
        let mut ctx = EnumCtx {
            pid,
            selected: hwnd,
            target_rect,
            rows: Vec::new(),
        };
        unsafe {
            let _ = EnumWindows(Some(callback), LPARAM(&mut ctx as *mut _ as isize));
        }
        ctx.rows.sort_by_key(|row| (row.0, row.1));
        for (_, _, candidate) in ctx.rows {
            push_unique(candidate);
        }
    }
    out
}

/// Change the owner of one of Neo's top-level popup helper windows.
///
/// For a top-level WS_POPUP, GWLP_HWNDPARENT changes the owner (not the
/// parent/child relationship). USER32 guarantees that an owned popup remains
/// above its owner, which is stronger than a passive sibling Z-order query on
/// the AMD/DWM reproduction. Both HWNDs must belong to this process.
pub fn set_owned_popup_owner(popup_hwnd: isize, owner_hwnd: isize) {
    if popup_hwnd == 0 || !is_window_valid(popup_hwnd) || !is_own_window(popup_hwnd) {
        return;
    }
    if owner_hwnd != 0
        && (!is_window_valid(owner_hwnd) || !is_own_window(owner_hwnd) || popup_hwnd == owner_hwnd)
    {
        return;
    }
    if window_owner(popup_hwnd) == owner_hwnd {
        return;
    }
    unsafe {
        let _ = SetWindowLongPtrW(HWND(popup_hwnd as *mut _), GWLP_HWNDPARENT, owner_hwnd);
    }
}

/// Attach/detach the native control panel as an owned popup of the magnified
/// overlay. The relationship is used only while GUI-topmost is OFF. It does
/// not involve the main GUI, so the panel never follows GUI minimize/restore or
/// always-on-top state. Panel visibility remains controlled solely by the
/// "Show control panel" setting / panel hotkey.
pub fn set_panel_overlay_owner(panel_hwnd: isize, overlay_hwnd: isize, attach: bool) {
    set_owned_popup_owner(panel_hwnd, if attach { overlay_hwnd } else { 0 });
}

/// Keep the floating control panel above the magnified overlay without
/// continuously promoting it to the front of the TOPMOST band.
///
/// The panel, cursor sprite and optional TOPMOST GUI are independent helper
/// HWNDs. Re-raising the panel on every housekeeping tick creates a visible
/// intermediate stack (panel > cursor/GUI) before the later cursor/GUI repair,
/// which DWM can scan out as a one-frame blink. Only perform the heavy repair
/// when USER32 says the panel actually fell below the overlay.
pub fn force_panel_above_overlay(panel_hwnd: isize, overlay_hwnd: isize) {
    let own_live = |hwnd: isize| {
        hwnd != 0 && is_window_valid(hwnd) && is_own_window(hwnd) && is_window_visible(hwnd)
    };
    if !own_live(panel_hwnd) || !own_live(overlay_hwnd) {
        return;
    }

    // Keep panel/overlay as independent top-level helpers. A stale owner can
    // couple panel lifetime to the overlay and is never part of the steady
    // ordering contract.
    if window_owner(panel_hwnd) != 0 {
        set_panel_overlay_owner(panel_hwnd, overlay_hwnd, false);
    }
    if !is_topmost(overlay_hwnd) {
        set_own_topmost(overlay_hwnd, true);
    }
    if !is_topmost(panel_hwnd) {
        set_own_topmost(panel_hwnd, true);
    }

    // Steady state is intentionally a no-op. In particular, do not call
    // raise_topmost()+DwmFlush while the cursor is hovering the panel or while
    // the main GUI overlaps it: that transiently places the panel over the
    // cursor/GUI and is the source of the visible flashing.
    if window_is_above(panel_hwnd, overlay_hwnd) {
        return;
    }

    log::debug!(
        "panel-zorder-repair: panel={panel_hwnd:#x} overlay={overlay_hwnd:#x} reason=panel-below-overlay"
    );
    raise_topmost(panel_hwnd);
    request_window_repaint(panel_hwnd);
    unsafe {
        let _ = DwmFlush();
    }
}

/// Keep the floating control panel independent from the main GUI's
/// always-on-top preference. When both are live the panel is always TOPMOST
/// and immediately above the magnified overlay. This function never reads or
/// mutates the main GUI state and never changes panel visibility.
pub fn normalize_panel_overlay_stack(panel_hwnd: isize, overlay_hwnd: isize) {
    let own_live = |hwnd: isize| {
        hwnd != 0 && is_window_valid(hwnd) && is_own_window(hwnd) && is_window_visible(hwnd)
    };
    let panel = if own_live(panel_hwnd) { panel_hwnd } else { 0 };
    let overlay = if own_live(overlay_hwnd) {
        overlay_hwnd
    } else {
        0
    };
    if overlay == 0 {
        return;
    }
    if panel != 0 {
        force_panel_above_overlay(panel, overlay);
    } else if !is_topmost(overlay) {
        set_own_topmost(overlay, true);
    }
}

/// Normalize Neo's owned topmost siblings without activating any of them.
/// Desired order is GUI > panel > overlay when GUI-topmost is ON. When it is
/// OFF, the GUI is left entirely alone and the panel is force-reinserted above
/// the overlay; this is the broken RX 9060 XT path that cannot trust a passive
/// GetWindow z-order query.
pub fn normalize_neo_topmost_stack(
    gui_hwnd: isize,
    gui_topmost: bool,
    panel_hwnd: isize,
    overlay_hwnd: isize,
) {
    let own_live = |hwnd: isize| {
        hwnd != 0 && is_window_valid(hwnd) && is_own_window(hwnd) && is_window_visible(hwnd)
    };
    let gui = if gui_topmost && own_live(gui_hwnd) && !is_minimized(gui_hwnd) {
        gui_hwnd
    } else {
        0
    };
    let panel_valid = panel_hwnd != 0 && is_window_valid(panel_hwnd) && is_own_window(panel_hwnd);
    let panel = if own_live(panel_hwnd) { panel_hwnd } else { 0 };
    let overlay = if own_live(overlay_hwnd) {
        overlay_hwnd
    } else {
        0
    };

    // v235 never coupled panel/cursor lifetime to the overlay through GW_OWNER.
    // Always clean up a stale owner relationship before doing ordinary TOPMOST
    // ordering. This makes GUI-topmost OFF, GUI minimize and panel visibility
    // independent again.
    if panel_valid && window_owner(panel_hwnd) != 0 {
        set_owned_popup_owner(panel_hwnd, 0);
    }

    if overlay == 0 {
        keep_gui_transition_snapshot_topmost();
        return;
    }
    if !is_topmost(overlay) {
        set_own_topmost(overlay, true);
    }

    if panel != 0 {
        // This is now a conditional repair. When panel > overlay is already
        // true it must not perturb the cursor/GUI sibling order.
        force_panel_above_overlay(panel, overlay);
    }

    if gui == 0 {
        // GUI-topmost OFF: leave the GUI untouched. Panel/cursor ordering is
        // stable because the steady panel path above performs no front-raise.
        keep_gui_transition_snapshot_topmost();
        return;
    }

    if !is_topmost(gui) {
        set_own_topmost(gui, true);
    }

    // GUI is the highest ordinary Neo surface. Do not re-raise it every tick;
    // doing so makes it race the panel/cursor TOPMOST helpers. Repair only when
    // the observed sibling order is actually wrong. The cursor sprite is raised
    // separately by the input owner and remains the visual pointer above GUI.
    let gui_below_panel = panel != 0 && !window_is_above(gui, panel);
    let gui_below_overlay = !window_is_above(gui, overlay);
    if gui_below_panel || gui_below_overlay {
        log::debug!(
            "neo-topmost-zorder-repair: gui={gui:#x} panel={panel:#x} overlay={overlay:#x} below_panel={gui_below_panel} below_overlay={gui_below_overlay}"
        );
        raise_topmost(gui);
    }
    keep_gui_transition_snapshot_topmost();
}

pub fn set_visible_no_activate(hwnd: isize, visible: bool) {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window visibility mutation rejected: hwnd={hwnd:#x} visible={visible} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        let _ = ShowWindow(h, if visible { SW_SHOWNOACTIVATE } else { SW_HIDE });
        if visible {
            let _ = SetWindowPos(
                h,
                Some(HWND_TOPMOST),
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING,
            );
        }
    }
}

/// Insert `hwnd` directly below `above` in the z-order (no activation).
pub fn place_below(hwnd: isize, above: isize) {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let a = HWND(above as *mut _);
        // skip if already directly below
        let next = GetWindow(a, GW_HWNDNEXT).unwrap_or_default();
        if next.0 as isize == hwnd {
            return;
        }
        let _ = SetWindowPos(
            h,
            Some(a),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_NOSENDCHANGING,
        );
    }
}

/// Relaunch this exe elevated (UAC prompt). Returns true if launched.
pub fn relaunch_as_admin() -> bool {
    let Ok(exe) = std::env::current_exe() else {
        return false;
    };
    let verb: Vec<u16> = "runas\0".encode_utf16().collect();
    let file: Vec<u16> = exe
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    unsafe {
        let r = windows::Win32::UI::Shell::ShellExecuteW(
            None,
            windows::core::PCWSTR(verb.as_ptr()),
            windows::core::PCWSTR(file.as_ptr()),
            None,
            None,
            SW_SHOWNORMAL,
        );
        r.0 as isize > 32
    }
}

/// Enumerate ordinary visible File Explorer top-level windows.
fn explorer_windows() -> Vec<isize> {
    struct Ctx {
        windows: Vec<isize>,
    }

    unsafe extern "system" fn callback(
        hwnd: HWND,
        lp: windows::Win32::Foundation::LPARAM,
    ) -> windows::core::BOOL {
        unsafe {
            let ctx = &mut *(lp.0 as *mut Ctx);
            let owner = GetWindow(hwnd, GW_OWNER).unwrap_or_default();
            if !owner.0.is_null() || !IsWindowVisible(hwnd).as_bool() {
                return true.into();
            }
            let mut class_buf = [0u16; 64];
            let count = GetClassNameW(hwnd, &mut class_buf);
            let class = String::from_utf16_lossy(&class_buf[..count.max(0) as usize]);
            if matches!(class.as_str(), "CabinetWClass" | "ExploreWClass") {
                ctx.windows.push(hwnd.0 as isize);
            }
            true.into()
        }
    }

    let mut ctx = Ctx {
        windows: Vec::new(),
    };
    unsafe {
        let _ = EnumWindows(
            Some(callback),
            windows::Win32::Foundation::LPARAM(&mut ctx as *mut _ as isize),
        );
    }
    ctx.windows
}

fn explorer_window_is_usable(hwnd: isize) -> bool {
    if hwnd == 0 {
        return false;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if !IsWindow(Some(h)).as_bool() || !IsWindowVisible(h).as_bool() {
            return false;
        }
    }
    matches!(
        window_class(hwnd).as_str(),
        "CabinetWClass" | "ExploreWClass"
    ) && !is_cloaked(hwnd)
        && window_rect(hwnd).is_some_and(|(_, _, width, height)| width >= 240 && height >= 180)
}

#[derive(Clone, Copy, Debug)]
enum SettingsFolderExplorerState {
    Idle,
    Opening,
    Open(isize),
}

fn settings_folder_explorer_state() -> &'static std::sync::Mutex<SettingsFolderExplorerState> {
    static STATE: std::sync::OnceLock<std::sync::Mutex<SettingsFolderExplorerState>> =
        std::sync::OnceLock::new();
    STATE.get_or_init(|| std::sync::Mutex::new(SettingsFolderExplorerState::Idle))
}

fn lock_settings_folder_explorer_state()
-> std::sync::MutexGuard<'static, SettingsFolderExplorerState> {
    settings_folder_explorer_state()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn reset_settings_folder_explorer_state() {
    *lock_settings_folder_explorer_state() = SettingsFolderExplorerState::Idle;
}

pub fn monitor_work_area_of(hwnd: isize) -> (i32, i32, i32, i32) {
    unsafe {
        let mon = MonitorFromWindow(HWND(hwnd as *mut _), MONITOR_DEFAULTTONEAREST);
        let mut info = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(mon, &mut info).as_bool() {
            let rect = info.rcWork;
            (
                rect.left,
                rect.top,
                rect.right - rect.left,
                rect.bottom - rect.top,
            )
        } else {
            monitor_rect_of(hwnd)
        }
    }
}

fn exposed_strip_around_gui(gui: (i32, i32, i32, i32), explorer: (i32, i32, i32, i32)) -> i32 {
    let (gx, gy, gw, gh) = gui;
    let (ex, ey, ew, eh) = explorer;
    let gr = gx.saturating_add(gw);
    let gb = gy.saturating_add(gh);
    let er = ex.saturating_add(ew);
    let eb = ey.saturating_add(eh);
    [
        gx.saturating_sub(ex),
        er.saturating_sub(gr),
        gy.saturating_sub(ey),
        eb.saturating_sub(gb),
    ]
    .into_iter()
    .max()
    .unwrap_or(0)
    .max(0)
}

fn explorer_minimal_nudge_x(
    gui: (i32, i32, i32, i32),
    work: (i32, i32, i32, i32),
    explorer: (i32, i32, i32, i32),
) -> Option<i32> {
    const MIN_VISIBLE_STRIP: i32 = 140;
    let current_strip = exposed_strip_around_gui(gui, explorer);
    if current_strip >= MIN_VISIBLE_STRIP {
        return None;
    }

    let (wx, _, ww, _) = work;
    let work_right = wx.saturating_add(ww.max(1));
    let (gx, _, gw, _) = gui;
    let gui_right = gx.saturating_add(gw);
    let (ex, _, ew, _) = explorer;
    let max_x = work_right.saturating_sub(ew).max(wx);

    // Keep Explorer's normal Windows-managed size and Y position. Move only
    // far enough horizontally to leave a useful strip visible beside an
    // always-on-top Neo GUI. Slight overlap is intentional and acceptable.
    let right = gui_right
        .saturating_add(MIN_VISIBLE_STRIP)
        .saturating_sub(ew)
        .clamp(wx, max_x);
    let left = gx.saturating_sub(MIN_VISIBLE_STRIP).clamp(wx, max_x);

    let right_strip = exposed_strip_around_gui(gui, (right, explorer.1, ew, explorer.3));
    let left_strip = exposed_strip_around_gui(gui, (left, explorer.1, ew, explorer.3));
    let target = if right_strip > left_strip {
        right
    } else if left_strip > right_strip {
        left
    } else if (right - ex).abs() <= (left - ex).abs() {
        right
    } else {
        left
    };
    (target != ex).then_some(target)
}

/// Leave Explorer completely under Windows' normal shell management. Only if
/// the newly opened window would be almost entirely hidden behind Neo do we
/// apply one horizontal, position-only nudge. Size, title bar, show state,
/// activation and non-client metrics are never modified.
fn nudge_explorer_if_hidden_by_gui(explorer_hwnd: isize, gui_hwnd: isize) -> bool {
    if explorer_hwnd == 0 || gui_hwnd == 0 {
        return false;
    }
    let Some(gui_rect) = window_rect(gui_hwnd) else {
        return false;
    };
    let Some(explorer_rect) = window_rect(explorer_hwnd) else {
        return false;
    };
    let work = monitor_work_area_of(gui_hwnd);
    let Some(target_x) = explorer_minimal_nudge_x(gui_rect, work, explorer_rect) else {
        return false;
    };
    unsafe {
        SetWindowPos(
            HWND(explorer_hwnd as *mut _),
            None,
            target_x,
            explorer_rect.1,
            0,
            0,
            SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_NOSENDCHANGING,
        )
        .is_ok()
    }
}

#[derive(Debug)]
pub enum OpenApplicationFolderResult {
    Opened(std::path::PathBuf),
    AlreadyOpen(std::path::PathBuf),
}

/// Open the directory containing the currently running executable in ordinary
/// Windows Explorer. The path is resolved at click time and is never hard-coded
/// or persisted, preserving the portable layout when the app is moved.
///
/// Explorer is launched normally and visibly. Neo never hides it, resizes it,
/// changes its title-bar state, or reconstructs its frame. At most one Explorer
/// window launched by the current Neo process is tracked at a time.
pub fn open_application_folder(gui_hwnd: isize) -> Result<OpenApplicationFolderResult, String> {
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::core::PCWSTR;

    let exe = std::env::current_exe()
        .map_err(|error| format!("could not resolve the running executable: {error}"))?;
    let dir = exe
        .parent()
        .ok_or_else(|| "the running executable has no parent directory".to_string())?
        .to_path_buf();

    {
        let mut state = lock_settings_folder_explorer_state();
        match *state {
            SettingsFolderExplorerState::Opening => {
                return Ok(OpenApplicationFolderResult::AlreadyOpen(dir));
            }
            SettingsFolderExplorerState::Open(hwnd) => {
                if explorer_window_is_usable(hwnd) {
                    unsafe {
                        let _ = SetForegroundWindow(HWND(hwnd as *mut _));
                    }
                    return Ok(OpenApplicationFolderResult::AlreadyOpen(dir));
                }
                *state = SettingsFolderExplorerState::Idle;
            }
            SettingsFolderExplorerState::Idle => {}
        }
        *state = SettingsFolderExplorerState::Opening;
    }

    let existing: std::collections::HashSet<isize> = explorer_windows().into_iter().collect();
    let verb: Vec<u16> = "open\0".encode_utf16().collect();
    let explorer: Vec<u16> = "explorer.exe\0".encode_utf16().collect();
    let parameters_text = format!("/n,\"{}\"", dir.display());
    let parameters: Vec<u16> = parameters_text
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let result = unsafe {
        ShellExecuteW(
            None,
            PCWSTR(verb.as_ptr()),
            PCWSTR(explorer.as_ptr()),
            PCWSTR(parameters.as_ptr()),
            None,
            SW_SHOWNORMAL,
        )
    };
    let code = result.0 as isize;
    if code <= 32 {
        reset_settings_folder_explorer_state();
        return Err(match code {
            2 => "Windows Explorer was not found".to_string(),
            3 => "the application folder path was not found".to_string(),
            5 => "access to the application folder was denied".to_string(),
            8 => "not enough memory to open Windows Explorer".to_string(),
            31 => "Windows could not open the application folder".to_string(),
            other => format!("ShellExecuteW failed with code {other}"),
        });
    }

    let dir_for_log = dir.clone();
    std::thread::spawn(move || {
        let mut candidate = 0isize;
        let mut consecutive = 0u32;

        // The Explorer window is already visible and fully interactive. This
        // loop only identifies the newly created window for duplicate blocking
        // and an optional minimal horizontal nudge. No show/hide/resize calls.
        for _ in 0..100 {
            std::thread::sleep(std::time::Duration::from_millis(25));
            let found = explorer_windows()
                .into_iter()
                .filter(|hwnd| !existing.contains(hwnd))
                .find(|hwnd| explorer_window_is_usable(*hwnd))
                .unwrap_or(0);

            if found == 0 {
                candidate = 0;
                consecutive = 0;
                continue;
            }
            if found == candidate {
                consecutive = consecutive.saturating_add(1);
            } else {
                candidate = found;
                consecutive = 1;
            }
            if consecutive < 4 {
                continue;
            }

            let nudged = nudge_explorer_if_hidden_by_gui(candidate, gui_hwnd);
            unsafe {
                let _ = SetForegroundWindow(HWND(candidate as *mut _));
            }
            *lock_settings_folder_explorer_state() = SettingsFolderExplorerState::Open(candidate);
            log::info!(
                "settings-folder-explorer-ready: explorer=0x{:x} gui=0x{:x} path={} mode=normal-shell-open nudged={}",
                candidate,
                gui_hwnd,
                dir_for_log.display(),
                nudged
            );
            return;
        }

        // Explorer was successfully requested even if its HWND could not be
        // identified (for example, a shell policy reused an existing window).
        // Do not keep the launcher permanently blocked in that case.
        reset_settings_folder_explorer_state();
        log::info!(
            "settings-folder-explorer-ready: path={} mode=normal-shell-open hwnd=untracked",
            dir_for_log.display()
        );
    });

    Ok(OpenApplicationFolderResult::Opened(dir))
}

/// System/shell windows the user can't meaningfully magnify.
pub fn is_system_window(hwnd: isize) -> bool {
    const BLOCK: &[&str] = &[
        "Progman",
        "WorkerW",
        "Shell_TrayWnd",
        "Shell_SecondaryTrayWnd",
        "Windows.UI.Core.CoreWindow",
        "XamlExplorerHostIslandWindow",
        "TaskListThumbnailWnd",
        "ForegroundStaging",
        "NotifyIconOverflowWindow",
        "TopLevelWindowForOverflowXamlIsland",
    ];
    let class = window_class(hwnd);
    BLOCK.iter().any(|b| *b == class) || is_cloaked(hwnd)
}

/// Window class name (system-window filtering).
pub fn window_class(hwnd: isize) -> String {
    unsafe {
        let mut buf = [0u16; 128];
        let n = GetClassNameW(HWND(hwnd as *mut _), &mut buf);
        String::from_utf16_lossy(&buf[..n.max(0) as usize])
    }
}

/// Temporarily cloak/uncloak an owned top-level window at the DWM layer.
/// Used only as a short visual shield around root-GUI WGPU surface resizes;
/// unlike ShowWindow(SW_HIDE), this keeps the HWND/style/z-order intact.
pub fn set_window_cloaked(hwnd: isize, cloaked: bool) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return false;
    }
    let value: i32 = if cloaked { 1 } else { 0 };
    unsafe {
        DwmSetWindowAttribute(
            HWND(hwnd as *mut _),
            windows::Win32::Graphics::Dwm::DWMWA_CLOAK,
            (&value as *const i32).cast(),
            size_of::<i32>() as u32,
        )
        .is_ok()
    }
}

/// DWM-cloaked windows (suspended UWP etc.) are invisible to the user.
pub fn is_cloaked(hwnd: isize) -> bool {
    unsafe {
        let mut cloaked: u32 = 0;
        let _ = DwmGetWindowAttribute(
            HWND(hwnd as *mut _),
            windows::Win32::Graphics::Dwm::DWMWA_CLOAKED,
            &mut cloaked as *mut _ as *mut _,
            4,
        );
        cloaked != 0
    }
}

pub fn window_rect(hwnd: isize) -> Option<(i32, i32, i32, i32)> {
    unsafe {
        let mut r = RECT::default();
        if GetWindowRect(HWND(hwnd as *mut _), &mut r).is_ok() {
            Some((r.left, r.top, r.right - r.left, r.bottom - r.top))
        } else {
            None
        }
    }
}

pub fn set_topmost(hwnd: isize, on: bool) {
    unsafe {
        let _ = SetWindowPos(
            HWND(hwnd as *mut _),
            Some(if on { HWND_TOPMOST } else { HWND_NOTOPMOST }),
            0,
            0,
            0,
            0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        );
    }
}

/// Topmost mutation for Neo-owned GUI/helper windows. Source windows use
/// `set_topmost`, because they intentionally belong to another process.
pub fn set_own_topmost(hwnd: isize, on: bool) {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        log::error!(
            "foreign-window own-topmost mutation rejected: hwnd={hwnd:#x} on={on} pid={} current_pid={}",
            window_pid(hwnd),
            unsafe { GetCurrentProcessId() }
        );
        return;
    }
    set_topmost(hwnd, on);
}

/// The monitor rect containing `hwnd` (for fullscreen overlay geometry).
pub fn monitor_rect_of(hwnd: isize) -> (i32, i32, i32, i32) {
    unsafe {
        let mon = MonitorFromWindow(HWND(hwnd as *mut _), MONITOR_DEFAULTTONEAREST);
        let mut mi = MONITORINFO {
            cbSize: std::mem::size_of::<MONITORINFO>() as u32,
            ..Default::default()
        };
        if GetMonitorInfoW(mon, &mut mi).as_bool() {
            let r = mi.rcMonitor;
            (r.left, r.top, r.right - r.left, r.bottom - r.top)
        } else {
            (0, 0, 1920, 1080)
        }
    }
}

/// Current refresh rate of the monitor containing `hwnd`.
pub fn monitor_refresh_hz(hwnd: isize) -> Option<f64> {
    unsafe {
        let mon = MonitorFromWindow(HWND(hwnd as *mut _), MONITOR_DEFAULTTONEAREST);
        let mut mi = MONITORINFOEXW::default();
        mi.monitorInfo.cbSize = std::mem::size_of::<MONITORINFOEXW>() as u32;
        if !GetMonitorInfoW(mon, &mut mi.monitorInfo).as_bool() {
            return None;
        }
        let mut mode = DEVMODEW::default();
        mode.dmSize = std::mem::size_of::<DEVMODEW>() as u16;
        if !EnumDisplaySettingsW(
            windows::core::PCWSTR(mi.szDevice.as_ptr()),
            ENUM_CURRENT_SETTINGS,
            &mut mode,
        )
        .as_bool()
        {
            return None;
        }
        let hz = mode.dmDisplayFrequency as f64;
        (hz.is_finite() && (20.0..=1000.0).contains(&hz)).then_some(hz)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        WS_CAPTION, WS_MAXIMIZEBOX, WS_MINIMIZEBOX, WS_SIZEBOX, WS_SYSMENU,
        caption_style_with_disabled_maximize, explorer_minimal_nudge_x, exposed_strip_around_gui,
        fit_axis_to_bounds, rect_covers_monitor,
    };

    #[test]
    fn caption_buttons_keep_close_and_minimize_but_disable_maximize() {
        let style =
            WS_CAPTION.0 | WS_SIZEBOX.0 | WS_SYSMENU.0 | WS_MINIMIZEBOX.0 | WS_MAXIMIZEBOX.0;
        let configured = caption_style_with_disabled_maximize(style);
        assert_ne!(configured & WS_CAPTION.0, 0);
        assert_ne!(configured & WS_SIZEBOX.0, 0);
        assert_ne!(configured & WS_SYSMENU.0, 0);
        assert_ne!(configured & WS_MINIMIZEBOX.0, 0);
        assert_eq!(configured & WS_MAXIMIZEBOX.0, 0);
    }

    #[test]
    fn explorer_is_not_moved_when_a_useful_strip_is_already_visible() {
        let gui = (0, 100, 920, 700);
        let explorer = (780, 120, 900, 700);
        assert!(exposed_strip_around_gui(gui, explorer) >= 140);
        assert_eq!(
            explorer_minimal_nudge_x(gui, (0, 0, 1920, 1040), explorer),
            None
        );
    }

    #[test]
    fn explorer_gets_only_a_horizontal_nudge_when_nearly_hidden() {
        let gui = (100, 100, 920, 700);
        let explorer = (110, 110, 900, 680);
        let target = explorer_minimal_nudge_x(gui, (0, 0, 1920, 1040), explorer)
            .expect("nearly hidden Explorer should be nudged");
        assert_ne!(target, explorer.0);
        assert!(exposed_strip_around_gui(gui, (target, explorer.1, explorer.2, explorer.3)) >= 140);
    }

    #[test]
    fn capture_client_origin_is_kept_inside_monitor() {
        assert_eq!(fit_axis_to_bounds(1239, 610, 0, 1920), 1239);
        assert_eq!(fit_axis_to_bounds(1239, 1920, 0, 1920), 0);
        assert_eq!(fit_axis_to_bounds(1600, 640, 0, 1920), 1280);
        assert_eq!(fit_axis_to_bounds(-200, 640, 0, 1920), 0);
    }

    #[test]
    fn fullscreen_detection_handles_secondary_monitor_coordinates() {
        let monitor = (2560, 0, 2560, 1440);
        assert!(rect_covers_monitor((2560, 0, 2560, 1440), monitor, 2));
        assert!(rect_covers_monitor((2559, -1, 2562, 1442), monitor, 2));
        assert!(!rect_covers_monitor((2560, 0, 2560, 1400), monitor, 2));
        assert!(!rect_covers_monitor((2600, 30, 2400, 1300), monitor, 2));
    }
}
