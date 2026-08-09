//! Win32 ctypes-style leaf helpers (window queries, monitor geometry).

use windows::Win32::Foundation::{HWND, POINT, RECT};
use windows::Win32::Graphics::Dwm::{
    DWMWA_CAPTION_COLOR, DWMWA_EXTENDED_FRAME_BOUNDS, DWMWA_TEXT_COLOR,
    DWMWA_USE_IMMERSIVE_DARK_MODE, DwmGetWindowAttribute, DwmSetWindowAttribute,
};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, DEVMODEW, ENUM_CURRENT_SETTINGS, EnumDisplaySettingsW, GetMonitorInfoW,
    MONITOR_DEFAULTTONEAREST, MONITORINFO, MONITORINFOEXW, MonitorFromWindow, RDW_ALLCHILDREN,
    RDW_FRAME, RDW_INVALIDATE, RDW_UPDATENOW, RedrawWindow,
};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_LBUTTON, VK_MBUTTON, VK_RBUTTON,
};
use windows::Win32::UI::WindowsAndMessaging::*;

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

/// Remove the native maximize command while retaining ordinary resizing and
/// minimize/close. The GUI has no useful maximized layout and changing its
/// desktop footprint during capture destabilizes cursor handoff geometry.
pub fn disable_maximize(hwnd: isize) {
    if hwnd == 0 || !is_window_valid(hwnd) {
        return;
    }
    unsafe {
        let h = HWND(hwnd as *mut _);
        if IsZoomed(h).as_bool() {
            let _ = ShowWindow(h, SW_RESTORE);
        }
        let style = GetWindowLongW(h, GWL_STYLE) as u32;
        let next = style & !WS_MAXIMIZEBOX.0;
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
        // keep the handle alive for the process lifetime (leak on purpose)
        if let Ok(h) = handle {
            // HANDLE is a Copy wrapper and has no Drop implementation. Leaving
            // it unclosed intentionally keeps the named mutex for process life.
            let _ = h;
        }
        !already
    }
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

pub fn window_input_passthrough(hwnd: isize) -> bool {
    if hwnd == 0 || !is_window_valid(hwnd) || !is_own_window(hwnd) {
        return false;
    }
    unsafe { GetWindowLongW(HWND(hwnd as *mut _), GWL_EXSTYLE) as u32 & WS_EX_TRANSPARENT.0 != 0 }
}

/// Move+resize without activation or z change.
pub fn set_window_rect(hwnd: isize, x: i32, y: i32, w: i32, h: i32) {
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
}

pub fn is_maximized(hwnd: isize) -> bool {
    unsafe { IsZoomed(HWND(hwnd as *mut _)).as_bool() }
}

/// True for a borderless/browser video window whose outer rectangle covers
/// its monitor. Resizing such a window changes the hidden browser surface
/// while Windows keeps fullscreen cursor coordinates, so capture-resolution
/// overrides must be ignored for that session.
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

/// Restore an exact outer-window rectangle. A currently maximized window must
/// first leave the maximized state or SetWindowPos only changes its hidden
/// normal-placement rectangle.
pub fn restore_window_rect(hwnd: isize, rect: (i32, i32, i32, i32), was_maximized: bool) -> bool {
    if !is_window_valid(hwnd) || rect.2 <= 0 || rect.3 <= 0 {
        return false;
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
    window_rect(hwnd).is_some()
}

pub fn any_mouse_button_down() -> bool {
    unsafe {
        [VK_LBUTTON, VK_RBUTTON, VK_MBUTTON]
            .into_iter()
            .any(|vk| GetAsyncKeyState(vk.0 as i32) < 0)
    }
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

pub fn window_title(hwnd: isize) -> String {
    unsafe {
        let h = HWND(hwnd as *mut _);
        let mut buf = [0u16; 512];
        let n = GetWindowTextW(h, &mut buf);
        String::from_utf16_lossy(&buf[..n.max(0) as usize])
    }
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

fn monitor_work_area_of(hwnd: isize) -> (i32, i32, i32, i32) {
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
        explorer_minimal_nudge_x, exposed_strip_around_gui, fit_axis_to_bounds, rect_covers_monitor,
    };

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
