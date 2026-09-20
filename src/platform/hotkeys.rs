//! Global hotkeys on a dedicated thread (RegisterHotKey + cooperative message pump).
//! Fired ids are pushed into an mpsc channel polled by the GUI.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, Sender, channel},
};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::*;
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, MSG, PM_NOREMOVE, PM_REMOVE, PeekMessageW, PostThreadMessageW, WM_HOTKEY,
    WM_QUIT,
};

pub const HK_TOGGLE: i32 = 1;
pub const HK_QUIT: i32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HotkeyValidationError {
    KeyCount,
    MissingModifier,
    MissingPrimary,
    MultiplePrimary,
    DuplicateKey,
    UnsupportedKey(String),
    WinModifier,
    Reserved,
}

fn primary_key(part: &str) -> Option<(u32, String)> {
    let upper = part.trim().to_ascii_uppercase();
    if upper.len() == 1 {
        let c = upper.chars().next()?;
        if c.is_ascii_alphanumeric() {
            return Some((c as u32, c.to_string()));
        }
    }
    if let Some(n) = upper.strip_prefix('F').and_then(|n| n.parse::<u32>().ok()) {
        if (1..=24).contains(&n) {
            return Some((VK_F1.0 as u32 + n - 1, format!("F{n}")));
        }
    }
    let (vk, name) = match upper.as_str() {
        "SPACE" => (VK_SPACE, "Space"),
        "ENTER" => (VK_RETURN, "Enter"),
        "HOME" => (VK_HOME, "Home"),
        "END" => (VK_END, "End"),
        "INSERT" => (VK_INSERT, "Insert"),
        "DELETE" => (VK_DELETE, "Delete"),
        "PAGEUP" => (VK_PRIOR, "PageUp"),
        "PAGEDOWN" => (VK_NEXT, "PageDown"),
        "UP" => (VK_UP, "Up"),
        "DOWN" => (VK_DOWN, "Down"),
        "LEFT" => (VK_LEFT, "Left"),
        "RIGHT" => (VK_RIGHT, "Right"),
        _ => return None,
    };
    Some((vk.0 as u32, name.into()))
}

fn parse_hotkey_checked(
    s: &str,
) -> Result<(HOT_KEY_MODIFIERS, u32, String), HotkeyValidationError> {
    let parts: Vec<&str> = s
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if !(2..=3).contains(&parts.len()) {
        return Err(HotkeyValidationError::KeyCount);
    }
    let mut mods = MOD_NOREPEAT;
    let mut modifier_count = 0usize;
    let mut primary = None;
    for part in parts {
        match part.trim().to_ascii_lowercase().as_str() {
            "ctrl" | "control" => {
                if mods.contains(MOD_CONTROL) {
                    return Err(HotkeyValidationError::DuplicateKey);
                }
                mods |= MOD_CONTROL;
                modifier_count += 1;
            }
            "alt" => {
                if mods.contains(MOD_ALT) {
                    return Err(HotkeyValidationError::DuplicateKey);
                }
                mods |= MOD_ALT;
                modifier_count += 1;
            }
            "shift" => {
                if mods.contains(MOD_SHIFT) {
                    return Err(HotkeyValidationError::DuplicateKey);
                }
                mods |= MOD_SHIFT;
                modifier_count += 1;
            }
            "win" => {
                if mods.contains(MOD_WIN) {
                    return Err(HotkeyValidationError::DuplicateKey);
                }
                mods |= MOD_WIN;
                modifier_count += 1;
            }
            k => {
                if primary.is_some() {
                    return Err(HotkeyValidationError::MultiplePrimary);
                }
                primary = Some(
                    primary_key(k)
                        .ok_or_else(|| HotkeyValidationError::UnsupportedKey(k.to_string()))?,
                );
            }
        }
    }
    if modifier_count == 0 {
        return Err(HotkeyValidationError::MissingModifier);
    }
    let (vk, primary_name) = primary.ok_or(HotkeyValidationError::MissingPrimary)?;
    let canonical = [
        mods.contains(MOD_CONTROL).then_some("Ctrl"),
        mods.contains(MOD_ALT).then_some("Alt"),
        mods.contains(MOD_SHIFT).then_some("Shift"),
        mods.contains(MOD_WIN).then_some("Win"),
    ]
    .into_iter()
    .flatten()
    .chain(std::iter::once(primary_name.as_str()))
    .collect::<Vec<_>>()
    .join("+");
    Ok((mods, vk, canonical))
}

/// Parse "Ctrl+Alt+Z" -> (modifiers, vk). Returns None if unparseable.
pub fn parse_hotkey(s: &str) -> Option<(HOT_KEY_MODIFIERS, u32)> {
    parse_hotkey_checked(s).ok().map(|(mods, vk, _)| (mods, vk))
}

pub fn validate_user_hotkey(s: &str) -> Result<String, HotkeyValidationError> {
    let (mods, _vk, canonical) = parse_hotkey_checked(s)?;
    if mods.contains(MOD_WIN) {
        return Err(HotkeyValidationError::WinModifier);
    }
    if matches!(canonical.as_str(), "Ctrl+Alt+Q" | "Ctrl+Alt+P" | "Alt+F4")
        || canonical.eq_ignore_ascii_case("Ctrl+Alt+Delete")
    {
        return Err(HotkeyValidationError::Reserved);
    }
    Ok(canonical)
}

/// Probe Windows registration without keeping the key. The caller should skip
/// this when the candidate is unchanged because its current hotkey thread owns
/// that registration already.
pub fn hotkey_is_available(s: &str) -> bool {
    let Some((mods, vk)) = parse_hotkey(s) else {
        return false;
    };
    const PROBE_ID: i32 = 0x4348;
    unsafe {
        if RegisterHotKey(None, PROBE_ID, mods, vk).is_err() {
            return false;
        }
        let _ = UnregisterHotKey(None, PROBE_ID);
    }
    true
}

#[derive(Clone, Debug)]
pub struct HotkeyEvent {
    pub id: i32,
    pub binding: String,
    pub received_at: Instant,
    pub handled_directly: bool,
    pub background_gui: BackgroundGui,
    /// Foreground window sampled before a minimized/hidden GUI is temporarily
    /// pumped. This preserves the user's intended next capture target even if
    /// the wake transaction changes desktop foreground ordering afterwards.
    pub foreground_hwnd: isize,
    pub cursor_target_hwnd: isize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackgroundGui {
    Foreground,
    Minimized,
    Hidden,
}

fn direct_background_stop_allowed(background_gui: BackgroundGui, capture_active: bool) -> bool {
    // Stop never needs the root GUI. Keeping both minimized and tray-hidden
    // stops on the engine's dedicated resident lane avoids briefly showing the
    // hidden eframe root while the capture overlay is being destroyed. That
    // show/teardown race could deliver CloseRequested to the root viewport and
    // terminate Neo after the first background stop.
    capture_active && background_gui != BackgroundGui::Foreground
}

pub struct HotkeyThread {
    pub rx: Receiver<HotkeyEvent>,
    pub registration_failures: Vec<(i32, String)>,
    thread_id: u32,
    stop_requested: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl HotkeyThread {
    /// `keys` = [(id, "Ctrl+Alt+Z"), ...]
    pub fn start(
        keys: Vec<(i32, String)>,
        minimized_stop: crate::engine::EngineStopHandle,
        wake_gui: std::sync::Arc<dyn Fn() + Send + Sync>,
    ) -> Self {
        let (tx, rx): (Sender<HotkeyEvent>, Receiver<HotkeyEvent>) = channel();
        let (id_tx, id_rx) = channel();
        let (ready_tx, ready_rx) = channel();
        let stop_requested = Arc::new(AtomicBool::new(false));
        let worker_stop_requested = Arc::clone(&stop_requested);
        let handle = std::thread::Builder::new()
            .name("hotkeys".into())
            .spawn(move || unsafe {
                let tid = windows::Win32::System::Threading::GetCurrentThreadId();
                // Create the thread message queue before publishing the TID. This
                // keeps PostThreadMessage usable, while the independent atomic stop
                // flag below guarantees shutdown even if a posted WM_QUIT is lost.
                let mut bootstrap = MSG::default();
                let _ = PeekMessageW(&mut bootstrap, None, 0, 0, PM_NOREMOVE);
                let _ = id_tx.send(tid);
                let mut failures = Vec::new();
                for (id, s) in &keys {
                    if let Some((mods, vk)) = parse_hotkey(s) {
                        if RegisterHotKey(None, *id, mods, vk).is_err() {
                            log::warn!("hotkey-registration-failed: id={id} binding='{s}' reason=already-in-use");
                            failures.push((*id, s.clone()));
                        } else {
                            log::info!("hotkey-registered: id={id} binding='{s}'");
                        }
                    } else {
                        failures.push((*id, s.clone()));
                    }
                }
                let _ = ready_tx.send(failures);
                let mut msg = MSG::default();
                loop {
                    if worker_stop_requested.load(Ordering::Acquire) {
                        break;
                    }
                    let mut drained_message = false;
                    while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                        drained_message = true;
                        if msg.message == WM_QUIT {
                            worker_stop_requested.store(true, Ordering::Release);
                            break;
                        }
                        if msg.message == WM_HOTKEY {
                        let id = msg.wParam.0 as i32;
                        let binding = keys
                            .iter()
                            .find(|(registered_id, _)| *registered_id == id)
                            .map(|(_, binding)| binding.clone())
                            .unwrap_or_else(|| format!("id:{id}"));
                        let received_at = Instant::now();
                        let foreground_hwnd = super::win32::foreground_window();
                        let (cursor_x, cursor_y) = super::win32::cursor_pos();
                        let cursor_target_hwnd =
                            super::win32::external_top_level_window_at_point(cursor_x, cursor_y);
                        // Background Stop goes straight to the resident engine.
                        // Every other command is GUI-owned and therefore wakes the
                        // hidden/minimized eframe root under a DWM cloak so the
                        // event is consumed without presenting the GUI.
                        let gui_hwnd = super::win32::main_gui_hwnd();
                        // An iconic window can also report as not visible.
                        // Preserve Minimized before applying the broader Hidden
                        // classification so Start restores the exact state.
                        let background_intent = super::win32::main_gui_background_intent();
                        let background_gui = if background_intent == 1 {
                            BackgroundGui::Minimized
                        } else if background_intent == 2 {
                            BackgroundGui::Hidden
                        } else if gui_hwnd != 0
                            && super::win32::is_minimized(gui_hwnd)
                        {
                            BackgroundGui::Minimized
                        } else if gui_hwnd != 0 && !super::win32::is_window_visible(gui_hwnd) {
                            BackgroundGui::Hidden
                        } else {
                            BackgroundGui::Foreground
                        };
                        let handled_directly = id == HK_TOGGLE
                            && direct_background_stop_allowed(
                                background_gui,
                                minimized_stop.is_active(),
                            );
                        if handled_directly {
                            // End the floating panel's visual/input lifetime at
                            // the same instant as the resident Stop request. Do
                            // not wait for a minimized eframe root to repaint.
                            let panel = super::win32::quiesce_panel_for_capture_stop();
                            log::info!(
                                "hotkey-background-direct-dispatch: id={id} binding='{binding}' gui={background_gui:?} action=stop resident=true panel_quiesced={panel:#x}"
                            );
                            minimized_stop.request_stop("background-global-hotkey");
                            // The engine stop lane is independent from eframe,
                            // but final GUI bookkeeping (idle epoch, anchor
                            // withdrawal, panel state) still needs one bounded
                            // background pump. Wake without activation; main.rs
                            // restores the exact previous state after cleanup.
                            let _ = super::win32::wake_background_gui(gui_hwnd);
                        } else if background_gui != BackgroundGui::Foreground {
                            // Every non-direct global command is owned by the GUI
                            // state machine (Start, panel toggle, GUI-topmost, Quit).
                            // A minimized/hidden eframe root otherwise receives no
                            // update frame, so the event can sit in rx indefinitely.
                            // Wake it DWM-cloaked; main.rs restores the exact previous
                            // background state after the command has committed.
                            let _ = super::win32::wake_background_gui(gui_hwnd);
                        }
                        if id == HK_QUIT {
                            // The janitor only starts its timeout clock here. It
                            // does NOT alter ordinary capture/input state. Neo
                            // gets its normal close path and normal 2 s stable
                            // shutdown grace plus the janitor safety margin first.
                            crate::input::notify_cursor_janitor_quit_requested(
                                "global-quit-hotkey",
                            );
                        }
                        log::info!("hotkey-received: id={id} binding='{binding}'");
                        let _ = tx.send(HotkeyEvent {
                            id,
                            binding,
                            received_at,
                            handled_directly,
                            background_gui,
                            foreground_hwnd,
                            cursor_target_hwnd,
                        });
                            wake_gui();
                        }
                        DispatchMessageW(&msg);
                    }
                    if worker_stop_requested.load(Ordering::Acquire) {
                        break;
                    }
                    // Global hotkeys do not need a permanently blocking GetMessage
                    // wait. A short idle sleep keeps CPU use negligible and, more
                    // importantly, lets the shared stop flag terminate this thread
                    // deterministically even if PostThreadMessage fails during GUI
                    // teardown. Maximum idle shutdown/hotkey polling latency is 4 ms.
                    if !drained_message {
                        std::thread::sleep(Duration::from_millis(4));
                    }
                }
                for (id, _) in &keys {
                    let _ = UnregisterHotKey(None, *id);
                }
                log::info!("hotkey-thread-stopped: tid={tid} registrations_released=true");
            })
            .expect("hotkey thread");
        let thread_id = id_rx.recv().unwrap_or(0);
        let registration_failures = ready_rx.recv().unwrap_or_default();
        Self {
            rx,
            registration_failures,
            thread_id,
            stop_requested,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        self.stop_requested.store(true, Ordering::Release);
        let post_quit_ok = unsafe {
            PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0)).is_ok()
        };
        log::debug!(
            "hotkey-thread-stop-requested: tid={} post_quit_ok={} atomic_stop=true",
            self.thread_id,
            post_quit_ok
        );
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        log::debug!("hotkey-thread-stop-complete: tid={}", self.thread_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_two_or_three_key_user_shortcuts() {
        assert_eq!(validate_user_hotkey("Ctrl+Z").unwrap(), "Ctrl+Z");
        assert_eq!(
            validate_user_hotkey("control+alt+f12").unwrap(),
            "Ctrl+Alt+F12"
        );
    }

    #[test]
    fn rejects_modifier_only_and_four_key_shortcuts() {
        assert_eq!(
            validate_user_hotkey("Ctrl+Shift"),
            Err(HotkeyValidationError::MissingPrimary)
        );
        assert_eq!(
            validate_user_hotkey("Ctrl+Alt+Shift+Z"),
            Err(HotkeyValidationError::KeyCount)
        );
    }

    #[test]
    fn every_background_stop_uses_the_resident_engine_lane() {
        assert!(direct_background_stop_allowed(BackgroundGui::Hidden, true));
        assert!(direct_background_stop_allowed(
            BackgroundGui::Minimized,
            true
        ));
        assert!(!direct_background_stop_allowed(BackgroundGui::Foreground, true));
        assert!(!direct_background_stop_allowed(BackgroundGui::Hidden, false));
        assert!(!direct_background_stop_allowed(BackgroundGui::Minimized, false));
    }

    #[test]
    fn repeated_tray_hidden_stops_never_fall_back_to_the_gui_wake_lane() {
        for _ in 0..10_000 {
            assert!(direct_background_stop_allowed(BackgroundGui::Hidden, true));
        }
    }

    #[test]
    fn ordinary_stop_and_quit_do_not_arm_emergency_input_recovery() {
        let source = include_str!("hotkeys.rs");
        let production = source.split("#[cfg(test)]").next().unwrap_or(source);
        assert!(!production.contains("request_stop_lockfree(\"global-hotkey-direct\""));
        assert!(!production.contains("request_emergency_input_release(\"global-quit-hotkey\""));
        assert!(production.contains("if id == HK_QUIT"));
        assert!(production.contains("wake_background_gui(gui_hwnd)"));
        assert!(production.contains("request_stop(\"background-global-hotkey\""));
        assert!(production.contains("notify_cursor_janitor_quit_requested"));
    }

    #[test]
    fn rejects_reserved_and_win_shortcuts() {
        assert_eq!(
            validate_user_hotkey("Ctrl+Alt+Q"),
            Err(HotkeyValidationError::Reserved)
        );
        assert_eq!(
            validate_user_hotkey("Win+Z"),
            Err(HotkeyValidationError::WinModifier)
        );
    }
}

impl Drop for HotkeyThread {
    fn drop(&mut self) {
        self.stop();
    }
}
