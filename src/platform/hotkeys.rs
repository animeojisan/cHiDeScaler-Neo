//! Global hotkeys on a dedicated thread (RegisterHotKey + GetMessage loop).
//! Fired ids are pushed into an mpsc channel polled by the GUI.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Instant;
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::*;
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, GetMessageW, MSG, PostThreadMessageW, WM_HOTKEY, WM_QUIT,
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
    pub handled_while_minimized: bool,
}

pub struct HotkeyThread {
    pub rx: Receiver<HotkeyEvent>,
    pub registration_failures: Vec<(i32, String)>,
    thread_id: u32,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl HotkeyThread {
    /// `keys` = [(id, "Ctrl+Alt+Z"), ...]
    pub fn start(
        keys: Vec<(i32, String)>,
        minimized_stop: crate::engine::EngineStopHandle,
    ) -> Self {
        let (tx, rx): (Sender<HotkeyEvent>, Receiver<HotkeyEvent>) = channel();
        let (id_tx, id_rx) = channel();
        let (ready_tx, ready_rx) = channel();
        let handle = std::thread::Builder::new()
            .name("hotkeys".into())
            .spawn(move || unsafe {
                let tid = windows::Win32::System::Threading::GetCurrentThreadId();
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
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    if msg.message == WM_HOTKEY {
                        let id = msg.wParam.0 as i32;
                        let binding = keys
                            .iter()
                            .find(|(registered_id, _)| *registered_id == id)
                            .map(|(_, binding)| binding.clone())
                            .unwrap_or_else(|| format!("id:{id}"));
                        let received_at = Instant::now();
                        let gui_hwnd = super::win32::main_gui_hwnd();
                        let handled_while_minimized = id == HK_TOGGLE
                            && gui_hwnd != 0
                            && super::win32::is_minimized(gui_hwnd);
                        if handled_while_minimized {
                            log::info!(
                                "hotkey-minimized-direct-dispatch: id={id} binding='{binding}' action=stop"
                            );
                            minimized_stop.request_stop("minimized-global-hotkey");
                        }
                        log::info!("hotkey-received: id={id} binding='{binding}'");
                        let _ = tx.send(HotkeyEvent {
                            id,
                            binding,
                            received_at,
                            handled_while_minimized,
                        });
                    }
                    DispatchMessageW(&msg);
                }
                for (id, _) in &keys {
                    let _ = UnregisterHotKey(None, *id);
                }
            })
            .expect("hotkey thread");
        let thread_id = id_rx.recv().unwrap_or(0);
        let registration_failures = ready_rx.recv().unwrap_or_default();
        Self {
            rx,
            registration_failures,
            thread_id,
            handle: Some(handle),
        }
    }

    pub fn stop(&mut self) {
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_QUIT, WPARAM(0), LPARAM(0));
        }
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
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
