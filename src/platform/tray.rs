//! Windows notification-area controller on an isolated message thread.
#![allow(unsafe_op_in_unsafe_fn)]

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, GetLastError, HINSTANCE, HWND, LPARAM, LRESULT, POINT, WPARAM,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Shell::{
    NIF_ICON, NIF_MESSAGE, NIF_SHOWTIP, NIF_TIP, NIM_ADD, NIM_DELETE, NIM_SETVERSION,
    NOTIFYICON_VERSION_4, NOTIFYICONDATAW, Shell_NotifyIconW,
};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{PCWSTR, w};

const WM_TRAY_ICON: u32 = WM_APP + 41;
const WM_TRAY_WAKE: u32 = WM_APP + 42;
const MENU_SHOW: usize = 4101;
const MENU_EXIT: usize = 4103;

#[derive(Clone, Debug)]
pub enum TrayEvent {
    ToggleGui,
    Exit,
}

#[derive(Clone, Debug)]
pub struct TrayLabels {
    pub show: String,
    pub exit: String,
}

enum Command {
    SetRunning(bool),
    SetLabels(TrayLabels),
    Shutdown,
}

struct Shared {
    event_tx: Sender<TrayEvent>,
    wake: Box<dyn Fn() + Send + Sync>,
    labels: TrayLabels,
    running: bool,
    hwnd: isize,
}

static SHARED: OnceLock<Mutex<Option<Shared>>> = OnceLock::new();

fn wide(value: &str) -> Vec<u16> {
    value.encode_utf16().chain(std::iter::once(0)).collect()
}

fn notify(event: TrayEvent) {
    if let Ok(guard) = SHARED.get_or_init(|| Mutex::new(None)).lock()
        && let Some(shared) = guard.as_ref()
    {
        let _ = shared.event_tx.send(event);
        (shared.wake)();
    }
}

unsafe fn add_icon(hwnd: HWND) {
    let mut data = NOTIFYICONDATAW::default();
    data.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
    data.hWnd = hwnd;
    data.uID = 1;
    data.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP | NIF_SHOWTIP;
    data.uCallbackMessage = WM_TRAY_ICON;
    let instance = GetModuleHandleW(None).unwrap_or_default();
    data.hIcon = LoadIconW(Some(HINSTANCE(instance.0)), PCWSTR(1usize as *const u16))
        .or_else(|_| LoadIconW(None, IDI_APPLICATION))
        .unwrap_or_default();
    let tip = wide("cHiDeScaler-Neo");
    let count = tip.len().min(data.szTip.len());
    data.szTip[..count].copy_from_slice(&tip[..count]);
    let _ = Shell_NotifyIconW(NIM_ADD, &data);
    data.Anonymous.uVersion = NOTIFYICON_VERSION_4;
    let _ = Shell_NotifyIconW(NIM_SETVERSION, &data);
}

unsafe fn delete_icon(hwnd: HWND) {
    let mut data = NOTIFYICONDATAW::default();
    data.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
    data.hWnd = hwnd;
    data.uID = 1;
    let _ = Shell_NotifyIconW(NIM_DELETE, &data);
}

unsafe fn popup_menu(hwnd: HWND) {
    let labels = SHARED
        .get_or_init(|| Mutex::new(None))
        .lock()
        .ok()
        .and_then(|guard| guard.as_ref().map(|s| s.labels.clone()))
        .unwrap_or(TrayLabels {
            show: "Show".into(),
            exit: "Exit".into(),
        });
    let Ok(menu) = CreatePopupMenu() else { return };
    let show = wide(&labels.show);
    let exit = wide(&labels.exit);
    let _ = AppendMenuW(menu, MF_STRING, MENU_SHOW, PCWSTR(show.as_ptr()));
    let _ = AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null());
    let _ = AppendMenuW(menu, MF_STRING, MENU_EXIT, PCWSTR(exit.as_ptr()));
    let mut point = POINT::default();
    let _ = GetCursorPos(&mut point);
    let _ = SetForegroundWindow(hwnd);
    let _ = TrackPopupMenu(menu, TPM_RIGHTBUTTON, point.x, point.y, Some(0), hwnd, None);
    let _ = DestroyMenu(menu);
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if message == WM_TRAY_ICON {
        // NOTIFYICON_VERSION_4 packs the event into LOWORD(lParam) and the
        // icon id into HIWORD(lParam). Comparing the full LPARAM makes every
        // click look unknown once Explorer accepts v4 notification semantics.
        let mouse_message = (lparam.0 as u32) & 0xffff;
        if mouse_message == WM_LBUTTONUP {
            // Keep notification-area clicks on the same GUI visibility toggle
            // as the context-menu item. Do not restore the HWND on this tray
            // thread before the main thread decides whether to show or hide it.
            notify(TrayEvent::ToggleGui);
            return LRESULT(0);
        }
        if matches!(mouse_message, WM_RBUTTONUP | WM_CONTEXTMENU) {
            popup_menu(hwnd);
            return LRESULT(0);
        }
    } else if message == WM_COMMAND {
        match wparam.0 & 0xffff {
            MENU_SHOW => {
                // The historical label says "Show GUI", but the command is a
                // visibility toggle: hidden -> show, visible -> hide. The main
                // GUI thread owns the actual HWND transition.
                notify(TrayEvent::ToggleGui);
            }
            MENU_EXIT => {
                // A minimized/hidden eframe root can remain event-starved even
                // after request_repaint(), so waiting for the main thread to
                // drain TrayEvent::Exit leaves the process in the taskbar until
                // the user activates it. Post WM_CLOSE directly to the real root;
                // its normal eframe on_exit path still owns every cleanup step.
                let root = super::win32::main_gui_hwnd();
                let posted = super::win32::request_main_gui_close(root);
                log::info!(
                    "task-tray-native-exit-dispatch: gui={root:#x} posted={posted}"
                );
                notify(TrayEvent::Exit);
            }
            _ => {}
        }
        return LRESULT(0);
    } else if message == WM_TRAY_WAKE {
        return LRESULT(0);
    } else if message == WM_DESTROY {
        PostQuitMessage(0);
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, message, wparam, lparam)
}

pub struct TrayThread {
    pub rx: Receiver<TrayEvent>,
    command_tx: Sender<Command>,
    thread_id: u32,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl TrayThread {
    pub fn start(labels: TrayLabels, wake: impl Fn() + Send + Sync + 'static) -> Option<Self> {
        let (event_tx, rx) = mpsc::channel();
        let (command_tx, command_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name("notification-tray".into())
            .spawn(move || unsafe {
                let tid = windows::Win32::System::Threading::GetCurrentThreadId();
                let instance = GetModuleHandleW(None).unwrap_or_default();
                let class = wide(&format!("cHiDeScalerNeoTray-{}", std::process::id()));
                let wc = WNDCLASSW {
                    lpfnWndProc: Some(wnd_proc),
                    hInstance: instance.into(),
                    lpszClassName: PCWSTR(class.as_ptr()),
                    ..Default::default()
                };
                if RegisterClassW(&wc) == 0 && GetLastError() != ERROR_CLASS_ALREADY_EXISTS {
                    let _ = ready_tx.send((tid, 0));
                    return;
                }
                let Ok(hwnd) = CreateWindowExW(
                    WINDOW_EX_STYLE::default(),
                    PCWSTR(class.as_ptr()),
                    w!("cHiDeScaler-Neo tray"),
                    WINDOW_STYLE::default(),
                    0,
                    0,
                    0,
                    0,
                    Some(HWND_MESSAGE),
                    None,
                    Some(instance.into()),
                    None,
                ) else {
                    let _ = ready_tx.send((tid, 0));
                    return;
                };
                if let Ok(mut guard) = SHARED.get_or_init(|| Mutex::new(None)).lock() {
                    *guard = Some(Shared {
                        event_tx,
                        wake: Box::new(wake),
                        labels,
                        running: false,
                        hwnd: hwnd.0 as isize,
                    });
                }
                add_icon(hwnd);
                let _ = ready_tx.send((tid, hwnd.0 as isize));
                let mut msg = MSG::default();
                loop {
                    while let Ok(command) = command_rx.try_recv() {
                        match command {
                            Command::SetRunning(value) => {
                                if let Ok(mut g) = SHARED.get_or_init(|| Mutex::new(None)).lock() {
                                    if let Some(s) = g.as_mut() {
                                        s.running = value;
                                    }
                                }
                            }
                            Command::SetLabels(value) => {
                                if let Ok(mut g) = SHARED.get_or_init(|| Mutex::new(None)).lock() {
                                    if let Some(s) = g.as_mut() {
                                        s.labels = value;
                                    }
                                }
                            }
                            Command::Shutdown => {
                                delete_icon(hwnd);
                                let _ = DestroyWindow(hwnd);
                            }
                        }
                    }
                    let result = GetMessageW(&mut msg, None, 0, 0);
                    if result.0 <= 0 {
                        break;
                    }
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
                delete_icon(hwnd);
                if let Ok(mut guard) = SHARED.get_or_init(|| Mutex::new(None)).lock() {
                    *guard = None;
                }
            })
            .ok()?;
        let (thread_id, hwnd) = ready_rx.recv().ok()?;
        if hwnd == 0 {
            let _ = handle.join();
            return None;
        }
        log::info!("task-tray-started: hwnd={hwnd:#x}");
        Some(Self {
            rx,
            command_tx,
            thread_id,
            handle: Some(handle),
        })
    }

    fn wake(&self) {
        let hwnd = SHARED
            .get_or_init(|| Mutex::new(None))
            .lock()
            .ok()
            .and_then(|g| g.as_ref().map(|s| s.hwnd))
            .unwrap_or(0);
        if hwnd != 0 {
            unsafe {
                let _ = PostMessageW(
                    Some(HWND(hwnd as *mut _)),
                    WM_TRAY_WAKE,
                    WPARAM(0),
                    LPARAM(0),
                );
            }
        }
    }
    pub fn set_running(&self, running: bool) {
        let _ = self.command_tx.send(Command::SetRunning(running));
        self.wake();
    }
    pub fn set_labels(&self, labels: TrayLabels) {
        let _ = self.command_tx.send(Command::SetLabels(labels));
        self.wake();
    }
    pub fn stop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        let _ = self.command_tx.send(Command::Shutdown);
        self.wake();
        unsafe {
            let _ = PostThreadMessageW(self.thread_id, WM_TRAY_WAKE, WPARAM(0), LPARAM(0));
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
            log::info!("task-tray-stopped");
        }
    }
}

impl Drop for TrayThread {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notification_thread_starts_updates_and_stops_cleanly() {
        let mut tray = TrayThread::start(
            TrayLabels {
                show: "Show GUI".into(),
                exit: "Exit".into(),
            },
            || {},
        )
        .expect("notification-area thread");
        tray.set_running(true);
        tray.set_running(false);
        tray.stop();
        assert!(tray.handle.is_none());

        // Toggling either tray option recreates the controller in the same
        // process. The registered class is intentionally reusable.
        let mut restarted = TrayThread::start(
            TrayLabels {
                show: "Show GUI".into(),
                exit: "Exit".into(),
            },
            || {},
        )
        .expect("notification-area thread restart");
        restarted.stop();
    }

    #[test]
    fn version_four_callback_uses_low_word_event() {
        let packed = ((1u32 << 16) | WM_RBUTTONUP) as isize;
        assert_eq!((packed as u32) & 0xffff, WM_RBUTTONUP);
    }
}
