//! Win32 window + WGL context creation (the only place that talks to WGL).
//!
//! Used for both the hidden offscreen context (tests / engine warm-up) and the
//! visible overlay window. The overlay window style is deliberately
//! NON-layered: WS_EX_LAYERED forces a slow DWM composition path for GL
//! swapchains; WS_EX_TRANSPARENT alone provides click-through.

use anyhow::{Result, anyhow};
use std::ffi::CString;
use windows::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::Graphics::Gdi::{GetDC, HBRUSH, HDC};
use windows::Win32::Graphics::OpenGL::*;
use windows::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress, LoadLibraryW};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::PCWSTR;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_NCHITTEST => LRESULT(HTTRANSPARENT as isize), // overlay: clicks fall through
        WM_ERASEBKGND => LRESULT(1),
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn ensure_class(instance: HINSTANCE) -> Result<Vec<u16>> {
    let name = wide("cHiDeScalerNeoGL");
    let wc = WNDCLASSW {
        style: CS_OWNDC | CS_HREDRAW | CS_VREDRAW,
        lpfnWndProc: Some(wndproc),
        hInstance: instance,
        lpszClassName: PCWSTR(name.as_ptr()),
        hCursor: HCURSOR::default(),
        hbrBackground: HBRUSH::default(),
        ..Default::default()
    };
    unsafe { RegisterClassW(&wc) }; // 0 if already registered: fine
    Ok(name)
}

pub struct GlWindow {
    pub hwnd: HWND,
    pub hdc: HDC,
    pub hglrc: HGLRC,
}

// HWND/HDC/HGLRC are used from the single render thread only.
unsafe impl Send for GlWindow {}

impl GlWindow {
    /// Create a (hidden by default) window with a WGL context, made current.
    pub fn new(w: i32, h: i32, ex_style: WINDOW_EX_STYLE, style: WINDOW_STYLE) -> Result<Self> {
        unsafe {
            let instance: HINSTANCE = GetModuleHandleW(PCWSTR::null())?.into();
            let class = ensure_class(instance)?;
            let title = wide("cHiDeScaler-Neo");
            let hwnd = CreateWindowExW(
                ex_style,
                PCWSTR(class.as_ptr()),
                PCWSTR(title.as_ptr()),
                style,
                0,
                0,
                w,
                h,
                None,
                None,
                Some(instance),
                None,
            )?;
            let hdc = GetDC(Some(hwnd));
            if hdc.is_invalid() {
                let _ = DestroyWindow(hwnd);
                return Err(anyhow!("GetDC failed"));
            }
            let pfd = PIXELFORMATDESCRIPTOR {
                nSize: std::mem::size_of::<PIXELFORMATDESCRIPTOR>() as u16,
                nVersion: 1,
                dwFlags: PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER,
                iPixelType: PFD_TYPE_RGBA,
                cColorBits: 32,
                cDepthBits: 0,
                cStencilBits: 0,
                iLayerType: 0, // PFD_MAIN_PLANE
                ..Default::default()
            };
            let pf = ChoosePixelFormat(hdc, &pfd);
            if pf == 0 {
                let _ = DestroyWindow(hwnd);
                return Err(anyhow!("ChoosePixelFormat failed"));
            }
            SetPixelFormat(hdc, pf, &pfd)?;
            let hglrc = wglCreateContext(hdc)?;
            wglMakeCurrent(hdc, hglrc)?;
            Ok(Self { hwnd, hdc, hglrc })
        }
    }

    pub fn hidden(w: i32, h: i32) -> Result<Self> {
        Self::new(w, h, WS_EX_TOOLWINDOW, WS_POPUP) // never shown
    }

    pub fn make_current(&self) -> Result<()> {
        unsafe { wglMakeCurrent(self.hdc, self.hglrc)? };
        Ok(())
    }

    pub fn swap_buffers(&self) {
        unsafe {
            let _ = SwapBuffers(self.hdc);
        }
    }

    /// vsync off by default (lowest latency); call with 1 to enable.
    pub fn set_swap_interval(&self, interval: i32) {
        unsafe {
            let name = CString::new("wglSwapIntervalEXT").unwrap();
            if let Some(p) = wglGetProcAddress(windows::core::PCSTR(name.as_ptr() as _)) {
                let f: extern "system" fn(i32) -> i32 = std::mem::transmute(p);
                f(interval);
            }
        }
    }

    /// GL function loader for glow: wglGetProcAddress with opengl32.dll fallback.
    pub fn loader(&self, symbol: &str) -> *const std::ffi::c_void {
        unsafe {
            let c = CString::new(symbol).unwrap();
            if let Some(p) = wglGetProcAddress(windows::core::PCSTR(c.as_ptr() as _)) {
                let addr = p as usize;
                // Some drivers return small sentinel values for unsupported fns
                if addr > 3 && addr != usize::MAX {
                    return addr as *const _;
                }
            }
            static mut OPENGL32: usize = 0;
            if OPENGL32 == 0 {
                let dll = wide("opengl32.dll");
                if let Ok(m) = LoadLibraryW(PCWSTR(dll.as_ptr())) {
                    OPENGL32 = m.0 as usize;
                }
            }
            if OPENGL32 != 0 {
                let m = windows::Win32::Foundation::HMODULE(OPENGL32 as *mut _);
                if let Some(p) = GetProcAddress(m, windows::core::PCSTR(c.as_ptr() as _)) {
                    return p as usize as *const _;
                }
            }
            std::ptr::null()
        }
    }

    /// Pump pending messages without blocking (call from the render loop).
    pub fn pump_messages(&self) {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, Some(self.hwnd), 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}

impl Drop for GlWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = wglMakeCurrent(HDC::default(), HGLRC::default());
            let _ = wglDeleteContext(self.hglrc);
            let _ = DestroyWindow(self.hwnd);
        }
    }
}
