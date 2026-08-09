//! Click-through overlay window + GL presentation (letterboxed).
//!
//! Style notes:
//! - WS_EX_LAYERED | WS_EX_TRANSPARENT (alpha=255) is REQUIRED for real
//!   cross-process click-through. WS_EX_TRANSPARENT/HTTRANSPARENT alone only
//!   passes clicks to windows of the SAME thread — an automated click-routing
//!   test proved the overlay swallowed clicks to the source without LAYERED
//!   (WindowFromPoint returned the overlay). Layered alpha=255 is composited
//!   natively by modern DWM; presentation fps is verified by test.
//! - WS_EX_NOACTIVATE|WS_EX_TOOLWINDOW keeps it out of focus and Alt-Tab.
//! - vsync off by default for lowest latency.

use crate::platform::gl_window::GlWindow;
use crate::render::gl::{GlContext, GpuTex};
use anyhow::Result;
use glow::HasContext;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Dwm::DwmFlush;
use windows::Win32::UI::WindowsAndMessaging::*;

const PRESENT_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = vec4(clamp(texture(tex, vec2(v_uv.x, 1.0 - v_uv.y)).rgb, 0.0, 1.0), 1.0); }\n";

pub struct OverlayWindow {
    pub win: GlWindow,
    width: i32,
    height: i32,
    visible: bool,
}

impl OverlayWindow {
    pub fn new(w: i32, h: i32) -> Result<Self> {
        let ex =
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST;
        let win = GlWindow::new(w, h, ex, WS_POPUP)?;
        unsafe {
            // Start transparent while hidden. A reused DWM surface can retain
            // the previous session's last frame even after hidden SwapBuffers.
            windows::Win32::UI::WindowsAndMessaging::SetLayeredWindowAttributes(
                win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                windows::Win32::UI::WindowsAndMessaging::LWA_ALPHA,
            )?;
        }
        win.set_swap_interval(0);
        Ok(Self {
            win,
            width: w,
            height: h,
            visible: false,
        })
    }

    pub fn hwnd(&self) -> HWND {
        self.win.hwnd
    }

    pub fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    pub fn show(&mut self) {
        unsafe {
            // Keep an already-composed window in DWM's tree between sessions.
            // Re-entering after SW_HIDE lets DWM briefly reuse the previous
            // redirection surface even though a new hidden buffer was swapped.
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                LWA_ALPHA,
            );
            if !IsWindowVisible(self.win.hwnd).as_bool() {
                let _ = ShowWindow(self.win.hwnd, SW_SHOWNOACTIVATE);
            }
            let _ = DwmFlush();
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                255,
                LWA_ALPHA,
            );
            // The source window may be resized immediately after this call.
            // Commit the opaque transition frame first so DWM can never expose
            // the source application's intermediate resize surface.
            let _ = DwmFlush();
        }
        self.visible = true;
    }

    pub fn hide(&mut self) {
        unsafe {
            // Logical hide only: retain the transparent, click-through window
            // in DWM's composition tree. Physically hiding and later showing
            // this persistent WGL HWND resurrected the previous session's
            // front surface for one composition on AMD.
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                LWA_ALPHA,
            );
            let _ = DwmFlush();
        }
        self.visible = false;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn reposition(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            let _ = SetWindowPos(
                self.win.hwnd,
                Some(HWND_TOPMOST),
                x,
                y,
                w,
                h,
                SWP_NOACTIVATE | SWP_NOSENDCHANGING,
            );
        }
        self.width = w;
        self.height = h;
    }

    pub fn current_rect(&self) -> (i32, i32, i32, i32) {
        unsafe {
            let mut r = windows::Win32::Foundation::RECT::default();
            let _ = GetWindowRect(self.win.hwnd, &mut r);
            (r.left, r.top, r.right - r.left, r.bottom - r.top)
        }
    }

    /// Present `tex` letterboxed into the window (aspect preserved, black bars).
    pub fn present(&self, gc: &mut GlContext, tex: GpuTex) -> Result<()> {
        self.present_with_swap_timing(gc, tex).map(|_| ())
    }

    /// Same presentation path, but return only the time spent inside
    /// SwapBuffers itself. The interpolation scheduler uses this to distinguish
    /// a real compositor/vblank wait from shader/resample work and Win32 message
    /// pumping; feeding those unrelated costs back into the x5 phase clock was
    /// the source of the ~100 fps plateau on otherwise under-utilised GPUs.
    pub fn present_with_swap_timing(&self, gc: &mut GlContext, tex: GpuTex) -> Result<Duration> {
        let prog = gc
            .program(PRESENT_FRAG)
            .map_err(|e| anyhow::anyhow!("present shader: {e}"))?;
        let (ww, wh) = (self.width.max(1), self.height.max(1));
        let (fw, fh) = (tex.w() as f64, tex.h() as f64);
        let scale = (ww as f64 / fw).min(wh as f64 / fh);
        let vw = (fw * scale).round() as i32;
        let vh = (fh * scale).round() as i32;
        let vx = (ww - vw) / 2;
        let vy = (wh - vh) / 2;
        let gl = gc.gl.clone();
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, ww, wh);
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            gl.viewport(vx, vy, vw.max(1), vh.max(1));
            gl.use_program(Some(prog));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex.tex));
            // The engine has already resized `final_tex` to the exact content
            // viewport with the user-selected kernel. Sampling that result
            // with LINEAR again softens even a 1:1, filter-free capture on
            // some drivers. Keep the final copy pixel-exact whenever no
            // geometric scaling remains; LINEAR is only a safety fallback for
            // a genuine size mismatch (for example, a transient resize).
            let filter = if tex.w() == vw && tex.h() == vh {
                glow::NEAREST
            } else {
                glow::LINEAR
            } as i32;
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MIN_FILTER, filter);
            gl.tex_parameter_i32(glow::TEXTURE_2D, glow::TEXTURE_MAG_FILTER, filter);
            if let Some(loc) = gl.get_uniform_location(prog, "tex") {
                gl.uniform_1_i32(Some(&loc), 0);
            }
            gl.bind_vertex_array(Some(gc.quad_vao));
            gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
            gl.bind_vertex_array(None);
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MIN_FILTER,
                glow::NEAREST as i32,
            );
            gl.tex_parameter_i32(
                glow::TEXTURE_2D,
                glow::TEXTURE_MAG_FILTER,
                glow::NEAREST as i32,
            );
        }
        let swap_started = Instant::now();
        self.win.swap_buffers();
        let swap_elapsed = swap_started.elapsed();
        self.win.pump_messages();
        Ok(swap_elapsed)
    }

    /// Read back the window framebuffer (tests only).
    pub fn read_front_pixels(&self, gc: &GlContext) -> Vec<u8> {
        let gl = &gc.gl;
        let (w, h) = (self.width as usize, self.height as usize);
        let mut buf = vec![0u8; w * h * 4];
        unsafe {
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.read_buffer(glow::BACK);
            gl.read_pixels(
                0,
                0,
                self.width,
                self.height,
                glow::RGBA,
                glow::UNSIGNED_BYTE,
                glow::PixelPackData::Slice(Some(&mut buf)),
            );
        }
        buf
    }
}
