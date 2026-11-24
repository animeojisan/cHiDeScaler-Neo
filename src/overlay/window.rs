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
    // True when the native WGL HWND has already been reinserted into DWM's
    // visible tree at alpha=0 and is waiting for the first safe reveal.
    // This preserves the v235 composition behaviour while still allowing the
    // hard SW_HIDE used by the modern Stop path to retire stale AMD surfaces.
    staged_hidden: bool,
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
            staged_hidden: false,
        })
    }

    pub fn hwnd(&self) -> HWND {
        self.win.hwnd
    }

    pub fn size(&self) -> (i32, i32) {
        (self.width, self.height)
    }

    /// Reinsert a physically hidden Stop-era WGL HWND into DWM while it is
    /// still fully transparent. v235 never removed the overlay from DWM's
    /// composition tree; the later hard Stop fix did, and showing the full-
    /// screen HWND only at the reveal boundary can let AMD promote it before
    /// the small panel/cursor helper surfaces are composed. Stage it early at
    /// alpha=0 after geometry is restored, then reveal by alpha only.
    pub fn prepare_hidden_for_reveal(&mut self) {
        if self.visible || self.staged_hidden {
            return;
        }
        unsafe {
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
        }
        self.staged_hidden = true;
        log::info!(
            "overlay-hidden-stage-ready: hwnd={:#x} native_visible={} alpha=0 size={}x{}",
            self.win.hwnd.0 as isize,
            unsafe { IsWindowVisible(self.win.hwnd).as_bool() },
            self.width,
            self.height
        );
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
        self.staged_hidden = false;
    }

    /// While the layered window is fully transparent, overwrite both WGL
    /// back buffers with black.  A persistent AMD/DWM redirection surface can
    /// otherwise resurrect the previous capture session's last frame after a
    /// hard Stop -> Start even though all GL/DML resources were retired.
    pub fn blank_hidden_buffers(&mut self, gc: &GlContext, reason: &str) {
        unsafe {
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                LWA_ALPHA,
            );
            let gl = &gc.gl;
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            gl.viewport(0, 0, self.width.max(1), self.height.max(1));
            gl.clear_color(0.0, 0.0, 0.0, 1.0);
            gl.clear(glow::COLOR_BUFFER_BIT);
            self.win.swap_buffers();
            gl.clear(glow::COLOR_BUFFER_BIT);
            self.win.swap_buffers();
            self.win.pump_messages();
            let _ = DwmFlush();
        }
        log::info!(
            "overlay-hidden-buffer-reset: hwnd={:#x} reason={} size={}x{} buffers=2 alpha=0",
            self.win.hwnd.0 as isize,
            reason,
            self.width.max(1),
            self.height.max(1)
        );
    }

    pub fn hide(&mut self) {
        unsafe {
            // Logical hide for ordinary in-session transitions: retain the
            // transparent, click-through window in DWM's composition tree.
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                LWA_ALPHA,
            );
            let _ = DwmFlush();
        }
        self.visible = false;
        // Native HWND remains visible in DWM at alpha=0.
        self.staged_hidden = true;
    }

    /// Final capture-stop hide. Stop is a hard visual boundary: after it
    /// returns, no WGL/DWM surface from the magnified view may remain on the
    /// desktop. Alpha=0 plus off-screen parking was insufficient on the AMD
    /// reproduction (the old full-screen frame could remain scanned out), so
    /// physically remove the overlay HWND from the visible window tree here.
    /// The persistent WGL context is retained; show() restores the HWND while
    /// alpha is still 0 and only exposes it after the next frame is double-primed.
    pub fn hide_for_stop(&mut self) {
        unsafe {
            let _ = SetLayeredWindowAttributes(
                self.win.hwnd,
                windows::Win32::Foundation::COLORREF(0),
                0,
                LWA_ALPHA,
            );
            let _ = DwmFlush();
            let _ = ShowWindow(self.win.hwnd, SW_HIDE);
            let _ = SetWindowPos(
                self.win.hwnd,
                None,
                -32000,
                -32000,
                1,
                1,
                SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOZORDER,
            );
            let _ = DwmFlush();
            let native_visible = IsWindowVisible(self.win.hwnd).as_bool();
            if native_visible {
                log::error!(
                    "overlay-stop-hide-postcondition-failed: hwnd={:#x} native_visible=true",
                    self.win.hwnd.0 as isize
                );
            } else {
                log::info!(
                    "overlay-stop-hide-complete: hwnd={:#x} native_visible=false rect=(-32000,-32000 1x1)",
                    self.win.hwnd.0 as isize
                );
            }
        }
        self.width = 1;
        self.height = 1;
        self.visible = false;
        self.staged_hidden = false;
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    /// Commit already-swapped overlay content to the Desktop Window Manager.
    /// Used only for one-shot semantic updates such as a filter-chain edit on
    /// a paused/change-driven WGC source; ordinary playback stays asynchronous.
    pub fn flush_compositor(&self) {
        unsafe {
            let _ = DwmFlush();
        }
    }

    /// Geometry-only reposition used while the main GUI itself is TOPMOST.
    /// This is the modern stable path: do not disturb the established
    /// GUI > panel > overlay sibling ordering.
    pub fn reposition(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            let _ = SetWindowPos(
                self.win.hwnd,
                None,
                x,
                y,
                w,
                h,
                SWP_NOACTIVATE | SWP_NOSENDCHANGING | SWP_NOZORDER,
            );
        }
        self.width = w;
        self.height = h;
    }

    /// v235-compatible reposition for GUI-topmost OFF. v235 always supplied
    /// HWND_TOPMOST when placing the overlay. Later GUI-topmost fixes changed
    /// this globally to SWP_NOZORDER so the TOPMOST GUI would not flicker.
    /// Keep that newer behaviour only for GUI-topmost ON; when OFF, restore the
    /// original overlay placement contract exactly.
    pub fn reposition_v235_topmost(&mut self, x: i32, y: i32, w: i32, h: i32) {
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
        self.prepare_back_buffer(gc, tex)?;
        self.swap_prepared_back_buffer()
    }

    /// Render once before the display deadline; no swap or message pumping.
    pub fn prepare_back_buffer(&self, gc: &mut GlContext, tex: GpuTex) -> Result<()> {
        content_diag_present(gc, tex);
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
            // The present shader is opaque (alpha=1.0) and blending/depth are
            // disabled. If the content viewport covers the complete overlay,
            // the following fullscreen draw overwrites every color pixel, so
            // clearing the same surface first is redundant. Preserve the old
            // black clear whenever letterbox/pillarbox pixels are present.
            let full_cover = vx == 0 && vy == 0 && vw == ww && vh == wh;
            if !full_cover {
                gl.viewport(0, 0, ww, wh);
                gl.clear_color(0.0, 0.0, 0.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
            }
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
        Ok(())
    }

    pub fn swap_prepared_back_buffer(&self) -> Result<Duration> {
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

/// v939 diagnostic only (no behaviour change): is the PRESENTED picture changing?
/// At most once per second the texture about to be presented is reduced to 64x36 and
/// read back (one small ReadPixels), hashed, and compared with the previous sample.
/// When the hash stays identical for >= 1 s while presents keep running, the output is
/// a frozen picture even though the presentation clock is alive (user report 2026-11-17:
/// XS at 1080p showed the same image for ~10 s right after start). Logs:
///   content-diag-present-static: static_ms presents   (warn, when a frozen run passes 1 s, then every 5 s)
///   content-diag-present-static-end: static_ms        (info, when the picture changes again)
/// `CHIDESCALER_CONTENT_DIAG=off` disables it.
fn content_diag_present(gc: &mut GlContext, tex: GpuTex) {
    use std::sync::Mutex;
    struct St {
        last_sample: Instant,
        hash: u64,
        same_since: Option<Instant>,
        presents: u64,
        presents_at_change: u64,
        warned_ms: u128,
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static ST: Mutex<Option<St>> = Mutex::new(None);
    if !*ENABLED.get_or_init(|| std::env::var("CHIDESCALER_CONTENT_DIAG").ok().as_deref() != Some("off")) {
        return;
    }
    if tex.w() < 64 || tex.h() < 36 {
        return;
    }
    let Ok(mut guard) = ST.lock() else { return };
    let now = Instant::now();
    let st = guard.get_or_insert_with(|| St {
        last_sample: now - Duration::from_secs(2),
        hash: 0,
        same_since: None,
        presents: 0,
        presents_at_change: 0,
        warned_ms: 0,
    });
    st.presents += 1;
    // a gap of > 3 s without presents = a new session: start over
    if now.duration_since(st.last_sample) > Duration::from_secs(3) {
        st.same_since = None;
        st.hash = 0;
        st.warned_ms = 0;
        st.presents_at_change = st.presents;
    }
    if now.duration_since(st.last_sample) < Duration::from_millis(1000) {
        return;
    }
    st.last_sample = now;
    let Ok(small) = crate::render::scaler::resample(gc, tex, 64, 36, crate::render::scaler::Kernel::Bilinear) else {
        return;
    };
    let px = gc.download_rgba8(small);
    gc.recycle(small);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in px {
        h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
    }
    if h == st.hash {
        let since = *st.same_since.get_or_insert(now);
        let ms = now.duration_since(since).as_millis() + 1000;
        if ms >= 1000 && (st.warned_ms == 0 || ms >= st.warned_ms + 5000) {
            st.warned_ms = ms;
            log::warn!(
                "content-diag-present-static: static_ms~{} presents_since_change={} size={}x{} (presentation alive, picture unchanged)",
                ms,
                st.presents - st.presents_at_change,
                tex.w(),
                tex.h()
            );
        }
    } else {
        if let Some(since) = st.same_since.take() {
            log::info!(
                "content-diag-present-static-end: static_ms~{} presents={}",
                now.duration_since(since).as_millis() + 1000,
                st.presents - st.presents_at_change
            );
        }
        st.hash = h;
        st.warned_ms = 0;
        st.presents_at_change = st.presents;
    }
}

#[cfg(test)]
mod prepared_present_tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn chain_transition_gpu_snapshot_survives_source_retirement() {
        let overlay = OverlayWindow::new(64, 64).unwrap();
        let gl = Rc::new(unsafe { glow::Context::from_loader_function(|n| overlay.win.loader(n)) });
        let mut gc = GlContext::new(gl);
        for (w, h) in [(960, 540), (1280, 720), (1920, 1080)] {
            let pixels: Vec<u8> = (0..w*h).flat_map(|i| [(i%251) as u8, ((i*3)%251) as u8, ((i*7)%251) as u8, (i%256) as u8]).collect();
            let src = gc.upload_rgba8(w, h, &pixels);
            let mut cpu_ms = Vec::new();
            let mut gpu_ms = Vec::new();
            for round in 0..12 {
                let start = std::time::Instant::now();
                let cpu = gc.download_rgba8(src);
                let old = gc.upload_rgba8(w, h, &cpu);
                gc.finish();
                if round > 1 { cpu_ms.push(start.elapsed().as_secs_f64()*1000.0); }
                assert_eq!(pixels, gc.download_rgba8(old));
                gc.recycle(old);
                let start = std::time::Instant::now();
                let owned = gc.snapshot_transition_rgba8(src).unwrap();
                if round > 1 { gpu_ms.push(start.elapsed().as_secs_f64()*1000.0); }
                assert_eq!(pixels, gc.download_rgba8(owned));
                gc.recycle(owned);
                gc.trim_transient_pool(0);
            }
            let owned = gc.snapshot_transition_rgba8(src).unwrap();
            gc.recycle(src);
            gc.trim_transient_pool(0);
            assert_eq!(pixels, gc.download_rgba8(owned));
            gc.release_frame(&[owned]);
            assert_eq!(pixels, gc.download_rgba8(owned));
            cpu_ms.sort_by(f64::total_cmp); gpu_ms.sort_by(f64::total_cmp);
            println!("transition-snapshot-bench: size={}x{} cpu_median_ms={:.3} gpu_median_ms={:.3} pixels=exact", w, h, cpu_ms[5], gpu_ms[5]);
            gc.recycle(owned); gc.trim_transient_pool(0);
            assert_eq!(gc.live_texture_count(), 0);
        }
    }

    #[test]
    fn prepared_back_buffer_matches_regular_draw_without_changing_texture() {
        let overlay=OverlayWindow::new(128,96).unwrap();
        let gl=Rc::new(unsafe{glow::Context::from_loader_function(|n|overlay.win.loader(n))});
        let mut gc=GlContext::new(gl);
        for (w,h) in [(128,96),(96,96),(64,32)] {
            let pixels:Vec<u8>=(0..w*h).flat_map(|i|[(i%251)as u8,((i*3)%251)as u8,((i*7)%251)as u8,255]).collect();
            let tex=gc.upload_rgba8(w,h,&pixels);
            overlay.present(&mut gc,tex).unwrap();
            overlay.prepare_back_buffer(&mut gc,tex).unwrap();gc.finish();
            let expected=overlay.read_front_pixels(&gc);
            overlay.swap_prepared_back_buffer().unwrap();
            overlay.prepare_back_buffer(&mut gc,tex).unwrap();gc.finish();
            assert_eq!(expected,overlay.read_front_pixels(&gc));
            assert_eq!(pixels,gc.download_rgba8(tex));
            assert!(expected.chunks_exact(4).any(|p|p[0]>0));
        }
    }
}
