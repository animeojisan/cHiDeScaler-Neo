//! Windows.Graphics.Capture frame source (event-driven, newest-wins).
//!
//! WGC only delivers frames when the source changes; the pipeline re-presents
//! the last frame while idle, so no pacing thread lives here. The callback
//! copies the newest frame into a double buffer under a mutex and signals a
//! condvar; the render loop waits on it (arrival-driven, no polling).

use anyhow::{Result, anyhow};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_MAP_READ, D3D11_MAPPED_SUBRESOURCE,
    D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
use windows::Win32::System::Threading::{
    AVRT_PRIORITY_HIGH, AvRevertMmThreadCharacteristics, AvSetMmThreadCharacteristicsW,
    AvSetMmThreadPriority, GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL,
};
use windows::core::PCWSTR;
use windows_capture::capture::{Context, GraphicsCaptureApiHandler};
use windows_capture::frame::Frame;
use windows_capture::graphics_capture_api::{GraphicsCaptureApi, InternalCaptureControl};
use windows_capture::settings::{
    ColorFormat, CursorCaptureSettings, DirtyRegionSettings, DrawBorderSettings,
    MinimumUpdateIntervalSettings, SecondaryWindowSettings, Settings,
};
use windows_capture::window::Window;

const CAPTURE_QUEUE_HARD_MAX: usize = 8;
const MAX_NATIVE_CAPTURE_FPS: f64 = 240.0;

fn native_capture_interval() -> MinimumUpdateIntervalSettings {
    if GraphicsCaptureApi::is_minimum_update_interval_supported().unwrap_or(false) {
        MinimumUpdateIntervalSettings::Custom(Duration::from_secs_f64(1.0 / MAX_NATIVE_CAPTURE_FPS))
    } else {
        MinimumUpdateIntervalSettings::Default
    }
}

#[derive(Default, Clone)]
pub struct FrameBuf {
    pub w: i32,
    pub h: i32,
    pub data: Vec<u8>, // tightly packed RGBA8, or RGBA16F bytes when hdr
    pub hdr: bool,
    pub seq: u64,
    pub received_at: Option<Instant>,
    pub source_time_100ns: Option<i64>,
}

pub struct Shared {
    pub buf: Mutex<FrameBuf>,
    pub queue: Mutex<VecDeque<FrameBuf>>,
    /// Reuse full-frame allocations returned by the render thread. This keeps
    /// the interpolation callback to one mapped-texture copy per frame.
    pub recycled_data: Mutex<Vec<Vec<u8>>>,
    pub ready: Condvar,
    pub seq: AtomicU64,
    pub queued_dropped: AtomicU64,
    pub alive: AtomicBool,
    pub queue_enabled: AtomicBool,
    /// crop to the client area (exclude title bar/borders); no-op for
    /// borderless windows. Recomputed per frame (cheap) so resizes track.
    pub client_only: AtomicBool,
    pub hwnd: std::sync::atomic::AtomicIsize,
    pub hdr: AtomicBool,
    /// Stable raw WGC geometry for monitor-cover fullscreen sessions. When
    /// present, only a +/-1px compositor drift is normalized; material size
    /// changes are left untouched for the engine's normal resize path.
    pub pixel_exact_raw_hint: Option<(u32, u32)>,
}

struct Handler {
    shared: Arc<Shared>,
    scratch: Vec<u8>,
    staging: Option<ID3D11Texture2D>,
    staging_key: Option<(u32, u32, ColorFormat)>,
    crop_cache: Option<(u32, u32, bool, isize, Option<ClientCrop>)>,
    last_crop_log: Option<(u32, u32, u32, u32, u32, u32)>,
    last_pad_log: Option<(u32, u32, u32, u32)>,
    last_missing_crop_log: Option<(u32, u32)>,
    last_pixel_lock_log: Option<(u32, u32, u32, u32)>,
    diag_frames: u32,
    diag_callback_ms: f64,
    diag_callback_max_ms: f64,
    diag_arrival_ms: f64,
    diag_arrival_max_ms: f64,
    diag_source_ms: f64,
    diag_source_max_ms: f64,
    last_received_at: Option<Instant>,
    last_source_time_100ns: Option<i64>,
    mmcss_handle: Option<isize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ClientCrop {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    reference: &'static str,
}

fn client_crop_for_frame(
    full: (u32, u32),
    client: (i32, i32, i32, i32),
    outer: Option<(i32, i32, i32, i32)>,
    dwm: Option<(i32, i32, i32, i32)>,
) -> Option<ClientCrop> {
    let mut best: Option<(u64, ClientCrop)> = None;
    for (reference, rect) in [("outer", outer), ("dwm", dwm)] {
        let Some((rx, ry, rw, rh)) = rect else {
            continue;
        };
        if rw <= 0 || rh <= 0 || client.2 <= 0 || client.3 <= 0 {
            continue;
        }
        let x0 = client.0 - rx;
        let y0 = client.1 - ry;
        let x1 = x0 + client.2;
        let y1 = y0 + client.3;
        if x0 < 0 || y0 < 0 || x1 <= x0 || y1 <= y0 || x1 as u32 > full.0 || y1 as u32 > full.1 {
            continue;
        }
        let score =
            (rw as i64 - full.0 as i64).unsigned_abs() + (rh as i64 - full.1 as i64).unsigned_abs();
        let crop = ClientCrop {
            x0: x0 as u32,
            y0: y0 as u32,
            x1: x1 as u32,
            y1: y1 as u32,
            reference,
        };
        if best.is_none_or(|(best_score, _)| score < best_score) {
            best = Some((score, crop));
        }
    }
    best.map(|(_, crop)| crop)
}

/// Extend only the right/bottom edge by replicating the nearest source pixel.
/// This is used by the fullscreen pixel-geometry lock before the ordinary
/// even-size pad, so a transient -1px WGC content-size drift never triggers a
/// rescale or changes the sampling origin.
fn pad_edge_to_size(
    data: &mut Vec<u8>,
    width: u32,
    height: u32,
    target_width: u32,
    target_height: u32,
    bytes_per_pixel: usize,
) -> (u32, u32) {
    if width == 0
        || height == 0
        || bytes_per_pixel == 0
        || target_width < width
        || target_height < height
    {
        return (width, height);
    }
    let old_row = width as usize * bytes_per_pixel;
    let new_row = target_width as usize * bytes_per_pixel;
    if target_width != width {
        data.resize(new_row * height as usize, 0);
        for y in (0..height as usize).rev() {
            let src = y * old_row;
            let dst = y * new_row;
            data.copy_within(src..src + old_row, dst);
            let last = dst + old_row - bytes_per_pixel;
            for x in width as usize..target_width as usize {
                let out = dst + x * bytes_per_pixel;
                data.copy_within(last..last + bytes_per_pixel, out);
            }
        }
    }
    if target_height != height {
        let old_len = new_row * height as usize;
        data.resize(new_row * target_height as usize, 0);
        for y in height as usize..target_height as usize {
            let dst = y * new_row;
            data.copy_within(old_len - new_row..old_len, dst);
        }
    }
    (target_width, target_height)
}

/// Preserve every captured source pixel while satisfying even-sized ONNX and
/// shader pipelines. Repeat only the nearest edge pixel; black padding would
/// create an artificial high-contrast boundary for sharpening/CNN filters.
fn pad_edge_to_even(
    data: &mut Vec<u8>,
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
) -> (u32, u32) {
    if width == 0 || height == 0 || bytes_per_pixel == 0 {
        return (width, height);
    }
    let padded_width = width + width % 2;
    let padded_height = height + height % 2;
    pad_edge_to_size(
        data,
        width,
        height,
        padded_width,
        padded_height,
        bytes_per_pixel,
    )
}

impl Handler {
    fn copy_frame_to_vec(
        &mut self,
        frame: &Frame<'_>,
        crop: Option<ClientCrop>,
        raw_lock_size: Option<(u32, u32)>,
        destination: &mut Vec<u8>,
    ) -> Result<(i32, i32)> {
        let (x0, y0, x1, y1) = crop
            .map(|crop| (crop.x0, crop.y0, crop.x1, crop.y1))
            .unwrap_or((0, 0, frame.width(), frame.height()));
        let width = x1 - x0;
        let height = y1 - y0;
        let format = frame.color_format();
        let key = (width, height, format);
        if self.staging_key != Some(key) {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: width,
                Height: height,
                MipLevels: 1,
                ArraySize: 1,
                Format: frame.desc().Format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut staging = None;
            unsafe {
                frame
                    .device()
                    .CreateTexture2D(&desc, None, Some(&mut staging))?;
            }
            self.staging = staging;
            self.staging_key = Some(key);
            log::debug!(
                "WGC persistent staging texture created: {}x{} format={format:?}",
                width,
                height
            );
        }
        let staging = self.staging.as_ref().expect("staging texture");
        unsafe {
            if x0 == 0 && y0 == 0 && x1 == frame.width() && y1 == frame.height() {
                frame
                    .device_context()
                    .CopyResource(staging, frame.as_raw_texture());
            } else {
                let region = D3D11_BOX {
                    left: x0,
                    top: y0,
                    front: 0,
                    right: x1,
                    bottom: y1,
                    back: 1,
                };
                frame.device_context().CopySubresourceRegion(
                    staging,
                    0,
                    0,
                    0,
                    0,
                    frame.as_raw_texture(),
                    0,
                    Some(&region),
                );
            }
        }
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            frame
                .device_context()
                .Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        }
        let bytes_per_pixel = if format == ColorFormat::Rgba16F { 8 } else { 4 };
        let row_bytes = width as usize * bytes_per_pixel;
        destination.resize(row_bytes * height as usize, 0);
        for y in 0..height as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped.pData.cast::<u8>().add(y * mapped.RowPitch as usize),
                    destination.as_mut_ptr().add(y * row_bytes),
                    row_bytes,
                );
            }
        }
        unsafe {
            frame.device_context().Unmap(staging, 0);
        }
        let (width, height) = if let Some((lock_w, lock_h)) = raw_lock_size {
            pad_edge_to_size(destination, width, height, lock_w, lock_h, bytes_per_pixel)
        } else {
            (width, height)
        };
        let (width, height) = pad_edge_to_even(destination, width, height, bytes_per_pixel);
        Ok((width as i32, height as i32))
    }
}

impl Drop for Handler {
    fn drop(&mut self) {
        if let Some(handle) = self.mmcss_handle.take() {
            unsafe {
                let _ = AvRevertMmThreadCharacteristics(HANDLE(handle as *mut _));
            }
        }
    }
}

impl GraphicsCaptureApiHandler for Handler {
    type Flags = Arc<Shared>;
    type Error = anyhow::Error;

    fn new(ctx: Context<Self::Flags>) -> Result<Self, Self::Error> {
        // This is the free-threaded WGC callback, not the GUI thread. A small
        // priority boost reduces missed frame-pool deadlines when a
        // mid-range system is simultaneously decoding and running DirectML.
        // It is thread-local, lasts only for this capture session, and does
        // not alter process/system policy.
        let mmcss_handle = unsafe {
            if SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL).is_err() {
                log::debug!(
                    "WGC callback priority boost unavailable; continuing at normal priority"
                );
            }
            let task_name: Vec<u16> = "Capture".encode_utf16().chain([0]).collect();
            let mut task_index = 0u32;
            match AvSetMmThreadCharacteristicsW(PCWSTR(task_name.as_ptr()), &mut task_index) {
                Ok(handle) => {
                    let _ = AvSetMmThreadPriority(handle, AVRT_PRIORITY_HIGH);
                    Some(handle.0 as isize)
                }
                Err(error) => {
                    log::debug!("WGC MMCSS registration unavailable: {error}");
                    None
                }
            }
        };
        Ok(Self {
            shared: ctx.flags,
            scratch: Vec::new(),
            staging: None,
            staging_key: None,
            crop_cache: None,
            last_crop_log: None,
            last_pad_log: None,
            last_missing_crop_log: None,
            last_pixel_lock_log: None,
            diag_frames: 0,
            diag_callback_ms: 0.0,
            diag_callback_max_ms: 0.0,
            diag_arrival_ms: 0.0,
            diag_arrival_max_ms: 0.0,
            diag_source_ms: 0.0,
            diag_source_max_ms: 0.0,
            last_received_at: None,
            last_source_time_100ns: None,
            mmcss_handle,
        })
    }

    fn on_frame_arrived(
        &mut self,
        frame: &mut Frame,
        _control: InternalCaptureControl,
    ) -> Result<(), Self::Error> {
        let received_at = Instant::now();
        let source_time_100ns = frame.timestamp().ok().map(|t| t.Duration);
        let full_w = frame.width();
        let full_h = frame.height();
        let hwnd = self.shared.hwnd.load(Ordering::Relaxed);
        // Monitor-cover fullscreen sessions have a stable session-origin
        // geometry. WGC/DWM can nevertheless report a one-pixel content-size
        // drift while the source HWND itself has not resized. Normalize only
        // that +/-1px WGC-only drift. If the HWND outer rect really changed,
        // even by one pixel, leave it authoritative for the engine's normal
        // source-resize path. The Win32 rect query therefore runs only on an
        // actual +/-1px mismatch, not on every capture callback.
        let pixel_lock = self
            .shared
            .pixel_exact_raw_hint
            .and_then(|(lock_w, lock_h)| {
                let dw = full_w.abs_diff(lock_w);
                let dh = full_h.abs_diff(lock_h);
                if dw > 1 || dh > 1 {
                    return None;
                }
                if (full_w, full_h) == (lock_w, lock_h) {
                    return Some((lock_w, lock_h));
                }
                let source_still_locked = crate::platform::win32::window_rect(hwnd)
                    .is_some_and(|(_, _, w, h)| w == lock_w as i32 && h == lock_h as i32);
                source_still_locked.then_some((lock_w, lock_h))
            });
        let pixel_lock_crop = pixel_lock.and_then(|(lock_w, lock_h)| {
            if full_w > lock_w || full_h > lock_h {
                Some(ClientCrop {
                    x0: 0,
                    y0: 0,
                    x1: full_w.min(lock_w),
                    y1: full_h.min(lock_h),
                    reference: "fullscreen-pixel-lock",
                })
            } else {
                None
            }
        });
        if let Some((lock_w, lock_h)) = pixel_lock {
            if (full_w, full_h) != (lock_w, lock_h) {
                let key = (full_w, full_h, lock_w, lock_h);
                if self.last_pixel_lock_log != Some(key) {
                    log::info!(
                        "wgc-pixel-exact-normalize: observed={}x{} locked={}x{} action=right/bottom-only trim-or-edge-replicate no-resize",
                        full_w,
                        full_h,
                        lock_w,
                        lock_h
                    );
                    self.last_pixel_lock_log = Some(key);
                }
            }
        }
        // WGC frame bounds vary by window type. Most windows match DWM's
        // extended frame, while some PIP windows match the Win32 outer rect.
        // Select the coordinate reference whose dimensions match this frame.
        let client_only = self.shared.client_only.load(Ordering::Relaxed);
        let cache_key_matches = self.crop_cache.is_some_and(|cached| {
            cached.0 == full_w && cached.1 == full_h && cached.2 == client_only && cached.3 == hwnd
        });
        let client_crop = if cache_key_matches {
            self.crop_cache.and_then(|cached| cached.4)
        } else if client_only {
            crate::platform::win32::client_rect_on_screen(hwnd).and_then(|client| {
                client_crop_for_frame(
                    (full_w, full_h),
                    client,
                    crate::platform::win32::window_rect(hwnd),
                    crate::platform::win32::extended_frame_bounds(hwnd),
                )
            })
        } else {
            None
        };
        if !cache_key_matches {
            self.crop_cache = Some((full_w, full_h, client_only, hwnd, client_crop));
        }
        if client_only
            && client_crop.is_none()
            && self.last_missing_crop_log != Some((full_w, full_h))
        {
            log::warn!(
                "wgc-client-crop unavailable: frame={}x{}; using full frame",
                full_w,
                full_h
            );
            self.last_missing_crop_log = Some((full_w, full_h));
        }
        let base_size = pixel_lock.unwrap_or_else(|| {
            client_crop
                .map(|crop| (crop.x1 - crop.x0, crop.y1 - crop.y0))
                .unwrap_or((full_w, full_h))
        });
        let crop = pixel_lock_crop.or(client_crop);
        if let Some(crop) = crop {
            let key = (full_w, full_h, crop.x0, crop.y0, crop.x1, crop.y1);
            if crop.reference != "fullscreen-pixel-lock" && self.last_crop_log != Some(key) {
                log::info!(
                    "wgc-client-crop: frame={}x{} reference={} crop=({}, {})-({}, {}) result={}x{}",
                    full_w,
                    full_h,
                    crop.reference,
                    crop.x0,
                    crop.y0,
                    crop.x1,
                    crop.y1,
                    crop.x1 - crop.x0,
                    crop.y1 - crop.y0
                );
                self.last_crop_log = Some(key);
            }
        }
        let valid_crop = crop.filter(|crop| {
            crop.x1 <= full_w && crop.y1 <= full_h && crop.x1 > crop.x0 && crop.y1 > crop.y0
        });
        let mut frame_data = self
            .shared
            .recycled_data
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| std::mem::take(&mut self.scratch));
        let (w, h) = self.copy_frame_to_vec(frame, valid_crop, pixel_lock, &mut frame_data)?;
        let pad_key = (base_size.0, base_size.1, w as u32, h as u32);
        if (w as u32, h as u32) != base_size && self.last_pad_log != Some(pad_key) {
            log::info!(
                "wgc-even-pad: original={}x{} result={}x{} method=right/bottom edge replication (no crop, no resize)",
                base_size.0,
                base_size.1,
                w,
                h
            );
            self.last_pad_log = Some(pad_key);
        }
        let seq = self.shared.seq.fetch_add(1, Ordering::AcqRel) + 1;
        if self.shared.queue_enabled.load(Ordering::Relaxed) {
            let mut queued = FrameBuf {
                w,
                h,
                hdr: self.shared.hdr.load(Ordering::Relaxed),
                seq,
                received_at: Some(received_at),
                source_time_100ns,
                ..FrameBuf::default()
            };
            queued.data = frame_data;
            let mut q = self.shared.queue.lock().unwrap();
            q.push_back(queued);
            while q.len() > CAPTURE_QUEUE_HARD_MAX {
                if let Some(mut dropped) = q.pop_front() {
                    dropped.data.clear();
                    self.shared.recycled_data.lock().unwrap().push(dropped.data);
                }
                self.shared.queued_dropped.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            let mut buf = self.shared.buf.lock().unwrap();
            buf.w = w;
            buf.h = h;
            buf.hdr = self.shared.hdr.load(Ordering::Relaxed);
            buf.seq = seq;
            buf.received_at = Some(received_at);
            buf.source_time_100ns = source_time_100ns;
            std::mem::swap(&mut buf.data, &mut frame_data);
            frame_data.clear();
            self.scratch = frame_data;
        }
        self.shared.ready.notify_all();
        if crate::logging::diagnostics_enabled() {
            let callback_ms = received_at.elapsed().as_secs_f64() * 1000.0;
            self.diag_frames += 1;
            self.diag_callback_ms += callback_ms;
            self.diag_callback_max_ms = self.diag_callback_max_ms.max(callback_ms);
            if let Some(previous) = self.last_received_at.replace(received_at) {
                let interval_ms = received_at.duration_since(previous).as_secs_f64() * 1000.0;
                self.diag_arrival_ms += interval_ms;
                self.diag_arrival_max_ms = self.diag_arrival_max_ms.max(interval_ms);
            }
            if let Some(current) = source_time_100ns {
                if let Some(previous) = self.last_source_time_100ns.replace(current) {
                    let interval_ms = current.saturating_sub(previous) as f64 / 10_000.0;
                    self.diag_source_ms += interval_ms;
                    self.diag_source_max_ms = self.diag_source_max_ms.max(interval_ms);
                }
            }
            if self.diag_frames >= 120 {
                let intervals = (self.diag_frames - 1).max(1) as f64;
                log::debug!(
                    "wgc-callback: frames={} callback_avg_ms={:.2} callback_max_ms={:.2} arrival_avg_ms={:.2} arrival_max_ms={:.2} source_avg_ms={:.2} source_max_ms={:.2}",
                    self.diag_frames,
                    self.diag_callback_ms / self.diag_frames as f64,
                    self.diag_callback_max_ms,
                    self.diag_arrival_ms / intervals,
                    self.diag_arrival_max_ms,
                    self.diag_source_ms / intervals,
                    self.diag_source_max_ms
                );
                self.diag_frames = 0;
                self.diag_callback_ms = 0.0;
                self.diag_callback_max_ms = 0.0;
                self.diag_arrival_ms = 0.0;
                self.diag_arrival_max_ms = 0.0;
                self.diag_source_ms = 0.0;
                self.diag_source_max_ms = 0.0;
            }
        } else if self.diag_frames != 0
            || self.last_received_at.is_some()
            || self.last_source_time_100ns.is_some()
        {
            // Reset once when diagnostics are turned off; do not add per-frame
            // writes to the default lightweight capture callback.
            self.diag_frames = 0;
            self.diag_callback_ms = 0.0;
            self.diag_callback_max_ms = 0.0;
            self.diag_arrival_ms = 0.0;
            self.diag_arrival_max_ms = 0.0;
            self.diag_source_ms = 0.0;
            self.diag_source_max_ms = 0.0;
            self.last_received_at = None;
            self.last_source_time_100ns = None;
        }
        Ok(())
    }

    fn on_closed(&mut self) -> Result<(), Self::Error> {
        self.shared.alive.store(false, Ordering::Release);
        self.shared.ready.notify_all();
        Ok(())
    }
}

pub struct WgcSource {
    shared: Arc<Shared>,
    control: Option<windows_capture::capture::CaptureControl<Handler, anyhow::Error>>,
    last_seq: u64,
}

impl WgcSource {
    /// Start capturing `hwnd`. WGC remains at the stable native/high-Hz request;
    /// `fps_cap` is applied by the engine before optional pre-processing and the filter chain.
    /// `client_only` crops away title bar/borders.
    pub fn start(hwnd: isize, fps_cap: Option<u32>, client_only: bool) -> Result<Self> {
        Self::start_fmt(hwnd, fps_cap, client_only, false)
    }

    /// `hdr`: capture scRGB fp16 (Rgba16F) instead of 8-bit.
    pub fn start_fmt(
        hwnd: isize,
        fps_cap: Option<u32>,
        client_only: bool,
        hdr: bool,
    ) -> Result<Self> {
        Self::start_fmt_with_pixel_lock(hwnd, fps_cap, client_only, hdr, None)
    }

    /// Start WGC with an optional stable raw-geometry hint. The hint is used
    /// only to normalize +/-1px fullscreen compositor drift; larger geometry
    /// changes remain visible to the engine.
    pub fn start_fmt_with_pixel_lock(
        hwnd: isize,
        fps_cap: Option<u32>,
        client_only: bool,
        hdr: bool,
        pixel_exact_raw_hint: Option<(u32, u32)>,
    ) -> Result<Self> {
        let shared = Arc::new(Shared {
            buf: Mutex::new(FrameBuf::default()),
            queue: Mutex::new(VecDeque::new()),
            recycled_data: Mutex::new(Vec::new()),
            ready: Condvar::new(),
            seq: AtomicU64::new(0),
            queued_dropped: AtomicU64::new(0),
            alive: AtomicBool::new(true),
            queue_enabled: AtomicBool::new(false),
            client_only: AtomicBool::new(client_only),
            hwnd: std::sync::atomic::AtomicIsize::new(hwnd),
            hdr: AtomicBool::new(false),
            pixel_exact_raw_hint,
        });
        // No is_valid() pre-check: it rejects same-process/tool windows that
        // WGC can actually capture. Real incompatibility (shell/UWP windows)
        // surfaces as a start error below, which callers handle.
        let window = Window::from_raw_hwnd(hwnd as *mut std::ffi::c_void);
        // Request up to 240 Hz from WGC and let the engine perform any user
        // cap. WGC's system default is 60 Hz, which otherwise discards half
        // of a genuine 120 Hz source before the engine ever sees it.
        let _ = fps_cap;
        let interval = native_capture_interval();
        let settings = Settings::new(
            window,
            CursorCaptureSettings::WithoutCursor,
            DrawBorderSettings::WithoutBorder,
            SecondaryWindowSettings::Default,
            interval,
            DirtyRegionSettings::Default,
            if hdr {
                ColorFormat::Rgba16F
            } else {
                ColorFormat::Rgba8
            },
            shared.clone(),
        );
        shared.hdr.store(hdr, Ordering::Relaxed);
        let control =
            Handler::start_free_threaded(settings).map_err(|e| anyhow!("WGC start failed: {e}"))?;
        log::info!(
            "WGC optimized path: native_request_hz={MAX_NATIVE_CAPTURE_FPS:.0} frame_pool_buffers=3 persistent_staging=true mapped_copy=single callback_priority=mmcss-capture/high"
        );
        Ok(Self {
            shared,
            control: Some(control),
            last_seq: 0,
        })
    }

    /// Capture a whole monitor (test utility).
    pub fn start_monitor(
        monitor: windows_capture::monitor::Monitor,
        fps_cap: Option<u32>,
    ) -> Result<Self> {
        let shared = Arc::new(Shared {
            buf: Mutex::new(FrameBuf::default()),
            queue: Mutex::new(VecDeque::new()),
            recycled_data: Mutex::new(Vec::new()),
            ready: Condvar::new(),
            seq: AtomicU64::new(0),
            queued_dropped: AtomicU64::new(0),
            alive: AtomicBool::new(true),
            queue_enabled: AtomicBool::new(false),
            client_only: AtomicBool::new(false),
            hwnd: std::sync::atomic::AtomicIsize::new(0),
            hdr: AtomicBool::new(false),
            pixel_exact_raw_hint: None,
        });
        // As with window capture, keep WGC native delivery open to high-Hz
        // sources and apply optional rate limiting in the engine before HDR-to-SDR and the filter chain.
        let _ = fps_cap;
        let interval = native_capture_interval();
        let settings = Settings::new(
            monitor,
            CursorCaptureSettings::WithoutCursor,
            DrawBorderSettings::WithoutBorder,
            SecondaryWindowSettings::Default,
            interval,
            DirtyRegionSettings::Default,
            ColorFormat::Rgba8,
            shared.clone(),
        );
        let control = Handler::start_free_threaded(settings)
            .map_err(|e| anyhow!("WGC monitor start failed: {e}"))?;
        log::info!(
            "WGC optimized path: native_request_hz={MAX_NATIVE_CAPTURE_FPS:.0} frame_pool_buffers=3 persistent_staging=true mapped_copy=single callback_priority=mmcss-capture/high"
        );
        Ok(Self {
            shared,
            control: Some(control),
            last_seq: 0,
        })
    }

    /// Total frames delivered by WGC (including ones we dropped via
    /// newest-wins) — the honest capture rate.
    pub fn delivered(&self) -> u64 {
        self.shared.seq.load(Ordering::Acquire)
    }

    pub fn queued_dropped(&self) -> u64 {
        self.shared.queued_dropped.load(Ordering::Acquire)
    }

    pub fn queued_len(&self) -> usize {
        self.shared.queue.lock().unwrap().len()
    }

    /// Inspect the oldest queued frame without consuming it. Used by the
    /// optional duplicate reducer for one-frame look-ahead; the closure should
    /// perform only a small sampled analysis while the queue is locked.
    pub fn inspect_next_queued<R>(&self, f: impl FnOnce(&FrameBuf) -> R) -> Option<R> {
        let q = self.shared.queue.lock().unwrap();
        q.front().map(f)
    }

    /// Borrow a frame-sized allocation from the WGC pool for pre-chain format
    /// conversion. This avoids allocating a fresh 4-byte-per-pixel SDR buffer
    /// on every HDR frame, including queued interpolation sessions.
    pub fn take_recycled_data_buffer(&self) -> Vec<u8> {
        self.shared
            .recycled_data
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_default()
    }

    /// Return a temporary frame allocation to the WGC pool. Both Rgba8 and
    /// Rgba16F buffers may live here; copy_frame_to_vec resizes whichever one
    /// it receives to the exact next-frame byte count.
    pub fn recycle_data_buffer(&self, mut data: Vec<u8>) {
        data.clear();
        let mut recycled = self.shared.recycled_data.lock().unwrap();
        if recycled.len() < CAPTURE_QUEUE_HARD_MAX * 2 {
            recycled.push(data);
        }
    }

    pub fn set_queue_enabled(&self, enabled: bool) {
        self.shared.queue_enabled.store(enabled, Ordering::Release);
        if !enabled {
            let mut queue = self.shared.queue.lock().unwrap();
            let mut recycled = self.shared.recycled_data.lock().unwrap();
            while let Some(mut frame) = queue.pop_front() {
                frame.data.clear();
                recycled.push(frame.data);
            }
        }
    }

    pub fn alive(&self) -> bool {
        self.shared.alive.load(Ordering::Acquire)
    }

    /// Block until a frame newer than the last returned one arrives (or
    /// timeout). Returns whether a new frame is available.
    pub fn wait_new(&self, timeout: Duration) -> bool {
        let cur = self.shared.seq.load(Ordering::Acquire);
        if cur > self.last_seq {
            return true;
        }
        let guard = self.shared.buf.lock().unwrap();
        let (_g, _res) = self
            .shared
            .ready
            .wait_timeout_while(guard, timeout, |_| {
                self.shared.seq.load(Ordering::Acquire) <= self.last_seq
                    && self.shared.alive.load(Ordering::Acquire)
            })
            .unwrap();
        self.shared.seq.load(Ordering::Acquire) > self.last_seq
    }

    /// Copy the newest frame into `out` if newer than the last taken one.
    pub fn take_latest(&mut self, out: &mut FrameBuf) -> bool {
        let cur = self.shared.seq.load(Ordering::Acquire);
        if cur <= self.last_seq {
            return false;
        }
        let buf = self.shared.buf.lock().unwrap();
        if buf.data.is_empty() || buf.seq != cur {
            return false;
        }
        out.w = buf.w;
        out.h = buf.h;
        out.hdr = buf.hdr;
        out.seq = cur;
        out.received_at = buf.received_at;
        out.source_time_100ns = buf.source_time_100ns;
        out.data.clear();
        out.data.extend_from_slice(&buf.data);
        self.last_seq = cur;
        true
    }

    /// Copy the oldest queued frame newer than the last returned one. Used by
    /// NeoFlow to preserve adjacent frame pairs while keeping latency bounded.
    pub fn take_next_queued(&mut self, out: &mut FrameBuf, max_queue: usize) -> bool {
        if !self.shared.queue_enabled.load(Ordering::Acquire) {
            return self.take_latest(out);
        }
        let mut q = self.shared.queue.lock().unwrap();
        while q.front().map(|f| f.seq <= self.last_seq).unwrap_or(false) {
            q.pop_front();
        }
        while q.len() > max_queue.max(1) {
            q.pop_front();
            self.shared.queued_dropped.fetch_add(1, Ordering::Relaxed);
        }
        let Some(fb) = q.pop_front() else {
            drop(q);
            return self.take_latest(out);
        };
        self.last_seq = fb.seq;
        let mut old_data = std::mem::take(&mut out.data);
        old_data.clear();
        self.shared.recycled_data.lock().unwrap().push(old_data);
        *out = fb;
        true
    }

    pub fn stop(&mut self) {
        if let Some(c) = self.control.take() {
            let _ = c.stop();
        }
        self.shared.alive.store(false, Ordering::Release);
    }
}

impl Drop for WgcSource {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_capture_request_supports_240fps() {
        assert_eq!(MAX_NATIVE_CAPTURE_FPS, 240.0);
        assert!((1.0 / MAX_NATIVE_CAPTURE_FPS - 1.0 / 240.0).abs() < f64::EPSILON);
    }

    #[test]
    fn client_crop_uses_outer_rect_for_pip_frame() {
        let crop = client_crop_for_frame(
            (656, 519),
            (108, 131, 640, 480),
            Some((100, 100, 656, 519)),
            Some((107, 100, 642, 511)),
        )
        .unwrap();
        assert_eq!(crop.reference, "outer");
        assert_eq!((crop.x0, crop.y0, crop.x1, crop.y1), (8, 31, 648, 511));
    }

    #[test]
    fn client_crop_uses_dwm_rect_when_wgc_matches_it() {
        let crop = client_crop_for_frame(
            (642, 511),
            (108, 131, 640, 480),
            Some((100, 100, 656, 519)),
            Some((107, 100, 642, 511)),
        )
        .unwrap();
        assert_eq!(crop.reference, "dwm");
        assert_eq!((crop.x0, crop.y0, crop.x1, crop.y1), (1, 31, 641, 511));
    }

    #[test]
    fn odd_height_is_padded_by_repeating_the_last_row() {
        let mut pixels = vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
        ];
        let (width, height) = pad_edge_to_even(&mut pixels, 2, 3, 4);
        assert_eq!((width, height), (2, 4));
        assert_eq!(&pixels[16..24], &pixels[24..32]);
    }

    #[test]
    fn odd_width_and_height_repeat_the_right_and_bottom_edges() {
        let mut pixels = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (width, height) = pad_edge_to_even(&mut pixels, 3, 3, 1);
        assert_eq!((width, height), (4, 4));
        assert_eq!(
            pixels,
            vec![1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9, 9, 7, 8, 9, 9,]
        );
    }

    #[test]
    fn even_dimensions_are_left_unchanged() {
        let mut pixels = vec![1, 2, 3, 4];
        let (width, height) = pad_edge_to_even(&mut pixels, 2, 2, 1);
        assert_eq!((width, height), (2, 2));
        assert_eq!(pixels, vec![1, 2, 3, 4]);
    }
}
