//! Optional RGBA8 transport. Untrusted/vendor GPU code executes outside Neo.
use super::dlssnr_backend::{DLSSNR_CAP_ZERO_GUIDANCE, DlssNrBackend, detect_dlssnr_backend_pack};
use crate::core::dlssnr::DlssNrOptions;
use super::gl::{GlContext, GpuTex};
use std::borrow::Cow;
use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use windows::Win32::{
    Foundation::HANDLE,
    Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D12::{
            D3D12_HEAP_FLAG_SHARED, D3D12_HEAP_PROPERTIES, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_DESC, D3D12_RESOURCE_DIMENSION_BUFFER,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON,
            D3D12_TEXTURE_LAYOUT_ROW_MAJOR, D3D12CreateDevice, ID3D12Device, ID3D12Resource,
        },
        Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
        Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_FLAG_SOFTWARE, IDXGIFactory1},
    },
};
use windows::core::PCWSTR;
use std::time::{Duration, Instant};

const READY: [u8; 4] = *b"NR03";
const MAX_BYTES: usize = 256 * 1024 * 1024;


static DLSSNR_SHARED_KEY: AtomicU64 = AtomicU64::new(0x4e52_0000_0000_0001);

#[link(name = "kernel32")]
unsafe extern "system" {
    #[link_name = "GetCurrentProcess"]
    fn raw_get_current_process() -> isize;
    #[link_name = "OpenProcess"]
    fn raw_open_process(desired_access: u32, inherit_handle: i32, process_id: u32) -> isize;
    #[link_name = "DuplicateHandle"]
    fn raw_duplicate_handle(
        source_process: isize,
        source_handle: isize,
        target_process: isize,
        target_handle: *mut isize,
        desired_access: u32,
        inherit_handle: i32,
        options: u32,
    ) -> i32;
    #[link_name = "CloseHandle"]
    fn raw_close_handle(handle: isize) -> i32;
}
const PROCESS_DUP_HANDLE_RAW: u32 = 0x0040;
const DUPLICATE_SAME_ACCESS_RAW: u32 = 0x0000_0002;

fn luid_bytes(luid: windows::Win32::Foundation::LUID) -> [u8; 8] {
    let value = ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64;
    value.to_le_bytes()
}

fn d3d12_device_for_luid(wanted: [u8; 8]) -> Result<ID3D12Device, String> {
    let factory = unsafe { CreateDXGIFactory1::<IDXGIFactory1>() }.map_err(|e| e.to_string())?;
    for index in 0u32.. {
        let Ok(adapter) = (unsafe { factory.EnumAdapters1(index) }) else { break; };
        let desc = unsafe { adapter.GetDesc1() }.map_err(|e| e.to_string())?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 || luid_bytes(desc.AdapterLuid) != wanted {
            continue;
        }
        let mut device = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }
            .map_err(|e| e.to_string())?;
        return device.ok_or_else(|| "D3D12CreateDevice returned no DLSSNR shared device".into());
    }
    Err(format!("no DXGI adapter matched DLSSNR LUID {wanted:02x?}"))
}

struct SharedRgba8Buffer {
    key: u64,
    resource: ID3D12Resource,
    device: ID3D12Device,
    byte_len: usize,
    allocation_byte_len: u64,
    luid: [u8; 8],
}

impl SharedRgba8Buffer {
    fn new(luid: u64, width: u32, height: u32) -> Result<Self, String> {
        let byte_len = frame_bytes(width, height)?;
        let luid = luid.to_le_bytes();
        let device = d3d12_device_for_luid(luid)?;
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: byte_len as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_UNKNOWN,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let allocation = unsafe { device.GetResourceAllocationInfo(0, &[desc]) };
        if allocation.SizeInBytes == 0 || allocation.SizeInBytes == u64::MAX {
            return Err("invalid DLSSNR shared D3D12 allocation size".into());
        }
        let mut resource = None;
        unsafe {
            device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_SHARED,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut resource,
            )
        }
        .map_err(|e| e.to_string())?;
        Ok(Self {
            key: DLSSNR_SHARED_KEY.fetch_add(1, Ordering::Relaxed),
            resource: resource.ok_or("DLSSNR shared D3D12 resource was not created")?,
            device,
            byte_len,
            allocation_byte_len: allocation.SizeInBytes,
            luid,
        })
    }

    fn shared_handle(&self) -> Result<HANDLE, String> {
        unsafe {
            self.device
                .CreateSharedHandle(&self.resource, None, 0x1000_0000, PCWSTR::null())
                .map_err(|e| e.to_string())
        }
    }
}

fn duplicate_shared_handle_into_child(source: HANDLE, child_pid: u32) -> Result<u64, String> {
    let source_raw = source.0 as isize;
    let target_process = unsafe { raw_open_process(PROCESS_DUP_HANDLE_RAW, 0, child_pid) };
    if target_process == 0 {
        unsafe { let _ = raw_close_handle(source_raw); }
        return Err("OpenProcess(PROCESS_DUP_HANDLE) failed for DLSSNR worker".into());
    }
    let mut duplicated = 0isize;
    let ok = unsafe {
        raw_duplicate_handle(
            raw_get_current_process(),
            source_raw,
            target_process,
            &mut duplicated,
            0,
            0,
            DUPLICATE_SAME_ACCESS_RAW,
        )
    };
    unsafe {
        let _ = raw_close_handle(target_process);
        let _ = raw_close_handle(source_raw);
    }
    if ok == 0 || duplicated == 0 {
        Err("DuplicateHandle failed for DLSSNR shared RGBA8 resource".into())
    } else {
        Ok(duplicated as u64)
    }
}

fn frame_bytes(w: u32, h: u32) -> Result<usize, String> {
    let bytes = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4));
    bytes
        .filter(|n| *n > 0 && *n <= MAX_BYTES)
        .ok_or_else(|| "invalid DLSSNR frame size".into())
}

fn align_to(value: u32, alignment: u32) -> Result<u32, String> {
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err("invalid DLSSNR alignment".into());
    }
    value
        .checked_add(alignment - 1)
        .map(|v| v & !(alignment - 1))
        .filter(|v| *v > 0)
        .ok_or_else(|| "DLSSNR aligned geometry overflow".into())
}

fn reflect_index(index: usize, len: usize) -> usize {
    if len <= 1 {
        return 0;
    }
    let period = (len - 1) * 2;
    let reflected = index % period;
    if reflected < len {
        reflected
    } else {
        period - reflected
    }
}

fn pad_rgba8_reflect(
    input: &[u8],
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
) -> Result<Cow<'_, [u8]>, String> {
    let input_bytes = frame_bytes(width, height)?;
    let padded_bytes = frame_bytes(padded_width, padded_height)?;
    if input.len() != input_bytes || padded_width < width || padded_height < height {
        return Err("invalid DLSSNR padding geometry".into());
    }
    if width == padded_width && height == padded_height {
        return Ok(Cow::Borrowed(input));
    }
    let src_stride = width as usize * 4;
    let dst_stride = padded_width as usize * 4;
    let src_w = width as usize;
    let src_h = height as usize;
    let mut output = vec![0u8; padded_bytes];
    for y in 0..padded_height as usize {
        let sy = reflect_index(y, src_h);
        let src = &input[sy * src_stride..(sy + 1) * src_stride];
        let dst = &mut output[y * dst_stride..(y + 1) * dst_stride];
        dst[..src_stride].copy_from_slice(src);
        for x in src_w..padded_width as usize {
            let sx = reflect_index(x, src_w);
            dst[x * 4..x * 4 + 4].copy_from_slice(&src[sx * 4..sx * 4 + 4]);
        }
    }
    Ok(Cow::Owned(output))
}

fn crop_rgba8(
    input: &[u8],
    padded_width: u32,
    padded_height: u32,
    width: u32,
    height: u32,
) -> Result<Cow<'_, [u8]>, String> {
    let padded_bytes = frame_bytes(padded_width, padded_height)?;
    if input.len() != padded_bytes || width > padded_width || height > padded_height {
        return Err("invalid DLSSNR crop geometry".into());
    }
    if width == padded_width && height == padded_height {
        return Ok(Cow::Borrowed(input));
    }
    let src_stride = padded_width as usize * 4;
    let dst_stride = width as usize * 4;
    let mut output = vec![0u8; frame_bytes(width, height)?];
    for y in 0..height as usize {
        output[y * dst_stride..(y + 1) * dst_stride]
            .copy_from_slice(&input[y * src_stride..y * src_stride + dst_stride]);
    }
    Ok(Cow::Owned(output))
}

#[derive(Debug, Clone, Copy)]
struct GpuTimingReply {
    frame_index: u64,
    input_copy_us: u64,
    evaluate_us: u64,
    output_copy_us: u64,
    total_us: u64,
}

#[derive(Debug)]
struct WorkerReply {
    pixels: Vec<u8>,
    pad_us: u64,
    backend_us: u64,
    crop_us: u64,
    shared_active: bool,
    gpu_timing: Option<GpuTimingReply>,
}

struct Worker {
    child: Child,
    jobs: Option<SyncSender<(Option<Vec<u8>>, bool, Option<DlssNrOptions>)>>,
    replies: Receiver<Result<WorkerReply, String>>,
    input_recycle: Receiver<Vec<u8>>,
    output_recycle: SyncSender<Vec<u8>>,
    ready: bool,
    started: Instant,
    initialized: Arc<AtomicBool>,
    shared_active: bool,
}

impl Worker {
    fn start(
        app: &std::path::Path,
        w: u32,
        h: u32,
        luid: u64,
        options: DlssNrOptions,
        shared_request: Option<(HANDLE, u64)>,
    ) -> Result<Self, String> {
        let bytes = frame_bytes(w, h)?;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(app.join("dlssnr-worker.log"))
            .map_err(|e| e.to_string())?;
        let child_result = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .arg("--dlssnr-worker")
            .arg(app)
            .arg(w.to_string())
            .arg(h.to_string())
            .arg(luid.to_string())
            .arg(options.preset.to_string())
            .arg(options.style.to_string())
            .arg(options.intensity.to_string())
            .arg(options.local_tone.to_string())
            .arg(options.local_structure.to_string())
            .arg(options.skin_structure.to_string())
            .arg(u8::from(options.auto_mask).to_string())
            .arg(u8::from(options.ui_correction).to_string())
            .creation_flags(0x08000000)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(log))
            .spawn();
        let mut child = match child_result {
            Ok(child) => child,
            Err(error) => {
                if let Some((handle, _)) = shared_request {
                    unsafe { let _ = raw_close_handle(handle.0 as isize); }
                }
                return Err(error.to_string());
            }
        };
        let shared_child = match shared_request {
            Some((handle, byte_len)) => match duplicate_shared_handle_into_child(handle, child.id()) {
                Ok(child_handle) => Some((child_handle, byte_len)),
                Err(error) => {
                    log::warn!("dlssnr-shared-init: stage=duplicate result=fallback-cpu reason={error}");
                    None
                }
            },
            None => None,
        };
        let mut input = child.stdin.take().ok_or("missing worker stdin")?;
        let mut output = child.stdout.take().ok_or("missing worker stdout")?;
        let (jobs, incoming) = mpsc::sync_channel::<(Option<Vec<u8>>, bool, Option<DlssNrOptions>)>(1);
        let (completed, replies) = mpsc::sync_channel(1);
        let (input_recycle_tx, input_recycle) = mpsc::sync_channel::<Vec<u8>>(1);
        let (output_recycle, output_recycle_rx) = mpsc::sync_channel::<Vec<u8>>(1);
        let initialized = Arc::new(AtomicBool::new(false));
        let ready_flag = initialized.clone();
        std::thread::spawn(move || {
            let mut transfer = || -> Result<(), String> {
                // One-time transport negotiation. A duplicated HANDLE value is valid only
                // inside this child process. Missing/newer bridge support simply selects CPU.
                match shared_child {
                    Some((child_handle, byte_len)) => {
                        input.write_all(&[1]).map_err(|e| format!("shared config: {e}"))?;
                        input.write_all(&child_handle.to_le_bytes()).map_err(|e| format!("shared handle: {e}"))?;
                        input.write_all(&byte_len.to_le_bytes()).map_err(|e| format!("shared bytes: {e}"))?;
                    }
                    None => input.write_all(&[0]).map_err(|e| format!("shared config: {e}"))?,
                }
                input.flush().map_err(|e| format!("shared config flush: {e}"))?;

                let mut ready = [0; 4];
                output
                    .read_exact(&mut ready)
                    .map_err(|e| format!("initialize: {e}"))?;
                if ready != READY {
                    return Err("invalid worker handshake".into());
                }
                let mut shared_status = [0u8; 1];
                output.read_exact(&mut shared_status).map_err(|e| format!("shared status: {e}"))?;
                let shared_active = shared_status[0] != 0;
                completed
                    .send(Ok(WorkerReply {
                        pixels: Vec::new(),
                        pad_us: 0,
                        backend_us: 0,
                        crop_us: 0,
                        shared_active,
                        gpu_timing: None,
                    }))
                    .map_err(|e| e.to_string())?;
                ready_flag.store(true, Ordering::Release);
                while let Ok((pixels, reset, options_update)) = incoming.recv() {
                    let use_shared = pixels.is_none();
                    if use_shared && !shared_active {
                        return Err("shared DLSSNR job submitted without active shared transport".into());
                    }
                    if let Some(ref pixels) = pixels {
                        if pixels.len() != bytes {
                            return Err("invalid input length".into());
                        }
                    }
                    input
                        .write_all(&[
                            u8::from(reset),
                            u8::from(options_update.is_some()),
                            u8::from(use_shared),
                        ])
                        .map_err(|e| format!("input: {e}"))?;
                    if let Some(options) = options_update {
                        for word in [
                            options.preset,
                            options.style,
                            options.intensity.to_bits(),
                            options.local_tone.to_bits(),
                            options.local_structure.to_bits(),
                            options.skin_structure.to_bits(),
                        ] {
                            input
                                .write_all(&word.to_le_bytes())
                                .map_err(|e| format!("input options: {e}"))?;
                        }
                        input
                            .write_all(&[u8::from(options.auto_mask), u8::from(options.ui_correction)])
                            .map_err(|e| format!("input options: {e}"))?;
                    }
                    if let Some(pixels) = pixels {
                        input.write_all(&pixels).map_err(|e| format!("input: {e}"))?;
                        let _ = input_recycle_tx.try_send(pixels);
                    }
                    input.flush().map_err(|e| format!("input flush: {e}"))?;

                    let mut timing = [0u8; 72];
                    output
                        .read_exact(&mut timing)
                        .map_err(|e| format!("evaluate timing: {e}"))?;
                    let pad_us = u64::from_le_bytes(timing[0..8].try_into().unwrap());
                    let backend_us = u64::from_le_bytes(timing[8..16].try_into().unwrap());
                    let crop_us = u64::from_le_bytes(timing[16..24].try_into().unwrap());
                    let gpu_valid = u64::from_le_bytes(timing[24..32].try_into().unwrap()) != 0;
                    let gpu_timing = gpu_valid.then(|| GpuTimingReply {
                        frame_index: u64::from_le_bytes(timing[32..40].try_into().unwrap()),
                        input_copy_us: u64::from_le_bytes(timing[40..48].try_into().unwrap()),
                        evaluate_us: u64::from_le_bytes(timing[48..56].try_into().unwrap()),
                        output_copy_us: u64::from_le_bytes(timing[56..64].try_into().unwrap()),
                        total_us: u64::from_le_bytes(timing[64..72].try_into().unwrap()),
                    });
                    let pixels = if use_shared {
                        Vec::new()
                    } else {
                        let mut pixels = output_recycle_rx
                            .try_recv()
                            .ok()
                            .filter(|v| v.len() == bytes)
                            .unwrap_or_else(|| vec![0; bytes]);
                        output
                            .read_exact(&mut pixels)
                            .map_err(|e| format!("evaluate: {e}"))?;
                        pixels
                    };
                    completed
                        .send(Ok(WorkerReply {
                            pixels,
                            pad_us,
                            backend_us,
                            crop_us,
                            shared_active,
                            gpu_timing,
                        }))
                        .map_err(|e| e.to_string())?;
                }
                Ok(())
            };
            if let Err(error) = transfer() {
                let _ = completed.send(Err(error));
                ready_flag.store(true, Ordering::Release);
            }
        });
        Ok(Self {
            child,
            jobs: Some(jobs),
            replies,
            input_recycle,
            output_recycle,
            ready: false,
            started: Instant::now(),
            initialized,
            shared_active: false,
        })
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Closing the request channel lets an idle child release Feature/NGX.
        // A hung vendor call is killed after a bounded grace period, never joined.
        self.jobs.take();
        let deadline = Instant::now() + Duration::from_millis(100);
        while Instant::now() < deadline {
            if self.child.try_wait().ok().flatten().is_some() {
                return;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

pub struct DlssNrStage {
    options: DlssNrOptions,
    options_dirty: bool,
    app: PathBuf,
    worker: Option<Worker>,
    geometry: Option<(u32, u32, u64)>,
    disabled: bool,
    evaluated: bool,
    last_frame: Option<Instant>,
    input_buffer: Vec<u8>,
    timing_frames: u64,
    shared_buffer: Option<SharedRgba8Buffer>,
    retired_shared_keys: Vec<u64>,
    shared_fallback_geometry: Option<(u32, u32, u64)>,
}

impl DlssNrStage {
    pub fn set_options(&mut self, options: DlssNrOptions) {
        let options = crate::core::dlssnr::sanitize_options(options);
        if self.options == options {
            return;
        }
        let recreate_feature = self.options.preset != options.preset;
        self.options = options;
        self.disabled = false;
        if recreate_feature {
            self.worker = None;
            if let Some(shared) = self.shared_buffer.take() {
                self.retired_shared_keys.push(shared.key);
            }
            self.geometry = None;
            self.evaluated = false;
            self.last_frame = None;
            self.options_dirty = false;
        } else {
            self.options_dirty = true;
        }
    }

    pub fn metric_ms(&self, elapsed: f64) -> f64 {
        if self.disabled {
            -2.0
        } else if !self.evaluated {
            -1.0
        } else {
            elapsed
        }
    }

    pub fn reset(&mut self) {
        self.worker = None;
        if let Some(shared) = self.shared_buffer.take() {
            self.retired_shared_keys.push(shared.key);
        }
        self.geometry = None;
        self.disabled = false;
        self.evaluated = false;
        self.last_frame = None;
        self.input_buffer.clear();
        self.timing_frames = 0;
        self.options_dirty = false;
        self.shared_fallback_geometry = None;
    }

    pub fn needs_refresh(&self) -> bool {
        !self.disabled
            && (self.options_dirty
                || self.worker.as_ref().is_some_and(|w| {
                    !w.ready
                        && (w.initialized.load(Ordering::Acquire)
                            || w.started.elapsed() > Duration::from_secs(60))
                }))
    }

    pub fn new(app: PathBuf, options: DlssNrOptions) -> Self {
        Self {
            options: crate::core::dlssnr::sanitize_options(options),
            options_dirty: false,
            app,
            worker: None,
            geometry: None,
            disabled: false,
            evaluated: false,
            last_frame: None,
            input_buffer: Vec::new(),
            timing_frames: 0,
            shared_buffer: None,
            retired_shared_keys: Vec::new(),
            shared_fallback_geometry: None,
        }
    }

    fn fail(&mut self, reason: &str) {
        log::warn!("dlssnr-bypass: reason={reason} action=disable-session original-frame=true");
        self.disabled = true;
        self.worker = None;
        if let Some(shared) = self.shared_buffer.take() {
            self.retired_shared_keys.push(shared.key);
        }
    }

    fn fallback_shared(
        &mut self,
        gc: &mut GlContext,
        geometry: (u32, u32, u64),
        reason: &str,
    ) {
        log::warn!(
            "dlssnr-shared-fallback: size={}x{} reason={} action=restart-v742-cpu-bridge",
            geometry.0,
            geometry.1,
            reason
        );
        self.worker = None;
        if let Some(shared) = self.shared_buffer.take() {
            gc.clear_external_buffer_key(shared.key);
        }
        self.geometry = None;
        self.shared_fallback_geometry = Some(geometry);
        self.evaluated = false;
        self.last_frame = None;
        self.input_buffer.clear();
        self.timing_frames = 0;
        self.options_dirty = false;
    }

    fn init_geometry(
        &mut self,
        gc: &mut GlContext,
        geometry: (u32, u32, u64),
    ) -> Result<(), String> {
        self.worker = None;
        if let Some(shared) = self.shared_buffer.take() {
            gc.clear_external_buffer_key(shared.key);
        }
        if self.shared_fallback_geometry.is_some_and(|old| old != geometry) {
            self.shared_fallback_geometry = None;
        }
        self.geometry = Some(geometry);
        self.evaluated = false;
        self.input_buffer.clear();
        self.timing_frames = 0;

        let aligned = (
            align_to(geometry.0, 64).unwrap_or(geometry.0),
            align_to(geometry.1, 16).unwrap_or(geometry.1),
        );
        let aligned_logical = aligned == (geometry.0, geometry.1);
        let allow_shared = aligned_logical && self.shared_fallback_geometry != Some(geometry);
        let mut shared = None;
        let mut shared_request = None;

        if allow_shared {
            match SharedRgba8Buffer::new(geometry.2, geometry.0, geometry.1) {
                Ok(candidate) => match candidate.shared_handle() {
                    Ok(gl_handle) => match gc.import_external_d3d12_buffer(
                        candidate.key,
                        gl_handle,
                        candidate.byte_len,
                        candidate.allocation_byte_len,
                        candidate.luid,
                    ) {
                        Ok(()) => match candidate.shared_handle() {
                            Ok(worker_handle) => {
                                shared_request = Some((worker_handle, candidate.byte_len as u64));
                                log::info!(
                                    "dlssnr-shared-init: size={}x{} key={} bytes={} allocation_bytes={} result=parent-import-ready",
                                    geometry.0,
                                    geometry.1,
                                    candidate.key,
                                    candidate.byte_len,
                                    candidate.allocation_byte_len,
                                );
                                shared = Some(candidate);
                            }
                            Err(error) => {
                                gc.clear_external_buffer_key(candidate.key);
                                log::warn!("dlssnr-shared-init: stage=worker-handle result=fallback-cpu reason={error}");
                            }
                        },
                        Err(error) => {
                            log::warn!("dlssnr-shared-init: stage=gl-import result=fallback-cpu reason={error}");
                        }
                    },
                    Err(error) => {
                        log::warn!("dlssnr-shared-init: stage=gl-handle result=fallback-cpu reason={error}");
                    }
                },
                Err(error) => {
                    log::warn!("dlssnr-shared-init: stage=d3d12-buffer result=fallback-cpu reason={error}");
                }
            }
        }

        let worker = match Worker::start(
            &self.app,
            geometry.0,
            geometry.1,
            geometry.2,
            self.options,
            shared_request,
        ) {
            Ok(worker) => worker,
            Err(error) => {
                if let Some(shared) = shared.take() {
                    gc.clear_external_buffer_key(shared.key);
                }
                return Err(error);
            }
        };
        self.shared_buffer = shared;
        self.worker = Some(worker);
        self.options_dirty = false;
        log::info!(
            "dlssnr-initialize: {}x{} RGBA8 working={}x{} alignment=width64-height16 reflect_pad={} luid={:016x} process-isolated=true shared_requested={} options={:?}",
            geometry.0,
            geometry.1,
            aligned.0,
            aligned.1,
            !aligned_logical,
            geometry.2,
            self.shared_buffer.is_some(),
            self.options
        );
        Ok(())
    }

    pub fn apply(&mut self, gc: &mut GlContext, source: GpuTex) -> GpuTex {
        for key in std::mem::take(&mut self.retired_shared_keys) {
            gc.clear_external_buffer_key(key);
        }
        if self.disabled {
            return source;
        }
        let luid = super::vulkan_gpu::production_selected_luid()
            .or_else(|| gc.external_device_luid().map(u64::from_le_bytes));
        let Some(luid) = luid.filter(|v| *v != 0) else {
            self.fail("render GPU identity unavailable; refusing alternate GPU");
            return source;
        };
        let geometry = (source.w() as u32, source.h() as u32, luid);
        if self.geometry != Some(geometry) {
            if let Err(error) = self.init_geometry(gc, geometry) {
                self.fail(&error);
            }
            return source;
        }
        if self.worker.is_none() {
            return source;
        }

        if self.worker.as_ref().is_some_and(|worker| !worker.ready) {
            let reply = self.worker.as_mut().unwrap().replies.try_recv();
            match reply {
                Ok(Ok(reply)) => {
                    let worker = self.worker.as_mut().unwrap();
                    worker.ready = true;
                    worker.shared_active = reply.shared_active;
                    if self.shared_buffer.is_some() && !reply.shared_active {
                        if let Some(shared) = self.shared_buffer.take() {
                            gc.clear_external_buffer_key(shared.key);
                        }
                        self.shared_fallback_geometry = Some(geometry);
                        log::info!(
                            "dlssnr-create: success=true transport=isolated-cpu-bridge-v742-reuse shared_result=backend-unavailable"
                        );
                    } else {
                        log::info!(
                            "dlssnr-create: success=true transport={}",
                            if reply.shared_active {
                                "isolated-d3d12-shared-rgba8-v743"
                            } else {
                                "isolated-cpu-bridge-v742-reuse"
                            }
                        );
                    }
                }
                Ok(Err(error)) => {
                    if self.shared_buffer.is_some() {
                        self.fallback_shared(gc, geometry, &error);
                    } else {
                        self.fail(&error);
                    }
                    return source;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.shared_buffer.is_some() {
                        self.fallback_shared(gc, geometry, "worker exited during shared creation");
                    } else {
                        self.fail("worker exited during creation");
                    }
                    return source;
                }
                Err(_) => {
                    if self.worker.as_ref().unwrap().started.elapsed() > Duration::from_secs(60) {
                        if self.shared_buffer.is_some() {
                            self.fallback_shared(gc, geometry, "shared initialization timeout");
                        } else {
                            self.fail("initialization timeout");
                        }
                    }
                    return source;
                }
            }
        }

        let now = Instant::now();
        let reset = self
            .last_frame
            .is_none_or(|t| now.duration_since(t) > Duration::from_secs(1));
        self.last_frame = Some(now);
        let options_update = self.options_dirty.then_some(self.options);
        let shared_active = self.worker.as_ref().is_some_and(|w| w.shared_active);

        if shared_active {
            let Some(key) = self.shared_buffer.as_ref().map(|shared| shared.key) else {
                self.fallback_shared(gc, geometry, "worker selected shared transport without parent buffer");
                return source;
            };
            if let Err(error) = gc.wait_external_buffer_idle(key) {
                self.fallback_shared(gc, geometry, &error);
                return source;
            }
            let pack_started = Instant::now();
            if let Err(error) = gc.rgba_texture_to_external_rgba8(key, source, source.w(), source.h()) {
                self.fallback_shared(gc, geometry, &error);
                return source;
            }
            let pack_us = pack_started.elapsed().as_micros() as u64;
            let send_result = self
                .worker
                .as_ref()
                .and_then(|worker| worker.jobs.as_ref())
                .ok_or(())
                .and_then(|jobs| jobs.try_send((None, reset, options_update)).map_err(|_| ()));
            if send_result.is_err() {
                self.fallback_shared(gc, geometry, "shared worker queue unavailable");
                return source;
            }
            if options_update.is_some() {
                self.options_dirty = false;
            }
            let roundtrip_started = Instant::now();
            let reply = self
                .worker
                .as_mut()
                .unwrap()
                .replies
                .recv_timeout(Duration::from_millis(1000));
            match reply {
                Ok(Ok(reply)) => {
                    let roundtrip_us = roundtrip_started.elapsed().as_micros() as u64;
                    let unpack_started = Instant::now();
                    let texture = match gc.external_rgba8_buffer_to_texture_with_source_alpha(
                        key,
                        source,
                        source.w(),
                        source.h(),
                    ) {
                        Ok(texture) => texture,
                        Err(error) => {
                            self.fallback_shared(gc, geometry, &error);
                            return source;
                        }
                    };
                    let unpack_us = unpack_started.elapsed().as_micros() as u64;
                    if !self.evaluated {
                        log::info!(
                            "dlssnr-evaluate: success=true output={}x{} RGBA8 transport=isolated-d3d12-shared-rgba8-v743",
                            source.w(),
                            source.h()
                        );
                        self.evaluated = true;
                    }
                    self.timing_frames = self.timing_frames.wrapping_add(1);
                    if self.timing_frames == 1 || self.timing_frames % 120 == 0 {
                        log::debug!(
                            "dlssnr-transport-timing: frame={} size={}x{} route=gpu-shared gl_pack_ms={:.3} worker_roundtrip_ms={:.3} backend_total_ms={:.3} gl_unpack_ms={:.3} cpu_frame_bytes=0",
                            self.timing_frames,
                            source.w(),
                            source.h(),
                            pack_us as f64 / 1000.0,
                            roundtrip_us as f64 / 1000.0,
                            reply.backend_us as f64 / 1000.0,
                            unpack_us as f64 / 1000.0,
                        );
                        if let Some(gpu) = reply.gpu_timing {
                            let overhead_us = reply.backend_us.saturating_sub(gpu.total_us);
                            log::debug!(
                                "dlssnr-backend-timing: frame={} bridge_frame={} size={}x{} input_copy_gpu_ms={:.3} evaluate_gpu_ms={:.3} output_copy_gpu_ms={:.3} gpu_total_ms={:.3} backend_wall_ms={:.3} cpu_driver_fence_ms={:.3} zero_guidance_state=persistent",
                                self.timing_frames,
                                gpu.frame_index + 1,
                                source.w(),
                                source.h(),
                                gpu.input_copy_us as f64 / 1000.0,
                                gpu.evaluate_us as f64 / 1000.0,
                                gpu.output_copy_us as f64 / 1000.0,
                                gpu.total_us as f64 / 1000.0,
                                reply.backend_us as f64 / 1000.0,
                                overhead_us as f64 / 1000.0,
                            );
                        }
                    }
                    texture
                }
                Ok(Err(error)) => {
                    self.fallback_shared(gc, geometry, &error);
                    source
                }
                Err(error) => {
                    self.fallback_shared(gc, geometry, &format!("shared evaluation timeout/disconnect: {error}"));
                    source
                }
            }
        } else {
            if let Ok(recycled) = self.worker.as_mut().unwrap().input_recycle.try_recv() {
                self.input_buffer = recycled;
            }
            let download_started = Instant::now();
            gc.download_rgba8_into(source, &mut self.input_buffer);
            let download_us = download_started.elapsed().as_micros() as u64;
            let pixels = std::mem::take(&mut self.input_buffer);
            if self
                .worker
                .as_ref()
                .and_then(|worker| worker.jobs.as_ref())
                .unwrap()
                .try_send((Some(pixels), reset, options_update))
                .is_err()
            {
                self.fail("worker queue unavailable");
                return source;
            }
            if options_update.is_some() {
                self.options_dirty = false;
            }
            let roundtrip_started = Instant::now();
            match self
                .worker
                .as_mut()
                .unwrap()
                .replies
                .recv_timeout(Duration::from_millis(1000))
            {
                Ok(Ok(reply)) => {
                    let roundtrip_us = roundtrip_started.elapsed().as_micros() as u64;
                    if !self.evaluated {
                        log::info!(
                            "dlssnr-evaluate: success=true output={}x{} RGBA8 transport=isolated-cpu-bridge-v742-reuse",
                            source.w(),
                            source.h()
                        );
                        self.evaluated = true;
                    }
                    let upload_started = Instant::now();
                    let texture = gc.upload_rgba8(source.w(), source.h(), &reply.pixels);
                    let upload_us = upload_started.elapsed().as_micros() as u64;
                    self.timing_frames = self.timing_frames.wrapping_add(1);
                    if self.timing_frames == 1 || self.timing_frames % 120 == 0 {
                        log::debug!(
                            "dlssnr-transport-timing: frame={} size={}x{} route=cpu-reuse gl_download_ms={:.3} worker_roundtrip_ms={:.3} worker_pad_ms={:.3} backend_total_ms={:.3} worker_crop_ms={:.3} gl_upload_ms={:.3} aligned_copy_elided={}",
                            self.timing_frames,
                            source.w(),
                            source.h(),
                            download_us as f64 / 1000.0,
                            roundtrip_us as f64 / 1000.0,
                            reply.pad_us as f64 / 1000.0,
                            reply.backend_us as f64 / 1000.0,
                            reply.crop_us as f64 / 1000.0,
                            upload_us as f64 / 1000.0,
                            source.w() as u32 % 64 == 0 && source.h() as u32 % 16 == 0,
                        );
                    }
                    let _ = self.worker.as_ref().unwrap().output_recycle.try_send(reply.pixels);
                    texture
                }
                Ok(Err(error)) => {
                    self.fail(&error);
                    source
                }
                Err(error) => {
                    self.fail(&format!("evaluation timeout/disconnect: {error}"));
                    source
                }
            }
        }
    }
}

pub fn run_worker() -> Result<(), String> {
    let args: Vec<_> = std::env::args_os().skip(2).collect();
    if args.len() != 12 {
        return Err("invalid worker arguments".into());
    }
    let app = PathBuf::from(&args[0]);
    let w: u32 = args[1]
        .to_string_lossy()
        .parse()
        .map_err(|_| "invalid width")?;
    let h: u32 = args[2]
        .to_string_lossy()
        .parse()
        .map_err(|_| "invalid height")?;
    let luid: u64 = args[3]
        .to_string_lossy()
        .parse()
        .map_err(|_| "invalid LUID")?;
    if luid == 0 {
        return Err("explicit GPU identity required".into());
    }
    let parse_u32 = |i: usize, name: &str| -> Result<u32, String> {
        args[i].to_string_lossy().parse().map_err(|_| format!("invalid {name}"))
    };
    let parse_f32 = |i: usize, name: &str| -> Result<f32, String> {
        args[i].to_string_lossy().parse().map_err(|_| format!("invalid {name}"))
    };
    let options = crate::core::dlssnr::sanitize_options(DlssNrOptions {
        preset: parse_u32(4, "preset")?,
        style: parse_u32(5, "style")?,
        intensity: parse_f32(6, "intensity")?,
        local_tone: parse_f32(7, "local tone")?,
        local_structure: parse_f32(8, "local structure")?,
        skin_structure: parse_f32(9, "skin structure")?,
        auto_mask: parse_u32(10, "automatic mask")? != 0,
        ui_correction: parse_u32(11, "UI correction")? != 0,
    });
    let bytes = frame_bytes(w, h)?;
    // Feature 18 is stable when the RGBA8 working row is 256-byte aligned.
    // RGBA8 is 4 bytes/pixel, so the private working width is aligned to
    // 64 pixels while height keeps the 16-pixel alignment verified by the
    // existing 720/912-height success cases. Logical geometry is unchanged.
    let work_w = align_to(w, 64)?;
    let work_h = align_to(h, 16)?;
    // Validate the padded working allocation before any backend/D3D12 session is created.
    // This keeps pathological dimensions on the normal fail-open path.
    let _ = frame_bytes(work_w, work_h)?;
    let availability = detect_dlssnr_backend_pack(&app);
    let backend = DlssNrBackend::load_verified(&availability)?;
    let caps = backend.capabilities();
    eprintln!(
        "DLSSNR ABI=1 capabilities={caps:#x} description={:?}",
        backend.backend_description()
    );
    if caps & DLSSNR_CAP_ZERO_GUIDANCE == 0 {
        return Err("Zero Guidance not supported".into());
    }
    let mut session = backend.create_session_with_options(Some(luid), work_w, work_h, options)?;
    // Also call the optional setter once. This keeps legacy eval-options-v1
    // bridges compatible while v710+ bridges already received the same values
    // before CreateFeature through the reserved create descriptor contract.
    session.set_options(options)?;
    eprintln!(
        "Feature 18 Create succeeded logical={w}x{h} working={work_w}x{work_h} RGBA8 reflect_pad={} luid={luid:016x} options={options:?}",
        work_w != w || work_h != h
    );
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();

    let mut shared_config = [0u8; 1];
    input.read_exact(&mut shared_config).map_err(|e| format!("shared config: {e}"))?;
    let mut shared_active = false;
    if shared_config[0] != 0 {
        let mut raw = [0u8; 16];
        input.read_exact(&mut raw).map_err(|e| format!("shared config: {e}"))?;
        let child_handle = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let shared_bytes = u64::from_le_bytes(raw[8..16].try_into().unwrap());
        if session.shared_rgba8_available() {
            match session.attach_shared_rgba8(child_handle, shared_bytes) {
                Ok(()) => {
                    shared_active = true;
                    eprintln!(
                        "DLSSNR shared RGBA8 attached logical={w}x{h} bytes={shared_bytes} handle=worker-local"
                    );
                }
                Err(error) => {
                    eprintln!("DLSSNR shared RGBA8 attach failed; CPU fallback: {error}");
                }
            }
        } else {
            unsafe { let _ = raw_close_handle(child_handle as isize); }
            eprintln!("DLSSNR bridge has no shared RGBA8 extension; CPU fallback");
        }
    }
    output
        .write_all(&READY)
        .and_then(|_| output.write_all(&[u8::from(shared_active)]))
        .and_then(|_| output.flush())
        .map_err(|e| e.to_string())?;

    let mut pixels = vec![0; bytes];
    let mut processed = vec![0u8; frame_bytes(work_w, work_h)?];
    let mut frames = 0u64;
    loop {
        let mut header = [0u8; 3];
        match input.read_exact(&mut header) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.to_string()),
        }
        if header[1] != 0 {
            let mut words = [0u32; 6];
            for word in &mut words {
                let mut bytes = [0u8; 4];
                input.read_exact(&mut bytes).map_err(|e| e.to_string())?;
                *word = u32::from_le_bytes(bytes);
            }
            let mut flags = [0u8; 2];
            input.read_exact(&mut flags).map_err(|e| e.to_string())?;
            let updated = crate::core::dlssnr::sanitize_options(DlssNrOptions {
                preset: words[0],
                style: words[1],
                intensity: f32::from_bits(words[2]),
                local_tone: f32::from_bits(words[3]),
                local_structure: f32::from_bits(words[4]),
                skin_structure: f32::from_bits(words[5]),
                auto_mask: flags[0] != 0,
                ui_correction: flags[1] != 0,
            });
            session.set_options(updated)?;
            eprintln!("Feature 18 live options updated: {updated:?}");
        }

        if header[2] != 0 {
            if !shared_active {
                return Err("shared frame requested after shared transport fallback".into());
            }
            let backend_started = Instant::now();
            session.process_shared_rgba8(header[0] != 0)?;
            let backend_us = backend_started.elapsed().as_micros() as u64;
            let gpu = session.last_gpu_timing();
            let values = if let Some(timing) = gpu {
                [
                    0u64, backend_us, 0u64, 1u64, timing.frame_index,
                    timing.input_copy_gpu_us, timing.evaluate_gpu_us,
                    timing.output_copy_gpu_us, timing.gpu_total_us,
                ]
            } else {
                [0u64, backend_us, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64]
            };
            for value in values {
                output.write_all(&value.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            output.flush().map_err(|e| e.to_string())?;
        } else {
            input.read_exact(&mut pixels).map_err(|e| e.to_string())?;
            let pad_started = Instant::now();
            let padded = pad_rgba8_reflect(&pixels, w, h, work_w, work_h)?;
            let pad_us = pad_started.elapsed().as_micros() as u64;
            let backend_started = Instant::now();
            session.process_rgba8_into(
                padded.as_ref(),
                work_w * 4,
                header[0] != 0,
                &mut processed,
                work_w * 4,
            )?;
            let backend_us = backend_started.elapsed().as_micros() as u64;
            let crop_started = Instant::now();
            let cropped = crop_rgba8(&processed, work_w, work_h, w, h)?;
            let crop_us = crop_started.elapsed().as_micros() as u64;
            for value in [pad_us, backend_us, crop_us, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64] {
                output.write_all(&value.to_le_bytes()).map_err(|e| e.to_string())?;
            }
            output
                .write_all(cropped.as_ref())
                .and_then(|_| output.flush())
                .map_err(|e| e.to_string())?;
        }
        frames += 1;
        if frames == 1 {
            eprintln!(
                "Feature 18 Evaluate succeeded transport={}",
                if shared_active { "d3d12-shared-rgba8" } else { "cpu-rgba8" }
            );
        }
    }
    drop(session);
    drop(backend);
    eprintln!("DLSSNR shutdown/release completed frames={frames}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_transport_sizes() {
        assert!(frame_bytes(0, 720).is_err());
        assert!(frame_bytes(u32::MAX, u32::MAX).is_err());
        assert_eq!(frame_bytes(1280, 720).unwrap(), 3686400);
    }

    #[test]
    fn dlssnr_working_alignment_covers_1440x810_and_preserves_known_good_modes() {
        assert_eq!(align_to(1440, 64).unwrap(), 1472);
        assert_eq!(align_to(810, 16).unwrap(), 816);
        assert_eq!(align_to(1280, 64).unwrap(), 1280);
        assert_eq!(align_to(720, 16).unwrap(), 720);
        assert_eq!(align_to(1600, 64).unwrap(), 1600);
        assert_eq!(align_to(900, 16).unwrap(), 912);
        assert_eq!(align_to(5120, 64).unwrap(), 5120);
        assert_eq!(align_to(2880, 16).unwrap(), 2880);
    }

    #[test]
    fn reflection_padding_and_crop_preserve_original_pixels() {
        let input = vec![
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let padded = pad_rgba8_reflect(&input, 2, 2, 4, 4).unwrap();
        assert_eq!(&padded[0..16], &[1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]);
        assert_eq!(&padded[16..32], &[9,10,11,12,13,14,15,16,9,10,11,12,13,14,15,16]);
        assert_eq!(&padded[32..48], &[1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8]);
        assert_eq!(crop_rgba8(&padded, 4, 4, 2, 2).unwrap().as_ref(), input.as_slice());
    }

    #[test]
    fn reflection_padding_handles_single_pixel_axis() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let padded = pad_rgba8_reflect(&input, 1, 2, 4, 4).unwrap();
        assert_eq!(&padded[0..16], &[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]);
        assert_eq!(&padded[16..32], &[5,6,7,8,5,6,7,8,5,6,7,8,5,6,7,8]);
        assert_eq!(crop_rgba8(&padded, 4, 4, 1, 2).unwrap().as_ref(), input.as_slice());
    }
}
