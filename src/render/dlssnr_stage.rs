//! Optional RGBA8 transport. Untrusted/vendor GPU code executes outside Neo.
use super::dlssnr_backend::{DLSSNR_CAP_ZERO_GUIDANCE, DlssNrBackend, detect_dlssnr_backend_pack};
use crate::core::dlssnr::DlssNrOptions;
use super::gl::{GlContext, GpuTex};
use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

const READY: [u8; 4] = *b"NR02";
const MAX_BYTES: usize = 256 * 1024 * 1024;

fn frame_bytes(w: u32, h: u32) -> Result<usize, String> {
    let bytes = (w as usize)
        .checked_mul(h as usize)
        .and_then(|n| n.checked_mul(4));
    bytes
        .filter(|n| *n > 0 && *n <= MAX_BYTES)
        .ok_or_else(|| "invalid DLSSNR frame size".into())
}

fn align4(value: u32) -> Result<u32, String> {
    value
        .checked_add(3)
        .map(|v| v & !3)
        .filter(|v| *v > 0)
        .ok_or_else(|| "DLSSNR aligned geometry overflow".into())
}

fn pad_rgba8_edge(
    input: &[u8],
    width: u32,
    height: u32,
    padded_width: u32,
    padded_height: u32,
) -> Result<Vec<u8>, String> {
    let input_bytes = frame_bytes(width, height)?;
    let padded_bytes = frame_bytes(padded_width, padded_height)?;
    if input.len() != input_bytes || padded_width < width || padded_height < height {
        return Err("invalid DLSSNR padding geometry".into());
    }
    if width == padded_width && height == padded_height {
        return Ok(input.to_vec());
    }
    let src_stride = width as usize * 4;
    let dst_stride = padded_width as usize * 4;
    let mut output = vec![0u8; padded_bytes];
    for y in 0..padded_height as usize {
        let sy = y.min(height as usize - 1);
        let src = &input[sy * src_stride..(sy + 1) * src_stride];
        let dst = &mut output[y * dst_stride..(y + 1) * dst_stride];
        dst[..src_stride].copy_from_slice(src);
        let edge = &src[src_stride - 4..src_stride];
        for x in width as usize..padded_width as usize {
            dst[x * 4..x * 4 + 4].copy_from_slice(edge);
        }
    }
    Ok(output)
}

fn crop_rgba8(
    input: &[u8],
    padded_width: u32,
    padded_height: u32,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, String> {
    let padded_bytes = frame_bytes(padded_width, padded_height)?;
    if input.len() != padded_bytes || width > padded_width || height > padded_height {
        return Err("invalid DLSSNR crop geometry".into());
    }
    if width == padded_width && height == padded_height {
        return Ok(input.to_vec());
    }
    let src_stride = padded_width as usize * 4;
    let dst_stride = width as usize * 4;
    let mut output = vec![0u8; frame_bytes(width, height)?];
    for y in 0..height as usize {
        output[y * dst_stride..(y + 1) * dst_stride]
            .copy_from_slice(&input[y * src_stride..y * src_stride + dst_stride]);
    }
    Ok(output)
}

struct Worker {
    child: Child,
    jobs: Option<SyncSender<(Vec<u8>, bool, Option<DlssNrOptions>)>>,
    replies: Receiver<Result<Vec<u8>, String>>,
    ready: bool,
    started: Instant,
    initialized: Arc<AtomicBool>,
}

impl Worker {
    fn start(
        app: &std::path::Path,
        w: u32,
        h: u32,
        luid: u64,
        options: DlssNrOptions,
    ) -> Result<Self, String> {
        let bytes = frame_bytes(w, h)?;
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(app.join("dlssnr-worker.log"))
            .map_err(|e| e.to_string())?;
        let mut child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
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
            .spawn()
            .map_err(|e| e.to_string())?;
        let mut input = child.stdin.take().ok_or("missing worker stdin")?;
        let mut output = child.stdout.take().ok_or("missing worker stdout")?;
        let (jobs, incoming) = mpsc::sync_channel::<(Vec<u8>, bool, Option<DlssNrOptions>)>(1);
        let (completed, replies) = mpsc::sync_channel(1);
        let initialized = Arc::new(AtomicBool::new(false));
        let ready_flag = initialized.clone();
        std::thread::spawn(move || {
            let mut transfer = || -> Result<(), String> {
                let mut ready = [0; 4];
                output
                    .read_exact(&mut ready)
                    .map_err(|e| format!("initialize: {e}"))?;
                if ready != READY {
                    return Err("invalid worker handshake".into());
                }
                completed.send(Ok(Vec::new())).map_err(|e| e.to_string())?;
                ready_flag.store(true, Ordering::Release);
                while let Ok((pixels, reset, options_update)) = incoming.recv() {
                    if pixels.len() != bytes {
                        return Err("invalid input length".into());
                    }
                    input
                        .write_all(&[u8::from(reset), u8::from(options_update.is_some())])
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
                    input
                        .write_all(&pixels)
                        .and_then(|_| input.flush())
                        .map_err(|e| format!("input: {e}"))?;
                    let mut pixels = vec![0; bytes];
                    output
                        .read_exact(&mut pixels)
                        .map_err(|e| format!("evaluate: {e}"))?;
                    completed.send(Ok(pixels)).map_err(|e| e.to_string())?;
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
            ready: false,
            started: Instant::now(),
            initialized,
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
            // The render/model preset is supplied through the create descriptor and can
            // be latched at CreateFeature. Recreate only this isolated DLSSNR
            // worker; capture/GL/Present stay untouched. Style and strengths use
            // the live option setter below.
            self.worker = None;
            self.geometry = None;
            self.evaluated = false;
            self.last_frame = None;
            self.options_dirty = false;
        } else {
            // Strength/mask controls are forwarded to the live backend session
            // on the next frame, avoiding a child/D3D12/Feature recreation for
            // every slider tick.
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
        self.geometry = None;
        self.disabled = false;
        self.evaluated = false;
        self.last_frame = None;
        self.options_dirty = false;
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
        }
    }

    fn fail(&mut self, reason: &str) {
        log::warn!("dlssnr-bypass: reason={reason} action=disable-session original-frame=true");
        self.disabled = true;
        self.worker = None;
    }

    pub fn apply(&mut self, gc: &mut GlContext, source: GpuTex) -> GpuTex {
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
            self.worker = None;
            self.geometry = Some(geometry);
            self.evaluated = false;
            match Worker::start(&self.app, geometry.0, geometry.1, luid, self.options) {
                Ok(worker) => {
                    let aligned = (align4(geometry.0).unwrap_or(geometry.0), align4(geometry.1).unwrap_or(geometry.1));
                    log::info!(
                        "dlssnr-initialize: {}x{} RGBA8 working={}x{} alignment=4 edge_pad={} luid={luid:016x} process-isolated=true options={:?}",
                        geometry.0,
                        geometry.1,
                        aligned.0,
                        aligned.1,
                        aligned != (geometry.0, geometry.1),
                        self.options
                    );
                    self.worker = Some(worker);
                    self.options_dirty = false;
                }
                Err(error) => self.fail(&error),
            }
            return source;
        }
        let Some(worker) = self.worker.as_mut() else {
            return source;
        };
        if !worker.ready {
            match worker.replies.try_recv() {
                Ok(Ok(_)) => {
                    worker.ready = true;
                    log::info!("dlssnr-create: success=true");
                }
                Ok(Err(e)) => {
                    self.fail(&e);
                    return source;
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    self.fail("worker exited during creation");
                    return source;
                }
                Err(_) => {
                    if worker.started.elapsed() > Duration::from_secs(60) {
                        self.fail("initialization timeout");
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
        let pixels = gc.download_rgba8(source);
        let options_update = self.options_dirty.then_some(self.options);
        if worker
            .jobs
            .as_ref()
            .unwrap()
            .try_send((pixels, reset, options_update))
            .is_err()
        {
            self.fail("worker queue unavailable");
            return source;
        }
        if options_update.is_some() {
            self.options_dirty = false;
        }
        match worker.replies.recv_timeout(Duration::from_millis(1000)) {
            Ok(Ok(output)) => {
                if !self.evaluated {
                    log::info!(
                        "dlssnr-evaluate: success=true output={}x{} RGBA8 transport=isolated-cpu-bridge",
                        source.w(),
                        source.h()
                    );
                    self.evaluated = true;
                }
                gc.upload_rgba8(source.w(), source.h(), &output)
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
    let work_w = align4(w)?;
    let work_h = align4(h)?;
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
        "Feature 18 Create succeeded logical={w}x{h} working={work_w}x{work_h} RGBA8 edge_pad={} luid={luid:016x} options={options:?}",
        work_w != w || work_h != h
    );
    let mut input = std::io::stdin().lock();
    let mut output = std::io::stdout().lock();
    output
        .write_all(&READY)
        .and_then(|_| output.flush())
        .map_err(|e| e.to_string())?;
    let mut pixels = vec![0; bytes];
    let mut frames = 0u64;
    loop {
        let mut header = [0u8; 2];
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
        input.read_exact(&mut pixels).map_err(|e| e.to_string())?;
        let padded = pad_rgba8_edge(&pixels, w, h, work_w, work_h)?;
        let processed = session.process_rgba8(&padded, work_w * 4, header[0] != 0)?;
        let cropped = crop_rgba8(&processed, work_w, work_h, w, h)?;
        output
            .write_all(&cropped)
            .and_then(|_| output.flush())
            .map_err(|e| e.to_string())?;
        frames += 1;
        if frames == 1 {
            eprintln!("Feature 18 Evaluate succeeded");
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
    fn four_pixel_alignment_covers_1440x810_without_changing_aligned_modes() {
        assert_eq!(align4(1440).unwrap(), 1440);
        assert_eq!(align4(810).unwrap(), 812);
        assert_eq!(align4(1280).unwrap(), 1280);
        assert_eq!(align4(720).unwrap(), 720);
        assert_eq!(align4(1600).unwrap(), 1600);
        assert_eq!(align4(900).unwrap(), 900);
    }

    #[test]
    fn edge_padding_and_crop_preserve_original_pixels() {
        let input = vec![
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
        ];
        let padded = pad_rgba8_edge(&input, 2, 2, 4, 4).unwrap();
        assert_eq!(&padded[0..16], &[1,2,3,4,5,6,7,8,5,6,7,8,5,6,7,8]);
        assert_eq!(&padded[48..64], &[9,10,11,12,13,14,15,16,13,14,15,16,13,14,15,16]);
        assert_eq!(crop_rgba8(&padded, 4, 4, 2, 2).unwrap(), input);
    }
}
