//! Runtime logger for Neo.
//!
//! Normal mode keeps only operational INFO/WARN/ERROR output on stderr.
//! Enabling the existing "Save log" option turns on the detailed diagnostic
//! stream (DEBUG + explicitly gated high-frequency diagnostics) and tees it to
//! `cHiDeScaler-Neo.log` next to the executable.
//!
//! Diagnostic files are bounded: the active file is capped at 10 MiB and Neo
//! keeps two older generations, for roughly 30 MiB maximum total log storage.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex,OnceLock,mpsc::{self,SyncSender,Receiver,TrySendError}};
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use std::sync::atomic::{AtomicBool, Ordering};

const LOG_FILE_NAME: &str = "cHiDeScaler-Neo.log";
const LOG_MAX_BYTES: u64 = 10 * 1024 * 1024;
const LOG_GENERATIONS: usize = 3; // active + .1 + .2 = about 30 MiB maximum

struct LogFileState {
    file: std::fs::File,
    path: PathBuf,
    bytes_written: u64,
}

pub struct TeeLogger {
    file: Mutex<Option<LogFileState>>,
    file_enabled: AtomicBool,
    diagnostics_enabled: AtomicBool,
    writer:OnceLock<Option<SyncSender<LogCommand>>>,
    file_epoch:AtomicU64,
    dropped:AtomicU64,
}

static LOGGER: TeeLogger = TeeLogger {
    file: Mutex::new(None),
    file_enabled: AtomicBool::new(false),
    diagnostics_enabled: AtomicBool::new(false),
    writer:OnceLock::new(),
    file_epoch:AtomicU64::new(0),
    dropped:AtomicU64::new(0),
};

fn rotated_log_path(active: &Path, generation: usize) -> PathBuf {
    let parent = active.parent().unwrap_or_else(|| Path::new(""));
    let stem = active
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("cHiDeScaler-Neo");
    match active.extension().and_then(|s| s.to_str()) {
        Some(ext) if !ext.is_empty() => parent.join(format!("{stem}.{generation}.{ext}")),
        _ => parent.join(format!("{stem}.{generation}")),
    }
}

/// Rotate active -> .1 -> .2 and open a fresh active file. The caller must have
/// already dropped the active file handle so Windows can rename it reliably.
fn rotate_and_open(active: &Path) -> std::io::Result<LogFileState> {
    for generation in (1..LOG_GENERATIONS).rev() {
        let from = if generation == 1 {
            active.to_path_buf()
        } else {
            rotated_log_path(active, generation - 1)
        };
        let to = rotated_log_path(active, generation);

        if to.exists() {
            fs::remove_file(&to)?;
        }
        if from.exists() {
            fs::rename(&from, &to)?;
        }
    }

    let file = OpenOptions::new().create(true).append(true).open(active)?;
    Ok(LogFileState {
        file,
        path: active.to_path_buf(),
        bytes_written: 0,
    })
}

fn open_bounded_log(active: &Path) -> std::io::Result<LogFileState> {
    let existing_len = fs::metadata(active).map(|m| m.len()).unwrap_or(0);
    if existing_len >= LOG_MAX_BYTES {
        return rotate_and_open(active);
    }

    let file = OpenOptions::new().create(true).append(true).open(active)?;
    Ok(LogFileState {
        file,
        path: active.to_path_buf(),
        bytes_written: existing_len,
    })
}

fn disable_diagnostic_file_logging_after_io_error(error: &std::io::Error, path: &Path) {
    LOGGER.file_enabled.store(false, Ordering::Release);
    LOGGER.diagnostics_enabled.store(false, Ordering::Release);
    log::set_max_level(log::LevelFilter::Info);
    // Do not recurse through the logger while handling a logger I/O failure.
    eprintln!(
        "cHiDeScaler-Neo: diagnostic logging disabled after log I/O error: path={} error={error}",
        path.display()
    );
}

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        if self.diagnostics_enabled.load(Ordering::Relaxed) {
            // Naga's parser/validator emits extremely verbose per-expression
            // DEBUG traces. A single Vulkan shader compile can otherwise add
            // tens of thousands of lines to Neo's diagnostic log. Keep Naga
            // INFO/WARN/ERROR (including compile failures), but suppress only
            // its internal DEBUG stream. Neo/wgpu diagnostics remain unchanged.
            if metadata.level() == log::Level::Debug && metadata.target().starts_with("naga") {
                return false;
            }
            metadata.level() <= log::Level::Debug
        } else {
            metadata.level() <= log::Level::Info
        }
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let line = format!(
            "[{}.{:03}] {} [{}] {}\n",
            ts / 1000,
            ts % 1000,
            record.level(),
            record.target().split("::").last().unwrap_or(""),
            record.args()
        );
        if !self.file_enabled.load(Ordering::Relaxed) {
            eprint!("{line}");
            return;
        }
        // Never take a disk/file mutex or wait for the writer on GPU threads.
        let epoch=self.file_epoch.load(Ordering::Acquire);
        match self.writer.get().and_then(Option::as_ref) {
            Some(writer)=>match writer.try_send(LogCommand::Line(epoch,line)) {
                Ok(())=>{},
                Err(TrySendError::Full(_))=>{self.dropped.fetch_add(1,Ordering::Relaxed);},
                Err(TrySendError::Disconnected(_))=>{
                    self.file_enabled.store(false,Ordering::Release);
                    self.diagnostics_enabled.store(false,Ordering::Release);
                    log::set_max_level(log::LevelFilter::Info);
                }
            },
            None=>{self.dropped.fetch_add(1,Ordering::Relaxed);}
        }
    }

    fn flush(&self) { self.drain_writer(); }

}

enum LogCommand { Line(u64,String), Flush(SyncSender<()>) }

fn run_writer(receiver:Receiver<LogCommand>,mut write:impl FnMut(u64,String),mut flush:impl FnMut()) {
    while let Ok(command)=receiver.recv(){
        match command{
            LogCommand::Line(epoch,line)=>write(epoch,line),
            LogCommand::Flush(ack)=>{flush();let _=ack.try_send(());}
        }
    }
    flush();
}

impl TeeLogger {
    fn write_file_line(&self,line:&str) {
        if let Ok(mut g) = self.file.lock() {
            let line_len = line.len() as u64;
            let need_rotate = g
                .as_ref()
                .map(|state| {
                    state.bytes_written > 0
                        && state.bytes_written.saturating_add(line_len) > LOG_MAX_BYTES
                })
                .unwrap_or(false);

            if need_rotate {
                let path = match g.as_ref() {
                    Some(state) => state.path.clone(),
                    None => return,
                };

                // Drop the open handle before rename; required for reliable
                // rotation on Windows. Rotation happens only once per 10 MiB,
                // so normal log writes never perform filesystem size queries.
                if let Some(mut old) = g.take() {
                    let _ = old.file.flush();
                }

                match rotate_and_open(&path) {
                    Ok(state) => *g = Some(state),
                    Err(error) => {
                        disable_diagnostic_file_logging_after_io_error(&error, &path);
                        return;
                    }
                }
            }

            let write_error = if let Some(state) = g.as_mut() {
                match state.file.write_all(line.as_bytes()) {
                    Ok(()) => {
                        state.bytes_written = state.bytes_written.saturating_add(line_len);
                        None
                    }
                    Err(error) => Some((error, state.path.clone())),
                }
            } else {
                None
            };

            if let Some((error, path)) = write_error {
                *g = None;
                disable_diagnostic_file_logging_after_io_error(&error, &path);
            }
        }
    }

    fn flush_file_now(&self) {
        if let Ok(mut state)=self.file.lock(){if let Some(state)=state.as_mut(){let _=state.file.flush();}}
    }

    fn start_writer(&self)->bool {
        self.writer.get_or_init(||{
            let(sender,receiver)=mpsc::sync_channel(2048);
            std::thread::Builder::new().name("diagnostic-log-writer".into()).spawn(move||{
                run_writer(receiver,|epoch,line|{
                    if epoch==LOGGER.file_epoch.load(Ordering::Acquire){
                        let _=std::io::stderr().write_all(line.as_bytes());
                        LOGGER.write_file_line(&line);
                    }
                    let dropped=LOGGER.dropped.swap(0,Ordering::Relaxed);
                    if dropped>0{
                        let ts=std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d|d.as_millis()).unwrap_or(0);
                        LOGGER.write_file_line(&format!("[{}.{:03}] WARN [logging] diagnostic-writer-overflow: dropped={} records queue_limit=2048 rendering_not_blocked=true\n",ts/1000,ts%1000,dropped));
                    }
                },||LOGGER.flush_file_now());
            }).ok().map(|_|sender)
        }).is_some()
    }

    fn drain_writer(&self) {
        if let Some(writer)=self.writer.get().and_then(Option::as_ref){
            let(tx,rx)=mpsc::sync_channel(1);
            let limit=std::time::Instant::now()+Duration::from_millis(500);
            let mut command=LogCommand::Flush(tx);
            loop{
                match writer.try_send(command){
                    Ok(())=>{let _=rx.recv_timeout(limit.saturating_duration_since(std::time::Instant::now()));break;},
                    Err(TrySendError::Full(returned)) if std::time::Instant::now()<limit=>{
                        command=returned;std::thread::sleep(Duration::from_millis(1));
                    },
                    Err(_)=>break
                }
            }
        }
        self.flush_file_now();
    }
}

/// Normal GUI exit drains already queued diagnostics after engine shutdown.
pub struct FlushOnExit;
impl Drop for FlushOnExit { fn drop(&mut self){log::logger().flush();} }

pub fn init() {
    let _ = log::set_logger(&LOGGER);
    // Default/first-launch mode intentionally excludes DEBUG diagnostics.
    log::set_max_level(log::LevelFilter::Info);
}

/// True only while the user-facing "Save log" option is enabled and the log
/// file could be opened. Hot diagnostic paths should gate any extra sampling,
/// aggregation or string construction with this function.
#[inline]
pub fn diagnostics_enabled() -> bool {
    LOGGER.diagnostics_enabled.load(Ordering::Relaxed)
}

/// Existing GUI option semantics:
/// - OFF: lightweight normal logging only, no file sink, diagnostics disabled.
/// - ON: detailed diagnostics enabled and written to a bounded rotating log.
///
/// If the file cannot be opened, Neo falls back to normal logging rather than
/// paying the diagnostic cost without producing the requested file.
pub fn set_file_logging(app_dir: &Path, on: bool) {
    if !on {
        LOGGER.diagnostics_enabled.store(false, Ordering::Release);
        log::set_max_level(log::LevelFilter::Info);
        LOGGER.file_enabled.store(false, Ordering::Release);
        LOGGER.drain_writer();
        LOGGER.file_epoch.fetch_add(1,Ordering::AcqRel);
        if let Ok(mut g) = LOGGER.file.lock() {
            if let Some(mut state) = g.take() {
                let _ = state.file.flush();
            }
        }
        return;
    }

    // Repeated UI/settings application is harmless. If already active, keep
    // the current file handle and diagnostic level intact.
    if LOGGER.file_enabled.load(Ordering::Acquire) {
        LOGGER.diagnostics_enabled.store(true, Ordering::Release);
        log::set_max_level(log::LevelFilter::Debug);
        return;
    }

    if !LOGGER.start_writer(){
        log::warn!("diagnostic writer could not start; file logging remains disabled");
        return;
    }
    LOGGER.file_epoch.fetch_add(1,Ordering::AcqRel);
    let path = app_dir.join(LOG_FILE_NAME);
    match open_bounded_log(&path) {
        Ok(state) => {
            if let Ok(mut g) = LOGGER.file.lock() {
                *g = Some(state);
            }
            LOGGER.file_enabled.store(true, Ordering::Release);
            LOGGER.diagnostics_enabled.store(true, Ordering::Release);
            log::set_max_level(log::LevelFilter::Debug);
            log::info!(
                "=== diagnostic file logging enabled ({}) rotation=10MiB x3 sink=async queue_limit=2048 ===",
                path.display()
            );
        }
        Err(error) => {
            LOGGER.file_enabled.store(false, Ordering::Release);
            LOGGER.diagnostics_enabled.store(false, Ordering::Release);
            log::set_max_level(log::LevelFilter::Info);
            log::warn!(
                "diagnostic file logging unavailable: path={} error={error}",
                path.display()
            );
        }
    }
}

#[cfg(test)]
mod async_writer_tests {
    use super::*;
    use std::sync::{Arc,atomic::AtomicUsize};
    #[test]
    fn async_writer_keeps_order_and_flush_is_a_barrier(){
        let lines=Arc::new(Mutex::new(Vec::new()));let writes=lines.clone();
        let flushed=Arc::new(AtomicUsize::new(0));let count=flushed.clone();
        let(tx,rx)=mpsc::sync_channel(16);
        let handle=std::thread::spawn(move||run_writer(rx,|epoch,line|writes.lock().unwrap().push((epoch,line)),||{count.fetch_add(1,Ordering::SeqCst);}));
        for text in ["one","two","three"]{tx.send(LogCommand::Line(7,text.into())).unwrap();}
        let(ack,done)=mpsc::sync_channel(1);tx.send(LogCommand::Flush(ack)).unwrap();done.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(*lines.lock().unwrap(),vec![(7,"one".into()),(7,"two".into()),(7,"three".into())]);
        assert_eq!(flushed.load(Ordering::SeqCst),1);drop(tx);handle.join().unwrap();
    }
    #[test]
    fn full_writer_queue_never_blocks_log_producer(){
        let(tx,rx)=mpsc::sync_channel(2);let(done_tx,done_rx)=mpsc::sync_channel(1);
        let producer=std::thread::spawn(move||{
            let writer=OnceLock::new();let _=writer.set(Some(tx));
            let logger=TeeLogger{file:Mutex::new(None),file_enabled:AtomicBool::new(true),diagnostics_enabled:AtomicBool::new(true),writer,file_epoch:AtomicU64::new(1),dropped:AtomicU64::new(0)};
            let record=log::Record::builder().level(log::Level::Debug).target("probe").args(format_args!("bounded queue test")).build();
            for _ in 0..1002{log::Log::log(&logger,&record);}
            let _=done_tx.send(logger.dropped.load(Ordering::Relaxed));
        });
        let result=done_rx.recv_timeout(Duration::from_secs(2));drop(rx);producer.join().unwrap();
        assert_eq!(result.unwrap(),1000);
    }
}
