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
use std::sync::Mutex;
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
}

static LOGGER: TeeLogger = TeeLogger {
    file: Mutex::new(None),
    file_enabled: AtomicBool::new(false),
    diagnostics_enabled: AtomicBool::new(false),
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
        eprint!("{line}");

        // Avoid even taking the file mutex in the default lightweight mode.
        if !self.file_enabled.load(Ordering::Relaxed) {
            return;
        }

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

    fn flush(&self) {
        if !self.file_enabled.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut g) = self.file.lock() {
            if let Some(state) = g.as_mut() {
                let _ = state.file.flush();
            }
        }
    }
}

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
                "=== diagnostic file logging enabled ({}) rotation=10MiB x3 ===",
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
