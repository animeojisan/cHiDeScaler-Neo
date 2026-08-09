//! Tee logger: always stderr, optionally a file next to the exe
//! (`cHiDeScaler-Neo.log`, toggled from the GUI, default OFF).

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Mutex;

pub struct TeeLogger {
    file: Mutex<Option<std::fs::File>>,
}

static LOGGER: TeeLogger = TeeLogger {
    file: Mutex::new(None),
};

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::Level::Debug
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
        if let Ok(mut g) = self.file.lock() {
            if let Some(f) = g.as_mut() {
                let _ = f.write_all(line.as_bytes());
            }
        }
    }

    fn flush(&self) {
        if let Ok(mut g) = self.file.lock() {
            if let Some(f) = g.as_mut() {
                let _ = f.flush();
            }
        }
    }
}

pub fn init() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Debug);
}

/// Enable/disable the file sink (app_dir/cHiDeScaler-Neo.log, append).
pub fn set_file_logging(app_dir: &std::path::Path, on: bool) {
    let mut g = LOGGER.file.lock().unwrap();
    if on && g.is_none() {
        let path = app_dir.join("cHiDeScaler-Neo.log");
        if let Ok(f) = OpenOptions::new().create(true).append(true).open(&path) {
            *g = Some(f);
            drop(g);
            log::info!("=== file logging enabled ({}) ===", path.display());
            return;
        }
    } else if !on {
        *g = None;
    }
}
