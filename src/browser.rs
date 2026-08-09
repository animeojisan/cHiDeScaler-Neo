//! Portable browser launcher for software video decoding.
//!
//! No browser policy, registry value, normal browser profile, or user-level
//! configuration is modified. All launcher-owned profile data lives below the
//! cHiDeScaler-Neo application directory.

use crate::core::config::BrowserKind;
use std::path::{Path, PathBuf};

const FIREFOX_USER_JS: &str = r#"// Managed by cHiDeScaler-Neo.
// This file belongs only to the dedicated portable browser profile.
user_pref("media.hardware-video-decoding.enabled", false);
user_pref("media.wmf.dxva.enabled", false);
user_pref("browser.shell.checkDefaultBrowser", false);
"#;

impl BrowserKind {
    pub fn display_name(self) -> &'static str {
        match self {
            Self::Edge => "Microsoft Edge",
            Self::Chrome => "Google Chrome",
            Self::Firefox => "Mozilla Firefox",
        }
    }

    fn executable_name(self) -> &'static str {
        match self {
            Self::Edge => "msedge.exe",
            Self::Chrome => "chrome.exe",
            Self::Firefox => "firefox.exe",
        }
    }

    fn profile_folder(self) -> &'static str {
        match self {
            Self::Edge => "Edge",
            Self::Chrome => "Chrome",
            Self::Firefox => "Firefox",
        }
    }
}

pub fn profile_dir(app_dir: &Path, browser: BrowserKind) -> PathBuf {
    app_dir
        .join("cache")
        .join("BrowserProfiles")
        .join(browser.profile_folder())
}

/// Launch a browser with hardware video decoding disabled.
///
/// The returned path is the dedicated profile directory. A successful return
/// means Windows accepted the launch request; browser startup may still report
/// its own error (for example, if the same Firefox profile is already open).
pub fn launch_hw_decode_off(app_dir: &Path, browser: BrowserKind) -> Result<PathBuf, String> {
    let profile = profile_dir(app_dir, browser);
    std::fs::create_dir_all(&profile).map_err(|error| {
        format!(
            "could not create portable browser profile {}: {error}",
            profile.display()
        )
    })?;

    if browser == BrowserKind::Firefox {
        prepare_firefox_profile(&profile)?;
    }

    log::info!(
        "browser-launch-request: browser={} profile={}",
        browser.display_name(),
        profile.display()
    );
    let launch_result = match browser {
        BrowserKind::Edge | BrowserKind::Chrome => {
            let params = chromium_parameters(&profile);
            shell_execute(browser.executable_name(), &params)
        }
        BrowserKind::Firefox => launch_firefox(&profile),
    };
    launch_result.map_err(|error| {
        log::warn!(
            "browser-launch-failed: browser={} error={error}",
            browser.display_name()
        );
        format!("{}: {error}", browser.display_name())
    })?;
    log::info!(
        "browser-launch-ok: browser={} profile={}",
        browser.display_name(),
        profile.display()
    );
    Ok(profile)
}

fn chromium_parameters(profile: &Path) -> String {
    let disk_cache = profile.join("DiskCache");
    format!(
        "--user-data-dir={} --disk-cache-dir={} --disable-accelerated-video-decode \
         --disable-background-mode --no-first-run --no-default-browser-check --new-window",
        quote_argument(profile),
        quote_argument(&disk_cache),
    )
}

#[cfg(target_os = "windows")]
fn launch_firefox(profile: &Path) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    use std::process::Command;
    use windows::Win32::System::Threading::CREATE_NO_WINDOW;

    let local_profile = profile.join("Local");
    std::fs::create_dir_all(&local_profile).map_err(|error| {
        format!(
            "could not create Firefox local profile {}: {error}",
            local_profile.display()
        )
    })?;

    // Firefox separates roaming profile data and machine-local caches. These
    // variables are applied only to the launched Firefox process tree, never
    // to Windows or the user's persistent environment.
    if let Some(executable) = find_firefox_executable() {
        Command::new(&executable)
            .arg("--profile")
            .arg(profile)
            .args(["--no-remote", "--new-instance", "--browser"])
            .env("XRE_PROFILE_PATH", profile)
            .env("XRE_PROFILE_LOCAL_PATH", &local_profile)
            .creation_flags(CREATE_NO_WINDOW.0)
            .spawn()
            .map_err(|error| {
                format!(
                    "could not start Firefox at {}: {error}",
                    executable.display()
                )
            })?;
        return Ok(());
    }

    // Fallback for non-standard installs registered with Windows App Paths.
    // cmd/start inherits the two temporary environment variables above and
    // returns immediately without modifying the user's normal Firefox setup.
    let command_line = format!(
        "start \"\" \"firefox.exe\" --profile {} --no-remote --new-instance --browser",
        quote_argument(profile)
    );
    let status = Command::new("cmd.exe")
        .args(["/D", "/C", &command_line])
        .env("XRE_PROFILE_PATH", profile)
        .env("XRE_PROFILE_LOCAL_PATH", &local_profile)
        .creation_flags(CREATE_NO_WINDOW.0)
        .status()
        .map_err(|error| format!("could not start the Firefox launcher: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Firefox launcher exited with status {}",
            status.code().unwrap_or(-1)
        ))
    }
}

#[cfg(target_os = "windows")]
fn find_firefox_executable() -> Option<PathBuf> {
    let candidates = [
        ("ProgramFiles", r"Mozilla Firefox\firefox.exe"),
        ("ProgramFiles(x86)", r"Mozilla Firefox\firefox.exe"),
        ("LOCALAPPDATA", r"Mozilla Firefox\firefox.exe"),
    ];
    candidates.into_iter().find_map(|(variable, relative)| {
        let path = std::env::var_os(variable)
            .map(PathBuf::from)?
            .join(relative);
        path.is_file().then_some(path)
    })
}

#[cfg(not(target_os = "windows"))]
fn launch_firefox(_profile: &Path) -> Result<(), String> {
    Err("browser launch is supported only on Windows".to_string())
}

fn prepare_firefox_profile(profile: &Path) -> Result<(), String> {
    let user_js = profile.join("user.js");
    std::fs::write(&user_js, FIREFOX_USER_JS)
        .map_err(|error| format!("could not write {}: {error}", user_js.display()))
}

fn quote_argument(path: &Path) -> String {
    // Windows file and directory names cannot contain a literal double quote.
    // The profile paths never end in a separator, so simple quoting is safe.
    format!("\"{}\"", path.display())
}

#[cfg(target_os = "windows")]
fn shell_execute(file: &str, parameters: &str) -> Result<(), String> {
    use windows::Win32::UI::Shell::ShellExecuteW;
    use windows::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;
    use windows::core::PCWSTR;

    let verb: Vec<u16> = "open\0".encode_utf16().collect();
    let file: Vec<u16> = file.encode_utf16().chain(std::iter::once(0)).collect();
    let parameters: Vec<u16> = parameters
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let result = unsafe {
        ShellExecuteW(
            None,
            PCWSTR(verb.as_ptr()),
            PCWSTR(file.as_ptr()),
            PCWSTR(parameters.as_ptr()),
            None,
            SW_SHOWNORMAL,
        )
    };
    let code = result.0 as isize;
    if code > 32 {
        Ok(())
    } else {
        Err(match code {
            2 => "browser executable was not found".to_string(),
            3 => "browser executable path was not found".to_string(),
            5 => "access was denied".to_string(),
            8 => "not enough memory to start the browser".to_string(),
            31 => "Windows could not open the browser executable".to_string(),
            other => format!("ShellExecuteW failed with code {other}"),
        })
    }
}

#[cfg(not(target_os = "windows"))]
fn shell_execute(_file: &str, _parameters: &str) -> Result<(), String> {
    Err("browser launch is supported only on Windows".to_string())
}

#[cfg(test)]
mod tests {
    use super::{FIREFOX_USER_JS, prepare_firefox_profile, profile_dir};
    use crate::core::config::BrowserKind;
    use std::path::Path;

    #[test]
    fn every_profile_stays_below_the_application_directory() {
        let app = Path::new(r"C:\Portable\cHiDeScaler-Neo");
        for browser in [BrowserKind::Edge, BrowserKind::Chrome, BrowserKind::Firefox] {
            let profile = profile_dir(app, browser);
            assert!(profile.starts_with(app));
            assert!(profile.to_string_lossy().contains("BrowserProfiles"));
        }
    }

    #[test]
    fn firefox_profile_disables_video_hardware_decoding() {
        let unique = format!(
            "chidescaler-neo-browser-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        prepare_firefox_profile(&dir).unwrap();
        let text = std::fs::read_to_string(dir.join("user.js")).unwrap();
        assert_eq!(text, FIREFOX_USER_JS);
        assert!(text.contains("media.hardware-video-decoding.enabled\", false"));
        assert!(text.contains("media.wmf.dxva.enabled\", false"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
