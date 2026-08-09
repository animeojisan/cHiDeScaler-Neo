//! Plain-data config types shared between GUI, engine and persistence.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageSpec {
    pub kind: StageKind,
    /// Path relative to the app dir (forward slashes) or absolute.
    pub path: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Per-instance values for mpv-style `//!PARAM` declarations.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, f32>,
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StageKind {
    Glsl,
    Onnx,
    /// built-in GPU filter (e.g. NeoFlow frame interpolation)
    Flow,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaleMode {
    /// Fullscreen overlay on the source's monitor.
    Auto,
    /// Floating window of source-size x ratio, centered over the source.
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UiLanguage {
    #[serde(rename = "ja-JP", alias = "ja")]
    JaJp,
    #[serde(rename = "zh-CN")]
    ZhCn,
    #[serde(rename = "zh-TW")]
    ZhTw,
    #[serde(rename = "ko-KR")]
    KoKr,
    #[serde(rename = "pt-BR")]
    PtBr,
    #[serde(rename = "es")]
    Es,
    #[serde(rename = "fr-FR")]
    FrFr,
    #[serde(rename = "de-DE")]
    DeDe,
    #[serde(rename = "en-US", alias = "en", other)]
    EnUs,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UiLanguageMode {
    #[serde(rename = "ja-JP", alias = "ja")]
    JaJp,
    #[serde(rename = "en-US", alias = "en")]
    EnUs,
    #[serde(rename = "zh-CN")]
    ZhCn,
    #[serde(rename = "zh-TW")]
    ZhTw,
    #[serde(rename = "ko-KR")]
    KoKr,
    #[serde(rename = "pt-BR")]
    PtBr,
    #[serde(rename = "es")]
    Es,
    #[serde(rename = "fr-FR")]
    FrFr,
    #[serde(rename = "de-DE")]
    DeDe,
    #[serde(rename = "auto", other)]
    Auto,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UiMode {
    Mini,
    Basic,
    Full,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BrowserKind {
    #[default]
    Edge,
    Chrome,
    Firefox,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OnnxBackendPreference {
    #[default]
    DirectML,
    TensorRT,
}

/// Legacy HDR-to-SDR profile values retained only for settings-file compatibility.
/// Current builds ignore the saved selector and use the conservative Low400 mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HdrSdrMode {
    /// Strongest highlight protection for entry-level HDR displays (~400 nit).
    #[default]
    #[serde(rename = "low-400")]
    Low400,
    /// Balanced mapping for mid-range HDR displays (~600 nit).
    #[serde(rename = "mid-600")]
    Mid600,
    /// Wider highlight range for high-brightness HDR displays (~1000 nit).
    #[serde(rename = "high-1000")]
    High1000,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureResolution {
    pub w: u32,
    pub h: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default = "default_ui_mode")]
    pub ui_mode: UiMode,
    #[serde(default = "default_language_mode")]
    pub language_mode: UiLanguageMode,
    #[serde(default = "default_language")]
    pub language: UiLanguage,
    /// Optional user-supplied locale catalog filename stem (e.g. "it-IT").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub custom_language: Option<String>,
    #[serde(default)]
    pub fps_cap_enabled: bool,
    #[serde(default = "default_fps")]
    pub fps_cap: u32,
    #[serde(default = "default_mode")]
    pub scale_mode: ScaleMode,
    #[serde(default = "default_ratio")]
    pub ratio: f32,
    #[serde(default)]
    pub stats_on: bool,
    #[serde(default = "default_hotkey")]
    pub hotkey_toggle: String,
    #[serde(default)]
    pub vsync: bool,
    /// Buffer a small number of captured frames and present them on a stable
    /// source cadence. This improves motion smoothness at the cost of latency.
    #[serde(default = "default_true_s")]
    pub smooth_pacing: bool,
    /// Reuse the last completed upscale when the captured picture is a
    /// perceptual duplicate. Presentation cadence is preserved.
    #[serde(default)]
    pub duplicate_frame_reduction: bool,
    /// Reuse exact previous ONNX output and infer only safely bounded changed
    /// regions for supported local CNN models.
    #[serde(default = "default_true_s")]
    pub neo_accel: bool,
    /// keep the GUI window always on top (user option, default off)
    #[serde(default)]
    pub gui_topmost: bool,
    /// show the floating control panel during capture (default on)
    #[serde(default = "default_true_s")]
    pub panel_show: bool,
    /// write cHiDeScaler-Neo.log next to the exe (default off)
    #[serde(default)]
    pub log_on: bool,
    /// visually hide the source window during magnification (single-window UX)
    #[serde(default = "default_true_s")]
    pub hide_source: bool,
    /// capture the client area only (exclude title bar / borders)
    #[serde(default = "default_true_s")]
    pub client_only: bool,
    /// auto-hide the virtual cursor after N seconds of no movement
    #[serde(default = "default_true_s")]
    pub cursor_autohide: bool,
    #[serde(default = "default_autohide_secs")]
    pub cursor_autohide_secs: f32,
    /// natural cursor speed over the magnified view (off = raw source-space)
    #[serde(default = "default_true_s")]
    pub cursor_speed_fix: bool,
    /// frame interpolation output multiplier (2 = double fps)
    #[serde(default = "default_interp_factor")]
    pub interp_factor: u32,
    /// relaunch elevated at startup (needed to control elevated windows)
    #[serde(default)]
    pub run_as_admin: bool,
    /// Browser used by the portable software-video-decode launcher.
    #[serde(default)]
    pub browser_kind: BrowserKind,
    /// remembered GUI window placement (logical points)
    #[serde(default)]
    pub win_pos: Option<(f32, f32)>,
    #[serde(default)]
    pub win_size: Option<(f32, f32)>,
    /// Remember the compact window independently so switching back to Full
    /// restores the user's previous editor size.
    #[serde(default)]
    pub basic_win_size: Option<(f32, f32)>,
    /// Remember the minimal launcher size independently.
    #[serde(default)]
    pub mini_win_size: Option<(f32, f32)>,
    /// final downscale filter (spline36/lanczos3/bicubic/bilinear/nearest)
    #[serde(default = "default_downscaler")]
    pub downscaler: String,
    /// capture HDR (scRGB fp16) windows and tone-map them to SDR
    #[serde(default)]
    pub hdr_capture: bool,
    /// Legacy saved value; retained for compatibility and ignored by the fixed Low400 mapper.
    #[serde(default)]
    pub hdr_sdr_mode: HdrSdrMode,
    /// Stable DXGI adapter LUID for DirectML. None = automatic high performance.
    #[serde(default)]
    pub gpu_adapter_luid: Option<u64>,
    /// Global ONNX backend preference. Presets intentionally remain backend
    /// agnostic; unavailable TensorRT installations resolve to DirectML.
    #[serde(default)]
    pub onnx_backend: OnnxBackendPreference,
    /// Source-window client resize before/while capturing. None = auto/no resize.
    #[serde(default)]
    pub capture_resolution: Option<CaptureResolution>,
}

fn default_downscaler() -> String {
    "spline36".into()
}

fn default_language() -> UiLanguage {
    UiLanguage::JaJp
}

fn default_language_mode() -> UiLanguageMode {
    UiLanguageMode::Auto
}

fn default_ui_mode() -> UiMode {
    UiMode::Full
}

fn default_autohide_secs() -> f32 {
    3.0
}
fn default_interp_factor() -> u32 {
    2
}

fn default_true_s() -> bool {
    true
}

fn default_fps() -> u32 {
    60
}
fn default_mode() -> ScaleMode {
    ScaleMode::Auto
}
fn default_ratio() -> f32 {
    2.0
}
fn default_hotkey() -> String {
    "Ctrl+Alt+Z".into()
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            ui_mode: UiMode::Full,
            language_mode: UiLanguageMode::Auto,
            language: UiLanguage::JaJp,
            custom_language: None,
            fps_cap_enabled: false,
            fps_cap: 60,
            scale_mode: ScaleMode::Auto,
            ratio: 2.0,
            stats_on: false,
            hotkey_toggle: default_hotkey(),
            vsync: false,
            smooth_pacing: true,
            duplicate_frame_reduction: false,
            neo_accel: false,
            gui_topmost: false,
            panel_show: true,
            log_on: false,
            hide_source: true,
            client_only: true,
            cursor_autohide: true,
            cursor_autohide_secs: 3.0,
            cursor_speed_fix: true,
            interp_factor: 2,
            run_as_admin: false,
            browser_kind: BrowserKind::Edge,
            win_pos: None,
            win_size: None,
            basic_win_size: None,
            mini_win_size: None,
            downscaler: "spline36".into(),
            hdr_capture: false,
            hdr_sdr_mode: HdrSdrMode::Low400,
            gpu_adapter_luid: None,
            onnx_backend: OnnxBackendPreference::DirectML,
            capture_resolution: None,
        }
    }
}

/// Resolve a spec path against the app dir; searches shaders/ and models/
/// by basename as a fallback (robust against moved preset files).
pub fn resolve_path(app_dir: &std::path::Path, p: &str) -> std::path::PathBuf {
    let pb = std::path::PathBuf::from(p);
    if pb.is_absolute() && pb.exists() {
        return pb;
    }
    let direct = app_dir.join(&pb);
    if direct.exists() {
        return direct;
    }
    if let Some(base) = pb.file_name() {
        for sub in ["shaders", "models", "."] {
            let c = app_dir.join(sub).join(base);
            if c.exists() {
                return c;
            }
        }
        // categorized default layout (shaders/CRT/…, models/RIFE/…): find by
        // basename recursively so presets saved with old flat paths keep
        // working after the folder reorganization
        for sub in ["shaders", "models"] {
            if let Some(found) = find_by_basename(&app_dir.join(sub), base, 3) {
                return found;
            }
        }
    }
    direct
}

fn find_by_basename(
    dir: &std::path::Path,
    base: &std::ffi::OsStr,
    depth: u32,
) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut subdirs = Vec::new();
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            subdirs.push(p);
        } else if p.file_name() == Some(base) {
            return Some(p);
        }
    }
    if depth > 1 {
        for d in subdirs {
            if let Some(found) = find_by_basename(&d, base, depth - 1) {
                return Some(found);
            }
        }
    }
    None
}

pub fn app_dir() -> std::path::PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| ".".into())
}

#[cfg(test)]
mod tests {
    use super::{BrowserKind, OnnxBackendPreference, Settings, UiLanguage, UiLanguageMode, UiMode};

    #[test]
    fn smooth_pacing_is_on_for_new_and_legacy_missing_settings() {
        assert!(Settings::default().smooth_pacing);
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert!(parsed.smooth_pacing);
    }

    #[test]
    fn explicitly_disabled_smooth_pacing_is_preserved() {
        let parsed: Settings = serde_json::from_str(r#"{"smooth_pacing":false}"#).unwrap();
        assert!(!parsed.smooth_pacing);
    }

    #[test]
    fn file_logging_is_off_for_first_launch_and_legacy_missing_settings() {
        assert!(!Settings::default().log_on);
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert!(!parsed.log_on);
    }

    #[test]
    fn legacy_settings_open_in_full_mode() {
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.ui_mode, UiMode::Full);
    }

    #[test]
    fn legacy_settings_default_to_edge_browser_launcher() {
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.browser_kind, BrowserKind::Edge);
    }

    #[test]
    fn legacy_settings_default_to_directml() {
        let parsed: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(parsed.onnx_backend, OnnxBackendPreference::DirectML);
    }

    #[test]
    fn tensorrt_preference_round_trips_without_touching_presets() {
        let mut settings = Settings::default();
        settings.onnx_backend = OnnxBackendPreference::TensorRT;
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.onnx_backend, OnnxBackendPreference::TensorRT);
    }

    #[test]
    fn basic_and_full_window_sizes_round_trip_independently() {
        let mut settings = Settings::default();
        settings.ui_mode = UiMode::Basic;
        settings.basic_win_size = Some((700.0, 214.0));
        settings.win_size = Some((960.0, 840.0));
        let json = serde_json::to_string(&settings).unwrap();
        let parsed: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.ui_mode, UiMode::Basic);
        assert_eq!(parsed.basic_win_size, Some((700.0, 214.0)));
        assert_eq!(parsed.win_size, Some((960.0, 840.0)));
    }

    #[test]
    fn legacy_ja_and_en_language_values_remain_compatible() {
        let ja: Settings =
            serde_json::from_str(r#"{"language_mode":"ja","language":"ja"}"#).unwrap();
        assert_eq!(ja.language_mode, UiLanguageMode::JaJp);
        assert_eq!(ja.language, UiLanguage::JaJp);
        let en: Settings =
            serde_json::from_str(r#"{"language_mode":"en","language":"en"}"#).unwrap();
        assert_eq!(en.language_mode, UiLanguageMode::EnUs);
        assert_eq!(en.language, UiLanguage::EnUs);
    }

    #[test]
    fn unknown_language_values_do_not_reset_other_settings() {
        let parsed: Settings = serde_json::from_str(
            r#"{"language_mode":"xx-YY","language":"xx-YY","fps_cap":144,"stats_on":true}"#,
        )
        .unwrap();
        assert_eq!(parsed.language_mode, UiLanguageMode::Auto);
        assert_eq!(parsed.language, UiLanguage::EnUs);
        assert_eq!(parsed.fps_cap, 144);
        assert!(parsed.stats_on);
    }
}
