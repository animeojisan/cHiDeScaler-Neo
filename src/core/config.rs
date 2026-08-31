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

/// How the final display-only aspect correction is derived. Manual preserves
/// the v576 width/height controls. Auto modes recompute from the current
/// pre-correction presentation aspect whenever that aspect changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AspectCorrectionMode {
    #[default]
    #[serde(rename = "manual")]
    Manual,
    #[serde(rename = "auto_4_3")]
    Auto4x3,
    #[serde(rename = "auto_16_9")]
    Auto16x9,
}

/// User crop relative to Neo's already-established capture image (after the
/// existing optional client/title-bar crop). Values are pixels removed from
/// each edge. Runtime code clamps them against the actual frame so malformed
/// or stale presets can never produce an empty image.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaptureCrop {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub left: u32,
    #[serde(default)]
    pub top: u32,
    #[serde(default)]
    pub right: u32,
    #[serde(default)]
    pub bottom: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AppliedCaptureCrop {
    pub left: u32,
    pub top: u32,
    pub right: u32,
    pub bottom: u32,
    /// Exact source pixels retained before Neo's existing even-size edge pad.
    pub content_w: u32,
    pub content_h: u32,
    /// Processing size after right/bottom edge replication for odd dimensions.
    pub output_w: u32,
    pub output_h: u32,
}

impl CaptureCrop {
    pub fn applied_to(self, width: u32, height: u32) -> AppliedCaptureCrop {
        let width = width.max(1);
        let height = height.max(1);
        // Absolute compatibility rule: when user crop is OFF, this helper is
        // geometry-transparent. v576 already owns any WGC right/bottom even
        // padding, so the new feature must not introduce a second implicit
        // size/aspect adjustment merely because an odd Win32 rect is observed.
        if !self.enabled {
            return AppliedCaptureCrop {
                left: 0,
                top: 0,
                right: 0,
                bottom: 0,
                content_w: width,
                content_h: height,
                output_w: width,
                output_h: height,
            };
        }
        let left = self.left.min(width.saturating_sub(1));
        let right = self.right.min(width.saturating_sub(left).saturating_sub(1));
        let top = self.top.min(height.saturating_sub(1));
        let bottom = self
            .bottom
            .min(height.saturating_sub(top).saturating_sub(1));
        let content_w = width.saturating_sub(left + right).max(1);
        let content_h = height.saturating_sub(top + bottom).max(1);
        let output_w = if content_w & 1 == 0 {
            content_w
        } else {
            content_w + 1
        };
        let output_h = if content_h & 1 == 0 {
            content_h
        } else {
            content_h + 1
        };
        AppliedCaptureCrop {
            left,
            top,
            right,
            bottom,
            content_w,
            content_h,
            output_w,
            output_h,
        }
    }
}

pub const ASPECT_CORRECTION_SCALE_MIN: f32 = 0.50;
pub const ASPECT_CORRECTION_SCALE_MAX: f32 = 2.00;

pub fn sanitize_aspect_correction_scale(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(ASPECT_CORRECTION_SCALE_MIN, ASPECT_CORRECTION_SCALE_MAX)
    } else {
        1.0
    }
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
    /// Stable DXGI adapter LUID selected by the user. None = Auto.
    /// Neo first asks Windows/driver to place WGPU/WGL on this adapter. If an
    /// explicit request cannot move the render GPU, ONNX still uses the
    /// selected adapter and falls back to the existing cross-GPU/CPU transfer
    /// paths rather than silently running AI inference on the iGPU.
    #[serde(default)]
    pub gpu_adapter_luid: Option<u64>,
    /// Force compatible GLSL stages through Vulkan on the explicitly selected
    /// GPU even when that adapter already owns the OpenGL presentation context.
    /// False keeps the normal same-GPU OpenGL fast path; cross-GPU explicit
    /// selection still uses Vulkan automatically so GLSL stays on the selected GPU.
    #[serde(default)]
    pub gpu_force_vulkan: bool,
    /// Global ONNX backend preference. Presets intentionally remain backend
    /// agnostic; unavailable TensorRT installations resolve to DirectML.
    #[serde(default)]
    pub onnx_backend: OnnxBackendPreference,
    /// Source-window client resize before/while capturing. None = auto/no resize.
    #[serde(default)]
    pub capture_resolution: Option<CaptureResolution>,
    /// Display-only non-uniform aspect correction. Capture/WGC and all
    /// filter-processing geometry remain unchanged; only the final
    /// presentation aspect is adjusted when enabled.
    #[serde(default)]
    pub aspect_correction: bool,
    #[serde(default)]
    pub aspect_correction_mode: AspectCorrectionMode,
    #[serde(default = "default_aspect_scale")]
    pub aspect_width_scale: f32,
    #[serde(default = "default_aspect_scale")]
    pub aspect_height_scale: f32,
    /// Optional user crop applied before GLSL/ONNX processing.
    #[serde(default)]
    pub capture_crop: CaptureCrop,
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
fn default_aspect_scale() -> f32 {
    1.0
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
            win_pos: None,
            win_size: None,
            basic_win_size: None,
            mini_win_size: None,
            downscaler: "spline36".into(),
            hdr_capture: false,
            hdr_sdr_mode: HdrSdrMode::Low400,
            gpu_adapter_luid: None,
            gpu_force_vulkan: false,
            onnx_backend: OnnxBackendPreference::DirectML,
            capture_resolution: None,
            aspect_correction: false,
            aspect_correction_mode: AspectCorrectionMode::Manual,
            aspect_width_scale: 1.0,
            aspect_height_scale: 1.0,
            capture_crop: CaptureCrop::default(),
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
    use super::{OnnxBackendPreference, Settings, UiLanguage, UiLanguageMode, UiMode};

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
    fn removed_browser_launcher_setting_is_ignored_for_legacy_files() {
        let parsed: Settings =
            serde_json::from_str(r#"{"browser_kind":"firefox","smooth_pacing":true}"#).unwrap();
        assert!(parsed.smooth_pacing);
        let serialized = serde_json::to_string(&parsed).unwrap();
        assert!(!serialized.contains("browser_kind"));
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
