//! GUI-only localization. The capture/render threads never access this module.

use crate::core::config::{UiLanguage, UiLanguageMode};
use std::{
    collections::HashMap,
    path::Path,
    sync::{LazyLock, RwLock},
};

const JA: &str = include_str!("../locales/ja-JP.json");
const EN: &str = include_str!("../locales/en-US.json");
const ZH_CN: &str = include_str!("../locales/zh-CN.json");
const ZH_TW: &str = include_str!("../locales/zh-TW.json");
const KO: &str = include_str!("../locales/ko-KR.json");
const PT: &str = include_str!("../locales/pt-BR.json");
const ES: &str = include_str!("../locales/es.json");
const FR: &str = include_str!("../locales/fr-FR.json");
const DE: &str = include_str!("../locales/de-DE.json");

fn parse(source: &'static str) -> HashMap<String, String> {
    serde_json::from_str(source).expect("embedded locale catalog must be valid")
}

static CATALOGS: LazyLock<[HashMap<String, String>; 9]> = LazyLock::new(|| {
    [
        parse(JA),
        parse(EN),
        parse(ZH_CN),
        parse(ZH_TW),
        parse(KO),
        parse(PT),
        parse(ES),
        parse(FR),
        parse(DE),
    ]
});

#[derive(Clone, Debug)]
pub struct CustomLocale {
    pub tag: String,
    pub name: String,
    pub short: String,
    catalog: HashMap<String, &'static str>,
}

static ACTIVE_CUSTOM: LazyLock<RwLock<Option<HashMap<String, &'static str>>>> =
    LazyLock::new(|| RwLock::new(None));

pub fn discover_custom_locales(app_dir: &Path) -> Vec<CustomLocale> {
    let locale_dir = app_dir.join("locales");
    let builtins = [
        "ja-jp", "en-us", "zh-cn", "zh-tw", "ko-kr", "pt-br", "es", "fr-fr", "de-de",
    ];
    let mut locales = Vec::new();
    let Ok(entries) = std::fs::read_dir(locale_dir) else {
        return locales;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.eq_ignore_ascii_case("json"))
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|value| value.to_str()) else {
            continue;
        };
        if builtins.iter().any(|tag| stem.eq_ignore_ascii_case(tag)) {
            continue;
        }
        let Ok(source) = std::fs::read_to_string(&path) else {
            continue;
        };
        let Ok(mut values) = serde_json::from_str::<HashMap<String, String>>(&source) else {
            log::warn!("custom locale ignored (invalid JSON): {}", path.display());
            continue;
        };
        let tag = values
            .remove("_language_tag")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| stem.to_owned());
        let name = values
            .remove("_language_name")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| tag.clone());
        let short = values
            .remove("_language_code")
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| {
                tag.split(['-', '_'])
                    .next()
                    .unwrap_or(&tag)
                    .chars()
                    .take(3)
                    .collect::<String>()
                    .to_uppercase()
            });
        if !values.contains_key("capture.start") {
            log::warn!(
                "custom locale ignored (copy en-US.json and translate its values): {}",
                path.display()
            );
            continue;
        }
        let catalog = values
            .into_iter()
            .map(|(key, value)| {
                let value: &'static str = Box::leak(value.into_boxed_str());
                (key, value)
            })
            .collect();
        locales.push(CustomLocale {
            tag,
            name,
            short,
            catalog,
        });
    }
    locales.sort_by(|a, b| a.tag.to_lowercase().cmp(&b.tag.to_lowercase()));
    locales
}

pub fn activate_custom_locale(locale: Option<&CustomLocale>) {
    if let Ok(mut active) = ACTIVE_CUSTOM.write() {
        *active = locale.map(|locale| locale.catalog.clone());
    }
}

pub const ALL_LANGUAGES: [UiLanguage; 9] = [
    UiLanguage::JaJp,
    UiLanguage::EnUs,
    UiLanguage::ZhCn,
    UiLanguage::ZhTw,
    UiLanguage::KoKr,
    UiLanguage::PtBr,
    UiLanguage::Es,
    UiLanguage::FrFr,
    UiLanguage::DeDe,
];

fn index(lang: UiLanguage) -> usize {
    match lang {
        UiLanguage::JaJp => 0,
        UiLanguage::EnUs => 1,
        UiLanguage::ZhCn => 2,
        UiLanguage::ZhTw => 3,
        UiLanguage::KoKr => 4,
        UiLanguage::PtBr => 5,
        UiLanguage::Es => 6,
        UiLanguage::FrFr => 7,
        UiLanguage::DeDe => 8,
    }
}

pub fn text(lang: UiLanguage, key: &str) -> &str {
    if let Ok(active) = ACTIVE_CUSTOM.read()
        && let Some(value) = active.as_ref().and_then(|catalog| catalog.get(key))
    {
        return value;
    }
    builtin_text(lang, key)
}

/// Looks up only the bundled catalogs.  Status/statistics use this path so
/// they remain English even while a user-supplied UI locale is active.
pub fn builtin_text(lang: UiLanguage, key: &str) -> &str {
    CATALOGS[index(lang)]
        .get(key)
        .filter(|s| !s.is_empty())
        .or_else(|| CATALOGS[1].get(key).filter(|s| !s.is_empty()))
        .map(String::as_str)
        .unwrap_or(key)
}

pub fn format_text(lang: UiLanguage, key: &str, values: &[(&str, String)]) -> String {
    let mut result = text(lang, key).to_owned();
    for (name, value) in values {
        result = result.replace(&format!("{{{name}}}"), value);
    }
    result
}

pub fn format_builtin_text(lang: UiLanguage, key: &str, values: &[(&str, String)]) -> String {
    let mut result = builtin_text(lang, key).to_owned();
    for (name, value) in values {
        result = result.replace(&format!("{{{name}}}"), value);
    }
    result
}

fn builtin_metadata(lang: UiLanguage, key: &str, fallback: &'static str) -> &'static str {
    CATALOGS[index(lang)]
        .get(key)
        .filter(|value| !value.trim().is_empty())
        .map(String::as_str)
        .unwrap_or(fallback)
}

/// BCP 47 tag shown in the Language list and used for locale diagnostics.
/// Bundled languages read this from the matching JSON catalog so the files
/// shipped to translators are also the single source of truth for display
/// metadata. The fallback is only a corruption guard for malformed packages.
pub fn tag(lang: UiLanguage) -> &'static str {
    builtin_metadata(
        lang,
        "_language_tag",
        match lang {
            UiLanguage::JaJp => "ja-JP",
            UiLanguage::EnUs => "en-US",
            UiLanguage::ZhCn => "zh-CN",
            UiLanguage::ZhTw => "zh-TW",
            UiLanguage::KoKr => "ko-KR",
            UiLanguage::PtBr => "pt-BR",
            UiLanguage::Es => "es",
            UiLanguage::FrFr => "fr-FR",
            UiLanguage::DeDe => "de-DE",
        },
    )
}

pub fn native_name(lang: UiLanguage) -> &'static str {
    builtin_metadata(
        lang,
        "_language_name",
        match lang {
            UiLanguage::JaJp => "日本語",
            UiLanguage::EnUs => "English",
            UiLanguage::ZhCn => "简体中文",
            UiLanguage::ZhTw => "繁體中文（台灣）",
            UiLanguage::KoKr => "한국어",
            UiLanguage::PtBr => "Português (Brasil)",
            UiLanguage::Es => "Español",
            UiLanguage::FrFr => "Français",
            UiLanguage::DeDe => "Deutsch",
        },
    )
}

pub fn short_name(lang: UiLanguage) -> &'static str {
    builtin_metadata(
        lang,
        "_language_code",
        match lang {
            UiLanguage::JaJp => "JP",
            UiLanguage::EnUs => "EN",
            UiLanguage::ZhCn => "CN",
            UiLanguage::ZhTw => "TW",
            UiLanguage::KoKr => "KO",
            UiLanguage::PtBr => "PT",
            UiLanguage::Es => "ES",
            UiLanguage::FrFr => "FR",
            UiLanguage::DeDe => "DE",
        },
    )
}

pub fn fixed_mode(lang: UiLanguage) -> UiLanguageMode {
    match lang {
        UiLanguage::JaJp => UiLanguageMode::JaJp,
        UiLanguage::EnUs => UiLanguageMode::EnUs,
        UiLanguage::ZhCn => UiLanguageMode::ZhCn,
        UiLanguage::ZhTw => UiLanguageMode::ZhTw,
        UiLanguage::KoKr => UiLanguageMode::KoKr,
        UiLanguage::PtBr => UiLanguageMode::PtBr,
        UiLanguage::Es => UiLanguageMode::Es,
        UiLanguage::FrFr => UiLanguageMode::FrFr,
        UiLanguage::DeDe => UiLanguageMode::DeDe,
    }
}

pub fn mode_language(mode: UiLanguageMode) -> Option<UiLanguage> {
    match mode {
        UiLanguageMode::Auto => None,
        UiLanguageMode::JaJp => Some(UiLanguage::JaJp),
        UiLanguageMode::EnUs => Some(UiLanguage::EnUs),
        UiLanguageMode::ZhCn => Some(UiLanguage::ZhCn),
        UiLanguageMode::ZhTw => Some(UiLanguage::ZhTw),
        UiLanguageMode::KoKr => Some(UiLanguage::KoKr),
        UiLanguageMode::PtBr => Some(UiLanguage::PtBr),
        UiLanguageMode::Es => Some(UiLanguage::Es),
        UiLanguageMode::FrFr => Some(UiLanguage::FrFr),
        UiLanguageMode::DeDe => Some(UiLanguage::DeDe),
    }
}

pub fn from_bcp47(raw: &str) -> UiLanguage {
    let tag = raw.trim().replace('_', "-").to_ascii_lowercase();
    if tag == "ja" || tag.starts_with("ja-") {
        UiLanguage::JaJp
    } else if tag == "zh-tw"
        || tag.starts_with("zh-hant")
        || tag == "zh-hk"
        || tag.starts_with("zh-hk-")
        || tag == "zh-mo"
        || tag.starts_with("zh-mo-")
    {
        UiLanguage::ZhTw
    } else if tag == "zh-cn"
        || tag == "zh-sg"
        || tag.starts_with("zh-hans")
        || tag.starts_with("zh-cn-")
        || tag.starts_with("zh-sg-")
    {
        UiLanguage::ZhCn
    } else if tag == "ko" || tag.starts_with("ko-") {
        UiLanguage::KoKr
    } else if tag == "pt" || tag.starts_with("pt-") {
        UiLanguage::PtBr
    } else if tag == "es" || tag.starts_with("es-") {
        UiLanguage::Es
    } else if tag == "fr" || tag.starts_with("fr-") {
        UiLanguage::FrFr
    } else if tag == "de" || tag.starts_with("de-") {
        UiLanguage::DeDe
    } else {
        UiLanguage::EnUs
    }
}

/// Compatibility bridge while call sites migrate to stable keys.
/// Every known phrase maps to a stable catalog key; unknown phrases safely use English.
pub fn legacy_text<'a>(lang: UiLanguage, ja: &'a str, en: &'a str) -> &'a str {
    let key = match en {
        "Stop" => "capture.stop",
        "Auto" => "common.auto",
        "Built-in" => "common.builtin",
        "(No filters found)" => "filter.none",
        "Display:" => "display.label",
        "Fullscreen" => "display.fullscreen",
        "Windowed" => "display.windowed",
        "Capture size:" => "capture.size",
        "Keep visible" => "panel.keep_visible",
        "■ Stop" => "capture.stop_icon",
        "▶ Start" => "capture.start",
        "Capture:" => "capture.short",
        "Save" => "common.save",
        "Save As" => "preset.save_as",
        "New" => "common.new",
        "New Preset" => "preset.new",
        "Delete" => "common.delete",
        "🎯 Target:" => "capture.target",
        "● Running" => "capture.running",
        "Stats" => "settings.stats",
        "Frame interpolation:" => "settings.frame_interpolation",
        "Duplicate reduction" => "settings.duplicate_reduction",
        "Resize:" => "settings.resize",
        "VSync" => "settings.vsync",
        "Smooth pacing" => "settings.smooth_pacing",
        "FPS cap" => "settings.fps_cap",
        "Keep GUI on top" => "settings.gui_topmost",
        "Show control panel" => "settings.panel_show",
        "Client area only" => "settings.client_only",
        "HDR to SDR" => "settings.hdr_sdr",
        "Save log" => "settings.save_log",
        "Auto-hide cursor" => "settings.cursor_autohide",
        " sec" => "common.seconds",
        "Natural cursor speed" => "settings.cursor_speed",
        "Restart as admin" => "settings.restart_admin",
        "Start/Stop" => "shortcut.start_stop",
        "Edit" => "common.edit",
        "Shortcuts" => "shortcut.title",
        "GUI" => "shortcut.gui",
        "Panel" => "shortcut.panel",
        "Quit" => "common.quit",
        " launched" => "browser.launched",
        "more filter errors" => "filter.more_errors",
        "Filter Chain" => "filter.chain",
        "+ Add Filter" => "filter.add_icon",
        "Enable/Disable" => "filter.toggle",
        "Move down" => "common.move_down",
        "Move up" => "common.move_up",
        "Cancel" => "common.cancel",
        "Resize scale" => "resize.title",
        "Scale" => "resize.scale",
        "Apply" => "common.apply",
        "Edit Shortcut" => "shortcut.edit",
        "Add Filter" => "filter.add",
        "Enter a preset name." => "preset.enter_name",
        "Overwrite" => "common.overwrite",
        "Delete Preset" => "preset.delete",
        "Use a total of two or three keys, including modifiers." => "hotkey.error.key_count",
        "Ctrl, Alt, or Shift is required." => "hotkey.error.modifier_required",
        "Modifier-only shortcuts cannot be registered. Add a letter, number, or function key." => {
            "hotkey.error.primary_required"
        }
        "Only one non-modifier key is allowed." => "hotkey.error.one_primary",
        "The same key is duplicated." => "hotkey.error.duplicate",
        "This key is not supported for global shortcuts." => "hotkey.error.unsupported",
        "Win-key combinations are blocked because they conflict with Windows shortcuts." => {
            "hotkey.error.win_blocked"
        }
        "This combination is reserved by the app or Windows." => "hotkey.error.reserved",
        "Auto keeps the current source size. A preset/custom value resizes the source client area." => {
            "capture.size_help"
        }
        "This shortcut is already used by Windows or another application." => {
            "hotkey.error.conflict"
        }
        "The shortcut became unavailable during registration. The previous shortcut was restored." => {
            "hotkey.error.registration"
        }
        "The source window cannot keep the requested capture resolution. Select Auto or a supported size. Capture was not started." => {
            "capture.resolution_failed"
        }
        "Screenshot" => "panel.screenshot",
        "Minimize panel (hover to restore)" => "panel.minimize_help",
        "Long press and drag to reorder" => "preset.reorder_help",
        "Select the capture resolution. Auto keeps the current source size." => {
            "capture.size_select_help"
        }
        "Overwrite this preset with the current chain" => "preset.overwrite_help",
        "(Click the window to magnify)" => "capture.click_target",
        "Reuses the previous completed output for confidently identical frames while preserving presentation timing." => {
            "duplicate.summary"
        }
        "Final resize filter used to fit the chain output to the window/monitor (upscale or downscale)" => {
            "resize.final_help"
        }
        "Sync presentation to the monitor refresh rate" => "vsync.help",
        "On: preserves the source cadence with up to one frame of buffering for smooth motion. Off: low-latency mode that processes frames immediately. Separate from VSync." => {
            "smooth.help"
        }
        "Skips confidently identical source frames and re-presents the completed output using the selected smooth/low-latency timing." => {
            "duplicate.help"
        }
        "Show the floating stop/collapse panel during capture" => "panel.show_help",
        "Capture only the window contents; applies next start" => "capture.client_help",
        "Capture HDR windows as fp16 and tone-map to SDR; applies next start"
        | "Capture HDR windows as FP16 and convert them to SDR before FPS limiting, duplicate reduction, and all filters; applies next start."
        | "Capture HDR windows as FP16, apply the FPS cap first, then convert only accepted frames to SDR while preserving ordinary luminance, colour differences, and fine gradation. Duplicate reduction and all filters run after conversion; applies next start."
        | "Capture HDR windows as FP16, apply the FPS cap first, then convert only accepted frames to SDR. Uses the conservative former 400-nit curve to prevent highlight clipping while preserving RGB colour ratios. Interpolated LUT sampling and neutral dithering protect fine same-colour gradation. Duplicate reduction and all filters run after conversion. Applies next start."
        | "Converts HDR video to SDR colours and brightness. Applies next start." => "hdr.help",
        "Write cHiDeScaler-Neo.log for troubleshooting" => "log.help",
        "Make the cursor feel like normal desktop speed over the magnified view" => {
            "cursor.speed_help"
        }
        "Required when controlling windows that are running as administrator" => {
            "admin.required_help"
        }
        "Output frame-rate multiplier used by frame-interpolation filters" => {
            "interpolation.factor_help"
        }
        "HW Acceleration" => "browser.hw_title",
        "Launch with HW decode OFF" => "browser.launch_sw",
        "Launch a dedicated browser with GPU video decoding disabled. Normal browser settings are unchanged; its profile and cache stay inside this tool's cache folder." => {
            "browser.launch_help"
        }
        " was not found or could not be launched" => "browser.launch_failed",
        "Total is the measured time from the captured source frame reaching Neo to the filtered frame being presented.\nDelay frames are total ms divided by the source frame interval, rounded down." => {
            "stats.help"
        }
        "(Hold and drag to reorder)" => "filter.reorder_help",
        "Edit resize scale" => "resize.edit",
        "No filters in this chain. Use + Add Filter to add one." => "filter.empty_help",
        "Administrator permission required" => "admin.dialog_title",
        "The selected window is running with administrator permission." => "admin.target_elevated",
        "To capture and control this window, restart cHiDeScaler-Neo as administrator, then select the window again." => {
            "admin.restart_instruction"
        }
        "Restart as administrator" => "admin.restart",
        "Could not restart as administrator. Allow the Windows confirmation prompt and try again." => {
            "admin.restart_failed"
        }
        "0.25–4.00 (default 0.75)" => "resize.range",
        "Hold Ctrl, Alt, or Shift and press a letter, number, function key, or navigation key." => {
            "hotkey.edit_help"
        }
        "Two or three keys total. Modifier-only, Win-key, and reserved shortcuts are blocked." => {
            "hotkey.rules"
        }
        "A preset with this name already exists." => "preset.exists",
        "The selected preset could not be found." => "preset.not_found",
        _ => return if lang == UiLanguage::JaJp { ja } else { en },
    };
    text(lang, key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn every_catalog_has_identical_nonempty_keys_and_placeholders() {
        let english: BTreeSet<_> = CATALOGS[1].keys().cloned().collect();
        for (i, catalog) in CATALOGS.iter().enumerate() {
            assert_eq!(
                catalog.keys().cloned().collect::<BTreeSet<_>>(),
                english,
                "catalog {i}"
            );
            for (key, value) in catalog {
                assert!(!value.trim().is_empty(), "{key} is empty in catalog {i}");
                assert_eq!(
                    placeholders(&CATALOGS[1][key]),
                    placeholders(value),
                    "{key} placeholders"
                );
                assert!(
                    !value
                        .chars()
                        .any(|c| c == '\u{fffd}' || (c.is_control() && c != '\n' && c != '\t'))
                );
            }
        }
    }

    fn placeholders(s: &str) -> BTreeSet<String> {
        s.split('{')
            .skip(1)
            .filter_map(|part| part.split('}').next())
            .map(str::to_owned)
            .collect()
    }

    #[test]
    fn option_and_panel_tooltips_are_translated_in_every_bundled_catalog() {
        let keys = [
            "smooth.help",
            "interpolation.factor_help",
            "panel.gui_topmost_enable_help",
            "panel.gui_topmost_disable_help",
            "tensorrt.preparing_label",
            "tensorrt.help",
            "tensorrt.cuda_fallback_count",
            "tensorrt.directml_fallback_count",
            "tensorrt.pack_required",
            "tensorrt.unavailable_help",
        ];
        for key in keys {
            let english = &CATALOGS[1][key];
            for (index, catalog) in CATALOGS.iter().enumerate() {
                let value = catalog.get(key).expect("tooltip key");
                assert!(!value.trim().is_empty(), "{key} empty in catalog {index}");
                if index != 1 {
                    assert_ne!(value, english, "{key} stayed English in catalog {index}");
                }
            }
        }
    }

    #[test]
    fn bcp47_mapping_matches_supported_aliases() {
        for (tag, expected) in [
            ("ja-JP", UiLanguage::JaJp),
            ("en-GB", UiLanguage::EnUs),
            ("zh-CN", UiLanguage::ZhCn),
            ("zh-Hans-SG", UiLanguage::ZhCn),
            ("zh-TW", UiLanguage::ZhTw),
            ("zh-Hant", UiLanguage::ZhTw),
            ("zh-HK", UiLanguage::ZhTw),
            ("ko-KR", UiLanguage::KoKr),
            ("pt-BR", UiLanguage::PtBr),
            ("pt-PT", UiLanguage::PtBr),
            ("es-MX", UiLanguage::Es),
            ("es-ES", UiLanguage::Es),
            ("fr-CA", UiLanguage::FrFr),
            ("de-AT", UiLanguage::DeDe),
            ("de-CH", UiLanguage::DeDe),
            ("it-IT", UiLanguage::EnUs),
        ] {
            assert_eq!(from_bcp47(tag), expected, "{tag}");
        }
    }

    #[test]
    fn missing_key_falls_back_without_panicking() {
        assert_eq!(text(UiLanguage::DeDe, "missing.key"), "missing.key");
    }

    #[test]
    fn bundled_json_files_put_required_metadata_first() {
        for source in [JA, EN, ZH_CN, ZH_TW, KO, PT, ES, FR, DE] {
            let mut keys = source
                .lines()
                .map(str::trim)
                .filter(|line| line.starts_with('"'))
                .take(3);
            assert!(
                keys.next()
                    .is_some_and(|line| line.starts_with("\"_language_name\""))
            );
            assert!(
                keys.next()
                    .is_some_and(|line| line.starts_with("\"_language_code\""))
            );
            assert!(
                keys.next()
                    .is_some_and(|line| line.starts_with("\"_language_tag\""))
            );
        }
    }

    #[test]
    fn bundled_language_buttons_are_driven_by_json_metadata() {
        for lang in ALL_LANGUAGES {
            let catalog = &CATALOGS[index(lang)];
            assert_eq!(tag(lang), catalog["_language_tag"].as_str());
            assert_eq!(native_name(lang), catalog["_language_name"].as_str());
            assert_eq!(short_name(lang), catalog["_language_code"].as_str());
            assert!((2..=4).contains(&short_name(lang).chars().count()));
        }
    }

    #[test]
    fn custom_json_locale_is_discovered_from_the_portable_locales_folder() {
        let root = std::env::temp_dir().join(format!("neo-custom-locale-{}", std::process::id()));
        let locales = root.join("locales");
        std::fs::create_dir_all(&locales).unwrap();
        std::fs::write(
            locales.join("it-IT.json"),
            r#"{
                "_language_name":"Italiano",
                "_language_code":"IT",
                "_language_tag":"it-IT",
                "capture.start":"Avvia"
            }"#,
        )
        .unwrap();
        let found = discover_custom_locales(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].tag, "it-IT");
        assert_eq!(found[0].name, "Italiano");
        assert_eq!(found[0].short, "IT");
        let _ = std::fs::remove_dir_all(root);
    }
}
