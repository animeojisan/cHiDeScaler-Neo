//! Named filter-chain presets + portable JSON persistence (atomic save).

use super::config::{Settings, StageKind, StageSpec, UiLanguage, UiLanguageMode};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(windows)]
fn replace_file(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    use windows::Win32::Storage::FileSystem::{
        MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH, MoveFileExW,
    };
    use windows::core::PCWSTR;

    let source: Vec<u16> = source.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = destination
        .as_os_str()
        .encode_wide()
        .chain(Some(0))
        .collect();
    unsafe {
        MoveFileExW(
            PCWSTR(source.as_ptr()),
            PCWSTR(destination.as_ptr()),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
        .map_err(|error| std::io::Error::other(error.to_string()))
    }
}

#[cfg(not(windows))]
fn replace_file(source: &std::path::Path, destination: &std::path::Path) -> std::io::Result<()> {
    std::fs::rename(source, destination)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    pub chain: Vec<StageSpec>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PresetFile {
    #[serde(default)]
    pub active: String,
    #[serde(default)]
    pub presets: Vec<Preset>,
}

pub struct PresetStore {
    path: PathBuf,
    pub data: PresetFile,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PresetEditError {
    EmptyName,
    NameInUse,
    ActivePresetMissing,
}

impl PresetStore {
    pub fn load(app_dir: &std::path::Path) -> Self {
        let path = app_dir.join("presets.json");
        let mut data: PresetFile = std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        if data.presets.is_empty() {
            data.presets = default_presets();
            data.active = data
                .presets
                .first()
                .map(|preset| preset.name.clone())
                .unwrap_or_default();
        }
        if data.active.is_empty() || !data.presets.iter().any(|p| p.name == data.active) {
            data.active = data.presets[0].name.clone();
        }
        let mut migrated = false;
        for preset in &mut data.presets {
            for stage in &mut preset.chain {
                if stage.path == "shaders/Deinterlace/NeoDeint_Anime_IVTC_HQ_RT.glsl" {
                    stage.path = "shaders/Deinterlace/NeoDeint.glsl".into();
                    migrated = true;
                }
            }
        }
        let store = Self { path, data };
        if migrated {
            store.save();
        }
        store
    }

    pub fn save(&self) {
        if let Ok(json) = serde_json::to_string_pretty(&self.data) {
            let tmp = self.path.with_extension("json.tmp");
            if std::fs::write(&tmp, &json).is_ok() {
                if let Err(error) = replace_file(&tmp, &self.path) {
                    log::error!(
                        "preset-save-failed: path={} error={error}",
                        self.path.display()
                    );
                    let _ = std::fs::remove_file(tmp);
                }
            }
        }
    }

    pub fn active(&self) -> Option<&Preset> {
        self.data
            .presets
            .iter()
            .find(|p| p.name == self.data.active)
    }

    pub fn unique_name(&self, base: &str) -> String {
        if !self.data.presets.iter().any(|p| p.name == base) {
            return base.to_string();
        }
        for i in 2.. {
            let n = format!("{base} ({i})");
            if !self.data.presets.iter().any(|p| p.name == n) {
                return n;
            }
        }
        unreachable!()
    }

    /// Rename and overwrite the currently selected preset.
    pub fn overwrite_active_as(
        &mut self,
        requested_name: &str,
        chain: &[StageSpec],
    ) -> Result<String, PresetEditError> {
        let name = requested_name.trim();
        if name.is_empty() {
            return Err(PresetEditError::EmptyName);
        }
        let active = self.data.active.clone();
        if self
            .data
            .presets
            .iter()
            .any(|preset| preset.name == name && preset.name != active)
        {
            return Err(PresetEditError::NameInUse);
        }
        let preset = self
            .data
            .presets
            .iter_mut()
            .find(|preset| preset.name == active)
            .ok_or(PresetEditError::ActivePresetMissing)?;
        preset.name = name.to_string();
        preset.chain = chain.to_vec();
        self.data.active = name.to_string();
        Ok(name.to_string())
    }

    /// Save the current chain as a new preset. Reusing the active name creates
    /// the familiar "(2)" copy; another existing name is rejected.
    pub fn save_as_new(
        &mut self,
        requested_name: &str,
        chain: &[StageSpec],
    ) -> Result<String, PresetEditError> {
        let requested_name = requested_name.trim();
        if requested_name.is_empty() {
            return Err(PresetEditError::EmptyName);
        }
        let name = if requested_name == self.data.active {
            self.unique_name(requested_name)
        } else if self
            .data
            .presets
            .iter()
            .any(|preset| preset.name == requested_name)
        {
            return Err(PresetEditError::NameInUse);
        } else {
            requested_name.to_string()
        };
        self.data.presets.push(Preset {
            name: name.clone(),
            chain: chain.to_vec(),
        });
        self.data.active = name.clone();
        Ok(name)
    }

    pub fn create_blank(&mut self, base_name: &str) -> String {
        let name = self.unique_name(base_name);
        self.data.presets.push(Preset {
            name: name.clone(),
            chain: Vec::new(),
        });
        self.data.active = name.clone();
        name
    }
}

fn default_presets() -> Vec<Preset> {
    serde_json::from_str::<PresetFile>(include_str!("../../presets.json"))
        .expect("bundled presets.json must be valid")
        .presets
}

pub fn load_settings(app_dir: &std::path::Path) -> Settings {
    let Some(text) = std::fs::read_to_string(app_dir.join("settings.json")).ok() else {
        return Settings::default();
    };
    let had_language_mode = serde_json::from_str::<serde_json::Value>(&text)
        .ok()
        .and_then(|v| v.get("language_mode").cloned())
        .is_some();
    let mut settings: Settings = serde_json::from_str(&text).unwrap_or_default();
    if !had_language_mode {
        settings.language_mode = match settings.language {
            UiLanguage::JaJp => UiLanguageMode::JaJp,
            UiLanguage::EnUs => UiLanguageMode::EnUs,
            UiLanguage::ZhCn => UiLanguageMode::ZhCn,
            UiLanguage::ZhTw => UiLanguageMode::ZhTw,
            UiLanguage::KoKr => UiLanguageMode::KoKr,
            UiLanguage::PtBr => UiLanguageMode::PtBr,
            UiLanguage::Es => UiLanguageMode::Es,
            UiLanguage::FrFr => UiLanguageMode::FrFr,
            UiLanguage::DeDe => UiLanguageMode::DeDe,
        };
    }
    settings.interp_factor = settings.interp_factor.clamp(2, 5);
    settings
}

pub fn save_settings(app_dir: &std::path::Path, s: &Settings) {
    if let Ok(json) = serde_json::to_string_pretty(s) {
        let _ = std::fs::write(app_dir.join("settings.json"), json);
    }
}

/// Discover drop-in filters below shaders/ and models/. Both legacy files
/// placed directly in those folders and files in arbitrarily deep user-made
/// subfolders are supported. Returns app-dir-relative forward-slash paths.
pub fn discover_filters(app_dir: &std::path::Path) -> Vec<(StageKind, String)> {
    let mut out = Vec::new();
    fn walk(
        root: &std::path::Path,
        dir: &std::path::Path,
        visited: &mut std::collections::HashSet<std::path::PathBuf>,
        out: &mut Vec<(StageKind, String)>,
    ) {
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        if !visited.insert(canonical) {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut entries: Vec<_> = entries.flatten().collect();
        entries.sort_by(|a, b| {
            filter_display_sort_key(&a.file_name().to_string_lossy())
                .cmp(&filter_display_sort_key(&b.file_name().to_string_lossy()))
        });
        for entry in entries {
            let p = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                walk(root, &p, visited, out);
                continue;
            }
            if !file_type.is_file() {
                continue;
            }
            let ext = p
                .extension()
                .map(|e| e.to_string_lossy().to_lowercase())
                .unwrap_or_default();
            let kind = match ext.as_str() {
                "glsl" => {
                    let is_external_neoflow =
                        std::fs::read_to_string(&p).ok().is_some_and(|source| {
                            (source.contains("NeoFlow external multi-pass shader source")
                                && (source.contains("NEOFLOW_PASS_COMPOSITE")
                                    || (source.contains("NEOFLOW_PASS_OWNER_CLEAR")
                                        && source.contains("NEOFLOW_PASS_FINAL"))))
                                || ((source.contains("NeoFlow GameDIS")
                                    || source.contains("NeoFlow GameMesh")
                                    || source.contains("NeoFlow HybridCadence")
                                    || source.contains("NeoFlow CausalStable"))
                                    && source.contains("NF_PASS_FINAL"))
                        });
                    // NeoFlow is frozen for this release. Keep its host and shader
                    // support in the binary/source so it can be restored later,
                    // but do not expose experimental NeoFlow files in the picker.
                    if is_external_neoflow {
                        continue;
                    }
                    StageKind::Glsl
                }
                "onnx" => StageKind::Onnx,
                _ => continue,
            };
            let Ok(rel_path) = p.strip_prefix(root) else {
                continue;
            };
            out.push((kind, rel_path.to_string_lossy().replace('\\', "/")));
        }
    }

    let mut visited = std::collections::HashSet::new();
    for folder in ["shaders", "models"] {
        walk(app_dir, &app_dir.join(folder), &mut visited, &mut out);
    }
    out
}

fn filter_display_sort_key(file_name: &str) -> (String, u8, String) {
    let stem = std::path::Path::new(file_name)
        .file_stem()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| file_name.to_string());
    let upper = stem.to_ascii_uppercase();
    const TIERS: [(&str, u8); 6] = [
        ("_S", 1),
        ("_M", 2),
        ("_L", 3),
        ("_VL", 4),
        ("_UL", 5),
        ("_UUL", 6),
    ];
    for (suffix, rank) in TIERS {
        if upper.ends_with(suffix) {
            let base_len = stem.len().saturating_sub(suffix.len());
            return (
                stem[..base_len].to_ascii_lowercase(),
                rank,
                stem.to_ascii_lowercase(),
            );
        }
    }
    (stem.to_ascii_lowercase(), 0, stem.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_filter_dir() -> std::path::PathBuf {
        let unique = format!(
            "chidescaler-filter-scan-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        std::env::temp_dir().join(unique)
    }

    #[test]
    fn discovers_legacy_and_arbitrarily_deep_user_filters() {
        let root = temp_filter_dir();
        let deep_shader = root.join("shaders/a/b/c/d/e/deep.glsl");
        let named_onnx_dir = root.join("models/onnx/custom/model.onnx");
        std::fs::create_dir_all(deep_shader.parent().unwrap()).unwrap();
        std::fs::create_dir_all(named_onnx_dir.parent().unwrap()).unwrap();
        std::fs::write(root.join("shaders/legacy.glsl"), "//!HOOK MAIN").unwrap();
        std::fs::write(&deep_shader, "//!HOOK MAIN").unwrap();
        std::fs::write(&named_onnx_dir, b"test").unwrap();
        std::fs::write(root.join("outside.glsl"), "//!HOOK MAIN").unwrap();

        let found = discover_filters(&root);
        assert!(found.contains(&(StageKind::Glsl, "shaders/legacy.glsl".into())));
        assert!(found.contains(&(StageKind::Glsl, "shaders/a/b/c/d/e/deep.glsl".into())));
        assert!(found.contains(&(StageKind::Onnx, "models/onnx/custom/model.onnx".into())));
        assert!(!found.iter().any(|(_, path)| path == "outside.glsl"));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn frozen_neoflow_is_not_exposed() {
        let root = temp_filter_dir();
        std::fs::create_dir_all(root.join("shaders")).unwrap();
        std::fs::write(
            root.join("shaders/NeoFlow_CausalStable_v0_5_EXPERIMENTAL.glsl"),
            "#version 330\n// NeoFlow CausalStable\n// NF_PASS_FINAL\n",
        )
        .unwrap();
        let found = discover_filters(&root);
        assert!(!found.iter().any(|(kind, path)| {
            *kind == StageKind::Flow
                || path.to_ascii_lowercase().contains("neoflow")
                || path.starts_with("builtin:")
        }));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn load_does_not_restore_presets_the_user_deleted() {
        let root = temp_filter_dir();
        std::fs::create_dir_all(&root).unwrap();
        let data = PresetFile {
            active: "Only Mine".into(),
            presets: vec![Preset {
                name: "Only Mine".into(),
                chain: vec![],
            }],
        };
        std::fs::write(
            root.join("presets.json"),
            serde_json::to_string_pretty(&data).unwrap(),
        )
        .unwrap();
        let loaded = PresetStore::load(&root);
        assert_eq!(loaded.data.active, "Only Mine");
        assert_eq!(loaded.data.presets.len(), 1);
        assert_eq!(loaded.data.presets[0].name, "Only Mine");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn first_launch_selects_latest_distribution_default() {
        let root = temp_filter_dir();
        std::fs::create_dir_all(&root).unwrap();
        let loaded = PresetStore::load(&root);
        assert_eq!(loaded.data.active, "Anime4K Restore S");
        assert_eq!(loaded.active().unwrap().name, "Anime4K Restore S");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn bundled_presets_include_registered_models_and_shaders() {
        let presets = default_presets();
        assert_eq!(presets.len(), 36);
        assert!(presets.iter().any(|preset| {
            preset.name == "AniSD+ [SD Anime]"
                && preset
                    .chain
                    .iter()
                    .any(|stage| stage.path == "shaders/Ani/AniSD_ArtCNN_C4F32_i4.glsl")
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "W2xEX  [SD Anime]"
                && preset
                    .chain
                    .iter()
                    .any(|stage| stage.path == "shaders/ESRGAN/W2xEX_AnimeVideo_Mini_x2.glsl")
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "AnimeJaNai HD V3.1"
                && preset.chain.iter().any(|stage| {
                    stage.path
                        == "models/AnimeJaNai/2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_fp16.onnx"
                })
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "AnimeJaNai HD V3.1 Sharp" && preset.chain.iter().any(|stage| {
                stage.path
                    == "models/AnimeJaNai/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_fp16.onnx"
            })
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "SD Anime V2NR"
                && preset.chain.iter().any(|stage| {
                    stage.path
                        == "models/AnimeJaNai/the_database_AnimeJaNaiV2L1_x2_fp16_opset14.onnx"
                })
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "umzi mahou  [HD Anime]"
                && preset
                    .chain
                    .iter()
                    .any(|stage| stage.path == "models/umzi/2x_umzi_mahou_rtmosr_fp16_op17.onnx")
        }));
        assert!(presets.iter().any(|preset| {
            preset.name == "LosOdy360  [X360/PS3]"
                && preset.chain.iter().any(|stage| {
                    stage.path == "models/LosOdy_X360/LosOdy_X360_UltraCompact_V_1-fp16.onnx"
                })
        }));
    }

    #[test]
    fn rescan_finds_filters_added_after_startup() {
        let root = temp_filter_dir();
        std::fs::create_dir_all(root.join("models/user/deeper")).unwrap();
        let before = discover_filters(&root);
        assert!(!before.iter().any(|(_, path)| path.ends_with("late.onnx")));

        std::fs::write(root.join("models/user/deeper/late.onnx"), b"test").unwrap();
        let after = discover_filters(&root);
        assert!(after.contains(&(StageKind::Onnx, "models/user/deeper/late.onnx".into())));

        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn anime4k_variants_follow_quality_order_and_stay_grouped() {
        let root = temp_filter_dir();
        let anime = root.join("shaders/Anime4K");
        std::fs::create_dir_all(&anime).unwrap();
        for name in [
            "Anime4K_Restore_CNN_VL.glsl",
            "Anime4K_Upscale_CNN_x2_L.glsl",
            "Anime4K_Restore_CNN_L.glsl",
            "Anime4K_Restore_CNN_S.glsl",
            "Anime4K_Restore_CNN_M.glsl",
        ] {
            std::fs::write(anime.join(name), "//!HOOK MAIN").unwrap();
        }

        let restore: Vec<_> = discover_filters(&root)
            .into_iter()
            .map(|(_, path)| path)
            .filter(|path| path.contains("Anime4K_Restore_CNN_"))
            .collect();
        assert_eq!(
            restore,
            [
                "shaders/Anime4K/Anime4K_Restore_CNN_S.glsl",
                "shaders/Anime4K/Anime4K_Restore_CNN_M.glsl",
                "shaders/Anime4K/Anime4K_Restore_CNN_L.glsl",
                "shaders/Anime4K/Anime4K_Restore_CNN_VL.glsl",
            ]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    fn sample_stage(path: &str) -> StageSpec {
        StageSpec {
            kind: StageKind::Glsl,
            path: path.into(),
            enabled: true,
            params: Default::default(),
        }
    }

    fn test_store() -> PresetStore {
        PresetStore {
            path: std::env::temp_dir().join("unused-presets.json"),
            data: PresetFile {
                active: "A".into(),
                presets: vec![
                    Preset {
                        name: "A".into(),
                        chain: vec![sample_stage("old.glsl")],
                    },
                    Preset {
                        name: "B".into(),
                        chain: vec![],
                    },
                ],
            },
        }
    }

    #[test]
    fn overwrite_can_rename_active_without_creating_a_copy() {
        let mut store = test_store();
        let chain = vec![sample_stage("new.glsl")];
        assert_eq!(
            store.overwrite_active_as("Renamed", &chain),
            Ok("Renamed".into())
        );
        assert_eq!(store.data.presets.len(), 2);
        assert_eq!(store.data.active, "Renamed");
        assert_eq!(store.active().unwrap().chain, chain);
    }

    #[test]
    fn overwrite_rejects_another_preset_name() {
        let mut store = test_store();
        assert_eq!(
            store.overwrite_active_as("B", &[]),
            Err(PresetEditError::NameInUse)
        );
        assert_eq!(store.data.active, "A");
    }

    #[test]
    fn save_as_current_name_creates_numbered_copy() {
        let mut store = test_store();
        assert_eq!(store.save_as_new("A", &[]), Ok("A (2)".into()));
        assert_eq!(store.data.active, "A (2)");
        assert_eq!(store.data.presets.len(), 3);
    }

    #[test]
    fn new_preset_starts_with_an_empty_chain() {
        let mut store = test_store();
        let name = store.create_blank("New Preset");
        assert_eq!(name, "New Preset");
        assert!(store.active().unwrap().chain.is_empty());
    }

    #[test]
    fn save_replaces_an_existing_presets_file() {
        let root = temp_filter_dir();
        std::fs::create_dir_all(&root).unwrap();
        let path = root.join("presets.json");
        std::fs::write(&path, "old content").unwrap();
        let store = PresetStore {
            path: path.clone(),
            data: test_store().data,
        };
        store.save();
        let saved: PresetFile =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(saved.active, "A");
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn built_in_defaults_match_the_shipped_preset_set() {
        let names: Vec<_> = default_presets()
            .into_iter()
            .map(|preset| preset.name)
            .collect();
        assert_eq!(names.len(), 34);
        assert_eq!(names.first().map(String::as_str), Some("Anime4K Restore S"));
        assert_eq!(names.get(1).map(String::as_str), Some("Anime4K Restore L"));
        assert!(names.iter().any(|name| name == "RIFE Lite S  FG"));
        assert!(names.iter().any(|name| name == "RIFE Lite   FG"));
        assert!(!names.iter().any(|name| name == "Anime4K Restore"));
        assert!(!names.iter().any(|name| name.contains("NeoFlow")));
    }
}
