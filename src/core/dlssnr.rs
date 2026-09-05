//! GUI-owned optional DLSS Neural Rendering processing and tuning persistence.
use super::config::{StageKind, StageSpec};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DlssNrOptions {
    pub preset: u32,
    pub style: u32,
    pub intensity: f32,
    pub local_tone: f32,
    pub local_structure: f32,
    /// -1 means "leave the runtime/model default untouched".
    pub skin_structure: f32,
    pub auto_mask: bool,
    pub ui_correction: bool,
}


const USER_PRESETS_FILE: &str = "dlssnr_presets.json";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DlssNrPresetFile {
    version: u32,
    presets: Vec<DlssNrOptions>,
}

pub fn factory_presets() -> [DlssNrOptions; 4] {
    let base = DlssNrOptions::default();
    [
        DlssNrOptions { preset: 0, ..base },
        DlssNrOptions { preset: 1, ..base },
        DlssNrOptions { preset: 2, ..base },
        DlssNrOptions { preset: 3, ..base },
    ]
}

pub fn load_user_presets(app_dir: &Path) -> [DlssNrOptions; 4] {
    let defaults = factory_presets();
    let Ok(mut text) = std::fs::read_to_string(app_dir.join(USER_PRESETS_FILE)) else {
        return defaults;
    };
    if text.starts_with('\u{feff}') {
        text.remove(0);
    }
    let Ok(file) = serde_json::from_str::<DlssNrPresetFile>(&text) else {
        log::warn!("dlssnr-presets-load: invalid json action=use-factory-defaults");
        return defaults;
    };
    let mut result = defaults;
    for (index, stored) in file.presets.into_iter().take(4).enumerate() {
        let mut stored = sanitize_options(stored);
        stored.preset = index as u32;
        result[index] = stored;
    }
    result
}

pub fn save_user_presets(app_dir: &Path, presets: &[DlssNrOptions; 4]) -> Result<(), String> {
    let mut clean = *presets;
    for (index, values) in clean.iter_mut().enumerate() {
        *values = sanitize_options(*values);
        values.preset = index as u32;
    }
    let file = DlssNrPresetFile {
        version: 1,
        presets: clean.to_vec(),
    };
    let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
    std::fs::write(app_dir.join(USER_PRESETS_FILE), json).map_err(|e| e.to_string())
}

impl Default for DlssNrOptions {
    fn default() -> Self {
        // Preserve v709 output by default: v709 always created Preset 1 and
        // explicitly supplied 1.0 for all four strength controls.
        Self {
            preset: 1,
            style: 0,
            intensity: 1.0,
            local_tone: 1.0,
            local_structure: 1.0,
            skin_structure: 1.0,
            auto_mask: false,
            ui_correction: false,
        }
    }
}

pub fn split(mut specs: Vec<StageSpec>) -> (Vec<StageSpec>, Option<StageSpec>) {
    let nr = specs.iter().find(|s| s.kind == StageKind::Dlssnr).cloned();
    specs.retain(|s| s.kind != StageKind::Dlssnr);
    (specs, nr)
}

/// DLSSNR is experimental and intentionally excluded from ordinary Neo presets.
/// Existing preset files may still contain a legacy DLSSNR row; callers should
/// use `split()` and ignore that row so ON/OFF always remains a manual action.
pub fn ordinary_preset_specs(chain: &[StageSpec]) -> Vec<StageSpec> {
    chain
        .iter()
        .filter(|stage| stage.kind != StageKind::Dlssnr)
        .cloned()
        .collect()
}

/// Runtime chain only: append the manually controlled DLSSNR stage without
/// making it part of ordinary preset persistence.
pub fn engine_specs(chain: &[StageSpec], nr: &Option<StageSpec>) -> Vec<StageSpec> {
    let mut specs = ordinary_preset_specs(chain);
    if let Some(nr) = nr {
        specs.push(nr.clone());
    }
    specs
}

pub fn new_spec() -> StageSpec {
    StageSpec {
        kind: StageKind::Dlssnr,
        path: "DLSS Neural Rendering (Experimental)".into(),
        enabled: false,
        params: Default::default(),
    }
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

pub fn sanitize_strength(value: f32) -> f32 {
    finite_or(value, 1.0).clamp(0.0, 1.0)
}

pub fn sanitize_skin(value: f32) -> f32 {
    finite_or(value, 1.0).clamp(-1.0, 1.0)
}

fn param_u32(spec: &StageSpec, key: &str, default: u32, max: u32) -> u32 {
    let value = spec.params.get(key).copied().unwrap_or(default as f32);
    if !value.is_finite() {
        return default;
    }
    value.round().clamp(0.0, max as f32) as u32
}

fn param_bool(spec: &StageSpec, key: &str, default: bool) -> bool {
    spec.params
        .get(key)
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| v >= 0.5)
        .unwrap_or(default)
}

pub const OPTION_KEYS: [&str; 8] = [
    "preset",
    "style",
    "intensity",
    "local_tone",
    "local_structure",
    "skin_structure",
    "auto_mask",
    "ui_correction",
];

pub fn options(spec: &StageSpec) -> DlssNrOptions {
    let defaults = DlssNrOptions::default();
    DlssNrOptions {
        preset: param_u32(spec, "preset", defaults.preset, 3),
        style: param_u32(spec, "style", defaults.style, 2),
        intensity: sanitize_strength(
            spec.params
                .get("intensity")
                .copied()
                .unwrap_or(defaults.intensity),
        ),
        local_tone: sanitize_strength(
            spec.params
                .get("local_tone")
                .copied()
                .unwrap_or(defaults.local_tone),
        ),
        local_structure: sanitize_strength(
            spec.params
                .get("local_structure")
                .copied()
                .unwrap_or(defaults.local_structure),
        ),
        skin_structure: sanitize_skin(
            spec.params
                .get("skin_structure")
                .copied()
                .unwrap_or(defaults.skin_structure),
        ),
        auto_mask: param_bool(spec, "auto_mask", defaults.auto_mask),
        ui_correction: param_bool(spec, "ui_correction", defaults.ui_correction),
    }
}

pub fn set_options(spec: &mut StageSpec, values: DlssNrOptions) {
    let values = sanitize_options(values);
    spec.params.insert("preset".into(), values.preset as f32);
    spec.params.insert("style".into(), values.style as f32);
    spec.params.insert("intensity".into(), values.intensity);
    spec.params.insert("local_tone".into(), values.local_tone);
    spec.params
        .insert("local_structure".into(), values.local_structure);
    spec.params
        .insert("skin_structure".into(), values.skin_structure);
    spec.params.insert(
        "auto_mask".into(),
        if values.auto_mask { 1.0 } else { 0.0 },
    );
    spec.params.insert(
        "ui_correction".into(),
        if values.ui_correction { 1.0 } else { 0.0 },
    );
}

pub fn sanitize_options(mut values: DlssNrOptions) -> DlssNrOptions {
    values.preset = values.preset.min(3);
    values.style = values.style.min(2);
    values.intensity = sanitize_strength(values.intensity);
    values.local_tone = sanitize_strength(values.local_tone);
    values.local_structure = sanitize_strength(values.local_structure);
    values.skin_structure = sanitize_skin(values.skin_structure);
    values
}

/// Only parameter changes may reuse the established DLSSNR stage object. The
/// stage forwards evaluation controls to the live session and performs a bounded
/// isolated worker/session restart only when the render preset (a create-time
/// model choice) changes. All other edits retain the established full-chain
/// transaction, including enable/disable and reorder.
pub fn options_only_change(old: &[StageSpec], new: &[StageSpec]) -> Option<DlssNrOptions> {
    if old.len() != new.len() {
        return None;
    }
    let mut result = None;
    for (a, b) in old.iter().zip(new) {
        if a.kind == StageKind::Dlssnr
            && a.enabled
            && b.enabled
            && a.kind == b.kind
            && a.path == b.path
        {
            if result.is_some() {
                return None;
            }
            result = Some(options(b));
        } else if a != b {
            return None;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordinary_preset_specs_never_persist_dlssnr() {
        let mut nr = new_spec();
        nr.enabled = true;
        let shader = StageSpec {
            kind: StageKind::Glsl,
            path: "x.glsl".into(),
            enabled: true,
            params: Default::default(),
        };
        let specs = ordinary_preset_specs(&[nr, shader.clone()]);
        assert_eq!(specs, vec![shader]);
    }

    #[test]
    fn engine_specs_can_append_manual_dlssnr_without_persisting_it() {
        let mut nr = new_spec();
        nr.enabled = true;
        let shader = StageSpec {
            kind: StageKind::Glsl,
            path: "x.glsl".into(),
            enabled: true,
            params: Default::default(),
        };
        let runtime = engine_specs(&[shader.clone()], &Some(nr.clone()));
        assert_eq!(runtime, vec![shader.clone(), nr]);
        assert_eq!(ordinary_preset_specs(&runtime), vec![shader]);
    }
    #[test]
    fn dlssnr_user_preset_slots_keep_their_render_preset_identity() {
        let presets = factory_presets();
        for (index, values) in presets.iter().enumerate() {
            assert_eq!(values.preset, index as u32);
        }
        let file = DlssNrPresetFile { version: 1, presets: presets.to_vec() };
        let encoded = serde_json::to_string(&file).unwrap();
        let decoded: DlssNrPresetFile = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.presets.len(), 4);
        assert_eq!(decoded.presets[3].preset, 3);
    }

    #[test]
    fn defaults_preserve_v709_output_and_advanced_ranges_are_sanitized() {
        assert_eq!(options(&new_spec()), DlssNrOptions::default());
        assert_eq!(sanitize_strength(f32::NAN), 1.0);
        assert_eq!(sanitize_strength(f32::INFINITY), 1.0);
        assert_eq!(sanitize_strength(-2.0), 0.0);
        assert_eq!(sanitize_strength(3.0), 1.0);
        assert_eq!(sanitize_skin(-5.0), -1.0);
        assert_eq!(sanitize_skin(3.0), 1.0);
    }
    #[test]
    fn parameter_fast_path_never_handles_enable_or_other_filter_changes() {
        let mut old = new_spec();
        old.enabled = true;
        let mut new = old.clone();
        let mut changed = DlssNrOptions::default();
        changed.intensity = 0.5;
        set_options(&mut new, changed);
        assert_eq!(options_only_change(&[old.clone()], &[new.clone()]), Some(changed));
        new.enabled = false;
        assert_eq!(options_only_change(&[old.clone()], &[new.clone()]), None);
        new.enabled = true;
        new.path = "different".into();
        assert_eq!(options_only_change(&[old], &[new]), None);
    }

}
