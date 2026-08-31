//! cHiDeScaler-Neo GUI (eframe/egui, dark, DPI-aware).
//!
//! Layout: top = start + preset bar / target frame / center = chain editor
//! (drag & drop reorder) / bottom = display mode, fps, options, stats.
//! A small always-on-top control panel (Ctrl+Alt+P) rides on the overlay so
//! beginners can always stop magnification even in fullscreen mode.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use chidescaler_neo::core::config::{
    ASPECT_CORRECTION_SCALE_MAX, ASPECT_CORRECTION_SCALE_MIN, AspectCorrectionMode, CaptureCrop,
    CaptureResolution, OnnxBackendPreference, ScaleMode, Settings, StageKind, StageSpec,
    UiLanguage, UiLanguageMode, UiMode, app_dir, sanitize_aspect_correction_scale,
};
use chidescaler_neo::core::metrics::StageStat;
use chidescaler_neo::core::presets::{
    PresetAspectCorrection, PresetEditError, PresetStore, discover_filters, load_settings,
    save_settings,
};
use chidescaler_neo::engine::{Cmd, EngineHandle, Status, panel_target_position};
use chidescaler_neo::i18n;
use chidescaler_neo::input;
use chidescaler_neo::logging;
use chidescaler_neo::platform::gpu::{self, GpuAdapter};
use chidescaler_neo::platform::hotkeys::{
    HotkeyEvent, HotkeyThread, HotkeyValidationError, hotkey_is_available, validate_user_hotkey,
};
use chidescaler_neo::platform::win32;
use chidescaler_neo::render::onnx_backend::{
    TensorRtAvailability, detect_tensorrt_backend, tensorrt_cache_root,
};
use chidescaler_neo::render::{vulkan_gpu, vulkan_onepass};
use eframe::egui;
use std::collections::HashSet;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, StartCause, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::window::WindowId;

mod resource_monitor;
use resource_monitor::ResourceMonitor;

const HK_TOGGLE: i32 = 1;
const HK_QUIT: i32 = 2;
const HK_PANEL: i32 = 3;
const HK_GUI_TOPMOST: i32 = 4;
// HDR highlight protection is frozen pending a later review. Keep the setting,
// translations, capture path and processing code intact so the feature can be
// restored by changing this single gate, but hide the checkbox and never pass
// an enabled HDR request to the engine while the gate is false.
const HDR_CAPTURE_OPTION_ENABLED: bool = false;

fn gpu_selector_visible(adapters: &[GpuAdapter]) -> bool {
    adapters.len() >= 2
}

/// Auto keeps compute on the actual WGL device so GPU-direct sharing remains
/// available. An explicit user choice is different: it remains authoritative
/// for ONNX even if Windows/the display driver refuses to move WGL. This keeps
/// a useful dGPU inference escape hatch on hybrid systems where full-process
/// GPU selection is not honored.
fn resolve_compute_gpu_luid(requested: Option<u64>, render: Option<u64>) -> Option<u64> {
    requested.or(render)
}

fn cross_gpu_compute_active(requested: Option<u64>, render: Option<u64>) -> bool {
    matches!((requested, render), (Some(requested), Some(render)) if requested != render)
}

fn tensorrt_option_visible(availability: &TensorRtAvailability) -> bool {
    availability.available
}

/// TensorRT Crop execution is allowed only for a geometry that belongs to the
/// selected/saved preset. The enabled flag is intentionally ignored here so a
/// saved Crop can still be toggled OFF/ON while TensorRT is active without
/// opening arbitrary shape editing.
fn capture_crop_geometry_matches(a: CaptureCrop, b: CaptureCrop) -> bool {
    a.left == b.left && a.top == b.top && a.right == b.right && a.bottom == b.bottom
}

fn tensorrt_crop_switch_allowed(current: CaptureCrop, saved: CaptureCrop) -> bool {
    !current.enabled || capture_crop_geometry_matches(current, saved)
}

const BUILD_ID: &str = "20260831-v0.99.1-public";
const FULL_DEFAULT_SIZE: [f32; 2] = [900.0, 840.0];
const FULL_MIN_SIZE: [f32; 2] = [880.0, 700.0];
const BASIC_DEFAULT_SIZE: [f32; 2] = [720.0, 390.0];
const BASIC_MIN_SIZE: [f32; 2] = [720.0, 350.0];
const BASIC_MAX_RESTORED_WIDTH: f32 = 870.0;
// v348z: the latched capture button gained more physical depth.  Mini needs
// a few logical points of real viewport breathing room below that control so
// the raised face/socket never reads as glued to the window edge.  These are
// egui logical points, so the extra clearance remains DPI-aware.
const MINI_DEFAULT_SIZE: [f32; 2] = [760.0, 105.0];
const MINI_MIN_SIZE: [f32; 2] = [760.0, 101.0];
const MINI_PRESET_WIDTH: f32 = 270.0;
const MAX_FPS_CAP: u32 = 240;
// Full mode follows the same model as Basic mode: measure the localized text
// that is actually present, add the known fixed widget widths, then reserve a
// stable amount for panel margins, the vertical scrollbar and a visible right
// edge. No previous-language or live unconstrained layout value is reused.
const FULL_LAYOUT_FIXED_RESERVE: f32 = 76.0;
const MAX_TRANSLATED_BUTTON_WIDTH: f32 = 176.0;
const MAX_TRANSLATED_CHECKBOX_LABEL_WIDTH: f32 = 210.0;
const MAX_TRANSLATED_PLAIN_LABEL_WIDTH: f32 = 220.0;
const MAX_CAPTURE_BUTTON_WIDTH: f32 = 220.0;
const CAPTURE_RESOLUTION_FIELD_MIN_WIDTH: f32 = 96.0;
const CAPTURE_RESOLUTION_FIELD_MAX_WIDTH: f32 = 156.0;

fn hdr_capture_requested(settings: &Settings) -> bool {
    HDR_CAPTURE_OPTION_ENABLED && settings.hdr_capture
}

// v415 diagnostic: v414 eliminated the visual split, but a live native-caption
// drag still steals enough GPU/desktop-compositor scheduling to reduce video
// throughput. Keep every Win32 move/input event native and real-time, but do
// not submit new WGPU GUI frames while the caption is physically held. DWM can
// move the already-composed front buffer as a static image. On button release,
// forward one deferred RedrawRequested so the final GUI state is refreshed.
// This gate never touches the independent WGL/OpenGL video overlay or filter
// engine and does not rate-limit mouse coordinates or native window movement.
struct GuiCaptionDragRedrawGate<'a> {
    inner: eframe::EframeWinitApplication<'a>,
    deferred_redraws: HashSet<WindowId>,
    // The root/main viewport exists from app startup, while the floating panel
    // is only created later when capture starts. Remember the first native
    // WindowId once and gate redraws for that window only.
    root_window_id: Option<WindowId>,
    root_window_identified: bool,
    // control_panel() is generated from the root App::ui() pass, so freezing
    // every root redraw also freezes the native GDI panel's FPS text. During a
    // caption drag allow a sparse root update only for fresh stats/state, while
    // keeping the bulk of the v416 drag-time WGPU load suppression.
    last_root_drag_redraw: Option<Instant>,
}

impl<'a> GuiCaptionDragRedrawGate<'a> {
    fn new(inner: eframe::EframeWinitApplication<'a>) -> Self {
        Self {
            inner,
            deferred_redraws: HashSet::new(),
            root_window_id: None,
            root_window_identified: false,
            last_root_drag_redraw: None,
        }
    }

    fn root_drag_redraw_due(&self, now: Instant) -> bool {
        self.last_root_drag_redraw.map_or(true, |last| {
            now.duration_since(last) >= Duration::from_millis(200)
        })
    }

    fn flush_deferred_redraws(&mut self, event_loop: &ActiveEventLoop) {
        if self.deferred_redraws.is_empty() {
            return;
        }

        if input::native_gui_caption_drag_active() {
            // A child/panel RedrawRequested cannot regenerate panel contents by
            // itself: the panel is built inside the root App::ui() pass. Let one
            // deferred root redraw through every 200 ms (~5 Hz) so the FPS text
            // remains live, without returning to continuous WGPU submissions.
            let Some(root) = self.root_window_id else {
                return;
            };
            let now = Instant::now();
            if self.deferred_redraws.contains(&root) && self.root_drag_redraw_due(now) {
                self.deferred_redraws.remove(&root);
                self.last_root_drag_redraw = Some(now);
                self.inner
                    .window_event(event_loop, root, WindowEvent::RedrawRequested);
            }
            return;
        }

        self.last_root_drag_redraw = None;
        let pending = self.deferred_redraws.drain().collect::<Vec<_>>();
        for window_id in pending {
            self.inner
                .window_event(event_loop, window_id, WindowEvent::RedrawRequested);
        }
    }
}

impl ApplicationHandler<eframe::UserEvent> for GuiCaptionDragRedrawGate<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.resumed(event_loop);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if !self.root_window_identified {
            self.root_window_id = Some(window_id);
            self.root_window_identified = true;
        }

        let is_root_window = self.root_window_id == Some(window_id);
        if is_root_window
            && matches!(&event, WindowEvent::RedrawRequested)
            && input::native_gui_caption_drag_active()
        {
            let now = Instant::now();
            if self.root_drag_redraw_due(now) {
                self.deferred_redraws.remove(&window_id);
                self.last_root_drag_redraw = Some(now);
            } else {
                self.deferred_redraws.insert(window_id);
                return;
            }
        }

        let destroyed = matches!(&event, WindowEvent::Destroyed);
        self.inner.window_event(event_loop, window_id, event);
        if destroyed {
            self.deferred_redraws.remove(&window_id);
        }
        self.flush_deferred_redraws(event_loop);
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        self.inner.new_events(event_loop, cause);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: eframe::UserEvent) {
        self.inner.user_event(event_loop, event);
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        device_id: DeviceId,
        event: DeviceEvent,
    ) {
        self.inner.device_event(event_loop, device_id, event);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.flush_deferred_redraws(event_loop);
        self.inner.about_to_wait(event_loop);
    }

    fn suspended(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.suspended(event_loop);
    }

    fn exiting(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.exiting(event_loop);
    }

    fn memory_warning(&mut self, event_loop: &ActiveEventLoop) {
        self.inner.memory_warning(event_loop);
    }
}

fn stage_file_name(path: &str) -> String {
    std::path::Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(path)
        .to_string()
}

fn metric_row_occurrence(label: &str) -> Option<usize> {
    label.rsplit_once(" #").and_then(|(_, suffix)| {
        (!suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
            .then(|| suffix.parse::<usize>().ok())
            .flatten()
    })
}

fn metric_row_base_name(label: &str) -> String {
    // ONNX rows append the active provider (" [DirectML]", " [TensorRT]"),
    // while duplicate rows append " #N". Neither suffix is part of the
    // user-visible filter identity used by the chain editor.
    let without_duplicate = label
        .rsplit_once(" #")
        .filter(|(_, suffix)| !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()))
        .map(|(base, _)| base)
        .unwrap_or(label);
    without_duplicate
        .split(" [")
        .next()
        .unwrap_or(without_duplicate)
        .to_string()
}

/// Render statistics in the exact top-to-bottom order of the GUI chain.
///
/// The interpolation worker is asynchronous and its timing can arrive before
/// or after neighbouring image stages. The GUI must therefore never trust
/// timing-arrival order (or a stale metrics order after a live drag). It uses
/// the chain editor itself as the authoritative order and only appends
/// diagnostic rows that do not correspond to an enabled filter.
fn stats_rows_in_filter_chain_order(
    chain: &[StageSpec],
    rows: Vec<(String, StageStat)>,
) -> Vec<(String, StageStat)> {
    let mut remaining = rows.into_iter().map(Some).collect::<Vec<_>>();
    let mut ordered = Vec::with_capacity(remaining.len());
    let mut totals = std::collections::HashMap::<String, usize>::new();
    for stage in chain.iter().filter(|stage| stage.enabled) {
        *totals
            .entry(stage_file_name(&stage.path).to_ascii_lowercase())
            .or_default() += 1;
    }
    let mut seen = std::collections::HashMap::<String, usize>::new();

    for stage in chain.iter().filter(|stage| stage.enabled) {
        let expected = stage_file_name(&stage.path);
        let key = expected.to_ascii_lowercase();
        let occurrence = seen.entry(key.clone()).or_default();
        *occurrence += 1;
        let require_occurrence = totals.get(&key).copied().unwrap_or(1) > 1;
        let match_index = remaining.iter().position(|entry| {
            entry.as_ref().is_some_and(|(label, _)| {
                let same_filter = metric_row_base_name(label).eq_ignore_ascii_case(&expected)
                    || (stage.kind == StageKind::Flow
                        && label.eq_ignore_ascii_case("Frame interpolation"));
                same_filter
                    && (!require_occurrence || metric_row_occurrence(label) == Some(*occurrence))
            })
        });
        if let Some(index) = match_index {
            if let Some(row) = remaining[index].take() {
                ordered.push(row);
            }
        } else {
            // A chain transition / capture-resolution reset clears measured
            // stage timings. ONNX/interpolation timing often returns before an
            // asynchronous GLSL timer query, so `rows` can be non-empty while
            // one or more live GLSL stages have no fresh timing yet. Keep every
            // enabled filter visible in its authoritative chain position rather
            // than making the row disappear until the next GPU timing sample.
            let kind = match stage.kind {
                StageKind::Glsl => "glsl",
                StageKind::Onnx => "onnx",
                StageKind::Flow => "gpu",
            };
            ordered.push((
                expected,
                StageStat {
                    kind: kind.to_string(),
                    ms: -1.0,
                },
            ));
        }
    }

    // Keep non-filter diagnostic rows visible, but never allow them to disturb
    // the declared filter order above.
    ordered.extend(remaining.into_iter().flatten());
    ordered
}

// NVIDIA Optimus and AMD PowerXpress inspect these exported DWORDs before the
// first WGL context is created. The linker export directives live in build.rs.
#[unsafe(no_mangle)]
pub static NvOptimusEnablement: u32 = 1;
#[unsafe(no_mangle)]
pub static AmdPowerXpressRequestHighPerformance: u32 = 1;
const PANEL_BAR_W_PTS: f32 = 270.0;
const PANEL_BAR_H_PTS: f32 = 30.0;
// The lurk chip is fully transparent and exists only as a rediscovery / hover
// hit area.  Keep its height unchanged, but make it twice as wide so users do
// not have to hunt for a 34pt-wide invisible target when restoring the panel.
const PANEL_CHIP_W_PTS: f32 = 68.0;
const PANEL_CHIP_H_PTS: f32 = 24.0;
const PANEL_LURK_ALPHA: u8 = 0;
const STATS_FONT_SIZE: f32 = 11.5;
const MINI_LANGUAGE_POPUP_HEIGHT: f32 = 58.0;
const FULL_LANGUAGE_POPUP_HEIGHT: f32 = 286.0;
const MINI_PRESET_POPUP_HEIGHT: f32 = 80.0;
const FULL_PRESET_POPUP_HEIGHT: f32 = 324.0;
const TARGET_ROW_HEIGHT: f32 = 26.0;
// Small optical correction for the target row only. The panel remains
// geometrically unchanged; only the painted glyphs move down by one point.
const TARGET_TEXT_Y_OFFSET: f32 = 1.0;
const PRESET_ROW_HEIGHT: f32 = 24.0;
const PRESET_ROW_GAP: f32 = 5.0;
const PRESET_POPUP_BOTTOM_GUTTER: f32 = 14.0;
// Full settings is rendered in a fixed bottom panel. The old size ended at
// the resource-meter border when the chain had two or three filters; the
// one-filter case only looked correct because FULL_MIN_SIZE supplied accidental
// spare height. Reserve a real bottom gutter so the frame stroke remains fully
// visible at every DPI and independently of the statistics state.
const FULL_SETTINGS_BOTTOM_GUTTER: f32 = 36.0;
// Draw an explicit visible gap after the last Full-mode row.  The remaining
// difference from FULL_SETTINGS_BOTTOM_GUTTER is a safety reserve for frame
// strokes, DPI rounding and ScrollArea bookkeeping.
const FULL_SETTINGS_VISIBLE_BOTTOM_PADDING: f32 = 20.0;
// Fallback reserve for frame/DPI/ScrollArea bookkeeping. The Full window now
// re-evaluates its target on every actual layout-state change (stats, chain,
// language, wrapping) and is allowed to shrink as well as grow. Keep this
// small fixed reserve only so the final row is never flush with the viewport.
const FULL_SETTINGS_MEASURED_EXTRA_HEIGHT: f32 = 28.0;
// Extra height per additional wrapped toolbar line. Other settings rows use
// their measured width with a 32pt line reserve; the toolbar buttons need a
// little more vertical room.
const FULL_TOOLBAR_WRAP_EXTRA_HEIGHT: f32 = 44.0;

fn language_popup_height(mode: UiMode) -> f32 {
    match mode {
        UiMode::Mini => MINI_LANGUAGE_POPUP_HEIGHT,
        UiMode::Basic | UiMode::Full => FULL_LANGUAGE_POPUP_HEIGHT,
    }
}

fn preset_popup_height(mode: UiMode) -> f32 {
    match mode {
        UiMode::Mini => MINI_PRESET_POPUP_HEIGHT,
        UiMode::Basic | UiMode::Full => FULL_PRESET_POPUP_HEIGHT,
    }
}

fn preset_popup_id(mode: UiMode) -> &'static str {
    match mode {
        UiMode::Mini => "preset_mini",
        UiMode::Basic => "preset_basic",
        UiMode::Full => "preset_full",
    }
}

fn preset_popup_egui_id(mode: UiMode) -> egui::Id {
    egui::Id::new(("preset_popup", preset_popup_id(mode)))
}

fn basic_restored_width(width: f32) -> f32 {
    width.clamp(BASIC_MIN_SIZE[0], BASIC_MAX_RESTORED_WIDTH)
}

fn gui_fallback_repaint_ms(running: bool, _gui_dt_seconds: f32) -> u64 {
    if !running {
        return 250;
    }
    // v409: while capture is steadily running, treat the main GUI like a
    // mostly static surface. Native pointer/keyboard/window events wake egui
    // immediately, and explicit short repaint requests still drive Start/Stop
    // press animation, preparing/stopping state, panel feedback, etc. This
    // fallback is only a safety heartbeat for asynchronous state changes.
    // Keep it entirely on the GUI thread: never sleep/yield/synchronize the
    // capture/filter/present thread, and do not touch cursor/input ownership.
    // v410: 521 ms is deliberately non-harmonic with the common 60 Hz /
    // 30 fps / 24 fps cadences. A 500 ms heartbeat repeatedly lands on the
    // same presentation phase (30 refreshes at 60 Hz, 12 frames at 24 fps),
    // so even a very low repaint rate can keep colliding with the same DWM
    // composition phase. Let the GUI heartbeat drift through the video phase.
    521
}

fn full_chain_height(filter_count: usize, stats_on: bool) -> f32 {
    // Statistics on: keep up to three actual filter rows visible; longer
    // chains scroll. Statistics off: every registered filter plus one
    // insertion row. The three-row floor prevents the resource meter and
    // statistics at the bottom from being clipped after preset changes.
    let visible_rows = if stats_on {
        filter_count.clamp(2, 3)
    } else {
        filter_count + 1
    };
    58.0 + visible_rows as f32 * 56.0
}

fn full_settings_height(stats_on: bool, stats_rows: usize) -> f32 {
    // Aspect correction and user crop each have a dedicated row. Keeping both
    // separate avoids language-dependent wrapping in the established timing
    // and options rows.
    394.0
        + FULL_SETTINGS_BOTTOM_GUTTER
        + FULL_SETTINGS_MEASURED_EXTRA_HEIGHT
        + if stats_on {
            // Includes the statistics divider, summary, optional monitor
            // line, row spacing, and a compact bottom padding matching the
            // divider-to-first-line spacing.
            // Summary + monitor + stage rows, followed by exactly one spare
            // text line. Avoid reserving a large fixed blank block.
            122.0 + stats_rows as f32 * 18.0
        } else {
            0.0
        }
}

fn full_wrapped_row_extra(row_width: f32, usable_width: f32, per_extra_line: f32) -> f32 {
    let usable_width = usable_width.max(1.0);
    if row_width <= usable_width + 0.5 {
        0.0
    } else {
        let lines = (row_width / usable_width).ceil().clamp(1.0, 3.0);
        (lines - 1.0) * per_extra_line.max(0.0)
    }
}

fn full_layout_key(width: f32, monitor_height: f32, desired_height: f32) -> i32 {
    (width.round() as i32)
        .wrapping_mul(31)
        .wrapping_add(monitor_height.round() as i32)
        .wrapping_mul(31)
        .wrapping_add(desired_height.round() as i32)
}

fn flow_accent_color() -> egui::Color32 {
    egui::Color32::from_rgb(255, 180, 90)
}

fn folder_icon_geometry(center: egui::Pos2) -> (egui::Rect, egui::Rect) {
    let body = egui::Rect::from_center_size(center + egui::vec2(0.0, 1.5), egui::vec2(17.0, 11.0));
    let tab = egui::Rect::from_min_size(
        body.left_top() + egui::vec2(1.5, -3.0),
        egui::vec2(7.0, 4.5),
    );
    (body, tab)
}

fn folder_icon_button(ui: &mut egui::Ui, tooltip: &str) -> egui::Response {
    let button_size = egui::vec2(30.0, 28.0);
    let (rect, response) = ui.allocate_exact_size(button_size, egui::Sense::click());
    let visuals = ui.style().interact(&response);
    ui.painter().rect(
        rect,
        visuals.corner_radius,
        visuals.bg_fill,
        visuals.bg_stroke,
        egui::StrokeKind::Inside,
    );

    // Paint the folder from geometry instead of a font glyph/emoji. This
    // keeps the icon centered and identical for every locale and font set.
    let (body, tab) = folder_icon_geometry(rect.center());
    let folder_fill = if response.hovered() {
        egui::Color32::from_rgb(247, 198, 86)
    } else {
        egui::Color32::from_rgb(225, 177, 70)
    };
    let folder_stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(111, 79, 24));
    ui.painter().rect(
        tab,
        1.5,
        folder_fill,
        folder_stroke,
        egui::StrokeKind::Inside,
    );
    ui.painter().rect(
        body,
        2.0,
        folder_fill,
        folder_stroke,
        egui::StrokeKind::Inside,
    );
    ui.painter().line_segment(
        [
            body.left_top() + egui::vec2(1.5, 3.0),
            body.right_top() + egui::vec2(-1.5, 3.0),
        ],
        egui::Stroke::new(0.8, egui::Color32::from_rgb(255, 224, 145)),
    );

    response.on_hover_text(tooltip)
}

fn preset_fg_tag_range(name: &str) -> Option<std::ops::Range<usize>> {
    name.match_indices("FG").find_map(|(start, marker)| {
        let before_ok = name[..start]
            .chars()
            .next_back()
            .is_none_or(|c| !c.is_alphanumeric());
        let end = start + marker.len();
        let after_ok = name[end..]
            .chars()
            .next()
            .is_none_or(|c| !c.is_alphanumeric());
        (before_ok && after_ok).then_some(start..end)
    })
}

fn small_badge_format(
    ui: &egui::Ui,
    color: egui::Color32,
    valign: egui::Align,
) -> egui::TextFormat {
    let mut font_id = egui::TextStyle::Small.resolve(ui.style());
    font_id.family = egui::FontFamily::Monospace;
    egui::TextFormat {
        font_id,
        color,
        valign,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PresetNameAccent {
    Gold,
    Cyan,
    Red,
}

fn preset_markup_segments(name: &str) -> Vec<(&str, Option<PresetNameAccent>)> {
    let mut segments = Vec::new();
    let mut plain_start = 0;
    let mut cursor = 0;
    while cursor < name.len() {
        if name.as_bytes()[cursor] != b'*' {
            cursor += name[cursor..].chars().next().unwrap().len_utf8();
            continue;
        }
        let stars = name.as_bytes()[cursor..]
            .iter()
            .take_while(|byte| **byte == b'*')
            .take(3)
            .count();
        let marker = &name[cursor..cursor + stars];
        let content_start = cursor + stars;
        let Some(relative_end) = name[content_start..].find(marker) else {
            cursor += stars;
            continue;
        };
        let content_end = content_start + relative_end;
        if content_end == content_start {
            cursor += stars;
            continue;
        }
        if plain_start < cursor {
            segments.push((&name[plain_start..cursor], None));
        }
        let accent = match stars {
            3 => PresetNameAccent::Red,
            2 => PresetNameAccent::Cyan,
            _ => PresetNameAccent::Gold,
        };
        segments.push((&name[content_start..content_end], Some(accent)));
        cursor = content_end + stars;
        plain_start = cursor;
    }
    if plain_start < name.len() {
        segments.push((&name[plain_start..], None));
    }
    segments
}

fn preset_name_job(ui: &egui::Ui, name: &str) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let normal = egui::TextFormat {
        font_id: egui::TextStyle::Button.resolve(ui.style()),
        color: ui.visuals().text_color(),
        ..Default::default()
    };
    for (text, markup) in preset_markup_segments(name) {
        if let Some(markup) = markup {
            let mut format = normal.clone();
            format.color = match markup {
                PresetNameAccent::Gold => egui::Color32::from_rgb(255, 215, 92),
                PresetNameAccent::Cyan => egui::Color32::from_rgb(100, 225, 255),
                PresetNameAccent::Red => egui::Color32::from_rgb(255, 100, 100),
            };
            job.append(text, 0.0, format);
        } else if let Some(range) = preset_fg_tag_range(text) {
            job.append(&text[..range.start], 0.0, normal.clone());
            // FG is a compact badge, but it must not inherit the text
            // baseline/descent of the active locale. Baseline or bottom
            // alignment varies across Japanese, CJK and Latin fonts and made
            // the badge drift vertically between languages. Align the badge
            // to the preset line's vertical center for every locale.
            let mut accent = small_badge_format(ui, flow_accent_color(), egui::Align::Center);
            // Keep the same font family as the surrounding preset name so the
            // glyph shape stays consistent while only its size/color differ.
            accent.font_id.family = normal.font_id.family.clone();
            job.append(&text[range.clone()], 0.0, accent);
            job.append(&text[range.end..], 0.0, normal.clone());
        } else {
            job.append(text, 0.0, normal.clone());
        }
    }
    job
}

fn chain_stage_name_job(
    ui: &egui::Ui,
    kind: StageKind,
    name: &str,
    enabled: bool,
) -> egui::text::LayoutJob {
    let (badge, badge_color) = match kind {
        StageKind::Glsl => ("GLSL", egui::Color32::from_rgb(90, 170, 255)),
        StageKind::Onnx => ("ONNX", egui::Color32::from_rgb(140, 220, 120)),
        StageKind::Flow => ("FG", flow_accent_color()),
    };
    let normal = egui::TextFormat {
        font_id: egui::FontId::proportional(14.0),
        color: if enabled {
            ui.visuals().text_color()
        } else {
            ui.visuals().weak_text_color()
        },
        ..Default::default()
    };
    let mut job = egui::text::LayoutJob::default();
    let mut badge_format = small_badge_format(ui, badge_color, egui::Align::Center);
    badge_format.font_id.family = normal.font_id.family.clone();
    job.append(badge, 0.0, badge_format);
    job.append("  ", 0.0, normal.clone());
    job.append(name, 0.0, normal);
    job
}
fn install_emergency_cleanup() {
    chidescaler_neo::input::startup_recover_input_state();
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        chidescaler_neo::input::emergency_release_all();
        previous(info);
    }));
}

fn apply_panel_action_state(
    chip_lurking: &mut bool,
    bar_shown: &mut bool,
    leave_at: &mut Option<Instant>,
    collapse: bool,
    expand: bool,
) {
    if collapse {
        *chip_lurking = true;
        *bar_shown = false;
        *leave_at = None;
    }
    if expand {
        *chip_lurking = false;
        *bar_shown = true;
        *leave_at = None;
    }
}

fn cleanup_removed_browser_launcher_artifacts(app_dir: &std::path::Path) {
    // v348n: the former "HW decode OFF" browser launcher has been permanently
    // removed. Old releases could leave a complete Chromium/Firefox profile
    // tree below cache/BrowserProfiles, including DiskCache/GPUCache/Code Cache.
    // It is entirely launcher-owned and no longer has any runtime consumer, so
    // remove it instead of carrying an unbounded dead cache forever.
    let legacy = app_dir.join("cache").join("BrowserProfiles");
    if !legacy.exists() {
        return;
    }
    match std::fs::remove_dir_all(&legacy) {
        Ok(()) => {
            log::info!(
                "legacy-browser-profile-cleanup: removed obsolete launcher data path={}",
                legacy.display()
            );
            let cache_root = app_dir.join("cache");
            let is_empty = std::fs::read_dir(&cache_root)
                .ok()
                .and_then(|mut entries| entries.next().transpose().ok())
                .flatten()
                .is_none();
            if is_empty {
                let _ = std::fs::remove_dir(&cache_root);
            }
        }
        Err(error) => {
            // A browser left running from an older build may still hold files
            // open. Keep startup safe and retry automatically next launch.
            log::warn!(
                "legacy-browser-profile-cleanup: retained path={} error={error}",
                legacy.display()
            );
        }
    }
}

fn main() -> eframe::Result {
    // One-shot cursor rescue is an isolated same-EXE helper.  It must branch
    // before logging, singleton setup, GUI creation, GPU detection, or any
    // render/capture initialization.
    if std::env::args().any(|arg| arg == "--cursor-rescue") {
        chidescaler_neo::input::run_cursor_rescue_once();
        return Ok(());
    }
    // The janitor is the same portable EXE running in a tiny no-GUI mode. It
    // must bypass logging, elevation and singleton setup so it can outlive and
    // recover the main process even after a hard kill.
    if let Some(parent_pid) = std::env::args().find_map(|arg| {
        arg.strip_prefix("--cursor-janitor=")
            .and_then(|v| v.parse::<u32>().ok())
    }) {
        chidescaler_neo::input::run_cursor_janitor(parent_pid);
        return Ok(());
    }

    logging::init();
    install_emergency_cleanup();
    log::info!("cHiDeScaler-Neo build {BUILD_ID}");
    // v617 production-test breadcrumb: record the opt-in as soon as the real
    // GUI process starts, before elevation, settings load, singleton handoff,
    // WGPU/WGL creation, or any Vulkan initialization.  This makes a second-
    // instance handoff distinguishable from a render-routing failure.
    if vulkan_onepass::production_one_pass_requested() {
        let sink_present = std::env::var("NEO_VULKAN_PROBE_RESULT")
            .map(|value| !value.trim().is_empty())
            .unwrap_or(false);
        let line = format!(
            "vulkan-production-glsl: phase=process-entry build={} env=enabled result_sink={} pid={} singleton=not-checked vulkan_init=false",
            BUILD_ID,
            sink_present,
            std::process::id(),
        );
        log::info!("{line}");
        vulkan_gpu::record_probe_result(&line);
    }
    // Keep the native egui event loop responsive when the render worker is
    // saturating the same low-end GPU. This is thread-local only: capture,
    // provider and process priorities are unchanged.
    win32::promote_current_thread_for_gui();
    // user opted into elevated mode: relaunch with UAC once
    let saved = {
        let dir = app_dir();
        cleanup_removed_browser_launcher_artifacts(&dir);
        let mut st = load_settings(&dir);
        if let Ok(mode) = std::env::var("NEO_GUI_SCREENSHOT_MODE") {
            st.ui_mode = match mode.to_ascii_lowercase().as_str() {
                "mini" => UiMode::Mini,
                "basic" | "basic_stats" => UiMode::Basic,
                "full" | "full_stats" => UiMode::Full,
                _ => st.ui_mode,
            };
        }
        if st.run_as_admin
            && !win32::own_process_elevated()
            && std::env::args().all(|a| a != "--no-elevate")
            && win32::relaunch_as_admin()
        {
            return Ok(());
        }
        let size = match st.ui_mode {
            UiMode::Mini => st
                .mini_win_size
                .map(|(width, _)| (width, MINI_DEFAULT_SIZE[1])),
            UiMode::Basic => st
                .basic_win_size
                .map(|(width, _)| (basic_restored_width(width), BASIC_DEFAULT_SIZE[1])),
            UiMode::Full => st.win_size,
        };
        (st.win_pos, size, st.ui_mode)
    };
    // single instance: two running copies would each install a WH_MOUSE_LL hook
    // and fight over ClipCursor/SetCursorPos, producing exactly the erratic
    // cursor conflicts (including "hotkey already in use"). If one
    // is already running, focus it and exit.
    let gui_screenshot_test = std::env::var_os("NEO_GUI_SCREENSHOT").is_some();
    if !win32::acquire_single_instance() && !gui_screenshot_test {
        log::warn!("another cHiDeScaler-Neo instance is already running; focusing it and exiting");
        if vulkan_onepass::production_one_pass_requested() {
            let line = format!(
                "vulkan-production-glsl: result=blocked reason=single-instance-already-running build={} pid={} action=close-existing-Neo-and-rerun vulkan_init=false",
                BUILD_ID,
                std::process::id(),
            );
            log::warn!("{line}");
            vulkan_gpu::record_probe_result(&line);
        }
        win32::activate_other_instance("cHiDeScaler-Neo");
        return Ok(());
    } else if gui_screenshot_test {
        log::info!("gui screenshot test: singleton focus handoff bypassed");
    }
    if !gui_screenshot_test {
        chidescaler_neo::input::spawn_cursor_janitor();
    }
    // Portable hybrid-GPU default: vendor driver export hints are embedded in
    // this EXE (NvOptimusEnablement / AmdPowerXpressRequestHighPerformance).
    // Normal startup never persists a Windows GPU preference. v643 keeps
    // WGPU/WGL as presentation devices and switches compute backends directly,
    // so a manual GPU change while capture is stopped requires no Neo restart.
    log::info!("gpu-preference: portable vendor hints enabled; persistent registry override=false");
    // Public releases keep the native title stable and user-facing.
    // Detailed build identification remains available in the diagnostic log.
    let (default_size, min_size) = match saved.2 {
        UiMode::Mini => (MINI_DEFAULT_SIZE, MINI_MIN_SIZE),
        UiMode::Basic => (BASIC_DEFAULT_SIZE, BASIC_MIN_SIZE),
        UiMode::Full => (FULL_DEFAULT_SIZE, FULL_MIN_SIZE),
    };
    let mut viewport = egui::ViewportBuilder::default()
        .with_inner_size(default_size)
        .with_min_inner_size(min_size)
        // Keep ordinary Windows Minimize and Close behavior. Maximize stays
        // disabled because Neo's mode-specific layout owns the window size.
        .with_close_button(true)
        .with_minimize_button(true)
        .with_maximize_button(false)
        .with_maximized(false)
        .with_title("cHiDeScaler-Neo");
    // restore the remembered window placement (sanity-checked)
    if let Some((w, h)) = saved.1 {
        if (min_size[0]..=4000.0).contains(&w) && (min_size[1]..=4000.0).contains(&h) {
            viewport = viewport.with_inner_size([w.max(min_size[0]), h.max(min_size[1])]);
        }
    }
    if let Some((x, y)) = saved.0 {
        if (-200.0..=8000.0).contains(&x) && (-50.0..=8000.0).contains(&y) {
            viewport = viewport.with_position([x, y]);
        }
    }
    if let Some(icon) = load_icon() {
        viewport = viewport.with_icon(std::sync::Arc::new(icon));
    }
    let mut options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };
    // Keep the established non-vsync eframe/WGPU GUI compositor path.
    // ONNX x3 now uses a strict integer interpolation timeline independently
    // of GUI/DWM presentation policy.
    options.renderer = eframe::Renderer::Wgpu;
    options.wgpu_options =
        eframe::WgpuConfiguration::default().with_surface_config(eframe::SurfaceConfig {
            present_mode: eframe::wgpu::PresentMode::AutoNoVsync,
            desired_maximum_frame_latency: Some(1),
        });
    // Own eframe's already-supported winit event loop only to gate WGPU
    // RedrawRequested while a native caption drag is active. The OS window
    // continues moving at full native rate; only fresh GUI GPU submissions are
    // deferred until release.
    let event_loop =
        winit::event_loop::EventLoop::<eframe::UserEvent>::with_user_event().build()?;
    let eframe_app = eframe::create_native(
        "cHiDeScaler-Neo",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
        &event_loop,
    );
    let mut app = GuiCaptionDragRedrawGate::new(eframe_app);
    event_loop.run_app(&mut app)?;

    Ok(())
}

fn load_icon() -> Option<egui::IconData> {
    let bytes: &[u8] = include_bytes!("../assets/icon_256.png");
    let decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    let mut reader = decoder.read_info().ok()?;
    let mut buf = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buf).ok()?;
    if info.color_type != png::ColorType::Rgba {
        return None;
    }
    buf.truncate(info.buffer_size());
    Some(egui::IconData {
        rgba: buf,
        width: info.width,
        height: info.height,
    })
}

fn locale_font_family(lang: UiLanguage) -> egui::FontFamily {
    let name = match lang {
        UiLanguage::JaJp => "ui_ja",
        UiLanguage::ZhCn => "ui_zh_cn",
        UiLanguage::ZhTw => "ui_zh_tw",
        UiLanguage::KoKr => "ui_ko",
        _ => "ui_latin",
    };
    egui::FontFamily::Name(name.into())
}

fn mixed_text_font_family(text: &str, ui_language: UiLanguage) -> egui::FontFamily {
    let has_japanese = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x3040..=0x30ff | 0x31f0..=0x31ff));
    let has_hangul = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x1100..=0x11ff | 0x3130..=0x318f | 0xac00..=0xd7af));
    let has_cjk = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x3400..=0x4dbf | 0x4e00..=0x9fff));
    let has_indic = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x0900..=0x0dff | 0x0f00..=0x0fff));
    let has_thai_lao = text.chars().any(|ch| matches!(ch as u32, 0x0e00..=0x0eff));
    let has_myanmar = text
        .chars()
        .any(|ch| matches!(ch as u32, 0x1000..=0x109f | 0xaa60..=0xaa7f));

    if has_indic {
        egui::FontFamily::Name("ui_indic".into())
    } else if has_thai_lao {
        egui::FontFamily::Name("ui_thai".into())
    } else if has_myanmar {
        egui::FontFamily::Name("ui_myanmar".into())
    } else if has_japanese {
        locale_font_family(UiLanguage::JaJp)
    } else if has_hangul {
        locale_font_family(UiLanguage::KoKr)
    } else if has_cjk {
        match ui_language {
            UiLanguage::ZhCn => locale_font_family(UiLanguage::ZhCn),
            UiLanguage::ZhTw => locale_font_family(UiLanguage::ZhTw),
            // A Japanese window title shown in an English or other GUI often
            // contains only kanji and Latin characters. Yu Gothic contains
            // both, keeping the whole title on one physical font baseline.
            _ => locale_font_family(UiLanguage::JaJp),
        }
    } else {
        locale_font_family(ui_language)
    }
}

fn install_ui_fonts(ctx: &egui::Context) {
    const FONTS: &[(&str, &str, f32)] = &[
        ("ui_segoe", "C:/Windows/Fonts/segoeui.ttf", 0.0),
        ("ui_ja_font", "C:/Windows/Fonts/YuGothM.ttc", 0.08),
        ("ui_zh_cn_font", "C:/Windows/Fonts/msyh.ttc", 0.025),
        ("ui_zh_tw_font", "C:/Windows/Fonts/msjh.ttc", 0.025),
        ("ui_ko_font", "C:/Windows/Fonts/malgun.ttf", 0.025),
        ("ui_indic_font", "C:/Windows/Fonts/Nirmala.ttc", 0.025),
        ("ui_thai_font", "C:/Windows/Fonts/LeelawUI.ttf", 0.025),
        ("ui_myanmar_font", "C:/Windows/Fonts/mmrtext.ttf", 0.025),
    ];
    let mut fonts = egui::FontDefinitions::default();
    let built_in_fallbacks = fonts
        .families
        .get(&egui::FontFamily::Proportional)
        .cloned()
        .unwrap_or_default();
    let mut loaded = std::collections::HashSet::new();
    for &(name, p, y_offset_factor) in FONTS {
        if let Ok(bytes) = std::fs::read(p) {
            let data = egui::FontData {
                tweak: egui::FontTweak {
                    y_offset_factor,
                    ..Default::default()
                },
                ..egui::FontData::from_owned(bytes)
            };
            fonts.font_data.insert(name.into(), data.into());
            loaded.insert(name);
            log::info!("ui-font: registered-once path={p}");
        }
    }

    let ordered_family = |native: &str| {
        // Keep Latin letters, numbers and punctuation in the active native
        // font whenever that font contains them. Mixing Segoe UI for Latin
        // with Yu Gothic/MS Gothic-family glyphs in one row gives each run a
        // different baseline. Native-first keeps mixed labels such as
        // "GUIを最前面に表示" on one visual baseline; the remaining fonts
        // are still available strictly as missing-glyph fallbacks.
        [
            native,
            "ui_segoe",
            "ui_ja_font",
            "ui_zh_cn_font",
            "ui_zh_tw_font",
            "ui_ko_font",
            "ui_indic_font",
            "ui_thai_font",
            "ui_myanmar_font",
        ]
        .into_iter()
        .filter(|name| loaded.contains(name))
        .fold(Vec::<String>::new(), |mut list, name| {
            if !list.iter().any(|existing| existing == name) {
                list.push(name.to_owned());
            }
            list
        })
        .into_iter()
        .chain(built_in_fallbacks.iter().cloned())
        .fold(Vec::<String>::new(), |mut list, name| {
            if !list.iter().any(|existing| existing == &name) {
                list.push(name);
            }
            list
        })
    };
    for (family, native) in [
        ("ui_latin", "ui_segoe"),
        ("ui_ja", "ui_ja_font"),
        ("ui_zh_cn", "ui_zh_cn_font"),
        ("ui_zh_tw", "ui_zh_tw_font"),
        ("ui_ko", "ui_ko_font"),
        ("ui_indic", "ui_indic_font"),
        ("ui_thai", "ui_thai_font"),
        ("ui_myanmar", "ui_myanmar_font"),
    ] {
        fonts.families.insert(
            egui::FontFamily::Name(family.into()),
            ordered_family(native),
        );
    }
    if let Some(list) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
        for name in ordered_family("ui_segoe").into_iter().rev() {
            list.insert(0, name);
        }
    }
    ctx.set_fonts(fonts);
}

fn select_ui_font_for_locale(ctx: &egui::Context, lang: UiLanguage) {
    let family = locale_font_family(lang);
    ctx.all_styles_mut(|style| {
        for font_id in style.text_styles.values_mut() {
            if font_id.family != egui::FontFamily::Monospace {
                font_id.family = family.clone();
            }
        }
    });
    log::info!("ui-font: selected cached locale={}", i18n::tag(lang));
}

fn ellipsize_to_width(ui: &egui::Ui, text: &str, font_id: &egui::FontId, max_width: f32) -> String {
    if ui
        .painter()
        .layout_no_wrap(text.to_owned(), font_id.clone(), egui::Color32::WHITE)
        .size()
        .x
        <= max_width
    {
        return text.to_owned();
    }
    let mut chars: Vec<char> = text.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate = format!("{}…", chars.iter().collect::<String>());
        if ui
            .painter()
            .layout_no_wrap(candidate.clone(), font_id.clone(), egui::Color32::WHITE)
            .size()
            .x
            <= max_width
        {
            return candidate;
        }
    }
    "…".to_owned()
}

fn measured_text_width(ui: &egui::Ui, text: &str, font_id: &egui::FontId) -> f32 {
    ui.painter()
        .layout_no_wrap(text.to_owned(), font_id.clone(), egui::Color32::WHITE)
        .size()
        .x
}

fn bounded_visible_text(
    ui: &egui::Ui,
    full_text: &str,
    font_id: &egui::FontId,
    max_width: f32,
) -> (String, bool) {
    if measured_text_width(ui, full_text, font_id) <= max_width {
        (full_text.to_owned(), false)
    } else {
        (ellipsize_to_width(ui, full_text, font_id, max_width), true)
    }
}

fn bounded_button_width(ui: &egui::Ui, text: &str) -> f32 {
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    (measured_text_width(ui, text, &font_id)
        .min(MAX_TRANSLATED_BUTTON_WIDTH - 2.0 * ui.spacing().button_padding.x)
        + 2.0 * ui.spacing().button_padding.x)
        .max(ui.spacing().interact_size.x)
}

fn bounded_checkbox_width(ui: &egui::Ui, text: &str) -> f32 {
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    ui.spacing().icon_width
        + ui.spacing().icon_spacing
        + measured_text_width(ui, text, &font_id).min(MAX_TRANSLATED_CHECKBOX_LABEL_WIDTH)
}

fn bounded_plain_label_width(ui: &egui::Ui, text: &str) -> f32 {
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    measured_text_width(ui, text, &font_id).min(MAX_TRANSLATED_PLAIN_LABEL_WIDTH)
}

fn capture_resolution_field_width(ui: &egui::Ui, lang: UiLanguage) -> f32 {
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    (capture_resolution_presets()
        .into_iter()
        .map(|preset| measured_text_width(ui, &capture_resolution_label(preset, lang), &font_id))
        .fold(0.0_f32, f32::max)
        + 22.0)
        .clamp(
            CAPTURE_RESOLUTION_FIELD_MIN_WIDTH,
            CAPTURE_RESOLUTION_FIELD_MAX_WIDTH,
        )
}

fn bounded_plain_label(
    ui: &mut egui::Ui,
    text: impl Into<String>,
    max_width: f32,
) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Body.resolve(ui.style());
    let (visible_text, truncated) = bounded_visible_text(ui, &full_text, &font_id, max_width);
    let response = ui.label(visible_text);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn language_choice_row(
    ui: &mut egui::Ui,
    selected: bool,
    left: &str,
    left_language: UiLanguage,
    right: &str,
    right_language: UiLanguage,
) -> egui::Response {
    let width = ui.available_width().max(140.0);
    let (rect, response) = ui.allocate_exact_size(egui::vec2(width, 24.0), egui::Sense::click());
    if selected || response.hovered() {
        let fill = if selected {
            egui::Color32::from_rgb(55, 88, 122)
        } else {
            egui::Color32::from_rgb(43, 47, 53)
        };
        ui.painter().rect_filled(rect, 4.0, fill);
    }

    let color = egui::Color32::from_rgb(235, 238, 242);
    let left_font = egui::FontId::new(12.5, locale_font_family(left_language));
    let right_font = egui::FontId::new(10.5, locale_font_family(right_language));
    let right_width = if right.is_empty() {
        0.0
    } else {
        ui.painter()
            .layout_no_wrap(right.to_owned(), right_font.clone(), color)
            .size()
            .x
            + 18.0
    };
    let left_text = ellipsize_to_width(ui, left, &left_font, rect.width() - 14.0 - right_width);
    let left_galley = ui
        .painter()
        .layout_no_wrap(left_text.clone(), left_font, color);
    let left_ink = left_galley.mesh_bounds;
    let left_target = rect.left_center() + egui::vec2(7.0, 0.0);
    ui.painter().galley(
        left_target - egui::vec2(left_ink.left(), left_ink.center().y),
        left_galley,
        color,
    );

    if !right.is_empty() {
        // Spaces have no painted mesh, so use their advance width rather
        // than mesh_bounds. The code is Latin-only and uses the stable Segoe
        // baseline independently from the native language name.
        let gap_width = ui
            .painter()
            .layout_no_wrap(
                "    ".to_owned(),
                egui::FontId::new(10.5, locale_font_family(UiLanguage::EnUs)),
                color,
            )
            .size()
            .x;
        let right_galley = ui
            .painter()
            .layout_no_wrap(right.to_owned(), right_font, color);
        let right_ink = right_galley.mesh_bounds;
        let right_target = egui::pos2(
            left_target.x + left_ink.width() + gap_width,
            rect.center().y,
        );
        ui.painter().galley(
            right_target - egui::vec2(right_ink.left(), right_ink.center().y),
            right_galley,
            color,
        );
    }
    if left_text != left {
        response.on_hover_text(left)
    } else {
        response
    }
}

fn visually_centered_label(
    ui: &mut egui::Ui,
    text: &str,
    font_id: egui::FontId,
    color: egui::Color32,
    strong: bool,
    row_height: f32,
) -> egui::Response {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        text,
        0.0,
        egui::TextFormat {
            font_id,
            color,
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
    let desired_width = galley.size().x.min(ui.available_width()).max(1.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(desired_width, row_height), egui::Sense::hover());
    let ink = galley.mesh_bounds;
    let pos = centered_ink_position(rect, ink);
    if strong {
        // A one-pixel second pass preserves the existing strong-title emphasis
        // while keeping mixed-script glyphs centered by their painted bounds.
        ui.painter()
            .galley(pos + egui::vec2(0.55, 0.0), galley.clone(), color);
    }
    ui.painter().galley(pos, galley, color);
    response
}

fn centered_ink_position(rect: egui::Rect, ink_bounds: egui::Rect) -> egui::Pos2 {
    rect.center() - ink_bounds.center().to_vec2()
}

fn target_centered_label_with_offset(
    ui: &mut egui::Ui,
    text: &str,
    font_id: egui::FontId,
    color: egui::Color32,
    row_height: f32,
    y_offset: f32,
) -> egui::Response {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        text,
        0.0,
        egui::TextFormat {
            font_id,
            color,
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
    let desired_width = galley.size().x.min(ui.available_width()).max(1.0);
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(desired_width, row_height), egui::Sense::hover());
    let ink = galley.mesh_bounds;
    let pos = centered_ink_position(rect, ink) + egui::vec2(0.0, y_offset);
    ui.painter().galley(pos, galley, color);
    response
}

fn target_centered_label(
    ui: &mut egui::Ui,
    text: &str,
    font_id: egui::FontId,
    color: egui::Color32,
    row_height: f32,
) -> egui::Response {
    target_centered_label_with_offset(ui, text, font_id, color, row_height, TARGET_TEXT_Y_OFFSET)
}

fn target_icon_and_label(
    ui: &mut egui::Ui,
    localized_label: &str,
    family: egui::FontFamily,
    color: egui::Color32,
) -> egui::Response {
    // The emoji and translated text must not share one mesh-bounds box.
    // Different scripts change the combined ink bounds and therefore made the
    // target icon drift vertically between languages. Keep one combined row
    // allocation (so horizontal spacing stays identical), but center the icon
    // and translated text independently by their own painted bounds.
    let label = localized_label
        .strip_prefix('🎯')
        .map(|label| label.trim_start())
        .unwrap_or(localized_label);
    let font_id = egui::FontId::new(13.0, family);
    let icon_galley = ui
        .painter()
        .layout_no_wrap("🎯".to_owned(), font_id.clone(), color);
    let gap_galley = ui
        .painter()
        .layout_no_wrap(" ".to_owned(), font_id.clone(), color);
    let label_galley = ui
        .painter()
        .layout_no_wrap(label.to_owned(), font_id, color);
    let icon_advance = icon_galley.size().x;
    let gap = gap_galley.size().x;
    let desired_width = (icon_advance + gap + label_galley.size().x)
        .min(ui.available_width())
        .max(1.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(desired_width, TARGET_ROW_HEIGHT),
        egui::Sense::hover(),
    );

    let icon_ink = icon_galley.mesh_bounds;
    let icon_pos = egui::pos2(
        rect.left() - icon_ink.left(),
        rect.center().y - icon_ink.center().y,
    );
    ui.painter().galley(icon_pos, icon_galley, color);

    let label_ink = label_galley.mesh_bounds;
    let label_pos = egui::pos2(
        rect.left() + icon_advance + gap - label_ink.left(),
        rect.center().y - label_ink.center().y + TARGET_TEXT_Y_OFFSET,
    );
    ui.painter().galley(label_pos, label_galley, color);
    response
}

fn control_text_job(ui: &egui::Ui, text: impl Into<String>) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    job.append(
        &text.into(),
        0.0,
        egui::TextFormat {
            font_id: egui::TextStyle::Button.resolve(ui.style()),
            color: ui.visuals().text_color(),
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    job
}

fn paint_control_text_centered(
    ui: &egui::Ui,
    response: &egui::Response,
    job: egui::text::LayoutJob,
    horizontal_center: bool,
) {
    let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
    let ink = galley.mesh_bounds;
    let x = if horizontal_center {
        response.rect.center().x - ink.center().x
    } else {
        response.rect.left() + ui.spacing().button_padding.x - ink.left()
    };
    let pos = egui::pos2(x, response.rect.center().y - ink.center().y);
    ui.painter()
        .with_clip_rect(response.rect.shrink2(egui::vec2(4.0, 0.0)))
        .galley(pos, galley, ui.visuals().text_color());
}

fn paint_dropdown_arrow(ui: &egui::Ui, response: &egui::Response) {
    ui.painter().text(
        egui::pos2(
            response.rect.right() - ui.spacing().button_padding.x - 5.0,
            response.rect.center().y,
        ),
        egui::Align2::CENTER_CENTER,
        "▼",
        egui::FontId::new(10.0, egui::FontFamily::Proportional),
        ui.visuals().weak_text_color(),
    );
}

fn transparent_control_job(mut job: egui::text::LayoutJob) -> egui::text::LayoutJob {
    for section in &mut job.sections {
        section.format.color = egui::Color32::TRANSPARENT;
    }
    job
}

fn checkbox_label_reference(text: &str) -> &'static str {
    let uses_cjk_or_hangul = text.chars().any(|c| {
        matches!(
            c,
            '\u{3040}'..='\u{30ff}'
                | '\u{3400}'..='\u{9fff}'
                | '\u{ac00}'..='\u{d7af}'
        )
    });
    if uses_cjk_or_hangul {
        // One shared optical reference keeps short labels such as 「統計」 on
        // the same text line as the longer Japanese/Chinese/Korean options.
        "統計漢字かな한글"
    } else {
        // Latin-script labels share a stable cap/x-height reference instead of
        // being vertically re-centered from each word's individual ink bounds.
        "Ag0123"
    }
}

fn checkbox_label_reference_center_y(ui: &egui::Ui, text: &str) -> f32 {
    let reference = control_text_job(ui, checkbox_label_reference(text));
    ui.fonts_mut(|fonts| fonts.layout_job(reference))
        .mesh_bounds
        .center()
        .y
}

/// Preserve egui's native checkbox geometry, interaction, disabled styling,
/// focus handling and wrapping width. Only the visible label is repainted.
/// Labels using the same script share one optical baseline, so neighboring
/// controls stay aligned while the checkbox icon and hit rectangle remain stock.
fn paint_ink_centered_checkbox_label(
    ui: &egui::Ui,
    response: &egui::Response,
    label: &str,
    mut job: egui::text::LayoutJob,
) {
    let color = ui.style().interact(response).text_color();
    for section in &mut job.sections {
        section.format.color = color;
    }
    job.wrap.max_width =
        (response.rect.width() - ui.spacing().icon_width - ui.spacing().icon_spacing).max(1.0);
    let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
    let shared_center_y = checkbox_label_reference_center_y(ui, label);
    let text_x = response.rect.min.x + ui.spacing().icon_width + ui.spacing().icon_spacing;
    let text_pos = egui::pos2(text_x, response.rect.center().y - shared_center_y);
    ui.painter()
        .with_clip_rect(response.rect)
        .galley(text_pos, galley, color);
}

fn ink_centered_checkbox(
    ui: &mut egui::Ui,
    checked: &mut bool,
    text: impl Into<String>,
) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let (text, truncated) = bounded_visible_text(
        ui,
        &full_text,
        &font_id,
        MAX_TRANSLATED_CHECKBOX_LABEL_WIDTH,
    );
    let job = control_text_job(ui, text.clone());
    let response = ui.add(egui::Checkbox::new(
        checked,
        transparent_control_job(job.clone()),
    ));
    paint_ink_centered_checkbox_label(ui, &response, &text, job);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn ink_centered_checkbox_enabled(
    ui: &mut egui::Ui,
    enabled: bool,
    checked: &mut bool,
    text: impl Into<String>,
) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let (text, truncated) = bounded_visible_text(
        ui,
        &full_text,
        &font_id,
        MAX_TRANSLATED_CHECKBOX_LABEL_WIDTH,
    );
    let job = control_text_job(ui, text.clone());
    let response = ui.add_enabled(
        enabled,
        egui::Checkbox::new(checked, transparent_control_job(job.clone())),
    );
    paint_ink_centered_checkbox_label(ui, &response, &text, job);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn control_row_button(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let max_text_width =
        (MAX_TRANSLATED_BUTTON_WIDTH - 2.0 * ui.spacing().button_padding.x).max(1.0);
    let (text, truncated) = bounded_visible_text(ui, &full_text, &font_id, max_text_width);
    let text_width = ui.fonts_mut(|fonts| {
        fonts
            .layout_no_wrap(text.clone(), font_id, ui.visuals().text_color())
            .size()
            .x
    });
    let width =
        (text_width + 2.0 * ui.spacing().button_padding.x).max(ui.spacing().interact_size.x);
    let job = control_text_job(ui, text);
    let response = ui.add_sized(
        [width, ui.spacing().interact_size.y + 2.0],
        egui::Button::new(transparent_control_job(job.clone())),
    );
    paint_control_text_centered(ui, &response, job, true);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn ink_centered_selectable_label(
    ui: &mut egui::Ui,
    selected: bool,
    text: impl Into<String>,
) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let max_text_width =
        (MAX_TRANSLATED_BUTTON_WIDTH - 2.0 * ui.spacing().button_padding.x).max(1.0);
    let (text, truncated) = bounded_visible_text(ui, &full_text, &font_id, max_text_width);
    let job = control_text_job(ui, text);
    let response = ui.selectable_label(selected, transparent_control_job(job.clone()));
    paint_control_text_centered(ui, &response, job, true);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn ink_centered_control_label(ui: &mut egui::Ui, text: impl Into<String>) -> egui::Response {
    let full_text = text.into();
    let font_id = egui::TextStyle::Button.resolve(ui.style());
    let (text, truncated) =
        bounded_visible_text(ui, &full_text, &font_id, MAX_TRANSLATED_PLAIN_LABEL_WIDTH);
    let job = control_text_job(ui, text);
    let ink_width = ui.fonts_mut(|fonts| fonts.layout_job(job.clone()).mesh_bounds.width());
    // paint_control_text_centered clips four points on each side. Reserve the
    // same amount here so the first/last glyph can never touch that clip.
    let width = ink_width + 8.0;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(width, ui.spacing().interact_size.y + 2.0),
        egui::Sense::hover(),
    );
    let response = response.with_new_rect(rect);
    paint_control_text_centered(ui, &response, job, true);
    if truncated {
        response.on_hover_text(full_text)
    } else {
        response
    }
}

fn truncated_target_label(
    ui: &mut egui::Ui,
    text: &str,
    font_id: egui::FontId,
    color: egui::Color32,
    width: f32,
) -> egui::Response {
    let width = width.max(1.0);
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = width;
    job.wrap.max_rows = 1;
    job.wrap.break_anywhere = true;
    job.wrap.overflow_character = Some('…');
    job.append(
        text,
        0.0,
        egui::TextFormat {
            font_id,
            color,
            valign: egui::Align::Center,
            ..Default::default()
        },
    );
    let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
    let (rect, response) =
        ui.allocate_exact_size(egui::vec2(width, TARGET_ROW_HEIGHT), egui::Sense::hover());
    let ink = galley.mesh_bounds;
    let pos = egui::pos2(
        rect.left() - ink.left(),
        rect.center().y - ink.center().y + TARGET_TEXT_Y_OFFSET,
    );
    ui.painter()
        .galley(pos + egui::vec2(0.55, 0.0), galley.clone(), color);
    ui.painter().galley(pos, galley, color);
    response
}

/// payload marker for preset drag & drop (distinct from chain's usize)
#[allow(dead_code)] // retained drag payload type for the disabled preset DnD path
#[derive(Clone, Copy)]
struct PresetIdx(usize);

/// filter picker tree: folders (below shaders/ / models/) become submenus
#[derive(Default)]
struct FilterNode {
    files: Vec<(StageKind, String, String)>, // (kind, full path, display)
    dirs: std::collections::BTreeMap<String, FilterNode>,
}

fn compact_status_line(ui: &mut egui::Ui, color: egui::Color32, msg: &str) {
    let one_line = msg.split_whitespace().collect::<Vec<_>>().join(" ");
    let max_chars = 140usize;
    let short = if one_line.chars().count() > max_chars {
        let mut s: String = one_line.chars().take(max_chars).collect();
        s.push_str("...");
        s
    } else {
        one_line
    };
    ui.colored_label(color, format!("! {short}"))
        .on_hover_text(msg);
}

fn tr<'a>(lang: UiLanguage, ja: &'a str, en: &'a str) -> &'a str {
    i18n::legacy_text(lang, ja, en)
}

fn hotkey_error_text(lang: UiLanguage, error: &HotkeyValidationError) -> String {
    match error {
        HotkeyValidationError::KeyCount => tr(
            lang,
            "キーは修飾キーを含めて2個または3個にしてください。",
            "Use a total of two or three keys, including modifiers.",
        ),
        HotkeyValidationError::MissingModifier => tr(
            lang,
            "Ctrl、Alt、Shiftのいずれかが必要です。",
            "Ctrl, Alt, or Shift is required.",
        ),
        HotkeyValidationError::MissingPrimary => tr(
            lang,
            "修飾キーだけでは登録できません。最後に文字・数字・Fキーなどを押してください。",
            "Modifier-only shortcuts cannot be registered. Add a letter, number, or function key.",
        ),
        HotkeyValidationError::MultiplePrimary => tr(
            lang,
            "通常キーは1個だけ指定できます。",
            "Only one non-modifier key is allowed.",
        ),
        HotkeyValidationError::DuplicateKey => tr(
            lang,
            "同じキーが重複しています。",
            "The same key is duplicated.",
        ),
        HotkeyValidationError::UnsupportedKey(_) => tr(
            lang,
            "このキーはグローバルショートカットには使用できません。",
            "This key is not supported for global shortcuts.",
        ),
        HotkeyValidationError::WinModifier => tr(
            lang,
            "Winキーを含む組み合わせはWindows操作と衝突するため使用できません。",
            "Win-key combinations are blocked because they conflict with Windows shortcuts.",
        ),
        HotkeyValidationError::Reserved => tr(
            lang,
            "この組み合わせはアプリまたはWindowsの予約操作と重複しています。",
            "This combination is reserved by the app or Windows.",
        ),
    }
    .to_string()
}

fn egui_hotkey_name(key: egui::Key) -> String {
    match key {
        egui::Key::ArrowUp => "Up".into(),
        egui::Key::ArrowDown => "Down".into(),
        egui::Key::ArrowLeft => "Left".into(),
        egui::Key::ArrowRight => "Right".into(),
        _ => key.name().to_string(),
    }
}

fn capture_hotkey_candidate(ctx: &egui::Context) -> Option<(String, egui::Key)> {
    ctx.input(|input| {
        input.events.iter().rev().find_map(|event| {
            let egui::Event::Key {
                key,
                pressed: true,
                repeat: false,
                modifiers,
                ..
            } = event
            else {
                return None;
            };
            if *key == egui::Key::Escape {
                return None;
            }
            let mut parts = Vec::new();
            if modifiers.ctrl {
                parts.push("Ctrl".to_string());
            }
            if modifiers.alt {
                parts.push("Alt".to_string());
            }
            if modifiers.shift {
                parts.push("Shift".to_string());
            }
            parts.push(egui_hotkey_name(*key));
            Some((parts.join("+"), *key))
        })
    })
}

fn held_hotkey_modifiers(ctx: &egui::Context) -> String {
    ctx.input(|input| {
        let mut parts = Vec::new();
        if input.modifiers.ctrl {
            parts.push("Ctrl");
        }
        if input.modifiers.alt {
            parts.push("Alt");
        }
        if input.modifiers.shift {
            parts.push("Shift");
        }
        parts.join("+")
    })
}

fn hotkey_chips(ui: &mut egui::Ui, hotkey: &str) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        for part in hotkey.split('+').filter(|p| !p.is_empty()) {
            egui::Frame::new()
                .fill(egui::Color32::from_rgb(58, 60, 64))
                .corner_radius(egui::CornerRadius::same(3))
                .inner_margin(egui::Margin::symmetric(7, 3))
                .show(ui, |ui| {
                    visually_centered_label(
                        ui,
                        part,
                        egui::FontId::new(11.5, egui::FontFamily::Proportional),
                        ui.visuals().text_color(),
                        true,
                        15.0,
                    );
                });
        }
    });
}

fn may_follow_foreground_target(
    running: bool,
    _own_process_elevated: bool,
    _candidate_elevated: Option<bool>,
) -> bool {
    // Selection is harmless and must remain understandable: even when this
    // process cannot control an elevated window, show it as the selected
    // target. Start() then explains why capture cannot begin and offers an
    // administrator relaunch.
    !running
}

fn save_gui_screenshot(path: &std::path::Path, image: &egui::ColorImage) -> anyhow::Result<()> {
    let file = std::fs::File::create(path)?;
    let mut encoder = png::Encoder::new(
        std::io::BufWriter::new(file),
        image.size[0] as u32,
        image.size[1] as u32,
    );
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let rgba: Vec<u8> = image
        .pixels
        .iter()
        .flat_map(|pixel| pixel.to_array())
        .collect();
    writer.write_image_data(&rgba)?;
    writer.finish()?;
    Ok(())
}

#[allow(dead_code)] // retained compact-language control for UI layout variants
fn language_pill(ui: &mut egui::Ui, lang: UiLanguage) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(84.0, 23.0), egui::Sense::click());
    let label = "Language";
    let active = i18n::short_name(lang);
    let bg = if resp.hovered() {
        egui::Color32::from_rgb(36, 42, 52)
    } else {
        egui::Color32::from_rgb(27, 32, 39)
    };
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 118, 145));
    ui.painter()
        .rect(rect, 11.5, bg, stroke, egui::StrokeKind::Inside);
    let chip_center = rect.right_center() - egui::vec2(14.0, 0.0);
    let chip_rect = egui::Rect::from_center_size(chip_center, egui::vec2(18.0, 18.0));
    ui.painter()
        .circle_filled(chip_center, 9.0, egui::Color32::from_rgb(78, 126, 176));
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(9.5),
        egui::Color32::from_rgb(225, 232, 238),
    );
    ui.painter().text(
        chip_rect.center(),
        egui::Align2::CENTER_CENTER,
        active,
        egui::FontId::proportional(8.2),
        egui::Color32::WHITE,
    );
    resp.on_hover_text("日本語 / English")
}

#[allow(dead_code)] // retained compact auto/manual language control
fn language_mode_pill(
    ui: &mut egui::Ui,
    mode: UiLanguageMode,
    effective: UiLanguage,
) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(96.0, 23.0), egui::Sense::click());
    let label = "Language";
    let active = if mode == UiLanguageMode::Auto {
        "Auto"
    } else {
        i18n::short_name(effective)
    };
    let tip = if mode == UiLanguageMode::Auto {
        format!("Auto / {}", i18n::native_name(effective))
    } else {
        format!(
            "{} [{}]",
            i18n::native_name(effective),
            i18n::tag(effective)
        )
    };
    let bg = if resp.hovered() {
        egui::Color32::from_rgb(36, 42, 52)
    } else {
        egui::Color32::from_rgb(27, 32, 39)
    };
    let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 118, 145));
    ui.painter()
        .rect(rect, 11.5, bg, stroke, egui::StrokeKind::Inside);
    let chip_width = if active == "Auto" { 31.0 } else { 23.0 };
    let chip_center = rect.right_center() - egui::vec2(5.0 + chip_width * 0.5, 0.0);
    let chip_rect = egui::Rect::from_center_size(chip_center, egui::vec2(chip_width, 19.0));
    ui.painter()
        .rect_filled(chip_rect, 8.5, egui::Color32::from_rgb(78, 126, 176));
    ui.painter().text(
        rect.left_center() + egui::vec2(8.0, 0.0),
        egui::Align2::LEFT_CENTER,
        label,
        egui::FontId::proportional(9.2),
        egui::Color32::from_rgb(225, 232, 238),
    );
    ui.painter().text(
        chip_rect.center(),
        egui::Align2::CENTER_CENTER,
        active,
        egui::FontId::proportional(if active == "Auto" { 6.7 } else { 8.2 }),
        egui::Color32::WHITE,
    );
    resp.on_hover_text(tip)
}

fn ui_mode_pill(ui: &mut egui::Ui, mode: UiMode) -> Option<UiMode> {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(144.0, 23.0), egui::Sense::click());
    let bg = if response.hovered() {
        egui::Color32::from_rgb(36, 42, 52)
    } else {
        egui::Color32::from_rgb(27, 32, 39)
    };
    ui.painter().rect(
        rect,
        11.5,
        bg,
        egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 118, 145)),
        egui::StrokeKind::Inside,
    );
    let segment = rect.width() / 3.0;
    let mini = egui::Rect::from_min_size(rect.min, egui::vec2(segment, rect.height()));
    let basic = mini.translate(egui::vec2(segment, 0.0));
    let full = basic.translate(egui::vec2(segment, 0.0));
    let selected = match mode {
        UiMode::Mini => mini,
        UiMode::Basic => basic,
        UiMode::Full => full,
    }
    .shrink(2.0);
    ui.painter()
        .rect_filled(selected, 9.5, egui::Color32::from_rgb(78, 126, 176));
    let text_color = egui::Color32::from_rgb(235, 240, 245);
    ui.painter().text(
        mini.center(),
        egui::Align2::CENTER_CENTER,
        "Mini",
        egui::FontId::proportional(8.5),
        text_color,
    );
    ui.painter().text(
        basic.center(),
        egui::Align2::CENTER_CENTER,
        "Basic",
        egui::FontId::proportional(8.5),
        text_color,
    );
    ui.painter().text(
        full.center(),
        egui::Align2::CENTER_CENTER,
        "Full",
        egui::FontId::proportional(8.5),
        text_color,
    );

    if response.clicked()
        && let Some(pointer) = response.interact_pointer_pos()
    {
        return Some(if pointer.x < rect.left() + segment {
            UiMode::Mini
        } else if pointer.x < rect.left() + segment * 2.0 {
            UiMode::Basic
        } else {
            UiMode::Full
        });
    }
    None
}

fn load_bar(ui: &mut egui::Ui, width: f32, label: &str, value: f32, low: egui::Color32) {
    ui.vertical(|ui| {
        ui.set_width(width);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(label).monospace().size(10.5).strong());
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    egui::RichText::new(format!("{:>3.0}%", value))
                        .monospace()
                        .size(10.5),
                );
            });
        });
        let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 8.0), egui::Sense::hover());
        ui.painter()
            .rect_filled(rect, 2.0, egui::Color32::from_rgb(45, 47, 51));
        let value = value.clamp(0.0, 100.0);
        let yellow = egui::Color32::from_rgb(232, 190, 55);
        let red = egui::Color32::from_rgb(220, 72, 72);
        for (start, end, color) in [
            (0.0_f32, value.min(70.0), low),
            (70.0, value.min(85.0), yellow),
            (85.0, value, red),
        ] {
            if end > start {
                let segment = egui::Rect::from_min_max(
                    egui::pos2(rect.left() + rect.width() * start / 100.0, rect.top()),
                    egui::pos2(rect.left() + rect.width() * end / 100.0, rect.bottom()),
                );
                ui.painter().rect_filled(segment, 0.0, color);
            }
        }
    });
}

fn effective_language(mode: UiLanguageMode) -> UiLanguage {
    i18n::mode_language(mode).unwrap_or_else(|| {
        let requested = win32::preferred_ui_language_tags()
            .into_iter()
            .next()
            .unwrap_or_else(|| "en-US".to_owned());
        let resolved = i18n::from_bcp47(&requested);
        if requested.to_ascii_lowercase().starts_with("pt-")
            && !requested.eq_ignore_ascii_case("pt-BR")
        {
            log::info!("locale auto alias: requested={requested} resolved=pt-BR");
        }
        resolved
    })
}

fn capture_resolution_fullscreen_guard(
    session_origin_fullscreen: Option<bool>,
    live_fullscreen: bool,
) -> bool {
    // Once a capture session has an origin snapshot, that snapshot is the only
    // fullscreen authority. The live HWND may be monitor-sized solely because
    // Neo just applied a 1920x1080 capture resolution.
    session_origin_fullscreen.unwrap_or(live_fullscreen)
}

#[cfg(test)]
fn capture_resolution_content_size(client_size: (i32, i32), crop: CaptureCrop) -> (i32, i32) {
    if !crop.enabled || client_size.0 <= 0 || client_size.1 <= 0 {
        return client_size;
    }
    let applied = crop.applied_to(client_size.0 as u32, client_size.1 as u32);
    (applied.content_w as i32, applied.content_h as i32)
}

fn capture_resolution_label(v: Option<CaptureResolution>, lang: UiLanguage) -> String {
    match v {
        Some(r) if u64::from(r.w) * 3 == u64::from(r.h) * 4 => {
            format!("{}x{} (4:3)", r.w, r.h)
        }
        Some(r) => format!("{}x{}", r.w, r.h),
        None => tr(lang, "自動", "Auto").to_string(),
    }
}

fn is_picture_in_picture_title(title: &str) -> bool {
    let lower = title.to_lowercase();
    title.contains("ピクチャー イン ピクチャー")
        || lower.contains("picture in picture")
        || lower.contains("picture-in-picture")
}

fn even_capture_neighbors(value: u32, min_value: u32, max_value: u32) -> (u32, u32) {
    let min_even = if min_value & 1 == 0 {
        min_value
    } else {
        min_value.saturating_add(1)
    };
    let max_even = max_value & !1;
    let bounded = value.clamp(min_even, max_even.max(min_even));
    let down = (bounded & !1).max(min_even).min(max_even.max(min_even));
    let up = bounded
        .saturating_add(bounded & 1)
        .max(min_even)
        .min(max_even.max(min_even));
    (down, up)
}

fn evenize_capture_fit(
    limit: CaptureResolution,
    proposed: CaptureResolution,
    current_aspect: f64,
) -> CaptureResolution {
    // WGC normalizes an odd right/bottom edge by replicating one pixel so the
    // GPU path always receives an even-sized frame. If the source HWND itself
    // remains odd-sized (for example 640x359), the capture-resolution state
    // machine waits forever for a frame shape WGC can never publish. Choose the
    // closest even client size up front so source geometry, WGC, ONNX and input
    // mapping all share one coordinate space.
    let (w0, w1) = even_capture_neighbors(proposed.w, 160, limit.w.max(160));
    let (h0, h1) = even_capture_neighbors(proposed.h, 120, limit.h.max(120));
    let mut best = CaptureResolution { w: w0, h: h0 };
    let mut best_error = f64::INFINITY;
    let mut best_area = 0u64;
    for w in [w0, w1] {
        for h in [h0, h1] {
            if w == 0 || h == 0 || w > limit.w || h > limit.h {
                continue;
            }
            let aspect = w as f64 / h as f64;
            let error = ((aspect / current_aspect) - 1.0).abs();
            let area = u64::from(w) * u64::from(h);
            if error < best_error - 1e-9 || ((error - best_error).abs() <= 1e-9 && area > best_area)
            {
                best = CaptureResolution { w, h };
                best_error = error;
                best_area = area;
            }
        }
    }
    best
}

fn fit_resolution_preserving_aspect(
    limit: CaptureResolution,
    current_size: (i32, i32),
) -> CaptureResolution {
    if current_size.0 <= 0 || current_size.1 <= 0 {
        return limit;
    }
    // PiP client sizes often differ from their nominal aspect by one physical
    // pixel (for example 758x426). Treat that as the same aspect and honor the
    // requested capture size exactly when possible. The final applied size is
    // nevertheless kept even because WGC's GPU path edge-pads odd dimensions.
    let current_aspect = current_size.0 as f64 / current_size.1 as f64;
    let requested_aspect = limit.w as f64 / limit.h as f64;
    let proposed = if ((current_aspect / requested_aspect) - 1.0).abs() <= 0.0025 {
        limit
    } else {
        let scale =
            (limit.w as f64 / current_size.0 as f64).min(limit.h as f64 / current_size.1 as f64);
        let mut w = (current_size.0 as f64 * scale).round().max(160.0) as u32;
        let mut h = (current_size.1 as f64 * scale).round().max(120.0) as u32;
        w = w.min(limit.w);
        h = h.min(limit.h);
        CaptureResolution { w, h }
    };
    evenize_capture_fit(limit, proposed, current_aspect)
}

fn parse_capture_resolution(text: &str) -> Option<Option<CaptureResolution>> {
    let s = text.trim();
    if s.is_empty()
        || s.eq_ignore_ascii_case("auto")
        || s.eq_ignore_ascii_case("none")
        || s == "自動"
    {
        return Some(None);
    }
    let s = s.strip_suffix("(4:3)").unwrap_or(s).trim();
    let normalized = s
        .replace('×', "x")
        .replace('X', "x")
        .replace('＊', "x")
        .replace('*', "x");
    let (w, h) = normalized.split_once('x')?;
    let w = w.trim().parse::<u32>().ok()?.clamp(160, 7680);
    let h = h.trim().parse::<u32>().ok()?.clamp(120, 4320);
    Some(Some(CaptureResolution { w, h }))
}

fn capture_resolution_presets() -> [Option<CaptureResolution>; 13] {
    [
        None,
        Some(CaptureResolution { w: 480, h: 360 }),
        Some(CaptureResolution { w: 640, h: 360 }),
        Some(CaptureResolution { w: 640, h: 480 }),
        Some(CaptureResolution { w: 768, h: 576 }),
        Some(CaptureResolution { w: 854, h: 480 }),
        Some(CaptureResolution { w: 960, h: 540 }),
        Some(CaptureResolution { w: 960, h: 720 }),
        Some(CaptureResolution { w: 1152, h: 648 }),
        Some(CaptureResolution { w: 1280, h: 720 }),
        Some(CaptureResolution { w: 1440, h: 810 }),
        Some(CaptureResolution { w: 1600, h: 900 }),
        Some(CaptureResolution { w: 1920, h: 1080 }),
    ]
}

fn build_filter_tree(available: &[(StageKind, String)]) -> FilterNode {
    let mut root = FilterNode::default();
    for (kind, path) in available {
        if let Some(name) = path.strip_prefix("builtin:") {
            let display = if name.to_ascii_lowercase().starts_with("neoflow") {
                "NeoFlow".to_string()
            } else {
                name.to_string()
            };
            root.files.push((*kind, path.clone(), display));
            continue;
        }
        let mut parts: Vec<&str> = path.split('/').collect();
        // hide the shaders/models root folder
        if parts.len() > 1 && (parts[0] == "shaders" || parts[0] == "models") {
            parts.remove(0);
        }
        let file = parts.pop().unwrap_or_default().to_string();
        let mut node = &mut root;
        for dir in parts {
            node = node.dirs.entry(dir.to_string()).or_default();
        }
        node.files.push((*kind, path.clone(), file));
    }
    root
}

fn is_frozen_neoflow_stage(stage: &StageSpec) -> bool {
    stage.kind == StageKind::Flow || stage.path.to_ascii_lowercase().contains("neoflow")
}

fn log_ui_test_rect(id: &str, response: &egui::Response) {
    #[cfg(test)]
    {
        test_ui_rects()
            .lock()
            .unwrap()
            .insert(id.to_string(), response.rect);
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var_os("NEO_UI_TEST_RECTS").is_some()) {
        return;
    }
    static SEEN: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        std::sync::OnceLock::new();
    let seen = SEEN.get_or_init(Default::default);
    if seen.lock().unwrap().insert(id.to_string()) {
        log::info!(
            "ui-test-rect: id={id} min=({:.1},{:.1}) max=({:.1},{:.1})",
            response.rect.min.x,
            response.rect.min.y,
            response.rect.max.x,
            response.rect.max.y
        );
    }
}

#[cfg(test)]
fn test_ui_rects() -> &'static std::sync::Mutex<std::collections::HashMap<String, egui::Rect>> {
    static RECTS: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<String, egui::Rect>>,
    > = std::sync::OnceLock::new();
    RECTS.get_or_init(Default::default)
}

fn render_filter_tree_picker(
    ui: &mut egui::Ui,
    node: &FilterNode,
    id_path: &str,
    lang: UiLanguage,
    add: &mut Option<(StageKind, String)>,
) {
    // Root files are rendered before folders. NeoFlow is a root-level built-in,
    // so it remains the first choice even when many categorized folders exist.
    for (kind, path, display) in &node.files {
        let badge = match kind {
            StageKind::Glsl => "GLSL",
            StageKind::Onnx => "ONNX",
            StageKind::Flow => tr(lang, "内蔵", "Built-in"),
        };
        let response = ui.selectable_label(false, format!("[{badge}] {display}"));
        log_ui_test_rect(&format!("filter-item:{path}"), &response);
        if response.hovered() && ui.input(|input| input.pointer.primary_pressed()) {
            log::info!("filter-picker-select-press: kind={kind:?} path={path}");
        }
        if response.clicked() {
            log::info!("filter-picker-select: kind={kind:?} path={path}");
            *add = Some((*kind, path.clone()));
        }
    }
    if !node.dirs.is_empty() && !node.files.is_empty() {
        ui.separator();
    }
    for (name, child) in &node.dirs {
        let child_id = format!("{id_path}/{name}");
        let response = egui::CollapsingHeader::new(name)
            .id_salt(&child_id)
            .show(ui, |ui| {
                render_filter_tree_picker(ui, child, &child_id, lang, add)
            });
        log_ui_test_rect(
            &format!("filter-folder:{child_id}"),
            &response.header_response,
        );
    }
    if node.dirs.is_empty() && node.files.is_empty() {
        ui.weak(tr(lang, "（フィルターがありません）", "(No filters found)"));
    }
}

fn append_filter_stage(chain: &mut Vec<StageSpec>, kind: StageKind, path: String) -> usize {
    let index = chain.len();
    let mut params = std::collections::BTreeMap::new();
    if is_resize_shader_path(&path) {
        params.insert("RESIZE_SCALE".to_string(), 0.75);
    }
    chain.push(StageSpec {
        kind,
        path,
        enabled: true,
        params,
    });
    index
}

fn is_resize_shader_path(path: &str) -> bool {
    path.replace('\\', "/")
        .to_ascii_lowercase()
        .starts_with("shaders/resize/")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PanelDiagSnapshot {
    rect: Option<(i32, i32, i32, i32)>,
    expected_size: (i32, i32),
    effective_show: bool,
    lurk: bool,
    native_visible: bool,
    layered: bool,
    passthrough: bool,
    topmost: bool,
}

#[derive(Clone, Copy, Debug)]
struct PendingUiModeTransition {
    mode: UiMode,
    /// True for an actual Mini/Basic/Full mode switch. False for a protected
    /// same-Full content resize (for example Stats ON -> OFF).
    mode_change: bool,
    target: egui::Vec2,
    armed_at: Instant,
    /// False while only the native/root surface is being resized. Once true,
    /// the target mode is already drawing behind the snapshot/cloak transition guard.
    mode_committed: bool,
    /// Keep at least two complete target-mode frames covered before revealing the
    /// root GUI. Fast AutoNoVsync paths can render those frames in only a few ms,
    /// so committed_at also enforces a minimum DWM-visible hold interval.
    hidden_warmup_frames: u8,
    committed_at: Option<Instant>,
    /// True when a native GDI snapshot shield is covering the old/root GUI
    /// representation while the WGPU surface is resized behind it.
    shield_applied: bool,
    /// DWM cloak is retained only as a fail-safe if snapshot capture/creation
    /// fails. The normal transition path never removes the GUI from DWM.
    cloak_applied: bool,
}

struct App {
    engine: EngineHandle,
    hotkeys: HotkeyThread,
    app_dir: std::path::PathBuf,
    settings: Settings,
    store: PresetStore,
    chain: Vec<StageSpec>,
    saved_chain: Vec<StageSpec>,
    saved_aspect_correction: PresetAspectCorrection,
    saved_crop: CaptureCrop,
    saved_capture_resolution: Option<CaptureResolution>,
    available: Vec<(StageKind, String)>,
    /// Hardware DXGI adapters exposed only in Full mode when two or more are
    /// present. The stable LUID, not the volatile list index, is persisted.
    gpu_adapters: Vec<GpuAdapter>,
    /// Legacy v635 restart-handoff signal. v636 no longer creates this for GPU
    /// selection, but honoring an inherited marker keeps upgrade handoffs safe.
    gpu_handoff_ready_path: Option<std::path::PathBuf>,
    gpu_handoff_signaled: bool,
    gpu_selection_verified: bool,
    /// Opt-in Vulkan LUID probe used only to validate the future GLSL backend.
    /// It never changes the active OpenGL/GLSL path.
    vulkan_gpu_probe_done: bool,
    filter_picker_open: bool,
    hotkey_editor_open: bool,
    hotkey_editor_candidate: String,
    hotkey_editor_error: Option<String>,
    hotkey_capture_pending: Option<(String, egui::Key)>,
    resize_scale_editor: Option<(usize, f32)>,
    gui_test_screenshot_path: Option<std::path::PathBuf>,
    gui_test_screenshot_requested: bool,
    gui_test_frame_count: u32,
    target_hwnd: isize,
    target_title: String,
    elevated_target_notice: Option<(isize, String)>,
    capture_resolution_fullscreen_notice_open: bool,
    capture_resolution_fullscreen_notice_seen_seq: u64,
    capture_resolution_reapply_seen_seq: u64,
    /// Last engine-side source-occlusion safety-stop notice consumed by the GUI.
    source_occlusion_notice_seen_seq: u64,
    /// A short, non-blocking topmost explanation shown after an occlusion stop.
    source_occlusion_notice_until: Option<Instant>,
    source_occlusion_notice_hwnd: isize,
    /// DirectML/GLSL crop preview stays live, but DragValue can emit near-60 Hz
    /// shape changes. Keep only the newest crop and publish it at a bounded
    /// cadence so the image continues moving without geometry-reset storms.
    live_crop_pending: Option<CaptureCrop>,
    live_crop_last_sent: Option<Instant>,
    /// Crop DragValue can change every GUI frame. Persist only the settled
    /// value after the gesture instead of rewriting settings.json per pixel.
    crop_settings_save_due: Option<Instant>,
    /// TensorRT Crop geometry is preset-owned in v597: numeric editing is
    /// DirectML-only. Keep the existing settled commit fields for the permitted
    /// saved-Crop ON/OFF toggle and as a defensive barrier for future callers.
    tensorrt_crop_commit_pending: Option<CaptureCrop>,
    tensorrt_crop_commit_due: Option<Instant>,
    last_poll: Instant,
    save_as_open: bool,
    save_as_name: String,
    save_as_error: Option<PresetEditError>,
    confirm_delete: bool,
    gui_hwnd: isize,
    gui_topmost_applied: Option<bool>,
    /// When turning GUI topmost OFF during capture, keep the native GUI topmost
    /// until the v459-compatible WGPU composition anchor is physically ready.
    /// This prevents a one-composition flash while DWM switches presentation paths.
    gui_topmost_off_pending: bool,
    /// v525: while the staged OFF transition prepares the first WGPU keep-alive
    /// surface, remove the still-TOPMOST root GUI from DWM composition. Otherwise
    /// the expensive first anchor Present can expose the old GUI for one frame
    /// immediately before it is demoted behind the fullscreen overlay.
    gui_topmost_off_cloak_applied: bool,
    gui_mouse_passthrough_applied: Option<bool>,
    gui_priority_sent: Option<(isize, bool)>,
    panel_visible: bool,
    /// long-press drag state: (list_id, index, press_start, activated)
    drag: Option<(u8, usize, Instant, bool)>,
    ratio_text: String,
    capture_resolution_text: String,
    panel_hwnd: isize,
    /// Tiny eframe/WGPU viewport used only as an AMD/DWM composition keep-alive
    /// while capture is running with the main GUI not-topmost.  The visible
    /// control panel remains the native GDI host; this viewport sits underneath
    /// it and never owns input.  v448-v454 intentionally kept the old WGPU panel
    /// alive for this exact compositor contract, which v465 removed.
    compositor_anchor_hwnd: isize,
    compositor_anchor_state_sent: Option<(bool, isize, isize, isize)>,
    compositor_anchor_last_warn_at: Instant,
    /// Single auxiliary dialog viewport used by Mini mode. Keeping one stable
    /// HWND avoids multiplying cursor ownership boundaries while still letting
    /// every warning/editor escape Mini's 105-point main viewport.
    mini_dialog_hwnd: isize,
    panel_metrics_sent: Option<bool>,
    panel_screenshot_feedback_until: Option<Instant>,
    panel_chip_lurking: bool,
    panel_bar_shown: bool,
    panel_leave_at: Option<Instant>,
    /// hover-restore arms only after the pointer has LEFT the chip once —
    /// otherwise the — click (pointer still on the chip) undoes itself
    panel_hover_restore_armed: bool,
    /// After collapsing the panel, the pointer is still over the chip. Keep
    /// hover-expand disarmed until the pointer has left the panel once.
    panel_hover_expand_armed: bool,
    panel_state_sent: Option<(bool, bool)>,
    panel_layout_sent: Option<(isize, (i32, i32), (i32, i32))>,
    panel_placed_for_run: bool,
    /// While restoring the full bar from the transparent lurk chip, keep the parent
    /// fully transparent until its final bar geometry is committed. The GDI mirror is
    /// then painted/shown first and the parent is revealed last, preventing a one-frame
    /// glimpse of the resizing WGPU surface.
    panel_gdi_reveal_pending: bool,
    /// Diagnostic-only control-panel compositor trace. These fields never
    /// influence layout, z-order, visibility, pacing, or input routing.
    panel_diag_seq: u64,
    panel_diag_last_frame_at: Option<Instant>,
    panel_diag_last_snapshot: Option<PanelDiagSnapshot>,
    panel_diag_last_trace_at: Instant,
    last_panel_hotkey: Option<Instant>,
    was_running: bool,
    was_capture_busy: bool,
    /// Direct Win32 Start/Stop presses are committed on pointer-down. This
    /// short bridge covers the few milliseconds before engine status reflects
    /// the new latch state; after that starting/running/stopping keeps the
    /// physical button depressed until the state transition really completes.
    main_control_press_until: Option<Instant>,
    capture_idle_since: Instant,
    qa_auto_start: bool,
    qa_panel_preview: bool,
    resource_monitor: ResourceMonitor,
    basic_stats_layout_applied: Option<bool>,
    basic_stats_rows_applied: Option<usize>,
    mini_language_width_applied: Option<i32>,
    basic_language_width_applied: Option<i32>,
    full_language_width_applied: Option<i32>,
    /// Extra vertical space required by wrapped Full settings rows. Computed
    /// from the same measured widths as the root Full target and fed back into
    /// the bottom panel on the next frame so narrow/localized layouts do not
    /// donate that space to the filter-chain panel instead.
    full_settings_wrap_extra: f32,
    /// A GUI mode change is resized first and committed only after the root
    /// viewport reports the requested final inner size. This prevents a fast
    /// WGPU/DWM path from presenting a half-transition layout.
    pending_ui_mode: Option<PendingUiModeTransition>,
    /// QA-only language override; never written to settings.json.
    locale_test_override: Option<UiLanguage>,
    custom_locales: Vec<i18n::CustomLocale>,
    tensorrt_availability: TensorRtAvailability,
    onnx_backend_selected: OnnxBackendPreference,
    onnx_backend_pending: Option<OnnxBackendPreference>,
    onnx_backend_revision_seen: u64,
}

impl App {
    fn hotkey_bindings(toggle: &str) -> Vec<(i32, String)> {
        vec![
            (HK_TOGGLE, toggle.to_string()),
            (HK_QUIT, "Ctrl+Alt+Q".into()),
            (HK_PANEL, "Ctrl+Alt+P".into()),
            (HK_GUI_TOPMOST, "Ctrl+Alt+G".into()),
        ]
    }

    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        cc.egui_ctx.set_theme(egui::ThemePreference::Dark);
        cc.egui_ctx.set_zoom_factor(1.1);
        cc.egui_ctx.all_styles_mut(|st| {
            st.spacing.button_padding = egui::vec2(14.0, 6.0);
            st.spacing.item_spacing = egui::vec2(10.0, 8.0);
            st.spacing.interact_size.y = 28.0;
            st.visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(7);
            st.visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(7);
            st.visuals.widgets.active.corner_radius = egui::CornerRadius::same(7);
        });

        let dir = {
            let d = app_dir();
            if !d.join("shaders").exists()
                && std::env::current_dir()
                    .map(|c| c.join("shaders").exists())
                    .unwrap_or(false)
            {
                std::env::current_dir().unwrap()
            } else {
                d
            }
        };
        // Portable root: all relative backend/cache paths resolve beneath the
        // executable directory. TensorRT receives ASCII-only relative paths,
        // which also keeps Japanese and other Unicode install paths working.
        if let Err(error) = std::env::set_current_dir(&dir) {
            log::warn!("portable-root: could not select {}: {error}", dir.display());
        }
        let gui_test_mode = std::env::var("NEO_GUI_SCREENSHOT_MODE").unwrap_or_default();
        let mut settings = load_settings(&dir);
        let gpu_adapters = gpu::enumerate_adapters();
        if let Some(saved_luid) = settings.gpu_adapter_luid
            && gpu::adapter_for_luid(&gpu_adapters, Some(saved_luid)).is_none()
        {
            log::warn!(
                "gpu-selection-startup-fallback: saved_luid={saved_luid:016x} reason=adapter-not-present -> Auto"
            );
            settings.gpu_adapter_luid = None;
            settings.gpu_force_vulkan = false;
            save_settings(&dir, &settings);
        } else if settings.gpu_adapter_luid.is_none() && settings.gpu_force_vulkan {
            // There is deliberately no "Auto [Vulkan]" entry. Older/corrupt
            // settings must not silently turn Auto into a Vulkan route.
            settings.gpu_force_vulkan = false;
            save_settings(&dir, &settings);
        }
        let gpu_handoff_ready_path =
            std::env::var_os("NEO_GPU_HANDOFF_READY").map(std::path::PathBuf::from);
        let custom_locales = i18n::discover_custom_locales(&dir);
        // QA-only override used by the mandatory external-locale layout test.
        // It never writes settings and has no effect unless explicitly set.
        if let Ok(tag) = std::env::var("NEO_GUI_CUSTOM_LANGUAGE") {
            if custom_locales
                .iter()
                .any(|locale| locale.tag.eq_ignore_ascii_case(&tag))
            {
                settings.language_mode = UiLanguageMode::EnUs;
                settings.custom_language = Some(tag);
            }
        }
        // QA-only: render the actual selected code inside the language frame.
        if let Ok(tag) = std::env::var("NEO_GUI_SELECTED_LANGUAGE") {
            let selected = i18n::from_bcp47(&tag);
            settings.language_mode = i18n::fixed_mode(selected);
            settings.custom_language = None;
        }
        let active_custom = settings.custom_language.as_ref().and_then(|tag| {
            custom_locales
                .iter()
                .find(|locale| locale.tag.eq_ignore_ascii_case(tag))
        });
        i18n::activate_custom_locale(active_custom);
        if active_custom.is_none() {
            settings.custom_language = None;
        }
        settings.fps_cap = settings.fps_cap.clamp(5, MAX_FPS_CAP);
        settings.aspect_width_scale = sanitize_aspect_correction_scale(settings.aspect_width_scale);
        settings.aspect_height_scale =
            sanitize_aspect_correction_scale(settings.aspect_height_scale);
        let cli_locale =
            std::env::args().find_map(|arg| arg.strip_prefix("--test-locale=").map(str::to_owned));
        let locale_test_override = cli_locale
            .or_else(|| std::env::var("NEO_GUI_LANGUAGE").ok())
            .map(|tag| i18n::from_bcp47(&tag));
        // NeoAccel is frozen: retain the backend for later evaluation, but
        // keep it disabled and hidden from the GUI in this release.
        chidescaler_neo::render::onnx_accel::set_neoaccel_enabled(false);
        settings.ui_mode = match gui_test_mode.to_ascii_lowercase().as_str() {
            "mini" => UiMode::Mini,
            "basic" | "basic_stats" => UiMode::Basic,
            "full" | "full_stats" => UiMode::Full,
            _ => settings.ui_mode,
        };
        if gui_test_mode.eq_ignore_ascii_case("basic_stats")
            || gui_test_mode.eq_ignore_ascii_case("full_stats")
        {
            settings.stats_on = true;
        }
        let initial_language =
            locale_test_override.unwrap_or_else(|| effective_language(settings.language_mode));
        install_ui_fonts(&cc.egui_ctx);
        select_ui_font_for_locale(&cc.egui_ctx, initial_language);
        logging::set_file_logging(&dir, settings.log_on);
        log::info!("cHiDeScaler-Neo build {BUILD_ID} app_dir={}", dir.display());
        // v628 compatibility inventory: parser/compiler only. It deliberately
        // runs before any selected-GPU Vulkan production route and never creates
        // a Vulkan instance or changes the stable OpenGL path.
        if vulkan_onepass::compatibility_scan_requested() {
            vulkan_onepass::scan_bundled_compatibility(&dir);
        }
        if chidescaler_neo::render::vulkan_multipass::compatibility_scan_requested() {
            chidescaler_neo::render::vulkan_multipass::scan_bundled_compatibility(&dir);
        }
        let ratio_text = format!("{:.1}", settings.ratio);
        let capture_resolution_text =
            capture_resolution_label(settings.capture_resolution, initial_language);
        let store = PresetStore::load(&dir);
        let active_aspect_correction = store
            .active()
            .map(|preset| preset.effective_aspect_correction())
            .unwrap_or_default();
        let active_crop = store
            .active()
            .map(|preset| preset.effective_crop())
            .unwrap_or_default();
        // v672: fixed capture-resolution metadata remains preset-owned, while
        // missing metadata means "inherit the current GUI value". This lets users
        // compare multiple presets at one capture resolution without each legacy /
        // unspecified preset forcing the selector back to Auto. A fixed preset still
        // keeps its saved value as the dirty-marker baseline, while an unspecified
        // preset adopts the current settings.json value as its baseline.
        let active_preset_capture_resolution =
            store.active().and_then(|preset| preset.capture_resolution);
        let active_capture_resolution_baseline =
            active_preset_capture_resolution.or(settings.capture_resolution);
        if let Some(saved_aspect) = store.active().and_then(|preset| preset.aspect_correction) {
            let saved_aspect = saved_aspect.sanitized();
            saved_aspect.apply_to_settings(&mut settings);
            log::info!(
                "preset-aspect-load: source=startup preset='{}' metadata=present enabled={} scale={:.2}x{:.2}",
                store.data.active,
                saved_aspect.enabled,
                saved_aspect.width_scale,
                saved_aspect.height_scale
            );
        } else {
            // Legacy/bundled presets have no aspect metadata. Preserve the
            // existing settings.json value on startup so v557's ordinary UI
            // persistence is not regressed. On a genuinely first launch,
            // Settings::default() is still OFF / 1.00 x 1.00. Explicitly
            // selecting a legacy preset later uses the safe preset default
            // (OFF / 1.00 x 1.00), preventing a prior CPS2 correction from
            // leaking into an unrelated preset.
            log::info!(
                "preset-aspect-load: source=startup preset='{}' metadata=legacy settings_preserved=true enabled={} scale={:.2}x{:.2}",
                store.data.active,
                settings.aspect_correction,
                settings.aspect_width_scale,
                settings.aspect_height_scale
            );
        }
        if let Some(saved_crop) = store.active().and_then(|preset| preset.crop) {
            settings.capture_crop = saved_crop;
            log::info!(
                "preset-crop-load: source=startup preset='{}' metadata=present enabled={} edges=({}, {}, {}, {})",
                store.data.active,
                saved_crop.enabled,
                saved_crop.left,
                saved_crop.top,
                saved_crop.right,
                saved_crop.bottom
            );
        } else {
            // Match the established aspect-metadata startup behavior: a legacy
            // active preset does not erase settings.json on launch, while an
            // explicit preset selection below uses the safe OFF/zero default.
            log::info!(
                "preset-crop-load: source=startup preset='{}' metadata=legacy settings_preserved=true enabled={} edges=({}, {}, {}, {})",
                store.data.active,
                settings.capture_crop.enabled,
                settings.capture_crop.left,
                settings.capture_crop.top,
                settings.capture_crop.right,
                settings.capture_crop.bottom
            );
        }
        let chain: Vec<StageSpec> = store
            .active()
            .map(|p| {
                p.chain
                    .iter()
                    .filter(|stage| !is_frozen_neoflow_stage(stage))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();
        let available = discover_filters(&dir);
        let tensorrt_availability = detect_tensorrt_backend(&dir, settings.gpu_adapter_luid);
        let tensorrt_crop_start_allowed =
            tensorrt_crop_switch_allowed(settings.capture_crop, active_crop);
        let onnx_backend_selected = if settings.onnx_backend == OnnxBackendPreference::TensorRT
            && tensorrt_availability.available
            && tensorrt_crop_start_allowed
        {
            OnnxBackendPreference::TensorRT
        } else {
            if settings.onnx_backend == OnnxBackendPreference::TensorRT
                && tensorrt_availability.available
                && !tensorrt_crop_start_allowed
            {
                log::warn!(
                    "onnx-backend-startup-fallback: requested=TensorRT active=DirectML reason=unsaved-crop-geometry current=({}, {}, {}, {}) saved=({}, {}, {}, {})",
                    settings.capture_crop.left,
                    settings.capture_crop.top,
                    settings.capture_crop.right,
                    settings.capture_crop.bottom,
                    active_crop.left,
                    active_crop.top,
                    active_crop.right,
                    active_crop.bottom
                );
                settings.onnx_backend = OnnxBackendPreference::DirectML;
            }
            OnnxBackendPreference::DirectML
        };
        let trt_cache_root =
            tensorrt_cache_root(&dir, &tensorrt_availability, settings.gpu_adapter_luid);
        chidescaler_neo::render::onnx_stage::maintain_tensorrt_cache(&trt_cache_root);
        let engine = EngineHandle::spawn_with_backend(
            dir.clone(),
            onnx_backend_selected,
            tensorrt_availability.device_id,
            trt_cache_root,
        );
        engine
            .metrics
            .set_enabled(settings.stats_on || logging::diagnostics_enabled());
        let hotkeys = HotkeyThread::start(
            Self::hotkey_bindings(&settings.hotkey_toggle),
            engine.stop_handle(),
        );
        let hotkey_editor_candidate = settings.hotkey_toggle.clone();
        let gui_test_screenshot_path = std::env::var_os("NEO_GUI_SCREENSHOT").map(Into::into);
        let gui_test_save_as = gui_test_mode.eq_ignore_ascii_case("save_as");
        let gui_test_save_as_name = if gui_test_save_as {
            store.data.active.clone()
        } else {
            String::new()
        };
        let qa_target_hwnd = std::env::var("NEO_QA_TARGET_HWND")
            .ok()
            .and_then(|value| value.parse::<isize>().ok())
            .filter(|hwnd| win32::is_window_valid(*hwnd))
            .unwrap_or(0);
        let qa_auto_start = qa_target_hwnd != 0 && std::env::var_os("NEO_QA_AUTO_START").is_some();
        let qa_panel_preview = std::env::var_os("NEO_QA_PANEL_PREVIEW").is_some();
        Self {
            engine,
            hotkeys,
            app_dir: dir,
            settings,
            saved_chain: chain.clone(),
            saved_aspect_correction: active_aspect_correction,
            saved_crop: active_crop,
            saved_capture_resolution: active_capture_resolution_baseline,
            chain,
            store,
            available,
            gpu_adapters,
            gpu_handoff_ready_path,
            gpu_handoff_signaled: false,
            gpu_selection_verified: false,
            vulkan_gpu_probe_done: false,
            filter_picker_open: gui_test_mode.eq_ignore_ascii_case("filters"),
            hotkey_editor_open: gui_test_mode.eq_ignore_ascii_case("hotkey"),
            hotkey_editor_candidate,
            hotkey_editor_error: None,
            hotkey_capture_pending: None,
            resize_scale_editor: None,
            gui_test_screenshot_path,
            gui_test_screenshot_requested: false,
            gui_test_frame_count: 0,
            target_hwnd: qa_target_hwnd,
            target_title: win32::window_title(qa_target_hwnd),
            elevated_target_notice: None,
            capture_resolution_fullscreen_notice_open: false,
            capture_resolution_fullscreen_notice_seen_seq: 0,
            capture_resolution_reapply_seen_seq: 0,
            source_occlusion_notice_seen_seq: 0,
            source_occlusion_notice_until: None,
            source_occlusion_notice_hwnd: 0,
            live_crop_pending: None,
            live_crop_last_sent: None,
            crop_settings_save_due: None,
            tensorrt_crop_commit_pending: None,
            tensorrt_crop_commit_due: None,
            last_poll: Instant::now() - Duration::from_secs(1),
            save_as_open: gui_test_save_as,
            save_as_name: gui_test_save_as_name,
            save_as_error: None,
            confirm_delete: false,
            gui_hwnd: 0,
            gui_topmost_applied: None,
            gui_topmost_off_pending: false,
            gui_topmost_off_cloak_applied: false,
            gui_mouse_passthrough_applied: None,
            gui_priority_sent: None,
            panel_visible: true,
            drag: None,
            ratio_text,
            capture_resolution_text,
            panel_hwnd: 0,
            compositor_anchor_hwnd: 0,
            compositor_anchor_state_sent: None,
            compositor_anchor_last_warn_at: Instant::now() - Duration::from_secs(2),
            mini_dialog_hwnd: 0,
            panel_metrics_sent: None,
            panel_screenshot_feedback_until: None,
            panel_chip_lurking: false,
            panel_bar_shown: true,
            panel_leave_at: None,
            panel_hover_restore_armed: false,
            panel_hover_expand_armed: true,
            panel_state_sent: None,
            panel_layout_sent: None,
            panel_placed_for_run: false,
            panel_gdi_reveal_pending: false,
            panel_diag_seq: 0,
            panel_diag_last_frame_at: None,
            panel_diag_last_snapshot: None,
            panel_diag_last_trace_at: Instant::now() - Duration::from_secs(1),
            last_panel_hotkey: None,
            was_running: false,
            was_capture_busy: false,
            main_control_press_until: None,
            capture_idle_since: Instant::now(),
            qa_auto_start,
            qa_panel_preview,
            resource_monitor: ResourceMonitor::new(),
            basic_stats_layout_applied: None,
            basic_stats_rows_applied: None,
            mini_language_width_applied: None,
            basic_language_width_applied: None,
            full_language_width_applied: None,
            full_settings_wrap_extra: 0.0,
            pending_ui_mode: None,
            locale_test_override,
            custom_locales,
            tensorrt_availability,
            onnx_backend_selected,
            onnx_backend_pending: None,
            onnx_backend_revision_seen: 0,
        }
    }

    fn effective_language(&self) -> UiLanguage {
        self.locale_test_override
            .unwrap_or_else(|| effective_language(self.settings.language_mode))
    }

    fn maybe_finalize_gpu_selection_startup(&mut self) {
        if self.gpu_selection_verified && self.gpu_handoff_signaled && self.vulkan_gpu_probe_done {
            return;
        }
        let status = self.engine.status.lock().unwrap().clone();
        if !status.render_gpu_ready {
            return;
        }

        if !self.gpu_selection_verified {
            let requested = self.settings.gpu_adapter_luid;
            match (requested, status.render_gpu_luid) {
                (Some(requested), Some(actual)) if requested == actual => {
                    let name = gpu::adapter_for_luid(&self.gpu_adapters, Some(requested))
                        .map(|adapter| adapter.name.as_str())
                        .unwrap_or("unknown");
                    log::info!(
                        "gpu-selection-verified: requested_luid={requested:016x} actual_gl_luid={actual:016x} result=match name='{}'",
                        name
                    );
                }
                (Some(requested), Some(actual)) => {
                    log::warn!(
                        "gpu-selection-verified: requested_luid={requested:016x} actual_gl_luid={actual:016x} result=mismatch renderer='{}'; compute remains on selected GPU (ONNX + compatible Vulkan GLSL), presentation stays on OpenGL GPU and cross-GPU staging may apply",
                        status.render_gpu_name
                    );
                }
                (Some(requested), None) => {
                    log::warn!(
                        "gpu-selection-verified: requested_luid={requested:016x} actual_gl_luid=unavailable result=unverified renderer='{}'",
                        status.render_gpu_name
                    );
                }
                (None, Some(actual)) => {
                    log::info!(
                        "gpu-selection-active: requested=Auto actual_gl_luid={actual:016x} renderer='{}'",
                        status.render_gpu_name
                    );
                }
                (None, None) => {
                    log::info!(
                        "gpu-selection-active: requested=Auto actual_gl_luid=unavailable renderer='{}'",
                        status.render_gpu_name
                    );
                }
            }
            self.gpu_selection_verified = true;
        }

        if !self.gpu_handoff_signaled {
            if let Some(path) = self.gpu_handoff_ready_path.take() {
                gpu::signal_relaunch_render_ready(
                    &path,
                    status.render_gpu_luid,
                    &status.render_gpu_name,
                );
            }
            self.gpu_handoff_signaled = true;
        }

        // v610 experimental foundation: retain the exact-LUID/GLSL probes and add
        // an opt-in synthetic RGBA storage-image probe. All probe results use the dedicated
        // result-file sink independent of the normal diagnostic log. None of these paths
        // touches a capture frame, GL texture, user filter, ONNX resource,
        // compositor, input state, or presentation resource.
        if !self.vulkan_gpu_probe_done {
            // v616: write an application-side breadcrumb before any capture session exists.
            // This separates "the BAT launched Neo but Neo did not see the opt-in" from
            // "the opt-in was seen but capture/session routing never started".  It does
            // not create a Vulkan instance or touch the render path.
            if vulkan_onepass::production_one_pass_requested() {
                let requested = self.settings.gpu_adapter_luid;
                let requested_text = requested
                    .map(|luid| format!("{luid:016x}"))
                    .unwrap_or_else(|| "Auto".to_string());
                let actual_text = status
                    .render_gpu_luid
                    .map(|luid| format!("{luid:016x}"))
                    .unwrap_or_else(|| "unavailable".to_string());
                let sink_present = std::env::var("NEO_VULKAN_PROBE_RESULT")
                    .map(|value| !value.trim().is_empty())
                    .unwrap_or(false);
                let line = format!(
                    "vulkan-production-glsl: phase=process-start build={} env=enabled result_sink={} requested_luid={} actual_gl_luid={} renderer='{}' vulkan_init=false next=session-start",
                    BUILD_ID,
                    sink_present,
                    requested_text,
                    actual_text,
                    status.render_gpu_name.replace('\r', " ").replace('\n', " "),
                );
                log::info!("{line}");
                vulkan_gpu::record_probe_result(&line);
            }

            let target_luid = self.settings.gpu_adapter_luid.or(status.render_gpu_luid);
            if vulkan_gpu::probe_requested() {
                match target_luid {
                    Some(target_luid) => match vulkan_gpu::probe_selected_luid(target_luid) {
                        Ok(result) => {
                            let line = format!(
                                "vulkan-gpu-probe: result=ready requested_luid={:016x} matched_luid={:016x} name='{}' vendor={:04x} device={:04x} api={} queue_family={} queue_flags={:?} path=diagnostic-only glsl_backend=OpenGL-unchanged",
                                target_luid,
                                result.luid,
                                result.name,
                                result.vendor_id,
                                result.device_id,
                                result.api_version_string(),
                                result.queue_family_index,
                                result.queue_flags,
                            );
                            log::info!("{line}");
                            vulkan_gpu::record_probe_result(&line);
                        }
                        Err(error) => {
                            let line = format!(
                                "vulkan-gpu-probe: result=unavailable requested_luid={target_luid:016x} error={error:#} fallback=OpenGL-unchanged"
                            );
                            log::warn!("{line}");
                            vulkan_gpu::record_probe_result(&line);
                        }
                    },
                    None => {
                        let line = "vulkan-gpu-probe: result=skipped reason=no-dxgi-luid fallback=OpenGL-unchanged";
                        log::warn!("{line}");
                        vulkan_gpu::record_probe_result(line);
                    }
                }
            }
            if vulkan_gpu::glsl_probe_requested() {
                match target_luid {
                    Some(target_luid) => {
                        match vulkan_gpu::probe_glsl_compute_selected_luid(target_luid) {
                            Ok(result) => {
                                let line = format!(
                                    "vulkan-glsl-probe: result=ready requested_luid={:016x} matched_luid={:016x} name='{}' vendor={:04x} device={:04x} api={} queue_family={} queue_flags={:?} compiler=naga glsl=450 spirv_words={} workload=u32x64 verified={}/64 checksum={} expected_checksum=4096 path=diagnostic-only user_glsl_backend=OpenGL-unchanged",
                                    target_luid,
                                    result.gpu.luid,
                                    result.gpu.name,
                                    result.gpu.vendor_id,
                                    result.gpu.device_id,
                                    result.gpu.api_version_string(),
                                    result.gpu.queue_family_index,
                                    result.gpu.queue_flags,
                                    result.spirv_words,
                                    result.verified_values,
                                    result.checksum,
                                );
                                log::info!("{line}");
                                vulkan_gpu::record_probe_result(&line);
                            }
                            Err(error) => {
                                let line = format!(
                                    "vulkan-glsl-probe: result=unavailable requested_luid={target_luid:016x} error={error:#} fallback=OpenGL-unchanged"
                                );
                                log::warn!("{line}");
                                vulkan_gpu::record_probe_result(&line);
                            }
                        }
                    }
                    None => {
                        let line = "vulkan-glsl-probe: result=skipped reason=no-dxgi-luid fallback=OpenGL-unchanged";
                        log::warn!("{line}");
                        vulkan_gpu::record_probe_result(line);
                    }
                }
            }
            if vulkan_gpu::image_probe_requested() {
                match target_luid {
                    Some(target_luid) => {
                        match vulkan_gpu::probe_glsl_image_selected_luid(target_luid) {
                            Ok(result) => {
                                let line = format!(
                                    "vulkan-image-probe: result=ready requested_luid={:016x} matched_luid={:016x} name='{}' vendor={:04x} device={:04x} api={} queue_family={} queue_flags={:?} compiler=naga glsl=450 spirv_words={} format=R8G8B8A8_UNORM image=8x8 transform=bgra-swap verified={}/64 checksum={:016x} expected_checksum={:016x} path=diagnostic-only user_glsl_backend=OpenGL-unchanged",
                                    target_luid,
                                    result.gpu.luid,
                                    result.gpu.name,
                                    result.gpu.vendor_id,
                                    result.gpu.device_id,
                                    result.gpu.api_version_string(),
                                    result.gpu.queue_family_index,
                                    result.gpu.queue_flags,
                                    result.spirv_words,
                                    result.verified_pixels,
                                    result.checksum,
                                    result.expected_checksum,
                                );
                                log::info!("{line}");
                                vulkan_gpu::record_probe_result(&line);
                            }
                            Err(error) => {
                                let line = format!(
                                    "vulkan-image-probe: result=unavailable requested_luid={target_luid:016x} error={error:#} fallback=OpenGL-unchanged"
                                );
                                log::warn!("{line}");
                                vulkan_gpu::record_probe_result(&line);
                            }
                        }
                    }
                    None => {
                        let line = "vulkan-image-probe: result=skipped reason=no-dxgi-luid fallback=OpenGL-unchanged";
                        log::warn!("{line}");
                        vulkan_gpu::record_probe_result(line);
                    }
                }
            }
            self.vulkan_gpu_probe_done = true;
        }
    }

    fn gpu_selector_control(&mut self, ui: &mut egui::Ui, lang: UiLanguage, enabled: bool) {
        if !gpu_selector_visible(&self.gpu_adapters) {
            return;
        }

        // One compact selector carries two independent pieces of state:
        // physical GPU LUID and compatible-GLSL routing policy. The duplicated
        // "[Vulkan]" rows are intentionally placed after the ordinary GPU rows
        // so users who only need to correct Windows' iGPU auto-selection can
        // keep the normal same-GPU OpenGL fast path.
        let mut choice = (
            self.settings.gpu_adapter_luid,
            self.settings.gpu_force_vulkan && self.settings.gpu_adapter_luid.is_some(),
        );
        let selected_text = match gpu::adapter_for_luid(&self.gpu_adapters, choice.0) {
            Some(adapter) if choice.1 => format!("{} [Vulkan]", adapter.name),
            Some(adapter) => adapter.name.clone(),
            None => i18n::text(lang, "common.auto").to_string(),
        };
        let original_choice = choice;
        let mut changed = false;
        let group = ui.add_enabled_ui(enabled, |ui| {
            ui.label("GPU:");
            let response = egui::ComboBox::from_id_salt("gpu_adapter_selector")
                .width(190.0)
                .selected_text(selected_text)
                .show_ui(ui, |ui| {
                    changed |= ui
                        .selectable_value(
                            &mut choice,
                            (None, false),
                            i18n::text(lang, "common.auto"),
                        )
                        .changed();
                    for adapter in &self.gpu_adapters {
                        changed |= ui
                            .selectable_value(
                                &mut choice,
                                (Some(adapter.luid), false),
                                &adapter.name,
                            )
                            .changed();
                    }
                    ui.separator();
                    for adapter in &self.gpu_adapters {
                        let label = format!("{} [Vulkan]", adapter.name);
                        changed |= ui
                            .selectable_value(&mut choice, (Some(adapter.luid), true), label)
                            .changed();
                    }
                })
                .response;
            response.on_hover_text(i18n::text(lang, "gpu.select_help"));
        });
        if !enabled {
            group
                .response
                .on_hover_text(i18n::text(lang, "gpu.stop_to_change"));
        }
        if changed && choice != original_choice {
            let (requested, force_vulkan_glsl) = choice;
            let render_gpu_luid = self.engine.status.lock().unwrap().render_gpu_luid;
            let effective_gpu_luid = resolve_compute_gpu_luid(requested, render_gpu_luid);
            let gpu_adapter = gpu::device_id_for_luid(&self.gpu_adapters, effective_gpu_luid);
            let requested_name = gpu::adapter_for_luid(&self.gpu_adapters, requested)
                .map(|adapter| {
                    if force_vulkan_glsl {
                        format!("{} [Vulkan]", adapter.name)
                    } else {
                        adapter.name.clone()
                    }
                })
                .unwrap_or_else(|| "Auto".to_string());

            // TensorRT's device id is CUDA-specific and cannot be reused after
            // a DXGI selection change. [Vulkan] changes only GLSL routing, so
            // ONNX/TensorRT identity remains the same physical selected GPU.
            let auto_tensorrt_unambiguous = requested.is_some()
                || self
                    .gpu_adapters
                    .iter()
                    .filter(|adapter| adapter.vendor_id == 0x10de)
                    .count()
                    == 1;
            let same_tensorrt_gpu = self.tensorrt_availability.available
                && auto_tensorrt_unambiguous
                && self.tensorrt_availability.gpu_luid == effective_gpu_luid;
            let new_tensorrt = if same_tensorrt_gpu {
                log::info!(
                    "tensorrt-gpu-map-reuse: requested_luid={} effective_luid={} cuda_device_id={:?} reason=same-physical-gpu",
                    requested
                        .map(|luid| format!("{luid:016x}"))
                        .unwrap_or_else(|| "Auto".to_string()),
                    effective_gpu_luid
                        .map(|luid| format!("{luid:016x}"))
                        .unwrap_or_else(|| "Auto".to_string()),
                    self.tensorrt_availability.device_id,
                );
                self.tensorrt_availability.clone()
            } else {
                detect_tensorrt_backend(&self.app_dir, requested)
            };
            let mut backend = self.onnx_backend_selected;
            if backend == OnnxBackendPreference::TensorRT && !new_tensorrt.available {
                log::warn!(
                    "gpu-selection-tensorrt-fallback: requested_luid={} name='{}' reason='{}' active=DirectML",
                    requested
                        .map(|luid| format!("{luid:016x}"))
                        .unwrap_or_else(|| "Auto".to_string()),
                    requested_name,
                    new_tensorrt
                        .reason
                        .as_deref()
                        .unwrap_or("selected GPU is not available to TensorRT"),
                );
                backend = OnnxBackendPreference::DirectML;
                self.settings.onnx_backend = OnnxBackendPreference::DirectML;
                self.onnx_backend_selected = OnnxBackendPreference::DirectML;
                self.onnx_backend_pending = None;
            }
            self.tensorrt_availability = new_tensorrt;
            self.settings.gpu_adapter_luid = requested;
            self.settings.gpu_force_vulkan = force_vulkan_glsl;
            save_settings(&self.app_dir, &self.settings);
            let cache_root =
                tensorrt_cache_root(&self.app_dir, &self.tensorrt_availability, requested);
            chidescaler_neo::render::onnx_stage::maintain_tensorrt_cache(&cache_root);
            self.engine.send(Cmd::SetGpuSelection {
                gpu_adapter,
                explicit_gpu_luid: requested,
                force_vulkan_glsl,
                backend,
                trt_device_id: self.tensorrt_availability.device_id,
                cache_root,
            });
            log::info!(
                "gpu-selection-requested: luid={} name='{}' force_vulkan={} effective_luid={} dml_device={:?} onnx_backend={backend:?} trt_device={:?} relaunch=false presentation_gpu_unchanged=true",
                requested
                    .map(|luid| format!("{luid:016x}"))
                    .unwrap_or_else(|| "Auto".to_string()),
                requested_name,
                force_vulkan_glsl,
                effective_gpu_luid
                    .map(|luid| format!("{luid:016x}"))
                    .unwrap_or_else(|| "Auto".to_string()),
                gpu_adapter,
                self.tensorrt_availability.device_id,
            );
        }
    }

    fn request_onnx_backend_switch(&mut self, backend: OnnxBackendPreference) {
        if backend == OnnxBackendPreference::TensorRT && !self.tensorrt_availability.available {
            return;
        }
        if backend == OnnxBackendPreference::TensorRT
            && !tensorrt_crop_switch_allowed(self.settings.capture_crop, self.saved_crop)
        {
            log::warn!(
                "onnx-backend-switch-blocked: requested=TensorRT reason=unsaved-crop-geometry current=({}, {}, {}, {}) saved=({}, {}, {}, {})",
                self.settings.capture_crop.left,
                self.settings.capture_crop.top,
                self.settings.capture_crop.right,
                self.settings.capture_crop.bottom,
                self.saved_crop.left,
                self.saved_crop.top,
                self.saved_crop.right,
                self.saved_crop.bottom
            );
            return;
        }
        // Reflect the user's requested state immediately. The render engine
        // remains authoritative and the revision handler rolls this back on a
        // failed switch, but the checkbox must not appear to reject an idle
        // DirectML <-> TensorRT click while the command crosses threads.
        self.onnx_backend_selected = backend;
        self.onnx_backend_pending = Some(backend);
        log::info!("onnx-backend-switch-requested: requested={backend:?}");
        self.engine.send(Cmd::SwitchOnnxBackend {
            backend,
            trt_device_id: self.tensorrt_availability.device_id,
            cache_root: tensorrt_cache_root(
                &self.app_dir,
                &self.tensorrt_availability,
                self.settings.gpu_adapter_luid,
            ),
            specs: self.chain.clone(),
        });
    }

    fn prepare_ui_mode_switch(&mut self, ui: &egui::Ui, lang: UiLanguage, mode: UiMode) {
        if mode == self.settings.ui_mode || self.pending_ui_mode.is_some() {
            return;
        }
        // A root WGPU surface resize can become visible to DWM before the first
        // target-layout frame is ready. Keep the old screen representation
        // visible with a native GDI snapshot shield while the real root HWND
        // resizes underneath it. Unlike v467's normal-path DWM cloak, this does
        // not expose the fullscreen video overlay where the GUI used to be.
        egui::Popup::close_all(ui.ctx());

        // Invalidate only the target mode's sizing latch. The current mode keeps
        // drawing while the native surface reaches the target geometry.
        let target = match mode {
            UiMode::Mini => {
                self.mini_language_width_applied = None;
                self.resize_mini_for_language(ui, lang, true)
            }
            UiMode::Basic => {
                self.basic_language_width_applied = None;
                let target = self.resize_basic_for_language(ui, lang, true);
                self.basic_stats_layout_applied = Some(self.settings.stats_on);
                self.basic_stats_rows_applied = Some(self.basic_stats_rows());
                target
            }
            UiMode::Full => {
                self.full_language_width_applied = None;
                self.resize_full_for_language(ui, lang)
            }
        };
        // Viewport commands above are queued by egui and are not applied until
        // after this update returns, so the root HWND still contains the fully
        // rendered old mode here. Capture the union of the current and target
        // footprints before DWM can resize it. The shield therefore preserves
        // exactly what the user was seeing: old GUI pixels plus the surrounding
        // video/desktop pixels that a growing GUI is about to cover.
        let pixels_per_point = ui.ctx().pixels_per_point().max(0.1);
        let shield_applied = self.gui_hwnd != 0
            && win32::show_gui_transition_snapshot(
                self.gui_hwnd,
                target.x,
                target.y,
                pixels_per_point,
            );
        // Fail-visible safety: if the snapshot cannot be created, continue the
        // resize without hiding the root GUI. A compositor failure must never
        // leave the application cloaked or waiting for an uncloak retry.
        let cloak_applied = false;

        self.pending_ui_mode = Some(PendingUiModeTransition {
            mode,
            mode_change: true,
            target,
            armed_at: Instant::now(),
            mode_committed: false,
            hidden_warmup_frames: 0,
            committed_at: None,
            shield_applied,
            cloak_applied,
        });
        log::info!(
            "ui-mode-transition: prepared from={:?} to={mode:?} target={:.1}x{:.1} shield={} cloaked_fallback={}",
            self.settings.ui_mode,
            target.x,
            target.y,
            shield_applied,
            cloak_applied
        );
        ui.ctx().request_repaint_after(Duration::from_millis(8));
    }

    fn commit_pending_ui_mode_if_ready(&mut self, ctx: &egui::Context) {
        let Some(mut transition) = self.pending_ui_mode else {
            return;
        };

        // After the mode has been committed, keep rendering it behind the
        // snapshot shield (or the cloak fail-safe) for two complete eframe
        // updates. The next update cannot run until the previous frame has gone
        // through the renderer/Present path, so the final layout has a stable
        // front buffer before the old visual representation is released.
        if transition.mode_committed {
            if transition.shield_applied {
                win32::keep_gui_transition_snapshot_topmost();
            }
            if transition.hidden_warmup_frames > 0 {
                transition.hidden_warmup_frames -= 1;
                self.pending_ui_mode = Some(transition);
                ctx.request_repaint_after(Duration::from_millis(8));
                return;
            }
            // Two eframe updates can complete in under 10 ms on a fast GPU.
            // Hold the old snapshot through at least two 60-Hz compositor
            // intervals so DWM cannot expose a resize frame after the helper is
            // removed but before the final root surface has actually composed.
            const SNAPSHOT_MIN_HOLD_AFTER_COMMIT: Duration = Duration::from_millis(40);
            if let Some(committed_at) = transition.committed_at {
                let elapsed = committed_at.elapsed();
                if elapsed < SNAPSHOT_MIN_HOLD_AFTER_COMMIT {
                    self.pending_ui_mode = Some(transition);
                    ctx.request_repaint_after(SNAPSHOT_MIN_HOLD_AFTER_COMMIT - elapsed);
                    return;
                }
            }

            if transition.cloak_applied && self.gui_hwnd != 0 {
                let reveal_requested = win32::set_window_cloaked(self.gui_hwnd, false);
                let still_cloaked = win32::is_cloaked(self.gui_hwnd);
                if !reveal_requested || still_cloaked {
                    // Fail-visible contract: never clear the transition latch
                    // while our own root GUI still reports cloaked. Retry on a
                    // later frame rather than leaving the application invisible.
                    self.pending_ui_mode = Some(transition);
                    log::warn!(
                        "ui-mode-transition: uncloak-retry mode={:?} request_ok={} still_cloaked={}",
                        transition.mode,
                        reveal_requested,
                        still_cloaked
                    );
                    ctx.request_repaint_after(Duration::from_millis(16));
                    return;
                }
            }
            if transition.shield_applied {
                // The snapshot already covers the transition. Never block the
                // GUI thread waiting synchronously for DWM during a surface
                // resize; fail-visible behavior is safer than a frozen GUI.
                win32::keep_gui_transition_snapshot_topmost();
                win32::hide_gui_transition_snapshot();
            }
            if transition.mode_change {
                log::info!(
                    "ui-mode-transition: revealed mode={:?} snapshot_released={} uncloaked_fallback={}",
                    transition.mode,
                    transition.shield_applied,
                    transition.cloak_applied
                );
            } else {
                log::info!(
                    "full-layout-transition: revealed target={:.1}x{:.1} snapshot_released={}",
                    transition.target.x,
                    transition.target.y,
                    transition.shield_applied
                );
            }
            self.pending_ui_mode = None;
            ctx.request_repaint();
            return;
        }

        if transition.shield_applied {
            win32::keep_gui_transition_snapshot_topmost();
        }
        let current = ctx.input(|input| input.viewport().inner_rect.map(|rect| rect.size()));
        let geometry_ready = current.as_ref().is_some_and(|size| {
            (size.x - transition.target.x).abs() <= 1.5
                && (size.y - transition.target.y).abs() <= 1.5
        });
        let timed_out = transition.armed_at.elapsed() >= Duration::from_millis(250);
        if !geometry_ready && !timed_out {
            ctx.request_repaint_after(Duration::from_millis(8));
            return;
        }

        // A same-Full shrink is cosmetic, never worth risking a stuck GUI. If
        // the native/WGPU surface did not reach the requested smaller geometry
        // inside the guard interval, remove the shield and keep the current
        // working size. Unlike an actual mode switch there is nothing that must
        // be committed. A later real layout change will produce a fresh key.
        if !geometry_ready && timed_out && !transition.mode_change {
            if transition.shield_applied {
                win32::hide_gui_transition_snapshot();
            }
            log::warn!(
                "full-layout-transition: resize-timeout-cancelled target={:.1}x{:.1} current={:?} action=keep-current",
                transition.target.x,
                transition.target.y,
                current.as_ref().map(|size| (size.x, size.y))
            );
            self.pending_ui_mode = None;
            ctx.request_repaint();
            return;
        }

        if transition.mode_change {
            self.settings.ui_mode = transition.mode;
            save_settings(&self.app_dir, &self.settings);
        }
        transition.mode_committed = true;
        transition.hidden_warmup_frames = 2;
        transition.committed_at = Some(Instant::now());
        self.pending_ui_mode = Some(transition);
        if transition.mode_change {
            if geometry_ready {
                log::info!(
                    "ui-mode-transition: committed-covered mode={:?} target={:.1}x{:.1} warmup_frames=2",
                    transition.mode,
                    transition.target.x,
                    transition.target.y
                );
            } else {
                log::warn!(
                    "ui-mode-transition: commit-timeout-covered mode={:?} target={:.1}x{:.1} current={:?} warmup_frames=2",
                    transition.mode,
                    transition.target.x,
                    transition.target.y,
                    current.as_ref().map(|size| (size.x, size.y))
                );
            }
            log::info!("ui-mode: switched to {:?}", transition.mode);
        } else if geometry_ready {
            log::info!(
                "full-layout-transition: committed-covered target={:.1}x{:.1} warmup_frames=2",
                transition.target.x,
                transition.target.y
            );
        } else {
            log::warn!(
                "full-layout-transition: commit-timeout-covered target={:.1}x{:.1} current={:?} warmup_frames=2",
                transition.target.x,
                transition.target.y,
                current.as_ref().map(|size| (size.x, size.y))
            );
        }
        ctx.request_repaint_after(Duration::from_millis(8));
    }

    fn basic_stats_rows(&self) -> usize {
        let measured = self.engine.metrics.snapshot().display_stages().len();
        let configured = self.chain.iter().filter(|stage| stage.enabled).count();
        measured.max(configured)
    }

    fn basic_stats_height(&self) -> f32 {
        // The fixed portion ends at the resource meter. Grow only for the
        // statistics header and the number of filter rows actually displayed.
        438.0 + self.basic_stats_rows() as f32 * 22.0
    }

    fn resize_mini_for_language(
        &mut self,
        ui: &egui::Ui,
        lang: UiLanguage,
        allow_shrink: bool,
    ) -> egui::Vec2 {
        let measure = |text: &str, size: f32| {
            ui.painter()
                .layout_no_wrap(
                    text.to_owned(),
                    egui::FontId::new(size, locale_font_family(lang)),
                    ui.visuals().text_color(),
                )
                .size()
                .x
        };
        let start = i18n::text(lang, "capture.start");
        let running = i18n::text(lang, "capture.running");
        let capture = i18n::text(lang, "capture.short");
        let localized_width =
            measure(start, 15.0).max(measure(running, 15.0)) + measure(capture, 12.0);
        let zoom = ui.ctx().zoom_factor().max(0.1);
        let monitor_width = ui
            .ctx()
            .input(|input| input.viewport().monitor_size.map(|size| size.x * zoom))
            .unwrap_or(1400.0);
        // Fixed contribution: start-button padding, preset selector, capture
        // selector, separators, gaps, and the right-aligned mode/language row.
        let desired = (localized_width * zoom + 650.0)
            .max(MINI_DEFAULT_SIZE[0])
            .min((monitor_width - 32.0).max(MINI_DEFAULT_SIZE[0]));
        let current_width = ui
            .ctx()
            .input(|input| input.viewport().inner_rect.map(|r| r.width() * zoom));
        let desired = if allow_shrink {
            desired
        } else {
            current_width.map_or(desired, |w| desired.max(w))
        };
        let width_key = desired.round() as i32;
        let target = egui::vec2(desired / zoom, MINI_DEFAULT_SIZE[1] / zoom);
        if self.mini_language_width_applied == Some(width_key) {
            return target;
        }
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                desired / zoom,
                MINI_MIN_SIZE[1] / zoom,
            )));
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                desired / zoom,
                MINI_DEFAULT_SIZE[1] / zoom,
            )));
        self.mini_language_width_applied = Some(width_key);
        log::debug!(
            "mini-layout: language={} width={desired:.0}",
            i18n::tag(lang)
        );
        target
    }

    fn resize_basic_for_stats(&self, ctx: &egui::Context) {
        let zoom = ctx.zoom_factor().max(0.1);
        let width = ctx
            .input(|input| input.viewport().inner_rect.map(|rect| rect.width() * zoom))
            .unwrap_or(BASIC_DEFAULT_SIZE[0]);
        let height = if self.settings.stats_on {
            self.basic_stats_height()
        } else {
            BASIC_DEFAULT_SIZE[1]
        };
        let current_height =
            ctx.input(|input| input.viewport().inner_rect.map(|rect| rect.height() * zoom));
        let height = current_height.map_or(height, |current| height.max(current));
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
            width.max(BASIC_MIN_SIZE[0]) / zoom,
            height / zoom,
        )));
    }

    fn resize_basic_for_language(
        &mut self,
        ui: &egui::Ui,
        lang: UiLanguage,
        allow_shrink: bool,
    ) -> egui::Vec2 {
        let label_width: f32 = [
            tr(lang, "", "Display:"),
            tr(lang, "", "Fullscreen"),
            tr(lang, "", "Windowed"),
            tr(lang, "", "Capture size:"),
        ]
        .into_iter()
        .map(|text| {
            ui.painter()
                .layout_no_wrap(
                    text.to_owned(),
                    egui::FontId::proportional(12.0),
                    ui.visuals().text_color(),
                )
                .size()
                .x
        })
        .sum();
        let zoom = ui.ctx().zoom_factor().max(0.1);
        let monitor_width = ui
            .ctx()
            .input(|input| input.viewport().monitor_size.map(|size| size.x * zoom))
            .unwrap_or(1400.0);
        // Fixed controls: two combo arrows, two text fields/selectors,
        // separators, button padding and all inter-widget gaps.
        let desired = (label_width * zoom + 490.0)
            .max(BASIC_DEFAULT_SIZE[0])
            .min((monitor_width - 32.0).max(BASIC_DEFAULT_SIZE[0]));
        let current_width = ui
            .ctx()
            .input(|input| input.viewport().inner_rect.map(|r| r.width() * zoom));
        let desired = if allow_shrink {
            desired
        } else {
            current_width.map_or(desired, |w| desired.max(w))
        };
        let width_key = desired.round() as i32;
        let height = if self.settings.stats_on {
            self.basic_stats_height()
        } else {
            BASIC_DEFAULT_SIZE[1]
        };
        let target = egui::vec2(desired / zoom, height / zoom);
        if self.basic_language_width_applied == Some(width_key) {
            return target;
        }
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                desired / zoom,
                BASIC_MIN_SIZE[1] / zoom,
            )));
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                desired / zoom,
                height / zoom,
            )));
        self.basic_language_width_applied = Some(width_key);
        log::debug!(
            "basic-layout: language={} width={desired:.0}",
            i18n::tag(lang)
        );
        target
    }

    fn resize_full_for_language(&mut self, ui: &egui::Ui, lang: UiLanguage) -> egui::Vec2 {
        // Full follows the same measured sizing model as Basic mode:
        // measure only the current language's visible text, add the fixed
        // widths of the widgets that are actually present, and reserve a
        // stable amount for panel margins, scrollbar and the right edge.
        // No live ScrollArea width and no value from the previous language is
        // fed back into this calculation.
        let item_gap = ui.spacing().item_spacing.x;
        let separator_width = 14.0;
        let button_font = egui::TextStyle::Button.resolve(ui.style());
        let body_font = egui::TextStyle::Body.resolve(ui.style());

        let display_label_width = measured_text_width(ui, tr(lang, "", "Display:"), &button_font)
            .min(MAX_TRANSLATED_PLAIN_LABEL_WIDTH)
            + 8.0;
        let display_mode_width = display_label_width
            + bounded_button_width(ui, tr(lang, "", "Fullscreen"))
            + bounded_button_width(ui, tr(lang, "", "Windowed"))
            + 52.0
            + 30.0
            + measured_text_width(ui, "x", &body_font)
            + 6.0 * item_gap;

        let resize_width =
            bounded_plain_label_width(ui, tr(lang, "", "Resize:")) + 104.0 + 2.0 * item_gap;
        let capture_width = bounded_plain_label_width(ui, tr(lang, "", "Capture size:"))
            + capture_resolution_field_width(ui, lang)
            + 28.0
            + 3.0 * item_gap;
        let row_display = display_mode_width
            + resize_width
            + capture_width
            + 2.0 * separator_width
            + 2.0 * item_gap;

        let row_aspect = bounded_checkbox_width(ui, i18n::text(lang, "settings.aspect_correction"))
            + bounded_plain_label_width(ui, i18n::text(lang, "settings.aspect_mode"))
            + 112.0
            + bounded_plain_label_width(ui, i18n::text(lang, "settings.aspect_width"))
            + bounded_plain_label_width(ui, i18n::text(lang, "settings.aspect_height"))
            // Two DragValue controls plus their gaps.
            + 132.0
            + 7.0 * item_gap;
        let row_crop = bounded_checkbox_width(ui, i18n::text(lang, "settings.crop"))
            + [
                "settings.crop_left",
                "settings.crop_top",
                "settings.crop_right",
                "settings.crop_bottom",
            ]
            .iter()
            .map(|key| bounded_plain_label_width(ui, i18n::text(lang, key)))
            .sum::<f32>()
            // Four compact integer DragValues.
            + 224.0
            + 9.0 * item_gap;

        // Requested order:
        // FPS cap -> duplicate reduction -> rendering stabilization
        // -> VSync -> TensorRT.
        let mut cadence_labels = vec![
            tr(lang, "", "FPS cap"),
            tr(lang, "", "Duplicate reduction"),
            tr(lang, "", "Smooth pacing"),
            tr(lang, "", "VSync"),
        ];
        if tensorrt_option_visible(&self.tensorrt_availability) {
            cadence_labels.push(tr(lang, "TensorRT（準備中）", "TensorRT (preparing)"));
        }
        let row_cadence = cadence_labels
            .iter()
            .map(|text| bounded_checkbox_width(ui, text))
            .sum::<f32>()
            // FPS DragValue; reserve it even while the checkbox is off so
            // toggling the option cannot suddenly clip the row.
            + 64.0
            + (cadence_labels.len() as f32 + 1.0) * item_gap;

        let option_labels = [
            tr(lang, "", "Stats"),
            tr(lang, "", "Keep GUI on top"),
            tr(lang, "", "Show control panel"),
            tr(lang, "", "Client area only"),
            tr(lang, "", "Save log"),
        ];
        let hdr_option_width = if HDR_CAPTURE_OPTION_ENABLED {
            bounded_checkbox_width(ui, tr(lang, "", "HDR to SDR")) + item_gap
        } else {
            0.0
        };
        let gpu_option_width = if gpu_selector_visible(&self.gpu_adapters) {
            bounded_plain_label_width(ui, "GPU:") + 190.0 + separator_width + 3.0 * item_gap
        } else {
            0.0
        };
        let row_options = option_labels
            .iter()
            .map(|text| bounded_checkbox_width(ui, text))
            .sum::<f32>()
            + hdr_option_width
            + gpu_option_width
            + 30.0
            + separator_width
            + (option_labels.len() as f32 + 3.0) * item_gap;

        let cursor_labels = [
            tr(lang, "", "Auto-hide cursor"),
            tr(lang, "", "Natural cursor speed"),
            tr(lang, "", "Restart as admin"),
        ];
        let row_cursor = cursor_labels
            .iter()
            .map(|text| bounded_checkbox_width(ui, text))
            .sum::<f32>()
            + 92.0
            + separator_width
            + (cursor_labels.len() as f32 + 1.0) * item_gap;

        let measure_locale = |text: &str, size: f32| -> f32 {
            ui.painter()
                .layout_no_wrap(
                    text.to_owned(),
                    egui::FontId::new(size, locale_font_family(lang)),
                    ui.visuals().text_color(),
                )
                .size()
                .x
        };
        let start_label = i18n::text(lang, "capture.start")
            .trim_start_matches(|c: char| c == '▶' || c.is_whitespace());
        let running_label = i18n::text(lang, "capture.running")
            .trim_start_matches(|c: char| c == '●' || c.is_whitespace());
        let start_button_width = (measure_locale(start_label, 15.0)
            .max(measure_locale(running_label, 15.0))
            + 10.0
            + 7.0
            + 36.0)
            .max(150.0)
            .min(MAX_CAPTURE_BUTTON_WIDTH);
        let widest_preset = self
            .store
            .data
            .presets
            .iter()
            .map(|preset| measure_locale(&preset.name, 12.0))
            .fold(0.0_f32, f32::max);
        let preset_width = (widest_preset + 2.0 * ui.spacing().button_padding.x + 18.0)
            .ceil()
            .clamp(190.0, 420.0);
        let action_labels = [
            tr(lang, "", "Save"),
            tr(lang, "", "Save As"),
            tr(lang, "", "New"),
            tr(lang, "", "Delete"),
        ];
        let action_width = action_labels
            .iter()
            .map(|text| bounded_button_width(ui, text))
            .sum::<f32>();
        let row_toolbar =
            start_button_width + preset_width + action_width + separator_width + 7.0 * item_gap;

        let required_row_width = row_display
            .max(row_aspect)
            .max(row_crop)
            .max(row_cadence)
            .max(row_options)
            .max(row_cursor)
            .max(row_toolbar);
        let zoom = ui.ctx().zoom_factor().max(0.1);
        let monitor_width = ui
            .ctx()
            .input(|input| input.viewport().monitor_size.map(|size| size.x * zoom))
            .unwrap_or(1600.0);
        let monitor_height = ui
            .ctx()
            .input(|input| input.viewport().monitor_size.map(|size| size.y * zoom))
            .unwrap_or(1000.0);
        // Cap Full mode to the actual Windows work area, not only egui's raw
        // monitor size. The outer Win32 frame/title bar is outside egui's
        // InnerSize, so subtract its live non-client extent before choosing the
        // maximum inner viewport. This guarantees Full never grows behind the
        // taskbar or beyond the monitor; extra content stays in the existing
        // chain/settings ScrollAreas instead.
        let monitor_cap_width = (monitor_width - 16.0).max(320.0);
        let monitor_cap_height = (monitor_height - 16.0).max(320.0);
        let (work_cap_width, work_cap_height, work_area) =
            if self.gui_hwnd != 0 && win32::is_window_valid(self.gui_hwnd) {
                let work = win32::monitor_work_area_of(self.gui_hwnd);
                let (frame_w, frame_h) = match (
                    win32::window_rect(self.gui_hwnd),
                    win32::client_rect_on_screen(self.gui_hwnd),
                ) {
                    (Some((_, _, ow, oh)), Some((_, _, cw, ch))) => {
                        ((ow - cw).max(0), (oh - ch).max(0))
                    }
                    _ => (16, 40),
                };
                (
                    (work.2 - frame_w - 8).max(320) as f32,
                    (work.3 - frame_h - 8).max(320) as f32,
                    Some(work),
                )
            } else {
                (monitor_cap_width, monitor_cap_height, None)
            };
        let max_inner_width = monitor_cap_width.min(work_cap_width);
        let max_inner_height = monitor_cap_height.min(work_cap_height);
        let effective_min_width = FULL_MIN_SIZE[0].min(max_inner_width);
        let effective_min_height = FULL_MIN_SIZE[1].min(max_inner_height);
        let preferred_width = FULL_DEFAULT_SIZE[0].min(max_inner_width);
        let desired_unclamped = (required_row_width + FULL_LAYOUT_FIXED_RESERVE) * zoom;
        let desired = desired_unclamped
            .max(preferred_width)
            .max(effective_min_width)
            .min(max_inner_width);
        // Full mode is content-sized, not grow-only. The layout key below
        // prevents resize spam, while allowing stats OFF, shorter chains, or a
        // narrower locale to shrink the window back immediately. User manual
        // resizing is still preserved until a real layout requirement changes.
        let usable_content_width = (desired / zoom - FULL_LAYOUT_FIXED_RESERVE).max(1.0);
        // Every settings row that uses horizontal_wrapped participates in the
        // height model.  Previously only the preset toolbar did, so a narrow
        // monitor or a wider localization could silently create another visual
        // row without increasing the native Full window and clip the bottom.
        let settings_wrap_extra = [
            row_display,
            row_aspect,
            row_crop,
            row_cadence,
            row_options,
            row_cursor,
        ]
        .into_iter()
        .map(|width| full_wrapped_row_extra(width, usable_content_width, 32.0))
        .sum::<f32>();
        let toolbar_wrap_extra = full_wrapped_row_extra(
            row_toolbar,
            usable_content_width,
            FULL_TOOLBAR_WRAP_EXTRA_HEIGHT,
        );
        let total_wrap_extra = settings_wrap_extra + toolbar_wrap_extra;
        // The bottom settings panel owns these wrapped rows, so its own exact
        // height must receive the same extra space as the outer Full window.
        // Otherwise the extra native height is accidentally donated to the
        // chain panel and the settings area scrolls/clips despite free space.
        self.full_settings_wrap_extra = total_wrap_extra;

        let desired_content_height = 166.0
            + total_wrap_extra
            + full_chain_height(self.chain.len(), self.settings.stats_on)
            + full_settings_height(self.settings.stats_on, self.basic_stats_rows());
        let desired_height = desired_content_height
            .max(effective_min_height)
            .min(max_inner_height);
        let layout_key = full_layout_key(desired, max_inner_height, desired_height);
        let target = egui::vec2(desired / zoom, desired_height / zoom);
        if self.full_language_width_applied == Some(layout_key) {
            return target;
        }

        // Never stack a second root-surface resize on top of an existing
        // protected transition. Keep the new key unapplied; once the current
        // transition reveals, the next frame re-measures and targets the newest
        // requirement.
        if self.pending_ui_mode.is_some() && self.settings.ui_mode == UiMode::Full {
            return target;
        }

        let current_size = ui
            .ctx()
            .input(|input| input.viewport().inner_rect.map(|r| r.size()));
        let shrinking_existing_full = self.settings.ui_mode == UiMode::Full
            && current_size
                .as_ref()
                .is_some_and(|size| target.x + 1.5 < size.x || target.y + 1.5 < size.y);

        // v570 deliberately avoided direct same-Full WGPU surface shrink after
        // it could stall the GUI. Preserve that safety while still allowing
        // Stats OFF / shorter chains to reclaim stale blank space: cover the
        // old GUI with the same native snapshot shield used by mode switches,
        // resize underneath it, then reveal only after target frames are warm.
        if shrinking_existing_full {
            egui::Popup::close_all(ui.ctx());
            let pixels_per_point = ui.ctx().pixels_per_point().max(0.1);
            let shield_applied = self.gui_hwnd != 0
                && win32::show_gui_transition_snapshot(
                    self.gui_hwnd,
                    target.x,
                    target.y,
                    pixels_per_point,
                );
            self.pending_ui_mode = Some(PendingUiModeTransition {
                mode: UiMode::Full,
                mode_change: false,
                target,
                armed_at: Instant::now(),
                mode_committed: false,
                hidden_warmup_frames: 0,
                committed_at: None,
                shield_applied,
                cloak_applied: false,
            });
            log::info!(
                "full-layout-transition: prepared current={:?} target={:.1}x{:.1} shield={}",
                current_size.as_ref().map(|size| (size.x, size.y)),
                target.x,
                target.y,
                shield_applied
            );
        }

        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                effective_min_width / zoom,
                effective_min_height / zoom,
            )));
        ui.ctx()
            .send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                desired / zoom,
                desired_height / zoom,
            )));
        self.full_language_width_applied = Some(layout_key);
        log::debug!(
            "full-layout: language={} width={desired:.0} height={desired_height:.0} stats={} rows={} model=workarea-scroll reserve={:.0} usable={usable_content_width:.0} required(display={:.0},aspect={:.0},crop={:.0},cadence={:.0},options={:.0},cursor={:.0},toolbar={:.0}) wrap_extra={total_wrap_extra:.0} toolbar_wrapped={} max_inner={:.0}x{:.0} work_area={:?} scroll_limited={}",
            i18n::tag(lang),
            self.settings.stats_on,
            self.basic_stats_rows(),
            FULL_LAYOUT_FIXED_RESERVE * zoom,
            row_display * zoom,
            row_aspect * zoom,
            row_crop * zoom,
            row_cadence * zoom,
            row_options * zoom,
            row_cursor * zoom,
            row_toolbar * zoom,
            toolbar_wrap_extra > 0.0,
            max_inner_width,
            max_inner_height,
            work_area,
            desired_content_height > max_inner_height + 0.5
        );
        target
    }

    fn resource_meter(&mut self, ui: &mut egui::Ui) {
        let (cpu, gpu) = self.resource_monitor.sample();
        egui::Frame::new()
            .fill(egui::Color32::from_rgb(27, 28, 30))
            .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(65, 68, 73)))
            .corner_radius(egui::CornerRadius::same(5))
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                ui.set_min_height(64.0);
                let gap = ui.spacing().item_spacing.x;
                let bar_width = ((ui.available_width() - gap) * 0.5).max(1.0);
                ui.horizontal(|ui| {
                    load_bar(
                        ui,
                        bar_width,
                        "CPU",
                        cpu,
                        egui::Color32::from_rgb(75, 150, 215),
                    );
                    load_bar(
                        ui,
                        bar_width,
                        "GPU",
                        gpu.unwrap_or(0.0),
                        egui::Color32::from_rgb(92, 186, 125),
                    );
                });
                if gpu.is_none() {
                    ui.label(
                        egui::RichText::new("GPU counter unavailable")
                            .size(9.0)
                            .weak(),
                    );
                }
            });
    }

    fn display_mode_controls(&mut self, ui: &mut egui::Ui, lang: UiLanguage) -> bool {
        let mut changed = false;
        ink_centered_control_label(ui, tr(lang, "表示:", "Display:"));
        let mut mode = self.settings.scale_mode;
        if ink_centered_selectable_label(
            ui,
            mode == ScaleMode::Auto,
            tr(lang, "全画面", "Fullscreen"),
        )
        .clicked()
        {
            mode = ScaleMode::Auto;
        }
        if ink_centered_selectable_label(ui, mode == ScaleMode::Fixed, tr(lang, "倍率", "Windowed"))
            .clicked()
        {
            mode = ScaleMode::Fixed;
        }
        if mode != self.settings.scale_mode {
            self.settings.scale_mode = mode;
            changed = true;
        }

        let enabled = self.settings.scale_mode == ScaleMode::Fixed;
        ui.add_enabled_ui(enabled, |ui| {
            let edit = ui.add_sized(
                [52.0, 26.0],
                egui::TextEdit::singleline(&mut self.ratio_text)
                    .horizontal_align(egui::Align::Center)
                    .vertical_align(egui::Align::Center)
                    .margin(egui::Margin {
                        left: 4,
                        right: 4,
                        top: 3,
                        bottom: 1,
                    }),
            );
            if edit.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                if let Ok(value) = self.ratio_text.trim().trim_end_matches('x').parse::<f32>() {
                    let value = value.clamp(1.0, 6.0);
                    if (value - self.settings.ratio).abs() > 0.001 {
                        self.settings.ratio = value;
                        changed = true;
                    }
                }
                self.ratio_text = format!("{:.1}", self.settings.ratio);
            }
            egui::ComboBox::from_id_salt("ratio_preset")
                .width(30.0)
                .selected_text("")
                .show_ui(ui, |ui| {
                    for i in 5..=30 {
                        let ratio = i as f32 * 0.2;
                        if ui
                            .selectable_label(
                                (self.settings.ratio - ratio).abs() < 0.01,
                                format!("{ratio:.1}x"),
                            )
                            .clicked()
                        {
                            self.settings.ratio = ratio;
                            self.ratio_text = format!("{ratio:.1}");
                            changed = true;
                        }
                    }
                });
            ui.label(egui::RichText::new("x").size(11.0).weak());
        });
        changed
    }

    fn capture_resolution_controls(
        &mut self,
        ui: &mut egui::Ui,
        lang: UiLanguage,
        running: bool,
    ) -> bool {
        let mut changed = false;
        if self.settings.ui_mode == UiMode::Full {
            bounded_plain_label(
                ui,
                tr(lang, "キャプチャ解像度:", "Capture size:"),
                MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
            );
        } else {
            ui.label(tr(lang, "キャプチャ解像度:", "Capture size:"));
        }
        let capture_field_width = if self.settings.ui_mode == UiMode::Full {
            capture_resolution_field_width(ui, lang)
        } else {
            88.0
        };
        let edit = ui.add_sized(
            [capture_field_width, 26.0],
            egui::TextEdit::singleline(&mut self.capture_resolution_text)
                .horizontal_align(egui::Align::Center)
                .vertical_align(egui::Align::Center)
                .margin(egui::Margin {
                    left: 4,
                    right: 4,
                    top: 3,
                    bottom: 1,
                }),
        );
        let capture_enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
        if edit.lost_focus() || capture_enter {
            if let Some(parsed) = parse_capture_resolution(&self.capture_resolution_text) {
                if parsed != self.settings.capture_resolution {
                    self.settings.capture_resolution = parsed;
                    changed = true;
                    if running {
                        let _ = self.apply_capture_resolution_to_target();
                    }
                }
                self.capture_resolution_text =
                    capture_resolution_label(self.settings.capture_resolution, lang);
            }
        }
        egui::ComboBox::from_id_salt("capture_resolution_preset")
            .width(28.0)
            .selected_text("")
            .show_ui(ui, |ui| {
                for preset in capture_resolution_presets() {
                    let label = capture_resolution_label(preset, lang);
                    if ui
                        .selectable_label(self.settings.capture_resolution == preset, label)
                        .clicked()
                    {
                        self.settings.capture_resolution = preset;
                        self.capture_resolution_text =
                            capture_resolution_label(self.settings.capture_resolution, lang);
                        changed = true;
                        if running {
                            let _ = self.apply_capture_resolution_to_target();
                        }
                    }
                }
            })
            .response
            .on_hover_text(i18n::text(lang, "capture.size_help"));
        changed
    }

    fn running(&self) -> bool {
        self.engine.status.lock().unwrap().running
    }

    fn commit_toggle_hotkey(&mut self, candidate: &str, lang: UiLanguage) -> Result<(), String> {
        let canonical = validate_user_hotkey(candidate).map_err(|e| hotkey_error_text(lang, &e))?;
        if canonical == self.settings.hotkey_toggle {
            return Ok(());
        }
        if !hotkey_is_available(&canonical) {
            return Err(tr(
                lang,
                "この組み合わせはWindowsまたは別のアプリで使用されています。",
                "This shortcut is already used by Windows or another application.",
            )
            .to_string());
        }

        let old = self.settings.hotkey_toggle.clone();
        self.hotkeys.stop();
        let replacement =
            HotkeyThread::start(Self::hotkey_bindings(&canonical), self.engine.stop_handle());
        if replacement
            .registration_failures
            .iter()
            .any(|(id, _)| *id == HK_TOGGLE)
        {
            drop(replacement);
            self.hotkeys =
                HotkeyThread::start(Self::hotkey_bindings(&old), self.engine.stop_handle());
            return Err(tr(
                lang,
                "登録中に競合が発生しました。以前のショートカットへ戻しました。",
                "The shortcut became unavailable during registration. The previous shortcut was restored.",
            )
            .to_string());
        }
        self.hotkeys = replacement;
        self.settings.hotkey_toggle = canonical.clone();
        self.hotkey_editor_candidate = canonical.clone();
        save_settings(&self.app_dir, &self.settings);
        log::info!("toggle-hotkey-updated: old={old} new={canonical}");
        Ok(())
    }

    fn capture_resolution_plan(&self) -> Option<(CaptureResolution, CaptureResolution)> {
        let Some(res) = self.settings.capture_resolution else {
            return None;
        };
        if self.target_hwnd == 0 || !win32::is_window_valid(self.target_hwnd) {
            return None;
        }
        let title = win32::window_title(self.target_hwnd);
        let current_size = win32::client_rect_on_screen(self.target_hwnd)
            .map(|(_, _, w, h)| (w, h))
            .unwrap_or((res.w as i32, res.h as i32));
        // Capture Resolution is the raw/pre-crop capture canvas. Crop is a
        // downstream view into that fixed canvas and must never resize the
        // foreign source HWND. This keeps live Crop responsive and prevents
        // WGC transitional-frame stalls. PIP keeps its existing aspect-safe fit
        // against the raw client aspect.
        let capture_target = if is_picture_in_picture_title(&title) {
            fit_resolution_preserving_aspect(res, current_size)
        } else {
            res
        };
        if capture_target != res {
            log::info!(
                "capture-resolution PIP aspect-safe fit: hwnd={:#x} requested={}x{} current_capture={}x{} applied_capture={}x{}",
                self.target_hwnd,
                res.w,
                res.h,
                current_size.0,
                current_size.1,
                capture_target.w,
                capture_target.h
            );
        }
        if self.settings.capture_crop.enabled {
            log::debug!(
                "capture-resolution fixed-canvas plan: requested={}x{} applied={}x{} crop=({}, {}, {}, {}) crop_stage=post-capture",
                res.w,
                res.h,
                capture_target.w,
                capture_target.h,
                self.settings.capture_crop.left,
                self.settings.capture_crop.top,
                self.settings.capture_crop.right,
                self.settings.capture_crop.bottom
            );
        }
        Some((res, capture_target))
    }

    fn capture_resolution_disabled_for_target(&self) -> bool {
        if self.settings.capture_resolution.is_none()
            || self.target_hwnd == 0
            || !win32::is_window_valid(self.target_hwnd)
        {
            return false;
        }

        // A source that was genuinely fullscreen at capture start always owns
        // its native WGC geometry. For a running windowed-origin session, the
        // engine publishes a separate live fullscreen state when the source
        // application later switches itself to monitor-covering presentation.
        // This avoids misclassifying a Neo-initiated 1920x1080 resize on a
        // 1080p monitor as fullscreen.
        let status = self
            .engine
            .status
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone();
        let session_origin_fullscreen =
            status
                .source_recovery
                .as_ref()
                .and_then(|(hwnd, _, rect, _, _, _)| {
                    if *hwnd != self.target_hwnd {
                        return None;
                    }
                    rect.as_ref()
                        .copied()
                        .map(|r| win32::is_rect_monitor_fullscreen(*hwnd, r))
                });
        if status.running || status.starting {
            return status.source_live_fullscreen;
        }
        capture_resolution_fullscreen_guard(
            session_origin_fullscreen,
            win32::is_monitor_fullscreen(self.target_hwnd),
        )
    }

    fn apply_capture_resolution_to_target(&mut self) -> bool {
        if self.settings.capture_resolution.is_none() {
            if self.running() {
                self.engine.send(Cmd::ClearCaptureGeometry);
            }
            return true;
        }
        if self.capture_resolution_disabled_for_target() {
            self.capture_resolution_fullscreen_notice_open = true;
            log::warn!(
                "capture-resolution ignored while source is fullscreen: hwnd={:#x}; preference retained for windowed mode and WGC native geometry preserved",
                self.target_hwnd
            );
            return true;
        }
        let Some((res, applied)) = self.capture_resolution_plan() else {
            return false;
        };
        let publish_live_resize_intent = self.running();
        if publish_live_resize_intent {
            self.engine
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .capture_resolution_resize_intent = Some((applied.w, applied.h));
        }
        if win32::resize_client_area(self.target_hwnd, applied.w, applied.h) {
            // Coordinate the live source resize with the render thread. Merely
            // resizing the PIP from the GUI let queued old-size WGC frames and
            // a shape-specialized DirectML session survive the transition.
            self.engine.send(Cmd::SetCaptureGeometry {
                requested: (res.w, res.h),
                applied: (applied.w, applied.h),
                capture_crop: self.settings.capture_crop,
            });
            log::info!(
                "capture-resolution applied: hwnd={:#x} requested={}x{} client={}x{} engine_transition=armed",
                self.target_hwnd,
                res.w,
                res.h,
                applied.w,
                applied.h
            );
            true
        } else {
            if publish_live_resize_intent {
                self.engine
                    .status
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .capture_resolution_resize_intent = None;
            }
            let actual = win32::client_rect_on_screen(self.target_hwnd)
                .map(|(_, _, w, h)| format!("{w}x{h}"))
                .unwrap_or_else(|| "unavailable".to_string());
            log::warn!(
                "capture-resolution rejected: hwnd={:#x} requested={}x{} applied={}x{} actual={actual}",
                self.target_hwnd,
                res.w,
                res.h,
                applied.w,
                applied.h
            );
            false
        }
    }

    fn poll(&mut self, running: bool) {
        // Input exclusion geometry is latency-sensitive state, unlike the
        // slower housekeeping below. Publish the actual current Win32 GUI
        // rectangle on every egui update (~30fps while running). EngineHandle
        // coalesces it latest-only, so language/DPI/mode/layout changes and
        // window dragging cannot accumulate stale rectangles behind heavy GPU
        // work.
        let mut live_no_engage = Vec::new();
        if running
            && self.settings.gui_topmost
            && self.gui_hwnd != 0
            && win32::is_window_valid(self.gui_hwnd)
            && !win32::is_minimized(self.gui_hwnd)
        {
            if let Some(r) = win32::window_rect(self.gui_hwnd) {
                live_no_engage.push((r.0, r.1, r.2, r.3, self.gui_hwnd));
            }
        }
        self.engine.send(Cmd::SetNoEngage(live_no_engage));

        if !running {
            self.live_crop_pending = None;
            self.live_crop_last_sent = None;
            // If capture stops during the short Crop save debounce, do not lose
            // the user's final value. Idle editing also lands here on the next
            // GUI update and is persisted immediately.
            if self.crop_settings_save_due.take().is_some() {
                save_settings(&self.app_dir, &self.settings);
                log::debug!(
                    "crop-settings-save: idle/stop flush enabled={} edges=({}, {}, {}, {})",
                    self.settings.capture_crop.enabled,
                    self.settings.capture_crop.left,
                    self.settings.capture_crop.top,
                    self.settings.capture_crop.right,
                    self.settings.capture_crop.bottom
                );
            }
            self.tensorrt_crop_commit_pending = None;
            self.tensorrt_crop_commit_due = None;
        } else {
            // Keep ordinary DirectML/GLSL Crop visibly live without publishing
            // every 1 px DragValue sample. 33 ms is fast enough for a fluid
            // preview while cutting shape-change rebuild pressure substantially.
            if let Some(crop) = self.live_crop_pending {
                let pointer_down = win32::left_mouse_button_down();
                let due = !pointer_down
                    || self.settings.capture_resolution.is_none()
                    || self
                        .live_crop_last_sent
                        .map_or(true, |sent| sent.elapsed() >= Duration::from_millis(33));
                if due {
                    self.live_crop_pending = None;
                    self.live_crop_last_sent = Some(Instant::now());
                    self.engine.send(Cmd::SetCaptureCrop { crop });
                    log::debug!(
                        "crop-live-throttle: publish enabled={} edges=({}, {}, {}, {}) pointer_down={} cadence_ms=33",
                        crop.enabled,
                        crop.left,
                        crop.top,
                        crop.right,
                        crop.bottom,
                        pointer_down
                    );
                }
            }

            // Crop persistence is deliberately decoupled from render updates.
            // Saving settings.json and re-sending SetMode for every 1 px edit
            // adds avoidable GUI/disk/command traffic and does not affect Crop.
            if self
                .crop_settings_save_due
                .is_some_and(|due| !win32::left_mouse_button_down() && Instant::now() >= due)
            {
                self.crop_settings_save_due = None;
                save_settings(&self.app_dir, &self.settings);
                log::debug!(
                    "crop-settings-save: settled enabled={} edges=({}, {}, {}, {})",
                    self.settings.capture_crop.enabled,
                    self.settings.capture_crop.left,
                    self.settings.capture_crop.top,
                    self.settings.capture_crop.right,
                    self.settings.capture_crop.bottom
                );
            }

            // TensorRT Crop execution accepts saved geometry only. Keep the
            // existing transaction as a final safety net for the ON/OFF toggle
            // (and any future non-DragValue caller): the render thread still sees
            // only one settled Crop state. DirectML owns numeric Crop editing.
            if self.tensorrt_crop_commit_pending.is_some() {
                if win32::left_mouse_button_down() {
                    self.tensorrt_crop_commit_due = None;
                } else if self.tensorrt_crop_commit_due.is_none() {
                    self.tensorrt_crop_commit_due =
                        Some(Instant::now() + Duration::from_millis(220));
                    log::debug!("tensorrt-crop-transaction: mouse-up settle armed debounce_ms=220");
                } else if self
                    .tensorrt_crop_commit_due
                    .is_some_and(|due| Instant::now() >= due)
                {
                    let crop = self
                        .tensorrt_crop_commit_pending
                        .take()
                        .unwrap_or(self.settings.capture_crop);
                    self.tensorrt_crop_commit_due = None;
                    // Capture Resolution is a fixed pre-crop canvas, so a Crop
                    // commit must never resize the foreign source HWND. TensorRT
                    // still receives only the final shape to avoid engine-build churn.
                    self.engine.send(Cmd::SetCaptureCrop { crop });
                    log::info!(
                        "tensorrt-crop-transaction: commit final enabled={} edges=({}, {}, {}, {}) geometry_transaction=false fixed_capture_canvas=true",
                        crop.enabled,
                        crop.left,
                        crop.top,
                        crop.right,
                        crop.bottom
                    );
                }
            }
        }

        let poll_ms = if running { 80 } else { 250 };
        if self.last_poll.elapsed() < Duration::from_millis(poll_ms) {
            return;
        }
        self.last_poll = Instant::now();
        if self.qa_auto_start {
            self.qa_auto_start = false;
            log::info!("QA auto-start: target={:#x}", self.target_hwnd);
            self.start();
        }
        let status = self.engine.status.lock().unwrap().clone();
        if status.capture_resolution_fullscreen_notice_seq
            > self.capture_resolution_fullscreen_notice_seen_seq
        {
            self.capture_resolution_fullscreen_notice_seen_seq =
                status.capture_resolution_fullscreen_notice_seq;
            if self.settings.capture_resolution.is_some() {
                self.capture_resolution_fullscreen_notice_open = true;
            }
        }
        if status.capture_resolution_reapply_seq > self.capture_resolution_reapply_seen_seq {
            self.capture_resolution_reapply_seen_seq = status.capture_resolution_reapply_seq;
            if running {
                if self.settings.capture_resolution.is_some() {
                    let _ = self.apply_capture_resolution_to_target();
                } else {
                    self.engine.send(Cmd::ClearCaptureGeometry);
                }
            }
        }
        let gui_priority = (self.gui_hwnd, self.settings.gui_topmost);
        if self.gui_priority_sent != Some(gui_priority) {
            self.engine.send(Cmd::SetGuiPriority {
                hwnd: gui_priority.0,
                topmost: gui_priority.1,
            });
            self.gui_priority_sent = Some(gui_priority);
        }
        // Fail-visible retry for explicit TOPMOST-ON and capture-stop
        // transitions. Normally DWMWA_CLOAK=0 succeeds immediately, but never
        // leave the root GUI hidden if DWM transiently keeps the cloak bit set.
        // While capture is running with GUI TOPMOST still OFF, the retained
        // cloak is intentional and must not be released here.
        if self.gui_topmost_off_cloak_applied
            // A staged TOPMOST-OFF request intentionally cloaks the GUI while
            // settings.gui_topmost is still true until the anchor commit.
            // Never let the fail-visible ON/idle retry undo that transition
            // cloak in this narrow pending window.
            && !self.gui_topmost_off_pending
            && (self.settings.gui_topmost || !running)
            && self.gui_hwnd != 0
            && win32::is_window_valid(self.gui_hwnd)
        {
            if self.settings.gui_topmost {
                win32::set_own_topmost(self.gui_hwnd, true);
                win32::raise_topmost(self.gui_hwnd);
            }
            let reveal_ok = win32::set_window_cloaked(self.gui_hwnd, false);
            let still_cloaked = win32::is_cloaked(self.gui_hwnd);
            if !still_cloaked {
                self.gui_topmost_off_cloak_applied = false;
                log::debug!(
                    "GUI topmost cloak release retry: gui={:#x} request_ok={} running={} topmost={} result=revealed",
                    self.gui_hwnd,
                    reveal_ok,
                    running,
                    self.settings.gui_topmost
                );
            }
        }
        let mut ne = Vec::new();
        // Stable reference policy:
        // - GUI topmost ON: keep the GUI above the scaled view and mark it as
        //   a no-engage zone so it stays clickable.
        // - GUI topmost OFF: keep the GUI below the source while running so
        //   source input cannot be stolen by our own menu.
        if running
            && self.gui_hwnd != 0
            && win32::is_window_valid(self.gui_hwnd)
            && !win32::is_minimized(self.gui_hwnd)
        {
            if self.settings.gui_topmost {
                // Avoid reissuing HWND_TOPMOST every housekeeping tick. During
                // a native title-bar move that needless Z-order traffic can
                // fight Windows' move/resize modal loop. Correct the style only
                // when it is actually lost; sibling ordering is handled below.
                if !win32::is_topmost(self.gui_hwnd) {
                    win32::set_own_topmost(self.gui_hwnd, true);
                }
                // Converge all Neo-owned topmost siblings on one stable order.
                // Never independently raise GUI/panel/overlay: those competing
                // lifts caused DWM to alternate which surface was in front
                // while the magnified window was dragged across the GUI.
                win32::normalize_neo_topmost_stack(
                    self.gui_hwnd,
                    true,
                    self.panel_hwnd,
                    status.overlay_hwnd,
                );
                chidescaler_neo::input::keep_cursor_sprite_on_top();
                if let Some(r) = win32::window_rect(self.gui_hwnd) {
                    ne.push((r.0, r.1, r.2, r.3, self.gui_hwnd));
                }
            } else {
                // OFF: behave like an ordinary window. Do NOT place it below the
                // source — the engine keeps the source WS_EX_TOPMOST while
                // running, and inserting a non-topmost window just under a
                // topmost one lands it at the TOP of the normal-window band, so
                // the GUI ends up floating in front of every other app (the
                // stale-topmost condition). Clear topmost only when needed and let normal
                // z-order rule.
                if win32::is_topmost(self.gui_hwnd) {
                    win32::set_own_topmost(self.gui_hwnd, false);
                }
            }
        }
        self.engine.send(Cmd::SetNoEngage(ne));
        // Pin the selected source for the whole session. Previously an
        // unrelated foreground window (most often Task Manager) silently
        // replaced target_hwnd while capture was running, so the next Start
        // targeted that window and appeared to have lost its filter.
        let foreground = win32::foreground_window();
        let own_elevated = win32::own_process_elevated();
        let candidate_elevated = win32::process_elevated(win32::window_pid(foreground));
        if may_follow_foreground_target(running, own_elevated, candidate_elevated) {
            let fg = win32::normalize_capture_target(foreground);
            if fg != 0
                && !win32::is_own_window(fg)
                && win32::is_window_valid(fg)
                && !win32::is_minimized(fg)
                && !win32::is_system_window(fg)
            {
                let title = win32::window_title(fg);
                if !title.is_empty() {
                    self.target_hwnd = fg;
                    self.target_title = title;
                }
            }
        }
    }

    fn start(&mut self) {
        if self.target_hwnd == 0 {
            return;
        }
        // A large owned presentation helper may become foreground after the
        // user clicks the video surface. Keep the user's logical source on the
        // structural root host; do not let an internal presentation HWND turn
        // into an independent fullscreen/windowed target.
        let normalized_target = win32::normalize_capture_target(self.target_hwnd);
        if normalized_target != self.target_hwnd {
            log::info!(
                "capture-start-target-normalized: selected={:#x} root={:#x} reason=owned-presentation-surface",
                self.target_hwnd,
                normalized_target
            );
            self.target_hwnd = normalized_target;
            self.target_title = win32::window_title(normalized_target);
        }
        if !win32::own_process_elevated()
            && win32::process_elevated(win32::window_pid(self.target_hwnd)) == Some(true)
        {
            let title = win32::window_title(self.target_hwnd);
            log::warn!(
                "elevated-target-start-blocked: hwnd={:#x} title='{}' action=show-admin-guidance",
                self.target_hwnd,
                title
            );
            self.elevated_target_notice = Some((self.target_hwnd, title));
            return;
        }
        // Snapshot before the optional capture-resolution resize. Stop must
        // restore the exact window the user selected, even after in-session
        // moves, resizes, fullscreen toggles, or an off-screen displacement.
        let source_restore_rect = win32::window_rect(self.target_hwnd);
        let source_restore_placement = win32::window_placement_snapshot(self.target_hwnd);
        let source_was_maximized = win32::is_maximized(self.target_hwnd);
        let source_pid = win32::window_pid(self.target_hwnd);
        if source_pid == 0 {
            let lang = self.effective_language();
            self.engine.status.lock().unwrap().last_error = Some(
                tr(
                    lang,
                    "キャプチャ対象が見つかりません。",
                    "The capture target is no longer available.",
                )
                .to_string(),
            );
            return;
        }
        let source_was_topmost = win32::is_topmost(self.target_hwnd);
        let source_was_layered =
            win32::source_window_layered(self.target_hwnd, source_pid).unwrap_or(false);
        let source_corner_preference = win32::window_corner_preference(self.target_hwnd);
        log::debug!(
            "capture-start-window-placement: hwnd={:#x} outer={source_restore_rect:?} normal={:?} maximized={source_was_maximized}",
            self.target_hwnd,
            source_restore_placement.map(|p| p.normal_rect_xywh())
        );
        // A monitor-covering browser video keeps its input coordinate system at
        // native fullscreen geometry. Resizing that HWND makes cursor mapping
        // diverge, so this session deliberately uses the source resolution.
        let capture_resolution_disabled = self.capture_resolution_disabled_for_target();
        if capture_resolution_disabled {
            log::warn!(
                "capture-resolution disabled for this fullscreen session: hwnd={:#x}; source resolution and cursor coordinates preserved",
                self.target_hwnd
            );
            if self.settings.capture_resolution.is_some() {
                self.capture_resolution_fullscreen_notice_open = true;
            }
        }
        // Resolve the PIP aspect-safe geometry once from the original source.
        // The applied size, not the nominal menu limit, is the authoritative
        // capture/input canvas for this session. Recomputing from a source that
        // Neo already resized can otherwise rebase the coordinate system.
        let capture_resolution_plan = if capture_resolution_disabled {
            None
        } else {
            self.capture_resolution_plan()
        };
        let deferred_capture_resolution = if self.settings.hide_source {
            capture_resolution_plan
                .map(|(requested, applied)| ((requested.w, requested.h), (applied.w, applied.h)))
        } else {
            None
        };
        let capture_resolution_ready = if self.settings.hide_source {
            self.settings.capture_resolution.is_none()
                || capture_resolution_disabled
                || capture_resolution_plan.is_some()
        } else {
            self.apply_capture_resolution_to_target()
        };
        let capture_canvas = capture_resolution_plan.map(|(_, applied)| (applied.w, applied.h));
        if !capture_resolution_ready {
            let restored = win32::restore_window_origin(
                self.target_hwnd,
                source_restore_rect,
                source_was_maximized,
                source_restore_placement,
            );
            log::warn!(
                "capture-resolution abort restore: hwnd={:#x} rect={source_restore_rect:?} normal_rect={:?} maximized={} ok={restored}",
                self.target_hwnd,
                source_restore_placement.map(|p| p.normal_rect_xywh()),
                source_was_maximized
            );
            let lang = self.effective_language();
            self.engine.status.lock().unwrap().last_error = Some(
                tr(
                    lang,
                    "指定したキャプチャ解像度をソースウィンドウが維持できません。キャプチャ解像度を「自動」にするか、対応するサイズを選んでください。キャプチャは開始しませんでした。",
                    "The source window cannot keep the requested capture resolution. Select Auto or a supported size. Capture was not started.",
                )
                .to_string(),
            );
            return;
        }
        self.panel_chip_lurking = false;
        self.panel_bar_shown = true;
        self.engine.send(Cmd::SetInputOpts {
            autohide_secs: if self.settings.cursor_autohide {
                self.settings.cursor_autohide_secs.max(0.5)
            } else {
                0.0
            },
            speed_fix: self.settings.cursor_speed_fix,
        });
        self.engine
            .send(Cmd::SetInterpFactor(self.settings.interp_factor));
        self.engine
            .send(Cmd::SetDownscaler(self.settings.downscaler.clone()));
        self.engine.send(Cmd::SetVsync(self.settings.vsync));
        self.engine
            .send(Cmd::SetSmoothPacing(self.settings.smooth_pacing));
        self.engine.send(Cmd::SetDuplicateFrameReduction(
            self.settings.duplicate_frame_reduction,
        ));
        // Auto keeps ONNX on the actual WGL adapter for the fastest GPU-direct
        // path. An explicit user choice remains authoritative for ONNX even if
        // Windows/driver did not move WGL: hybrid systems may still gain far
        // more from dGPU inference than they lose to the existing cross-GPU
        // transfer fallback. Never silently force an explicit dGPU request back
        // onto an iGPU merely to preserve sharing.
        let render_gpu_luid = self.engine.status.lock().unwrap().render_gpu_luid;
        let requested_gpu_luid = self.settings.gpu_adapter_luid;
        let effective_gpu_luid = resolve_compute_gpu_luid(requested_gpu_luid, render_gpu_luid);
        if cross_gpu_compute_active(requested_gpu_luid, render_gpu_luid) {
            let requested = requested_gpu_luid.unwrap();
            let actual = render_gpu_luid.unwrap();
            log::warn!(
                "gpu-selection-cross-gpu: render_luid={actual:016x} onnx_luid={requested:016x} mode=explicit-compute-override gpu_direct_sharing=disabled-or-fallback reason=user-selected-gpu"
            );
        }
        let gpu_adapter = gpu::device_id_for_luid(&self.gpu_adapters, effective_gpu_luid);
        let gpu_name = gpu::adapter_for_luid(&self.gpu_adapters, effective_gpu_luid)
            .map(|adapter| adapter.name.as_str())
            .unwrap_or("Auto");
        log::info!(
            "capture-start-request: hwnd={:#x} title='{}' filters={} gpu_requested_luid={} gpu_force_vulkan={} gpu_effective_luid={} gpu_device_id={:?} gpu_name='{}'",
            self.target_hwnd,
            self.target_title,
            self.chain.iter().filter(|stage| stage.enabled).count(),
            requested_gpu_luid
                .map(|luid| format!("{luid:016x}"))
                .unwrap_or_else(|| "Auto".to_string()),
            self.settings.gpu_force_vulkan,
            effective_gpu_luid
                .map(|luid| format!("{luid:016x}"))
                .unwrap_or_else(|| "Auto".to_string()),
            gpu_adapter,
            gpu_name
        );
        log::info!(
            "privilege-diag: neo_elevated={} source_pid={} source_elevated={:?}",
            win32::own_process_elevated(),
            win32::window_pid(self.target_hwnd),
            win32::process_elevated(win32::window_pid(self.target_hwnd))
        );
        // An explicit capture resolution describes the retained content area.
        // With user crop enabled Neo enlarges the source client by those margins
        // so the post-crop pixels equal the selected size. Client-only WGC is
        // still mandatory so decorations cannot enter that geometry.
        let effective_client_only =
            self.settings.client_only || self.settings.capture_resolution.is_some();
        if effective_client_only != self.settings.client_only {
            log::info!("capture-resolution: forcing client-area WGC crop");
        }
        chidescaler_neo::input::publish_janitor_source_recovery(
            self.target_hwnd,
            source_pid,
            source_restore_rect,
            source_restore_placement,
            source_was_maximized,
            source_was_topmost,
            source_was_layered,
            source_corner_preference,
        );
        self.engine.send(Cmd::Start {
            hwnd: self.target_hwnd,
            source_pid,
            source_was_topmost,
            specs: self.chain.clone(),
            mode: self.settings.scale_mode,
            ratio: self.settings.ratio,
            aspect_correction: self.settings.aspect_correction,
            aspect_correction_mode: self.settings.aspect_correction_mode,
            aspect_width_scale: sanitize_aspect_correction_scale(self.settings.aspect_width_scale),
            aspect_height_scale: sanitize_aspect_correction_scale(
                self.settings.aspect_height_scale,
            ),
            capture_crop: self.settings.capture_crop,
            fps_cap: self
                .settings
                .fps_cap_enabled
                .then_some(self.settings.fps_cap),
            hide_source: self.settings.hide_source,
            client_only: effective_client_only,
            hdr: hdr_capture_requested(&self.settings),
            hdr_sdr_mode: self.settings.hdr_sdr_mode,
            gpu_adapter,
            // Physical GPU selection and GLSL backend policy are independent.
            // A normal explicit selection keeps same-GPU OpenGL, cross-GPU
            // selection uses Vulkan automatically, and a [Vulkan] row forces
            // compatible GLSL through Vulkan on that selected adapter.
            explicit_gpu_luid: requested_gpu_luid,
            force_vulkan_glsl: self.settings.gpu_force_vulkan,
            source_restore_rect,
            source_restore_placement,
            source_was_maximized,
            deferred_capture_resolution,
            capture_canvas,
        });
    }

    fn should_dispatch_panel_stop(
        starting: bool,
        running: bool,
        stopping: bool,
        provider_preparing: bool,
    ) -> bool {
        !stopping && (starting || running || provider_preparing)
    }

    fn capture_busy_now(&self) -> (bool, bool, bool, bool) {
        let status = self.engine.status.lock().unwrap().clone();
        let provider_preparing = chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
        (
            status.starting,
            status.running,
            status.stopping,
            provider_preparing,
        )
    }

    fn dispatch_capture_stop_known(
        &mut self,
        source: &str,
        starting: bool,
        running: bool,
        stopping: bool,
        provider_preparing: bool,
    ) -> bool {
        if stopping {
            log::info!(
                "capture-stop-request: source={source} result=ignored-already-stopping starting={starting} running={running} preparing={provider_preparing}"
            );
            return false;
        }
        if starting || running || provider_preparing {
            log::info!(
                "capture-stop-request: source={source} result=dispatched starting={starting} running={running} preparing={provider_preparing}"
            );
            self.engine.send(Cmd::Stop);
            true
        } else {
            log::info!(
                "capture-stop-request: source={source} result=ignored-idle starting={starting} running={running} preparing={provider_preparing}"
            );
            false
        }
    }

    fn request_capture_stop(&mut self, source: &str) -> bool {
        let (starting, running, stopping, provider_preparing) = self.capture_busy_now();
        self.dispatch_capture_stop_known(source, starting, running, stopping, provider_preparing)
    }

    fn request_capture_start(&mut self, source: &str) -> bool {
        let (starting, running, stopping, provider_preparing) = self.capture_busy_now();
        if starting || running || stopping || provider_preparing {
            log::info!(
                "capture-start-request-ignored: source={source} starting={starting} running={running} stopping={stopping} preparing={provider_preparing}"
            );
            false
        } else {
            log::info!("capture-start-dispatch: source={source}");
            self.start();
            true
        }
    }

    fn dispatch_toggle_hotkey(&mut self, event: &HotkeyEvent) {
        let age_ms = event.received_at.elapsed().as_secs_f64() * 1000.0;
        let (starting, running, stopping, provider_preparing) = self.capture_busy_now();
        if starting || running || stopping || provider_preparing {
            log::info!(
                "hotkey-toggle-dispatch: binding='{}' action=stop age_ms={:.1} starting={} running={} stopping={} preparing={}",
                event.binding,
                age_ms,
                starting,
                running,
                stopping,
                provider_preparing
            );
            let _ = self.request_capture_stop("hotkey-toggle");
        } else if event.received_at <= self.capture_idle_since {
            log::info!(
                "hotkey-toggle-stale-ignored: binding='{}' age_ms={:.1} reason=received-before-current-idle-epoch",
                event.binding,
                age_ms
            );
        } else {
            log::info!(
                "hotkey-toggle-dispatch: binding='{}' action=start age_ms={:.1}",
                event.binding,
                age_ms
            );
            let _ = self.request_capture_start("hotkey-toggle");
        }
    }

    fn apply_live(&self) {
        if self.running() {
            self.engine.send(Cmd::ApplyChain {
                specs: self.chain.clone(),
                aspect_correction: self.settings.aspect_correction,
                aspect_correction_mode: self.settings.aspect_correction_mode,
                aspect_width_scale: sanitize_aspect_correction_scale(
                    self.settings.aspect_width_scale,
                ),
                aspect_height_scale: sanitize_aspect_correction_scale(
                    self.settings.aspect_height_scale,
                ),
                capture_crop: self.settings.capture_crop,
            });
        }
    }

    fn refresh_filters(&mut self, reason: &str) {
        let previous = self.available.len();
        let discovered = discover_filters(&self.app_dir);
        let changed = discovered != self.available;
        log::info!(
            "filter-scan: reason={reason} found={} previous={} changed={changed}",
            discovered.len(),
            previous
        );
        self.available = discovered;
    }

    fn add_filter_to_chain(&mut self, kind: StageKind, path: String) {
        let before = self.chain.len();
        let index = append_filter_stage(&mut self.chain, kind, path.clone());
        log::info!(
            "filter-chain-add: index={index} kind={kind:?} path={path} before={before} after={} running={}",
            self.chain.len(),
            self.running()
        );
        self.apply_live();
    }

    fn select_preset(&mut self, name: &str) {
        if let Some(preset) = self.store.data.presets.iter().find(|p| p.name == name) {
            let chain = preset.chain.clone();
            let aspect_correction = preset.effective_aspect_correction();
            let crop = preset.effective_crop();
            let preset_capture_resolution = preset.capture_resolution;
            let capture_resolution_was_specified = preset_capture_resolution.is_some();
            // v672: a preset only owns the capture resolution when it actually
            // stores a fixed value. Missing metadata inherits the current GUI value
            // instead of forcing Auto, which keeps comparison runs at one resolution.
            let selected_capture_resolution =
                preset_capture_resolution.or(self.settings.capture_resolution);
            self.chain = chain
                .iter()
                .filter(|stage| !is_frozen_neoflow_stage(stage))
                .cloned()
                .collect();
            self.saved_chain = chain;
            aspect_correction.apply_to_settings(&mut self.settings);
            self.settings.capture_crop = crop;
            self.settings.capture_resolution = selected_capture_resolution;
            let lang = self.effective_language();
            self.capture_resolution_text =
                capture_resolution_label(self.settings.capture_resolution, lang);
            self.saved_aspect_correction = aspect_correction;
            self.saved_crop = crop;
            // For an unspecified preset, the inherited GUI value becomes the
            // comparison baseline. Therefore selecting it does not immediately
            // create a false dirty '*' marker; later user changes still do.
            self.saved_capture_resolution = selected_capture_resolution;
            self.store.data.active = name.to_string();
            self.store.save();
            save_settings(&self.app_dir, &self.settings);
            let running = self.running();
            // Preserve the established preset chain/aspect/crop transaction first.
            // Only a preset with an explicit fixed resolution is allowed to invoke
            // the live capture-resolution state machine; unspecified presets leave
            // the source geometry and GUI selector untouched.
            self.apply_live();
            if running && capture_resolution_was_specified {
                let _ = self.apply_capture_resolution_to_target();
            }
            log::info!(
                "preset-state-load: source=selection preset='{}' aspect_enabled={} aspect_mode={:?} aspect_scale={:.2}x{:.2} crop_enabled={} crop=({}, {}, {}, {}) capture_resolution_saved={} capture_resolution_effective={} inherited={}",
                name,
                aspect_correction.enabled,
                aspect_correction.mode,
                aspect_correction.width_scale,
                aspect_correction.height_scale,
                crop.enabled,
                crop.left,
                crop.top,
                crop.right,
                crop.bottom,
                preset_capture_resolution
                    .map(|r| format!("{}x{}", r.w, r.h))
                    .unwrap_or_else(|| "unspecified".to_string()),
                selected_capture_resolution
                    .map(|r| format!("{}x{}", r.w, r.h))
                    .unwrap_or_else(|| "Auto".to_string()),
                !capture_resolution_was_specified
            );
        }
    }

    fn current_preset_aspect_correction(&self) -> PresetAspectCorrection {
        PresetAspectCorrection::from_settings(&self.settings)
    }

    fn current_preset_crop(&self) -> CaptureCrop {
        self.settings.capture_crop
    }

    fn current_preset_capture_resolution(&self) -> Option<CaptureResolution> {
        self.settings.capture_resolution
    }

    fn dirty(&self) -> bool {
        self.chain != self.saved_chain
            || self.current_preset_aspect_correction() != self.saved_aspect_correction
            || self.current_preset_crop() != self.saved_crop
            || self.current_preset_capture_resolution() != self.saved_capture_resolution
    }

    /// soft, rounded little move button ("∧" / "∨" look)
    fn soft_button(ui: &mut egui::Ui, glyph: &str, enabled: bool) -> egui::Response {
        let text = egui::RichText::new(glyph)
            .size(11.0)
            .color(egui::Color32::from_gray(200));
        ui.add_enabled(
            enabled,
            egui::Button::new(text)
                .min_size(egui::vec2(26.0, 22.0))
                .corner_radius(egui::CornerRadius::same(8)),
        )
    }

    fn apply_panel_action(&mut self, collapse: bool, expand: bool) {
        apply_panel_action_state(
            &mut self.panel_chip_lurking,
            &mut self.panel_bar_shown,
            &mut self.panel_leave_at,
            collapse,
            expand,
        );
    }

    fn toggle_panel_hotkey(&mut self, running: bool, ctx: &egui::Context) {
        let now = Instant::now();
        if self
            .last_panel_hotkey
            .is_some_and(|t| now.duration_since(t) < Duration::from_millis(180))
        {
            log::info!("panel hotkey ignored: debounce");
            return;
        }
        self.last_panel_hotkey = Some(now);
        self.panel_visible = !self.panel_visible;
        self.panel_chip_lurking = false;
        self.panel_bar_shown = true;
        self.panel_leave_at = None;
        self.panel_state_sent = None;
        self.panel_layout_sent = None;
        // Commit native panel hit-testing/visibility immediately in the hotkey
        // frame. The v465 panel has no WGPU child surface to keep alive.
        if self.panel_hwnd != 0
            && win32::is_window_valid(self.panel_hwnd)
            && win32::is_own_window(self.panel_hwnd)
        {
            if self.panel_visible && running && self.settings.panel_show {
                win32::set_window_input_passthrough(self.panel_hwnd, false);
                win32::set_window_opaque_unlayered(self.panel_hwnd);
                // Reveal after control_panel has refreshed the GDI pixels.
            } else {
                win32::set_window_alpha(self.panel_hwnd, 0);
                win32::set_window_input_passthrough(self.panel_hwnd, true);
                win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
            }
        }
        if self.panel_visible {
            if self.panel_hwnd != 0
                && (!win32::is_window_valid(self.panel_hwnd)
                    || !win32::is_own_window(self.panel_hwnd))
            {
                self.panel_hwnd = 0;
                self.panel_layout_sent = None;
            }
            // Z-order is normalized by the single shared hierarchy manager on
            // the next GUI/engine tick. Do not lift the panel independently.
            ctx.request_repaint_after(Duration::from_millis(16));
        }
        // Keep the engine's panel visibility metadata in sync with the native
        // GDI host. No eframe child viewport or secondary GPU surface exists.
        self.engine.send(Cmd::SetPanelState {
            visible: running && self.panel_visible && self.settings.panel_show,
            chip: false,
        });
        log::info!(
            "panel hotkey: visible={} running={}",
            self.panel_visible,
            running
        );
        ctx.request_repaint();
    }

    fn commit_gui_topmost_off(&mut self, reason: &str) {
        self.gui_topmost_off_pending = false;
        self.settings.gui_topmost = false;
        if self.gui_hwnd != 0 && win32::is_window_valid(self.gui_hwnd) {
            win32::set_own_topmost(self.gui_hwnd, false);

            // v538: keep the GUI cloaked until the entire TOPMOST helper stack
            // has been recommitted *after* the GUI demotion. The first panel
            // GUI-button click may have to create the v459 keep-alive anchor,
            // and that first Present can take tens of milliseconds. Releasing
            // the cloak after only the GUI demotion allowed DWM to expose the
            // newly-uncloaked root for one composition frame before the
            // overlay/anchor/panel order settled: disappear -> flash -> hide.
            //
            // Reuse the existing GUI-TOPMOST-OFF recommit path and preserve the
            // established sibling order: overlay < anchor < panel < cursor.
            // This runs only for the already-cloaked OFF transition; normal
            // capture, panel, cursor and render paths are untouched.
            if self.gui_topmost_off_cloak_applied {
                let status = self
                    .engine
                    .status
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                let overlay = status.overlay_hwnd;
                if overlay != 0 && win32::is_window_valid(overlay) && win32::is_own_window(overlay)
                {
                    // First recommit the fullscreen overlay into TOPMOST without
                    // lifting the panel yet. Then rebuild the helper stack in
                    // deterministic bottom-to-top order.
                    win32::recommit_overlay_below_helpers(0, overlay);

                    let anchor = self.compositor_anchor_hwnd;
                    if anchor != 0 && win32::is_window_valid(anchor) && win32::is_own_window(anchor)
                    {
                        win32::raise_topmost(anchor);
                    }
                    if self.panel_hwnd != 0
                        && win32::is_window_valid(self.panel_hwnd)
                        && win32::is_own_window(self.panel_hwnd)
                        && win32::is_window_visible(self.panel_hwnd)
                    {
                        win32::raise_topmost(self.panel_hwnd);
                    }
                    chidescaler_neo::input::keep_cursor_sprite_on_top();

                    log::debug!(
                        "GUI topmost off pre-uncloak restack: gui={:#x} overlay={:#x} anchor={:#x} panel={:#x} anchor_above_overlay={} panel_above_anchor={}",
                        self.gui_hwnd,
                        overlay,
                        self.compositor_anchor_hwnd,
                        self.panel_hwnd,
                        self.compositor_anchor_hwnd == 0
                            || win32::window_is_above(self.compositor_anchor_hwnd, overlay),
                        self.compositor_anchor_hwnd == 0
                            || self.panel_hwnd == 0
                            || win32::window_is_above(self.panel_hwnd, self.compositor_anchor_hwnd),
                    );
                }

                // v539: while capture is still running, do NOT uncloak the
                // ordinary GUI after demotion. A DWM-uncloaked normal window can
                // still be sampled for one composition frame even when the
                // overlay/helper sibling order is already correct. That is the
                // remaining disappear -> reappear -> disappear symptom seen in
                // v538. Keeping the GUI cloaked for the whole TOPMOST-OFF state
                // makes the visual contract deterministic: the first click
                // cloaks it once, and it cannot re-enter composition until the
                // user explicitly turns GUI TOPMOST back on or capture stops.
                let keep_cloaked = status.running
                    && !status.stopping
                    && overlay != 0
                    && win32::is_window_valid(overlay)
                    && win32::is_own_window(overlay);
                win32::sync_panel_composition_with_dwm();
                if keep_cloaked {
                    log::debug!(
                        "GUI topmost off cloak retained: gui={:#x} reason={} until=topmost-on-or-capture-stop",
                        self.gui_hwnd,
                        reason
                    );
                } else {
                    let reveal_ok = win32::set_window_cloaked(self.gui_hwnd, false);
                    let still_cloaked = win32::is_cloaked(self.gui_hwnd);
                    self.gui_topmost_off_cloak_applied = still_cloaked;
                    log::debug!(
                        "GUI topmost off cloak release: gui={:#x} ok={} still_cloaked={} reason={} capture_active=false",
                        self.gui_hwnd,
                        reveal_ok,
                        still_cloaked,
                        reason
                    );
                }
            }
        }
        self.gui_topmost_applied = Some(false);
        self.gui_priority_sent = None;
        save_settings(&self.app_dir, &self.settings);
        log::info!("GUI topmost toggled: false transition={reason}");
    }

    fn toggle_gui_topmost(&mut self) {
        // Persistent always-on-top is independent from GUI minimize/restore.
        // Only the dedicated GUI setting/hotkey may change this preference.
        //
        // While capture is running, turning TOPMOST off is staged until the
        // v459-compatible WGPU composition anchor is already alive underneath
        // the panel. Demoting the root GUI first and creating the anchor
        // afterwards caused the occasional one-frame "blink, then disappear"
        // transition on AMD. TOPMOST on remains immediate.
        if self.gui_topmost_off_pending {
            // A second toggle before the staged OFF commits means the user
            // wants to stay ON. Nothing native has been demoted yet. Restore
            // only the cloak owned by this transition; do not disturb any
            // unrelated GUI transition guard.
            self.gui_topmost_off_pending = false;
            if self.gui_topmost_off_cloak_applied
                && self.gui_hwnd != 0
                && win32::is_window_valid(self.gui_hwnd)
            {
                let reveal_ok = win32::set_window_cloaked(self.gui_hwnd, false);
                win32::raise_topmost(self.gui_hwnd);
                log::debug!(
                    "GUI topmost off cloak cancelled: gui={:#x} ok={}",
                    self.gui_hwnd,
                    reveal_ok
                );
            }
            self.gui_topmost_off_cloak_applied = false;
            log::info!("GUI topmost off staging cancelled: requested_state=true");
            return;
        }

        if self.settings.gui_topmost {
            let status = self
                .engine
                .status
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone();
            let anchor_needed = status.running
                && !status.stopping
                && status.overlay_hwnd != 0
                && win32::is_window_valid(status.overlay_hwnd)
                && self.panel_visible
                && self.settings.panel_show
                && self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_own_window(self.panel_hwnd)
                && win32::is_window_visible(self.panel_hwnd);

            if anchor_needed {
                // v525: make the old TOPMOST GUI disappear from DWM *before*
                // constructing the first keep-alive WGPU surface. On affected
                // systems that first Present can take tens of milliseconds;
                // leaving the GUI visible until afterwards creates the visible
                // one-frame flash reported when the panel GUI button is turned
                // OFF. Cloaking changes composition visibility only; the HWND,
                // input state and TOPMOST bit remain untouched until commit.
                self.gui_topmost_off_cloak_applied = self.gui_hwnd != 0
                    && win32::is_window_valid(self.gui_hwnd)
                    && win32::set_window_cloaked(self.gui_hwnd, true);
                if self.gui_topmost_off_cloak_applied {
                    win32::sync_panel_composition_with_dwm();
                }
                self.gui_topmost_off_pending = true;
                log::debug!(
                    "GUI topmost off staged: gui={:#x} panel={:#x} overlay={:#x} action=cloak-before-anchor cloak_applied={}",
                    self.gui_hwnd,
                    self.panel_hwnd,
                    status.overlay_hwnd,
                    self.gui_topmost_off_cloak_applied
                );
                return;
            }

            self.commit_gui_topmost_off("immediate-anchor-unneeded");
            return;
        }

        self.settings.gui_topmost = true;
        if self.gui_hwnd != 0 && win32::is_window_valid(self.gui_hwnd) {
            // If TOPMOST-OFF owns a persistent DWM cloak, rebuild the visible
            // TOPMOST relationship *before* releasing it. The GUI therefore
            // returns exactly once in the requested ON state instead of being
            // briefly composed as an ordinary window underneath the overlay.
            win32::set_own_topmost(self.gui_hwnd, true);
            win32::raise_topmost(self.gui_hwnd);
            if self.gui_topmost_off_cloak_applied {
                let status = self
                    .engine
                    .status
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .clone();
                if status.overlay_hwnd != 0
                    && win32::is_window_valid(status.overlay_hwnd)
                    && win32::is_own_window(status.overlay_hwnd)
                {
                    win32::normalize_neo_topmost_stack(
                        self.gui_hwnd,
                        true,
                        self.panel_hwnd,
                        status.overlay_hwnd,
                    );
                    chidescaler_neo::input::keep_cursor_sprite_on_top();
                }
                win32::sync_panel_composition_with_dwm();
                let reveal_ok = win32::set_window_cloaked(self.gui_hwnd, false);
                let still_cloaked = win32::is_cloaked(self.gui_hwnd);
                self.gui_topmost_off_cloak_applied = still_cloaked;
                if still_cloaked {
                    log::warn!(
                        "GUI topmost on cloak release pending: gui={:#x} request_ok={} still_cloaked=true",
                        self.gui_hwnd,
                        reveal_ok
                    );
                } else {
                    log::debug!(
                        "GUI topmost on cloak release: gui={:#x} request_ok={} order=topmost-before-uncloak",
                        self.gui_hwnd,
                        reveal_ok
                    );
                }
            }
            win32::activate_window(self.gui_hwnd);
            win32::raise_topmost(self.gui_hwnd);
        }
        self.gui_topmost_applied = Some(true);
        self.gui_priority_sent = None;
        save_settings(&self.app_dir, &self.settings);
        log::info!("GUI topmost toggled: true");
    }

    // ------------- control panel (old cHiDeScaler port) -------------
    // The panel is a native GDI window; the engine owns its position and moves
    // it every tick in lockstep with the overlay. The main WGPU GUI never has
    // to repaint a second panel swapchain.
    fn control_panel(&mut self, ctx: &egui::Context, status: &Status) {
        let lang = self.effective_language();
        let show =
            status.running && !status.stopping && self.panel_visible && self.settings.panel_show;
        let effective_show = show && self.panel_placed_for_run;
        // v465 has no hidden WGPU keep-alive panel. Physical visibility follows
        // the logical/native GDI panel visibility exactly.
        let physical_show = effective_show;
        let lurk = self.panel_chip_lurking && !self.panel_bar_shown;
        if self.panel_state_sent != Some((effective_show, lurk)) {
            if self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_own_window(self.panel_hwnd)
            {
                // Apply the native host's alpha/input state immediately. There
                // is no child GPU surface or vendor-specific keep-alive path.
                if effective_show {
                    win32::set_window_input_passthrough(self.panel_hwnd, false);
                    if lurk {
                        win32::set_window_alpha(self.panel_hwnd, PANEL_LURK_ALPHA);
                    } else {
                        win32::set_window_opaque_unlayered(self.panel_hwnd);
                    }
                } else {
                    win32::set_window_alpha(self.panel_hwnd, 0);
                    win32::set_window_input_passthrough(self.panel_hwnd, true);
                    if win32::is_panel_gdi_host(self.panel_hwnd) {
                        win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
                    }
                }
            }
            self.engine.send(Cmd::SetPanelState {
                visible: effective_show,
                chip: lurk,
            });
            self.panel_state_sent = Some((effective_show, lurk));
        }
        if self.panel_metrics_sent != Some(show) {
            self.engine.metrics.set_panel_enabled(show);
            self.panel_metrics_sent = Some(show);
        }
        if !status.running {
            win32::hide_panel_gdi_mirror();
            if self.panel_hwnd != 0 && win32::is_panel_gdi_host(self.panel_hwnd) {
                // Ownership is a running-session ordering aid only. Clear it
                // after Stop so the panel remains logically independent from
                // both GUI and persistent overlay while idle.
                win32::set_panel_overlay_owner(self.panel_hwnd, status.overlay_hwnd, false);
                win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
            }
            self.panel_diag_last_frame_at = None;
            self.panel_diag_last_snapshot = None;
            return;
        }
        let panel_diag_started = Instant::now();
        let panel_diag_gap_ms = self
            .panel_diag_last_frame_at
            .replace(panel_diag_started)
            .map(|previous| panel_diag_started.duration_since(previous).as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        self.panel_diag_seq = self.panel_diag_seq.wrapping_add(1);
        let panel_diag_seq = self.panel_diag_seq;
        let present_fps = self.engine.metrics.snapshot().present_fps;
        let z = ctx.zoom_factor();
        let nppp = ctx
            .input(|i| i.viewport().native_pixels_per_point)
            .unwrap_or(1.0)
            .max(0.5);
        // Floating bar: [stop][present fps][screenshot][GUI visibility][collapse].
        let (bar_w, bar_h) = (PANEL_BAR_W_PTS, PANEL_BAR_H_PTS); // egui pts
        let (chip_w, chip_h) = (PANEL_CHIP_W_PTS, PANEL_CHIP_H_PTS);
        let bar_px = (
            (bar_w * z * nppp).ceil() as i32,
            (bar_h * z * nppp).ceil() as i32,
        );
        let chip_px = (
            (chip_w * z * nppp).ceil() as i32,
            (chip_h * z * nppp).ceil() as i32,
        );
        let panel_px = (
            ((if lurk { chip_w } else { bar_w }) * z * nppp).ceil() as i32,
            ((if lurk { chip_h } else { bar_h }) * z * nppp).ceil() as i32,
        );
        let content_rect = if status.content_rect.2 > 0 && status.content_rect.3 > 0 {
            status.content_rect
        } else {
            status.overlay_rect
        };
        let initial_px = panel_target_position(
            self.settings.scale_mode,
            status.overlay_rect,
            content_rect,
            panel_px,
        );
        let mut panel_stop_requested = false;
        let mut collapse = false;
        let mut expand = false;
        let mut take_screenshot = false;
        let mut toggle_gui_topmost = false;
        let mut hovered_any = false;
        let panel_actions = chidescaler_neo::input::take_panel_actions();
        if panel_actions & chidescaler_neo::input::PANEL_ACTION_STOP != 0 {
            panel_stop_requested = true;
        }
        if panel_actions & chidescaler_neo::input::PANEL_ACTION_COLLAPSE != 0 {
            collapse = true;
        }
        if panel_actions & chidescaler_neo::input::PANEL_ACTION_EXPAND != 0 {
            expand = true;
        }
        if panel_actions & chidescaler_neo::input::PANEL_ACTION_SCREENSHOT != 0 {
            take_screenshot = true;
        }
        if panel_actions & chidescaler_neo::input::PANEL_ACTION_GUI_TOPMOST != 0 {
            toggle_gui_topmost = true;
        }
        let now = Instant::now();
        if self
            .panel_screenshot_feedback_until
            .is_some_and(|until| now >= until)
        {
            self.panel_screenshot_feedback_until = None;
        }
        if take_screenshot {
            self.panel_screenshot_feedback_until = Some(now + Duration::from_millis(900));
        }
        let screenshot_feedback = self.panel_screenshot_feedback_until.is_some();
        if screenshot_feedback {
            ctx.request_repaint_after(Duration::from_millis(50));
        }
        let stable_hover_action = (show && self.panel_hwnd != 0).then_some(()).and_then(|_| {
            let (rx, ry, rw, rh) = win32::window_rect(self.panel_hwnd)?;
            let inside = |(x, y): (i32, i32)| {
                (x >= rx && x < rx + rw && y >= ry && y < ry + rh).then_some((x, y))
            };
            // The visible cursor over the floating panel is normally Neo's
            // sprite, not the real cursor.  virtual_cursor_pos() uses try_lock
            // and can therefore return None for a single repaint while the LL
            // hook owns State; that one-frame miss was enough to alternate the
            // custom button fill between hover/idle and produce a subtle blink.
            // Read the already-published sprite target lock-free first, then
            // fall back to the native cursor only when the sprite is not the
            // visible owner.
            let pointer = chidescaler_neo::input::panel_virtual_cursor_pos_lockfree()
                .and_then(inside)
                .or_else(|| inside(win32::cursor_pos()))?;
            chidescaler_neo::input::panel_action_for_relative_x(rw, rh, pointer.0 - rx)
        });
        // v465: the floating control panel no longer needs an eframe/WGPU
        // child viewport. v448+ already renders the visible pixels with the
        // cached GDI mirror and input.rs already commits every panel action via
        // the lock-free Win32 hook. Keeping show_viewport_immediate here made
        // every unrelated root-GUI hover/mode-switch repaint submit another
        // WGPU surface. On the low-spec reproduction that lurk/full-bar child
        // Present repeatedly stalled for 15-27 ms even while the panel was in
        // transparent lurk mode. Use a native GDI host instead: same HWND,
        // geometry, topmost and direct-input contract, zero panel GPU Present.
        let panel_host_update_started = Instant::now();
        if self.panel_hwnd == 0 {
            if let Some(h) = win32::ensure_panel_gdi_host(panel_px.0, panel_px.1) {
                self.panel_hwnd = h;
                self.panel_gdi_reveal_pending = false;
                self.panel_state_sent = None;
                self.panel_layout_sent = None;
                win32::set_window_rect(h, initial_px.0, initial_px.1, panel_px.0, panel_px.1);
                log::info!(
                    "panel-backend: native-gdi isolated hwnd={:#x} no_wgpu_surface=true",
                    h
                );
            }
        }
        // Keep the historical diagnostic field name so old/new logs remain
        // directly comparable. It now measures only native-host bookkeeping;
        // there is deliberately no child WGPU Present in this path.
        let panel_viewport_draw_ms = panel_host_update_started.elapsed().as_secs_f64() * 1000.0;

        if self.panel_hwnd != 0
            && (!win32::is_window_valid(self.panel_hwnd) || !win32::is_own_window(self.panel_hwnd))
        {
            self.panel_hwnd = 0;
            self.panel_gdi_reveal_pending = false;
            self.panel_state_sent = None;
            self.panel_layout_sent = None;
        }
        if self.panel_hwnd == 0 {
            if let Some(h) = win32::ensure_panel_gdi_host(panel_px.0, panel_px.1) {
                self.panel_hwnd = h;
                self.panel_gdi_reveal_pending = false;
                self.panel_state_sent = None;
                self.panel_layout_sent = None;
            }
        }
        if self.panel_hwnd != 0 {
            if status.running && !self.panel_placed_for_run {
                let active_size = if lurk { chip_px } else { bar_px };
                let (px, py) = panel_target_position(
                    self.settings.scale_mode,
                    status.overlay_rect,
                    content_rect,
                    active_size,
                );
                win32::set_window_rect(self.panel_hwnd, px, py, active_size.0, active_size.1);
                self.panel_placed_for_run = true;
                self.panel_state_sent = None;
                log::info!(
                    "panel initial placement committed before show: rect=({}, {}, {}, {})",
                    px,
                    py,
                    active_size.0,
                    active_size.1
                );
            }
            if show && self.panel_placed_for_run {
                win32::normalize_neo_topmost_stack(
                    self.gui_hwnd,
                    self.settings.gui_topmost,
                    self.panel_hwnd,
                    status.overlay_hwnd,
                );
                chidescaler_neo::input::keep_cursor_sprite_on_top();
            }
            if show && let Some((rx, ry, rw, rh)) = win32::window_rect(self.panel_hwnd) {
                // No invisible hover moat around the panel. Expanding the chip
                // while the pointer is still outside its visible rectangle can
                // make the enlarged bar appear underneath a passing cursor and
                // feel like magnetic attraction. Hover begins only after the
                // visible cursor is physically inside the current panel rect.
                let over = |x: i32, y: i32| x >= rx && x < rx + rw && y >= ry && y < ry + rh;
                let (cx, cy) = win32::cursor_pos();
                if over(cx, cy) {
                    hovered_any = true;
                }
                // While engaged the REAL cursor is confined to the source and
                // can never land on the chip; the VIRTUAL cursor (sprite) is
                // what visually hovers it, so hit-test that too (the cursor-routing design's
                // _check_panel_hover). This is what auto-expands the chip in
                // fullscreen without any teleport.
                if let Some((vx, vy)) = chidescaler_neo::input::panel_virtual_cursor_pos_lockfree()
                {
                    if over(vx, vy) {
                        hovered_any = true;
                    }
                }
            }
            let layout = (self.panel_hwnd, bar_px, chip_px);
            if self.panel_layout_sent != Some(layout) {
                self.engine.send(Cmd::SetPanel {
                    hwnd: self.panel_hwnd,
                    bar: bar_px,
                    chip: chip_px,
                });
                self.panel_layout_sent = Some(layout);
            }
        }

        // old-tool lurk behaviour: hover expands temporarily; leaving for
        // 350ms lurks again; the — button toggles the lurking style
        self.apply_panel_action(collapse, expand);
        if collapse {
            // the pointer is still on the chip right after the — click; without
            // this the hover-expand below re-opens the bar on the very next
            // frame and — appears to do nothing
            self.panel_hover_expand_armed = false;
        }
        if !hovered_any {
            self.panel_hover_expand_armed = true;
        }
        if collapse {
            self.panel_hover_restore_armed = false;
        }
        if !hovered_any {
            self.panel_hover_restore_armed = true;
        }
        if self.panel_chip_lurking {
            if lurk {
                // Hovering the fully transparent chip area restores the persistent panel.
                // The restore remains one-way until collapse is requested again,
                // and is disarmed until the pointer leaves once after a collapse.
                if hovered_any && self.panel_hover_restore_armed {
                    self.panel_chip_lurking = false;
                    self.panel_bar_shown = true;
                    self.panel_leave_at = None;
                    ctx.request_repaint();
                }
            } else {
                // bar shown temporarily: re-lurk after leaving for 350ms
                if hovered_any {
                    self.panel_leave_at = None;
                } else {
                    let due = *self
                        .panel_leave_at
                        .get_or_insert_with(|| Instant::now() + Duration::from_millis(350));
                    if Instant::now() >= due {
                        self.panel_bar_shown = false;
                        self.panel_leave_at = None;
                    }
                }
            }
        }
        let final_lurk = self.panel_chip_lurking && !self.panel_bar_shown;
        let final_show = show && self.panel_placed_for_run;
        let restoring_from_lurk = final_show
            && !final_lurk
            && self
                .panel_state_sent
                .is_some_and(|(was_show, was_lurk)| was_show && was_lurk);
        let becoming_visible = final_show
            && !final_lurk
            && self
                .panel_state_sent
                .is_none_or(|(was_show, was_lurk)| !was_show || was_lurk);
        if restoring_from_lurk {
            // v528: v527 can resize/draw the complete 297x33 bar here while
            // the render thread still owns the previous queued chip=true state.
            // Until that thread consumes the final chip=false state below, tell
            // its geometry follower to keep using bar geometry rather than
            // replaying stale lurk-chip geometry over the freshly restored host.
            chidescaler_neo::engine::begin_panel_lurk_restore_geometry_guard();
        }
        if becoming_visible {
            // Atomic panel reveal: prepare the cached GDI pixels first and,
            // when GUI-topmost is OFF, also wait for the proven v459 WGPU
            // composition anchor.  The host itself stays alpha=0 until the
            // complete visual stack is ready, preventing the old two-stage
            // start/restore appearance.
            self.panel_gdi_reveal_pending = true;

            // Lurk restore used to wait for the render-engine SetPanel command
            // to resize the host from the transparent lurk hit area back to
            // the 297x33 bar. That left one GUI frame where the logical state
            // was already "bar" but USER32 still clipped the GDI child to the
            // old chip rectangle. On some DWM/AMD timings the child then became
            // visible while Windows was still validating only pieces of the
            // enlarged region, making the panel appear to fill in gradually as
            // the pointer moved. Resize the still-transparent host
            // synchronously at this exact transition; the engine receives the
            // same layout below and simply observes that it is already correct.
            if restoring_from_lurk
                && self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_own_window(self.panel_hwnd)
            {
                let (px, py) = panel_target_position(
                    self.settings.scale_mode,
                    status.overlay_rect,
                    content_rect,
                    bar_px,
                );
                win32::set_window_rect(self.panel_hwnd, px, py, bar_px.0, bar_px.1);
                log::debug!(
                    "panel-lurk-restore-prelayout: hwnd={:#x} rect=({}, {}, {}, {}) action=resize-hidden-before-gdi-reveal",
                    self.panel_hwnd,
                    px,
                    py,
                    bar_px.0,
                    bar_px.1
                );
            }
        }
        if self.panel_state_sent != Some((final_show, final_lurk)) {
            // Apply alpha/input state on the GUI thread in the same frame as the
            // — button action. SetPanelState is metadata-only; without this,
            // panel_state_sent could advance before alpha changed and the panel
            // would remain opaque instead of returning to its transparent lurk area.
            if self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_own_window(self.panel_hwnd)
            {
                if final_show {
                    win32::set_window_input_passthrough(self.panel_hwnd, false);
                    if final_lurk {
                        self.panel_gdi_reveal_pending = false;
                        win32::set_window_alpha(self.panel_hwnd, PANEL_LURK_ALPHA);
                    } else if self.panel_gdi_reveal_pending {
                        // The GDI mirror is a child of the native host. Alpha=0 lets
                        // us prepare the full mirror without showing either layer.
                        win32::set_window_alpha(self.panel_hwnd, 0);
                    } else {
                        win32::set_window_opaque_unlayered(self.panel_hwnd);
                    }
                } else {
                    self.panel_gdi_reveal_pending = false;
                    // Withdraw the visual WGPU anchor first.  It remains a
                    // separate HWND for the AMD composition contract, so hiding
                    // it before the parent panel prevents a one-composition
                    // trailing layer during Stop/panel-hide.  The anchor method
                    // below will retire the viewport normally in the same frame.
                    if self.compositor_anchor_hwnd != 0
                        && win32::is_window_valid(self.compositor_anchor_hwnd)
                        && win32::is_own_window(self.compositor_anchor_hwnd)
                    {
                        win32::set_window_alpha(self.compositor_anchor_hwnd, 0);
                        win32::set_visible_no_activate(self.compositor_anchor_hwnd, false);
                    }
                    win32::set_window_alpha(self.panel_hwnd, 0);
                    win32::set_window_input_passthrough(self.panel_hwnd, true);
                    win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
                }
            }
            self.engine.send(Cmd::SetPanelState {
                visible: final_show,
                chip: final_lurk,
            });
            self.panel_state_sent = Some((final_show, final_lurk));
            self.panel_layout_sent = None;
            ctx.request_repaint();
        }

        // v465: the cached GDI child is now the only panel drawing surface.
        // It covers both the ordinary bar and the tiny lurk chip; the native
        // host remains the hit-test/z-order/alpha owner and has no GPU surface.
        let mirror_hover_slot = match stable_hover_action {
            Some(chidescaler_neo::input::PANEL_ACTION_STOP) => 1,
            Some(chidescaler_neo::input::PANEL_ACTION_SCREENSHOT) => 2,
            Some(chidescaler_neo::input::PANEL_ACTION_GUI_TOPMOST) => 3,
            Some(chidescaler_neo::input::PANEL_ACTION_COLLAPSE) => 4,
            _ => 0,
        };
        let mirror_size = if final_lurk { chip_px } else { bar_px };
        let panel_geometry_ready = self.panel_hwnd != 0
            && win32::window_rect(self.panel_hwnd).is_some_and(|(_, _, w, h)| {
                (w - mirror_size.0).abs() <= 3 && (h - mirror_size.1).abs() <= 3
            });
        // Pre-v465 behavior: lurk is a completely invisible hit area.  Never
        // leave the cached GDI child visible as a dark lurk chip.
        let mirror_visible =
            final_show && !final_lurk && (!self.panel_gdi_reveal_pending || panel_geometry_ready);
        win32::update_panel_gdi_mirror(
            self.panel_hwnd,
            mirror_size.0,
            mirror_size.1,
            mirror_visible,
            final_lurk,
            i18n::text(lang, "capture.stop"),
            present_fps,
            mirror_hover_slot,
            screenshot_feedback,
        );
        let anchor_required_for_reveal = final_show
            && !final_lurk
            && !self.settings.gui_topmost
            && status.overlay_hwnd != 0
            && win32::is_window_valid(status.overlay_hwnd);
        let anchor_ready_for_reveal = !anchor_required_for_reveal
            || (self.compositor_anchor_hwnd != 0
                && win32::is_window_valid(self.compositor_anchor_hwnd)
                && win32::is_own_window(self.compositor_anchor_hwnd)
                && win32::is_window_visible(self.compositor_anchor_hwnd)
                && win32::window_is_above(self.compositor_anchor_hwnd, status.overlay_hwnd));
        if self.panel_gdi_reveal_pending && panel_geometry_ready && anchor_ready_for_reveal {
            // Reveal only after every visible layer is ready.  This is one
            // Show/alpha commit, not host-then-child or panel-then-anchor.
            win32::set_window_opaque_unlayered(self.panel_hwnd);
            win32::set_panel_gdi_host_visible(self.panel_hwnd, true);

            // The mirror is fully primed while the host is still hidden, but on
            // rare USER32/DWM timings the first visible composition can expose
            // only part of those cached pixels. Re-publish the same complete
            // 297x33 frame once *after* the host becomes visible. Keep ordinary
            // FPS/hover updates on their existing small dirty-region path.
            let reveal_republish = win32::republish_panel_gdi_mirror_full(self.panel_hwnd);
            self.panel_gdi_reveal_pending = false;
            log::debug!(
                "panel-atomic-reveal-commit: hwnd={:#x} size={}x{} anchor_required={} order=mirror-anchor-host-last reveal_republish={}",
                self.panel_hwnd,
                mirror_size.0,
                mirror_size.1,
                anchor_required_for_reveal,
                reveal_republish
            );
        }
        if final_show && !self.panel_gdi_reveal_pending {
            win32::set_panel_gdi_host_visible(self.panel_hwnd, true);
        } else if !final_show {
            win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
        }

        if logging::diagnostics_enabled() {
            let expected_size = if final_lurk { chip_px } else { bar_px };
            let rect = if self.panel_hwnd != 0 && win32::is_window_valid(self.panel_hwnd) {
                win32::window_rect(self.panel_hwnd)
            } else {
                None
            };
            let native_visible = self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_window_visible(self.panel_hwnd);
            let layered = self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_window_layered(self.panel_hwnd);
            let passthrough = self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::window_input_passthrough(self.panel_hwnd);
            let topmost = self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_topmost(self.panel_hwnd);
            let (gdi_mirror_valid, gdi_mirror_visible) =
                win32::panel_gdi_mirror_status(self.panel_hwnd);
            let snapshot = PanelDiagSnapshot {
                rect,
                expected_size,
                effective_show: final_show,
                lurk: final_lurk,
                native_visible,
                layered,
                passthrough,
                topmost,
            };
            let state_changed = self.panel_diag_last_snapshot != Some(snapshot);
            if state_changed {
                log::info!(
                    "panel-compositor-state: seq={} hwnd={:#x} rect={:?} expected={}x{} show={} lurk={} native_visible={} layered={} passthrough={} topmost={}",
                    panel_diag_seq,
                    self.panel_hwnd,
                    rect,
                    expected_size.0,
                    expected_size.1,
                    final_show,
                    final_lurk,
                    native_visible,
                    layered,
                    passthrough,
                    topmost
                );
                self.panel_diag_last_snapshot = Some(snapshot);
            }

            let size_mismatch = final_show
                && rect.is_some_and(|(_, _, w, h)| {
                    (w - expected_size.0).abs() > 3 || (h - expected_size.1).abs() > 3
                });
            let visibility_mismatch = physical_show && self.panel_placed_for_run && !native_visible;
            let layered_mismatch =
                final_show && ((!final_lurk && layered) || (final_lurk && !layered));
            let passthrough_mismatch = final_show && passthrough;
            let topmost_mismatch = final_show && !topmost;
            // In lurk mode the cached GDI child is deliberately hidden: the
            // parent HWND alone remains as the transparent hover target.  The
            // old diagnostic treated that expected hidden state as an anomaly,
            // producing a WARN every trace interval while the panel was
            // lurking.  Validate the mirror against the state we actually
            // require instead: visible for the bar, hidden for the lurk chip.
            let mirror_should_be_visible = final_show && !final_lurk;
            let mirror_mismatch =
                final_show && (!gdi_mirror_valid || gdi_mirror_visible != mirror_should_be_visible);
            let draw_stall = panel_viewport_draw_ms >= 12.0;
            let heartbeat_stall = panel_diag_gap_ms >= 900.0;
            let anomaly = size_mismatch
                || visibility_mismatch
                || layered_mismatch
                || passthrough_mismatch
                || topmost_mismatch
                || mirror_mismatch
                || draw_stall
                || heartbeat_stall;
            // Pointer movement can wake egui at high frequency. Keep the
            // diagnostic timeline dense enough to correlate a blink without
            // turning ordinary mouse movement into a log flood.
            let should_trace = state_changed
                || anomaly
                || self.panel_diag_last_trace_at.elapsed() >= Duration::from_millis(250);
            if should_trace {
                self.panel_diag_last_trace_at = Instant::now();
                log::debug!(
                    "panel-frame-diag: seq={} gap_ms={:.2} draw_ms={:.2} hwnd={:#x} show={} effective_show={} physical_show={} lurk_before={} lurk_final={} hovered={} fps={:.1} expected={}x{} rect={:?} native_visible={} layered={} passthrough={} topmost={} gdi_mirror_valid={} gdi_mirror_visible={} state_changed={} anomalies=size:{} visible:{} layered:{} passthrough:{} topmost:{} mirror:{} draw_stall:{} heartbeat_stall:{}",
                    panel_diag_seq,
                    panel_diag_gap_ms,
                    panel_viewport_draw_ms,
                    self.panel_hwnd,
                    show,
                    final_show,
                    physical_show,
                    lurk,
                    final_lurk,
                    hovered_any,
                    present_fps,
                    expected_size.0,
                    expected_size.1,
                    rect,
                    native_visible,
                    layered,
                    passthrough,
                    topmost,
                    gdi_mirror_valid,
                    gdi_mirror_visible,
                    state_changed,
                    size_mismatch,
                    visibility_mismatch,
                    layered_mismatch,
                    passthrough_mismatch,
                    topmost_mismatch,
                    mirror_mismatch,
                    draw_stall,
                    heartbeat_stall
                );

                if anomaly {
                    log::warn!(
                        "panel-compositor-anomaly: seq={} hwnd={:#x} gap_ms={:.2} draw_ms={:.2} rect={:?} expected={}x{} show={} lurk={} visible={} layered={} passthrough={} topmost={} gdi_mirror_valid={} gdi_mirror_visible={} flags=[size={},visible={},layered={},passthrough={},topmost={},mirror={},draw_stall={},heartbeat_stall={}]",
                        panel_diag_seq,
                        self.panel_hwnd,
                        panel_diag_gap_ms,
                        panel_viewport_draw_ms,
                        rect,
                        expected_size.0,
                        expected_size.1,
                        final_show,
                        final_lurk,
                        native_visible,
                        layered,
                        passthrough,
                        topmost,
                        gdi_mirror_valid,
                        gdi_mirror_visible,
                        size_mismatch,
                        visibility_mismatch,
                        layered_mismatch,
                        passthrough_mismatch,
                        topmost_mismatch,
                        mirror_mismatch,
                        draw_stall,
                        heartbeat_stall
                    );
                }
            }
        }

        if panel_stop_requested {
            if self.qa_panel_preview {
                self.qa_panel_preview = false;
            } else {
                // The floating panel is a STOP-only control, not a Start/Stop
                // toggle. On a heavily loaded machine the low-level input hook
                // can deliver the direct Stop immediately while egui sees the
                // same physical click much later. Calling toggle() here made
                // that delayed duplicate click restart capture after Stop had
                // already completed. Treat both paths as the same idempotent
                // Stop request and ignore stale duplicates once capture is idle.
                let current = self.engine.status.lock().unwrap().clone();
                let provider_preparing =
                    chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
                if Self::should_dispatch_panel_stop(
                    current.starting,
                    current.running,
                    current.stopping,
                    provider_preparing,
                ) {
                    let _ = self.request_capture_stop("floating-panel");
                } else {
                    log::debug!("panel-stop-stale-click-ignored: capture already idle or stopping");
                }
            }
        }
        if take_screenshot {
            self.panel_screenshot_feedback_until =
                Some(Instant::now() + Duration::from_millis(900));
            ctx.request_repaint();
            self.request_filtered_screenshot();
        }
        if toggle_gui_topmost {
            // Deliberately share the exact same implementation as Ctrl+Alt+G.
            self.toggle_gui_topmost();
        }
    }

    /// Restore only the compositor keep-alive part of the pre-v465 floating
    /// panel architecture.  The actual panel is still native GDI and all cursor
    /// / input ownership remains on the existing Win32 path.
    ///
    /// v448-v454 deliberately kept an eframe/WGPU panel HWND physically alive
    /// because it was part of the AMD composition contract.  v465 removed that
    /// WGPU surface to eliminate child-swapchain Present stalls.  On affected
    /// AMD systems the remaining fullscreen WGL overlay can then be scanned out
    /// independently while ordinary GDI/layered helper HWNDs are logically above
    /// it but absent from the final screen.  A source popup or topmost main GUI
    /// forces normal composition again, which exactly matches the field symptom.
    ///
    /// This anchor is intentionally NOT a replacement panel.  It is first
    /// created hidden, placed underneath the GDI panel, made input-transparent,
    /// then shown without activation.  GUI-topmost ON never uses the anchor.
    fn compositor_keepalive_anchor(&mut self, ctx: &egui::Context, status: &Status) {
        const TITLE: &str = "NeoCompositorKeepalivePanel";

        let overlay_valid = status.overlay_hwnd != 0 && win32::is_window_valid(status.overlay_hwnd);
        let panel_valid = self.panel_hwnd != 0
            && win32::is_window_valid(self.panel_hwnd)
            && win32::is_own_window(self.panel_hwnd);
        // v459-proven contract: the extra WGPU top-level surface exists only
        // while the ordinary floating panel is physically present and the main
        // GUI is not itself topmost.  Do not create a tiny always-on fallback
        // when the panel is disabled: that was a v505 experiment, not part of
        // the known-good pre-v465 behavior.
        let active = status.running
            && (!self.settings.gui_topmost || self.gui_topmost_off_pending)
            && overlay_valid
            && panel_valid
            && self.panel_visible
            && self.settings.panel_show
            && (win32::is_window_visible(self.panel_hwnd) || self.panel_gdi_reveal_pending);

        if !active {
            // If capture/panel state changed before an OFF transition could
            // need an anchor, the overlay no longer requires the composition
            // bridge. Commit the requested ordinary GUI state directly.
            if self.gui_topmost_off_pending
                && (!status.running
                    || status.stopping
                    || !overlay_valid
                    || !panel_valid
                    || !self.panel_visible
                    || !self.settings.panel_show)
            {
                self.commit_gui_topmost_off("anchor-became-unneeded");
            }

            let old_anchor = self.compositor_anchor_hwnd;
            if old_anchor != 0
                && win32::is_window_valid(old_anchor)
                && win32::is_own_window(old_anchor)
            {
                win32::set_visible_no_activate(old_anchor, false);
                win32::set_own_topmost(old_anchor, false);
            }
            if self.compositor_anchor_state_sent
                != Some((false, old_anchor, self.panel_hwnd, status.overlay_hwnd))
            {
                self.compositor_anchor_state_sent =
                    Some((false, old_anchor, self.panel_hwnd, status.overlay_hwnd));
                if old_anchor != 0 {
                    log::info!(
                        "compositor-keepalive-anchor: active=false hwnd={:#x} action=withdraw-v459-panel-contract",
                        old_anchor
                    );
                }
            }
            // Not submitting an immediate viewport destroys the retained child.
            self.compositor_anchor_hwnd = 0;
            return;
        }

        let Some((panel_x, panel_y, panel_w, panel_h)) = win32::window_rect(self.panel_hwnd) else {
            return;
        };
        // Keep the proven v459-compatible WGPU surface underneath the native
        // GDI panel. Only its inset follows the 5 px rounded host so no WGPU
        // pixels can peek through the clipped corners; classification/lifetime
        // and z-order rules remain identical to v508.
        let inset = if panel_w > 12 && panel_h > 12 { 5 } else { 1 };
        let px = panel_x + inset;
        let py = panel_y + inset;
        let pw = (panel_w - inset * 2).max(2);
        let ph = (panel_h - inset * 2).max(2);

        let z = ctx.zoom_factor().max(0.5);
        let nppp = ctx
            .input(|i| i.viewport().native_pixels_per_point)
            .unwrap_or(1.0)
            .max(0.5);
        let logical_per_px = z / nppp;
        let size_pts = [
            (pw as f32 * logical_per_px).max(2.0),
            (ph as f32 * logical_per_px).max(2.0),
        ];
        let pos_pts = [px as f32 * logical_per_px, py as f32 * logical_per_px];

        // Match the known-good v454/v459 panel viewport classification as
        // closely as possible.  In particular, do NOT use with_active(false)
        // or with_mouse_passthrough(true): those change the native extended
        // style/classification and were not present before v465.
        let creating_anchor = self.compositor_anchor_hwnd == 0;
        let viewport = egui::ViewportBuilder::default()
            .with_title(TITLE)
            .with_decorations(false)
            .with_always_on_top()
            .with_resizable(false)
            .with_taskbar(false)
            // Never expose the WGPU backing layer before it has been placed
            // underneath the completed GDI panel.  This changes only initial
            // visibility, not the v459 HWND/surface classification.
            .with_visible(!creating_anchor)
            .with_inner_size(size_pts)
            .with_position(pos_pts);

        let present_started = Instant::now();
        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("neo_compositor_keepalive_panel_v459"),
            viewport,
            |ctx2, _| {
                egui::CentralPanel::default()
                    .frame(egui::Frame::new().fill(egui::Color32::from_rgb(43, 43, 43)))
                    .show(ctx2, |_ui| {});
            },
        );
        let anchor_present_ms = present_started.elapsed().as_secs_f64() * 1000.0;

        if self.compositor_anchor_hwnd != 0
            && (!win32::is_window_valid(self.compositor_anchor_hwnd)
                || !win32::is_own_window(self.compositor_anchor_hwnd))
        {
            self.compositor_anchor_hwnd = 0;
            self.compositor_anchor_state_sent = None;
        }
        if self.compositor_anchor_hwnd == 0 {
            if let Some(hwnd) = win32::find_own_window(TITLE) {
                self.compositor_anchor_hwnd = hwnd;
                self.compositor_anchor_state_sent = None;
                log::info!(
                    "compositor-keepalive-anchor-created: hwnd={:#x} backend=eframe-wgpu contract=v459-normal-top-level",
                    hwnd
                );
            }
        }

        let anchor = self.compositor_anchor_hwnd;
        if anchor == 0 || !win32::is_window_valid(anchor) || !win32::is_own_window(anchor) {
            ctx.request_repaint_after(Duration::from_millis(16));
            return;
        }

        // Keep the composition surface alive in lurk mode, and also keep it
        // transparent while a full panel reveal is being prepared.  The old
        // v512 ordering made this backing layer opaque before the GDI host, so
        // a dark/offset second layer could be seen for one composition frame.
        let anchor_lurk = self.panel_chip_lurking && !self.panel_bar_shown;
        let anchor_must_stay_transparent = anchor_lurk || self.panel_gdi_reveal_pending;
        if anchor_must_stay_transparent {
            if !win32::is_window_layered(anchor) {
                win32::set_window_alpha(anchor, 0);
            }
        }

        // Geometry follows the native panel, but avoid continuous TOPMOST
        // churn.  Prepare the hidden/transparent backing surface completely
        // before it can become visible.
        if win32::window_rect(anchor) != Some((px, py, pw, ph)) {
            win32::set_window_rect(anchor, px, py, pw, ph);
        }
        if !win32::is_topmost(anchor) {
            win32::set_own_topmost(anchor, true);
        }
        if !win32::window_is_above(anchor, status.overlay_hwnd) {
            win32::place_below(status.overlay_hwnd, anchor);
        }
        if !win32::window_is_above(self.panel_hwnd, anchor) {
            win32::place_below(anchor, self.panel_hwnd);
        }
        if !win32::is_window_visible(anchor) {
            // At first creation the viewport itself is hidden.  Show it only
            // after alpha/geometry/z-order have been committed.
            win32::set_visible_no_activate(anchor, true);
        }

        // control_panel() is the sole owner of revealing the GDI host.  Once it
        // has completed that commit, make the already-covered WGPU anchor
        // opaque underneath it and wait for one DWM boundary.  Front layer
        // first, backing layer second: there is never an exposed WGPU rectangle.
        if !anchor_lurk
            && !self.panel_gdi_reveal_pending
            && win32::is_window_visible(self.panel_hwnd)
            && win32::window_is_above(self.panel_hwnd, anchor)
            && win32::is_window_layered(anchor)
        {
            win32::set_window_opaque_unlayered(anchor);
            win32::sync_panel_composition_with_dwm();
            log::debug!(
                "panel-layer-sync-commit: panel={:#x} anchor={:#x} order=panel-first-anchor-second",
                self.panel_hwnd,
                anchor
            );
        } else if !anchor_lurk
            && !self.panel_gdi_reveal_pending
            && !win32::is_window_layered(anchor)
        {
            // Normal steady state: preserve the proven opaque v459 surface.
        }

        // The anchor is now fully placed, visible, topmost and covered by the
        // GDI panel. Only at this boundary may a staged GUI-TOPMOST OFF request
        // demote the root GUI. DWM therefore sees one stable composition change
        // instead of "GUI down -> create anchor -> restack".
        if self.gui_topmost_off_pending
            && !anchor_lurk
            && !self.panel_gdi_reveal_pending
            && win32::is_window_visible(anchor)
            && win32::window_is_above(anchor, status.overlay_hwnd)
            && win32::window_is_above(self.panel_hwnd, anchor)
            && !win32::is_window_layered(anchor)
        {
            self.commit_gui_topmost_off("anchor-prepared-before-demote");
            win32::sync_panel_composition_with_dwm();
            log::debug!(
                "GUI topmost off atomic-commit: gui={:#x} panel={:#x} anchor={:#x} overlay={:#x}",
                self.gui_hwnd,
                self.panel_hwnd,
                anchor,
                status.overlay_hwnd
            );
        }

        chidescaler_neo::input::keep_cursor_sprite_on_top();

        let state = (active, anchor, self.panel_hwnd, status.overlay_hwnd);
        if self.compositor_anchor_state_sent != Some(state) {
            self.compositor_anchor_state_sent = Some(state);
            log::info!(
                "compositor-keepalive-anchor: active=true contract=v459 hwnd={:#x} panel={:#x} overlay={:#x} anchor_above_overlay={} panel_above_anchor={} rect={:?} present_ms={:.2}",
                anchor,
                self.panel_hwnd,
                status.overlay_hwnd,
                win32::window_is_above(anchor, status.overlay_hwnd),
                win32::window_is_above(self.panel_hwnd, anchor),
                win32::window_rect(anchor),
                anchor_present_ms,
            );
        }

        if anchor_present_ms >= 8.0
            && self.compositor_anchor_last_warn_at.elapsed() >= Duration::from_secs(1)
        {
            self.compositor_anchor_last_warn_at = Instant::now();
            log::warn!(
                "compositor-keepalive-anchor-present-stall: ms={:.2} hwnd={:#x} contract=v459 diagnostic_only=true",
                anchor_present_ms,
                anchor
            );
        }
    }

    fn request_filtered_screenshot(&self) {
        use std::time::{SystemTime, UNIX_EPOCH};
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let path = self
            .app_dir
            .join("screenshots")
            .join(format!("cHiDeScaler-Neo_{stamp}.png"));
        self.engine.send(Cmd::SaveScreenshot(path));
    }
}

impl App {
    /// Three-second, non-blocking explanation for the windowed-source occlusion
    /// safety stop. The actual Stop/recovery has already completed in the engine;
    /// this viewport is GUI-only and never participates in capture/input routing.
    fn render_source_occlusion_notice(&mut self, ctx: &egui::Context, lang: UiLanguage) {
        let Some(until) = self.source_occlusion_notice_until else {
            if self.source_occlusion_notice_hwnd != 0
                && !win32::is_window_valid(self.source_occlusion_notice_hwnd)
            {
                self.source_occlusion_notice_hwnd = 0;
            }
            return;
        };
        let now = Instant::now();
        if now >= until {
            self.source_occlusion_notice_until = None;
            self.source_occlusion_notice_hwnd = 0;
            log::info!("source-occlusion-notice: action=expired");
            return;
        }

        let monitor_size = ctx
            .input(|input| input.viewport().monitor_size)
            .unwrap_or(egui::vec2(1280.0, 720.0));
        let size = [
            500.0_f32.min((monitor_size.x - 32.0).max(320.0)),
            112.0_f32.min((monitor_size.y - 48.0).max(96.0)),
        ];
        let mut viewport = egui::ViewportBuilder::default()
            .with_title("cHiDeScaler-Neo Safety Notice")
            .with_decorations(false)
            .with_always_on_top()
            .with_resizable(false)
            .with_taskbar(false)
            .with_inner_size(size);
        if self.source_occlusion_notice_hwnd == 0 {
            let z = ctx.zoom_factor();
            if let Some(outer) = ctx.input(|input| input.viewport().outer_rect) {
                let center = outer.center();
                viewport = viewport.with_position([
                    (center.x * z - size[0] * 0.5).max(8.0),
                    (center.y * z - size[1] * 0.5).max(8.0),
                ]);
            }
        }

        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("neo_source_occlusion_notice"),
            viewport,
            |ctx2, _| {
                let frame = egui::Frame::new()
                    .fill(egui::Color32::from_rgb(28, 25, 18))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(214, 164, 66),
                    ))
                    .corner_radius(egui::CornerRadius::same(7))
                    .inner_margin(egui::Margin::same(14));
                egui::CentralPanel::default().frame(frame).show(ctx2, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(
                            egui::RichText::new(i18n::text(lang, "source_occlusion.notice_title"))
                                .family(locale_font_family(lang))
                                .strong()
                                .size(13.0),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new(i18n::text(lang, "source_occlusion.notice"))
                                .family(locale_font_family(lang))
                                .size(12.0),
                        );
                    });
                });
            },
        );

        if self.source_occlusion_notice_hwnd == 0
            && let Some(hwnd) = win32::find_own_window("cHiDeScaler-Neo Safety Notice")
        {
            self.source_occlusion_notice_hwnd = hwnd;
            log::info!("source-occlusion-notice: hwnd={hwnd:#x} action=shown duration_ms=3000");
        }
        if self.source_occlusion_notice_hwnd != 0
            && win32::is_window_valid(self.source_occlusion_notice_hwnd)
        {
            if !win32::is_topmost(self.source_occlusion_notice_hwnd) {
                win32::set_own_topmost(self.source_occlusion_notice_hwnd, true);
            }
            win32::raise_topmost(self.source_occlusion_notice_hwnd);
        }
        ctx.request_repaint_after((until - now).min(Duration::from_millis(100)));
    }

    /// Mini overload warning must stay on the already-existing root GUI surface.
    ///
    /// v465 removed the floating panel's WGPU viewport after low-spec machines
    /// exposed horizontal/sandstorm-like corruption when an auxiliary WGPU
    /// swapchain was created/repainted under heavy 3D pressure. The generic
    /// Mini dialog host reintroduced that same class of risk for the GLSL
    /// overload notice because the notice appears precisely when GPU headroom is
    /// exhausted. Draw this informational warning as a foreground Area inside
    /// the root GUI instead: no new HWND, swapchain, surface configure, resize,
    /// DWM restack, capture change, or video-path synchronization is involved.
    fn render_mini_glsl_overload_notice(
        &self,
        ctx: &egui::Context,
        lang: UiLanguage,
        status: &Status,
        running: bool,
    ) {
        if self.settings.ui_mode != UiMode::Mini || !running || !status.glsl_overload_notice_latched
        {
            return;
        }

        egui::Area::new(egui::Id::new("mini-glsl-overload-inline"))
            // Keep the warning wholly inside Mini's first row.  It must never
            // grow downward into the capture button row, even while the locale
            // font is being swapped in the same egui frame.
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(10.0, 6.0))
            .order(egui::Order::Foreground)
            .interactable(false)
            .show(ctx, |ui| {
                // Keep the warning icon out of the text run.  Locale-specific
                // fonts give U+26A0 (WARNING SIGN) different ascent/baseline
                // metrics, so a single text galley can look vertically centered
                // in Japanese while the triangle shifts in Latin/CJK/Korean
                // locales.  The triangle is now pure geometry centered on the
                // physical warning rect; only the message uses locale fonts.
                let warning = i18n::text(lang, "glsl_overload.auto_stop");
                let font = egui::FontId::new(11.5, locale_font_family(lang));
                // Do not use Label/Frame auto-layout here.  During a language
                // switch egui can briefly inherit a narrow available width;
                // Label then wraps to two rows and the floating warning reaches
                // the capture button below.  A pre-laid-out galley has no wrap
                // path at all, so the warning remains exactly one line.
                let galley =
                    ui.painter()
                        .layout_no_wrap(warning.to_owned(), font, egui::Color32::WHITE);
                let ink = galley.mesh_bounds;
                const WARNING_H: f32 = 25.0;
                const WARNING_MIN_W: f32 = 104.0;
                const WARNING_PAD_X: f32 = 12.0;
                const ICON_W: f32 = 12.0;
                const ICON_H: f32 = 11.0;
                const ICON_TEXT_GAP: f32 = 5.0;
                let group_w = ICON_W + ICON_TEXT_GAP + ink.width();
                let warning_w = (group_w + WARNING_PAD_X * 2.0).max(WARNING_MIN_W);
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(warning_w, WARNING_H), egui::Sense::hover());
                ui.painter()
                    .rect_filled(rect, 5.0, egui::Color32::from_rgb(50, 39, 18));
                ui.painter().rect_stroke(
                    rect,
                    5.0,
                    egui::Stroke::new(1.0, egui::Color32::from_rgb(214, 164, 66)),
                    egui::StrokeKind::Inside,
                );

                let group_left = rect.center().x - group_w * 0.5;
                let icon_center = egui::pos2(group_left + ICON_W * 0.5, rect.center().y);
                let half_w = ICON_W * 0.5;
                let half_h = ICON_H * 0.5;
                let icon_stroke = egui::Stroke::new(1.15, egui::Color32::WHITE);
                let top = egui::pos2(icon_center.x, icon_center.y - half_h);
                let left = egui::pos2(icon_center.x - half_w, icon_center.y + half_h);
                let right = egui::pos2(icon_center.x + half_w, icon_center.y + half_h);
                ui.painter().line_segment([top, left], icon_stroke);
                ui.painter().line_segment([left, right], icon_stroke);
                ui.painter().line_segment([right, top], icon_stroke);
                ui.painter().line_segment(
                    [
                        egui::pos2(icon_center.x, icon_center.y - 2.6),
                        egui::pos2(icon_center.x, icon_center.y + 1.5),
                    ],
                    egui::Stroke::new(1.25, egui::Color32::WHITE),
                );
                ui.painter().circle_filled(
                    egui::pos2(icon_center.x, icon_center.y + 3.6),
                    0.85,
                    egui::Color32::WHITE,
                );

                // Center the visible message ink independently of the icon.
                // The icon's geometric center and the message ink center both
                // sit exactly on the yellow border's vertical center in every
                // locale, while the whole icon+text group remains horizontally
                // centered as one unit.
                let desired_ink_left = group_left + ICON_W + ICON_TEXT_GAP;
                let galley_pos = egui::pos2(
                    desired_ink_left - ink.min.x,
                    rect.center().y - ink.center().y,
                );
                ui.painter()
                    .galley(galley_pos, galley, egui::Color32::WHITE);
            });
    }

    fn render_mini_dialog_host(
        &mut self,
        ctx: &egui::Context,
        lang: UiLanguage,
        status: &Status,
        running: bool,
    ) {
        if self.mini_dialog_hwnd != 0 && !win32::is_window_valid(self.mini_dialog_hwnd) {
            self.mini_dialog_hwnd = 0;
        }
        if self.settings.ui_mode != UiMode::Mini {
            return;
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum Kind {
            TensorRt,
            Elevated,
            FullscreenCaptureNotice,
            ResizeScale,
            Hotkey,
            FilterPicker,
            SaveAs,
            DeletePreset,
        }

        let trt_progress = chidescaler_neo::render::onnx_stage::tensorrt_build_progress();
        let kind = if trt_progress.is_some() {
            Some(Kind::TensorRt)
        } else if self.elevated_target_notice.is_some() {
            Some(Kind::Elevated)
        } else if self.capture_resolution_fullscreen_notice_open {
            Some(Kind::FullscreenCaptureNotice)
        } else if self.resize_scale_editor.is_some() {
            Some(Kind::ResizeScale)
        } else if self.hotkey_editor_open {
            Some(Kind::Hotkey)
        } else if self.filter_picker_open {
            Some(Kind::FilterPicker)
        } else if self.save_as_open {
            Some(Kind::SaveAs)
        } else if self.confirm_delete {
            Some(Kind::DeletePreset)
        } else {
            None
        };

        let Some(kind) = kind else {
            // show_viewport_immediate destroys the retained child after it is no
            // longer submitted. Do not keep a stale HWND in cursor/z-order code.
            if self.mini_dialog_hwnd != 0 && !win32::is_window_valid(self.mini_dialog_hwnd) {
                self.mini_dialog_hwnd = 0;
            }
            return;
        };

        let (display_title, requested_size): (&str, [f32; 2]) = match kind {
            Kind::TensorRt => (
                tr(
                    lang,
                    "TensorRT用エンジンを作成中",
                    "Preparing TensorRT engine",
                ),
                [430.0, 285.0],
            ),
            Kind::Elevated => (
                tr(
                    lang,
                    "管理者権限が必要です",
                    "Administrator permission required",
                ),
                [440.0, 330.0],
            ),
            Kind::FullscreenCaptureNotice => (
                i18n::text(lang, "capture.fullscreen_notice_title"),
                [400.0, 280.0],
            ),
            Kind::ResizeScale => (tr(lang, "リサイズ倍率", "Resize scale"), [340.0, 225.0]),
            Kind::Hotkey => (
                tr(lang, "ショートカット編集", "Edit Shortcut"),
                [430.0, 340.0],
            ),
            Kind::FilterPicker => (tr(lang, "フィルター追加", "Add Filter"), [470.0, 520.0]),
            Kind::SaveAs => (tr(lang, "別名で保存", "Save As"), [430.0, 250.0]),
            Kind::DeletePreset => (
                tr(lang, "プリセットの削除", "Delete Preset"),
                [390.0, 230.0],
            ),
        };

        // Never let a dialog exceed the current monitor. This is especially
        // important for Mini on low-resolution / high-DPI desktops: the body
        // becomes scrollable while the action row remains fixed and reachable.
        let monitor_size = ctx
            .input(|input| input.viewport().monitor_size)
            .unwrap_or(egui::vec2(1280.0, 720.0));
        let size = [
            requested_size[0].min((monitor_size.x - 32.0).max(300.0)),
            requested_size[1].min((monitor_size.y - 48.0).max(190.0)),
        ];
        let mut viewport = egui::ViewportBuilder::default()
            .with_title("cHiDeScaler-Neo Mini Dialog")
            .with_decorations(false)
            .with_always_on_top()
            .with_resizable(false)
            .with_taskbar(false)
            .with_inner_size(size);
        if self.mini_dialog_hwnd == 0 {
            let z = ctx.zoom_factor();
            if let Some(outer) = ctx.input(|input| input.viewport().outer_rect) {
                let center = outer.center();
                viewport = viewport.with_position([
                    (center.x * z - size[0] * 0.5).max(8.0),
                    (center.y * z - size[1] * 0.5).max(8.0),
                ]);
            }
        }

        let mut close_notice = false;
        let mut trt_stop = false;
        let mut elevated_restart = false;
        let mut elevated_cancel = false;
        let mut resize_apply = false;
        let mut resize_cancel = false;
        let mut resize_value = self
            .resize_scale_editor
            .map(|(_, value)| value)
            .unwrap_or(0.75);
        let mut hotkey_save = false;
        let mut hotkey_cancel = false;
        let mut filter_add: Option<(StageKind, String)> = None;
        let mut filter_cancel = false;
        let mut save_as_overwrite = false;
        let mut save_as_new = false;
        let mut save_as_cancel = false;
        let mut delete_confirm = false;
        let mut delete_cancel = false;

        let elevated_snapshot = self.elevated_target_notice.clone();
        let trt_snapshot = trt_progress.clone();
        let filter_tree = (kind == Kind::FilterPicker).then(|| build_filter_tree(&self.available));

        ctx.show_viewport_immediate(
            egui::ViewportId::from_hash_of("neo_mini_dialog_host"),
            viewport,
            |ctx2, _| {
                let dialog_ctx = ctx2.ctx().clone();
                let frame = egui::Frame::new()
                    .fill(egui::Color32::from_rgb(16, 18, 22))
                    .stroke(egui::Stroke::new(1.0, egui::Color32::from_rgb(72, 84, 100)))
                    .corner_radius(egui::CornerRadius::same(7))
                    .inner_margin(egui::Margin::same(12));
                egui::CentralPanel::default().frame(frame).show(ctx2, |ui| {
                    ui.set_min_size(egui::vec2(size[0] - 24.0, size[1] - 24.0));
                    ui.label(
                        egui::RichText::new(display_title)
                            .family(locale_font_family(lang))
                            .strong()
                            .size(13.0),
                    );
                    ui.separator();

                    let has_footer = true;
                    let footer_reserve = 50.0;
                    let body_height = (ui.available_height() - footer_reserve).max(56.0);
                    egui::ScrollArea::vertical()
                        .id_salt("mini_dialog_body")
                        .max_height(body_height)
                        .auto_shrink([false, true])
                        .show(ui, |ui| match kind {
                            Kind::TensorRt => {
                                if let Some((model, elapsed, completed, total, model_active)) =
                                    trt_snapshot.as_ref()
                                {
                                    ui.horizontal(|ui| {
                                        ui.spinner();
                                        ui.label(if *model_active {
                                            tr(
                                                lang,
                                                "現在のONNX用エンジンを作成しています。",
                                                "Building the engine for the current ONNX model.",
                                            )
                                        } else {
                                            tr(
                                                lang,
                                                "次のONNXの処理開始を待っています。",
                                                "Waiting for the next ONNX model to begin processing.",
                                            )
                                        });
                                    });
                                    ui.add_space(8.0);
                                    let model_label = if model.to_ascii_lowercase().ends_with(".onnx") {
                                        model.clone()
                                    } else {
                                        format!("{model}.onnx")
                                    };
                                    ui.label(egui::RichText::new(model_label).monospace().size(11.0));
                                    ui.label(format!(
                                        "{}: {}/{}",
                                        tr(lang, "エンジン", "Engine"),
                                        (*completed + 1).min(*total),
                                        total
                                    ));
                                    if *model_active {
                                        ui.label(format!(
                                            "{}: {:.1}s",
                                            tr(lang, "このエンジンの経過時間", "Current engine elapsed"),
                                            elapsed.as_secs_f64()
                                        ));
                                    }
                                    if chidescaler_neo::render::onnx_stage::tensorrt_cancel_requested() {
                                        ui.add_space(6.0);
                                        ui.label(tr(
                                            lang,
                                            "停止要求を受け付けました。現在のTensorRT処理が完了し次第停止します。",
                                            "Stop requested. Capture will close after the current TensorRT operation finishes.",
                                        ));
                                    }
                                }
                            }
                            Kind::Elevated => {
                                if let Some((_, title)) = elevated_snapshot.as_ref() {
                                    ui.label(tr(
                                        lang,
                                        "選択したウィンドウは管理者権限で動作しています。",
                                        "The selected window is running with administrator permission.",
                                    ));
                                    ui.label(egui::RichText::new(title).strong());
                                    ui.add_space(4.0);
                                    ui.label(tr(
                                        lang,
                                        "このウィンドウを拡大・操作するには、cHiDeScaler-Neoも管理者として再起動してください。再起動後、対象ウィンドウをもう一度選択してください。",
                                        "To capture and control this window, restart cHiDeScaler-Neo as administrator, then select the window again.",
                                    ));
                                }
                            }
                            Kind::FullscreenCaptureNotice => {
                                ui.label(i18n::text(lang, "capture.fullscreen_notice_body"));
                            }
                            Kind::ResizeScale => {
                                ui.horizontal(|ui| {
                                    ui.label(tr(lang, "倍率", "Scale"));
                                    ui.add(
                                        egui::DragValue::new(&mut resize_value)
                                            .range(0.25..=4.0)
                                            .speed(0.01)
                                            .fixed_decimals(2),
                                    );
                                });
                                ui.label(tr(
                                    lang,
                                    "0.25～4.00（初期値 0.75）",
                                    "0.25–4.00 (default 0.75)",
                                ));
                            }
                            Kind::Hotkey => {
                                let captured = capture_hotkey_candidate(&dialog_ctx);
                                let captured_this_frame = captured.is_some();
                                if let Some(pending) = captured {
                                    self.hotkey_capture_pending = Some(pending);
                                    self.hotkey_editor_error = None;
                                }
                                let released = self
                                    .hotkey_capture_pending
                                    .as_ref()
                                    .is_some_and(|(_, key)| {
                                        dialog_ctx.input(|input| {
                                            !input.key_down(*key) && input.modifiers.is_none()
                                        })
                                    });
                                if released
                                    && let Some((candidate, _)) = self.hotkey_capture_pending.take()
                                {
                                    match validate_user_hotkey(&candidate) {
                                        Ok(candidate) => {
                                            self.hotkey_editor_candidate = candidate;
                                            self.hotkey_editor_error = None;
                                        }
                                        Err(error) => {
                                            self.hotkey_editor_error =
                                                Some(hotkey_error_text(lang, &error));
                                        }
                                    }
                                }
                                if dialog_ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
                                    hotkey_cancel = true;
                                }
                                let held = held_hotkey_modifiers(&dialog_ctx);
                                ui.label(tr(
                                    lang,
                                    "Ctrl / Alt / Shiftを押しながら、文字・数字・Fキーなどを押してください。",
                                    "Hold Ctrl, Alt, or Shift and press a letter, number, function key, or navigation key.",
                                ));
                                ui.add_space(10.0);
                                let pending_display = self
                                    .hotkey_capture_pending
                                    .as_ref()
                                    .map(|(candidate, _)| candidate.as_str());
                                let display = if let Some(candidate) = pending_display {
                                    candidate
                                } else if !captured_this_frame && !held.is_empty() {
                                    held.as_str()
                                } else {
                                    self.hotkey_editor_candidate.as_str()
                                };
                                hotkey_chips(ui, display);
                                ui.add_space(8.0);
                                if let Some(error) = &self.hotkey_editor_error {
                                    ui.colored_label(egui::Color32::LIGHT_RED, error);
                                } else {
                                    ui.label(
                                        egui::RichText::new(tr(
                                            lang,
                                            "合計2～3キー。修飾キーのみ、Winキー、予約済みキーは使用できません。",
                                            "Two or three keys total. Modifier-only, Win-key, and reserved shortcuts are blocked.",
                                        ))
                                        .size(10.5)
                                        .weak(),
                                    );
                                }
                            }
                            Kind::FilterPicker => {
                                if let Some(tree) = filter_tree.as_ref() {
                                    render_filter_tree_picker(
                                        ui,
                                        tree,
                                        "mini-filter-picker",
                                        lang,
                                        &mut filter_add,
                                    );
                                }
                            }
                            Kind::SaveAs => {
                                let edit = ui.text_edit_singleline(&mut self.save_as_name);
                                if edit.changed() {
                                    self.save_as_error = None;
                                }
                                if let Some(error) = self.save_as_error {
                                    let message = match error {
                                        PresetEditError::EmptyName => tr(
                                            lang,
                                            "名前を入力してください。",
                                            "Enter a preset name.",
                                        ),
                                        PresetEditError::NameInUse => tr(
                                            lang,
                                            "同じ名前のプリセットが既にあります。",
                                            "A preset with this name already exists.",
                                        ),
                                        PresetEditError::ActivePresetMissing => tr(
                                            lang,
                                            "選択中のプリセットが見つかりません。",
                                            "The selected preset could not be found.",
                                        ),
                                    };
                                    ui.colored_label(egui::Color32::LIGHT_RED, message);
                                }
                            }
                            Kind::DeletePreset => {
                                ui.label(
                                    i18n::text(lang, "preset.delete_confirm")
                                        .replace("{name}", &self.store.data.active),
                                );
                            }
                        });

                    if has_footer {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(4.0);
                        ui.horizontal_centered(|ui| match kind {
                            Kind::TensorRt => {
                                if chidescaler_neo::render::onnx_stage::tensorrt_cancel_requested() {
                                    ui.add_enabled_ui(false, |ui| {
                                        let _ = control_row_button(ui, tr(lang, "停止", "Stop"));
                                    });
                                } else if control_row_button(ui, tr(lang, "停止", "Stop")).clicked() {
                                    trt_stop = true;
                                }
                            }
                            Kind::Elevated => {
                                if control_row_button(
                                    ui,
                                    tr(lang, "管理者として再起動", "Restart as administrator"),
                                )
                                .clicked()
                                {
                                    elevated_restart = true;
                                }
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    elevated_cancel = true;
                                }
                            }
                            Kind::FullscreenCaptureNotice => {
                                if control_row_button(ui, "OK").clicked() {
                                    close_notice = true;
                                }
                            }
                            Kind::ResizeScale => {
                                if control_row_button(ui, tr(lang, "適用", "Apply")).clicked() {
                                    resize_apply = true;
                                }
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    resize_cancel = true;
                                }
                            }
                            Kind::Hotkey => {
                                if control_row_button(ui, tr(lang, "保存", "Save")).clicked() {
                                    hotkey_save = true;
                                }
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    hotkey_cancel = true;
                                }
                            }
                            Kind::FilterPicker => {
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    filter_cancel = true;
                                }
                            }
                            Kind::SaveAs => {
                                if control_row_button(ui, tr(lang, "上書き保存", "Overwrite")).clicked() {
                                    save_as_overwrite = true;
                                }
                                if control_row_button(ui, tr(lang, "別名で保存", "Save As")).clicked() {
                                    save_as_new = true;
                                }
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    save_as_cancel = true;
                                }
                            }
                            Kind::DeletePreset => {
                                if control_row_button(ui, tr(lang, "削除", "Delete")).clicked() {
                                    delete_confirm = true;
                                }
                                if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                                    delete_cancel = true;
                                }
                            }
                        });
                    }
                });
            },
        );

        // Discover the single retained viewport after creation and make it a
        // normal interactive Neo-owned topmost window. Cursor ownership can then
        // use the existing process-owned HWND fallback without adding another
        // special cursor state machine or altering the v350g/v361/v362 contract.
        if self.mini_dialog_hwnd == 0
            && let Some(hwnd) = win32::find_own_window("cHiDeScaler-Neo Mini Dialog")
        {
            self.mini_dialog_hwnd = hwnd;
            win32::set_window_input_passthrough(hwnd, false);
            log::info!("mini-dialog-host: hwnd={hwnd:#x} registered cursor_route=own-window");
        }
        if self.mini_dialog_hwnd != 0 {
            if !win32::is_window_valid(self.mini_dialog_hwnd) {
                self.mini_dialog_hwnd = 0;
            } else {
                if !win32::is_topmost(self.mini_dialog_hwnd) {
                    win32::set_own_topmost(self.mini_dialog_hwnd, true);
                }
                win32::set_window_input_passthrough(self.mini_dialog_hwnd, false);
                if running
                    && status.overlay_hwnd != 0
                    && !win32::window_is_above(self.mini_dialog_hwnd, status.overlay_hwnd)
                {
                    win32::raise_topmost(self.mini_dialog_hwnd);
                    chidescaler_neo::input::keep_cursor_sprite_on_top();
                }
            }
        }

        match kind {
            Kind::TensorRt => {
                if trt_stop {
                    let _ = self.request_capture_stop("tensorrt-progress-dialog");
                }
                ctx.request_repaint_after(Duration::from_millis(100));
            }
            Kind::Elevated => {
                if elevated_restart {
                    if let Some((hwnd, _)) = elevated_snapshot {
                        self.settings.run_as_admin = true;
                        save_settings(&self.app_dir, &self.settings);
                        log::info!(
                            "elevated-target-guidance accepted: hwnd={hwnd:#x}; relaunching as administrator"
                        );
                        if win32::relaunch_as_admin() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        } else {
                            self.engine.status.lock().unwrap().last_error = Some(
                                tr(
                                    lang,
                                    "管理者として再起動できませんでした。Windowsの確認画面で許可してから、もう一度お試しください。",
                                    "Could not restart as administrator. Allow the Windows confirmation prompt and try again.",
                                )
                                .to_string(),
                            );
                        }
                    }
                    self.elevated_target_notice = None;
                } else if elevated_cancel {
                    self.elevated_target_notice = None;
                }
            }
            Kind::FullscreenCaptureNotice => {
                if close_notice {
                    self.capture_resolution_fullscreen_notice_open = false;
                }
            }
            Kind::ResizeScale => {
                if let Some((index, _)) = self.resize_scale_editor {
                    let value = resize_value.clamp(0.25, 4.0);
                    if resize_apply && index < self.chain.len() {
                        self.chain[index]
                            .params
                            .insert("RESIZE_SCALE".to_string(), value);
                        log::info!(
                            "resize-scale-edit: index={index} value={value:.2} path={}",
                            self.chain[index].path
                        );
                        self.resize_scale_editor = None;
                        self.apply_live();
                    } else if resize_cancel {
                        self.resize_scale_editor = None;
                    } else {
                        self.resize_scale_editor = Some((index, value));
                    }
                }
            }
            Kind::Hotkey => {
                if hotkey_save {
                    let candidate = self.hotkey_editor_candidate.clone();
                    match self.commit_toggle_hotkey(&candidate, lang) {
                        Ok(()) => {
                            self.hotkey_editor_open = false;
                            self.hotkey_editor_error = None;
                            self.hotkey_capture_pending = None;
                        }
                        Err(error) => self.hotkey_editor_error = Some(error),
                    }
                } else if hotkey_cancel {
                    self.hotkey_editor_open = false;
                    self.hotkey_editor_candidate = self.settings.hotkey_toggle.clone();
                    self.hotkey_editor_error = None;
                    self.hotkey_capture_pending = None;
                }
            }
            Kind::FilterPicker => {
                if let Some((stage_kind, path)) = filter_add {
                    self.filter_picker_open = false;
                    self.add_filter_to_chain(stage_kind, path);
                } else if filter_cancel {
                    self.filter_picker_open = false;
                    log::info!("filter-picker-close: cancelled=true source=mini-dialog-host");
                }
            }
            Kind::SaveAs => {
                if save_as_overwrite {
                    let aspect_correction = self.current_preset_aspect_correction();
                    let crop = self.current_preset_crop();
                    let capture_resolution = self.current_preset_capture_resolution();
                    match self.store.overwrite_active_as(
                        &self.save_as_name,
                        &self.chain,
                        aspect_correction,
                        crop,
                        capture_resolution,
                    ) {
                        Ok(name) => {
                            self.saved_chain = self.chain.clone();
                            self.saved_aspect_correction = aspect_correction;
                            self.saved_crop = crop;
                            self.saved_capture_resolution = capture_resolution;
                            self.store.save();
                            self.save_as_open = false;
                            self.save_as_error = None;
                            log::info!(
                                "preset-overwrite: name={name} aspect_enabled={} aspect_scale={:.2}x{:.2}",
                                aspect_correction.enabled,
                                aspect_correction.width_scale,
                                aspect_correction.height_scale
                            );
                        }
                        Err(error) => self.save_as_error = Some(error),
                    }
                } else if save_as_new {
                    let aspect_correction = self.current_preset_aspect_correction();
                    let crop = self.current_preset_crop();
                    let capture_resolution = self.current_preset_capture_resolution();
                    match self.store.save_as_new(
                        &self.save_as_name,
                        &self.chain,
                        aspect_correction,
                        crop,
                        capture_resolution,
                    ) {
                        Ok(name) => {
                            self.saved_chain = self.chain.clone();
                            self.saved_aspect_correction = aspect_correction;
                            self.saved_crop = crop;
                            self.saved_capture_resolution = capture_resolution;
                            self.store.save();
                            self.save_as_open = false;
                            self.save_as_error = None;
                            log::info!(
                                "preset-save-as: name={name} aspect_enabled={} aspect_scale={:.2}x{:.2}",
                                aspect_correction.enabled,
                                aspect_correction.width_scale,
                                aspect_correction.height_scale
                            );
                        }
                        Err(error) => self.save_as_error = Some(error),
                    }
                } else if save_as_cancel {
                    self.save_as_open = false;
                    self.save_as_error = None;
                }
            }
            Kind::DeletePreset => {
                if delete_confirm {
                    let active = self.store.data.active.clone();
                    self.store
                        .data
                        .presets
                        .retain(|preset| preset.name != active);
                    if let Some(first) = self.store.data.presets.first() {
                        let name = first.name.clone();
                        self.select_preset(&name);
                    }
                    self.store.save();
                    self.confirm_delete = false;
                } else if delete_cancel {
                    self.confirm_delete = false;
                }
            }
        }
    }
}

impl eframe::App for App {
    fn ui(&mut self, root: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = root.ctx().clone();
        let ctx = &ctx;
        self.commit_pending_ui_mode_if_ready(ctx);
        // Diagnostic sampling stays active regardless of UI mode/stats layout.
        // It does not change the GPU bar calculation; ResourceMonitor logs the
        // same legacy value plus raw per-engine/per-process evidence so GPU
        // behavior can be diagnosed without external screenshots.
        let _ = self.resource_monitor.sample();
        self.maybe_finalize_gpu_selection_startup();
        if let Some(path) = self.gui_test_screenshot_path.clone() {
            let screenshot = ctx.input(|input| {
                input.events.iter().find_map(|event| match event {
                    egui::Event::Screenshot { image, .. } => Some(image.clone()),
                    _ => None,
                })
            });
            if let Some(image) = screenshot {
                match save_gui_screenshot(&path, &image) {
                    Ok(()) => log::info!("gui-test-screenshot: saved={}", path.display()),
                    Err(error) => log::error!("gui-test-screenshot failed: {error:#}"),
                }
                self.gui_test_screenshot_path = None;
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
        if self.gui_hwnd == 0 {
            use raw_window_handle::{HasWindowHandle, RawWindowHandle};
            if let Ok(h) = frame.window_handle() {
                if let RawWindowHandle::Win32(w) = h.as_raw() {
                    self.gui_hwnd = w.hwnd.get();
                    win32::set_dark_title_bar(self.gui_hwnd);
                    win32::disable_native_maximize_button(self.gui_hwnd);
                    win32::install_main_gui_native_minimize(self.gui_hwnd);
                }
            }
        }
        // WS_EX_TRANSPARENT alone does not reliably suppress winit/egui hover
        // delivery when the hidden source cursor physically overlaps this
        // topmost window. Keep the framework's native hit-test state tied to
        // the input owner's visible cursor contract as well.
        let gui_mouse_passthrough = chidescaler_neo::input::main_gui_mouse_passthrough();
        let native_passthrough_matches = self.gui_hwnd == 0
            || win32::window_input_passthrough(self.gui_hwnd) == gui_mouse_passthrough;
        if self.gui_mouse_passthrough_applied != Some(gui_mouse_passthrough)
            || !native_passthrough_matches
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(
                gui_mouse_passthrough,
            ));
            self.gui_mouse_passthrough_applied = Some(gui_mouse_passthrough);
        }
        // Defense in depth for Win+Up, title-bar double click, Snap and stale
        // settings from older builds. The button is absent at creation, and a
        // platform-level maximize request is immediately returned to normal.
        if self.gui_hwnd != 0 && win32::restore_if_maximized(self.gui_hwnd) {
            log::info!("gui-maximize-request-blocked: restored normal window");
        }
        // apply the GUI always-on-top option
        if self.gui_hwnd != 0 && self.gui_topmost_applied != Some(self.settings.gui_topmost) {
            win32::set_own_topmost(self.gui_hwnd, self.settings.gui_topmost);
            self.gui_topmost_applied = Some(self.settings.gui_topmost);
        }
        // Global hotkeys. Capture the complete configured binding and reception
        // time on the RegisterHotKey thread so even a badly stalled GUI can
        // distinguish a fresh Start request from a Stop keypress that sat in
        // the queue until after shutdown completed.
        let hotkey_status = self.engine.status.lock().unwrap().clone();
        let hotkey_provider_preparing =
            chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
        let capture_busy_now = hotkey_status.starting
            || hotkey_status.running
            || hotkey_status.stopping
            || hotkey_provider_preparing;
        let capture_became_idle = self.was_capture_busy && !capture_busy_now;
        if capture_became_idle {
            self.capture_idle_since = Instant::now();
            // A confirmed ordinary busy->idle transition has completed the
            // proven source restoration path. The janitor must not later undo
            // user changes made to that source while Neo remains open.
            chidescaler_neo::input::clear_janitor_source_recovery();
            log::debug!(
                "capture-idle-epoch-advanced: stale queued toggle hotkeys can no longer start capture"
            );
        }
        self.was_capture_busy = capture_busy_now;

        // v348u: Start and Stop now share the same physical pointer-DOWN path.
        // This removes the v348t asymmetry where Stop fired on DOWN but Start
        // depended on a later egui release that could be swallowed or lost.
        let main_actions = chidescaler_neo::input::take_main_actions();
        if main_actions != 0 {
            let now = Instant::now();
            self.main_control_press_until = Some(now + Duration::from_millis(115));
        }
        if main_actions & chidescaler_neo::input::MAIN_ACTION_STOP != 0 {
            let current = self.engine.status.lock().unwrap().clone();
            let provider_preparing = chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
            if current.starting || current.running || provider_preparing {
                log::info!("main-capture-control: action=stop trigger=win32-lockfree-down");
                let _ = self.request_capture_stop("main-gui-direct");
            } else {
                log::debug!("main-capture-control: action=stop result=stale-direct-click-ignored");
            }
        }
        if main_actions & chidescaler_neo::input::MAIN_ACTION_START != 0 {
            let current = self.engine.status.lock().unwrap().clone();
            let provider_preparing = chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
            if !current.starting && !current.running && !current.stopping && !provider_preparing {
                log::info!("main-capture-control: action=start trigger=win32-lockfree-down");
                let _ = self.request_capture_start("main-gui-direct");
            } else {
                log::debug!(
                    "main-capture-control: action=start result=stale-direct-click-ignored starting={} running={} stopping={} preparing={}",
                    current.starting,
                    current.running,
                    current.stopping,
                    provider_preparing
                );
            }
        }

        let mut toggle_hotkey_handled = false;
        while let Ok(event) = self.hotkeys.rx.try_recv() {
            if event.handled_directly {
                log::info!(
                    "hotkey-dispatch-ignored: id={} binding='{}' reason=already-handled-on-hotkey-thread",
                    event.id,
                    event.binding
                );
                continue;
            }
            if self.hotkey_editor_open {
                log::info!(
                    "hotkey-dispatch-ignored: id={} binding='{}' reason=editor-open",
                    event.id,
                    event.binding
                );
                continue;
            }
            match event.id {
                HK_TOGGLE => {
                    if toggle_hotkey_handled {
                        log::info!(
                            "hotkey-toggle-coalesced: binding='{}' age_ms={:.1} reason=same-ui-drain-batch",
                            event.binding,
                            event.received_at.elapsed().as_secs_f64() * 1000.0
                        );
                    } else {
                        toggle_hotkey_handled = true;
                        self.dispatch_toggle_hotkey(&event);
                    }
                }
                HK_PANEL => {
                    log::info!(
                        "hotkey-dispatch: binding='{}' action=panel-toggle",
                        event.binding
                    );
                    let running_now = self.engine.status.lock().unwrap().running;
                    self.toggle_panel_hotkey(running_now, ctx);
                }
                HK_GUI_TOPMOST => {
                    log::info!(
                        "hotkey-dispatch: binding='{}' action=gui-topmost-toggle",
                        event.binding
                    );
                    self.toggle_gui_topmost();
                }
                HK_QUIT => {
                    log::info!("hotkey-dispatch: binding='{}' action=quit", event.binding);
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                }
                _ => log::warn!(
                    "hotkey-dispatch-ignored: id={} binding='{}' reason=unknown-id",
                    event.id,
                    event.binding
                ),
            }
        }
        // remember the window placement for the next launch.
        // NOTE: viewport rects are in egui (zoomed) points; ViewportBuilder
        // wants OS-logical points — convert with the zoom factor, otherwise
        // the window shrinks by 1/zoom on every launch.
        if self.gui_hwnd == 0 || !win32::is_maximized(self.gui_hwnd) {
            let z = ctx.zoom_factor();
            let vp = ctx.input(|i| i.viewport().clone());
            if let Some(outer) = vp.outer_rect {
                self.settings.win_pos = Some((outer.min.x * z, outer.min.y * z));
            }
            if let Some(inner) = vp.inner_rect {
                let size = Some((inner.width() * z, inner.height() * z));
                match self.settings.ui_mode {
                    UiMode::Mini => self.settings.mini_win_size = size,
                    UiMode::Basic => self.settings.basic_win_size = size,
                    UiMode::Full => self.settings.win_size = size,
                }
            }
        }
        let mut status = self.engine.status.lock().unwrap().clone();

        if status.source_occlusion_notice_seq != self.source_occlusion_notice_seen_seq {
            self.source_occlusion_notice_seen_seq = status.source_occlusion_notice_seq;
            self.source_occlusion_notice_until = Some(Instant::now() + Duration::from_secs(3));
            log::info!(
                "source-occlusion-notice: seq={} action=armed duration_ms=3000",
                status.source_occlusion_notice_seq
            );
        }

        // v555: the existing GLSL overload detector remains the sole authority.
        // Once it has armed a deadline, keep the warning readable for the full
        // grace period and then enter the exact same Stop path as the user's
        // normal Stop button. Engine::send(Cmd::Stop) therefore retains all of
        // the proven immediate cursor/source/overlay recovery and provider
        // cancellation behavior.
        if status.running && !status.stopping {
            if let Some(deadline) = status.glsl_overload_auto_stop_deadline {
                let now = Instant::now();
                if now >= deadline {
                    log::info!(
                        "glsl-overload-auto-stop: deadline-reached action=dispatch-normal-stop"
                    );
                    let _ = self.request_capture_stop("glsl-overload-auto-stop");
                    status = self.engine.status.lock().unwrap().clone();
                } else {
                    // Event-driven GUI modes must still wake at the deadline even
                    // when the pointer is elsewhere or the GUI is behind the overlay.
                    ctx.request_repaint_after((deadline - now).min(Duration::from_millis(100)));
                }
            }
        }

        if status.onnx_backend_revision != self.onnx_backend_revision_seen {
            self.onnx_backend_revision_seen = status.onnx_backend_revision;
            if let Some(requested) = self.onnx_backend_pending.take() {
                if status.onnx_backend_error.is_none() && status.onnx_backend == requested {
                    self.onnx_backend_selected = requested;
                    self.settings.onnx_backend = requested;
                    save_settings(&self.app_dir, &self.settings);
                } else {
                    self.onnx_backend_selected = status.onnx_backend;
                }
            } else {
                self.onnx_backend_selected = status.onnx_backend;
            }
        }
        let running = status.running;
        if !self.was_running && running {
            // The retained native panel host still has the previous session's
            // desktop position. Keep it hidden until control_panel commits
            // the new overlay-relative rectangle with its final pixel size.
            self.panel_placed_for_run = false;
            self.panel_gdi_reveal_pending = false;
            self.panel_state_sent = None;
            self.panel_layout_sent = None;
            if self.panel_hwnd != 0
                && win32::is_window_valid(self.panel_hwnd)
                && win32::is_own_window(self.panel_hwnd)
            {
                win32::set_window_alpha(self.panel_hwnd, 0);
                win32::set_window_input_passthrough(self.panel_hwnd, true);
                if win32::is_panel_gdi_host(self.panel_hwnd) {
                    win32::set_panel_gdi_host_visible(self.panel_hwnd, false);
                }
            }
        }
        if self.was_running && !running {
            // v539 TOPMOST-OFF keeps the root GUI DWM-cloaked for the whole
            // running interval. Once the overlay session ends, release only
            // that owned cloak and restore the ordinary non-topmost GUI.
            if self.gui_topmost_off_cloak_applied
                && self.gui_hwnd != 0
                && win32::is_window_valid(self.gui_hwnd)
            {
                let reveal_ok = win32::set_window_cloaked(self.gui_hwnd, false);
                let still_cloaked = win32::is_cloaked(self.gui_hwnd);
                self.gui_topmost_off_cloak_applied = still_cloaked;
                log::debug!(
                    "GUI topmost off cloak release on capture stop: gui={:#x} request_ok={} still_cloaked={}",
                    self.gui_hwnd,
                    reveal_ok,
                    still_cloaked
                );
            }
            // A fullscreen source may re-assert its own ClipCursor while it is
            // foreground. Put our topmost GUI back in the foreground first,
            // then perform a final process-wide cursor release.
            if self.settings.gui_topmost && self.gui_hwnd != 0 {
                win32::activate_window(self.gui_hwnd);
            }
            chidescaler_neo::input::emergency_release_all();
            // There are two independent passthrough owners: our direct Win32
            // recovery and eframe/winit's queued ViewportCommand. The native repair
            // guaranteed the former. If a stale MousePassthrough(true) command
            // lands just after Stop, an idle event-driven GUI receives no mouse
            // event and therefore cannot repair itself. Explicitly restore BOTH
            // layers on the running->idle edge and schedule a follow-up repaint.
            if self.gui_hwnd != 0 && win32::is_window_valid(self.gui_hwnd) {
                win32::set_window_input_passthrough(self.gui_hwnd, false);
                ctx.send_viewport_cmd(egui::ViewportCommand::MousePassthrough(false));
                // Force the scheduled follow-up repaint to re-send the same
                // framework command once more after this transition frame.
                self.gui_mouse_passthrough_applied = None;
                ctx.request_repaint();
                ctx.request_repaint_after(Duration::from_millis(50));
                log::info!(
                    "gui-stop-route-recovery: native_interactive=true viewport_passthrough=false repaint_guard_ms=50"
                );
            }
            log::info!("capture stopped: auxiliary windows closed and cursor state released");
        }
        self.was_running = running;
        self.poll(running);
        let lang = self.effective_language();
        if self.locale_test_override.is_none() {
            self.settings.language = lang;
        }

        // v555: when the root GUI is not physically visible above the video
        // overlay (GUI topmost OFF/cloaked, minimized, or simply behind the
        // windowed overlay), mirror the same overload-stop notice directly on
        // the magnified content using a cached native GDI helper. This does not
        // create another WGPU surface and therefore preserves v551's low-spec
        // corruption fix. content_rect is physical desktop geometry for both
        // fullscreen and windowed magnification, so moves/resizes naturally
        // reposition the notice without touching WGC/GLSL/Present.
        let root_gui_notice_visible = self.gui_hwnd != 0
            && win32::is_window_valid(self.gui_hwnd)
            && win32::is_window_visible(self.gui_hwnd)
            && !win32::is_minimized(self.gui_hwnd)
            && !win32::is_cloaked(self.gui_hwnd)
            && status.overlay_hwnd != 0
            && win32::window_is_above(self.gui_hwnd, status.overlay_hwnd);
        if running
            && !status.stopping
            && status.glsl_overload_auto_stop_deadline.is_some()
            && !root_gui_notice_visible
        {
            win32::show_overload_notice_gdi(
                status.overlay_hwnd,
                status.content_rect,
                i18n::text(lang, "glsl_overload.auto_stop"),
            );
        } else {
            win32::hide_overload_notice_gdi();
        }

        // Low-spec GLSL protection is intentionally visible. v555 keeps the
        // established overload thresholds unchanged, but a proven overload now
        // enters a six-second readable warning grace and then the ordinary Stop route. During
        // the grace the engine holds the last complete filtered frame whenever
        // that is semantics-safe, so cursor/GUI/DWM recovery gets GPU headroom.
        if self.settings.ui_mode != UiMode::Mini && running && status.glsl_overload_notice_latched {
            egui::Window::new("glsl-overload-pause-notice")
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 58.0))
                .collapsible(false)
                .resizable(false)
                .movable(false)
                .title_bar(false)
                .frame(
                    egui::Frame::new()
                        .fill(egui::Color32::from_rgb(34, 29, 18))
                        .stroke(egui::Stroke::new(
                            1.0,
                            egui::Color32::from_rgb(214, 164, 66),
                        ))
                        .corner_radius(6.0)
                        .inner_margin(12.0),
                )
                .show(ctx, |ui| {
                    ui.set_max_width(580.0);
                    ui.label(
                        egui::RichText::new(i18n::text(lang, "glsl_overload.auto_stop"))
                            .family(locale_font_family(lang))
                            .strong(),
                    );
                });
        }

        if self.settings.ui_mode != UiMode::Mini
            && let Some((model, elapsed, completed, total, model_active)) =
                chidescaler_neo::render::onnx_stage::tensorrt_build_progress()
        {
            let cancel_requested = chidescaler_neo::render::onnx_stage::tensorrt_cancel_requested();
            egui::Window::new(tr(
                lang,
                "TensorRT用エンジンを作成中",
                "Preparing TensorRT engine",
            ))
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(8, 10, 13))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(78, 126, 176),
                    ))
                    .corner_radius(6.0)
                    .inner_margin(14.0),
            )
            .show(ctx, |ui| {
                ui.set_min_width(390.0);
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(if model_active {
                        tr(
                            lang,
                            "現在のONNX用エンジンを作成しています。",
                            "Building the engine for the current ONNX model.",
                        )
                    } else {
                        tr(
                            lang,
                            "次のONNXの処理開始を待っています。",
                            "Waiting for the next ONNX model to begin processing.",
                        )
                    });
                });
                ui.add_space(8.0);
                let model_label = if model.to_ascii_lowercase().ends_with(".onnx") {
                    model
                } else {
                    format!("{model}.onnx")
                };
                ui.label(egui::RichText::new(model_label).monospace().size(11.0));
                ui.label(format!(
                    "{}: {}/{}",
                    tr(lang, "エンジン", "Engine"),
                    (completed + 1).min(total),
                    total
                ));
                if model_active {
                    ui.label(format!(
                        "{}: {:.1}s",
                        tr(lang, "このエンジンの経過時間", "Current engine elapsed"),
                        elapsed.as_secs_f64()
                    ));
                }
                ui.add_space(6.0);
                if cancel_requested {
                    ui.label(tr(
                        lang,
                        "停止要求を受け付けました。現在のTensorRT処理が完了し次第停止します。",
                        "Stop requested. Capture will close after the current TensorRT operation finishes.",
                    ));
                } else if control_row_button(ui, tr(lang, "停止", "Stop")).clicked() {
                    let _ = self.request_capture_stop("tensorrt-progress-dialog");
                }
            });
            ctx.request_repaint_after(Duration::from_millis(100));
        }
        let mini_ui = self.settings.ui_mode == UiMode::Mini;
        let basic_ui = self.settings.ui_mode == UiMode::Basic;
        let full_ui = self.settings.ui_mode == UiMode::Full;
        // QA-only transition path: start with statistics off, then enable it
        // without resetting layout state. This catches stale resize keys that
        // a screenshot launched directly in stats-on mode cannot detect.
        if full_ui
            && std::env::var_os("NEO_GUI_TEST_TOGGLE_STATS").is_some()
            && self.gui_test_frame_count >= 5
            && !self.settings.stats_on
        {
            self.settings.stats_on = true;
            self.engine.metrics.set_enabled(true);
            log::info!("gui-test: toggled Full statistics on after initial layout");
        }
        let basic_stats_rows = self.basic_stats_rows();
        if basic_ui
            && (self.basic_stats_layout_applied != Some(self.settings.stats_on)
                || self.basic_stats_rows_applied != Some(basic_stats_rows))
        {
            self.resize_basic_for_stats(ctx);
            self.basic_stats_layout_applied = Some(self.settings.stats_on);
            self.basic_stats_rows_applied = Some(basic_stats_rows);
        }

        // ---------- top bar ----------
        let mut language_changed = false;
        egui::Panel::top("top").show(root, |ui| {
            if mini_ui {
                self.resize_mini_for_language(ui, lang, false);
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let language_response = ui
                        .scope(|ui| {
                            ui.set_width(96.0);
                            ui.spacing_mut().button_padding = egui::vec2(8.0, 3.0);
                            ui.spacing_mut().interact_size.y = 23.0;
                            let visuals = ui.visuals_mut();
                            let language_bg = egui::Color32::from_rgb(27, 32, 39);
                            let language_hover = egui::Color32::from_rgb(36, 42, 52);
                            let language_stroke =
                                egui::Stroke::new(1.0, egui::Color32::from_rgb(92, 118, 145));
                            visuals.widgets.inactive.bg_fill = language_bg;
                            visuals.widgets.inactive.weak_bg_fill = language_bg;
                            visuals.widgets.inactive.bg_stroke = language_stroke;
                            visuals.widgets.hovered.bg_fill = language_hover;
                            visuals.widgets.hovered.weak_bg_fill = language_hover;
                            visuals.widgets.hovered.bg_stroke = language_stroke;
                            visuals.widgets.active.bg_fill = language_hover;
                            visuals.widgets.active.weak_bg_fill = language_hover;
                            visuals.widgets.active.bg_stroke = language_stroke;
                            visuals.widgets.inactive.corner_radius = egui::CornerRadius::same(11);
                            visuals.widgets.hovered.corner_radius = egui::CornerRadius::same(11);
                            visuals.widgets.active.corner_radius = egui::CornerRadius::same(11);
                            let active_language =
                                if self.settings.language_mode == UiLanguageMode::Auto {
                                    "Auto".to_owned()
                                } else if let Some(custom) =
                                    self.settings.custom_language.as_ref().and_then(|tag| {
                                        self.custom_locales
                                            .iter()
                                            .find(|locale| locale.tag.eq_ignore_ascii_case(tag))
                                    })
                                {
                                    custom.short.chars().take(4).collect::<String>()
                                } else {
                                    i18n::short_name(lang).to_owned()
                                };
                            let (menu_response, _) = egui::menu::MenuButton::from_button(
                                egui::Button::new("").min_size(egui::vec2(88.0, 23.0)),
                            )
                            .ui(ui, |ui| {
                                let popup_height = language_popup_height(self.settings.ui_mode);
                                let popup_width = if self.settings.ui_mode == UiMode::Mini {
                                    148.0
                                } else {
                                    188.0
                                };
                                ui.set_min_width(popup_width);
                                ui.set_max_width(popup_width);
                                // menu_button otherwise permits its popup child
                                // to collapse to egui's ~2-row default height.
                                // Pin both bounds so Full/Basic cannot inherit
                                // Mini's compact viewport after any mode switch.
                                ui.set_min_height(popup_height);
                                ui.set_max_height(popup_height);
                                // Mini deliberately remains compact. The
                                // language list itself scrolls so all current
                                // and future locales remain reachable.
                                egui::ScrollArea::vertical()
                                    .id_salt("language_picker_scroll")
                                    .max_height(popup_height)
                                    .min_scrolled_height(popup_height)
                                    .auto_shrink([false, false])
                                    .show(ui, |ui| {
                                        // Compact rows let Full/Basic show
                                        // every currently supported language.
                                        // Mini still exposes two rows and
                                        // scrolls the same complete list.
                                        ui.spacing_mut().item_spacing.y = 2.0;
                                        let auto_language =
                                            effective_language(UiLanguageMode::Auto);
                                        if language_choice_row(
                                            ui,
                                            self.settings.language_mode == UiLanguageMode::Auto,
                                            "Auto (Windows)",
                                            UiLanguage::EnUs,
                                            "",
                                            auto_language,
                                        )
                                        .clicked()
                                        {
                                            self.settings.language_mode = UiLanguageMode::Auto;
                                            self.settings.custom_language = None;
                                            i18n::activate_custom_locale(None);
                                            language_changed = true;
                                            ui.close();
                                        }
                                        ui.separator();
                                        for candidate in i18n::ALL_LANGUAGES {
                                            let mode = i18n::fixed_mode(candidate);
                                            let code = format!("[{}]", i18n::short_name(candidate));
                                            if language_choice_row(
                                                ui,
                                                self.settings.custom_language.is_none()
                                                    && self.settings.language_mode == mode,
                                                i18n::native_name(candidate),
                                                candidate,
                                                &code,
                                                UiLanguage::EnUs,
                                            )
                                            .clicked()
                                            {
                                                self.settings.language_mode = mode;
                                                self.settings.custom_language = None;
                                                i18n::activate_custom_locale(None);
                                                language_changed = true;
                                                ui.close();
                                            }
                                        }
                                        for custom in &self.custom_locales {
                                            let code = format!("[{}]", custom.short);
                                            if language_choice_row(
                                                ui,
                                                self.settings.custom_language.as_ref().is_some_and(
                                                    |tag| tag.eq_ignore_ascii_case(&custom.tag),
                                                ),
                                                &custom.name,
                                                UiLanguage::EnUs,
                                                &code,
                                                UiLanguage::EnUs,
                                            )
                                            .clicked()
                                            {
                                                self.settings.language_mode = UiLanguageMode::EnUs;
                                                self.settings.custom_language =
                                                    Some(custom.tag.clone());
                                                i18n::activate_custom_locale(Some(custom));
                                                language_changed = true;
                                                ui.close();
                                            }
                                        }
                                    });
                            });
                            ui.painter().text(
                                menu_response.rect.left_center() + egui::vec2(8.0, 0.0),
                                egui::Align2::LEFT_CENTER,
                                "Language",
                                egui::FontId::proportional(8.5),
                                egui::Color32::from_rgb(225, 232, 238),
                            );
                            let chip_center = egui::pos2(
                                menu_response.rect.right() - 14.0,
                                menu_response.rect.center().y,
                            );
                            ui.painter().circle_filled(
                                chip_center,
                                9.0,
                                egui::Color32::from_rgb(78, 126, 176),
                            );
                            ui.painter().text(
                                chip_center,
                                egui::Align2::CENTER_CENTER,
                                &active_language,
                                egui::FontId::proportional(if active_language == "Auto" {
                                    6.7
                                } else {
                                    8.2
                                }),
                                egui::Color32::WHITE,
                            );
                            menu_response
                        })
                        .inner;
                    if std::env::var_os("NEO_GUI_TEST_LANGUAGE_MENU_OPEN").is_some() {
                        egui::Popup::open_id(
                            ctx,
                            egui::Popup::default_response_id(&language_response),
                        );
                    }
                    language_response.on_hover_text(
                        if let Some(tag) = &self.settings.custom_language {
                            self.custom_locales
                                .iter()
                                .find(|locale| locale.tag.eq_ignore_ascii_case(tag))
                                .map(|locale| format!("{} [{}]", locale.name, locale.tag))
                                .unwrap_or_else(|| tag.clone())
                        } else if self.settings.language_mode == UiLanguageMode::Auto {
                            format!("Auto / {}", i18n::native_name(lang))
                        } else {
                            format!("{} [{}]", i18n::native_name(lang), i18n::tag(lang))
                        },
                    );
                    let selected_lang = self.effective_language();
                    if selected_lang != self.settings.language || language_changed {
                        self.settings.language = selected_lang;
                        select_ui_font_for_locale(ctx, selected_lang);
                        self.capture_resolution_text = capture_resolution_label(
                            self.settings.capture_resolution,
                            selected_lang,
                        );
                        // Test overrides are deliberately ephemeral.
                        if self.locale_test_override.is_none() {
                            save_settings(&self.app_dir, &self.settings);
                        }
                        ctx.request_repaint();
                    }
                    ui.add_space(5.0);
                    if let Some(mode) = ui_mode_pill(ui, self.settings.ui_mode) {
                        self.prepare_ui_mode_switch(ui, lang, mode);
                    }
                });
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                // Size from the actual localized galley so left/right padding
                // remains equal for long French, Portuguese and German text.
                {
                    let provider_preparing =
                        chidescaler_neo::render::onnx_stage::tensorrt_is_preparing();
                    let preparing = status.starting || provider_preparing;
                    if preparing || status.stopping {
                        // Keep the transient button state visibly live while
                        // filters/providers are being prepared or Stop cleanup
                        // is unwinding; idle GUI fallback can otherwise be too
                        // slow to show the state before it has already changed.
                        ctx.request_repaint_after(Duration::from_millis(33));
                    }
                    // Keep the compact two-label control: idle/preparing shows
                    // Start, while a confirmed running/teardown state shows Stop.
                    // v408 changes only the visual latch: active capture is now
                    // indicated by a raised red Stop face rather than a face held down.
                    let label = if running || status.stopping {
                        i18n::text(lang, "capture.running")
                    } else {
                        i18n::text(lang, "capture.start")
                    };
                    let label_text = label.trim_start_matches(|c: char| {
                        c == '▶' || c == '■' || c == '●' || c.is_whitespace()
                    });
                    let font = egui::FontId::new(15.0, locale_font_family(lang));
                    let galley = ui.painter().layout_no_wrap(
                        label_text.to_owned(),
                        font.clone(),
                        egui::Color32::WHITE,
                    );
                    // Use the actual painted glyph bounds, not the font's
                    // advance/line box. CJK and Latin fonts have very
                    // different side bearings and ascenders, so centering
                    // the logical galley makes the visible padding drift.
                    let ink_bounds = galley.mesh_bounds;
                    let group_width = ink_bounds.width();
                    let button_width = (group_width + 44.0).max(150.0);
                    let (rect, resp) = ui
                        .allocate_exact_size(egui::vec2(button_width, 36.0), egui::Sense::click());

                    let now = Instant::now();
                    if self
                        .main_control_press_until
                        .is_some_and(|until| now >= until)
                    {
                        self.main_control_press_until = None;
                    }
                    // v408: keep the tactile push animation, but do not keep
                    // the button mechanically latched while capture is running.
                    // A real pointer/keyboard DOWN still depresses the face for
                    // the existing short 115 ms feedback window; once capture
                    // starts, the face rises again and the red Stop state alone
                    // communicates that capture is active.
                    let pressed = resp.is_pointer_button_down_on()
                        || chidescaler_neo::input::main_control_direct_pressed()
                        || self
                            .main_control_press_until
                            .is_some_and(|until| now < until);
                    if pressed {
                        ctx.request_repaint_after(Duration::from_millis(16));
                    }

                    // Idle/starting remains neutral. Running and Stop cleanup use
                    // a raised red face so the active state is obvious without a
                    // permanently recessed control.
                    let base = if running || status.stopping {
                        egui::Color32::from_rgb(176, 58, 58)
                    } else {
                        egui::Color32::from_rgb(76, 82, 92)
                    };
                    let press_y = if pressed { 3.0 } else { 0.0 };
                    let depth_y = if pressed { 3.8 } else { 5.2 };

                    let depth_rect = rect.translate(egui::vec2(0.0, depth_y));
                    ui.painter().rect_filled(
                        depth_rect,
                        9.0,
                        base.gamma_multiply(0.42),
                    );
                    let face_rect = rect.translate(egui::vec2(0.0, press_y));
                    let fill = if pressed {
                        base.gamma_multiply(0.92)
                    } else if resp.hovered() {
                        base.gamma_multiply(1.15)
                    } else {
                        base
                    };

                    ui.painter().rect_filled(face_rect, 7.4, fill);
                    ui.painter().rect_stroke(
                        face_rect.shrink(0.6),
                        6.8,
                        egui::Stroke::new(
                            0.8,
                            if !pressed {
                                egui::Color32::WHITE.gamma_multiply(0.14)
                            } else {
                                egui::Color32::WHITE.gamma_multiply(0.06)
                            },
                        ),
                        egui::StrokeKind::Inside,
                    );

                    // The text follows the transient physical face so the short
                    // click depression still feels identical to the old button.
                    let text_center = face_rect.center();
                    let galley_pos = text_center - ink_bounds.center().to_vec2();
                    ui.painter()
                        .galley(galley_pos, galley, egui::Color32::WHITE);

                    // Publish the *painted* control rectangle in Win32 client
                    // pixels. Start and Stop both commit on the same physical DOWN edge;
                    // only the published mode changes with the latch state.
                    let control_screen_rect = if self.gui_hwnd != 0 {
                        let ppp = ui.ctx().pixels_per_point().max(0.5);
                        let left = (rect.min.x * ppp).floor() as i32;
                        let top = (rect.min.y * ppp).floor() as i32;
                        let right = (rect.max.x * ppp).ceil() as i32;
                        let bottom = (rect.max.y * ppp).ceil() as i32;
                        Some((left, top, (right - left).max(1), (bottom - top).max(1)))
                    } else {
                        None
                    };
                    // The Start/Stop control is also hit-tested by the global
                    // WH_MOUSE_LL path so it can commit on physical DOWN. An
                    // egui::Window (such as the filter picker) shares the same
                    // native HWND as the main GUI, so Win32 z-order alone cannot
                    // distinguish a click on the foreground picker from the
                    // Start/Stop button underneath it. Disable the lock-free
                    // surface while the picker is open; egui then owns those
                    // clicks exclusively and the surface is republished on the
                    // first frame after the picker closes.
                    let control_mode = if self.filter_picker_open || status.stopping {
                        chidescaler_neo::input::MAIN_CONTROL_DISABLED
                    } else if preparing || running {
                        chidescaler_neo::input::MAIN_CONTROL_STOP
                    } else {
                        chidescaler_neo::input::MAIN_CONTROL_START
                    };
                    chidescaler_neo::input::set_main_control_surface(
                        self.gui_hwnd,
                        control_screen_rect,
                        control_mode,
                    );

                    // Native mouse actions are already committed on physical
                    // pointer-DOWN by WH_MOUSE_LL. Never toggle again from the
                    // matching egui release: that exact duplicate was able to
                    // turn a completed Stop into an unintended Start in v348t.
                    if resp.clicked_by(egui::PointerButton::Primary) {
                        log::debug!(
                            "main-capture-control: action=release result=presentation-only"
                        );
                    }
                    // Keep keyboard activation available without reintroducing
                    // release-based mouse semantics.
                    let keyboard_activate = resp.has_focus()
                        && ui.input(|input| input.key_pressed(egui::Key::Enter));
                    if keyboard_activate {
                        self.main_control_press_until =
                            Some(now + Duration::from_millis(115));
                        if status.stopping {
                            log::info!(
                                "main-capture-control: action=keyboard result=ignored-already-stopping"
                            );
                        } else if preparing || running {
                            let _ = self.request_capture_stop("main-gui-keyboard");
                        } else {
                            let _ = self.request_capture_start("main-gui-keyboard");
                        }
                    }
                }
                ui.separator();
                // preset bar (drag to reorder inside the dropdown)
                let active = self.store.data.active.clone();
                let display = if self.dirty() {
                    format!("{active} *")
                } else {
                    active.clone()
                };
                let mut selected: Option<String> = None;
                let mut moved: Option<(usize, usize)> = None;
                // Keep Mini physically small. Its viewport only exposes about
                // two rows, so its *scroll area's* max height must match that
                // visible opening. Giving it the Full/Basic multi-row extent
                // lets scrolling stop while the final rows are still clipped
                // outside the Mini viewport. Full/Basic use a large,
                // DPI-independent logical height.
                // Leave a few logical points below the final row. At 200%
                // DPI the popup frame consumes slightly more than its 100%
                // counterpart; keep an explicit lower breathing space instead
                // of placing the final button directly on the clip boundary.
                let popup_mode = self.settings.ui_mode;
                let preset_popup_height = preset_popup_height(popup_mode);
                // Popup/scroll memory must not be shared between modes.
                // Otherwise Mini's compact two-row viewport can be restored
                // in Basic/Full after a locale or mode switch.
                let preset_menu = |ui: &mut egui::Ui| {
                    // Do not constrain the content Ui itself. In a short
                    // Basic window, set_max_height clamps the cursor at the
                    // lower edge and every remaining preset is painted at
                    // that same y coordinate. ComboBox::height already
                    // constrains its outer ScrollArea viewport.
                    ui.label(
                        egui::RichText::new(tr(
                            lang,
                            "長押しドラッグで並べ替え",
                            "Long press and drag to reorder",
                        ))
                        .size(9.5)
                        .weak(),
                    );
                    // Do not inherit a locale/font-dependent menu spacing.
                    // Every preset keeps the same row separation from the
                    // first entry through the scroll area's final entry.
                    ui.spacing_mut().item_spacing.y = PRESET_ROW_GAP;
                    let n = self.store.data.presets.len();
                    let mut rects = Vec::with_capacity(n);
                    for i in 0..n {
                        let name = self.store.data.presets[i].name.clone();
                        let is_active = name == active;
                        let fill = if is_active {
                            ui.visuals().selection.bg_fill
                        } else {
                            egui::Color32::TRANSPARENT
                        };
                        let resp = ui.add_sized(
                            [ui.available_width().max(160.0), PRESET_ROW_HEIGHT],
                            egui::Button::new(preset_name_job(ui, &name))
                                .fill(fill)
                                .sense(egui::Sense::click_and_drag()),
                        );
                        rects.push(resp.rect);
                        // click (released before the hold delay) = select
                        if resp.clicked() {
                            selected = Some(name.clone());
                        }
                        if resp.drag_started() {
                            self.drag = Some((1, i, Instant::now(), false));
                        }
                    }
                    // long-press drag handling
                    if let Some((1, from, t0, ref mut act)) = self.drag {
                        let held = t0.elapsed() >= Duration::from_millis(300);
                        let down = ui.input(|inp| inp.pointer.primary_down());
                        if held && down {
                            if !*act {
                                self.drag = Some((1, from, t0, true));
                            }
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                            if let Some(pos) = ui.ctx().pointer_interact_pos() {
                                // insertion indicator
                                let mut to = rects.len();
                                for (i, r) in rects.iter().enumerate() {
                                    if pos.y < r.center().y {
                                        to = i;
                                        break;
                                    }
                                }
                                let y = if to < rects.len() {
                                    rects[to].top()
                                } else {
                                    rects.last().map(|r| r.bottom()).unwrap_or(pos.y)
                                };
                                if let Some(r0) = rects.first() {
                                    ui.painter().hline(
                                        r0.x_range(),
                                        y,
                                        egui::Stroke::new(2.0, egui::Color32::LIGHT_BLUE),
                                    );
                                }
                            }
                        }
                        if !down {
                            if self.drag.map(|d| d.3).unwrap_or(false) {
                                if let Some(pos) = ui.ctx().pointer_interact_pos() {
                                    let mut to = rects.len();
                                    for (i, r) in rects.iter().enumerate() {
                                        if pos.y < r.center().y {
                                            to = i;
                                            break;
                                        }
                                    }
                                    moved = Some((from, to));
                                }
                            }
                            self.drag = None;
                        }
                    }
                    // The final preset must never sit directly on or under
                    // the popup clip edge. This gutter is part of the
                    // scrollable content, so it remains available at any
                    // DPI and with any future number of presets.
                    ui.add_space(PRESET_POPUP_BOTTOM_GUTTER);
                };
                let preset_button_response = if mini_ui {
                    let mut selected_job = preset_name_job(ui, &display);
                    selected_job.wrap.max_width =
                        MINI_PRESET_WIDTH - 2.0 * ui.spacing().button_padding.x - 18.0;
                    selected_job.wrap.max_rows = 1;
                    selected_job.wrap.break_anywhere = true;
                    selected_job.wrap.overflow_character = Some('…');
                    let response = ui.add_sized(
                        [MINI_PRESET_WIDTH, ui.spacing().interact_size.y + 2.0],
                        egui::Button::new(transparent_control_job(selected_job.clone())),
                    );
                    paint_control_text_centered(ui, &response, selected_job, true);
                    paint_dropdown_arrow(ui, &response);
                    if std::env::var_os("NEO_GUI_TEST_PRESET_MENU_OPEN").is_some() {
                        egui::Popup::open_id(ctx, preset_popup_egui_id(popup_mode));
                    }
                    egui::Popup::menu(&response)
                        .id(preset_popup_egui_id(popup_mode))
                        .align(egui::RectAlign::TOP_START)
                        .align_alternatives(&[])
                        .width(MINI_PRESET_WIDTH)
                        .show(|ui| {
                            ui.set_min_height(preset_popup_height);
                            egui::ScrollArea::vertical()
                                .id_salt(("preset_scroll", preset_popup_id(popup_mode)))
                                .max_height(preset_popup_height)
                                .auto_shrink([false, false])
                                .show(ui, preset_menu);
                        });
                    response
                } else {
                    // Read the live mode here. The header can switch modes in
                    // this same egui frame, while the cached booleans are
                    // refreshed later; using them briefly forced 190 px and
                    // that narrowed size could then be persisted.
                    let expanded_preset =
                        matches!(self.settings.ui_mode, UiMode::Basic | UiMode::Full);
                    let preset_width = if expanded_preset {
                        let widest_name = self
                            .store
                            .data
                            .presets
                            .iter()
                            .map(|preset| {
                                let job = preset_name_job(ui, &preset.name);
                                ui.fonts_mut(|fonts| fonts.layout_job(job).size().x)
                            })
                            .fold(0.0_f32, f32::max);
                        (widest_name + 2.0 * ui.spacing().button_padding.x + 18.0)
                            .ceil()
                            .clamp(190.0, 420.0)
                    } else {
                        190.0
                    };
                    let mut selected_job = preset_name_job(ui, &display);
                    selected_job.wrap.max_width =
                        preset_width - 2.0 * ui.spacing().button_padding.x - 18.0;
                    selected_job.wrap.max_rows = 1;
                    selected_job.wrap.break_anywhere = true;
                    selected_job.wrap.overflow_character = Some('…');
                    let response = ui.add_sized(
                        [preset_width, ui.spacing().interact_size.y + 2.0],
                        egui::Button::new(transparent_control_job(selected_job.clone())),
                    );
                    paint_control_text_centered(ui, &response, selected_job, true);
                    paint_dropdown_arrow(ui, &response);
                    egui::Popup::menu(&response)
                        .id(preset_popup_egui_id(popup_mode))
                        .align(egui::RectAlign::TOP_START)
                        .align_alternatives(&[])
                        .width(preset_width)
                        .show(|ui| {
                            egui::ScrollArea::vertical()
                                .id_salt(("preset_scroll", preset_popup_id(popup_mode)))
                                .max_height(preset_popup_height)
                                .auto_shrink([false, false])
                                .show(ui, preset_menu);
                        });
                    response
                };
                if preset_button_response.clicked() {
                    log::debug!(
                        "preset-popup-open: mode={popup_mode:?} height={preset_popup_height:.0} presets={}",
                        self.store.data.presets.len()
                    );
                }
                if std::env::var_os("NEO_GUI_TEST_PRESET_MENU_OPEN").is_some() {
                    egui::Popup::open_id(ctx, preset_popup_egui_id(popup_mode));
                }
                if let Some((f, mut t)) = moved {
                    if t > f {
                        t -= 1;
                    }
                    if t != f && f < self.store.data.presets.len() {
                        let p = self.store.data.presets.remove(f);
                        self.store
                            .data
                            .presets
                            .insert(t.min(self.store.data.presets.len()), p);
                        self.store.save();
                    }
                } else if let Some(n) = selected {
                    self.select_preset(&n);
                }
                if mini_ui {
                    ui.separator();
                    ink_centered_control_label(ui, tr(lang, "キャプチャ:", "Capture:"));
                    let current_capture =
                        capture_resolution_label(self.settings.capture_resolution, lang);
                    let current_capture_job = control_text_job(ui, current_capture);
                    let response = egui::ComboBox::from_id_salt("mini_capture_resolution")
                        .width(96.0)
                        .selected_text(transparent_control_job(current_capture_job.clone()))
                        .show_ui(ui, |ui| {
                            for preset in capture_resolution_presets() {
                                let label = capture_resolution_label(preset, lang);
                                if ui
                                    .selectable_label(
                                        self.settings.capture_resolution == preset,
                                        label,
                                    )
                                    .clicked()
                                {
                                    self.settings.capture_resolution = preset;
                                    self.capture_resolution_text = capture_resolution_label(
                                        self.settings.capture_resolution,
                                        lang,
                                    );
                                    if running {
                                        let _ = self.apply_capture_resolution_to_target();
                                    }
                                    save_settings(&self.app_dir, &self.settings);
                                }
                            }
                        })
                        .response;
                    paint_control_text_centered(ui, &response, current_capture_job, false);
                    response.on_hover_text(i18n::text(lang, "capture.size_select_help"));
                }
                if full_ui
                    && control_row_button(ui, tr(lang, "保存", "Save"))
                        .on_hover_text(i18n::text(lang, "preset.overwrite_help"))
                        .clicked()
                {
                    let active = self.store.data.active.clone();
                    let aspect_correction = self.current_preset_aspect_correction();
                    let crop = self.current_preset_crop();
                    let capture_resolution = self.current_preset_capture_resolution();
                    if let Some(p) = self
                        .store
                        .data
                        .presets
                        .iter_mut()
                        .find(|p| p.name == active)
                    {
                        p.chain = self.chain.clone();
                        p.aspect_correction = Some(aspect_correction);
                        p.crop = Some(crop);
                        p.capture_resolution = capture_resolution;
                        self.saved_chain = self.chain.clone();
                        self.saved_aspect_correction = aspect_correction;
                        self.saved_crop = crop;
                        self.saved_capture_resolution = capture_resolution;
                        self.store.save();
                        log::info!(
                            "preset-aspect-save: preset='{}' enabled={} scale={:.2}x{:.2}",
                            active,
                            aspect_correction.enabled,
                            aspect_correction.width_scale,
                            aspect_correction.height_scale
                        );
                    }
                }
                if full_ui && control_row_button(ui, tr(lang, "別名で保存", "Save As")).clicked()
                {
                    self.save_as_open = true;
                    self.save_as_name = self.store.data.active.clone();
                    self.save_as_error = None;
                }
                if full_ui && control_row_button(ui, tr(lang, "新規", "New")).clicked() {
                    let default_name = tr(lang, "新規プリセット", "New Preset");
                    let aspect_correction = self.current_preset_aspect_correction();
                    let crop = self.current_preset_crop();
                    let capture_resolution = self.current_preset_capture_resolution();
                    let name = self.store.create_blank(
                        default_name,
                        aspect_correction,
                        crop,
                        capture_resolution,
                    );
                    self.chain.clear();
                    self.saved_chain.clear();
                    self.saved_aspect_correction = aspect_correction;
                    self.saved_crop = crop;
                    self.saved_capture_resolution = capture_resolution;
                    self.store.save();
                    self.apply_live();
                    log::info!("preset-created: name={name} chain=empty");
                }
                if full_ui
                    && control_row_button(ui, tr(lang, "削除", "Delete")).clicked()
                    && self.store.data.presets.len() > 1
                {
                    self.confirm_delete = true;
                }
            });
            // v348z: Mini's deeper raised/latched capture button visually
            // occupies more of the lower edge than the old flat control. Keep
            // an explicit DPI-aware footer gutter without changing Basic/Full.
            ui.add_space(if mini_ui { 13.0 } else { 8.0 });
        });

        // ---------- target frame ----------
        if !mini_ui {
            egui::Panel::top("target").show(root, |ui| {
                ui.add_space(5.0);
                // Never use `horizontal_centered` directly in an auto-sized
                // top panel. It claims the panel's remaining height and feeds
                // that value back into the next frame, which makes this row
                // visibly grow while the app is idle. Give the row one fixed
                // height; individual labels still use painted-glyph centering.
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), TARGET_ROW_HEIGHT),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        let family = locale_font_family(lang);
                        let text_color = ui.visuals().text_color();
                        target_icon_and_label(
                            ui,
                            tr(lang, "🎯 拡大対象:", "🎯 Target:"),
                            family.clone(),
                            text_color,
                        );
                        let title = if running {
                            &status.target_title
                        } else {
                            &self.target_title
                        };
                        let title_text = if title.is_empty() {
                            tr(
                                lang,
                                "（拡大したいウィンドウをクリックしてください）",
                                "(Click the window to magnify)",
                            )
                        } else {
                            title
                        };
                        let running_text = tr(lang, "● 拡大中", "● Running");
                        let running_width = if running {
                            ui.painter()
                                .layout_no_wrap(
                                    running_text.to_owned(),
                                    egui::FontId::new(13.0, family.clone()),
                                    egui::Color32::LIGHT_GREEN,
                                )
                                .size()
                                .x
                                + ui.spacing().item_spacing.x
                        } else {
                            0.0
                        };
                        let title_width = (ui.available_width() - running_width).max(24.0);
                        truncated_target_label(
                            ui,
                            title_text,
                            egui::FontId::new(13.0, mixed_text_font_family(title_text, lang)),
                            text_color,
                            title_width,
                        );
                        if running {
                            target_centered_label(
                                ui,
                                running_text,
                                egui::FontId::new(13.0, family),
                                egui::Color32::LIGHT_GREEN,
                                TARGET_ROW_HEIGHT,
                            );
                        }
                    },
                );
                ui.add_space(5.0);
            });
        }

        // ---------- compact controls / full settings ----------
        if basic_ui {
            egui::Panel::top("basic_controls")
                .show_separator_line(false)
                .show(root, |ui| {
                    self.resize_basic_for_language(ui, lang, false);
                    ui.add_space(6.0);
                    let mut settings_changed = false;
                    // Give horizontal_wrapped rows breathing room when labels
                    // such as Display and Frame interpolation land one above the
                    // other. This panel has its own Ui, so the spacing stays local.
                    ui.spacing_mut().item_spacing.y = 6.0;
                    // Measure translated labels before laying out the row.
                    // French/Spanish/Portuguese/German can be substantially wider
                    // than English. Splitting the two logical control groups keeps
                    // the capture preset arrow fully visible instead of clipping
                    // the final widget at the right edge.
                    let label_width: f32 = [
                        tr(lang, "表示:", "Display:"),
                        tr(lang, "全画面", "Fullscreen"),
                        tr(lang, "倍率", "Windowed"),
                        tr(lang, "キャプチャ解像度:", "Capture size:"),
                    ]
                    .into_iter()
                    .map(|text| {
                        ui.painter()
                            .layout_no_wrap(
                                text.to_owned(),
                                egui::FontId::proportional(12.0),
                                ui.visuals().text_color(),
                            )
                            .size()
                            .x
                    })
                    .sum();
                    // Fixed contribution includes both text fields, both combo
                    // arrows, separators, widget padding and item spacing.
                    let _required_row_width = label_width + 460.0;
                    ui.spacing_mut().item_spacing.x = 10.0;
                    ui.horizontal(|ui| {
                        settings_changed |= self.display_mode_controls(ui, lang);
                        ui.separator();
                        settings_changed |= self.capture_resolution_controls(ui, lang, running);
                    });
                    // Keep these as two real rows at every width. Automatic
                    // wrapping alone made the vertical gap width-dependent.
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        if ink_centered_checkbox(
                            ui,
                            &mut self.settings.stats_on,
                            tr(lang, "統計", "Stats"),
                        )
                        .changed()
                        {
                            self.engine.metrics.set_enabled(
                                self.settings.stats_on || logging::diagnostics_enabled(),
                            );
                            self.resize_basic_for_stats(ui.ctx());
                            self.basic_stats_layout_applied = Some(self.settings.stats_on);
                            self.basic_stats_rows_applied = Some(self.basic_stats_rows());
                            settings_changed = true;
                        }
                        ui.separator();
                        ui.label(tr(lang, "フレーム補間:", "Frame interpolation:"));
                        egui::ComboBox::from_id_salt("basic_interp_factor")
                            .width(62.0)
                            .selected_text(format!("x{}", self.settings.interp_factor))
                            .show_ui(ui, |ui| {
                                for factor in [2u32, 3, 4, 5] {
                                    if ui
                                        .selectable_value(
                                            &mut self.settings.interp_factor,
                                            factor,
                                            format!("x{factor}"),
                                        )
                                        .changed()
                                    {
                                        settings_changed = true;
                                        self.engine.send(Cmd::SetInterpFactor(factor));
                                    }
                                }
                            });
                        ui.separator();
                        if ink_centered_checkbox(
                            ui,
                            &mut self.settings.duplicate_frame_reduction,
                            tr(lang, "重複削減", "Duplicate reduction"),
                        )
                        .on_hover_text(i18n::text(lang, "duplicate.summary"))
                        .changed()
                        {
                            settings_changed = true;
                            self.engine.send(Cmd::SetDuplicateFrameReduction(
                                self.settings.duplicate_frame_reduction,
                            ));
                        }
                    });
                    ui.add_space(5.0);
                    self.resource_meter(ui);
                    if self.settings.stats_on {
                        let snap = self.engine.metrics.snapshot();
                        ui.separator();
                        let internal_size = if snap.internal_size.0 > 0 && snap.internal_size.1 > 0
                        {
                            snap.internal_size
                        } else {
                            snap.out_size
                        };
                        let stats_line = i18n::format_builtin_text(
                            lang,
                            "stats.main",
                            &[
                                ("in_w", snap.in_size.0.to_string()),
                                ("in_h", snap.in_size.1.to_string()),
                                ("internal_w", internal_size.0.to_string()),
                                ("internal_h", internal_size.1.to_string()),
                                ("out_w", snap.out_size.0.to_string()),
                                ("out_h", snap.out_size.1.to_string()),
                                ("total_ms", format!("{:.2}", snap.total_ms)),
                                ("lag_frames", snap.lag_frames.to_string()),
                                ("present_fps", format!("{:.1}", snap.present_fps)),
                                ("capture_fps", format!("{:.1}", snap.capture_fps)),
                            ],
                        );
                        ui.label(
                            egui::RichText::new(stats_line)
                                .family(locale_font_family(lang))
                                .size(STATS_FONT_SIZE),
                        );
                        let rows =
                            stats_rows_in_filter_chain_order(&self.chain, snap.display_stages());
                        if rows.is_empty() {
                            for stage in self.chain.iter().filter(|stage| stage.enabled) {
                                let name = std::path::Path::new(&stage.path)
                                    .file_name()
                                    .and_then(|name| name.to_str())
                                    .unwrap_or(&stage.path);
                                ui.label(
                                    egui::RichText::new(format!(
                                        "  --.--ms  [{:?}] {name}",
                                        stage.kind
                                    ))
                                    .family(locale_font_family(UiLanguage::EnUs))
                                    .size(STATS_FONT_SIZE),
                                );
                            }
                        } else {
                            for (name, stage) in rows {
                                let line = if stage.ms >= 0.0 {
                                    format!("  {:>5.2}ms  [{}] {}", stage.ms, stage.kind, name)
                                } else {
                                    format!("  --.--ms  [{}] {}", stage.kind, name)
                                };
                                ui.label(
                                    egui::RichText::new(line)
                                        .family(locale_font_family(UiLanguage::EnUs))
                                        .size(STATS_FONT_SIZE),
                                );
                            }
                        }
                    }
                    // Keep the resource meter or final statistics row away from
                    // the physical bottom edge in both stats states.
                    ui.add_space(12.0);
                    if settings_changed {
                        save_settings(&self.app_dir, &self.settings);
                        if running {
                            self.engine.send(Cmd::SetMode {
                                mode: self.settings.scale_mode,
                                ratio: self.settings.ratio,
                            });
                        }
                    }
                });
        } else if full_ui {
            // Keep the chain header and up to three filter rows visible even
            // when statistics add several lines to the settings panel. The
            // settings remain available through their own vertical scroll.
            let available_height = root.available_height();
            let desired_chain_height = full_chain_height(self.chain.len(), self.settings.stats_on);
            // Keep settings usable on short displays. If an unusually long
            // chain cannot fit in full, its existing ScrollArea handles it.
            let chain_height = desired_chain_height.min((available_height - 240.0).max(150.0));
            let desired_bottom_height =
                full_settings_height(self.settings.stats_on, self.basic_stats_rows())
                    + self.full_settings_wrap_extra;
            let bottom_height =
                desired_bottom_height.min((available_height - chain_height).max(240.0));
            egui::Panel::bottom("bottom")
                .exact_size(bottom_height)
                .show(root, |ui| {
                    egui::ScrollArea::vertical()
                .id_salt("full_settings_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
            // Full mode follows the current content in both directions. Stats
            // ON/OFF, chain-length changes, locale changes and toolbar wrapping
            // all change the layout key, so the native window grows or shrinks
            // exactly once for that new requirement instead of retaining stale
            // blank space from a previous state.
            self.resize_full_for_language(ui, lang);
            ui.add_space(6.0);
            let mut settings_changed = false;
            let mut aspect_changed = false;
            let mut crop_changed = false;
            // row 1: display mode / ratio / fps cap
            ui.horizontal_wrapped(|ui| {
                settings_changed |= self.display_mode_controls(ui, lang);
                ui.separator();
                bounded_plain_label(
                    ui,
                    tr(lang, "リサイズ:", "Resize:"),
                    MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
                );
                egui::ComboBox::from_id_salt("downscaler")
                    .width(104.0)
                    .selected_text(self.settings.downscaler.clone())
                    .show_ui(ui, |ui| {
                        for k in ["spline36", "lanczos3", "bicubic", "bilinear", "nearest"] {
                            if ui
                                .selectable_label(self.settings.downscaler == k, k)
                                .clicked()
                            {
                                self.settings.downscaler = k.to_string();
                                settings_changed = true;
                                self.engine.send(Cmd::SetDownscaler(k.to_string()));
                            }
                        }
                    })
                    .response
                    .on_hover_text(i18n::text(lang, "resize.final_help"));
                ui.separator();
                settings_changed |= self.capture_resolution_controls(ui, lang, running);
            });
            ui.add_space(2.0);
            // Display-only aspect correction. Capture/WGC geometry and every
            // ONNX/GLSL processing stage remain on their original pixels.
            // Auto modes derive the target ratio from the current pre-correction
            // presentation aspect; Manual preserves v576's numeric controls.
            ui.horizontal_wrapped(|ui| {
                let aspect_response = ink_centered_checkbox(
                    ui,
                    &mut self.settings.aspect_correction,
                    i18n::text(lang, "settings.aspect_correction"),
                )
                .on_hover_text(i18n::text(lang, "aspect.help"));
                if aspect_response.changed() {
                    settings_changed = true;
                    aspect_changed = true;
                }

                ui.add_enabled_ui(self.settings.aspect_correction, |ui| {
                    bounded_plain_label(
                        ui,
                        i18n::text(lang, "settings.aspect_mode"),
                        MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
                    );
                    let selected_mode = match self.settings.aspect_correction_mode {
                        AspectCorrectionMode::Manual => "Manual",
                        AspectCorrectionMode::Auto4x3 => "4:3 (Auto)",
                        AspectCorrectionMode::Auto16x9 => "16:9 (Auto)",
                    };
                    egui::ComboBox::from_id_salt("aspect_correction_mode")
                        .width(112.0)
                        .selected_text(selected_mode)
                        .show_ui(ui, |ui| {
                            for (mode, label) in [
                                (AspectCorrectionMode::Manual, "Manual"),
                                (AspectCorrectionMode::Auto4x3, "4:3 (Auto)"),
                                (AspectCorrectionMode::Auto16x9, "16:9 (Auto)"),
                            ] {
                                if ui
                                    .selectable_label(
                                        self.settings.aspect_correction_mode == mode,
                                        label,
                                    )
                                    .clicked()
                                {
                                    self.settings.aspect_correction_mode = mode;
                                    settings_changed = true;
                                    aspect_changed = true;
                                }
                            }
                        })
                        .response
                        .on_hover_text(i18n::text(lang, "aspect.help"));
                });

                let manual_values_enabled = self.settings.aspect_correction
                    && self.settings.aspect_correction_mode == AspectCorrectionMode::Manual;
                ui.add_enabled_ui(manual_values_enabled, |ui| {
                    bounded_plain_label(
                        ui,
                        i18n::text(lang, "settings.aspect_width"),
                        MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
                    );
                    let mut width_scale = self.settings.aspect_width_scale;
                    if ui
                        .add(
                            egui::DragValue::new(&mut width_scale)
                                .range(ASPECT_CORRECTION_SCALE_MIN..=ASPECT_CORRECTION_SCALE_MAX)
                                .speed(0.01)
                                .fixed_decimals(2),
                        )
                        .on_hover_text(i18n::text(lang, "aspect.help"))
                        .changed()
                    {
                        self.settings.aspect_width_scale =
                            sanitize_aspect_correction_scale(width_scale);
                        settings_changed = true;
                        aspect_changed = true;
                    }

                    bounded_plain_label(
                        ui,
                        i18n::text(lang, "settings.aspect_height"),
                        MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
                    );
                    let mut height_scale = self.settings.aspect_height_scale;
                    if ui
                        .add(
                            egui::DragValue::new(&mut height_scale)
                                .range(ASPECT_CORRECTION_SCALE_MIN..=ASPECT_CORRECTION_SCALE_MAX)
                                .speed(0.01)
                                .fixed_decimals(2),
                        )
                        .on_hover_text(i18n::text(lang, "aspect.help"))
                        .changed()
                    {
                        self.settings.aspect_height_scale =
                            sanitize_aspect_correction_scale(height_scale);
                        settings_changed = true;
                        aspect_changed = true;
                    }
                });
            });
            ui.add_space(2.0);
            // User crop is performed after Neo's existing client/title-bar crop
            // and before GLSL/ONNX. Values are persisted with each preset.
            // TensorRT may execute a saved Crop, but arbitrary shape editing is
            // DirectML-only so a DragValue scrub can never fan out into engine
            // builds for intermediate dimensions.
            ui.horizontal_wrapped(|ui| {
                let backend_switching =
                    self.onnx_backend_pending.is_some() || status.onnx_backend_switching;
                let tensorrt_crop_locked = self.onnx_backend_selected
                    == OnnxBackendPreference::TensorRT
                    || status.onnx_backend == OnnxBackendPreference::TensorRT
                    || backend_switching;
                let crop_response = ink_centered_checkbox_enabled(
                    ui,
                    !backend_switching,
                    &mut self.settings.capture_crop.enabled,
                    i18n::text(lang, "settings.crop"),
                )
                .on_hover_text(i18n::text(lang, "crop.help"));
                if crop_response.changed() {
                    if self.settings.capture_crop.enabled && tensorrt_crop_locked {
                        // TensorRT can re-enable only the selected preset's saved
                        // rectangle. Any unsaved DirectML edit is deliberately
                        // ignored rather than becoming a new TensorRT shape.
                        self.settings.capture_crop.left = self.saved_crop.left;
                        self.settings.capture_crop.top = self.saved_crop.top;
                        self.settings.capture_crop.right = self.saved_crop.right;
                        self.settings.capture_crop.bottom = self.saved_crop.bottom;
                    }
                    crop_changed = true;
                }

                let crop_value_editor = ui.add_enabled_ui(
                    self.settings.capture_crop.enabled && !tensorrt_crop_locked,
                    |ui| {
                        for (key, value) in [
                            ("settings.crop_left", &mut self.settings.capture_crop.left),
                            ("settings.crop_top", &mut self.settings.capture_crop.top),
                            ("settings.crop_right", &mut self.settings.capture_crop.right),
                            ("settings.crop_bottom", &mut self.settings.capture_crop.bottom),
                        ] {
                            bounded_plain_label(
                                ui,
                                i18n::text(lang, key),
                                MAX_TRANSLATED_PLAIN_LABEL_WIDTH,
                            );
                            if ui
                                .add(egui::DragValue::new(value).range(0..=8192).speed(1.0))
                                .on_hover_text(i18n::text(lang, "crop.help"))
                                .changed()
                            {
                                crop_changed = true;
                            }
                        }
                    },
                );
                if tensorrt_crop_locked {
                    crop_value_editor
                        .response
                        .on_hover_text(i18n::text(lang, "crop.tensorrt_edit_locked"));
                }
            });
            // Visually separate source geometry controls (aspect/crop) from
            // timing and rendering options below. Keep this as a single native
            // egui separator so it adds minimal height and follows the current
            // theme/DPI automatically.
            ui.separator();
            // Row 2 starts after capture resolution at every window width.
            // The important timing controls keep the requested fixed order in
            // every language: FPS cap -> duplicate reduction -> rendering
            // stabilization -> VSync -> TensorRT (only when the backend pack is detected).
            ui.horizontal_wrapped(|ui| {
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.fps_cap_enabled,
                    tr(lang, "FPS上限", "FPS cap"),
                )
                    .changed()
                {
                    settings_changed = true;
                }
                if self.settings.fps_cap_enabled {
                    let mut fps = self.settings.fps_cap as i32;
                    if ui
                        .add(egui::DragValue::new(&mut fps).range(5..=MAX_FPS_CAP as i32))
                        .changed()
                    {
                        self.settings.fps_cap = fps as u32;
                        settings_changed = true;
                    }
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.duplicate_frame_reduction,
                    tr(lang, "重複削減", "Duplicate reduction"),
                )
                    .on_hover_text(i18n::text(lang, "duplicate.help"))
                    .changed()
                {
                    settings_changed = true;
                    self.engine.send(Cmd::SetDuplicateFrameReduction(
                        self.settings.duplicate_frame_reduction,
                    ));
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.smooth_pacing,
                    tr(lang, "描画安定化", "Smooth pacing"),
                )
                    .on_hover_text(i18n::text(lang, "smooth.help"))
                    .changed()
                {
                    settings_changed = true;
                    self.engine
                        .send(Cmd::SetSmoothPacing(self.settings.smooth_pacing));
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.vsync,
                    tr(lang, "VSync", "VSync"),
                )
                    .on_hover_text(i18n::text(lang, "vsync.help"))
                    .changed()
                {
                    settings_changed = true;
                    self.engine.send(Cmd::SetVsync(self.settings.vsync));
                }

                if tensorrt_option_visible(&self.tensorrt_availability) {
                    let backend_switching =
                        self.onnx_backend_pending.is_some() || status.onnx_backend_switching;
                    let mut tensorrt_on =
                        self.onnx_backend_selected == OnnxBackendPreference::TensorRT;
                    let crop_switch_allowed =
                        tensorrt_crop_switch_allowed(self.settings.capture_crop, self.saved_crop);
                    // An active TensorRT session must always remain switchable
                    // OFF. The saved-Crop gate applies only when entering TRT.
                    let backend_control_enabled =
                        !backend_switching && (tensorrt_on || crop_switch_allowed);
                    let backend_label = if backend_switching {
                        i18n::text(lang, "tensorrt.preparing_label")
                    } else {
                        "TensorRT"
                    };
                    let response = ink_centered_checkbox_enabled(
                        ui,
                        backend_control_enabled,
                        &mut tensorrt_on,
                        backend_label,
                    )
                    .on_hover_text({
                        let mut help = i18n::text(lang, "tensorrt.help").to_owned();
                        if !crop_switch_allowed && !tensorrt_on {
                            help.push('\n');
                            help.push_str(i18n::text(lang, "tensorrt.crop_unsaved_help"));
                        }
                        if status.onnx_cuda_stages > 0 {
                            help.push('\n');
                            help.push_str(&i18n::format_text(
                                lang,
                                "tensorrt.cuda_fallback_count",
                                &[("count", status.onnx_cuda_stages.to_string())],
                            ));
                        }
                        if status.onnx_directml_fallbacks > 0 {
                            help.push('\n');
                            help.push_str(&i18n::format_text(
                                lang,
                                "tensorrt.directml_fallback_count",
                                &[("count", status.onnx_directml_fallbacks.to_string())],
                            ));
                        }
                        help
                    });
                    let requested_tensorrt = if response.changed() {
                        Some(tensorrt_on)
                    } else if response.clicked() {
                        // Defensive fallback for a platform/widget path that
                        // reports the click but not the local bool mutation.
                        Some(self.onnx_backend_selected != OnnxBackendPreference::TensorRT)
                    } else {
                        None
                    };
                    if let Some(tensorrt_on) = requested_tensorrt {
                        self.request_onnx_backend_switch(if tensorrt_on {
                            OnnxBackendPreference::TensorRT
                        } else {
                            OnnxBackendPreference::DirectML
                        });
                    }
                }
            });
            ui.add_space(2.0);
            // row 2: options (wrapped so a narrow window never clips items)
            ui.horizontal_wrapped(|ui| {
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.stats_on,
                    tr(lang, "統計", "Stats"),
                )
                    .changed()
                {
                    self.engine.metrics.set_enabled(self.settings.stats_on || logging::diagnostics_enabled());
                    settings_changed = true;
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.gui_topmost,
                    tr(lang, "GUIを最前面に表示", "Keep GUI on top"),
                )
                    .changed()
                {
                    settings_changed = true;
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.panel_show,
                    tr(lang, "操作パネルを表示", "Show control panel"),
                )
                    .on_hover_text(i18n::text(lang, "panel.show_help"))
                    .changed()
                {
                    settings_changed = true;
                }
                // A fixed capture resolution defines the raw/pre-crop capture canvas;
                // client-only capture is mandatory while a resolution is selected.
                // Keep the user's manual setting untouched underneath the forced
                // visual state so switching Capture Resolution back to Auto restores
                // exactly the previous ON/OFF preference.
                let capture_resolution_forces_client_only =
                    self.settings.capture_resolution.is_some();
                let mut client_only_display = if capture_resolution_forces_client_only {
                    true
                } else {
                    self.settings.client_only
                };
                let mut client_only_response = ink_centered_checkbox_enabled(
                    ui,
                    !capture_resolution_forces_client_only,
                    &mut client_only_display,
                    tr(lang, "タイトルバー除外", "Client area only"),
                );
                if capture_resolution_forces_client_only {
                    // `add_enabled(false, ...)` intentionally suppresses interaction,
                    // including hover sensing. Add a hover-only response over the
                    // exact disabled checkbox rect so the reason for the forced ON
                    // state remains discoverable without making the control clickable.
                    ui.interact(
                        client_only_response.rect,
                        client_only_response.id.with("forced-client-only-help"),
                        egui::Sense::hover(),
                    )
                    .on_hover_text(i18n::text(lang, "capture.client_forced_help"));
                } else {
                    client_only_response =
                        client_only_response.on_hover_text(i18n::text(lang, "capture.client_help"));
                }
                if !capture_resolution_forces_client_only && client_only_response.changed() {
                    self.settings.client_only = client_only_display;
                    settings_changed = true;
                }
                if HDR_CAPTURE_OPTION_ENABLED
                    && ink_centered_checkbox(
                        ui,
                        &mut self.settings.hdr_capture,
                        tr(lang, "HDR→SDR", "HDR to SDR"),
                    )
                    .on_hover_text(i18n::text(lang, "hdr.help"))
                    .changed()
                {
                    settings_changed = true;
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.log_on,
                    tr(lang, "ログを保存", "Save log"),
                )
                    .on_hover_text(i18n::text(lang, "log.help"))
                    .changed()
                {
                    logging::set_file_logging(&self.app_dir, self.settings.log_on);
                    // Detailed per-stage timers serve either the visible Stats
                    // panel or the diagnostic log. Keep them off only when
                    // neither feature needs them.
                    self.engine.metrics.set_enabled(
                        self.settings.stats_on || logging::diagnostics_enabled(),
                    );
                    settings_changed = true;
                }
                if gpu_selector_visible(&self.gpu_adapters) {
                    ui.separator();
                    let gpu_change_enabled = !(
                        status.starting
                            || status.running
                            || status.stopping
                            || status.onnx_backend_switching
                            || self.onnx_backend_pending.is_some()
                            || chidescaler_neo::render::onnx_stage::tensorrt_is_preparing()
                    );
                    self.gpu_selector_control(ui, lang, gpu_change_enabled);
                }
                ui.separator();
                let open_folder =
                    folder_icon_button(ui, i18n::text(lang, "settings.open_folder"));
                if open_folder.clicked() {
                    match win32::open_application_folder(self.gui_hwnd) {
                        Ok(win32::OpenApplicationFolderResult::Opened(path)) => {
                            log::info!("settings-folder-open-requested: {}", path.display());
                        }
                        Ok(win32::OpenApplicationFolderResult::AlreadyOpen(path)) => {
                            log::info!(
                                "settings-folder-open-cancelled: already-open path={}",
                                path.display()
                            );
                        }
                        Err(error) => {
                            log::warn!("settings-folder-open-failed: {error}");
                            self.engine.status.lock().unwrap().last_error = Some(format!(
                                "{}: {error}",
                                i18n::text(lang, "settings.open_folder_failed")
                            ));
                        }
                    }
                }
            });
            ui.add_space(2.0);
            // row 2.5: cursor & interpolation options
            let mut input_opts_changed = false;
            ui.horizontal_wrapped(|ui| {
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.cursor_autohide,
                    tr(lang, "カーソル自動非表示", "Auto-hide cursor"),
                )
                    .changed()
                {
                    settings_changed = true;
                    input_opts_changed = true;
                }
                if self.settings.cursor_autohide {
                    let mut secs = self.settings.cursor_autohide_secs;
                    if ui
                        .add(
                            egui::DragValue::new(&mut secs)
                                .range(0.5..=30.0)
                                .speed(0.1)
                                .suffix(tr(lang, " 秒", " sec")),
                        )
                        .changed()
                    {
                        self.settings.cursor_autohide_secs = secs;
                        settings_changed = true;
                        input_opts_changed = true;
                    }
                }
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.cursor_speed_fix,
                    tr(lang, "カーソル速度補正", "Natural cursor speed"),
                )
                    .on_hover_text(i18n::text(lang, "cursor.speed_help"))
                    .changed()
                {
                    settings_changed = true;
                    input_opts_changed = true;
                }
                ui.separator();
                if ink_centered_checkbox(
                    ui,
                    &mut self.settings.run_as_admin,
                    tr(lang, "管理者として再起動", "Restart as admin"),
                )
                    .on_hover_text(i18n::text(lang, "admin.required_help"))
                    .changed()
                {
                    settings_changed = true;
                    save_settings(&self.app_dir, &self.settings);
                    if self.settings.run_as_admin && !win32::own_process_elevated() {
                        if win32::relaunch_as_admin() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                }
            });
            ui.add_space(2.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(tr(lang, "フレーム補間:", "Frame interpolation:"));
                egui::ComboBox::from_id_salt("interp_factor")
                    .width(64.0)
                    .selected_text(format!("x{}", self.settings.interp_factor))
                    .show_ui(ui, |ui| {
                        for f in [2u32, 3, 4, 5] {
                            if ui
                                .selectable_label(
                                    self.settings.interp_factor == f,
                                    format!("x{f}"),
                                )
                                .clicked()
                            {
                                self.settings.interp_factor = f;
                                settings_changed = true;
                                self.engine.send(Cmd::SetInterpFactor(f));
                            }
                        }
                    })
                    .response
                    .on_hover_text(i18n::text(lang, "interpolation.factor_help"));
            });
            if input_opts_changed {
                self.engine.send(Cmd::SetInputOpts {
                    autohide_secs: if self.settings.cursor_autohide {
                        self.settings.cursor_autohide_secs.max(0.5)
                    } else {
                        0.0
                    },
                    speed_fix: self.settings.cursor_speed_fix,
                });
            }
            ui.add_space(2.0);
            ui.columns(2, |columns| {
                egui::Frame::new()
                    .fill(egui::Color32::from_rgb(27, 28, 30))
                    .stroke(egui::Stroke::new(
                        1.0,
                        egui::Color32::from_rgb(65, 68, 73),
                    ))
                    .corner_radius(egui::CornerRadius::same(5))
                    .inner_margin(egui::Margin::same(8))
                    .show(&mut columns[0], |ui| {
                        ui.set_min_width(ui.available_width());
                        ui.set_min_height(64.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(
                                egui::RichText::new(tr(lang, "開始/停止", "Start/Stop"))
                                    .size(12.0)
                                    .strong(),
                            );
                            hotkey_chips(ui, &self.settings.hotkey_toggle);
                            if control_row_button(ui, tr(lang, "編集", "Edit")).clicked() {
                                self.hotkey_editor_candidate = self.settings.hotkey_toggle.clone();
                                self.hotkey_editor_error = None;
                                self.hotkey_capture_pending = None;
                                self.hotkey_editor_open = true;
                            }
                        });
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(format!(
                                "{}: Ctrl+Alt+G {}　Ctrl+Alt+P {}　Ctrl+Alt+Q {}",
                                tr(lang, "ショートカット", "Shortcuts"),
                                tr(lang, "GUI", "GUI"),
                                tr(lang, "操作パネル", "Panel"),
                                tr(lang, "終了", "Quit")
                            ))
                            .size(10.5)
                            .weak(),
                        );
                    });

                self.resource_meter(&mut columns[1]);
            });
            if aspect_changed && running {
                self.engine.send(Cmd::SetAspectCorrection {
                    enabled: self.settings.aspect_correction,
                    mode: self.settings.aspect_correction_mode,
                    width_scale: sanitize_aspect_correction_scale(
                        self.settings.aspect_width_scale,
                    ),
                    height_scale: sanitize_aspect_correction_scale(
                        self.settings.aspect_height_scale,
                    ),
                });
            }
            if crop_changed {
                // Persist the settled Crop once after the gesture. This remains
                // independent of the render-path update cadence below.
                self.crop_settings_save_due =
                    Some(Instant::now() + Duration::from_millis(160));
            }
            if crop_changed && running {
                if self.onnx_backend_selected == OnnxBackendPreference::TensorRT {
                    // Numeric Crop editing is disabled while TensorRT is
                    // active. Keep the established final-commit transaction for
                    // the permitted saved-Crop ON/OFF change.
                    self.tensorrt_crop_commit_pending = Some(self.settings.capture_crop);
                    self.tensorrt_crop_commit_due = if win32::left_mouse_button_down() {
                        None
                    } else {
                        Some(Instant::now() + Duration::from_millis(220))
                    };
                    log::debug!(
                        "tensorrt-crop-transaction: staged enabled={} edges=({}, {}, {}, {}) pointer_down={}",
                        self.settings.capture_crop.enabled,
                        self.settings.capture_crop.left,
                        self.settings.capture_crop.top,
                        self.settings.capture_crop.right,
                        self.settings.capture_crop.bottom,
                        win32::left_mouse_button_down()
                    );
                } else if self.settings.capture_resolution.is_some() {
                    // Only fixed Capture Resolution needs throttling. Capture=Auto
                    // keeps v591's exact immediate live-crop behavior.
                    let crop = self.settings.capture_crop;
                    let pointer_down = win32::left_mouse_button_down();
                    let can_publish_now = !pointer_down
                        || self
                            .live_crop_last_sent
                            .map_or(true, |sent| sent.elapsed() >= Duration::from_millis(33));
                    if can_publish_now {
                        self.live_crop_pending = None;
                        self.live_crop_last_sent = Some(Instant::now());
                        self.engine.send(Cmd::SetCaptureCrop { crop });
                    } else {
                        self.live_crop_pending = Some(crop);
                    }
                } else {
                    self.live_crop_pending = None;
                    self.live_crop_last_sent = None;
                    self.engine.send(Cmd::SetCaptureCrop {
                        crop: self.settings.capture_crop,
                    });
                }
            }
            if settings_changed {
                save_settings(&self.app_dir, &self.settings);
                if running {
                    self.engine.send(Cmd::SetMode {
                        mode: self.settings.scale_mode,
                        ratio: self.settings.ratio,
                    });
                }
            }
            // stats
            if self.settings.stats_on {
                let snap = self.engine.metrics.snapshot();
                ui.separator();
                let internal_size = if snap.internal_size.0 > 0 && snap.internal_size.1 > 0 {
                    snap.internal_size
                } else {
                    snap.out_size
                };
                let stats_line = i18n::format_builtin_text(
                    lang,
                    "stats.main",
                    &[
                        ("in_w", snap.in_size.0.to_string()),
                        ("in_h", snap.in_size.1.to_string()),
                        ("internal_w", internal_size.0.to_string()),
                        ("internal_h", internal_size.1.to_string()),
                        ("out_w", snap.out_size.0.to_string()),
                        ("out_h", snap.out_size.1.to_string()),
                        ("total_ms", format!("{:.2}", snap.total_ms)),
                        ("lag_frames", snap.lag_frames.to_string()),
                        ("present_fps", format!("{:.1}", snap.present_fps)),
                        ("capture_fps", format!("{:.1}", snap.capture_fps)),
                    ],
                );
                ui.label(
                    egui::RichText::new(stats_line)
                        .family(locale_font_family(lang))
                        .size(STATS_FONT_SIZE),
                )
                .on_hover_text(i18n::builtin_text(lang, "stats.help"));
                if snap.monitor_size.0 > 0 && snap.monitor_size.1 > 0 {
                    let monitor_line = i18n::format_builtin_text(
                        lang,
                        "stats.monitor",
                        &[
                            ("width", snap.monitor_size.0.to_string()),
                            ("height", snap.monitor_size.1.to_string()),
                            ("refresh", format!("{:.2}", snap.monitor_refresh_hz)),
                        ],
                    );
                    ui.label(
                        egui::RichText::new(monitor_line)
                            .family(locale_font_family(lang))
                            .size(STATS_FONT_SIZE),
                    );
                }
                let rows = stats_rows_in_filter_chain_order(&self.chain, snap.display_stages());
                if rows.is_empty() {
                    for stage in self.chain.iter().filter(|stage| stage.enabled) {
                        let name = std::path::Path::new(&stage.path)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or(&stage.path);
                        ui.label(
                            egui::RichText::new(format!(
                                "  --.--ms  [{:?}] {name}",
                                stage.kind
                            ))
                            .family(locale_font_family(UiLanguage::EnUs))
                            .size(STATS_FONT_SIZE),
                        );
                    }
                } else {
                    for (name, st) in rows {
                        let line = if st.ms >= 0.0 {
                            format!("  {:>5.2}ms  [{}] {}", st.ms, st.kind, name)
                        } else {
                            format!("  --.--ms  [{}] {}", st.kind, name)
                        };
                        ui.label(
                            egui::RichText::new(line)
                                .family(locale_font_family(UiLanguage::EnUs))
                                .size(STATS_FONT_SIZE),
                        );
                    }
                }
            }
            if let Some(w) = &status.warning {
                compact_status_line(ui, egui::Color32::ORANGE, w);
            }
            if let Some(e) = &status.last_error {
                compact_status_line(ui, egui::Color32::LIGHT_RED, e);
            }
            if let Some(e) = &status.onnx_backend_error {
                compact_status_line(
                    ui,
                    egui::Color32::YELLOW,
                    &format!(
                        "{}: {e}",
                        tr(
                            lang,
                            "TensorRTへの切替に失敗したため、現在のバックエンドを継続しています",
                            "TensorRT switch failed; the current backend remains active"
                        )
                    ),
                );
            }
            let shown_errors = status.chain_errors.len().min(3);
            for e in status.chain_errors.iter().take(shown_errors) {
                compact_status_line(ui, egui::Color32::YELLOW, e);
            }
            if status.chain_errors.len() > shown_errors {
                ui.colored_label(
                    egui::Color32::YELLOW,
                    format!(
                        "! +{} {}",
                        status.chain_errors.len() - shown_errors,
                        tr(lang, "件の追加フィルターエラー", "more filter errors")
                    ),
                )
                .on_hover_text(status.chain_errors[shown_errors..].join("\n\n"));
            }
            if false {
            if let Some(w) = &status.warning {
                ui.colored_label(egui::Color32::ORANGE, format!("⚠ {w}"));
            }
            if let Some(e) = &status.last_error {
                ui.colored_label(egui::Color32::LIGHT_RED, format!("⚠ {e}"));
            }
            for e in &status.chain_errors {
                ui.colored_label(egui::Color32::YELLOW, format!("⚠ {e}"));
            }
            }
            ui.add_space(FULL_SETTINGS_VISIBLE_BOTTOM_PADDING);
                });
                });
        }

        // ---------- center: chain editor ----------
        if !full_ui {
            egui::CentralPanel::default().show(root, |_ui| {});
        } else {
            egui::CentralPanel::default().show(root, |ui| {
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), 32.0),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        visually_centered_label(
                            ui,
                            tr(lang, "フィルターチェーン", "Filter Chain"),
                            egui::FontId::new(18.0, locale_font_family(lang)),
                            ui.visuals().text_color(),
                            false,
                            32.0,
                        );
                        let add_filter_button =
                            control_row_button(ui, tr(lang, "＋ フィルター追加", "+ Add Filter"));
                        log_ui_test_rect("add-filter-button", &add_filter_button);
                        if add_filter_button.hovered()
                            && ui.input(|input| input.pointer.primary_pressed())
                        {
                            log::info!("filter-picker-press: chain_len={}", self.chain.len());
                        }
                        if add_filter_button.clicked() {
                            self.refresh_filters("picker-open");
                            self.filter_picker_open = true;
                            log::info!(
                                "filter-picker-open: available={} chain_len={} running={}",
                                self.available.len(),
                                self.chain.len(),
                                running
                            );
                        }
                        ui.label(
                            egui::RichText::new(tr(
                                lang,
                                "（長押しドラッグで並べ替え）",
                                "(Hold and drag to reorder)",
                            ))
                            .size(10.5)
                            .weak(),
                        );
                    },
                );
                ui.separator();
                let mut changed = false;
                let mut remove: Option<usize> = None;
                let mut edit_resize: Option<usize> = None;
                let mut swap: Option<(usize, usize)> = None;
                let mut dnd_move: Option<(usize, usize)> = None;
                let len = self.chain.len();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let mut row_rects: Vec<egui::Rect> = Vec::with_capacity(len);
                    for i in 0..len {
                        let spec_kind = self.chain[i].kind;
                        let spec_name = self.chain[i]
                            .path
                            .strip_prefix("builtin:")
                            .map(|name| {
                                if name.to_ascii_lowercase().starts_with("neoflow") {
                                    "NeoFlow".to_string()
                                } else {
                                    name.to_string()
                                }
                            })
                            .unwrap_or_else(|| {
                                self.chain[i]
                                    .path
                                    .rsplit(['/', '\\'])
                                    .next()
                                    .unwrap_or(&self.chain[i].path)
                                    .to_string()
                            });
                        let dragging_this = matches!(self.drag, Some((2, di, _, true)) if di == i);
                        let row = ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new("≡")
                                    .size(14.0)
                                    .color(egui::Color32::from_gray(110)),
                            );
                            let spec = &mut self.chain[i];
                            if ui
                                .checkbox(&mut spec.enabled, "")
                                .on_hover_text(i18n::text(lang, "filter.toggle"))
                                .changed()
                            {
                                log::info!(
                                    "filter-chain-toggle: index={i} enabled={} path={}",
                                    spec.enabled,
                                    spec.path
                                );
                                changed = true;
                            }
                            // grab zone: from the name up to just left of the
                            // buttons (long-press to start moving)
                            let btn_zone = if is_resize_shader_path(&spec.path) {
                                165.0
                            } else {
                                130.0
                            };
                            let grab_w = (ui.available_width() - btn_zone).max(60.0);
                            let (rect, grab) = ui.allocate_exact_size(
                                egui::vec2(grab_w, 24.0),
                                egui::Sense::click_and_drag(),
                            );
                            let job = chain_stage_name_job(ui, spec_kind, &spec_name, spec.enabled);
                            let galley = ui.fonts_mut(|fonts| fonts.layout_job(job));
                            let pos = egui::Align2::LEFT_CENTER
                                .align_size_within_rect(galley.size(), rect)
                                .min;
                            ui.painter().galley(pos, galley, ui.visuals().text_color());
                            if grab.drag_started() {
                                self.drag = Some((2, i, Instant::now(), false));
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(egui::RichText::new("🗑").size(11.0))
                                                .min_size(egui::vec2(28.0, 24.0))
                                                .corner_radius(egui::CornerRadius::same(8)),
                                        )
                                        .on_hover_text(i18n::text(lang, "common.delete"))
                                        .clicked()
                                    {
                                        remove = Some(i);
                                    }
                                    if is_resize_shader_path(&spec.path) {
                                        let response = ui
                                            .add(
                                                egui::Button::new("")
                                                    .min_size(egui::vec2(28.0, 24.0))
                                                    .corner_radius(egui::CornerRadius::same(8)),
                                            )
                                            .on_hover_text(i18n::text(lang, "resize.edit"));
                                        let center = response.rect.center();
                                        let color = ui.style().interact(&response).fg_stroke.color;
                                        let stroke = egui::Stroke::new(1.7, color);
                                        // Draw the pen ourselves: unlike a Unicode glyph this cannot
                                        // become a missing-font square on another Windows install.
                                        let tip = center + egui::vec2(-5.0, 5.0);
                                        let upper = center + egui::vec2(3.2, -3.2);
                                        let cap = center + egui::vec2(5.0, -5.0);
                                        let side = egui::vec2(1.35, 1.35);
                                        let painter = ui.painter();
                                        painter.line_segment([tip - side, upper - side], stroke);
                                        painter.line_segment([tip + side, upper + side], stroke);
                                        painter.line_segment([upper - side, cap - side], stroke);
                                        painter.line_segment([upper + side, cap + side], stroke);
                                        painter.line_segment([cap - side, cap + side], stroke);
                                        painter.line_segment([tip - side, tip + side], stroke);
                                        painter.line_segment(
                                            [tip, tip + egui::vec2(-1.4, 1.4)],
                                            stroke,
                                        );
                                        if response.clicked() {
                                            edit_resize = Some(i);
                                        }
                                    }
                                    if Self::soft_button(ui, "∨", i + 1 < len)
                                        .on_hover_text(i18n::text(lang, "common.move_down"))
                                        .clicked()
                                    {
                                        swap = Some((i, i + 1));
                                    }
                                    if Self::soft_button(ui, "∧", i > 0)
                                        .on_hover_text(i18n::text(lang, "common.move_up"))
                                        .clicked()
                                    {
                                        swap = Some((i, i - 1));
                                    }
                                },
                            );
                        });
                        if dragging_this {
                            // clear "you're holding this row" feedback
                            ui.painter().rect_filled(
                                row.response.rect.expand2(egui::vec2(2.0, 1.0)),
                                6.0,
                                egui::Color32::from_rgba_unmultiplied(90, 160, 255, 36),
                            );
                            ui.painter().rect_stroke(
                                row.response.rect.expand2(egui::vec2(2.0, 1.0)),
                                6.0,
                                egui::Stroke::new(1.5, egui::Color32::from_rgb(110, 180, 255)),
                                egui::StrokeKind::Outside,
                            );
                        }
                        row_rects.push(row.response.rect);
                        ui.separator();
                    }
                    // long-press drag: activate after 300ms hold, then show the
                    // move cursor + insertion line; drop on release
                    if let Some((2, from, t0, act)) = self.drag {
                        let held = t0.elapsed() >= Duration::from_millis(300);
                        let down = ui.input(|inp| inp.pointer.primary_down());
                        if held && down && !act {
                            self.drag = Some((2, from, t0, true));
                        }
                        if held && down {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                            if let Some(pos) = ui.ctx().pointer_interact_pos() {
                                let mut to = row_rects.len();
                                for (i, r) in row_rects.iter().enumerate() {
                                    if pos.y < r.center().y {
                                        to = i;
                                        break;
                                    }
                                }
                                let y = if to < row_rects.len() {
                                    row_rects[to].top()
                                } else {
                                    row_rects.last().map(|r| r.bottom()).unwrap_or(pos.y)
                                };
                                if let Some(r0) = row_rects.first() {
                                    ui.painter().hline(
                                        r0.x_range(),
                                        y,
                                        egui::Stroke::new(2.5, egui::Color32::LIGHT_BLUE),
                                    );
                                }
                            }
                        }
                        if !down {
                            if self.drag.map(|d| d.3).unwrap_or(false) {
                                if let Some(pos) = ui.ctx().pointer_interact_pos() {
                                    let mut to = row_rects.len();
                                    for (i, r) in row_rects.iter().enumerate() {
                                        if pos.y < r.center().y {
                                            to = i;
                                            break;
                                        }
                                    }
                                    dnd_move = Some((from, to));
                                }
                            }
                            self.drag = None;
                        }
                    }
                    if len == 0 {
                        ui.weak(tr(
                            lang,
                            "フィルターがありません。「＋ フィルター追加」から追加してください。",
                            "No filters in this chain. Use + Add Filter to add one.",
                        ));
                    }
                });
                if let Some((f, mut t)) = dnd_move {
                    if t > f {
                        t -= 1;
                    }
                    if t != f && f < self.chain.len() {
                        let path = self.chain[f].path.clone();
                        let item = self.chain.remove(f);
                        self.chain.insert(t.min(self.chain.len()), item);
                        log::info!("filter-chain-move: from={f} to={t} path={path}");
                        changed = true;
                    }
                }
                if let Some(i) = edit_resize {
                    let value = self.chain[i]
                        .params
                        .get("RESIZE_SCALE")
                        .copied()
                        .unwrap_or(0.75)
                        .clamp(0.25, 4.0);
                    self.resize_scale_editor = Some((i, value));
                }
                if let Some((a, b)) = swap {
                    log::info!(
                        "filter-chain-swap: a={a} b={b} path_a={} path_b={}",
                        self.chain[a].path,
                        self.chain[b].path
                    );
                    self.chain.swap(a, b);
                    changed = true;
                }
                if let Some(i) = remove {
                    log::info!("filter-chain-remove: index={i} path={}", self.chain[i].path);
                    self.chain.remove(i);
                    changed = true;
                }
                if changed {
                    self.apply_live();
                }
            });
        }

        // ---------- floating control panel ----------
        let mut panel_status = status.clone();
        panel_status.running |= self.qa_panel_preview;
        self.control_panel(ctx, &panel_status);
        // v505: restore only the pre-v465 AMD/WGPU composition keep-alive.
        // The native GDI panel and the existing independent cursor sprite remain
        // authoritative for visuals/input; this viewport is only a DWM anchor.
        self.compositor_keepalive_anchor(ctx, &panel_status);

        // ---------- modals ----------
        if self.settings.ui_mode != UiMode::Mini && self.capture_resolution_fullscreen_notice_open {
            let mut close_notice = false;
            let notice_title = i18n::text(lang, "capture.fullscreen_notice_title");
            let notice_body = i18n::text(lang, "capture.fullscreen_notice_body");
            egui::Window::new(notice_title)
                .id(egui::Id::new("capture_resolution_fullscreen_notice"))
                .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 68.0))
                .collapsible(false)
                .resizable(false)
                .movable(false)
                .show(ctx, |ui| {
                    // Size this one-shot notice from the actual localized glyph width
                    // instead of forcing the old 430-point minimum. Short CJK text stays
                    // compact, while longer Latin translations get only the width they
                    // need. The clamp prevents either a cramped column or a very wide
                    // release-notice window on high-DPI / localized systems.
                    let font_id = egui::TextStyle::Body.resolve(ui.style());
                    let text_color = ui.visuals().text_color();
                    let unwrapped_body_width = ui.fonts_mut(|fonts| {
                        fonts
                            .layout_no_wrap(notice_body.to_owned(), font_id, text_color)
                            .size()
                            .x
                    });
                    let notice_width = (unwrapped_body_width / 4.8).clamp(300.0, 390.0);
                    ui.set_min_width(notice_width);
                    ui.set_max_width(notice_width);
                    ui.label(notice_body);
                    ui.add_space(8.0);
                    ui.horizontal_centered(|ui| {
                        // Use the same glyph-bounds centering path as the other corrected
                        // Neo popup buttons. The stock button baseline can look vertically
                        // high/low depending on the active Windows font and DPI.
                        if control_row_button(ui, "OK").clicked() {
                            close_notice = true;
                        }
                    });
                });
            if close_notice {
                self.capture_resolution_fullscreen_notice_open = false;
            }
        }

        if self.settings.ui_mode != UiMode::Mini
            && let Some((hwnd, title)) = self.elevated_target_notice.clone()
        {
            let mut keep_open = true;
            let mut restart = false;
            egui::Window::new(tr(
                lang,
                "管理者権限が必要です",
                "Administrator permission required",
            ))
            .id(egui::Id::new("elevated_target_notice"))
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, egui::Vec2::ZERO)
            .show(ctx, |ui| {
                ui.set_min_width(390.0);
                ui.label(tr(
                    lang,
                    "選択したウィンドウは管理者権限で動作しています。",
                    "The selected window is running with administrator permission.",
                ));
                ui.label(egui::RichText::new(&title).strong());
                ui.add_space(4.0);
                ui.label(tr(
                    lang,
                    "このウィンドウを拡大・操作するには、cHiDeScaler-Neoも管理者として再起動してください。再起動後、対象ウィンドウをもう一度選択してください。",
                    "To capture and control this window, restart cHiDeScaler-Neo as administrator, then select the window again.",
                ));
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    if control_row_button(
                        ui,
                        tr(
                            lang,
                            "管理者として再起動",
                            "Restart as administrator",
                        ),
                    )
                    .clicked()
                    {
                        restart = true;
                        keep_open = false;
                    }
                    if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                        keep_open = false;
                    }
                });
            });
            if restart {
                self.settings.run_as_admin = true;
                save_settings(&self.app_dir, &self.settings);
                log::info!(
                    "elevated-target-guidance accepted: hwnd={hwnd:#x}; relaunching as administrator"
                );
                if win32::relaunch_as_admin() {
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                } else {
                    self.engine.status.lock().unwrap().last_error = Some(
                        tr(
                            lang,
                            "管理者として再起動できませんでした。Windowsの確認画面で許可してから、もう一度お試しください。",
                            "Could not restart as administrator. Allow the Windows confirmation prompt and try again.",
                        )
                        .to_string(),
                    );
                }
            }
            if keep_open {
                self.elevated_target_notice = Some((hwnd, title));
            } else {
                self.elevated_target_notice = None;
            }
        }
        if self.settings.ui_mode != UiMode::Mini
            && let Some((index, mut value)) = self.resize_scale_editor.take()
        {
            let mut keep_open = true;
            let mut apply = false;
            egui::Window::new(tr(lang, "リサイズ倍率", "Resize scale"))
                .collapsible(false)
                .resizable(false)
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(tr(lang, "倍率", "Scale"));
                        ui.add(
                            egui::DragValue::new(&mut value)
                                .range(0.25..=4.0)
                                .speed(0.01)
                                .fixed_decimals(2),
                        );
                    });
                    ui.label(tr(
                        lang,
                        "0.25～4.00（初期値 0.75）",
                        "0.25–4.00 (default 0.75)",
                    ));
                    ui.horizontal(|ui| {
                        if control_row_button(ui, tr(lang, "適用", "Apply")).clicked() {
                            apply = true;
                            keep_open = false;
                        }
                        if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                            keep_open = false;
                        }
                    });
                });
            if apply && index < self.chain.len() {
                let value = value.clamp(0.25, 4.0);
                self.chain[index]
                    .params
                    .insert("RESIZE_SCALE".to_string(), value);
                log::info!(
                    "resize-scale-edit: index={index} value={value:.2} path={}",
                    self.chain[index].path
                );
                self.apply_live();
            } else if keep_open {
                self.resize_scale_editor = Some((index, value.clamp(0.25, 4.0)));
            }
        }
        if self.settings.ui_mode != UiMode::Mini && self.hotkey_editor_open {
            let mut open = true;
            let mut save = false;
            let mut cancel = false;
            let captured = capture_hotkey_candidate(ctx);
            let captured_this_frame = captured.is_some();
            if let Some(pending) = captured {
                self.hotkey_capture_pending = Some(pending);
                self.hotkey_editor_error = None;
            }
            let released = self
                .hotkey_capture_pending
                .as_ref()
                .is_some_and(|(_, key)| {
                    ctx.input(|input| !input.key_down(*key) && input.modifiers.is_none())
                });
            if released {
                if let Some((candidate, _)) = self.hotkey_capture_pending.take() {
                    match validate_user_hotkey(&candidate) {
                        Ok(candidate) => {
                            self.hotkey_editor_candidate = candidate;
                            self.hotkey_editor_error = None;
                        }
                        Err(error) => {
                            self.hotkey_editor_error = Some(hotkey_error_text(lang, &error));
                        }
                    }
                }
            }
            if ctx.input(|input| input.key_pressed(egui::Key::Escape)) {
                cancel = true;
            }
            let held = held_hotkey_modifiers(ctx);
            egui::Window::new(tr(lang, "ショートカット編集", "Edit Shortcut"))
                .id(egui::Id::new("hotkey_editor_window"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    ui.set_min_width(380.0);
                    ui.label(tr(
                        lang,
                        "Ctrl / Alt / Shiftを押しながら、文字・数字・Fキーなどを押してください。",
                        "Hold Ctrl, Alt, or Shift and press a letter, number, function key, or navigation key.",
                    ));
                    ui.add_space(10.0);
                    let pending_display = self
                        .hotkey_capture_pending
                        .as_ref()
                        .map(|(candidate, _)| candidate.as_str());
                    let display = if let Some(candidate) = pending_display {
                        candidate
                    } else if !captured_this_frame && !held.is_empty() {
                        held.as_str()
                    } else {
                        self.hotkey_editor_candidate.as_str()
                    };
                    hotkey_chips(ui, display);
                    ui.add_space(8.0);
                    if let Some(error) = &self.hotkey_editor_error {
                        ui.colored_label(egui::Color32::LIGHT_RED, error);
                    } else {
                        ui.label(
                            egui::RichText::new(tr(
                                lang,
                                "合計2～3キー。修飾キーのみ、Winキー、予約済みキーは使用できません。",
                                "Two or three keys total. Modifier-only, Win-key, and reserved shortcuts are blocked.",
                            ))
                            .size(10.5)
                            .weak(),
                        );
                    }
                    ui.add_space(10.0);
                    ui.horizontal(|ui| {
                        if control_row_button(ui, tr(lang, "保存", "Save")).clicked() {
                            save = true;
                        }
                        if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                            cancel = true;
                        }
                    });
                });
            if save {
                let candidate = self.hotkey_editor_candidate.clone();
                match self.commit_toggle_hotkey(&candidate, lang) {
                    Ok(()) => {
                        self.hotkey_editor_open = false;
                        self.hotkey_editor_error = None;
                        self.hotkey_capture_pending = None;
                    }
                    Err(error) => self.hotkey_editor_error = Some(error),
                }
            } else if cancel || !open {
                self.hotkey_editor_open = false;
                self.hotkey_editor_candidate = self.settings.hotkey_toggle.clone();
                self.hotkey_editor_error = None;
                self.hotkey_capture_pending = None;
            }
        }
        if self.settings.ui_mode != UiMode::Mini && self.filter_picker_open {
            let tree = build_filter_tree(&self.available);
            let mut open = true;
            let mut add: Option<(StageKind, String)> = None;
            egui::Window::new(tr(lang, "フィルター追加", "Add Filter"))
                .id(egui::Id::new("filter_picker_window"))
                .collapsible(false)
                .resizable(true)
                .default_size(egui::vec2(440.0, 520.0))
                .open(&mut open)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        render_filter_tree_picker(ui, &tree, "filter-picker", lang, &mut add);
                    });
                });
            if let Some((kind, path)) = add {
                self.filter_picker_open = false;
                self.add_filter_to_chain(kind, path);
            } else if !open {
                self.filter_picker_open = false;
                log::info!("filter-picker-close: cancelled=true");
            }
        }
        if self.settings.ui_mode != UiMode::Mini && self.save_as_open {
            let mut open = true;
            egui::Window::new(tr(lang, "別名で保存", "Save As"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .open(&mut open)
                .show(ctx, |ui| {
                    let edit = ui.text_edit_singleline(&mut self.save_as_name);
                    if edit.changed() {
                        self.save_as_error = None;
                    }
                    if let Some(error) = self.save_as_error {
                        let message = match error {
                            PresetEditError::EmptyName => {
                                tr(lang, "名前を入力してください。", "Enter a preset name.")
                            }
                            PresetEditError::NameInUse => tr(
                                lang,
                                "同じ名前のプリセットが既にあります。",
                                "A preset with this name already exists.",
                            ),
                            PresetEditError::ActivePresetMissing => tr(
                                lang,
                                "選択中のプリセットが見つかりません。",
                                "The selected preset could not be found.",
                            ),
                        };
                        ui.colored_label(egui::Color32::LIGHT_RED, message);
                    }
                    ui.horizontal(|ui| {
                        if control_row_button(ui, tr(lang, "上書き保存", "Overwrite")).clicked()
                        {
                            let aspect_correction = self.current_preset_aspect_correction();
                            let crop = self.current_preset_crop();
                            let capture_resolution = self.current_preset_capture_resolution();
                            match self.store.overwrite_active_as(
                                &self.save_as_name,
                                &self.chain,
                                aspect_correction,
                                crop,
                                capture_resolution,
                            ) {
                                Ok(name) => {
                                    self.saved_chain = self.chain.clone();
                                    self.saved_aspect_correction = aspect_correction;
                                    self.saved_crop = crop;
                                    self.saved_capture_resolution = capture_resolution;
                                    self.store.save();
                                    self.save_as_open = false;
                                    log::info!(
                                        "preset-overwrite: name={name} aspect_enabled={} aspect_scale={:.2}x{:.2}",
                                        aspect_correction.enabled,
                                        aspect_correction.width_scale,
                                        aspect_correction.height_scale
                                    );
                                }
                                Err(error) => self.save_as_error = Some(error),
                            }
                        }
                        if control_row_button(ui, tr(lang, "別名で保存", "Save As")).clicked()
                        {
                            let aspect_correction = self.current_preset_aspect_correction();
                            let crop = self.current_preset_crop();
                            let capture_resolution = self.current_preset_capture_resolution();
                            match self.store.save_as_new(
                                &self.save_as_name,
                                &self.chain,
                                aspect_correction,
                                crop,
                                capture_resolution,
                            ) {
                                Ok(name) => {
                                    self.saved_chain = self.chain.clone();
                                    self.saved_aspect_correction = aspect_correction;
                                    self.saved_crop = crop;
                                    self.saved_capture_resolution = capture_resolution;
                                    self.store.save();
                                    self.save_as_open = false;
                                    log::info!(
                                        "preset-save-as: name={name} aspect_enabled={} aspect_scale={:.2}x{:.2}",
                                        aspect_correction.enabled,
                                        aspect_correction.width_scale,
                                        aspect_correction.height_scale
                                    );
                                }
                                Err(error) => self.save_as_error = Some(error),
                            }
                        }
                        if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                            self.save_as_open = false;
                            self.save_as_error = None;
                        }
                    });
                });
            if !open {
                self.save_as_open = false;
                self.save_as_error = None;
            }
        }
        if self.settings.ui_mode != UiMode::Mini && self.confirm_delete {
            egui::Window::new(tr(lang, "プリセットの削除", "Delete Preset"))
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.label(
                        i18n::text(lang, "preset.delete_confirm")
                            .replace("{name}", &self.store.data.active),
                    );
                    ui.horizontal(|ui| {
                        if control_row_button(ui, tr(lang, "削除", "Delete")).clicked() {
                            let active = self.store.data.active.clone();
                            self.store.data.presets.retain(|p| p.name != active);
                            if let Some(first) = self.store.data.presets.first() {
                                let name = first.name.clone();
                                self.select_preset(&name);
                            }
                            self.store.save();
                            self.confirm_delete = false;
                        }
                        if control_row_button(ui, tr(lang, "キャンセル", "Cancel")).clicked() {
                            self.confirm_delete = false;
                        }
                    });
                });
        }

        self.render_source_occlusion_notice(ctx, lang);
        self.render_mini_glsl_overload_notice(ctx, lang, &status, running);
        self.render_mini_dialog_host(ctx, lang, &status, running);

        if self.gui_test_screenshot_path.is_some() && !self.gui_test_screenshot_requested {
            self.gui_test_frame_count += 1;
            let required_frames = std::env::var("NEO_GUI_SCREENSHOT_FRAMES")
                .ok()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(5)
                .clamp(1, 300);
            if self.gui_test_frame_count >= required_frames {
                self.gui_test_screenshot_requested = true;
                ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(egui::UserData::default()));
                ctx.request_repaint();
            }
        }
        // Keep GUI responsiveness isolated from the video fast path. Pointer/
        // keyboard input still wakes egui immediately. On a genuinely slow PC,
        // only the periodic GUI fallback is reduced; filter/WGC/Present cadence
        // never receives a sleep/glFinish/yield for GUI responsiveness.
        let gui_dt = ctx.input(|i| i.stable_dt);
        ctx.request_repaint_after(Duration::from_millis(gui_fallback_repaint_ms(
            running, gui_dt,
        )));
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        // v618 production-test exit guard. The ordinary product path is untouched.
        // If a driver/helper teardown stalls after the GUI has already entered the
        // normal close path, guarantee that this opt-in test process cannot remain
        // resident forever. External cursor/source state is restored below before
        // the watchdog deadline can fire; the cursor janitor remains the final
        // process-level recovery net if hard termination becomes necessary.
        if vulkan_onepass::production_one_pass_requested() {
            let line = format!(
                "vulkan-production-glsl: phase=on-exit-begin build={} pid={} watchdog_ms=4500",
                BUILD_ID,
                std::process::id(),
            );
            vulkan_onepass::record_route_once("on-exit-begin", &line);
            let pid = std::process::id();
            let _ = std::thread::Builder::new()
                .name("vulkan-production-exit-watchdog".into())
                .spawn(move || {
                    std::thread::sleep(Duration::from_millis(4_500));
                    let line = format!(
                        "vulkan-production-glsl: result=forced-process-exit reason=shutdown-timeout pid={pid} watchdog_ms=4500 action=process-exit"
                    );
                    vulkan_gpu::record_probe_result(&line);
                    std::process::exit(0);
                });
        }
        // Never leave a temporary transition helper alive during shutdown.
        win32::hide_gui_transition_snapshot();
        if self.gui_hwnd != 0 && win32::is_cloaked(self.gui_hwnd) {
            let _ = win32::set_window_cloaked(self.gui_hwnd, false);
        }
        if std::env::var_os("NEO_GUI_SCREENSHOT").is_none() {
            save_settings(&self.app_dir, &self.settings);
            self.store.save();
        }
        // v583 restores the stable close contract. The caption WndProc already
        // hid the GUI, so allow the render/provider thread its bounded 2.5 s
        // graceful shutdown window without making the user stare at a blocked
        // window. v586 stops native input before slow provider-cache teardown;
        // if the remaining provider cleanup still exceeds the bound, the
        // independent janitor remains the final process-level safety net.
        chidescaler_neo::input::emergency_release_all();
        if vulkan_onepass::production_one_pass_requested() {
            // v622: the selected-GPU Vulkan runtime is render-thread TLS. Avoid
            // vendor vkDestroy* calls from the TLS destructor on application
            // close, then use the ordinary bounded shutdown/join again. This
            // lets the render thread terminate normally instead of detaching a
            // live worker and relying on a later process watchdog.
            vulkan_onepass::prepare_runtime_for_process_exit();
        }
        if chidescaler_neo::render::vulkan_multipass::production_requested() {
            chidescaler_neo::render::vulkan_multipass::prepare_runtime_for_process_exit();
        }
        self.engine.shutdown();
        if vulkan_onepass::production_one_pass_requested() {
            let line = format!(
                "vulkan-production-glsl: phase=engine-shutdown-returned build={} pid={} mode=bounded-join-tls-abandon",
                BUILD_ID,
                std::process::id(),
            );
            vulkan_onepass::record_route_once("engine-shutdown-returned", &line);
        }
        chidescaler_neo::input::emergency_release_all();
        self.hotkeys.stop();
        if vulkan_onepass::production_one_pass_requested() {
            let line = format!(
                "vulkan-production-glsl: phase=on-exit-complete build={} pid={}",
                BUILD_ID,
                std::process::id(),
            );
            vulkan_onepass::record_route_once("on-exit-complete", &line);
        }
    }
}

#[cfg(test)]
mod app_tests {
    use super::*;

    #[test]
    fn gpu_auto_follows_actual_render_adapter() {
        assert_eq!(resolve_compute_gpu_luid(None, Some(0x22)), Some(0x22));
        assert!(!cross_gpu_compute_active(None, Some(0x22)));
    }

    #[test]
    fn explicit_gpu_remains_onnx_authority_when_wgl_mismatches() {
        assert_eq!(resolve_compute_gpu_luid(Some(0x44), Some(0x22)), Some(0x44));
        assert!(cross_gpu_compute_active(Some(0x44), Some(0x22)));
    }

    #[test]
    fn tensorrt_crop_gate_allows_only_saved_active_geometry() {
        let saved = CaptureCrop {
            enabled: true,
            left: 12,
            top: 34,
            right: 56,
            bottom: 78,
        };

        // ON/OFF is intentionally not part of the geometry lock: TensorRT may
        // toggle a preset's saved rectangle without opening numeric editing.
        let same_geometry_disabled = CaptureCrop {
            enabled: false,
            ..saved
        };
        assert!(capture_crop_geometry_matches(same_geometry_disabled, saved));
        assert!(tensorrt_crop_switch_allowed(same_geometry_disabled, saved));

        let same_geometry_enabled = CaptureCrop {
            enabled: true,
            ..saved
        };
        assert!(tensorrt_crop_switch_allowed(same_geometry_enabled, saved));

        // An active unsaved DirectML edit must not be allowed to create a new
        // TensorRT shape merely by toggling the backend.
        let unsaved_active = CaptureCrop {
            enabled: true,
            left: saved.left + 1,
            ..saved
        };
        assert!(!tensorrt_crop_switch_allowed(unsaved_active, saved));

        // Disabled Crop is shape-neutral, so TensorRT can still be entered. If
        // Crop is enabled later under TRT, the UI restores saved geometry first.
        let unsaved_disabled = CaptureCrop {
            enabled: false,
            left: saved.left + 1,
            ..saved
        };
        assert!(tensorrt_crop_switch_allowed(unsaved_disabled, saved));
    }

    #[test]
    fn tensorrt_crop_ui_keeps_numeric_editing_locked_to_directml() {
        let source = include_str!("main.rs");
        assert!(source.contains("self.settings.capture_crop.enabled && !tensorrt_crop_locked"));
        assert!(source.contains("self.settings.capture_crop.left = self.saved_crop.left"));
        assert!(source.contains(
            "onnx-backend-switch-blocked: requested=TensorRT reason=unsaved-crop-geometry"
        ));
        assert!(source.contains("onnx-backend-startup-fallback: requested=TensorRT active=DirectML reason=unsaved-crop-geometry"));
    }

    #[test]
    fn gui_surfaces_use_autonovsync_and_sparse_caption_drag_redraws() {
        let source = include_str!("main.rs");
        assert!(source.contains("present_mode: eframe::wgpu::PresentMode::AutoNoVsync"));
        let runtime = source.split("#[cfg(test)]").next().unwrap_or(source);
        assert!(runtime.contains("fn root_drag_redraw_due"));
        assert!(runtime.contains("Duration::from_millis(200)"));
    }

    #[test]
    fn gui_fallback_throttles_only_the_gui_side() {
        assert_eq!(gui_fallback_repaint_ms(false, 0.200), 250);
        assert_eq!(gui_fallback_repaint_ms(true, 0.016), 521);
        assert_eq!(gui_fallback_repaint_ms(true, 0.050), 521);
        assert_eq!(gui_fallback_repaint_ms(true, 0.100), 521);
    }

    #[test]
    fn frozen_hdr_option_is_hidden_and_cannot_start_hdr_processing() {
        assert!(!HDR_CAPTURE_OPTION_ENABLED);
        let mut settings = Settings::default();
        settings.hdr_capture = true;
        assert!(!hdr_capture_requested(&settings));
    }

    #[test]
    fn full_layout_retargets_both_directions_when_requirements_change() {
        let source = include_str!("main.rs");
        let start = source
            .find("fn resize_full_for_language")
            .expect("resize_full_for_language must exist");
        let tail = &source[start..];
        let end = tail
            .find("fn resource_meter")
            .expect("resource_meter must follow resize_full_for_language");
        let resize = &tail[..end];
        assert!(resize.contains("full_layout_key"));
        assert!(!resize.contains("allow_shrink"));
        assert!(!resize.contains("current_size.map_or"));
    }

    #[test]
    fn full_layout_shrink_uses_snapshot_guard_instead_of_uncovered_surface_shrink() {
        let source = include_str!("main.rs");
        let start = source
            .find("fn resize_full_for_language")
            .expect("resize_full_for_language must exist");
        let tail = &source[start..];
        let end = tail
            .find("fn resource_meter")
            .expect("resource_meter must follow resize_full_for_language");
        let resize = &tail[..end];
        assert!(resize.contains("shrinking_existing_full"));
        assert!(resize.contains("show_gui_transition_snapshot"));
        assert!(resize.contains("mode_change: false"));

        let commit_start = source
            .find("fn commit_pending_ui_mode_if_ready")
            .expect("transition commit function");
        let commit_tail = &source[commit_start..];
        let commit_end = commit_tail
            .find("fn basic_stats_rows")
            .expect("basic_stats_rows boundary");
        let commit = &commit_tail[..commit_end];
        assert!(commit.contains("resize-timeout-cancelled"));
    }

    #[test]
    fn full_chain_height_follows_statistics_mode() {
        assert_eq!(full_chain_height(8, true), 226.0);
        assert_eq!(full_chain_height(3, true), 226.0);
        assert_eq!(full_chain_height(1, true), 170.0);
        assert_eq!(full_chain_height(2, false), 226.0);
        assert_eq!(full_chain_height(4, false), 338.0);
    }

    #[test]
    fn full_settings_reserve_no_statistics_space_when_disabled() {
        assert_eq!(full_settings_height(false, 12), 458.0);
        assert_eq!(full_settings_height(true, 2), 616.0);
    }

    #[test]
    fn full_layout_key_changes_when_statistics_need_more_height() {
        let stats_off = full_layout_key(918.0, 1080.0, 700.0);
        let stats_on = full_layout_key(918.0, 1080.0, 892.0);
        assert_ne!(stats_off, stats_on);
    }

    #[test]
    fn full_wrapped_rows_reserve_only_the_extra_visual_lines() {
        assert_eq!(full_wrapped_row_extra(700.0, 800.0, 32.0), 0.0);
        assert_eq!(full_wrapped_row_extra(801.0, 800.0, 32.0), 32.0);
        assert_eq!(full_wrapped_row_extra(1601.0, 800.0, 32.0), 64.0);
    }

    #[test]
    fn full_bottom_panel_consumes_the_measured_settings_wrap_extra() {
        let source = include_str!("main.rs");
        assert!(source.contains("self.full_settings_wrap_extra = settings_wrap_extra"));
        assert!(source.contains(") + self.full_settings_wrap_extra;"));
    }

    #[test]
    fn fps_cap_ui_reaches_native_240fps_limit() {
        assert_eq!(MAX_FPS_CAP, 240);
    }

    #[test]
    fn aspect_correction_defaults_are_noop_and_bounds_are_conservative() {
        let settings = Settings::default();
        assert!(!settings.aspect_correction);
        assert_eq!(settings.aspect_width_scale, 1.0);
        assert_eq!(settings.aspect_height_scale, 1.0);
        assert_eq!(ASPECT_CORRECTION_SCALE_MIN, 0.50);
        assert_eq!(ASPECT_CORRECTION_SCALE_MAX, 2.00);
    }

    #[test]
    fn japanese_cadence_labels_use_requested_names_and_casing() {
        assert_eq!(i18n::text(UiLanguage::JaJp, "settings.fps_cap"), "FPS上限");
        assert_eq!(
            i18n::text(UiLanguage::JaJp, "settings.smooth_pacing"),
            "描画安定化"
        );
        assert_eq!(i18n::text(UiLanguage::JaJp, "settings.vsync"), "VSync");
    }

    #[test]
    fn full_cadence_row_uses_requested_order_and_contains_tensorrt_only_once() {
        let source = include_str!("main.rs");
        let row = source
            .split("// Row 2 starts after capture resolution")
            .nth(1)
            .expect("cadence row")
            .split("// row 2: options")
            .next()
            .expect("cadence row end");
        let fps_pos = row
            .find("&mut self.settings.fps_cap_enabled")
            .expect("FPS cap");
        let drag_pos = row
            .find("egui::DragValue::new(&mut fps)")
            .expect("FPS value");
        let duplicate_pos = row
            .find("&mut self.settings.duplicate_frame_reduction")
            .expect("duplicate reduction");
        let smooth_pos = row
            .find("&mut self.settings.smooth_pacing")
            .expect("rendering stabilization");
        let vsync_pos = row.find("&mut self.settings.vsync").expect("VSync");
        let tensorrt_pos = row.find("let backend_switching").expect("TensorRT control");
        assert!(fps_pos < drag_pos);
        assert!(drag_pos < duplicate_pos);
        assert!(duplicate_pos < smooth_pos);
        assert!(smooth_pos < vsync_pos);
        assert!(vsync_pos < tensorrt_pos);

        let options = source
            .split("// row 2: options")
            .nth(1)
            .expect("options row")
            .split("// row 2.5: cursor")
            .next()
            .expect("options row end");
        assert!(!options.contains("let backend_switching"));
    }

    #[test]
    fn tensorrt_option_is_hidden_until_backend_is_detected() {
        let unavailable = TensorRtAvailability::default();
        assert!(!tensorrt_option_visible(&unavailable));

        let mut available = TensorRtAvailability::default();
        available.available = true;
        assert!(tensorrt_option_visible(&available));

        let source = include_str!("main.rs");
        let row = source
            .split("// Row 2 starts after capture resolution")
            .nth(1)
            .expect("cadence row")
            .split("// row 2: options")
            .next()
            .expect("cadence row end");
        assert!(row.contains("if tensorrt_option_visible(&self.tensorrt_availability)"));
    }

    #[test]
    fn tensorrt_checkbox_optimistically_reflects_the_requested_backend() {
        let source = include_str!("main.rs");
        let request = source
            .split("fn request_onnx_backend_switch")
            .nth(1)
            .expect("backend request function")
            .split("fn prepare_ui_mode_switch")
            .next()
            .expect("backend request function end");
        let selected = request
            .find("self.onnx_backend_selected = backend")
            .expect("optimistic selection");
        let pending = request
            .find("self.onnx_backend_pending = Some(backend)")
            .expect("pending backend");
        let send = request
            .find("Cmd::SwitchOnnxBackend")
            .expect("backend command");
        assert!(selected < pending && pending < send);
    }

    #[test]
    fn full_width_uses_basic_style_static_measurement_only() {
        let source = include_str!("main.rs");
        let function = source
            .split("fn resize_full_for_language")
            .nth(1)
            .expect("Full width function")
            .split("fn resource_meter")
            .next()
            .expect("Full width function end");
        assert!(function.contains("FULL_LAYOUT_FIXED_RESERVE"));
        assert!(function.contains("required_row_width"));
        assert!(!function.contains("current_content_width"));
        assert!(!function.contains("ui.available_width()"));
        assert!(!function.contains("live(inner="));
    }

    #[test]
    fn popup_heights_keep_mini_compact_and_restore_basic_full_capacity() {
        assert_eq!(language_popup_height(UiMode::Mini), 58.0);
        assert_eq!(language_popup_height(UiMode::Basic), 286.0);
        assert_eq!(language_popup_height(UiMode::Full), 286.0);
        assert_eq!(preset_popup_height(UiMode::Mini), 80.0);
        assert_eq!(preset_popup_height(UiMode::Basic), 324.0);
        assert_eq!(preset_popup_height(UiMode::Full), 324.0);
        assert!(language_popup_height(UiMode::Basic) > MINI_LANGUAGE_POPUP_HEIGHT * 4.0);
        assert!(language_popup_height(UiMode::Full) > MINI_LANGUAGE_POPUP_HEIGHT * 4.0);
        assert!(preset_popup_height(UiMode::Basic) > MINI_PRESET_POPUP_HEIGHT * 4.0);
        assert!(preset_popup_height(UiMode::Full) > MINI_PRESET_POPUP_HEIGHT * 4.0);
    }

    #[test]
    fn preset_popup_state_is_isolated_per_ui_mode() {
        let mini = preset_popup_id(UiMode::Mini);
        let basic = preset_popup_id(UiMode::Basic);
        let full = preset_popup_id(UiMode::Full);
        assert_ne!(mini, basic);
        assert_ne!(mini, full);
        assert_ne!(basic, full);
        assert_ne!(
            preset_popup_egui_id(UiMode::Mini),
            preset_popup_egui_id(UiMode::Basic)
        );
        assert_ne!(
            preset_popup_egui_id(UiMode::Mini),
            preset_popup_egui_id(UiMode::Full)
        );
        assert_ne!(
            preset_popup_egui_id(UiMode::Basic),
            preset_popup_egui_id(UiMode::Full)
        );
    }

    #[test]
    fn target_row_has_a_fixed_non_expanding_height() {
        assert_eq!(TARGET_ROW_HEIGHT, 26.0);
    }

    #[test]
    fn preset_rows_and_scroll_end_keep_explicit_breathing_room() {
        assert_eq!(PRESET_ROW_HEIGHT, 24.0);
        assert!(PRESET_ROW_GAP >= 4.0);
        assert!(PRESET_POPUP_BOTTOM_GUTTER >= 12.0);
    }

    #[test]
    fn preset_popup_height_is_a_viewport_limit_not_a_content_clamp() {
        // The popup content may reserve its viewport height, but must not clamp
        // future preset rows to that height.
        let source = include_str!("main.rs");
        let forbidden = ["ui.set_max_height(", "preset_popup_height)"].concat();
        assert!(!source.contains(&forbidden));
    }

    #[test]
    fn preset_popup_and_scroll_ids_are_bound_to_live_ui_mode() {
        let source = include_str!("main.rs");
        assert!(source.contains(".id(preset_popup_egui_id(popup_mode))"));
        assert!(source.contains(".id_salt((\"preset_scroll\", preset_popup_id(popup_mode)))"));
        assert!(source.contains("egui::Popup::close_all(ctx)"));
    }

    #[test]
    fn full_settings_reserves_a_real_bottom_border_gutter() {
        assert!(FULL_SETTINGS_BOTTOM_GUTTER >= 32.0);
        assert!(FULL_SETTINGS_VISIBLE_BOTTOM_PADDING >= 18.0);
        assert!(FULL_SETTINGS_BOTTOM_GUTTER > FULL_SETTINGS_VISIBLE_BOTTOM_PADDING);
        assert_eq!(full_settings_height(false, 0), 458.0);
        assert!(full_settings_height(true, 3) > full_settings_height(false, 0));
    }

    #[test]
    fn full_stats_off_two_and_three_filter_layouts_exceed_old_clipping_sizes() {
        let two_filters = 166.0 + full_chain_height(2, false) + full_settings_height(false, 2);
        let three_filters = 166.0 + full_chain_height(3, false) + full_settings_height(false, 3);
        assert_eq!(two_filters, 850.0);
        assert_eq!(three_filters, 906.0);
        assert!(two_filters > 722.0);
        assert!(three_filters > 778.0);
    }

    #[test]
    fn full_toolbar_wrap_has_a_dedicated_height_reserve() {
        assert!(FULL_TOOLBAR_WRAP_EXTRA_HEIGHT >= 40.0);
        let one_line = 166.0 + full_chain_height(3, false) + full_settings_height(false, 3);
        let wrapped = one_line + FULL_TOOLBAR_WRAP_EXTRA_HEIGHT;
        assert_eq!(one_line, 906.0);
        assert!(wrapped >= 946.0);
    }

    #[test]
    fn settings_folder_control_is_an_icon_only_portable_action() {
        let source = include_str!("main.rs");
        assert!(
            source.contains("folder_icon_button(ui, i18n::text(lang, \"settings.open_folder\"))")
        );
        assert!(source.contains("win32::open_application_folder(self.gui_hwnd)"));
        assert!(!source.contains("button(i18n::text(lang, \"settings.open_folder\"))"));
    }

    #[test]
    fn localized_checkbox_fix_preserves_geometry_and_uses_a_shared_script_line() {
        let source = include_str!("main.rs");
        assert!(source.contains("ui.add(egui::Checkbox::new("));
        assert!(source.contains("transparent_control_job(job.clone())"));
        assert!(source.contains("response.rect.min.x + ui.spacing().icon_width"));
        assert!(source.contains("checkbox_label_reference(text)"));
        assert_eq!(
            checkbox_label_reference("統計"),
            checkbox_label_reference("描画安定化")
        );
        assert_eq!(
            checkbox_label_reference("Stats"),
            checkbox_label_reference("Keep GUI on top")
        );
    }

    #[test]
    fn modal_action_buttons_all_use_the_centered_control_painter() {
        let source = include_str!("main.rs");
        let modal_source = source
            .split("// ---------- modals ----------")
            .nth(1)
            .expect("modal section")
            .split("if self.gui_test_screenshot_path")
            .next()
            .expect("modal runtime end");
        let stock_button_call = ["ui", ".button("].concat();
        assert!(!modal_source.contains(&stock_button_call));
        for label in [
            "削除",
            "キャンセル",
            "適用",
            "保存",
            "上書き保存",
            "別名で保存",
        ] {
            assert!(modal_source.contains(label), "missing modal label: {label}");
        }
    }

    #[test]
    fn mini_mode_routes_all_window_modals_through_one_dialog_host() {
        let source = include_str!("main.rs");
        assert!(source.contains("neo_mini_dialog_host"));
        assert!(source.contains("cHiDeScaler-Neo Mini Dialog"));
        assert!(source.contains("with_always_on_top()"));
        assert!(source.contains("mini-dialog-host: hwnd="));
        assert!(source.contains("cursor_route=own-window"));
        for variant in [
            "TensorRt",
            "Elevated",
            "FullscreenCaptureNotice",
            "ResizeScale",
            "Hotkey",
            "FilterPicker",
            "SaveAs",
            "DeletePreset",
        ] {
            assert!(source.contains(&format!("Kind::{variant}")));
        }
        // Body content may scroll, but the action row is painted after the
        // ScrollArea so OK/Cancel/Stop remains reachable even in Mini.
        let host = source
            .split("fn render_mini_dialog_host")
            .nth(1)
            .expect("mini dialog host")
            .split("impl eframe::App for App")
            .next()
            .expect("mini dialog host end");
        let scroll = host.find("ScrollArea::vertical()").expect("scroll body");
        let footer = host
            .find("ui.horizontal_centered(|ui| match kind")
            .expect("fixed footer");
        assert!(scroll < footer);
        // The overload warning is intentionally NOT another Mini child viewport:
        // it appears exactly when low-spec GPUs have no WGPU headroom.
        assert!(source.contains("mini-glsl-overload-inline"));
        assert!(!host.contains("GlslOverload"));
        assert!(!host.contains("glsl_overload.pause"));

        // The inline warning must never re-enter egui's wrapping layout.  It
        // is intentionally a fixed-height, pre-laid-out one-line galley. The
        // warning triangle is geometry (not a locale-font glyph), so its center
        // cannot drift when switching fonts/languages.
        let warning = source
            .split("fn render_mini_glsl_overload_notice")
            .nth(1)
            .expect("mini overload warning")
            .split("fn render_mini_dialog_host")
            .next()
            .expect("mini overload warning end");
        assert!(warning.contains("layout_no_wrap"));
        assert!(warning.contains("const WARNING_H: f32 = 25.0"));
        assert!(warning.contains("let icon_center = egui::pos2"));
        assert!(warning.contains("rect.center().y - ink.center().y"));
        assert!(warning.contains("line_segment([top, left]"));
        assert!(!warning.contains("⚠ 性能不足"));
        assert!(!warning.contains("ui.label("));
        assert!(!warning.contains("egui::Frame::new()"));
        assert!(warning.contains("glsl_overload.auto_stop"));
    }

    #[test]
    fn proven_glsl_overload_uses_bounded_grace_then_normal_stop() {
        let main = include_str!("main.rs");
        let engine = include_str!("engine.rs");
        let win32 = include_str!("platform/win32.rs");
        assert!(
            engine.contains("GLSL_OVERLOAD_AUTO_STOP_GRACE: Duration = Duration::from_secs(6)")
        );
        assert!(engine.contains("glsl_overload_auto_stop_deadline"));
        assert!(engine.contains("action=warn-then-normal-stop"));
        assert!(engine.contains("if auto_stop_hold"));
        assert!(main.contains("request_capture_stop(\"glsl-overload-auto-stop\")"));
        assert!(main.contains("deadline-reached action=dispatch-normal-stop"));
        // GUI-hidden warning uses the live magnified content rect and a cached
        // native GDI helper; never reintroduce an overload-time WGPU viewport.
        assert!(main.contains("show_overload_notice_gdi"));
        assert!(main.contains("status.content_rect"));
        assert!(win32.contains("NeoOverloadNoticeGdi"));
        assert!(win32.contains("WS_EX_LAYERED | WS_EX_TRANSPARENT"));
        assert!(win32.contains("backend=native-gdi-no-wgpu"));
        // No separate force-close path: the existing Cmd::Stop route remains
        // the sole shutdown mechanism.
        assert!(!main.contains("Cmd::Shutdown // glsl-overload"));
    }

    #[test]
    fn bundled_distribution_presets_match_the_latest_35_entry_set() {
        let value: serde_json::Value =
            serde_json::from_str(include_str!("../presets.json")).expect("valid presets.json");
        assert_eq!(value["active"], "Anime4K Restore S");
        assert_eq!(value["presets"].as_array().map(Vec::len), Some(35));
        assert_eq!(value["presets"][0]["name"], "Anime4K Restore S");
        assert_eq!(value["presets"][1]["name"], "Anime4K Restore L");
        assert_eq!(value["presets"][17]["name"], "SD Anime V2NR");
        assert_eq!(value["presets"][18]["name"], "SD Anime ***∞***");
        assert_eq!(
            value["presets"][34]["name"],
            "RIFE Lite + Anime4K  [1080p]  FG"
        );
    }

    #[test]
    fn target_and_mini_capture_use_local_optical_centering_only() {
        let source = include_str!("main.rs");
        assert_eq!(TARGET_TEXT_Y_OFFSET, 1.0);
        assert!(source.contains("target_centered_label("));
        assert!(source.contains("target_icon_and_label("));
        assert!(source.contains("strip_prefix('🎯')"));
        assert!(source.contains("rect.center().y - icon_ink.center().y"));
        assert!(source.contains("rect.center().y - label_ink.center().y + TARGET_TEXT_Y_OFFSET"));
        assert!(
            source
                .contains(r#"ink_centered_control_label(ui, tr(lang, "キャプチャ:", "Capture:"))"#)
        );
    }

    #[test]
    fn settings_folder_icon_geometry_is_exactly_centered() {
        let center = egui::pos2(100.0, 50.0);
        let (body, tab) = folder_icon_geometry(center);
        let left = body.left().min(tab.left());
        let right = body.right().max(tab.right());
        let top = body.top().min(tab.top());
        let bottom = body.bottom().max(tab.bottom());
        assert!(((left + right) * 0.5 - center.x).abs() < f32::EPSILON);
        assert!(((top + bottom) * 0.5 - center.y).abs() < f32::EPSILON);
    }

    #[test]
    fn full_width_estimator_measures_localized_capture_toolbar() {
        let source = include_str!("main.rs");
        assert!(source.contains("let row_toolbar = start_button_width + preset_width"));
        assert!(source.contains(".max(row_toolbar)"));
    }

    #[test]
    fn basic_saved_width_can_never_restore_wider_than_full() {
        assert_eq!(basic_restored_width(1133.0), BASIC_MAX_RESTORED_WIDTH);
        assert_eq!(basic_restored_width(720.0), 720.0);
        assert_eq!(basic_restored_width(400.0), BASIC_MIN_SIZE[0]);
        assert!(BASIC_MAX_RESTORED_WIDTH < FULL_MIN_SIZE[0]);
    }

    #[test]
    fn mixed_window_titles_use_one_native_font_family() {
        assert_eq!(
            mixed_text_font_family("YouTube 日本語タイトル", UiLanguage::EnUs),
            locale_font_family(UiLanguage::JaJp)
        );
        assert_eq!(
            mixed_text_font_family("Chrome 한글 제목", UiLanguage::EnUs),
            locale_font_family(UiLanguage::KoKr)
        );
        assert_eq!(
            mixed_text_font_family("Chrome 中文标题", UiLanguage::ZhCn),
            locale_font_family(UiLanguage::ZhCn)
        );
        assert_eq!(
            mixed_text_font_family("Chrome English title", UiLanguage::EnUs),
            locale_font_family(UiLanguage::EnUs)
        );
        assert_eq!(
            mixed_text_font_family("Chrome हिन्दी शीर्षक", UiLanguage::EnUs),
            egui::FontFamily::Name("ui_indic".into())
        );
        assert_eq!(
            mixed_text_font_family("Chrome ชื่อภาษาไทย", UiLanguage::EnUs),
            egui::FontFamily::Name("ui_thai".into())
        );
    }

    #[test]
    fn random_external_minor_locale_is_discovered_and_visually_centerable() {
        // A different non-built-in locale can be selected on each run. This is
        // deliberately mandatory: user-added locales must remain portable and
        // must not reintroduce font-baseline-specific positioning.
        let candidates = [
            ("ka-GE", "ქართული", "KA", "დაწყება"),
            ("hy-AM", "Հայերեն", "HY", "Սկսել"),
            ("is-IS", "Íslenska", "IS", "Hefja"),
        ];
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as usize;
        let (tag, name, short, start) = candidates[seed % candidates.len()];
        let root = std::env::temp_dir().join(format!(
            "neo-random-minor-locale-{}-{seed}",
            std::process::id()
        ));
        let locale_dir = root.join("locales");
        std::fs::create_dir_all(&locale_dir).unwrap();
        let json = serde_json::json!({
            "_language_tag": tag,
            "_language_name": name,
            "_language_code": short,
            "capture.start": start,
            "capture.target": format!("{name} — 日本語 Window")
        });
        std::fs::write(
            locale_dir.join(format!("{tag}.json")),
            serde_json::to_vec_pretty(&json).unwrap(),
        )
        .unwrap();

        let found = i18n::discover_custom_locales(&root);
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].tag, tag);
        assert_eq!(found[0].name, name);
        assert_eq!(found[0].short, short);

        // The common placement calculation is script-independent. Verify that
        // an intentionally off-baseline painted box lands on the exact visual
        // center used by every target-row language and mixed-script title.
        let row = egui::Rect::from_min_size(egui::pos2(20.0, 10.0), egui::vec2(280.0, 26.0));
        let mixed_script_ink =
            egui::Rect::from_min_max(egui::pos2(-2.5, 3.75), egui::pos2(173.0, 19.25));
        let pos = centered_ink_position(row, mixed_script_ink);
        let placed = mixed_script_ink.translate(pos.to_vec2());
        assert!((placed.center().x - row.center().x).abs() < f32::EPSILON);
        assert!((placed.center().y - row.center().y).abs() < f32::EPSILON);

        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn fg_tag_detection_colors_only_the_standalone_marker() {
        assert_eq!(preset_fg_tag_range("RIFE 4.25 Lite  FG"), Some(16..18));
        assert_eq!(preset_fg_tag_range("RIFE 4.25 Lite  FGen"), None);
        assert_eq!(preset_fg_tag_range("FSRCNNX x2"), None);
    }

    #[test]
    fn preset_fg_badge_is_center_aligned_for_every_locale() {
        let source = include_str!("main.rs");
        let start = source.find("fn preset_name_job").expect("preset_name_job");
        let end = source[start..]
            .find("fn chain_stage_name_job")
            .map(|offset| start + offset)
            .expect("chain_stage_name_job");
        let body = &source[start..end];
        assert!(body.contains("small_badge_format(ui, flow_accent_color(), egui::Align::Center)"));
        assert!(!body.contains("UiLanguage::JaJp"));
        assert!(!body.contains("egui::Align::BOTTOM"));
    }

    #[test]
    fn preset_name_markup_hides_delimiters_and_selects_color() {
        assert_eq!(
            preset_markup_segments("A *Σ* **Ω** ***∞*** Z"),
            vec![
                ("A ", None),
                ("Σ", Some(PresetNameAccent::Gold)),
                (" ", None),
                ("Ω", Some(PresetNameAccent::Cyan)),
                (" ", None),
                ("∞", Some(PresetNameAccent::Red)),
                (" Z", None),
            ]
        );
        assert_eq!(
            preset_markup_segments("plain * unmatched"),
            vec![("plain * unmatched", None)]
        );
    }

    #[test]
    fn capture_resolution_crop_off_is_geometry_transparent() {
        let crop = CaptureCrop {
            enabled: false,
            bottom: 23,
            ..CaptureCrop::default()
        };
        assert_eq!(
            capture_resolution_content_size((960, 540), crop),
            (960, 540)
        );
    }

    #[test]
    fn capture_resolution_crop_is_post_capture_and_does_not_expand_canvas() {
        let crop = CaptureCrop {
            enabled: true,
            bottom: 23,
            ..CaptureCrop::default()
        };
        // The raw fixed capture canvas stays 960x540; Crop removes pixels only
        // after capture instead of growing the foreign source to 960x563.
        assert_eq!(
            capture_resolution_content_size((960, 540), crop),
            (960, 517)
        );
    }

    #[test]
    fn capture_resolution_fullscreen_guard_uses_session_origin() {
        // Neo-created 1920x1080 must not block a later 854x480 request.
        assert!(!capture_resolution_fullscreen_guard(Some(false), true));
        // A source that was genuinely fullscreen when capture began remains
        // protected even if its current live rectangle later changes.
        assert!(capture_resolution_fullscreen_guard(Some(true), false));
        // Before a session snapshot exists, fall back to the current HWND.
        assert!(capture_resolution_fullscreen_guard(None, true));
        assert!(!capture_resolution_fullscreen_guard(None, false));
    }

    #[test]
    fn capture_resolution_labels_follow_ui_language() {
        assert_eq!(capture_resolution_label(None, UiLanguage::JaJp), "自動");
        assert_eq!(capture_resolution_label(None, UiLanguage::EnUs), "Auto");
        let size = Some(CaptureResolution { w: 640, h: 480 });
        assert_eq!(
            capture_resolution_label(size, UiLanguage::JaJp),
            "640x480 (4:3)"
        );
        assert_eq!(
            capture_resolution_label(size, UiLanguage::EnUs),
            "640x480 (4:3)"
        );
    }

    #[test]
    fn unspecified_preset_capture_resolution_inherits_current_gui_value() {
        let current = Some(CaptureResolution { w: 1280, h: 720 });
        let preset = None;
        let selected = preset.or(current);
        assert_eq!(selected, current);
    }

    #[test]
    fn fixed_preset_capture_resolution_overrides_current_gui_value() {
        let current = Some(CaptureResolution { w: 1280, h: 720 });
        let preset = Some(CaptureResolution { w: 640, h: 480 });
        let selected = preset.or(current);
        assert_eq!(selected, preset);
    }

    #[test]
    fn capture_resolution_parser_accepts_both_languages() {
        assert_eq!(parse_capture_resolution("自動"), Some(None));
        assert_eq!(parse_capture_resolution("Auto"), Some(None));
        assert_eq!(
            parse_capture_resolution("640x480 (4:3)"),
            Some(Some(CaptureResolution { w: 640, h: 480 }))
        );
        let presets = capture_resolution_presets();
        assert!(presets.contains(&Some(CaptureResolution { w: 480, h: 360 })));
        assert!(presets.contains(&Some(CaptureResolution { w: 960, h: 720 })));
        assert!(presets.contains(&Some(CaptureResolution { w: 1152, h: 648 })));
        assert!(presets.contains(&Some(CaptureResolution { w: 1600, h: 900 })));
        let index_of = |target: CaptureResolution| {
            presets
                .iter()
                .position(|preset| *preset == Some(target))
                .unwrap()
        };
        assert!(
            index_of(CaptureResolution { w: 960, h: 720 })
                < index_of(CaptureResolution { w: 1152, h: 648 })
        );
        assert!(
            index_of(CaptureResolution { w: 1152, h: 648 })
                < index_of(CaptureResolution { w: 1280, h: 720 })
        );
    }

    #[test]
    fn resize_filter_defaults_to_point_seven_five() {
        let mut chain = Vec::new();
        append_filter_stage(
            &mut chain,
            StageKind::Glsl,
            "shaders/Resize/Spline36_Neo.glsl".to_string(),
        );
        assert_eq!(chain[0].params.get("RESIZE_SCALE"), Some(&0.75));
    }

    #[test]
    fn pip_capture_resolution_preserves_current_content_aspect() {
        assert_eq!(
            fit_resolution_preserving_aspect(CaptureResolution { w: 1280, h: 720 }, (758, 426)),
            CaptureResolution { w: 1280, h: 720 }
        );
        assert_eq!(
            fit_resolution_preserving_aspect(CaptureResolution { w: 1280, h: 720 }, (568, 426)),
            CaptureResolution { w: 960, h: 720 }
        );
        assert_eq!(
            fit_resolution_preserving_aspect(CaptureResolution { w: 640, h: 480 }, (854, 480)),
            CaptureResolution { w: 640, h: 360 }
        );
        // Regression: 762x428 used to round to 640x359. WGC then edge-padded
        // that frame to 640x360 while the engine waited forever for 640x359,
        // causing a transition blackout and releasing cursor mapping.
        assert_eq!(
            fit_resolution_preserving_aspect(CaptureResolution { w: 640, h: 480 }, (762, 428)),
            CaptureResolution { w: 640, h: 360 }
        );
        assert!(is_picture_in_picture_title("ピクチャー イン ピクチャー"));
        assert!(is_picture_in_picture_title("Picture-in-Picture"));
    }

    #[test]
    fn foreground_target_is_pinned_during_capture() {
        assert!(!may_follow_foreground_target(true, false, Some(false)));
        assert!(!may_follow_foreground_target(true, true, Some(false)));
    }

    #[test]
    fn elevated_foreground_can_be_selected_but_not_started_without_admin() {
        assert!(may_follow_foreground_target(false, false, Some(true)));
        assert!(may_follow_foreground_target(false, false, Some(false)));
        assert!(may_follow_foreground_target(false, true, Some(true)));
    }

    #[test]
    fn panel_collapse_stays_lurking_chip() {
        let mut chip_lurking = false;
        let mut bar_shown = true;
        let mut leave_at = Some(Instant::now());
        apply_panel_action_state(
            &mut chip_lurking,
            &mut bar_shown,
            &mut leave_at,
            true,
            false,
        );
        assert!(chip_lurking);
        assert!(!bar_shown);
        assert!(leave_at.is_none());
    }

    #[test]
    fn panel_expand_pins_open_bar() {
        let mut chip_lurking = true;
        let mut bar_shown = false;
        let mut leave_at = Some(Instant::now());
        apply_panel_action_state(
            &mut chip_lurking,
            &mut bar_shown,
            &mut leave_at,
            false,
            true,
        );
        assert!(!chip_lurking);
        assert!(bar_shown);
        assert!(leave_at.is_none());
    }

    #[test]
    fn panel_stop_is_idempotent_and_never_becomes_start() {
        assert!(App::should_dispatch_panel_stop(false, true, false, false));
        assert!(App::should_dispatch_panel_stop(true, false, false, false));
        assert!(App::should_dispatch_panel_stop(false, false, false, true));
        assert!(!App::should_dispatch_panel_stop(false, false, true, false));
        assert!(!App::should_dispatch_panel_stop(false, false, false, false));

        let source = include_str!("main.rs");
        let panel_state = source
            .find("let mut panel_stop_requested = false;")
            .expect("panel stop state");
        let dispatch_start = source[panel_state..]
            .find("if panel_stop_requested {")
            .map(|offset| panel_state + offset)
            .expect("panel stop dispatch");
        let dispatch_end = source[dispatch_start..]
            .find("if take_screenshot {")
            .map(|offset| dispatch_start + offset)
            .expect("panel stop dispatch end");
        let body = &source[dispatch_start..dispatch_end];
        assert!(body.contains("self.request_capture_stop(\"floating-panel\")"));
        assert!(!body.contains("self.toggle()"));
    }

    #[test]
    fn capture_controls_do_not_use_ambiguous_toggle_for_stop() {
        let source = include_str!("main.rs");
        assert!(!source.contains("HK_TOGGLE => self.toggle()"));
        assert!(!source.contains(
            "if resp.clicked() && !status.stopping {\n                        self.toggle();"
        ));
        assert!(source.contains("request_capture_stop(\"main-gui-direct\")"));
        assert!(source.contains("request_capture_stop(\"main-gui-keyboard\")"));
        assert!(source.contains("request_capture_stop(\"floating-panel\")"));
        assert!(source.contains("dispatch_toggle_hotkey(&event)"));
    }

    #[test]
    fn hotkey_trace_uses_configured_binding_and_idle_epoch() {
        let source = include_str!("main.rs");
        assert!(source.contains("event.binding"));
        assert!(source.contains("event.received_at <= self.capture_idle_since"));
        assert!(source.contains("hotkey-toggle-stale-ignored"));
        assert!(source.contains("hotkey-toggle-coalesced"));
    }

    #[test]
    fn hover_help_uses_stable_locale_keys_instead_of_english_phrase_matching() {
        let source = include_str!("main.rs");
        let legacy_hover = [".on_hover_text(", "tr("].concat();
        assert!(!source.contains(&legacy_hover));
        for key in [
            "smooth.help",
            "interpolation.factor_help",
            "tensorrt.help",
            "tensorrt.pack_required",
            "tensorrt.unavailable_help",
            "panel.gui_topmost_enable_help",
            "panel.gui_topmost_disable_help",
        ] {
            assert!(source.contains(key), "missing keyed tooltip: {key}");
        }
        for obsolete in [
            ["Disable GUI always ", "on top"].concat(),
            ["Keep GUI always ", "on top"].concat(),
            ["CUDA fallback: {} ", "filter(s)"].concat(),
            ["DirectML fallback: {} ", "filter(s)"].concat(),
        ] {
            assert!(!source.contains(&obsolete));
        }
    }

    #[test]
    fn gui_statistics_follow_live_filter_chain_order() {
        let chain = vec![
            StageSpec {
                kind: StageKind::Onnx,
                path: "models/RIFE/rife_v4.22_lite_fp16.onnx".into(),
                enabled: true,
                params: Default::default(),
            },
            StageSpec {
                kind: StageKind::Onnx,
                path: "models/AnimeJaNai/2x_AnimeJaNai.onnx".into(),
                enabled: true,
                params: Default::default(),
            },
        ];
        // Simulate a stale/asynchronous arrival order.
        let rows = vec![
            (
                "2x_AnimeJaNai.onnx [DirectML]".to_string(),
                StageStat {
                    kind: "onnx".into(),
                    ms: 14.0,
                },
            ),
            (
                "rife_v4.22_lite_fp16.onnx [DirectML]".to_string(),
                StageStat {
                    kind: "onnx".into(),
                    ms: 10.0,
                },
            ),
        ];

        let ordered = stats_rows_in_filter_chain_order(&chain, rows);
        assert_eq!(ordered[0].0, "rife_v4.22_lite_fp16.onnx [DirectML]");
        assert_eq!(ordered[1].0, "2x_AnimeJaNai.onnx [DirectML]");
    }

    #[test]
    fn gui_statistics_keep_unmeasured_live_shader_visible() {
        let chain = vec![
            StageSpec {
                kind: StageKind::Onnx,
                path: "models/rife_v4.22_lite_fp16.onnx".into(),
                enabled: true,
                params: Default::default(),
            },
            StageSpec {
                kind: StageKind::Glsl,
                path: "shaders/FSRCNNX/FSRCNNX_x2_16_0_4_1.glsl".into(),
                enabled: true,
                params: Default::default(),
            },
        ];
        // This is the transient state seen after metrics.reset(): the ONNX
        // worker reports immediately while the asynchronous GLSL timer has not
        // produced a new sample yet. The GLSL row must remain visible.
        let rows = vec![(
            "rife_v4.22_lite_fp16.onnx [DirectML]".to_string(),
            StageStat {
                kind: "onnx".into(),
                ms: 13.8,
            },
        )];

        let ordered = stats_rows_in_filter_chain_order(&chain, rows);
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].0, "rife_v4.22_lite_fp16.onnx [DirectML]");
        assert_eq!(ordered[1].0, "FSRCNNX_x2_16_0_4_1.glsl");
        assert!(ordered[1].1.ms < 0.0);
        assert_eq!(ordered[1].1.kind, "glsl");
    }

    #[test]
    fn gui_statistics_preserve_duplicate_filter_occurrences() {
        let chain = vec![
            StageSpec {
                kind: StageKind::Glsl,
                path: "shaders/Test.glsl".into(),
                enabled: true,
                params: Default::default(),
            },
            StageSpec {
                kind: StageKind::Glsl,
                path: "shaders/Test.glsl".into(),
                enabled: true,
                params: Default::default(),
            },
        ];
        let rows = vec![
            (
                "Test.glsl #2".to_string(),
                StageStat {
                    kind: "glsl".into(),
                    ms: 2.0,
                },
            ),
            (
                "Test.glsl #1".to_string(),
                StageStat {
                    kind: "glsl".into(),
                    ms: 1.0,
                },
            ),
        ];

        let ordered = stats_rows_in_filter_chain_order(&chain, rows);
        assert_eq!(ordered.len(), 2);
        assert_eq!(ordered[0].0, "Test.glsl #1");
        assert_eq!(ordered[1].0, "Test.glsl #2");
    }

    #[test]
    fn stopped_chain_accepts_repeated_filter_additions() {
        let mut chain = Vec::new();
        append_filter_stage(
            &mut chain,
            StageKind::Onnx,
            "models/custom/first.onnx".into(),
        );
        append_filter_stage(
            &mut chain,
            StageKind::Glsl,
            "shaders/custom/second.glsl".into(),
        );
        append_filter_stage(
            &mut chain,
            StageKind::Onnx,
            "models/custom/third.onnx".into(),
        );
        assert_eq!(chain.len(), 3);
        assert!(chain.iter().all(|stage| stage.enabled));
        assert_eq!(chain[2].path, "models/custom/third.onnx");
    }

    #[test]
    fn picker_tree_preserves_deep_model_and_shader_folders() {
        let available = vec![
            (
                StageKind::Onnx,
                "models/people/anime/rife/model.onnx".into(),
            ),
            (StageKind::Glsl, "shaders/anime/restore/filter.glsl".into()),
        ];
        let tree = build_filter_tree(&available);
        assert!(
            tree.dirs["people"].dirs["anime"].dirs["rife"]
                .files
                .iter()
                .any(|(_, path, _)| path.ends_with("model.onnx"))
        );
        assert!(
            tree.dirs["anime"].dirs["restore"]
                .files
                .iter()
                .any(|(_, path, _)| path.ends_with("filter.glsl"))
        );
    }

    #[test]
    fn builtin_neoflow_is_rendered_above_filter_folders() {
        let available = vec![
            (StageKind::Flow, "builtin:NeoFlow".into()),
            (StageKind::Glsl, "shaders/Anime4K/filter.glsl".into()),
        ];
        let tree = build_filter_tree(&available);
        let ctx = egui::Context::default();
        test_ui_rects().lock().unwrap().clear();
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let mut add = None;
            render_filter_tree_picker(ui, &tree, "order-test", UiLanguage::JaJp, &mut add);
        });
        let rects = test_ui_rects().lock().unwrap();
        let builtin = rects["filter-item:builtin:NeoFlow"];
        let folder = rects["filter-folder:order-test/Anime4K"];
        assert!(builtin.center().y < folder.center().y);
    }

    #[test]
    fn picker_widget_accepts_repeated_real_click_events() {
        fn raw_input(events: Vec<egui::Event>) -> egui::RawInput {
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(800.0, 600.0),
                )),
                events,
                ..Default::default()
            }
        }

        fn picker_frame(
            ctx: &egui::Context,
            tree: &FilterNode,
            events: Vec<egui::Event>,
        ) -> Option<(StageKind, String)> {
            let mut add = None;
            let _ = ctx.run_ui(raw_input(events), |ui| {
                render_filter_tree_picker(ui, tree, "repeat-test", UiLanguage::JaJp, &mut add);
            });
            add
        }

        let path = "models/repeat.onnx".to_string();
        let tree = build_filter_tree(&[(StageKind::Onnx, path.clone())]);
        let ctx = egui::Context::default();
        assert!(picker_frame(&ctx, &tree, Vec::new()).is_none());
        let rect = test_ui_rects().lock().unwrap()[&format!("filter-item:{path}")];
        let click = || {
            vec![
                egui::Event::PointerMoved(rect.center()),
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos: rect.center(),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ]
        };

        let first = picker_frame(&ctx, &tree, click());
        assert_eq!(first, Some((StageKind::Onnx, path.clone())));
        let _ = picker_frame(&ctx, &tree, Vec::new());
        let second = picker_frame(&ctx, &tree, click());
        assert_eq!(second, Some((StageKind::Onnx, path)));
    }

    #[test]
    fn hotkey_editor_captures_three_key_combination() {
        let ctx = egui::Context::default();
        let mut captured = None;
        let modifiers = egui::Modifiers {
            ctrl: true,
            alt: true,
            ..Default::default()
        };
        let input = egui::RawInput {
            events: vec![egui::Event::Key {
                key: egui::Key::Z,
                physical_key: Some(egui::Key::Z),
                pressed: true,
                repeat: false,
                modifiers,
            }],
            ..Default::default()
        };
        let _ = ctx.run_ui(input, |ui| {
            captured = capture_hotkey_candidate(ui.ctx());
        });
        assert_eq!(captured, Some(("Ctrl+Alt+Z".into(), egui::Key::Z)));
    }
    #[test]
    fn control_panel_uses_native_gdi_host_without_wgpu_child_viewport() {
        let source = include_str!("main.rs");
        let control = source
            .split("fn control_panel")
            .nth(1)
            .expect("control_panel source")
            .split("fn compositor_keepalive_anchor")
            .next()
            .expect("control_panel boundary");
        assert!(control.contains("ensure_panel_gdi_host"));
        assert!(control.contains("update_panel_gdi_mirror"));
        assert!(!control.contains("show_viewport_immediate("));
        assert!(!control.contains("show_viewport_deferred("));
    }

    #[test]
    fn compositor_keepalive_is_separate_from_native_panel_and_off_only() {
        let source = include_str!("main.rs");
        let anchor = source
            .split("fn compositor_keepalive_anchor")
            .nth(1)
            .expect("compositor anchor source")
            .split("fn request_filtered_screenshot")
            .next()
            .expect("compositor anchor boundary");
        assert!(anchor.contains("(!self.settings.gui_topmost || self.gui_topmost_off_pending)"));
        assert!(anchor.contains("show_viewport_immediate("));
        assert!(!anchor.contains("with_mouse_passthrough(true)"));
        assert!(anchor.contains("place_below(status.overlay_hwnd, anchor)"));
        assert!(anchor.contains("keep_cursor_sprite_on_top()"));
    }
    #[test]
    fn gui_topmost_off_cloaks_before_anchor_creation_and_retains_cloak_while_running() {
        let source = include_str!("main.rs");
        let toggle = source
            .split("fn toggle_gui_topmost")
            .nth(1)
            .expect("toggle_gui_topmost source")
            .split("fn control_panel")
            .next()
            .expect("toggle boundary");
        let cloak = toggle
            .find("set_window_cloaked(self.gui_hwnd, true)")
            .expect("OFF cloak");
        let pending = toggle
            .find("self.gui_topmost_off_pending = true")
            .expect("OFF pending");
        assert!(
            cloak < pending,
            "GUI must be cloaked before anchor creation is armed"
        );
        assert!(
            toggle.contains("GUI topmost on cloak release"),
            "TOPMOST ON must explicitly release the retained OFF cloak"
        );

        let update = source
            .split("fn update(&mut self")
            .nth(1)
            .expect("update source")
            .split("fn commit_gui_topmost_off")
            .next()
            .expect("update boundary");
        assert!(
            update.contains("&& !self.gui_topmost_off_pending"),
            "fail-visible retry must not uncloak a staged TOPMOST-OFF transition"
        );

        let commit = source
            .split("fn commit_gui_topmost_off")
            .nth(1)
            .expect("commit source")
            .split("fn toggle_gui_topmost")
            .next()
            .expect("commit boundary");
        let demote = commit
            .find("set_own_topmost(self.gui_hwnd, false)")
            .expect("GUI demote");
        let restack = commit
            .find("recommit_overlay_below_helpers(0, overlay)")
            .expect("TOPMOST helper restack");
        let retain = commit
            .find("GUI topmost off cloak retained")
            .expect("running OFF cloak retention");
        assert!(demote < restack, "GUI must be demoted while still cloaked");
        assert!(
            restack < retain,
            "helper order must settle before the OFF cloak is retained"
        );
    }
}
