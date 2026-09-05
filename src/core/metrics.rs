//! Cross-cutting metrics: per-stage ms (EWMA), fps, resolutions.
//! Zero overhead when disabled (the engine passes probe=None).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

const DELAY_FRAME_MS_60FPS: f64 = 1000.0 / 60.0;

#[derive(Clone, Debug, Default)]
pub struct StageStat {
    pub kind: String,
    pub ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub stage_order: Vec<String>,
    pub stages: HashMap<String, StageStat>,
    pub total_ms: f64,
    pub present_fps: f64,
    pub capture_fps: f64,
    /// Timestamp-derived content cadence. Unlike capture_fps, this is not
    /// reduced merely because the render thread was busy.
    pub source_fps: f64,
    pub monitor_size: (i32, i32),
    pub monitor_refresh_hz: f64,
    /// User-facing processing delay in source-frame units, derived from the
    /// measured source/capture-to-present latency.
    pub lag_frames: u64,
    pub in_size: (i32, i32),
    /// Resolution produced by the enabled filter chain before final display fit.
    pub internal_size: (i32, i32),
    /// Resolution actually presented by the magnified overlay.
    pub out_size: (i32, i32),
    pub onnx_note: String,
}

impl Snapshot {
    pub fn display_stages(&self) -> Vec<(String, StageStat)> {
        let mut rows = Vec::new();
        let interp_summary_name = self.stage_order.iter().find_map(|name| {
            self.stages
                .get(name)
                .filter(|st| is_frame_interpolation_summary(name, st))
                .map(|_| name.clone())
        });
        let mut interp_row: Option<(usize, String, StageStat)> = None;
        for name in &self.stage_order {
            let Some(st) = self.stages.get(name) else {
                continue;
            };
            let is_interp_summary = interp_summary_name.as_deref() == Some(name.as_str());
            if is_interp_summary || is_frame_interpolation_detail(name) {
                let label = interp_summary_name
                    .clone()
                    .unwrap_or_else(|| "Frame interpolation".to_string());
                let stat = StageStat {
                    kind: st.kind.clone(),
                    ms: st.ms,
                };
                match &mut interp_row {
                    Some((_, _, best)) if best.ms >= stat.ms => {}
                    Some((_, best_label, best)) => {
                        *best_label = label;
                        *best = stat;
                    }
                    None => interp_row = Some((rows.len(), label, stat)),
                }
            } else {
                rows.push((name.clone(), st.clone()));
            }
        }
        if let Some((idx, label, stat)) = interp_row {
            rows.insert(idx.min(rows.len()), (label, stat));
        }
        rows
    }

    /// Produce rows from an explicit current-chain order. This is used by the
    /// engine log path so a live drag cannot leave the previous chain order in
    /// a metrics snapshot. Rows not present in the declared chain are appended
    /// afterward as diagnostics.
    pub fn display_stages_in_order(&self, declared_order: &[String]) -> Vec<(String, StageStat)> {
        let mut ordered = self.clone();
        ordered.stage_order.clear();
        for name in declared_order {
            if !ordered.stage_order.iter().any(|existing| existing == name) {
                ordered.stage_order.push(name.clone());
            }
        }
        for name in &self.stage_order {
            if self.stages.contains_key(name)
                && !ordered.stage_order.iter().any(|existing| existing == name)
            {
                ordered.stage_order.push(name.clone());
            }
        }
        ordered.display_stages()
    }
}

fn is_frame_interpolation_summary(name: &str, _st: &StageStat) -> bool {
    let lower = name.split(" [").next().unwrap_or(name).to_ascii_lowercase();
    lower == "neoflow"
        || (lower.ends_with(".onnx")
            && (lower.contains("rife")
                || lower.contains("drba")
                || lower.contains("interp")
                || lower.contains("interpolation")
                || lower.contains("dain")
                || lower.contains("ifrnet")))
}

fn is_frame_interpolation_detail(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("frame interpolation")
        || lower.contains("neoflow interp")
        || lower.starts_with("interp ")
}

#[derive(Clone, Default)]
pub struct Metrics {
    inner: Arc<Mutex<Inner>>,
}

#[derive(Default)]
struct Inner {
    gui_enabled: bool,
    panel_enabled: bool,
    configured_stage_order: Vec<String>,
    snap: Snapshot,
    presents: u32,
    captures: u32,
    window_start: Option<std::time::Instant>,
    frame_stage_max_ms: f64,
}

impl Metrics {
    fn clear_snapshot_preserving_monitor(g: &mut Inner) {
        let monitor_size = g.snap.monitor_size;
        let monitor_refresh_hz = g.snap.monitor_refresh_hz;
        g.snap = Snapshot::default();
        g.snap.stage_order = g.configured_stage_order.clone();
        g.snap.monitor_size = monitor_size;
        g.snap.monitor_refresh_hz = monitor_refresh_hz;
    }

    /// Pin statistics rows to the declared GUI filter-chain order.
    ///
    /// Asynchronous interpolation can report its timing before an earlier
    /// image stage finishes. Probe arrival order must therefore never become
    /// the user-facing filter order. Detail-only interpolation probes that are
    /// not part of the chain are still appended after these configured rows.
    pub fn set_stage_order(&self, order: Vec<String>) {
        let mut g = self.inner.lock().unwrap();
        let mut configured = Vec::with_capacity(order.len());
        for name in order {
            if !configured.iter().any(|existing| existing == &name) {
                configured.push(name);
            }
        }
        g.configured_stage_order = configured.clone();

        g.snap.stages.retain(|name, stat| {
            configured.iter().any(|current| current == name)
                || is_frame_interpolation_detail(name)
                || is_frame_interpolation_summary(name, stat)
        });
        let mut stage_order = configured;
        for name in g.snap.stage_order.clone() {
            if g.snap.stages.contains_key(&name)
                && is_frame_interpolation_detail(&name)
                && !stage_order.iter().any(|existing| existing == &name)
            {
                stage_order.push(name);
            }
        }
        g.snap.stage_order = stage_order;
    }

    pub fn set_enabled(&self, on: bool) {
        let mut g = self.inner.lock().unwrap();
        g.gui_enabled = on;
        if !g.gui_enabled && !g.panel_enabled {
            Self::clear_snapshot_preserving_monitor(&mut g);
        }
    }

    /// Lightweight control-panel statistics. This deliberately does not enable
    /// per-stage GPU timing; the movable popup can stay open without adding a
    /// synchronization point to every filter pass.
    pub fn set_panel_enabled(&self, on: bool) {
        let mut g = self.inner.lock().unwrap();
        g.panel_enabled = on;
        if !g.gui_enabled && !g.panel_enabled {
            Self::clear_snapshot_preserving_monitor(&mut g);
        }
    }

    pub fn enabled(&self) -> bool {
        let g = self.inner.lock().unwrap();
        g.gui_enabled || g.panel_enabled
    }

    pub fn detailed_enabled(&self) -> bool {
        self.inner.lock().unwrap().gui_enabled
    }

    pub fn probe(&self, name: &str, kind: &str, ms: f64) {
        let mut g = self.inner.lock().unwrap();
        if !g.gui_enabled {
            return;
        }
        // Asynchronous GL timer results may arrive after a preset switch.
        // Never let a completed query from the previous chain create a stale
        // filter row in the current preset's statistics.
        if !g.configured_stage_order.is_empty()
            && !g
                .configured_stage_order
                .iter()
                .any(|configured| configured == name)
            && !is_frame_interpolation_detail(name)
        {
            return;
        }
        if !g.snap.stages.contains_key(name)
            && !g
                .snap
                .stage_order
                .iter()
                .any(|existing| existing.as_str() == name)
        {
            g.snap.stage_order.push(name.to_string());
        }
        let stage_ms = {
            let e = g.snap.stages.entry(name.to_string()).or_default();
            e.kind = kind.to_string();
            e.ms = if e.ms == 0.0 || (kind == "dlssnr" && (ms < 0.0 || e.ms < 0.0)) {
                ms
            } else {
                e.ms * 0.9 + ms * 0.1
            };
            e.ms
        };
        g.frame_stage_max_ms = g.frame_stage_max_ms.max(ms).max(stage_ms);
    }

    pub fn frame(
        &self,
        presented: bool,
        captures: u32,
        present_latency_ms: Option<f64>,
        pipeline_ms: Option<f64>,
        in_size: (i32, i32),
        out_size: (i32, i32),
    ) {
        let mut g = self.inner.lock().unwrap();
        if !g.gui_enabled && !g.panel_enabled {
            return;
        }
        let now = std::time::Instant::now();
        if presented {
            g.presents += 1;
        }
        g.captures += captures;
        g.snap.in_size = in_size;
        // Idle/duplicate ticks pass (0, 0): they do not replace the displayed image.
        if out_size.0 > 0 && out_size.1 > 0 {
            g.snap.out_size = out_size;
        }
        if let Some(mut total_ms) = present_latency_ms.filter(|ms| ms.is_finite() && *ms >= 0.0) {
            if let Some(process_ms) = pipeline_ms.filter(|ms| ms.is_finite() && *ms >= 0.0) {
                total_ms = total_ms.max(process_ms);
            }
            if g.frame_stage_max_ms.is_finite() {
                total_ms = total_ms.max(g.frame_stage_max_ms);
            }
            g.snap.total_ms = total_ms;
            g.snap.lag_frames = latency_ms_to_frames(total_ms, DELAY_FRAME_MS_60FPS);
        }
        g.frame_stage_max_ms = 0.0;
        match g.window_start {
            None => g.window_start = Some(now),
            Some(t0) => {
                let dt = now.duration_since(t0).as_secs_f64();
                if dt >= 1.0 {
                    g.snap.present_fps = g.presents as f64 / dt;
                    g.snap.capture_fps = g.captures as f64 / dt;
                    g.presents = 0;
                    g.captures = 0;
                    g.window_start = Some(now);
                }
            }
        }
    }

    /// Record a successful overlay presentation that reused an already
    /// processed texture. This keeps the user-facing final-output FPS honest
    /// without pretending that WGC delivered another captured frame or
    /// disturbing per-stage timing from the next real processed frame.
    pub fn present_only(&self) {
        let mut g = self.inner.lock().unwrap();
        if !g.gui_enabled && !g.panel_enabled {
            return;
        }
        let now = std::time::Instant::now();
        g.presents += 1;
        match g.window_start {
            None => g.window_start = Some(now),
            Some(t0) => {
                let dt = now.duration_since(t0).as_secs_f64();
                if dt >= 1.0 {
                    g.snap.present_fps = g.presents as f64 / dt;
                    g.snap.capture_fps = g.captures as f64 / dt;
                    g.presents = 0;
                    g.captures = 0;
                    g.window_start = Some(now);
                }
            }
        }
    }

    pub fn set_note(&self, note: String) {
        self.inner.lock().unwrap().snap.onnx_note = note;
    }

    pub fn set_source_fps(&self, fps: Option<f64>) {
        let mut g = self.inner.lock().unwrap();
        g.snap.source_fps = fps
            .filter(|fps| fps.is_finite() && *fps > 0.0)
            .unwrap_or(0.0);
    }

    /// Record the filter-chain output independently from the final overlay fit.
    pub fn set_internal_size(&self, size: (i32, i32)) {
        let mut g = self.inner.lock().unwrap();
        if g.gui_enabled || g.panel_enabled {
            g.snap.internal_size = size;
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        self.inner.lock().unwrap().snap.clone()
    }

    pub fn set_monitor(&self, size: (i32, i32), refresh_hz: Option<f64>) {
        let mut g = self.inner.lock().unwrap();
        g.snap.monitor_size = size;
        g.snap.monitor_refresh_hz = refresh_hz.unwrap_or(0.0);
    }

    pub fn reset(&self) {
        let mut g = self.inner.lock().unwrap();
        let gui_enabled = g.gui_enabled;
        let panel_enabled = g.panel_enabled;
        let configured_stage_order = g.configured_stage_order.clone();
        let monitor_size = g.snap.monitor_size;
        let monitor_refresh_hz = g.snap.monitor_refresh_hz;
        *g = Inner::default();
        g.gui_enabled = gui_enabled;
        g.panel_enabled = panel_enabled;
        g.configured_stage_order = configured_stage_order.clone();
        g.snap.stage_order = configured_stage_order;
        g.snap.monitor_size = monitor_size;
        g.snap.monitor_refresh_hz = monitor_refresh_hz;
    }
}

fn latency_ms_to_frames(latency_ms: f64, frame_ms: f64) -> u64 {
    if latency_ms <= 0.0 || frame_ms <= 0.0 {
        0
    } else {
        (latency_ms / frame_ms).floor() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::{Metrics, latency_ms_to_frames};

    #[test]
    fn idle_ticks_preserve_displayed_resolution_until_reset() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.frame(true, 1, Some(4.0), Some(4.0), (640, 360), (1280, 720));
        metrics.frame(true, 0, None, None, (640, 360), (0, 0));
        assert_eq!(metrics.snapshot().out_size, (1280, 720));
        metrics.frame(false, 0, None, Some(0.0), (640, 360), (0, 0));
        assert_eq!(metrics.snapshot().out_size, (1280, 720));
        metrics.reset();
        assert_eq!(metrics.snapshot().out_size, (0, 0));
    }

    #[test]
    fn panel_metrics_are_independent_and_do_not_enable_stage_probes() {
        let metrics = Metrics::default();
        metrics.set_panel_enabled(true);
        assert!(metrics.enabled());
        assert!(!metrics.detailed_enabled());

        metrics.probe("expensive stage", "onnx", 40.0);
        metrics.frame(true, 1, Some(12.0), Some(12.0), (640, 360), (1280, 720));
        let snap = metrics.snapshot();
        assert!(snap.stages.is_empty());
        assert!((snap.total_ms - 12.0).abs() < 0.001);

        metrics.set_enabled(true);
        assert!(metrics.detailed_enabled());
        metrics.set_panel_enabled(false);
        assert!(metrics.enabled(), "GUI statistics must remain enabled");
    }

    #[test]
    fn monitor_information_survives_statistics_reset() {
        let metrics = Metrics::default();
        metrics.set_monitor((2560, 1440), Some(120.0));
        metrics.set_enabled(true);
        metrics.reset();
        let snap = metrics.snapshot();
        assert_eq!(snap.monitor_size, (2560, 1440));
        assert_eq!(snap.monitor_refresh_hz, 120.0);
    }

    #[test]
    fn lag_frames_are_floor_of_processing_latency() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.frame(true, 3, Some(8.7), None, (640, 360), (1280, 720));
        assert_eq!(metrics.snapshot().lag_frames, 0);

        metrics.frame(true, 0, None, None, (640, 360), (1280, 720));
        assert_eq!(
            metrics.snapshot().lag_frames,
            0,
            "idle frames must not overwrite the last measured processing delay"
        );

        metrics.frame(true, 1, Some(35.0), None, (640, 360), (1280, 720));
        assert_eq!(metrics.snapshot().lag_frames, 2);
    }

    #[test]
    fn latency_below_one_frame_is_zero_frames() {
        let frame_ms = 1000.0 / 60.0;
        assert_eq!(latency_ms_to_frames(frame_ms - 0.001, frame_ms), 0);
        assert_eq!(latency_ms_to_frames(frame_ms, frame_ms), 1);
        assert_eq!(latency_ms_to_frames(frame_ms * 2.0, frame_ms), 2);
    }

    #[test]
    fn total_ms_is_never_below_stage_work_for_that_frame() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.probe("Frame interpolation", "onnx", 16.2);
        metrics.frame(true, 1, Some(12.0), Some(12.0), (640, 360), (1280, 720));

        let snap = metrics.snapshot();
        assert!(
            snap.total_ms >= 16.2,
            "total_ms={} must include the slowest blocking stage",
            snap.total_ms
        );
        assert_eq!(snap.lag_frames, 0);
    }

    #[test]
    fn total_ms_uses_sixty_fps_frame_unit_not_present_interval() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.frame(true, 1, Some(24.0), Some(24.0), (640, 360), (1280, 720));

        let snap = metrics.snapshot();
        assert_eq!(snap.lag_frames, 1);
        assert!((snap.total_ms - 24.0).abs() < 0.001);
    }

    #[test]
    fn twenty_ms_is_one_delay_frame_at_sixty_fps() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.frame(true, 1, Some(20.0), Some(20.0), (640, 360), (1280, 720));

        assert_eq!(metrics.snapshot().lag_frames, 1);
    }

    #[test]
    fn total_ms_uses_pipeline_time_if_frame_timestamp_underreports() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.frame(true, 1, Some(12.0), Some(20.0), (640, 360), (1280, 720));

        let snap = metrics.snapshot();
        assert!((snap.total_ms - 20.0).abs() < 0.001);
        assert_eq!(snap.lag_frames, 1);
    }

    #[test]
    fn display_stages_preserves_directml_provider_suffix() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.probe("rife_v4.22_lite_fp16.onnx [DirectML]", "onnx", 5.5);
        metrics.probe("Frame interpolation run", "onnx", 5.0);

        let rows = metrics.snapshot().display_stages();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "rife_v4.22_lite_fp16.onnx [DirectML]");
        assert_eq!(rows[0].1.kind, "onnx");
    }

    #[test]
    fn display_stages_collapse_frame_interpolation_details_to_one_row() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);

        metrics.probe("interp RGBA->RGB", "cpu", 1.0);
        metrics.probe("Frame interpolation pack", "cpu", 2.0);
        metrics.probe("Frame interpolation run", "onnx", 16.0);
        metrics.probe("Frame interpolation output", "cpu", 3.0);
        metrics.probe("rife_v4.25_lite.onnx [TensorRT]", "onnx", 18.0);
        metrics.probe("interp RGB upload", "gpu", 0.8);
        metrics.probe("Anime4K", "glsl", 4.0);

        let rows = metrics.snapshot().display_stages();
        let interp_rows: Vec<_> = rows
            .iter()
            .filter(|(name, _)| name == "rife_v4.25_lite.onnx [TensorRT]")
            .collect();
        assert_eq!(interp_rows.len(), 1);
        assert!((interp_rows[0].1.ms - 18.0).abs() < 0.001);
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn display_stages_follow_configured_chain_order_not_probe_arrival_order() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.set_stage_order(vec![
            "2x_AnimeJaNai.onnx [DirectML]".to_string(),
            "rife_v4.22_lite_fp16.onnx [DirectML]".to_string(),
        ]);

        // The asynchronous interpolation worker reports first, but it is the
        // second filter in the user's chain.
        metrics.probe("rife_v4.22_lite_fp16.onnx [DirectML]", "onnx", 10.0);
        metrics.probe("2x_AnimeJaNai.onnx [DirectML]", "onnx", 14.0);

        let rows = metrics.snapshot().display_stages();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].0, "2x_AnimeJaNai.onnx [DirectML]");
        assert_eq!(rows[1].0, "rife_v4.22_lite_fp16.onnx [DirectML]");
    }

    #[test]
    fn configured_stage_order_survives_metrics_reset() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.set_stage_order(vec![
            "rife_v4.22_lite_fp16.onnx [TensorRT]".to_string(),
            "2x_AnimeJaNai.onnx [TensorRT]".to_string(),
        ]);
        metrics.reset();
        metrics.probe("2x_AnimeJaNai.onnx [TensorRT]", "onnx", 4.2);
        metrics.probe("rife_v4.22_lite_fp16.onnx [TensorRT]", "onnx", 2.7);

        let rows = metrics.snapshot().display_stages();
        assert_eq!(rows[0].0, "rife_v4.22_lite_fp16.onnx [TensorRT]");
        assert_eq!(rows[1].0, "2x_AnimeJaNai.onnx [TensorRT]");
    }

    #[test]
    fn stale_async_gpu_result_cannot_add_previous_preset_row() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.set_stage_order(vec!["Current.glsl".to_string()]);
        metrics.probe("Current.glsl", "glsl", 1.5);
        metrics.probe("PreviousPreset.glsl", "glsl", 8.0);

        let rows = metrics.snapshot().display_stages();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].0, "Current.glsl");
    }

    #[test]
    fn explicit_live_chain_order_overrides_stale_snapshot_order() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.set_stage_order(vec![
            "2x_AnimeJaNai.onnx [DirectML]".to_string(),
            "rife_v4.22_lite_fp16.onnx [DirectML]".to_string(),
        ]);
        metrics.probe("2x_AnimeJaNai.onnx [DirectML]", "onnx", 14.0);
        metrics.probe("rife_v4.22_lite_fp16.onnx [DirectML]", "onnx", 10.0);

        let live_order = vec![
            "rife_v4.22_lite_fp16.onnx [DirectML]".to_string(),
            "2x_AnimeJaNai.onnx [DirectML]".to_string(),
        ];
        let rows = metrics.snapshot().display_stages_in_order(&live_order);
        assert_eq!(rows[0].0, "rife_v4.22_lite_fp16.onnx [DirectML]");
        assert_eq!(rows[1].0, "2x_AnimeJaNai.onnx [DirectML]");
    }

    #[test]
    fn internal_size_is_reported_separately_from_display_output() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        metrics.set_internal_size((2560, 1920));
        metrics.frame(true, 1, Some(8.0), Some(8.0), (640, 480), (1280, 960));

        let snap = metrics.snapshot();
        assert_eq!(snap.in_size, (640, 480));
        assert_eq!(snap.internal_size, (2560, 1920));
        assert_eq!(snap.out_size, (1280, 960));
    }

    #[test]
    fn dlssnr_initializing_and_failure_are_not_averaged_as_processing_time() {
        let metrics = Metrics::default();
        metrics.set_enabled(true);
        let name = "DLSS Neural Rendering [D3D12]";
        metrics.probe(name, "dlssnr", -1.0);
        assert_eq!(metrics.snapshot().stages[name].ms, -1.0);
        metrics.probe(name, "dlssnr", 4.0);
        assert_eq!(metrics.snapshot().stages[name].ms, 4.0);
        metrics.probe(name, "dlssnr", -2.0);
        assert_eq!(metrics.snapshot().stages[name].ms, -2.0);
        metrics.probe(name, "dlssnr", 3.0);
        assert_eq!(metrics.snapshot().stages[name].ms, 3.0);
    }
}
