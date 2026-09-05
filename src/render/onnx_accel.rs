//! NeoAccel ONNX acceleration experiments.
//!
//! v691 keeps the rejected v684-v686 temporal-detail/motion-warp experiment
//! retired. Active NeoAccel is backend-aware: TensorRT may probe FP16+INT8,
//! while the DirectML preference may probe the optional Windows ML MIGraphX
//! AMD provider. No active NeoAccel path reuses or synthesizes video frames.

use anyhow::{Result, anyhow};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::{collections::HashMap, fs, path::Path};

static NEOACCEL_GUI_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn set_neoaccel_enabled(enabled: bool) {
    NEOACCEL_GUI_ENABLED.store(enabled, Ordering::Relaxed);
}

const DEFAULT_BLOCK: usize = 64;
const DEFAULT_ALIGNMENT: usize = 16;
const DEFAULT_MAX_PATCH_EDGE: usize = 384;
const DEFAULT_MAX_AFFECTED_RATIO: f32 = 0.42;
const DEFAULT_ARM_FRAMES: u32 = 2;
const DEFAULT_NOISE_TOLERANCE: u8 = 3;
const DEFAULT_PROOF_TOLERANCE: usize = 1;
const DEFAULT_PROOF_MAX_MISMATCH_RATIO: f32 = 0.001;
const DEFAULT_PROOF_PASSES: u32 = 2;
const DEFAULT_REPROOF_INTERVAL: u32 = 300;

// v684 temporal-detail reuse experiment. NeoAccel never chains reconstructed
// frames: one exact ONNX frame may feed at most one reconstructed frame, then
// the next source frame is forced through the original model again.
const TEMPORAL_BLOCK: usize = 32;
const TEMPORAL_SEARCH_RADIUS: i32 = 12;
const TEMPORAL_SEARCH_SAMPLE_STEP: usize = 16;
const TEMPORAL_VALIDATE_SAMPLE_STEP: usize = 8;
const TEMPORAL_MIN_CONFIDENT_RATIO: f32 = 0.70;
const TEMPORAL_MAX_MEAN_ERROR: f32 = 18.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Rect {
    pub x0: usize,
    pub y0: usize,
    pub x1: usize,
    pub y1: usize,
}

impl Rect {
    pub fn width(self) -> usize {
        self.x1.saturating_sub(self.x0)
    }

    pub fn height(self) -> usize {
        self.y1.saturating_sub(self.y0)
    }

    pub fn area(self) -> usize {
        self.width().saturating_mul(self.height())
    }

    pub fn expand(self, x: usize, y: usize, width: usize, height: usize) -> Self {
        Self {
            x0: self.x0.saturating_sub(x),
            y0: self.y0.saturating_sub(y),
            x1: self.x1.saturating_add(x).min(width),
            y1: self.y1.saturating_add(y).min(height),
        }
    }

    pub fn align_out(self, alignment: usize, width: usize, height: usize) -> Self {
        let alignment = alignment.max(1);
        Self {
            x0: (self.x0 / alignment) * alignment,
            y0: (self.y0 / alignment) * alignment,
            x1: self
                .x1
                .div_ceil(alignment)
                .saturating_mul(alignment)
                .min(width),
            y1: self
                .y1
                .div_ceil(alignment)
                .saturating_mul(alignment)
                .min(height),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalModelPlan {
    pub halo_x: usize,
    pub halo_y: usize,
    pub scale_hint: usize,
    pub conv_count: usize,
    pub alignment: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CropGeometry {
    pub crop_w: usize,
    pub crop_h: usize,
    pub touches_left: bool,
    pub touches_top: bool,
    pub touches_right: bool,
    pub touches_bottom: bool,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GeometryValidation {
    passes: u8,
    uses_since_proof: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PatchPlan {
    /// Input-coordinate output area whose pixels can have changed.
    pub affected: Rect,
    /// Input crop needed to evaluate `affected` exactly.
    pub crop: Rect,
    pub geometry: CropGeometry,
}

#[derive(Clone, Debug)]
pub struct TemporalFlowPlan {
    /// RGBA8 coarse motion field. R/G encode signed current->previous source
    /// displacement as byte(value + 128), B is confidence, A stores validation
    /// MAE (255 marks a border cell excluded from global confidence).
    pub rgba: Vec<u8>,
    pub grid_w: usize,
    pub grid_h: usize,
    pub confident_ratio: f32,
    pub mean_error: f32,
}

#[derive(Debug)]
pub enum Decision {
    Bypass,
    EstablishBaseline,
    Reuse,
    Patch(PatchPlan),
}

#[derive(Debug)]
pub struct NeoAccelState {
    pub plan: LocalModelPlan,
    pub previous_input: Vec<u8>,
    pub cached_output: Vec<u8>,
    pub input_size: Option<(usize, usize)>,
    pub output_size: Option<(usize, usize)>,
    pub active: bool,
    pub permanently_disabled: bool,
    pub low_motion_streak: u32,
    pub validated_geometries: HashMap<CropGeometry, GeometryValidation>,
    pub texture_key: String,
    /// v684: persistent exact-AI output, matching ordinary Spline36 baseline,
    /// and coarse current->previous motion field used only on the one skipped
    /// frame between exact ONNX runs.
    pub temporal_ai_key: String,
    pub temporal_base_key: String,
    pub temporal_flow_key: String,
    pub temporal_ready: bool,
    pub temporal_skip_allowed: bool,
    pub temporal_full_frames: u64,
    pub temporal_reused_frames: u64,
    pub proof_tolerance: u8,
    pub proof_max_mismatch_ratio: f32,
    proof_passes_required: u8,
    reproof_interval: u32,
    block: usize,
    max_patch_edge: usize,
    max_affected_ratio: f32,
    arm_frames: u32,
    noise_tolerance: u8,
}

impl NeoAccelState {
    pub fn new(plan: LocalModelPlan, texture_key: String) -> Self {
        let temporal_ai_key = format!("{texture_key}:temporal-ai");
        let temporal_base_key = format!("{texture_key}:temporal-base");
        let temporal_flow_key = format!("{texture_key}:temporal-flow");
        Self {
            plan,
            previous_input: Vec::new(),
            cached_output: Vec::new(),
            input_size: None,
            output_size: None,
            active: false,
            permanently_disabled: false,
            low_motion_streak: 0,
            validated_geometries: HashMap::new(),
            texture_key,
            temporal_ai_key,
            temporal_base_key,
            temporal_flow_key,
            temporal_ready: false,
            temporal_skip_allowed: false,
            temporal_full_frames: 0,
            temporal_reused_frames: 0,
            proof_tolerance: env_usize("CHIDE_NEOACCEL_PROOF_TOLERANCE", DEFAULT_PROOF_TOLERANCE)
                .clamp(0, 4) as u8,
            proof_max_mismatch_ratio: env_f32(
                "CHIDE_NEOACCEL_PROOF_MAX_MISMATCH_RATIO",
                DEFAULT_PROOF_MAX_MISMATCH_RATIO,
            )
            .clamp(0.0, 0.05),
            proof_passes_required: env_u32("CHIDE_NEOACCEL_PROOF_PASSES", DEFAULT_PROOF_PASSES)
                .clamp(1, 8) as u8,
            reproof_interval: env_u32("CHIDE_NEOACCEL_REPROOF_INTERVAL", DEFAULT_REPROOF_INTERVAL)
                .min(10_000),
            block: env_usize("CHIDE_NEOACCEL_BLOCK", DEFAULT_BLOCK).clamp(16, 256),
            max_patch_edge: env_usize("CHIDE_NEOACCEL_MAX_PATCH", DEFAULT_MAX_PATCH_EDGE)
                .clamp(64, 1024),
            max_affected_ratio: env_f32("CHIDE_NEOACCEL_MAX_RATIO", DEFAULT_MAX_AFFECTED_RATIO)
                .clamp(0.05, 0.90),
            arm_frames: env_u32("CHIDE_NEOACCEL_ARM_FRAMES", DEFAULT_ARM_FRAMES).clamp(1, 30),
            noise_tolerance: env_usize(
                "CHIDE_NEOACCEL_NOISE_TOLERANCE",
                DEFAULT_NOISE_TOLERANCE as usize,
            )
            .clamp(0, 8) as u8,
        }
    }

    pub fn reset_for_size(&mut self, width: usize, height: usize) {
        self.previous_input.clear();
        self.cached_output.clear();
        self.input_size = Some((width, height));
        self.output_size = None;
        self.active = false;
        self.temporal_ready = false;
        self.temporal_skip_allowed = false;
        self.low_motion_streak = 0;
        self.validated_geometries.clear();
    }

    pub fn disable_permanently(&mut self) {
        self.permanently_disabled = true;
        self.active = false;
        self.temporal_ready = false;
        self.temporal_skip_allowed = false;
        self.cached_output.clear();
        self.validated_geometries.clear();
    }

    pub fn commit_input(&mut self, rgba: &[u8]) {
        self.previous_input.clear();
        self.previous_input.extend_from_slice(rgba);
    }

    /// Commit a real full-frame ONNX result as the only source allowed to feed
    /// one temporal reuse frame. Reconstructed frames are never committed.
    pub fn temporal_commit_full(
        &mut self,
        width: usize,
        height: usize,
        rgba: &[u8],
        output_size: (usize, usize),
    ) {
        if self.input_size != Some((width, height)) {
            self.reset_for_size(width, height);
        }
        self.commit_input(rgba);
        self.output_size = Some(output_size);
        self.temporal_ready = true;
        self.temporal_skip_allowed = true;
        self.temporal_full_frames = self.temporal_full_frames.saturating_add(1);
    }

    pub fn temporal_commit_reuse(&mut self) {
        // Force the following source frame through the exact model. This is the
        // hard no-drift invariant of the v684 experiment.
        self.temporal_skip_allowed = false;
        self.temporal_reused_frames = self.temporal_reused_frames.saturating_add(1);
    }

    pub fn temporal_can_attempt(&self, width: usize, height: usize, rgba_len: usize) -> bool {
        !self.permanently_disabled
            && self.temporal_ready
            && self.temporal_skip_allowed
            && self.input_size == Some((width, height))
            && self.previous_input.len() == rgba_len
            && self.output_size.is_some()
    }

    /// Estimate a deliberately coarse current->previous motion field from the
    /// already CPU-visible WGC RGBA buffers. This is not frame interpolation:
    /// it only tells the shader where the previous exact AI *detail residual*
    /// came from. Ambiguous/changed blocks receive low confidence and therefore
    /// contribute no old AI detail.
    pub fn temporal_flow_plan(
        &self,
        width: usize,
        height: usize,
        rgba: &[u8],
    ) -> Option<TemporalFlowPlan> {
        if !self.temporal_can_attempt(width, height, rgba.len()) {
            return None;
        }
        let grid_w = width.div_ceil(TEMPORAL_BLOCK);
        let grid_h = height.div_ceil(TEMPORAL_BLOCK);
        let cells = grid_w.checked_mul(grid_h)?;
        if cells == 0 {
            return None;
        }

        let prev = &self.previous_input;
        let mut rgba_flow = vec![0u8; cells * 4];
        rgba_flow
            .par_chunks_mut(4)
            .enumerate()
            .for_each(|(cell, out)| {
                let bx = cell % grid_w;
                let by = cell / grid_w;
                let x0 = bx * TEMPORAL_BLOCK;
                let y0 = by * TEMPORAL_BLOCK;
                let x1 = (x0 + TEMPORAL_BLOCK).min(width);
                let y1 = (y0 + TEMPORAL_BLOCK).min(height);

                #[inline]
                fn luma(buf: &[u8], width: usize, x: usize, y: usize) -> i32 {
                    let i = (y * width + x) * 4;
                    (54 * buf[i] as i32 + 183 * buf[i + 1] as i32 + 19 * buf[i + 2] as i32) >> 8
                }

                // Exhaustive integer search, but only four samples per 32x32
                // block. This costs roughly the same number of comparisons as
                // the old 16-sample coarse search while avoiding its fatal
                // assumption that nearby integer shifts remain correlated.
                let mut search_samples = [(0usize, 0usize, 0i32); 4];
                let mut search_count = 0usize;
                let mut y = y0 + (TEMPORAL_SEARCH_SAMPLE_STEP / 2).min(y1.saturating_sub(y0));
                while y < y1 {
                    let mut x = x0 + (TEMPORAL_SEARCH_SAMPLE_STEP / 2).min(x1.saturating_sub(x0));
                    while x < x1 {
                        if x >= TEMPORAL_SEARCH_RADIUS as usize
                            && y >= TEMPORAL_SEARCH_RADIUS as usize
                            && x + (TEMPORAL_SEARCH_RADIUS as usize) < width
                            && y + (TEMPORAL_SEARCH_RADIUS as usize) < height
                            && search_count < search_samples.len()
                        {
                            search_samples[search_count] = (x, y, luma(rgba, width, x, y));
                            search_count += 1;
                        }
                        x = x.saturating_add(TEMPORAL_SEARCH_SAMPLE_STEP);
                    }
                    y = y.saturating_add(TEMPORAL_SEARCH_SAMPLE_STEP);
                }
                if search_count == 0 {
                    // A narrow border block cannot safely search the full
                    // radius. It contributes no old detail and is excluded from
                    // the global reuse-confidence denominator (A=255 sentinel).
                    out.copy_from_slice(&[128, 128, 0, 255]);
                    return;
                }

                let mut best_dx = 0i32;
                let mut best_dy = 0i32;
                let mut best_sad = u64::MAX;
                for dy in -TEMPORAL_SEARCH_RADIUS..=TEMPORAL_SEARCH_RADIUS {
                    for dx in -TEMPORAL_SEARCH_RADIUS..=TEMPORAL_SEARCH_RADIUS {
                        let mut sad = 0u64;
                        for &(x, y, current_luma) in &search_samples[..search_count] {
                            let px = (x as i32 + dx) as usize;
                            let py = (y as i32 + dy) as usize;
                            sad += (current_luma - luma(prev, width, px, py)).unsigned_abs() as u64;
                        }
                        // Resolve flat/repeating-area ties toward smaller motion.
                        let penalty = (dx.unsigned_abs() + dy.unsigned_abs()) as u64;
                        let cost = sad.saturating_mul(16).saturating_add(penalty);
                        let best_cost = best_sad.saturating_mul(16).saturating_add(
                            (best_dx.unsigned_abs() + best_dy.unsigned_abs()) as u64,
                        );
                        if cost < best_cost {
                            best_sad = sad;
                            best_dx = dx;
                            best_dy = dy;
                        }
                    }
                }

                // Validate the selected vector on a denser 4x4 sample lattice.
                // A moving object, occlusion, subtitle change, fade, etc. can
                // win the sparse search accidentally; this second pass turns
                // such blocks into confidence=0 rather than ghosting old detail.
                let mut validation_sad = 0u64;
                let mut validation_count = 0usize;
                let mut y = y0 + (TEMPORAL_VALIDATE_SAMPLE_STEP / 2).min(y1.saturating_sub(y0));
                while y < y1 {
                    let mut x = x0 + (TEMPORAL_VALIDATE_SAMPLE_STEP / 2).min(x1.saturating_sub(x0));
                    while x < x1 {
                        let px = x as i32 + best_dx;
                        let py = y as i32 + best_dy;
                        if px >= 0 && py >= 0 && px < width as i32 && py < height as i32 {
                            validation_sad += (luma(rgba, width, x, y)
                                - luma(prev, width, px as usize, py as usize))
                            .unsigned_abs() as u64;
                            validation_count += 1;
                        }
                        x = x.saturating_add(TEMPORAL_VALIDATE_SAMPLE_STEP);
                    }
                    y = y.saturating_add(TEMPORAL_VALIDATE_SAMPLE_STEP);
                }
                if validation_count == 0 {
                    out.copy_from_slice(&[128, 128, 0, 255]);
                    return;
                }

                let mae = validation_sad as f32 / validation_count as f32;
                let boundary = best_dx.unsigned_abs() as i32 >= TEMPORAL_SEARCH_RADIUS
                    || best_dy.unsigned_abs() as i32 >= TEMPORAL_SEARCH_RADIUS;
                let mut confidence = if mae <= 3.0 {
                    255u8
                } else if mae <= 8.0 {
                    230
                } else if mae <= 14.0 {
                    190
                } else if mae <= TEMPORAL_MAX_MEAN_ERROR {
                    150
                } else {
                    0
                };
                if boundary {
                    confidence = confidence.min(96);
                }
                out[0] = (best_dx + 128).clamp(0, 255) as u8;
                out[1] = (best_dy + 128).clamp(0, 255) as u8;
                out[2] = confidence;
                out[3] = mae.clamp(0.0, 254.0) as u8;
            });

        let mut confident = 0usize;
        let mut valid = 0usize;
        let mut error_sum = 0u64;
        for cell in rgba_flow.chunks_exact(4) {
            if cell[3] == 255 {
                continue;
            }
            valid += 1;
            if cell[2] >= 150 {
                confident += 1;
            }
            error_sum += cell[3] as u64;
        }
        if valid == 0 {
            return None;
        }
        let confident_ratio = confident as f32 / valid as f32;
        let mean_error = error_sum as f32 / valid as f32;
        if confident_ratio < TEMPORAL_MIN_CONFIDENT_RATIO || mean_error > TEMPORAL_MAX_MEAN_ERROR {
            return None;
        }
        Some(TemporalFlowPlan {
            rgba: rgba_flow,
            grid_w,
            grid_h,
            confident_ratio,
            mean_error,
        })
    }

    pub fn geometry_needs_proof(&self, geometry: CropGeometry) -> bool {
        let Some(validation) = self.validated_geometries.get(&geometry) else {
            return true;
        };
        validation.passes < self.proof_passes_required
            || (self.reproof_interval != 0 && validation.uses_since_proof >= self.reproof_interval)
    }

    pub fn record_geometry_proof(&mut self, geometry: CropGeometry) {
        let validation = self.validated_geometries.entry(geometry).or_default();
        validation.passes = validation.passes.saturating_add(1);
        validation.uses_since_proof = 0;
    }

    pub fn record_geometry_use(&mut self, geometry: CropGeometry) {
        if let Some(validation) = self.validated_geometries.get_mut(&geometry) {
            validation.uses_since_proof = validation.uses_since_proof.saturating_add(1);
        }
    }

    pub fn decide(&mut self, width: usize, height: usize, rgba: &[u8]) -> Decision {
        if self.permanently_disabled || rgba.len() < width.saturating_mul(height).saturating_mul(4)
        {
            return Decision::Bypass;
        }
        if self.input_size != Some((width, height)) {
            self.reset_for_size(width, height);
        }
        if self.previous_input.len() != rgba.len() {
            self.commit_input(rgba);
            return Decision::Bypass;
        }

        let dirty = dirty_bounds_rgba8(
            &self.previous_input,
            rgba,
            width,
            height,
            self.block,
            self.noise_tolerance,
        );

        if !self.active {
            let low_motion = dirty
                .map(|rect| rect.area() as f32 / (width * height).max(1) as f32 <= 0.20)
                .unwrap_or(true);
            self.low_motion_streak = if low_motion {
                self.low_motion_streak.saturating_add(1)
            } else {
                0
            };
            if self.low_motion_streak >= self.arm_frames {
                self.low_motion_streak = 0;
                return Decision::EstablishBaseline;
            }
            self.commit_input(rgba);
            return Decision::Bypass;
        }

        let Some(dirty) = dirty else {
            // Keep the last inferred input as the reference. Random ±1..3
            // compression noise remains suppressed, while a real gradual
            // fade accumulates until it crosses the structural threshold.
            return Decision::Reuse;
        };

        // Changed inputs affect outputs within one receptive-field radius. To
        // compute those outputs with identical context, the input crop needs a
        // second radius around that affected area.
        let affected = dirty
            .expand(self.plan.halo_x, self.plan.halo_y, width, height)
            .align_out(self.plan.alignment, width, height);
        let minimum_crop = affected
            .expand(self.plan.halo_x, self.plan.halo_y, width, height)
            .align_out(self.plan.alignment, width, height);

        let affected_ratio = affected.area() as f32 / (width * height).max(1) as f32;
        if affected_ratio > self.max_affected_ratio
            || minimum_crop.width() > self.max_patch_edge
            || minimum_crop.height() > self.max_patch_edge
            || minimum_crop.width() < 8
            || minimum_crop.height() < 8
        {
            self.active = false;
            self.cached_output.clear();
            self.output_size = None;
            self.low_motion_streak = 0;
            self.validated_geometries.clear();
            self.commit_input(rgba);
            return Decision::Bypass;
        }

        // Keep a stable DirectML graph shape instead of recompiling for every
        // slightly different dirty rectangle. Interior patches normally use
        // one fixed WxH shape; only small source frames and edge contact vary.
        let target_w = stable_patch_extent(self.max_patch_edge, self.plan.alignment, width);
        let target_h = stable_patch_extent(self.max_patch_edge, self.plan.alignment, height);
        let crop = fit_rect_to_extent(
            minimum_crop,
            target_w,
            target_h,
            self.plan.alignment,
            width,
            height,
        );

        Decision::Patch(PatchPlan {
            affected,
            crop,
            geometry: CropGeometry {
                crop_w: crop.width(),
                crop_h: crop.height(),
                touches_left: crop.x0 == 0,
                touches_top: crop.y0 == 0,
                touches_right: crop.x1 == width,
                touches_bottom: crop.y1 == height,
            },
        })
    }
}

pub fn neoaccel_enabled() -> bool {
    if let Ok(value) = std::env::var("CHIDE_NEOACCEL") {
        return !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "directml" | "disabled"
        );
    }
    NEOACCEL_GUI_ENABLED.load(Ordering::Relaxed)
}

/// Conservatively recognize image models that can be evaluated on independent
/// crops without changing their result. Unsupported/global graphs return None.
pub fn analyze_local_image_model(path: &Path) -> Result<Option<LocalModelPlan>> {
    if !neoaccel_enabled() {
        return Ok(None);
    }
    let bytes = fs::read(path)?;
    let graph = proto_fields(&bytes)
        .find_map(|field| (field.number == 7).then_some(field.bytes).flatten())
        .ok_or_else(|| anyhow!("ONNX ModelProto has no graph"))?;

    let mut conv_count = 0usize;
    let mut halo_x = 0usize;
    let mut halo_y = 0usize;
    let mut scale_hint = 1usize;
    let mut spatial_rearrangement_seen = false;

    for node in proto_fields(graph).filter(|field| field.number == 1) {
        let Some(node_bytes) = node.bytes else {
            return Ok(None);
        };
        let mut op_type = None::<String>;
        let mut attrs = Vec::<Attribute>::new();
        for field in proto_fields(node_bytes) {
            match (field.number, field.bytes) {
                (4, Some(value)) => {
                    op_type = Some(String::from_utf8_lossy(value).into_owned());
                }
                (5, Some(value)) => attrs.push(parse_attribute(value)?),
                _ => {}
            }
        }
        let Some(op) = op_type else {
            return Ok(None);
        };
        match op.as_str() {
            "Constant" | "Cast" | "Identity" | "PRelu" | "Relu" | "LeakyRelu" | "Add" | "Sub"
            | "Mul" | "Div" | "Clip" | "Sigmoid" | "Tanh" | "Softplus" => {}
            "Conv" => {
                // A convolution after Resize/DepthToSpace lives in a different
                // coordinate system. Reject it rather than risk underestimating
                // the input-space receptive field.
                if spatial_rearrangement_seen {
                    return Ok(None);
                }
                conv_count += 1;
                let kernel = attr_ints(&attrs, "kernel_shape").unwrap_or_else(|| vec![3, 3]);
                let dilation = attr_ints(&attrs, "dilations").unwrap_or_else(|| vec![1, 1]);
                let strides = attr_ints(&attrs, "strides").unwrap_or_else(|| vec![1, 1]);
                if kernel.len() != 2
                    || dilation.len() != 2
                    || strides.as_slice() != &[1, 1]
                    || kernel.iter().any(|value| *value == 0 || *value % 2 == 0)
                {
                    return Ok(None);
                }
                let ky = usize::try_from(kernel[0]).ok();
                let kx = usize::try_from(kernel[1]).ok();
                let dy = usize::try_from(dilation[0]).ok();
                let dx = usize::try_from(dilation[1]).ok();
                let (Some(ky), Some(kx), Some(dy), Some(dx)) = (ky, kx, dy, dx) else {
                    return Ok(None);
                };
                halo_x = halo_x.saturating_add(((kx - 1) * dx).div_ceil(2));
                halo_y = halo_y.saturating_add(((ky - 1) * dy).div_ceil(2));
            }
            "DepthToSpace" => {
                let block = attr_i(&attrs, "blocksize").unwrap_or(1);
                let Ok(block) = usize::try_from(block) else {
                    return Ok(None);
                };
                if block == 0 || block > 8 {
                    return Ok(None);
                }
                scale_hint = scale_hint.saturating_mul(block);
                spatial_rearrangement_seen = true;
            }
            "Resize" => {
                let mode = attr_s(&attrs, "mode").unwrap_or(b"nearest");
                let coordinate =
                    attr_s(&attrs, "coordinate_transformation_mode").unwrap_or(b"asymmetric");
                if mode != b"nearest" || coordinate != b"asymmetric" {
                    return Ok(None);
                }
                spatial_rearrangement_seen = true;
            }
            // Spatial reductions, warps, shape-dependent padding, transposed
            // convolutions, normalization and attention are intentionally not
            // accepted. Falling back is cheaper than risking a seam.
            _ => return Ok(None),
        }
    }

    if conv_count == 0
        || conv_count > 64
        || halo_x > 128
        || halo_y > 128
        || !(1..=8).contains(&scale_hint)
    {
        return Ok(None);
    }

    Ok(Some(LocalModelPlan {
        halo_x: halo_x.max(1),
        halo_y: halo_y.max(1),
        scale_hint,
        conv_count,
        // DepthToSpace has a spatial phase. Aligning every crop origin to the
        // least common multiple keeps that phase identical to full-frame
        // inference while retaining RDNA 4-friendly 16-pixel dimensions.
        alignment: lcm(DEFAULT_ALIGNMENT, scale_hint),
    }))
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let remainder = a % b;
        a = b;
        b = remainder;
    }
    a.max(1)
}

fn lcm(a: usize, b: usize) -> usize {
    (a / gcd(a, b)).saturating_mul(b).max(1)
}

fn stable_patch_extent(max_patch_edge: usize, alignment: usize, frame_extent: usize) -> usize {
    if frame_extent <= max_patch_edge {
        return frame_extent;
    }
    let aligned = (max_patch_edge / alignment.max(1)) * alignment.max(1);
    aligned.max(alignment.max(1)).min(frame_extent)
}

fn fit_rect_to_extent(
    required: Rect,
    target_w: usize,
    target_h: usize,
    alignment: usize,
    frame_w: usize,
    frame_h: usize,
) -> Rect {
    fn axis(
        required_start: usize,
        required_end: usize,
        target: usize,
        alignment: usize,
        frame: usize,
    ) -> (usize, usize) {
        let required_len = required_end.saturating_sub(required_start);
        let target = target.max(required_len).min(frame);
        if target == frame {
            return (0, frame);
        }
        let center = required_start.saturating_add(required_len / 2);
        let mut start = center.saturating_sub(target / 2).min(frame - target);
        if start > required_start {
            start = required_start;
        }
        if start.saturating_add(target) < required_end {
            start = required_end.saturating_sub(target);
        }
        let alignment = alignment.max(1);
        start = (start / alignment) * alignment;
        if start.saturating_add(target) < required_end {
            start = required_end
                .saturating_sub(target)
                .div_ceil(alignment)
                .saturating_mul(alignment)
                .min(frame - target);
        }
        (start, start + target)
    }

    let (x0, x1) = axis(required.x0, required.x1, target_w, alignment, frame_w);
    let (y0, y1) = axis(required.y0, required.y1, target_h, alignment, frame_h);
    Rect { x0, y0, x1, y1 }
}

pub fn crop_rgba8(rgba: &[u8], full_width: usize, rect: Rect) -> Result<Vec<u8>> {
    let row_bytes = rect.width().saturating_mul(4);
    let mut out = vec![0u8; row_bytes.saturating_mul(rect.height())];
    for row in 0..rect.height() {
        let src_start = ((rect.y0 + row) * full_width + rect.x0) * 4;
        let src_end = src_start + row_bytes;
        let dst_start = row * row_bytes;
        let Some(src) = rgba.get(src_start..src_end) else {
            return Err(anyhow!("NeoAccel crop exceeds input"));
        };
        out[dst_start..dst_start + row_bytes].copy_from_slice(src);
    }
    Ok(out)
}

pub fn extract_output_patch(
    crop_output: &[u8],
    crop_output_width: usize,
    scale: usize,
    crop: Rect,
    affected: Rect,
) -> Result<Vec<u8>> {
    let src_x = affected.x0.saturating_sub(crop.x0).saturating_mul(scale);
    let src_y = affected.y0.saturating_sub(crop.y0).saturating_mul(scale);
    let patch_w = affected.width().saturating_mul(scale);
    let patch_h = affected.height().saturating_mul(scale);
    let row_bytes = patch_w.saturating_mul(4);
    let mut patch = vec![0u8; row_bytes.saturating_mul(patch_h)];
    for row in 0..patch_h {
        let src_start = ((src_y + row) * crop_output_width + src_x) * 4;
        let src_end = src_start + row_bytes;
        let dst_start = row * row_bytes;
        let Some(src) = crop_output.get(src_start..src_end) else {
            return Err(anyhow!("NeoAccel patch exceeds crop output"));
        };
        patch[dst_start..dst_start + row_bytes].copy_from_slice(src);
    }
    Ok(patch)
}

pub fn write_output_patch(
    full_output: &mut [u8],
    full_output_width: usize,
    scale: usize,
    affected: Rect,
    patch: &[u8],
) -> Result<()> {
    let dst_x = affected.x0.saturating_mul(scale);
    let dst_y = affected.y0.saturating_mul(scale);
    let patch_w = affected.width().saturating_mul(scale);
    let patch_h = affected.height().saturating_mul(scale);
    let row_bytes = patch_w.saturating_mul(4);
    if patch.len() != row_bytes.saturating_mul(patch_h) {
        return Err(anyhow!("NeoAccel patch byte count mismatch"));
    }
    for row in 0..patch_h {
        let dst_start = ((dst_y + row) * full_output_width + dst_x) * 4;
        let dst_end = dst_start + row_bytes;
        let Some(dst) = full_output.get_mut(dst_start..dst_end) else {
            return Err(anyhow!("NeoAccel patch exceeds cached output"));
        };
        let src_start = row * row_bytes;
        dst.copy_from_slice(&patch[src_start..src_start + row_bytes]);
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct PatchProof {
    pub accepted: bool,
    pub mismatched_channels: usize,
    pub compared_channels: usize,
    pub max_delta: u8,
}

/// Compare only the affected RGB output. Shape-dependent GPU convolution
/// kernels can change a tiny number of rounded RGBA8 values by one LSB even
/// when the underlying image is equivalent. The default policy accepts at
/// most a one-step delta in at most 0.1% of RGB channels; both limits are
/// configurable and setting tolerance to zero restores byte-exact proof.
pub fn compare_output_patch(
    full_output: &[u8],
    full_output_width: usize,
    scale: usize,
    affected: Rect,
    patch: &[u8],
    tolerance: u8,
    max_mismatch_ratio: f32,
) -> PatchProof {
    let dst_x = affected.x0.saturating_mul(scale);
    let dst_y = affected.y0.saturating_mul(scale);
    let patch_w = affected.width().saturating_mul(scale);
    let patch_h = affected.height().saturating_mul(scale);
    let row_bytes = patch_w.saturating_mul(4);
    let compared_channels = patch_w.saturating_mul(patch_h).saturating_mul(3);
    if patch.len() != row_bytes.saturating_mul(patch_h) || compared_channels == 0 {
        return PatchProof {
            accepted: false,
            mismatched_channels: 0,
            compared_channels,
            max_delta: u8::MAX,
        };
    }

    let mut mismatched_channels = 0usize;
    let mut max_delta = 0u8;
    let mut within_tolerance = true;
    for row in 0..patch_h {
        let dst_start = ((dst_y + row) * full_output_width + dst_x) * 4;
        let dst_end = dst_start + row_bytes;
        let Some(dst) = full_output.get(dst_start..dst_end) else {
            return PatchProof {
                accepted: false,
                mismatched_channels,
                compared_channels,
                max_delta: u8::MAX,
            };
        };
        let src_start = row * row_bytes;
        let src = &patch[src_start..src_start + row_bytes];
        for (full_px, patch_px) in dst.chunks_exact(4).zip(src.chunks_exact(4)) {
            for channel in 0..3 {
                let delta = full_px[channel].abs_diff(patch_px[channel]);
                max_delta = max_delta.max(delta);
                if delta != 0 {
                    mismatched_channels = mismatched_channels.saturating_add(1);
                }
                if delta > tolerance {
                    within_tolerance = false;
                }
            }
            // Alpha is generated as 255 by both conversion paths. Any alpha
            // discrepancy indicates corruption rather than harmless rounding.
            if full_px[3] != patch_px[3] {
                within_tolerance = false;
                max_delta = max_delta.max(full_px[3].abs_diff(patch_px[3]));
            }
        }
    }
    let mismatch_ratio = mismatched_channels as f32 / compared_channels as f32;
    PatchProof {
        accepted: within_tolerance && mismatch_ratio <= max_mismatch_ratio,
        mismatched_channels,
        compared_channels,
        max_delta,
    }
}

fn dirty_bounds_rgba8(
    previous: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
    block: usize,
    noise_tolerance: u8,
) -> Option<Rect> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0usize;
    let mut max_y = 0usize;
    let mut any = false;

    for y0 in (0..height).step_by(block) {
        let y1 = (y0 + block).min(height);
        for x0 in (0..width).step_by(block) {
            let x1 = (x0 + block).min(width);
            let pixels = (x1 - x0).saturating_mul(y1 - y0);
            // A handful of codec ringing outliers must not dirty a whole
            // block. Real edges/motion exceed this density very quickly.
            let required_strong = (pixels / 256).max(4);
            let mut strong = 0usize;
            for y in y0..y1 {
                let start = (y * width + x0) * 4;
                for x in 0..(x1 - x0) {
                    let i = start + x * 4;
                    let delta = previous[i]
                        .abs_diff(current[i])
                        .max(previous[i + 1].abs_diff(current[i + 1]))
                        .max(previous[i + 2].abs_diff(current[i + 2]));
                    if delta > noise_tolerance {
                        strong += 1;
                        if strong >= required_strong {
                            break;
                        }
                    }
                }
                if strong >= required_strong {
                    break;
                }
            }
            let changed = strong >= required_strong;
            if changed {
                any = true;
                min_x = min_x.min(x0);
                min_y = min_y.min(y0);
                max_x = max_x.max(x1);
                max_y = max_y.max(y1);
            }
        }
    }

    any.then_some(Rect {
        x0: min_x,
        y0: min_y,
        x1: max_x,
        y1: max_y,
    })
}

#[derive(Default, Debug)]
struct Attribute {
    name: String,
    i: Option<u64>,
    s: Option<Vec<u8>>,
    ints: Vec<u64>,
}

fn parse_attribute(bytes: &[u8]) -> Result<Attribute> {
    let mut attr = Attribute::default();
    for field in proto_fields(bytes) {
        match field.number {
            1 => {
                if let Some(value) = field.bytes {
                    attr.name = String::from_utf8_lossy(value).into_owned();
                }
            }
            3 => attr.i = field.varint,
            4 => attr.s = field.bytes.map(ToOwned::to_owned),
            8 => {
                if let Some(value) = field.varint {
                    attr.ints.push(value);
                }
                if let Some(value) = field.bytes {
                    attr.ints.extend(read_packed_varints(value)?);
                }
            }
            _ => {}
        }
    }
    Ok(attr)
}

fn attr_i(attrs: &[Attribute], name: &str) -> Option<u64> {
    attrs.iter().find(|attr| attr.name == name)?.i
}

fn attr_s<'a>(attrs: &'a [Attribute], name: &str) -> Option<&'a [u8]> {
    attrs.iter().find(|attr| attr.name == name)?.s.as_deref()
}

fn attr_ints(attrs: &[Attribute], name: &str) -> Option<Vec<u64>> {
    Some(attrs.iter().find(|attr| attr.name == name)?.ints.clone())
}

#[derive(Clone, Copy)]
struct ProtoField<'a> {
    number: u32,
    varint: Option<u64>,
    bytes: Option<&'a [u8]>,
}

struct ProtoFields<'a> {
    bytes: &'a [u8],
    cursor: usize,
    failed: bool,
}

fn proto_fields(bytes: &[u8]) -> ProtoFields<'_> {
    ProtoFields {
        bytes,
        cursor: 0,
        failed: false,
    }
}

impl<'a> Iterator for ProtoFields<'a> {
    type Item = ProtoField<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed || self.cursor >= self.bytes.len() {
            return None;
        }
        let key = match read_varint(self.bytes, &mut self.cursor) {
            Ok(value) => value,
            Err(_) => {
                self.failed = true;
                return None;
            }
        };
        let number = (key >> 3) as u32;
        let wire = (key & 7) as u8;
        let field = match wire {
            0 => read_varint(self.bytes, &mut self.cursor).map(|value| ProtoField {
                number,
                varint: Some(value),
                bytes: None,
            }),
            1 => take_bytes(self.bytes, &mut self.cursor, 8).map(|value| ProtoField {
                number,
                varint: None,
                bytes: Some(value),
            }),
            2 => read_varint(self.bytes, &mut self.cursor).and_then(|len| {
                let len = usize::try_from(len).map_err(|_| anyhow!("protobuf length overflow"))?;
                take_bytes(self.bytes, &mut self.cursor, len).map(|value| ProtoField {
                    number,
                    varint: None,
                    bytes: Some(value),
                })
            }),
            5 => take_bytes(self.bytes, &mut self.cursor, 4).map(|value| ProtoField {
                number,
                varint: None,
                bytes: Some(value),
            }),
            _ => Err(anyhow!("unsupported protobuf wire type {wire}")),
        };
        match field {
            Ok(field) => Some(field),
            Err(_) => {
                self.failed = true;
                None
            }
        }
    }
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| anyhow!("truncated protobuf varint"))?;
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(anyhow!("protobuf varint is too long"))
}

fn read_packed_varints(bytes: &[u8]) -> Result<Vec<u64>> {
    let mut cursor = 0usize;
    let mut values = Vec::new();
    while cursor < bytes.len() {
        values.push(read_varint(bytes, &mut cursor)?);
    }
    Ok(values)
}

fn take_bytes<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = (*cursor)
        .checked_add(len)
        .ok_or_else(|| anyhow!("protobuf offset overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| anyhow!("truncated protobuf field"))?;
    *cursor = end;
    Ok(value)
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

fn env_f32(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirty_area_and_double_halo_cover_influence() {
        let mut state = NeoAccelState::new(
            LocalModelPlan {
                halo_x: 10,
                halo_y: 8,
                scale_hint: 2,
                conv_count: 10,
                alignment: 16,
            },
            "test".into(),
        );
        state.active = true;
        state.input_size = Some((256, 256));
        state.previous_input = vec![0; 256 * 256 * 4];
        state.cached_output = vec![0; 512 * 512 * 4];
        state.output_size = Some((512, 512));
        let mut current = state.previous_input.clone();
        for x in 96..112 {
            current[(100 * 256 + x) * 4] = 8;
        }
        let Decision::Patch(plan) = state.decide(256, 256, &current) else {
            panic!("expected patch");
        };
        assert!(plan.affected.x0 <= 90 && plan.affected.x1 >= 111);
        assert!(plan.crop.x0 <= 80 && plan.crop.x1 >= 121);
    }

    #[test]
    fn codec_noise_is_reused_but_accumulated_real_change_is_detected() {
        let previous = vec![100u8; 64 * 64 * 4];
        let mut noise = previous.clone();
        for pixel in noise.chunks_exact_mut(4) {
            pixel[0] = 102;
            pixel[1] = 99;
        }
        assert!(dirty_bounds_rgba8(&previous, &noise, 64, 64, 64, 3).is_none());

        let mut fade = previous.clone();
        for pixel in fade.chunks_exact_mut(4) {
            pixel[0] = 104;
            pixel[1] = 104;
            pixel[2] = 104;
        }
        assert!(dirty_bounds_rgba8(&previous, &fade, 64, 64, 64, 3).is_some());
    }

    #[test]
    fn stable_crop_shape_is_aligned_and_contains_required_area() {
        let required = Rect {
            x0: 777,
            y0: 333,
            x1: 913,
            y1: 469,
        };
        let crop = fit_rect_to_extent(required, 384, 384, 16, 1920, 1080);
        assert_eq!(crop.width(), 384);
        assert_eq!(crop.height(), 384);
        assert_eq!(crop.x0 % 16, 0);
        assert_eq!(crop.y0 % 16, 0);
        assert!(crop.x0 <= required.x0 && crop.x1 >= required.x1);
        assert!(crop.y0 <= required.y0 && crop.y1 >= required.y1);
    }

    #[test]
    fn depth_to_space_phase_alignment_keeps_matrix_alignment() {
        assert_eq!(lcm(16, 2), 16);
        assert_eq!(lcm(16, 3), 48);
        assert_eq!(lcm(16, 8), 16);
    }

    #[test]
    fn geometry_requires_multiple_proofs_and_periodic_reproof() {
        let mut state = NeoAccelState::new(
            LocalModelPlan {
                halo_x: 1,
                halo_y: 1,
                scale_hint: 1,
                conv_count: 1,
                alignment: 16,
            },
            "proof-test".into(),
        );
        state.proof_passes_required = 2;
        state.reproof_interval = 3;
        let geometry = CropGeometry {
            crop_w: 384,
            crop_h: 384,
            touches_left: false,
            touches_top: false,
            touches_right: false,
            touches_bottom: false,
        };
        assert!(state.geometry_needs_proof(geometry));
        state.record_geometry_proof(geometry);
        assert!(state.geometry_needs_proof(geometry));
        state.record_geometry_proof(geometry);
        assert!(!state.geometry_needs_proof(geometry));
        for _ in 0..3 {
            state.record_geometry_use(geometry);
        }
        assert!(state.geometry_needs_proof(geometry));
    }

    #[test]
    fn one_lsb_rounding_is_accepted_only_when_sparse() {
        let affected = Rect {
            x0: 0,
            y0: 0,
            x1: 16,
            y1: 16,
        };
        let full = vec![100u8; 16 * 16 * 4];
        let mut patch = full.clone();
        patch[0] = 101;
        let proof = compare_output_patch(&full, 16, 1, affected, &patch, 1, 0.0015);
        assert!(proof.accepted);
        assert_eq!(proof.mismatched_channels, 1);
        assert_eq!(proof.max_delta, 1);

        patch[4] = 102;
        let proof = compare_output_patch(&full, 16, 1, affected, &patch, 1, 0.01);
        assert!(!proof.accepted);
        assert_eq!(proof.max_delta, 2);
    }

    #[test]
    fn temporal_flow_tracks_translation_and_rejects_scene_change() {
        let (width, height) = (384usize, 256usize);
        let mut previous = vec![0u8; width * height * 4];
        let mut value = 0x1234_5678u32;
        for pixel in previous.chunks_exact_mut(4) {
            value = value.wrapping_mul(1664525).wrapping_add(1013904223);
            pixel[0] = (value >> 24) as u8;
            pixel[1] = (value >> 16) as u8;
            pixel[2] = (value >> 8) as u8;
            pixel[3] = 255;
        }
        let mut state = NeoAccelState::new(
            LocalModelPlan {
                halo_x: 0,
                halo_y: 0,
                scale_hint: 2,
                conv_count: 0,
                alignment: 1,
            },
            "temporal-test".into(),
        );
        state.temporal_commit_full(width, height, &previous, (width * 2, height * 2));

        // Current picture is previous shifted right 5 / down 3, therefore a
        // current pixel maps back to previous at (-5,-3).
        let mut shifted = vec![0u8; previous.len()];
        for y in 3..height {
            for x in 5..width {
                let dst = (y * width + x) * 4;
                let src = ((y - 3) * width + (x - 5)) * 4;
                shifted[dst..dst + 4].copy_from_slice(&previous[src..src + 4]);
            }
        }
        let flow = state
            .temporal_flow_plan(width, height, &shifted)
            .expect("translated picture should be reusable");
        let center_x = flow.grid_w / 2;
        let center_y = flow.grid_h / 2;
        let i = (center_y * flow.grid_w + center_x) * 4;
        assert_eq!(flow.rgba[i] as i32 - 128, -5);
        assert_eq!(flow.rgba[i + 1] as i32 - 128, -3);
        assert!(flow.confident_ratio >= TEMPORAL_MIN_CONFIDENT_RATIO);

        // A deterministic unrelated scene must fail the global confidence gate
        // and force a real ONNX run.
        let mut scene = vec![0u8; previous.len()];
        let mut value = 0x8765_4321u32;
        for pixel in scene.chunks_exact_mut(4) {
            value = value.wrapping_mul(22695477).wrapping_add(1);
            pixel[0] = (value >> 24) as u8;
            pixel[1] = (value >> 16) as u8;
            pixel[2] = (value >> 8) as u8;
            pixel[3] = 255;
        }
        assert!(state.temporal_flow_plan(width, height, &scene).is_none());
    }

    #[test]
    fn patch_round_trip() {
        let affected = Rect {
            x0: 1,
            y0: 1,
            x1: 3,
            y1: 3,
        };
        let mut full = vec![0u8; 8 * 8 * 4];
        let patch = vec![7u8; 4 * 4 * 4];
        write_output_patch(&mut full, 8, 2, affected, &patch).unwrap();
        let proof = compare_output_patch(&full, 8, 2, affected, &patch, 0, 0.0);
        assert!(proof.accepted);
    }
}
