//! NeoFlow Kari - experimental built-in frame interpolation.
//!
//! This is not an MVTools code port. It borrows the classic idea of block
//! motion estimation + motion-compensated placement, but keeps the
//! implementation original and self-contained. The goal is a second built-in
//! option with a different quality/speed tradeoff from the all-GPU NeoFlow:
//! heavier, x2-oriented, and allowed to spend several source-frame intervals
//! when quality matters.

use super::gl::{GlContext, GpuTex};
use anyhow::{Result, anyhow};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug)]
struct BlockMotion {
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    confidence: f32,
}

#[derive(Clone, Debug)]
struct MotionComponent {
    pixels: Vec<(i32, i32)>,
    cx: f32,
    cy: f32,
    mean: [f32; 3],
    area: usize,
}

fn pix(rgb: &[u8], w: usize, h: usize, x: i32, y: i32) -> [u8; 3] {
    let xx = x.clamp(0, w as i32 - 1) as usize;
    let yy = y.clamp(0, h as i32 - 1) as usize;
    let i = (yy * w + xx) * 3;
    [rgb[i], rgb[i + 1], rgb[i + 2]]
}

fn color_cost(a: [u8; 3], b: [u8; 3]) -> u32 {
    let dr = a[0].abs_diff(b[0]) as u32;
    let dg = a[1].abs_diff(b[1]) as u32;
    let db = a[2].abs_diff(b[2]) as u32;
    let ya = (77 * a[0] as u32 + 150 * a[1] as u32 + 29 * a[2] as u32) >> 8;
    let yb = (77 * b[0] as u32 + 150 * b[1] as u32 + 29 * b[2] as u32) >> 8;
    dr + dg + db + 2 * ya.abs_diff(yb)
}

fn patch_cost(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    x: i32,
    y: i32,
    dx: i32,
    dy: i32,
    half: i32,
) -> u32 {
    let mut sum = 0u32;
    for oy in -half..=half {
        for ox in -half..=half {
            sum += color_cost(
                pix(a, w, h, x + ox, y + oy),
                pix(b, w, h, x + dx + ox, y + dy + oy),
            );
        }
    }
    sum
}

fn patch_activity(a: &[u8], w: usize, h: usize, x: i32, y: i32, half: i32) -> u32 {
    let center = pix(a, w, h, x, y);
    let mut sum = 0u32;
    for oy in -half..=half {
        for ox in -half..=half {
            sum += color_cost(center, pix(a, w, h, x + ox, y + oy));
        }
    }
    sum
}

fn estimate_blocks(a: &[u8], b: &[u8], w: usize, h: usize) -> Vec<BlockMotion> {
    let max_dim = w.max(h);
    let step = if max_dim <= 720 {
        4
    } else if max_dim >= 1600 {
        12
    } else if max_dim >= 1000 {
        10
    } else {
        8
    };
    let radius = if max_dim >= 1600 { 56 } else { 80 };
    let patch_half = if max_dim <= 720 || max_dim >= 1600 {
        2
    } else {
        3
    };
    let coarse_step = 4;
    let margin = radius + patch_half + 1;
    let xs: Vec<i32> = (patch_half + 1..w as i32 - patch_half - 1)
        .step_by(step)
        .collect();
    let ys: Vec<i32> = (patch_half + 1..h as i32 - patch_half - 1)
        .step_by(step)
        .collect();
    let coords: Vec<(i32, i32)> = ys
        .iter()
        .flat_map(|&y| xs.iter().map(move |&x| (x, y)))
        .collect();

    coords
        .par_iter()
        .map(|&(x, y)| {
            let zero = patch_cost(a, b, w, h, x, y, 0, 0, patch_half);
            let activity = patch_activity(a, w, h, x, y, patch_half);
            let mut best = zero;
            let mut best_dx = 0;
            let mut best_dy = 0;

            let rx0 = -radius;
            let rx1 = radius;
            for dy in (rx0..=rx1).step_by(coarse_step as usize) {
                let yy = y + dy;
                if yy < patch_half || yy >= h as i32 - patch_half {
                    continue;
                }
                for dx in (rx0..=rx1).step_by(coarse_step as usize) {
                    let xx = x + dx;
                    if xx < patch_half || xx >= w as i32 - patch_half {
                        continue;
                    }
                    let motion_penalty = ((dx * dx + dy * dy) as u32) / 10;
                    let c = patch_cost(a, b, w, h, x, y, dx, dy, patch_half) + motion_penalty;
                    if c < best {
                        best = c;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            let refine_base = (best_dx, best_dy);
            for dy in refine_base.1 - 4..=refine_base.1 + 4 {
                let yy = y + dy;
                if yy < patch_half || yy >= h as i32 - patch_half {
                    continue;
                }
                for dx in refine_base.0 - 4..=refine_base.0 + 4 {
                    let xx = x + dx;
                    if xx < patch_half || xx >= w as i32 - patch_half {
                        continue;
                    }
                    let motion_penalty = ((dx * dx + dy * dy) as u32) / 12;
                    let c = patch_cost(a, b, w, h, x, y, dx, dy, patch_half) + motion_penalty;
                    if c < best {
                        best = c;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }
            }

            let motion = ((best_dx * best_dx + best_dy * best_dy) as f32).sqrt();
            let improvement = (zero.saturating_sub(best)) as f32 / (zero as f32 + 32.0);
            let textured = activity > (patch_half * 2 + 1).pow(2) as u32 * 24;
            if (!textured && improvement < 0.35) || improvement < 0.08 || motion < 1.25 {
                best_dx = 0;
                best_dy = 0;
            }
            let near_edge =
                x < margin || y < margin || x >= w as i32 - margin || y >= h as i32 - margin;
            let edge_scale = if near_edge { 0.75 } else { 1.0 };
            let confidence = (improvement * 3.2).clamp(0.0, 1.0) * edge_scale;
            BlockMotion {
                x,
                y,
                dx: best_dx,
                dy: best_dy,
                confidence,
            }
        })
        .collect()
}

fn add_sample(
    accum: &mut [[f32; 3]],
    weights: &mut [f32],
    w: usize,
    h: usize,
    x: f32,
    y: f32,
    color: [f32; 3],
    weight: f32,
) {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    for oy in 0..=1 {
        let py = y0 + oy;
        if py < 0 || py >= h as i32 {
            continue;
        }
        let wy = if oy == 0 { 1.0 - fy } else { fy };
        for ox in 0..=1 {
            let px = x0 + ox;
            if px < 0 || px >= w as i32 {
                continue;
            }
            let wx = if ox == 0 { 1.0 - fx } else { fx };
            let ww = weight * wx * wy;
            if ww <= 0.0 {
                continue;
            }
            let i = py as usize * w + px as usize;
            accum[i][0] += color[0] * ww;
            accum[i][1] += color[1] * ww;
            accum[i][2] += color[2] * ww;
            weights[i] += ww;
        }
    }
}

fn build_diff_mask(a: &[u8], b: &[u8], w: usize, h: usize) -> Vec<bool> {
    let mut mask = vec![false; w * h];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 3;
            let ca = [a[i], a[i + 1], a[i + 2]];
            let cb = [b[i], b[i + 1], b[i + 2]];
            mask[y * w + x] = color_cost(ca, cb) > 105;
        }
    }
    mask
}

fn components(frame: &[u8], mask: &[bool], w: usize, h: usize) -> Vec<MotionComponent> {
    let mut seen = vec![false; w * h];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for sy in 0..h {
        for sx in 0..w {
            let start = sy * w + sx;
            if seen[start] || !mask[start] {
                continue;
            }
            let seed_color = pix(frame, w, h, sx as i32, sy as i32);
            seen[start] = true;
            stack.clear();
            stack.push((sx as i32, sy as i32));
            let mut pixels = Vec::new();
            let mut sx_sum = 0.0f32;
            let mut sy_sum = 0.0f32;
            let mut rgb_sum = [0.0f32; 3];
            while let Some((x, y)) = stack.pop() {
                pixels.push((x, y));
                sx_sum += x as f32;
                sy_sum += y as f32;
                let pi = (y as usize * w + x as usize) * 3;
                rgb_sum[0] += frame[pi] as f32;
                rgb_sum[1] += frame[pi + 1] as f32;
                rgb_sum[2] += frame[pi + 2] as f32;
                for (nx, ny) in [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)] {
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let ni = ny as usize * w + nx as usize;
                    if !seen[ni] && mask[ni] {
                        let nc = pix(frame, w, h, nx, ny);
                        if color_cost(seed_color, nc) > 90 {
                            continue;
                        }
                        seen[ni] = true;
                        stack.push((nx, ny));
                    }
                }
            }
            let area = pixels.len();
            if area < 6 {
                continue;
            }
            let inv = 1.0 / area as f32;
            out.push(MotionComponent {
                pixels,
                cx: sx_sum * inv,
                cy: sy_sum * inv,
                mean: [rgb_sum[0] * inv, rgb_sum[1] * inv, rgb_sum[2] * inv],
                area,
            });
        }
    }
    out
}

fn mean_dist(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dr = a[0] - b[0];
    let dg = a[1] - b[1];
    let db = a[2] - b[2];
    (dr * dr + dg * dg + db * db).sqrt()
}

fn salient_component(c: &MotionComponent) -> bool {
    let max_c = c.mean[0].max(c.mean[1]).max(c.mean[2]);
    let min_c = c.mean[0].min(c.mean[1]).min(c.mean[2]);
    let luma = 0.299 * c.mean[0] + 0.587 * c.mean[1] + 0.114 * c.mean[2];
    luma > 72.0 || max_c - min_c > 42.0
}

fn component_assist(
    a: &[u8],
    b: &[u8],
    w: usize,
    h: usize,
    t: f32,
    accum: &mut [[f32; 3]],
    weights: &mut [f32],
    fallback_override: &mut [Option<[f32; 3]>],
) {
    let mask = build_diff_mask(a, b, w, h);
    let prev = components(a, &mask, w, h);
    let cur = components(b, &mask, w, h);
    for pc in prev.iter().filter(|c| c.area >= 8 && salient_component(c)) {
        let mut best: Option<(usize, f32)> = None;
        for (i, cc) in cur.iter().enumerate() {
            if cc.area < 8 || !salient_component(cc) {
                continue;
            }
            let area_ratio = pc.area as f32 / cc.area.max(1) as f32;
            if !(0.25..=4.0).contains(&area_ratio) {
                continue;
            }
            let cd = mean_dist(pc.mean, cc.mean);
            if cd > 34.0 {
                continue;
            }
            let dx = cc.cx - pc.cx;
            let dy = cc.cy - pc.cy;
            let motion = (dx * dx + dy * dy).sqrt();
            if motion < 1.5 {
                continue;
            }
            let score = cd * 3.8 + (area_ratio.ln().abs() * 28.0) + motion * 0.004;
            if best.is_none_or(|(_, s)| score < s) {
                best = Some((i, score));
            }
        }
        let Some((ci, _)) = best else {
            continue;
        };
        let cc = &cur[ci];
        let dx = cc.cx - pc.cx;
        let dy = cc.cy - pc.cy;
        let motion = (dx * dx + dy * dy).sqrt();
        let weight = (18.0 + (motion / 4.0).min(18.0)).clamp(18.0, 36.0);
        for &(x, y) in &pc.pixels {
            let bx = (x as f32 + dx).round() as i32;
            let by = (y as f32 + dy).round() as i32;
            if bx < 0 || by < 0 || bx >= w as i32 || by >= h as i32 {
                continue;
            }
            let ca = pix(a, w, h, x, y);
            let cb = pix(b, w, h, bx, by);
            if color_cost(ca, cb) > 120 {
                continue;
            }
            let src_i = y as usize * w + x as usize;
            let dst_i = by as usize * w + bx as usize;
            let src_bg = pix(b, w, h, x, y);
            let dst_bg = pix(a, w, h, bx, by);
            fallback_override[src_i] = Some([src_bg[0] as f32, src_bg[1] as f32, src_bg[2] as f32]);
            fallback_override[dst_i] = Some([dst_bg[0] as f32, dst_bg[1] as f32, dst_bg[2] as f32]);
            let color = [
                ca[0] as f32 * (1.0 - t) + cb[0] as f32 * t,
                ca[1] as f32 * (1.0 - t) + cb[1] as f32 * t,
                ca[2] as f32 * (1.0 - t) + cb[2] as f32 * t,
            ];
            add_sample(
                accum,
                weights,
                w,
                h,
                x as f32 + dx * t,
                y as f32 + dy * t,
                color,
                weight,
            );
        }
    }
}

fn interpolate_rgb(a: &[u8], b: &[u8], w: usize, h: usize, t: f32) -> Vec<u8> {
    let mut accum = vec![[0.0f32; 3]; w * h];
    let mut weights = vec![0.0f32; w * h];
    let mut fallback_override = vec![None; w * h];

    let max_dim = w.max(h);

    if max_dim > 720 {
        let blocks = estimate_blocks(a, b, w, h);
        let step = if max_dim >= 1600 {
            12
        } else if max_dim >= 1000 {
            10
        } else {
            8
        };
        let splat_half = (step / 2 + 2).max(5);
        for m in &blocks {
            let motion = ((m.dx * m.dx + m.dy * m.dy) as f32).sqrt();
            let moving = motion > 1.25 && m.confidence > 0.10;
            let base_weight = if moving {
                8.0 + m.confidence * 18.0 + (motion / 7.0).min(9.0)
            } else {
                0.12
            };
            for oy in -splat_half..=splat_half {
                for ox in -splat_half..=splat_half {
                    let ax = m.x + ox;
                    let ay = m.y + oy;
                    let bx = ax + m.dx;
                    let by = ay + m.dy;
                    if ax < 0
                        || ay < 0
                        || bx < 0
                        || by < 0
                        || ax >= w as i32
                        || ay >= h as i32
                        || bx >= w as i32
                        || by >= h as i32
                    {
                        continue;
                    }
                    let ca = pix(a, w, h, ax, ay);
                    let cb = pix(b, w, h, bx, by);
                    let color = [
                        ca[0] as f32 * (1.0 - t) + cb[0] as f32 * t,
                        ca[1] as f32 * (1.0 - t) + cb[1] as f32 * t,
                        ca[2] as f32 * (1.0 - t) + cb[2] as f32 * t,
                    ];
                    let rr = (ox * ox + oy * oy) as f32 / (splat_half * splat_half).max(1) as f32;
                    let spatial = (1.0 - rr * 0.55).clamp(0.15, 1.0);
                    let mx = ax as f32 + m.dx as f32 * t;
                    let my = ay as f32 + m.dy as f32 * t;
                    add_sample(
                        &mut accum,
                        &mut weights,
                        w,
                        h,
                        mx,
                        my,
                        color,
                        base_weight * spatial,
                    );
                }
            }
        }
    }

    component_assist(
        a,
        b,
        w,
        h,
        t,
        &mut accum,
        &mut weights,
        &mut fallback_override,
    );

    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let ai = i * 3;
            let ca = [a[ai], a[ai + 1], a[ai + 2]];
            let cb = [b[ai], b[ai + 1], b[ai + 2]];
            let blend = [
                a[ai] as f32 * (1.0 - t) + b[ai] as f32 * t,
                a[ai + 1] as f32 * (1.0 - t) + b[ai + 1] as f32 * t,
                a[ai + 2] as f32 * (1.0 - t) + b[ai + 2] as f32 * t,
            ];
            let same_place_change = color_cost(ca, cb);
            let fallback = if let Some(bg) = fallback_override[i] {
                bg
            } else if same_place_change > 55 {
                let la = 77 * ca[0] as u32 + 150 * ca[1] as u32 + 29 * ca[2] as u32;
                let lb = 77 * cb[0] as u32 + 150 * cb[1] as u32 + 29 * cb[2] as u32;
                if la <= lb {
                    [ca[0] as f32, ca[1] as f32, ca[2] as f32]
                } else {
                    [cb[0] as f32, cb[1] as f32, cb[2] as f32]
                }
            } else {
                blend
            };
            let mut c = fallback;
            if weights[i] > 0.001 {
                let inv = 1.0 / weights[i];
                let splat = [accum[i][0] * inv, accum[i][1] * inv, accum[i][2] * inv];
                let trust = (weights[i] / (weights[i] + 0.22)).clamp(0.0, 1.0);
                c = [
                    fallback[0] * (1.0 - trust) + splat[0] * trust,
                    fallback[1] * (1.0 - trust) + splat[1] * trust,
                    fallback[2] * (1.0 - trust) + splat[2] * trust,
                ];
            }
            out[ai] = c[0].round().clamp(0.0, 255.0) as u8;
            out[ai + 1] = c[1].round().clamp(0.0, 255.0) as u8;
            out[ai + 2] = c[2].round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Synthesize the frame at time `t` (0..1) between `prev` and `cur`.
pub fn interpolate(gc: &mut GlContext, prev: GpuTex, cur: GpuTex, t: f32) -> Result<GpuTex> {
    let (w, h) = (prev.w(), prev.h());
    if cur.w() != w || cur.h() != h {
        return Err(anyhow!("NeoFlowKari: frame size changed"));
    }
    let t = t.clamp(0.0, 1.0);
    let a = gc.download_rgb8(prev);
    let b = gc.download_rgb8(cur);
    let out = interpolate_rgb(&a, &b, w as usize, h as usize, t);
    Ok(gc.upload_rgb8(w, h, &out))
}
