//! Separable polyphase resampler (down/up) with selectable kernels:
//! Spline36 (default) / Lanczos3 / Bicubic (Catmull-Rom) / Bilinear / Nearest.
//! Two passes (H then V); kernel footprint scales with the ratio so
//! downscaling is properly antialiased (like mpv's dscale).

use super::gl::{Dtype, GlContext, GpuTex};
use anyhow::{Result, anyhow};
use glow::HasContext;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kernel {
    Spline36,
    Lanczos3,
    Bicubic,
    Bilinear,
    Nearest,
}

impl Kernel {
    pub fn from_name(s: &str) -> Self {
        match s.to_ascii_lowercase().as_str() {
            "lanczos" | "lanczos3" => Self::Lanczos3,
            "bicubic" | "catmullrom" => Self::Bicubic,
            "bilinear" => Self::Bilinear,
            "nearest" | "nearestneighbor" => Self::Nearest,
            _ => Self::Spline36,
        }
    }
    pub fn name(&self) -> &'static str {
        match self {
            Self::Spline36 => "spline36",
            Self::Lanczos3 => "lanczos3",
            Self::Bicubic => "bicubic",
            Self::Bilinear => "bilinear",
            Self::Nearest => "nearest",
        }
    }
    fn radius(&self) -> f32 {
        match self {
            Self::Spline36 | Self::Lanczos3 => 3.0,
            Self::Bicubic => 2.0,
            Self::Bilinear => 1.0,
            Self::Nearest => 0.5,
        }
    }
    fn glsl_fn(&self) -> &'static str {
        match self {
            Self::Spline36 => {
                r#"
float kern(float x){
  x = abs(x);
  if (x < 1.0) return ((13.0/11.0*x - 453.0/209.0)*x - 3.0/209.0)*x + 1.0;
  if (x < 2.0) { x -= 1.0; return ((-6.0/11.0*x + 270.0/209.0)*x - 156.0/209.0)*x; }
  if (x < 3.0) { x -= 2.0; return ((1.0/11.0*x - 45.0/209.0)*x + 26.0/209.0)*x; }
  return 0.0;
}"#
            }
            Self::Lanczos3 => {
                r#"
float sinc(float x){ if (abs(x) < 1e-5) return 1.0; float p = 3.14159265*x; return sin(p)/p; }
float kern(float x){ x = abs(x); return x < 3.0 ? sinc(x)*sinc(x/3.0) : 0.0; }"#
            }
            Self::Bicubic => {
                r#"
float kern(float x){ // Catmull-Rom
  x = abs(x);
  if (x < 1.0) return (1.5*x - 2.5)*x*x + 1.0;
  if (x < 2.0) return ((-0.5*x + 2.5)*x - 4.0)*x + 2.0;
  return 0.0;
}"#
            }
            Self::Bilinear => {
                r#"
float kern(float x){ x = abs(x); return x < 1.0 ? 1.0 - x : 0.0; }"#
            }
            Self::Nearest => "",
        }
    }
}

fn pass_frag(k: Kernel) -> String {
    format!(
        r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D tex;
uniform int axis;        // 0 = horizontal, 1 = vertical
uniform float src_len;   // source size along axis (pixels)
uniform float dst_len;   // destination size along axis (pixels)
uniform vec2 src_pt;     // 1/source size (both axes)
{kern}
void main() {{
  float dpos = (axis == 0 ? v_uv.x : v_uv.y) * dst_len;   // dest px + .5
  float scale = max(src_len / dst_len, 1.0);              // AA footprint
  float center = dpos * src_len / dst_len;                // in source px
  float r = {radius:.1} * scale;
  int lo = int(floor(center - r + 0.5));
  int hi = int(ceil(center + r - 0.5));
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  for (int i = lo; i <= hi && i - lo < 64; i++) {{
    float sp = float(i) + 0.5;
    float w = kern((center - sp) / scale);
    if (w == 0.0) continue;
    vec2 uv = axis == 0 ? vec2(sp * src_pt.x, v_uv.y) : vec2(v_uv.x, sp * src_pt.y);
    acc += texture(tex, uv) * w;
    wsum += w;
  }}
  frag = wsum > 0.0 ? acc / wsum : texture(tex, v_uv);
}}
"#,
        kern = k.glsl_fn(),
        radius = k.radius(),
    )
}

/// Safety fallback for an fp16 HDR frame that could not be normalized by the
/// engine's pre-chain CPU stage. It mirrors the 100-nit target normalization
/// and luminance/gamut shoulder used there instead of compressing every pixel
/// with Reinhard (which made mid-tones and colour look washed out).
pub fn tonemap_hdr(gc: &mut GlContext, src: GpuTex) -> Result<GpuTex> {
    const FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvec3 linear_to_srgb(vec3 c){\n  bvec3 low = lessThanEqual(c, vec3(0.0031308));\n  vec3 lo = 12.92 * c;\n  vec3 hi = 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055;\n  return mix(hi, lo, low);\n}\nvoid main(){\n  const float exposure = 80.0 / 100.0;\n  const float knee = 0.85;\n  vec3 c = texture(tex, v_uv).rgb * exposure;\n  float y = dot(c, vec3(0.2126, 0.7152, 0.0722));\n  if (y <= 1e-6) { frag = vec4(0.0, 0.0, 0.0, 1.0); return; }\n  if (y > knee) {\n    float span = 1.0 - knee;\n    float distance = y - knee;\n    float mapped = knee + span * distance / (distance + span);\n    c *= mapped / y;\n    y = mapped;\n  }\n  float low = min(c.r, min(c.g, c.b));\n  if (low < 0.0) {\n    float amount = clamp(y / (y - low), 0.0, 1.0);\n    c = vec3(y) + (c - vec3(y)) * amount;\n  }\n  float peak = max(c.r, max(c.g, c.b));\n  if (peak > 1.0) {\n    float amount = clamp((1.0 - y) / max(peak - y, 1e-6), 0.0, 1.0);\n    c = vec3(y) + (c - vec3(y)) * amount;\n  }\n  frag = vec4(linear_to_srgb(clamp(c, 0.0, 1.0)), 1.0);\n}\n";
    let prog = gc.program(FRAG).map_err(|e| anyhow!(e))?;
    Ok(run(gc, prog, src, src.w(), src.h(), 0, 0.0, 0.0))
}

/// Resample `src` to (dw, dh). Returns a pooled F16 texture.
pub fn resample(gc: &mut GlContext, src: GpuTex, dw: i32, dh: i32, k: Kernel) -> Result<GpuTex> {
    if src.w() == dw && src.h() == dh {
        return Ok(src);
    }
    if k == Kernel::Nearest {
        // plain NEAREST blit
        let frag = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = texture(tex, v_uv); }\n";
        let prog = gc.program(frag).map_err(|e| anyhow!(e))?;
        return Ok(run(gc, prog, src, dw, dh, 0, 0.0, 0.0));
    }
    let prog = gc.program(&pass_frag(k)).map_err(|e| anyhow!(e))?;
    // horizontal, then vertical
    let mid = run(gc, prog, src, dw, src.h(), 0, src.w() as f32, dw as f32);
    Ok(run(gc, prog, mid, dw, dh, 1, src.h() as f32, dh as f32))
}

fn run(
    gc: &mut GlContext,
    prog: glow::Program,
    src: GpuTex,
    ow: i32,
    oh: i32,
    axis: i32,
    src_len: f32,
    dst_len: f32,
) -> GpuTex {
    let tgt = gc.make_tex(ow.max(1), oh.max(1), 4, Dtype::F16);
    gc.bind_target(tgt);
    let gl = gc.gl.clone();
    unsafe {
        gl.use_program(Some(prog));
        gl.active_texture(glow::TEXTURE0);
        gl.bind_texture(glow::TEXTURE_2D, Some(src.tex));
        for (n, v) in [("axis", axis)] {
            if let Some(loc) = gl.get_uniform_location(prog, n) {
                gl.uniform_1_i32(Some(&loc), v);
            }
        }
        for (n, v) in [("src_len", src_len), ("dst_len", dst_len)] {
            if let Some(loc) = gl.get_uniform_location(prog, n) {
                gl.uniform_1_f32(Some(&loc), v);
            }
        }
        if let Some(loc) = gl.get_uniform_location(prog, "src_pt") {
            gl.uniform_2_f32(Some(&loc), 1.0 / src.w() as f32, 1.0 / src.h() as f32);
        }
        if let Some(loc) = gl.get_uniform_location(prog, "tex") {
            gl.uniform_1_i32(Some(&loc), 0);
        }
        gl.bind_vertex_array(Some(gc.quad_vao));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
    }
    gc.unbind_target();
    tgt
}
