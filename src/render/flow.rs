//! NeoFlow — real-time bidirectional block-pyramid interpolation.
//!
//! Motion-compensated interpolation structure:
//! independent forward/backward fields, coarse-to-fine search, spatial vector
//! coherence, forward/backward validation, and visibility-aware dual warping.
//! It is an original GPU implementation; no third-party implementation code is
//! embedded here.

use super::gl::{Dtype, GlContext, GpuTex};
use anyhow::{Result, anyhow};
use glow::HasContext;
use std::hash::{DefaultHasher, Hash, Hasher};

const NEOFLOW_CPU_CUT_PROBE: bool = false;

const LUMA_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D tex;
void main(){
  // Compare area content rather than one aliased point per block.
  vec2 p=1.0/vec2(textureSize(tex,0));
  vec3 c=vec3(0.0);
  c+=texture(tex,v_uv+vec2(-1.5,-1.5)*p).rgb;
  c+=texture(tex,v_uv+vec2( 1.5,-1.5)*p).rgb;
  c+=texture(tex,v_uv+vec2(-1.5, 1.5)*p).rgb;
  c+=texture(tex,v_uv+vec2( 1.5, 1.5)*p).rgb;
  c*=0.25;
  float y=dot(c,vec3(0.299,0.587,0.114));
  frag=vec4(y,c.r-c.b,c.g-0.5*(c.r+c.b),1.0);
}
"#;

// coarse search on the 1/16 pyramid: no init, wide range
const SEARCH16_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D srcL; uniform sampler2D dstL; uniform vec2 pt;
float sad(vec2 a, vec2 b){
  float s = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    s += abs(texture(srcL, a+o).r - texture(dstL, b+o).r);
  }
  return s;
}
void main(){
  vec2 best = vec2(0.0);
  float bc = sad(v_uv, v_uv);
  for (int dy=-6; dy<=6; dy+=2)
  for (int dx=-6; dx<=6; dx+=2){
    vec2 v = vec2(dx,dy);
    float c = sad(v_uv, v_uv + v*pt) + 0.12*length(v);
    if (c < bc){ bc = c; best = v; }
  }
  vec2 b2 = best; float c2 = bc;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 v = best + vec2(dx,dy);
    float c = sad(v_uv, v_uv + v*pt) + 0.12*length(v);
    if (c < c2){ c2 = c; b2 = v; }
  }
  frag = vec4(b2, c2, 1.0);
}
"#;

// refine at 1/8 around an upsampled init vector (init texels are 1/16-scale)
const REFINE8_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D srcL; uniform sampler2D dstL; uniform sampler2D init;
uniform vec2 pt; uniform vec2 ipt;
float sad(vec2 a, vec2 b){
  float s = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 d = abs(texture(srcL,a+o).rgb-texture(dstL,b+o).rgb);
    s += d.r + 0.28*(d.g+d.b);
  }
  return s/9.0;
}
float score(vec2 v, vec2 pred){
  vec2 q=v_uv+v*pt;
  if(any(lessThan(q,vec2(0.0)))||any(greaterThan(q,vec2(1.0)))) return 1000.0;
  return sad(v_uv,q)+0.030*length(v-pred)+0.025*length(v);
}
void main(){
  vec2 v0=texture(init,v_uv).xy*2.0;
  vec2 vl=texture(init,v_uv-vec2(ipt.x,0.0)).xy*2.0;
  vec2 vu=texture(init,v_uv-vec2(0.0,ipt.y)).xy*2.0;
  vec2 vr=texture(init,v_uv+vec2(ipt.x,0.0)).xy*2.0;
  vec2 vd=texture(init,v_uv+vec2(0.0,ipt.y)).xy*2.0;
  vec2 pred=(v0+vl+vu+vr+vd)/5.0;
  vec2 best=v0; float bc=score(best,pred);
  vec2 candidates[6]=vec2[6](v0,vl,vu,vr,vd,vec2(0.0));
  for(int k=0;k<6;k++){
    float c=score(candidates[k],pred);
    if(c<bc){bc=c;best=candidates[k];}
  }
  vec2 seed=best;
  for (int dy=-2; dy<=2; dy++)
  for (int dx=-2; dx<=2; dx++){
    vec2 v=seed+vec2(dx,dy);
    float c=score(v,pred);
    if (c < bc){ bc = c; best = v; }
  }
  seed=best;
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
    vec2 v=seed+vec2(dx,dy)*0.0625; float c=score(v,pred);
    if(c<bc){bc=c;best=v;}
  }
  frag = vec4(best, bc, 1.0);
}
"#;

// refine at 1/4 around an upsampled 1/8 vector
#[allow(dead_code)]
const REFINE4_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D srcL; uniform sampler2D dstL; uniform sampler2D init;
uniform vec2 pt; uniform vec2 ipt;
float sad(vec2 a, vec2 b){
  float s = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 d=abs(texture(srcL,a+o).rgb-texture(dstL,b+o).rgb);
    s += d.r+0.28*(d.g+d.b);
  }
  return s/9.0;
}
float score(vec2 v,vec2 pred){
  vec2 q=v_uv+v*pt;
  if(any(lessThan(q,vec2(0.0)))||any(greaterThan(q,vec2(1.0)))) return 1000.0;
  return sad(v_uv,q)+0.010*length(v-pred)+0.001*length(v);
}
void main(){
  vec2 v0=texture(init,v_uv).xy*2.0;
  vec2 vl=texture(init,v_uv-vec2(ipt.x,0.0)).xy*2.0;
  vec2 vu=texture(init,v_uv-vec2(0.0,ipt.y)).xy*2.0;
  vec2 vr=texture(init,v_uv+vec2(ipt.x,0.0)).xy*2.0;
  vec2 vd=texture(init,v_uv+vec2(0.0,ipt.y)).xy*2.0;
  vec2 pred=(v0+vl+vu+vr+vd)/5.0;
  vec2 best=v0; float bc=score(best,pred);
  vec2 candidates[6]=vec2[6](v0,vl,vu,vr,vd,vec2(0.0));
  for(int k=0;k<6;k++){
    float c=score(candidates[k],pred);
    if(c<bc){bc=c;best=candidates[k];}
  }
  vec2 seed=best;
  for (int dy=-2; dy<=2; dy++)
  for (int dx=-2; dx<=2; dx++){
    vec2 v=seed+vec2(dx,dy); float c=score(v,pred);
    if (c < bc){ bc = c; best = v; }
  }
  seed=best;
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
    vec2 v=seed+vec2(dx,dy)*0.5; float c=score(v,pred);
    if(c<bc){bc=c;best=v;}
  }
  frag = vec4(best, bc, 1.0);
}
"#;

// Select a spatially coherent candidate without averaging across boundaries.
// z is reliability; w is a compression/divergence occlusion risk.
const CONSIST_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D fw; uniform sampler2D bw; uniform sampler2D srcL; uniform sampler2D dstL; uniform vec2 pt;
float sad(vec2 v){
  vec2 q=v_uv+v*pt;
  if(any(lessThan(q,vec2(0.0)))||any(greaterThan(q,vec2(1.0)))) return 1000.0;
  float s=0.0;
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
    vec2 o=vec2(dx,dy)*pt;
    vec3 d=abs(texture(srcL,v_uv+o).rgb-texture(dstL,q+o).rgb);
    s+=d.r+0.28*(d.g+d.b);
  }
  return s/9.0;
}
void main(){
  vec2 center=texture(fw,v_uv).xy;
  vec2 mean=vec2(0.0);
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
    mean+=texture(fw,v_uv+vec2(dx,dy)*pt).xy;
  mean/=9.0;
  vec2 best=center; float bc=sad(center)+0.045*length(center-mean)+0.012*length(center);
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
    vec2 v=texture(fw,v_uv+vec2(dx,dy)*pt).xy;
    float c=sad(v)+0.045*length(v-mean)+0.012*length(v);
    if(c<bc){bc=c;best=v;}
  }
  vec2 vb=texture(bw,v_uv+best*pt).xy;
  float fb=length(best+vb)/(1.0+0.08*length(best));
  vec2 vx0=texture(fw,v_uv-vec2(pt.x,0.0)).xy;
  vec2 vx1=texture(fw,v_uv+vec2(pt.x,0.0)).xy;
  vec2 vy0=texture(fw,v_uv-vec2(0.0,pt.y)).xy;
  vec2 vy1=texture(fw,v_uv+vec2(0.0,pt.y)).xy;
  float div=abs((vx1.x-vx0.x)+(vy1.y-vy0.y))*0.5;
  float conf=(1.0-smoothstep(0.25,1.25,fb))*(1.0-smoothstep(0.075,0.30,bc));
  float occ=smoothstep(1.0,4.5,div);
  frag=vec4(best,clamp(conf,0.02,1.0),occ);
}
"#;

const SCENE_SCORE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D flow;
void main(){
  float bad=smoothstep(0.42,1.35,texture(flow,v_uv).z);
  frag=vec4(bad,bad,bad,1.0);
}
"#;

const REDUCE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D tex; uniform vec2 pt;
void main(){
  float v=texture(tex,v_uv+vec2(-0.5,-0.5)*pt).r
         +texture(tex,v_uv+vec2( 0.5,-0.5)*pt).r
         +texture(tex,v_uv+vec2(-0.5, 0.5)*pt).r
         +texture(tex,v_uv+vec2( 0.5, 0.5)*pt).r;
  v*=0.25; frag=vec4(v,v,v,1.0);
}
"#;

#[allow(dead_code)]
const HISTORY_ACTIVITY_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prev2C; uniform sampler2D prevC; uniform sampler2D curC;
uniform vec2 px;
float diff9(sampler2D a,sampler2D b){
  float s=0.0;
  for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
    vec2 o=vec2(dx,dy)*px*4.0;
    s+=dot(abs(texture(a,v_uv+o).rgb-texture(b,v_uv+o).rgb),vec3(0.333333));
  }
  return s/9.0;
}
void main(){
  frag=vec4(diff9(prev2C,prevC),diff9(prevC,curC),diff9(prev2C,curC),1.0);
}
"#;

#[allow(dead_code)]
const HISTORY_GLOBAL_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D activity;
void main(){
  float meanPast=0.0,meanNow=0.0,meanReturn=0.0;
  float loNow=1.0,changed=0.0;
  for(int y=0;y<9;y++) for(int x=0;x<16;x++){
    vec3 a=texture(activity,(vec2(x,y)+0.5)/vec2(16.0,9.0)).rgb;
    meanPast+=a.r; meanNow+=a.g; meanReturn+=a.b;
    loNow=min(loNow,a.g); changed+=step(0.075,a.g);
  }
  meanPast/=144.0; meanNow/=144.0; meanReturn/=144.0;
  float ratio=changed/144.0;
  float hardCut=smoothstep(0.11,0.22,loNow)*smoothstep(0.16,0.30,meanNow);
  float sudden=smoothstep(0.045,0.12,meanNow)
              *(1.0-smoothstep(0.025,0.085,meanPast));
  float reversal=smoothstep(0.045,0.12,min(meanPast,meanNow))
                *(1.0-smoothstep(0.025,0.080,meanReturn));
  float broad=smoothstep(0.08,0.22,ratio)*smoothstep(0.015,0.060,meanNow);
  frag=vec4(hardCut,max(max(sudden,reversal),broad),meanNow,ratio);
}
"#;

#[allow(dead_code)]
const GLOBAL_MOTION_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D fw; uniform sampler2D activity;
void main(){
  vec2 meanV=vec2(0.0); float weight=0.0,confidence=0.0;
  for(int y=0;y<9;y++) for(int x=0;x<16;x++){
    vec2 p=(vec2(x,y)+0.5)/vec2(16.0,9.0);
    float a=smoothstep(0.025,0.12,texture(activity,p).g);
    vec4 m=texture(fw,p);
    meanV+=m.xy*a; confidence+=m.z*a; weight+=a;
  }
  meanV/=max(weight,0.001); confidence/=max(weight,0.001);
  float variance=0.0;
  for(int y=0;y<9;y++) for(int x=0;x<16;x++){
    vec2 p=(vec2(x,y)+0.5)/vec2(16.0,9.0);
    float a=smoothstep(0.025,0.12,texture(activity,p).g);
    vec2 d=texture(fw,p).xy-meanV;
    variance+=dot(d,d)*a;
  }
  variance/=max(weight,0.001);
  float coherent=1.0-smoothstep(0.55,2.0,sqrt(variance));
  float reliable=smoothstep(0.22,0.62,confidence)*coherent;
  frag=vec4(reliable,confidence,sqrt(variance),1.0);
}
"#;

#[allow(dead_code)]
const ZERO_GLOBAL_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
void main(){ frag=vec4(0.0); }
"#;

#[allow(dead_code)]
const LEGACY_COMPOSITE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevC; uniform sampler2D curC; uniform sampler2D mot;
uniform float t; uniform vec2 px;

// NEOFLOW_TEXTFX_GUARD_BEGIN
// Protect transparent text, logos, anime OP text effects and glowing thin lines.
// Screen-space high-frequency regions that also move/flicker or sit on low flow
// confidence are exactly where 2-frame optical-flow interpolation tears them
// apart. There we pull flow confidence DOWN and prefer a temporal blend over a
// hard nearest-frame pick, so those regions degrade to a soft cross-fade instead
// of grinding/splitting. Normal footage (no persistent high-freq edges) is
// untouched. No new history, no extra passes, no UI, motion search unchanged.
float luma_nf(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float edge_luma_nf(sampler2D tex, vec2 uv, vec2 p){
  float l = luma_nf(texture(tex, uv - vec2(p.x, 0.0)).rgb);
  float r = luma_nf(texture(tex, uv + vec2(p.x, 0.0)).rgb);
  float u = luma_nf(texture(tex, uv - vec2(0.0, p.y)).rgb);
  float d = luma_nf(texture(tex, uv + vec2(0.0, p.y)).rgb);
  return max(abs(r - l), abs(d - u));
}
// NEOFLOW_TEXTFX_GUARD_END

void main(){
  vec4 m = texture(mot, v_uv);
  vec2 v = m.xy * 4.0 * px;              // 1/4 texels -> full-res uv offset
  vec3 p0 = texture(prevC, v_uv - v * t).rgb;
  vec3 p1 = texture(curC,  v_uv + v * (1.0 - t)).rgb;
  float err = dot(abs(p0 - p1), vec3(1.0/3.0));
  float conf = m.z;
  conf *= 1.0 - smoothstep(0.085, 0.34, err);        // warped frames agree?

  // NEOFLOW_TEXTFX_GUARD_BEGIN
  vec3 raw0 = texture(prevC, v_uv).rgb;
  vec3 raw1 = texture(curC,  v_uv).rgb;
  // pull confidence toward the 3x3 minimum so a thin glyph is not half
  // interpolated / half fallback, which can create partial-interpolation flicker. mot is the small
  // 1/8-res motion field, so the 9 taps are cheap.
  float confMin = conf;
  for (int dy = -1; dy <= 1; dy++)
  for (int dx = -1; dx <= 1; dx++)
    confMin = min(confMin, texture(mot, v_uv + vec2(dx, dy) * px * 4.0).z);
  conf = mix(conf, confMin, 0.12);

  float e0 = edge_luma_nf(prevC, v_uv, px);
  float e1 = edge_luma_nf(curC,  v_uv, px);
  float edge          = max(e0, e1);   // any contour
  float persistentEdge = min(e0, e1);  // same-place edge = logo/subtitle/UI
  float frameDiff     = length(raw0 - raw1);

  float edgeRisk    = smoothstep(0.05, 0.16, edge);
  float persistRisk = smoothstep(0.035, 0.12, persistentEdge);
  float diffRisk    = smoothstep(0.08, 0.25, frameDiff);
  float lowConfRisk = 1.0 - conf;

  float textFxRisk = 0.0;
  textFxRisk = max(textFxRisk, edgeRisk * diffRisk);   // OP text / glow / telop
  textFxRisk = max(textFxRisk, persistRisk * 0.75);    // transparent logo / subs
  textFxRisk = max(textFxRisk, edgeRisk * lowConfRisk);// misflowed glyph outline
  textFxRisk = clamp(textFxRisk, 0.0, 1.0);

  conf *= 1.0 - textFxRisk * 0.45;                     // trust flow less there
  vec3 nearestFallback = (t <= 0.5) ? raw0 : raw1;
  vec3 blendFallback   = mix(raw0, raw1, t);
  vec3 fallback = mix(nearestFallback, blendFallback, textFxRisk * 0.45);
  // NEOFLOW_TEXTFX_GUARD_END

  vec3 flow_mix = mix(p0, p1, t);
  frag = vec4(mix(fallback, flow_mix, conf), 1.0);
}
"#;

const COMPOSITE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevC; uniform sampler2D curC;
uniform sampler2D fw; uniform sampler2D bw; uniform sampler2D scene;
uniform float t; uniform vec2 px;
void main(){
  vec4 mf=texture(fw,v_uv);
  vec4 mb=texture(bw,v_uv);
  vec2 q=v_uv-mf.xy*8.0*px*t;
  vec2 r=v_uv-mb.xy*8.0*px*(1.0-t);
  vec3 a=texture(prevC,q).rgb;
  vec3 b=texture(curC,r).rgb;
  vec3 raw0=texture(prevC,v_uv).rgb;
  vec3 raw1=texture(curC,v_uv).rgb;
  // Basic FlowFPS visibility masks: unreliable or compressed blocks hand
  // ownership to the opposite compensated endpoint, not to a held frame.
  float va=max(0.04,mf.z)*(1.0-0.82*mf.w);
  float vb=max(0.04,mb.z)*(1.0-0.82*mb.w);
  float wa=(1.0-t)*va;
  float wb=t*vb;
  vec3 motion=(a*wa+b*wb)/max(wa+wb,1e-5);
  vec3 temporal=mix(raw0,raw1,t);
  // The reduction stores the fraction of unmatched blocks. Treat only a
  // near-global mismatch as a cut; ordinary pans and busy scenes must remain
  // motion compensated (the former threshold classified them as blends).
  float cut=smoothstep(0.82,0.94,texture(scene,vec2(0.5)).r);
  frag=vec4(mix(motion,temporal,cut),1.0);
}
"#;

const COPY_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = texture(tex, v_uv); }\n";

pub(crate) fn run_pass(
    gc: &mut GlContext,
    prog: glow::Program,
    binds: &[(&str, GpuTex)],
    floats: &[(&str, f32)],
    vec2s: &[(&str, (f32, f32))],
    ow: i32,
    oh: i32,
    comps: u8,
) -> GpuTex {
    let tgt = gc.make_tex(ow.max(1), oh.max(1), comps, Dtype::F16);
    gc.bind_target(tgt);
    let gl = gc.gl.clone();
    unsafe {
        gl.use_program(Some(prog));
        for (unit, (name, tex)) in binds.iter().enumerate() {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex.tex));
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_1_i32(Some(&loc), unit as i32);
            }
        }
        for (name, v) in floats {
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_1_f32(Some(&loc), *v);
            }
        }
        for (name, (x, y)) in vec2s {
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_2_f32(Some(&loc), *x, *y);
            }
        }
        gl.bind_vertex_array(Some(gc.quad_vao));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
    }
    gc.unbind_target();
    tgt
}

/// mean of one channel of a small texture via CPU readback (cut probe).
fn mean_ch(gc: &mut GlContext, t: GpuTex, ch: usize) -> f32 {
    let v = gc.download_f32(t);
    let mut sum = 0.0f64;
    let n = (t.w() * t.h()) as usize;
    for i in 0..n {
        sum += v[i * 4 + ch] as f64;
    }
    (sum / n.max(1) as f64) as f32
}

/// Synthesize the frame at time `t` (0..1) between `prev` and `cur`.
pub fn interpolate(gc: &mut GlContext, prev: GpuTex, cur: GpuTex, t: f32) -> Result<GpuTex> {
    let mut frames = interpolate_many(gc, prev, cur, &[t])?;
    frames
        .pop()
        .ok_or_else(|| anyhow!("NeoFlow: no interpolation phase"))
}

pub fn interpolate3(
    gc: &mut GlContext,
    _prev2: Option<GpuTex>,
    prev: GpuTex,
    cur: GpuTex,
    t: f32,
) -> Result<GpuTex> {
    interpolate(gc, prev, cur, t)
}

/// Synthesize several phases for one captured pair. Motion analysis is shared,
/// so x3 performs one vector search plus two inexpensive composites instead of
/// repeating the complete forward/backward search twice.
pub fn interpolate_many(
    gc: &mut GlContext,
    prev: GpuTex,
    cur: GpuTex,
    phases: &[f32],
) -> Result<Vec<GpuTex>> {
    interpolate_many_with_source(gc, prev, cur, phases, None)
}

/// Run a bridge-format NeoFlow GLSL. The same source is compiled once per
/// `NEOFLOW_PASS_*` macro, matching the contract documented in the file.
pub fn interpolate_many_external(
    gc: &mut GlContext,
    prev: GpuTex,
    cur: GpuTex,
    phases: &[f32],
    source: &str,
) -> Result<Vec<GpuTex>> {
    if (source.contains("NeoFlow GameDIS")
        || source.contains("NeoFlow GameMesh")
        || source.contains("NeoFlow HybridCadence")
        || source.contains("NeoFlow CausalStable"))
        && source.contains("NF_PASS_FINAL")
    {
        return interpolate_many_game_dis(gc, prev, cur, phases, source);
    }
    if source.contains("NEOFLOW_PASS_OWNER_CLEAR") && source.contains("NEOFLOW_PASS_FINAL") {
        return interpolate_many_game_reconstruct(gc, prev, cur, phases, source);
    }
    interpolate_many_with_source(gc, prev, cur, phases, Some(source))
}

fn external_nf_pass_source(source: &str, pass: &str) -> Result<String> {
    let newline = source
        .find('\n')
        .ok_or_else(|| anyhow!("external NeoFlow source has no version line"))?;
    anyhow::ensure!(
        source[..newline].trim().starts_with("#version"),
        "external NeoFlow source must start with #version"
    );
    Ok(format!(
        "{}\n#define NF_PASS_{pass}\n{}",
        &source[..newline],
        &source[newline + 1..]
    ))
}

fn run_pass_ints(
    gc: &mut GlContext,
    prog: glow::Program,
    binds: &[(&str, GpuTex)],
    floats: &[(&str, f32)],
    ints: &[(&str, i32)],
    ow: i32,
    oh: i32,
    comps: u8,
) -> GpuTex {
    let target = gc.make_tex(ow.max(1), oh.max(1), comps, Dtype::F16);
    gc.bind_target(target);
    let gl = gc.gl.clone();
    unsafe {
        gl.use_program(Some(prog));
        for (unit, (name, tex)) in binds.iter().enumerate() {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex.tex));
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_1_i32(Some(&loc), unit as i32);
            }
        }
        for (name, value) in floats {
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_1_f32(Some(&loc), *value);
            }
        }
        for (name, value) in ints {
            if let Some(loc) = gl.get_uniform_location(prog, name) {
                gl.uniform_1_i32(Some(&loc), *value);
            }
        }
        gl.bind_vertex_array(Some(gc.quad_vao));
        gl.draw_arrays(glow::TRIANGLE_STRIP, 0, 4);
        gl.bind_vertex_array(None);
    }
    gc.unbind_target();
    target
}

fn interpolate_many_game_dis(
    gc: &mut GlContext,
    prev: GpuTex,
    cur: GpuTex,
    phases: &[f32],
    source: &str,
) -> Result<Vec<GpuTex>> {
    let is_causal = source.contains("NeoFlow CausalStable");
    let mut causal_state = if is_causal {
        gc.take_external_neoflow_state()
    } else {
        Default::default()
    };
    let mut hasher = DefaultHasher::new();
    source.hash(&mut hasher);
    let fingerprint = hasher.finish();
    anyhow::ensure!(
        cur.w() == prev.w() && cur.h() == prev.h(),
        "GameDIS: frame size changed"
    );
    // GameDIS/GameMesh only synthesize one midpoint. The display cadence may
    // request a slightly shifted phase while it settles (or after a WGC
    // arrival wobble); rejecting that request disabled interpolation for all
    // following frames. Keep the filter's x2 contract and render its certified
    // t=0.5 midpoint for the single generated slot.
    anyhow::ensure!(
        phases.len() == 1,
        "NeoFlow GameDIS/GameMesh supports x2 only (requested phases: {phases:?})"
    );
    let names = [
        "FEATURE",
        "FLOW_INIT",
        "FLOW_REFINE",
        "FLOW_REGULARIZE",
        "FLOW_VARIATIONAL",
        "CONFIDENCE",
        "SCENE_SCORE",
        "REDUCE",
        "MIDMAP_INIT",
        "MIDMAP_ITER",
        "WARP",
        "VISIBILITY",
        "VIS_SMOOTH",
        "UI_MASK",
        "MASK_DILATE",
        "FINAL",
    ];
    let mut p = Vec::with_capacity(names.len());
    for name in names {
        p.push(
            gc.program(&external_nf_pass_source(source, name)?)
                .map_err(|e| anyhow!("GameDIS {name}: {e}"))?,
        );
    }
    let (w, h) = (prev.w(), prev.h());
    if is_causal
        && (causal_state.shader_fingerprint != fingerprint || causal_state.size != Some((w, h)))
    {
        for texture in [
            causal_state.mode_state,
            causal_state.previous_stable_forward,
            causal_state.previous_backward,
            causal_state.previous_stable_risk,
        ]
        .into_iter()
        .flatten()
        {
            gc.recycle(texture);
        }
        causal_state = Default::default();
        causal_state.shader_fingerprint = fingerprint;
        causal_state.size = Some((w, h));
    }
    let (w2, h2) = ((w + 1) / 2, (h + 1) / 2);
    let (w4, h4) = ((w + 3) / 4, (h + 3) / 4);
    let (w8, h8) = ((w + 7) / 8, (h + 7) / 8);
    gc.set_filter_linear(prev, true);
    gc.set_filter_linear(cur, true);
    let features = |gc: &mut GlContext, color: GpuTex| {
        let full = run_pass_ints(
            gc,
            p[0],
            &[("tex", color)],
            &[],
            &[("u_downsample", 0)],
            w,
            h,
            4,
        );
        let f2 = run_pass_ints(
            gc,
            p[0],
            &[("tex", color)],
            &[],
            &[("u_downsample", 1)],
            w2,
            h2,
            4,
        );
        gc.set_filter_linear(f2, true);
        let f4 = run_pass_ints(
            gc,
            p[0],
            &[("tex", f2)],
            &[],
            &[("u_downsample", 1)],
            w4,
            h4,
            4,
        );
        gc.set_filter_linear(f4, true);
        let f8 = run_pass_ints(
            gc,
            p[0],
            &[("tex", f4)],
            &[],
            &[("u_downsample", 1)],
            w8,
            h8,
            4,
        );
        for tex in [full, f8] {
            gc.set_filter_linear(tex, true);
        }
        [full, f2, f4, f8]
    };
    let pf = features(gc, prev);
    let cf = features(gc, cur);
    let mut hybrid_anime = 0.0f32;
    let mut causal_mode_tex = None;
    let mut causal_pan_tex = None;
    let mut causal_lines = None;
    let mut causal_class = [2.0f32, 0.0, 2.0, 0.0];
    if source.contains("NeoFlow HybridCadence") || is_causal {
        let pan_pass = if is_causal {
            "PAN_WARP_CAUSAL"
        } else {
            "PAN_WARP"
        };
        let hp = [
            "LINE_TOPOLOGY",
            "CADENCE_SCORE",
            "CADENCE_AUX",
            "GLOBAL_PAN_SEARCH",
            "CADENCE_CLASSIFY",
            pan_pass,
            "COPY",
        ];
        let mut programs = Vec::with_capacity(hp.len());
        for name in hp {
            programs.push(
                gc.program(&external_nf_pass_source(source, name)?)
                    .map_err(|e| anyhow!("HybridCadence {name}: {e}"))?,
            );
        }
        let line_a = run_pass(gc, programs[0], &[("colorTex", prev)], &[], &[], w, h, 4);
        let line_b = run_pass(gc, programs[0], &[("colorTex", cur)], &[], &[], w, h, 4);
        if is_causal {
            causal_lines = Some((line_a, line_b));
        }
        let score = run_pass(
            gc,
            programs[1],
            &[
                ("prevColor", prev),
                ("curColor", cur),
                ("prevLine", line_a),
                ("curLine", line_b),
            ],
            &[],
            &[],
            w,
            h,
            4,
        );
        let aux = run_pass(
            gc,
            programs[2],
            &[
                ("prevColor", prev),
                ("curColor", cur),
                ("prevLine", line_a),
                ("curLine", line_b),
            ],
            &[],
            &[],
            w,
            h,
            4,
        );
        let reduce_1x1 = |gc: &mut GlContext, mut tex: GpuTex| {
            let (mut rw, mut rh) = (w, h);
            while rw > 1 || rh > 1 {
                rw = (rw + 1) / 2;
                rh = (rh + 1) / 2;
                tex = run_pass(gc, p[7], &[("tex", tex)], &[], &[], rw, rh, 4);
            }
            tex
        };
        let score1 = reduce_1x1(gc, score);
        let aux1 = reduce_1x1(gc, aux);
        let pan = run_pass(
            gc,
            programs[3],
            &[("prevF16", pf[3]), ("curF16", cf[3])],
            &[],
            &[],
            1,
            1,
            4,
        );
        let class_tex = run_pass(
            gc,
            programs[4],
            &[
                ("cadenceScore1x1", score1),
                ("cadenceAux1x1", aux1),
                ("panResult1x1", pan),
            ],
            &[],
            &[("u_frameSize", (w as f32, h as f32))],
            1,
            1,
            4,
        );
        let class_tex = if is_causal {
            let stabilize = gc
                .program(&external_nf_pass_source(source, "MODE_STABILIZE")?)
                .map_err(|e| anyhow!("CausalStable MODE_STABILIZE: {e}"))?;
            let previous = causal_state.mode_state.unwrap_or(class_tex);
            let stable = run_pass(
                gc,
                stabilize,
                &[
                    ("currentClass1x1", class_tex),
                    ("previousState1x1", previous),
                ],
                &[(
                    "u_hasHistory",
                    if causal_state.mode_history_valid {
                        1.0
                    } else {
                        0.0
                    },
                )],
                &[],
                1,
                1,
                4,
            );
            if let Some(old) = causal_state.mode_state.replace(stable) {
                gc.recycle(old);
            }
            causal_state.mode_history_valid = true;
            causal_mode_tex = Some(stable);
            stable
        } else {
            class_tex
        };
        gc.finish();
        let class = gc.download_f32(class_tex);
        if is_causal {
            for (dst, src) in causal_class.iter_mut().zip(class.iter().copied()) {
                *dst = src;
            }
        }
        let class_id = class.first().copied().unwrap_or(2.0).round() as i32;
        hybrid_anime = class.get(1).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        if is_causal {
            causal_pan_tex = Some(run_pass(
                gc,
                programs[5],
                &[("prevColor", prev), ("panResult1x1", pan)],
                &[("u_t", 0.5), ("u_animeWeight", hybrid_anime)],
                &[],
                w,
                h,
                4,
            ));
        }
        if !is_causal && (class_id == 0 || class_id == 4) {
            let endpoint = if class_id == 0 { prev } else { cur };
            return Ok(vec![run_pass(
                gc,
                programs[6],
                &[("tex", endpoint)],
                &[],
                &[],
                w,
                h,
                4,
            )]);
        }
        if !is_causal && class_id == 1 {
            let binds = if is_causal {
                vec![("prevColor", prev), ("panResult1x1", pan)]
            } else {
                vec![
                    ("prevColor", prev),
                    ("curColor", cur),
                    ("panResult1x1", pan),
                ]
            };
            return Ok(vec![run_pass(
                gc,
                programs[5],
                &binds,
                &[("u_t", 0.5), ("u_animeWeight", hybrid_anime)],
                &[],
                w,
                h,
                4,
            )]);
        }
    }
    let causal_line_programs = if is_causal {
        Some((
            gc.program(&external_nf_pass_source(source, "LINE_FLOW_REFINE")?)
                .map_err(|e| anyhow!(e))?,
            gc.program(&external_nf_pass_source(source, "FLAT_FLOW_PROPAGATE")?)
                .map_err(|e| anyhow!(e))?,
        ))
    } else {
        None
    };
    let build_flow = |gc: &mut GlContext,
                      src: &[GpuTex; 4],
                      dst: &[GpuTex; 4],
                      src_c: GpuTex,
                      dst_c: GpuTex,
                      src_line: Option<GpuTex>,
                      dst_line: Option<GpuTex>| {
        let f8 = run_pass(
            gc,
            p[1],
            &[("srcFeat", src[3]), ("dstFeat", dst[3])],
            &[],
            &[],
            w8,
            h8,
            4,
        );
        gc.set_filter_linear(f8, true);
        let mut f4 = run_pass(
            gc,
            p[2],
            &[("srcFeat", src[2]), ("dstFeat", dst[2]), ("initFlow", f8)],
            &[("u_searchStepPixels", 2.0), ("u_smoothPenalty", 0.0035)],
            &[],
            w4,
            h4,
            4,
        );
        gc.set_filter_linear(f4, true);
        f4 = run_pass(
            gc,
            p[2],
            &[("srcFeat", src[2]), ("dstFeat", dst[2]), ("initFlow", f4)],
            &[("u_searchStepPixels", 1.0), ("u_smoothPenalty", 0.0035)],
            &[],
            w4,
            h4,
            4,
        );
        gc.set_filter_linear(f4, true);
        f4 = run_pass(
            gc,
            p[3],
            &[("flowTex", f4), ("guideFeat", src[2])],
            &[("u_strength", 0.55)],
            &[],
            w4,
            h4,
            4,
        );
        gc.set_filter_linear(f4, true);
        let settings = [(2.0, 0.0030), (1.0, 0.0030), (0.5, 0.0025)];
        let mut f2 = f4;
        for (step, penalty) in settings {
            f2 = run_pass(
                gc,
                p[2],
                &[("srcFeat", src[1]), ("dstFeat", dst[1]), ("initFlow", f2)],
                &[("u_searchStepPixels", step), ("u_smoothPenalty", penalty)],
                &[],
                w2,
                h2,
                4,
            );
            gc.set_filter_linear(f2, true);
        }
        f2 = run_pass(
            gc,
            p[3],
            &[("flowTex", f2), ("guideFeat", src[1])],
            &[("u_strength", 0.48)],
            &[],
            w2,
            h2,
            4,
        );
        gc.set_filter_linear(f2, true);
        if let (Some((line_refine, flat_propagate)), Some(src_line), Some(dst_line)) =
            (causal_line_programs, src_line, dst_line)
        {
            f2 = run_pass(
                gc,
                line_refine,
                &[
                    ("flowTex", f2),
                    ("srcColor", src_c),
                    ("dstColor", dst_c),
                    ("srcLine", src_line),
                    ("dstLine", dst_line),
                ],
                &[
                    ("u_lineWeight", 0.10 + 0.26 * hybrid_anime),
                    ("u_stepPixels", 1.0),
                ],
                &[],
                w2,
                h2,
                4,
            );
            for _ in 0..2 {
                f2 = run_pass(
                    gc,
                    flat_propagate,
                    &[("flowTex", f2), ("lineTex", src_line)],
                    &[("u_strength", 0.44)],
                    &[],
                    w2,
                    h2,
                    4,
                );
            }
            gc.set_filter_linear(f2, true);
        }
        let mut full = f2;
        for _ in 0..5 {
            full = run_pass(
                gc,
                p[4],
                &[("srcColor", src_c), ("dstColor", dst_c), ("flowTex", full)],
                &[
                    ("u_dataGain", 0.72),
                    ("u_smoothGain", 0.24),
                    ("u_maxCorrectionPixels", 0.70),
                ],
                &[],
                w,
                h,
                4,
            );
            gc.set_filter_linear(full, true);
        }
        full
    };
    let (line_a, line_b) = causal_lines.unwrap_or((prev, cur));
    let line_pair = if is_causal {
        (Some(line_a), Some(line_b))
    } else {
        (None, None)
    };
    let mut raw_fw = build_flow(gc, &pf, &cf, prev, cur, line_pair.0, line_pair.1);
    let raw_bw = build_flow(gc, &cf, &pf, cur, prev, line_pair.1, line_pair.0);
    if is_causal {
        let temporal = gc
            .program(&external_nf_pass_source(source, "FLOW_TEMPORAL_STABILIZE")?)
            .map_err(|e| anyhow!("CausalStable FLOW_TEMPORAL_STABILIZE: {e}"))?;
        let previous_forward = causal_state.previous_stable_forward.unwrap_or(raw_fw);
        let previous_backward = causal_state.previous_backward.unwrap_or(raw_bw);
        raw_fw = run_pass(
            gc,
            temporal,
            &[
                ("currentFlow", raw_fw),
                ("previousStableFlow", previous_forward),
                ("previousBackwardFlow", previous_backward),
            ],
            &[
                (
                    "u_hasHistory",
                    if causal_state.flow_history_valid {
                        1.0
                    } else {
                        0.0
                    },
                ),
                ("u_strength", 0.24),
            ],
            &[("u_frameSize", (w as f32, h as f32))],
            w,
            h,
            4,
        );
        gc.set_filter_linear(raw_fw, true);
    }
    let is_hybrid = source.contains("NeoFlow HybridCadence") || is_causal;
    let is_mesh = source.contains("NeoFlow GameMesh") || is_hybrid;
    let (selected_fw, selected_bw, safe_fw, safe_bw, stable_risk) = if is_mesh {
        let extra_names = [
            "MESH_REDUCE",
            "MESH_MEDIAN",
            "MESH_SMOOTH",
            "FLOW_RISK",
            "MASK_MAX",
            "MASK_MIN",
            "MASK_BLUR",
            "FLOW_COMBINE",
        ];
        let mut ep = Vec::with_capacity(extra_names.len());
        for name in extra_names {
            ep.push(
                gc.program(&external_nf_pass_source(source, name)?)
                    .map_err(|e| anyhow!("GameMesh {name}: {e}"))?,
            );
        }
        let hysteresis = if is_causal {
            Some(
                gc.program(&external_nf_pass_source(source, "TEMPORAL_HYSTERESIS")?)
                    .map_err(|e| anyhow!("CausalStable TEMPORAL_HYSTERESIS: {e}"))?,
            )
        } else {
            None
        };
        let mesh_for = |gc: &mut GlContext, raw: GpuTex, guide: GpuTex| {
            let half = run_pass(gc, ep[0], &[("flowTex", raw)], &[], &[], w2, h2, 4);
            gc.set_filter_linear(half, true);
            let quarter = run_pass(gc, ep[0], &[("flowTex", half)], &[], &[], w4, h4, 4);
            gc.set_filter_linear(quarter, true);
            let eighth = run_pass(gc, ep[0], &[("flowTex", quarter)], &[], &[], w8, h8, 4);
            gc.set_filter_linear(eighth, true);
            let mut mesh = run_pass(gc, ep[1], &[("flowTex", eighth)], &[], &[], w8, h8, 4);
            gc.set_filter_linear(mesh, true);
            for _ in 0..3 {
                mesh = run_pass(
                    gc,
                    ep[2],
                    &[("flowTex", mesh), ("guideFeat", guide)],
                    &[("u_strength", 0.55)],
                    &[],
                    w8,
                    h8,
                    4,
                );
                gc.set_filter_linear(mesh, true);
            }
            mesh
        };
        let mesh_fw = mesh_for(gc, raw_fw, pf[3]);
        let mesh_bw = mesh_for(gc, raw_bw, cf[3]);
        let select_for = |gc: &mut GlContext,
                          raw: GpuTex,
                          opposite: GpuTex,
                          mesh: GpuTex,
                          src_c: GpuTex,
                          dst_c: GpuTex,
                          use_history: bool| {
            let mut mask = run_pass(
                gc,
                ep[3],
                &[
                    ("rawFlow", raw),
                    ("oppositeRawFlow", opposite),
                    ("meshFlow", mesh),
                    ("srcColor", src_c),
                    ("dstColor", dst_c),
                ],
                &[],
                &[],
                w,
                h,
                1,
            );
            gc.set_filter_linear(mask, true);
            for (program, radius) in [(ep[4], 3.0), (ep[5], 3.0), (ep[5], 2.0), (ep[4], 2.0)] {
                mask = run_pass(
                    gc,
                    program,
                    &[("maskTex", mask)],
                    &[("u_radiusPixels", radius)],
                    &[],
                    w,
                    h,
                    1,
                );
                gc.set_filter_linear(mask, true);
            }
            for _ in 0..2 {
                mask = run_pass(gc, ep[6], &[("maskTex", mask)], &[], &[], w, h, 1);
                gc.set_filter_linear(mask, true);
            }
            if use_history && let Some(program) = hysteresis {
                let previous_mask = causal_state.previous_stable_risk.unwrap_or(mask);
                let previous_backward = causal_state.previous_backward.unwrap_or(opposite);
                mask = run_pass(
                    gc,
                    program,
                    &[
                        ("currentMask", mask),
                        ("previousStableMask", previous_mask),
                        ("previousBackwardFlow", previous_backward),
                    ],
                    &[
                        (
                            "u_hasHistory",
                            if causal_state.flow_history_valid {
                                1.0
                            } else {
                                0.0
                            },
                        ),
                        ("u_onThreshold", 0.54),
                        ("u_offThreshold", 0.28),
                        ("u_decay", 0.86),
                    ],
                    &[],
                    w,
                    h,
                    1,
                );
            }
            (
                run_pass(
                    gc,
                    ep[7],
                    &[("rawFlow", raw), ("meshFlow", mesh), ("selectMask", mask)],
                    &[],
                    &[],
                    w,
                    h,
                    4,
                ),
                mask,
            )
        };
        let forward = select_for(gc, raw_fw, raw_bw, mesh_fw, prev, cur, is_causal);
        let backward = select_for(gc, raw_bw, raw_fw, mesh_bw, cur, prev, false);
        (forward.0, backward.0, mesh_fw, mesh_bw, Some(forward.1))
    } else {
        (raw_fw, raw_bw, raw_fw, raw_bw, None)
    };
    gc.set_filter_linear(selected_fw, true);
    gc.set_filter_linear(selected_bw, true);
    let fw = run_pass(
        gc,
        p[5],
        &[
            ("flowTex", selected_fw),
            ("oppositeFlow", selected_bw),
            ("srcColor", prev),
            ("dstColor", cur),
        ],
        &[],
        &[],
        w,
        h,
        4,
    );
    let bw = run_pass(
        gc,
        p[5],
        &[
            ("flowTex", selected_bw),
            ("oppositeFlow", selected_fw),
            ("srcColor", cur),
            ("dstColor", prev),
        ],
        &[],
        &[],
        w,
        h,
        4,
    );
    gc.set_filter_linear(fw, true);
    gc.set_filter_linear(bw, true);
    let scene0 = run_pass(
        gc,
        p[6],
        &[("prevColor", prev), ("curColor", cur), ("forwardConf", fw)],
        &[],
        &[],
        w4,
        h4,
        4,
    );
    let mut scene_levels = vec![scene0];
    let (mut sw, mut sh) = (w4, h4);
    while sw > 1 || sh > 1 {
        sw = (sw + 1) / 2;
        sh = (sh + 1) / 2;
        let next = run_pass(
            gc,
            p[7],
            &[("tex", *scene_levels.last().unwrap())],
            &[],
            &[],
            sw,
            sh,
            4,
        );
        gc.set_filter_linear(next, true);
        scene_levels.push(next);
    }
    let scene = *scene_levels.last().unwrap();
    let midmap = |gc: &mut GlContext, flow: GpuTex| {
        let mut map = run_pass(
            gc,
            p[8],
            &[("flowConf", flow)],
            &[("u_phase", 0.5)],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(map, true);
        for _ in 0..4 {
            map = run_pass(
                gc,
                p[9],
                &[("previousMap", map), ("flowConf", flow)],
                &[("u_phase", 0.5), ("u_damping", 0.85)],
                &[],
                w,
                h,
                4,
            );
            gc.set_filter_linear(map, true);
        }
        map
    };
    let prev_map = midmap(gc, fw);
    let cur_map = midmap(gc, bw);
    let warped_prev = run_pass(
        gc,
        p[10],
        &[("colorTex", prev), ("sourceMap", prev_map)],
        &[],
        &[],
        w,
        h,
        4,
    );
    if is_causal {
        let causal = gc
            .program(&external_nf_pass_source(source, "CAUSAL_WARP")?)
            .map_err(|e| anyhow!("CausalStable CAUSAL_WARP: {e}"))?;
        let causal_detail = run_pass(
            gc,
            causal,
            &[("prevColor", prev), ("sourceMap", prev_map)],
            &[("u_animeWeight", hybrid_anime)],
            &[],
            w,
            h,
            4,
        );
        let safe_forward_conf = run_pass(
            gc,
            p[5],
            &[
                ("flowTex", safe_fw),
                ("oppositeFlow", safe_bw),
                ("srcColor", prev),
                ("dstColor", cur),
            ],
            &[],
            &[],
            w,
            h,
            4,
        );
        let safe_map = midmap(gc, safe_forward_conf);
        let causal_safe = run_pass(
            gc,
            causal,
            &[("prevColor", prev), ("sourceMap", safe_map)],
            &[("u_animeWeight", hybrid_anime)],
            &[],
            w,
            h,
            4,
        );
        let severe_prog = gc
            .program(&external_nf_pass_source(source, "SEVERE_MASK")?)
            .map_err(|e| anyhow!("CausalStable SEVERE_MASK: {e}"))?;
        let mask_max = gc
            .program(&external_nf_pass_source(source, "MASK_MAX")?)
            .map_err(|e| anyhow!(e))?;
        let mask_min = gc
            .program(&external_nf_pass_source(source, "MASK_MIN")?)
            .map_err(|e| anyhow!(e))?;
        let mask_blur = gc
            .program(&external_nf_pass_source(source, "MASK_BLUR")?)
            .map_err(|e| anyhow!(e))?;
        let mut severe = run_pass(
            gc,
            severe_prog,
            &[("warpedPrev", causal_detail), ("warpedCur", causal_safe)],
            &[
                (
                    "u_pairThreshold",
                    if hybrid_anime > 0.5 { 0.045 } else { 0.065 },
                ),
                (
                    "u_reliabilityThreshold",
                    if hybrid_anime > 0.5 { 0.10 } else { 0.13 },
                ),
            ],
            &[],
            w,
            h,
            1,
        );
        for (program, radius) in [
            (mask_max, 8.0),
            (mask_min, 8.0),
            (mask_min, 3.0),
            (mask_max, 3.0),
            (mask_max, 5.0),
        ] {
            severe = run_pass(
                gc,
                program,
                &[("maskTex", severe)],
                &[("u_radiusPixels", radius)],
                &[],
                w,
                h,
                1,
            );
        }
        severe = run_pass(gc, mask_blur, &[("maskTex", severe)], &[], &[], w, h, 1);
        let (line_a, _) = causal_lines.expect("causal lines");
        let ui_prog = gc
            .program(&external_nf_pass_source(source, "UI_MASK")?)
            .map_err(|e| anyhow!(e))?;
        let dilate = gc
            .program(&external_nf_pass_source(source, "MASK_DILATE")?)
            .map_err(|e| anyhow!(e))?;
        let mut ui = run_pass(
            gc,
            ui_prog,
            &[
                ("prevColor", prev),
                ("curColor", cur),
                ("warpedPrev", causal_detail),
                ("warpedCur", causal_safe),
            ],
            &[],
            &[],
            w,
            h,
            1,
        );
        ui = run_pass(gc, dilate, &[("maskTex", ui)], &[], &[], w, h, 1);
        ui = run_pass(gc, dilate, &[("maskTex", ui)], &[], &[], w, h, 1);
        let effect_prog = gc
            .program(&external_nf_pass_source(source, "EFFECT_MASK")?)
            .map_err(|e| anyhow!(e))?;
        let effect = run_pass(
            gc,
            effect_prog,
            &[
                ("prevColor", prev),
                ("curColor", cur),
                ("lineTex", line_a),
                ("forwardConf", fw),
            ],
            &[],
            &[],
            w,
            h,
            4,
        );
        let final_prog = gc
            .program(&external_nf_pass_source(source, "CAUSAL_FINAL")?)
            .map_err(|e| anyhow!("CausalStable CAUSAL_FINAL: {e}"))?;
        let out = run_pass(
            gc,
            final_prog,
            &[
                ("prevColor", prev),
                ("curColor", cur),
                ("causalDetail", causal_detail),
                ("causalSafe", causal_safe),
                ("causalPan", causal_pan_tex.expect("causal pan")),
                ("severeMask", severe),
                ("uiMask", ui),
                ("effectMask", effect),
                ("modeState1x1", causal_mode_tex.expect("causal mode")),
            ],
            &[("u_animeWeight", hybrid_anime)],
            &[],
            w,
            h,
            4,
        );
        let stable_mode = causal_class[0].round() as i32;
        if matches!(stable_mode, 2 | 3) {
            for (slot, value) in [
                (&mut causal_state.previous_stable_forward, selected_fw),
                (&mut causal_state.previous_backward, raw_bw),
                (
                    &mut causal_state.previous_stable_risk,
                    stable_risk.unwrap_or(severe),
                ),
            ] {
                if let Some(old) = slot.replace(value) {
                    gc.recycle(old);
                }
            }
            causal_state.flow_history_valid = true;
        } else {
            causal_state.flow_history_valid = false;
        }
        log::debug!(
            "causal-v05: stable_mode={} candidate={:.0} count={:.0} anime_weight={:.3} mode_history={} flow_history={} path=CAUSAL_FINAL",
            stable_mode,
            causal_class[2],
            causal_class[3],
            hybrid_anime,
            causal_state.mode_history_valid,
            causal_state.flow_history_valid
        );
        gc.set_external_neoflow_state(causal_state);
        return Ok(vec![out]);
    }
    let warped_cur = run_pass(
        gc,
        p[10],
        &[("colorTex", cur), ("sourceMap", cur_map)],
        &[],
        &[],
        w,
        h,
        4,
    );
    gc.set_filter_linear(warped_prev, true);
    gc.set_filter_linear(warped_cur, true);
    let mut vis = run_pass(
        gc,
        p[11],
        &[("warpedPrev", warped_prev), ("warpedCur", warped_cur)],
        &[],
        &[],
        w,
        h,
        4,
    );
    gc.set_filter_linear(vis, true);
    for _ in 0..4 {
        vis = run_pass(
            gc,
            p[12],
            &[
                ("visibilityTex", vis),
                ("warpedPrev", warped_prev),
                ("warpedCur", warped_cur),
            ],
            &[],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(vis, true);
    }
    let ui0 = run_pass(
        gc,
        p[13],
        &[
            ("prevColor", prev),
            ("curColor", cur),
            ("warpedPrev", warped_prev),
            ("warpedCur", warped_cur),
        ],
        &[],
        &[],
        w,
        h,
        1,
    );
    gc.set_filter_linear(ui0, true);
    let ui1 = run_pass(gc, p[14], &[("maskTex", ui0)], &[], &[], w, h, 1);
    gc.set_filter_linear(ui1, true);
    let ui2 = run_pass(gc, p[14], &[("maskTex", ui1)], &[], &[], w, h, 1);
    gc.set_filter_linear(ui2, true);
    let base = run_pass(
        gc,
        p[15],
        &[
            ("prevColor", prev),
            ("curColor", cur),
            ("warpedPrev", warped_prev),
            ("warpedCur", warped_cur),
            ("visibilityTex", vis),
            ("uiMask", ui2),
            ("scene1x1", scene),
        ],
        &[("u_t", 0.5), ("u_animeWeight", hybrid_anime)],
        &[],
        w,
        h,
        4,
    );
    let out = if is_mesh {
        let severe = gc
            .program(&external_nf_pass_source(source, "SEVERE_MASK")?)
            .map_err(|e| anyhow!("GameMesh SEVERE_MASK: {e}"))?;
        let mask_max = gc
            .program(&external_nf_pass_source(source, "MASK_MAX")?)
            .map_err(|e| anyhow!("GameMesh MASK_MAX: {e}"))?;
        let mask_min = gc
            .program(&external_nf_pass_source(source, "MASK_MIN")?)
            .map_err(|e| anyhow!("GameMesh MASK_MIN: {e}"))?;
        let mask_blur = gc
            .program(&external_nf_pass_source(source, "MASK_BLUR")?)
            .map_err(|e| anyhow!("GameMesh MASK_BLUR: {e}"))?;
        let guard = gc
            .program(&external_nf_pass_source(source, "FINAL_GUARD")?)
            .map_err(|e| anyhow!("GameMesh FINAL_GUARD: {e}"))?;
        let mut mask = run_pass(
            gc,
            severe,
            &[("warpedPrev", warped_prev), ("warpedCur", warped_cur)],
            &[],
            &[],
            w,
            h,
            1,
        );
        gc.set_filter_linear(mask, true);
        for (program, radius) in [(mask_max, 3.0), (mask_min, 3.0), (mask_max, 2.0)] {
            mask = run_pass(
                gc,
                program,
                &[("maskTex", mask)],
                &[("u_radiusPixels", radius)],
                &[],
                w,
                h,
                1,
            );
            gc.set_filter_linear(mask, true);
        }
        for _ in 0..2 {
            mask = run_pass(gc, mask_blur, &[("maskTex", mask)], &[], &[], w, h, 1);
            gc.set_filter_linear(mask, true);
        }
        run_pass(
            gc,
            guard,
            &[
                ("generatedColor", base),
                ("currentRealColor", cur),
                ("severeMask", mask),
            ],
            &[],
            &[],
            w,
            h,
            4,
        )
    } else {
        base
    };
    log::info!(
        "NeoFlow {} host active: x2 midpoint",
        if is_hybrid {
            "HybridCadence v0.4"
        } else if is_mesh {
            "GameMesh v0.3"
        } else {
            "GameDIS v0.2"
        }
    );
    Ok(vec![out])
}

fn run_compute_owner(
    gc: &mut GlContext,
    program: glow::Program,
    owner: GpuTex,
    binds: &[(&str, GpuTex)],
    t: Option<f32>,
    dispatch: (i32, i32),
) {
    let gl = gc.gl.clone();
    unsafe {
        gl.use_program(Some(program));
        gl.bind_image_texture(
            0,
            Some(owner.tex),
            0,
            false,
            0,
            glow::READ_WRITE,
            glow::R32UI,
        );
        for (unit, (name, tex)) in binds.iter().enumerate() {
            gl.active_texture(glow::TEXTURE0 + unit as u32);
            gl.bind_texture(glow::TEXTURE_2D, Some(tex.tex));
            if let Some(loc) = gl.get_uniform_location(program, name) {
                gl.uniform_1_i32(Some(&loc), unit as i32);
            }
        }
        if let Some(value) = t
            && let Some(loc) = gl.get_uniform_location(program, "t")
        {
            gl.uniform_1_f32(Some(&loc), value);
        }
        gl.dispatch_compute(
            (dispatch.0.max(1) as u32).div_ceil(8),
            (dispatch.1.max(1) as u32).div_ceil(8),
            1,
        );
    }
}

/// Expanded game-reconstruction contract: shared bidirectional analysis,
/// integer owner-map forward splatting, conservative hole fill and final
/// endpoint fallback. OpenGL 4.3 is required by the supplied compute passes.
fn interpolate_many_game_reconstruct(
    gc: &mut GlContext,
    prev: GpuTex,
    cur: GpuTex,
    phases: &[f32],
    source: &str,
) -> Result<Vec<GpuTex>> {
    if phases.is_empty() {
        return Ok(Vec::new());
    }
    let (w, h) = (prev.w(), prev.h());
    anyhow::ensure!(
        cur.w() == w && cur.h() == h,
        "GameReconstruct: frame size changed"
    );
    let frag_names = [
        "FEATURE",
        "GLOBAL_SEARCH",
        "SEARCH16",
        "REFINE8",
        "REFINE4",
        "CONSIST",
        "SCENE_SCORE",
        "REDUCE",
        "UI_MASK",
        "UI_DILATE",
        "OWNER_RESOLVE",
        "HOLE_FILL",
        "FINAL",
    ];
    let mut fp = Vec::with_capacity(frag_names.len());
    for name in frag_names {
        fp.push(
            gc.program(&external_pass_source(source, name)?)
                .map_err(|e| anyhow!("GameReconstruct {name}: {e}"))?,
        );
    }
    let compute_names = ["OWNER_CLEAR", "SPLAT_PREV", "SPLAT_CUR"];
    let mut cp = Vec::with_capacity(compute_names.len());
    for name in compute_names {
        cp.push(
            gc.compute_program(&external_pass_source(source, name)?)
                .map_err(|e| anyhow!("GameReconstruct {name}: {e}"))?,
        );
    }
    let (w2, h2) = ((w + 1) / 2, (h + 1) / 2);
    let (w4, h4) = ((w + 3) / 4, (h + 3) / 4);
    let (w8, h8) = ((w + 7) / 8, (h + 7) / 8);
    let (w16, h16) = ((w + 15) / 16, (h + 15) / 16);
    gc.set_filter_linear(prev, true);
    gc.set_filter_linear(cur, true);
    let feature_chain = |gc: &mut GlContext, input: GpuTex| {
        let f2 = run_pass(gc, fp[0], &[("tex", input)], &[], &[], w2, h2, 4);
        gc.set_filter_linear(f2, true);
        let f4 = run_pass(gc, fp[0], &[("tex", f2)], &[], &[], w4, h4, 4);
        gc.set_filter_linear(f4, true);
        let f8 = run_pass(gc, fp[0], &[("tex", f4)], &[], &[], w8, h8, 4);
        gc.set_filter_linear(f8, true);
        let f16 = run_pass(gc, fp[0], &[("tex", f8)], &[], &[], w16, h16, 4);
        gc.set_filter_linear(f16, true);
        [f2, f4, f8, f16]
    };
    let pf = feature_chain(gc, prev);
    let cf = feature_chain(gc, cur);
    let global_fw = run_pass(
        gc,
        fp[1],
        &[("srcF16", pf[3]), ("dstF16", cf[3])],
        &[],
        &[],
        1,
        1,
        4,
    );
    let global_bw = run_pass(
        gc,
        fp[1],
        &[("srcF16", cf[3]), ("dstF16", pf[3])],
        &[],
        &[],
        1,
        1,
        4,
    );
    let search_fw = run_pass(
        gc,
        fp[2],
        &[
            ("srcF16", pf[3]),
            ("dstF16", cf[3]),
            ("globalFlow", global_fw),
        ],
        &[],
        &[],
        w16,
        h16,
        4,
    );
    let search_bw = run_pass(
        gc,
        fp[2],
        &[
            ("srcF16", cf[3]),
            ("dstF16", pf[3]),
            ("globalFlow", global_bw),
        ],
        &[],
        &[],
        w16,
        h16,
        4,
    );
    for tex in [global_fw, global_bw, search_fw, search_bw] {
        gc.set_filter_linear(tex, true);
    }
    let raw_fw8 = run_pass(
        gc,
        fp[3],
        &[
            ("srcF8", pf[2]),
            ("dstF8", cf[2]),
            ("initFlow16", search_fw),
            ("globalFlow", global_fw),
        ],
        &[],
        &[],
        w8,
        h8,
        4,
    );
    let raw_bw8 = run_pass(
        gc,
        fp[3],
        &[
            ("srcF8", cf[2]),
            ("dstF8", pf[2]),
            ("initFlow16", search_bw),
            ("globalFlow", global_bw),
        ],
        &[],
        &[],
        w8,
        h8,
        4,
    );
    gc.set_filter_linear(raw_fw8, true);
    gc.set_filter_linear(raw_bw8, true);
    let raw_fw4 = run_pass(
        gc,
        fp[4],
        &[("srcF4", pf[1]), ("dstF4", cf[1]), ("initFlow8", raw_fw8)],
        &[],
        &[],
        w4,
        h4,
        4,
    );
    let raw_bw4 = run_pass(
        gc,
        fp[4],
        &[("srcF4", cf[1]), ("dstF4", pf[1]), ("initFlow8", raw_bw8)],
        &[],
        &[],
        w4,
        h4,
        4,
    );
    gc.set_filter_linear(raw_fw4, true);
    gc.set_filter_linear(raw_bw4, true);
    let fw4 = run_pass(
        gc,
        fp[5],
        &[
            ("forwardRaw4", raw_fw4),
            ("backwardRaw4", raw_bw4),
            ("srcF4", pf[1]),
            ("dstF4", cf[1]),
        ],
        &[],
        &[],
        w4,
        h4,
        4,
    );
    let bw4 = run_pass(
        gc,
        fp[5],
        &[
            ("forwardRaw4", raw_bw4),
            ("backwardRaw4", raw_fw4),
            ("srcF4", cf[1]),
            ("dstF4", pf[1]),
        ],
        &[],
        &[],
        w4,
        h4,
        4,
    );
    gc.set_filter_linear(fw4, true);
    gc.set_filter_linear(bw4, true);
    let scene0 = run_pass(
        gc,
        fp[6],
        &[
            ("prevF16", pf[3]),
            ("curF16", cf[3]),
            ("globalForward", global_fw),
        ],
        &[],
        &[],
        w16,
        h16,
        4,
    );
    let mut scene_levels = vec![scene0];
    let (mut sw, mut sh) = (w16, h16);
    while sw > 1 || sh > 1 {
        sw = (sw + 1) / 2;
        sh = (sh + 1) / 2;
        let next = run_pass(
            gc,
            fp[7],
            &[("tex", *scene_levels.last().unwrap())],
            &[],
            &[],
            sw,
            sh,
            4,
        );
        gc.set_filter_linear(next, true);
        scene_levels.push(next);
    }
    let scene = *scene_levels.last().unwrap();
    let ui0 = run_pass(
        gc,
        fp[8],
        &[
            ("prevC", prev),
            ("curC", cur),
            ("forward4", fw4),
            ("backward4", bw4),
        ],
        &[],
        &[],
        w,
        h,
        1,
    );
    gc.set_filter_linear(ui0, true);
    let ui1 = run_pass(gc, fp[9], &[("maskTex", ui0)], &[], &[], w, h, 1);
    gc.set_filter_linear(ui1, true);
    let ui2 = run_pass(gc, fp[9], &[("maskTex", ui1)], &[], &[], w, h, 1);
    gc.set_filter_linear(ui2, true);
    let mut outputs = Vec::with_capacity(phases.len());
    for &phase in phases {
        let t = phase.clamp(0.0, 1.0);
        let owner = gc.make_tex(w, h, 1, Dtype::U32);
        run_compute_owner(gc, cp[0], owner, &[], None, (w, h));
        unsafe {
            gc.gl.memory_barrier(glow::SHADER_IMAGE_ACCESS_BARRIER_BIT);
        }
        run_compute_owner(
            gc,
            cp[1],
            owner,
            &[("sourceColor", prev), ("sourceFlow4", fw4), ("uiMask", ui2)],
            Some(t),
            (w, h),
        );
        run_compute_owner(
            gc,
            cp[2],
            owner,
            &[("sourceColor", cur), ("sourceFlow4", bw4), ("uiMask", ui2)],
            Some(t),
            (w, h),
        );
        unsafe {
            gc.gl.memory_barrier(
                glow::SHADER_IMAGE_ACCESS_BARRIER_BIT | glow::TEXTURE_FETCH_BARRIER_BIT,
            );
        }
        let resolved = run_pass(
            gc,
            fp[10],
            &[("ownerTex", owner), ("prevC", prev), ("curC", cur)],
            &[],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(resolved, true);
        let fill1 = run_pass(
            gc,
            fp[11],
            &[("inputTex", resolved), ("prevC", prev), ("curC", cur)],
            &[("t", t), ("fillStep", 1.0)],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(fill1, true);
        let fill2 = run_pass(
            gc,
            fp[11],
            &[("inputTex", fill1), ("prevC", prev), ("curC", cur)],
            &[("t", t), ("fillStep", 2.0)],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(fill2, true);
        let fill3 = run_pass(
            gc,
            fp[11],
            &[("inputTex", fill2), ("prevC", prev), ("curC", cur)],
            &[("t", t), ("fillStep", 4.0)],
            &[],
            w,
            h,
            4,
        );
        gc.set_filter_linear(fill3, true);
        outputs.push(run_pass(
            gc,
            fp[12],
            &[
                ("generatedTex", fill3),
                ("ownerTex", owner),
                ("prevC", prev),
                ("curC", cur),
                ("forward4", fw4),
                ("backward4", bw4),
                ("uiMask", ui2),
                ("scene1x1", scene),
            ],
            &[("t", t)],
            &[],
            w,
            h,
            4,
        ));
    }
    log::info!("NeoFlow GameReconstruct expanded host active: fragment=13 compute=3 owner=R32UI");
    Ok(outputs)
}

fn external_pass_source(source: &str, pass: &str) -> Result<String> {
    let newline = source
        .find('\n')
        .ok_or_else(|| anyhow!("external NeoFlow source has no version line"))?;
    if !source[..newline].trim().starts_with("#version") {
        return Err(anyhow!("external NeoFlow source must start with #version"));
    }
    Ok(format!(
        "{}\n#define NEOFLOW_PASS_{pass}\n{}",
        &source[..newline],
        &source[newline + 1..]
    ))
}

fn interpolate_many_with_source(
    gc: &mut GlContext,
    prev: GpuTex,
    cur: GpuTex,
    phases: &[f32],
    external: Option<&str>,
) -> Result<Vec<GpuTex>> {
    if phases.is_empty() {
        return Ok(Vec::new());
    }
    let (w, h) = (prev.w(), prev.h());
    if cur.w() != w || cur.h() != h {
        return Err(anyhow!("NeoFlow: frame size changed"));
    }
    let (w8, h8) = ((w + 7) / 8, (h + 7) / 8);
    let (w16, h16) = ((w + 15) / 16, (h + 15) / 16);

    let external_sources = external
        .map(|source| {
            Ok::<_, anyhow::Error>([
                external_pass_source(source, "LUMA")?,
                external_pass_source(source, "SEARCH16")?,
                external_pass_source(source, "REFINE8")?,
                external_pass_source(source, "CONSIST")?,
                external_pass_source(source, "SCENE_SCORE")?,
                external_pass_source(source, "REDUCE")?,
                external_pass_source(source, "COMPOSITE")?,
                external_pass_source(source, "COPY")?,
            ])
        })
        .transpose()?;
    let shader = |index: usize, builtin: &'static str| -> &str {
        external_sources
            .as_ref()
            .map(|sources| sources[index].as_str())
            .unwrap_or(builtin)
    };
    let luma = gc.program(shader(0, LUMA_FRAG)).map_err(|e| anyhow!(e))?;
    let s16 = gc
        .program(shader(1, SEARCH16_FRAG))
        .map_err(|e| anyhow!(e))?;
    let r8 = gc
        .program(shader(2, REFINE8_FRAG))
        .map_err(|e| anyhow!(e))?;
    let cons = gc
        .program(shader(3, CONSIST_FRAG))
        .map_err(|e| anyhow!(e))?;
    let scene_score = gc
        .program(shader(4, SCENE_SCORE_FRAG))
        .map_err(|e| anyhow!(e))?;
    let reduce = gc.program(shader(5, REDUCE_FRAG)).map_err(|e| anyhow!(e))?;
    let comp = gc
        .program(shader(6, COMPOSITE_FRAG))
        .map_err(|e| anyhow!(e))?;
    let copy = gc.program(shader(7, COPY_FRAG)).map_err(|e| anyhow!(e))?;
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            log::info!(
                "NeoFlow: bidirectional pyramid flow + consistency/occlusion resolve enabled"
            )
        });
    }

    gc.set_filter_linear(prev, true);
    gc.set_filter_linear(cur, true);

    // pyramid
    let p8 = run_pass(gc, luma, &[("tex", prev)], &[], &[], w8, h8, 1);
    let c8 = run_pass(gc, luma, &[("tex", cur)], &[], &[], w8, h8, 1);
    gc.set_filter_linear(p8, true);
    gc.set_filter_linear(c8, true);
    let p16 = run_pass(gc, luma, &[("tex", p8)], &[], &[], w16, h16, 1);
    let c16 = run_pass(gc, luma, &[("tex", c8)], &[], &[], w16, h16, 1);
    gc.set_filter_linear(p16, true);
    gc.set_filter_linear(c16, true);

    let pt16 = (1.0 / w16 as f32, 1.0 / h16 as f32);
    let pt8 = (1.0 / w8 as f32, 1.0 / h8 as f32);
    // coarse forward search first: its mean best-match COST is the scene-cut
    // probe (a cut matches nowhere even after searching; a fast scroll still
    // finds good matches, unlike a raw frame-difference metric)
    let fw16 = run_pass(
        gc,
        s16,
        &[("srcL", p16), ("dstL", c16)],
        &[],
        &[("pt", pt16)],
        w16,
        h16,
        4,
    );
    let cut = if NEOFLOW_CPU_CUT_PROBE {
        mean_ch(gc, fw16, 2)
    } else {
        0.0
    };
    let mut scene_levels = Vec::new();
    let scene0 = run_pass(
        gc,
        scene_score,
        &[("flow", fw16), ("srcL", p16), ("dstL", c16)],
        &[],
        &[],
        w16,
        h16,
        4,
    );
    gc.set_filter_linear(scene0, true);
    scene_levels.push(scene0);
    let (mut sw, mut sh) = (w16, h16);
    while sw > 1 || sh > 1 {
        let src_size = (sw, sh);
        sw = (sw + 1) / 2;
        sh = (sh + 1) / 2;
        let next = run_pass(
            gc,
            reduce,
            &[("tex", *scene_levels.last().unwrap())],
            &[],
            &[("pt", (1.0 / src_size.0 as f32, 1.0 / src_size.1 as f32))],
            sw,
            sh,
            1,
        );
        gc.set_filter_linear(next, true);
        scene_levels.push(next);
    }
    let scene = *scene_levels.last().unwrap();
    let results = if cut > 0.15 {
        // hard cut: hold the PREVIOUS frame — the new scene must appear
        // exactly with its real frame (never leak early, never ghost)
        phases
            .iter()
            .map(|_| run_pass(gc, copy, &[("tex", prev)], &[], &[], w, h, 4))
            .collect()
    } else {
        // both directions: coarse then refine
        let bw16 = run_pass(
            gc,
            s16,
            &[("srcL", c16), ("dstL", p16)],
            &[],
            &[("pt", pt16)],
            w16,
            h16,
            4,
        );
        gc.set_filter_linear(fw16, true);
        gc.set_filter_linear(bw16, true);
        let fw8 = run_pass(
            gc,
            r8,
            &[("srcL", p8), ("dstL", c8), ("init", fw16)],
            &[],
            &[("pt", pt8), ("ipt", pt16)],
            w8,
            h8,
            4,
        );
        let bw8 = run_pass(
            gc,
            r8,
            &[("srcL", c8), ("dstL", p8), ("init", bw16)],
            &[],
            &[("pt", pt8), ("ipt", pt16)],
            w8,
            h8,
            4,
        );
        gc.set_filter_linear(fw8, true);
        gc.set_filter_linear(bw8, true);
        let fw_valid = run_pass(
            gc,
            cons,
            &[("fw", fw8), ("bw", bw8), ("srcL", p8), ("dstL", c8)],
            &[],
            &[("pt", pt8)],
            w8,
            h8,
            4,
        );
        let bw_valid = run_pass(
            gc,
            cons,
            &[("fw", bw8), ("bw", fw8), ("srcL", c8), ("dstL", p8)],
            &[],
            &[("pt", pt8)],
            w8,
            h8,
            4,
        );
        gc.set_filter_linear(fw_valid, true);
        gc.set_filter_linear(bw_valid, true);
        let outputs = phases
            .iter()
            .map(|phase| {
                run_pass(
                    gc,
                    comp,
                    &[
                        ("prevC", prev),
                        ("curC", cur),
                        ("fw", fw_valid),
                        ("bw", bw_valid),
                        ("scene", scene),
                    ],
                    &[("t", phase.clamp(0.0, 1.0))],
                    &[("px", (1.0 / w as f32, 1.0 / h as f32))],
                    w,
                    h,
                    4,
                )
            })
            .collect();
        for tex in [fw16, bw16, fw8, bw8, fw_valid, bw_valid] {
            gc.set_filter_linear(tex, false);
        }
        outputs
    };

    for tex in [prev, cur, p8, c8, p16, c16] {
        gc.set_filter_linear(tex, false);
    }
    for tex in scene_levels {
        gc.set_filter_linear(tex, false);
    }
    Ok(results)
}
