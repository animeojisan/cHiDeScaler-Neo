//! NeoFlow - unified built-in GPU frame interpolation.
//!
//! The pipeline is designed for generic real-time content:
//!
//! * SYMMETRIC (bilateral) motion estimation, the MEMC technique used in
//!   TV frame-rate converters: the field is anchored at the OUTPUT pixel grid
//!   and each candidate displacement `d` is scored by comparing
//!   prev(x − d/2) against cur(x + d/2). The standard NeoFlow anchors flow at
//!   the PREV frame and gathers it at the output position — for small fast
//!   objects the gathered vector belongs to the background, so the object
//!   can vanish from the in-between frame. Symmetric anchoring makes the object
//!   vote for itself where it actually needs to be drawn.
//! * FULL search at EVERY pyramid level (not coarse-to-fine refinement only).
//!   A 10px object is invisible at 1/16 resolution, so a coarse-init refine
//!   chain can miss it, so later levels repeat a bounded local search.
//! * SPATIAL CANDIDATE PROPAGATION at each refine level (the classic
//!   multi-candidate trick): neighbouring vectors compete at every pixel, so a
//!   lock-on at an object's centre spreads over the whole object, and crossing
//!   objects keep separate fields on each side of the crossing.
//! * MEDIAN-OF-3 composite: median(warped-prev, warped-cur, temporal blend)
//!   per channel. Where motion is right the median equals the warp; where one
//!   side is occluded (objects crossing) the median silently drops the
//!   outlier instead of ghosting. This lets NeoFlow interpolate aggressively
//!   without the "artifacts vs smoothness" trade-off collapsing to blend.
//! * Zero-motion bias + match-ambiguity confidence: repetitive textures such as
//!   wire fences produce many equally good matches; the ambiguity gap between
//!   best and runner-up feeds confidence so those areas degrade to a clean
//!   blend instead of crawling, while the fence itself (static, zero-biased)
//!   stays pixel-stable.
//! * The TextFX guard protects transparent logos, subtitles, and thin lines.
//! * A three-component Y/Co/Cg feature pyramid distinguishes objects that
//!   differ in color even when their luminance is similar.
//! * Three-frame context is used only to protect broad discontinuities such as
//!   scene cuts. It does not freeze local two/three-frame animation.
//!
//! The production path uses pooled F16 textures and performs no CPU readback.

use super::flow::run_pass;
use super::gl::{GlContext, GpuTex};
use anyhow::{Result, anyhow};

const LUMA_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ vec3 c = texture(tex, v_uv).rgb; float y=dot(c,vec3(0.299,0.587,0.114)); float co=c.r-c.b; float cg=c.g-0.5*(c.r+c.b); frag=vec4(y,co,cg,1.0); }\n";

// luma->luma half-res downsample. NOT the LUMA_FRAG: applying the luma dot to
// an already-luma (R,0,0) texture multiplies by 0.299 per level, and by 1/16
// the pyramid had 2.7% of the real contrast — the coarse search saw nothing
// Reapplying luma conversion at each level would progressively destroy contrast.
const DOWN_FRAG: &str = "#version 330\nin vec2 v_uv; out vec4 frag;\nuniform sampler2D tex;\nvoid main(){ frag = vec4(texture(tex, v_uv).rgb, 1.0); }\n";

// ---- symmetric coarse search at 1/16: full ±6 texels step 2 + ±1 refine ----
// Output: xy = displacement in 1/16 texels (full prev→cur), z = best cost,
// w = ambiguity gap (runner-up − best; small = repetitive/flat = untrustworthy).
// Cost = symmetric SAD + zero-bias − annihilation bonus. The bonus solves the
// classic MEMC "small object vanishes" failure: over a flat background, the
// object's midpoint pixel scores ~0 for BOTH (object, correct d) and
// (background, d=0), and the zero-bias then erases the object from the
// in-between frame. A candidate whose two endpoint samples BOTH changed
// between the frames (the object left prev there AND arrived in cur there) is
// the object hypothesis — reward it so it wins that tie. Static/flat regions
// have zero activity, so the bias still pins them to zero motion.
const SYM16_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevL; uniform sampler2D curL; uniform vec2 pt;
// x = SAD, y = structure (mean deviation of the two windows). A match with
// no structure is uninformative: two flat background patches always agree, so
// without a flatness tax a bg-bg ghost vector (SAD ~0) beats a slightly
// misaligned TRUE object match (SAD ~0.3) every time — measured exactly that
// on the stress scene. The tax makes structured matches win; genuinely static
// flat areas still settle at v=0 through the |v| bias (all candidates pay the
// same tax there).
vec2 sad2(vec2 d){
  vec3 ac = texture(prevL, v_uv - d*0.5).rgb;
  vec3 bc = texture(curL, v_uv + d*0.5).rgb;
  float s = 0.0; float st = 0.0;
  // 5x5 color window: wide support + chroma separate true matches from
  // flat-background ghosts that fooled luma-only matching.
  for (int dy=-2; dy<=2; dy++)
  for (int dx=-2; dx<=2; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 a = texture(prevL, v_uv - d*0.5 + o).rgb;
    vec3 b = texture(curL, v_uv + d*0.5 + o).rgb;
    s += dot(abs(a - b), vec3(0.12));
    st += dot(abs(a - ac) + abs(b - bc), vec3(0.12));
  }
  return vec2(s, st);
}
float sad(vec2 d){ return sad2(d).x; }
float act(vec2 d){
  float ap = dot(abs(texture(prevL, v_uv - d*0.5).rgb - texture(curL, v_uv - d*0.5).rgb), vec3(0.3333));
  float ac = dot(abs(texture(prevL, v_uv + d*0.5).rgb - texture(curL, v_uv + d*0.5).rgb), vec3(0.3333));
  return min(ap, ac);
}
float oob(vec2 d){
  vec2 a = v_uv - d*0.5; vec2 b = v_uv + d*0.5;
  vec2 lo = min(min(a,b), vec2(0.0));
  vec2 hi = max(max(a,b), vec2(1.0));
  return (length(-lo) + length(hi - vec2(1.0))) * 40.0;
}
float cost(vec2 v){
  vec2 d = v*pt;
  vec2 sv = sad2(d);
  float flat_tax = 0.45 * (1.0 - smoothstep(0.03, 0.20, sv.y));
  return sv.x + 0.02*length(v) + oob(d) + flat_tax - 1.0*act(d);
}
void main(){
  // step-2 grid scan keeping the TOP-3 candidates. Refining only around the
  // single winner missed the true motion entirely: its step-2 neighbours pay a
  // full misaligned-row penalty and lose to background noise vectors, so the
  // ±1 polish never got to look near the truth. Poking around three seeds
  // fixes that for ~30% more cost.
  vec2 v1 = vec2(0.0); float c1 = cost(vec2(0.0));
  vec2 v2 = v1; float cB = 1e9;
  vec2 v3 = v1; float cC = 1e9;
  for (int dy=-12; dy<=12; dy+=2)
  for (int dx=-12; dx<=12; dx+=2){
    if (dx==0 && dy==0) continue;
    vec2 v = vec2(dx,dy);
    float c = cost(v);
    if (c < c1){ cC = cB; v3 = v2; cB = c1; v2 = v1; c1 = c; v1 = v; }
    else if (c < cB){ cC = cB; v3 = v2; cB = c; v2 = v; }
    else if (c < cC){ cC = c; v3 = v; }
  }
  float gap = max(cB - c1, 0.0);
  vec2 best = v1; float bc = c1;
  for (int k = 0; k < 3; k++){
    vec2 seed = (k==0) ? v1 : (k==1) ? v2 : v3;
    for (int dy=-1; dy<=1; dy++)
    for (int dx=-1; dx<=1; dx++){
      vec2 v = seed + vec2(dx,dy);
      float c = cost(v);
      if (c < bc){ bc = c; best = v; }
    }
  }
  // fractional polish: the true optimum often sits at a half-texel phase
  // (downsampled edges land mid-texel); the CPU landscape dump showed the
  // fractional minimum 30% below the best integer candidate.
  vec2 seedf = best;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    if (dx==0 && dy==0) continue;
    vec2 v = seedf + vec2(dx,dy)*0.5;
    float c = cost(v);
    if (c < bc){ bc = c; best = v; }
  }
  frag = vec4(best, max(bc, 0.0), gap);
}
"#;

// ---- symmetric search+refine at 1/8: candidates ∪ full ±6 step 2 ∪ ±1 ----
// Candidates: upsampled coarse init and its 4 neighbours (spatial propagation)
// plus zero motion. Then an independent full search so that small objects
// invisible at 1/16 are still caught, then a ±1 polish around the winner.
#[allow(dead_code)] // retained experimental 1/8 symmetric search shader
const SYM8_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevL; uniform sampler2D curL; uniform sampler2D init;
uniform vec2 pt; uniform vec2 ipt;
// x = SAD, y = structure (mean deviation of the two windows). A match with
// no structure is uninformative: two flat background patches always agree, so
// without a flatness tax a bg-bg ghost vector (SAD ~0) beats a slightly
// misaligned TRUE object match (SAD ~0.3) every time — measured exactly that
// on the stress scene. The tax makes structured matches win; genuinely static
// flat areas still settle at v=0 through the |v| bias (all candidates pay the
// same tax there).
vec2 sad2(vec2 d){
  vec3 ac = texture(prevL, v_uv - d*0.5).rgb;
  vec3 bc = texture(curL, v_uv + d*0.5).rgb;
  float s = 0.0; float st = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 a = texture(prevL, v_uv - d*0.5 + o).rgb;
    vec3 b = texture(curL, v_uv + d*0.5 + o).rgb;
    s += dot(abs(a - b), vec3(0.3333));
    st += dot(abs(a - ac) + abs(b - bc), vec3(0.3333));
  }
  return vec2(s, st);
}
float sad(vec2 d){ return sad2(d).x; }
float act(vec2 d){
  float ap = dot(abs(texture(prevL, v_uv - d*0.5).rgb - texture(curL, v_uv - d*0.5).rgb), vec3(0.3333));
  float ac = dot(abs(texture(prevL, v_uv + d*0.5).rgb - texture(curL, v_uv + d*0.5).rgb), vec3(0.3333));
  return min(ap, ac);
}
float oob(vec2 d){
  vec2 a = v_uv - d*0.5; vec2 b = v_uv + d*0.5;
  vec2 lo = min(min(a,b), vec2(0.0));
  vec2 hi = max(max(a,b), vec2(1.0));
  return (length(-lo) + length(hi - vec2(1.0))) * 40.0;
}
float cost(vec2 v){
  vec2 d = v*pt;
  vec2 sv = sad2(d);
  float flat_tax = 0.45 * (1.0 - smoothstep(0.03, 0.20, sv.y));
  return sv.x + 0.012*length(v) + oob(d) + flat_tax - 1.0*act(d);
}
void main(){
  vec2 best = vec2(0.0);
  float bc = cost(vec2(0.0));
  float second = 1e9;
  // spatial candidate propagation from the coarse field
  for (int i = 0; i < 5; i++){
    vec2 off = (i==1) ? vec2( 1.5, 0.0) : (i==2) ? vec2(-1.5, 0.0)
             : (i==3) ? vec2(0.0,  1.5) : (i==4) ? vec2(0.0, -1.5) : vec2(0.0);
    vec2 v = texture(init, v_uv + off*ipt).xy * 2.0;   // 1/16 -> 1/8 texels
    float c = cost(v);
    if (c < bc){ second = bc; bc = c; best = v; }
    else if (c < second){ second = c; }
  }
  // independent full search: catches small objects the coarse level missed
  for (int dy=-6; dy<=6; dy+=2)
  for (int dx=-6; dx<=6; dx+=2){
    vec2 v = vec2(dx,dy);
    float c = cost(v);
    if (c < bc){ second = bc; bc = c; best = v; }
    else if (c < second){ second = c; }
  }
  vec2 b2 = best; float c2 = bc;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 v = best + vec2(dx,dy);
    float c = cost(v);
    if (c < c2){ c2 = c; b2 = v; }
  }
  frag = vec4(b2, max(c2, 0.0), max(second - bc, 0.0));
}
"#;

// ---- symmetric refine at 1/4 and 1/2: candidates + ±2/±1 jitter ----
const SYM_REFINE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevL; uniform sampler2D curL; uniform sampler2D init;
uniform vec2 pt; uniform vec2 ipt; uniform float jitter; uniform float allowZero;
// x = SAD, y = structure (mean deviation of the two windows). A match with
// no structure is uninformative: two flat background patches always agree, so
// without a flatness tax a bg-bg ghost vector (SAD ~0) beats a slightly
// misaligned TRUE object match (SAD ~0.3) every time — measured exactly that
// on the stress scene. The tax makes structured matches win; genuinely static
// flat areas still settle at v=0 through the |v| bias (all candidates pay the
// same tax there).
vec2 sad2(vec2 d){
  vec3 ac = texture(prevL, v_uv - d*0.5).rgb;
  vec3 bc = texture(curL, v_uv + d*0.5).rgb;
  float s = 0.0; float st = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 a = texture(prevL, v_uv - d*0.5 + o).rgb;
    vec3 b = texture(curL, v_uv + d*0.5 + o).rgb;
    s += dot(abs(a - b), vec3(0.3333));
    st += dot(abs(a - ac) + abs(b - bc), vec3(0.3333));
  }
  return vec2(s, st);
}
float sad(vec2 d){ return sad2(d).x; }
float act(vec2 d){
  float ap = dot(abs(texture(prevL, v_uv - d*0.5).rgb - texture(curL, v_uv - d*0.5).rgb), vec3(0.3333));
  float ac = dot(abs(texture(prevL, v_uv + d*0.5).rgb - texture(curL, v_uv + d*0.5).rgb), vec3(0.3333));
  return min(ap, ac);
}
float oob(vec2 d){
  vec2 a = v_uv - d*0.5; vec2 b = v_uv + d*0.5;
  vec2 lo = min(min(a,b), vec2(0.0));
  vec2 hi = max(max(a,b), vec2(1.0));
  return (length(-lo) + length(hi - vec2(1.0))) * 40.0;
}
float cost(vec2 v){
  vec2 d = v*pt;
  vec2 sv = sad2(d);
  float flat_tax = 0.45 * (1.0 - smoothstep(0.03, 0.20, sv.y));
  return sv.x + 0.008*length(v) + oob(d) + flat_tax - 1.0*act(d);
}
void main(){
  vec4 c0 = texture(init, v_uv);
  vec2 best = c0.xy * 2.0;
  float bc = cost(best);
  float gap = c0.w;
  // neighbour candidates (spatial propagation across object boundaries)
  for (int i = 1; i < 5; i++){
    vec2 off = (i==1) ? vec2( 1.5, 0.0) : (i==2) ? vec2(-1.5, 0.0)
             : (i==3) ? vec2(0.0,  1.5) : vec2(0.0, -1.5);
    vec2 v = texture(init, v_uv + off*ipt).xy * 2.0;
    float c = cost(v);
    if (c < bc){ bc = c; best = v; }
  }
  if (allowZero > 0.5) { // static overlays and fences, before final full-res refine
    float c = cost(vec2(0.0));
    if (c < bc){ bc = c; best = vec2(0.0); }
  }
  vec2 b2 = best; float c2 = bc;
  // ±3: a small CG element moving against a panning background needs up to
  // ~12px of correction away from the background seed (anime mixed-fps test)
  for (int dy=-3; dy<=3; dy++)
  for (int dx=-3; dx<=3; dx++){
    vec2 v = best + vec2(dx,dy)*jitter;
    float c = cost(v);
    if (c < c2){ c2 = c; b2 = v; }
  }
  frag = vec4(b2, max(c2, 0.0), gap);
}
"#;

// ---- support-weighted vector propagation ----
// The interior of a moving object over a flat background is INFORMATION-FREE
// for pixel-local matching: both (object, correct d) and (ghost, wrong d)
// score ~zero SAD, and the probe showed the zero-bias then locks in a wrong
// low-magnitude vector (orange-rect midpoint got (−8.9,1) instead of (+38,−4)).
// But the object's EDGE pixels are unambiguous (the window sees structure), and
// their ambiguity gap (.w = runner-up − best) is high. This pass lets each
// pixel adopt a nearby vector when that candidate matches at least as well
// locally AND its source pixel had real support — so edge-resolved vectors
// flood the ambiguous interior. Wrong vectors on flat background are harmless
// (bg warps to bg); vectors that would distort real content are rejected by
// the local re-score.
const PROP_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D fieldT; uniform sampler2D prevL; uniform sampler2D curL; uniform vec2 pt;
// x = SAD, y = structure (mean deviation of the two windows). A match with
// no structure is uninformative: two flat background patches always agree, so
// without a flatness tax a bg-bg ghost vector (SAD ~0) beats a slightly
// misaligned TRUE object match (SAD ~0.3) every time — measured exactly that
// on the stress scene. The tax makes structured matches win; genuinely static
// flat areas still settle at v=0 through the |v| bias (all candidates pay the
// same tax there).
vec2 sad2(vec2 d){
  vec3 ac = texture(prevL, v_uv - d*0.5).rgb;
  vec3 bc = texture(curL, v_uv + d*0.5).rgb;
  float s = 0.0; float st = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    vec3 a = texture(prevL, v_uv - d*0.5 + o).rgb;
    vec3 b = texture(curL, v_uv + d*0.5 + o).rgb;
    s += dot(abs(a - b), vec3(0.3333));
    st += dot(abs(a - ac) + abs(b - bc), vec3(0.3333));
  }
  return vec2(s, st);
}
float sad(vec2 d){ return sad2(d).x; }
// Re-score with the SAME cost model as the search (flatness tax included!) —
// a plain-SAD re-score let flat-background ghost vectors (SAD≈0) overwrite
// true structured matches during propagation.
float pscore(vec2 v){
  vec2 d = v*pt;
  vec2 sv = sad2(d);
  float flat_tax = 0.45 * (1.0 - smoothstep(0.03, 0.20, sv.y));
  float ap = dot(abs(texture(prevL, v_uv - d*0.5).rgb - texture(curL, v_uv - d*0.5).rgb), vec3(0.3333));
  float ac = dot(abs(texture(prevL, v_uv + d*0.5).rgb - texture(curL, v_uv + d*0.5).rgb), vec3(0.3333));
  return sv.x + 0.01*length(v) + flat_tax - 1.0*min(ap, ac);
}
void main(){
  vec4 c0 = texture(fieldT, v_uv);
  vec2 best = c0.xy;
  float bgap = c0.w;
  float bscore = pscore(best) - 0.8*min(c0.w, 1.0);
  for (int i = 0; i < 8; i++){
    vec2 off = (i==0) ? vec2( 1.0, 0.0) : (i==1) ? vec2(-1.0, 0.0)
             : (i==2) ? vec2(0.0,  1.0) : (i==3) ? vec2(0.0, -1.0)
             : (i==4) ? vec2( 2.0, 0.0) : (i==5) ? vec2(-2.0, 0.0)
             : (i==6) ? vec2(0.0,  2.0) : vec2(0.0, -2.0);
    vec4 cn = texture(fieldT, v_uv + off*pt);
    float score = pscore(cn.xy) - 0.8*min(cn.w, 1.0);
    if (score < bscore){ bscore = score; best = cn.xy; bgap = cn.w * 0.8; }
  }
  frag = vec4(best, max(pscore(best), 0.0), bgap);
}
"#;

// ---- edge-aware smoothing + confidence (cost, chaos, ambiguity) at 1/2 ----
//
// The naive gates (cost/chaos, inherited from the standard NeoFlow) kill
// confidence at exactly the pixels that matter: a small moving object's field
// is a lone island of vectors in a sea of zeros, so the chaos gate fires and
// the smoothing dilutes the island toward zero — the object then falls back to
// blend and effectively vanishes. The STRONG-MATCH OVERRIDE below rescues it:
// a match is trusted regardless of neighbourhood chaos when its cost is tiny
// AND there is real temporal activity at BOTH endpoint samples along the
// vector (the object left one spot and arrived at the other). A wire-fence
// alias cannot fake this — a static lattice has no temporal activity — so the
// override never resurrects the fence-crawl vectors.
const SIGMA_CONS_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D field; uniform sampler2D prevL; uniform sampler2D curL; uniform vec2 pt;
void main(){
  vec4 m0 = texture(field, v_uv);
  vec3 yc = texture(prevL, v_uv).rgb;
  vec2 acc = vec2(0.0); float wsum = 0.0; float chaos = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 uv = v_uv + vec2(dx,dy)*pt;
    vec4 m = texture(field, uv);
    vec3 yw = texture(prevL, uv).rgb;
    float edge_w = exp(-dot(abs(yw - yc), vec3(6.0)));
    float w = edge_w / (0.05 + m.z);
    acc += m.xy * w; wsum += w;
    chaos += length(m.xy - m0.xy);
  }
  vec2 vs = acc / max(wsum, 1e-5);
  float conf = 1.0 - smoothstep(0.55, 1.5, m0.z);        // bad match
  conf *= 1.0 - smoothstep(3.5, 10.0, chaos);            // field chaos
  // ambiguity: repetitive texture (wire fence) -> many equal matches -> gap≈0.
  // Do not kill motion entirely (fences scroll too) — just turn it cautious.
  conf *= mix(0.55, 1.0, smoothstep(0.015, 0.12, m0.w));
  // strong-match override: tiny cost + activity at both endpoints along d
  vec2 d = m0.xy * pt;
  float ap = dot(abs(texture(prevL, v_uv - d*0.5).rgb - texture(curL, v_uv - d*0.5).rgb), vec3(0.3333));
  float ac = dot(abs(texture(prevL, v_uv + d*0.5).rgb - texture(curL, v_uv + d*0.5).rgb), vec3(0.3333));
  float evidence = min(ap, ac);
  float anyEvidence = max(ap, ac);
  // Full-resolution RGB SAD is naturally larger than the half-resolution
  // Y/Co/Cg cost. The old 0.30 ceiling rejected a uniquely matched anime cel
  // (foot probe: cost .496, gap .131) after the pyramid had tracked it well.
  float strong = (1.0 - smoothstep(0.18, 0.85, m0.z)) * smoothstep(0.05, 0.16, evidence);
  // ambiguity veto: repetitive high-contrast patterns (zigzag rooflines,
  // fences) match themselves at many offsets with near-zero cost AND show
  // compression-noise "evidence" — the override then trusted a garbage vector
  // and smeared dark roof pixels into the sky as flickering marks. Require an
  // unambiguous match (real best-vs-runner-up gap) before overriding.
  strong *= smoothstep(0.03, 0.10, m0.w);
  conf = max(conf, strong * 0.95);
  // At a foreground/background boundary neighbouring vectors legitimately
  // disagree. A clearly separated best candidate should survive that chaos;
  // repetitive fences cannot pass because their runner-up gap is near zero.
  float unique = smoothstep(0.070, 0.145, m0.w)
               * (1.0 - smoothstep(0.50, 1.00, m0.z))
               * smoothstep(0.75, 2.50, length(m0.xy))
               * smoothstep(0.035, 0.120, anyEvidence);
  conf = max(conf, unique * 0.90);
  // where the match is strong, keep ITS vector — the edge-aware smoothing
  // dilutes a small object's lone island of vectors toward the background zero
  vec2 v = mix(vs, m0.xy, max(strong, unique));
  // Preserve the best-vs-runner-up gap for the final compositor. Repetitive
  // fences have a tiny gap; a uniquely tracked moving outline has a large one.
  frag = vec4(v, conf, m0.w);
}
"#;

// ---- transition classification mask ----
// R protects frame-wide cuts. G marks local appearance changes that must not
// be interpreted as motion: a sudden light/sign switch, or an A-B-A flash.
const CADENCE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prev2C; uniform sampler2D prevC; uniform sampler2D curC;
uniform vec2 pt;
float act3at(sampler2D a, sampler2D b, vec2 center){
  float s = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++){
    vec2 o = vec2(dx,dy)*pt;
    s += dot(abs(texture(a, center+o).rgb - texture(b, center+o).rgb), vec3(0.3333));
  }
  return s / 9.0;
}
// WIDE-window activity for the "was it held before?" question. A flat-shaded
// object gliding a few px per frame has zero POINT diff in its interior, so a
// local test misreads its advancing edge as a held cel and freezes it (the
// 60fps CG disc stopped dead in the anime stress test). A cel stepped on
// twos/threes is still over its WHOLE region, so demand stillness across a
// broad neighbourhood before granting the hold.
float act9(sampler2D a, sampler2D b){
  float m = 0.0;
  // ±8 full-res px: wide enough to see a small flat CG element's own moving
  // edges (its leading sliver is <8px), narrow enough that a held cel's
  // interior isn't contaminated by the panning background — the rim that IS
  // contaminated gets covered by the ±8px dilation of the hold mask.
  // MEAN (not max): real broadcast/stream anime carries mosquito noise that
  // spikes a max-statistic past any threshold and dissolved the hold on real
  // footage (hold_err 121-128 on the video bench); a true cel step still
  // dominates the mean, noise does not.
  for (int dy=-4; dy<=4; dy+=2)
  for (int dx=-4; dx<=4; dx+=2){
    vec2 o = vec2(dx,dy)*pt*2.0;
    m += dot(abs(texture(a, v_uv+o).rgb - texture(b, v_uv+o).rgb), vec3(0.3333));
  }
  return m / 25.0;
}
void main(){
  float past = act3at(prev2C, prevC, v_uv);
  float now = act3at(prevC, curC, v_uv);
  float returned = act3at(prev2C, curC, v_uv);
  // Static -> changed catches lights and signs switching state. A-B-A catches
  // alternating flashes even after the first transition. Motion that matches
  // spatially is allowed later by the correspondence gate in the composite.
  float sudden = smoothstep(0.008, 0.060, now)
               * (1.0 - smoothstep(0.020, 0.080, past));
  float reversal = smoothstep(0.015, 0.080, now)
                 * smoothstep(0.015, 0.080, past)
                 * (1.0 - smoothstep(0.010, 0.055, returned));
  float localAppearance = max(sudden, reversal);

  // B carries current-interval activity. The following dilation propagates
  // moving edges into flat object interiors; a truly repeated frame stays 0.
  frag = vec4(past, localAppearance, now, 1.0);
}
"#;

// Reduce the current-interval activity map to one screen-wide classification.
// R = full-frame cut, G = near-duplicate frame. This must be global: the old
// local +/-32px test mistook a large character fade for a scene cut, while a
// local max mistook compression noise in a repeated frame for real motion.
const GLOBAL_TRANSITION_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D activity;
void main(){
  float lo = 1.0;
  float hi = 0.0;
  float mean = 0.0;
  float pastHi = 0.0;
  float pastMean = 0.0;
  float changed = 0.0;
  for (int y=0; y<9; y++)
  for (int x=0; x<16; x++) {
    vec2 p = (vec2(x,y) + 0.5) / vec2(16.0,9.0);
    float a = texture(activity,p).b;
    float pa = texture(activity,p).r;
    lo = min(lo,a);
    hi = max(hi,a);
    mean += a;
    changed += step(0.10,a);
    pastHi = max(pastHi,pa);
    pastMean += pa;
  }
  mean /= 144.0;
  pastMean /= 144.0;
  float hardCut = smoothstep(0.14,0.28,lo) * smoothstep(0.18,0.34,mean);
  // Stream compression changes a handful of edge pixels even when the source
  // drawing is held.  Treat that low-energy residue as a repeat so the motion
  // search cannot animate codec noise. A genuinely moving small object still
  // raises either the screen mean or sampled local peak beyond these bounds.
  float repeated = (1.0 - smoothstep(0.0025,0.0120,mean))
                 * (1.0 - smoothstep(0.100,0.220,hi));
  float pastRepeated = (1.0 - smoothstep(0.0025,0.0120,pastMean))
                     * (1.0 - smoothstep(0.100,0.220,pastHi));
  // Broad incoherent camera/object transitions are where local block flow can
  // create scratches across otherwise smooth surfaces.  Motion coherence in
  // the composite still preserves a genuinely trackable pan; these thresholds
  // only make the fallback decisive when more than roughly half the screen
  // changes without a trustworthy common motion.
  float changedRatio = changed/144.0;
  float broadArea = smoothstep(0.40,0.58,changedRatio)
                  * smoothstep(0.06,0.15,mean);
  // A high-contrast effect can occupy only a quarter of the image (ink
  // splashes, silhouettes, rhythm-game notes) yet still be globally
  // untrackable.  Keep this signal separate from motion reliability: a
  // coherent camera pan is retained later, while an incoherent effect uses a
  // real endpoint instead of manufacturing detached shards.
  float broadIntensity = smoothstep(0.07,0.13,mean)
                       * smoothstep(0.14,0.25,changedRatio);
  float localizedIntensity = smoothstep(0.035,0.070,mean)
                           * smoothstep(0.10,0.18,changedRatio)
                           * smoothstep(0.25,0.50,hi);
  float broad = max(max(broadArea,broadIntensity),localizedIntensity);
  frag = vec4(hardCut,repeated,pastRepeated,broad);
}
"#;

// Reduce the final motion field to a frame-level reliability decision.
// R = coherent-flow reliability, G = fraction of the screen undergoing a
// broad change. Static overlays are ignored by weighting with activity.
const GLOBAL_MOTION_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D mot; uniform sampler2D activity;
void main(){
  vec2 meanV = vec2(0.0);
  float activeWeight = 0.0;
  float confidence = 0.0;
  float changed = 0.0;
  for (int y=0; y<9; y++)
  for (int x=0; x<16; x++) {
    vec2 p = (vec2(x,y)+0.5)/vec2(16.0,9.0);
    float a = smoothstep(0.035,0.14,texture(activity,p).b);
    vec4 m = texture(mot,p);
    meanV += m.xy*a;
    confidence += m.z*a;
    activeWeight += a;
    changed += step(0.32,a);
  }
  meanV /= max(activeWeight,0.001);
  confidence /= max(activeWeight,0.001);
  float variance = 0.0;
  for (int y=0; y<9; y++)
  for (int x=0; x<16; x++) {
    vec2 p = (vec2(x,y)+0.5)/vec2(16.0,9.0);
    float a = smoothstep(0.035,0.14,texture(activity,p).b);
    vec2 d = texture(mot,p).xy-meanV;
    variance += dot(d,d)*a;
  }
  variance /= max(activeWeight,0.001);
  float coherent = 1.0-smoothstep(2.5,8.0,sqrt(variance));
  float reliable = smoothstep(0.28,0.62,confidence)*coherent;
  float broad = smoothstep(0.18,0.52,changed/144.0);
  frag = vec4(reliable,broad,0.0,1.0);
}
"#;

// Grow both masks so anti-aliased rims and glow around a transition are covered.
const DILATE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D tex; uniform vec2 pt;
void main(){
  vec4 m = vec4(0.0);
  // Sparse 25px-radius growth at half resolution. Held-cel motion is visible
  // at the outline but its flat interior has no matching evidence; carrying
  // the transition class inward prevents a raw 50/50 double image there.
  for (int dy=-12; dy<=12; dy+=2)
  for (int dx=-12; dx<=12; dx+=2)
    m = max(m, texture(tex, v_uv + vec2(dx,dy)*pt));
  frag = m;
}
"#;

const ZERO_FRAG: &str = "#version 330
in vec2 v_uv; out vec4 frag;
void main(){ frag = vec4(0.0, 0.0, 1.0, 1.0); }
";

const GLOBAL_ZERO_FRAG: &str = "#version 330
in vec2 v_uv; out vec4 frag;
void main(){ frag = vec4(0.0); }
";

// ---- full-res composite: median-of-3 + TextFX guard ----
const SIGMA_COMPOSITE_FRAG: &str = r#"#version 330
in vec2 v_uv; out vec4 frag;
uniform sampler2D prevC; uniform sampler2D curC; uniform sampler2D mot;
uniform sampler2D cad; uniform sampler2D globalCad; uniform sampler2D globalMotion;
uniform float t; uniform vec2 px; uniform float motScale;

// NEOFLOW_TEXTFX_GUARD_BEGIN (shared design with the standard NeoFlow)
float luma_nf(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float edge_luma_nf(sampler2D tex, vec2 uv, vec2 p){
  float l = luma_nf(texture(tex, uv - vec2(p.x, 0.0)).rgb);
  float r = luma_nf(texture(tex, uv + vec2(p.x, 0.0)).rgb);
  float u = luma_nf(texture(tex, uv - vec2(0.0, p.y)).rgb);
  float d = luma_nf(texture(tex, uv + vec2(0.0, p.y)).rgb);
  return max(abs(r - l), abs(d - u));
}
// NEOFLOW_TEXTFX_GUARD_END

vec3 median3(vec3 a, vec3 b, vec3 c){
  return max(min(a, b), min(max(a, b), c));
}

void main(){
  vec4 m = texture(mot, v_uv);
  vec2 v = m.xy * motScale * px;           // 1/2 texels -> full-res uv offset
  // symmetric field: full displacement d, prev sampled at −d·t, cur at +d·(1−t)
  vec3 p0 = texture(prevC, v_uv - v * t).rgb;
  vec3 p1 = texture(curC,  v_uv + v * (1.0 - t)).rgb;
  float err = dot(abs(p0 - p1), vec3(1.0/3.0));
  float conf = m.z;
  conf *= 1.0 - smoothstep(0.06, 0.24, err);

  vec3 raw0 = texture(prevC, v_uv).rgb;
  vec3 raw1 = texture(curC,  v_uv).rgb;
  vec3 blendRaw = mix(raw0, raw1, t);
  float e0 = max(edge_luma_nf(prevC, v_uv, px), edge_luma_nf(prevC, v_uv, px*2.0));
  float e1 = max(edge_luma_nf(curC,  v_uv, px), edge_luma_nf(curC,  v_uv, px*2.0));
  float edge = max(e0,e1);
  float persistentEdge = min(e0,e1);
  float frameDiff = length(raw0-raw1);

  // A confident vector in an almost-static region can still introduce a tiny
  // crawl. Require evidence that content actually left/arrived along the
  // vector. This keeps true moving objects active while static detail falls
  // back to the exact temporal blend.
  float motionActivity = max(length(raw0 - p0), length(raw1 - p1));
  conf *= smoothstep(0.008, 0.050, motionActivity);

  // A low matching cost is not sufficient by itself around occlusion edges:
  // isolated vectors can latch onto another piece of a repeated cel and paint
  // small fragments between the two poses.  Require the local field to agree
  // before treating a match as a real, uniquely tracked midpoint.
  float vectorSupport = 0.0;
  float vectorSamples = 0.0;
  float maxVectorDeviation = 0.0;
  for (int dy = -1; dy <= 1; dy++)
  for (int dx = -1; dx <= 1; dx++) {
    vec2 nearby = texture(mot, v_uv + vec2(dx, dy) * px * motScale).xy;
    float deviation = length(nearby-m.xy);
    vectorSupport += 1.0 - smoothstep(1.5,4.5,deviation);
    maxVectorDeviation = max(maxVectorDeviation,deviation);
    vectorSamples += 1.0;
  }
  // Robust neighbourhood vote. An occlusion sliver is supported by only a few
  // pixels, while a real moving surface has a local majority even when one side
  // of the window crosses its boundary. This keeps object interiors moving
  // without accepting every isolated low-cost vector.
  for (int r = 3; r <= 6; r += 3) {
    vec2 a = texture(mot, v_uv + vec2( r,0) * px * motScale).xy;
    vec2 b = texture(mot, v_uv + vec2(-r,0) * px * motScale).xy;
    vec2 c = texture(mot, v_uv + vec2(0, r) * px * motScale).xy;
    vec2 d = texture(mot, v_uv + vec2(0,-r) * px * motScale).xy;
    vectorSupport += 1.0 - smoothstep(1.5,4.5,length(a-m.xy));
    vectorSupport += 1.0 - smoothstep(1.5,4.5,length(b-m.xy));
    vectorSupport += 1.0 - smoothstep(1.5,4.5,length(c-m.xy));
    vectorSupport += 1.0 - smoothstep(1.5,4.5,length(d-m.xy));
    maxVectorDeviation = max(maxVectorDeviation,length(a-m.xy));
    maxVectorDeviation = max(maxVectorDeviation,length(b-m.xy));
    maxVectorDeviation = max(maxVectorDeviation,length(c-m.xy));
    maxVectorDeviation = max(maxVectorDeviation,length(d-m.xy));
    vectorSamples += 4.0;
  }
  float supportRatio = vectorSupport / vectorSamples;
  float majorityCoherence = smoothstep(0.36,0.72,supportRatio);
  float strictCoherence = 1.0 - smoothstep(1.5,5.0,maxVectorDeviation);
  float boundaryRisk = smoothstep(0.045,0.14,edge)
                     * smoothstep(0.045,0.18,frameDiff);
  float stateChangeRisk = texture(cad,v_uv).g;
  float motionCoherence = mix(majorityCoherence,strictCoherence,
                              max(boundaryRisk,stateChangeRisk));
  // The confidence estimator can be fooled by a small island of mutually
  // inconsistent vectors, especially on blinking gradients and occlusion
  // boundaries.  Previously coherence only selected the direct-warp path;
  // the ordinary flow path could still paint those islands as scratches.
  conf *= motionCoherence;
  float uniqueWarp = smoothstep(0.45, 2.20, length(m.xy))
                   * smoothstep(0.48, 0.86, m.z)
                   * smoothstep(0.075, 0.20, m.w)
                   * (1.0 - smoothstep(0.028, 0.095, err))
                   * motionCoherence;
  // Occlusion-tolerant default: median drops one bad side. For a unique,
  // confident correspondence use the actual bidirectionally warped midpoint;
  // keeping raw temporal blend in that median caused visible double feet.
  vec3 warpedMid = mix(p0, p1, t);
  vec3 robustMid = median3(warpedMid, mix(p0, p1, step(0.5, t) * 1.0), blendRaw);
  vec3 flowTerm = mix(robustMid, warpedMid, uniqueWarp);
  // (mix(p0,p1,step) = the nearer-time warped side; keeps thin objects crisp)

  // NEOFLOW_TEXTFX_GUARD_BEGIN
  float confMin = conf;
  for (int dy = -1; dy <= 1; dy++)
  for (int dx = -1; dx <= 1; dx++)
    confMin = min(confMin, texture(mot, v_uv + vec2(dx, dy) * px * motScale).z);
  conf = mix(conf, confMin, 0.12);

  // Distinguish a stationary shape changing brightness from real translation.
  // A blinking shaded sphere keeps the direction of its local gradient even
  // while its luminance changes substantially. Treating that as motion made
  // the search align unrelated points on the gradient and produced dark spots.
  float pL = luma_nf(texture(prevC, v_uv - vec2(px.x,0.0)).rgb);
  float pR = luma_nf(texture(prevC, v_uv + vec2(px.x,0.0)).rgb);
  float pU = luma_nf(texture(prevC, v_uv - vec2(0.0,px.y)).rgb);
  float pD = luma_nf(texture(prevC, v_uv + vec2(0.0,px.y)).rgb);
  float cL = luma_nf(texture(curC,  v_uv - vec2(px.x,0.0)).rgb);
  float cR = luma_nf(texture(curC,  v_uv + vec2(px.x,0.0)).rgb);
  float cU = luma_nf(texture(curC,  v_uv - vec2(0.0,px.y)).rgb);
  float cD = luma_nf(texture(curC,  v_uv + vec2(0.0,px.y)).rgb);
  vec2 grad0 = vec2(pR-pL,pD-pU);
  vec2 grad1 = vec2(cR-cL,cD-cU);
  float gm0 = length(grad0), gm1 = length(grad1);
  float gradientAlignment = dot(grad0,grad1) / (gm0*gm1 + 0.00005);
  float gradientMagnitudeAgreement = min(gm0,gm1) / (max(gm0,gm1) + 0.00005);
  float stationaryStructure = smoothstep(0.72,0.94,gradientAlignment)
                            * smoothstep(0.010,0.045,min(gm0,gm1))
                            * smoothstep(0.18,0.62,gradientMagnitudeAgreement);

  float edgeRisk    = smoothstep(0.05, 0.16, edge);
  float persistRisk = smoothstep(0.035, 0.12, persistentEdge);
  float diffRisk    = smoothstep(0.08, 0.25, frameDiff);
  float lowConfRisk = 1.0 - conf;

  float textFxRisk = 0.0;
  textFxRisk = max(textFxRisk, edgeRisk * diffRisk);
  textFxRisk = max(textFxRisk, persistRisk * 0.75);
  textFxRisk = max(textFxRisk, edgeRisk * lowConfRisk);
  textFxRisk = clamp(textFxRisk, 0.0, 1.0);
  float historyAppearance = texture(cad, v_uv).g;
  // A held shape whose edge remains at the same screen position is changing
  // state (lamp, digit, glow), not translating. Do not let an accidentally
  // unique match override that temporal evidence.
  float samePlaceStateChange = historyAppearance * persistRisk;
  float uniqueTrackedMotion = uniqueWarp
                            * (1.0 - stationaryStructure)
                            * (1.0 - samePlaceStateChange);
  conf *= 1.0 - max(stationaryStructure * 0.96,
                    samePlaceStateChange * 0.98);
  textFxRisk *= 1.0 - uniqueTrackedMotion * 0.72;
  // Σ trusts its symmetric field more than the standard NeoFlow does: risk
  // only softens confidence (0.55) instead of nearly zeroing it (0.85) —
  // otherwise every moving object EDGE counts as "text" and motion dies.
  conf *= 1.0 - textFxRisk * 0.55;
  // ...EXCEPT persistent same-place edges (wire fences, logos, subtitles, UI):
  // a repetitive lattice matches itself at its period, so the err gate cannot
  // catch the alias — kill flow hard there; the temporal-blend fallback keeps
  // a static lattice pixel-perfect.
  conf *= 1.0 - persistRisk * mix(0.90, 0.20, uniqueTrackedMotion);
  // Local flashing and texture/state replacement have no valid intermediate
  // geometry. If warping cannot make the endpoints agree, use the temporally
  // nearest real frame instead of stretching fragments of both states. The
  // correspondence requirement keeps ordinary moving objects interpolated.
  float correspondenceFailure = smoothstep(0.035, 0.120, err);
  float unmatchedChange = smoothstep(0.035, 0.16, frameDiff)
                        * correspondenceFailure
                        * (1.0 - smoothstep(0.35, 0.80, m.z));
  // Held cel -> translated cel has the same temporal pattern as a flash
  // (static then change). A unique, confident correspondence is the deciding
  // evidence: interpolate that motion; keep ambiguous changes protected.
  float historyRisk = historyAppearance * (1.0 - uniqueTrackedMotion);
  float appearanceRisk = clamp(max(historyRisk, unmatchedChange), 0.0, 1.0);
  conf *= 1.0 - appearanceRisk;
  // Never carry a vector from prev2 into a repeated prev==cur interval. B is
  // a dilated current-interval activity mask, so flat interiors of genuinely
  // moving objects inherit activity from their moving boundary while an
  // unchanged foot/UI/frame remains exactly unchanged.
  float currentActivity = texture(cad, v_uv).b;
  conf *= smoothstep(0.004, 0.040, currentActivity);
  // This is a frame-level decision, not a per-pixel blend weight.  A partial
  // value leaves a faint fraction of a bogus flow field visible on held
  // drawings, so commit once the global classifier crosses its midpoint.
  float repeatedFrame = step(0.5, texture(globalCad, vec2(0.5)).g);
  conf *= 1.0 - repeatedFrame;
  // Limited animation frequently holds a drawing for two frames and then
  // moves it a long distance. In that interval, only a very strong coherent
  // correspondence is safe; partial matches create the characteristic dirty
  // fragments between the two poses.
  float heldStep = texture(globalCad, vec2(0.5)).b * (1.0 - repeatedFrame);
  float heldStepTrust = smoothstep(0.68, 0.94, uniqueTrackedMotion);
  conf *= mix(1.0, heldStepTrust, heldStep * historyAppearance);
  // A spatially coherent signed luminance change is a genuine fade/lighting
  // transition. It must avoid motion warping, but a temporal blend is correct.
  // Mixed-sign changes indicate digits, textures, particles, or shapes being
  // replaced; those use a hard nearest-frame switch to avoid residual pieces.
  float signedChange = 0.0;
  float absoluteChange = 0.0;
  float prevMean = 0.0;
  float curMean = 0.0;
  for (int dy=-1; dy<=1; dy++)
  for (int dx=-1; dx<=1; dx++) {
    vec2 q = v_uv + vec2(dx,dy) * px * 2.0;
    float yp = luma_nf(texture(prevC,q).rgb);
    float yc = luma_nf(texture(curC,q).rgb);
    float d = yc - yp;
    signedChange += d;
    absoluteChange += abs(d);
    prevMean += yp;
    curMean += yc;
  }
  float fadeCoherence = abs(signedChange) / (absoluteChange + 0.0001);
  prevMean /= 9.0;
  curMean /= 9.0;
  float darker = min(prevMean, curMean);
  float brighter = max(prevMean, curMean);
  float darkEmergence = (1.0 - smoothstep(0.018, 0.075, darker))
                      * smoothstep(0.045, 0.14, brighter);
  float replacementRisk = appearanceRisk
                        * max(historyAppearance,
                              1.0 - smoothstep(0.72, 0.94, fadeCoherence))
                        * (1.0 - darkEmergence);
  replacementRisk = max(replacementRisk,
                        heldStep * historyAppearance * (1.0 - heldStepTrust));
  // BLEND floor: on the real-footage bench (Neoflow-test01..03) the nearest-
  // frame fallback scored BELOW a plain 50/50 blend wherever the field was
  // uncertain and can produce melted ghosting. Falling back to the
  // temporal blend makes the WORST case equal to a dumb blend while confident
  // pixels still get true motion.
  vec3 nearestRaw = mix(raw0, raw1, step(0.5001, t));
  vec3 fallback = mix(blendRaw, nearestRaw, replacementRisk);
  // NEOFLOW_TEXTFX_GUARD_END

  vec3 outc = mix(fallback, flowTerm, conf);
  float hardCut = texture(globalCad, vec2(0.5)).r;
  outc = mix(outc, blendRaw, repeatedFrame);
  vec2 globalMotionState = texture(globalMotion,vec2(0.5)).rg;
  float broadUnreliable = texture(globalCad,vec2(0.5)).a
                        * (1.0-globalMotionState.x);
  // On a broad incoherent transition there is no defensible synthetic shape.
  // Keep one real endpoint instead of exposing fragments from many objects.
  outc = mix(outc,nearestRaw,broadUnreliable);
  frag = vec4(mix(outc, raw0, hardCut), 1.0);
}
"#;

/// Synthesize the frame at time `t` (0..1) between `prev` and `cur`.
pub fn interpolate(gc: &mut GlContext, prev: GpuTex, cur: GpuTex, t: f32) -> Result<GpuTex> {
    interpolate3(gc, None, prev, cur, t)
}

/// Three-frame synthesis. `prev2` protects broad transitions without adding
/// latency because it is already in the past.
pub fn interpolate3(
    gc: &mut GlContext,
    prev2: Option<GpuTex>,
    prev: GpuTex,
    cur: GpuTex,
    t: f32,
) -> Result<GpuTex> {
    let (w, h) = (prev.w(), prev.h());
    if cur.w() != w || cur.h() != h {
        return Err(anyhow!("NeoFlow: frame size changed"));
    }
    let (w2, h2) = ((w + 1) / 2, (h + 1) / 2);
    let (w4, h4) = ((w + 3) / 4, (h + 3) / 4);
    let (w8, h8) = ((w + 7) / 8, (h + 7) / 8);

    let luma = gc.program(LUMA_FRAG).map_err(|e| anyhow!(e))?;
    let down = gc.program(DOWN_FRAG).map_err(|e| anyhow!(e))?;
    let wide = gc.program(SYM16_FRAG).map_err(|e| anyhow!(e))?; // wide ±12 step 2 (used at 1/8)
    let refine = gc.program(SYM_REFINE_FRAG).map_err(|e| anyhow!(e))?;
    let prop = gc.program(PROP_FRAG).map_err(|e| anyhow!(e))?;
    let cons = gc.program(SIGMA_CONS_FRAG).map_err(|e| anyhow!(e))?;
    let comp = gc.program(SIGMA_COMPOSITE_FRAG).map_err(|e| anyhow!(e))?;
    let cadence = gc.program(CADENCE_FRAG).map_err(|e| anyhow!(e))?;
    let global_transition = gc.program(GLOBAL_TRANSITION_FRAG).map_err(|e| anyhow!(e))?;
    let global_motion_program = gc.program(GLOBAL_MOTION_FRAG).map_err(|e| anyhow!(e))?;
    let dilate = gc.program(DILATE_FRAG).map_err(|e| anyhow!(e))?;
    let zero = gc.program(ZERO_FRAG).map_err(|e| anyhow!(e))?;
    let global_zero = gc.program(GLOBAL_ZERO_FRAG).map_err(|e| anyhow!(e))?;
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            log::info!(
                "NeoFlow: unified YCoCg motion pipeline ready (scene-cut guard + median composite + TextFX guard)"
            )
        });
    }

    gc.set_filter_linear(prev, true);
    gc.set_filter_linear(cur, true);

    // ---- broad-transition mask (needs the frame before `prev`) ----
    let (wm, hm) = (w2, h2);
    let (cad, global_cad) = match prev2 {
        Some(p2f) => {
            gc.set_filter_linear(p2f, true);
            let raw_mask = run_pass(
                gc,
                cadence,
                &[("prev2C", p2f), ("prevC", prev), ("curC", cur)],
                &[],
                &[("pt", (1.0 / w as f32, 1.0 / h as f32))],
                wm,
                hm,
                4,
            );
            gc.set_filter_linear(raw_mask, true);
            let global = run_pass(
                gc,
                global_transition,
                &[("activity", raw_mask)],
                &[],
                &[],
                1,
                1,
                4,
            );
            gc.set_filter_linear(global, true);
            if std::env::var_os("NEOFLOW_GLOBAL_PROBE").is_some() {
                let values = gc.download_f32(raw_mask);
                let mut mean = 0.0f32;
                let mut changed = 0usize;
                let mut high = 0.0f32;
                for y in 0..9usize {
                    for x in 0..16usize {
                        let sx =
                            (((x as f32 + 0.5) * wm as f32 / 16.0) as usize).min(wm as usize - 1);
                        let sy =
                            (((y as f32 + 0.5) * hm as f32 / 9.0) as usize).min(hm as usize - 1);
                        let now = values[(sy * wm as usize + sx) * 4 + 2];
                        mean += now;
                        changed += usize::from(now >= 0.10);
                        high = high.max(now);
                    }
                }
                eprintln!(
                    "NEOFLOW global-raw: mean={:.4} changed={:.4} high={:.4}",
                    mean / 144.0,
                    changed as f32 / 144.0,
                    high
                );
            }
            let grown = run_pass(
                gc,
                dilate,
                &[("tex", raw_mask)],
                &[],
                &[("pt", (1.0 / wm as f32, 1.0 / hm as f32))],
                wm,
                hm,
                4,
            );
            gc.set_filter_linear(p2f, false);
            gc.set_filter_linear(raw_mask, false);
            gc.recycle(raw_mask);
            gc.set_filter_linear(grown, true);
            (grown, global)
        }
        // no history yet: local activity=1 and no global cut/repeat decision
        None => (
            run_pass(gc, zero, &[], &[], &[], 4, 4, 4),
            run_pass(gc, global_zero, &[], &[], &[], 1, 1, 4),
        ),
    };
    if std::env::var_os("NEOFLOW_GLOBAL_PROBE").is_some() {
        let v = gc.download_f32(global_cad);
        eprintln!(
            "NEOFLOW global-cadence: cut={:.4} repeat={:.4} past-repeat={:.4} broad={:.4}",
            v[0], v[1], v[2], v[3]
        );
    }

    // luma pyramid: 1/2 -> 1/4 -> 1/8 (a 1/16 level proved useless: its 16px
    // quantization made TRUE matches cost more than flat-background ghosts)
    let p2 = run_pass(gc, luma, &[("tex", prev)], &[], &[], w2, h2, 4);
    let c2 = run_pass(gc, luma, &[("tex", cur)], &[], &[], w2, h2, 4);
    gc.set_filter_linear(p2, true);
    gc.set_filter_linear(c2, true);
    let p4 = run_pass(gc, down, &[("tex", p2)], &[], &[], w4, h4, 4);
    let c4 = run_pass(gc, down, &[("tex", c2)], &[], &[], w4, h4, 4);
    gc.set_filter_linear(p4, true);
    gc.set_filter_linear(c4, true);
    let p8 = run_pass(gc, down, &[("tex", p4)], &[], &[], w8, h8, 4);
    let c8 = run_pass(gc, down, &[("tex", c4)], &[], &[], w8, h8, 4);
    gc.set_filter_linear(p8, true);
    gc.set_filter_linear(c8, true);
    let pt8 = (1.0 / w8 as f32, 1.0 / h8 as f32);
    let pt4 = (1.0 / w4 as f32, 1.0 / h4 as f32);
    let pt2 = (1.0 / w2 as f32, 1.0 / h2 as f32);
    let pt1 = (1.0 / w as f32, 1.0 / h as f32);

    // wide symmetric search at 1/8: ±12 texels = ±96px full displacement
    let f8 = run_pass(
        gc,
        wide,
        &[("prevL", p8), ("curL", c8)],
        &[],
        &[("pt", pt8)],
        w8,
        h8,
        4,
    );
    gc.set_filter_linear(f8, true);
    // support-weighted propagation: edge-resolved vectors flood ambiguous
    // object interiors (two rounds = ±4 texels = ±32px reach at 1/8)
    let g8 = run_pass(
        gc,
        prop,
        &[("fieldT", f8), ("prevL", p8), ("curL", c8)],
        &[],
        &[("pt", pt8)],
        w8,
        h8,
        4,
    );
    gc.set_filter_linear(g8, true);
    let g8b = run_pass(
        gc,
        prop,
        &[("fieldT", g8), ("prevL", p8), ("curL", c8)],
        &[],
        &[("pt", pt8)],
        w8,
        h8,
        4,
    );
    gc.set_filter_linear(g8b, true);
    let f4 = run_pass(
        gc,
        refine,
        &[("prevL", p4), ("curL", c4), ("init", g8b)],
        &[("jitter", 1.0), ("allowZero", 1.0)],
        &[("pt", pt4), ("ipt", pt8)],
        w4,
        h4,
        4,
    );
    gc.set_filter_linear(f4, true);
    let g4 = run_pass(
        gc,
        prop,
        &[("fieldT", f4), ("prevL", p4), ("curL", c4)],
        &[],
        &[("pt", pt4)],
        w4,
        h4,
        4,
    );
    gc.set_filter_linear(g4, true);
    let f2 = run_pass(
        gc,
        refine,
        &[("prevL", p2), ("curL", c2), ("init", g4)],
        &[("jitter", 0.5), ("allowZero", 1.0)],
        &[("pt", pt2), ("ipt", pt4)],
        w2,
        h2,
        4,
    );
    gc.set_filter_linear(f2, true);
    // debug probe: NEOFLOW_SIGMA_PROBE="x,y" (full-res px)
    if let Ok(spec) = std::env::var("NEOFLOW_SIGMA_PROBE") {
        if let Some((sx, sy)) = spec.split_once(',') {
            if let (Ok(px_), Ok(py_)) = (sx.trim().parse::<i32>(), sy.trim().parse::<i32>()) {
                let dump = |gc: &mut GlContext, tex: GpuTex, level: i32, tag: &str| {
                    let v = gc.download_f32(tex);
                    let (tw, th) = (tex.w(), tex.h());
                    let tx = (px_ / level).clamp(0, tw - 1);
                    let ty = (py_ / level).clamp(0, th - 1);
                    let i = ((ty * tw + tx) * 4) as usize;
                    eprintln!(
                        "SIGMA {tag} @L{level}({tx},{ty}): v=({:.2},{:.2}) z={:.3} w={:.3}",
                        v[i],
                        v[i + 1],
                        v[i + 2],
                        v[i + 3]
                    );
                };
                dump(gc, p8, 8, "p8 ");
                dump(gc, c8, 8, "c8 ");
                dump(gc, f8, 8, "f8 ");
                dump(gc, g8b, 8, "g8b");
                dump(gc, f4, 4, "f4 ");
                dump(gc, f2, 2, "f2 ");
            }
        }
    }
    // Full-resolution sub-pixel refinement. The previous pipeline stopped at
    // 1/2 resolution, so small lights and thin low-resolution outlines shared
    // a vector with their background. Two GPU-only passes sharpen that field
    // without adding CPU synchronization or buffered frames.
    let f1 = run_pass(
        gc,
        refine,
        &[("prevL", prev), ("curL", cur), ("init", f2)],
        &[("jitter", 0.5), ("allowZero", 0.0)],
        &[("pt", pt1), ("ipt", pt2)],
        w,
        h,
        4,
    );
    gc.set_filter_linear(f1, true);
    let mot = run_pass(
        gc,
        cons,
        &[("field", f1), ("prevL", prev), ("curL", cur)],
        &[],
        &[("pt", pt1)],
        w,
        h,
        4,
    );
    gc.set_filter_linear(mot, true);
    let global_motion = run_pass(
        gc,
        global_motion_program,
        &[("mot", mot), ("activity", cad)],
        &[],
        &[],
        1,
        1,
        2,
    );
    gc.set_filter_linear(global_motion, true);
    if std::env::var_os("NEOFLOW_GLOBAL_PROBE").is_some() {
        let v = gc.download_f32(global_motion);
        eprintln!(
            "NEOFLOW global-motion: reliable={:.4} broad={:.4}",
            v[0], v[1]
        );
    }
    if let Ok(spec) = std::env::var("NEOFLOW_SIGMA_PROBE") {
        if let Some((sx, sy)) = spec.split_once(',') {
            if let (Ok(px_), Ok(py_)) = (sx.trim().parse::<i32>(), sy.trim().parse::<i32>()) {
                let dump = |gc: &mut GlContext, tex: GpuTex, level: i32, tag: &str| {
                    let v = gc.download_f32(tex);
                    let (tw, th) = (tex.w(), tex.h());
                    let tx = (px_ / level).clamp(0, tw - 1);
                    let ty = (py_ / level).clamp(0, th - 1);
                    let i = ((ty * tw + tx) * 4) as usize;
                    eprintln!(
                        "SIGMA {tag} @L{level}({tx},{ty}): v=({:.2},{:.2}) z={:.3} w={:.3}",
                        v[i],
                        v[i + 1],
                        v[i + 2],
                        v[i + 3]
                    );
                };
                dump(gc, f1, 1, "f1 ");
                dump(gc, mot, 1, "mot");
                // CPU cost-landscape dump at the probe texel (1/8 level):
                // brute-force the wide-search cost for every step-1 candidate
                // and print the top-8 — ground truth for tuning the shader.
                let pv = gc.download_f32(p8);
                let cv = gc.download_f32(c8);
                let (tw, th) = (p8.w(), p8.h());
                let sample = |v: &Vec<f32>, x: f32, y: f32| -> [f32; 3] {
                    let xf = x.clamp(0.0, (tw - 1) as f32);
                    let yf = y.clamp(0.0, (th - 1) as f32);
                    let (x0, y0) = (xf.floor() as i32, yf.floor() as i32);
                    let (x1, y1) = ((x0 + 1).min(tw - 1), (y0 + 1).min(th - 1));
                    let (fx, fy) = (xf - x0 as f32, yf - y0 as f32);
                    let g = |xx: i32, yy: i32, c: usize| v[((yy * tw + xx) * 4) as usize + c];
                    let mut out = [0f32; 3];
                    for (c, o) in out.iter_mut().enumerate() {
                        let a = g(x0, y0, c) * (1.0 - fx) + g(x1, y0, c) * fx;
                        let b = g(x0, y1, c) * (1.0 - fx) + g(x1, y1, c) * fx;
                        *o = a * (1.0 - fy) + b * fy;
                    }
                    out
                };
                let (bx, by) = ((px_ as f32) / 8.0, (py_ as f32) / 8.0);
                let cost_at = |vx: f32, vy: f32| -> (f32, f32, f32) {
                    let (ax, ay) = (bx - vx * 0.5, by - vy * 0.5);
                    let (gx, gy) = (bx + vx * 0.5, by + vy * 0.5);
                    let ac = sample(&pv, ax, ay);
                    let bc = sample(&cv, gx, gy);
                    let mut s = 0.0f32;
                    let mut st = 0.0f32;
                    for dy in -2..=2 {
                        for dx in -2..=2 {
                            let a = sample(&pv, ax + dx as f32, ay + dy as f32);
                            let b = sample(&cv, gx + dx as f32, gy + dy as f32);
                            for c in 0..3 {
                                s += (a[c] - b[c]).abs() * 0.12;
                                st += ((a[c] - ac[c]).abs() + (b[c] - bc[c]).abs()) * 0.12;
                            }
                        }
                    }
                    let pa = sample(&pv, ax, ay);
                    let ca = sample(&cv, ax, ay);
                    let pb = sample(&pv, gx, gy);
                    let cb = sample(&cv, gx, gy);
                    let ap: f32 = pa
                        .iter()
                        .zip(ca.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / 3.0;
                    let bp: f32 = pb
                        .iter()
                        .zip(cb.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>()
                        / 3.0;
                    let act = ap.min(bp);
                    let flat = 0.45 * (1.0 - ((st - 0.03) / 0.17).clamp(0.0, 1.0));
                    let len = (vx * vx + vy * vy).sqrt();
                    (s + 0.012 * len + flat - 0.3 * act, s, st)
                };
                let mut all: Vec<(f32, i32, i32, f32, f32)> = Vec::new();
                for vy in -12..=12 {
                    for vx in -12..=12 {
                        let (c, s_, st_) = cost_at(vx as f32, vy as f32);
                        all.push((c, vx, vy, s_, st_));
                    }
                }
                all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                for (c, vx, vy, s_, st_) in all.iter().take(8) {
                    eprintln!("SIGMA cpu-top: v=({vx},{vy}) cost={c:.3} sad={s_:.3} st={st_:.3}");
                }
                let (c, s_, st_) = cost_at(9.5, -0.9);
                eprintln!("SIGMA cpu-true: v=(9.5,-0.9) cost={c:.3} sad={s_:.3} st={st_:.3}");
            }
        }
    }
    let out = run_pass(
        gc,
        comp,
        &[
            ("prevC", prev),
            ("curC", cur),
            ("mot", mot),
            ("cad", cad),
            ("globalCad", global_cad),
            ("globalMotion", global_motion),
        ],
        &[("t", t), ("motScale", 1.0)],
        &[("px", (1.0 / w as f32, 1.0 / h as f32))],
        w,
        h,
        4,
    );

    for tex in [
        prev,
        cur,
        p2,
        c2,
        p4,
        c4,
        p8,
        c8,
        f8,
        g8,
        g8b,
        f4,
        g4,
        f2,
        f1,
        mot,
        cad,
        global_cad,
        global_motion,
    ] {
        gc.set_filter_linear(tex, false);
    }
    for tex in [
        p2,
        c2,
        p4,
        c4,
        p8,
        c8,
        f8,
        g8,
        g8b,
        f4,
        g4,
        f2,
        f1,
        mot,
        cad,
        global_cad,
        global_motion,
    ] {
        gc.recycle(tex);
    }
    Ok(out)
}
