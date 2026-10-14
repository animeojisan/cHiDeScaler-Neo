// Preset: aliens (official cinegrain2 parameters)
// Based on https://github.com/mr-berndt/cinegrain2 ; original shader order preserved.
// cinegrain2 - realistic film grain for mpv
// https://github.com/mr-berndt/cinegrain2
//
// Grain model: white noise, a separable Gaussian blur that sets the grain
// size, and multiplicative application on the picture, as film density
// works. A five-zone equalizer shapes the grain strength over luma:
//   Black 0.00, Shadow 0.13, Mid 0.30, High 0.60, White 1.00
// with smoothstep blending between neighbouring zones.
//
// All parameters are shader PARAMs and are set live by cinegrain2-control.lua
// through glsl-shader-opts; the values below are only the start-up defaults.
// Grain size is calibrated for 2160p output and scales with the output
// height, so it looks the same at 1080p and 4K.

//!PARAM LEVEL
//!DESC Grain amplitude
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 5.0
0.240

//!PARAM GAIN_1
//!DESC EQ zone Black (luma 0.00): -1 = no grain, 0 = flat, +3 = four times
//!TYPE float
//!MINIMUM -3.0
//!MAXIMUM 3.0
3.000

//!PARAM GAIN_2
//!DESC EQ zone Shadow (luma 0.13)
//!TYPE float
//!MINIMUM -3.0
//!MAXIMUM 3.0
0.900

//!PARAM GAIN_3
//!DESC EQ zone Mid (luma 0.30)
//!TYPE float
//!MINIMUM -3.0
//!MAXIMUM 3.0
-0.150

//!PARAM GAIN_4
//!DESC EQ zone High (luma 0.60)
//!TYPE float
//!MINIMUM -3.0
//!MAXIMUM 3.0
-0.450

//!PARAM GAIN_5
//!DESC EQ zone White (luma 1.00)
//!TYPE float
//!MINIMUM -3.0
//!MAXIMUM 3.0
-0.950

//!PARAM GRAIN_SIZE
//!DESC Gaussian blur sigma (0=35mm white noise, ~2.5=8mm heavy clumps)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 3.0
0.500

//!PARAM CHROMA
//!DESC Chroma grain strength (set by the control script)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.05

//!PARAM DEMO
//!DESC Test picture: 0 = off, 0.5 = 50 % grey card, 1 = grey staircase
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM SOFTBLUR
//!DESC Grain-field softening radius (pixels). 0 = off (zero cost).
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 10.0
0.800

//!HOOK OUTPUT
//!SAVE GRAIN_H
//!COMPONENTS 3
//!WIDTH HOOKED.w
//!HEIGHT HOOKED.h
//!DESC cinegrain2 grain h-blur

#define MAX_R 3
#define MIN_SIGMA 0.001

// Irwin-Hall with N=4. Sum of 4 uniform[0,1] hash draws has mean=2 and
// variance=1/3, so (sum - 2) has mean=0, std≈0.577 directly — no extra
// scaling needed (matches downstream pipeline's expected std). Reduced
// from N=12 for laptop-class GPUs: 3× less hash ALU in Pass 1. Side
// benefit: shorter Gaussian tails than N=12 → fewer 3σ+ extreme events
// → fewer bright spikes on dark pixels under density-domain exp().
float hash_noise(vec2 p, float seed)
{
    float sum = 0.0;
    for (int i = 0; i < 4; i++) {
        float fi = float(i);
        vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) + seed + fi * 7.31);
        p3 += dot(p3, p3.yzx + 33.33);
        sum += fract((p3.x + p3.y) * p3.z);
    }
    return sum - 2.0;
}

// Resolution-independent: GRAIN_SIZE calibrated for UHD (2160p)
#define REF_HEIGHT 2160.0

vec4 hook()
{
    // 3 sigmas in parallel: R coarse (1.20×), G medium, B fine (0.80×).
    float scale = target_size.y / REF_HEIGHT;
    float sigma_g = max(GRAIN_SIZE * scale, MIN_SIGMA);
    float sigma_r = sigma_g * 1.20;
    float sigma_b = sigma_g * 0.80;

    vec2 pos = gl_FragCoord.xy;

    // Early-exit for very small sigma: when σ<0.5 px, first neighbor weight
    // exp(-1/(2σ²)) < 0.14, and further taps are numerically negligible.
    // Skipping the loop loses <~5% variance (absorbed by var-comp in Pass
    // 1.5), saves 8 hash calls at default GRAIN_SIZE=0.4 (UHD).
    if (sigma_r < 0.5) {
        float n = hash_noise(pos, random);
        return vec4(vec3(n * 0.5 + 0.5), 1.0);
    }

    vec3 nrcp = -1.0 / (2.0 * vec3(sigma_r, sigma_g, sigma_b)
                              * vec3(sigma_r, sigma_g, sigma_b));
    vec3 sum  = vec3(hash_noise(pos, random));  // center tap, w=1
    vec3 wsum = vec3(1.0);
    for (int i = 1; i <= MAX_R; i++) {
        float d = float(i);
        float d2 = d * d;
        vec3 w = exp(d2 * nrcp);
        float n_minus = hash_noise(pos + vec2(-d, 0.0), random);
        float n_plus  = hash_noise(pos + vec2( d, 0.0), random);
        sum  += (n_minus + n_plus) * w;
        wsum += 2.0 * w;
    }
    vec3 blurred = sum / wsum;

    return vec4(blurred * 0.5 + 0.5, 1.0);
}

//!HOOK OUTPUT
//!BIND GRAIN_H
//!SAVE GRAIN_V
//!COMPONENTS 3
//!WIDTH HOOKED.w
//!HEIGHT HOOKED.h
//!DESC cinegrain2 grain v-blur

#define MAX_R 3
#define MIN_SIGMA 0.001

vec4 hook()
{
    float scale = GRAIN_H_size.y / 2160.0;
    float sigma_g = max(GRAIN_SIZE * scale, MIN_SIGMA);
    float sigma_r = sigma_g * 1.20;
    float sigma_b = sigma_g * 0.80;

    // Early-exit at small sigma: pass GRAIN_H through unchanged. Matches
    // Pass 1's early-exit so both passes stay consistent — std is preserved
    // at the input level, no var-comp needed when only the center tap runs.
    if (sigma_r < 0.5) {
        return GRAIN_H_tex(GRAIN_H_pos);
    }

    vec3 nrcp = -1.0 / (2.0 * vec3(sigma_r, sigma_g, sigma_b)
                              * vec3(sigma_r, sigma_g, sigma_b));
    vec3 sum  = (GRAIN_H_tex(GRAIN_H_pos).rgb * 2.0 - 1.0);
    vec3 wsum = vec3(1.0);
    vec3 wsq  = vec3(1.0);
    for (int i = 1; i <= MAX_R; i++) {
        float d = float(i);
        float d2 = d * d;
        vec3 w = exp(d2 * nrcp);
        vec3 v_minus = GRAIN_H_tex(GRAIN_H_pos + vec2(0.0, -d) * GRAIN_H_pt).rgb * 2.0 - 1.0;
        vec3 v_plus  = GRAIN_H_tex(GRAIN_H_pos + vec2(0.0,  d) * GRAIN_H_pt).rgb * 2.0 - 1.0;
        sum  += (v_minus + v_plus) * w;
        wsum += 2.0 * w;
        wsq  += 2.0 * w * w;
    }
    vec3 blurred = (sum / wsum) * (wsum * wsum) / wsq;
    return vec4(blurred * 0.5 + 0.5, 1.0);
}

//!HOOK OUTPUT
//!BIND HOOKED
//!BIND GRAIN_V
//!DESC cinegrain2 apply

#define MAX_R 3
#define MIN_SIGMA 0.001
#define GRAIN_SCALE 0.702

float hash_noise(vec2 p, float seed)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973) + seed);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z) * 2.0 - 1.0;
}

// Five-zone equalizer over luma, smoothstep between neighbouring zones.
float luma_weight(float luma)
{
    float l = clamp(luma, 0.0, 1.0);
    float g, t;
    if (l <= 0.13) {
        t = l / 0.13;                    t = t*t*(3.0 - 2.0*t);
        g = mix(GAIN_1, GAIN_2, t);
    } else if (l <= 0.30) {
        t = (l - 0.13) / 0.17;           t = t*t*(3.0 - 2.0*t);
        g = mix(GAIN_2, GAIN_3, t);
    } else if (l <= 0.60) {
        t = (l - 0.30) / 0.30;           t = t*t*(3.0 - 2.0*t);
        g = mix(GAIN_3, GAIN_4, t);
    } else {
        t = (l - 0.60) / 0.40;           t = t*t*(3.0 - 2.0*t);
        g = mix(GAIN_4, GAIN_5, t);
    }
    return max(0.0, 1.0 + g);
}

vec4 hook()
{
    vec4 color = HOOKED_tex(HOOKED_pos);

    if (DEMO > 0.99) {
        float step = floor(gl_FragCoord.x / HOOKED_size.x * 16.0);
        color.rgb = vec3(step / 15.0);
    } else if (DEMO > 0.0) {
        color.rgb = vec3(DEMO);
    }

    // Per-channel RGB grain: sample GRAIN_V at 3 different large shifts
    // (≫ grain correlation length → independent draws). Pick the component
    // that matches the desired grain size: .r = coarse, .g = medium, .b = fine.
    // TILED shifts (fract-wrap instead of clamp-to-edge). Large shifts give
    // RGB decorrelation beyond grain correlation length; fract() wraps
    // out-of-bounds samples to the opposite texture edge → no edge-clamp
    // artifact. Independent noise field means the wrap produces no visible
    // seam (sampled values at wrapped positions are just more uncorrelated
    // noise).
    vec2 SHIFT_R = vec2(137.0, -73.0) * GRAIN_V_pt;
    vec2 SHIFT_B = vec2(-82.0, 235.0) * GRAIN_V_pt;
    vec2 pos_r = fract(HOOKED_pos + SHIFT_R);
    vec2 pos_b = fract(HOOKED_pos + SHIFT_B);

    vec3 grain_raw = vec3(
        GRAIN_V_tex(pos_r     ).r * 2.0 - 1.0,
        GRAIN_V_tex(HOOKED_pos).g * 2.0 - 1.0,
        GRAIN_V_tex(pos_b     ).b * 2.0 - 1.0
    );

    if (SOFTBLUR > 0.01) {
        float r = SOFTBLUR;
        float aoff = hash_noise(gl_FragCoord.xy + vec2(13.37, 71.3), random) * 6.2831853;
        vec3 gn = vec3(0.0);
        for (int j = 0; j < 8; j++) {
            float a = float(j) * 0.7853981634 + aoff;
            vec2 noff = vec2(cos(a), sin(a)) * r * GRAIN_V_pt;
            gn.r += GRAIN_V_tex(fract(pos_r      + noff)).r * 2.0 - 1.0;
            gn.g += GRAIN_V_tex(fract(HOOKED_pos + noff)).g * 2.0 - 1.0;
            gn.b += GRAIN_V_tex(fract(pos_b      + noff)).b * 2.0 - 1.0;
        }
        gn *= 0.125;
        float t = clamp(SOFTBLUR / 10.0, 0.0, 1.0);
        grain_raw = mix(grain_raw, gn, t);
    }

    float grain_y = dot(grain_raw, vec3(0.2126, 0.7152, 0.0722));
    grain_raw = mix(vec3(grain_y), grain_raw, CHROMA);

    // GRAIN_V already has per-channel variance compensation baked in.
    vec3 grain = grain_raw * GRAIN_SCALE;

    // Vision3-like per-channel amplitude, slightly bumped from 250D
    // baseline (1.10/1.00/0.92) toward mid-speed stocks (500T-ish).
    const vec3 channel_amp = vec3(1.15, 1.00, 0.88);

    float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    float weight = luma_weight(luma);

    color.rgb *= exp(LEVEL * weight * grain * channel_amp);

    color.rgb = max(color.rgb, vec3(0.0));
    return color;
}
