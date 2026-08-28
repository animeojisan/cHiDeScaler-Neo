//!PARAM STRENGTH
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.25
1.0

//!PARAM RADIUS
//!TYPE float
//!MINIMUM 6.0
//!MAXIMUM 30.0
28.0

//!PARAM RANGE
//!TYPE float
//!MINIMUM 0.008
//!MAXIMUM 0.100
0.075

//!PARAM LINE_GUARD
//!TYPE float
//!MINIMUM 0.001
//!MAXIMUM 0.030
0.006

//!PARAM DITHER
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.06

//!PARAM MAX_SHIFT
//!TYPE float
//!MINIMUM 0.002
//!MAXIMUM 0.040
0.016

//!HOOK MAIN
//!BIND HOOKED
//!SAVE NEO_DB_H
//!DESC [Neo Deband] horizontal surface

float db_max3(vec3 v) { return max(v.r, max(v.g, v.b)); }

float db_luma(vec3 v) { return dot(v, vec3(0.2126, 0.7152, 0.0722)); }

vec4 hook() {
    vec4 src = HOOKED_texOff(0);
    vec3 sum = src.rgb * 0.35;
    float total = 0.35;
    float spacing = RADIUS / 12.0;
    for (int i = -12; i <= 12; i++) {
        if (i == 0) continue;
        float fi = float(i);
        vec3 s = HOOKED_texOff(vec2(fi * spacing, 0.0)).rgb;
        float spatial = exp(-0.5 * (fi * fi) / 42.0);
        float delta = db_max3(abs(s - src.rgb));
        float rangeWeight = exp(-0.5 * delta * delta / max(RANGE * RANGE, 1e-7));
        rangeWeight *= 1.0 - smoothstep(RANGE * 0.55, RANGE * 0.95, delta);
        float w = spatial * rangeWeight;
        sum += s * w;
        total += w;
    }
    return vec4(sum / max(total, 1e-6), src.a);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND NEO_DB_H
//!SAVE NEO_DB_BASE
//!DESC [Neo Deband] vertical surface

float db_max3(vec3 v) { return max(v.r, max(v.g, v.b)); }

vec4 hook() {
    vec4 src = HOOKED_texOff(0);
    vec3 center = NEO_DB_H_tex(NEO_DB_H_pos).rgb;
    vec3 sum = center * 0.35;
    float total = 0.35;
    float spacing = RADIUS / 12.0;
    for (int i = -12; i <= 12; i++) {
        if (i == 0) continue;
        float fi = float(i);
        vec3 s = NEO_DB_H_texOff(vec2(0.0, fi * spacing)).rgb;
        vec3 guide = HOOKED_texOff(vec2(0.0, fi * spacing)).rgb;
        float spatial = exp(-0.5 * (fi * fi) / 42.0);
        float delta = db_max3(abs(guide - src.rgb));
        float rangeWeight = exp(-0.5 * delta * delta / max(RANGE * RANGE, 1e-7));
        rangeWeight *= 1.0 - smoothstep(RANGE * 0.55, RANGE * 0.95, delta);
        float w = spatial * rangeWeight;
        sum += s * w;
        total += w;
    }
    return vec4(sum / max(total, 1e-6), src.a);
}

//!HOOK MAIN
//!BIND HOOKED
//!BIND NEO_DB_BASE
//!DESC [Neo Deband]

float db_hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float db_max3(vec3 v) { return max(v.r, max(v.g, v.b)); }

float db_luma(vec3 v) { return dot(v, vec3(0.2126, 0.7152, 0.0722)); }

float db_ridge(vec3 a, vec3 b, vec3 c) {
    vec3 da = a - c;
    vec3 db = b - c;
    return db_max3(min(abs(da), abs(db)) * step(vec3(0.0), da * db));
}

vec4 hook() {
    vec4 src = HOOKED_texOff(0);
    vec3 c = src.rgb;
    vec3 base = NEO_DB_BASE_tex(NEO_DB_BASE_pos).rgb;
    float sourceY = db_luma(c);
    float liftScale = mix(0.15, 0.55, smoothstep(0.08, 0.60, sourceY));
    float darkenScale = mix(0.55, 0.18, smoothstep(0.40, 0.90, sourceY));
    float filteredY = db_luma(base);
    float lumaShift = clamp(filteredY - sourceY,
                            -MAX_SHIFT * darkenScale,
                             MAX_SHIFT * liftScale);
    lumaShift *= mix(0.45, 1.0, step(0.0, lumaShift));
    // Keep the source chroma exactly. Independent RGB smoothing can turn a
    // small luminance correction into a visible hue shift on dark gradients.
    base = c + vec3(lumaShift);
    vec3 l1 = HOOKED_texOff(vec2(-1.0, 0.0)).rgb;
    vec3 r1 = HOOKED_texOff(vec2(1.0, 0.0)).rgb;
    vec3 u1 = HOOKED_texOff(vec2(0.0, -1.0)).rgb;
    vec3 d1 = HOOKED_texOff(vec2(0.0, 1.0)).rgb;
    vec3 l2 = HOOKED_texOff(vec2(-2.0, 0.0)).rgb;
    vec3 r2 = HOOKED_texOff(vec2(2.0, 0.0)).rgb;
    vec3 u2 = HOOKED_texOff(vec2(0.0, -2.0)).rgb;
    vec3 d2 = HOOKED_texOff(vec2(0.0, 2.0)).rgb;
    vec3 l4 = HOOKED_texOff(vec2(-4.0, 0.0)).rgb;
    vec3 r4 = HOOKED_texOff(vec2(4.0, 0.0)).rgb;
    vec3 u4 = HOOKED_texOff(vec2(0.0, -4.0)).rgb;
    vec3 d4 = HOOKED_texOff(vec2(0.0, 4.0)).rgb;
    vec3 nw = HOOKED_texOff(vec2(-2.0, -2.0)).rgb;
    vec3 se = HOOKED_texOff(vec2(2.0, 2.0)).rgb;
    vec3 ne = HOOKED_texOff(vec2(2.0, -2.0)).rgb;
    vec3 sw = HOOKED_texOff(vec2(-2.0, 2.0)).rgb;

    float slope = max(db_max3(abs(l1 - r1)), db_max3(abs(u1 - d1)));
    float curve1 = max(db_max3(abs(l1 + r1 - 2.0 * c)), db_max3(abs(u1 + d1 - 2.0 * c)));
    float curve2 = max(db_max3(abs(l2 + r2 - 2.0 * c)), db_max3(abs(u2 + d2 - 2.0 * c)));
    float curve4 = max(db_max3(abs(l4 + r4 - 2.0 * c)), db_max3(abs(u4 + d4 - 2.0 * c)));
    float ridge = max(max(db_ridge(l1, r1, c), db_ridge(u1, d1, c)),
                      max(max(db_ridge(l2, r2, c), db_ridge(u2, d2, c)),
                          max(max(db_ridge(l4, r4, c), db_ridge(u4, d4, c)),
                              max(db_ridge(nw, se, c), db_ridge(ne, sw, c)))));
    float lineMask = smoothstep(0.0007, LINE_GUARD, ridge);
    float edgeEnergy = max(slope * 0.45, max(curve1, max(curve2 * 0.75, curve4 * 0.50)));
    float edgeMask = smoothstep(0.010, 0.042, edgeEnergy);
    float protection = max(lineMask, edgeMask);
    float appliedShift = lumaShift * clamp(STRENGTH * (1.0 - protection), 0.0, 1.0);
    float appliedCodes = appliedShift * 255.0;
    // Accept subtle band corrections from 0.40 code values while retaining
    // symmetric treatment of brighter and darker transitions.
    appliedShift = sign(appliedCodes) * floor(abs(appliedCodes) + 0.60) / 255.0;
    vec3 result = c + vec3(appliedShift);

    float sameL = 1.0 - smoothstep(0.35 / 255.0, 1.35 / 255.0, db_max3(abs(l1 - c)));
    float sameR = 1.0 - smoothstep(0.35 / 255.0, 1.35 / 255.0, db_max3(abs(r1 - c)));
    float sameU = 1.0 - smoothstep(0.35 / 255.0, 1.35 / 255.0, db_max3(abs(u1 - c)));
    float sameD = 1.0 - smoothstep(0.35 / 255.0, 1.35 / 255.0, db_max3(abs(d1 - c)));
    float plateau = (sameL + sameR + sameU + sameD) * 0.25;
    float quantized = smoothstep(0.28, 0.72, plateau) * (1.0 - protection);
    vec2 pixel = floor(HOOKED_pos * HOOKED_size);
    float noise = db_hash(pixel + vec2(17.0, 29.0)) - db_hash(pixel.yx + vec2(71.0, 31.0));
    float headroom = smoothstep(0.0, 0.02, sourceY) * (1.0 - smoothstep(0.98, 1.0, sourceY));
    result += vec3(noise * (DITHER / 255.0) * quantized * headroom);
    return vec4(clamp(result, 0.0, 1.0), src.a);
}
