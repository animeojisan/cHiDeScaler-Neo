// CRT-Geom
// Based on the libretro common-shaders CRT-Geom implementation.

/*
    CRT-interlaced

    Copyright (C) 2010-2012 cgwg, Themaister and DOLLS

    This program is free software; you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the Free
    Software Foundation; either version 2 of the License, or (at your option)
    any later version.

    (cgwg gave their consent to have the original version of this shader
    distributed under the GPL in this message:

        http://board.byuu.org/viewtopic.php?p=26075#p26075

        "Feel free to distribute my shaders under the GPL. After all, the
        barrel distortion code was taken from the Curvature shader, which is
        under the GPL."
    )
    This shader variant is pre-configured with screen Curvature.
*/

//!PARAM CRTGamma
//!DESC Target Gamma
//!TYPE CONSTANT float
//!MINIMUM 0.1
//!MAXIMUM 5.0
2.4

//!PARAM MonitorGamma
//!DESC Monitor Gamma
//!TYPE CONSTANT float
//!MINIMUM 0.1
//!MAXIMUM 5.0
2.2

//!PARAM CRTDistance
//!DESC Distance
//!TYPE CONSTANT float
//!MINIMUM 0.1
//!MAXIMUM 3.0
1.5

//!PARAM Curvature
//!DESC Curvature
//!TYPE CONSTANT int
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM CurvatureRadius
//!DESC Curvature Radius
//!TYPE CONSTANT float
//!MINIMUM 0.1
//!MAXIMUM 10.0
2.0

//!PARAM CornerSize
//!DESC Corner Size
//!TYPE CONSTANT float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.03

//!PARAM CornerSmoothness
//!DESC Corner Smoothness
//!TYPE CONSTANT int
//!MINIMUM 80
//!MAXIMUM 2000
1000

//!PARAM HorizontalTilt
//!DESC Horizontal Tilt
//!TYPE CONSTANT float
//!MINIMUM -0.5
//!MAXIMUM 0.5
0.0

//!PARAM VerticalTilt
//!DESC Vertical Tilt
//!TYPE CONSTANT float
//!MINIMUM -0.5
//!MAXIMUM 0.5
0.0

//!PARAM HorizontalOverscan
//!DESC Horizontal Overscan
//!TYPE CONSTANT int
//!MINIMUM -125
//!MAXIMUM 125
100

//!PARAM VerticalOverscan
//!DESC Vertical Overscan
//!TYPE CONSTANT int
//!MINIMUM -125
//!MAXIMUM 125
100

//!PARAM DotMask
//!DESC Dot Mask
//!TYPE CONSTANT float
//!MINIMUM 0.0
//!MAXIMUM 0.3
0.3

//!PARAM Sharpness
//!DESC Sharpness
//!TYPE CONSTANT int
//!MINIMUM 1
//!MAXIMUM 3
1

//!PARAM ScanlineWeight
//!DESC Scanline Weight
//!TYPE CONSTANT float
//!MINIMUM 0.1
//!MAXIMUM 0.5
0.3

//!PARAM LuminanceBoost
//!DESC Luminance Boost
//!TYPE CONSTANT float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!HOOK OUTPUT
//!BIND HOOKED
//!BIND MAIN
//!DESC CRT Geom
//!COMPONENTS 4

#define FIX(c) max(abs(c), 1e-5)
#define PI 3.141592653589
#define aspect vec2(1.0, 0.75)

vec4 pointSample(vec2 uv)
{
    ivec2 sizePx = textureSize(MAIN_raw, 0);
    ivec2 p = ivec2(floor(uv * vec2(sizePx)));
    p = clamp(p, ivec2(0), sizePx - ivec2(1));
    return texelFetch(MAIN_raw, p, 0);
}

vec4 TEX2D(vec2 uv)
{
    return pow(pointSample(uv), vec4(CRTGamma));
}

float intersectGeom(vec2 xy, vec4 sin_cos_angle)
{
    float A = dot(xy, xy) + CRTDistance * CRTDistance;
    float B = 2.0 * (CurvatureRadius * (dot(xy, sin_cos_angle.xy)
        - CRTDistance * sin_cos_angle.z * sin_cos_angle.w) - CRTDistance * CRTDistance);
    float C = CRTDistance * CRTDistance
        + 2.0 * CurvatureRadius * CRTDistance * sin_cos_angle.z * sin_cos_angle.w;
    return (-B - sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
}

vec2 bkwtrans(vec2 xy, vec4 sin_cos_angle)
{
    float c = intersectGeom(xy, sin_cos_angle);
    vec2 point_ = c * xy;
    point_ += CurvatureRadius * sin_cos_angle.xy;
    point_ /= CurvatureRadius;
    vec2 tang = sin_cos_angle.xy / sin_cos_angle.zw;
    vec2 poc = point_ / sin_cos_angle.zw;
    float A = dot(tang, tang) + 1.0;
    float B = -2.0 * dot(poc, tang);
    float C = dot(poc, poc) - 1.0;
    float a = (-B + sqrt(B * B - 4.0 * A * C)) / (2.0 * A);
    vec2 uv = (point_ - a * sin_cos_angle.xy) / sin_cos_angle.zw;
    float r = FIX(CurvatureRadius * acos(a));
    return uv * r / sin(r / CurvatureRadius);
}

vec2 fwtrans(vec2 uv, vec4 sin_cos_angle)
{
    float r = FIX(sqrt(dot(uv, uv)));
    uv *= sin(r / CurvatureRadius) / r;
    float x = 1.0 - cos(r / CurvatureRadius);
    float D = CRTDistance / CurvatureRadius
        + x * sin_cos_angle.z * sin_cos_angle.w
        + dot(uv, sin_cos_angle.xy);
    return CRTDistance * (uv * sin_cos_angle.zw - x * sin_cos_angle.xy) / D;
}

vec3 maxscale(vec4 sin_cos_angle)
{
    vec2 c = bkwtrans(
        -CurvatureRadius * sin_cos_angle.xy
            / (1.0 + CurvatureRadius / CRTDistance * sin_cos_angle.z * sin_cos_angle.w),
        sin_cos_angle);
    vec2 a = 0.5 * aspect;
    vec2 lo = vec2(
        fwtrans(vec2(-a.x, c.y), sin_cos_angle).x,
        fwtrans(vec2(c.x, -a.y), sin_cos_angle).y) / aspect;
    vec2 hi = vec2(
        fwtrans(vec2(+a.x, c.y), sin_cos_angle).x,
        fwtrans(vec2(c.x, +a.y), sin_cos_angle).y) / aspect;
    return vec3((hi + lo) * aspect * 0.5, max(hi.x - lo.x, hi.y - lo.y));
}

vec4 scanlineWeights(float distance1, vec4 color)
{
    vec4 wid = 2.0 + 2.0 * pow(color, vec4(4.0));
    float v = distance1 / ScanlineWeight;
    vec4 weights = vec4(v);
    return (LuminanceBoost + 1.4)
        * exp(-pow(weights * inversesqrt(0.5 * wid), wid))
        / (0.6 + 0.2 * wid);
}

vec4 hook()
{
    vec2 pos = HOOKED_pos;
    vec2 outputSize = target_size;
    vec2 inputSize = MAIN_size;

    vec4 sin_cos_angle = vec4(
        sin(vec2(HorizontalTilt, VerticalTilt)),
        cos(vec2(HorizontalTilt, VerticalTilt)));
    vec3 stretch = maxscale(sin_cos_angle);
    vec2 TextureSize = vec2(float(Sharpness) * inputSize.x, inputSize.y);
    // Derive dot-mask parity from the normalized output position and the
    // actual pass target size, matching the shader's original pixel grid.
    float mod_factor = pos.x * outputSize.x;
    vec2 ilfac = vec2(1.0, clamp(floor(inputSize.y / 1000.0), 1.0, 2.0));
    vec2 one = ilfac / TextureSize;

    vec2 xy = vec2(0.0);
    if (Curvature > 0) {
        vec2 cd = pos;
        cd = (cd - 0.5) * aspect * stretch.z + stretch.xy;
        xy = bkwtrans(cd, sin_cos_angle)
            / vec2(float(HorizontalOverscan) / 100.0, float(VerticalOverscan) / 100.0)
            / aspect + vec2(0.5, 0.5);
    } else {
        xy = pos;
    }

    vec2 cd2 = xy;
    cd2 = (cd2 - 0.5) * vec2(float(HorizontalOverscan), float(VerticalOverscan)) / 100.0 + 0.5;
    cd2 = min(cd2, 1.0 - cd2) * aspect;
    vec2 cdist = vec2(CornerSize, CornerSize);
    cd2 = cdist - min(cd2, cdist);
    float dist = sqrt(dot(cd2, cd2));
    float cval = clamp((cdist.x - dist) * float(CornerSmoothness), 0.0, 1.0);

    vec2 ratio_scale = (xy * TextureSize - 0.5) / ilfac;
    float scanFilter = inputSize.y / outputSize.y;
    vec2 uv_ratio = fract(ratio_scale);

    xy = (floor(ratio_scale) * ilfac + 0.5) / TextureSize;

    vec4 coeffs = PI * vec4(
        1.0 + uv_ratio.x,
        uv_ratio.x,
        1.0 - uv_ratio.x,
        2.0 - uv_ratio.x);
    coeffs = FIX(coeffs);
    coeffs = 2.0 * sin(coeffs) * sin(coeffs / 2.0) / (coeffs * coeffs);
    coeffs /= dot(coeffs, vec4(1.0));

    vec4 col = clamp(
          coeffs.x * TEX2D(xy + vec2(-one.x, 0.0))
        + coeffs.y * TEX2D(xy)
        + coeffs.z * TEX2D(xy + vec2(one.x, 0.0))
        + coeffs.w * TEX2D(xy + vec2(2.0 * one.x, 0.0)),
        0.0, 1.0);

    vec4 col2 = clamp(
          coeffs.x * TEX2D(xy + vec2(-one.x, one.y))
        + coeffs.y * TEX2D(xy + vec2(0.0, one.y))
        + coeffs.z * TEX2D(xy + one)
        + coeffs.w * TEX2D(xy + vec2(2.0 * one.x, one.y)),
        0.0, 1.0);

    col = pow(col, vec4(CRTGamma));
    col2 = pow(col2, vec4(CRTGamma));

    vec4 weights = scanlineWeights(uv_ratio.y, col);
    vec4 weights2 = scanlineWeights(1.0 - uv_ratio.y, col2);

    uv_ratio.y = uv_ratio.y + 1.0 / 3.0 * scanFilter;
    weights = (weights + scanlineWeights(uv_ratio.y, col)) / 3.0;
    weights2 = (weights2 + scanlineWeights(abs(1.0 - uv_ratio.y), col2)) / 3.0;
    uv_ratio.y = uv_ratio.y - 2.0 / 3.0 * scanFilter;
    weights = weights + scanlineWeights(abs(uv_ratio.y), col) / 3.0;
    weights2 = weights2 + scanlineWeights(abs(1.0 - uv_ratio.y), col2) / 3.0;

    vec3 mul_res = (col * weights + col2 * weights2).rgb;
    mul_res *= vec3(cval, cval, cval);

    vec3 dotMaskWeights = mix(
        vec3(1.0, 1.0 - DotMask, 1.0),
        vec3(1.0 - DotMask, 1.0, 1.0 - DotMask),
        floor(mod(mod_factor, 2.0)));
    mul_res *= dotMaskWeights;

    mul_res = pow(mul_res, vec3(1.0 / MonitorGamma));
    return vec4(mul_res, 1.0);
}
