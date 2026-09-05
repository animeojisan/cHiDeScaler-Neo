// CRT Beam Simulator-Neo
// cHiDeScaler-Neo refresh-cycle adaptation of Blur Busters crt-beam-simulator.
// Original project: https://github.com/blurbusters/crt-beam-simulator
// Original license: MIT
// Copyright 2024 Mark Rejhon (@BlurBusters) & Timothy Lottes (@NOTimothyLottes)
// See: CRT Beam Simulator LICENSE.txt
//
// Neo extension:
// - //!DISPLAY_HZ asks Neo to retain the image immediately before this stage
//   and rerun this stage + all following stages at the physical monitor Hz.
// - neo_display_frame advances per display refresh.
// - neo_refresh_hz is the detected monitor refresh rate.
// - MAIN hook is intentional: Neo's RGB runner treats MAIN as the authoritative
//   colour output, so writing MAIN guarantees the beam result is presented.
//
// Recommended placement: near the end of the chain. Neo deliberately does not
// reorder filters; placing expensive filters after this shader makes them run
// at display Hz too.

//!PARAM SimulatedHz
//!DESC Simulated CRT scan frequency
//!TYPE float
//!MINIMUM 24.0
//!MAXIMUM 120.0
60.0

//!PARAM EffectStrength
//!DESC CRT beam effect strength
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
1.0

//!PARAM Gamma
//!DESC Display gamma used by beam brightness math
//!TYPE float
//!MINIMUM 1.8
//!MAXIMUM 3.0
2.4

//!PARAM GainVsBlur
//!DESC Brightness vs motion-clarity tradeoff
//!TYPE float
//!MINIMUM 0.20
//!MAXIMUM 1.00
0.70

//!PARAM ScanDirection
//!DESC 0=global 1=top-down 2=bottom-up 3=left-right 4=right-left
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 4
1

//!PARAM LcdAntiRetention
//!DESC 1=slightly slew even scan ratios to reduce static LCD retention patterns
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
1

//!PARAM SplitScreen
//!DESC 0=full effect 1=left-side comparison
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM SplitX
//!DESC Split-screen boundary
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!DISPLAY_HZ
//!HOOK MAIN
//!BIND HOOKED
//!DESC CRT Beam Simulator-Neo
//!COMPONENTS 4

#define LCD_SLEW 0.001
#define SPLIT_BORDER_PX 2.0

float selF(float a, float b, bool p) { return p ? b : a; }

float linear2srgb1(float c)
{
    float lo = c * 12.92;
    float hi = 1.055 * pow(max(c, 0.0), 1.0 / Gamma) - 0.055;
    return clamp(selF(lo, hi, c > 0.0031308), 0.0, 1.0);
}

vec3 linear2srgb3(vec3 c)
{
    return vec3(linear2srgb1(c.r), linear2srgb1(c.g), linear2srgb1(c.b));
}

float srgb2linear1(float c)
{
    float lo = c / 12.92;
    float hi = pow(max((c + 0.055) / 1.055, 0.0), Gamma);
    return selF(lo, hi, c > 0.04045);
}

vec3 srgb2linear3(vec3 c)
{
    return vec3(srgb2linear1(c.r), srgb2linear1(c.g), srgb2linear1(c.b));
}

float effectiveFramesPerHz()
{
    float simulated = max(SimulatedHz, 1.0);
    float f = max(neo_refresh_hz / simulated, 1.0);
    float fi = floor(f + 0.0001);
    bool isInteger = abs(f - fi) < 0.0001;
    bool isEvenInteger = isInteger && mod(fi, 2.0) < 0.5;
    if (LcdAntiRetention != 0 && isEvenInteger)
        f += LCD_SLEW;
    return f;
}

vec3 getPixelFromOrigFrameNeo(vec2 uv)
{
    return textureLod(HOOKED_raw, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).rgb;
}

vec3 getPixelFromSimulatedCRTNeo(vec2 uv, float crtRasterPos, float framesPerHz)
{
    // The refresh-cycle path reuses the same completed source image between
    // content updates. The moving beam phase changes every physical refresh.
    vec3 pixel = srgb2linear3(getPixelFromOrigFrameNeo(uv));

    float brightnessScale = framesPerHz * GainVsBlur;
    vec3 colorPrev2 = pixel * brightnessScale;
    vec3 colorPrev1 = colorPrev2;
    vec3 colorCurr  = colorPrev2;

    float tubePos;
    if (ScanDirection == 0)
        tubePos = 0.0;
    else if (ScanDirection == 1)
        tubePos = 1.0 - uv.y;
    else if (ScanDirection == 2)
        tubePos = uv.y;
    else if (ScanDirection == 3)
        tubePos = uv.x;
    else
        tubePos = 1.0 - uv.x;

    vec3 result = vec3(0.0);
    float tubeFrame = tubePos * framesPerHz;
    float fStart = crtRasterPos * framesPerHz;
    float fEnd = fStart + 1.0;

    for (int ch = 0; ch < 3; ++ch)
    {
        float Lprev2 = colorPrev2[ch];
        float Lprev1 = colorPrev1[ch];
        float Lcurr  = colorCurr[ch];

        float startPrev2 = tubeFrame - framesPerHz;
        float endPrev2   = startPrev2 + Lprev2;
        float startPrev1 = tubeFrame;
        float endPrev1   = startPrev1 + Lprev1;
        float startCurr  = tubeFrame + framesPerHz;
        float endCurr    = startCurr + Lcurr;

        float overlapPrev2 = max(0.0, min(endPrev2, fEnd) - max(startPrev2, fStart));
        float overlapPrev1 = max(0.0, min(endPrev1, fEnd) - max(startPrev1, fStart));
        float overlapCurr  = max(0.0, min(endCurr,  fEnd) - max(startCurr,  fStart));

        result[ch] = overlapPrev2 + overlapPrev1 + overlapCurr;
    }

    return linear2srgb3(result);
}

vec4 hook()
{
    vec2 uv = HOOKED_pos;
    vec4 src = HOOKED_tex(uv);
    float framesPerHz = effectiveFramesPerHz();
    float crtRasterPos = fract(float(neo_display_frame) / framesPerHz);
    vec3 crt = getPixelFromSimulatedCRTNeo(uv, crtRasterPos, framesPerHz);
    vec3 effected = mix(src.rgb, crt, clamp(EffectStrength, 0.0, 1.0));

    if (SplitScreen != 0)
    {
        float borderPx = abs(uv.x * HOOKED_size.x - SplitX * HOOKED_size.x);
        if (borderPx < SPLIT_BORDER_PX)
            return vec4(1.0, 1.0, 1.0, src.a);
        if (uv.x > SplitX)
            return src;
    }

    return vec4(effected, src.a);
}
