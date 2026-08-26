// GTU v0.50
// Author: aliaspider - aliaspider@gmail.com
// License: GPLv3
// Source: https://github.com/libretro/common-shaders/tree/master/crt/shaders/gtu-v050

//!PARAM compositeConnection
//!DESC Composite Connection
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM noScanlines
//!DESC No Scanlines
//!TYPE int
//!MINIMUM 0
//!MAXIMUM 1
0

//!PARAM signalResolution
//!DESC Signal Resolution Y
//!TYPE int
//!MINIMUM 16
//!MAXIMUM 1024
256

//!PARAM signalResolutionI
//!DESC Signal Resolution I
//!TYPE int
//!MINIMUM 1
//!MAXIMUM 350
83

//!PARAM signalResolutionQ
//!DESC Signal Resolution Q
//!TYPE int
//!MINIMUM 1
//!MAXIMUM 350
25

//!PARAM tvVerticalResolution
//!DESC TV Vertical Resolution
//!TYPE int
//!MINIMUM 20
//!MAXIMUM 1000
250

//!PARAM blackLevel
//!DESC Black Level
//!TYPE float
//!MINIMUM -0.3
//!MAXIMUM 0.3
0.07

//!PARAM contrast
//!DESC Contrast
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 2.0
1.0


//!HOOK OUTPUT
//!BIND HOOKED
//!BIND MAIN
//!SAVE GTU_H
//!WIDTH OUTPUT.w
//!HEIGHT MAIN.h
//!COMPONENTS 4
//!DESC GTU v0.50 horizontal signal

#define GTU_PI 3.14159265358

float gtu_d(float x, float b)
{
    return GTU_PI * b * min(abs(x) + 0.5, 1.0 / b);
}

float gtu_e(float x, float b)
{
    return GTU_PI * b * min(max(abs(x) - 0.5, -1.0 / b), 1.0 / b);
}

float gtu_stu(float x, float b)
{
    return (gtu_d(x, b) + sin(gtu_d(x, b))
          - gtu_e(x, b) - sin(gtu_e(x, b))) / (2.0 * GTU_PI);
}

vec4 gtu_point_main(vec2 uv)
{
    ivec2 size_i = max(ivec2(MAIN_size), ivec2(1));
    ivec2 p = ivec2(floor(uv * MAIN_size));
    p = clamp(p, ivec2(0), size_i - ivec2(1));
    return texelFetch(MAIN_raw, p, 0);
}

vec3 gtu_rgb_to_yiq(vec3 c)
{
    return vec3(
        0.299000 * c.r + 0.587000 * c.g + 0.114000 * c.b,
        0.595716 * c.r - 0.274453 * c.g - 0.321263 * c.b,
        0.211456 * c.r - 0.522591 * c.g + 0.311135 * c.b
    );
}

vec3 gtu_yiq_to_rgb(vec3 c)
{
    return vec3(
        c.x + 0.9563 * c.y + 0.6210 * c.z,
        c.x - 0.2721 * c.y - 0.6474 * c.z,
        c.x - 1.1070 * c.y + 1.7046 * c.z
    );
}

vec4 hook()
{
    vec2 pos = MAIN_pos;
    vec2 inputSize = MAIN_size;
    vec2 inputPt = MAIN_pt;

    float offset = fract((pos.x * inputSize.x) - 0.5);
    vec3 tempColor = vec3(0.0);

    if (compositeConnection != 0) {
        float lowestResolution = min(
            min(float(signalResolution), float(signalResolutionI)),
            float(signalResolutionQ)
        );
        float range = ceil(0.5 + inputSize.x / lowestResolution);

        for (float i = -range; i < range + 2.0; i += 1.0) {
            float X = offset - i;
            vec3 c = gtu_point_main(
                vec2(pos.x - X * inputPt.x, pos.y)
            ).rgb;

            c = gtu_rgb_to_yiq(c);

            tempColor += vec3(
                c.x * gtu_stu(X, float(signalResolution)  * inputPt.x),
                c.y * gtu_stu(X, float(signalResolutionI) * inputPt.x),
                c.z * gtu_stu(X, float(signalResolutionQ) * inputPt.x)
            );
        }

        tempColor = clamp(gtu_yiq_to_rgb(tempColor), 0.0, 1.0);
    } else {
        float range = ceil(
            0.5 + inputSize.x / float(signalResolution)
        );

        for (float i = -range; i < range + 2.0; i += 1.0) {
            float X = offset - i;
            vec3 c = gtu_point_main(
                vec2(pos.x - X * inputPt.x, pos.y)
            ).rgb;

            tempColor += c * gtu_stu(
                X, float(signalResolution) * inputPt.x
            );
        }

        tempColor = clamp(tempColor, 0.0, 1.0);
    }

    return vec4(tempColor, 1.0);
}


//!HOOK OUTPUT
//!BIND HOOKED
//!BIND GTU_H
//!COMPONENTS 4
//!DESC GTU v0.50 vertical CRT response

#define GTU_PI 3.14159265358

float gtu_normal_gauss(float x)
{
    return exp(-(x * x) * 0.5) / sqrt(2.0 * GTU_PI);
}

float gtu_normal_gauss_integral(float x)
{
    float a1 = 0.4361836;
    float a2 = -0.1201676;
    float a3 = 0.9372980;
    float p = 0.3326700;
    float t = 1.0 / (1.0 + p * abs(x));

    return (
        0.5
        - gtu_normal_gauss(x)
        * (t * (a1 + t * (a2 + a3 * t)))
    ) * sign(x);
}

float gtu_d(float x, float b)
{
    return GTU_PI * b * min(abs(x) + 0.5, 1.0 / b);
}

float gtu_e(float x, float b)
{
    return GTU_PI * b * min(max(abs(x) - 0.5, -1.0 / b), 1.0 / b);
}

float gtu_stu(float x, float b)
{
    return (gtu_d(x, b) + sin(gtu_d(x, b))
          - gtu_e(x, b) - sin(gtu_e(x, b))) / (2.0 * GTU_PI);
}

vec4 gtu_point_h(vec2 uv)
{
    ivec2 size_i = max(ivec2(GTU_H_size), ivec2(1));
    ivec2 p = ivec2(floor(uv * GTU_H_size));
    p = clamp(p, ivec2(0), size_i - ivec2(1));
    return texelFetch(GTU_H_raw, p, 0);
}

vec3 gtu_scanlines(float x, vec3 c)
{
    float inputHeight = GTU_H_size.y;
    float inputPtY = GTU_H_pt.y;
    float outputHeight = HOOKED_size.y;
    float outputPtY = HOOKED_pt.y;

    float temp = sqrt(2.0 * GTU_PI)
               * (float(tvVerticalResolution) * inputPtY);

    float rrr = 0.5 * (inputHeight * outputPtY);
    float x1 = (x + rrr) * temp;
    float x2 = (x - rrr) * temp;

    float beam = gtu_normal_gauss_integral(x1)
               - gtu_normal_gauss_integral(x2);

    c *= beam;
    c *= outputHeight / max(inputHeight, 1.0);

    return c;
}

vec4 hook()
{
    vec2 pos = HOOKED_pos;
    vec2 inputSize = GTU_H_size;
    vec2 inputPt = GTU_H_pt;

    vec2 offset = fract(
        pos * vec2(HOOKED_size.x, inputSize.y) - 0.5
    );

    vec3 tempColor = vec3(0.0);

    float range = ceil(
        0.5 + inputSize.y / float(tvVerticalResolution)
    );

    if (noScanlines != 0) {
        for (float i = -range; i < range + 2.0; i += 1.0) {
            float y = offset.y - i;

            vec3 Cj = gtu_point_h(
                vec2(
                    pos.x,
                    pos.y - y * inputPt.y
                )
            ).rgb;

            tempColor += Cj * gtu_stu(
                y,
                float(tvVerticalResolution) * inputPt.y
            );
        }
    } else {
        for (float i = -range; i < range + 2.0; i += 1.0) {
            float y = offset.y - i;

            vec3 Cj = gtu_point_h(
                vec2(
                    pos.x,
                    pos.y - y * inputPt.y
                )
            ).rgb;

            tempColor += gtu_scanlines(y, Cj);
        }
    }

    tempColor -= vec3(blackLevel);
    tempColor *= contrast / (1.0 - blackLevel);

    return vec4(tempColor, 1.0);
}
