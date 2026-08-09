/*
    Catmull-Rom Resize Adjustable for cHiDeScaler-Neo v1.0
    -----------------------------
    - Adjustable resize from 0.25x to 4.00x.
    - Two separable passes: horizontal then vertical.
    - Kernel evaluation is fully inline for Neo's per-pass parser.
    - Downscaling widens the footprint for antialiasing.
*/

//!PARAM RESIZE_SCALE
//!DESC Resize scale (0.25x - 4.00x)
//!TYPE float
//!MINIMUM 0.25
//!MAXIMUM 4.00
0.75

//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w RESIZE_SCALE *
//!HEIGHT HOOKED.h
//!DESC Catmull-Rom Resize Adjustable Horizontal v1.0
vec4 hook()
{
    float src_len = HOOKED_size.x;
    float dst_len = max(floor(src_len * RESIZE_SCALE), 1.0);

    float dst_pos = HOOKED_pos.x * dst_len;
    float center = dst_pos * src_len / dst_len;

    float aa_scale = max(src_len / dst_len, 1.0);
    float radius = 2.0 * aa_scale;

    int lo = int(floor(center - radius + 0.5));
    int hi = int(ceil(center + radius - 0.5));

    vec4 accum = vec4(0.0);
    float weight_sum = 0.0;

    for (int tap = 0; tap < 64; ++tap)
    {
        int i = lo + tap;
        if (i > hi)
            break;

        float sample_pos = float(i) + 0.5;
        float x = abs((center - sample_pos) / aa_scale);
        float weight = 0.0;

        if (x < 1.0)
        {
            weight = ((1.5 * x - 2.5) * x) * x + 1.0;
        }
        else if (x < 2.0)
        {
            weight = ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
        }

        if (weight != 0.0)
        {
            vec2 uv = vec2(sample_pos * HOOKED_pt.x, HOOKED_pos.y);
            accum += HOOKED_tex(uv) * weight;
            weight_sum += weight;
        }
    }

    if (abs(weight_sum) > 1e-6)
        return accum / weight_sum;

    return HOOKED_tex(HOOKED_pos);
}

//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w
//!HEIGHT HOOKED.h RESIZE_SCALE *
//!DESC Catmull-Rom Resize Adjustable Vertical v1.0
vec4 hook()
{
    float src_len = HOOKED_size.y;
    float dst_len = max(floor(src_len * RESIZE_SCALE), 1.0);

    float dst_pos = HOOKED_pos.y * dst_len;
    float center = dst_pos * src_len / dst_len;

    float aa_scale = max(src_len / dst_len, 1.0);
    float radius = 2.0 * aa_scale;

    int lo = int(floor(center - radius + 0.5));
    int hi = int(ceil(center + radius - 0.5));

    vec4 accum = vec4(0.0);
    float weight_sum = 0.0;

    for (int tap = 0; tap < 64; ++tap)
    {
        int i = lo + tap;
        if (i > hi)
            break;

        float sample_pos = float(i) + 0.5;
        float x = abs((center - sample_pos) / aa_scale);
        float weight = 0.0;

        if (x < 1.0)
        {
            weight = ((1.5 * x - 2.5) * x) * x + 1.0;
        }
        else if (x < 2.0)
        {
            weight = ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
        }

        if (weight != 0.0)
        {
            vec2 uv = vec2(HOOKED_pos.x, sample_pos * HOOKED_pt.y);
            accum += HOOKED_tex(uv) * weight;
            weight_sum += weight;
        }
    }

    if (abs(weight_sum) > 1e-6)
        return accum / weight_sum;

    return HOOKED_tex(HOOKED_pos);
}
