/*
    Nearest Resize Adjustable for cHiDeScaler-Neo v1.0
    -------------------------
    - Adjustable resize from 0.25x to 4.00x.
    - Two separable passes: horizontal then vertical.
    - Preserves hard pixel edges with no interpolation or antialiasing.
    - Kernel logic is fully inline for Neo's per-pass parser.
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
//!DESC Nearest Resize Adjustable Horizontal v1.0
vec4 hook()
{
    float src_len = HOOKED_size.x;
    float dst_len = max(floor(src_len * RESIZE_SCALE), 1.0);

    float dst_pos = HOOKED_pos.x * dst_len;
    float center = dst_pos * src_len / dst_len;
    float nearest_pos = floor(center) + 0.5;
    nearest_pos = clamp(nearest_pos, 0.5, max(src_len - 0.5, 0.5));

    vec2 uv = vec2(nearest_pos * HOOKED_pt.x, HOOKED_pos.y);
    return HOOKED_tex(uv);
}

//!HOOK MAIN
//!BIND HOOKED
//!WIDTH HOOKED.w
//!HEIGHT HOOKED.h RESIZE_SCALE *
//!DESC Nearest Resize Adjustable Vertical v1.0
vec4 hook()
{
    float src_len = HOOKED_size.y;
    float dst_len = max(floor(src_len * RESIZE_SCALE), 1.0);

    float dst_pos = HOOKED_pos.y * dst_len;
    float center = dst_pos * src_len / dst_len;
    float nearest_pos = floor(center) + 0.5;
    nearest_pos = clamp(nearest_pos, 0.5, max(src_len - 0.5, 0.5));

    vec2 uv = vec2(HOOKED_pos.x, nearest_pos * HOOKED_pt.y);
    return HOOKED_tex(uv);
}
