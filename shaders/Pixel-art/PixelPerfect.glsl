/*
    Pixel Perfect for cHiDeScaler-Neo
    ---------------------------------
    Final-display integer scaling for pixel art.

    Behavior:
    - Uses the pre-display-resize MAIN texture directly.
    - If MAIN fits inside the final OUTPUT at 1x or larger, selects the
      largest integer scale that fits on both axes.
    - Centers the integer-scaled image and fills unused pixels with black.
    - If MAIN is larger than OUTPUT on either axis, falls back to a normal
      nearest-neighbor fit to OUTPUT so the image is not cropped.

    Intended to be used by itself for true pixel-preserving presentation.
    Other filters placed before it remain part of MAIN and are therefore
    preserved; the integer scaling is applied to their result.
*/

//!HOOK OUTPUT
//!BIND HOOKED
//!BIND MAIN
//!DESC Pixel Perfect

vec4 pixel_perfect_fetch(ivec2 p)
{
    ivec2 size_i = max(ivec2(MAIN_size), ivec2(1));
    p = clamp(p, ivec2(0), size_i - ivec2(1));
    return texelFetch(MAIN_raw, p, 0);
}

vec4 hook()
{
    vec2 src_size = max(MAIN_size, vec2(1.0));
    vec2 dst_size = max(HOOKED_size, vec2(1.0));

    // Integer destination pixel coordinate (0 .. dst_size-1).
    ivec2 dst_px = ivec2(clamp(
        floor(HOOKED_pos * dst_size),
        vec2(0.0),
        dst_size - vec2(1.0)
    ));

    // Largest common integer upscale that fits in the final display area.
    float integer_scale = floor(min(dst_size.x / src_size.x,
                                    dst_size.y / src_size.y));

    if (integer_scale >= 1.0) {
        vec2 scaled_size = src_size * integer_scale;

        // Pixel-aligned centering. If an odd number of pixels remains, the
        // opposite side simply receives the one extra black pixel.
        ivec2 origin = ivec2(floor((dst_size - scaled_size) * 0.5));
        ivec2 extent = ivec2(scaled_size);

        if (dst_px.x < origin.x || dst_px.y < origin.y ||
            dst_px.x >= origin.x + extent.x ||
            dst_px.y >= origin.y + extent.y) {
            return vec4(0.0, 0.0, 0.0, 1.0);
        }

        ivec2 local_px = dst_px - origin;
        ivec2 src_px = ivec2(floor(vec2(local_px) / integer_scale));
        return pixel_perfect_fetch(src_px);
    }

    // MAIN is already larger than the display area. Preserve Neo's normal
    // no-crop behavior by fitting it to OUTPUT with nearest-neighbor sampling.
    vec2 src_pos = (vec2(dst_px) + vec2(0.5)) * src_size / dst_size;
    ivec2 src_px = ivec2(floor(src_pos));
    return pixel_perfect_fetch(src_px);
}
