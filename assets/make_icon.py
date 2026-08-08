# cHiDeScaler-Neo icon: gunmetal metallic emblem ONLY (no plate, no
# background — everything outside the metal is transparent).
# C-ring + up arrow + bar steps, rendered 4x supersampled.
import math
from PIL import Image, ImageDraw

S = 4
N = 1024 * S
OUT = 1024

def make():
    def s(v):
        return v * S

    cx, cy = s(470), s(540)
    R_OUT, R_IN = s(330), s(198)
    shaft_w = s(118)
    shaft_x = cx + s(285)
    shaft_top = cy - s(250)
    head_w, head_h = s(122), s(170)
    apex_y = shaft_top - head_h
    tile = s(105)
    gap = s(24)
    bars_bottom = cy + s(300)

    emblem = Image.new("L", (N, N), 0)
    e = ImageDraw.Draw(emblem)

    # ring with an upper-right gap for the arrow
    e.ellipse([cx - R_OUT, cy - R_OUT, cx + R_OUT, cy + R_OUT], fill=255)
    e.ellipse([cx - R_IN, cy - R_IN, cx + R_IN, cy + R_IN], fill=0)
    wedge = [(cx, cy)]
    for a in range(-58, 49, 2):
        rad = math.radians(a)
        wedge.append((cx + 1.6 * R_OUT * math.cos(rad), cy + 1.6 * R_OUT * math.sin(rad)))
    e.polygon(wedge, fill=0)

    # arrow shaft + head
    e.rounded_rectangle([shaft_x - shaft_w // 2, shaft_top,
                         shaft_x + shaft_w // 2, cy + s(60)], radius=s(14), fill=255)
    e.polygon([(shaft_x - head_w, shaft_top),
               (shaft_x + head_w, shaft_top),
               (shaft_x, apex_y)], fill=255)

    # clear a uniform margin around the bar area so the ring never touches it
    pad = s(30)
    bars_left = shaft_x - 2 * (tile + gap) - tile // 2
    bars_top = bars_bottom - 2 * tile - gap
    e.rounded_rectangle([bars_left - pad, bars_top - pad,
                         shaft_x + tile // 2 + pad, bars_bottom + pad],
                        radius=s(16), fill=0)

    # staircase bars
    for cxx, n_tiles in [(shaft_x - 2 * (tile + gap), 1),
                         (shaft_x - (tile + gap), 2),
                         (shaft_x, 2)]:
        for i in range(n_tiles):
            y1 = bars_bottom - i * (tile + gap)
            e.rounded_rectangle([cxx - tile // 2, y1 - tile, cxx + tile // 2, y1],
                                radius=s(12), fill=255)

    # gunmetal vertical gradient
    met = Image.new("RGBA", (N, N))
    md = ImageDraw.Draw(met)
    stops = [(0.00, (232, 235, 238)),
             (0.40, (170, 176, 183)),
             (0.62, (118, 125, 132)),
             (1.00, (74, 81, 89))]
    for y in range(N):
        t = y / (N - 1)
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= t <= t1:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0
                col = tuple(int(c0[k] + (c1[k] - c0[k]) * f) for k in range(3))
                break
        md.line([(0, y), (N, y)], fill=col + (255,))

    img = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    img.paste(met, (0, 0), emblem)

    # crop to content, center on a square canvas with a small margin
    bbox = img.getbbox()
    img = img.crop(bbox)
    side = max(img.width, img.height)
    margin = int(side * 0.04)
    canvas = Image.new("RGBA", (side + 2 * margin, side + 2 * margin), (0, 0, 0, 0))
    canvas.paste(img, (margin + (side - img.width) // 2, margin + (side - img.height) // 2))

    out = canvas.resize((OUT, OUT), Image.LANCZOS)
    out.save("icon_1024.png")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    out.resize((256, 256), Image.LANCZOS).save(
        "cHiDeScaler-Neo.ico", sizes=[(x, x) for x in sizes])
    out.resize((64, 64), Image.LANCZOS).save("icon_64.png")
    out.resize((256, 256), Image.LANCZOS).save("icon_256.png")
    print("written: icon_1024.png / cHiDeScaler-Neo.ico / icon_64.png / icon_256.png")

if __name__ == "__main__":
    make()
