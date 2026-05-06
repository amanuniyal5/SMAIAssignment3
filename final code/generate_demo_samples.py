"""
generate_demo_samples.py
────────────────────────
Creates synthetic-looking leaf placeholder images for the demo.
Run once to populate sample_images/ with test images you can click in the UI.

Usage:
    python generate_demo_samples.py
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

OUT = Path("sample_images")
OUT.mkdir(exist_ok=True)

SAMPLES = [
    ("tomato_late_blight",   (30, 90, 40),  [(100,120), (180,80), (60,160)], "dark"),
    ("potato_early_blight",  (60, 120, 50), [(130,100), (70,170), (190,140)], "spots"),
    ("corn_gray_leaf_spot",  (80, 140, 60), [(90,90), (160,130)], "stripe"),
    ("apple_scab",           (50, 110, 45), [(120,80), (80,150), (170,100)], "scab"),
    ("tomato_healthy",       (40, 160, 60), [], "healthy"),
    ("grape_black_rot",      (35, 95, 40),  [(100,100), (160,120), (130,170)], "dark"),
]

rng = np.random.RandomState(42)

def make_leaf(bg_color, spots, style, size=(300, 300)):
    img = Image.new("RGB", size, (30, 30, 20))
    draw = ImageDraw.Draw(img)

    # Leaf body (ellipse)
    r, g, b = bg_color
    draw.ellipse([20, 30, 280, 270], fill=(r, g, b, 255))

    # Leaf veins
    for i in range(5):
        x = 150 + rng.randint(-10, 10)
        y0 = 40 + i * 40
        draw.line([(x, y0), (x + rng.randint(-40, 40), y0 + 35)], fill=(max(r-20,0), max(g-30,0), max(b-20,0)), width=2)

    # Disease spots / patterns
    for (sx, sy) in spots:
        if style == "spots":
            for _ in range(4):
                ox, oy = rng.randint(-15, 15), rng.randint(-15, 15)
                draw.ellipse([sx+ox-8, sy+oy-8, sx+ox+8, sy+oy+8],
                             fill=(90, 60, 20))
        elif style == "dark":
            draw.ellipse([sx-15, sy-15, sx+15, sy+15], fill=(20, 40, 15))
            draw.ellipse([sx-8,  sy-8,  sx+8,  sy+8],  fill=(80, 50, 10))
        elif style == "stripe":
            draw.rectangle([sx-30, sy-4, sx+30, sy+4], fill=(160, 140, 80))
        elif style == "scab":
            draw.ellipse([sx-10, sy-10, sx+10, sy+10], fill=(60, 40, 10))
            draw.ellipse([sx-5,  sy-5,  sx+5,  sy+5],  fill=(100, 80, 30))

    img = img.filter(ImageFilter.GaussianBlur(1.2))
    # Noise
    arr = np.array(img, dtype=np.float32)
    arr += rng.normal(0, 6, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

for name, bg, spots, style in SAMPLES:
    img = make_leaf(bg, spots, style)
    path = OUT / f"{name}.jpg"
    img.save(path, quality=90)
    print(f"  ✓ {path}")

print(f"\nGenerated {len(SAMPLES)} sample images in {OUT}/")
