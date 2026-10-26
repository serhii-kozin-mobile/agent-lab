"""Screenshot degradations. Each condition is a function (img, rng) -> img.

Rendering + noise is what the agent "sees". All conditions are deterministic
given the rng, so a run is reproducible from its seed.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from table import W, H

CHIP_COLORS = [(200, 30, 30), (30, 60, 180), (20, 20, 20), (30, 140, 60), (230, 230, 230), (120, 40, 140)]


def jpeg(img, q):
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def scale(img, s):
    return img.resize((int(W * s), int(H * s)), Image.BILINEAR)


def blur(img, r):
    return img.filter(ImageFilter.GaussianBlur(r))


def chips(img, rng, n_min=1, n_max=3):
    """Animated chip stacks sliding across the centre of the table.

    They land anywhere in the band between the two players, so they can cover
    board cards, the pot or the bet plates, as they do mid-animation in a real
    client."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(n_min, n_max + 1)):
        cx = int(rng.integers(300, 660))
        cy = int(rng.integers(200, 420))
        col = CHIP_COLORS[rng.integers(len(CHIP_COLORS))]
        for k in range(int(rng.integers(2, 6))):
            y = cy - k * 5
            d.ellipse([cx - 16, y - 7, cx + 16, y + 7], fill=col, outline=(250, 250, 250), width=2)
    return img


CONDITIONS = {
    # name: (theme, fn(img, rng) -> img)
    "clean": ("classic", lambda im, r: im),
    "jpeg_q20": ("classic", lambda im, r: jpeg(im, 20)),
    "scale_0.5": ("classic", lambda im, r: scale(im, 0.5)),
    "blur_1.5": ("classic", lambda im, r: blur(im, 1.5)),
    "chip_overlap": ("classic", lambda im, r: chips(im, r)),
    "theme_four_color": ("four_color", lambda im, r: im),
    "theme_dark": ("dark", lambda im, r: im),
    "realistic_mix": ("classic", lambda im, r: jpeg(blur(scale(chips(im, r, 0, 2), 0.75), 0.6), 55)),
}
