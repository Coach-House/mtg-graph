"""Render a still of the 2D MTG embedding as a hero image for the splash page.

Reads public/2d.json (point_positions + point_colors, the same data the 2D
viewer draws) and writes public/hero-2d.png. Run manually after the data
changes:  uv run python scripts/render_hero.py
"""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "public" / "2d.json"
OUT = ROOT / "public" / "hero-2d.png"

SIZE = 2200          # output is SIZE x SIZE px (data is square)
MARGIN = 40          # px of empty border inside the canvas
BG = np.array([6, 7, 9], dtype=np.float32)   # near-black, matches page
POINT_BRIGHTNESS = 4.2   # per-point light multiplier (raise = brighter dots)
KERNEL_SIGMA = 1.35   # soft dot radius in px


def gaussian_kernel(sigma):
    """Return (offsets, weights) for a 5x5 normalized gaussian splat."""
    offs = []
    ws = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            w = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            offs.append((dy, dx))
            ws.append(w)
    ws = np.array(ws, dtype=np.float32)
    ws /= ws.sum()
    return offs, ws


def main():
    data = json.loads(SRC.read_text())
    pos = np.asarray(data["point_positions"], dtype=np.float32).reshape(-1, 2)
    col = np.asarray(data["point_colors"], dtype=np.float32).reshape(-1, 4)
    assert pos.shape[0] == col.shape[0], "positions and colors length mismatch"

    # Scale data coords into [MARGIN, SIZE - MARGIN] preserving the square aspect.
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    span = (hi - lo).max()
    scale = (SIZE - 2 * MARGIN) / span
    xy = (pos - lo) * scale + MARGIN
    xi = xy[:, 0].astype(np.int32)
    yi = xy[:, 1].astype(np.int32)

    # Per-point light = rgb (0..1) * alpha * 255 * brightness.
    light = col[:, :3] * (col[:, 3:4] * 255.0 * POINT_BRIGHTNESS)

    acc = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    offsets, weights = gaussian_kernel(KERNEL_SIGMA)
    for (dy, dx), w in zip(offsets, weights):
        ys = np.clip(yi + dy, 0, SIZE - 1)
        xs = np.clip(xi + dx, 0, SIZE - 1)
        contrib = light * w
        np.add.at(acc, (ys, xs), contrib)

    # Compose additively over the background and clip to 8-bit.
    out = BG[None, None, :] + acc
    out = np.clip(out, 0, 255).astype(np.uint8)

    Image.fromarray(out, mode="RGB").save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB), {pos.shape[0]} points")


if __name__ == "__main__":
    main()
