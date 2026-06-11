# Planeswalk Splash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `public/index.html` with a clean, single-screen "Planeswalk" splash page (split layout) backed by a pre-rendered still of the 2D embedding.

**Architecture:** A one-off Python script reads `public/2d.json` (which already holds point positions and per-point RGBA colors) and renders an additive-glow scatter to `public/hero-2d.png`. A new self-contained static `public/index.html` displays that image in a right column beside a left content column. No build step, no runtime JS for the core page.

**Tech Stack:** Python 3.12 via `uv` (numpy + Pillow) for the image generator; plain HTML + inline CSS for the page.

---

## File Structure

- **Create** `scripts/render_hero.py` — standalone dev-time generator. Reads `public/2d.json`, writes `public/hero-2d.png`. One responsibility: turn embedding coords+colors into a hero image.
- **Create** `public/hero-2d.png` — committed output, served statically.
- **Modify (replace)** `public/index.html` — the splash page.
- **Modify** `pyproject.toml` + `uv.lock` — add `pillow` dependency.

Color/position data already lives in `public/2d.json`:
- `point_positions`: flat list, length 67366 = 33683 × 2, `[x0,y0,x1,y1,…]`, values in ~[200, 3896].
- `point_colors`: flat list, length 134732 = 33683 × 4, `[r,g,b,a,…]`, floats in [0,1].

---

## Task 1: Add the Pillow dependency

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Add pillow via uv**

Run:
```bash
uv add pillow
```
Expected: `pyproject.toml` gains `pillow` under dependencies; `uv.lock` updates; venv installs Pillow.

- [ ] **Step 2: Verify Pillow imports**

Run:
```bash
uv run python -c "import PIL, numpy; print('PIL', PIL.__version__, 'numpy', numpy.__version__)"
```
Expected: prints versions, e.g. `PIL 11.x.x numpy 2.4.6` — no ImportError.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: add pillow for hero image rendering"
```

---

## Task 2: Write the hero image generator

**Files:**
- Create: `scripts/render_hero.py`

This script is run manually, once, to (re)generate the hero. It loads the JSON,
scales positions into a square canvas, additively splats each point with a small
gaussian kernel weighted by its RGBA color, tone-maps over a near-black
background, and saves an optimized PNG.

- [ ] **Step 1: Create the script**

Create `scripts/render_hero.py`:
```python
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
POINT_BRIGHTNESS = 2.0   # per-point light multiplier (raise = brighter dots)
KERNEL_SIGMA = 1.1   # soft dot radius in px


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
```

- [ ] **Step 2: Run the generator**

Run:
```bash
uv run python scripts/render_hero.py
```
Expected: prints `wrote .../public/hero-2d.png (NNNN KB), 33683 points` and exits 0.

- [ ] **Step 3: Sanity-check the output**

Run:
```bash
uv run python -c "
from PIL import Image
im = Image.open('public/hero-2d.png')
print('size', im.size, 'mode', im.mode)
ex = im.convert('RGB').getextrema()
print('channel extrema', ex)
assert im.size == (2200, 2200)
assert max(hi for _, hi in ex) > 60, 'image looks empty/too dark'
print('OK')
"
```
Expected: `size (2200, 2200) mode RGB`, channel maxima well above 60 (bright dots present), prints `OK`. If it looks too dark/bright when opened, adjust `POINT_BRIGHTNESS` in the script and re-run.

- [ ] **Step 4: Eyeball it**

Run:
```bash
open public/hero-2d.png
```
Expected: a square dark image with colored clusters of points (white/blue/black/red/green/gold), recognizably the 2D embedding. Tweak `POINT_BRIGHTNESS` / `KERNEL_SIGMA` and re-run Step 2 if it reads muddy or too sparse.

- [ ] **Step 5: Commit**

```bash
git add scripts/render_hero.py public/hero-2d.png
git commit -m "feat: render Planeswalk hero image from 2D embedding"
```

---

## Task 3: Write the splash page

**Files:**
- Modify (replace): `public/index.html`

Self-contained HTML + inline CSS. Split layout on desktop, stacks on mobile.
Reuses the viewer theme (dark + gold). Links use `2d.html` / `3d.html` so they
work both when opening the file locally and on Vercel (whose `cleanUrls` still
serves those paths).

- [ ] **Step 1: Replace the file**

Replace the entire contents of `public/index.html` with:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Planeswalk — a map of 33,683 Magic cards</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="description" content="A map of 33,683 Magic: The Gathering cards and their relationship to each other." />
<style>
  :root {
    --bg: #0a0a0a; --bg-deep: #060709; --panel-border: #2a2a2a;
    --text: #e8e8e8; --text-dim: #888; --accent: #c9a227; --accent-bright: #fbbf24;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; height: 100%; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
  }
  .wrap { display: grid; grid-template-columns: 42% 58%; min-height: 100vh; }

  /* left: content */
  .content { display: flex; flex-direction: column; justify-content: center; padding: 56px 56px; max-width: 560px; }
  h1 { font-size: 52px; font-weight: 600; letter-spacing: -0.025em; margin: 0 0 18px; color: var(--accent-bright); }
  .blurb { font-size: 17px; line-height: 1.6; color: var(--text); margin: 0 0 8px; }
  .blurb .dim { color: var(--text-dim); font-size: 15px; }
  .btns { display: flex; gap: 12px; margin-top: 32px; }
  .btn {
    font-size: 15px; padding: 11px 22px; border-radius: 8px; text-decoration: none;
    border: 1px solid var(--accent); color: var(--accent-bright);
    background: rgba(201, 162, 39, 0.08); transition: transform .15s, background .15s;
  }
  .btn:hover { transform: translateY(-2px); }
  .btn.primary { background: var(--accent); color: #0a0a0a; font-weight: 600; }
  .btn.primary:hover { background: var(--accent-bright); }
  .credit { margin-top: 48px; font-size: 13px; color: var(--text-dim); }
  .credit a { color: var(--accent); text-decoration: none; }
  .credit a:hover { text-decoration: underline; }

  /* right: hero */
  .hero { position: relative; background: var(--bg-deep); overflow: hidden; }
  .hero img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .hero::after {
    content: ""; position: absolute; inset: 0; pointer-events: none;
    box-shadow: inset 0 0 120px 40px var(--bg-deep);
    background: radial-gradient(120% 100% at 60% 50%, transparent 55%, rgba(6,7,9,0.55) 100%);
  }

  @media (max-width: 820px) {
    .wrap { grid-template-columns: 1fr; min-height: 100vh; }
    .content { order: 1; padding: 44px 28px 32px; max-width: none; }
    .hero { order: 2; min-height: 46vh; }
    h1 { font-size: 40px; }
    .blurb { font-size: 16px; }
  }
</style>
</head>
<body>
<div class="wrap">
  <main class="content">
    <h1>Planeswalk</h1>
    <p class="blurb">A map of 33,683 Magic: The Gathering cards and their relationship to each other.</p>
    <p class="blurb dim">Cards that sit close together share mechanics, themes, or text.</p>
    <div class="btns">
      <a class="btn primary" href="2d.html">2D map →</a>
      <a class="btn" href="3d.html">3D space →</a>
    </div>
    <p class="credit">Produced by <a href="https://coachhouse.so">Coach House</a></p>
  </main>
  <div class="hero">
    <img src="hero-2d.png" alt="A scatter map of 33,683 Magic: The Gathering cards, each a colored point placed by semantic similarity and tinted by color identity." />
  </div>
</div>
</body>
</html>
```

- [ ] **Step 2: Open and verify locally**

Run:
```bash
open public/index.html
```
Expected in the browser:
- Left column shows "Planeswalk", the two blurb lines, two buttons, and "Produced by Coach House".
- Right column shows the hero image filling the column with a soft vignette.
- No vertical scrollbar at a normal desktop window height.

- [ ] **Step 3: Verify links and credit target**

Run:
```bash
grep -E 'href="(2d|3d)\.html"|coachhouse\.so' public/index.html
```
Expected: three matches — `2d.html`, `3d.html`, and `https://coachhouse.so`.

- [ ] **Step 4: Verify mobile stacking**

In the browser, narrow the window below ~820px (or use devtools device mode).
Expected: columns stack — content on top, hero image as a band beneath; layout
stays readable.

- [ ] **Step 5: Commit**

```bash
git add public/index.html
git commit -m "feat: Planeswalk split-layout splash page"
```

---

## Task 4: Final verification

**Files:** none (review only)

- [ ] **Step 1: Confirm the deployed file set is consistent**

Run:
```bash
ls -la public/index.html public/hero-2d.png && grep -c "Planeswalk" public/index.html
```
Expected: both files exist; `Planeswalk` appears at least once.

- [ ] **Step 2: Confirm git is clean**

Run:
```bash
git status --short
```
Expected: no uncommitted changes related to this work (the `.gitignore`/spec from
brainstorming were already committed separately).

---

## Self-Review

**Spec coverage:**
- Replace `index.html` with split splash → Task 3. ✓
- Name "Planeswalk", wordmark + 2-sentence blurb → Task 3 markup. ✓
- 2D-primary + 3D buttons linking the viewers → Task 3. ✓
- "Produced by Coach House" → coachhouse.so → Task 3, verified Step 3. ✓
- Static pre-rendered hero PNG from 2D data → Tasks 1–2. ✓
- Color identity matching the viewer → uses `point_colors` straight from `2d.json` (the viewer's own colors). ✓
- Dark/gold theme, system fonts → Task 3 CSS. ✓
- Mobile stacking → Task 3 media query, verified Step 4. ✓
- Non-scroll desktop → Task 3, verified Step 2. ✓
- Alt text for accessibility → Task 3 `<img alt>`. ✓
- Image committed so column never blank → Task 2 Step 5. ✓

**Placeholder scan:** No TBD/TODO; all code blocks are complete and runnable.

**Type consistency:** Script reads keys `point_positions` / `point_colors` (verified to exist with lengths 67366 / 134732). Output path `public/hero-2d.png` matches the `<img src="hero-2d.png">` in the page. Canvas size `2200` matches the Task 2 Step 3 assertion.
