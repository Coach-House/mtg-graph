# Planeswalk — landing splash design

**Date:** 2026-06-11
**Status:** Approved (pending spec review)

## Goal

Replace the current `public/index.html` with a cleaner, more shareable splash
page for the MTG card-embedding project. The page gives a visitor just enough
background to understand what they're looking at, shows a still of the 2D
embedding, and sends them into the viewers. It is not a marketing page — minimal
content, single screen, clean.

This work also establishes the project's brand name: **Planeswalk** (a real
Magic: The Gathering term — the act of traveling between planes — that doubles as
the action of moving through the map).

## Scope

- A single static `public/index.html`. No build step, no framework.
- A pre-rendered static hero image of the 2D embedding, shipped in `public/`.
- A one-off script that generates that hero image from existing data.

Out of scope: changes to the 2D/3D viewers, the data pipeline, or `vercel.json`.

## Layout

Composition **B — Split** (chosen over a full-bleed hero). Two columns on
desktop, vertically centered, non-scrolling at common viewport heights.

**Left column (~42% width)** — quiet content column:
- Wordmark: **Planeswalk**
- Background blurb (2 sentences):
  > A map of 33,683 Magic: The Gathering cards and their relationship to each
  > other. Cards that sit close together share mechanics, themes, or text.
- Two entry buttons:
  - **2D map →** — solid gold (primary)
  - **3D space →** — outline (secondary)
  - Link to `2d.html` and `3d.html` respectively.
- Footer mark: **Produced by Coach House** — links to `https://coachhouse.so`.

**Right column (~58% width)** — full-height still of the 2D embedding:
- Static PNG, colored by card color identity (matching the viewer palette).
- Subtle vignette / inner shadow to seat it against the dark frame.

**Mobile (narrow viewports):** Columns stack — wordmark, blurb, and buttons on
top; the embedding image as a band beneath. Minor scrolling on small phones is
acceptable. Coach House mark remains at the bottom.

## Visual style

Reuse the existing viewer theme so the splash and viewers feel like one product:
- Background `#0a0a0a`, panel `#161616`, border `#2a2a2a`
- Text `#e8e8e8`, dim `#888`
- Accent gold `#c9a227`, bright `#fbbf24`
- System font stack (`-apple-system, BlinkMacSystemFont, …`)

## Hero image

Pre-rendered **static PNG**, generated once from `public/2d.json` (or the
upstream `output/` data the JSON is built from) by a small standalone script.

- Rendered at ~2× target display size for retina sharpness.
- Points colored by color identity using the same mapping as the viewers.
- Dark background matching the page so it blends into the right column.
- Output written to `public/` (e.g. `public/hero-2d.png`).

Rationale: a live in-page render would re-fetch the ~95 MB `2d.json` payload,
which defeats the goal of a light, fast, shareable page. A static image keeps the
splash to a few KB of HTML plus one optimized PNG, and it screenshots cleanly.

The generator script is a dev-time tool (not part of any deploy build). It reads
the point coordinates + color identity and draws the scatter. Exact tooling
(Python via the existing `.venv`, matplotlib/Pillow, or a headless canvas)
to be decided in the implementation plan based on what the data files expose.

## Components

1. **`public/index.html`** — the splash. Self-contained HTML + inline CSS, no JS
   required for the core page (matches the current file's approach).
2. **Hero generator script** — standalone, run once, writes `public/hero-2d.png`.
   Lives alongside the pipeline source (e.g. under `src/`) or `scripts/`.
3. **`public/hero-2d.png`** — committed output, served statically.

## Data flow

Build-time (manual, once): `2d` embedding data → generator script → `hero-2d.png`.
Runtime: visitor loads `index.html` (tiny) + `hero-2d.png` → clicks a button →
enters `2d.html` / `3d.html` (the existing large viewers, unchanged).

## Error handling / edge cases

- Static page — no runtime error surface beyond a missing image. Ensure
  `hero-2d.png` is committed so the right column is never blank.
- Provide descriptive `alt` text on the image for accessibility.
- Buttons are plain `<a>` links — work without JS.

## Testing / verification

- Open `public/index.html` locally; confirm layout, both links resolve, Coach
  House link points to coachhouse.so.
- Check non-scroll on a typical desktop viewport and graceful stack on a narrow
  one.
- Confirm the hero image renders and matches the viewer's color identity palette.

## Open questions for the plan

- Which data source the generator reads (the deployed `public/2d.json` vs the
  upstream `output/` artifacts) and which drawing library — resolved during
  planning by inspecting the actual data files.
