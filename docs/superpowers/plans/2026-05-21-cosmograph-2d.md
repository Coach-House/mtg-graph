# Cosmograph 2D Viewer Implementation Plan

**Goal:** Swap the 2D viewer from Plotly to Cosmograph (`@cosmos.gl/graph` v2.6.4) for native graph interactivity. Keep Plotly 3D unchanged.

**Architecture:** Same Python pipeline; visualize.py grows a Cosmograph rendering path that emits typed-array data + a new HTML template. The 3D path is untouched. The existing side panel (search, card details, neighbors grid, mode switch) wraps both viewers via a `selectCard(oid)` integration point.

**Tech stack:** Cosmograph via ESM CDN, vanilla JS, plain HTML. Python: pandas, numpy, json.

**Spec:** `docs/superpowers/specs/2026-05-21-cosmograph-2d-design.md`

---

## Task 1: Cosmograph 2D template skeleton

**Files:**
- Create: `src/mtg_graph/templates/viewer_2d.html`

Build a hand-rolled HTML with:
- Same layout/CSS as `viewer.html` (split: chart left, side panel right)
- `<canvas id="chart">` for Cosmograph instead of a Plotly div
- Same side panel HTML (search, card details, neighbors grid, mode switch)
- ESM import: `import { Graph } from "https://esm.sh/@cosmos.gl/graph@2.6.4"`
- Three JSON `<script>` blocks for: `pointMeta` (per-card detail dict), `linksByOid` (KNN map for the neighbors grid), and `clusterLabels`
- Three typed-array `<script>` blocks for: `pointPositions`, `pointColors`, `links` — serialized as base64-encoded Float32Array bytes for efficient transfer

Reuse the side panel JS verbatim from `viewer.html` (search, renderPanel, renderNeighbors, mode toggle button, escape helpers).

The new JS surface is just:
- `initGraph()` — instantiates Cosmograph with the typed arrays, registers `onClick`
- `selectCard(oid)` — calls Cosmograph's `focusPointByIndex` + `setOutlinedPointsByIndices(neighborIndices)` AND renders the side panel
- `clearSelection()` — calls Cosmograph's `unfocusPoint()` + `setOutlinedPointsByIndices([])`
- `updateClusterLabels()` — repositions HTML label divs via Cosmograph's `spaceToScreenPosition`; called on init and on Cosmograph's `onZoom`/`onSimulationTick` (we won't simulate, but pan/zoom events trigger this)

---

## Task 2: Cosmograph data serialization in visualize.py

**Files:**
- Modify: `src/mtg_graph/visualize.py`

Add functions:
- `_normalize_coords(x, y, target_extent=2000)` — scales UMAP coords to ±target_extent so they sit cleanly in Cosmograph's 4096 spaceSize
- `_serialize_cosmograph_arrays(merged, model_key, knn)` — returns dict with:
  - `point_positions_b64`: base64 of Float32Array([x1,y1,x2,y2,...])
  - `point_colors_b64`: base64 of Float32Array([r,g,b,a, ...]) — RGBA per point from MTG color identity
  - `links_b64`: base64 of Float32Array([src,tgt, ...]) — 5000 × K=20 = 100k edges
  - `point_meta`: list of {oracle_id, name, type_line, mana_cost, ...} in order (matches positions index)
  - `cluster_labels`: list of {x_normalized, y_normalized, text}
  - `name_to_idx`: dict for search lookups
  - `oid_to_idx`: dict for KNN lookup → index conversion
  - `idx_to_oid`: list (inverse of oid_to_idx)

Add render function `_render_2d_cosmograph(merged, model_key, cluster_labels_dict, knn, output_path, counterpart_href, counterpart_label)` that fills `viewer_2d.html` template placeholders.

Update `run()` so 2D path goes through Cosmograph render, 3D path keeps Plotly as-is.

Delete dead Plotly-2D code paths: `_build_2d_traces`, `_overlay_traces_2d`, `_cluster_label_trace_2d`, `_layout_2d`.

---

## Task 3: Execute pipeline + verify in browser

- Re-render via `uv run mtg-graph-poc --open 2d`
- Manual checks:
  1. 5000 points render at UMAP positions
  2. Click a card → focus + neighbors outlined + edges visible
  3. Search returns matches, click selects + focuses
  4. Mode switch link goes to 3D Plotly HTML
  5. Side panel displays card image + details + neighbors grid
  6. Pan/zoom feels smooth
- Iterate on visual polish (point sizes, colors, link opacity)

---

## Task 4: Commit

```bash
git add -A
git commit -m "Swap 2D viewer from Plotly to Cosmograph

Native graph rendering with on-select neighbor highlighting + edge drawing
that Plotly's general-purpose plotting fundamentally couldn't deliver
without breaking its own controls. 3D viewer (Plotly) unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Out of scope (later)

- 3D Cosmograph (library is 2D-only; would need deck.gl or three.js)
- URL deep-linking to a selected card
- Edge filtering / pruning
