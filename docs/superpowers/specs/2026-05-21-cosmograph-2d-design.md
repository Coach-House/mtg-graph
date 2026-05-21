# Cosmograph 2D + 3d-force-graph 3D Viewer — Design

**Status:** Approved (scope expanded to include 3D after initial 2D swap)
**Date:** 2026-05-21
**Author:** Zach Wardlaw + Claude

## Goal

Replace Plotly with purpose-built graph libraries for both 2D and 3D so we get native graph features that Plotly fundamentally couldn't deliver:

1. Click a card → its 20 nearest neighbors light up + drawn edges connect them ("spiderweb" effect from the IG video)
2. Smooth performance at 5k–100k+ nodes/edges
3. Native click/highlight/hover with no `restyle`-breaks-the-renderer footguns

**2D:** Cosmograph (`@cosmos.gl/graph`) — WebGL graph viz.
**3D:** `3d-force-graph` (three.js-based) — uses node `fx/fy/fz` to pin UMAP positions, no force simulation drift. Cosmograph itself is 2D-only, so `3d-force-graph` is the right tool for 3D parity.

## Why Cosmograph

Plotly Scatter / Scattergl required us to bolt graph interactivity onto a general-purpose plotting library. Every interactive feature (selection, neighbor highlight, edge drawing on click) hit edges of Plotly's design. Cosmograph is purpose-built for this exact use case:

- `setPointPositions(Float32Array)` accepts our UMAP coords directly
- `setLinks(Float32Array)` accepts our KNN edge list (100k edges fine)
- `enableSimulation: false` keeps the UMAP layout (no force-directed drift)
- `onClick` + `focusedPointIndex` / `outlinedPointIndices` — native selection
- `getNeighboringPointIndices()` queries the graph topology

## Scope

**In scope:**
- New 2D viewer template using Cosmograph via ESM CDN (`https://esm.sh/@cosmos.gl/graph@2.6.4`)
- Visualize stage emits both 2D (Cosmograph) and 3D (Plotly) HTMLs as before
- Existing side panel (search, card details, neighbors grid, mode switcher) stays intact and works against both viewers
- Cluster labels: HTML overlays positioned via Cosmograph's screen projection
- Native edge rendering: 5000 × 20 = 100k KNN edges rendered at very low baseline opacity (~0.02); selected card's outgoing edges brighten to ~0.9

**Out of scope:**
- 3D — Cosmograph doesn't do 3D; Plotly 3D stays as-is
- Changes to the Python pipeline (load / embed / reduce / cluster / knn)
- Multi-model viewer (still one model per HTML page)

## Architecture

```
Python pipeline (unchanged)
    │
    ▼
visualize.py (new path for 2D)
    │
    ├─ existing _build_2d_traces, _layout_2d — DELETED
    ├─ new _serialize_for_cosmograph() — emits:
    │     * pointPositions: Float32Array of [x1,y1,x2,y2,...]
    │     * pointColors:    Float32Array of [r,g,b,a, ...]
    │     * pointSizes:     Float32Array of constant size
    │     * links:          Float32Array of [src,tgt, ...]
    │     * pointMeta:      JS array of {oracle_id, name, type, ...} for lookup
    │     * clusterLabels:  [{x, y, text}, ...]
    └─ renders viewer_2d.html template

3D path unchanged → renders viewer.html template (Plotly)
```

## Data marshaling

UMAP coords from `coords.parquet`: `x_local`, `y_local` (range typically -10 to +20).
Cosmograph default `spaceSize` is 4096. We normalize coords to ±2000 range so they sit comfortably in Cosmograph's space.

KNN edges: for each card `i` with neighbors `[n1, n2, ..., n20]`, emit pairs `(i, idx_of_n1)`, `(i, idx_of_n2)`, ... Total: 100k edges as `Float32Array(200000)`.

## File output

Same names as today:
- `output/mtg_graph_poc.html` — Cosmograph 2D viewer (was Plotly 2D)
- `output/mtg_graph_poc_3d.html` — Plotly 3D viewer (unchanged)

## Side panel

Unchanged. The panel works against a `selectCard(oid)` JS function which is the integration point. We re-implement that function to call Cosmograph's API instead of Plotly's.

## Cluster labels

Cosmograph doesn't render arbitrary text annotations at world coordinates. We use HTML divs positioned absolutely over the canvas, updated on each frame via Cosmograph's `getPointPositionByIndex` or by subscribing to view-transform events.

For v1, simpler: position labels using Cosmograph's `spaceToScreenPosition()` once on init and again on every camera change (zoom/pan events).

## Success criteria

Open the new 2D HTML:
1. 5000 points render in their UMAP positions
2. Click any card → that card and its 20 neighbors highlight + edges between them visible
3. Search bar still works, selects a card from the dropdown
4. Side panel shows full card details on selection
5. Click another card → previous highlight clears, new one appears
6. Pan/zoom smooth at 60 fps
7. 3D viewer (Plotly) still works as before — toggle link between 2D and 3D works both ways

## Out-of-scope follow-ups (later)

- 3D Cosmograph equivalent (would need a different library — deck.gl, three.js, or sigma.js)
- Custom edge filtering (e.g., "only show edges shorter than X")
- URL state persistence (deep link to a selected card)
