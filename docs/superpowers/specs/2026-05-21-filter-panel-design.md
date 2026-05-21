# Filter & Overlay Panel — Phase 1 Design

**Status:** Approved
**Date:** 2026-05-21
**Author:** Zach Wardlaw + Claude

## Goal

Add a "Filters & Overlays" section to both viewers (2D Cosmograph, 3D 3d-force-graph) plus an LLM-driven cluster label upgrade. All Phase 1 features work on the existing 5k card set — no infrastructure change. Phase 2 (scale to 37k + OpenAI re-embed) is a separate effort.

## Features in scope

### A. Color overlays (recolor existing points by a different axis)
- **Color identity** (current default)
- **EDHREC rank** — popularity heatmap (red = top-played, blue = obscure)
- **Price USD** — log-scale heatmap (cheap → expensive)
- **Card type** — categorical by primary type (Creature / Instant / Sorcery / Artifact / Enchantment / Land / Planeswalker)
- **Mana value** — gradient 0 → 7+

### B. Filters (hide points that don't match)
- **Format filter** — checkboxes for Modern / Pioneer / Legacy / Vintage / Commander / Standard / Pauper / Brawl (multi-select; show cards legal in at least one selected format, or "any" if none selected)
- **Set filter** — searchable dropdown ("type to filter") of all sets present in the dataset; multi-select
- **Mana value filter** — range slider 0–10+ (any CMC outside the range is hidden)
- **Hide-by-color toggles** — color legend chips become clickable to toggle visibility of that color bucket

### C. Substring oracle text search
- Extend the existing search input: queries now match against both `name` *and* `oracle_text`
- Case-insensitive substring match
- Multi-word query = AND across all words (so "flying lifelink" matches cards with both words in name/text)
- Search results highlight in the dropdown as before

### D. LLM-named clusters
- Replace TF-IDF labels ("ninja / ninjutsu") with one-line LLM summaries ("Ninjutsu & ninja tribal")
- Adds an opt-in pipeline stage that calls OpenAI's chat completions (`gpt-4o-mini`) once per labeled cluster
- Caches results in `output/cluster_labels.json` (same file, different content)
- Skipped gracefully if `OPENAI_API_KEY` not set; falls back to TF-IDF labels

## Data additions

To support filters/overlays, `load_and_filter.py` needs to keep three more fields:

- `cmc` (mana value, float)
- `legalities` (dict mapping format → "legal"/"not_legal"/"banned"/etc.)
- `set` (set code, e.g. "blb" — we already have `set_name` but the short code is more compact in filter UI)

Then `_build_cosmograph_data()` and `_build_forcegraph_data()` include these in `point_meta`.

## UI layout

A new collapsible "Filters" section in the side panel between the mode-switch row and the search input. Visible state by default; clicking the header collapses it.

```
┌───────────────────────────────────┐
│ 2D                  Switch to 3D →│  ← mode row (existing)
├───────────────────────────────────┤
│ ▼ Filters & Overlays              │  ← new section
│                                   │
│   Color by:    [Color identity ▾] │
│                                   │
│   Format:                         │
│   ☑ Modern  ☑ Commander           │
│   ☐ Standard ☐ Pauper             │
│   ☐ Legacy   ☐ Vintage            │
│   ☐ Pioneer  ☐ Brawl              │
│                                   │
│   Set:        [Type to filter…  ] │
│                                   │
│   Mana value: ●────●        0–7   │
│                                   │
│   Colors visible:                 │
│   [W] [U] [B] [R] [G] [M] [C]     │  ← click to toggle
├───────────────────────────────────┤
│ [Search card name or text…      ] │  ← existing, scope expanded
├───────────────────────────────────┤
│ ...card detail + neighbors...     │
└───────────────────────────────────┘
```

## Architecture

Both viewers share the same data; filter/overlay logic lives in JS per viewer.

### Per-card "visible" state

Each card has a base `visible` boolean derived from currently-active filters. On filter change:
1. Recompute visibility for all 5000 cards
2. Apply to the viewer (Cosmograph: alpha channel of `pointColors`; 3d-force-graph: per-sprite `visible` or material opacity)

### Color overlay

Stored as the current "color axis." On change:
1. Build a fresh `Float32Array` of RGBA colors for all cards
2. Cosmograph: `setPointColors(arr)`
3. 3d-force-graph: re-invoke `nodeColor(accessor)` (the closure picks up the new axis from shared state)

### Search expansion

The existing search dropdown indexes against `NAME_INDEX`. Extend to also match against `oracle_text` (lowercased once at init). When a match is purely from oracle text (not name), prefix the result item with a small "[text]" indicator so the user knows why it matched.

### LLM cluster labels

`cluster.py` grows an optional final step:
1. For each cluster with ≥ MIN_LABEL_SIZE cards, gather: top 5 cards by EDHREC rank within the cluster + their name + type + first line of oracle text + the TF-IDF terms as a fallback hint
2. Send to `gpt-4o-mini` with prompt: "Summarize this group of MTG cards as a short label (3-6 words). Use mechanic names or tribal names when applicable. Examples: 'Counterspells', 'Ninja tribal', 'Treasure tokens & ramp'."
3. Use returned label; cache by (cluster signature, model) so a re-run skips already-labeled clusters

## File output

Same filenames as today. Sizes will grow modestly with the new metadata fields (~10% per card). Total HTML stays under 12 MB.

## Success criteria

- Filter panel renders in side panel; collapsible
- All filter controls update the viewer in <100ms
- Color overlay switches without re-rendering all 5000 nodes from scratch (just color array update)
- Searching "flying" highlights all flying creatures in the dropdown
- LLM cluster labels read more naturally than the TF-IDF ones (subjective; we'll eyeball)
- 3D click + orbit still works (filter changes can't break controls per past Plotly lessons — we're using Cosmograph and three.js which handle runtime updates fine)

## Out of scope

- Semantic search (Phase 3)
- 37k card scale (Phase 2)
- Deck-aware mode (Phase 3)
- Multi-select / path between cards (Phase 3)
- RAG over rules text (Phase 3)
- Embedding model comparison (deferrable)
- Save / share filter state via URL (future polish)
