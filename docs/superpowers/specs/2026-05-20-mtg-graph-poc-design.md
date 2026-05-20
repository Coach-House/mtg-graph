# MTG Graph — 5k Proof of Concept

**Status:** Approved (design phase)
**Date:** 2026-05-20
**Author:** Zach Wardlaw + Claude

## Goal

Validate that semantic embeddings of Magic: The Gathering card text produce intuitive 2D clusters before scaling to the full ~37k unique cards. The deliverable is a single self-contained HTML file with two side-by-side scatterplots comparing local vs. OpenAI embeddings.

Success means: opening the HTML, you can zoom/pan, hover any point to see card name + image, and eyeball-verify clusters are sensible (counterspells near counterspells, fetch lands near fetch lands, Slivers form their own island).

## Scope

**In scope:**
- Top 5,000 cards by `edhrec_rank` from Scryfall's `oracle-cards` dump
- Two embedding generators: local `all-mpnet-base-v2` and OpenAI `text-embedding-3-small`
- UMAP reduction to 2D, separately for each embedding
- Plotly side-by-side scatter, single HTML output
- Caching of embeddings on disk

**Out of scope:**
- The full 37k cards (POC only; production run is a separate effort)
- Drawn edges between cards (points-only scatter — see Section 3)
- Graph embeddings (Node2Vec / DeepWalk)
- Tests (visual output is the verification)
- Deployment / hosting

## Data source

`data/oracle-cards-20260520090335.json` — Scryfall's deduped oracle dump, 37,472 unique cards, 165MB. Already present in the repo.

## Architecture

Five stages, each a separate module under `src/mtg_graph/`. Each stage reads from disk and writes to disk so it can be rerun independently. A CLI entrypoint chains them.

```
data/oracle-cards-*.json
        │
        ▼
[1] load_and_filter.py    → output/cards_top5k.parquet
        │
        ▼
[2] build_text_profiles.py → adds text_profile column
        │
        ▼
[3] embed.py               → output/embeddings_local.npy
        │                    output/embeddings_openai.npy
        ▼
[4] reduce.py              → output/coords.parquet
        │                    (x_local, y_local, x_openai, y_openai)
        ▼
[5] visualize.py           → output/mtg_graph_poc.html
```

### Stage 1: Load and filter

Read the Scryfall JSON, filter out non-card layouts, sort by `edhrec_rank` ascending (lower rank = more played), take the top 5,000.

**Layouts kept:** `normal`, `transform`, `modal_dfc`, `split`, `adventure`, `saga`, `class`, `case`, `flip`, `leveler`, `meld`, `mutate`, `prototype`.

**Layouts dropped:** `art_series`, `token`, `double_faced_token`, `planar`, `scheme`, `emblem`, `vanguard`, `prepare`, `host`, `augment`.

Cards without `edhrec_rank` are dropped (they're typically art series, tokens, or never-played cards).

Output: `output/cards_top5k.parquet` with columns `oracle_id`, `name`, `mana_cost`, `type_line`, `oracle_text`, `keywords`, `colors`, `color_identity`, `rarity`, `edhrec_rank`, `image_small` (URL from `image_uris.small`).

### Stage 2: Build text profiles

For each card, construct a single string that the embedding model will consume. Format:

```
{name} | {type_line} | {mana_cost} | {oracle_text}
```

Pipe-separated, no labels. Empty fields (e.g., lands have no mana cost) are skipped, not left blank. Keywords are already present in `oracle_text` so we don't append them separately.

Output: same parquet plus a `text_profile` string column.

### Stage 3: Embed

Two embedders run in sequence and write separate `.npy` files.

**Local embedder:** `sentence-transformers/all-mpnet-base-v2`. 768-dim vectors. Batch size 64. Expected runtime: ~3–5 min for 5k cards on a Mac.

**OpenAI embedder:** `text-embedding-3-small`. 1536-dim vectors. One batch request (the model accepts up to 2048 inputs per call, so this needs 3 batches). Expected cost: ~$0.05.

**Caching:** each embedder writes two files — `embeddings_<model>.npy` and a sidecar `embeddings_<model>.json` containing `{"model_id": ..., "card_hash": ...}` where `card_hash` is a SHA-256 of the sorted oracle_id list. On rerun, the embedder reads the sidecar; if `model_id` and `card_hash` match the current run, it skips regeneration. If the top-5k set changes, the hash changes and the cache is invalidated.

**Graceful degradation:** if `OPENAI_API_KEY` is not set (checked via `python-dotenv` reading `.env`), skip the OpenAI side. Stage 5 then renders a single-panel HTML instead of two.

### Stage 4: Reduce

UMAP from 768d (local) and 1536d (OpenAI) down to 2D. Parameters:

- `n_neighbors=15`
- `min_dist=0.1`
- `metric='cosine'`
- `random_state=42` (reproducibility)

Output: `output/coords.parquet` — same row order as the cards parquet, with added columns `x_local`, `y_local`, `x_openai`, `y_openai`.

### Stage 5: Visualize

Plotly subplots, two panels side-by-side. Each panel uses `scattergl` (WebGL) for smooth pan/zoom.

**Point styling:**
- Color: primary color identity bucket — White, Blue, Black, Red, Green, Multicolor (2+), Colorless. Standard MTG color hex codes.
- Size: constant (no encoding of rank — adds noise without value).
- Opacity: 0.7 to handle overlap.

**Hover content** (via `hovertemplate` + `customdata`):
- Card name (bold)
- Type line
- Mana cost
- Rarity
- An `<img>` of `image_uris.small` (Scryfall CDN — no local image storage needed)

**Layout:**
- Two panels: left = local mpnet, right = OpenAI 3-small.
- Shared legend (one color legend covers both).
- Title above each panel naming the model.
- Output written to a single HTML file via `fig.write_html(..., include_plotlyjs='cdn')` so the file stays small (Plotly JS loaded from CDN at view time).

If OpenAI was skipped, render only the local panel.

## CLI

Single entrypoint via `uv run mtg-graph-poc` (or `python -m mtg_graph.cli`). Runs all five stages in order. Each stage is idempotent — reruns are cheap because of caching. Optional flags:

- `--n-cards N` (default 5000) — to experiment with smaller subsets during development
- `--skip-openai` — force-skip even if the API key is present
- `--force-reembed` — invalidate the cache and regenerate

## Project layout

```
mtg_graph/
├── .git/
├── pyproject.toml
├── uv.lock
├── README.md
├── .gitignore
├── .env.example
├── data/
│   └── oracle-cards-*.json        (gitignored)
├── src/mtg_graph/
│   ├── __init__.py
│   ├── load_and_filter.py
│   ├── build_text_profiles.py
│   ├── embed.py
│   ├── reduce.py
│   ├── visualize.py
│   └── cli.py
├── output/                        (gitignored)
└── docs/superpowers/specs/
    └── 2026-05-20-mtg-graph-poc-design.md  (this file)
```

## Dependencies

- `pandas` + `pyarrow` — dataframes and parquet I/O
- `sentence-transformers` — local embeddings
- `openai` — API embeddings
- `umap-learn` — dimensionality reduction
- `plotly` — visualization
- `numpy` — array math
- `python-dotenv` — `.env` loading

Python 3.11+. Managed by `uv`.

## Visualization choice rationale

We considered three options:

1. **Plotly (chosen)** — single HTML output from Python, `scattergl` handles 37k+ points smoothly, no separate JS toolchain. Best fit for points-only UMAP scatter.
2. **Cosmograph** — WebGL-native, designed for true graphs with drawn edges. Required if we wanted the "spiderweb" IG-video aesthetic. We chose points-only, so this isn't needed.
3. **Bokeh** — similar to Plotly but card-image hover is more work.

The viz code as designed carries forward to the full 37k run; no rewrite needed.

## Success criteria

Opening `output/mtg_graph_poc.html` in a browser:

1. Both panels render and are responsive to zoom/pan
2. Hovering any point shows card name, type, mana cost, and an image thumbnail
3. Manually spot-checking a handful of cards, clusters are intuitive (e.g., Lightning Bolt is near other red burn; Counterspell is near other counterspells; basic lands cluster by color)
4. If both panels look like noise, the pipeline is broken. If only one looks bad, we know which model to drop for the full 37k run.

## Open questions

None at design time. Card-selection criteria, embedding model, viz library, and project setup are all decided.

## Out of scope (for future work)

- Full 37k production run
- Drawn KNN edges (Cosmograph)
- Structural edges (cards referencing other cards by name in oracle text)
- Node2Vec / DeepWalk graph embeddings
- Filtering by format (Modern, Commander, etc.)
- Sizing by edhrec_rank or price
- 3D view
