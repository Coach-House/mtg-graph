# MTG Graph — 5k POC

Vector-embedding map of the top 5,000 Magic: The Gathering cards (by EDHREC rank).
Embeds each card with two models, reduces to 2D with UMAP, and renders an
interactive side-by-side scatter as a single HTML file.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies (creates .venv/)
uv sync

# 2. (Optional) Add your OpenAI key for the second panel
cp .env.example .env
# edit .env and paste your OPENAI_API_KEY

# 3. Drop the Scryfall oracle-cards JSON into ./data/
#    Download from https://scryfall.com/docs/api/bulk-data (pick "Oracle Cards")
```

## Run

```bash
uv run mtg-graph-poc
```

Then open `output/mtg_graph_poc.html` in any browser.

## Flags

- `--n-cards N` — use a different number of cards (default 5000)
- `--skip-openai` — skip the OpenAI panel even if `OPENAI_API_KEY` is set
- `--force-reembed` — invalidate the embedding cache and regenerate

## What you'll see

Two side-by-side scatterplots — one per embedding model. Cards are placed by
UMAP-reduced semantic similarity and colored by color identity. Hover any point
to see card name, type, mana cost, and image.

## Cost & runtime

First run: ~5–10 min (model download + local embeddings). OpenAI embeddings cost ~$0.05.
Subsequent runs: ~30 seconds (cache hits).

## Design / planning docs

- Spec: `docs/superpowers/specs/2026-05-20-mtg-graph-poc-design.md`
- Plan: `docs/superpowers/plans/2026-05-20-mtg-graph-poc.md`
