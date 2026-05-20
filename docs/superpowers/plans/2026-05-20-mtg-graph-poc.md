# MTG Graph 5k POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python pipeline that loads the top 5,000 MTG cards by EDHREC rank, embeds them with two models (local mpnet + OpenAI 3-small), reduces to 2D via UMAP, and produces a single self-contained HTML file with two side-by-side Plotly scatterplots.

**Architecture:** Five sequential stages, each its own module under `src/mtg_graph/`. Each stage reads from disk and writes to disk so it can be rerun independently. Embeddings are cached on disk with a SHA-256 hash sidecar for invalidation. A `cli.py` chains the stages; the CLI is registered as a `uv run` script entry point.

**Tech Stack:** Python 3.11+, uv (package manager), pandas + pyarrow (data), sentence-transformers (local embeddings), openai (API embeddings), umap-learn (reduction), plotly (viz), python-dotenv (config).

**Testing approach:** Per the spec, no unit tests — visual HTML output is the verification. Each module has a sanity-check step (run it, eyeball output shape) before moving on.

**Spec reference:** `docs/superpowers/specs/2026-05-20-mtg-graph-poc-design.md`

---

## File Structure

Files this plan creates:

```
pyproject.toml                          # uv project config + dependencies
.env.example                            # Sample OPENAI_API_KEY
README.md                               # How to run
src/mtg_graph/__init__.py               # Package marker
src/mtg_graph/load_and_filter.py        # Stage 1
src/mtg_graph/build_text_profiles.py    # Stage 2
src/mtg_graph/embed.py                  # Stage 3
src/mtg_graph/reduce.py                 # Stage 4
src/mtg_graph/visualize.py              # Stage 5
src/mtg_graph/cli.py                    # Chains all stages
```

Each module exposes one function (`run(...)`) so `cli.py` is a thin orchestrator. State flows through parquet/npy files on disk, not in-memory passing.

---

## Task 1: Project setup with uv

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/mtg_graph/__init__.py`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`

Expected: prints a version like `uv 0.5.x` or similar. If "command not found", install with `brew install uv` and re-run.

- [ ] **Step 2: Create `pyproject.toml`**

Write this exact content to `pyproject.toml`:

```toml
[project]
name = "mtg-graph"
version = "0.1.0"
description = "Vector-embedding network graph of Magic: The Gathering cards"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.2.0",
    "pyarrow>=15.0.0",
    "numpy>=1.26.0",
    "sentence-transformers>=2.7.0",
    "openai>=1.30.0",
    "umap-learn>=0.5.5",
    "plotly>=5.20.0",
    "python-dotenv>=1.0.0",
]

[project.scripts]
mtg-graph-poc = "mtg_graph.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/mtg_graph"]
```

- [ ] **Step 3: Create package marker**

Write to `src/mtg_graph/__init__.py`:

```python
"""MTG card vector-embedding graph — proof of concept."""

__version__ = "0.1.0"
```

- [ ] **Step 4: Create `.env.example`**

Write to `.env.example`:

```
# Copy this file to .env and fill in your key.
# If OPENAI_API_KEY is unset, the pipeline runs local embeddings only.
OPENAI_API_KEY=sk-...
```

- [ ] **Step 5: Sync dependencies**

Run: `cd /Users/zwardlaw/Projects/mtg_graph && uv sync`

Expected: creates `.venv/` and `uv.lock`. Prints "Resolved N packages" and "Installed N packages". This will take 2–5 minutes the first time because `sentence-transformers` pulls in PyTorch.

- [ ] **Step 6: Verify the package imports**

Run: `uv run python -c "import mtg_graph; print(mtg_graph.__version__)"`

Expected output: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .env.example src/
git commit -m "Set up uv project with dependencies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Stage 1 — load and filter cards

**Files:**
- Create: `src/mtg_graph/load_and_filter.py`

- [ ] **Step 1: Write the module**

Write to `src/mtg_graph/load_and_filter.py`:

```python
"""Stage 1: Load Scryfall oracle JSON, filter to playable cards, take top 5,000 by edhrec_rank."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

KEEP_LAYOUTS = {
    "normal", "transform", "modal_dfc", "split", "adventure",
    "saga", "class", "case", "flip", "leveler", "meld",
    "mutate", "prototype",
}


def run(
    data_dir: Path = Path("data"),
    output_dir: Path = Path("output"),
    n_cards: int = 5000,
) -> Path:
    """Load Scryfall oracle JSON, filter, take top N by EDHREC rank.

    Returns the path to the written parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(glob.glob(str(data_dir / "oracle-cards-*.json")))
    if not candidates:
        raise FileNotFoundError(
            f"No oracle-cards-*.json found in {data_dir}. "
            "Download from https://scryfall.com/docs/api/bulk-data"
        )
    source = Path(candidates[-1])
    print(f"Loading {source.name}...")

    with source.open() as f:
        raw = json.load(f)

    df = pd.DataFrame(raw)
    print(f"  Loaded {len(df):,} unique cards from JSON")

    df = df[df["layout"].isin(KEEP_LAYOUTS)]
    print(f"  After layout filter: {len(df):,}")

    df = df.dropna(subset=["edhrec_rank"])
    print(f"  After dropping cards without edhrec_rank: {len(df):,}")

    df = df.sort_values("edhrec_rank").head(n_cards).reset_index(drop=True)
    print(f"  Top {len(df):,} by edhrec_rank")

    df["image_small"] = df["image_uris"].apply(
        lambda u: u.get("small") if isinstance(u, dict) else None
    )

    cols = [
        "oracle_id", "name", "mana_cost", "type_line", "oracle_text",
        "keywords", "colors", "color_identity", "rarity",
        "edhrec_rank", "image_small",
    ]
    for col in cols:
        if col not in df.columns:
            df[col] = None
    df = df[cols]

    out = output_dir / "cards_top5k.parquet"
    df.to_parquet(out, index=False)
    print(f"  Wrote {out}")
    return out


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Sanity-check run**

Run: `uv run python -m mtg_graph.load_and_filter`

Expected output (numbers approximate):
```
Loading oracle-cards-20260520090335.json...
  Loaded 37,472 unique cards from JSON
  After layout filter: 34,xxx
  After dropping cards without edhrec_rank: 27,xxx
  Top 5,000 by edhrec_rank
  Wrote output/cards_top5k.parquet
```

- [ ] **Step 3: Spot-check the output**

Run:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('output/cards_top5k.parquet')
print(df.shape)
print(df[['name', 'mana_cost', 'type_line', 'edhrec_rank']].head(10))
print(df[['name', 'edhrec_rank']].tail(5))
"
```

Expected: shape `(5000, 11)`, top 10 should include very recognizable Commander staples (Sol Ring, Command Tower, Arcane Signet, basic lands, etc.). Bottom 5 rows have `edhrec_rank` around 5000.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/load_and_filter.py
git commit -m "Add stage 1: load and filter cards by EDHREC rank

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Stage 2 — build text profiles

**Files:**
- Create: `src/mtg_graph/build_text_profiles.py`

- [ ] **Step 1: Write the module**

Write to `src/mtg_graph/build_text_profiles.py`:

```python
"""Stage 2: Build a single text profile string per card for embedding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _profile(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("name"):
        parts.append(str(row["name"]))
    if row.get("type_line"):
        parts.append(str(row["type_line"]))
    mana = row.get("mana_cost")
    if isinstance(mana, str) and mana:
        parts.append(mana)
    text = row.get("oracle_text")
    if isinstance(text, str) and text:
        parts.append(text)
    return " | ".join(parts)


def run(
    input_path: Path = Path("output/cards_top5k.parquet"),
    output_path: Path = Path("output/cards_top5k.parquet"),
) -> Path:
    """Add a `text_profile` column to the cards parquet."""
    df = pd.read_parquet(input_path)
    print(f"Building text profiles for {len(df):,} cards...")
    df["text_profile"] = df.apply(_profile, axis=1)
    df.to_parquet(output_path, index=False)
    print(f"  Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Sanity-check run**

Run: `uv run python -m mtg_graph.build_text_profiles`

Expected output:
```
Building text profiles for 5,000 cards...
  Wrote output/cards_top5k.parquet
```

- [ ] **Step 3: Eyeball a few profiles**

Run:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('output/cards_top5k.parquet')
for _, row in df.head(5).iterrows():
    print('---')
    print(row['text_profile'])
"
```

Expected: 5 cards with pipe-separated profiles. Lightning Bolt-like cards should have name, type, mana cost, and rules text all joined with `|`. Lands should have no mana cost section.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/build_text_profiles.py
git commit -m "Add stage 2: build text profiles for embedding

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Stage 3a — local embeddings (mpnet)

**Files:**
- Create: `src/mtg_graph/embed.py`

This task implements only the local embedder. The OpenAI side comes in Task 5.

- [ ] **Step 1: Write the module skeleton**

Write to `src/mtg_graph/embed.py`:

```python
"""Stage 3: Generate embeddings with two models. Cached on disk via SHA-256 sidecar.

Each model writes:
  output/embeddings_<model>.npy   — (N, D) float32 array
  output/embeddings_<model>.json  — {"model_id": ..., "card_hash": ...}

On rerun, if the sidecar's card_hash matches the current card list, regeneration is skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

LOCAL_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"
OPENAI_MODEL_ID = "text-embedding-3-small"
OPENAI_BATCH_SIZE = 2048


def _card_hash(oracle_ids: list[str]) -> str:
    h = hashlib.sha256()
    for oid in sorted(oracle_ids):
        h.update(oid.encode())
    return h.hexdigest()


def _cache_is_valid(sidecar: Path, model_id: str, card_hash: str) -> bool:
    if not sidecar.exists():
        return False
    try:
        meta = json.loads(sidecar.read_text())
    except json.JSONDecodeError:
        return False
    return meta.get("model_id") == model_id and meta.get("card_hash") == card_hash


def _write_cache(npy_path: Path, sidecar: Path, vectors: np.ndarray, model_id: str, card_hash: str) -> None:
    np.save(npy_path, vectors)
    sidecar.write_text(json.dumps({"model_id": model_id, "card_hash": card_hash}, indent=2))


def _embed_local(profiles: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading {LOCAL_MODEL_ID} (first run downloads ~420MB)...")
    model = SentenceTransformer(LOCAL_MODEL_ID)
    print(f"  Embedding {len(profiles):,} profiles (this takes 3–5 minutes)...")
    vectors = model.encode(
        profiles,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)


def run(
    input_path: Path = Path("output/cards_top5k.parquet"),
    output_dir: Path = Path("output"),
    force: bool = False,
    skip_openai: bool = False,
) -> dict[str, Path]:
    """Generate embeddings with both models. Returns {model_name: npy_path}.

    If OPENAI_API_KEY is missing or skip_openai is True, only local is generated.
    """
    load_dotenv()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    profiles = df["text_profile"].tolist()
    oracle_ids = df["oracle_id"].tolist()
    card_hash = _card_hash(oracle_ids)
    print(f"Card list hash: {card_hash[:12]}...")

    results: dict[str, Path] = {}

    # Local
    local_npy = output_dir / "embeddings_local.npy"
    local_sidecar = output_dir / "embeddings_local.json"
    if not force and _cache_is_valid(local_sidecar, LOCAL_MODEL_ID, card_hash):
        print(f"Local cache valid — skipping ({local_npy})")
    else:
        print("Generating local embeddings...")
        vectors = _embed_local(profiles)
        _write_cache(local_npy, local_sidecar, vectors, LOCAL_MODEL_ID, card_hash)
        print(f"  Wrote {local_npy} (shape {vectors.shape})")
    results["local"] = local_npy

    # OpenAI side — added in Task 5.
    return results


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run local embedding**

Run: `uv run python -m mtg_graph.embed`

Expected: First run downloads the model (~420MB, one-time). Then prints a progress bar for embedding. Takes 3–10 minutes depending on hardware. Final output: `Wrote output/embeddings_local.npy (shape (5000, 768))`.

- [ ] **Step 3: Verify cache works**

Run: `uv run python -m mtg_graph.embed`

Expected: prints `Local cache valid — skipping (output/embeddings_local.npy)` and exits in under a second.

- [ ] **Step 4: Spot-check the embeddings**

Run:

```bash
uv run python -c "
import numpy as np
v = np.load('output/embeddings_local.npy')
print('shape:', v.shape)
print('dtype:', v.dtype)
print('norm range:', np.linalg.norm(v, axis=1).min(), 'to', np.linalg.norm(v, axis=1).max())
print('first vec head:', v[0][:5])
"
```

Expected: shape `(5000, 768)`, dtype `float32`, norms all positive (typically 1.0 if normalized, or various values if not).

- [ ] **Step 5: Commit**

```bash
git add src/mtg_graph/embed.py
git commit -m "Add stage 3a: local mpnet embeddings with cache

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Stage 3b — OpenAI embeddings + graceful degradation

**Files:**
- Modify: `src/mtg_graph/embed.py`

- [ ] **Step 1: Add the OpenAI embedder function**

Open `src/mtg_graph/embed.py`. Replace the line `# OpenAI side — added in Task 5.` and the lines below it with this block (everything from the comment to the `return results` line):

```python
    # OpenAI
    if skip_openai:
        print("Skipping OpenAI (skip_openai=True)")
    elif not os.environ.get("OPENAI_API_KEY"):
        print("Skipping OpenAI (OPENAI_API_KEY not set in environment or .env)")
    else:
        openai_npy = output_dir / "embeddings_openai.npy"
        openai_sidecar = output_dir / "embeddings_openai.json"
        if not force and _cache_is_valid(openai_sidecar, OPENAI_MODEL_ID, card_hash):
            print(f"OpenAI cache valid — skipping ({openai_npy})")
        else:
            print(f"Generating OpenAI embeddings ({OPENAI_MODEL_ID})...")
            vectors = _embed_openai(profiles)
            _write_cache(openai_npy, openai_sidecar, vectors, OPENAI_MODEL_ID, card_hash)
            print(f"  Wrote {openai_npy} (shape {vectors.shape})")
        results["openai"] = openai_npy

    return results
```

- [ ] **Step 2: Add the `_embed_openai` function**

In the same file, add this function below `_embed_local` (above `def run(`):

```python
def _embed_openai(profiles: list[str]) -> np.ndarray:
    from openai import OpenAI

    client = OpenAI()
    all_vectors: list[np.ndarray] = []
    n_batches = (len(profiles) + OPENAI_BATCH_SIZE - 1) // OPENAI_BATCH_SIZE
    for i in range(0, len(profiles), OPENAI_BATCH_SIZE):
        batch = profiles[i : i + OPENAI_BATCH_SIZE]
        batch_idx = i // OPENAI_BATCH_SIZE + 1
        print(f"  Batch {batch_idx}/{n_batches} ({len(batch)} items)...")
        resp = client.embeddings.create(model=OPENAI_MODEL_ID, input=batch)
        all_vectors.extend(np.array(item.embedding, dtype=np.float32) for item in resp.data)
    return np.stack(all_vectors)
```

- [ ] **Step 3: Test graceful degradation first**

Run: `uv run python -c "
import os
os.environ.pop('OPENAI_API_KEY', None)
from mtg_graph import embed
embed.run()
"`

Expected: prints `Local cache valid — skipping ...` then `Skipping OpenAI (OPENAI_API_KEY not set ...)`. No errors. No openai file written.

- [ ] **Step 4: Run with API key**

If you haven't yet, copy the env template and add your key:

```bash
cp .env.example .env
# Edit .env and paste your real OPENAI_API_KEY
```

Then run: `uv run python -m mtg_graph.embed`

Expected: prints `Local cache valid — skipping`, then `Generating OpenAI embeddings`, prints 3 batch progress lines (batch sizes 2048, 2048, 904), writes `output/embeddings_openai.npy`. Total cost ≈ $0.05. Takes ~30 seconds.

- [ ] **Step 5: Spot-check OpenAI output**

Run:

```bash
uv run python -c "
import numpy as np
v = np.load('output/embeddings_openai.npy')
print('shape:', v.shape, 'dtype:', v.dtype)
"
```

Expected: shape `(5000, 1536)`, dtype `float32`.

- [ ] **Step 6: Verify OpenAI cache works**

Run: `uv run python -m mtg_graph.embed`

Expected: both lines print "cache valid — skipping". Exits in under a second.

- [ ] **Step 7: Commit**

```bash
git add src/mtg_graph/embed.py
git commit -m "Add stage 3b: OpenAI embeddings with graceful degradation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Stage 4 — UMAP dimensionality reduction

**Files:**
- Create: `src/mtg_graph/reduce.py`

- [ ] **Step 1: Write the module**

Write to `src/mtg_graph/reduce.py`:

```python
"""Stage 4: UMAP-reduce each embedding set to 2D and join coordinates onto the card table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)


def _reduce(vectors: np.ndarray, label: str) -> np.ndarray:
    import umap

    print(f"  UMAP-reducing {label} embeddings (shape {vectors.shape})...")
    reducer = umap.UMAP(n_components=2, **UMAP_KWARGS)
    coords = reducer.fit_transform(vectors)
    print(f"    -> coords shape {coords.shape}")
    return coords


def run(
    cards_path: Path = Path("output/cards_top5k.parquet"),
    output_dir: Path = Path("output"),
) -> Path:
    """Reduce each available embedding set to 2D and write a combined coords parquet."""
    df = pd.read_parquet(cards_path)
    coords_df = df[["oracle_id"]].copy()

    local_npy = output_dir / "embeddings_local.npy"
    if local_npy.exists():
        vec = np.load(local_npy)
        xy = _reduce(vec, "local")
        coords_df["x_local"] = xy[:, 0]
        coords_df["y_local"] = xy[:, 1]
    else:
        print(f"  No local embeddings found at {local_npy} — skipping")

    openai_npy = output_dir / "embeddings_openai.npy"
    if openai_npy.exists():
        vec = np.load(openai_npy)
        xy = _reduce(vec, "openai")
        coords_df["x_openai"] = xy[:, 0]
        coords_df["y_openai"] = xy[:, 1]
    else:
        print(f"  No OpenAI embeddings found at {openai_npy} — skipping")

    out = output_dir / "coords.parquet"
    coords_df.to_parquet(out, index=False)
    print(f"  Wrote {out} ({len(coords_df.columns)-1} coord columns)")
    return out


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Run UMAP**

Run: `uv run python -m mtg_graph.reduce`

Expected: prints two "UMAP-reducing" lines (local and openai) and "Wrote output/coords.parquet". UMAP on 5k × 768d takes ~30s; on 5k × 1536d takes ~60s. Total ~90s.

- [ ] **Step 3: Eyeball the coordinates**

Run:

```bash
uv run python -c "
import pandas as pd
df = pd.read_parquet('output/coords.parquet')
print(df.shape)
print(df.head())
print()
print('x_local range:', df['x_local'].min(), 'to', df['x_local'].max())
print('y_local range:', df['y_local'].min(), 'to', df['y_local'].max())
"
```

Expected: shape `(5000, 5)` if both embeddings exist (oracle_id, x_local, y_local, x_openai, y_openai). Coordinate ranges are typically -10 to +20 — UMAP doesn't normalize.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/reduce.py
git commit -m "Add stage 4: UMAP reduction to 2D

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Stage 5 — interactive HTML visualization

**Files:**
- Create: `src/mtg_graph/visualize.py`

- [ ] **Step 1: Write the module**

Write to `src/mtg_graph/visualize.py`:

```python
"""Stage 5: Render side-by-side Plotly scatterplots and write a self-contained HTML file."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_MAP = {
    "White": "#F8F6D8",
    "Blue": "#0E68AB",
    "Black": "#150B00",
    "Red": "#D3202A",
    "Green": "#00733E",
    "Multicolor": "#C9A227",
    "Colorless": "#A0A0A0",
}


def _color_bucket(color_identity) -> str:
    # color_identity may be a list, numpy array, or None depending on parquet engine.
    if color_identity is None:
        return "Colorless"
    try:
        n = len(color_identity)
    except TypeError:
        return "Colorless"
    if n == 0:
        return "Colorless"
    if n > 1:
        return "Multicolor"
    mapping = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}
    return mapping.get(color_identity[0], "Colorless")


def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "<i>%{customdata[2]}</i>  "
        "<span style='color:#888'>%{customdata[3]}</span>"
        "<br><img src='%{customdata[4]}' width='146'>"
        "<extra></extra>"
    )


def _build_traces(
    cards: pd.DataFrame,
    x_col: str,
    y_col: str,
    show_legend: bool,
) -> list[go.Scattergl]:
    traces: list[go.Scattergl] = []
    for bucket, color in COLOR_MAP.items():
        mask = cards["_color_bucket"] == bucket
        if not mask.any():
            continue
        sub = cards.loc[mask]
        customdata = sub[["name", "type_line", "mana_cost", "rarity", "image_small"]].fillna("").to_numpy()
        traces.append(
            go.Scattergl(
                x=sub[x_col],
                y=sub[y_col],
                mode="markers",
                name=bucket,
                marker=dict(color=color, size=6, opacity=0.7, line=dict(width=0.5, color="#222")),
                customdata=customdata,
                hovertemplate=_hover_template(),
                showlegend=show_legend,
                legendgroup=bucket,
            )
        )
    return traces


def run(
    cards_path: Path = Path("output/cards_top5k.parquet"),
    coords_path: Path = Path("output/coords.parquet"),
    output_path: Path = Path("output/mtg_graph_poc.html"),
) -> Path:
    cards = pd.read_parquet(cards_path)
    coords = pd.read_parquet(coords_path)
    merged = cards.merge(coords, on="oracle_id", how="inner")
    merged["_color_bucket"] = merged["color_identity"].apply(_color_bucket)

    has_local = "x_local" in merged.columns
    has_openai = "x_openai" in merged.columns
    if not has_local and not has_openai:
        raise RuntimeError("No coordinate columns present; run stage 4 first.")

    if has_local and has_openai:
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Local: all-mpnet-base-v2 (768d)", "OpenAI: text-embedding-3-small (1536d)"),
            horizontal_spacing=0.06,
        )
        for trace in _build_traces(merged, "x_local", "y_local", show_legend=True):
            fig.add_trace(trace, row=1, col=1)
        for trace in _build_traces(merged, "x_openai", "y_openai", show_legend=False):
            fig.add_trace(trace, row=1, col=2)
    elif has_local:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("Local: all-mpnet-base-v2 (768d)",))
        for trace in _build_traces(merged, "x_local", "y_local", show_legend=True):
            fig.add_trace(trace, row=1, col=1)
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=("OpenAI: text-embedding-3-small (1536d)",))
        for trace in _build_traces(merged, "x_openai", "y_openai", show_legend=True):
            fig.add_trace(trace, row=1, col=1)

    fig.update_layout(
        title=f"MTG Card Embedding Map — Top {len(merged):,} by EDHREC Rank",
        template="plotly_dark",
        height=820,
        hoverlabel=dict(bgcolor="#111", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5),
        margin=dict(l=30, r=30, t=80, b=60),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Render the HTML**

Run: `uv run python -m mtg_graph.visualize`

Expected: prints `Wrote output/mtg_graph_poc.html`. File should be 5–15 MB (custom hover data per point).

- [ ] **Step 3: Open and eyeball**

Run: `open output/mtg_graph_poc.html`

Manual checks:
- Two panels render side-by-side, both populated with ~5,000 points
- Pan and zoom work smoothly
- Hovering a point shows card name (bold), type line, mana cost, rarity, and a card image thumbnail
- Spot-check a handful: Lightning Bolt should sit near other red instants; Counterspell near other blue counters; Sol Ring near other colorless mana rocks; basic lands near each other by color.

If clusters look like random noise in both panels, stop and report — the pipeline has a bug.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/visualize.py
git commit -m "Add stage 5: side-by-side Plotly visualization

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: CLI entrypoint

**Files:**
- Create: `src/mtg_graph/cli.py`

- [ ] **Step 1: Write the CLI**

Write to `src/mtg_graph/cli.py`:

```python
"""Single CLI entrypoint that chains all five stages.

Usage:
    uv run mtg-graph-poc [--n-cards N] [--skip-openai] [--force-reembed]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mtg_graph import build_text_profiles, embed, load_and_filter, reduce, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="MTG card embedding pipeline (5k POC)")
    parser.add_argument("--n-cards", type=int, default=5000, help="How many cards to include (default 5000)")
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI even if API key is set")
    parser.add_argument("--force-reembed", action="store_true", help="Invalidate embedding cache")
    args = parser.parse_args()

    data_dir = Path("data")
    output_dir = Path("output")

    print("\n=== Stage 1: load and filter ===")
    load_and_filter.run(data_dir=data_dir, output_dir=output_dir, n_cards=args.n_cards)

    print("\n=== Stage 2: build text profiles ===")
    build_text_profiles.run(input_path=output_dir / "cards_top5k.parquet")

    print("\n=== Stage 3: embed ===")
    embed.run(
        input_path=output_dir / "cards_top5k.parquet",
        output_dir=output_dir,
        force=args.force_reembed,
        skip_openai=args.skip_openai,
    )

    print("\n=== Stage 4: reduce ===")
    reduce.run(cards_path=output_dir / "cards_top5k.parquet", output_dir=output_dir)

    print("\n=== Stage 5: visualize ===")
    out = visualize.run(
        cards_path=output_dir / "cards_top5k.parquet",
        coords_path=output_dir / "coords.parquet",
        output_path=output_dir / "mtg_graph_poc.html",
    )
    print(f"\nDone. Open {out} in your browser.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full pipeline end-to-end**

Run: `uv run mtg-graph-poc`

Expected: prints five "=== Stage N: ... ===" headers. Stages 1, 2, 4, 5 each finish in seconds. Stage 3 uses caches (both should report "cache valid — skipping"). Final line: `Done. Open output/mtg_graph_poc.html in your browser.`

- [ ] **Step 3: Verify the entry point works without `python -m`**

Run: `uv run mtg-graph-poc --help`

Expected: standard argparse help output with the three flags.

- [ ] **Step 4: Commit**

```bash
git add src/mtg_graph/cli.py
git commit -m "Add CLI entrypoint chaining all five stages

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: README and final cleanup

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Write to `README.md`:

```markdown
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
```

- [ ] **Step 2: Verify final repo layout**

Run: `git ls-files`

Expected output:

```
.env.example
.gitignore
README.md
docs/superpowers/plans/2026-05-20-mtg-graph-poc.md
docs/superpowers/specs/2026-05-20-mtg-graph-poc-design.md
pyproject.toml
src/mtg_graph/__init__.py
src/mtg_graph/build_text_profiles.py
src/mtg_graph/cli.py
src/mtg_graph/embed.py
src/mtg_graph/load_and_filter.py
src/mtg_graph/reduce.py
src/mtg_graph/visualize.py
uv.lock
```

- [ ] **Step 3: Final commit**

```bash
git add README.md docs/superpowers/plans/2026-05-20-mtg-graph-poc.md
git commit -m "Add README and implementation plan

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Done

Open `output/mtg_graph_poc.html` and explore. If clusters look sensible (burn near burn, counters near counters, lands near lands), the POC validates the approach and we can scale to the full ~37k card set as a follow-up.

If clusters look like noise, debug starting from `output/embeddings_local.npy` — load it and run a quick KNN lookup on Lightning Bolt's neighbors to see if the embeddings themselves are healthy before blaming UMAP.
