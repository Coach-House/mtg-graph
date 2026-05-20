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
