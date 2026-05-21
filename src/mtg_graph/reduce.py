"""Stage 4: UMAP-reduce each embedding set to 2D and 3D, join coordinates onto the card table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

UMAP_KWARGS = dict(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)


def _reduce(vectors: np.ndarray, label: str, n_components: int) -> np.ndarray:
    import umap

    print(f"  UMAP-reducing {label} to {n_components}D (input shape {vectors.shape})...")
    reducer = umap.UMAP(n_components=n_components, **UMAP_KWARGS)
    coords = reducer.fit_transform(vectors)
    print(f"    -> coords shape {coords.shape}")
    return coords


def run(
    cards_path: Path = Path("output/cards_top5k.parquet"),
    output_dir: Path = Path("output"),
) -> Path:
    """Reduce each available embedding set to both 2D and 3D."""
    df = pd.read_parquet(cards_path)
    coords_df = df[["oracle_id"]].copy()

    for label, npy_name in [("local", "embeddings_local.npy"), ("openai", "embeddings_openai.npy")]:
        npy = output_dir / npy_name
        if not npy.exists():
            print(f"  No {label} embeddings at {npy} — skipping")
            continue
        vec = np.load(npy)
        xy = _reduce(vec, label, n_components=2)
        coords_df[f"x_{label}"] = xy[:, 0]
        coords_df[f"y_{label}"] = xy[:, 1]
        xyz = _reduce(vec, label, n_components=3)
        coords_df[f"x3_{label}"] = xyz[:, 0]
        coords_df[f"y3_{label}"] = xyz[:, 1]
        coords_df[f"z3_{label}"] = xyz[:, 2]

    out = output_dir / "coords.parquet"
    coords_df.to_parquet(out, index=False)
    print(f"  Wrote {out} ({len(coords_df.columns) - 1} coord columns)")
    return out


if __name__ == "__main__":
    run()
