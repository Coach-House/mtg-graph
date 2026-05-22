"""Stage 4.6: Precompute top-K nearest neighbors per card in raw embedding space.

KNN is computed on the raw (un-reduced) embeddings via cosine distance — this gives
semantically truer neighbors than what's visible after UMAP's 2D/3D projection.

Output: output/knn.json
    {
      "local":  {oracle_id: [neighbor_oracle_id, ...]},   # K entries, self excluded
      "openai": {oracle_id: [...]},
    }
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

K = 20


def _knn_for(vectors: np.ndarray, oracle_ids: list[str]) -> dict[str, list[str]]:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=K + 1, metric="cosine", algorithm="brute")
    nn.fit(vectors)
    _, idx = nn.kneighbors(vectors)
    out: dict[str, list[str]] = {}
    for i, neighbors in enumerate(idx):
        # neighbors[0] is the point itself; skip it.
        out[oracle_ids[i]] = [oracle_ids[j] for j in neighbors[1:]]
    return out


def run(
    cards_path: Path = Path("output/cards.parquet"),
    output_dir: Path = Path("output"),
) -> Path:
    df = pd.read_parquet(cards_path)
    oracle_ids = df["oracle_id"].tolist()

    result: dict[str, dict[str, list[str]]] = {}
    for label, npy_name in [("local", "embeddings_local.npy"), ("openai", "embeddings_openai.npy")]:
        npy = output_dir / npy_name
        if not npy.exists():
            print(f"  No {label} embeddings — skipping")
            continue
        vec = np.load(npy)
        print(f"  KNN (K={K}) for {label} on shape {vec.shape}...")
        result[label] = _knn_for(vec, oracle_ids)
        print(f"    -> {len(result[label]):,} cards indexed")

    out = output_dir / "knn.json"
    out.write_text(json.dumps(result))
    print(f"  Wrote {out}")
    return out


if __name__ == "__main__":
    run()
