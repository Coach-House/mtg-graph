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


if __name__ == "__main__":
    run()
