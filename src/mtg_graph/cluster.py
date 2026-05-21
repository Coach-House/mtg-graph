"""Stage 4.5: HDBSCAN-cluster the embeddings and label each cluster via TF-IDF.

For each card, assigns a cluster_id (or -1 for noise). For each non-noise cluster,
computes a short distinctive label by comparing TF-IDF terms within the cluster
against the rest of the cards.

Output:
  output/clusters.parquet — columns: oracle_id, cluster_local, cluster_openai (whichever exist)
  output/cluster_labels.json — {"local": {cluster_id: "label", ...}, "openai": {...}}
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CLUSTER_SIZE = 15
MIN_LABEL_SIZE = 20  # only render labels for clusters with at least this many cards
N_LABEL_TERMS = 2

# MTG-specific stopwords on top of sklearn's english list.
EXTRA_STOPWORDS = {
    "card", "cards", "target", "creature", "creatures", "player", "players",
    "spell", "spells", "land", "lands", "permanent", "permanents", "battlefield",
    "library", "graveyard", "hand", "turn", "cast", "control", "controls",
    "ability", "abilities", "mana", "color", "colors", "deals", "damage",
    "enters", "enter", "draw", "draws", "create", "creates", "may", "instead",
    "end", "beginning", "each", "one", "two", "three", "type", "name",
}


def _run_hdbscan(vectors: np.ndarray) -> np.ndarray:
    from sklearn.cluster import HDBSCAN

    print(f"  HDBSCAN on shape {vectors.shape}...")
    clusterer = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=3,
        metric="euclidean",
        cluster_selection_epsilon=0.0,
    )
    labels = clusterer.fit_predict(vectors)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"    -> {n_clusters} clusters, {n_noise} noise points ({n_noise / len(labels):.0%})")
    return labels


def _label_clusters(text_profiles: list[str], cluster_ids: np.ndarray) -> dict[int, str]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    stopwords = list({*EXTRA_STOPWORDS})
    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
        ngram_range=(1, 2),
        max_features=2000,
    )
    tfidf = vectorizer.fit_transform(text_profiles)
    feature_names = vectorizer.get_feature_names_out()
    stop_set = set(stopwords)

    labels: dict[int, str] = {}
    unique = [c for c in np.unique(cluster_ids) if c != -1]
    for cid in unique:
        mask = cluster_ids == cid
        if mask.sum() < MIN_LABEL_SIZE:
            continue
        in_cluster = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        out_cluster = np.asarray(tfidf[~mask].mean(axis=0)).ravel()
        distinctive = in_cluster - out_cluster
        order = np.argsort(distinctive)[::-1]
        picked: list[str] = []
        seen_words: set[str] = set()
        for idx in order:
            term = feature_names[idx]
            words = term.split()
            if any(w in stop_set for w in words):
                continue
            # avoid picking a 2-gram that shares all words with what we already have
            if any(w in seen_words for w in words):
                continue
            picked.append(term)
            seen_words.update(words)
            if len(picked) >= N_LABEL_TERMS:
                break
        if picked:
            labels[int(cid)] = " / ".join(picked)
    return labels


LLM_MODEL = "gpt-4o-mini"
LLM_PROMPT = """You are labeling clusters of Magic: The Gathering cards.

Given the cards below from one cluster, produce a short label (3–7 words) that
captures what they have in common. Prefer mechanic names, tribal names, or
specific effect categories. Avoid generic terms like "cards" or "spells".

Examples of good labels:
- "Counterspells"
- "Ninja tribal & Ninjutsu"
- "Treasure-token generators"
- "Mass exile sweepers"
- "Cycling lands"
- "Fetch lands (search & sacrifice)"

Cards in this cluster:
{cards}

Hint terms from this cluster (may be noisy): {hints}

Reply with ONLY the label, no preamble, no quotes."""


def _llm_label_clusters(
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    tfidf_labels: dict[int, str],
) -> dict[int, str]:
    """Use an LLM to give each cluster a natural-language name.
    Returns dict[int, str]. Falls back to TF-IDF label on per-cluster API error."""
    try:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        return tfidf_labels

    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("    LLM labels skipped (OPENAI_API_KEY not set) — using TF-IDF")
        return tfidf_labels

    client = OpenAI()
    out: dict[int, str] = {}
    unique = [c for c in np.unique(cluster_ids) if c != -1 and c in {int(k) for k in tfidf_labels.keys()}]
    print(f"    LLM-labeling {len(unique)} clusters via {LLM_MODEL}...")
    for cid in unique:
        sub = df[cluster_ids == cid].sort_values("edhrec_rank").head(8)
        cards_block = "\n".join(
            f"- {r['name']} ({r['type_line']}): {('' if pd.isna(r['oracle_text']) else str(r['oracle_text'])).replace(chr(10), ' ')[:160]}"
            for _, r in sub.iterrows()
        )
        prompt = LLM_PROMPT.format(cards=cards_block, hints=tfidf_labels.get(int(cid), ""))
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=24,
            )
            label = resp.choices[0].message.content.strip().strip('"').strip("'")
            out[int(cid)] = label or tfidf_labels[int(cid)]
        except Exception as e:
            print(f"    cluster {cid}: LLM call failed ({e}); falling back to TF-IDF")
            out[int(cid)] = tfidf_labels[int(cid)]
    return out


def run(
    cards_path: Path = Path("output/cards_top5k.parquet"),
    output_dir: Path = Path("output"),
) -> Path:
    """Cluster on the 2D UMAP coords (HDBSCAN works much better in low-dim post-UMAP space)."""
    df = pd.read_parquet(cards_path)
    coords = pd.read_parquet(output_dir / "coords.parquet")
    merged = df[["oracle_id", "text_profile", "name", "type_line", "oracle_text", "edhrec_rank"]].merge(coords, on="oracle_id", how="left")

    clusters_df = df[["oracle_id"]].copy()
    all_labels: dict[str, dict[int, str]] = {}

    for label in ("local", "openai"):
        x_col, y_col = f"x_{label}", f"y_{label}"
        if x_col not in merged.columns:
            print(f"  No {label} UMAP coords — skipping")
            continue
        xy = merged[[x_col, y_col]].to_numpy()
        cids = _run_hdbscan(xy)
        clusters_df[f"cluster_{label}"] = cids
        cluster_labels = _label_clusters(merged["text_profile"].tolist(), cids)
        cluster_labels = _llm_label_clusters(merged, cids, cluster_labels)
        all_labels[label] = {str(k): v for k, v in cluster_labels.items()}
        print(f"    {len(cluster_labels)} labeled clusters")

    out = output_dir / "clusters.parquet"
    clusters_df.to_parquet(out, index=False)
    print(f"  Wrote {out}")

    labels_out = output_dir / "cluster_labels.json"
    labels_out.write_text(json.dumps(all_labels, indent=2))
    print(f"  Wrote {labels_out}")
    return out


if __name__ == "__main__":
    run()
