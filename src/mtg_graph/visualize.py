"""Stage 5: Render interactive HTML viewers.

2D viewer: Cosmograph (@cosmos.gl/graph) — native graph rendering with
           click-to-select neighbor highlighting + KNN edges.
3D viewer: 3d-force-graph (three.js-based) — same interactivity in 3D,
           with fixed UMAP positions (no force simulation drift).

Both viewers share the same side panel (search, card details, neighbors grid,
mode-switch link).

Always writes (per model):
  output/mtg_graph_poc.html      — 2D Cosmograph viewer
  output/mtg_graph_poc_3d.html   — 3D 3d-force-graph viewer
If OpenAI embeddings exist, both pairs are also emitted with _openai suffix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATE_2D = TEMPLATES_DIR / "viewer_2d.html"
TEMPLATE_3D = TEMPLATES_DIR / "viewer_3d.html"

# Hex colors used by both viewers; Cosmograph consumes the RGBA conversion below.
COLOR_MAP = {
    "White": "#F5EEDC",
    "Blue": "#3B82F6",
    "Black": "#7A6FA0",
    "Red": "#EF4444",
    "Green": "#22C55E",
    "Multicolor": "#FBBF24",
    "Colorless": "#9CA3AF",
}

MODEL_LABELS = {
    "local": "all-mpnet-base-v2 (768d)",
    "openai": "text-embedding-3-small (1536d)",
}


# ---------- shared helpers ----------

def _color_bucket(color_identity) -> str:
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


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0, alpha)


# ---------- Cosmograph (2D) serialization ----------

# Cosmograph's coordinate space runs [0, spaceSize] (NOT centered on origin).
# We match its spaceSize=4096 config and leave a small padding margin so
# fitViewOnInit has room to breathe.
COSMO_SPACE_SIZE = 4096.0
COSMO_PADDING = 200.0  # margin from each edge


def _normalize_coords_centered(values: np.ndarray, extent: float) -> np.ndarray:
    """Center on 0 and scale to roughly ±extent. Used by 3d-force-graph."""
    vmin, vmax = float(values.min()), float(values.max())
    center = (vmin + vmax) / 2.0
    half_range = max((vmax - vmin) / 2.0, 1e-6)
    return ((values - center) / half_range * extent).astype(np.float32)


def _normalize_coords_offset(
    values: np.ndarray,
    out_min: float = COSMO_PADDING,
    out_max: float = COSMO_SPACE_SIZE - COSMO_PADDING,
) -> np.ndarray:
    """Linearly map values into [out_min, out_max]. Used by Cosmograph (2D)."""
    vmin, vmax = float(values.min()), float(values.max())
    span = max(vmax - vmin, 1e-6)
    return ((values - vmin) / span * (out_max - out_min) + out_min).astype(np.float32)


def _build_cosmograph_data(
    merged: pd.DataFrame,
    model_key: str,
    cluster_labels: dict,
    knn: dict,
) -> dict:
    """Produce all the JSON payloads the Cosmograph viewer template needs."""
    x_col, y_col = f"x_{model_key}", f"y_{model_key}"
    df = merged.reset_index(drop=True)
    xs = _normalize_coords_offset(df[x_col].to_numpy(dtype=np.float64))
    ys = _normalize_coords_offset(df[y_col].to_numpy(dtype=np.float64))
    n = len(df)

    # Interleaved [x1, y1, x2, y2, ...] for Cosmograph.setPointPositions.
    point_positions = np.empty(n * 2, dtype=np.float32)
    point_positions[0::2] = xs
    point_positions[1::2] = ys

    # RGBA per point, values 0–1.
    rgba_lookup = {bucket: _hex_to_rgba(hex_, 1.0) for bucket, hex_ in COLOR_MAP.items()}
    point_colors = np.empty(n * 4, dtype=np.float32)
    for i, bucket in enumerate(df["_color_bucket"].tolist()):
        r, g, b, a = rgba_lookup.get(bucket, rgba_lookup["Colorless"])
        point_colors[i * 4] = r
        point_colors[i * 4 + 1] = g
        point_colors[i * 4 + 2] = b
        point_colors[i * 4 + 3] = a

    # KNN links → [src1, tgt1, src2, tgt2, ...]. Each card has K neighbors → N*K edges.
    oid_to_idx = {oid: i for i, oid in enumerate(df["oracle_id"].tolist())}
    idx_to_oid = df["oracle_id"].tolist()
    model_knn = (knn or {}).get(model_key, {})
    link_pairs: list[int] = []
    for src_oid, neighbor_oids in model_knn.items():
        src = oid_to_idx.get(src_oid)
        if src is None:
            continue
        for tgt_oid in neighbor_oids:
            tgt = oid_to_idx.get(tgt_oid)
            if tgt is None:
                continue
            link_pairs.append(src)
            link_pairs.append(tgt)
    links_arr = np.array(link_pairs, dtype=np.float32)

    # Per-card metadata used by the side panel. Indexed by position (matches positions order).
    meta_fields = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "flavor_text", "rarity", "set", "set_name", "legalities", "edhrec_rank",
        "color_identity", "released_at",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
    cluster_col_2d = f"cluster_{model_key}"
    has_cluster = cluster_col_2d in df.columns
    point_meta = []
    for _, row in df.iterrows():
        rec = {}
        for f in meta_fields:
            val = row.get(f)
            if val is None:
                rec[f] = None
            elif isinstance(val, float) and np.isnan(val):
                rec[f] = None
            elif isinstance(val, (list, np.ndarray)):
                rec[f] = list(val)
            else:
                rec[f] = val
        # Normalize the model-suffixed column name to a generic "cluster" key so
        # the viewer JS doesn't need to know which embedding model produced the data.
        if has_cluster:
            cv = row.get(cluster_col_2d)
            rec["cluster"] = None if (cv is None or (isinstance(cv, float) and np.isnan(cv))) else int(cv)
        point_meta.append(rec)

    # Cluster labels in NORMALIZED space coords so they line up with Cosmograph's frame.
    # Uses the SAME [out_min, out_max] mapping as the point positions above —
    # critical for label-overlay alignment when rescalePositions=false in the viewer.
    raw_x = df[x_col].to_numpy(dtype=np.float64)
    raw_y = df[y_col].to_numpy(dtype=np.float64)
    x_min, x_max = float(raw_x.min()), float(raw_x.max())
    y_min, y_max = float(raw_y.min()), float(raw_y.max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    out_min = COSMO_PADDING
    out_max = COSMO_SPACE_SIZE - COSMO_PADDING
    out_range = out_max - out_min
    cluster_col = f"cluster_{model_key}"
    label_entries = []
    for cid_str, label_text in (cluster_labels or {}).get(model_key, {}).items():
        if cluster_col not in df.columns:
            continue
        cid = int(cid_str)
        sub = df[df[cluster_col] == cid]
        if sub.empty:
            continue
        raw_cx = float(sub[x_col].mean())
        raw_cy = float(sub[y_col].mean())
        nx = (raw_cx - x_min) / x_span * out_range + out_min
        ny = (raw_cy - y_min) / y_span * out_range + out_min
        label_entries.append({"x": nx, "y": ny, "text": label_text, "cluster_id": cid})

    # KNN as oid-keyed map for the neighbors grid in the side panel.
    knn_by_oid = {oid: list(neighbors) for oid, neighbors in model_knn.items()}

    return {
        "point_positions": point_positions.tolist(),
        "point_colors": point_colors.tolist(),
        "links": links_arr.tolist(),
        "point_meta": point_meta,
        "cluster_labels": label_entries,
        "oid_to_idx": oid_to_idx,
        "idx_to_oid": idx_to_oid,
        "knn_by_oid": knn_by_oid,
    }


def _safe_json(obj) -> str:
    """JSON-encode for embedding inside a <script> tag — escape `</` so the
    browser parser doesn't close the script early on a stray `</script>`."""
    return json.dumps(obj, default=str).replace("</", "<\\/")


def _render_2d_cosmograph(
    template: str,
    data: dict,
    title: str,
    mode_label: str,
    counterpart_href: str,
    counterpart_label: str,
    status_hint: str,
) -> str:
    return (
        template
        .replace("{{TITLE}}", title)
        .replace("{{STATUS_HINT}}", status_hint)
        .replace("{{MODE_LABEL}}", mode_label)
        .replace("{{COUNTERPART_HREF}}", counterpart_href)
        .replace("{{COUNTERPART_LABEL}}", counterpart_label)
        .replace("{{POINT_META_JSON}}", _safe_json(data["point_meta"]))
        .replace("{{POINT_POSITIONS_JSON}}", _safe_json(data["point_positions"]))
        .replace("{{POINT_COLORS_JSON}}", _safe_json(data["point_colors"]))
        .replace("{{LINKS_JSON}}", _safe_json(data["links"]))
        .replace("{{CLUSTER_LABELS_JSON}}", _safe_json(data["cluster_labels"]))
        .replace("{{OID_TO_IDX_JSON}}", _safe_json(data["oid_to_idx"]))
        .replace("{{IDX_TO_OID_JSON}}", _safe_json(data["idx_to_oid"]))
        .replace("{{KNN_BY_OID_JSON}}", _safe_json(data["knn_by_oid"]))
    )


# ---------- 3d-force-graph (3D) serialization ----------

# 3D extent — smaller than 2D Cosmograph's 2000 because three.js camera defaults work
# well around the ±200 range for unit-less coordinates.
FORCEGRAPH_EXTENT = 200.0


def _build_forcegraph_data(
    merged: pd.DataFrame,
    model_key: str,
    cluster_labels: dict,
    knn: dict,
) -> dict:
    """Produce JSON payloads the 3D viewer template needs."""
    x_col, y_col, z_col = f"x3_{model_key}", f"y3_{model_key}", f"z3_{model_key}"
    df = merged.reset_index(drop=True)
    xs = _normalize_coords_centered(df[x_col].to_numpy(dtype=np.float64), extent=FORCEGRAPH_EXTENT)
    ys = _normalize_coords_centered(df[y_col].to_numpy(dtype=np.float64), extent=FORCEGRAPH_EXTENT)
    zs = _normalize_coords_centered(df[z_col].to_numpy(dtype=np.float64), extent=FORCEGRAPH_EXTENT)
    n = len(df)

    # Interleaved [x1, y1, z1, x2, y2, z2, ...] for the template to consume by index.
    positions = np.empty(n * 3, dtype=np.float32)
    positions[0::3] = xs
    positions[1::3] = ys
    positions[2::3] = zs

    # Per-point hex colors from MTG color identity.
    colors_hex = [COLOR_MAP.get(b, COLOR_MAP["Colorless"]) for b in df["_color_bucket"].tolist()]

    oid_to_idx = {oid: i for i, oid in enumerate(df["oracle_id"].tolist())}

    meta_fields = [
        "oracle_id", "name", "mana_cost", "cmc", "type_line", "oracle_text",
        "flavor_text", "rarity", "set", "set_name", "legalities", "edhrec_rank",
        "color_identity", "released_at",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
    cluster_col_3d = f"cluster_{model_key}"
    has_cluster = cluster_col_3d in df.columns
    point_meta = []
    for _, row in df.iterrows():
        rec = {}
        for f in meta_fields:
            val = row.get(f)
            if val is None:
                rec[f] = None
            elif isinstance(val, float) and np.isnan(val):
                rec[f] = None
            elif isinstance(val, (list, np.ndarray)):
                rec[f] = list(val)
            else:
                rec[f] = val
        if has_cluster:
            cv = row.get(cluster_col_3d)
            rec["cluster"] = None if (cv is None or (isinstance(cv, float) and np.isnan(cv))) else int(cv)
        point_meta.append(rec)

    model_knn = (knn or {}).get(model_key, {})
    knn_by_oid = {oid: list(neighbors) for oid, neighbors in model_knn.items()}

    # Cluster label 3D positions (centroid of each labeled cluster).
    raw_x = df[x_col].to_numpy(dtype=np.float64)
    raw_y = df[y_col].to_numpy(dtype=np.float64)
    raw_z = df[z_col].to_numpy(dtype=np.float64)
    label_entries: list[dict] = []
    if has_cluster:
        x_center = (float(raw_x.min()) + float(raw_x.max())) / 2.0
        y_center = (float(raw_y.min()) + float(raw_y.max())) / 2.0
        z_center = (float(raw_z.min()) + float(raw_z.max())) / 2.0
        x_half = max((float(raw_x.max()) - float(raw_x.min())) / 2.0, 1e-6)
        y_half = max((float(raw_y.max()) - float(raw_y.min())) / 2.0, 1e-6)
        z_half = max((float(raw_z.max()) - float(raw_z.min())) / 2.0, 1e-6)
        for cid_str, label_text in (cluster_labels or {}).get(model_key, {}).items():
            cid = int(cid_str)
            sub = df[df[cluster_col_3d] == cid]
            if sub.empty:
                continue
            cx = (float(sub[x_col].mean()) - x_center) / x_half * FORCEGRAPH_EXTENT
            cy = (float(sub[y_col].mean()) - y_center) / y_half * FORCEGRAPH_EXTENT
            cz = (float(sub[z_col].mean()) - z_center) / z_half * FORCEGRAPH_EXTENT
            label_entries.append({"x": cx, "y": cy, "z": cz, "text": label_text, "cluster_id": cid})

    return {
        "positions": positions.tolist(),
        "colors_hex": colors_hex,
        "point_meta": point_meta,
        "oid_to_idx": oid_to_idx,
        "knn_by_oid": knn_by_oid,
        "cluster_labels": label_entries,
    }


def _render_3d_forcegraph(
    template: str,
    data: dict,
    title: str,
    mode_label: str,
    counterpart_href: str,
    counterpart_label: str,
    status_hint: str,
) -> str:
    return (
        template
        .replace("{{TITLE}}", title)
        .replace("{{STATUS_HINT}}", status_hint)
        .replace("{{MODE_LABEL}}", mode_label)
        .replace("{{COUNTERPART_HREF}}", counterpart_href)
        .replace("{{COUNTERPART_LABEL}}", counterpart_label)
        .replace("{{POINT_META_JSON}}", _safe_json(data["point_meta"]))
        .replace("{{POINT_POSITIONS_3D_JSON}}", _safe_json(data["positions"]))
        .replace("{{POINT_COLORS_HEX_JSON}}", _safe_json(data["colors_hex"]))
        .replace("{{OID_TO_IDX_JSON}}", _safe_json(data["oid_to_idx"]))
        .replace("{{KNN_BY_OID_JSON}}", _safe_json(data["knn_by_oid"]))
        .replace("{{CLUSTER_LABELS_JSON}}", _safe_json(data["cluster_labels"]))
    )


# ---------- orchestration ----------

def run(
    cards_path: Path = Path("output/cards.parquet"),
    coords_path: Path = Path("output/coords.parquet"),
    output_dir: Path = Path("output"),
) -> dict[str, Path]:
    cards = pd.read_parquet(cards_path)
    coords = pd.read_parquet(coords_path)
    merged = cards.merge(coords, on="oracle_id", how="inner")
    merged["_color_bucket"] = merged["color_identity"].apply(_color_bucket)

    clusters_path = output_dir / "clusters.parquet"
    if clusters_path.exists():
        clusters = pd.read_parquet(clusters_path)
        merged = merged.merge(clusters, on="oracle_id", how="left")
    labels_path = output_dir / "cluster_labels.json"
    cluster_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

    knn_path = output_dir / "knn.json"
    knn = json.loads(knn_path.read_text()) if knn_path.exists() else {}

    template_2d = TEMPLATE_2D.read_text()
    template_3d = TEMPLATE_3D.read_text()
    outputs: dict[str, Path] = {}

    for model_key in ("local", "openai"):
        if f"x_{model_key}" not in merged.columns:
            continue
        model_label = MODEL_LABELS[model_key]
        suffix = "" if model_key == "local" else f"_{model_key}"
        out2d_name = f"mtg_graph_poc{suffix}.html"
        out3d_name = f"mtg_graph_poc_3d{suffix}.html"

        # 2D (Cosmograph)
        data_2d = _build_cosmograph_data(merged, model_key, cluster_labels, knn)
        html2d = _render_2d_cosmograph(
            template_2d,
            data_2d,
            title=f"MTG Graph — {model_label}",
            mode_label="2D",
            counterpart_href=out3d_name,
            counterpart_label="Switch to 3D",
            status_hint="Click any card · scroll to zoom · drag to pan",
        )
        out2d = output_dir / out2d_name
        out2d.write_text(html2d)
        outputs[f"2d_{model_key}"] = out2d
        print(f"  Wrote {out2d}")

        # 3D (3d-force-graph)
        if f"x3_{model_key}" not in merged.columns:
            continue
        data_3d = _build_forcegraph_data(merged, model_key, cluster_labels, knn)
        html3d = _render_3d_forcegraph(
            template_3d,
            data_3d,
            title=f"MTG Graph 3D — {model_label}",
            mode_label="3D",
            counterpart_href=out2d_name,
            counterpart_label="Switch to 2D",
            status_hint="Click any card · drag to orbit · scroll to zoom",
        )
        out3d = output_dir / out3d_name
        out3d.write_text(html3d)
        outputs[f"3d_{model_key}"] = out3d
        print(f"  Wrote {out3d}")

    return outputs


if __name__ == "__main__":
    run()
