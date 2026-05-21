"""Stage 5: Render interactive HTML viewers.

2D viewer:  Cosmograph (@cosmos.gl/graph) — native graph rendering with
            click-to-select neighbor highlighting + KNN edges.
3D viewer:  Plotly Scatter3d — side-panel-only interactivity (Plotly's 3D
            mode is hostile to runtime trace mutation).

Always writes (per model):
  output/mtg_graph_poc.html      — 2D Cosmograph viewer
  output/mtg_graph_poc_3d.html   — 3D Plotly viewer
If OpenAI embeddings exist, both pairs are also emitted with _openai suffix.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATE_3D = TEMPLATES_DIR / "viewer.html"
TEMPLATE_2D = TEMPLATES_DIR / "viewer_2d.html"

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

LABEL_TEXT_COLOR = "#F0F0F0"
LABEL_FONT_SIZE = 11

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

COSMO_SPACE_EXTENT = 2000.0  # ±2000 inside Cosmograph's default 4096 spaceSize


def _normalize_coords(values: np.ndarray, extent: float = COSMO_SPACE_EXTENT) -> np.ndarray:
    """Center on 0 and scale to roughly ±extent."""
    vmin, vmax = float(values.min()), float(values.max())
    center = (vmin + vmax) / 2.0
    half_range = max((vmax - vmin) / 2.0, 1e-6)
    return ((values - center) / half_range * extent).astype(np.float32)


def _build_cosmograph_data(
    merged: pd.DataFrame,
    model_key: str,
    cluster_labels: dict,
    knn: dict,
) -> dict:
    """Produce all the JSON payloads the Cosmograph viewer template needs."""
    x_col, y_col = f"x_{model_key}", f"y_{model_key}"
    df = merged.reset_index(drop=True)
    xs = _normalize_coords(df[x_col].to_numpy(dtype=np.float64))
    ys = _normalize_coords(df[y_col].to_numpy(dtype=np.float64))
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
        "oracle_id", "name", "mana_cost", "type_line", "oracle_text",
        "flavor_text", "rarity", "set_name", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
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
        point_meta.append(rec)

    # Cluster labels in NORMALIZED space coords so they line up with Cosmograph's frame.
    raw_x = df[x_col].to_numpy(dtype=np.float64)
    raw_y = df[y_col].to_numpy(dtype=np.float64)
    x_center = (float(raw_x.min()) + float(raw_x.max())) / 2.0
    y_center = (float(raw_y.min()) + float(raw_y.max())) / 2.0
    x_half = max((float(raw_x.max()) - float(raw_x.min())) / 2.0, 1e-6)
    y_half = max((float(raw_y.max()) - float(raw_y.min())) / 2.0, 1e-6)
    cluster_col = f"cluster_{model_key}"
    label_entries = []
    for cid_str, label_text in (cluster_labels or {}).get(model_key, {}).items():
        if cluster_col not in df.columns:
            continue
        sub = df[df[cluster_col] == int(cid_str)]
        if sub.empty:
            continue
        raw_cx = float(sub[x_col].mean())
        raw_cy = float(sub[y_col].mean())
        nx = (raw_cx - x_center) / x_half * COSMO_SPACE_EXTENT
        ny = (raw_cy - y_center) / y_half * COSMO_SPACE_EXTENT
        label_entries.append({"x": nx, "y": ny, "text": label_text})

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


# ---------- Plotly (3D) — unchanged from prior version ----------

def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "<i>%{customdata[2]}</i>  "
        "<span style='color:#888'>%{customdata[3]}</span>"
        "<extra></extra>"
    )


def _customdata(sub: pd.DataFrame) -> np.ndarray:
    return sub[["name", "type_line", "mana_cost", "rarity", "image_small", "oracle_id"]].fillna("").to_numpy()


def _cards_payload(cards: pd.DataFrame) -> dict[str, dict]:
    """Keyed-by-oracle_id card dict for the 3D side panel (Plotly viewer uses this)."""
    fields = [
        "oracle_id", "name", "mana_cost", "type_line", "oracle_text",
        "flavor_text", "rarity", "set_name", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil", "scryfall_uri",
    ]
    out: dict[str, dict] = {}
    for _, row in cards.iterrows():
        rec: dict = {}
        for f in fields:
            val = row.get(f)
            if val is None:
                rec[f] = None
            elif isinstance(val, float) and np.isnan(val):
                rec[f] = None
            else:
                rec[f] = val if isinstance(val, str) else (val if isinstance(val, (int, float)) else str(val))
        out[row["oracle_id"]] = rec
    return out


def _build_3d_traces(cards: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    for bucket, color in COLOR_MAP.items():
        mask = cards["_color_bucket"] == bucket
        if not mask.any():
            continue
        sub = cards.loc[mask]
        traces.append(go.Scatter3d(
            x=sub[x_col], y=sub[y_col], z=sub[z_col],
            mode="markers",
            name=bucket,
            marker=dict(color=color, size=3, opacity=0.78, line=dict(width=0)),
            customdata=_customdata(sub),
            hovertemplate=_hover_template(),
            legendgroup=bucket,
        ))
    return traces


def _cluster_label_trace_3d(merged: pd.DataFrame, x_col: str, y_col: str, z_col: str, cluster_col: str, labels: dict) -> go.Scatter3d | None:
    xs, ys, zs, texts = [], [], [], []
    for cid_str, label in labels.items():
        sub = merged[merged[cluster_col] == int(cid_str)]
        if sub.empty:
            continue
        xs.append(float(sub[x_col].mean()))
        ys.append(float(sub[y_col].mean()))
        zs.append(float(sub[z_col].mean()))
        texts.append(label)
    if not texts:
        return None
    return go.Scatter3d(
        x=xs, y=ys, z=zs, text=texts, mode="text",
        textfont=dict(color=LABEL_TEXT_COLOR, size=LABEL_FONT_SIZE, family="Arial Black, sans-serif"),
        hoverinfo="skip", showlegend=False, name="_cluster_labels",
    )


def _layout_3d(n_cards: int, model_label: str) -> dict:
    return dict(
        title=f"MTG Card Embedding Map (3D) — {model_label} ({n_cards:,} cards)",
        template="plotly_dark",
        paper_bgcolor="#0a0a0a",
        hoverlabel=dict(bgcolor="#111", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=50, b=20),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            bgcolor="#000",
        ),
        autosize=True,
    )


def _render_3d_plotly(
    template: str,
    figure: go.Figure,
    cards_payload: dict,
    knn_payload: dict,
    title: str,
    model_key: str,
    interaction_hint: str,
    status_hint: str,
    mode_label: str,
    counterpart_href: str,
    counterpart_label: str,
) -> str:
    fig_json = figure.to_json().replace("</", "<\\/")
    cards_json = _safe_json(cards_payload)
    knn_json = _safe_json(knn_payload)
    return (
        template
        .replace("{{TITLE}}", title)
        .replace("{{MODEL_KEY}}", model_key)
        .replace("{{IS_3D}}", "true")
        .replace("{{INTERACTION_HINT}}", interaction_hint)
        .replace("{{STATUS_HINT}}", status_hint)
        .replace("{{MODE_LABEL}}", mode_label)
        .replace("{{COUNTERPART_HREF}}", counterpart_href)
        .replace("{{COUNTERPART_LABEL}}", counterpart_label)
        .replace("{{FIGURE_JSON}}", fig_json)
        .replace("{{CARDS_JSON}}", cards_json)
        .replace("{{KNN_JSON}}", knn_json)
    )


# ---------- orchestration ----------

def run(
    cards_path: Path = Path("output/cards_top5k.parquet"),
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
    cards_payload_for_3d = _cards_payload(cards)
    outputs: dict[str, Path] = {}

    for model_key in ("local", "openai"):
        if f"x_{model_key}" not in merged.columns:
            continue
        model_label = MODEL_LABELS[model_key]
        suffix = "" if model_key == "local" else f"_{model_key}"
        out2d_name = f"mtg_graph_poc{suffix}.html"
        out3d_name = f"mtg_graph_poc_3d{suffix}.html"

        # 2D (Cosmograph)
        data = _build_cosmograph_data(merged, model_key, cluster_labels, knn)
        html2d = _render_2d_cosmograph(
            template_2d,
            data,
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

        # 3D (Plotly)
        x3, y3, z3 = f"x3_{model_key}", f"y3_{model_key}", f"z3_{model_key}"
        if x3 not in merged.columns:
            continue
        fig3 = go.Figure()
        for trace in _build_3d_traces(merged, x3, y3, z3):
            fig3.add_trace(trace)
        if model_key in cluster_labels:
            lbl3 = _cluster_label_trace_3d(merged, x3, y3, z3, f"cluster_{model_key}", cluster_labels[model_key])
            if lbl3 is not None:
                fig3.add_trace(lbl3)
        fig3.update_layout(**_layout_3d(len(merged), model_label))

        out3d = output_dir / out3d_name
        html3d = _render_3d_plotly(
            template_3d, fig3, cards_payload_for_3d, knn,
            title=f"MTG Graph 3D — {model_label}",
            model_key=model_key,
            interaction_hint="Click without dragging to inspect a card · drag to orbit · scroll to zoom.",
            status_hint="Click any card to inspect · drag to orbit · scroll to zoom · (highlights are 2D-only)",
            mode_label="3D",
            counterpart_href=out2d_name,
            counterpart_label="Switch to 2D",
        )
        out3d.write_text(html3d)
        outputs[f"3d_{model_key}"] = out3d
        print(f"  Wrote {out3d}")

    return outputs


if __name__ == "__main__":
    run()
