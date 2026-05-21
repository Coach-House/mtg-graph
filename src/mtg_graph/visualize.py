"""Stage 5: Render Plotly scatterplots (2D and 3D) with cluster labels.

Always writes BOTH outputs when run:
  output/mtg_graph_poc.html     — side-by-side 2D panels
  output/mtg_graph_poc_3d.html  — side-by-side 3D panels
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Brighter palette tuned for dark background — preserves MTG color identity recognition
# while keeping all colors independently legible against the plotly_dark canvas.
COLOR_MAP = {
    "White": "#F5EEDC",
    "Blue": "#3B82F6",
    "Black": "#7A6FA0",  # muted purple-gray; truly black would vanish on dark bg
    "Red": "#EF4444",
    "Green": "#22C55E",
    "Multicolor": "#FBBF24",
    "Colorless": "#9CA3AF",
}

MARKER_OUTLINE = "#1F1F1F"
LABEL_TEXT_COLOR = "#F0F0F0"
LABEL_FONT_SIZE = 11


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


def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b><br>"
        "%{customdata[1]}<br>"
        "<i>%{customdata[2]}</i>  "
        "<span style='color:#888'>%{customdata[3]}</span>"
        "<br><img src='%{customdata[4]}' width='146'>"
        "<extra></extra>"
    )


def _cluster_centroids_2d(merged: pd.DataFrame, x_col: str, y_col: str, cluster_col: str, labels: dict[int, str]) -> tuple[list[float], list[float], list[str]]:
    xs, ys, texts = [], [], []
    for cid_str, label in labels.items():
        cid = int(cid_str)
        sub = merged[merged[cluster_col] == cid]
        if sub.empty:
            continue
        xs.append(float(sub[x_col].mean()))
        ys.append(float(sub[y_col].mean()))
        texts.append(label)
    return xs, ys, texts


def _cluster_centroids_3d(merged: pd.DataFrame, x_col: str, y_col: str, z_col: str, cluster_col: str, labels: dict[int, str]) -> tuple[list[float], list[float], list[float], list[str]]:
    xs, ys, zs, texts = [], [], [], []
    for cid_str, label in labels.items():
        cid = int(cid_str)
        sub = merged[merged[cluster_col] == cid]
        if sub.empty:
            continue
        xs.append(float(sub[x_col].mean()))
        ys.append(float(sub[y_col].mean()))
        zs.append(float(sub[z_col].mean()))
        texts.append(label)
    return xs, ys, zs, texts


def _build_2d_traces(cards: pd.DataFrame, x_col: str, y_col: str, show_legend: bool) -> list[go.Scattergl]:
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
                marker=dict(color=color, size=6, opacity=0.78, line=dict(width=0.5, color=MARKER_OUTLINE)),
                customdata=customdata,
                hovertemplate=_hover_template(),
                showlegend=show_legend,
                legendgroup=bucket,
            )
        )
    return traces


def _build_3d_traces(cards: pd.DataFrame, x_col: str, y_col: str, z_col: str, show_legend: bool) -> list[go.Scatter3d]:
    traces: list[go.Scatter3d] = []
    for bucket, color in COLOR_MAP.items():
        mask = cards["_color_bucket"] == bucket
        if not mask.any():
            continue
        sub = cards.loc[mask]
        customdata = sub[["name", "type_line", "mana_cost", "rarity", "image_small"]].fillna("").to_numpy()
        traces.append(
            go.Scatter3d(
                x=sub[x_col],
                y=sub[y_col],
                z=sub[z_col],
                mode="markers",
                name=bucket,
                marker=dict(color=color, size=3, opacity=0.78, line=dict(width=0)),
                customdata=customdata,
                hovertemplate=_hover_template(),
                showlegend=show_legend,
                legendgroup=bucket,
            )
        )
    return traces


def _label_trace_2d(xs: list[float], ys: list[float], texts: list[str]) -> go.Scatter:
    return go.Scatter(
        x=xs, y=ys, text=texts,
        mode="text",
        textfont=dict(color=LABEL_TEXT_COLOR, size=LABEL_FONT_SIZE, family="Arial Black, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
    )


def _label_trace_3d(xs: list[float], ys: list[float], zs: list[float], texts: list[str]) -> go.Scatter3d:
    return go.Scatter3d(
        x=xs, y=ys, z=zs, text=texts,
        mode="text",
        textfont=dict(color=LABEL_TEXT_COLOR, size=LABEL_FONT_SIZE, family="Arial Black, sans-serif"),
        hoverinfo="skip",
        showlegend=False,
    )


def _render_2d(
    merged: pd.DataFrame,
    has_local: bool,
    has_openai: bool,
    cluster_labels: dict[str, dict[int, str]],
    output_path: Path,
) -> Path:
    n_panels = sum([has_local, has_openai])
    titles = []
    if has_local:
        titles.append("Local: all-mpnet-base-v2 (768d)")
    if has_openai:
        titles.append("OpenAI: text-embedding-3-small (1536d)")
    fig = make_subplots(rows=1, cols=n_panels, subplot_titles=titles, horizontal_spacing=0.06)

    col = 1
    if has_local:
        for trace in _build_2d_traces(merged, "x_local", "y_local", show_legend=True):
            fig.add_trace(trace, row=1, col=col)
        if "local" in cluster_labels:
            xs, ys, texts = _cluster_centroids_2d(merged, "x_local", "y_local", "cluster_local", cluster_labels["local"])
            if texts:
                fig.add_trace(_label_trace_2d(xs, ys, texts), row=1, col=col)
        col += 1
    if has_openai:
        for trace in _build_2d_traces(merged, "x_openai", "y_openai", show_legend=not has_local):
            fig.add_trace(trace, row=1, col=col)
        if "openai" in cluster_labels:
            xs, ys, texts = _cluster_centroids_2d(merged, "x_openai", "y_openai", "cluster_openai", cluster_labels["openai"])
            if texts:
                fig.add_trace(_label_trace_2d(xs, ys, texts), row=1, col=col)

    fig.update_layout(
        title=f"MTG Card Embedding Map (2D) — Top {len(merged):,} by EDHREC Rank",
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


def _render_3d(
    merged: pd.DataFrame,
    has_local: bool,
    has_openai: bool,
    cluster_labels: dict[str, dict[int, str]],
    output_path: Path,
) -> Path:
    n_panels = sum([has_local, has_openai])
    titles = []
    if has_local:
        titles.append("Local: all-mpnet-base-v2 (768d)")
    if has_openai:
        titles.append("OpenAI: text-embedding-3-small (1536d)")

    specs = [[{"type": "scene"} for _ in range(n_panels)]]
    fig = make_subplots(rows=1, cols=n_panels, specs=specs, subplot_titles=titles, horizontal_spacing=0.04)

    col = 1
    if has_local:
        for trace in _build_3d_traces(merged, "x3_local", "y3_local", "z3_local", show_legend=True):
            fig.add_trace(trace, row=1, col=col)
        if "local" in cluster_labels:
            xs, ys, zs, texts = _cluster_centroids_3d(merged, "x3_local", "y3_local", "z3_local", "cluster_local", cluster_labels["local"])
            if texts:
                fig.add_trace(_label_trace_3d(xs, ys, zs, texts), row=1, col=col)
        col += 1
    if has_openai:
        for trace in _build_3d_traces(merged, "x3_openai", "y3_openai", "z3_openai", show_legend=not has_local):
            fig.add_trace(trace, row=1, col=col)
        if "openai" in cluster_labels:
            xs, ys, zs, texts = _cluster_centroids_3d(merged, "x3_openai", "y3_openai", "z3_openai", "cluster_openai", cluster_labels["openai"])
            if texts:
                fig.add_trace(_label_trace_3d(xs, ys, zs, texts), row=1, col=col)

    scene_kwargs = dict(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        zaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        bgcolor="#000",
    )
    layout_kwargs = {f"scene{'' if i == 0 else i + 1}": scene_kwargs for i in range(n_panels)}

    fig.update_layout(
        title=f"MTG Card Embedding Map (3D) — Top {len(merged):,} by EDHREC Rank",
        template="plotly_dark",
        height=900,
        hoverlabel=dict(bgcolor="#111", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, xanchor="center", x=0.5),
        margin=dict(l=0, r=0, t=70, b=20),
        **layout_kwargs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    print(f"  Wrote {output_path}")
    return output_path


def _load_cluster_labels(labels_path: Path) -> dict[str, dict[int, str]]:
    if not labels_path.exists():
        return {}
    return json.loads(labels_path.read_text())


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
    cluster_labels = _load_cluster_labels(output_dir / "cluster_labels.json")

    has_local = "x_local" in merged.columns
    has_openai = "x_openai" in merged.columns
    if not has_local and not has_openai:
        raise RuntimeError("No coordinate columns present; run stage 4 first.")

    outputs = {}
    outputs["2d"] = _render_2d(merged, has_local, has_openai, cluster_labels, output_dir / "mtg_graph_poc.html")

    has_3d_local = "x3_local" in merged.columns
    has_3d_openai = "x3_openai" in merged.columns
    if has_3d_local or has_3d_openai:
        outputs["3d"] = _render_3d(merged, has_3d_local, has_3d_openai, cluster_labels, output_dir / "mtg_graph_poc_3d.html")
    return outputs


if __name__ == "__main__":
    run()
