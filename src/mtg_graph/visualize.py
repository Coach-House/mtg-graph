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
