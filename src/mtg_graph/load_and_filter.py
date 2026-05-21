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


def _pick_image(uris, key: str) -> str | None:
    return uris.get(key) if isinstance(uris, dict) else None


def _pick_price(prices, key: str) -> str | None:
    return prices.get(key) if isinstance(prices, dict) else None


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

    df["image_small"] = df["image_uris"].apply(lambda u: _pick_image(u, "small"))
    df["image_normal"] = df["image_uris"].apply(lambda u: _pick_image(u, "normal"))
    df["image_art_crop"] = df["image_uris"].apply(lambda u: _pick_image(u, "art_crop"))
    df["price_usd"] = df["prices"].apply(lambda p: _pick_price(p, "usd"))
    df["price_usd_foil"] = df["prices"].apply(lambda p: _pick_price(p, "usd_foil"))

    cols = [
        "oracle_id", "name", "mana_cost", "type_line", "oracle_text",
        "flavor_text", "keywords", "colors", "color_identity", "rarity",
        "set_name", "edhrec_rank",
        "image_small", "image_normal", "image_art_crop",
        "price_usd", "price_usd_foil",
        "scryfall_uri",
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
