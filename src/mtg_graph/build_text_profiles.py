"""Stage 2: Build a single text profile string per card for embedding."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _profile(row: pd.Series) -> str:
    parts: list[str] = []
    if row.get("name"):
        parts.append(str(row["name"]))
    if row.get("type_line"):
        parts.append(str(row["type_line"]))
    mana = row.get("mana_cost")
    if isinstance(mana, str) and mana:
        parts.append(mana)
    text = row.get("oracle_text")
    if isinstance(text, str) and text:
        parts.append(text)
    return " | ".join(parts)


def run(
    input_path: Path = Path("output/cards.parquet"),
    output_path: Path = Path("output/cards.parquet"),
) -> Path:
    """Add a `text_profile` column to the cards parquet."""
    df = pd.read_parquet(input_path)
    print(f"Building text profiles for {len(df):,} cards...")
    df["text_profile"] = df.apply(_profile, axis=1)
    df.to_parquet(output_path, index=False)
    print(f"  Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    run()
