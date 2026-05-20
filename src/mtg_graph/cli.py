"""Single CLI entrypoint that chains all five stages.

Usage:
    uv run mtg-graph-poc [--n-cards N] [--skip-openai] [--force-reembed]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mtg_graph import build_text_profiles, embed, load_and_filter, reduce, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="MTG card embedding pipeline (5k POC)")
    parser.add_argument("--n-cards", type=int, default=5000, help="How many cards to include (default 5000)")
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI even if API key is set")
    parser.add_argument("--force-reembed", action="store_true", help="Invalidate embedding cache")
    args = parser.parse_args()

    data_dir = Path("data")
    output_dir = Path("output")

    print("\n=== Stage 1: load and filter ===")
    load_and_filter.run(data_dir=data_dir, output_dir=output_dir, n_cards=args.n_cards)

    print("\n=== Stage 2: build text profiles ===")
    build_text_profiles.run(input_path=output_dir / "cards_top5k.parquet")

    print("\n=== Stage 3: embed ===")
    embed.run(
        input_path=output_dir / "cards_top5k.parquet",
        output_dir=output_dir,
        force=args.force_reembed,
        skip_openai=args.skip_openai,
    )

    print("\n=== Stage 4: reduce ===")
    reduce.run(cards_path=output_dir / "cards_top5k.parquet", output_dir=output_dir)

    print("\n=== Stage 5: visualize ===")
    out = visualize.run(
        cards_path=output_dir / "cards_top5k.parquet",
        coords_path=output_dir / "coords.parquet",
        output_path=output_dir / "mtg_graph_poc.html",
    )
    print(f"\nDone. Open {out} in your browser.")


if __name__ == "__main__":
    main()
