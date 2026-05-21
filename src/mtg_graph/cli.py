"""Single CLI entrypoint that chains all pipeline stages.

Usage:
    uv run mtg-graph-poc [--n-cards N] [--skip-openai] [--force-reembed] [--open {2d,3d,none}]

Always writes both `output/mtg_graph_poc.html` (2D) and `output/mtg_graph_poc_3d.html` (3D).
`--open` controls which one is auto-launched in the browser (default: none).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mtg_graph import build_text_profiles, cluster, embed, load_and_filter, reduce, visualize


def main() -> None:
    parser = argparse.ArgumentParser(description="MTG card embedding pipeline (5k POC)")
    parser.add_argument("--n-cards", type=int, default=5000, help="How many cards to include (default 5000)")
    parser.add_argument("--skip-openai", action="store_true", help="Skip OpenAI even if API key is set")
    parser.add_argument("--force-reembed", action="store_true", help="Invalidate embedding cache")
    parser.add_argument("--open", choices=["2d", "3d", "none"], default="none", help="Auto-open the rendered HTML")
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

    print("\n=== Stage 4.5: cluster ===")
    cluster.run(cards_path=output_dir / "cards_top5k.parquet", output_dir=output_dir)

    print("\n=== Stage 5: visualize ===")
    outputs = visualize.run(
        cards_path=output_dir / "cards_top5k.parquet",
        coords_path=output_dir / "coords.parquet",
        output_dir=output_dir,
    )
    print()
    for key, path in outputs.items():
        print(f"  {key}: {path}")

    if args.open != "none" and args.open in outputs:
        subprocess.run(["open", str(outputs[args.open])], check=False)


if __name__ == "__main__":
    main()
