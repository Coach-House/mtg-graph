"""MTG card vector-embedding graph — proof of concept."""

from __future__ import annotations

from mtg_graph import build_text_profiles, embed, load_and_filter, reduce, visualize

__version__ = "0.1.0"

__all__ = [
    "build_text_profiles",
    "embed",
    "load_and_filter",
    "reduce",
    "visualize",
]
