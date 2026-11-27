"""Command-line flags shared by the experiment scripts."""

from __future__ import annotations

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", default="data/mvtec", help="Path to the extracted MVTec AD root.")
    parser.add_argument("--model", default="ViT-B-16", help="open_clip model name.")
    parser.add_argument("--pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda.")
    parser.add_argument("--cache", default="cache", help="Feature-cache directory (not committed).")
    parser.add_argument("--results", default="results")
    parser.add_argument("--assets", default="assets")
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional subset of MVTec categories. Default: all 15.",
    )
    return parser
