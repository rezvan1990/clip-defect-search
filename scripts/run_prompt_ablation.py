#!/usr/bin/env python3
"""Prompt-ensemble ablation on the same cached image features."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.plots import grouped_bars
from defectclip.session import start
from defectclip.util import relativize_paths, write_json
from defectclip.zeroshot import evaluate_zeroshot

VARIANTS = ["minimal", "inspection", "ensemble", "defect_vocab"]


def main() -> None:
    parser = build_parser("Compare prompt sets for zero-shot anomaly AUROC.")
    args = parser.parse_args()
    session = start(args, split="test", with_patches=False)
    per_variant = {}
    for variant in VARIANTS:
        result = evaluate_zeroshot(
            session["records"],
            session["table"]["features"],
            session["model"],
            session["tokenizer"],
            session["device"],
            variant=variant,
            categories=args.categories,
        )
        result.pop("_rows")
        per_variant[variant] = result
        print(f"{variant:14s} mean AUROC {result['mean_auroc']:.4f}")

    cats = list(next(iter(per_variant.values()))["per_category"].keys())
    grouped_bars(
        cats,
        {
            variant: [per_variant[variant]["per_category"][c]["auroc"] for c in cats]
            for variant in VARIANTS
        },
        ylabel="Image AUROC",
        title=f"Prompt ablation ({args.model})",
        path=Path(args.assets) / "prompt_ablation.png",
    )
    payload = {
        "seed": args.seed,
        "model": args.model,
        "pretrained": args.pretrained,
        "mean_auroc": {variant: per_variant[variant]["mean_auroc"] for variant in VARIANTS},
        "per_category": {
            variant: per_variant[variant]["per_category"] for variant in VARIANTS
        },
        "variants": {
            "minimal": "One normal sentence and one anomaly sentence.",
            "inspection": "Six normal and six anomaly states, industrial-inspection template only.",
            "ensemble": "Six states crossed with six templates. Main method.",
            "defect_vocab": "Ensemble plus sentences that name the category's defect types.",
        },
    }
    out = Path(args.results) / "prompt_ablation.json"
    write_json(out, relativize_paths(payload, str(session["root"])))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
