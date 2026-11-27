#!/usr/bin/env python3
"""Zero-shot anomaly classification with a prompt ensemble."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.data import MVTEC_CATEGORIES
from defectclip.plots import barh
from defectclip.session import start
from defectclip.util import relativize_paths, write_json
from defectclip.zeroshot import evaluate_zeroshot


def main() -> None:
    parser = build_parser("Zero-shot image AUROC on MVTec AD.")
    parser.add_argument("--variant", default="ensemble", choices=["minimal", "ensemble", "inspection", "defect_vocab"])
    args = parser.parse_args()
    # Patches are saved alongside image features so later localization does not encode twice.
    session = start(args, split="test", with_patches=True)
    session["table"].pop("patches", None)
    result = evaluate_zeroshot(
        session["records"],
        session["table"]["features"],
        session["model"],
        session["tokenizer"],
        session["device"],
        variant=args.variant,
        categories=args.categories,
    )
    rows = result.pop("_rows")
    cats = [c for c in MVTEC_CATEGORIES if c in result["per_category"]]
    values = [result["per_category"][c]["auroc"] for c in cats]
    barh(
        cats,
        values,
        xlabel="Image AUROC",
        title=f"Zero-shot anomaly AUROC ({args.model}, {args.variant})",
        path=Path(args.assets) / "zeroshot_auroc_by_category.png",
    )
    payload = {
        "seed": args.seed,
        "model": args.model,
        "pretrained": args.pretrained,
        "variant": args.variant,
        "mean_auroc": result["mean_auroc"],
        "per_category": result["per_category"],
        "definition": "Image AUROC of the cosine gap between the anomaly prompt prototype and the normal prompt prototype.",
    }
    out = Path(args.results) / "zeroshot.json"
    write_json(out, relativize_paths(payload, str(session["root"])))
    print(f"mean image AUROC ({args.variant}): {result['mean_auroc']:.4f}")
    for cat in cats:
        print(f"  {cat:12s} {result['per_category'][cat]['auroc']:.4f}")
    print(f"wrote {out}")
    del rows


if __name__ == "__main__":
    main()
