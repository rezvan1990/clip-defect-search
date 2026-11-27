#!/usr/bin/env python3
"""Compare OpenAI CLIP backbones on the same protocol."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.localize import evaluate_localization
from defectclip.plots import grouped_bars
from defectclip.retrieval import evaluate_retrieval
from defectclip.session import start
from defectclip.util import relativize_paths, write_json
from defectclip.zeroshot import evaluate_zeroshot

BACKBONES = [
    ("ViT-B-32", "openai"),
    ("ViT-B-16", "openai"),
    ("RN50", "openai"),
]


def main() -> None:
    parser = build_parser("Image AUROC, retrieval mAP, and pixel AUROC for three CLIP backbones.")
    args = parser.parse_args()
    summaries = {}
    root = None
    for model_name, pretrained in BACKBONES:
        args.model = model_name
        args.pretrained = pretrained
        session = start(args, split="test", with_patches=True)
        root = session["root"]
        zeroshot = evaluate_zeroshot(
            session["records"],
            session["table"]["features"],
            session["model"],
            session["tokenizer"],
            session["device"],
            variant="ensemble",
            categories=args.categories,
        )
        zeroshot.pop("_rows")
        retrieval = evaluate_retrieval(
            session["records"],
            session["table"]["features"],
            session["model"],
            session["tokenizer"],
            session["device"],
            categories=args.categories,
        )
        localization = evaluate_localization(
            session["records"],
            session["table"]["patches"],
            session["model"],
            session["tokenizer"],
            session["device"],
            session["mask_transform"],
            categories=args.categories,
            windows=(1,),
        )
        summaries[model_name] = {
            "pretrained": pretrained,
            "mean_image_auroc": zeroshot["mean_auroc"],
            "macro_mAP": retrieval["macro_mAP"],
            "macro_recall@5": retrieval["macro_recall@5"],
            "macro_hit@5": retrieval["macro_hit@5"],
            "mean_pixel_auroc_generic_anomaly_images": localization["mean_pixel_auroc_generic_anomaly_images"],
            "mean_pixel_auroc_defect_text_anomaly_images": localization["mean_pixel_auroc_defect_text_anomaly_images"],
            "image_auroc_per_category": {c: row["auroc"] for c, row in zeroshot["per_category"].items()},
            "map_per_category": {c: row["mAP"] for c, row in retrieval["per_category"].items()},
        }
        print(
            f"{model_name:10s} AUROC {zeroshot['mean_auroc']:.4f}  "
            f"mAP {retrieval['macro_mAP']:.4f}  "
            f"pixAUROC {localization['mean_pixel_auroc_generic_anomaly_images']:.4f}"
        )
        # Free the patch tensor before the next backbone.
        del session

    labels = [name for name, _pre in BACKBONES]
    grouped_bars(
        ["image AUROC", "retrieval mAP", "pixel AUROC"],
        {
            name: [
                summaries[name]["mean_image_auroc"],
                summaries[name]["macro_mAP"],
                summaries[name]["mean_pixel_auroc_generic_anomaly_images"],
            ]
            for name in labels
        },
        ylabel="Score",
        title="CLIP backbone comparison (OpenAI weights)",
        path=Path(args.assets) / "backbone_comparison.png",
    )
    payload = {
        "seed": args.seed,
        "backbones": summaries,
        "pixel_auroc": "Generic contrast map, anomalous test images only, single patch scale.",
        "retrieval": "Category-macro mAP of defect-type queries.",
    }
    out = Path(args.results) / "backbone_comparison.json"
    write_json(out, relativize_paths(payload, str(root)))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
