#!/usr/bin/env python3
"""Text-to-image retrieval of MVTec defect types."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.data import MVTEC_CATEGORIES
from defectclip.plots import FIGURE_CATEGORIES, barh, median_query_per_category, retrieval_grid
from defectclip.retrieval import evaluate_retrieval
from defectclip.session import start
from defectclip.util import relativize_paths, write_json


def main() -> None:
    parser = build_parser("Text-to-image defect retrieval on MVTec AD.")
    args = parser.parse_args()
    session = start(args, split="test", with_patches=False)
    result = evaluate_retrieval(
        session["records"],
        session["table"]["features"],
        session["model"],
        session["tokenizer"],
        session["device"],
        categories=args.categories,
    )
    cats = [c for c in MVTEC_CATEGORIES if c in result["per_category"]]
    barh(
        cats,
        [result["per_category"][c]["mAP"] for c in cats],
        xlabel="mAP (mean over defect queries)",
        title=f"Defect retrieval mAP ({args.model})",
        path=Path(args.assets) / "retrieval_map_by_category.png",
    )
    figure_cats = [c for c in FIGURE_CATEGORIES if c in result["per_category"]]
    chosen = median_query_per_category(result["queries"], figure_cats)
    if chosen:
        k = min(5, min(len(query["top"]) for query in chosen))
        retrieval_grid(chosen, Path(args.assets) / "retrieval_grid.png", k=k)

    # Drop the bulky ranked lists from the public JSON after the figure is drawn,
    # but keep the top-5 so the figure can be audited.
    public_queries = []
    for query in result["queries"]:
        public_queries.append(
            {
                **{k: v for k, v in query.items() if k != "top"},
                "top5": query["top"][:5],
            }
        )
    payload = {
        "seed": args.seed,
        "model": args.model,
        "pretrained": args.pretrained,
        "n_queries": result["n_queries"],
        "macro_mAP": result["macro_mAP"],
        "micro_mAP": result["micro_mAP"],
        **{k: result[k] for k in result if k.startswith("macro_")},
        "per_category": result["per_category"],
        "queries": public_queries,
        "definitions": {
            "gallery": "All test images of the queried category, including good images and other defects.",
            "relevant": "Test images whose defect-folder name equals the query.",
            "recall@k": "Fraction of relevant images that fall in the top k.",
            "hit@k": "1 if the top k contains at least one relevant image.",
            "mAP": "Mean average precision. macro_* averages categories equally; micro_mAP averages queries equally.",
            "figure": "For each figure category, the query with the median AP. Not the best query.",
        },
    }
    out = Path(args.results) / "retrieval.json"
    write_json(out, relativize_paths(payload, str(session["root"])))
    print(f"macro mAP: {result['macro_mAP']:.4f}")
    print(f"macro recall@1/5/10: {result['macro_recall@1']:.4f} / {result['macro_recall@5']:.4f} / {result['macro_recall@10']:.4f}")
    print(f"macro hit@1/5/10: {result['macro_hit@1']:.4f} / {result['macro_hit@5']:.4f} / {result['macro_hit@10']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
