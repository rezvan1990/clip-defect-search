#!/usr/bin/env python3
"""Patch similarity maps and pixel AUROC.

Runs the single-patch map and a 1/2/3-window max map. The figure uses the
defect-text map (the query a person would type) for the median image of each
predefined category.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.cli import build_parser
from defectclip.data import MVTEC_CATEGORIES
from defectclip.localize import contrast_grid, evaluate_localization, similarity_grid
from defectclip.plots import FIGURE_CATEGORIES, _crop_for_display, grouped_bars, localization_grid, median_image_per_category
from defectclip.prompts import retrieval_prompts
from defectclip.session import start
from defectclip.encoder import encode_texts, load_aligned_mask
from defectclip.util import relativize_paths, write_json


def _public(result: dict) -> dict:
    return {k: v for k, v in result.items() if not k.startswith("_")}


def main() -> None:
    parser = build_parser("Patch-level defect localization on MVTec AD.")
    parser.add_argument(
        "--stem",
        default="localization",
        help="Output name. The default writes localization.json and the README figures. Use another stem to compare a second backbone without overwriting those files.",
    )
    args = parser.parse_args()
    stem = args.stem
    if stem == "localization":
        examples_path = Path(args.assets) / "localization_examples.png"
        auroc_path = Path(args.assets) / "localization_auroc_by_category.png"
        json_path = Path(args.results) / "localization.json"
    else:
        examples_path = Path(args.assets) / f"{stem}_examples.png"
        auroc_path = Path(args.assets) / f"{stem}_auroc.png"
        json_path = Path(args.results) / f"{stem}.json"
    session = start(args, split="test", with_patches=True)
    patches = session["table"]["patches"]
    single = evaluate_localization(
        session["records"],
        patches,
        session["model"],
        session["tokenizer"],
        session["device"],
        session["mask_transform"],
        categories=args.categories,
        windows=(1,),
    )
    multi = evaluate_localization(
        session["records"],
        patches,
        session["model"],
        session["tokenizer"],
        session["device"],
        session["mask_transform"],
        categories=args.categories,
        windows=(1, 2, 3),
    )

    cats = [c for c in MVTEC_CATEGORIES if c in single["per_category"]]
    grouped_bars(
        cats,
        {
            "generic, all pixels": [single["per_category"][c]["pixel_auroc_generic_all"] for c in cats],
            "defect text, anomaly images": [
                single["per_category"][c]["pixel_auroc_defect_text_anomaly_images"] for c in cats
            ],
        },
        ylabel="Pixel AUROC",
        title=f"Localization ({args.model}, single patch scale)",
        path=auroc_path,
    )

    by_cat = {cat: row["images"] for cat, row in single["_per_category_full"].items()}
    chosen = median_image_per_category(by_cat, [c for c in FIGURE_CATEGORIES if c in by_cat], "guided_pixel_auroc")
    figure_rows = []
    proto_cache = {}
    for row in chosen:
        key = (row["category"], row["defect"])
        if key not in proto_cache:
            proto_cache[key] = encode_texts(
                session["model"],
                session["tokenizer"],
                retrieval_prompts(row["category"], row["defect"]),
                session["device"],
            ).numpy()
        heat = similarity_grid(patches[row["index"]], proto_cache[key], windows=(1,))
        mask = load_aligned_mask(row["mask_path"], session["mask_transform"]).numpy()
        auroc = row["guided_pixel_auroc"]
        auroc_txt = "n/a" if auroc is None else f"{auroc:.2f}"
        figure_rows.append(
            {
                "image": _crop_for_display(row["path"], session["preprocess"]),
                "heat": heat,
                "mask": mask,
                "title": f"{row['category']} / {row['defect'].replace('_', ' ')}\npixel AUROC {auroc_txt}",
            }
        )
    if figure_rows:
        localization_grid(figure_rows, examples_path)

    generic_chosen = median_image_per_category(by_cat, [c for c in FIGURE_CATEGORIES if c in by_cat], "generic_pixel_auroc")
    generic_rows = []
    generic_proto = {}
    from defectclip.zeroshot import category_prototypes

    for row in generic_chosen:
        if row["category"] not in generic_proto:
            names = sorted({r.defect for r in session["records"] if r.category == row["category"] and r.is_anomaly})
            generic_proto[row["category"]] = category_prototypes(
                session["model"],
                session["tokenizer"],
                row["category"],
                session["device"],
                variant="ensemble",
                defect_names=names,
            )
        normal_proto, anomaly_proto, _n = generic_proto[row["category"]]
        heat = contrast_grid(patches[row["index"]], normal_proto, anomaly_proto, windows=(1,))
        mask = load_aligned_mask(row["mask_path"], session["mask_transform"]).numpy()
        auroc = row["generic_pixel_auroc"]
        auroc_txt = "n/a" if auroc is None else f"{auroc:.2f}"
        generic_rows.append(
            {
                "image": _crop_for_display(row["path"], session["preprocess"]),
                "heat": heat,
                "mask": mask,
                "title": f"{row['category']} / {row['defect'].replace('_', ' ')}\ngeneric AUROC {auroc_txt}",
            }
        )
    generic_path = examples_path.with_name(examples_path.stem + "_generic" + examples_path.suffix)
    if generic_rows:
        localization_grid(
            generic_rows,
            generic_path,
            suptitle="Generic anomaly-minus-normal map (no defect name), next to the mask.",
        )

    def _slim(result: dict) -> dict:
        full = result["_per_category_full"]
        out = _public(result)
        out["per_category"] = {}
        for cat, row in full.items():
            out["per_category"][cat] = {
                k: v for k, v in row.items() if k != "images"
            }
            out["per_category"][cat]["per_image"] = [
                {k: v for k, v in image.items() if k != "index"} for image in row["images"]
            ]
        return out

    payload = {
        "seed": args.seed,
        "model": args.model,
        "pretrained": args.pretrained,
        "single_scale": _slim(single),
        "multiscale_1_2_3": _slim(multi),
        "definitions": {
            "generic": "Cosine to the anomaly prototype minus cosine to the normal prototype. No defect name.",
            "defect_text": "Cosine to the text description of that image's defect type. Uses the type name as the query.",
            "pixels": "Masks and scores are compared on the 224 CLIP crop, not the original resolution.",
            "figure": "Median per-image defect-text pixel AUROC inside each predefined category.",
        },
    }
    out = json_path
    write_json(out, relativize_paths(payload, str(session["root"])))
    print("single-scale mean pixel AUROC")
    print(f"  generic, all test pixels:          {single['mean_pixel_auroc_generic_all']:.4f}")
    print(f"  generic, anomaly images:           {single['mean_pixel_auroc_generic_anomaly_images']:.4f}")
    print(f"  defect text, anomaly images:       {single['mean_pixel_auroc_defect_text_anomaly_images']:.4f}")
    print("windows 1+2+3 mean pixel AUROC")
    print(f"  generic, all test pixels:          {multi['mean_pixel_auroc_generic_all']:.4f}")
    print(f"  generic, anomaly images:           {multi['mean_pixel_auroc_generic_anomaly_images']:.4f}")
    print(f"  defect text, anomaly images:       {multi['mean_pixel_auroc_defect_text_anomaly_images']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
