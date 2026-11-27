#!/usr/bin/env python3
"""Normal-only nearest neighbors and a light linear probe, versus zero-shot."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectclip.adapt import evaluate_adaptation
from defectclip.cache import ensure_features
from defectclip.cli import build_parser
from defectclip.data import load_mvtec
from defectclip.plots import barh
from defectclip.session import start
from defectclip.util import pick_device, relativize_paths, set_seed, write_json

METHOD_ORDER = [
    ("zero_shot", "zero-shot"),
    ("nn4", "NN-4"),
    ("nn16", "NN-16"),
    ("nn_all", "NN-all"),
    ("zero_shot_heldout", "zero-shot, held-out"),
    ("probe_gap_nn16", "probe, 2 features"),
    ("probe_embedding", "probe, embedding"),
]


def main() -> None:
    parser = build_parser("Few-shot adaptation versus the text-only score.")
    args = parser.parse_args()
    set_seed(args.seed)
    session = start(args, split="test", with_patches=False)
    train_records = load_mvtec(session["root"], split="train", categories=args.categories)
    print(f"train: {len(train_records)} images")
    train_table = ensure_features(
        train_records,
        split="train",
        model_name=args.model,
        pretrained=args.pretrained,
        device=pick_device(args.device),
        batch_size=args.batch_size,
        cache_root=args.cache,
        with_patches=False,
    )
    result = evaluate_adaptation(
        session["records"],
        session["table"]["features"],
        train_records,
        train_table["features"],
        session["model"],
        session["tokenizer"],
        session["device"],
        seed=args.seed,
        categories=args.categories,
    )
    labels = [label for _key, label in METHOD_ORDER]
    values = [result["mean_auroc"][key] for key, _label in METHOD_ORDER]
    barh(
        labels,
        values,
        xlabel="Mean image AUROC",
        title=f"Adaptation versus zero-shot ({args.model})",
        path=Path(args.assets) / "fewshot_adaptation.png",
        show_mean=False,
    )
    # per_category contains absolute calibration paths; relativize below.
    payload = {
        "seed": args.seed,
        "model": args.model,
        "pretrained": args.pretrained,
        "data_root_name": session["root"].name,
        "mean_auroc": result["mean_auroc"],
        "per_category": result["per_category"],
        "protocol_counts": result["protocol_counts"],
        "notes": result["notes"],
    }
    out = Path(args.results) / "fewshot.json"
    write_json(out, relativize_paths(payload, str(session["root"])))
    for key, label in METHOD_ORDER:
        value = result["mean_auroc"][key]
        text = "n/a" if value is None else f"{value:.4f}"
        print(f"{label:22s} {text}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
