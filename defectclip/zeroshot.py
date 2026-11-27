"""Image-level zero-shot anomaly classification.

Each category gets a normal text prototype and an anomaly text prototype.
The score of an image is the cosine gap between those two prototypes.
For two prototypes, a softmax over the same gap is a monotone transform, so
image AUROC would be identical. The gap is reported directly.
"""

from __future__ import annotations

import numpy as np
import torch

from defectclip.data import MVTEC_CATEGORIES, Record
from defectclip.encoder import encode_texts
from defectclip.metrics import safe_auroc
from defectclip.prompts import state_prompts
from defectclip.util import mean_ignore_none


def category_prototypes(
    model,
    tokenizer,
    category: str,
    device: torch.device,
    variant: str,
    defect_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    normal_prompts, anomaly_prompts = state_prompts(category, variant=variant, defect_names=defect_names)
    normal = encode_texts(model, tokenizer, normal_prompts, device).numpy()
    anomaly = encode_texts(model, tokenizer, anomaly_prompts, device).numpy()
    return normal, anomaly, len(normal_prompts)


def score_images(features: np.ndarray, normal_proto: np.ndarray, anomaly_proto: np.ndarray) -> np.ndarray:
    return features @ anomaly_proto - features @ normal_proto


def evaluate_zeroshot(
    records: list[Record],
    features: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    variant: str = "ensemble",
    categories: list[str] | None = None,
) -> dict:
    if len(records) != len(features):
        raise ValueError("records and features have different lengths")
    cats = categories or [c for c in MVTEC_CATEGORIES if any(r.category == c for r in records)]
    per_category = {}
    for category in cats:
        index = np.array([i for i, r in enumerate(records) if r.category == category])
        if len(index) == 0:
            continue
        labels = np.array([records[i].is_anomaly for i in index], dtype=int)
        defect_names = sorted({records[i].defect for i in index if records[i].is_anomaly})
        normal, anomaly, n_prompts = category_prototypes(
            model,
            tokenizer,
            category,
            device,
            variant,
            defect_names=defect_names,
        )
        scores = score_images(features[index], normal, anomaly)
        per_category[category] = {
            "auroc": safe_auroc(labels, scores),
            "n": int(len(index)),
            "n_normal": int((labels == 0).sum()),
            "n_anomaly": int((labels == 1).sum()),
            "n_prompts_per_state": int(n_prompts) if variant != "defect_vocab" else None,
            "n_normal_prompts": int(n_prompts),
            "n_anomaly_prompts": int(
                len(state_prompts(category, variant, defect_names)[1])
            ),
            "scores": scores,
            "index": index,
            "labels": labels,
        }
    summary = {
        "variant": variant,
        "mean_auroc": mean_ignore_none(v["auroc"] for v in per_category.values()),
        "per_category": {
            cat: {
                "auroc": row["auroc"],
                "n": row["n"],
                "n_normal": row["n_normal"],
                "n_anomaly": row["n_anomaly"],
                "n_normal_prompts": row["n_normal_prompts"],
                "n_anomaly_prompts": row["n_anomaly_prompts"],
            }
            for cat, row in per_category.items()
        },
    }
    summary["_rows"] = per_category
    return summary
