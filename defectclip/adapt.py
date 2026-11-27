"""Few-shot adaptation on frozen CLIP embeddings.

Two references use only normal training images (no anomaly labels):

- NN-k: anomaly score is 1 minus the cosine similarity to the nearest of k
  normal training embeddings. k is 4, 16, or the full training split.
  The k-shot subsets are nested prefixes of one permutation.

The linear probes use a handful of labeled anomalies and are scored only on
the images that were not used to fit them:

- one anomalous test image per defect type, plus 8 normal training images
  that are disjoint from the 16-shot support
- a 2-d probe on (zero-shot gap, NN-16 distance)
- a 512-d probe on the CLIP embedding itself, with strong L2 regularization

Zero-shot AUROC is reported on the full test set and again on the same
held-out images as the probes, so those rows can be compared.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from defectclip.data import MVTEC_CATEGORIES, Record, defect_types
from defectclip.metrics import safe_auroc
from defectclip.util import mean_ignore_none
from defectclip.zeroshot import category_prototypes, score_images


def _nn_distance(query: np.ndarray, support: np.ndarray) -> np.ndarray:
    if len(support) == 0:
        raise RuntimeError("Nearest-neighbor support is empty")
    sims = query @ support.T
    return 1.0 - sims.max(axis=1)


def _fit_probe(train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, C: float) -> np.ndarray | None:
    if len(np.unique(train_y)) < 2:
        return None
    scaler = StandardScaler()
    model = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
    model.fit(scaler.fit_transform(train_x), train_y)
    return model.decision_function(scaler.transform(eval_x))


def evaluate_adaptation(
    test_records: list[Record],
    test_features: np.ndarray,
    train_records: list[Record],
    train_features: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    seed: int,
    categories: list[str] | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    cats = categories or [c for c in MVTEC_CATEGORIES if any(r.category == c for r in test_records)]
    methods = [
        "zero_shot",
        "nn4",
        "nn16",
        "nn_all",
        "zero_shot_heldout",
        "probe_gap_nn16",
        "probe_embedding",
    ]
    buckets = {name: [] for name in methods}
    per_category = {}
    protocol = []

    for category in cats:
        test_index = np.array([i for i, r in enumerate(test_records) if r.category == category])
        train_index = np.array([i for i, r in enumerate(train_records) if r.category == category and not r.is_anomaly])
        if len(test_index) == 0 or len(train_index) == 0:
            continue
        test_labels = np.array([test_records[i].is_anomaly for i in test_index], dtype=int)
        names = defect_types(test_records, category)
        normal_proto, anomaly_proto, _n = category_prototypes(
            model, tokenizer, category, device, variant="ensemble", defect_names=names
        )
        test_gap = score_images(test_features[test_index], normal_proto, anomaly_proto)
        train_gap = score_images(train_features[train_index], normal_proto, anomaly_proto)

        order = rng.permutation(len(train_index))
        k4 = min(4, len(order))
        k16 = min(16, len(order))
        n_cal = min(8, max(len(order) - k16, 0))
        support4 = train_features[train_index[order[:k4]]]
        support16 = train_features[train_index[order[:k16]]]
        support_all = train_features[train_index]
        cal_normal_local = order[k16 : k16 + n_cal]

        nn4 = _nn_distance(test_features[test_index], support4)
        nn16 = _nn_distance(test_features[test_index], support16)
        nn_all = _nn_distance(test_features[test_index], support_all)

        held_out = []
        for defect in names:
            local_pos = [j for j, i in enumerate(test_index) if test_records[i].defect == defect]
            if not local_pos:
                continue
            pick = int(rng.choice(local_pos))
            held_out.append(pick)
        held_out = np.array(held_out, dtype=int)
        eval_mask = np.ones(len(test_index), dtype=bool)
        eval_mask[held_out] = False

        row = {
            "n_train_normal": int(len(train_index)),
            "n_test": int(len(test_index)),
            "k4": int(k4),
            "k16": int(k16),
            "n_calibration_normals": int(n_cal),
            "n_calibration_anomalies": int(len(held_out)),
            "n_eval": int(eval_mask.sum()),
            "calibration_anomaly_paths": [test_records[test_index[j]].path for j in held_out],
            "auroc": {},
        }
        row["auroc"]["zero_shot"] = safe_auroc(test_labels, test_gap)
        row["auroc"]["nn4"] = safe_auroc(test_labels, nn4)
        row["auroc"]["nn16"] = safe_auroc(test_labels, nn16)
        row["auroc"]["nn_all"] = safe_auroc(test_labels, nn_all)
        row["auroc"]["zero_shot_heldout"] = safe_auroc(test_labels[eval_mask], test_gap[eval_mask])

        probe_gap = None
        probe_emb = None
        if n_cal > 0 and len(held_out) > 0 and eval_mask.any():
            cal_nn = _nn_distance(train_features[train_index[cal_normal_local]], support16)
            X_cal = np.concatenate(
                [
                    np.stack([train_gap[cal_normal_local], cal_nn], axis=1),
                    np.stack([test_gap[held_out], nn16[held_out]], axis=1),
                ],
                axis=0,
            )
            y_cal = np.concatenate(
                [
                    np.zeros(len(cal_normal_local), dtype=int),
                    np.ones(len(held_out), dtype=int),
                ]
            )
            probe_gap = _fit_probe(X_cal, y_cal, np.stack([test_gap[eval_mask], nn16[eval_mask]], axis=1), C=1.0)

            X_cal_emb = np.concatenate(
                [
                    train_features[train_index[cal_normal_local]],
                    test_features[test_index[held_out]],
                ],
                axis=0,
            )
            probe_emb = _fit_probe(X_cal_emb, y_cal, test_features[test_index[eval_mask]], C=0.01)
        if probe_gap is not None:
            row["auroc"]["probe_gap_nn16"] = safe_auroc(test_labels[eval_mask], probe_gap)
        else:
            row["auroc"]["probe_gap_nn16"] = None
        if probe_emb is not None:
            row["auroc"]["probe_embedding"] = safe_auroc(test_labels[eval_mask], probe_emb)
        else:
            row["auroc"]["probe_embedding"] = None

        for name in methods:
            buckets[name].append(row["auroc"][name])
        per_category[category] = row
        protocol.append(
            {
                "category": category,
                "k4": row["k4"],
                "k16": row["k16"],
                "n_calibration_normals": row["n_calibration_normals"],
                "n_calibration_anomalies": row["n_calibration_anomalies"],
                "n_eval": row["n_eval"],
            }
        )

    return {
        "seed": seed,
        "mean_auroc": {name: mean_ignore_none(values) for name, values in buckets.items()},
        "per_category": per_category,
        "protocol_counts": protocol,
        "notes": {
            "nn": "1 - max cosine to k normal training images. No anomaly labels. Full test set.",
            "probes": (
                "Logistic regression fit on 8 normal training images plus one test anomaly "
                "per defect type. Scored on the remaining test images. "
                "probe_gap_nn16 uses two features (zero-shot gap, NN-16 distance), C=1. "
                "probe_embedding uses the CLIP vector, C=0.01."
            ),
            "zero_shot_heldout": "Same text score as zero_shot, restricted to the probe's eval images.",
        },
    }
