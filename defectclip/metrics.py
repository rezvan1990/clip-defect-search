"""Retrieval and detection metrics.

Retrieval uses the information-retrieval definitions, not the single-positive
Recall@K from the CLIP paper:

- Recall@k is the fraction of relevant gallery images that appear in the top k.
- Hit@k is 1 when at least one relevant image is in the top k.
- Average precision uses sklearn's implementation (area under the precision-recall curve).

Image and pixel AUROC are the usual ranking scores: 1.0 is perfect, 0.5 is chance.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def rank_relevance(relevance: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Boolean relevance sorted from the highest score to the lowest."""
    relevance = np.asarray(relevance).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="mergesort")
    return relevance[order]


def recall_at_k(relevance_ranked: np.ndarray, k: int) -> float | None:
    n_rel = int(relevance_ranked.sum())
    if n_rel == 0:
        return None
    k = min(k, int(relevance_ranked.shape[0]))
    return float(relevance_ranked[:k].sum() / n_rel)


def hit_at_k(relevance_ranked: np.ndarray, k: int) -> float | None:
    if int(relevance_ranked.sum()) == 0:
        return None
    k = min(k, int(relevance_ranked.shape[0]))
    return float(bool(relevance_ranked[:k].any()))


def average_precision(relevance: np.ndarray, scores: np.ndarray) -> float | None:
    relevance = np.asarray(relevance).astype(int)
    if relevance.sum() == 0 or relevance.sum() == len(relevance):
        return None
    return float(average_precision_score(relevance, np.asarray(scores, dtype=np.float64)))


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels).astype(int)
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))
