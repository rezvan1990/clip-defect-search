"""Text-to-image retrieval of a named defect type.

The gallery for a query is every test image of that category, including
defect-free images and the other defect types. An image is relevant when its
folder name matches the queried defect.
"""

from __future__ import annotations

import numpy as np
import torch

from defectclip.data import MVTEC_CATEGORIES, Record, defect_types
from defectclip.encoder import encode_texts
from defectclip.metrics import average_precision, hit_at_k, rank_relevance, recall_at_k
from defectclip.prompts import retrieval_prompts, retrieval_query_text
from defectclip.util import mean_ignore_none

K_LIST = (1, 5, 10)


def evaluate_retrieval(
    records: list[Record],
    features: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    categories: list[str] | None = None,
) -> dict:
    cats = categories or [c for c in MVTEC_CATEGORIES if any(r.category == c for r in records)]
    queries = []
    per_category = {}
    for category in cats:
        index = np.array([i for i, r in enumerate(records) if r.category == category])
        gallery = features[index]
        names = defect_types(records, category)
        cat_queries = []
        for defect in names:
            texts = retrieval_prompts(category, defect)
            proto = encode_texts(model, tokenizer, texts, device).numpy()
            scores = gallery @ proto
            relevance = np.array([records[i].defect == defect for i in index], dtype=bool)
            ranked = rank_relevance(relevance, scores)
            order = np.argsort(-scores, kind="mergesort")
            row = {
                "category": category,
                "defect": defect,
                "query": retrieval_query_text(category, defect),
                "n_gallery": int(len(index)),
                "n_relevant": int(relevance.sum()),
                "ap": average_precision(relevance, scores),
            }
            for k in K_LIST:
                row[f"recall@{k}"] = recall_at_k(ranked, k)
                row[f"hit@{k}"] = hit_at_k(ranked, k)
            row["top"] = [
                {
                    "path": records[index[j]].path,
                    "defect": records[index[j]].defect,
                    "relevant": bool(relevance[j]),
                    "score": float(scores[j]),
                }
                for j in order[:10]
            ]
            cat_queries.append(row)
            queries.append(row)
        per_category[category] = {
            "n_queries": len(cat_queries),
            "mAP": mean_ignore_none(q["ap"] for q in cat_queries),
            **{
                f"recall@{k}": mean_ignore_none(q[f"recall@{k}"] for q in cat_queries)
                for k in K_LIST
            },
            **{
                f"hit@{k}": mean_ignore_none(q[f"hit@{k}"] for q in cat_queries)
                for k in K_LIST
            },
        }
    summary = {
        "n_queries": len(queries),
        "macro_mAP": mean_ignore_none(v["mAP"] for v in per_category.values()),
        "micro_mAP": mean_ignore_none(q["ap"] for q in queries),
        "per_category": per_category,
        "queries": queries,
    }
    for k in K_LIST:
        summary[f"macro_recall@{k}"] = mean_ignore_none(v[f"recall@{k}"] for v in per_category.values())
        summary[f"macro_hit@{k}"] = mean_ignore_none(v[f"hit@{k}"] for v in per_category.values())
    return summary
