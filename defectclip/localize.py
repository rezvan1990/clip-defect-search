"""Patch-level similarity maps.

Two maps are computed for every anomalous test image:

- generic: cosine to the anomaly prototype minus cosine to the normal prototype.
  This does not use the defect name.
- defect: cosine to the text description of that image's own defect type.
  This is text-guided localization. It uses the defect name as the query,
  which is what a user types, and it is not a fair unsupervised score.

Pixel labels are the MVTec masks after the same resize and center crop the
image encoder applied. Scores are bilinearly upsampled to that 224 crop.
"""

from __future__ import annotations

import numpy as np
import torch

from defectclip.data import MVTEC_CATEGORIES, Record
from defectclip.encoder import load_aligned_mask, upsample_map
from defectclip.metrics import safe_auroc
from defectclip.prompts import defect_phrase, object_phrase, retrieval_prompts
from defectclip.util import mean_ignore_none
from defectclip.zeroshot import category_prototypes


def _pool_windows(tokens_hw: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return tokens_hw
    pad = k // 2
    padded = np.pad(tokens_hw, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    height, width, _dim = tokens_hw.shape
    out = np.empty_like(tokens_hw)
    for i in range(height):
        for j in range(width):
            out[i, j] = padded[i : i + k, j : j + k].mean(axis=(0, 1))
    return out


def similarity_grid(tokens: np.ndarray, prototype: np.ndarray, windows: tuple[int, ...] = (1,)) -> np.ndarray:
    """tokens: [P, D] normalized. Returns a 224x224 similarity map.

    With more than one window the maps are upsampled and combined by a max,
    which keeps the sharper of the small-window and large-window responses.
    """
    n_tokens, _dim = tokens.shape
    grid = int(round(n_tokens ** 0.5))
    if grid * grid != n_tokens:
        raise RuntimeError(f"Patch count {n_tokens} is not a square grid")
    spatial = tokens.reshape(grid, grid, _dim)
    maps = []
    for k in windows:
        pooled = _pool_windows(spatial, k)
        norm = np.linalg.norm(pooled, axis=-1, keepdims=True)
        pooled = pooled / np.clip(norm, 1e-8, None)
        score = pooled @ prototype
        maps.append(upsample_map(score))
    if len(maps) == 1:
        return maps[0]
    return np.maximum.reduce(maps)


def contrast_grid(tokens: np.ndarray, normal_proto: np.ndarray, anomaly_proto: np.ndarray, windows=(1,)) -> np.ndarray:
    """Per-window anomaly-minus-normal cosine, upsampled, then max over windows."""
    n_tokens, dim = tokens.shape
    grid = int(round(n_tokens ** 0.5))
    if grid * grid != n_tokens:
        raise RuntimeError(f"Patch count {n_tokens} is not a square grid")
    spatial = tokens.reshape(grid, grid, dim)
    maps = []
    for k in windows:
        pooled = _pool_windows(spatial, k)
        pooled = pooled / np.clip(np.linalg.norm(pooled, axis=-1, keepdims=True), 1e-8, None)
        score = pooled @ anomaly_proto - pooled @ normal_proto
        maps.append(upsample_map(score))
    if len(maps) == 1:
        return maps[0]
    return np.maximum.reduce(maps)


def evaluate_localization(
    records: list[Record],
    patches: np.ndarray,
    model,
    tokenizer,
    device: torch.device,
    mask_transform,
    categories: list[str] | None = None,
    windows: tuple[int, ...] = (1,),
) -> dict:
    cats = categories or [c for c in MVTEC_CATEGORIES if any(r.category == c for r in records)]
    per_category = {}
    # Kept for the figure script: one entry per anomalous image.
    for category in cats:
        index = [i for i, r in enumerate(records) if r.category == category]
        if not index:
            continue
        defect_names = sorted({records[i].defect for i in index if records[i].is_anomaly})
        normal, anomaly, _n = category_prototypes(
            model, tokenizer, category, device, variant="ensemble", defect_names=defect_names
        )
        from defectclip.encoder import encode_texts

        defect_proto = {}
        for name in defect_names:
            defect_proto[name] = encode_texts(
                model, tokenizer, retrieval_prompts(category, name), device
            ).numpy()

        generic_scores = []
        generic_labels = []
        generic_anom_scores = []
        generic_anom_labels = []
        guided_scores = []
        guided_labels = []
        image_rows = []

        for i in index:
            record = records[i]
            heat = contrast_grid(patches[i], normal, anomaly, windows=windows)
            mask = load_aligned_mask(record.mask_path, mask_transform).numpy()
            generic_scores.append(heat.reshape(-1))
            generic_labels.append((mask > 0).reshape(-1).astype(int))
            if record.is_anomaly and record.mask_path:
                generic_anom_scores.append(heat.reshape(-1))
                generic_anom_labels.append((mask > 0).reshape(-1).astype(int))
                guided = similarity_grid(patches[i], defect_proto[record.defect], windows=windows)
                guided_scores.append(guided.reshape(-1))
                guided_labels.append((mask > 0).reshape(-1).astype(int))
                image_auroc = safe_auroc((mask > 0).astype(int).reshape(-1), guided.reshape(-1))
                generic_image_auroc = safe_auroc((mask > 0).astype(int).reshape(-1), heat.reshape(-1))
                image_rows.append(
                    {
                        "path": record.path,
                        "category": category,
                        "defect": record.defect,
                        "query": f"a photo of a {object_phrase(category)} with a {defect_phrase(record.defect)}",
                        "guided_pixel_auroc": image_auroc,
                        "generic_pixel_auroc": generic_image_auroc,
                        "mask_path": record.mask_path,
                        "index": int(i),
                    }
                )

        def _cat_auroc(labels, scores):
            if not labels:
                return None
            return safe_auroc(np.concatenate(labels), np.concatenate(scores))

        per_category[category] = {
            "pixel_auroc_generic_all": _cat_auroc(generic_labels, generic_scores),
            "pixel_auroc_generic_anomaly_images": _cat_auroc(generic_anom_labels, generic_anom_scores),
            "pixel_auroc_defect_text_anomaly_images": _cat_auroc(guided_labels, guided_scores),
            "n_images": len(index),
            "n_anomaly_with_mask": len(image_rows),
            "images": image_rows,
        }

    def _mean(key):
        return mean_ignore_none(v[key] for v in per_category.values())

    public = {}
    for cat, row in per_category.items():
        public[cat] = {k: v for k, v in row.items() if k != "images"}
        public[cat]["n_images_scored"] = len(row["images"])
    return {
        "windows": list(windows),
        "mean_pixel_auroc_generic_all": _mean("pixel_auroc_generic_all"),
        "mean_pixel_auroc_generic_anomaly_images": _mean("pixel_auroc_generic_anomaly_images"),
        "mean_pixel_auroc_defect_text_anomaly_images": _mean("pixel_auroc_defect_text_anomaly_images"),
        "per_category": public,
        "_per_category_full": per_category,
    }
