"""Disk cache for CLIP features.

Image embeddings are small. Patch tokens are larger, so they are stored
per category and left out of git (see .gitignore). A cache is reused only
when the model, the image list, and CACHE_VERSION all match.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from defectclip.data import Record
from defectclip.encoder import encode_images, load_clip, mask_preprocess

CACHE_VERSION = 1


def model_dir(cache_root: Path | str, model_name: str, pretrained: str) -> Path:
    safe_model = model_name.replace("/", "-")
    safe_pre = pretrained.replace("/", "-")
    return Path(cache_root) / f"{safe_model}__{safe_pre}"


def _meta_matches(path: Path, n: int, split: str) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as blob:
            if int(blob["version"]) != CACHE_VERSION:
                return False
            if str(blob["split"]) != split:
                return False
            return int(blob["features"].shape[0]) == n
    except Exception:
        return False


def _save_table(path: Path, records: list[Record], features: np.ndarray, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        version=np.int32(CACHE_VERSION),
        split=np.array(split),
        path=np.array([r.path for r in records]),
        category=np.array([r.category for r in records]),
        defect=np.array([r.defect for r in records]),
        is_anomaly=np.array([r.is_anomaly for r in records], dtype=np.bool_),
        mask_path=np.array([r.mask_path or "" for r in records]),
        features=features.astype(np.float16),
    )


def _load_table(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        features = blob["features"].astype(np.float32)
        features = features / np.clip(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8, None)
        return {
            "path": blob["path"].astype(str),
            "category": blob["category"].astype(str),
            "defect": blob["defect"].astype(str),
            "is_anomaly": blob["is_anomaly"].astype(bool),
            "mask_path": blob["mask_path"].astype(str),
            "features": features,
        }


def _save_patches(path: Path, records: list[Record], tokens: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens.astype(np.float16))
    np.save(path.with_suffix(".paths.npy"), np.array([r.path for r in records]))


def _load_patches(path: Path, records: list[Record]) -> np.ndarray:
    tokens = np.load(path).astype(np.float32)
    stored_paths = np.load(path.with_suffix(".paths.npy")).astype(str)
    wanted = np.array([r.path for r in records])
    if stored_paths.shape != wanted.shape or not np.array_equal(stored_paths, wanted):
        raise RuntimeError(f"Patch cache {path} does not match the current image list")
    tokens = tokens / np.clip(np.linalg.norm(tokens, axis=-1, keepdims=True), 1e-8, None)
    return tokens


def ensure_features(
    records: list[Record],
    split: str,
    model_name: str,
    pretrained: str,
    device: torch.device,
    batch_size: int,
    cache_root: Path | str,
    with_patches: bool,
) -> dict:
    """Encode `records` or load them from the cache. Order follows `records`."""
    if not records:
        raise RuntimeError(f"No records to encode for split={split}")
    root = model_dir(cache_root, model_name, pretrained)
    image_file = root / f"{split}_images.npz"
    patch_file = root / f"{split}_patches.npy"
    if _meta_matches(image_file, len(records), split) and (not with_patches or patch_file.is_file()):
        table = _load_table(image_file)
        if not np.array_equal(table["path"], np.array([r.path for r in records])):
            # Same count, different files: re-encode rather than return a silent mismatch.
            pass
        else:
            if with_patches:
                table["patches"] = _load_patches(patch_file, records)
            table["model"] = model_name
            table["pretrained"] = pretrained
            return table

    model, preprocess, _tokenizer = load_clip(model_name, pretrained, device)
    features, patches = encode_images(
        model,
        preprocess,
        [r.path for r in records],
        device,
        batch_size,
    )
    _save_table(image_file, records, features.numpy(), split)
    if with_patches:
        _save_patches(patch_file, records, patches.numpy())
    # Drop the model before the caller maybe loads another one.
    del model
    table = _load_table(image_file)
    if with_patches:
        table["patches"] = _load_patches(patch_file, records)
    table["model"] = model_name
    table["pretrained"] = pretrained
    return table


def load_model_bundle(model_name: str, pretrained: str, device: torch.device):
    model, preprocess, tokenizer = load_clip(model_name, pretrained, device)
    masks = mask_preprocess(preprocess)
    return model, preprocess, tokenizer, masks
