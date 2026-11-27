"""Shared helpers: seeding, devices, JSON."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (str, int, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2) + "\n")


def relativize_paths(obj, root: str):
    """Replace absolute paths under `root` so committed JSON stays portable."""
    prefix = str(root).rstrip("/") + "/"
    if isinstance(obj, dict):
        return {k: relativize_paths(v, root) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relativize_paths(v, root) for v in obj]
    if isinstance(obj, str) and obj.startswith(prefix):
        return obj[len(prefix):]
    return obj


def mean_ignore_none(values) -> float | None:
    nums = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not nums:
        return None
    return float(sum(nums) / len(nums))
