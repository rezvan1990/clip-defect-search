"""NumPy window pooling for patch grids. No torch import, so tests can use it."""

from __future__ import annotations

import numpy as np


def pool_windows(tokens_hw: np.ndarray, k: int) -> np.ndarray:
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
