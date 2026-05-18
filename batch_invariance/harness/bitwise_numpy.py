"""Bitwise helpers for numpy bf16 arrays."""

from __future__ import annotations

import numpy as np


def bitwise_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return a.tobytes() == b.tobytes()


def max_abs_bf16(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())
