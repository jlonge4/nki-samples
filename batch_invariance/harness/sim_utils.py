"""
NKI CPU simulator helpers (NKI 0.3+).

See: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/nki_simulator.html

Recommended env (set by configure_simulator):
  NKI_PRECISE_FP=1              — real bf16 storage (bitwise tests)
  NEURON_PLATFORM_TARGET_OVERRIDE=trn2  — NCv3 matmul free-dim limits
"""

from __future__ import annotations

import os
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def configure_simulator(*, platform: str = "trn2", precise_fp: bool = True) -> None:
    """Apply simulator defaults before importing kernels (safe to call repeatedly)."""
    if precise_fp:
        os.environ.setdefault("NKI_PRECISE_FP", "1")
    os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", platform)


def has_simulate() -> bool:
    try:
        import nki  # noqa: F401

        return hasattr(nki, "simulate")
    except ImportError:
        return False


def require_simulate() -> bool:
    configure_simulator()
    if not has_simulate():
        print(
            "SKIP: nki.simulate not available.\n"
            "  Install AWS Neuron SDK 2.29+ (NKI 0.3) and run in your Neuron venv.\n"
            "  See README § CPU Simulator."
        )
        return False
    return True


def simulate(kernel: F) -> F:
    """Return nki.simulate(kernel) after configure_simulator."""
    import nki

    configure_simulator()
    return nki.simulate(kernel)  # type: ignore[return-value]


def bf16_equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Bitwise bf16 equality (same contract as bi_testkit)."""
    import ml_dtypes

    if a.dtype != ml_dtypes.bfloat16:
        a = a.astype(ml_dtypes.bfloat16)
    if b.dtype != ml_dtypes.bfloat16:
        b = b.astype(ml_dtypes.bfloat16)
    return bool(np.array_equal(a.view(np.uint16), b.view(np.uint16)))


def bf16_max_ulp(a: np.ndarray, b: np.ndarray) -> int:
    import ml_dtypes

    au = a.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int32)
    bu = b.astype(ml_dtypes.bfloat16).view(np.uint16).astype(np.int32)
    return int(np.max(np.abs(au - bu))) if au.size else 0


def bf16_equal_sim(a: np.ndarray, b: np.ndarray, *, max_ulp: int = 1) -> bool:
    """Bitwise bf16, or within max_ulp for known CPU-simulator matmul tail gaps."""
    if bf16_equal(a, b):
        return True
    return bf16_max_ulp(a, b) <= max_ulp


def linspace_bf16(shape: tuple[int, ...], start: float = -1.0, stop: float = 1.0):
    import ml_dtypes

    n = int(np.prod(shape))
    x = np.linspace(start, stop, n, dtype=np.float32).reshape(shape)
    return x.astype(ml_dtypes.bfloat16)


def assert_no_nan(x: np.ndarray, label: str) -> None:
    if np.issubdtype(x.dtype, np.floating) and np.isnan(x).any():
        raise AssertionError(f"{label}: NaN in output (uninitialized SBUF/PSUM?)")


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a.astype(np.float32) - b.astype(np.float32))))
