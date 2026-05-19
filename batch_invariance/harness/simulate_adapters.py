"""
NumPy ops for bi_testkit via nki.simulate (no Trainium device / torch required).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from harness.sim_utils import simulate

try:
    import ml_dtypes

    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

    _HAS_NKI = True
except ImportError:
    _HAS_NKI = False
    ml_dtypes = None  # type: ignore

_WEIGHT_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}

_sim_matmul = None
_sim_rmsnorm = None


def _lazy_sim_kernels():
    global _sim_matmul, _sim_rmsnorm
    if _sim_matmul is None:
        _sim_matmul = simulate(nki_matmul_kernel_isa)
        _sim_rmsnorm = simulate(nki_rmsnorm_kernel_isa)


def _get_weight(name: str, k: int, n: int, seed: int = 0) -> np.ndarray:
    key = (name, k, n)
    if key not in _WEIGHT_CACHE:
        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((k, n), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16)
        _WEIGHT_CACHE[key] = w
    return _WEIGHT_CACHE[key]


def make_sim_rmsnorm_op(k: int) -> Callable[[np.ndarray], np.ndarray]:
    _lazy_sim_kernels()
    g = np.ones(k, dtype=ml_dtypes.bfloat16)

    def fn(x: np.ndarray) -> np.ndarray:
        return _sim_rmsnorm(x, g, deterministic=True)

    return fn


def make_sim_matmul_op(
    k: int, n: int, weight_name: str = "linear"
) -> Callable[[np.ndarray], np.ndarray]:
    _lazy_sim_kernels()
    w = _get_weight(weight_name, k, n)

    def fn(x: np.ndarray) -> np.ndarray:
        return _sim_matmul(x.T.copy(), w, deterministic=True)

    return fn
