"""
Adapters: batch-invariant NKI kernels -> bi_testkit op contract (M, K) -> (M, N).

All serving tests use deterministic=True (fixed tiles). L0 tile A vs B is separate.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

try:
    from harness.nanochat_shapes import N_EMBD
except ImportError:
    from nanochat_shapes import N_EMBD  # type: ignore

try:
    import ml_dtypes
    import nki  # noqa: F401
    try:
        from harness.neuron_device import init_nki_runtime
    except ImportError:
        from neuron_device import init_nki_runtime  # type: ignore

    init_nki_runtime()  # import torch_neuronx when available (TorchNeuron + NKI)
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

    _HAS_NEURON = True
except ImportError:
    _HAS_NEURON = False
    ml_dtypes = None  # type: ignore

_WEIGHT_CACHE: Dict[Tuple[str, int, int], np.ndarray] = {}

# RMSNorm kernel requires M % BATCH_TILE == 0.  Inputs are padded to the next
# multiple of this value and outputs are sliced back to the original M.
_RMSNORM_BATCH_TILE = 128


def has_neuron() -> bool:
    return _HAS_NEURON


def _bf16_linspace(shape: tuple, scale: float = 0.02) -> np.ndarray:
    n = int(np.prod(shape))
    x = np.linspace(-scale, scale, n, dtype=np.float32).reshape(shape)
    return x.astype(ml_dtypes.bfloat16)


def _get_weight(name: str, k: int, n: int, seed: int = 0) -> np.ndarray:
    key = (name, k, n)
    if key not in _WEIGHT_CACHE:
        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((k, n), dtype=np.float32) * 0.02).astype(ml_dtypes.bfloat16)
        _WEIGHT_CACHE[key] = w
    return _WEIGHT_CACHE[key]


def _pad_to_tile(x: np.ndarray, tile: int) -> tuple[np.ndarray, int]:
    """Pad first dimension to a multiple of tile. Returns (padded, original_M)."""
    M = x.shape[0]
    M_padded = ((M + tile - 1) // tile) * tile
    if M_padded > M:
        pad_shape = (M_padded - M,) + x.shape[1:]
        x = np.concatenate([x, np.zeros(pad_shape, dtype=x.dtype)], axis=0)
    return x, M


def make_rmsnorm_op(k: int = N_EMBD) -> Callable[[np.ndarray], np.ndarray]:
    g = np.ones(k, dtype=ml_dtypes.bfloat16)

    def fn(x: np.ndarray) -> np.ndarray:
        x_padded, M = _pad_to_tile(x, _RMSNORM_BATCH_TILE)
        result = nki_rmsnorm_kernel_isa(x_padded, g, deterministic=True)
        return result[:M]

    return fn


# Matmul kernel requires M % M_TILE == 0 (M_TILE=128).
_MATMUL_M_TILE = 128


def make_matmul_op(k: int, n: int, weight_name: str = "linear") -> Callable[[np.ndarray], np.ndarray]:
    w = _get_weight(weight_name, k, n)

    def fn(x: np.ndarray) -> np.ndarray:
        # x: (M, K), w: (K, N) -> kernel(a=[K,M], b=[K,N]) -> (M, N)
        x_padded, M = _pad_to_tile(x, _MATMUL_M_TILE)
        result = nki_matmul_kernel_isa(x_padded.T.copy(), w, deterministic=True)
        return result[:M]

    return fn
