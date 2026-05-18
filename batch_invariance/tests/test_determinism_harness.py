"""
Determinism harness — run-to-run + cross-invocation stability (deterministic=True).

Combines per-kernel run-to-run with L1-style spot checks.
"""

from __future__ import annotations

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.run_utils  # noqa: F401
from harness.run_utils import require_neuron
from harness.bitwise_numpy import bitwise_equal
from harness.nanochat_shapes import N_EMBD
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def _rand(m, k, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((m, k), dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16)


def test_run_to_run(fn, label: str, n: int = 5) -> bool:
    x = _rand(128, N_EMBD, seed=0)
    ref = fn(x)
    for i in range(1, n):
        y = fn(x)
        if not bitwise_equal(ref, y):
            print(f"  {label} run {i}: FAIL")
            return False
    print(f"  {label}: PASS ({n} runs)")
    return True


def test_cross_m_embedding(fn, k: int) -> bool:
    """Same logical rows at M=128 alone vs top of M=256."""
    m_s, m_b = 128, 256
    x_s = _rand(m_s, k, seed=1)
    x_b = np.concatenate([x_s, _rand(m_b - m_s, k, seed=2)], axis=0)
    y_s = fn(x_s)
    y_b = fn(x_b)[:m_s]
    ok = bitwise_equal(y_s, y_b)
    print(f"  cross-M ({m_s},{m_b}) k={k}: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    if not require_neuron():
        return 0
    from harness.serving_adapters import make_matmul_op, make_rmsnorm_op

    rms = make_rmsnorm_op(N_EMBD)
    mm = make_matmul_op(N_EMBD, N_EMBD, "c_q")
    print("\n--- Determinism harness ---")
    ok = test_run_to_run(rms, "RMSNorm")
    ok = test_run_to_run(mm, "MatMul c_q") and ok
    ok = test_cross_m_embedding(rms, N_EMBD) and ok
    ok = test_cross_m_embedding(mm, N_EMBD) and ok
    print(f"\nDeterminism harness: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
