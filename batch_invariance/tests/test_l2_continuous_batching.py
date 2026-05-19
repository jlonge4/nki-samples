"""
L2 — Continuous batching / request packing (numpy + Neuron kernels).

Position, neighbor isolation, whole-block with multiple fillers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import harness.run_utils  # noqa: F401
from harness.bitwise_numpy import bitwise_equal, max_abs_bf16
from harness.nanochat_shapes import N_EMBD
from harness.run_utils import require_neuron

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def _make_sparse(shape, seed, scale=5.0, frac=0.1):
    rng = np.random.default_rng(seed)
    vals = rng.standard_normal(shape, dtype=np.float32) * scale
    mask = (rng.random(shape, dtype=np.float32) < frac).astype(np.float32)
    return (vals * mask).astype(ml_dtypes.bfloat16)


def _make_randn(shape, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(ml_dtypes.bfloat16)


def test_position(rms_fn, k: int) -> bool:
    print("\n--- L2: RMSNorm position invariance ---")
    m_values = [128, 255, 256]
    positions_map = {128: [0, 1, 63, 64, 127], 255: [0, 128, 254], 256: [0, 128, 255]}
    all_ok = True
    for m in m_values:
        positions = [p for p in positions_map.get(m, [0, m // 2, m - 1]) if p < m]
        x_base = _make_randn((m, k), seed=42)
        refs = []
        for p in positions:
            x = x_base.copy()
            if p != 0:
                tmp = x[0].copy()
                x[0] = x[p]
                x[p] = tmp
            y = rms_fn(x)
            refs.append(y[p])
        ref = refs[0]
        for p, out in zip(positions, refs):
            ok = bitwise_equal(ref, out)
            if not ok:
                all_ok = False
            tag = "PASS" if ok else f"FAIL max_abs={max_abs_bf16(ref, out):.2e}"
            print(f"  M={m:>4} p={p:>4}: {tag}")
    return all_ok


def test_neighbor(rms_fn, k: int) -> bool:
    print("\n--- L2: RMSNorm neighbor independence ---")
    configs = [(128, 0), (255, 128), (256, 128)]
    fillers = [
        ("zeros", lambda s, seed: np.zeros(s, dtype=ml_dtypes.bfloat16)),
        ("randn", lambda s, seed: _make_randn(s, seed)),
        ("sparse", lambda s, seed: _make_sparse(s, seed)),
    ]
    rng = np.random.default_rng(99)
    probe = (rng.standard_normal(k, dtype=np.float32) * 0.5).astype(ml_dtypes.bfloat16)
    all_ok = True
    for m, p in configs:
        outs = []
        for fname, fgen in fillers:
            for seed in (0, 1):
                x = fgen((m, k), seed)
                x[p] = probe
                outs.append(rms_fn(x)[p])
        ref = outs[0]
        ok = all(o.shape == ref.shape and bitwise_equal(ref, o) for o in outs[1:])
        if not ok:
            all_ok = False
        print(f"  M={m} p={p}: {'PASS' if ok else 'FAIL'} ({len(outs)} mutations)")
    return all_ok


def test_whole_block_filler(rms_fn, k: int) -> bool:
    print("\n--- L2: RMSNorm whole-block + filler tails ---")
    pairs = [(1, 128), (255, 256), (128, 2048)]
    all_ok = True
    for m_s, m_b in pairs:
        x_s = _make_randn((m_s, k), seed=7)
        y_s = rms_fn(x_s)
        for fname, fgen in [
            ("zeros", lambda s, sd: np.zeros(s, dtype=ml_dtypes.bfloat16)),
            ("sparse", _make_sparse),
        ]:
            tail = fgen((m_b - m_s, k), seed=1000)
            x_b = np.concatenate([x_s, tail], axis=0)
            y_b = rms_fn(x_b)[:m_s]
            ok = bitwise_equal(y_s, y_b)
            if not ok:
                all_ok = False
            print(f"  ({m_s},{m_b}) fill={fname}: {'PASS' if ok else 'FAIL'}")
    return all_ok


def main() -> int:
    if not require_neuron():
        return 0
    from harness.serving_adapters import make_rmsnorm_op

    rms_fn = make_rmsnorm_op(N_EMBD)
    ok = test_position(rms_fn, N_EMBD)
    ok = test_neighbor(rms_fn, N_EMBD) and ok
    ok = test_whole_block_filler(rms_fn, N_EMBD) and ok
    print(f"\nL2 overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
