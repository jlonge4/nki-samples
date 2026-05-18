"""
L3 — Attention packing + seqlen sweep (Part B-lite on sample SDPA).

B1: same logical sequence, padded seq_k with zero filler.
B4: run-to-run determinism at nanochat seqlen boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.bitwise_numpy import bitwise_equal
from harness.nanochat_shapes import ATTN_SEQLENS, HEAD_DIM
from harness.run_utils import require_neuron

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def _linspace_qkv(seq: int, d: int):
    n = seq * d
    base = np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(seq, d)
    q = base.astype(ml_dtypes.bfloat16)
    k = (base * 0.9 + 0.01).astype(ml_dtypes.bfloat16)
    v = (base * 1.1 - 0.01).astype(ml_dtypes.bfloat16)
    return q, k, v


def test_b1_packing(attn_fn) -> bool:
    print("\n--- L3 B1: attention seq_k packing (zero-padded KV) ---")
    seq_q = 128
    d = HEAD_DIM
    q, k, v = _linspace_qkv(seq_q, d)
    out_base = attn_fn(q, k, v, deterministic=True)

    all_ok = True
    for seq_k_big in (256, 512):
        pad = seq_k_big - seq_q
        k_pad = np.concatenate([k, np.zeros((pad, d), dtype=ml_dtypes.bfloat16)], axis=0)
        v_pad = np.concatenate([v, np.zeros((pad, d), dtype=ml_dtypes.bfloat16)], axis=0)
        # Key-padding mask: 0 for real KV positions, -1e9 for zero-padded positions
        attn_bias = np.zeros((seq_q, seq_k_big), dtype=np.float32)
        attn_bias[:, seq_q:] = -1e9
        out_pad = attn_fn(q, k_pad, v_pad, deterministic=True, attn_bias=attn_bias)
        ok = bitwise_equal(out_base, out_pad)
        if not ok:
            all_ok = False
        print(f"  seq_q={seq_q} seq_k={seq_k_big}: {'PASS' if ok else 'FAIL'}")
    return all_ok


def test_b4_run_to_run(attn_fn) -> bool:
    print("\n--- L3 B4: run-to-run determinism (seqlen sweep) ---")
    all_ok = True
    for t in ATTN_SEQLENS:
        q, k, v = _linspace_qkv(t, HEAD_DIM)
        o1 = attn_fn(q, k, v, deterministic=True)
        o2 = attn_fn(q, k, v, deterministic=True)
        ok = bitwise_equal(o1, o2)
        if not ok:
            all_ok = False
        print(f"  T={t:>5}: {'PASS' if ok else 'FAIL'}")
    return all_ok


def main() -> int:
    if not require_neuron():
        return 0
    from kernels.attention_batch_invariant import nki_attention_kernel_isa

    ok = test_b1_packing(nki_attention_kernel_isa) and test_b4_run_to_run(nki_attention_kernel_isa)
    print(f"\nL3 overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
