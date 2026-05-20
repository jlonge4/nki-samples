"""
nki-library ``attention_cte`` (flash prefill) wrappers for this sample.

Layouts (defaults tp_q=True, tp_k=False, tp_out=False):
  q: (batch, seqlen_q, d)
  k: (batch_kv, d, seqlen_kv)
  v: (batch_kv, seqlen_kv, d)
  out: (batch, seqlen_q, d)

Install: https://github.com/aws-neuron/nki-library
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from harness.bi_testkit import bitwise_equal_ndarray
from harness.nanochat_shapes import HEAD_DIM, N_HEAD

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None  # type: ignore

AttentionFn = Callable[..., np.ndarray]

ATTN_SCALE = HEAD_DIM**-0.5


def _ensure_nkilib_path() -> None:
    """Wheel layout is site-packages/nkilib_src/nkilib/ (top_level.txt = nkilib_src only)."""
    import sys
    from pathlib import Path

    try:
        import nkilib  # noqa: F401

        return
    except ImportError:
        pass
    import site

    for sp in site.getsitepackages():
        root = Path(sp) / "nkilib_src"
        if (root / "nkilib").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
            return


def import_attention_cte() -> Optional[Callable[..., Any]]:
    try:
        _ensure_nkilib_path()
        from nkilib.core.attention.attention_cte import attention_cte

        return attention_cte
    except ImportError:
        return None


def require_attention_cte() -> Optional[Callable[..., Any]]:
    fn = import_attention_cte()
    if fn is None:
        print(
            "SKIP: nki-library required for attention_cte. "
            "pip install nki-library or add PYTHONPATH=.../nki-library/src/nkilib_src"
        )
    return fn


def pack_qkv(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    batch: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat (seq, d) or batched (batch, seq, d) -> attention_cte layouts."""
    if q.ndim == 2:
        q_b = q[np.newaxis, ...]
        k_b = k[np.newaxis, ...]
        v_b = v[np.newaxis, ...]
    else:
        q_b, k_b, v_b = q, k, v
    if batch > q_b.shape[0]:
        raise ValueError(f"batch={batch} exceeds q batch {q_b.shape[0]}")
    q_b = q_b[:batch]
    k_b = k_b[:batch]
    v_b = v_b[:batch]
    k_cte = np.transpose(k_b, (0, 2, 1)).copy()
    return q_b, k_cte, v_b


def attention_cte_run(
    attention_cte: Callable[..., Any],
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    batch: int = 1,
    causal_mask: bool = True,
    bound_min: Optional[np.ndarray] = None,
    bound_max: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Run flash attention; return (batch, seq, d)."""
    q_b, k_b, v_b = pack_qkv(q, k, v, batch=batch)
    out = attention_cte(
        q_b,
        k_b,
        v_b,
        scale=kwargs.pop("scale", ATTN_SCALE),
        causal_mask=causal_mask,
        bound_min=bound_min,
        bound_max=bound_max,
        **kwargs,
    )
    return out


def attention_cte_single(
    attention_cte: Callable[..., Any],
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    **kwargs: Any,
) -> np.ndarray:
    """One sequence: q,k,v (seq, d) -> (seq, d)."""
    out = attention_cte_run(attention_cte, q, k, v, batch=1, **kwargs)
    return out[0]


def make_qkv_nanochat(
    batch: int,
    seq: int,
    seed: int,
    *,
    d: int = HEAD_DIM,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    bf16: Any = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nanochat-style (batch=H heads) q,k,v for Part B tests."""
    if bf16 is None:
        import ml_dtypes

        bf16 = ml_dtypes.bfloat16
    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((batch, seq, d), dtype=np.float32) * q_scale).astype(bf16)
    v = (rng.standard_normal((batch, seq, d), dtype=np.float32) * kv_scale).astype(bf16)
    k = (rng.standard_normal((batch, seq, d), dtype=np.float32) * kv_scale).astype(bf16)
    return q, k, v


def _linspace_qkv(seq: int, d: int = HEAD_DIM):
    n = seq * d
    base = np.linspace(-0.5, 0.5, n, dtype=np.float32).reshape(seq, d)
    q = base.astype(ml_dtypes.bfloat16)
    k = (base * 0.9 + 0.01).astype(ml_dtypes.bfloat16)
    v = (base * 1.1 - 0.01).astype(ml_dtypes.bfloat16)
    return q, k, v


def check_padded_kv(
    attn_fn: AttentionFn, *, seq_q: int = 128, seq_k_big: Sequence[int] = (256, 512)
) -> bool:
    """Part B: longer physical KV must not change output (causal)."""
    d = HEAD_DIM
    q, k, v = _linspace_qkv(seq_q, d)
    out_base = attention_cte_single(attn_fn, q, k, v, causal_mask=True)
    all_ok = True
    for sk in seq_k_big:
        pad = sk - seq_q
        k_pad = np.concatenate([k, np.zeros((pad, d), dtype=ml_dtypes.bfloat16)], axis=0)
        v_pad = np.concatenate([v, np.zeros((pad, d), dtype=ml_dtypes.bfloat16)], axis=0)
        out_pad = attention_cte_single(attn_fn, q, k_pad, v_pad, causal_mask=True)
        ok = bitwise_equal_ndarray(out_base, out_pad)
        all_ok = all_ok and ok
        print(f"  padded KV seq_k={sk}: {'PASS' if ok else 'FAIL'}")
    return all_ok


def check_batch_schedule(
    attn_fn: AttentionFn,
    *,
    batch_heads: int = N_HEAD,
    seq: int = 128,
    multipliers: Sequence[int] = (1, 2),
) -> bool:
    """Part B B1: same request alone vs co-packed with filler batch slots."""
    q_r, k_r, v_r = make_qkv_nanochat(batch_heads, seq, seed=17)
    results = []
    for mult in multipliers:
        batch_total = batch_heads * mult
        if mult == 1:
            q, k, v = q_r, k_r, v_r
        else:
            q_f, k_f, v_f = make_qkv_nanochat(batch_total - batch_heads, seq, seed=batch_total + 100)
            q = np.concatenate([q_r, q_f], axis=0)
            k = np.concatenate([k_r, k_f], axis=0)
            v = np.concatenate([v_r, v_f], axis=0)
        out = attention_cte_run(attn_fn, q, k, v, batch=batch_total, causal_mask=True)
        results.append((batch_total, out[:batch_heads].copy()))
    ref_batch, ref = results[0]
    all_ok = True
    for batch_total, out_r in results[1:]:
        ok = bitwise_equal_ndarray(ref, out_r)
        all_ok = all_ok and ok
        print(f"  batch {ref_batch} vs {batch_total}: {'PASS' if ok else 'FAIL'}")
    return all_ok


def check_run_to_run(
    attn_fn: AttentionFn,
    seqlens: Iterable[int],
    *,
    batch_heads: int = N_HEAD,
) -> bool:
    """Run-to-run: same inputs twice → bitwise same (not batch invariance, but required)."""
    all_ok = True
    for t in seqlens:
        q, k, v = make_qkv_nanochat(batch_heads, t, seed=t + 1)
        o1 = attention_cte_run(attn_fn, q, k, v, batch=batch_heads, causal_mask=True)
        o2 = attention_cte_run(attn_fn, q, k, v, batch=batch_heads, causal_mask=True)
        ok = bitwise_equal_ndarray(o1, o2)
        all_ok = all_ok and ok
        print(f"  seqlen {t}: {'PASS' if ok else 'FAIL'}")
    return all_ok
