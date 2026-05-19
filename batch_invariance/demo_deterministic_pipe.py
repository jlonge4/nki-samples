#!/usr/bin/env python3
"""
Toy DeepSeek-style deterministic inference pipe (no nanochat clone).

Story (bf16, deterministic=True, fixed NKI tiles):
  1. Run-to-run — same request → bitwise-identical logits.
  2. Continuous batching — prefix tokens unchanged when co-packed with filler.
  3. (optional) nondeterministic tiles — same layout drifts (shows why det=True).

Architecture (demo dims, fits matmul N_TILE on Trn2):
  embed (fixed table) → 2× pre-norm block → final RMSNorm → lm_head → logits

Run on Trainium:
    source <neuron-venv>/bin/activate
    cd batch_invariance
    NEURON_RT_VISIBLE_CORES=0 python demo_deterministic_pipe.py

Laptop (kernels only):  ./run.sh sim
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Demo “mini decoder” — SBUF N limit keeps lm_head small
D_MODEL = 256
D_HEAD = 128
D_FFN = 512
N_LAYERS = 2
SEQ = 256
SEQ_PACKED = 512
VOCAB = 256
PREFIX = 128


def _bitwise_bf16(a, b) -> bool:
    import torch

    return bool(
        torch.equal(
            a.detach().cpu().contiguous().view(torch.int16),
            b.detach().cpu().contiguous().view(torch.int16),
        )
    )


def _make_toy_weights(dtype=None):
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    scale = 0.02
    blocks = []
    for _ in range(N_LAYERS):
        blocks.append(
            {
                "wq": torch.randn(D_MODEL, D_HEAD, dtype=dtype) * scale,
                "wk": torch.randn(D_MODEL, D_HEAD, dtype=dtype) * scale,
                "wv": torch.randn(D_MODEL, D_HEAD, dtype=dtype) * scale,
                "wo": torch.randn(D_HEAD, D_MODEL, dtype=dtype) * scale,
                "w1": torch.randn(D_MODEL, D_FFN, dtype=dtype) * scale,
                "w2": torch.randn(D_FFN, D_MODEL, dtype=dtype) * scale,
                "g_attn": torch.ones(D_MODEL, dtype=dtype),
                "g_ffn": torch.ones(D_MODEL, dtype=dtype),
            }
        )
    return {
        "blocks": blocks,
        "g_out": torch.ones(D_MODEL, dtype=dtype),
        "w_lm": torch.randn(D_MODEL, VOCAB, dtype=dtype) * scale,
        "embed": torch.randn(VOCAB, D_MODEL, dtype=dtype) * 0.01,
    }


def _forward_neuron(token_ids, weights, *, deterministic: bool, packed_len: int | None):
    import torch

    from harness.neuron_device import (
        get_device,
        init_nki_runtime,
        sync_device,
        to_neuron,
    )
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa
    from transformer_block import nki_transformer_block

    init_nki_runtime()
    device = get_device()

    def mm(a, b):
        return nki_matmul_kernel_isa(a.T.contiguous(), b, deterministic=deterministic)

    def rms(a, g):
        return nki_rmsnorm_kernel_isa(a, g, deterministic=deterministic)

    w_blocks_dev = [{kk: to_neuron(vv) for kk, vv in block.items()} for block in weights["blocks"]]
    g_out = to_neuron(weights["g_out"])
    w_lm = to_neuron(weights["w_lm"])
    embed = to_neuron(weights["embed"])

    ids = to_neuron(token_ids.long())
    x = embed[ids]  # [seq, d_model]

    if packed_len is not None and packed_len > x.shape[0]:
        filler = torch.linspace(
            -0.1, 0.1, (packed_len - x.shape[0]) * D_MODEL, device=device, dtype=x.dtype
        )
        filler = filler.reshape(packed_len - x.shape[0], D_MODEL)
        x = torch.cat([x, filler], dim=0)

    h = x
    for block_w in w_blocks_dev:
        h = nki_transformer_block(h, block_w, deterministic=deterministic)

    h = rms(h, g_out)
    logits = mm(h, w_lm)
    sync_device()
    return h, logits


def run_neuron_demo() -> int:
    import torch

    from harness.neuron_device import detect_backend, init_nki_runtime

    if detect_backend() is None:
        print(
            "SKIP: need TorchNeuron (torch_neuronx) or torch_xla — use --sim-kernels-only on laptop"
        )
        return 0

    init_nki_runtime()
    torch.manual_seed(0)
    weights = _make_toy_weights()
    # Fixed “request” — PREFIX tokens; packed run appends unrelated filler tokens
    base_ids = torch.arange(PREFIX, dtype=torch.long) % VOCAB
    filler_ids = torch.arange(PREFIX, SEQ_PACKED, dtype=torch.long) % VOCAB
    packed_ids = torch.cat([base_ids, filler_ids])

    print("=" * 70)
    print("  Toy deterministic inference pipe (Trainium)")
    print(f"  {N_LAYERS} blocks  d_model={D_MODEL}  seq={SEQ}  vocab={VOCAB}")
    print("=" * 70)

    # 1) Run-to-run
    print("\n[1] Run-to-run determinism (same token_ids, det=True)")
    _, logits_a = _forward_neuron(base_ids, weights, deterministic=True, packed_len=None)
    _, logits_b = _forward_neuron(base_ids, weights, deterministic=True, packed_len=None)
    ok_rr = _bitwise_bf16(logits_a, logits_b)
    print(f"    logits bitwise match: {'PASS' if ok_rr else 'FAIL'}")

    # 2) Continuous batching / prefix invariance
    print("\n[2] Continuous batching — prefix invariant under co-packed seq (det=True)")
    h_small, _ = _forward_neuron(base_ids, weights, deterministic=True, packed_len=None)
    h_big, _ = _forward_neuron(packed_ids, weights, deterministic=True, packed_len=None)
    ok_prefix = _bitwise_bf16(h_small[:PREFIX], h_big[:PREFIX])
    print(
        f"    hidden[0:{PREFIX}] packed {PREFIX} vs {SEQ_PACKED}: {'PASS' if ok_prefix else 'FAIL'}"
    )

    # 3) Show drift without deterministic tiles (inference still “runs”, not invariant)
    print("\n[3] Same request, det=False (tile schedule changes — expect drift)")
    _, logits_det = _forward_neuron(base_ids, weights, deterministic=True, packed_len=None)
    _, logits_nondet = _forward_neuron(base_ids, weights, deterministic=False, packed_len=None)
    ok_drift = _bitwise_bf16(logits_det, logits_nondet)
    print(f"    logits det vs nondet: {'same (unexpected)' if ok_drift else 'DIFFER (expected)'}")

    # 4) Last-token logits shape smoke
    print("\n[4] Output contract")
    print(f"    logits shape: {tuple(logits_a.shape)}  (seq, vocab)")
    print(f"    last-token logit sample[0:4]: {logits_a[-1, :4].float().cpu().tolist()}")

    all_ok = ok_rr and ok_prefix
    print("\n" + "=" * 70)
    print(f"  Pipe demo: {'PASS' if all_ok else 'FAIL'}")
    print("=" * 70)
    return 0 if all_ok else 1


def main() -> int:
    return run_neuron_demo()


if __name__ == "__main__":
    raise SystemExit(main())
