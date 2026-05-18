"""
Transformer block tile-invariance test (TorchNeuron / neuron device).

Run from batch_invariance/:
    source <neuron-venv>/bin/activate   # TorchNeuron or torch-neuronx
    NEURON_RT_VISIBLE_CORES=0 python3 test_block_invariance.py
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from harness.neuron_device import get_device, init_nki_runtime, sync_device, to_neuron
from transformer_block import make_block_weights, nki_transformer_block

init_nki_runtime()


def check(label, t):
    c = t.cpu().float()
    nan_n = c.isnan().sum().item()
    inf_n = c.isinf().sum().item()
    maxabs = c[~c.isnan() & ~c.isinf()].abs().max().item() if nan_n + inf_n < c.numel() else float("nan")
    print(f"  {label:35s}  shape={tuple(t.shape)}  dtype={t.dtype}  max={maxabs:.3e}  nan={nan_n}  inf={inf_n}")


def run_invariance_test(seq, d_model, d_head, d_ffn, dtype):
    print(f"\n{'─' * 65}")
    print(f"  seq={seq}  d_model={d_model}  d_head={d_head}  d_ffn={d_ffn}  dtype={dtype}")
    print(f"  device={get_device()}")
    print(f"{'─' * 65}")

    weights = {k: to_neuron(v, dtype) for k, v in make_block_weights(d_model, d_head, d_ffn, dtype=dtype).items()}
    x = to_neuron(torch.linspace(-1, 1, seq * d_model).reshape(seq, d_model), dtype)

    out_det = nki_transformer_block(x, weights, deterministic=True)
    sync_device()
    check("out (det=True)", out_det)

    out_nondet = nki_transformer_block(x, weights, deterministic=False)
    sync_device()
    check("out (det=False)", out_nondet)

    out_det_f = out_det.cpu().float()
    out_nondet_f = out_nondet.cpu().float()
    diff = (out_det_f - out_nondet_f).abs().max().item()
    has_nan = out_det_f.isnan().any().item() or out_nondet_f.isnan().any().item()
    has_inf = out_det_f.isinf().any().item() or out_nondet_f.isinf().any().item()

    if has_nan:
        status = "FAIL (NaN in output)"
    elif has_inf:
        status = "FAIL (Inf in output)"
    elif diff == 0.0:
        status = "PASS — diff=0 (invariant)"
    else:
        status = f"diff={diff:.2e} (not invariant, expected for float32)"

    print(f"\n  det/nondet max diff: {diff}")
    print(f"  Result: [{status}]")
    return diff == 0.0 and not has_nan and not has_inf


if __name__ == "__main__":
    print("NKI Transformer Block — Tile Invariance Test")
    print("det=True  (KV_TILE=128, K_TILE=128)")
    print("det=False (KV_TILE=64,  K_TILE=64 )\n")

    results = {}
    for dtype in (torch.bfloat16, torch.float32):
        results[dtype] = run_invariance_test(seq=512, d_model=256, d_head=128, d_ffn=512, dtype=dtype)

    print(f"\n{'=' * 65}")
    bf16_ok = results[torch.bfloat16] is True
    f32_ok = results[torch.float32] is False
    overall = bf16_ok and f32_ok
    print(f"  bfloat16 diff=0 (invariant):      {'PASS' if bf16_ok else 'FAIL'}")
    print(f"  float32  diff>0 (not invariant):  {'PASS' if f32_ok else 'FAIL'}")
    print(f"  Overall: {'PASS' if overall else 'FAIL'}")
    print(f"{'=' * 65}")
