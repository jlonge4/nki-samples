"""
L0 — Per-kernel tile invariance (det=True vs det=False) on Trainium.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.run_utils import require_neuron


def main() -> int:
    if not require_neuron():
        return 0

    import torch

    from harness.neuron_device import (
        get_device,
        init_nki_runtime,
        sync_device,
        to_neuron,
    )
    from kernels.attention_batch_invariant import nki_attention_kernel_isa
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

    init_nki_runtime()

    def linspace_tensor(shape, start=-1.0, stop=1.0):
        n = int(torch.tensor(shape).prod().item())
        return torch.linspace(start, stop, n).reshape(shape)

    def run_case(kernel_fn, inputs, label, nondet=False, atol=0.0):
        device_inputs = [to_neuron(x, torch.bfloat16) for x in inputs]
        out_det = kernel_fn(*device_inputs, deterministic=True)
        sync_device()
        out_nondet = kernel_fn(*device_inputs, deterministic=nondet)
        sync_device()
        diff = (out_det.cpu().float() - out_nondet.cpu().float()).abs().max().item()
        ok = diff <= atol
        print(f"  [{'PASS' if ok else f'FAIL diff={diff:.2e}':>18s}]  {label}")
        return ok

    seq, d_head = 512, 128
    attn_inputs = [linspace_tensor((seq, d_head))] * 3
    mm_inputs = [linspace_tensor((512, 512)), linspace_tensor((512, 512))]
    rms_inputs = [linspace_tensor((128, 512)), torch.ones(512)]

    torch.manual_seed(0)
    attn_random = [torch.randn(seq, d_head) for _ in range(3)]

    print(f"=== L0: tile invariance  device={get_device()} ===\n")
    ok = True
    ok &= run_case(nki_attention_kernel_isa, attn_inputs, "attention bf16 (linspace)")
    ok &= run_case(nki_matmul_kernel_isa, mm_inputs, "matmul bf16 (linspace)")
    ok &= run_case(nki_rmsnorm_kernel_isa, rms_inputs, "rmsnorm bf16 (linspace)")
    ok &= run_case(
        nki_attention_kernel_isa,
        attn_random,
        "attention bf16 (random) [may ~1 ULP]",
        atol=2e-5,  # softmax concentration can cause 1 BF16 ULP diff at small output magnitudes
    )
    print(f"\nL0 overall: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
