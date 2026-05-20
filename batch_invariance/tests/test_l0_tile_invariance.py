"""Reduction-tile invariance: MatMul/RMSNorm det K/H tile 128 vs 64, same inputs."""

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

    from harness.neuron_device import get_device, init_nki_runtime, sync_device, to_neuron
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

    init_nki_runtime()

    def linspace_tensor(shape, start=-1.0, stop=1.0):
        n = int(torch.tensor(shape).prod().item())
        return torch.linspace(start, stop, n).reshape(shape)

    def run_case(kernel_fn, inputs, label):
        device_inputs = [to_neuron(x, torch.bfloat16) for x in inputs]
        out_det = kernel_fn(*device_inputs, deterministic=True)
        sync_device()
        out_nondet = kernel_fn(*device_inputs, deterministic=False)
        sync_device()
        diff = (out_det.cpu().float() - out_nondet.cpu().float()).abs().max().item()
        ok = diff == 0.0
        print(f"  [{'PASS' if ok else f'FAIL diff={diff:.2e}':>18s}]  {label}")
        return ok

    mm_inputs = [linspace_tensor((512, 512)), linspace_tensor((512, 512))]
    rms_inputs = [linspace_tensor((128, 512)), torch.ones(512)]

    print(f"=== Reduction-tile invariance  device={get_device()} ===\n")
    ok = run_case(nki_matmul_kernel_isa, mm_inputs, "matmul")
    ok = run_case(nki_rmsnorm_kernel_isa, rms_inputs, "rmsnorm") and ok
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
