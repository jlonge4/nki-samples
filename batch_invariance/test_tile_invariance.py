"""
Tile invariance test for batch-invariant NKI kernels.

Verifies that bfloat16 outputs are bit-exact regardless of tile size.

Device: TorchNeuron (``torch.device("neuron")``) with ``import torch_neuronx`` for NKI;
falls back to torch_xla on older stacks.
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
from kernels.attention_batch_invariant import nki_attention_kernel_isa
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

init_nki_runtime()


def linspace_tensor(shape, start=-1.0, stop=1.0):
    n = int(torch.tensor(shape).prod().item())
    return torch.linspace(start, stop, n).reshape(shape)


def test_tile_invariance(kernel_fn, inputs, dtype, deterministic, label):
    device_inputs = [to_neuron(x, dtype) for x in inputs]

    out_det = kernel_fn(*device_inputs, deterministic=True)
    sync_device()
    out_nondet = kernel_fn(*device_inputs, deterministic=deterministic)
    sync_device()

    diff = (out_det.cpu().float() - out_nondet.cpu().float()).abs().max().item()
    return {"label": label, "dtype": str(dtype), "diff": diff, "invariant": diff == 0.0}


if __name__ == "__main__":
    seq, d_head = 512, 128

    attn_inputs = [
        linspace_tensor((seq, d_head)),
        linspace_tensor((seq, d_head)),
        linspace_tensor((seq, d_head)),
    ]
    mm_inputs = [linspace_tensor((512, 512)), linspace_tensor((512, 512))]
    rms_inputs = [linspace_tensor((128, 512)), torch.ones(512)]

    torch.manual_seed(0)
    attn_random = [
        torch.randn(seq, d_head),
        torch.randn(seq, d_head),
        torch.randn(seq, d_head),
    ]

    cases = [
        (nki_attention_kernel_isa, attn_inputs, torch.bfloat16, False, "attention  bf16 det/nondet (linspace)"),
        (nki_matmul_kernel_isa, mm_inputs, torch.bfloat16, False, "matmul     bf16 det/nondet (linspace)"),
        (nki_rmsnorm_kernel_isa, rms_inputs, torch.bfloat16, False, "rmsnorm    bf16 det/nondet (linspace)"),
        (nki_attention_kernel_isa, attn_random, torch.bfloat16, False, "attention  bf16 det/nondet (random)   [~1 ULP expected]"),
    ]

    print(f"Tile invariance tests  device={get_device()}\n")
    for kernel_fn, inputs, dtype, det, label in cases:
        r = test_tile_invariance(kernel_fn, inputs, dtype, det, label)
        status = "PASS" if r["invariant"] else f"diff={r['diff']:.2e}"
        print(f"  [{status:>12s}]  {r['label']}")
