# Paste into test_batch_invariance.ipynb (new cell after §1 MatMul).
# Requires: device from harness.neuron_device (TorchNeuron) or legacy xm.

import torch

from harness.neuron_device import get_device, init_nki_runtime, sync_device, to_neuron
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa

init_nki_runtime()
device = get_device()


def matmul_mk(x_mk, w_kn, deterministic=True):
    return nki_matmul_kernel_isa(x_mk.T.contiguous(), w_kn, deterministic=deterministic)


# --- Your pattern: (512, 512) @ (512, 2048) ---
M, K, N = 512, 512, 2048
x = to_neuron(torch.linspace(-1, 1, M * K, dtype=torch.bfloat16).reshape(M, K))
w = to_neuron(torch.linspace(-0.02, 0.02, K * N, dtype=torch.bfloat16).reshape(K, N))

y_full = matmul_mk(x, w, deterministic=True)
sync_device()

row = 1
y_row_only = matmul_mk(x[row : row + 1], w, deterministic=True)
sync_device()

ok = torch.equal(y_full[row].cpu().view(torch.int16), y_row_only[0].cpu().view(torch.int16))
print(f"row {row}: full[i] vs matmul(x[i:i+1], W)[0] -> {'PASS' if ok else 'FAIL'}")

# --- Prefix batch (128 rows alone vs first 128 of 512) ---
y_prefix_only = matmul_mk(x[:128], w, deterministic=True)
sync_device()
ok2 = torch.equal(y_prefix_only.cpu().view(torch.int16), y_full[:128].cpu().view(torch.int16))
print(f"prefix 128/512: {'PASS' if ok2 else 'FAIL'}")
