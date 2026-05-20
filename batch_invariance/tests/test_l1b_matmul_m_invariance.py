"""Row schedule invariance (Part A spot): Y[i] from full M vs isolated row; prefix rows."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from harness.neuron_device import get_device, init_nki_runtime, sync_device, to_neuron
from harness.run_utils import require_neuron
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa


def matmul_rows(a_mk: torch.Tensor, w_kn: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    """Standard (M, K) @ (K, N) via NKI kernel (M tail slabs, no batch padding)."""
    return nki_matmul_kernel_isa(a_mk.T.contiguous(), w_kn, deterministic=deterministic)


def assert_row_matches_isolated(
    x_mk: torch.Tensor,
    w_kn: torch.Tensor,
    row: int,
    *,
    deterministic: bool = True,
) -> bool:
    """Y[i] from full batch vs matmul of only row i."""
    y_full = matmul_rows(x_mk, w_kn, deterministic=deterministic)
    sync_device()
    y_one = matmul_rows(x_mk[row : row + 1], w_kn, deterministic=deterministic)
    sync_device()
    a = y_full[row].cpu().view(torch.int16)
    b = y_one[0].cpu().view(torch.int16)
    return bool(torch.equal(a, b))


def assert_prefix_matches_small_run(
    x_big: torch.Tensor,
    w_kn: torch.Tensor,
    n_prefix: int,
    *,
    deterministic: bool = True,
) -> bool:
    """op(X[:n]) vs op(X_big)[:n] — same as notebook §5b but explicit."""
    x_small = x_big[:n_prefix].contiguous()
    y_small = matmul_rows(x_small, w_kn, deterministic=deterministic)
    sync_device()
    y_big = matmul_rows(x_big, w_kn, deterministic=deterministic)
    sync_device()
    a = y_small.cpu().view(torch.int16)
    b = y_big[:n_prefix].cpu().view(torch.int16)
    return bool(torch.equal(a, b))


def main() -> int:
    if not require_neuron():
        return 0

    init_nki_runtime()
    dtype = torch.bfloat16
    device = get_device()
    print(f"MatMul M-invariance  device={device}\n")

    # User-shaped example: (512, 512) @ (512, 2048) -> (512, 2048)
    m, k, n = 512, 512, 2048
    x_cpu = torch.linspace(-1, 1, m * k, dtype=dtype).reshape(m, k)
    w_cpu = torch.linspace(-0.02, 0.02, k * n, dtype=dtype).reshape(k, n)
    x = to_neuron(x_cpu)
    w = to_neuron(w_cpu)

    all_ok = True

    print("--- Row i: full matmul then slice vs matmul of row i only ---")
    for row in (0, 1, 63, 127, 255, 511):
        ok = assert_row_matches_isolated(x, w, row, deterministic=True)
        if not ok:
            all_ok = False
        print(f"  row={row:>4}: {'PASS' if ok else 'FAIL'}")

    print("\n--- Prefix: X[:128]@W vs (X[512]@W)[:128] (co-packed batch) ---")
    ok = assert_prefix_matches_small_run(x, w, n_prefix=128, deterministic=True)
    if not ok:
        all_ok = False
    print(f"  prefix 128/512: {'PASS' if ok else 'FAIL'}")

    print("\n--- M tail slab (255 rows, K=512, N=2048) ---")
    m_tail, k_tail, n_tail = 255, 512, 2048
    x_tail_cpu = torch.linspace(-1, 1, m_tail * k_tail, dtype=dtype).reshape(m_tail, k_tail)
    w_tail_cpu = torch.linspace(-0.02, 0.02, k_tail * n_tail, dtype=dtype).reshape(k_tail, n_tail)
    x_tail = to_neuron(x_tail_cpu)
    w_tail = to_neuron(w_tail_cpu)
    ok_tail = assert_row_matches_isolated(x_tail, w_tail, row=254, deterministic=True)
    if not ok_tail:
        all_ok = False
    print(f"  row=254 (M=255 tail): {'PASS' if ok_tail else 'FAIL'}")

    print(f"\nOverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
