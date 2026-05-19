#!/usr/bin/env python3
"""Debug kernels on CPU via nki.simulate. Usage: ./run.sh debug {psum|m-tail}"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cmd_psum() -> int:
    import ml_dtypes
    import nki
    import nki.isa as nisa
    import nki.language as nl
    import numpy as np

    from harness.sim_utils import configure_simulator, require_simulate

    configure_simulator()
    if not require_simulate():
        return 0

    @nki.jit
    def matmul_dump_psum(a, b, k_tile):
        K, M = a.shape
        N = b.shape[1]
        M_TILE = 128
        n_tiles = K // k_tile
        snapshots = nl.ndarray((n_tiles, M_TILE, N), dtype=nl.float32, buffer=nl.shared_hbm)
        c_psum = nl.ndarray((M_TILE, N), dtype=nl.float32, buffer=nl.psum)
        for k in nl.static_range(n_tiles):
            a_tile = nl.ndarray((k_tile, M_TILE), dtype=a.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=a_tile, src=a[k * k_tile : (k + 1) * k_tile, 0:M_TILE])
            b_tile = nl.ndarray((k_tile, N), dtype=b.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=b_tile, src=b[k * k_tile : (k + 1) * k_tile, 0:N])
            nisa.nc_matmul(dst=c_psum, stationary=a_tile, moving=b_tile)
            snap = nl.ndarray((M_TILE, N), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=snap, src=c_psum)
            nisa.dma_copy(dst=snapshots[k, 0:M_TILE, 0:N], src=snap)
        return snapshots

    def inspect(dtype, label):
        K, M, N = 512, 128, 512
        a = np.linspace(-1, 1, K * M, dtype=np.float32).reshape(K, M).astype(dtype)
        b = np.linspace(-1, 1, K * N, dtype=np.float32).reshape(K, N).astype(dtype)
        snaps_128 = nki.simulate(matmul_dump_psum)(a, b, 128)
        snaps_64 = nki.simulate(matmul_dump_psum)(a, b, 64)
        diff = np.max(np.abs(snaps_128[0].astype(np.float32) - snaps_64[1].astype(np.float32)))
        final_diff = np.max(
            np.abs(snaps_128[-1].astype(np.float32) - snaps_64[-1].astype(np.float32))
        )
        print(f"\n{label}")
        print(f"  PSUM after first 128 K: diff={diff:.6e}")
        print(f"  PSUM after all 512 K:   diff={final_diff:.6e}")

    print("PSUM inspect (K=512, M=N=128)")
    inspect(ml_dtypes.bfloat16, "bfloat16:")
    inspect(np.float32, "float32:")
    return 0


def cmd_m_tail() -> int:
    import numpy as np

    from harness.sim_utils import (
        configure_simulator,
        linspace_bf16,
        require_simulate,
        simulate,
    )
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa

    configure_simulator()
    if not require_simulate():
        return 0

    sim_mm = simulate(nki_matmul_kernel_isa)
    k, n = 512, 512
    for m in (255, 256):
        y = sim_mm(linspace_bf16((m, k)).T.copy(), linspace_bf16((k, n), -0.02, 0.02), True)
        print(f"\nM={m}  last row cols 0:4: {y[m - 1, :4].astype(np.float32)}")
    x = linspace_bf16((255, k))
    w = linspace_bf16((k, n), -0.02, 0.02)
    y_full = sim_mm(x.T.copy(), w, True)
    y_one = sim_mm(x[254:255].T.copy(), w, True)
    diff = np.max(np.abs(y_full[254].astype(np.float32) - y_one[0].astype(np.float32)))
    print(f"\nM=255 row 254 max_abs_diff: {diff:.6e}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Inspect kernels via nki.simulate")
    p.add_argument("command", choices=("psum", "m-tail"))
    args = p.parse_args()
    return cmd_psum() if args.command == "psum" else cmd_m_tail()


if __name__ == "__main__":
    raise SystemExit(main())
