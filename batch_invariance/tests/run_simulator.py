#!/usr/bin/env python3
"""
CPU simulator smoke (no Trainium). Invoked by: ./run.sh sim

  SIM_FULL=1 ./run.sh sim   # nanochat H=1280 + full bi_testkit pairs
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.sim_utils import (
    assert_no_nan,
    bf16_equal,
    bf16_equal_sim,
    bf16_max_ulp,
    configure_simulator,
    linspace_bf16,
    max_abs_diff,
    require_simulate,
    simulate,
)

FAST_WB_PAIRS = [(1, 128), (127, 128), (255, 256), (128, 256)]
FAST_ATTN_SEQLENS = (128, 256)
FULL_ATTN_SEQLENS = (128, 256, 512, 1024)


def _sim_full() -> bool:
    return os.environ.get("SIM_FULL", "").strip() in ("1", "true", "yes")


def _hidden_dim() -> int:
    return 1280 if _sim_full() else 512


def _matmul_kn() -> tuple[int, int]:
    return (512, 2048) if _sim_full() else (512, 512)


def _wb_pairs():
    if _sim_full():
        from harness.nanochat_shapes import MATMUL_WB_PAIRS, NANOCHAT_WB_PAIRS

        return NANOCHAT_WB_PAIRS, MATMUL_WB_PAIRS
    return FAST_WB_PAIRS, FAST_WB_PAIRS


def _tile_pair(sim_fn, inputs):
    out_det = sim_fn(*inputs, deterministic=True)
    out_nondet = sim_fn(*inputs, deterministic=False)
    assert_no_nan(out_det, "det")
    assert_no_nan(out_nondet, "nondet")
    diff = max_abs_diff(out_det, out_nondet)
    return diff, diff == 0.0


def test_tile_invariance() -> int:
    import ml_dtypes
    import numpy as np

    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
    from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa

    sim_mm = simulate(nki_matmul_kernel_isa)
    sim_rms = simulate(nki_rmsnorm_kernel_isa)

    k, m, n, h = 512, 128, 512, _hidden_dim()
    cases = [
        (
            "matmul bf16",
            lambda: _tile_pair(
                sim_mm,
                [linspace_bf16((k, m)), linspace_bf16((k, n))],
            ),
        ),
        (
            "rmsnorm bf16",
            lambda: _tile_pair(
                sim_rms,
                [linspace_bf16((m, h)), np.ones(h, dtype=ml_dtypes.bfloat16)],
            ),
        ),
    ]

    fails = 0
    print("\n=== Reduction-tile invariance (simulate) ===")
    for label, run in cases:
        diff, ok = run()
        status = "PASS" if ok else f"FAIL (max_abs={diff:.3e})"
        if not ok:
            fails += 1
        print(f"  [{status:>28s}]  {label}")
    return fails


def test_attention_cte_sim() -> int:
    from harness.attention_cte_ops import import_attention_cte
    from harness.attention_cte_ops import check_batch_schedule, check_padded_kv, check_run_to_run
    from harness.nanochat_shapes import N_HEAD

    attention_cte = import_attention_cte()
    if attention_cte is None:
        print("\n=== attention_cte (nki.simulate) ===")
        print("  SKIP: install nki-library (see README)")
        return 0

    sim_cte = simulate(attention_cte)
    batch_heads = N_HEAD if _sim_full() else 2
    seqlens = FULL_ATTN_SEQLENS if _sim_full() else FAST_ATTN_SEQLENS
    pad_sizes = (256, 512) if _sim_full() else (256,)

    fails = 0
    print(f"\n=== Attention batch invariance (simulate, heads={batch_heads}) ===\nPadded KV:")
    if not check_padded_kv(sim_cte, seq_k_big=pad_sizes):
        fails += 1

    print("\nAlone vs co-packed batch:")
    if not check_batch_schedule(sim_cte, batch_heads=batch_heads, multipliers=(1, 2)):
        fails += 1

    print("\nRun-to-run:")
    if not check_run_to_run(sim_cte, seqlens, batch_heads=batch_heads):
        fails += 1

    return fails


def test_matmul_m_invariance() -> int:
    from kernels.matmul_batch_invariant import nki_matmul_kernel_isa

    sim_mm = simulate(nki_matmul_kernel_isa)
    k, n = _matmul_kn()

    def mm(x_mk):
        return sim_mm(x_mk.T.copy(), linspace_bf16((k, n), -0.02, 0.02), deterministic=True)

    fails = 0
    print("\n=== MatMul M invariance (tail slabs, no padding) ===")
    m_full = 512 if _sim_full() else 256
    x_full = linspace_bf16((m_full, k))
    row = min(254, m_full - 1)
    y_full, y_one = mm(x_full), mm(x_full[row : row + 1])
    ok = bf16_equal_sim(y_full[row], y_one[0])
    ulp = bf16_max_ulp(y_full[row], y_one[0])
    print(f"  row={row} full vs isolated: {'PASS' if ok else 'FAIL'} (ulp={ulp})")
    fails += 0 if ok else 1

    n_prefix = 128
    y_small, y_big = mm(x_full[:n_prefix]), mm(x_full)
    ok2 = bf16_equal(y_small, y_big[:n_prefix])
    print(f"  prefix {n_prefix}/{m_full}: {'PASS' if ok2 else 'FAIL'}")
    fails += 0 if ok2 else 1

    x_tail = linspace_bf16((255, k))
    w_tail = linspace_bf16((k, n), -0.02, 0.02)
    y_tail = sim_mm(x_tail.T.copy(), w_tail, deterministic=True)
    y_row = sim_mm(x_tail[254:255].T.copy(), w_tail, deterministic=True)
    ok3 = bf16_equal_sim(y_tail[254], y_row[0])
    ulp3 = bf16_max_ulp(y_tail[254], y_row[0])
    print(f"  M=255 tail row 254: {'PASS' if ok3 else 'FAIL'} (ulp={ulp3}, sim ≤1)")
    fails += 0 if ok3 else 1
    return fails


def test_serving_battery() -> int:
    from harness.bi_testkit import numpy_bf16_op, print_report, run_battery
    from harness.run_utils import count_failures
    from harness.simulate_adapters import make_sim_matmul_op, make_sim_rmsnorm_op

    h = _hidden_dim()
    k_mm, n_mm = _matmul_kn()
    rms_pairs, mm_pairs = _wb_pairs()
    kwargs = dict(
        position_M_values=(128, 255) if not _sim_full() else (128, 255, 256),
        neighbor_configs=((128, 0), (255, 128)),
        seeds=(0,) if not _sim_full() else (0, 1),
        include_adversarial=False,
    )

    fails = 0
    print(f"\n=== Row schedule invariance (simulate, H={h}) ===")
    for name, op_fn, k, pairs in (
        ("RMSNorm", make_sim_rmsnorm_op(h), h, rms_pairs),
        ("MatMul", make_sim_matmul_op(k_mm, n_mm, "sim_linear"), k_mm, mm_pairs),
    ):
        res = run_battery(numpy_bf16_op(op_fn), k, whole_block_pairs=pairs, **kwargs)
        print_report(res)
        fails += count_failures(res)
    return fails


def main() -> int:
    configure_simulator()
    if not require_simulate():
        return 0

    print(f"NKI CPU simulator  mode={'FULL' if _sim_full() else 'fast'}")
    fails = (
        test_tile_invariance()
        + test_attention_cte_sim()
        + test_matmul_m_invariance()
        + test_serving_battery()
    )
    print("\n" + "=" * 60)
    print("  simulator: PASS" if fails == 0 else f"  simulator: FAIL ({fails} groups)")
    print("=" * 60)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
