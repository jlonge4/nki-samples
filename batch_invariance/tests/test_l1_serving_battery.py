"""
L1 — Part A serving contract via bi_testkit on batch-invariant kernels (nanochat shapes).

Run from batch_invariance/:
    python tests/test_l1_serving_battery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.nanochat_shapes import (  # noqa: E402
    LINEAR_SHAPES,
    MATMUL_WB_PAIRS,
    N_EMBD,
    NANOCHAT_WB_PAIRS,
    NEIGHBOR_CONFIGS,
    POSITION_M_VALUES,
)
from harness.run_utils import count_failures, require_neuron


def main() -> int:
    if not require_neuron():
        return 0

    from harness.bi_testkit import numpy_bf16_op, print_report, run_battery
    from harness.serving_adapters import make_matmul_op, make_rmsnorm_op

    kwargs = dict(
        position_M_values=POSITION_M_VALUES,
        neighbor_configs=NEIGHBOR_CONFIGS,
        seeds=(0, 1),
        include_adversarial=False,
    )
    exit_code = 0

    print("\n#### RMSNorm @ nanochat H=1280 (deterministic=True) ####")
    op_rms = numpy_bf16_op(make_rmsnorm_op(N_EMBD))
    res_rms = run_battery(
        op_rms,
        N_EMBD,
        whole_block_pairs=NANOCHAT_WB_PAIRS,
        **kwargs,
    )
    print_report(res_rms)
    if count_failures(res_rms):
        exit_code = 1

    for name, (k, n) in LINEAR_SHAPES.items():
        if name == "lm_head":
            print(f"\n#### MatMul {name} K={k} N={n} — SKIPPED (large N; enable manually) ####")
            continue
        print(f"\n#### MatMul {name} K={k} N={n} (deterministic=True) ####")
        op_mm = numpy_bf16_op(make_matmul_op(k, n, weight_name=name))
        res_mm = run_battery(
            op_mm,
            k,
            whole_block_pairs=MATMUL_WB_PAIRS,
            **kwargs,
        )
        print_report(res_mm)
        if count_failures(res_mm):
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
