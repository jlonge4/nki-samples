"""Attention batch invariance (Part B) via nki-library attention_cte."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.attention_cte_ops import (
    check_batch_schedule,
    check_padded_kv,
    check_run_to_run,
    require_attention_cte,
)
from harness.nanochat_shapes import ATTN_SEQLENS, N_HEAD
from harness.run_utils import require_neuron


def main() -> int:
    if not require_neuron():
        return 0
    attention_cte = require_attention_cte()
    if attention_cte is None:
        return 0

    print("=== Attention batch invariance (Part B) ===\n")
    print("Padded KV (causal):")
    ok1 = check_padded_kv(attention_cte)
    print("\nAlone vs co-packed batch:")
    ok2 = check_batch_schedule(attention_cte, batch_heads=N_HEAD)
    print("\nRun-to-run:")
    ok3 = check_run_to_run(attention_cte, ATTN_SEQLENS, batch_heads=N_HEAD)

    ok = ok1 and ok2 and ok3
    print(f"\n{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
