#!/usr/bin/env bash
# E2E determinism + batch-invariance harness (L0–L4 + determinism).
set -euo pipefail
cd "$(dirname "$0")"
export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

PY="${PYTHON:-python3}"
FAIL=0
SKIP=0

run() {
  local name="$1"
  shift
  echo ""
  echo "========================================================================"
  echo "  $name"
  echo "========================================================================"
  if "$PY" "$@"; then
    echo ">>> $name: OK"
  else
    code=$?
    if [ "$code" -eq 0 ]; then
      echo ">>> $name: SKIP"
      SKIP=$((SKIP + 1))
    else
      echo ">>> $name: FAIL (exit $code)"
      FAIL=$((FAIL + 1))
    fi
  fi
}

run "L0 tile invariance" tests/test_l0_tile_invariance.py
run "L1 serving battery" tests/test_l1_serving_battery.py
run "L1b matmul M invariance" tests/test_matmul_m_invariance.py
run "L2 continuous batching" tests/test_l2_continuous_batching.py
run "L3 attention packing" tests/test_l3_attention_packing.py
run "L4 block E2E" tests/test_l4_block_e2e.py
run "Determinism harness" tests/test_determinism_harness.py

echo ""
echo "========================================================================"
echo "  E2E harness summary"
echo "========================================================================"
if [ "$FAIL" -eq 0 ]; then
  echo "  All stages completed (some may have SKIP without Neuron/XLA)."
  exit 0
fi
echo "  FAILURES: $FAIL"
exit 1
