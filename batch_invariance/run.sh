#!/usr/bin/env bash
# ./run.sh e2e | sim | setup-sim | demo | debug
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${PYTHON:-python3}"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh e2e              Trainium: properties in README table
  ./run.sh sim              CPU: reduction-tile + row battery + attention_cte sim
  SIM_FULL=1 ./run.sh sim   Larger H and attention seqlen sweep
  ./run.sh setup-sim        nki-cpu-sim venv
  ./run.sh demo             Toy inference (Trainium)
  ./run.sh debug psum|m-tail
EOF
}

run_e2e() {
  export NEURON_RT_VISIBLE_CORES="${NEURON_RT_VISIBLE_CORES:-0}"
  local FAIL=0
  run_one() {
    local name="$1"
    shift
    echo ""
    echo "========================================================================"
    echo "  $name"
    echo "========================================================================"
    if "$PY" "$@"; then
      echo ">>> $name: OK"
    else
      echo ">>> $name: FAIL (exit $?)"
      FAIL=$((FAIL + 1))
    fi
  }
  run_one "Reduction-tile invariance" tests/test_l0_tile_invariance.py
  run_one "Row schedule invariance (Part A battery)" tests/test_l1_serving_battery.py
  run_one "Row schedule invariance (MatMul spot)" tests/test_l1b_matmul_m_invariance.py
  run_one "Attention batch invariance (Part B)" tests/test_l3_attention_packing.py
  run_one "Composed block" tests/test_l4_block_e2e.py
  run_one "Run-to-run stability" tests/test_determinism_harness.py
  echo ""
  echo "========================================================================"
  if [ "$FAIL" -eq 0 ]; then echo "  e2e: PASS"; exit 0; fi
  echo "  e2e: FAIL ($FAIL stages)"; exit 1
}

run_sim() {
  export NKI_PRECISE_FP="${NKI_PRECISE_FP:-1}"
  export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
  "$PY" tools/kernel_debug.py psum
  echo ""
  "$PY" tools/kernel_debug.py m-tail
  echo ""
  "$PY" tests/run_simulator.py "$@"
}

CMD="${1:-}"
shift || true
case "$CMD" in
  e2e|trainium) run_e2e ;;
  sim|simulate) run_sim "$@" ;;
  setup-sim) exec ./setup_sim.sh ;;
  demo) "$PY" demo_deterministic_pipe.py "$@" ;;
  debug|inspect)
    SUB="${1:-}"
    shift || true
    [ -n "$SUB" ] || { usage; exit 1; }
    export NKI_PRECISE_FP="${NKI_PRECISE_FP:-1}"
    export NEURON_PLATFORM_TARGET_OVERRIDE="${NEURON_PLATFORM_TARGET_OVERRIDE:-trn2}"
    "$PY" tools/kernel_debug.py "$SUB" "$@"
    ;;
  -h|--help|help|"") usage; [ -z "$CMD" ] && exit 1 || exit 0 ;;
  *) echo "Unknown command: $CMD"; usage; exit 1 ;;
esac
