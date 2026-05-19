# Batch-invariant NKI kernel library (forward)

DeepSeek-V4 §3.3-style goal for **forward** ops on AWS Neuron: bitwise batch invariance and run-to-run determinism when using `deterministic=True` (fixed reduction tiles, NKI ISA, bf16).

## Altitude ladder

| Layer | What | Script |
|-------|------|--------|
| L0 | Tile A vs B (mechanism) | `tests/test_l0_tile_invariance.py` |
| L1 | Part A: M, position, neighbors @ nanochat K | `tests/test_l1_serving_battery.py` |
| L2 | Continuous batching (packing, neighbors) | `tests/test_l2_continuous_batching.py` |
| L3 | Attention packing + seqlen sweep | `tests/test_l3_attention_packing.py` |
| L4 | Composed block (demo dims) | `tests/test_l4_block_e2e.py` |
| — | Run-to-run + cross-M | `tests/test_determinism_harness.py` |

Run all: `./run_e2e_harness.sh`

## Kernel status (fill after running on Trainium)

| Kernel | Batch-invariant | Deterministic forward | Evidence |
|--------|-----------------|----------------------|----------|
| MatMul (ISA) | TBD | TBD | L0 + L1 |
| RMSNorm (ISA) | TBD | TBD | L0 + L1 + L2 |
| Attention (ISA) | TBD | TBD | L0 + L3 |
| Decoder block | TBD | TBD | L4 |

Bitwise pass/fail: `torch.equal` on bf16 int16 views (see `harness/bi_testkit.py`, vendored from peer study).

## Out of scope (v1)

- Attention / MoE **backward** deterministic atomics
- In-repo nanochat dependency
- Prefix-cache split API (Part B B3) on sample SDPA

See [NANOCHAT_INTEGRATION.md](NANOCHAT_INTEGRATION.md) for optional external nanochat hooks.
