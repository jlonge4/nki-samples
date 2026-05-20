# Batch invariance on Neuron

Bitwise-identical bf16 when the **logical** computation is unchanged but **physical** scheduling is not (batch size, row index, neighbors, reduction tiles, packed attention).

Nanochat d20 shapes. PSUM / K-tile mechanism: [EXPLAINER.md](EXPLAINER.md).

## Install

Trainium: Neuron SDK + `torch_neuronx` + [nki-library](https://github.com/aws-neuron/nki-library): `pip install "git+https://github.com/aws-neuron/nki-library.git"` (not PyPI 0.0.1).

Laptop: `./run.sh setup-sim` → `source nki-cpu-sim/bin/activate` (same nki-library for attention sim).

## Run

```bash
./run.sh e2e    # Trainium — table below
./run.sh sim    # CPU — tile + row battery + attention_cte simulate
./run.sh demo   # Trainium toy 2-block pipe
```

Pass/fail: `torch.equal` on bf16 `int16` views (no `allclose`).

## Properties under test

Three **batch-invariance** claims (plus run-to-run sanity). Each maps to one script — no “L0–L4” ladder.

| Property | What must stay identical | What we vary | Script |
|----------|-------------------------|--------------|--------|
| **Reduction-tile invariance** | Full tensor output | MatMul/RMSNorm `deterministic=True` (K/H tile 128) vs `False` (64) | `tests/test_l0_tile_invariance.py` |
| **Row schedule invariance** (Part A) | Each output row for a fixed input row | Batch size `M`, row index, filler neighbors | `tests/test_l1_serving_battery.py` (`bi_testkit`: whole-block, position, neighbor) |
| **Row schedule invariance** (MatMul spot) | `Y[i]` from `X @ W` | Full `M` vs isolated row; prefix rows | `tests/test_l1b_matmul_m_invariance.py` |
| **Attention batch invariance** (Part B) | Same logical request’s output | Padded KV; co-packed batch heads | `tests/test_l3_attention_packing.py` |
| **Run-to-run stability** | Output on repeat calls | Same inputs, same shape, re-invoke | `tests/test_determinism_harness.py`, `test_l3` (attention) |
| **Composed block** | Above on a wired pre-norm block | Tile flag (matmul/RMSNorm only), packed seq | `tests/test_l4_block_e2e.py` |

**Row schedule invariance** = Part A predicates: (1) `M` does not matter, (2) row index does not matter, (3) other rows’ values do not matter.

**Attention batch invariance** = Part B: same logical token under the same attention graph when batch layout or physical KV length changes.

`./run.sh sim` runs reduction-tile + row battery (simulate) + the same Part B attention checks via `nki.simulate(attention_cte)`.

## Code

- `kernels/matmul_batch_invariant.py`, `rmsnorm_batch_invariant.py` — tile flag
- `harness/attention_cte_ops.py` — flash prefill + Part B checks
- `harness/bi_testkit.py` — Part A battery
