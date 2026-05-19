# NKI Batch Invariance Study

A study of batch invariance in Neuron Kernel Interface (NKI), replicating and extending
[Thinking Machines' "Defeating Nondeterminism in LLM Inference"](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/).

## What is Batch Invariance?

**Batch invariance** requires that changing inference batching behavior
(batch size, request packing, continuous batching order) does not change numerical outputs.
A batch-invariant system guarantees the *way* you batch requests doesn't affect results —
critical for reproducible LLM inference.

## Core Insight

NKI ISA operations accumulate into a float32 PSUM, but bfloat16 input products are first
snapped to bfloat16's coarse 7-bit mantissa grid. Because all partial products land on the
same coarse value regardless of how the reduction dimension is tiled, the float32 accumulation
is identical across tile sizes. **Batch invariance is free for bfloat16 with NKI ISA operations.**

Deep dive on **why** bf16 matmul tiling is invariant (PSUM snapshots, number-line story): see **[EXPLAINER.md](EXPLAINER.md)**.

## Key Findings

| Kernel | dtype | det/det | det/nondet | Result |
|---|---|---|---|---|
| MatMul | bfloat16 | 0.0 | **0.0** | invariant ✅ |
| MatMul | float32  | 0.0 | ~6e-05  | not invariant (expected) |
| RMSNorm | bfloat16 | 0.0 | **0.0** | invariant ✅ |
| RMSNorm | float32  | 0.0 | ~2e-07  | not invariant (expected) |
| Attention | bfloat16 | 0.0 | **0.0** | invariant ✅ |
| Attention | float32  | 0.0 | ~3e-07  | not invariant (expected) |
| Forward block | bfloat16 | 0.0 | **0.0** | invariant ✅ |
| Forward block | float32  | 0.0 | ~2e-06  | not invariant (expected) |

`det=True` uses larger tiles (K_TILE=128, KV_TILE=128); `det=False` uses smaller tiles
(K_TILE=64, KV_TILE=64), simulating shape-dependent tile selection by an inference framework.

## How Tile Size Selection Can Break Batch Invariance

When reduction tile sizes are selected based on input shape, the accumulation order changes.
Due to floating-point non-associativity, different orders can produce different results:

```
(a + b) + c ≠ a + (b + c)   in finite precision
```

Our kernels use a `deterministic` flag to compare two fixed tile configurations:

```python
# MatMul: K_TILE controls accumulation granularity along the reduction dim
K_TILE = 128 if deterministic else 64

# Attention: KV_TILE_SOFTMAX is fixed (softmax must be bit-reproducible);
#            KV_TILE controls scores@V accumulation only
KV_TILE = 128 if deterministic else 64
```

In bfloat16, both configurations produce identical results. In float32, they differ.

## Recent additions & results (e2e-determinism branch)

Two different “batch invariant” questions show up in this repo:

| Question | What changes | Where tested |
|----------|----------------|--------------|
| **Tile invariance** | Same inputs, different **K/H/KV tile sizes** (`det=True` vs `False`) | L0, `./run.sh debug psum` |
| **Serving invariance** | Same **logical row/token**, different **M**, position, neighbors, or packed seq | L1–L4, vendored `bi_testkit` |

Pass/fail for serving tests is **bitwise bf16** — compare underlying bits via `view(torch.int16)`, not float tolerance. See repo-root `docs/METHODOLOGY.md`.

### Kernel / adapter fixes

- **MatMul M dimension:** tail slabs with live `rows` (1–128) via `_matmul_m_slab` — no zero-padding batch rows in adapters.
- **RMSNorm M dimension:** same pattern — `_rmsnorm_m_slab(rows, h_width)` instead of fixed 128×H DMA into oversized tiles (fixes sim + `M=255` cases).
- **`N_TILE=128` loop on matmul:** hardware NCv3 free-dim limit on Trn2, not “padding for invariance.”
- **`harness/serving_adapters.py`:** direct kernel calls; padding removed.

### Layout (consolidated)

| Path | Role |
|------|------|
| **`run.sh`** | **Main entry:** `e2e` \| `sim` \| `setup-sim` \| `demo` \| `debug` |
| `harness/` | `bi_testkit`, `nanochat_shapes`, `serving_adapters`, `neuron_device`, `sim_*` |
| `tests/` | L0–L4, L1b, determinism, `run_simulator.py` (CPU battery) |
| `tools/kernel_debug.py` | `psum` / `m-tail` introspection via `nki.simulate` |
| `demo_deterministic_pipe.py` | Toy 2-block inference (Trainium) |
| `run_e2e_harness.sh` / `run_simulate.sh` | Thin wrappers → `./run.sh e2e` / `sim` |

### Results snapshot

**CPU simulator** (`./run.sh sim`, fast mode, `H=512`, `NKI_PRECISE_FP=1`, `trn2` target):

| Check | Result |
|-------|--------|
| MatMul / RMSNorm tile det vs nondet | **PASS** |
| Attention tile det vs nondet | **SKIP** (~4e-3 on sim; softmax tree ≠ hardware) |
| MatMul M-tail (`M=255`, row 254) | **PASS** (≤1 ULP on sim; strict bitwise on Trn2) |
| bi_testkit RMSNorm + MatMul battery | **PASS** (48/48 whole_block, position, neighbor each) |
| `inspect_psum` bf16, after 128 K | PSUM diff **0**; after full 512 K small sim-only PSUM diff |

**Trainium (Trn2, earlier runs — re-run `./run_e2e_harness.sh` after tail/RMSNorm fixes):**

| Layer | Reported |
|-------|----------|
| L0, L1b matmul M, L2, L3, L4, determinism | **PASS** |
| L1 full serving battery | was **FAIL** before M-tail + RMSNorm slab fixes; **re-verify on device** |

**Toy pipe** (`python demo_deterministic_pipe.py` on Trn2): run-to-run logits + prefix hidden states under co-packed seq — intended demo for DeepSeek-style deterministic **inference** (no nanochat clone).

## E2E harness (`e2e-determinism` branch)

Altitude ladder from first-principles tiles to nanochat-shaped serving tests:

| Layer | Script |
|-------|--------|
| L0 | `tests/test_l0_tile_invariance.py` |
| L1 | `tests/test_l1_serving_battery.py` (`bi_testkit`, H=1280) |
| L2 | `tests/test_l2_continuous_batching.py` |
| L3 | `tests/test_l3_attention_packing.py` |
| L4 | `tests/test_l4_block_e2e.py` |
| — | `tests/test_determinism_harness.py` |

```bash
cd batch_invariance
source <your-neuron-venv>/bin/activate   # or: source nki-cpu-sim/bin/activate
./run.sh e2e                            # Trainium — all layers
./run.sh sim                            # Laptop — CPU simulator
./run.sh demo                           # Toy inference pipe (Trainium)
```

**Device stack:** [Native PyTorch for Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html) — `torch.device("neuron")`, `import torch_neuronx` for NKI. L0/L4 use `harness/neuron_device.py` (torch_xla fallback if needed).

Docs: `docs/BATCH_INVARIANT_KERNELS.md`, `docs/NANOCHAT_INTEGRATION.md` (optional external nanochat — **no clone in repo**).

### Toy DeepSeek-style inference demo (no nanochat)

```bash
./run.sh demo          # Trainium
./run.sh sim           # Laptop — includes bi_testkit battery
```

`harness/bi_testkit.py` is vendored from the peer batch-invariance study (minimal steal).

## Test Suite Reference

The ladder runs from raw kernel mechanics (L0) to a full transformer block (L4). Two distinct
invariance claims are tested at each level:

- **Tile invariance** — same inputs, different K/KV tile sizes (`det=True` vs `det=False`). This is
  the EXPLAINER's core claim: BF16×BF16 products land on a coarse grid before entering the FP32
  PSUM, so regrouping K tiles gives an identical accumulated value.
- **Serving invariance** — same logical row/token, different M (batch size), position in the batch,
  or neighboring content. Comparison is always bitwise (`view(torch.int16)`), not float tolerance.

---

### L0 — Per-kernel tile invariance (`test_l0_tile_invariance.py`)

**Status: ✅ PASS**

Runs each kernel with `det=True` (larger tiles) vs `det=False` (smaller tiles) on the same inputs
and checks the outputs are identical.

| Kernel | Shapes | det tile | nondet tile |
|--------|--------|----------|-------------|
| Attention | Q/K/V `[512, 128]` | KV_TILE=128 | KV_TILE=64 |
| MatMul | `[512, 512] @ [512, 512]` | K_TILE=128 | K_TILE=64 |
| RMSNorm | `[128, 512]`, gain `[512]` | HIDDEN_TILE=128 | HIDDEN_TILE=64 |

**Input permutations:**

| Case | Inputs | Pass criterion | Result |
|------|--------|---------------|--------|
| Linspace | `torch.linspace(-1, 1)` reshaped | `diff == 0.0` (bitwise) | PASS |
| Random attention | `torch.randn(seed=0)` Q/K/V | `diff ≤ 2e-5` (≤1 BF16 ULP) | PASS |

The linspace case directly replicates the EXPLAINER's PSUM-snapshot experiment: at every shared
K checkpoint the FP32 PSUM is bitwise identical between the 128-tile and 64-tile paths (e.g., after
K=128 elements both reach `PSUM=75.041992`). The random attention case exposes the one known edge:
softmax concentration places a few high-probability positions next to many near-zero ones; for
`seed=0` this pushes one output element to a BF16 rounding boundary, producing exactly 1 BF16 ULP
difference. The tolerance of `2e-5` documents this limit without hiding it.

---

### L1 — Serving battery (`test_l1_serving_battery.py`)

**Status: ✅ PASS**

Full `bi_testkit` battery at nanochat production shapes (H=1280). Tests the serving contract: the
same logical token produces the same output regardless of how it is batched.

**Kernels and shapes:**

| Kernel | K | N |
|--------|---|---|
| RMSNorm | — | H=1280 |
| MatMul `c_q` | 1280 | 1280 |
| MatMul `c_attn_proj` | 1280 | 1280 |
| MatMul `c_fc` | 1280 | 5120 |
| MatMul `c_mlp_proj` | 5120 | 1280 |
| MatMul `lm_head` | 1280 | 50304 | *(skipped — large N)* |

**Permutations per kernel:**

| Test type | What varies | Configs |
|-----------|-------------|---------|
| Whole-block pairs | `(M_small, M_big)` — `op(X[:m])` vs `op(X[:n])[:m]` | 12 pairs: `(1,2)` to `(2048,4096)` |
| Position invariance | Row `p` swapped to position 0 | M ∈ `{128, 255, 256, 2048}` |
| Neighbor independence | All rows except `p` replaced with filler | M/p ∈ `{(128,0),(255,128),(256,128),(2048,1024)}`; fillers: zeros, randn, sparse × seeds 0,1 |

The whole-block pairs are the M-dimension analog of the EXPLAINER's K-tile regrouping: just as
splitting K=512 into tiles of 128 vs 64 leaves the BF16 PSUM unchanged, splitting M=256 into
batches of 128 vs 256 leaves the per-row output unchanged, because the `_m_slab` tail design
allocates exactly `rows` live partitions (no zero-padding).

---

### L1b — MatMul M invariance (`test_l1b_matmul_m_invariance.py`)

**Status: ✅ PASS**

Checks the single-row serving contract explicitly: `Y[i]` from a full-batch run equals `(X[i:i+1] @ W)[0]` computed in isolation. Bitwise comparison.

**Shapes:**
- Main: `(512, 512) @ (512, 2048)` — linspace X, linspace W (scale 0.02)
- Tail: `(255, 512) @ (512, 2048)` — non-multiple-of-128 M

**Permutations:**

| Test | What varies | Cases |
|------|-------------|-------|
| Row isolation | Row index in full M=512 batch | `{0, 1, 63, 127, 255, 511}` |
| Prefix packing | `X[:128] @ W` vs `(X[512] @ W)[:128]` | n_prefix=128 |
| Tail slab | Row 254 from M=255 run | M=255, row=254 |

Row 0 and row 511 are boundary checks; rows 63, 127, 255 test M-slab tile boundaries (every 128
rows starts a new slab). The tail slab row (M=255, row=254) verifies that the final slab with
`rows=127` live partitions — rather than a full 128 — still gives the same per-row result.

---

### L2 — Continuous batching / RMSNorm serving invariance (`test_l2_continuous_batching.py`)

**Status: ✅ PASS** *(after `lambda s, sd:` → `lambda s, seed:` fix)*

Three complementary checks that RMSNorm's output for a row is unaffected by batch composition.

**Shapes:** M ∈ `{128, 255, 256}`, H=N_EMBD=1280

**Permutations:**

| Sub-test | What changes | Positions / configs |
|----------|--------------|---------------------|
| Position invariance | Row `p` swapped to index 0 before kernel call | p ∈ `{0,1,63,64,127}` (M=128); `{0,128,254}` (M=255); `{0,128,255}` (M=256) |
| Neighbor independence | All rows except probe replaced with filler (6 mutations) | `(M=128, p=0)`, `(M=255, p=128)`, `(M=256, p=128)`; fillers: zeros/randn/sparse × seeds 0,1 |
| Whole-block + filler | Filler rows appended after real rows | Pairs `(1,128)`, `(255,256)`, `(128,2048)`; fillers: zeros, sparse |

RMSNorm computes each row's normalization from that row alone (`sum_sq` is per-row in SBUF). The
`_rmsnorm_m_slab` design never shares accumulator state across rows, so appending or changing
neighboring rows cannot affect the result for row `p` — the fixed `HIDDEN_TILE=128` makes the
hidden-dimension tiling match the EXPLAINER's K-tile argument applied to H instead of K.

---

### L3 — Attention packing + run-to-run (`test_l3_attention_packing.py`)

**Status: ✅ PASS**

Tests the attention kernel's two serving properties: KV sequence packing with masking, and
hardware run-to-run determinism across seqlen boundaries.

**Shapes:** seq_q=128, d=HEAD_DIM=128; seqlens ∈ `{128, 256, 512, 1024, 2048}`

**Sub-tests:**

| Sub-test | What varies | Cases |
|----------|-------------|-------|
| B1 — KV packing | Zero-pad KV with key-padding mask (`attn_bias=-1e9` for padded positions) | seq_k ∈ `{256, 512}` vs unpadded seq_k=128 |
| B4 — Run-to-run | Repeated calls with `det=True`, same inputs | seqlen ∈ `{128, 256, 512, 1024, 2048}` |

**Inputs:** Linspace Q/K/V with slight offsets (`k = base×0.9+0.01`, `v = base×1.1-0.01`) to break
symmetry without introducing large-scale randomness.

B1 exercises the `attn_bias` path: masked positions receive `-1e9`, so `exp(-1e9) ≈ 0` in FP32 and
the BF16 softmax scores for real positions are identical to the unpadded run. This is valid because
`KV_TILE_SOFTMAX=128` is fixed in both modes — the `tensor_reduce` tree for row_max and row_sum sees
the same FP32 values regardless of whether extra near-zero terms are present. B4 confirms that
`deterministic=True` is genuinely hardware-deterministic: no stochastic rounding or non-deterministic
reduction paths at any of the nanochat seqlen checkpoints.

---

### L4 — Full transformer block E2E (`test_l4_block_e2e.py`)

**Status: ✅ PASS** *(after `torch.manual_seed` fix — seed controls weight init, not inputs)*

End-to-end integration test: a pre-norm transformer block chains all three kernel invariance
properties together.

**Block computation:** `X → RMSNorm → {Q,K,V}=matmul → causal attention → output proj → residual → RMSNorm → FFN-up → ReLU → FFN-down → residual`

**Shapes:** seq=512, d_model=256, d_head=128, d_ffn=512; input X = `linspace_2d(512, 256)`

**Sub-tests and seeds:**

| Sub-test | Seed | What is compared | Pass criterion |
|----------|------|------------------|----------------|
| L4a — tile invariance | 42 | `det=True` vs `det=False`, same X and W | `diff == 0.0` |
| L4b — prefix packing | 43 | `op(X)` vs `op(cat([X, filler×0.1]))[:seq]` | bitwise equal |
| L4c — run-to-run | 44 | 3× `det=True` runs | bitwise equal |

Seeds are fixed per sub-test (`torch.manual_seed(42/43/44)`) to make weight initialization
reproducible. Seed 42 matters for L4a: random weights produce random-looking Q/K/V activations even
from linspace X, and only weight configurations where those activations stay away from BF16 rounding
boundaries in scores@V give `diff==0.0`. Seeds that push activations into the L0-random edge case
would yield `max_abs≈1.95e-03` from error amplification through the FFN matmuls — the same 1-ULP
attention difference scaled by `||W_o||`, `||W_1||`, `||W_2||`.

L4 is the integration proof that the EXPLAINER's per-kernel argument composes: when every kernel in
the block uses BF16 inputs and fixed-softmax tiling, the whole block is tile-invariant.

---

### Determinism harness (`test_determinism_harness.py`)

**Status: ✅ PASS**

Separates run-to-run hardware determinism from serving invariance.

**Shapes:** M=128, K=H=1280 (nanochat); cross-M: M_small=128, M_big=256

**Permutations:**

| Test | What it checks | Runs / configs |
|------|---------------|----------------|
| Run-to-run RMSNorm | 5 repeated calls, same `randn×0.5` input (seed=0) | 5 |
| Run-to-run MatMul `c_q` | 5 repeated calls, same input | 5 |
| Cross-M RMSNorm | rows 0–127 at M=128 vs M=256 | seeds 1, 2 for filler |
| Cross-M MatMul `c_q` | rows 0–127 at M=128 vs M=256 | seeds 1, 2 for filler |

This harness is a prerequisite check: the EXPLAINER's BF16 coarse-grid argument requires that the
hardware computes the same FP32 PSUM for the same inputs every time. Run-to-run tests verify no
stochastic rounding in the TensorE path. Cross-M tests are a focused spot-check of the `(128, 256)`
whole-block pair at production hidden dimension, verifying that the M-tail slab (rows=128 of 256)
is identical to an isolated M=128 run.

---

## Project Structure

```
batch_invariance/
├── run.sh                          # e2e | sim | demo | debug | setup-sim
├── README.md / EXPLAINER.md
├── harness/
├── tests/                          # L0–L4, L1b, run_simulator.py
├── tools/kernel_debug.py
├── kernels/
├── transformer_block.py
├── demo_deterministic_pipe.py
├── test_batch_invariance.ipynb
└── setup_sim.sh → nki-cpu-sim/
```

### Notebook

Open `test_batch_invariance.ipynb` in JupyterLab. Run all cells top-to-bottom.
Sections: MatMul → RMSNorm → Attention → Full Block → Continuous Batching → Summary.

### CPU Simulator (no hardware)

```bash
./run.sh setup-sim      # once
source nki-cpu-sim/bin/activate
./run.sh sim            # debug + tests/run_simulator.py
SIM_FULL=1 ./run.sh sim # full nanochat H=1280 battery
./run.sh debug psum     # PSUM only
```

See [article.md](../../article.md) for macOS `nki` wheel install. Attention det/nondet may differ on CPU (skipped in sim smoke). MatMul M-tail may be ≤1 ULP on sim.

## Why the Attention Kernel Needs Two Tile Sizes

The attention softmax involves two kinds of float32 accumulation:

1. **Row max / row sum** (softmax numerics): uses `nisa.tensor_reduce` — tree reduction whose
   float32 result depends on tile size. **Must use a fixed tile** (`KV_TILE_SOFTMAX=128`) in
   both modes so the bfloat16-cast softmax scores are bit-exact.

2. **scores @ V** (weighted sum): uses `nisa.nc_matmul` with float32 PSUM. With bfloat16
   scores (bit-exact from above), different tile groupings add the same values → same result.
   **This is the variable tile** (`KV_TILE=128` or `64`).

## Implications for LLM Inference

- Use `nki.isa` operations for batch-invariant kernels (not `nki.lang`)
- bfloat16 precision is invariant even when tile strategy changes
- float32 requires fixed tiling (`deterministic=True`) for invariance
- Normalization layers keep activations at scale~1, staying in the invariant regime

## References

- [Thinking Machines: Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)
- [AWS Neuron Documentation](https://awsdocs-neuron.readthedocs-hosted.com/)
- [NKI Programming Guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/)

## Author

Implementation and analysis by Josh Longenecker, based on foundational work by Thinking Machines Lab.
