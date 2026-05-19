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
