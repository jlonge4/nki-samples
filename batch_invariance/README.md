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

## E2E harness (`e2e-determinism` branch)

Altitude ladder from first-principles tiles to nanochat-shaped serving tests:

| Layer | Script |
|-------|--------|
| L0 | `test_tile_invariance.py` (det vs nondet) |
| L1 | `tests/test_l1_serving_battery.py` (`bi_testkit`, H=1280) |
| L2 | `tests/test_l2_continuous_batching.py` |
| L3 | `tests/test_l3_attention_packing.py` |
| L4 | `tests/test_l4_block_e2e.py` |
| — | `tests/test_determinism_harness.py` |

```bash
cd batch_invariance
# TorchNeuron (PyTorch Native on Trn2/Trn3) or legacy torch-neuronx venv
source <your-neuron-venv>/bin/activate
./run_e2e_harness.sh
```

**Device stack:** [Native PyTorch for Trainium](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html) — `torch.device("neuron")`, `import torch_neuronx` for NKI. L0/L4 use `harness/neuron_device.py` (torch_xla fallback if needed).

Docs: `docs/BATCH_INVARIANT_KERNELS.md`, `docs/NANOCHAT_INTEGRATION.md` (optional external nanochat — **no clone in repo**).

`harness/bi_testkit.py` is vendored from the peer batch-invariance study (minimal steal).

## Project Structure

```
batch_invariance/
├── README.md
├── run_e2e_harness.sh
├── harness/                        # serving tests + vendored bi_testkit
├── tests/                          # L1–L4 + determinism
├── docs/
├── kernels/
├── transformer_block.py
├── test_tile_invariance.py
├── test_block_invariance.py
├── test_batch_invariance.ipynb
├── simulate_batch_invariance.py
└── inspect_psum.py
```

## Running the Tests

### Standalone scripts (recommended first)

```bash
cd batch_invariance
source <your-neuron-venv>/bin/activate

python3 test_tile_invariance.py      # device=neuron (TorchNeuron)
python3 test_block_invariance.py
```

### Notebook

Open `test_batch_invariance.ipynb` in JupyterLab. Run all cells top-to-bottom.
Sections: MatMul → RMSNorm → Attention → Full Block → Continuous Batching → Summary.

### CPU Simulator (no hardware required)

```bash
NKI_PRECISE_FP=1 python3 simulate_batch_invariance.py
```

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
