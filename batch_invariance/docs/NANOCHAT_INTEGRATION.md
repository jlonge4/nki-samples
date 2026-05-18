# Optional nanochat integration (external)

This repo does **not** clone [nanochat](https://github.com/karpathy/nanochat). Shapes follow d20 (`n_embd=1280`, `head_dim=128`, …) via `harness/nanochat_shapes.py`.

## External demo (paper-style)

1. Clone nanochat elsewhere; install Neuron + this kernel package on `PYTHONPATH`.
2. Replace `F.rms_norm(x, (H,))` with `nki_rmsnorm_kernel_isa(x, ones(H), deterministic=True)` on `x.to("neuron")` (TorchNeuron; `import torch_neuronx` for NKI).
3. Run two forwards that differ only in batch layout, e.g.:
   - `B=1, T=2048` vs `B=8, T=256` (same total tokens if data aligned)
4. Forward-hook layer `L`; compare activations for the same logical token — expect **bitwise** match if batch invariant.

## What in-repo harness already proves

- **L1–L2**: same logical **rows** at nanochat `H=1280` under M / position / neighbor changes
- **L3**: same logical **attention** prefix under padded KV
- **L4**: same block **prefix** under padded sequence (demo `d_model=256`)

That is sufficient for the kernel-library claim without an in-tree nanochat run.
