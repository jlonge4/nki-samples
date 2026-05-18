# Harness

- `bi_testkit.py` — vendored Part A three-predicate battery.
- `neuron_device.py` — **TorchNeuron first** (`torch.device("neuron")`, `torch_neuronx`), torch_xla fallback.
- `nanochat_shapes.py` — d20 shape constants (no nanochat import).
- `serving_adapters.py` — NumPy wrappers for `@nki.jit` kernels (`deterministic=True`).
- `run_utils.py` — `require_neuron()` checks nki + Trainium PyTorch backend.

PyTorch Native: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html
