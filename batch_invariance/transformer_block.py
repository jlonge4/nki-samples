"""
NKI Transformer Block — pre-norm decoder block.

  - MatMul / RMSNorm: local batch-invariant teaching kernels (``deterministic`` tile flag)
  - Attention: nki-library ``attention_cte`` (flash prefill)

Shape constraints:
  d_head == 128          (attention_cte head dim)
  seq    % 128 == 0      (attention_cte Q groups)
  d_model % 128 == 0     (matmul K tiles)
  d_ffn   % 128 == 0

Demo values: seq=512, d_model=256, d_head=128, d_ffn=512
"""

import torch

from harness.attention_cte_torch import attention_cte_forward
from kernels.matmul_batch_invariant import nki_matmul_kernel_isa
from kernels.rmsnorm_batch_invariant import nki_rmsnorm_kernel_isa


def make_block_weights(d_model, d_head, d_ffn, dtype=torch.bfloat16):
    """
    Returns a dict of CPU tensors.  Move to device before passing to the block:
        weights = make_block_weights(...)
        weights = {k: v.to(device) for k, v in weights.items()}
    """
    scale = 0.02
    return {
        "wq": torch.randn(d_model, d_head, dtype=dtype) * scale,
        "wk": torch.randn(d_model, d_head, dtype=dtype) * scale,
        "wv": torch.randn(d_model, d_head, dtype=dtype) * scale,
        "wo": torch.randn(d_head, d_model, dtype=dtype) * scale,
        "w1": torch.randn(d_model, d_ffn, dtype=dtype) * scale,
        "w2": torch.randn(d_ffn, d_model, dtype=dtype) * scale,
        "g_attn": torch.ones(d_model, dtype=dtype),
        "g_ffn": torch.ones(d_model, dtype=dtype),
    }


def nki_transformer_block(x, weights, deterministic=True):
    """
    Pre-norm transformer block:
      x -> RMSNorm -> QKV proj -> attention_cte -> out proj -> residual
        -> RMSNorm -> FFN up -> ReLU -> FFN down -> residual

    ``deterministic`` applies to matmul and RMSNorm only (attention_cte has no tile knob).
    """
    device = x.device
    w = {k: v.to(device) for k, v in weights.items()}

    def mm(a, b):
        return nki_matmul_kernel_isa(a.T.contiguous(), b, deterministic=deterministic)

    def rms(a, g):
        return nki_rmsnorm_kernel_isa(a, g, deterministic=deterministic)

    def attn(q, k, v):
        return attention_cte_forward(q, k, v, causal_mask=True)

    x_norm = rms(x, w["g_attn"])
    q = mm(x_norm, w["wq"])
    k = mm(x_norm, w["wk"])
    v = mm(x_norm, w["wv"])
    attn_out = attn(q, k, v)
    x = x + mm(attn_out, w["wo"])
    x_norm = rms(x, w["g_ffn"])
    h = mm(x_norm, w["w1"])
    h = torch.relu(h)
    x = x + mm(h, w["w2"])
    return x
