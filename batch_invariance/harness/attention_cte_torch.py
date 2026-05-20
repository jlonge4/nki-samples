"""TorchNeuron wrapper: (seq, d) tensors -> attention_cte -> (seq, d)."""

from __future__ import annotations

from typing import Any, Optional

import torch

from harness.attention_cte_ops import ATTN_SCALE, import_attention_cte


def attention_cte_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal_mask: bool = True,
    bound_min: Optional[torch.Tensor] = None,
    bound_max: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """q, k, v: [seq, d] bfloat16 on Trainium -> [seq, d]."""
    attention_cte = import_attention_cte()
    if attention_cte is None:
        raise RuntimeError("nki-library (nkilib) is required for attention_cte")

    q_b = q.unsqueeze(0)
    k_cte = k.unsqueeze(0).transpose(1, 2).contiguous()
    v_b = v.unsqueeze(0)
    kwargs: dict[str, Any] = dict(scale=ATTN_SCALE, causal_mask=causal_mask)
    if bound_min is not None and bound_max is not None:
        kwargs["bound_min"] = bound_min
        kwargs["bound_max"] = bound_max
    out = attention_cte(q_b, k_cte, v_b, **kwargs)
    return out[0] if out.ndim == 3 else out.squeeze(0)
