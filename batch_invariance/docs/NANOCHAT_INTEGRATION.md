# Optional nanochat (external)

No in-repo nanochat clone. Shapes: `harness/nanochat_shapes.py`.

In-repo harness already covers row schedule + attention batch invariance at d20 shapes — see [README.md](../README.md).

To hook nanochat: swap `F.rms_norm` / attention for `nki_rmsnorm_kernel_isa` + `attention_cte`; compare layer hooks for the same logical token across batch layouts (bitwise bf16).
