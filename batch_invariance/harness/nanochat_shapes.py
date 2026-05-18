"""Nanochat d20 shape targets (no nanochat dependency)."""

# Model
N_EMBD = 1280
N_HEAD = 10
HEAD_DIM = 128
FFN_DIM = 5120
VOCAB_SIZE = 50304
MAX_SEQ_LEN = 2048

# Linear (K, N) for matmul batteries
LINEAR_SHAPES = {
    "c_q": (N_EMBD, N_EMBD),
    "c_attn_proj": (N_EMBD, N_EMBD),
    "c_fc": (N_EMBD, FFN_DIM),
    "c_mlp_proj": (FFN_DIM, N_EMBD),
    "lm_head": (N_EMBD, VOCAB_SIZE),
}

# Part A whole-block M pairs (nanochat-relevant + Neuron boundary cases)
NANOCHAT_WB_PAIRS = [
    (1, 2), (1, 128), (1, 256),
    (2, 128),
    (127, 128), (127, 256),
    (128, 256), (128, 2048),
    (255, 256), (255, 2048),
    (1024, 2048),
    (2048, 4096),
]

# Matmul kernel requires M % 128 == 0 (full M_TILE slabs only)
MATMUL_WB_PAIRS = [
    (128, 256),
    (128, 2048),
    (1024, 2048),
    (2048, 4096),
]

POSITION_M_VALUES = (128, 255, 256, 2048)
NEIGHBOR_CONFIGS = ((128, 0), (255, 128), (256, 128), (2048, 1024))

# Demo block (SBUF N limit in matmul kernel)
DEMO_D_MODEL = 256
DEMO_D_FFN = 512
DEMO_SEQ = 512

# Attention Part B
ATTN_SEQLENS = (128, 256, 512, 1024, 2048)
