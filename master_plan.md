# Parameter Golf Optimization Master Plan

## Context

**Challenge**: Train the best LM fitting in 16,000,000-byte artifact, trainable in <10 min on 8×H100. Metric: BPB on FineWeb validation set (lower = better).
**Timeline**: March 18 – April 30, 2026. Compute credits: $1M via RunPod.
**Submission**: GitHub PR to `records/track_10min_16mb/` with `train_gpt.py`, `train.log`, `submission.json`. Must beat SOTA by **≥0.005 nats, p<0.01**.

**Old Baseline** (weight-shared arch): BPB = 1.2244, ~17M params, int8+zlib
**Winner Reference**: BPB = 1.1243, ~27M params, int6+zstd

**Key Insight**: The winner uses **int6 + zstd quantization** (~0.59 bytes/param) to fit ~27M params in 16MB vs our old ~17M params with int8+zlib (~0.92 bytes/param). More params > clever weight sharing at this scale.

---

## Current Architecture (Winner-Based)

Implemented in `train_gpt.py`:

- **11 independent blocks** (no weight sharing), dim=512, 8 heads, 4 KV heads
- **3× MLP** with ReLU² activation
- **Partial RoPE**: 16/64 dims + NTK scaling for long context
- **int6 quantization** (mlp/attn) + int8 (embeddings) + zstd-22 compression
- **Late QAT**: STE fake-quantization enabled when lr_scale < 0.18
- **SmearGate**: Sigmoid-gated, single shared across all layers
- **BigramHash**: 2048 buckets, dim=128 → cheap bigram features
- **Value Embeddings**: Shared table, layers 8,9,10 (one extra vs winner)
- **XSA**: Last 4 layers — removes self-attention bias
- **LN Scale**: 1/sqrt(layer_idx+1) for deep layer stability
- **EMA**: decay=0.997, fp32 accumulation
- **SWA**: From EMA, every 40 steps (tighter than winner's 50)
- **U-Net skip connections**: Encoder-decoder style with learned skip weights
- **FA3/SDPA auto-detection**: FA3 on Hopper, SDPA fallback on Ada/Ampere
- **Orthogonal init**: With proj scaling 1/sqrt(2*num_layers)
- **AdamW weight decay**: 0.04 for both Adam and Muon params
- **Grad clip**: 0.3

### Our Improvements Over Winner

| Change | Winner | Ours | Rationale |
|--------|--------|------|-----------|
| VE layers | 9,10 | 8,9,10 | More layers benefit from token identity |
| Late QAT threshold | 0.15 | 0.18 | ~20% more QAT training steps |
| SWA frequency | every 50 | every 40 | ~25% more averaging checkpoints |
| Warmdown | 3500 iters | 3750 iters | More convergence tail |
| Muon momentum warmup | 1500 steps | 1750 steps | Smoother transition |

---

## Quick Reference Commands

### 4090 Local Quick Sanity (~5 min)
```bash
COMPILE_MODE=off \
ITERATIONS=500 \
VAL_LOSS_EVERY=100 \
VAL_BATCH_SIZE=65536 \
TRAIN_BATCH_TOKENS=65536 \
TRAIN_SEQ_LEN=1024 \
WARMDOWN_ITERS=200 \
WARMUP_STEPS=5 \
MAX_WALLCLOCK_SECONDS=300 \
SWA_ENABLED=0 \
EMA_ENABLED=0 \
LATE_QAT_THRESHOLD=0 \
QAT_ENABLED=0 \
python train_gpt.py
```

### Artifact Size Check
```bash
SIZE_ONLY=1 python train_gpt.py
```

### Full 8×H100 Run
```bash
SEED=1337 NUM_LAYERS=11 MLP_MULT=3.0 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=40 LATE_QAT_THRESHOLD=0.18 WARMDOWN_ITERS=3750 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=8,9,10 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 ADAM_WD=0.04 MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM_WARMUP_STEPS=1750 \
torchrun --nproc_per_node=8 train_gpt.py
```

---

## Verification Checklist

1. **Artifact size**: `SIZE_ONLY=1 python train_gpt.py` → must be < 16MB
2. **Loss decreases**: Quick local run → loss should drop from ~7.0 to < 6.0
3. **VRAM fits**: Single 4050/4090 with 65K batch tokens → < 24GB
4. **Quant roundtrip**: int6+zstd → decompress → dequantize → eval works
5. **SDPA fallback**: Works on 4050/4090 (no flash_attn_interface needed)
6. **Sliding window eval**: Produces BPB metric correctly
7. **EMA + SWA**: Weight averaging runs and loads correctly
8. **Full local run**: 500 iterations on 4050/4090, loss decreasing

---

## Key Dependencies

- `zstandard` Python package (for zstd compression)
- `flash_attn_interface` — only on Hopper GPUs (RunPod H100), NOT needed for 4050/4090
- `sentencepiece`, `torch`, `numpy`

## Critical Files
- `train_gpt.py` — All model + training code
- `winner_train_gpt.py` — Original winner code for reference
- `master_plan.md` — This file
- `data/cached_challenge_fineweb.py` — Download more shards
- `initial_info/evaluation.md` — Official rules
