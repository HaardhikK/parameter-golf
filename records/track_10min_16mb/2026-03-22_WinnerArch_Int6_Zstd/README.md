# WinnerArch Int6+Zstd — 11L 512dim, EMA+SWA, Late QAT

**Author:** HaardhikK
**Date:** 2026-03-22
**Score (sliding window BPB, stride=64):** FILL_IN
**Artifact size:** FILL_IN bytes

---

## Summary

This submission implements the winning architecture from the community record (1.1243 BPB) with five conservative improvements. The key insight is that **int6+zstd compression allows ~27M parameters in 16MB** vs ~17M with the naive int8+zlib baseline — more parameters dominates at this scale.

See `explanation.md` at the repo root for a detailed writeup of every architectural and training decision.

---

## Architecture

- **11 independent transformer blocks**, dim=512, 8 heads, 4 KV heads (GQA)
- **3× MLP** with ReLU² activation (hidden=1536)
- **Partial RoPE**: 16/64 head dims + NTK scaling for long context
- **U-Net skip connections**: 5 encoder → 6 decoder layers
- **SmearGate**: sigmoid-gated 1-token lookback, per-dimension
- **BigramHash**: 2048 buckets, dim=128
- **Value Embeddings**: shared table dim=128, applied at layers 8, 9, 10
- **XSA**: self-attention bias removal on last 4 layers
- **LN Scale**: 1/√(layer_idx+1) per block
- **Logit softcap**: ±30
- **Tied embeddings**, orthogonal init with proj scaling

## Quantization

- MLP + attention weights: **int6 per-row** (6-bit, 64 levels)
- Embeddings: **int8 per-row**
- Control tensors: **fp32** passthrough
- Compression: **zstd level 22**
- Result: ~0.59 bytes/param → ~27M params in ~15 MB trained artifact

## Training

- **Muon** (matrices): lr=0.025, momentum 0.92→0.99 over 1750 steps, WD=0.04
- **AdamW** (embeddings + scalars): lr=0.035/0.025, WD=0.04
- Gradient clip: 0.3
- Batch: 786,432 tokens/step, seq_len=2048
- Warmdown: 3750 iters (wallclock-based, last portion of 10 min)
- **EMA**: decay=0.997, float32 accumulation every step
- **SWA**: collects EMA snapshots every 40 steps when scale<0.2
- **Late QAT**: STE int6 fake-quantization enabled when lr_scale < 0.18
- **Sliding window eval**: stride=64 (submission score)

## Improvements Over Prior Record (1.1243 BPB)

| Change | Prior Record | This Submission | Rationale |
|--------|-------------|-----------------|-----------|
| Value embedding layers | 9, 10 | **8, 9, 10** | More layers benefit |
| Late QAT threshold | 0.15 | **0.18** | ~20% more QAT steps |
| SWA frequency | every 50 | **every 40** | More averaging checkpoints |
| Warmdown | 3500 iters | **3750 iters** | Longer convergence tail |
| Muon momentum warmup | 1500 steps | **1750 steps** | Smoother transition |

## Run Command

```bash
NCCL_IB_DISABLE=1 \
SEED=1337 \
RUN_ID=winner_arch_s1337 \
NUM_LAYERS=11 MLP_MULT=3.0 XSA_LAST_N=4 ROPE_DIMS=16 LN_SCALE=1 \
SWA_ENABLED=1 SWA_EVERY=40 LATE_QAT_THRESHOLD=0.18 WARMDOWN_ITERS=3750 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=8,9,10 \
EMA_ENABLED=1 EMA_DECAY=0.997 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 ADAM_WD=0.04 MUON_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM_WARMUP_STEPS=1750 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Key Metrics

FILL_IN after runs complete:

| Seed | Steps | Sliding BPB (s64) | Post-avg BPB | Quant gap | Artifact |
|------|-------|-------------------|--------------|-----------|---------|
| 1337 | FILL | FILL | FILL | FILL | FILL MB |
| 42   | FILL | FILL | FILL | FILL | FILL MB |
| 7    | FILL | FILL | FILL | FILL | FILL MB |

**Best:** FILL | **Mean:** FILL | **Std:** FILL

## Files

- `train_gpt.py` — exact script used for this run
- `train.log` — full training log (seed 1337)
- `submission.json` — leaderboard metadata
