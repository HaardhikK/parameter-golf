# Parameter Golf Optimization Master Plan

## Context

**Challenge**: Train the best LM fitting in 16,000,000-byte artifact, trainable in <10 min on 8×H100. Metric: BPB on FineWeb validation set (lower = better).
**Timeline**: March 18 – April 30, 2026.
**Submission**: GitHub PR to `records/track_10min_16mb/` with `train_gpt.py`, `train.log`, `submission.json`. Must beat SOTA by **≥0.005 nats, p<0.01**.

**Winner Reference**: BPB = 1.0745 (3-seed mean), ~27M params, int5/int6+zstd, 5-expert Hedge mixer + TTT
**Our Target**: ~1.050-1.060 BPB

---

## Current Architecture (Winner + 10 Improvements)

Implemented in `train_gpt.py` (based on `winner_train_gpt.py`):

- **11 independent blocks**, dim=512, 8 heads, 8 KV heads
- **3× MLP** with LeakyReLU(0.5)² activation
- **Partial RoPE**: 16/64 dims
- **Mixed int5/int6 GPTQ** + Hadamard rotation + zstd-22 compression
- **Late soft-round QAT**: Enabled when lr_scale < 0.15, alpha anneals 1→16
- **BigramHash**: 10240 buckets, dim=128
- **Value Embeddings**: Shared table, layers 8,9,10
- **XSA**: All 11 layers
- **LN Scale**: 1/sqrt(layer_idx+1)
- **EMA**: decay=0.997 + SWA every 40 steps
- **3% magnitude pruning**
- **7-expert LogisticContextMixer** (Hedge/AdaHedge algorithm):
  - Neural, Unigram, Bigram, Trigram(128K), Entropy, 4-gram(32K), Skip-bigram
- **Score-first TTT**: 262K chunk, 2 epochs, 3 blocks unfrozen, per-layer LR groups
- **Weight decay warmup**: Linear 0→0.04 over first 500 steps
- **Finer GPTQ grid**: 12 percentile thresholds + 256 calibration samples

### Improvements Over Winner

| # | Change | Winner | Ours | Expected BPB Gain |
|---|--------|--------|------|-------------------|
| 3 | Mixer experts | 5 (neural/uni/bi/tri/entropy) | 7 (+4gram, +skip-bigram) | 0.003-0.008 |
| 2 | TTT LR groups | Flat LR all params | Per-layer (3.0/1.5/0.5/1.0) | 0.002-0.005 |
| 8 | Hadamard rotation | None | Block-diagonal Hadamard before GPTQ | 0.002-0.005 |
| 5 | Soft-round QAT | Disabled | Enabled, threshold=0.15 | 0.002-0.004 |
| 1 | GPTQ grid | 5 percentiles | 12 percentiles + 256 samples | 0.001-0.003 |
| 4 | TTT chunks | 131K/3ep | 262K/2ep | 0.001-0.003 |
| 7 | TTT blocks | 2 unfrozen | 3 unfrozen | 0.001-0.003 |
| 6 | BigramHash/MLP | 6144/3.5x | 10240/3.0x | 0.001-0.002 |
| 10 | Mixer eta | Fixed 0.1 | AdaHedge adaptive | 0.001-0.002 |
| 9 | Weight decay | Constant 0.04 | Warmup 0→0.04 over 500 steps | 0.0005-0.001 |
| - | VE layers | 9,10 | 8,9,10 | ~0.001 |
| - | SWA frequency | every 50 | every 40 | ~0.0005 |
| - | Warmdown iters | 3500 | 3750 | ~0.0005 |
| - | Muon warmup | 1500 steps | 1750 steps | ~0.0005 |

---

## Quick Reference Commands

### Artifact Size Check
```bash
SIZE_ONLY=1 python train_gpt.py
```

### 1×H100 Training Only (no TTT, ~10 min)
```bash
RUN_ID=improved_v1 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TTT_ENABLED=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### 1×H100 With TTT (partial val, ~30 min)
```bash
RUN_ID=improved_v1_ttt \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
TTT_ENABLED=1 \
QUICK_VAL_FRAC=0.25 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

### Full 8×H100 Run (3 seeds)
```bash
for SEED in 1337 42 7; do
  RUN_ID=improved_final_s${SEED} \
  DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 SEED=$SEED \
  TTT_ENABLED=1 USE_MIXER=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
done
```

---

## Verification Checklist

1. **Artifact size**: `SIZE_ONLY=1 python train_gpt.py` → must be < 16,000,000 bytes
2. **Training runs**: Loss decreases on 1×H100, no NaN/crash
3. **Quant roundtrip**: int5/int6+Hadamard+zstd → decompress → dequant+inverse Hadamard → eval
4. **Mixer correctness**: 7 experts produce valid log-probs, weights update, no NaN
5. **TTT smoke test**: `TTT_ENABLED=1 QUICK_VAL_FRAC=0.1 TTT_EPOCHS=1` completes
6. **Per-layer LR**: Log confirms different LR per group during TTT
7. **Soft-round QAT**: Log shows QAT activation when scale < 0.15
8. **Weight decay warmup**: WD increases during first 500 steps
9. **SDPA fallback**: Works without flash_attn_interface on non-Hopper GPUs

---

## Key Dependencies

- `zstandard` Python package (for zstd compression)
- `flash_attn_interface` — only on Hopper GPUs (RunPod H100)
- `sentencepiece`, `torch`, `numpy`

## Critical Files
- `train_gpt.py` — All model + training + eval code (based on winner + 10 improvements)
- `winner_train_gpt.py` — Original winner code for reference
- `master_plan.md` — This file
- `data/cached_challenge_fineweb.py` — Download data shards
