# RunPod Session Init Prompt

Copy-paste the block below as your first message to the new Claude session on RunPod.

---

## PROMPT TO SEND

```
I'm running the Parameter Golf challenge (OpenAI Model Craft Challenge, March–April 2026).
Goal: train the best language model fitting in a 16MB artifact, trainable in <10 min on 8×H100.
Metric: bits-per-byte (BPB) on FineWeb validation set. Lower = better.
Current SOTA in the repo: 1.2244 BPB (naive baseline).

You are on a fresh RunPod 8×H100 SXM pod. Do NOT do any testing or experimentation.
Run the full 10-minute submission run directly, save the output, and prepare the submission files.

## Setup

The repo is at https://github.com/HaardhikK/parameter-golf (branch: runpod).

Run this to set up:
```bash
cd /workspace
git clone -b runpod --single-branch https://github.com/HaardhikK/parameter-golf.git
cd parameter-golf
bash setup.sh
source .venv/bin/activate
python3 data/cached_challenge_fineweb.py --variant sp1024
```

## What the code does

train_gpt.py implements the winning architecture (~1.1243 BPB reference):
- 11 independent transformer layers, dim=512, 8 heads, 4 KV heads (GQA), 3× MLP ReLU²
- int6+zstd compression: MLP/attn → int6 per-row, embeddings → int8, zstd level 22
  (allows ~27M params in ~15MB trained artifact vs the old 17M with int8+zlib)
- Late QAT: STE int6 fake-quantization enabled when lr_scale < 0.18
- EMA(0.997) + SWA(every 40 steps) stacked weight averaging
- FA3 on H100 (auto-detected), SDPA fallback otherwise
- Sliding window eval at stride=64 (the submission score)
- Our 5 improvements over the prior record: VE layers 8,9,10, QAT 0.18, SWA every 40,
  warmdown 3750 iters, Muon momentum warmup 1750 steps

See explanation.md for full architectural details.

## What you need to do

### Step 1 — Run seed 1337 (the primary submission)

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
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The log saves to logs/winner_arch_s1337.txt automatically.

### Step 2 — Run seed 42 (for statistical significance p<0.01)

Same command but:
- SEED=42
- RUN_ID=winner_arch_s42

### Step 3 — Run seed 7 (third seed)

Same command but:
- SEED=7
- RUN_ID=winner_arch_s7

### Step 4 — Save submission files

After all 3 runs complete, record the results. For each run, find these lines in the log:
- `final_int6_sliding_window_exact val_bpb:X.XXXXXXXX` ← SUBMISSION SCORE
- `final_int6_roundtrip_exact val_bpb:X.XXXXXXXX` ← post-quant BPB
- `DIAGNOSTIC post_avg val_bpb:X.XXXX` ← pre-quant BPB
- `Total submission size int6+zstd: XXXXXXXX bytes` ← must be < 16,000,000
- `stopping_early: wallclock_cap ... step:XXXX/20000` ← how many steps completed

Then:

```bash
SUBMISSION_DIR="records/track_10min_16mb/2026-03-22_WinnerArch_Int6_Zstd"

# Copy the log (use -f because *.log is gitignored globally)
cp logs/winner_arch_s1337.txt "$SUBMISSION_DIR/train.log"

# Copy the exact script
cp train_gpt.py "$SUBMISSION_DIR/train_gpt.py"
```

Update the submission files:

1. **$SUBMISSION_DIR/submission.json** — fill in:
   - "date": current UTC datetime in ISO 8601
   - "val_bpb": sliding window BPB from seed 1337 (the best/primary result)
   - "val_loss": the val_loss from final_int6_roundtrip line for seed 1337
   - "bytes_total": the total submission size in bytes
   - "bytes_code": the code_size in bytes (from the log)
   - "seed_bpbs": [seed1337_bpb, seed42_bpb, seed7_bpb]

2. **$SUBMISSION_DIR/README.md** — fill in the FILL_IN placeholders with actual numbers.

### Step 5 — Commit and push

```bash
git add -f "$SUBMISSION_DIR/train.log"   # -f needed due to *.log gitignore
git add "$SUBMISSION_DIR/"
git commit -m "Add submission: WinnerArch int6+zstd, BPB=X.XXXX (3 seeds)"
git push origin runpod
```

## Key things to check

- `Total submission size int6+zstd` must be **< 16,000,000 bytes** for every seed
- `final_int6_sliding_window_exact val_bpb` is the actual submission score
- If artifact > 16MB: do NOT submit, report back
- If the flash_attn_interface import fails: that's fine, SDPA fallback kicks in automatically
  (but H100s should have it — check log for `attention_backend:FA3`)

## Statistical significance note

The submission rules require p<0.01 that improvement is ≥0.005 nats over SOTA (1.2244 BPB).
We expect ~1.12 BPB — a ~0.10 BPB improvement. With std ~0.001 across seeds, 3 seeds gives
overwhelming statistical significance. Report all 3 seed BPBs in README.md.

## After pushing

Tell me:
1. The 3 seed BPBs (sliding window)
2. The artifact size for each
3. The number of steps completed (wallclock usually stops around 7000 steps on 8×H100)
4. The git commit hash

Then we'll open the PR to openai/parameter-golf to submit to the leaderboard.
```
