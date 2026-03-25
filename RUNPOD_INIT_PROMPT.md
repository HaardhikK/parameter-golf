# RunPod Session Init Prompt

Copy-paste everything inside the triple-backtick block below as your first message to the new Claude session on RunPod.

---

```
I'm running the Parameter Golf challenge (OpenAI Model Craft Challenge, March–April 2026).
Goal: train the best language model fitting in a 16MB artifact, trainable in <10 min on 8×H100.
Metric: bits-per-byte (BPB) on FineWeb validation set. Lower = better.
Current SOTA in the repo: 1.2244 BPB (naive baseline).

You are on a fresh RunPod 8×H100 SXM pod. Do NOT do any testing or experimentation.
Run the full 10-minute submission run directly, save the output, and prepare the submission files.

## Step 0 — Setup

```bash
cd /workspace
git clone -b runpod --single-branch https://github.com/HaardhikK/parameter-golf.git
cd parameter-golf
bash setup.sh
above part done
continue from here :
source .venv/bin/activate
python3 data/cached_challenge_fineweb.py --variant sp1024
NOTE: FA3 MUST BE INSTALLED IF NOT, ITS VERY IMPORTANT 
```

The data download gets the full validation split + 80 training shards (~8B tokens).
It takes ~10–20 min. Wait for it to fully complete before training.

## What train_gpt.py does

Implements the winning architecture (~1.1243 BPB community reference):
- 11 independent transformer layers, dim=512, 8 heads, 4 KV heads (GQA), 3× MLP ReLU²
- int6+zstd compression: ~27M params fits in ~15MB trained artifact
- Late QAT (lr_scale < 0.18), EMA(0.997) + SWA(every 40 steps), FA3 on H100
- Sliding window eval stride=64 is the submission score
- 5 improvements over the prior record: VE layers 8,9,10, QAT 0.18, SWA every 40,
  warmdown 3750 iters, Muon momentum warmup 1750 steps

## Step 1 — Run seed 1337 (primary submission)

```bash
NCCL_IB_DISABLE=1 \
SEED=1337 \
RUN_ID=winner_arch_s1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
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

Log saves to: logs/winner_arch_s1337.txt

## Step 2 — Run seed 42

Same command, change only:
- SEED=42
- RUN_ID=winner_arch_s42

## Step 3 — Run seed 7

Same command, change only:
- SEED=7
- RUN_ID=winner_arch_s7

We need 3 seeds to satisfy p<0.01 statistical significance (required by submission rules).

## Step 4 — Extract key metrics from each log

For each log file, find:
- `attention_backend:` → should say FA3 (not SDPA) on H100
- `stopping_early: wallclock_cap ... step:XXXX/20000` → how many steps
- `DIAGNOSTIC post_avg val_bpb:X.XXXX` → pre-quantization BPB
- `final_int6_roundtrip_exact val_loss:X val_bpb:X` → post-quant BPB and loss
- `final_int6_sliding_window_exact val_bpb:X.XXXXXXXX` → SUBMISSION SCORE
- `Total submission size int6+zstd: XXXXXXXX bytes` → MUST be < 16,000,000

If any seed has artifact > 16,000,000 bytes: STOP and tell me immediately.

## Step 5 — Save submission files

```bash
SUBMISSION_DIR="records/track_10min_16mb/2026-03-22_WinnerArch_Int6_Zstd"

# Copy seed 1337 log (use -f to bypass *.log gitignore)
cp logs/winner_arch_s1337.txt "$SUBMISSION_DIR/train.log"

# Copy the exact training script
cp train_gpt.py "$SUBMISSION_DIR/train_gpt.py"
```

Edit $SUBMISSION_DIR/submission.json — fill in every FILL_IN field:
- "date": current UTC time in ISO 8601 (e.g. "2026-03-22T15:30:00Z")
- "val_bpb": sliding window BPB from seed 1337
- "val_loss": val_loss from final_int6_roundtrip line for seed 1337
- "bytes_total": total submission size in bytes from seed 1337 log
- "bytes_code": code_size in bytes from the log
- "seed_bpbs": [seed1337_bpb, seed42_bpb, seed7_bpb]

Edit $SUBMISSION_DIR/README.md — fill in all FILL_IN placeholders with actual numbers.

## Step 6 — Commit to our fork

```bash
git add -f "$SUBMISSION_DIR/train.log"   # -f needed due to *.log gitignore
git add "$SUBMISSION_DIR/"
git commit -m "Add submission: WinnerArch int6+zstd seed 1337 BPB=X.XXXX"
git push origin runpod
```

## Step 7 — Create a clean PR branch for openai/parameter-golf

The PR to openai/parameter-golf must ONLY add the records folder — not our other changes.
Create a dedicated branch based on upstream main:

```bash
git remote add upstream https://github.com/openai/parameter-golf.git
git fetch upstream
git checkout -b submission/winner-arch-int6-zstd upstream/main
git checkout runpod -- records/track_10min_16mb/2026-03-22_WinnerArch_Int6_Zstd/
git add records/track_10min_16mb/2026-03-22_WinnerArch_Int6_Zstd/
git commit -m "Add submission: WinnerArch int6+zstd, BPB=X.XXXX (3 seeds, p<0.01)"
git push origin submission/winner-arch-int6-zstd
```

Then open the PR from HaardhikK/parameter-golf:submission/winner-arch-int6-zstd → openai/parameter-golf:main.

## Key checks before opening PR

- [ ] All 3 seed artifacts < 16,000,000 bytes
- [ ] attention_backend:FA3 (not SDPA) in the log — confirms H100 FA3 was used
- [ ] train.log, train_gpt.py, submission.json, README.md all present in submission folder
- [ ] submission.json has no FILL_IN placeholders remaining
- [ ] The PR diff shows ONLY the new records folder (nothing else)

## After all done, tell me:
1. The 3 seed sliding window BPBs
2. Artifact size for each seed
3. Steps completed per seed (should be ~7000 on 8×H100)
4. Whether FA3 or SDPA was used
5. The PR URL once opened
```
