# Running Parameter Golf on RunPod

Step-by-step guide for a fresh Ubuntu 22.04/24.04 environment (RunPod container).

---

## Quick Start (3 commands after SSH)

```bash
git clone -b runpod --single-branch https://github.com/HaardhikK/parameter-golf.git
cd parameter-golf
bash setup.sh
```

Then download data and train (see steps below).

---

## 1. Launch a RunPod Pod

### For the final leaderboard submission (8×H100 SXM)

1. Go to [console.runpod.io](https://console.runpod.io) → **GPU Cloud** → **+ New Pod**.
2. Select **8× H100 SXM** (not PCIe — must be SXM for NVLink bandwidth).
3. Deploy using the official Parameter Golf template:
   `https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th`
   - Enable **SSH terminal access**.
   - Set **Volume: 200 GB** (or attach a Network Volume for dataset persistence).
4. Once running, SSH in. You will land in `/workspace/`.

> Cost: ~$20/hr. A single 10-minute run costs ~$3.30. Run 3 seeds = ~$10 total.

---

## 2. Clone the Repository

```bash
cd /workspace
git clone -b runpod --single-branch https://github.com/HaardhikK/parameter-golf.git
cd parameter-golf
```

---

## 3. Run Setup

```bash
bash setup.sh
source .venv/bin/activate
```

This script:
- Installs system packages and Node.js
- Installs Claude Code (`claude auth login` to authenticate)
- Creates `.venv/` and installs Python deps from `requirements.txt`
- Installs `zstandard` (required for int6+zstd quantization)
- Checks for FlashAttention 3 (available on H100, auto-falls back to SDPA otherwise)

---

## 4. Download the Dataset

```bash
python3 data/cached_challenge_fineweb.py --variant sp1024
```

Downloads:
- Full validation split (`fineweb_val_*`) — required for scoring
- 80 training shards (~8B tokens)

Files land in `data/datasets/fineweb10B_sp1024/` and `data/tokenizers/`.

> **Tip**: Attach a RunPod **Network Volume** at `/workspace/data` to avoid re-downloading across pod restarts.

---

## 5. Verify Artifact Size (Before Training)

```bash
SIZE_ONLY=1 python train_gpt.py
```

Expected output:
- `model_params: ~27M`
- `compressor: zstd`
- `artifact_size: ~8-10 MB` (untrained; trained will be larger but should stay under 16 MB)
- `budget_remaining: positive`

---

## 6. Full Submission Run (8×H100, Seed 1337)

This is the exact command for the leaderboard submission run. Run it **3 times with different seeds** (1337, 42, 7) to satisfy the p<0.01 statistical significance requirement.

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

For seed 42: change `SEED=1337` → `SEED=42` and `RUN_ID=winner_arch_s42`.
For seed 7: change `SEED=1337` → `SEED=7` and `RUN_ID=winner_arch_s7`.

### Key metrics to watch during training:
- `step:0/20000 val_bpb:` — should be ~4.1 (random)
- `step:200/20000 val_bpb:` — should be dropping toward 2.5
- `stopping_early: wallclock_cap` — confirms 10-min cap was hit
- `DIAGNOSTIC post_avg val_bpb:` — pre-quantization BPB (target: ~1.14)
- `final_int6_roundtrip_exact val_bpb:` — post-quant BPB (target: ~1.15)
- `final_int6_sliding_window_exact val_bpb:` — **submission score** (target: ~1.12)
- `Total submission size int6+zstd:` — must be **< 16,000,000 bytes**

The log file is saved to `logs/{RUN_ID}.txt`.

---

## 7. Save Results for Submission

After each run completes, create the submission folder and save files:

```bash
SUBMISSION_DATE=$(date +%Y-%m-%d)
SUBMISSION_DIR="records/track_10min_16mb/${SUBMISSION_DATE}_WinnerArch_Int6_Zstd"
mkdir -p "$SUBMISSION_DIR"

# Copy the log (requires -f because *.log is gitignored globally)
cp logs/winner_arch_s1337.txt "$SUBMISSION_DIR/train.log"

# Copy the exact script used
cp train_gpt.py "$SUBMISSION_DIR/train_gpt.py"
```

Then fill in `submission.json` and `README.md` (see templates in the submission folder).

Commit and push:
```bash
git add -f "$SUBMISSION_DIR/train.log"   # -f needed due to *.log gitignore
git add "$SUBMISSION_DIR/"
git commit -m "Add submission: WinnerArch int6+zstd seed 1337"
git push origin runpod
```

---

## 8. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `SEED` | 1337 | Random seed |
| `RUN_ID` | auto UUID | Name for this run's log file |
| `MAX_WALLCLOCK_SECONDS` | 600 | 10-minute cap (set to `0` for unlimited) |
| `VAL_LOSS_EVERY` | 4000 | Validate every N steps (200 gives good curves) |
| `TRAIN_LOG_EVERY` | 500 | Log train loss every N steps |
| `SIZE_ONLY` | 0 | Set to `1` to measure artifact size only |
| `COMPILE_MODE` | fullgraph | `fullgraph`/`default`/`off` |
| `NUM_LAYERS` | 11 | Transformer layers |
| `MLP_MULT` | 3.0 | MLP hidden multiplier |
| `LATE_QAT_THRESHOLD` | 0.18 | Enable QAT when lr_scale drops below this |
| `SWA_EVERY` | 40 | SWA checkpoint every N steps |
| `EMA_DECAY` | 0.997 | EMA decay factor |
| `VE_LAYERS` | 8,9,10 | Value embedding layers |

---

## 9. Statistical Significance Requirement

The submission rules require **p < 0.01** that the improvement is ≥ 0.005 nats over SOTA.

- Current SOTA: **1.2244 BPB** (naive baseline)
- Target: **~1.12 BPB** (based on winner's architecture)
- Required improvement: ≥ 0.00721 BPB (= 0.005 nats / ln(2))
- Effect size is so large (~0.10 BPB) that even 1 seed would be significant, but run 3 seeds to be safe.

Run all 3 seeds before opening the PR.

---

## 10. Persisting Work Between Sessions

RunPod containers are **ephemeral** — filesystem is wiped when the pod stops.

```bash
# Before stopping the pod, commit and push:
git add -A
git commit -m "checkpoint: describe changes"
git push origin runpod
```

Model weights (`final_model.pt`, `final_model.int6.ptz`) are `.gitignore`d — they are regenerated by training. Mount a **Network Volume** to persist the dataset.

---

## 11. Using Claude Code on RunPod

```bash
claude auth login   # one-time auth
claude              # start Claude Code in the project root
```

---

## 12. Troubleshooting

**`torch` not found after `setup.sh`**
Run `source .venv/bin/activate`

**`flash_attn_interface` not found**
Expected on non-Hopper GPUs. The code auto-falls back to `F.scaled_dot_product_attention`.

**`zstandard` not found**
Run `pip install zstandard` (or re-run `bash setup.sh`).

**OOM on 8×H100**
Reduce `TRAIN_BATCH_TOKENS` or `TRAIN_SEQ_LEN`. Default (786,432 tokens, seq=2048) fits comfortably in 80 GB H100.

**`torchrun` not found**
Make sure venv is activated: `source .venv/bin/activate`

---

## File Layout Reference

```
parameter-golf/
├── setup.sh                  # One-command setup
├── info_runpod.md            # This file
├── train_gpt.py              # Main training script (winner arch + improvements)
├── explanation.md            # Detailed approach explanation
├── master_plan.md            # Optimization roadmap
├── requirements.txt          # Python dependencies (includes zstandard)
├── data/
│   ├── cached_challenge_fineweb.py   # Dataset downloader
│   └── tokenizer_specs.json
├── records/
│   └── track_10min_16mb/
│       ├── 2026-03-17_NaiveBaseline/   # Existing baseline (1.2244 BPB)
│       └── 2026-03-22_WinnerArch_Int6_Zstd/  # Our submission (template)
└── initial_info/             # Challenge rules, evaluation, insights
```
