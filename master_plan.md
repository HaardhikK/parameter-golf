# Parameter Golf Optimization Master Plan

## Context

**Challenge**: Train the best LM fitting in 16,000,000-byte artifact, trainable in <10 min on 8×H100. Metric: BPB on FineWeb validation set (lower = better).
**Timeline**: March 18 – April 30, 2026. Compute credits: $1M via RunPod.
**Submission**: GitHub PR to `records/track_10min_16mb/` with `train_gpt.py`, `train.log`, `submission.json`. Must beat SOTA by **≥0.005 nats, p<0.01**.

**Current Baseline** (10-min run on 8×H100):
- Architecture: 9 blocks, dim=512, 8 heads, 4 KV heads, 2× MLP, vocab 1024, tied embeddings
- ~17M params, post-quant BPB: **1.2244**
- Artifact: 15,863,489 bytes (**only 136 KB headroom!**)
- Int8 payload: 17,178,912 bytes, zlib compresses ~8.2% (random-looking int8 data)
- zlib compression gives ~0.918 ratio on trained int8 weights

**Reference** (4-hour unlimited run, same arch):
- Pre-quant BPB: 1.1749, Post-quant BPB: 1.2074
- **Quantization gap: 0.0325 BPB** — huge degradation target

**Sizing formula** (empirically verified):
`artifact_bytes ≈ (param_count + 225K) × 0.918 + 48K`
Max params fitting in 16 MB: ~17.15M. We are already at the ceiling.

**Cloud setup**: RunPod 1×H100 80GB, 80 training shards, ~635 steps in 10 min at 945ms/step.

---

## Key Strategic Insight

We cannot add more parameters — the budget is fully used. The strategy is:
1. **Depth recurrence**: Use fewer unique blocks repeated multiple times → same param count but more virtual depth AND wider per-block → fundamentally more capable model
2. **Reduce quantization gap**: QAT to bridge the 0.03 BPB degradation
3. **Zero-cost architecture tweaks**: Value residuals, smear module — add expressiveness without adding params
4. **Training optimization**: Better optimizer, schedule, data diversity

---

## Implementation Status

### Phase 0: Infrastructure ✅ DONE
- `update.md` experiment ledger created
- `SIZE_ONLY=1` mode: instantiates model, quantizes, compresses, prints artifact size
- `COMPILE_MODE` env var (fullgraph/default/off) for local testing
- 10 training shards downloaded

### Phase 1: Depth Recurrence

#### Step 1.1: Implement Weight-Shared Blocks ✅ DONE
Physical blocks contain: `CausalSelfAttention`, `MLP` — the large matrices.
Virtual layer params stored in `nn.ParameterList`: `attn_scales`, `mlp_scales`, `resid_mixes`, `q_gains`.
`GPT.forward` loops over virtual layers, using `physical_blocks[i % num_physical]` for compute.

#### Step 1.2: Config B ✅ DONE
- 5 physical blocks × 3 = 15 virtual layers, dim=672, 12 heads, 6 KV heads (head_dim=56)
- **16,548,852 params**, estimated trained artifact ~14.7 MB (~1.3 MB headroom)
- Verified: untrained artifact = 4.87 MB, loss decreases 6.95→6.54 in 4 steps, memory 1037 MiB

#### Step 1.3: Hyperparameter Tuning ✅ DONE
Tuned on 1×H100 via 2-min sweeps validated with full 10-min run.
- **matrix_lr**: 0.04→0.08 (higher works due to Muon gradient normalization)
- **tied_embed_lr**: 0.05→0.02 (lower is better for tied embeddings)
- **warmdown_iters**: 1200→200 (1200 was catastrophic — LR never reached full value)
- **scalar_lr**: 0.04 (kept default, not yet swept)

### Phase 2: Quantization-Aware Training

#### Step 2.1: STE QAT ✅ DONE
Always-on in `CastedLinear.forward` — STE pattern `w + (w_q - w).detach()`.
`torch.compile` compatible (standard ops only).

#### Step 2.2: Entropy Regularization 🟡 DEFERRED
Add small penalty encouraging weight clustering (→ better zlib compression).
Only if headroom becomes tight.

### Phase 3: Zero-Cost Architecture Improvements

#### Step 3.1: Value Residuals ✅ DONE (Option B)
x0 skip after attention with zero-init per-layer scalar alpha. No matrix needed.
Combined with back-out mechanism (Step Exp D).

#### Step 3.2: Smear Module ✅ DONE
Per-virtual-layer causal 1-token lookback: `F.pad(x[:,:-1,:], (0,0,1,0))`.
Avoids data leakage (causal, not `torch.roll`).

#### Step 3.3: Partial RoPE ✅ DONE
50% of head dims (28 of 56) get RoPE; rest are position-invariant.

### Phase 4: Training Optimization

#### Step 4.1: NorMuon Optimizer 🔴 TODO
After Newton-Schulz step, normalize each output neuron's update magnitude proportionally to that neuron's current weight norm → maintains uniform neuron conditioning.
Add as optional path in Muon.step, controlled by env var.
Expected: 0.005–0.01 BPB

#### Step 4.2: Heterogeneous Embedding Updates ✅ DONE
Accumulate grad for 2 steps, only step/zero-grad tok_embed every 2nd step.
**Bug fix**: must NOT zero tok grad before forward — only zero after stepping.

#### Step 4.3: Gradient Clipping ✅ DONE
Default now 1.0 (was 0.0 / disabled).

### Phase 5: Advanced Techniques

#### Step 5.1: Long-Short Attention Windows 🔴 TODO
2–3 global heads + remaining local (128-token window) using FlexAttention.
Saves attention FLOPs → more steps in 10 minutes.
Risk: torch.compile + FlexAttention compatibility needs testing.
Expected: 0.005–0.015 BPB (speed gain)

#### Step 5.2: Test-Time Longer Context 🔴 TODO
At eval time use 2048-token sequences with RoPE NTK scaling.
Expected: 0.005–0.02 BPB (uncertain)

#### Step 5.3: Better Compression (zstd) 🟡 DEFERRED
Replace zlib with zstandard. Marginal on random int8 data.
Risk: must verify zstd availability on RunPod evaluation environment.

---

## New Experimental Ideas

### Exp A: SwiGLU MLP ✅ DONE
Replaced ReLU² with SwiGLU: gate(672→896) + fc(672→896) + proj(896→672) = same 1,806,336 params.
Part of EXP-002 combined result: 0.027 BPB improvement.

### Exp B: RMSNorm With Learned Scale 🔴 TODO
Current `RMSNorm` has no learned parameters. Adding per-dim scale adds expressiveness.
Cost: dim params per norm call (negligible).
With weight sharing, norms inside PhysicalBlock are shared — per-virtual-layer norms would go in virtual param lists.

### Exp C: ILP Architecture Search 🔴 TODO
Use `scipy.optimize.milp` or `PuLP` to solve:
"Find (P, V, D, H, K) that maximizes proxy quality score SUBJECT TO artifact_size(P,V,D,H,K) ≤ 16MB."
Proxy quality: `C × (total_flops_per_step) × (steps_in_10_min)` — calibrate from cloud runs.
Pure Python script, no changes to train_gpt.py.

### Exp D: Back-Out Mechanism ✅ DONE
At end of each virtual layer: `x = x + backout_weight * x_pre`
Zero-init scalar per layer. Part of EXP-002 combined result.

### Exp E: EoS-Aligned Batching 🔴 TODO
Align training sequence boundaries with End-of-Sequence tokens to avoid cross-document attention bleeding.
Small but free quality improvement. Needs changes to `DistributedTokenLoader`.

---

## README / Rules Points Not To Miss

1. **Statistical significance**: Must beat SOTA by ≥0.005 nats with p<0.01 — run multiple seeds if needed
2. **submission.json** required — contains run_id, BPB, git hash, timestamp
3. **No external network calls during evaluation** — submission must be fully self-contained
4. **PR format**: Goes to `records/track_10min_16mb/YYYY-MM-DD_Description/` with train.log, train_gpt.py, submission.json
5. **Artifact = code + compressed weights**: `code_bytes` = UTF-8 encoded train_gpt.py size — keep code lean (limit ~1500 lines)
6. **Evaluation runs 10+10 minutes**: 10-min train + 10-min eval. Eval must complete within its window
7. **Back-out mechanism** and **EoS-aligned batching**: documented speedrun wins not yet implemented
8. **Attention sinks/gating**: may help with recurrent arch stability

---

## Local Smoke Test Protocol

For every change:
```bash
# Quick correctness check (no compile, tiny model):
ITERATIONS=4 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=65536 WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=256 COMPILE_MODE=off \
NUM_LAYERS=3 NUM_PHYSICAL_BLOCKS=1 MODEL_DIM=128 NUM_HEADS=4 NUM_KV_HEADS=2 \
.venv/bin/python train_gpt.py

# Artifact size check (full model, no training):
SIZE_ONLY=1 .venv/bin/python train_gpt.py

# Full model correctness (with compile):
ITERATIONS=2 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=65536 WARMUP_STEPS=0 \
TRAIN_BATCH_TOKENS=16384 TRAIN_SEQ_LEN=256 COMPILE_MODE=fullgraph \
.venv/bin/python train_gpt.py
```

**Check**: (1) loss decreases, (2) artifact size < 16 MB, (3) compile succeeds, (4) roundtrip quant works

---

## Implementation Progress Tracker

| Phase | Step | Status | BPB Impact | Notes |
|---|---|---|---|---|
| 0 | Infrastructure | ✅ Done | — | SIZE_ONLY, 10 shards, update.md |
| 1.1 | Weight sharing | ✅ Done | Enabler | PhysicalBlock + ParameterList |
| 1.2 | Config B | ✅ Done | 0.02–0.04? | dim=672, 15 virtual, ~14.7 MB est. |
| 1.3 | HP tuning | ✅ Done | 0.027 combined | matrix_lr=0.08, embed_lr=0.02, warmdown=200 |
| 2.1 | STE QAT | ✅ Done | 0.01–0.03? | Always-on, STE pattern |
| 2.2 | Entropy reg | 🟡 Deferred | small | Only if tight on space |
| 3.1 | Value residuals | ✅ Done | part of 0.027 | Option B: x0 skip, zero-init alpha |
| 3.2 | Smear module | ✅ Done | 0.003–0.008? | Per-virtual-layer, causal |
| 3.3 | Partial RoPE | ✅ Done | 0.002–0.005? | 50% dims, no params |
| 4.1 | NorMuon | 🔴 TODO | 0.005–0.01 | Research impl first |
| 4.2 | Hetero updates | ✅ Done | 0.002–0.005? | Fixed zero-grad bug |
| 4.3 | Grad clipping | ✅ Done | stability | Default 1.0 |
| 5.1 | Long-short attn | 🔴 TODO | 0.005–0.015 | FlexAttention risk |
| 5.2 | Test-time ctx | 🔴 TODO | 0.005–0.02 | RoPE NTK scaling |
| 5.3 | zstd | 🟡 Deferred | small | Check RunPod availability |
| Exp A | SwiGLU MLP | ✅ Done | part of 0.027 | Same param count, 2/3 hidden |
| Exp B | RMSNorm+scale | 🔴 TODO | small | Per-virtual-layer norms |
| Exp C | ILP arch search | 🔴 TODO | enabler | Pure Python, scipy.optimize |
| Exp D | Back-out | ✅ Done | part of 0.027 | 1 scalar per virtual layer |
| Exp E | EoS batching | 🔴 TODO | small | Data pipeline change |
| — | submission.json | 🔴 TODO | required | Needed for final PR |

**Realistic combined target**: 0.04–0.08 BPB improvement → final BPB of **1.14–1.18** (vs baseline 1.2244)

---

## AutoResearch Integration (Phase 6)

### What AutoResearch Is
Andrej Karpathy's [AutoResearch](https://github.com/karpathy/autoresearch) is an open-source agentic system that automates the ML research loop: reading literature, generating hypotheses, editing training code, running short experiments, analyzing metrics, and iterating — independently. It uses Claude Sonnet/Opus as reasoning agents and repeatedly modifies a NanoGPT-style trainer while running fixed-length (e.g., 5-minute) runs to search architecture and hyperparameter space.

This is **conceptually aligned** with Parameter Golf: both involve short repeated training runs on a small model to optimize bits-per-byte under a resource constraint.

### How to Use AutoResearch with Parameter Golf

**Step 1 — Adapt the domain code**
Point AutoResearch's editable training script at a trimmed version of `train_gpt.py` (or the full file). The key constraint: any edit it proposes must still satisfy the 16 MB artifact limit. Add a pre-run check: `SIZE_ONLY=1 python train_gpt.py` — reject edits that exceed the limit.

**Step 2 — Redefine the objective in `program.md`**
Instead of validating on Shakespeare/nanochat, define the goal as:
> "Minimize FineWeb validation BPB. The trained model + code must compress to ≤16,000,000 bytes. Training must complete in ≤10 minutes on 8×H100."
Include the sizing formula and the `SIZE_ONLY=1` check command so the agent can self-validate.

**Step 3 — Configure short local runs for cheap iteration**
Use reduced-data, shorter-wallclock runs that still correlate with full-run BPB:
```bash
ITERATIONS=500 VAL_LOSS_EVERY=100 TRAIN_SEQ_LEN=512 \
TRAIN_BATCH_TOKENS=65536 NUM_SHARDS=2 .venv/bin/python train_gpt.py
```
Run on RTX 4050 locally (~2–3 min) to generate signal before invoking 8×H100.

**Step 4 — Candidate curation workflow**
1. AutoResearch generates N candidate variants (arch changes, LR schedules, etc.)
2. Filter: any that fail `SIZE_ONLY=1` are discarded immediately
3. Run short local experiments (~2 min) to rank survivors by proxy BPB
4. Run top 2–3 on full 8×H100 for final BPB numbers
5. Record all results in `update.md`

**Step 5 — Evaluation-time independence**
AutoResearch uses external Claude API calls — use it ONLY during development. The official submission artifact must be fully self-contained (no network calls during eval). AutoResearch is a development tool, not part of the submission.

### What to Give AutoResearch to Search
Good search targets for AutoResearch to explore autonomously:
- LR schedules: matrix_lr, embed_lr, warmdown shape
- MLP variant: ReLU² vs SwiGLU vs GeGLU (same param budget)
- Virtual depth: P=4,V=16 vs P=5,V=15 vs P=6,V=12 etc.
- Smear module: per-dim vs scalar, 1-token vs 2-token lookback
- RoPE fraction: 25% vs 50% vs 75% of head dims
- QAT: always-on vs delayed to final N steps

### Important: Still Do Manual Research for Specific Techniques
For well-defined techniques (NorMuon, FlexAttention), do NOT rely on AutoResearch to figure out the implementation — research it manually first:
- NorMuon: fetch Keller Jordan's modded-nanogpt repo for exact implementation
- FlexAttention: check PyTorch docs and fullgraph=True compatibility
- Read `initial_info/nanogpt_insights.md` first for any local notes on these

---

## Critical Files
- `train_gpt.py` — All model code (~1232 lines, limit 1500)
- `update.md` — Experiment ledger
- `records/track_10min_16mb/` — Submission folder
- `data/cached_challenge_fineweb.py` — Download more shards
- `initial_info/nanogpt_insights.md` — Speedrun techniques reference
- `initial_info/evaluation.md` — Official rules
- `records/track_10min_16mb/2026-03-17_NaiveBaseline/train.log` — Baseline convergence curve

---

## Starting Prompt for New Session

```
About this project: I'm participating in the OpenAI Parameter Golf challenge
(March–April 2026). Challenge details are in initial_info/challenge.md.
The rules are in initial_info/evaluation.md. Inspiration from speedruns
is in initial_info/nanogpt_insights.md.

The goal: train the best language model fitting in 16MB artifact, trainable
in <10 min on 8×H100s. Metric: BPB on FineWeb validation set (lower = better).
Current SOTA (baseline): 1.2244 BPB. Need to beat by ≥0.005 nats.

What's been done so far (all in train_gpt.py):
1. Weight-shared depth recurrence: 5 physical blocks × 3 = 15 virtual layers,
   dim=672, 12 heads, 6 KV heads, 16.55M params (~14.7 MB trained artifact)
2. STE QAT: always-on fake quantization in CastedLinear, directly attacks the
   0.0325 BPB quantization gap
3. Smear module: causal 1-token lookback per virtual layer (free local context)
4. Partial RoPE: 50% of head dims get rotary, rest position-invariant
5. Heterogeneous embedding updates: grad accumulates 2 steps for embeddings
6. Grad clipping enabled (1.0), 10 training shards downloaded
7. SIZE_ONLY=1 mode for instant artifact size check
8. COMPILE_MODE env var (fullgraph/default/off) for local testing

What failed / gotchas:
- Rotary caching caused inference-mode tensor bug (removed caching)
- Heterogeneous updates: must NOT zero embedding grad before forward,
  only after stepping (fixed)
- fullgraph=True compilation takes ~3 min on RTX 4050 (fast on H100)

What still needs doing (priority order):
1. HP tuning for new arch on 8×H100 (matrix_lr, embed_lr, warmdown_iters)
2. Value residuals (need shared W_v0 projection or simpler skip variant)
3. NorMuon optimizer
4. SwiGLU MLPs (same param count as ReLU², potentially better)
5. Long-short attention windows (FlexAttention)
6. ILP architecture search (scipy.optimize.milp to find optimal config)
7. Back-out mechanism, EoS-aligned batching

The experiment ledger is in update.md. Memory files are in
~/.claude/projects/-home-kunder-parameter-golf/memory/

Baseline run logs: records/track_10min_16mb/2026-03-17_NaiveBaseline/
Local setup: RTX 4050 5GB VRAM, WSL2 Ubuntu 24.04, Python 3.13, PyTorch 2.10
Cloud: RunPod 8×H100, use torchrun for distributed training
```
