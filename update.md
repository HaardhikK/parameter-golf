# Experiment Ledger

## Format
Each entry: **Date | Experiment | Goal | Changes | Artifact Size | BPB | Lessons**

---

## Baseline (2026-03-17)
- **Arch**: 9 blocks, dim=512, 8 heads, 4 KV heads, 2x MLP, vocab 1024, tied embeddings
- **Params**: ~17M
- **Artifact**: 15,863,489 bytes (136 KB headroom)
- **Post-quant BPB**: 1.2244
- **Quantization gap**: 0.0325 BPB

---

## Experiments

### EXP-001: Weight Sharing + Config B + QAT + Architectural Improvements
- **Date**: 2026-03-19
- **Goal**: Implement depth recurrence, QAT, smear, partial RoPE, grad clipping
- **Changes**:
  - Weight-shared blocks: 5 physical blocks x 3 = 15 virtual layers
  - Config B: dim=672, 12 heads, 6 KV heads (head_dim=56)
  - STE QAT on all CastedLinear layers (always-on, torch.compile compatible)
  - Smear module (1-token lookback per virtual layer, init to zero)
  - Partial RoPE (50% of head dims = 28 dims rotated, 28 pass-through)
  - Gradient clipping enabled (norm=1.0, was 0.0)
  - Heterogeneous embedding updates (every 2 steps)
  - SIZE_ONLY mode for quick artifact size measurement
  - COMPILE_MODE env var (fullgraph/default/off)
  - Rotary: removed caching to fix inference-mode tensor issue
  - 10 training shards downloaded (was 1)
- **Params**: 16,548,852
- **Artifact Size (untrained)**: 4.87 MB (estimated ~14.7 MB trained, ~1.3 MB headroom)
- **BPB**: Awaiting cloud run on 8xH100
- **Local verification**:
  - SIZE_ONLY mode: works, reports correct sizes
  - torch.compile fullgraph=True: works with full model
  - 4-step training: loss decreases 6.95 → 6.54 (correct)
  - Quantization roundtrip: works (val_bpb gap ~0.05 on untrained)
  - Memory: 1037 MiB (fits RTX 4050 with room to spare)
- **Lessons**:
  - Rotary caching causes inference tensor issues when first forward is during validation (WARMUP_STEPS=0)
  - Heterogeneous updates: must NOT zero tok grad before forward, only after stepping
  - fullgraph=True compilation time scales with virtual layers (~3 min for 15 layers on RTX 4050)

---

## Session Summary (2026-03-20)

### Local Verification Results (RTX 4050, WSL2)
- `SIZE_ONLY=1` reports untrained artifact = **4.87 MB**; estimated trained ≈ **14.7 MB** (~1.3 MB headroom)
- `torch.compile fullgraph=True`: compiles cleanly with full 16.55M model (~3 min on RTX 4050)
- 4-step training: loss 6.95 → 6.54 (loss decreasing correctly)
- Memory: **1037 MiB** VRAM on RTX 4050 (ample headroom)
- Roundtrip quant (int8 → zlib): works end-to-end, val_bpb gap ~0.05 on untrained model

### Bugs Found and Fixed
1. **Rotary caching bug**: Cached cos/sin buffers were registered as tensors in inference mode during the first validation forward pass, then reused during training — caused "inference-mode tensor in autograd graph" error. Fix: removed caching entirely, recompute per-step (negligible overhead).
2. **Heterogeneous updates zero-grad bug**: Original code zeroed `tok_embed.grad` before the forward pass in the off-step branches. This erased gradients from accumulation. Fix: only zero and step tok_embed optimizer every 2nd step, never zero grad before forward.

### Key Architectural Decisions Locked In
- Config B: dim=672, 12 heads, 6 KV heads, 5 phys blocks × 3 virtual layers = 15 total
- QAT is always-on (not delayed to final 20%) — simpler and compile-compatible
- Smear uses causal 1-token lookback: `F.pad(x[:,:-1,:], (0,0,1,0))` avoids data leakage

### What Was NOT Implemented (and why)
- **Value residuals (Phase 3.1)**: With GQA, kv_dim (336) ≠ model_dim (672); a per-block projection adds ~226K params — not negligible. Options: shared W_v0 across all blocks (Option A) or x0 skip after attention (Option B). Left for next session.
- **NorMuon (Phase 4.1)**: Implementation details unclear, no authoritative reference found at time of coding. Needs research.
- **Phases 5+**: Lower priority, not attempted.

### Next Steps (Priority Order)
1. **HP tuning for Config B** — LRs were set for old 9-block dim=512 arch; new wider recurrent arch almost certainly needs retuning. Run 2–3 min cloud sweeps of matrix_lr (try 0.01–0.03), tied_embed_lr (try 0.02–0.08), warmdown_iters (scale to ~5000–7000 steps).
2. **Value residuals** — Pick Option A (shared W_v0) or Option B (x0 skip), implement, test locally.
3. **NorMuon** — Research modded-nanogpt repo for exact implementation, then add as env-var-gated path.
4. **SwiGLU MLP** — Same param count as current ReLU², potentially better. Ablate against Config B baseline.
5. **ILP architecture search** — Use scipy.optimize.milp to exhaustively search (P, V, D, H, K) space under the 16 MB constraint.
6. **Long-short attention** (FlexAttention), **back-out mechanism**, **EoS batching**.
