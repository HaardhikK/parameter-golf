# Approach Explanation: Parameter Golf

## The Challenge

Train the best language model that fits in a 16 MB artifact, trainable in under 10 minutes on 8×H100 GPUs. Scored by bits-per-byte (BPB) on the FineWeb validation set — lower is better.

The challenge is fundamentally a **packing problem**: given a fixed byte budget, maximize model quality.

---

## The Key Insight: Compression is the Bottleneck

Most people's first instinct is to ask "how can I make a better architecture?" But the real question is "how many parameters can I fit in 16 MB?"

With naive float32 storage: 16 MB / 4 bytes = 4M parameters. That's tiny.
With int8 + zlib compression: ~17M parameters fit. That was our starting point.
With **int6 + zstd compression**: ~27M parameters fit. That's a 60% increase — for free.

The winning approach identified that the compression format was the real lever, not the architecture. Going from int8+zlib to int6+zstd gives ~0.59 bytes/parameter vs ~0.92 bytes/parameter — allowing 60% more parameters in the same budget. At this model scale, more parameters beats any architectural cleverness.

---

## Architecture

**11 independent transformer blocks**, 512-dim, 8 attention heads, 4 KV heads (GQA).

### Grouped Query Attention (GQA)
Standard multi-head attention stores one key/value head per query head. GQA shares KV heads across groups — 4 KV heads serving 8 query heads here. This cuts the KV parameter cost in half with minimal quality loss.

### 3× MLP with ReLU²
Each block's MLP expands to 3× the model dimension (hidden size = 1536) and uses ReLU² activation (apply ReLU then square the result). The squaring sharpens the activation — it emphasizes large activations and suppresses small ones more aggressively than regular ReLU, which empirically improves language modeling.

### Partial RoPE with NTK Scaling
Rotary position embeddings are applied to only 16 of 64 head dimensions (25%). The remaining dimensions are position-invariant, letting the model attend based on content alone when position doesn't matter. NTK scaling automatically adjusts the rotation frequencies when evaluating on sequences longer than the training length.

### U-Net Skip Connections
The 11 layers are split into 5 "encoder" layers and 6 "decoder" layers. Each decoder layer receives a skip connection from the corresponding encoder layer (like U-Net in image segmentation). This gives the model two paths of information flow — a deep path through all layers and a shortcut — which helps gradients flow and allows the model to combine low-level and high-level features.

### SmearGate
A learned sigmoid gate that blends each position's embedding with the previous position's embedding. Each dimension has its own gate weight, letting the model learn which features benefit from 1-token lookback context. This is essentially a very cheap local attention mechanism — no quadratic cost, just a single learned blend per dimension.

### BigramHash Embedding
A hash table of 2048 bigram buckets, embedded into 128 dimensions and projected into model space. For each position, the embedding is keyed on the XOR-hash of the current and previous token IDs. This gives the model cheap access to bigram statistics without adding a full bigram matrix (which would cost vocab² parameters).

### Value Embeddings
At layers 8, 9, and 10, the token identity is injected directly into the attention value vectors before the attention computation. This lets deep layers "remember" what token they started from, fighting the tendency of deep networks to lose token identity information. A single shared embedding table (dim=128 → KV dim) with per-layer learned scales keeps the parameter cost low.

### XSA (Cross-Self Attention Removal)
In the last 4 layers, after computing attention, the component of each output vector that lies in the direction of the corresponding value vector is subtracted out. This removes the "self-attention bias" — the tendency for a position to attend primarily to itself — encouraging the model to use its attention capacity for cross-token information.

### LN Scale Factor
Each block's RMSNorm output is scaled by `1/√(layer_index + 1)` before attention and MLP. This shrinks the norm of activations in deeper layers, which stabilizes training by preventing very deep layers from dominating the gradient signal.

### Logit Softcap
Final logits are passed through `30 × tanh(logit / 30)`. This bounds logits to ±30, preventing the model from becoming overconfident and improving training stability, especially early in training.

---

## Quantization: The Real Secret Weapon

After training in bfloat16, the weights are quantized to int6 before saving.

**Int6 quantization**: weights are rounded to 64 discrete levels (−32 to +31) using a per-row scale factor. The scale is stored as float16. This uses ~6.1 bits per weight instead of 16 bits — a 2.6× compression.

**Why int6 and not int8?** Int8 uses 256 levels. The extra precision sounds good, but those extra levels don't compress as efficiently with zstd. Int6 values (only 64 distinct levels) create much more repetition in the byte stream, which zstd's entropy coding exploits heavily.

**zstd level 22**: zstandard at maximum compression. On trained int6 weights it achieves ~0.59 bytes/parameter. For comparison, zlib on int8 gives ~0.92 bytes/parameter. This 36% difference in compression ratio is what unlocks 10M extra parameters.

**Mixed precision**: MLP and attention weights get int6. Embeddings get int8 (they need more precision since they're directly looked up, not multiplied through many layers). Small control tensors (scalars, scales) are kept in float32.

---

## Training: Reducing the Quantization Gap

The quantization gap is the BPB increase caused by quantizing from bfloat16 to int6. Minimizing this gap is as important as minimizing the pre-quantization BPB.

### Late Quantization-Aware Training (QAT)
For most of training, the model trains in full bfloat16. When the learning rate schedule decays below 18% of its peak (the warmdown phase), fake quantization is enabled: each weight matrix is quantized to int6 and immediately dequantized, and the model trains through this rounding. The straight-through estimator (STE) passes gradients through the rounding operation as if it were an identity. This teaches the model to be robust to int6 rounding — the weights cluster around values that survive quantization well.

Starting QAT at scale=0.18 (vs 0.15 in the prior record) gives ~20% more training steps under fake quantization, which further closes the gap.

### EMA + SWA Stacking
Two weight averaging mechanisms are stacked:

**EMA (Exponential Moving Average)**: After every training step, a running float32 average of all weights is maintained with decay=0.997. This is a smooth average that weights recent checkpoints more heavily.

**SWA (Stochastic Weight Averaging)**: During the warmdown phase (when LR scale < 0.2), snapshots of the EMA weights are collected every 40 steps. At the end of training, these snapshots are uniformly averaged.

The combination is powerful because EMA smooths out step-to-step noise, and SWA then averages across multiple points along the loss landscape's valley floor. The result is a flatter minimum that generalizes better and also quantizes more cleanly (flat minima are less sensitive to perturbations like rounding).

### Muon Optimizer for Matrices
Large weight matrices (attention projections, MLP weights) are optimized with Muon — a momentum-based optimizer that applies a Newton-Schulz orthogonalization to each gradient update before applying it. This effectively normalizes the update direction to be "maximally orthogonal" and unit-norm, which makes the effective learning rate uniform across all matrix parameters regardless of their scale. The Muon momentum warms up from 0.92 to 0.99 over 1750 steps for a smoother start.

**AdamW for everything else**: Embeddings and scalar parameters use AdamW with weight decay 0.04.

### Wallclock-Based Warmdown
Rather than fixed iteration counts, the warmdown is timed against the actual training clock. The learning rate decays linearly to zero over the last portion of the 10-minute budget. This means the model uses every available second productively regardless of how fast each step runs.

---

## Evaluation: Sliding Window BPB

The final score is computed with a sliding window evaluation at stride=64. Instead of scoring each token with only the preceding 2048 tokens of context, each token is scored with the maximum available context (up to 2048 tokens). For token position 100, it uses 100 tokens of context. For token position 5000, it uses the full 2048 tokens. This eliminates the "cold start" penalty from the beginning of each window and gives a fairer measure of the model's actual compression ability.

---

## Our Improvements Over the Prior Record

Starting from the record-setting architecture (1.1243 BPB), we made five conservative adjustments:

| Change | Prior Record | Our Approach | Reasoning |
|--------|-------------|--------------|-----------|
| Value embedding layers | 9, 10 | **8, 9, 10** | One more layer benefits from token identity injection |
| Late QAT threshold | scale < 0.15 | **scale < 0.18** | ~20% more QAT steps, further closes quant gap |
| SWA frequency | every 50 steps | **every 40 steps** | ~25% more averaging checkpoints, smoother final weights |
| Warmdown length | 3500 iters | **3750 iters** | Slightly longer convergence tail |
| Muon momentum warmup | 1500 steps | **1750 steps** | More gradual momentum ramp-up |

Each change is small enough that it's unlikely to hurt, and the combination may yield a small additional BPB reduction over the baseline.

---

## What Didn't Work (Prior Approach)

Before adopting the winning architecture, we tried a weight-shared depth-recurrent approach:
- 5 physical blocks repeated 3 times = 15 virtual layers at width 672
- The idea: same parameter count but more virtual depth → more compute per token

This reached 1.2244 BPB. The fundamental problem was that int8+zlib only allowed ~17M parameters — the weight-sharing trick couldn't compensate for having fewer parameters than the competing approaches that used better compression.

The lesson: **at this scale, the compression format determines how many parameters you can fit, and more parameters (from better compression) dominates architectural cleverness.**

---

## Hardware

- **Training**: 8× H100 SXM GPUs, FlashAttention 3 (Hopper-specific)
- **Local dev**: RTX 4050 (6 GB VRAM), PyTorch's scaled_dot_product_attention as FA3 fallback
- **Batch**: 786,432 tokens/step across 8 GPUs (grad accum × 1 per GPU)
- **Sequence length**: 2048 tokens
