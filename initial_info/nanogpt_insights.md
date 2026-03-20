# NanoGPT Speedrun Insights (for Parameter Golf)

Here is a consolidated list of NanoGPT Speedrun “winner” insights, including Muon, NorMuon, Polar Express, smear, etc., phrased so you can see how they might matter for Parameter Golf.

---

## 1. Architecture and activations

- Modernized GPT‑style block: rotary positional embeddings (RoPE), QK‑Norm on attention inputs, and ReLU² MLPs instead of GELU.  
- Zero‑init projection and classification layers (muP‑like) so early training is stable and updates are better conditioned.
- Skip connections from the token embedding into all layers (“embedding shortcut”), and value residual paths that feed value streams forward separately from the main residual stream.  
- Extra value embeddings: embeddings injected specifically into attention values, distinct from the standard token embedding. 
- “Back‑out” mechanism that lets the model subtract contributions from early layers before the final logits, effectively allowing the residual stream to carry information that doesn’t have to directly affect predictions.
- Smear module: a cheap 1‑token look‑back mechanism that allows local context reuse without invoking full attention, improving efficiency.

**Relevance to Parameter Golf:** these are all ways to get more expressive power per parameter; most are structural or routing tricks that add relatively few weights while unlocking better use of a tiny model.

---

## 2. Attention, context, and masking

- FlexAttention / FlashAttention‑style kernels for efficient long‑context attention with tiling and fused operations.
- Long–short window attention: some heads see long context windows while most heads use short, sliding windows; window size can be warmed up during training (YaRN‑style window warmup).
- Partial RoPE: apply rotary embeddings only to a subset (e.g., 50%) of the head dimensions; the “stationary” dimensions can carry more global, position‑invariant information, which works well with long‑window heads. 
- Attention “sinks” and gating: mechanisms to avoid wasting attention mass on BOS tokens or padding‑like sinks, plus sparse attention gating per head. 
- Short‑long windows are explicitly tuned so long heads focus on pattern‑matching similar activations across positions, while short heads handle local syntax/semantics.  

**Relevance to Parameter Golf:** long/short windows and partial RoPE are high‑impact, parameter‑cheap ways to get better compression from longer context, without making the model huge; you just have to keep the kernel code small enough for the 16 MB artifact.

---

## 3. Optimizers and training dynamics

- **Muon optimizer:** replaces AdamW with an optimizer that orthogonalizes momentum via a Newton–Schulz step (maintains low update condition number and more even neuron norms), leading to better data efficiency and faster convergence than AdamW on modded‑NanoGPT.
- **NorMuon:** a refinement of Muon that adds neuron‑wise adaptive learning rates based on second‑order statistics, maintaining uniform neuron norms while preserving Muon’s good conditioning; shown to outperform Muon on 124M/350M NanoGPT‑style models.
- **Polar Express:** optimized implementation of Muon’s orthogonalization using symmetric matmuls plus specialized distributed scheduling to reduce overhead.
- Momentum warmup: gradually increase momentum instead of jumping to a high value immediately, avoiding early training instabilities when using Muon‑style optimizers. 
- Learning rate cooldown: a linear LR cooldown phase toward the end of training to squeeze out a bit more loss improvement without overfitting or destabilizing training. 
- Heterogeneous update frequency: embeddings and LM head are only updated every 2 steps (using gradient accumulation) while the rest of the network updates every step, effectively giving them larger batch sizes and reducing communication cost. 

**Relevance to Parameter Golf:** Muon/NorMuon‑style optimizers and careful warmup/cooldown schedules are some of the clearest “loss‑per‑token” wins; they don’t cost many bytes if implemented compactly and are ideal for making a small model learn as much as possible within 10 minutes.

---

## 4. Numerics, precision, and quantization‑ready design

- Mixed precision: bfloat16 activations and loss computation, with some sensitive weights in float32.
- FP8 matmul head: the LM head is computed in FP8 with appropriate scaling, allowing cheaper logits computation while keeping accuracy via careful rescaling and softcapping.
- Tanh logit softcap (inspired by Gemma 2): clamp extreme logits via a smooth tanh cap to keep them in a numerically stable range; this particularly helps when using low precision in the head.
- Asymmetric rescaling of activations/gradients so FP8/bfloat16 operations remain stable and gradients neither explode nor vanish in deep stacks. 

**Relevance to Parameter Golf:** Golf explicitly rewards quantization and compression; Speedrun’s experience shows where you can push to FP8/low‑bit safely and where you should retain higher precision, which directly informs a small, highly compressed Golf model.

---

## 5. Data handling, packing, and batching

- Align batch boundaries with EoS: batches always start at document boundaries and respect a max document length so sequences are composed of complete docs instead of arbitrary chunks. 
- Careful masking: attention masks respect intra‑document structure, with minimal cross‑document leakage, improving modeling of real text structure.
- Efficient data pipeline: asynchronous CPU data loading, pre‑indexing of shards, and overlapping I/O with GPU compute to keep GPUs saturated. 
- Gradient accumulation and batch scheduling: use smaller per‑device microbatches with accumulation to hit an effective large batch size while fitting memory and reducing communication overhead for certain parameters.

**Relevance to Parameter Golf:** you have a fixed 10‑minute wall‑clock; getting more *useful* tokens (aligned, well‑masked, diverse) into that window yields better BPB for essentially no byte cost, and the required code is relatively small.

---

## 6. Distributed training and systems tricks

- Distributed Muon / NorMuon: pack parameters so optimizer states live in sharded form, and use reduce‑scatter plus all‑gather patterns to amortize the cost of orthogonalization across GPUs.
- Custom Triton or fused kernels for Muon’s symmetric matrix operations (Polar Express), reducing optimizer overhead at large scale.
- FlexAttention / FlashAttention integration using `torch.compile` or fused CUDA kernels to cut attention overhead with long windows.
- Communication‑aware parameter layouts so high‑traffic tensors are co‑located or updated less frequently.

**Relevance to Parameter Golf:** some of these systems optimizations might inflate code size, so you have to cherry‑pick those that bring big throughput benefits with minimal implementation footprint, but conceptually the lesson is “fuse and shard aggressively so your 8×H100 cluster is fully utilized for those 10 minutes.”

---

## 7. Regularization, initialization, and stability details

- Zero‑initialized projections and classifier head (“muP‑like”) to encourage smoother early training and muP‑style scaling behavior.
- Gradient and activation scaling strategies inspired by maximal update parameterization and its sparse/generalized variants (e.g., SμPar), to keep signal propagation balanced across depth and width.
- Careful choice of norm layers (QK‑Norm on attention inputs) to stabilize attention scores and help with training at high learning rates or under low precision.

**Relevance to Parameter Golf:** small models can be brittle; stable, muP‑like initialization and normalization give you more leeway to push learning rates and low precision without collapse, so you get more progress in 10 minutes.

---

## 8. Empirical process and search strategy

- Treat each improvement as an A/B‑testable change: modded‑NanoGPT logs individual contributions (RoPE, Muon, QK‑Norm, ReLU², FlexAttention, etc.) against a shared FineWeb benchmark.
- Recognize that later gains became smaller and more systems‑heavy: once big wins like RoPE and Muon were in, further record drops came primarily from better kernels, communication patterns, and long‑context tricks. 
- Use modded‑NanoGPT and variants as a “research platform” on which many separate groups tried things (new optimizers, new attention layouts, better data pipelines) and only kept what systematically reduced time‑to‑target loss.

**Relevance to Parameter Golf:** use the same mindset: define a cheap proxy run (fewer tokens / smaller model) inside the Parameter Golf repo, then iterate systematically on optimizers, normalization, attention layout, and packing; let AutoResearch or similar tools explore these spaces while you enforce strict parameter and artifact limits.

---
## 10. XLA and Ray actors

- XLA (e.g., via PJIT / Pmap on JAX or XLA‑backed PyTorch) can compile the training step into a fused graph, reducing Python overhead and improving kernel fusion, similar in spirit to what Flex/FlashAttention and fused Muon kernels aim to do. Used well, it increases tokens‑per‑second without changing the model’s math, but you must keep the compiled step and glue code small enough that it does not bloat the 16 MB artifact.  
- Ray actors (or similar actor‑based frameworks) can be useful during *development* for orchestrating many short experiments across multiple machines or GPUs, managing AutoResearch‑style agents, and queueing runs; however, for the final Parameter Golf submission, heavyweight orchestration frameworks are usually a bad fit because they add a lot of code and are unnecessary when evaluation is a single, fixed 8×H100 run.  

**Relevance to Parameter Golf:** XLA‑style compilation is conceptually aligned with the Speedrun focus on fused, high‑throughput kernels (FlashAttention, Triton, Polar Express), whereas Ray‑style actors are better kept as an out‑of‑band tooling choice for experiment management rather than something that lives inside the final 16 MB artifact.

---
## 9. Practical takeaway for Parameter Golf

All together, NanoGPT Speedrun winners showed that the big levers in this regime are:

- **Architecture:** RoPE + QK‑Norm + ReLU² + clever residual/value paths, smear module, back‑out mechanism. 
- **Optimizers:** Muon, NorMuon, Polar Express‑style implementations, with momentum warmup and LR cooldown.
- **Attention:** Flex/FlashAttention, long–short windows, partial RoPE, attention sinks and gating, window warmup.  
- **Numerics:** mixed precision (bfloat16, FP8 head), tanh logit softcap, safe rescaling.
- **Data / systems:** document‑aligned packing, efficient loaders, heterogeneous updates, distributed optimizer engineering.
For Parameter Golf you would borrow these *categories* of tricks, but always re‑evaluate them through two lenses: “Does this help BPB on FineWeb?” and “How many bytes of code/weights does it cost inside a 16 MB artifact?”