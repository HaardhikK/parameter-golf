## What the Parameter Golf challenge is

- OpenAI’s “Model Craft: Parameter Golf” is a challenge to train the best possible language model that fits inside a 16 MB artifact and can be trained from scratch in under 10 minutes on 8×H100 GPUs.
- The core metric is bits per byte (BPB) on a fixed FineWeb validation set; lower BPB = better compressor/predictor.
- Everyone uses a fixed FineWeb export (about 8–10B training tokens plus a fixed validation split), so improvements come from model and training tricks, not data differences.
- The challenge is free to enter, runs roughly from March 18, 2026 to April 30, 2026, and is open to most countries where OpenAI’s API is available (18+).
- OpenAI and Runpod are providing up to 1M USD in GPU compute credits, so participants can train on H100s without paying cash if they get credits.

## What makes it unique / design philosophy

- It’s “code‑golf‑style” for models: optimize performance under strict size and compute caps instead of just scaling up model size.
- Constraints force you to think about:
  - Better architectures at tiny scale.
  - Extreme compression and quantization.
  - Making the most of 10 minutes of training on 8×H100s.
- It is explicitly framed as optimizing L(N): best achievable loss at a fixed parameter / artifact budget, in contrast to:
  - NanoGPT Speedrun (L(T) – how fast to hit a target loss).
  - “Slowrun” style ideas (L(D) – best loss for a fixed data budget).
- It’s essentially a “tiny model scaling‑laws lab”: you are exploring the small‑N corner of scaling laws and trying to beat naive scaling behavior with clever design.

## High‑level rules / constraints (for Claude or AutoResearch to respect)

- Artifact size:
  - Entire submission (code + compressed weights + tokenizer + config etc.) must be ≤ 16,000,000 bytes.
- Compute budget:
  - Model must be trainable from scratch in under 10 minutes on 8×H100 GPUs for leaderboard eligibility.
- Dataset:
  - Use the provided FineWeb export via the repo’s data scripts (a fixed training set and validation set).
- Evaluation:
  - Metric is bits per byte (BPB) on the FineWeb validation set (tokenizer‑agnostic).
- Environment:
  - No external network calls during evaluation; the artifact must be self‑contained (no online APIs, no downloading additional data or code).
- Submission:
  - Submit as a public GitHub pull request to `openai/parameter-golf` containing:
    - Model code and configuration.
    - Training script.
    - Weights / checkpoints encoded into the 16 MB artifact.
    - Logs and description required by the repo.

## Repo and baseline structure (things Claude can hook into)

- The `openai/parameter-golf` repo provides:
  - Training scripts like `train_gpt.py` (PyTorch) and `train_gpt_mlx.py` (Apple MLX).
  - Data helpers under `data/` to download and cache FineWeb shards and tokenizers.
  - Example “records” under `records/track_10min_16mb/...` with a naive baseline config, hyperparameters, and logs.
  - GitHub Actions / evaluation workflows wired to the leaderboard.
- Data layout (important for any automation):
  - `data/datasets/<dataset_name>/` – tokenized training/validation shards.
  - `data/tokenizers/` – tokenizer models (e.g., 1024‑token SentencePiece/BPE).
  - Helper script (illustrative):
    - `python data/cached_challenge_fineweb.py --variant sp1024` to populate the canonical FineWeb export.

## Baseline model details (good starting point for Claude)

- Example baseline from `records/track_10min_16mb/...` uses:
  - Vocabulary size: 1024.
  - Layers: 9 transformer blocks.
  - Model dimension: 512.
  - Attention heads: 8, with 4 key/value heads (KV sharing).
  - MLP multiplier: 2.
  - Training batch tokens: ≈ 524k, sequence length 1024.
- The baseline demonstrates:
  - How to set distributed training over 8 GPUs (torchrun, NCCL env vars).
  - How to log validation loss, BPB, and compressed artifact size.
  - That ~7–8B tokens can be processed in 600 seconds on 8×H100 with a small GPT.

## Key technical levers (what the “research agent” should explore)

High impact design axes that fit the spirit of Parameter Golf:

- Architecture / parameter‑efficiency:
  - Depth recurrence: reuse the same block parameters multiple times across depth to simulate a deep network with a tiny parameter set.
  - Aggressive parameter tying: tie input/output embeddings, reuse projections, share weights across layers wherever possible.
  - Low‑rank factorizations in attention and MLP layers.
  - Bitnets / ultra‑low‑precision designs.

- Compression / quantization:
  - Quantization‑aware training to keep performance after int8 or lower quantization.
  - Exploring mixed‑precision layouts where “important” layers keep higher precision and others are aggressively compressed.
  - Weight packing plus general compression (zlib or similar) tuned to make best use of the 16 MB limit.

- Tokenization:
  - Non‑standard tokenizers and small vocabulary sizes to shrink embeddings while preserving compression quality.
  - Trade‑off: smaller vocab = smaller embedding matrix but longer sequences; must be evaluated by BPB, not just token‑level perplexity.

- Training strategy:
  - Scheduling the 10 minutes carefully: number of steps, learning‑rate schedules, warm‑up, optimizer choice, etc.
  - Curriculum or data‑ordering strategies on the fixed dataset (e.g., easy‑to‑hard ordering) within what the rules allow.

- Test‑time compute / test‑time training:
  - Within the rules, you can use test‑time compute tricks (ensembles, iterative decoding, extra passes) as long as they fit in the artifact.
  - Limited forms of test‑time training (small updates near evaluation) may be allowed if they use only the provided data and stay within constraints; must be checked carefully against the latest rules.

## Participation logistics (for agent planning)

- Eligibility and cost:
  - Free to enter, 18+, must be in an eligible country.
  - No entry fee; you can either use your own hardware or apply for Runpod GPU credits.
- Runpod credits:
  - OpenAI’s challenge page links to a form where you can request a specific credit tier (e.g., ≈25 / 500 / 1000 USD equivalent).
  - Approved credits appear in a Runpod account, where you can launch a pre‑configured Parameter Golf template with 8×H100.
- Attempts:
  - You can submit multiple PRs / iterations; there is no “one‑shot” restriction on the 8×H100 run.
  - Practical limits are GPU credits, time, and review bandwidth.

## Practical workflow sketch (how an agent could operate)

- Code layout expectations:
  - There is a main training script (e.g., `train_gpt.py`) and a config section / file that controls model size, optimizer, scheduler, etc.
  - Data loader and tokenizer live under `data/`; evaluation script computes BPB on validation shards.
- Typical loop (for a “research agent” or for manual work):
  1. Start from baseline config; verify it runs on a subset of FineWeb.
  2. Implement one family of changes at a time (e.g., quantization scheme, different tokenizer, depth recurrence).
  3. Run short, cheaper experiments (fewer tokens, smaller model) to estimate if the change looks promising.
  4. Periodically run full 8×H100, 10‑minute trials for promising configs to get official‑style BPB.
  5. Track artifact size and BPB together for all experiments to learn what moves the Pareto frontier.



