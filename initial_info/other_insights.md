# insights.md

## High‑level research and system‑design insights

- Parameter Golf is a perfect playground for “research‑agent” workflows because it:
  - Has a single clean metric (BPB).
  - Has fixed data and strict constraints.
  - Allows many fast iterations with short training runs.
- A strong approach is to treat it as an automated architecture search problem with a human in the loop: use agents to propose changes and run 5–10 minute mini‑experiments, but keep a human gating what gets promoted to full 8×H100 trials.

## Andrej Karpathy’s AutoResearch – core ideas you can reuse

- AutoResearch is an open‑source system by Andrej Karpathy that automates the ML research loop:
  - LLM agents read a goal spec (e.g., in `program.md`).
  - They edit a PyTorch training script and config.
  - They launch short training runs (≈5 minutes), read logs, and iteratively improve the code and hyperparameters.
- In the demo, AutoResearch uses Claude models (Sonnet / Opus) as reasoning agents and a NanoGPT‑style trainer as the domain code.
- The key pattern to copy:
  - Separate “research logic” (LLM agents deciding what to try next) from “domain code” (your `train_gpt.py` etc.).
  - Keep experiment cycles short and cheap, so you can do hundreds of iterations.

## How to adapt AutoResearch‑style loops to Parameter Golf

- Swap the domain:
  - Point AutoResearch (or your Claude‑code agent) at your fork of `openai/parameter-golf`.
  - Give it permission to modify:
    - Model config (depth, width, heads, vocab size, etc.).
    - Architectural choices (parameter tying, depth recurrence, low‑rank layers).
    - Quantization and compression code.
    - Training hyperparameters (optimizer, LR schedule, batch sizes).
- Redefine the objective:
  - In the AutoResearch “goal” document, make the objective:
    - “Minimize FineWeb validation BPB under a 16 MB artifact and a 10‑minute wall‑clock training budget on 8×H100.”
  - Use proxy objectives for cheaper iterations:
    - BPB / loss after a fixed small number of tokens.
    - Artifact size measured on a small model configuration.
- Guardrails:
  - Hard constraints: never exceed 16 MB artifact, never introduce external network calls, don’t change the dataset beyond allowed options.
  - Limit code‑editing scope (for safety and stability) to model config and explicit “hooks” instead of arbitrary files in the repo.

## Concrete design axes for automated exploration

- Architecture:
  - Depth recurrence: experiment with different numbers of “virtual layers” using a small number of physical blocks.
  - Parameter tying patterns: tie embeddings, attention projections, MLP layers, even multiple blocks; systematically sweep through patterns and record their effect on BPB and artifact size.
  - Low‑rank factorization: vary ranks of attention and MLP matrices; see how far ranks can be pushed down before BPB degrades too much.
  - Bitnets / low‑bit layers: try 1–2 bit weights in some layers, higher precision in others.

- Tokenization:
  - Search over vocab sizes: e.g., 256, 512, 1024, 2048, etc., and maybe domain‑specific tokenization schemes that compress web text better.
  - Record effect on:
    - Artifact size (embeddings + LM head).
    - Sequence length distribution and resulting BPB.

- Training and optimization:
  - Learning‑rate schedules: warm‑up length, cosine vs linear decay, restarts.
  - Optimizers: AdamW vs variants; weight decay strength; gradient clipping strategies.
  - Data ordering: try different shuffling strategies or curriculum in the fixed dataset to see if early‑stage convergence improves.

- Compression / quantization:
  - Systematically evaluate post‑training quantization vs quantization‑aware training, and mixed‑precision layouts.
  - Include artifact compression in the loop:
    - Checkpoint format.
    - Choice of entropy coding / compression parameters.
    - Layout of weights for better compressibility.

## Insights specific to BPB and tiny‑model scaling

- BPB is a more “compression‑native” metric than token‑level loss:
  - It is tokenizer‑agnostic and punishes wasteful encodings.
  - It rewards models that align their internal units with the true structure of bytes in the data.
- For tiny models, you often cannot rely on brute‑force depth/width scaling; you need:
  - Strong inductive biases (architectures that “bake in” assumptions about language).
  - Better tokenization.
  - Smarter use of test‑time compute.
- Neural scaling law perspective:
  - At small N, naive scaling laws often under‑predict what is possible with architectural tricks and better optimization.
  - The game is to “beat the scaling curve” at the tiny‑parameter regime.

## How Claude (or any LLM) can fit into your workflow

- Use Claude‑code as:
  - A config / code generator:
    - You provide high‑level design directions.
    - Claude edits the config and model code under strict constraints (e.g., only in `model.py` and `config.py`).
  - An experiment‑planner:
    - Given a log of past runs (config + BPB + artifact size), ask it to propose the next few experiments that move along the Pareto frontier (better BPB or smaller artifact).
  - A “research summarizer”:
    - Periodically feed it your logs and commit history and ask for synthesized insights on what’s working and why.

- Important constraint:
  - The LLM is only used during your development loop.
  - The final submitted artifact must run offline with no external API calls.

## Practical tips for a beginner (especially on Windows)

- Dev environment:
  - Use WSL2 with Ubuntu on Windows for local experiments; this aligns better with the Linux environment used on Runpod and in the official evaluations.
  - Start with tiny CPU or single‑GPU runs to smoke‑test changes before using expensive GPUs.
- Cloud usage:
  - Use Google Colab or small cloud GPUs for prototyping on toy data or small FineWeb subsets; they are great for debugging but not powerful enough for final 10‑minute 8×H100 runs.
  - Use Runpod or similar for serious experiments and final runs; leverage the challenge‑provided templates.
- Workflow pattern:
  - Debug locally → mini‑experiments on cheaper GPUs → full 8×H100 experiments → submission PR.
  - Automate logging and artifact‑size checks early to avoid silent rule violations.

## Mindset / meta‑level insights

- Treat the 16 MB + 10‑minute constraint as a feature, not a bug:
  - It forces you to learn about compression, quantization, tokenizer design, and efficient training—skills that matter for real‑world deployment.
- Think in terms of Pareto frontiers:
  - Every change shifts you somewhere on the BPB vs artifact‑size vs training‑time surface.
  - You want a set of configurations that lie on or near the efficient frontier; automated tools can help you find it.
- Community leverage:
  - Watch the `openai/parameter-golf` issues and PRs for emerging techniques and pitfalls.
  - Use other people’s records as seeds for your own AutoResearch / Claude explorations (e.g., starting from the naive baseline record).



