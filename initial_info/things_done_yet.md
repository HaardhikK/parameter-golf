# things done yet :
## Environment & Infrastructure
* **Repository:** Forked `openai/parameter-golf` and cloned it into a WSL2 Ubuntu environment.
* **OS/Platform:** Windows 11 with WSL2 (Ubuntu 24.04).
* **Python Management:** Installed Miniconda inside WSL2; created a dedicated `.venv`.
* **IDE:** VS Code with WSL Extension for remote development.
* **AI Tooling:** Started Claude Code within the terminal for automated development and analysis.

## Hardware & Dependencies
* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (5 GB VRAM).
* **CUDA:** 12.8.
* **PyTorch:** 2.10.0+cu128.
* **Python:** 3.13.12.
* **Packages:** Installed `torch`, `numpy`, `sentencepiece`, `huggingface-hub`, `datasets`, `tqdm`, `tiktoken`, and required CUDA libraries.
* **Fixes:** Installed `gcc` and `build-essential` via `apt-get` to resolve Triton JIT-compilation errors.

## Data Preparation
* **Variant:** `sp1024` with `--train-shards 1`.
* **Files Downloaded:**
    * `data/datasets/fineweb10B_sp1024/fineweb_train_000000.bin`
    * `data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin`
    * `data/tokenizers/fineweb_1024_bpe.model`

## Core Constraint Analysis
* **16MB Limit:** Enforced post-training via int8 quantization and `zlib` compression (level 9).
    * *Baseline Size:* 5.39 MB (~10.6 MB headroom remaining).
    * *Total Calculation:* `quant_file_bytes + code_bytes`.
* **10-Minute Timer:** Enforced by `max_wallclock_seconds = 600.0`.
    * The script uses a wallclock cap check at each step.
    * LR schedule (warmdown) automatically adapts based on remaining time.

## Current Status
* **Smoke Test:** PASSED.
* **Command used:** `ITERATIONS=10 VAL_LOSS_EVERY=10 WARMUP_STEPS=0 MAX_WALLCLOCK_SECONDS=300 .venv/bin/python train_gpt.py`.
* **Next Steps:** Optimize model architecture and training hyperparameters to improve BPB score while staying within the 16MB/10-minute envelope.