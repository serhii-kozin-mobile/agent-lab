#!/usr/bin/env bash
# Reproduce everything on a CPU box (no GPU, no paid API). ~1.5 h on 8 shared cores (llama.cpp at 3 threads per process).
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("Qwen/Qwen2.5-3B-Instruct-GGUF", "qwen2.5-3b-instruct-q4_k_m.gguf", local_dir="models")
PY
export MODEL=models/qwen2.5-3b-instruct-q4_k_m.gguf THREADS=3
python fetch_data.py
python features.py
python leakage_check.py
python run_llm.py stats 0.0 0 & python run_llm.py stats_odds 0.0 0 & wait
python run_llm.py stats 0.7 1 120 & python run_llm.py stats 0.7 2 120 & wait   # rerun check on the first 120 matches
python report.py
python elo_sensitivity.py   # review check: Elo baseline over a K x home-advantage grid
