#!/usr/bin/env bash
# Reproduce the local study end to end. Free: no API keys, no model calls, no network
# (except pip). Phase 2 against Slumbot's public API is opt-in: ./run_all.sh --slumbot
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
python3 -m venv .venv && . .venv/bin/activate
pip install -q -r requirements.txt
mkdir -p results charts
cd src
python3 perception_bench.py 2000 ../results/perception_bench.csv   # 2,000 real states x 8 conditions x 2 readers
python3 run_matches.py 3000 ../results "${WORKERS:-4}"              # GT + 16 vision configs, 3,000 duplicate hands each
python3 examples.py
python3 review_anchor_fix.py 3000   # review experiment: confirmed chip-conservation anchor (~10 min, single process)
if [[ "${1:-}" == "--slumbot" ]]; then
  # Polite: one request at a time per stream, 0.5 s pause between requests. Resumable.
  mkdir -p ../results/slumbot
  python3 slumbot_client.py gt     "${SLUMBOT_HANDS:-1000}" ../results/slumbot/slumbot_gt.csv     realistic_mix 0.5
  python3 slumbot_client.py vision "${SLUMBOT_HANDS:-1000}" ../results/slumbot/slumbot_vision.csv realistic_mix 0.5
  python3 slumbot_field_check.py ../results/slumbot/slumbot_vision.csv ../results/slumbot/field_check.csv
fi
python3 analyze.py
python3 review_checks.py   # independent review checks on the results above
