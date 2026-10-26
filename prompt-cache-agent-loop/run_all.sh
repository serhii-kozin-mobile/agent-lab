#!/usr/bin/env bash
# Reproduce the study end to end. Free: no API keys, no model calls.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv && . .venv/bin/activate
pip install -q -r requirements.txt
python3 fetch_data.py
python3 measure_tokens.py
python3 cost_model.py > /dev/null
python3 report.py
