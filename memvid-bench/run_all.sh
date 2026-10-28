#!/usr/bin/env bash
# Reproduce the Memvid vs BM25/dense baseline vs SQLite benchmark end to end.
# CPU only. ~1 GB download, ~20-40 min on 8 vCPU (index build dominates).
set -euo pipefail
cd "$(dirname "$0")"
# 1. Prep env: corpus download, chunking, bge-small embeddings, BM25 (the "lab" baseline)
[ -x prep/.venv/bin/python ] || python3 -m venv prep/.venv
prep/.venv/bin/python -m pip install -q -r prep/requirements.txt
(cd prep && bash scripts/fetch_data.sh && .venv/bin/python lab/build_corpus.py && .venv/bin/python scripts/build_indexes.py)
# 2. Bench env: memvid-sdk + sqlite-vec
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install -q "memvid-sdk==2.0.160" "numpy==2.5.4" "psutil==7.2.2" sqlite-vec
P=prep/.venv/bin/python
$P scripts/prep_queries.py            # 110 answerable questions -> query vectors (bge-small + query prefix)
$P scripts/bench_baseline.py --variant fixed
$P scripts/bench_baseline.py --variant ctx
$P scripts/time_query_encode.py
.venv/bin/python scripts/bench_sqlite.py --variant fixed
.venv/bin/python scripts/bench_sqlite.py --variant ctx
.venv/bin/python scripts/bench_memvid.py --variant fixed   # 4 shards: free tier caps one .mv2 at 50 MB
.venv/bin/python scripts/bench_memvid.py --variant ctx
for k in 30 100 300; do KLEX=$k .venv/bin/python scripts/shard_sanity.py; done
python3 scripts/paired_ci.py
python3 scripts/by_category.py
python3 scripts/summarize.py
