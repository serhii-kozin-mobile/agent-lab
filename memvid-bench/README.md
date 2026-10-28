# memvid-bench

Memvid (`memvid-sdk==2.0.160`, free tier) vs a plain BM25 + dense baseline vs SQLite (FTS5 + `sqlite-vec`)
on the same chunks, the same `BAAI/bge-small-en-v1.5` vectors and the same 110 questions. CPU only.

## Data (all public)

- Python 3.13 docs (HTML), Model Context Protocol spec (3 revisions, commit `54b91cd`), 10 NIST publications.
  `prep/scripts/fetch_data.sh` downloads them; `prep/data/sha256_raw.txt` holds the checksums of our run
  (Python docs are rebuilt on each patch release, so those may differ).
- `prep/data/golden.json`: 122 questions drafted by an LLM from sampled corpus chunks, kept only when the
  evidence quote is found verbatim in the corpus (150 passed review, sampled down to fixed per-category sizes).
  12 are deliberately unanswerable; the benchmark uses the other **110** (`data/queries.json`).
- A retrieved chunk is a hit when it overlaps a gold span by at least half of the shorter of the two.

`prep/` is the baseline pipeline ("the lab"): corpus builder, two chunkers (`fixed`: 200-word windows,
40-word overlap; `ctx`: section-aware chunks with a `title > section` header), BM25 (`rank_bm25`) and dense
retrieval. Memvid and SQLite index exactly the chunk text and vectors the lab produces.

## Reproduce

```bash
git clone https://github.com/serhii-kozin-mobile/agent-lab && cd agent-lab/memvid-bench
./run_all.sh
```

Needs Python 3.13, git, curl, ~3 GB free disk. Outputs go to `results/` (overwriting ours).

## Versions

memvid-sdk 2.0.160 · sqlite-vec (pip latest at run time, Oct 2026) · numpy 2.5.4 · psutil 7.2.2 ·
sentence-transformers 6.2.0 · torch 2.14.1 (CPU) · rank-bm25 0.2.2 · bge-small-en-v1.5.

## Results

`results/summary_table.md` (all systems), `results/paired_ci_hit5.json` (paired bootstrap 95% CI on hit@5
differences), `results/by_category_hit5.json`, per-query CSVs and run logs. Memvid `lex-*` rows are query
variants: `raw` = question as-is, `or` = terms OR-joined, `score` = shards merged by raw score, `deep` = larger k.

Not included: the built `.mv2` shards, SQLite databases and model weights (rebuild with `run_all.sh`).
