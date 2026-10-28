# llmwiki-vs-rag

**Question:** Does Karpathy's "LLM wiki" pattern (an LLM compiles raw docs into a linked wiki, then answers by reading the index and picking pages) beat plain BM25 retrieval over chunks of the same docs?

**Short answer:** Not in this setup. BM25 RAG over 200-word chunks scored 95.7%. The best wiki variant scored 63.0%. About 35% of answerable questions had the needed fact lost when the docs were compressed into the wiki.

## What was tested

- **Corpus:** 100 public technical docs (Python 3.13 docs, MCP spec revisions, NIST PDFs): 46 docs holding the gold answers plus 54 random distractors, 1,509,759 raw chars. Long docs capped at 36,000 chars; wiki and RAG saw the same text.
- **Questions:** 46 answerable (single 24, table 8, near-duplicate/version-specific 8, multi-hop 6) and 10 with no answer in the corpus. Plus 8 cross-source synthesis questions.
- **Wiki:** built one doc per LLM call following the schema in `CLAUDE.md` (source page + up to 3 entity/concept pages per doc, Python-rebuilt index). Final wiki: 303 pages, 530,307 chars, 48,642-char index. Compile cost ~935k estimated tokens.
- **Arms:**
  - `rag`: BM25 top-5 over 200-word chunks (40-word overlap, `Title > heading` prefix), 1,339 chunks.
  - `wiki`: read `index.md[:16000]`, pick up to 5 pages, answer.
  - `wiki_full`: same, whole index.
  - `wiki_links`: whole index + one optional `[[link]]` hop.
  - `wiki_bm25`: BM25 top-5 over wiki pages.
- BM25 is identical across arms: `rank_bm25` BM25Okapi (k1=1.5, b=0.75, epsilon=0.25), regex tokenizer `[a-z0-9]+(?:[._-][a-z0-9]+)*`, small stopword list. Only the unit differs.

## Models

- Answerer (all arms and wiki compile): Claude Sonnet 4.5 (`claude-sonnet-4-5`) via the `claude` CLI, `--effort low`, tools off, empty temp dir.
- Judge: Claude Opus 5.5, verdicts correct / partial / wrong / correct_abstention / wrong_abstention. A second judge agreed on 19/20 sampled answers.

Unlike the other studies in this repo, this one used paid hosted models.

## Results (46 answerable; accuracy = (correct + 0.5·partial)/n; paired bootstrap vs RAG, 10,000 resamples, seed 7)

| arm | acc (half) | strict | correct/partial/wrong | no-answer abstention | est. tokens/query | latency/query | diff vs RAG [95% CI] |
|---|---|---|---|---|---|---|---|
| rag | **95.7%** | 93.5% | 43/2/1 | 10/10 | 2,044 | 8.6 s | — |
| rag (repeat) | 95.7% | 93.5% | 43/2/1 | 10/10 | 2,043 | 9.0 s | 0.000 [−0.033, 0.033] |
| wiki (index[:16000]) | 32.6% | 28.3% | 13/4/29 | 10/10 | 5,330 | 15.8 s | −0.630 [−0.772, −0.489] |
| wiki_full (whole index) | 52.2% | 50.0% | 23/2/21 | 10/10 | 13,842 | 17.2 s | −0.435 [−0.587, −0.283] |
| wiki_full, 36k context cap | 58.7% | — | 26/2/18 | — | 13,937 | — | −0.370 [−0.522, −0.217] |
| wiki_links (full index + 1 hop) | 62.0% | 58.7% | 27/3/16 | 10/10 | 14,893 | 23.0 s | −0.337 [−0.478, −0.196] |
| wiki_bm25 | 63.0% | 60.9% | 28/2/16 | 10/10 | 2,488 | 8.5 s | −0.326 [−0.478, −0.174] |

- Link hop: the model asked for extra pages on only 4/56 queries. With matched context caps, wiki_links vs wiki_full is +0.033 [−0.076, 0.141]: no measurable effect.
- Synthesis questions (8): wiki 31%, RAG 69%.
- **Compile loss estimate:** for 16 of 46 answerable questions (35%) the fact is absent from the wiki (5 of the 8 table questions). A wider recheck reading 15–18 pages per question still found 16/16 absent. That puts the ceiling for any wiki-only arm near 30/46 = 65.2%; wiki_bm25 and wiki_links sit close to it. RAG answered 15 of those 16 correctly.
- Every gold quote sits inside at least one RAG chunk, and the RAG settings were chosen in an earlier study on the same golden set, so RAG has a home advantage.
- Token counts are estimates (chars/4). Latency is answerer wall time.

## What is omitted and why

- **Question text, reference answers and gold quotes:** they come from a private golden set (not released), so `results/` holds question ids and verdicts only.
- **`data/`, `raw/`, `wiki/`:** the corpus copies and generated wiki are not redistributed. The source docs are public (docs.python.org, modelcontextprotocol.io, NIST).
- **Lab retriever module:** the scripts import `retrieval.tokenize` from a separate lab directory set by `LAB_DIR` (default `./lab`). Its tokenizer is fully specified above.

## What this does not prove

It measures this implementation of the pattern at ~100 docs with a 3-page-per-doc compile cap (the gist suggests 10–15). A richer compile or an agentic loop that follows links might do better. The compile-loss count rests on an LLM judge and is an estimate.

## Rerun

Needs Python 3.13, `rank_bm25`, a logged-in `claude` CLI, a judge command at `~/bin/claude-llm` (any wrapper that takes a prompt argument and prints the reply), and your own golden set in the lab directory.

```bash
export LAB_DIR=/path/to/lab          # retrieval.py + data/golden.json + corpus
python scripts/select_subset.py      # build raw/ (seeds 42/43/123)
python scripts/ingest.py             # compile the wiki
python scripts/build_rag.py          # BM25 indexes
python scripts/answer.py --system rag   # also wiki, wiki_full, wiki_links, wiki_bm25
python scripts/judge.py
python scripts/compile_loss.py && python scripts/compile_loss_wide.py
python scripts/metrics.py && python scripts/r3_stats.py
```

Check each script's `--help` for labels and caps.
