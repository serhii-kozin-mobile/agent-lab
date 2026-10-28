# LLM Wiki schema (loaded every session)

This wiki implements Karpathy's LLM Wiki pattern over the docs in `raw/` (immutable).
The LLM writes and maintains everything under `wiki/`; humans only curate `raw/` and ask questions.

## Layout
- `raw/<slug>.md` — one immutable source document. Never edit.
- `wiki/sources/<slug>.md` — one summary page per raw source. Frontmatter: `doc_id`, `title`.
  Body: 4-10 bullet facts with exact values/names, then `## Links` with `[[entities/...]]` / `[[concepts/...]]` links.
- `wiki/entities/<name>.md` — a module, tool, spec feature, or publication. Facts attributed as `(source: <doc_id>)`.
- `wiki/concepts/<name>.md` — a cross-cutting concept (e.g. transports, sampling, logging). Same attribution rule.
- `wiki/index.md` — catalog: one line per page, `- [<title>](<path>) — <one-line summary>`, grouped by Sources / Entities / Concepts.
- `wiki/log.md` — append-only: `## [ingest] <doc_id>` per ingested source.
- `output/` — filed query answers worth keeping.

## Ingest (one source at a time)
1. Read the new raw source fully.
2. Write/update its `wiki/sources/` page: keep exact numbers, names, defaults, and version strings verbatim.
3. Create or update the entity/concept pages it touches (max 3 per source). Merge new facts; never delete
   a fact unless a newer source contradicts it — then keep both and add `> CONTRADICTION: ...` noting versions.
4. Regenerate `wiki/index.md` lines for touched pages; append one log line.
5. Link every entity/concept mention with `[[...]]`. No page may link to a page that does not exist.

## Query
1. Read `wiki/index.md` first; pick the pages that can answer the question.
2. Read only those pages. Answer using ONLY the wiki. Cite pages as `(wiki: <path>)`.
3. If the wiki does not contain the answer, reply exactly `NOT_FOUND`.

## Lint
Flag: contradictions between pages, stale claims superseded by newer sources, orphan pages,
broken `[[links]]`, concepts mentioned but lacking a page.
