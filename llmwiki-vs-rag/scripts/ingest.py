"""Incremental wiki ingest: one raw doc per answerer call, in fixed ingest order.

Waves of WAVE parallel calls; outputs are applied sequentially in ingest order.
Python (not the LLM) rebuilds wiki/index.md deterministically from page summaries,
which keeps the index consistent and saves tokens; the LLM still writes every page,
proposes links, and flags contradictions (per CLAUDE.md). Snapshots of wiki/ are
copied to results/snapshots/wiki_{25,50,100} at 25/50/100% of ingest.
Resumable: docs already in wiki/log.md are skipped.
"""
import json, re, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, call_answerer  # noqa: E402

WAVE = 3
SNAPSHOTS = {25: "wiki_25", 50: "wiki_50", 100: "wiki_100"}


def slugify(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")[:60] or "page"


def parse_pages(text: str):
    pages = {}
    parts = re.split(r"^=== PAGE: (.+?) ===\s*$", text, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        pages[parts[i].strip()] = parts[i + 1].strip()
    return pages


def first_summary(body: str) -> str:
    for line in body.splitlines():
        line = line.strip().lstrip("-* ").strip()
        if line and not line.startswith(("#", ">", "doc_id", "title", "---")):
            return line[:140]
    return ""


def rebuild_index(wiki: Path):
    lines = ["# Wiki Index", ""]
    for sub, title in (("sources", "Sources"), ("entities", "Entities"), ("concepts", "Concepts")):
        lines.append(f"## {title}")
        for p in sorted((wiki / sub).glob("*.md")):
            lines.append(f"- [{p.stem}]({sub}/{p.name}) — {first_summary(p.read_text())}")
        lines.append("")
    (wiki / "index.md").write_text("\n".join(lines))


def build_prompt(schema, index_text, existing, raw_file, raw_text):
    ex = "\n\n".join(f"--- EXISTING PAGE {p} ---\n{c}" for p, c in existing) or "(none)"
    return f"""You maintain a wiki per this schema:\n\n{schema}\n\nCURRENT INDEX:\n{index_text[:12000]}\n\nEXISTING PAGES YOU MAY UPDATE:\n{ex}\n\nNEW SOURCE TO INGEST (file raw/{raw_file}):\n{raw_text}\n\nIngest this source now. Output ONLY blocks in this exact format, no preamble:\n=== PAGE: sources/<slug-of-doc>.md ===\n# <doc title>\ndoc_id: <doc_id from the source header>\n- <fact with exact values verbatim>\n(more bullets, 4-10)\n## Links\n- [[entities/...]] ...\n=== PAGE: entities/<kebab-name>.md ===\n(or concepts/<kebab-name>.md; create/update at most 3 such pages total; each: '# Name' then bullets, every fact ending with (source: <doc_id>); if this source contradicts an existing page fact, keep both and add a line starting '> CONTRADICTION:')\n\nRules: exact numbers, names, defaults, version strings verbatim. Never invent facts not in the source. Source page slug: lowercase, non-alphanumerics as '-'."""


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    wiki = ROOT / "wiki"
    for sub in ("sources", "entities", "concepts"):
        (wiki / sub).mkdir(parents=True, exist_ok=True)
    if not (wiki / "log.md").exists():
        (wiki / "log.md").write_text("# Wiki Log\n")
    schema = (ROOT / "CLAUDE.md").read_text()
    docs = {json.loads(l)["doc_id"]: json.loads(l) for l in open(ROOT / "data/subset_docs.jsonl")}
    order = json.load(open(ROOT / "data/ingest_order.json"))
    done = set(re.findall(r"ingest \| (\S+)", (wiki / "log.md").read_text()))
    todo = [d for d in order if d not in done]
    if limit:
        todo = todo[:limit]
    print(f"ingest: {len(done)} done, {len(todo)} todo", flush=True)
    stats_file = ROOT / "results/ingest_stats.jsonl"

    def one(doc_id):
        d = docs[doc_id]
        raw_text = (ROOT / "raw" / d["raw_file"]).read_text()
        index_text = (wiki / "index.md").read_text() if (wiki / "index.md").exists() else "(empty)"
        toks = set(re.findall(r"[a-z0-9]+", (d["title"] + " " + doc_id).lower()))
        existing = []
        for sub in ("entities", "concepts"):
            for p in sorted((wiki / sub).glob("*.md")):
                score = len(toks & set(re.findall(r"[a-z0-9]+", p.stem.replace("-", " "))))
                if score:
                    existing.append((f"{sub}/{p.name}", p.read_text()[:2500], score))
        existing = [(p, c) for p, c, _ in sorted(existing, key=lambda x: -x[2])[:4]]
        r = call_answerer(build_prompt(schema, index_text, existing, d["raw_file"], raw_text), tag=f"ingest:{doc_id}")
        return doc_id, r

    t_start = time.time()
    applied = len(done)
    for i in range(0, len(todo), WAVE):
        wave = todo[i: i + WAVE]
        with ThreadPoolExecutor(max_workers=WAVE) as ex:
            results = list(ex.map(one, wave))
        for doc_id, r in results:
            pages = parse_pages(r["text"]) if r["ok"] else {}
            written = []
            for rel, body in pages.items():
                if not re.fullmatch(r"(sources|entities|concepts)/[a-zA-Z0-9._-]+\.md", rel) or ".." in rel:
                    continue
                (wiki / rel).write_text(body + "\n")
                written.append(rel)
            if not any(w.startswith("sources/") for w in written):
                # fallback source page so every doc is represented in the wiki
                d = docs[doc_id]
                slug = re.sub(r"[^a-zA-Z0-9]+", "-", doc_id).strip("-")[:90]
                body = f"# {d['title']}\ndoc_id: {doc_id}\n- Ingest output unparsable; see raw/{d['raw_file']}.\n"
                (wiki / "sources" / f"{slug}.md").write_text(body)
                written.append(f"sources/{slug}.md (fallback)")
            rebuild_index(wiki)
            with open(wiki / "log.md", "a") as fh:
                fh.write(f"## [ingest] ingest | {doc_id} | pages: {', '.join(written)}\n")
            applied += 1
            with open(stats_file, "a") as fh:
                fh.write(json.dumps({"doc_id": doc_id, "ok": r["ok"], "pages": written,
                                     "latency_s": round(r["latency_s"], 2), "prompt_chars": r["prompt_chars"],
                                     "output_chars": r["output_chars"], "error": r["error"]}) + "\n")
            print(f"[{applied}/100] {doc_id} ok={r['ok']} pages={len(written)} {r['latency_s']:.0f}s", flush=True)
            if applied in SNAPSHOTS:
                dst = ROOT / "results/snapshots" / SNAPSHOTS[applied]
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(wiki, dst)
                print(f"snapshot saved: {dst}", flush=True)
    print(f"ingest wall-clock total (this run): {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
