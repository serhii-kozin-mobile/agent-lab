"""Build a normalized corpus (docs.jsonl) from three public sources.

Each doc is a list of sections. A section has a heading path, text, and an
optional page number. Tables are flattened to 'col: value' rows so that
keyword and dense retrieval can both see the cell text.
"""
import json, re, glob, os, html
from pathlib import Path
from bs4 import BeautifulSoup
from pypdf import PdfReader

RAW = Path("data/raw")
OUT = Path("data/corpus/docs.jsonl")

PY_DIRS = ["library", "howto", "tutorial", "faq"]
MCP_REVS = ["2025-06-18", "2025-11-25", "2026-07-28"]


def clean(t):
    t = html.unescape(t)
    t = t.replace("\u00b6", "").replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n\n", t)
    return t.strip()


def table_to_text(tbl):
    rows = []
    for tr in tbl.find_all("tr"):
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["td", "th"])]
        if cells:
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def html_doc_full(path, rel):
    # Simpler, more faithful approach: walk text in document order, split on headings.
    soup = BeautifulSoup(open(path, encoding="utf-8").read(), "lxml")
    body = soup.select_one("div.body") or soup.body
    for bad in body.select("a.headerlink, script, style"):
        bad.decompose()
    for tbl in body.find_all("table"):
        tbl.replace_with(soup.new_string("\n[TABLE]\n" + table_to_text(tbl) + "\n[/TABLE]\n"))
    for h in body.find_all(["h1", "h2", "h3", "h4"]):
        h.replace_with(soup.new_string(f"\n@@H{h.name[1]}@@{h.get_text(' ', strip=True)}\n"))
    for br in body.find_all(["p", "pre", "li", "dt", "dd", "div"]):
        br.append(soup.new_string("\n"))
    text = body.get_text("")
    title = soup.title.get_text().split(" \u2014 ")[0].strip() if soup.title else rel
    sections, stack, buf = [], [], []
    def flush():
        t = clean("\n".join(buf))
        if t:
            sections.append({"heading": " > ".join(stack) or title, "text": t, "page": None})
        buf.clear()
    for line in text.split("\n"):
        m = re.match(r"@@H(\d)@@(.*)", line)
        if m:
            flush(); lvl = int(m.group(1)); stack[:] = stack[: lvl - 1]; stack.append(clean(m.group(2)))
        else:
            buf.append(line)
    flush()
    return title, sections


def mdx_doc(path):
    raw = open(path, encoding="utf-8").read()
    title = None
    fm = re.match(r"---\n(.*?)\n---\n", raw, re.S)
    if fm:
        m = re.search(r"^title:\s*\"?(.*?)\"?$", fm.group(1), re.M)
        title = m.group(1) if m else None
        raw = raw[fm.end():]
    raw = re.sub(r"^import .*$", "", raw, flags=re.M)
    raw = re.sub(r"</?[A-Z][A-Za-z]*[^>]*>", "", raw)  # strip JSX components
    sections, stack, buf, in_code = [], [], [], False
    def flush():
        t = clean("\n".join(buf))
        if t:
            sections.append({"heading": " > ".join([title or ""] + stack).strip(" >"), "text": t, "page": None})
        buf.clear()
    for line in raw.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush(); lvl = len(m.group(1)); stack[:] = stack[: lvl - 1]; stack.append(m.group(2).strip())
        else:
            buf.append(line)
    flush()
    return title or Path(path).stem, sections


def pdf_doc(path):
    r = PdfReader(path)
    title = (r.metadata.title if r.metadata and r.metadata.title else Path(path).stem)
    sections = []
    for i, p in enumerate(r.pages):
        t = clean(p.extract_text() or "")
        if len(t) > 40:
            sections.append({"heading": f"{Path(path).stem} p.{i+1}", "text": t, "page": i + 1})
    return title, sections


def main():
    docs = []
    base = RAW / "py/python-3.13-docs-html"
    for d in PY_DIRS:
        for f in sorted(glob.glob(str(base / d / "*.html"))):
            rel = os.path.relpath(f, base)
            title, secs = html_doc_full(f, rel)
            if secs:
                docs.append({"doc_id": f"py:{rel}", "source": "python-docs-3.13", "format": "html", "title": title, "sections": secs})
    for rev in MCP_REVS:
        root = RAW / "mcp_repo/docs/specification" / rev
        for f in sorted(glob.glob(str(root / "**/*.mdx"), recursive=True)):
            rel = os.path.relpath(f, root)
            if rel == "schema.mdx":
                continue  # generated reference, very large
            title, secs = mdx_doc(f)
            if secs:
                docs.append({"doc_id": f"mcp:{rev}/{rel}", "source": f"mcp-spec-{rev}", "format": "mdx", "title": f"{title} (MCP {rev})", "sections": secs})
    for f in sorted(glob.glob(str(RAW / "nist/*.pdf"))):
        title, secs = pdf_doc(f)
        docs.append({"doc_id": f"nist:{Path(f).name}", "source": "nist", "format": "pdf", "title": title, "sections": secs})
    with open(OUT, "w") as fh:
        for d in docs:
            fh.write(json.dumps(d) + "\n")
    from collections import Counter
    c = Counter(d["format"] for d in docs)
    chars = sum(len(s["text"]) for d in docs for s in d["sections"])
    print(f"docs={len(docs)} by_format={dict(c)} chars={chars}")

if __name__ == "__main__":
    main()
