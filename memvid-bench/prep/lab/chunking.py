"""Chunking variants.

All chunks carry (doc_id, start, end) char offsets into the canonical doc text,
so gold evidence spans can be matched across chunking strategies.

fixed:   200-word sliding windows, 40-word overlap, ignores structure.
struct:  splits on section boundaries, packs paragraphs up to 200 words,
         merges small adjacent sections. Tables and code blocks stay whole
         when they fit.
fixedh:  fixed windows with the ctx-style header (control for ctx vs fixed).
ctx:     same chunks as struct, but the indexed text is prefixed with
         'Title > heading path' (contextual chunk header).
"""
import json, re
from pathlib import Path

MAX_WORDS = 200
OVERLAP = 40


def load_docs(path="data/corpus/docs.jsonl"):
    return [json.loads(l) for l in open(path)]


def canonical(doc):
    """Return canonical text and section offsets."""
    parts, offs, pos = [], [], 0
    for s in doc["sections"]:
        offs.append((pos, pos + len(s["text"]), s["heading"], s.get("page")))
        parts.append(s["text"])
        pos += len(s["text"]) + 2
    return "\n\n".join(parts), offs


def _word_spans(text, base=0):
    return [(m.start() + base, m.end() + base) for m in re.finditer(r"\S+", text)]


def fixed_chunks(doc):
    text, offs = canonical(doc)
    ws = _word_spans(text)
    out, i, n = [], 0, 0
    while i < len(ws):
        j = min(i + MAX_WORDS, len(ws))
        st, en = ws[i][0], ws[j - 1][1]
        sec = next((o for o in offs if o[0] <= st < o[1] + 2), None)
        page = sec[3] if sec else None
        out.append({"chunk_id": f"{doc['doc_id']}#f{n}", "doc_id": doc["doc_id"], "start": st, "end": en,
                    "text": text[st:en], "heading": "", "page": page,
                    "start_heading": sec[2] if sec else ""})
        n += 1
        if j == len(ws):
            break
        i = j - OVERLAP
    return out


def struct_chunks(doc):
    text, offs = canonical(doc)
    out, n = [], 0
    pending = None  # (start, end, heading, page, words)

    def emit(st, en, heading, page):
        nonlocal n
        out.append({"chunk_id": f"{doc['doc_id']}#s{n}", "doc_id": doc["doc_id"], "start": st, "end": en,
                    "text": text[st:en], "heading": heading, "page": page})
        n += 1

    for (s0, s1, heading, page) in offs:
        # paragraph spans inside the section
        paras = [(m.start() + s0, m.end() + s0) for m in re.finditer(r"(?:.+\n?)+", text[s0:s1])]
        groups, cur, cw = [], None, 0
        for (p0, p1) in paras:
            w = len(text[p0:p1].split())
            if w > MAX_WORDS:  # split long paragraph by words
                if cur: groups.append(cur); cur, cw = None, 0
                ws = _word_spans(text[p0:p1], p0)
                for k in range(0, len(ws), MAX_WORDS - OVERLAP):
                    seg = ws[k:k + MAX_WORDS]
                    groups.append([seg[0][0], seg[-1][1], len(seg)])
                    if k + MAX_WORDS >= len(ws): break
                continue
            if cur and cw + w > MAX_WORDS:
                groups.append(cur); cur, cw = None, 0
            cur = [p0, p1, 0] if cur is None else [cur[0], p1, 0]
            cw += w; cur[2] = cw
        if cur: groups.append(cur)
        for g in groups:
            if pending and pending[4] + g[2] <= MAX_WORDS and pending[4] < 60:
                pending = (pending[0], g[1], pending[2], pending[3], pending[4] + g[2])
                continue
            if pending: emit(*pending[:4])
            pending = (g[0], g[1], heading, page, g[2])
    if pending: emit(*pending[:4])
    return out


def build(variant, docs):
    titles = {d["doc_id"]: d["title"] for d in docs}
    chunks = []
    for d in docs:
        cs = fixed_chunks(d) if variant in ("fixed", "fixedh") else struct_chunks(d)
        for c in cs:
            if variant == "fixedh":
                # control: fixed windows + the same kind of header (title > heading of the section the window starts in)
                sh = c.get("start_heading", "")
                head = sh if sh and sh != titles[d["doc_id"]] else ""
                c["index_text"] = f"{titles[d['doc_id']]}" + (f" > {head}" if head else "") + "\n" + c["text"]
            elif variant == "ctx":
                head = c["heading"] if c["heading"] and c["heading"] != titles[d["doc_id"]] else ""
                c["index_text"] = f"{titles[d['doc_id']]}" + (f" > {head}" if head else "") + "\n" + c["text"]
            else:
                c["index_text"] = c["text"]
            chunks.append(c)
    return chunks
