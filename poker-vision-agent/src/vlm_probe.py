"""Small local open VLM as the 'eyes' (SmolVLM-256M-Instruct, CPU, free).

Asks the model to read the same screenshots the CV pipeline reads and scores
each field. Separate venv (.venv-vlm) because torch is heavy; optional."""
from __future__ import annotations

import json
import os
import re
import sys
import time
import warnings

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from noise import CONDITIONS
from perception_bench import collect_states
from seeds import stable_seed
from table import render

warnings.filterwarnings("ignore")
torch.set_num_threads(int(os.environ.get("VLM_THREADS", "2")))
MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
PROMPT = ("This is a screenshot of a heads-up poker table. The player at the bottom is the hero. "
          "Answer with JSON only, keys: hero_cards (list like [\"Ah\",\"Td\"]), board (list), pot (int), "
          "hero_stack (int), villain_stack (int), hero_bet (int), villain_bet (int), button (\"hero\" or \"villain\").")
CARD = re.compile(r"\b(10|[2-9TJQKA])\s*([shdc♠♥♦♣])", re.I)
SUITMAP = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}


def parse(txt):
    out = {}
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            out = json.loads(m.group(0))
        except Exception:
            out = {}
    def cards(v):
        if not isinstance(v, list):
            return None
        res = []
        for c in v:
            mm = CARD.search(str(c))
            if mm:
                r = "T" if mm.group(1) == "10" else mm.group(1).upper()
                res.append(r + SUITMAP.get(mm.group(2), mm.group(2).lower()))
        return res
    return dict(hero_cards=cards(out.get("hero_cards")), board=cards(out.get("board")),
                **{k: (int(out[k]) if str(out.get(k, "")).isdigit() else None)
                   for k in ("pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet")},
                button=out.get("button"))


def main(n, conds, out):
    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.float32).eval()
    states = collect_states(n, seed=11)
    rows = []
    for cond in conds:
        theme, fn = CONDITIONS[cond]
        for k, (ts, off, i, d) in enumerate(states):
            img = fn(render(ts, theme, off), np.random.default_rng(stable_seed(11, "bench", k, cond)))
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, add_generation_prompt=True)
            inp = proc(text=text, images=[img], return_tensors="pt")
            t = time.perf_counter()
            with torch.no_grad():
                g = model.generate(**inp, max_new_tokens=120, do_sample=False)
            ms = (time.perf_counter() - t) * 1000
            ans = proc.batch_decode(g[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            seen, true = parse(ans), ts.as_dict()
            row = dict(condition=cond, k=k, ms=ms, raw=ans.strip()[:400])
            for f in true:
                row[f"ok_{f}"] = seen[f] == true[f]
            tc = true["hero_cards"] + true["board"]
            sc = (seen["hero_cards"] or []) + (seen["board"] or [])
            row["cards_total"] = len(tc)
            row["cards_correct"] = sum(1 for a in tc if a in sc)
            row["state_correct"] = all(row[f"ok_{f}"] for f in true)
            rows.append(row)
            print(cond, k, f"{ms:.0f}ms", ans.strip()[:120].replace("\n", " "), flush=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2].split(","), sys.argv[3])
