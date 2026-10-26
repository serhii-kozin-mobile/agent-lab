"""Ask a small local open model (Qwen2.5-3B-Instruct, Q4_K_M GGUF, CPU) for H/D/A probabilities.
Usage: python run_llm.py <condition: stats|stats_odds> <temperature> <seed> [limit]
Resumable: appends to out/llm_<condition>_t<temp>_s<seed>.jsonl"""
import json, sys, time, pathlib, os
from llama_cpp import Llama
cond, temp, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None
MODEL = os.environ.get("MODEL", "models/qwen2.5-3b-instruct-q4_k_m.gguf")
M = json.load(open("out/matches_2526.json"))["matches"][:limit]
out = pathlib.Path(f"out/llm_{cond}_t{temp}_s{seed}.jsonl")
done = {json.loads(l)["i"] for l in out.open()} if out.exists() else set()
llm = Llama(model_path=MODEL, n_ctx=1024, n_threads=int(os.environ.get("THREADS", "3")), n_threads_batch=int(os.environ.get("THREADS", "3")), seed=seed, verbose=False)
SYS = ("You are a football forecaster. Give calibrated probabilities for the full-time result of a "
       "Premier League match. Reply only with JSON: {\"home\": int, \"draw\": int, \"away\": int} in percent, summing to 100.")
schema = {"type": "object", "properties": {k: {"type": "integer"} for k in ("home", "draw", "away")},
          "required": ["home", "draw", "away"]}
for m in M:
    if m["i"] in done: continue
    u = (f"Match date: {m['date']}. Home: {m['home']}. Away: {m['away']}.\n"
         f"{m['home']}: {m['home_desc']}.\n{m['away']}: {m['away_desc']}.\n")
    if cond == "stats_odds":
        o = m["odds_open"]
        u += f"Average bookmaker decimal odds before kickoff: home {o[0]}, draw {o[1]}, away {o[2]}.\n"
    u += "Probabilities for home win, draw, away win?"
    t0 = time.time()
    r = llm.create_chat_completion(messages=[{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                                   temperature=temp, max_tokens=40, seed=seed,
                                   response_format={"type": "json_object", "schema": schema})
    txt = r["choices"][0]["message"]["content"]
    try:
        j = json.loads(txt); raw = [max(0, j["home"]), max(0, j["draw"]), max(0, j["away"])]
        s = sum(raw); p = [x / s for x in raw] if s > 0 else None
    except Exception:
        raw, p = None, None
    rec = dict(i=m["i"], raw=raw, p=p, text=txt, secs=round(time.time() - t0, 2),
               prompt_tokens=r["usage"]["prompt_tokens"], completion_tokens=r["usage"]["completion_tokens"])
    with out.open("a") as f: f.write(json.dumps(rec) + "\n")
    print(m["i"], m["home"], m["away"], raw, rec["secs"], flush=True)
