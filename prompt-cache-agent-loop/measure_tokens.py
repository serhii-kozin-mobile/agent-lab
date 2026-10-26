"""Turn each trajectory into the sequence of model calls an agent loop actually makes.

Model call k sends every message before the k-th assistant message (system prompt, task,
all earlier assistant turns and tool observations). We count tokens per message with
tiktoken o200k_base as a proxy tokenizer (provider tokenizers differ by a few percent to
tens of percent; the ratios this study reports are what matters).

Output: out/calls.csv (one row per model call) and out/runs.csv (one row per trajectory).
"""
import os, sys, random
import pandas as pd
import tiktoken

SEED = 20261025
N_SWEAGENT = int(os.environ.get("N_SWEAGENT", "2000"))
enc = tiktoken.get_encoding("o200k_base")

def ntok(s):
    return len(enc.encode(s or "", disallowed_special=()))

def calls_from_messages(msgs):
    """msgs: list of (role, text). Returns list of (input_tokens, output_tokens)."""
    calls, running = [], 0
    for role, text in msgs:
        t = ntok(text)
        if role in ("ai", "assistant"):
            calls.append((running, t))
        running += t
    return calls

def load_sweagent():
    p = "data/nebius__SWE-agent-trajectories/data/train-00000-of-00012.parquet"
    df = pd.read_parquet(p)
    idx = list(range(len(df)))
    random.Random(SEED).shuffle(idx)
    for i in idx[:N_SWEAGENT]:
        r = df.iloc[i]
        msgs = []
        for m in r.trajectory:
            text = m.get("system_prompt") if m["role"] == "system" else m.get("text")
            msgs.append((m["role"], text))
        yield "swe-agent", f"{r.instance_id}#{i}", r.model_name, bool(r.target), msgs

def load_openhands():
    p = "data/SWE-Gym__OpenHands-SFT-Trajectories/data/train.success.oss-00000-of-00001.parquet"
    df = pd.read_parquet(p)
    for i, r in df.iterrows():
        msgs = [(m["role"], str(m.get("content") or "")) for m in r.messages]
        yield "openhands", f"oh#{i}", "unknown(SWE-Gym SFT)", True, msgs

if __name__ == "__main__":
    os.makedirs("out", exist_ok=True)
    call_rows, run_rows = [], []
    for loader in (load_sweagent, load_openhands):
        for harness, run_id, model, resolved, msgs in loader():
            calls = calls_from_messages(msgs)
            if not calls:
                continue
            for k, (inp, out) in enumerate(calls, 1):
                call_rows.append(dict(harness=harness, run_id=run_id, call=k, input_tokens=inp, output_tokens=out))
            unique = sum(ntok(t) for _, t in msgs)
            run_rows.append(dict(harness=harness, run_id=run_id, model=model, resolved=resolved,
                                 calls=len(calls), total_input=sum(c[0] for c in calls),
                                 total_output=sum(c[1] for c in calls), unique_tokens=unique,
                                 first_call_input=calls[0][0], last_call_input=calls[-1][0]))
    pd.DataFrame(call_rows).to_csv("out/calls.csv", index=False)
    pd.DataFrame(run_rows).to_csv("out/runs.csv", index=False)
    print("runs", len(run_rows), "calls", len(call_rows))
