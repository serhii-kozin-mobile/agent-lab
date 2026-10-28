"""Shared helpers: LLM calls (answerer = Claude Sonnet via claude CLI, judge = claude-llm/Opus),
usage logging, token estimation (chars/4, stated in REPORT.md).

muse exec note: nested `muse exec` fails in this sandbox (session-writer lease on a
read-only FS + unreadable user settings; verified 2026-10-27, see REPORT.md). The answerer
therefore uses the same Claude CLI family with a pinned model id, run from an empty temp
dir with tools disabled, prompt on stdin. The judge is ~/bin/claude-llm (Opus), a
different model, as required.
"""
import json, os, subprocess, tempfile, time
from pathlib import Path

ROOT = Path(".")
RESULTS = ROOT / "results"
USAGE_LOG = RESULTS / "llm_usage.jsonl"
ANSWERER_MODEL = "claude-sonnet-4-5"
JUDGE_CMD = str(Path.home() / "bin" / "claude-llm")


def est_tokens(chars: int) -> int:
    return int(round(chars / 4))


def _log_usage(rec: dict):
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(USAGE_LOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


def call_answerer(prompt: str, tag: str, effort: str = "low", timeout: int = 300) -> dict:
    """One answerer call. Returns {text, latency_s, prompt_chars, output_chars, ok, error}."""
    env = dict(os.environ)
    env["PATH"] = subprocess.run(["bash", "-lc", "npm prefix -g"], capture_output=True, text=True).stdout.strip() + "/bin:" + env.get("PATH", "")
    # auth: the claude CLI uses whatever login/env the user already has configured
    t0 = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="llmwiki_ans_") as td:
            p = subprocess.run(
                ["claude", "-p", "--model", ANSWERER_MODEL, "--effort", effort,
                 "--disallowedTools", "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,NotebookEdit"],
                input=prompt, capture_output=True, text=True, timeout=timeout, cwd=td, env=env)
        dt = time.time() - t0
        text = (p.stdout or "").strip()
        ok = p.returncode == 0 and len(text) > 0
        err = (p.stderr or "")[:500] if not ok else ""
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        text, ok, err = "", False, f"{type(e).__name__}: {e}"[:500]
    rec = {"tag": tag, "model": ANSWERER_MODEL, "latency_s": round(dt, 2),
           "prompt_chars": len(prompt), "output_chars": len(text),
           "prompt_tokens_est": est_tokens(len(prompt)), "output_tokens_est": est_tokens(len(text)), "ok": ok}
    _log_usage(rec)
    return {"text": text, "latency_s": dt, "prompt_chars": len(prompt), "output_chars": len(text), "ok": ok, "error": err}


def call_judge(prompt: str, tag: str, timeout: int = 600) -> dict:
    """One judge call via ~/bin/claude-llm (Claude Opus, different model from answerer)."""
    t0 = time.time()
    try:
        p = subprocess.run([JUDGE_CMD, prompt], capture_output=True, text=True, timeout=timeout)
        dt = time.time() - t0
        text = (p.stdout or "").strip()
        ok = p.returncode == 0 and len(text) > 0
        err = (p.stderr or "")[:500] if not ok else ""
    except Exception as e:  # noqa: BLE001
        dt = time.time() - t0
        text, ok, err = "", False, f"{type(e).__name__}: {e}"[:500]
    rec = {"tag": tag, "model": "claude-opus-5-5 (claude-llm)", "latency_s": round(dt, 2),
           "prompt_chars": len(prompt), "output_chars": len(text),
           "prompt_tokens_est": est_tokens(len(prompt)), "output_tokens_est": est_tokens(len(text)), "ok": ok}
    _log_usage(rec)
    return {"text": text, "latency_s": dt, "prompt_chars": len(prompt), "output_chars": len(text), "ok": ok, "error": err}
