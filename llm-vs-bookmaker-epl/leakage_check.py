"""Does the model already know 2025-26 results? If it does, its forecasts are memory, not forecasting."""
import json
from llama_cpp import Llama
llm = Llama(model_path=__import__("os").environ.get("MODEL", "models/qwen2.5-3b-instruct-q4_k_m.gguf"), n_ctx=1024, n_threads=2, n_threads_batch=2, verbose=False, seed=0)
Q = ["Who won the 2025-26 English Premier League title?",
     "Which three clubs were promoted to the Premier League for the 2025-26 season?",
     "What was the score of Liverpool vs Bournemouth on 15 August 2025?",
     "Who was relegated from the Premier League at the end of the 2025-26 season?",
     "What is your training data cutoff date?"]
out = []
for q in Q:
    r = llm.create_chat_completion(messages=[{"role": "user", "content": q}], temperature=0, max_tokens=80)
    out.append(dict(q=q, a=r["choices"][0]["message"]["content"])); print(out[-1], flush=True)
json.dump(out, open("out/leakage_check.json", "w"), indent=1)
