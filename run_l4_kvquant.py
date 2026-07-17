"""
KV cache tax -- quantized-KV leg, L4/vLLM side (the cross-hardware answer to the
M3 finding).

THE QUESTION: on M3/MLX, quantized KV (--kv-bits) was all cost and no benefit --
it RAISED peak RAM (quantized SDPA materializes the full score matrix), kv8
decoded ~4x slower, and it OOM'd at 32k where fp16 fit. Is that finding about
KV quantization, or about MLX's implementation of it? vLLM treats fp8 KV cache
as a first-class feature (--kv-cache-dtype fp8): halves KV bytes/token, so the
same reservation holds ~2x the tokens. Same model, same transcript, same
lengths, actively-cooled GPU -- if quantized KV works ANYWHERE, it's here.

TWO ARMS, one box, same session (restart the server between arms):
  arm "fp16": vLLM default KV dtype ("auto" -> model dtype) -- the baseline.
  arm "fp8":  --kv-cache-dtype fp8  (E4M3 by default on CUDA; RECORD the exact
              dtype line from the startup log).
Run this client once per arm with KVTAX_KV_DTYPE set to the arm name. Records
append to results_l4_kvquant_raw.jsonl; each client run also (re)writes its
arm's aggregated rows into results_l4_kvquant.jsonl (other arm's rows kept).

Metrics per (arm, length): TTFT, decode tok/s, 5-fact recall -- plus, per ARM,
the vLLM startup log's KV-cache capacity line ("GPU KV cache size: N tokens" /
"# GPU blocks"), which is the memory result: same reservation, how many tokens
fit. Capacity should ~double under fp8; whether decode pays for it is the point.

Mirrors run_l4.py's rigor: warmup pass discarded (JIT/compile), 3 recorded
passes, interleaved length order, temperature=0, max_tokens=64, prefix caching
OFF, one request at a time. Paired cold/warm alternation is NOT needed here --
the two arms can't share a process anyway (different server flags), the L4 is
actively cooled (v1: 3 passes agreed to 2 decimals), and each arm gets its own
within-arm medians; the comparison is arm-median vs arm-median with [min,max]
shown.

ON THE BOX (see L4_RUNBOOK_KVQUANT.md for the full paste-in-order sequence):
  export KVTAX_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
  export KVTAX_GPU="L4-24GB"
  export KVTAX_KV_DTYPE="fp16"     # then "fp8" for the second arm
  python -u run_l4_kvquant.py
"""
import os, time, json, statistics
from openai import OpenAI
from transformers import AutoTokenizer

MODEL = os.environ.get("KVTAX_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
GPU = os.environ.get("KVTAX_GPU", "UNKNOWN-GPU")
BASE = os.environ.get("KVTAX_BASE", "http://localhost:8000/v1")
KV_DTYPE = os.environ.get("KVTAX_KV_DTYPE")
if KV_DTYPE not in ("fp16", "fp8"):
    raise SystemExit("set KVTAX_KV_DTYPE=fp16 or fp8 (must match the running server's flag)")
LENGTHS = [2000, 8000, 16000, 32000]
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_l4_kvquant_raw.jsonl")
OUT = os.path.join(HERE, "results_l4_kvquant.jsonl")

# identical transcript content to run_m3.py / run_l4.py so contexts match
FACTS = [("Aurora", "QX-4471"), ("Basalt", "ZR-9028"), ("Cinder", "MP-1357"),
         ("Dovewing", "KT-6604"), ("Ember", "VN-8813")]
QUESTION = ("List the access codes for all five vaults mentioned above, in order "
            "(Aurora, Basalt, Cinder, Dovewing, Ember). Answer with just the five codes.")
FILLER = (
    "The facilities report continued with routine notes about the north campus. "
    "Maintenance logged the quarterly inspection of the ventilation ducts and the "
    "backup generators, both of which passed without incident. The grounds team "
    "trimmed the hedges along the east walkway and repainted the loading-bay lines. "
    "A minor water-pressure fluctuation in Building C was traced to a partially "
    "closed valve and corrected the same afternoon. Nothing in this section requires "
    "escalation, and the summary is provided only for completeness of the record. ")

tok = AutoTokenizer.from_pretrained(MODEL)
_FILLER_TOK = len(tok(FILLER)["input_ids"])
_MSG_CACHE = {}


def build_messages(target):
    """One-pass build (the O(n^2) loop was a v1 bug the L4 exposed)."""
    if target in _MSG_CACHE:
        return _MSG_CACHE[target]
    n_blocks = max(len(FACTS) * 3 + 2, int(target / max(_FILLER_TOK, 1)) + 4)
    parts, fi = [], 0
    for b in range(n_blocks):
        parts.append(FILLER)
        if fi < len(FACTS) and b > 0 and b % 3 == 0:
            n, c = FACTS[fi]
            parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. ")
            fi += 1
    while fi < len(FACTS):
        n, c = FACTS[fi]
        parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. ")
        fi += 1
    msgs = [{"role": "user", "content": "".join(parts) + "\n\n" + QUESTION}]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    n_tok = len(tok(text)["input_ids"])
    _MSG_CACHE[target] = (msgs, n_tok)
    return msgs, n_tok


def one_run(client, target):
    msgs, ctx = build_messages(target)
    rec = {"stack": "vllm-l4", "gpu": GPU, "model": MODEL, "kv_dtype": KV_DTYPE,
           "target": target, "context_tokens": ctx}
    t0 = time.perf_counter()
    ttft = None
    text = ""
    n = 0
    usage_tokens = None
    stream = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0,
                                            max_tokens=64, stream=True,
                                            stream_options={"include_usage": True})
    for chunk in stream:
        if chunk.usage is not None:
            usage_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        d = chunk.choices[0].delta.content or ""
        if d and ttft is None:
            ttft = time.perf_counter() - t0
        text += d
        n += 1
    total = time.perf_counter() - t0
    # Token count, not chunk count (v1's known stream-chunk caveat, fixed here
    # because chunking could differ between the fp16 and fp8 arms): prefer the
    # server's own usage.completion_tokens; fall back to tokenizing the
    # accumulated text.
    gen_tokens = usage_tokens if usage_tokens else len(tok(text)["input_ids"])
    rec.update(ttft_s=round(ttft, 3) if ttft else None,
               decode_tps=(round((gen_tokens - 1) / (total - ttft), 2)
                           if ttft and total > ttft and gen_tokens > 1 else None),
               gen_tokens=gen_tokens, gen_tokens_source="usage" if usage_tokens else "retokenized",
               gen_chunks=n, facts_recalled=sum(1 for _, c in FACTS if c in text))
    return rec


OOM_MARKERS = ("out of memory", "outofmemory", "kv cache", "no available memory",
               "insufficient memory", "cuda error", "resource exhausted")


def classify_failure(e):
    """OOM only when the error actually says so -- a timeout or connection drop
    recorded as OOM would publish a false memory conclusion."""
    msg = f"{type(e).__name__}: {str(e)[:300]}"
    is_oom = any(m in msg.lower() for m in OOM_MARKERS)
    return is_oom, msg


def main():
    client = OpenAI(base_url=BASE, api_key="EMPTY")
    # Fail fast if the client's arm tag and the server don't agree on port:
    # (can't introspect the server's kv dtype over the API -- the runbook makes
    # the operator record the startup-log line per arm instead)
    client.models.list()

    import random
    def sh(seed):
        x = LENGTHS[:]
        random.Random(seed).shuffle(x)
        return x
    passes = [("warmup", sh(1), True), ("p1", sh(2), False), ("p2", sh(3), False), ("p3", sh(4), False)]

    raw = []
    for pname, order, cold in passes:
        print(f"\n### [{KV_DTYPE}] {pname} order={order}{' (discard)' if cold else ''}", flush=True)
        for target in order:
            try:
                r = one_run(client, target)
                r["pass"] = pname
                r["cold_discarded"] = cold
                r["oom"] = False
                print(f"  {target}: ctx {r['context_tokens']} | TTFT {r['ttft_s']}s | "
                      f"{r['decode_tps']} tok/s | recall {r['facts_recalled']}/5", flush=True)
            except Exception as e:
                is_oom, msg = classify_failure(e)
                r = {"stack": "vllm-l4", "kv_dtype": KV_DTYPE, "target": target, "pass": pname,
                     "cold_discarded": cold, "oom": is_oom, "failed": True, "failure": msg}
                print(f"  {target}: FAIL{' (OOM)' if is_oom else ''} {msg}", flush=True)
            raw.append(r)
            with open(RAW, "a") as f:
                f.write(json.dumps(r) + "\n")

    # Aggregate THIS arm; keep the other arm's rows in OUT if present.
    rec = [r for r in raw if not r["cold_discarded"] and not r.get("failed")]
    failed = [r for r in raw if not r["cold_discarded"] and r.get("failed")]
    n_expected = sum(1 for _, _, cold in passes if not cold)
    other = []
    if os.path.exists(OUT):
        other = [json.loads(l) for l in open(OUT) if json.loads(l).get("kv_dtype") != KV_DTYPE]
    rows = []
    for target in LENGTHS:
        s = [r for r in rec if r["target"] == target]
        fails = [r for r in failed if r["target"] == target]
        n_oom = sum(1 for r in fails if r["oom"])
        if not s:
            rows.append({"kv_dtype": KV_DTYPE, "target": target, "all_failed": True,
                         "n_ok": 0, "n_expected": n_expected,
                         "n_oom": n_oom, "n_other_failures": len(fails) - n_oom,
                         "failures": [r["failure"] for r in fails]})
            continue
        def ms(k):
            v = [r[k] for r in s if r[k] is not None]
            return ({"median": round(statistics.median(v), 3), "min": min(v), "max": max(v)}
                    if v else None)
        # Recall across ALL recorded passes (min/median/max), not just the
        # first -- degradation across passes is exactly what this experiment
        # exists to detect.
        rv = sorted(r["facts_recalled"] for r in s)
        rows.append({"gpu": GPU, "model": MODEL, "kv_dtype": KV_DTYPE, "target": target,
                     "context_tokens": s[0]["context_tokens"], "n_ok": len(s),
                     "n_expected": n_expected, "incomplete": len(s) < n_expected,
                     "n_oom": n_oom, "n_other_failures": len(fails) - n_oom,
                     "ttft_s": ms("ttft_s"), "decode_tps": ms("decode_tps"),
                     "recall": {"median": statistics.median(rv), "min": rv[0], "max": rv[-1],
                                "of": 5}})
    with open(OUT, "w") as f:
        for r in other + rows:
            f.write(json.dumps(r) + "\n")

    print(f"\n=== [{KV_DTYPE}] aggregated ({n_expected} recorded passes) ===", flush=True)
    for r in rows:
        if r.get("all_failed"):
            print(f"  {r['target']}: ALL FAILED ({r['n_oom']} OOM, "
                  f"{r['n_other_failures']} other) -- see failures[] in {OUT}", flush=True)
            continue
        flag = "  <- INCOMPLETE" if r["incomplete"] else ""
        rc = r["recall"]
        rc_str = (f"{rc['median']:g}/5" if rc["min"] == rc["max"]
                  else f"{rc['median']:g}/5 [{rc['min']},{rc['max']}]")
        print(f"  {r['target']}: TTFT {r['ttft_s']['median']}s | decode "
              f"{r['decode_tps']['median']} tok/s | recall {rc_str} | "
              f"n={r['n_ok']}/{r['n_expected']}{flag}", flush=True)
    print(f"\nwrote {OUT} (this arm) + raw appended to {RAW}.\n"
          f"NOW: save this arm's server startup log (the 'GPU KV cache size' / "
          f"'# GPU blocks' + 'kv cache dtype' lines) -- that's the capacity half "
          f"of the result.", flush=True)


if __name__ == "__main__":
    main()
