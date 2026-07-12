"""
KV cache tax — eviction leg (rotating/windowed KV vs. the v1 full-KV baseline).

Same model, same 2k/8k/16k/32k sweep, same thermal discipline as run_m3.py.
The only variable: prompt_cache is built as RotatingKVCache(max_size=WINDOW,
keep=KEEP) per layer instead of the default unbounded KVCache.

Llama-3.1-8B defines its own make_cache() (see mlx_lm/models/llama.py), which
only swaps in RotatingKVCache for layers with native sliding-window attention
-- none of which this model has. That means make_prompt_cache(model,
max_kv_size=N) is a silent no-op for this checkpoint (this is exactly the bug
mlx-lm PR #1343 fixes). So the cache is built directly here instead of going
through make_prompt_cache().

Question this answers: what recall do you trade to buy back memory, at a
fixed window size, as context grows past that window?

RUN UNDER caffeinate:
  caffeinate -dimsu <venv>/bin/python run_m3_eviction.py
  (smoke test one length: add  --lengths 2000)
"""

import time, json, argparse, os, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import RotatingKVCache

from run_m3 import MODEL, FACTS, build_prompt

WINDOW = 4096
KEEP = 4

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_m3_eviction_raw.jsonl")
OUT = os.path.join(HERE, "results_m3_eviction.jsonl")


def one_run(model, tok, target, cold):
    mx.reset_peak_memory()
    ids, actual = build_prompt(tok, target)
    cache = [RotatingKVCache(max_size=WINDOW, keep=KEEP) for _ in model.layers]
    rec = {
        "stack": "mlx-m3-rotating",
        "window": WINDOW,
        "keep": KEEP,
        "target": target,
        "context_tokens": actual,
        "cold_discarded": cold,
        "oom": False,
    }
    try:
        t0 = time.perf_counter()
        ttft = None
        text = ""
        last = None
        for r in stream_generate(
            model, tok, ids, max_tokens=64, prompt_cache=cache
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += r.text
            last = r
        recalled = [code in text for _, code in FACTS]
        rec.update(
            ttft_s=round(ttft, 3),
            decode_tps=round(last.generation_tps, 2),
            gen_tokens=last.generation_tokens,
            peak_gb=round(mx.get_peak_memory() / 1e9, 3),
            facts_recalled=sum(recalled),
            recalled_which=recalled,
        )
    except Exception as e:
        rec.update(
            oom=True,
            peak_gb=round(mx.get_peak_memory() / 1e9, 3),
            failure=f"{type(e).__name__}: {str(e)[:160]}",
        )
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2000, 8000, 16000, 32000])
    args = ap.parse_args()
    L = args.lengths

    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)
    print(f"window={WINDOW} keep={KEEP}")

    import random

    def shuffled(seed):
        x = L[:]
        random.Random(seed).shuffle(x)
        return x

    passes = [
        ("warmup", shuffled(1)),
        ("p1", shuffled(2)),
        ("p2", shuffled(3)),
        ("p3", shuffled(4)),
    ]

    open(RAW, "w").close()
    for pname, order in passes:
        cold = pname == "warmup"
        print(f"\n### pass {pname}  order={order}{'  (DISCARDED)' if cold else ''}")
        for target in order:
            rec = one_run(model, tok, target, cold)
            rec["pass"] = pname
            with open(RAW, "a") as f:
                f.write(json.dumps(rec) + "\n")
            if rec["oom"]:
                print(f"  {target}: OOM at {rec['context_tokens']} tok, peak {rec['peak_gb']}GB")
            else:
                print(
                    f"  {target}: ctx {rec['context_tokens']} | TTFT {rec['ttft_s']}s | "
                    f"{rec['decode_tps']} tok/s | peak {rec['peak_gb']}GB | "
                    f"recall {rec['facts_recalled']}/5 {rec['recalled_which']}"
                    + ("  [warmup, discarded]" if cold else "")
                )

    raw = [json.loads(l) for l in open(RAW)]
    rec_only = [r for r in raw if not r["cold_discarded"] and not r["oom"]]
    agg = []
    for target in L:
        s = [r for r in rec_only if r["target"] == target]
        if not s:
            oomed = [r for r in raw if r["target"] == target and r["oom"]]
            agg.append(
                {
                    "target": target,
                    "oom": True,
                    "n": 0,
                    "peak_gb": oomed[0]["peak_gb"] if oomed else None,
                }
            )
            continue

        def ms(k):
            v = [r[k] for r in s]
            return {"median": round(statistics.median(v), 3), "min": min(v), "max": max(v)}

        agg.append(
            {
                "target": target,
                "context_tokens": s[0]["context_tokens"],
                "n": len(s),
                "oom": False,
                "window": WINDOW,
                "keep": KEEP,
                "ttft_s": ms("ttft_s"),
                "decode_tps": ms("decode_tps"),
                "peak_gb": ms("peak_gb"),
                "recall": f"{s[0]['facts_recalled']}/5",
                "recalled_which": s[0]["recalled_which"],
            }
        )
    with open(OUT, "w") as f:
        for a in agg:
            f.write(json.dumps(a) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
