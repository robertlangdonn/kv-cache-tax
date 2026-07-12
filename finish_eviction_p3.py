"""
Finish the interrupted p3 pass (8000, 32000 only -- 2000 and 16000 already
recorded) and append to the existing raw log, then aggregate.

The original run_m3_eviction.py hit severe swap pressure (90% of 8GB swap
used) after ~14 consecutive one_run() calls without releasing MLX's internal
buffer cache between runs. This adds an explicit mx.clear_cache() after each
run to avoid repeating that.
"""

import json, os, statistics
import mlx.core as mx
from mlx_lm import load

from run_m3_eviction import MODEL, WINDOW, KEEP, one_run

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_m3_eviction_raw.jsonl")
OUT = os.path.join(HERE, "results_m3_eviction.jsonl")

MISSING = [8000, 32000]

print(f"loading {MODEL} ...")
model, tok = load(MODEL)

for target in MISSING:
    print(f"p3 {target} ...")
    rec = one_run(model, tok, target, cold=False)
    rec["pass"] = "p3"
    with open(RAW, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if rec["oom"]:
        print(f"  OOM at {rec['context_tokens']} tok, peak {rec['peak_gb']}GB")
    else:
        print(
            f"  ctx {rec['context_tokens']} | TTFT {rec['ttft_s']}s | "
            f"{rec['decode_tps']} tok/s | peak {rec['peak_gb']}GB | "
            f"recall {rec['facts_recalled']}/5 {rec['recalled_which']}"
        )
    mx.clear_cache()

# Re-aggregate the full set now that all 3 recorded passes (p1/p2/p3) exist.
L = [2000, 8000, 16000, 32000]
raw = [json.loads(l) for l in open(RAW)]
rec_only = [r for r in raw if not r["cold_discarded"] and not r["oom"]]
agg = []
for target in L:
    s = [r for r in rec_only if r["target"] == target]
    if not s:
        oomed = [r for r in raw if r["target"] == target and r["oom"]]
        agg.append(
            {"target": target, "oom": True, "n": 0, "peak_gb": oomed[0]["peak_gb"] if oomed else None}
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
for a in agg:
    print(a)
