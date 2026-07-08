"""
KV cache tax — M3 baseline leg (TIGHTENED: warmed, median-of-3, interleaved).

Measures what growing context costs on a 16 GB M3, one 8B model, mlx-lm.
Rigor (per gpt-5.5 consult 2026-07-07):
  - 1 discarded warmup pass per length (kills Metal JIT-compile artifact)
  - 3 recorded passes, each in a DIFFERENT (interleaved) length order, so no
    length always runs last on a hotter chip (fanless-M3 thermal control)
  - report median [min,max] per length
Per run: TTFT, decode tok/s, peak RAM, OOM (first-class), 5-fact recall.
Raw samples -> results_m3_raw.jsonl ; aggregated table -> results_m3.jsonl

RUN UNDER caffeinate so the machine doesn't nap mid-32k prefill:
  caffeinate -dimsu ~/Work/ondevice-bench/.venv/bin/python run_m3.py
  (smoke test one length: add  --lengths 2000)
Expect ~35-45 min for the full 4-length x 4-pass run (32k prefill is ~6 min each).
"""

import time, json, argparse, os, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_m3_raw.jsonl")
OUT = os.path.join(HERE, "results_m3.jsonl")

FACTS = [("Aurora","QX-4471"),("Basalt","ZR-9028"),("Cinder","MP-1357"),
         ("Dovewing","KT-6604"),("Ember","VN-8813")]
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

# cache built prompts so we don't rebuild the 32k prompt every pass
_PROMPT_CACHE = {}

def build_prompt(tok, target):
    if target in _PROMPT_CACHE:
        return _PROMPT_CACHE[target]
    parts, fi, block = [], 0, 0
    while True:
        parts.append(FILLER)
        if fi < len(FACTS) and block > 0 and block % 3 == 0:
            name, code = FACTS[fi]; parts.append(f"IMPORTANT: The access code for the {name} vault is {code}. "); fi += 1
        block += 1
        msg = [{"role":"user","content":"".join(parts)+"\n\n"+QUESTION}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True)
        if (len(ids) >= target and fi >= len(FACTS)) or block > 4000:
            _PROMPT_CACHE[target] = (ids, len(ids)); return ids, len(ids)

def one_run(model, tok, target, cold):
    mx.reset_peak_memory()
    ids, actual = build_prompt(tok, target)
    rec = {"stack":"mlx-m3","target":target,"context_tokens":actual,"cold_discarded":cold,"oom":False}
    try:
        t0 = time.perf_counter(); ttft=None; text=""; last=None
        for r in stream_generate(model, tok, ids, max_tokens=64):
            if ttft is None: ttft = time.perf_counter()-t0
            text += r.text; last = r
        rec.update(ttft_s=round(ttft,3), decode_tps=round(last.generation_tps,2),
                   gen_tokens=last.generation_tokens, peak_gb=round(mx.get_peak_memory()/1e9,3),
                   facts_recalled=sum(1 for _,c in FACTS if c in text))
    except Exception as e:
        rec.update(oom=True, peak_gb=round(mx.get_peak_memory()/1e9,3),
                   failure=f"{type(e).__name__}: {str(e)[:160]}")
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2000,8000,16000,32000])
    args = ap.parse_args()
    L = args.lengths

    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)

    # interleaved pass orders (Codex): warmup + 3 recorded, each shuffled
    import random
    def shuffled(seed): x=L[:]; random.Random(seed).shuffle(x); return x
    passes = [("warmup", shuffled(1)), ("p1", shuffled(2)), ("p2", shuffled(3)), ("p3", shuffled(4))]

    open(RAW,"w").close()  # fresh raw log
    for pname, order in passes:
        cold = (pname == "warmup")
        print(f"\n### pass {pname}  order={order}{'  (DISCARDED)' if cold else ''}")
        for target in order:
            rec = one_run(model, tok, target, cold)
            rec["pass"] = pname
            with open(RAW,"a") as f: f.write(json.dumps(rec)+"\n")
            if rec["oom"]:
                print(f"  {target}: OOM at {rec['context_tokens']} tok, peak {rec['peak_gb']}GB")
            else:
                print(f"  {target}: ctx {rec['context_tokens']} | TTFT {rec['ttft_s']}s | "
                      f"{rec['decode_tps']} tok/s | peak {rec['peak_gb']}GB | recall {rec['facts_recalled']}/5"
                      + ("  [warmup, discarded]" if cold else ""))

    # aggregate the 3 recorded passes -> median [min,max]
    raw = [json.loads(l) for l in open(RAW)]
    rec_only = [r for r in raw if not r["cold_discarded"] and not r["oom"]]
    agg = []
    for target in L:
        s = [r for r in rec_only if r["target"]==target]
        if not s:  # all OOM'd or failed
            oomed = [r for r in raw if r["target"]==target and r["oom"]]
            agg.append({"target":target,"oom":True,"n":0,
                        "peak_gb": oomed[0]["peak_gb"] if oomed else None}); continue
        def ms(k): v=[r[k] for r in s]; return {"median":round(statistics.median(v),3),"min":min(v),"max":max(v)}
        agg.append({"target":target,"context_tokens":s[0]["context_tokens"],"n":len(s),"oom":False,
                    "ttft_s":ms("ttft_s"),"decode_tps":ms("decode_tps"),"peak_gb":ms("peak_gb"),
                    "recall":f"{s[0]['facts_recalled']}/5"})
    with open(OUT,"w") as f:
        for a in agg: f.write(json.dumps(a)+"\n")

    print(f"\n=== AGGREGATED (median [min,max], n={len(passes)-1} recorded passes) ===")
    print(f"{'ctx':>7} | {'TTFT s med [min,max]':>24} | {'tok/s med':>10} | {'peak GB med':>11} | recall")
    for a in agg:
        if a["oom"]: print(f"{a['target']:>7} | {'OOM':>24} | {'-':>10} | {a.get('peak_gb','-'):>11} | -"); continue
        t,d,p = a["ttft_s"],a["decode_tps"],a["peak_gb"]
        print(f"{a['context_tokens']:>7} | {str(t['median'])+' ['+str(t['min'])+','+str(t['max'])+']':>24} | "
              f"{d['median']:>10} | {p['median']:>11} | {a['recall']}")
    print(f"\nraw -> {RAW}\nclean -> {OUT}")

if __name__ == "__main__":
    main()
