"""
Thermal control probe (Codex-designed, ~8 min).

Proves whether the M3 decode slowdown under sustained load is THERMAL vs a measurement bug,
before we trust "decode is thermally confounded" in the writeup. The test:

    cold 2k  ->  32k (heats the chip)  ->  immediately hot 2k

If hot-2k decode is much slower than cold-2k decode, AND `powermetrics` shows CPU/GPU
frequency dropping (thermal throttle) while swap did NOT spike -> it's thermal, not a bug.
If cold-2k == hot-2k -> the earlier ~2x was a bug/other, investigate before publishing.

Also hardens the decode timer per Codex: mx.eval() to force MLX lazy-eval before stopping
the timer (a classic mlx timing trap), and measures decode separately from prefill.

RUN (in one terminal, plugged in, Low Power Mode OFF):
    caffeinate -dimsu ~/Work/ondevice-bench/.venv/bin/python thermal_control.py
OPTIONAL (second terminal, needs sudo) to capture the throttle directly:
    sudo powermetrics --samplers smc,cpu_power -i 2000 | grep -iE "temp|frequency|throttle"
Then eyeball: did clocks drop during the 32k run? did hot-2k decode drop vs cold-2k?
"""
import time, json, os
import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thermal_control.jsonl")

FACTS=[("Aurora","QX-4471"),("Basalt","ZR-9028"),("Cinder","MP-1357"),("Dovewing","KT-6604"),("Ember","VN-8813")]
Q=("List the access codes for all five vaults mentioned above, in order "
   "(Aurora, Basalt, Cinder, Dovewing, Ember). Answer with just the five codes.")
FILLER=("The facilities report continued with routine notes about the north campus. "
   "Maintenance logged the quarterly inspection of the ventilation ducts and the backup "
   "generators, both of which passed without incident. The grounds team trimmed the hedges "
   "along the east walkway and repainted the loading-bay lines. A minor water-pressure "
   "fluctuation in Building C was traced to a partially closed valve and corrected the same "
   "afternoon. Nothing in this section requires escalation, and the summary is provided only "
   "for completeness of the record. ")

def build(tok, target):
    parts, fi, block = [], 0, 0
    while True:
        parts.append(FILLER)
        if fi < len(FACTS) and block>0 and block%3==0:
            n,c=FACTS[fi]; parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. "); fi+=1
        block+=1
        ids=tok.apply_chat_template([{"role":"user","content":"".join(parts)+"\n\n"+Q}], add_generation_prompt=True)
        if (len(ids)>=target and fi>=len(FACTS)) or block>4000: return ids, len(ids)

def measure(model, tok, target, label):
    """Hardened timing: separate prefill(TTFT) from decode; force mx.eval before stopping."""
    ids,ctx = build(tok, target)
    t0=time.perf_counter(); ttft=None; toks=[]; last=None
    for r in stream_generate(model, tok, ids, max_tokens=64):
        if ttft is None: ttft=time.perf_counter()-t0
        toks.append(r); last=r
    mx.eval(mx.array([1]))  # barrier: ensure all GPU work is flushed before we trust the clock
    total=time.perf_counter()-t0
    n=last.generation_tokens if last else 0
    decode_tps=(n-1)/(total-ttft) if ttft and total>ttft else None
    rec={"label":label,"target":target,"ctx":ctx,"ttft_s":round(ttft,3),
         "decode_tps_wall":round(decode_tps,2) if decode_tps else None,
         "decode_tps_mlx":round(last.generation_tps,2) if last else None,  # mlx's own number, for cross-check
         "peak_gb":round(mx.get_peak_memory()/1e9,3)}
    print(f"  [{label}] ctx {ctx} | TTFT {rec['ttft_s']}s | decode wall {rec['decode_tps_wall']} / mlx {rec['decode_tps_mlx']} tok/s | peak {rec['peak_gb']}GB")
    return rec

def main():
    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)
    # small warmup to remove JIT (not the point of this test)
    for _ in stream_generate(model, tok, tok.apply_chat_template([{"role":"user","content":"hi"}],add_generation_prompt=True), max_tokens=4): pass

    print("\n=== THERMAL CONTROL: cold-2k -> 32k(heat) -> hot-2k ===")
    mx.reset_peak_memory()
    cold = measure(model, tok, 2000, "cold-2k")
    heat = measure(model, tok, 32000, "heat-32k")   # ~7 min, drives the chip hot
    hot  = measure(model, tok, 2000, "hot-2k")      # immediately after, same tiny task

    with open(OUT,"w") as f:
        for r in (cold,heat,hot): f.write(json.dumps(r)+"\n")

    cw, hw = cold["decode_tps_wall"], hot["decode_tps_wall"]
    drop = round(100*(cw-hw)/cw,1) if cw and hw else None
    print("\n=== VERDICT ===")
    print(f"cold-2k decode {cw} tok/s  vs  hot-2k decode {hw} tok/s   -> hot is {drop}% slower")
    if drop is not None and drop > 15:
        print("=> LARGE drop on an identical 2k task after heating = THERMAL throttling confirmed.")
        print("   (cross-check: did `powermetrics` show clocks dropping during heat-32k? did swap stay flat?)")
    elif drop is not None and drop < 5:
        print("=> hot-2k ~= cold-2k. NOT clearly thermal. The earlier 2x may be a bug/other -> investigate before publishing.")
    else:
        print("=> ambiguous mild drop; run once more or check powermetrics/memory_pressure.")
    print(f"\nwrote -> {OUT}")

if __name__ == "__main__":
    main()
