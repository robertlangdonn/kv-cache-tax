"""
Thermal control v2 — 3 cold/hot cycles + cooldown recovery (upgrades n=1 -> defensible).

Codex flagged the n=1 control as "evidence, not proof." This runs the cold->heat->hot pattern
THREE times, plus a final cooldown-recovery check, so we can say "repeatable" not "one data point".
Pairs with a powermetrics capture (run separately, see bottom) to show clocks actually drop.

Pattern per cycle:  cold-2k  ->  32k (heat)  ->  hot-2k
Then: sleep 8 min (cooldown)  ->  recovered-2k   (should climb back toward cold speed = proves it's
reversible heat, not a permanent state/allocator effect).

RUN:  caffeinate -dimsu ~/Work/ondevice-bench/.venv/bin/python thermal_control_v2.py
OPTIONAL 2nd terminal (proves clocks drop):
  sudo powermetrics --samplers cpu_power,gpu_power -i 3000 | grep -iE "frequency|freq|throttle" > /tmp/pm.log
"""
import time, json, os, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate

MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thermal_control_v2.jsonl")

FACTS=[("Aurora","QX-4471"),("Basalt","ZR-9028"),("Cinder","MP-1357"),("Dovewing","KT-6604"),("Ember","VN-8813")]
Q=("List the access codes for all five vaults mentioned above, in order "
   "(Aurora, Basalt, Cinder, Dovewing, Ember). Answer with just the five codes.")
FILLER=("The facilities report continued with routine notes about the north campus. Maintenance "
   "logged the quarterly inspection of the ventilation ducts and the backup generators, both of "
   "which passed without incident. The grounds team trimmed the hedges along the east walkway and "
   "repainted the loading-bay lines. A minor water-pressure fluctuation in Building C was traced to "
   "a partially closed valve and corrected the same afternoon. Nothing here requires escalation. ")

_C={}
def build(tok,target):
    if target in _C: return _C[target]
    parts,fi,b=[],0,0
    while True:
        parts.append(FILLER)
        if fi<len(FACTS) and b>0 and b%3==0:
            n,c=FACTS[fi]; parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. "); fi+=1
        b+=1
        ids=tok.apply_chat_template([{"role":"user","content":"".join(parts)+"\n\n"+Q}],add_generation_prompt=True)
        if (len(ids)>=target and fi>=len(FACTS)) or b>4000: _C[target]=(ids,len(ids)); return _C[target]

def decode_tps(model,tok,target):
    ids,ctx=build(tok,target)
    t0=time.perf_counter(); ttft=None; last=None
    for r in stream_generate(model,tok,ids,max_tokens=64):
        if ttft is None: ttft=time.perf_counter()-t0
        last=r
    mx.eval(mx.array([1]))
    total=time.perf_counter()-t0; n=last.generation_tokens
    return round((n-1)/(total-ttft),2), ctx

def main():
    print(f"loading {MODEL} ...")
    model,tok=load(MODEL)
    for _ in stream_generate(model,tok,tok.apply_chat_template([{"role":"user","content":"hi"}],add_generation_prompt=True),max_tokens=4): pass
    rows=[]
    for cyc in range(1,4):
        print(f"\n=== cycle {cyc}: cold-2k -> heat-32k -> hot-2k ===")
        cold,_=decode_tps(model,tok,2000);  print(f"  cold-2k: {cold} tok/s")
        _=decode_tps(model,tok,32000);        print(f"  heat-32k done")
        hot,_=decode_tps(model,tok,2000);    print(f"  hot-2k:  {hot} tok/s  ({round(100*(cold-hot)/cold,1)}% drop)")
        rows.append({"cycle":cyc,"cold_2k_tps":cold,"hot_2k_tps":hot,"drop_pct":round(100*(cold-hot)/cold,1)})
    print("\n=== cooldown 8 min, then recovered-2k (should climb back = reversible heat) ===")
    time.sleep(480)
    rec,_=decode_tps(model,tok,2000); print(f"  recovered-2k: {rec} tok/s")
    rows.append({"cycle":"recovered","cold_2k_tps":None,"hot_2k_tps":rec,"drop_pct":None})
    with open(OUT,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    drops=[r["drop_pct"] for r in rows if r.get("drop_pct") is not None]
    print("\n=== VERDICT ===")
    print(f"3 cold/hot cycles, hot-2k slower by: {drops}  (median {statistics.median(drops)}%)")
    print(f"recovered-2k after cooldown: {rec} tok/s (vs cold ~{rows[0]['cold_2k_tps']})")
    if all(d>15 for d in drops) and rec > rows[-2]["hot_2k_tps"]:
        print("=> REPEATABLE + REVERSIBLE throttle across 3 cycles, recovers on cooldown = THERMAL, confirmed.")
    print(f"\nwrote -> {OUT}")

if __name__=="__main__":
    main()
