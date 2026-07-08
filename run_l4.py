"""
KV cache tax — L4 / vLLM leg (the cross-hardware half).

⚠️ UNTESTED locally (no CUDA on the M3). Runs ON the rented GPU box. Mirrors run_m3.py:
same model, same 5-planted-fact transcript grown to the same 2k/8k/16k/32k targets, same
metrics (TTFT, decode tok/s, recall) so the CURVES are comparable. Memory is handled per
Codex: do NOT compare raw peak GB (vLLM pre-reserves KV) — report theoretical KV/token +
whatever vLLM's startup log says it reserved + nvidia-smi, separately.

Governing rule (Codex): within-stack curves are primary; cross-stack absolutes get
artifact-labeled (M3=MLX-4bit-dynamic vs L4=vLLM-4bit/fp16-reserved). Record the EXACT GPU.

────────────────────────────────────────────────────────────────────────────────────────
ON THE BOX — one-time setup (paste this after you SSH in):
  pip install vllm openai transformers
  # PRIMARY run (4-bit, matches M3). AWQ build of Llama-3.1-8B:
  vllm serve hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
      --quantization awq --max-model-len 34000 --max-num-seqs 1 \
      --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8000
  # (CONTROL run later: swap to  meta-llama/Llama-3.1-8B-Instruct  with no --quantization;
  #  needs the HF gated-model token: huggingface-cli login)
  # watch the startup log for the line reporting GPU KV cache size / # KV blocks — RECORD IT.

THEN in another shell on the box:
  export KVTAX_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
  export KVTAX_GPU="L4-24GB"      # or whatever GPU you actually got — RECORD IT
  python run_l4.py
  nvidia-smi                      # note the memory number alongside the results
────────────────────────────────────────────────────────────────────────────────────────

RUN:  python run_l4.py   (talks to the local vLLM server on :8000)
"""
import os, time, json, statistics
from openai import OpenAI
from transformers import AutoTokenizer

MODEL = os.environ.get("KVTAX_MODEL", "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
GPU   = os.environ.get("KVTAX_GPU", "UNKNOWN-GPU")   # RECORD the exact GPU
BASE  = os.environ.get("KVTAX_BASE", "http://localhost:8000/v1")
LENGTHS = [2000, 8000, 16000, 32000]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_l4.jsonl")

# ── identical transcript content to run_m3.py so contexts match ──
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

tok = AutoTokenizer.from_pretrained(MODEL)

_FILLER_TOK = len(tok(FILLER)["input_ids"])   # measure once
_MSG_CACHE = {}
def build_messages(target):
    """Efficient one-pass build (O(n), not O(n^2)): estimate #filler blocks from a measured
    per-block token count, drop the 5 facts across the first portion, tokenize ONCE to get the
    real length. Cached per target so repeated passes are instant."""
    if target in _MSG_CACHE: return _MSG_CACHE[target]
    n_blocks = max(len(FACTS)*3 + 2, int(target / max(_FILLER_TOK, 1)) + 4)
    parts, fi = [], 0
    for b in range(n_blocks):
        parts.append(FILLER)
        if fi < len(FACTS) and b > 0 and b % 3 == 0:
            n, c = FACTS[fi]; parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. "); fi += 1
    # ensure all facts placed even if n_blocks was small
    while fi < len(FACTS):
        n, c = FACTS[fi]; parts.append(f"IMPORTANT: The access code for the {n} vault is {c}. "); fi += 1
    msgs = [{"role":"user","content":"".join(parts)+"\n\n"+QUESTION}]
    # render template to a STRING, then tokenize the string -> unambiguous flat token count
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    n_tok = len(tok(text)["input_ids"])
    _MSG_CACHE[target] = (msgs, n_tok)
    return msgs, n_tok

# theoretical KV bytes/token for Llama-3.1-8B (fp16 KV): 2(K,V) * 32 layers * 8 kv-heads * 128 head-dim * 2 bytes
KV_BYTES_PER_TOKEN = 2 * 32 * 8 * 128 * 2   # = 131072 bytes = 128 KiB/token

def one_run(client, target):
    msgs, ctx = build_messages(target)
    rec = {"stack":"vllm-l4","gpu":GPU,"model":MODEL,"target":target,"context_tokens":ctx,
           "theoretical_kv_gb": round(KV_BYTES_PER_TOKEN*ctx/1e9, 3)}
    t0 = time.perf_counter(); ttft=None; text=""; n=0
    stream = client.chat.completions.create(model=MODEL, messages=msgs, temperature=0,
                                            max_tokens=64, stream=True)
    for chunk in stream:
        d = chunk.choices[0].delta.content or ""
        if d and ttft is None: ttft = time.perf_counter()-t0
        text += d; n += 1
    total = time.perf_counter()-t0
    rec.update(ttft_s=round(ttft,3) if ttft else None,
               decode_tps=round((n-1)/(total-ttft),2) if ttft and total>ttft else None,
               gen_chunks=n, facts_recalled=sum(1 for _,c in FACTS if c in text))
    return rec

def main():
    client = OpenAI(base_url=BASE, api_key="EMPTY")
    # warmup+3 recorded, interleaved (same rigor as M3)
    import random
    def sh(seed): x=LENGTHS[:]; random.Random(seed).shuffle(x); return x
    passes = [("warmup",sh(1),True),("p1",sh(2),False),("p2",sh(3),False),("p3",sh(4),False)]
    raw=[]
    for pname, order, cold in passes:
        print(f"\n### {pname} order={order}{' (discard)' if cold else ''}")
        for target in order:
            try:
                r = one_run(client, target); r["pass"]=pname; r["cold_discarded"]=cold; r["oom"]=False
                print(f"  {target}: ctx {r['context_tokens']} | TTFT {r['ttft_s']}s | {r['decode_tps']} tok/s | recall {r['facts_recalled']}/5")
            except Exception as e:
                r = {"target":target,"pass":pname,"cold_discarded":cold,"oom":True,"failure":f"{type(e).__name__}: {str(e)[:160]}"}
                print(f"  {target}: FAIL {r['failure']}")
            raw.append(r)
    # aggregate recorded, non-failed
    rec=[r for r in raw if not r["cold_discarded"] and not r.get("oom")]
    with open(OUT,"w") as f:
        for target in LENGTHS:
            s=[r for r in rec if r["target"]==target]
            if not s: f.write(json.dumps({"target":target,"oom":True})+"\n"); continue
            def ms(k): v=[r[k] for r in s if r[k] is not None]; return {"median":round(statistics.median(v),3),"min":min(v),"max":max(v)} if v else None
            f.write(json.dumps({"gpu":GPU,"model":MODEL,"target":target,"context_tokens":s[0]["context_tokens"],
                "n":len(s),"ttft_s":ms("ttft_s"),"decode_tps":ms("decode_tps"),
                "theoretical_kv_gb":s[0]["theoretical_kv_gb"],"recall":f"{s[0]['facts_recalled']}/5"})+"\n")
    print(f"\nwrote {OUT}. ALSO record: the vLLM startup 'GPU KV cache' line + `nvidia-smi` memory + exact GPU name.")

if __name__ == "__main__":
    main()
