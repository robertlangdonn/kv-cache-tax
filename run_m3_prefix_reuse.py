"""
KV cache tax -- prefix-reuse leg (shared system-prompt / tool-def prefix, cached
once, reused across "requests" -- the agent case named in the Lab spec's 4th
column). Same model, same 2k/8k/16k/32k total-context sweep, same thermal
discipline as run_m3.py / run_m3_eviction.py.

Structure per prompt: [FIXED SHARED PREFIX, ~500 tok] + [UNIQUE SUFFIX, grows
to reach target total length, facts embedded inside it]. Two conditions,
measured within the SAME harness so cold and warm are apples-to-apples (not
reusing v1 baseline data -- v1's fact placement is structurally different,
would be a confound):

  cold: fresh KVCache every call, full prefix+suffix prefilled from scratch
        (models: no caching, or a cold cache miss).
  warm: the shared prefix is prefilled ONCE per length (not timed -- models an
        agent whose system prompt/tool defs were already cached from turn 1),
        saved via mlx_lm's own LRUPromptCache.insert_cache(). Each "request"
        then calls fetch_nearest_cache() -- the real production prefix-cache
        lookup mlx-lm's server uses -- which returns a deep-copied prefix
        cache + only the new suffix tokens to prefill. TTFT measured on that
        remainder-only call.

Question this answers: prefix reuse buys back a roughly FIXED amount of
prefill time (the prefix's own prefill cost) -- so what does that fixed
saving look like as a fraction of TTFT as context grows? (Expected shape:
big relative win at 2k, shrinking relative win at 32k -- opposite direction
from the eviction leg's memory plateau, same "fixed absolute effect" family.)
Also sanity-checks that fetch_nearest_cache's deepcopy+trim path doesn't
silently break recall (no eviction is happening here, so recall should stay
5/5 in both conditions -- if it doesn't, that's a real bug worth its own PR).

Thermal note (learned from an aborted first sweep, raw preserved in
results_m3_prefix_reuse_raw_ABORTED_cold-first-confound.jsonl): running
cold-then-warm in a fixed order hands warm a systematic penalty -- at 32k the
chip is heat-soaked by cold's own ~6-min prefill and warm measured SLOWER
(471s vs 373s) despite doing strictly less work. Condition order therefore
alternates per grid cell, and each record carries ran_first_in_cell +
t_start_s so the ordering/thermal effect stays analyzable post-hoc.

RUN UNDER caffeinate so the machine doesn't nap mid-32k prefill:
  caffeinate -dimsu ~/Work/ondevice-bench/.venv/bin/python run_m3_prefix_reuse.py
  (smoke test one length: add --lengths 2000)
"""

import time, json, argparse, os, statistics
import mlx.core as mx
from mlx_lm import load, stream_generate
from mlx_lm.models.cache import (
    LRUPromptCache,
    make_prompt_cache,
    can_trim_prompt_cache,
    trim_prompt_cache,
)

from run_m3 import MODEL, FACTS, QUESTION, FILLER

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results_m3_prefix_reuse_raw.jsonl")
OUT = os.path.join(HERE, "results_m3_prefix_reuse.jsonl")

# Stamped on every record so --resume can detect a pair whose halves ran in
# different process lifetimes (both halves EXIST, so pair-completeness alone
# can't catch it -- learned when the first resume kept exactly such a pair).
PROC_ID = f"{int(time.time())}-{os.getpid()}"

PREFIX_TARGET = 500  # tokens -- matches the inference-audit post's RAG-shaped shared prefix
PREFIX_TEXT = (
    "SYSTEM: You are a facilities records assistant. You will be given an "
    "internal report followed by a question. Answer only using information "
    "explicitly present in the report. Do not speculate. The following tool "
    "definitions are available but not required for this task: "
    "get_vault_status(name), list_recent_incidents(building), "
    "escalate_to_security(vault, reason). Do not call a tool unless asked to. "
)

_PROMPT_IDS = {}
_SHARED_PREFIX = {}


def build_full_prompt(tok, target):
    """Fixed prefix text + growing suffix (facts embedded in the suffix only),
    tokenized as ONE chat turn so cold/warm see byte-identical prompt text --
    only the caching strategy differs between conditions."""
    if target in _PROMPT_IDS:
        return _PROMPT_IDS[target]
    parts, fi, block = [], 0, 0
    while True:
        parts.append(FILLER)
        if fi < len(FACTS) and block > 0 and block % 3 == 0:
            name, code = FACTS[fi]
            parts.append(f"IMPORTANT: The access code for the {name} vault is {code}. ")
            fi += 1
        block += 1
        body = PREFIX_TEXT + "".join(parts) + "\n\n" + QUESTION
        msg = [{"role": "user", "content": body}]
        ids = tok.apply_chat_template(msg, add_generation_prompt=True)
        if (len(ids) >= target and fi >= len(FACTS)) or block > 4000:
            _PROMPT_IDS[target] = ids
            return ids


def shared_prefix_ids(tok, all_targets):
    """The cacheable shared prefix, as TOKENS OF THE FULL TEMPLATED PROMPT.

    Encoding PREFIX_TEXT standalone does NOT yield a token-prefix of the full
    prompt -- apply_chat_template prepends header tokens and tokenization at
    the text boundary differs -- so the trie lookup would never match and
    "warm" would silently run cold. Instead: tokenize every target's full
    prompt, take their common token prefix (chat header + PREFIX_TEXT + the
    shared start of the filler, identical across lengths by construction),
    and cap it at PREFIX_TARGET tokens. Verified at runtime, not assumed.
    """
    key = tuple(all_targets)
    if key in _SHARED_PREFIX:
        return _SHARED_PREFIX[key]
    id_lists = [build_full_prompt(tok, t) for t in all_targets]
    n = min(len(x) for x in id_lists)
    common = 0
    while common < n and all(x[common] == id_lists[0][common] for x in id_lists):
        common += 1
    if common <= PREFIX_TARGET:
        raise RuntimeError(
            f"shared token prefix across targets is only {common} tokens "
            f"(< {PREFIX_TARGET}); prompt construction violated the "
            f"identical-shared-start assumption"
        )
    prefix = id_lists[0][:PREFIX_TARGET]
    _SHARED_PREFIX[key] = prefix
    return prefix


def run_cold(model, tok, target):
    mx.reset_peak_memory()
    ids = build_full_prompt(tok, target)
    rec = {"stack": "mlx-m3-prefix-cold", "target": target, "context_tokens": len(ids), "oom": False}
    try:
        t0 = time.perf_counter()
        ttft = None
        text = ""
        last = None
        for r in stream_generate(model, tok, ids, max_tokens=64):
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += r.text
            last = r
        rec.update(
            ttft_s=round(ttft, 3),
            decode_tps=round(last.generation_tps, 2),
            gen_tokens=last.generation_tokens,
            peak_gb=round(mx.get_peak_memory() / 1e9, 3),
            facts_recalled=sum(1 for _, c in FACTS if c in text),
        )
    except Exception as e:
        rec.update(oom=True, peak_gb=round(mx.get_peak_memory() / 1e9, 3),
                    failure=f"{type(e).__name__}: {str(e)[:160]}")
    return rec


def run_warm(model, tok, target, lru):
    mx.reset_peak_memory()
    ids = build_full_prompt(tok, target)
    rec = {"stack": "mlx-m3-prefix-warm", "target": target, "context_tokens": len(ids), "oom": False}
    try:
        # LRUPromptCache keys on a HASHABLE model key, not the nn.Module itself
        # (mlx_lm/server.py passes model_provider.model_key; passing the module
        # raises "unhashable type: 'Model'" since mlx nn.Module subclasses dict).
        cache, remainder = lru.fetch_nearest_cache(MODEL, ids)
        rec["cache_hit"] = cache is not None
        rec["prefix_reused_tokens"] = len(ids) - len(remainder)
        if cache is None:
            cache = make_prompt_cache(model)
            remainder = ids
        t0 = time.perf_counter()
        ttft = None
        text = ""
        last = None
        for r in stream_generate(model, tok, remainder, max_tokens=64, prompt_cache=cache):
            if ttft is None:
                ttft = time.perf_counter() - t0
            text += r.text
            last = r
        rec.update(
            ttft_s=round(ttft, 3),
            decode_tps=round(last.generation_tps, 2),
            gen_tokens=last.generation_tokens,
            peak_gb=round(mx.get_peak_memory() / 1e9, 3),
            facts_recalled=sum(1 for _, c in FACTS if c in text),
        )
    except Exception as e:
        rec.update(oom=True, peak_gb=round(mx.get_peak_memory() / 1e9, 3),
                    failure=f"{type(e).__name__}: {str(e)[:160]}")
    return rec


def warm_prefix_cache(model, tok, lru, prefix_ids):
    """Prefill the shared prefix once and insert it -- NOT timed (this models
    an agent whose system prompt was cached on turn 1, before the measured
    request loop starts).

    Pure forward pass, NOT stream_generate(max_tokens=1): generating even one
    token leaves that sampled token in the cache, so the cache would be one
    position longer than the token list it's registered under in the trie --
    every warm fetch would then resume prefill at a misaligned offset.
    """
    cache = make_prompt_cache(model)
    logits = model(mx.array(prefix_ids)[None], cache=cache)
    mx.eval(logits)
    assert cache[0].offset == len(prefix_ids), (
        f"prefix cache offset {cache[0].offset} != {len(prefix_ids)}"
    )
    lru.insert_cache(MODEL, prefix_ids, cache, cache_type="system")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[2000, 8000, 16000, 32000])
    ap.add_argument("--resume", action="store_true",
                    help="keep existing raw records and skip already-recorded "
                         "(pass, target, condition) cells -- for continuing after "
                         "a hard Metal OOM abort, which kills the process outright")
    args = ap.parse_args()
    L = args.lengths

    print(f"loading {MODEL} ...")
    model, tok = load(MODEL)

    import random
    def shuffled(seed):
        x = L[:]
        random.Random(seed).shuffle(x)
        return x

    # 4 recorded passes (even count, Codex P1): condition order alternates at
    # the PASS level -- cold-first in p1/p3, warm-first in p2/p4 -- so every
    # target is cold-first in exactly 2 of 4 recorded cells. The earlier
    # (pass+position)%2 parity scheme did NOT guarantee this per target once
    # lengths are shuffled (a length could land warm-first 2-of-3).
    passes = [("warmup", shuffled(1)), ("p1", shuffled(2)), ("p2", shuffled(3)),
              ("p3", shuffled(4)), ("p4", shuffled(5))]

    prefix_ids = shared_prefix_ids(tok, L)
    print(f"shared prefix: {len(prefix_ids)} tokens (token-exact across all targets)")

    done = set()
    if args.resume and os.path.exists(RAW):
        # Pair-level resume (Codex round 2, P1): a (pass, target) cell only
        # counts as done if BOTH conditions were recorded -- in the same
        # process, which is guaranteed by requiring the pair, since an abort
        # between the two halves leaves an orphan. An orphaned half is
        # DISCARDED and the whole pair reruns: keeping it would pair a
        # measurement from one process lifetime with one from another, and
        # both would have "run first" thermally, breaking the paired design.
        existing = [json.loads(l) for l in open(RAW)]
        seen = {}
        for r in existing:
            key = (r["pass"], r["target"])
            seen.setdefault(key, {})[r["stack"].split("-")[-1]] = r.get("proc_id")
        # A pair is valid only if both halves exist AND share a proc_id --
        # a pair split across two process lifetimes has both halves present
        # but neither is thermally second, so it must rerun. (Records from
        # before proc_id stamping compare as None == None, i.e. same-process;
        # any known-split legacy pair has to be deleted from the raw file by
        # hand, which is how the p2/32k split pair was handled.)
        complete = {k for k, conds in seen.items()
                    if set(conds) == {"cold", "warm"} and conds["cold"] == conds["warm"]}
        kept = [r for r in existing if (r["pass"], r["target"]) in complete]
        dropped = len(existing) - len(kept)
        with open(RAW, "w") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        for pname_, target_ in complete:
            done.add((pname_, target_, "cold"))
            done.add((pname_, target_, "warm"))
        print(f"--resume: {len(complete)} complete pairs kept, "
              f"{dropped} orphaned half-pair record(s) discarded for rerun", flush=True)
        if done and len(done) < len(passes) * len(L) * 2:
            # Fresh process, but the recorded warmup pass belongs to a previous
            # lifetime -- without this, the first rerun would carry the Metal
            # JIT-compile artifact the warmup pass exists to absorb.
            print("--resume: one discarded JIT-warmup run before rerunning pairs", flush=True)
            run_cold(model, tok, min(L))
            mx.clear_cache()
    else:
        open(RAW, "w").close()
    t_process_start = time.perf_counter()
    for pi, (pname, order) in enumerate(passes):
        cold_warmup = pname == "warmup"
        # fresh LRU each pass so warm's "prefix already cached" story is clean
        # (mirrors an agent session boundary, not cross-pass contamination)
        lru = LRUPromptCache(max_size=8)
        warm_prefix_cache(model, tok, lru, prefix_ids)
        print(f"\n### pass {pname}  order={order}{'  (DISCARDED)' if cold_warmup else ''}")
        for li, target in enumerate(order):
            # Alternate which condition runs FIRST, at the pass level. A fixed
            # cold-then-warm order hands warm a systematic thermal penalty on
            # this fanless machine (the aborted first sweep measured it: warm
            # 471s vs cold 373s at 32k in the same pass -- the chip was
            # heat-soaked by cold's own 6-min prefill). Pass-level alternation
            # over an even number of recorded passes makes each condition take
            # the hot second slot exactly half the time FOR EVERY TARGET.
            conds = [("cold", run_cold), ("warm", run_warm)]
            if pi % 2 == 0 and not cold_warmup:
                conds.reverse()
            for ci, (cond, fn) in enumerate(conds):
                if (pname, target, cond) in done:
                    print(f"  {target} [{cond}]: already recorded, skipped (--resume)", flush=True)
                    continue
                t_start = round(time.perf_counter() - t_process_start, 1)
                if cond == "cold":
                    rec = fn(model, tok, target)
                else:
                    rec = fn(model, tok, target, lru)
                rec["pass"] = pname
                rec["cold_discarded"] = cold_warmup
                rec["ran_first_in_cell"] = ci == 0
                rec["t_start_s"] = t_start
                rec["proc_id"] = PROC_ID
                with open(RAW, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                if rec["oom"]:
                    print(f"  {target} [{cond}]: OOM at {rec['context_tokens']} tok, peak {rec['peak_gb']}GB", flush=True)
                else:
                    extra = f" | reused {rec['prefix_reused_tokens']}tok" if cond == "warm" else ""
                    print(f"  {target} [{cond}]: ctx {rec['context_tokens']} | TTFT {rec['ttft_s']}s | "
                          f"{rec['decode_tps']} tok/s | peak {rec['peak_gb']}GB | recall {rec['facts_recalled']}/5"
                          f"{extra}" + ("  [warmup, discarded]" if cold_warmup else ""), flush=True)
                # Per RUN, not per pass -- the eviction leg already learned this
                # (bd5f526): MLX's buffer cache accumulates across fresh per-run
                # cache lists, and 17 runs in, a 32k prefill aborted the whole
                # process with a hard Metal command-buffer OOM (libc++abi abort,
                # NOT a catchable Python exception -- the try/except in one_run
                # never sees it).
                mx.clear_cache()

    raw = [json.loads(l) for l in open(RAW)]
    recorded = [r for r in raw if not r["cold_discarded"]]
    n_expected = len(passes) - 1  # recorded passes
    agg = []
    for target in L:
        row = {"target": target}
        for cond in ("cold", "warm"):
            all_s = [r for r in recorded if r["target"] == target and r["stack"].endswith(cond)]
            s = [r for r in all_s if not r["oom"]]
            n_failed = len(all_s) - len(s)
            if not s:
                row[cond] = {"oom": True, "n_ok": 0, "n_expected": n_expected,
                             "peak_gb": all_s[0]["peak_gb"] if all_s else None}
                continue
            def ms(k):
                v = [r[k] for r in s]
                return {"median": round(statistics.median(v), 3), "min": min(v), "max": max(v)}
            # Codex P2: a partially-failed condition must say so, not present
            # a clean median as if every recorded pass had completed.
            row[cond] = {
                "oom": False, "n_ok": len(s), "n_expected": n_expected,
                "n_failed": n_failed, "incomplete": n_failed > 0,
                "context_tokens": s[0]["context_tokens"],
                "ttft_s": ms("ttft_s"), "decode_tps": ms("decode_tps"), "peak_gb": ms("peak_gb"),
                "recall": f"{s[0]['facts_recalled']}/5",
            }
            if cond == "warm":
                row[cond]["prefix_reused_tokens"] = s[0]["prefix_reused_tokens"]
        # Paired estimator (Codex round 2, P2): with paired cold/warm
        # observations, difference-of-marginal-medians is the wrong statistic
        # -- it mixes values from different passes and overstated the 16k/32k
        # savings in the first cut of this analysis. Compute each pass's
        # cold-minus-warm first, then summarize THOSE.
        by_pass = {}
        for r in [x for x in recorded if x["target"] == target and not x["oom"]]:
            by_pass.setdefault(r["pass"], {})[r["stack"].split("-")[-1]] = r["ttft_s"]
        pairs = [(p, v["cold"] - v["warm"], 100 * (v["cold"] - v["warm"]) / v["cold"])
                 for p, v in sorted(by_pass.items()) if {"cold", "warm"} <= set(v)]
        if pairs:
            row["paired"] = {
                "n_pairs": len(pairs),
                "saved_s": {"median": round(statistics.median(d for _, d, _ in pairs), 3),
                            "min": round(min(d for _, d, _ in pairs), 3),
                            "max": round(max(d for _, d, _ in pairs), 3)},
                "saved_pct": {"median": round(statistics.median(p for _, _, p in pairs), 2),
                              "min": round(min(p for _, _, p in pairs), 2),
                              "max": round(max(p for _, _, p in pairs), 2)},
            }
        agg.append(row)
    with open(OUT, "w") as f:
        for a in agg:
            f.write(json.dumps(a) + "\n")

    print(f"\n=== AGGREGATED ({n_expected} recorded passes; saved = paired per-pass median [min,max]) ===")
    print(f"{'ctx':>7} | {'TTFT cold':>10} | {'TTFT warm':>10} | {'paired saved':>22} | {'saved%':>18} | {'n c/w':>7} | recall(c/w)")
    for a in agg:
        c, w = a["cold"], a["warm"]
        if c.get("oom") or w.get("oom"):
            print(f"{a['target']:>7} | ALL RUNS FAILED "
                  f"(cold {c.get('n_ok', 0)}/{n_expected}, warm {w.get('n_ok', 0)}/{n_expected})")
            continue
        pr = a.get("paired")
        ns = f"{c['n_ok']}/{w['n_ok']}"
        flag = "  ⚠ INCOMPLETE" if (c["incomplete"] or w["incomplete"]) else ""
        if pr:
            d, p = pr["saved_s"], pr["saved_pct"]
            saved_str = f"{d['median']:.2f}s [{d['min']:.1f},{d['max']:.1f}]"
            pct_str = f"{p['median']:.1f}% [{p['min']:.1f},{p['max']:.1f}]"
            if pr["n_pairs"] < n_expected:
                flag += f"  ⚠ only {pr['n_pairs']} pairs"
        else:
            saved_str, pct_str = "no pairs", "-"
        print(f"{a['target']:>7} | {c['ttft_s']['median']:>10} | {w['ttft_s']['median']:>10} | "
              f"{saved_str:>22} | {pct_str:>18} | {ns:>7} | {c['recall']}/{w['recall']}{flag}")
    print(f"\nraw -> {RAW}\nclean -> {OUT}")


if __name__ == "__main__":
    main()
