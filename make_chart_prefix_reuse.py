"""
Prefix-reuse-leg chart: cold prefill vs warm (500-token shared prefix served
from LRUPromptCache), M3 only. The saving is a fixed absolute chunk of prefill
work, so its RELATIVE value should decay as context grows -- panel A shows the
TTFT curves, panel B shows the saved% trend directly.

Data-driven: reads results_m3_prefix_reuse.jsonl, no hand-entered numbers.
Skips (and labels) any length whose row is incomplete/OOM per the harness's
n_ok/n_expected accounting -- never plots a partial median as if it were clean.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
SERIF = "/Users/prasad/Work/prasadkhake/og-fonts/Serif-Regular.ttf"
MONO = "/Users/prasad/Work/prasadkhake/og-fonts/SpaceMono-Regular.ttf"
fm.fontManager.addfont(SERIF)
fm.fontManager.addfont(MONO)
serif = fm.FontProperties(fname=SERIF)
mono = fm.FontProperties(fname=MONO)

PAPER = "#f2ede4"
INK = "#1a1814"
MUTED = "#7a7468"
BLUE = "#2f5d8a"
RUST = "#a4552f"
GREY = "#b8b0a2"

rows = [json.loads(l) for l in open(os.path.join(HERE, "results_m3_prefix_reuse.jsonl"))]

clean, flagged = [], []
for r in rows:
    c, w = r["cold"], r["warm"]
    if c.get("oom") or w.get("oom") or c.get("incomplete") or w.get("incomplete"):
        flagged.append(r["target"])
        continue
    clean.append(r)
if not clean:
    raise SystemExit("no complete rows to plot")

ctx = [r["cold"]["context_tokens"] / 1000 for r in clean]
cold_med = [r["cold"]["ttft_s"]["median"] for r in clean]
warm_med = [r["warm"]["ttft_s"]["median"] for r in clean]
cold_lo = [r["cold"]["ttft_s"]["min"] for r in clean]
cold_hi = [r["cold"]["ttft_s"]["max"] for r in clean]
warm_lo = [r["warm"]["ttft_s"]["min"] for r in clean]
warm_hi = [r["warm"]["ttft_s"]["max"] for r in clean]
saved_pct = [100 * (c - w) / c for c, w in zip(cold_med, warm_med)]
prefix_tok = clean[0]["warm"]["prefix_reused_tokens"]
n_passes = clean[0]["cold"]["n_expected"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor(PAPER)
for ax in (axA, axB):
    ax.set_facecolor(PAPER)
    for s in ax.spines.values():
        s.set_color(MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontproperties(mono)
        lab.set_fontsize(10)

# Panel A: TTFT, cold vs warm, median with min-max band.
axA.plot(ctx, cold_med, "-o", color=RUST, lw=2.5, ms=6, label="cold (full prefill)")
axA.fill_between(ctx, cold_lo, cold_hi, color=RUST, alpha=0.15, lw=0)
axA.plot(ctx, warm_med, "-o", color=BLUE, lw=2.5, ms=6,
         label=f"warm ({prefix_tok}-token prefix cached)")
axA.fill_between(ctx, warm_lo, warm_hi, color=BLUE, alpha=0.15, lw=0)
axA.set_title("Time to first token", fontproperties=serif, fontsize=17, color=INK, pad=10)
axA.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axA.set_ylabel("seconds", fontproperties=mono, fontsize=11, color=MUTED)
leg = axA.legend(prop=serif, fontsize=10.5, frameon=False, loc="upper left")
for t in leg.get_texts():
    t.set_color(INK)

# Panel B: the point of the leg -- relative saving decays with context.
# Medians alone would hide that at 16k-32k the per-pass spread is larger than
# the effect (32k paired savings ranged 0.3s..51.6s around a ~9s prediction),
# so every pass's paired saving is plotted as its own dot.
raw = [json.loads(l) for l in open(os.path.join(HERE, "results_m3_prefix_reuse_raw.jsonl"))]
raw = [r for r in raw if not r["cold_discarded"] and not r["oom"]]
for r in clean:
    t = r["target"]
    by_pass = {}
    for s in raw:
        if s["target"] == t:
            by_pass.setdefault(s["pass"], {})[s["stack"].split("-")[-1]] = s["ttft_s"]
    pair_pct = [100 * (v["cold"] - v["warm"]) / v["cold"]
                for v in by_pass.values() if "cold" in v and "warm" in v]
    axB.scatter([r["cold"]["context_tokens"] / 1000] * len(pair_pct), pair_pct,
                s=22, color=MUTED, alpha=0.6, zorder=2,
                label="per-pass paired saving" if r is clean[0] else None)
axB.plot(ctx, saved_pct, "-o", color=INK, lw=2.5, ms=6, zorder=3, label="median saving")
prop = [100 * prefix_tok / (r["cold"]["context_tokens"]) for r in clean]
axB.plot(ctx, prop, ":", color=GREY, lw=2,
         label="prefix share of prompt (proportional bound)")
axB.axhline(0, color=MUTED, lw=0.8)
axB.set_title("TTFT saved by prefix reuse (%)", fontproperties=serif, fontsize=17, color=INK, pad=10)
axB.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axB.set_ylabel("% of cold TTFT", fontproperties=mono, fontsize=11, color=MUTED)
leg2 = axB.legend(prop=serif, fontsize=10.5, frameon=False, loc="upper right")
for t in leg2.get_texts():
    t.set_color(INK)

flag_note = f" · {','.join(str(t//1000)+'k' for t in flagged)} omitted (incomplete)" if flagged else ""
fig.suptitle(
    "Prefix caching: a fixed saving that context growth dilutes",
    fontproperties=serif, fontsize=20, color=INK, y=0.99,
)
fig.text(
    0.5, 0.005,
    f"Llama-3.1-8B, 4-bit, M3 16GB · {prefix_tok}-token shared prefix via LRUPromptCache · "
    f"median of {n_passes} passes, condition order alternated{flag_note} · Prasad Khake · prasadkhake.com",
    ha="center", fontproperties=mono, fontsize=9, color=MUTED,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.955])
out = os.path.join(HERE, "kv_cache_prefix_reuse_chart.png")
fig.savefig(out, dpi=150, facecolor=PAPER)
print("wrote", out)
if flagged:
    print(f"NOTE: omitted incomplete lengths: {flagged}")
