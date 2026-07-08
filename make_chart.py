"""Build the KV-cache-tax hero chart: M3 vs L4, TTFT (log) + decode, cream/serif site style."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
SERIF = "/Users/prasad/Work/prasadkhake/og-fonts/Serif-Regular.ttf"
MONO  = "/Users/prasad/Work/prasadkhake/og-fonts/SpaceMono-Regular.ttf"
fm.fontManager.addfont(SERIF); fm.fontManager.addfont(MONO)
serif = fm.FontProperties(fname=SERIF); mono = fm.FontProperties(fname=MONO)

PAPER="#f2ede4"; INK="#1a1814"; MUTED="#7a7468"; BLUE="#2f5d8a"; RUST="#a4552f"; GREY="#b8b0a2"

def load(p):
    rows=[json.loads(l) for l in open(os.path.join(HERE,p))]
    return rows
m3=load("results_m3.jsonl"); l4=load("results_l4.jsonl")

def xs(rows): return [r["context_tokens"]/1000 for r in rows]
def ttft(rows): return [r["ttft_s"]["median"] for r in rows]
def dec(rows): return [r["decode_tps"]["median"] for r in rows]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5.2))
fig.patch.set_facecolor(PAPER)
for ax in (axA, axB):
    ax.set_facecolor(PAPER)
    for s in ax.spines.values(): s.set_color(MUTED)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(colors=MUTED)
    for lab in ax.get_xticklabels()+ax.get_yticklabels(): lab.set_fontproperties(mono); lab.set_fontsize(10)

# Panel A: TTFT (log y — spans 0.8s to 430s)
axA.set_yscale("log")
axA.plot(xs(m3), ttft(m3), "-o", color=RUST, lw=2.5, ms=6, label="M3 16GB (MLX, 4-bit)")
axA.plot(xs(l4), ttft(l4), "-o", color=BLUE, lw=2.5, ms=6, label="L4 24GB (vLLM, AWQ 4-bit)")
axA.annotate("~7 min", (xs(m3)[-1], ttft(m3)[-1]), textcoords="offset points", xytext=(-6,8),
             fontproperties=mono, fontsize=11, color=RUST, ha="right")
axA.annotate("15 s", (xs(l4)[-1], ttft(l4)[-1]), textcoords="offset points", xytext=(-4,-16),
             fontproperties=mono, fontsize=11, color=BLUE, ha="right")
axA.set_title("Time to first token (prefill)", fontproperties=serif, fontsize=17, color=INK, pad=10)
axA.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axA.set_ylabel("seconds  (log scale)", fontproperties=mono, fontsize=11, color=MUTED)
leg=axA.legend(prop=serif, fontsize=11, frameon=False, loc="upper left")
for t in leg.get_texts(): t.set_color(INK)

# Panel B: decode tok/s (linear). M3 greyed (thermally confounded), L4 clean.
axB.plot(xs(l4), dec(l4), "-o", color=BLUE, lw=2.5, ms=6, label="L4 (clean, actively cooled)")
axB.plot(xs(m3), dec(m3), "--o", color=GREY, lw=2, ms=5, label="M3 (thermally throttled*)")
axB.set_title("Decode throughput", fontproperties=serif, fontsize=17, color=INK, pad=10)
axB.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axB.set_ylabel("tokens / second", fontproperties=mono, fontsize=11, color=MUTED)
axB.set_ylim(0, 55)
leg2=axB.legend(prop=serif, fontsize=11, frameon=False, loc="upper right")
for t in leg2.get_texts(): t.set_color(INK)
axB.text(0.02, 0.04, "*fanless M3 under sustained load: proven ~35% throttle on\nan identical task (see post). Report as observed, not a clean curve.",
         transform=axB.transAxes, fontproperties=mono, fontsize=8.5, color=MUTED, va="bottom")

fig.suptitle("The KV-cache tax: what long-context costs on an M3 vs an L4",
             fontproperties=serif, fontsize=20, color=INK, y=0.99)
fig.text(0.5, 0.005, "Llama-3.1-8B, 4-bit · 5 planted facts, recall 5/5 at every length on both · Prasad Khake · prasadkhake.com",
         ha="center", fontproperties=mono, fontsize=9, color=MUTED)
fig.tight_layout(rect=[0,0.03,1,0.955])
out=os.path.join(HERE,"kv_cache_tax_chart.png")
fig.savefig(out, dpi=150, facecolor=PAPER)
print("wrote", out)
