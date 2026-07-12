"""
Eviction-leg chart: full KV baseline (v1) vs rotating/windowed KV (window=4096,
keep=4), M3 only. What you buy back in memory vs what you pay in recall.
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


def load(p):
    return [json.loads(l) for l in open(os.path.join(HERE, p))]


full = load("results_m3.jsonl")
evict = load("results_m3_eviction.jsonl")


def xs(rows):
    return [r["context_tokens"] / 1000 for r in rows]


def peak(rows):
    return [r["peak_gb"]["median"] for r in rows]


def recall_n(rows):
    return [int(r["recall"].split("/")[0]) for r in rows]


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

# Panel A: peak memory -- full KV climbs, rotating plateaus at the window.
axA.plot(xs(full), peak(full), "-o", color=RUST, lw=2.5, ms=6, label="full KV cache (v1 baseline)")
axA.plot(xs(evict), peak(evict), "-o", color=BLUE, lw=2.5, ms=6, label="rotating KV, window=4096, keep=4")
axA.axvline(4.096, color=GREY, lw=1, ls=":")
axA.annotate(
    "window",
    (4.096, axA.get_ylim()[0] if axA.get_ylim()[0] else 5.0),
    textcoords="offset points",
    xytext=(4, 4),
    fontproperties=mono,
    fontsize=9,
    color=MUTED,
)
axA.set_title("Peak memory", fontproperties=serif, fontsize=17, color=INK, pad=10)
axA.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axA.set_ylabel("GB", fontproperties=mono, fontsize=11, color=MUTED)
leg = axA.legend(prop=serif, fontsize=10.5, frameon=False, loc="upper left")
for t in leg.get_texts():
    t.set_color(INK)

# Panel B: recall -- full KV stays 5/5, rotating collapses to 0/5 past the window.
axB.plot(xs(full), recall_n(full), "-o", color=RUST, lw=2.5, ms=6, label="full KV cache (v1 baseline)")
axB.plot(xs(evict), recall_n(evict), "-o", color=BLUE, lw=2.5, ms=6, label="rotating KV, window=4096, keep=4")
axB.axvline(4.096, color=GREY, lw=1, ls=":")
axB.set_ylim(-0.3, 5.3)
axB.set_title("Fact recall (of 5 planted)", fontproperties=serif, fontsize=17, color=INK, pad=10)
axB.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axB.set_ylabel("facts recalled", fontproperties=mono, fontsize=11, color=MUTED)
leg2 = axB.legend(prop=serif, fontsize=10.5, frameon=False, loc="center right")
for t in leg2.get_texts():
    t.set_color(INK)
axB.text(
    0.02,
    0.04,
    "keep=4 sink tokens (StreamingLLM default) protect position 0-3 only --\nthe planted facts live in the document body, not the prompt's start.",
    transform=axB.transAxes,
    fontproperties=mono,
    fontsize=8.5,
    color=MUTED,
    va="bottom",
)

fig.suptitle(
    "The eviction tax: what a rotating KV window buys and costs",
    fontproperties=serif,
    fontsize=20,
    color=INK,
    y=0.99,
)
fig.text(
    0.5,
    0.005,
    "Llama-3.1-8B, 4-bit, M3 16GB · window=4096 keep=4 · n=3 per point except 32k (n=2, memory-constrained) · Prasad Khake · prasadkhake.com",
    ha="center",
    fontproperties=mono,
    fontsize=9,
    color=MUTED,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.955])
out = os.path.join(HERE, "kv_cache_eviction_chart.png")
fig.savefig(out, dpi=150, facecolor=PAPER)
print("wrote", out)
