"""
Quantized-KV-leg chart (L4/vLLM): fp16 vs fp8 KV cache. Panel A = decode
throughput by context (the headline: fp8 pulls AHEAD as context grows —
bandwidth-bound decode reads half the KV bytes). Panel B = the fp8 speedup vs
context, with the 2x-capacity fact annotated. Data-driven from
results_l4_kvquant.jsonl; skips/labels incomplete rows.
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

rows = [json.loads(l) for l in open(os.path.join(HERE, "results_l4_kvquant.jsonl"))]
by_arm = {}
flagged = []
for r in rows:
    if r.get("all_failed") or r.get("incomplete"):
        flagged.append((r.get("kv_dtype"), r["target"]))
        continue
    by_arm.setdefault(r["kv_dtype"], []).append(r)
for arm in by_arm:
    by_arm[arm].sort(key=lambda r: r["target"])
fp16, fp8 = by_arm["fp16"], by_arm["fp8"]
assert [r["target"] for r in fp16] == [r["target"] for r in fp8], "arm length mismatch"

ctx = [r["context_tokens"] / 1000 for r in fp16]
d16 = [r["decode_tps"]["median"] for r in fp16]
d8 = [r["decode_tps"]["median"] for r in fp8]
speedup = [100 * (b - a) / a for a, b in zip(d16, d8)]

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

axA.plot(ctx, d16, "-o", color=RUST, lw=2.5, ms=6, label="fp16 KV cache (default)")
axA.plot(ctx, d8, "-o", color=BLUE, lw=2.5, ms=6, label="fp8 KV cache")
axA.set_title("Decode throughput", fontproperties=serif, fontsize=17, color=INK, pad=10)
axA.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axA.set_ylabel("tokens / second", fontproperties=mono, fontsize=11, color=MUTED)
leg = axA.legend(prop=serif, fontsize=10.5, frameon=False, loc="lower left")
for t in leg.get_texts():
    t.set_color(INK)

axB.plot(ctx, speedup, "-o", color=INK, lw=2.5, ms=6)
axB.axhline(0, color=MUTED, lw=0.8)
axB.set_title("fp8 decode speedup over fp16 (%)", fontproperties=serif, fontsize=17, color=INK, pad=10)
axB.set_xlabel("context length (thousands of tokens)", fontproperties=mono, fontsize=11, color=MUTED)
axB.set_ylabel("% faster", fontproperties=mono, fontsize=11, color=MUTED)
axB.text(
    0.03, 0.93,
    "same 13.37 GiB reservation:\n109,504 tokens (fp16) -> 219,008 (fp8) = 2.0x capacity\nrecall 5/5 at every length, both arms",
    transform=axB.transAxes, fontproperties=mono, fontsize=9, color=MUTED, va="top",
)

flag_note = f" · incomplete rows omitted: {flagged}" if flagged else ""
fig.suptitle(
    "fp8 KV cache on an L4: the win MLX's --kv-bits promised and didn't deliver",
    fontproperties=serif, fontsize=19, color=INK, y=0.99,
)
fig.text(
    0.5, 0.005,
    f"Llama-3.1-8B AWQ-INT4, vLLM 0.25.1, NVIDIA L4 24GB · batch 1, prefix caching off · "
    f"median of 3 interleaved passes{flag_note} · Prasad Khake · prasadkhake.com",
    ha="center", fontproperties=mono, fontsize=9, color=MUTED,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.955])
out = os.path.join(HERE, "kv_cache_kvquant_chart.png")
fig.savefig(out, dpi=150, facecolor=PAPER)
print("wrote", out)
