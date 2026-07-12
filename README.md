# kv-cache-tax

What long context actually costs, measured on two very different machines: a 16 GB fanless Apple M3
(mlx-lm) and a rented NVIDIA L4 (vLLM). One 8B model, context grown 2k → 32k, measuring the KV-cache
"tax" — time to first token, decode throughput, peak memory, and fact recall.

Write-up: **prasadkhake.com/blog/kv-cache-tax-m3-vs-l4** · one slice of a cross-hardware inference map.

## The result

| context | M3 TTFT | M3 decode | M3 peak RAM | L4 TTFT | L4 decode | recall |
|---|---|---|---|---|---|---|
| 2k | 20 s | ~10 tok/s* | 5.3 GB | 0.8 s | 50 tok/s | 5/5 |
| 8k | 77 s | *thermal* | 6.1 GB | 2.9 s | 44 tok/s | 5/5 |
| 16k | 176 s | *thermal* | 7.2 GB | 6.2 s | 38 tok/s | 5/5 |
| 32k | **430 s** | *thermal* | 9.3 GB | **14.6 s** | 29 tok/s | 5/5 |

- **Prefill is the tax.** M3 TTFT balloons ~22× (superlinear) to 7 minutes at 32k; the L4 is ~30× faster.
- **\*M3 decode is thermally throttled, not cleanly measurable.** Proven with a 3-cycle cold/hot test
  (`thermal_control_v2.py`): an identical 2k task ran 15–42% slower once the chip was hot, and
  recovered after an 8-min cooldown; CPU clocks slid ~3.2 → ~1.1 GHz. The L4, actively cooled, gave a
  clean monotonic curve (3 passes agreeing to 2 decimals).
- **"Peak memory" differs by runtime:** MLX grows the KV cache dynamically (→9.3 GB); vLLM
  pre-reserves it (13.4 GiB). Shared honest metric: **~128 KiB/token** (matches vLLM's own reservation,
  13.37 GiB / 109,504 tokens).

## Eviction leg: what a rotating KV window costs

Same model, same 4 lengths, one new variable: `prompt_cache` built as `RotatingKVCache(max_size=4096,
keep=4)` per layer instead of the default unbounded cache — the memory-saving lever most inference
stacks offer for long context.

| context | full KV peak | rotating KV peak | full KV recall | rotating KV recall |
|---|---|---|---|---|
| 2k | 5.3 GB | 5.3 GB | 5/5 | 5/5 |
| 8k | 6.1 GB | 5.9 GB | 5/5 | **0/5** |
| 16k | 7.2 GB | 6.0 GB | 5/5 | **0/5** |
| 32k | 9.3 GB | 6.0 GB | 5/5 | **0/5** |

- **Peak memory plateaus at the window** (~6 GB regardless of context length) instead of growing
  linearly — 3.3 GB saved at 32k (36%).
- **Recall doesn't degrade, it collapses** — 5/5 to 0/5, the instant context crosses 4096 tokens, on
  every single trial (8/8 recorded above-window runs, zero exceptions — 11/11 counting the 3
  discarded warmup runs, which also all showed 0/5). The 5 facts are planted near the front
  of the document (see caveat below); a rotating window keeps the *most recent* N tokens, so early
  content is exactly what gets evicted first — this is the expected failure mode, now measured.
- **`keep=4` (StreamingLLM's own default sink size) provides zero protection here.** Attention sinks
  exist to keep the softmax from destabilizing under eviction (garbage/repetition), not to preserve
  arbitrary important content — those are two different problems. Don't conflate "keeps output fluent"
  with "keeps your facts."
- **Methodology note:** n=3 (median, interleaved) at every length except 32k, which is n=2 — the
  machine hit real memory pressure (swap >90%) partway through the third 32k pass, twice, so that run
  was killed rather than risk degrading the rest of the system. Both recorded 32k trials agree with
  each other and with all 8 recorded above-window trials combined.
- **Correction: the effective window during prefill is larger than the configured `max_size`.**
  MLX's `RotatingKVCache` trims strictly to `max_size` during decode (`_update_in_place`, one
  token at a time) — but during prefill (`_update_concat`, multi-token chunks) it can transiently
  hold up to `max_size + S - 1` tokens, where `S` is the prefill chunk size. mlx-lm's
  `stream_generate` defaults to `prefill_step_size=2048`, so for a prompt processed in multiple
  chunks the peak footprint during prefill can reach ~6143 tokens' worth of cache, not a strict
  4096 — confirmed directly against `mlx_lm/models/cache.py`'s own source comment ("The largest
  size is `self.max_size + S - 1`"). `mx.get_peak_memory()` captures the all-time high across the
  whole run, so the peak-GB numbers above likely reflect this larger prefill-time transient, not
  the tighter decode-time steady state. **The plateau finding itself is unaffected** — peak memory
  is still capped by a fixed constant independent of total context length, which is the actual
  claim — but "window=4096" overstates the precision of what's being measured; the true effective
  window during prefill sits somewhere in the 4096–6143 range, not pinned down exactly here.
  Caught by a Codex review before this repo's commits were pushed; not re-run with an explicit
  smaller `prefill_step_size` to nail the exact number, which is the natural follow-up if this
  matters for a specific claim later.

Chart: `kv_cache_eviction_chart.png`.

## Honest caveats (read before quoting)

- Two different 4-bit schemes: MLX 4-bit (M3) vs AWQ INT4 (L4). Read the two as **within-stack curves**,
  not a controlled hardware A/B.
- **Early-needle recall:** the 5 facts are placed near the front, not scattered through the document.
- **L4 decode is stream-chunk-counted** (vLLM streams ~1 token/chunk); exact token-count is a v2 fix.
- n as noted; M3 medians of 3 interleaved passes, warmed.

## Reproduce

```bash
# M3 leg (Apple Silicon, mlx-lm):
pip install mlx mlx-lm transformers
caffeinate -dimsu python run_m3.py            # 2k/8k/16k/32k, warmed, median-of-3 interleaved
python thermal_control_v2.py                  # 3 cold/hot cycles + cooldown (proves the throttle)

# L4 leg (rented NVIDIA GPU, vLLM) — see L4_RUNBOOK.md for the full on-box sequence:
#   docker run ... vllm/vllm-openai:latest --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 ...
python run_l4.py                              # same sweep against the vLLM server

python make_chart.py                          # renders kv_cache_tax_chart.png

# Eviction leg (M3 only, reuses the same model/prompts):
caffeinate -dimsu python run_m3_eviction.py    # full KV vs rotating window (4096, keep=4)
python make_chart_eviction.py                  # renders kv_cache_eviction_chart.png
```

## What's here

| File | Does |
|---|---|
| `run_m3.py` | M3 sweep (mlx-lm): TTFT, decode, peak RAM, recall; median-of-3 interleaved. |
| `run_l4.py` | L4 sweep (vLLM via OpenAI client): same metrics, apples-to-apples build. |
| `thermal_control.py` / `_v2.py` | The controlled thermal test (v2 = 3 cycles + cooldown recovery). |
| `make_chart.py` | Renders the two-panel chart. |
| `L4_RUNBOOK.md` | Exact on-box setup for the rented GPU (Docker + vLLM). |
| `results_m3.jsonl`, `results_l4.jsonl`, `thermal_control_v2.jsonl` | Raw measured data. |
| `l4_env/` | L4 provenance: vLLM startup log (KV reservation line), GPU env. |
| `run_m3_eviction.py` | Eviction leg: full KV vs `RotatingKVCache(max_size=4096, keep=4)`, same sweep. |
| `finish_eviction_p3.py` | Recovery script — completes an interrupted pass without re-running everything (see methodology note above). |
| `make_chart_eviction.py` | Renders the eviction-leg comparison chart. |
| `results_m3_eviction.jsonl`, `results_m3_eviction_raw.jsonl` | Eviction leg raw + aggregated data. |

## License

MIT
