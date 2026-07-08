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

## License

MIT
