# vLLM docs-PR supporting run (2026-07-19)

Data behind the "Performance Considerations" section proposed for vLLM's
`docs/features/quantization/quantized_kvcache.md`: Qwen/Qwen3-4B on a rented
NVIDIA L4 24GB, vLLM 0.25.1, `auto` (BF16) vs `fp8` KV cache, batch 1, prefix
caching off, `--max-model-len 32768`, median of 3 interleaved passes after a
discarded warmup pass. Client: `../run_l4_kvquant.py`.

- `kv-results/startup_{auto,fp8}.log` — engine startup capacity lines (83,488
  vs 166,992 tokens at the same 11.47 GiB reservation; concurrency 2.55x/5.10x).
- `results_l4_kvquant.jsonl` / `_raw.jsonl` — sweep data.
- `kv-results/example{1,2}.log` — the docs page's two example snippets run
  verbatim with the replacement model.
- Caveat: the recall column in this run is invalid — Qwen3-4B's thinking mode
  consumes the 64-token generation budget, so the probe measures thinking
  verbosity, not retention. Not used in the docs PR.
