# Quantized-KV leg — on-box runbook (Jarvislabs L4)

Two arms, one box, one session: baseline fp16 KV vs `--kv-cache-dtype fp8`.
Budget: ~1.5 hr on L4 ₹41.31/hr ≈ **₹65–75** (standing ceiling ₹200).
**DESTROY THE VM the moment both arms' results + logs are copied back.**

Lesson from v1 carried forward: use image `vllm/vllm-openai:latest`, NOT `v0.24.0`
(v0.24.0 hit a CUDA-803 driver mismatch on this provider's L4 image).

## 0. Launch (Jarvislabs)
- L4 · 24 GB · ₹41.31/hr · storage 100 GB · VM name `kv-quant-l4`
- Ubuntu image with Docker + NVIDIA runtime.

## 1. Sanity-check
```bash
nvidia-smi && df -h && docker --version
```

## 2. Copy the client onto the box (from the Mac)
```bash
scp -P <PORT> ~/"Work/Open-Source Contributions/kv-cache-tax/run_l4_kvquant.py" root@<HOST>:~/
```

## 3. ARM 1 — baseline (fp16 KV)
```bash
# Model name INLINED in the tmux command below -- an unexported shell variable
# is invisible inside the tmux session (\$MODEL would expand empty and the
# server would fail while the readiness loop waits forever, on the clock).
mkdir -p ~/hf-cache ~/kv-results
docker pull vllm/vllm-openai:latest

tmux new -d -s vllm "docker run --rm --gpus all --ipc=host --name vllm-l4 \
  -p 8000:8000 -v \$HOME/hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq --dtype half \
  --max-model-len 34000 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8000"

# readiness loop with a visible heartbeat + a hard timeout so a dead server
# can't silently burn the rental
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5; docker logs --tail 3 vllm-l4 2>&1 | tail -1
  [ "$i" = 120 ] && { echo "SERVER NEVER CAME UP -- check docker logs vllm-l4"; exit 1; }
done
docker logs vllm-l4 2>&1 | grep -iE "kv cache|gpu blocks|dtype" > ~/kv-results/startup_fp16.log
docker logs vllm-l4 > ~/kv-results/vllm_startup_fp16_full.log
```
⭐ `startup_fp16.log` must contain the **"GPU KV cache size" / "# GPU blocks"** line — record it.

```bash
python3 -m venv ~/kv-client && source ~/kv-client/bin/activate
pip install -U pip && pip install openai transformers

export KVTAX_MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
export KVTAX_GPU="L4-24GB"
export KVTAX_KV_DTYPE="fp16"
python -u run_l4_kvquant.py 2>&1 | tee ~/kv-results/run_fp16.log
```

## 4. ARM 2 — fp8 KV (same box, restart server with one extra flag)
```bash
docker stop vllm-l4; sleep 5

tmux new -d -s vllm8 "docker run --rm --gpus all --ipc=host --name vllm-l4 \
  -p 8000:8000 -v \$HOME/hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --quantization awq --dtype half --kv-cache-dtype fp8 \
  --max-model-len 34000 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8000"

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null && break
  sleep 5; docker logs --tail 3 vllm-l4 2>&1 | tail -1
  [ "$i" = 120 ] && { echo "SERVER NEVER CAME UP -- check docker logs vllm-l4"; exit 1; }
done
docker logs vllm-l4 2>&1 | grep -iE "kv cache|gpu blocks|dtype" > ~/kv-results/startup_fp8.log
docker logs vllm-l4 > ~/kv-results/vllm_startup_fp8_full.log

export KVTAX_KV_DTYPE="fp8"
python -u run_l4_kvquant.py 2>&1 | tee ~/kv-results/run_fp8.log
```
⭐ The fp8 startup log's KV-capacity line vs fp16's = the **memory half of the result**
(same 0.90 reservation — capacity in tokens should ~double). Also record the exact
fp8 variant it chose (e4m3/e5m2) from the dtype line.

## 5. Env provenance + copy back + DESTROY
```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/kv-results/l4_env.csv
```
From the Mac:
```bash
scp -P <PORT> root@<HOST>:~/results_l4_kvquant*.jsonl ~/"Work/Open-Source Contributions/kv-cache-tax/"
scp -P <PORT> -r root@<HOST>:~/kv-results ~/"Work/Open-Source Contributions/kv-cache-tax/l4_kvquant_env/"
```
Then **Destroy the VM** in the Jarvislabs UI (billed until destroyed).

## What this answers (the truth-table cell)
M3/MLX: quantized KV = higher RAM + kv8 4× slower + OOM @32k (all cost, no benefit).
L4/vLLM fp8: does capacity double at the same reservation, what does decode/TTFT pay,
does 5/5 recall survive? Both directions are a real result.
