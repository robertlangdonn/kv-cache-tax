# L4 / vLLM leg — on-box runbook (Jarvislabs)

Paste blocks in order. The ₹41/hr clock is running — but the sweep itself is ~15 min; total ~₹60.
**DESTROY THE VM the moment `results_l4.jsonl` is written.**

## 0. Launch config (Jarvislabs web UI)
- **Resource:** L4 · IN2 · 24 GB · ₹41.31/hr
- **VM name:** `kv-tax-l4-awq`
- **Storage:** **100 GB** (cheap safety; avoids a full-disk mid-download)
- **Image:** Ubuntu 22.04/24.04 with Docker + NVIDIA runtime (preferred), OR a PyTorch CUDA 12.x image. **Avoid CUDA 11.x.**

## 1. SSH in, sanity-check the box
```bash
nvidia-smi          # confirm L4 shows up
df -h               # confirm ~100GB free
docker --version    # if this errors, use the pip fallback in section 6
```

## 2. Get run_l4.py onto the box
From your Mac (Jarvislabs shows the ssh host/port):
```bash
scp -P <PORT> ~/"Work/Open-Source Contributions/kv-cache-tax/run_l4.py" root@<HOST>:~/
```
(or just drag-drop it into the Jarvislabs Jupyter file browser)

## 3. Start the vLLM server (Docker path — least likely to break)
```bash
MODEL="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
mkdir -p ~/hf-cache ~/kv-results

docker pull vllm/vllm-openai:v0.24.0

tmux new -d -s vllm "docker run --rm --gpus all --ipc=host --name vllm-l4 \
  -p 8000:8000 -v \$HOME/hf-cache:/root/.cache/huggingface \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  vllm/vllm-openai:v0.24.0 \
  --model \$MODEL --quantization awq --dtype half \
  --max-model-len 34000 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8000"
```

## 4. Wait until ready (downloads ~6GB first time, ~2-4 min)
```bash
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do sleep 5; docker logs --tail 20 vllm-l4; done
curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool   # should list the model
```
**⭐ In the startup log, FIND + SCREENSHOT the line reporting "GPU KV cache size" / "# GPU blocks" / max concurrency — that's the reserved-KV number for the apples-to-apples memory report.**

## 5. Run the client + record environment
```bash
python3 -m venv ~/kv-client && source ~/kv-client/bin/activate
pip install -U pip && pip install openai transformers

# record env for the writeup
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > ~/kv-results/l4_env.csv
docker logs vllm-l4 > ~/kv-results/vllm_startup.log     # has the KV-cache line
nvidia-smi -q -x > ~/kv-results/nvidia_smi_q.xml

export KVTAX_MODEL="$MODEL"
export KVTAX_GPU="L4-24GB"     # exact GPU name — RECORD IT
python -u run_l4.py 2>&1 | tee ~/kv-results/run_l4.log
```

## 6. FALLBACK if Docker isn't available on the image
```bash
python3 -m venv ~/vllm-env && source ~/vllm-env/bin/activate
pip install -U pip
pip install "vllm==0.24.0" --extra-index-url https://download.pytorch.org/whl/cu129
pip install openai transformers
# then in a tmux/second shell:
vllm serve "$MODEL" --quantization awq --dtype half --max-model-len 34000 \
  --max-num-seqs 1 --gpu-memory-utilization 0.90 --no-enable-prefix-caching --port 8000
# wait for ready (section 4), then run the client (section 5, skip the docker logs line)
```

## 7. Get results back to your Mac, then DESTROY the VM
```bash
# on your Mac:
scp -P <PORT> root@<HOST>:~/results_l4.jsonl ~/"Work/Open-Source Contributions/kv-cache-tax/"
scp -P <PORT> -r root@<HOST>:~/kv-results ~/"Work/Open-Source Contributions/kv-cache-tax/l4_env/"
```
Then in the Jarvislabs UI: **Destroy the VM.** (Billed until you do.)

## Most likely break (Codex) + fix
CUDA/Torch/vLLM mismatch → symptoms: "CUDA driver version is insufficient", "undefined symbol",
Triton import error. **Fix: use the Docker path (section 3).** If already Docker and still broken,
relaunch the VM on a newer-CUDA image. Keep `VLLM_ENABLE_CUDA_COMPATIBILITY=1`.

## What to bring back here (for the chart + post)
`results_l4.jsonl` + the `kv-results/` folder (env + vllm_startup.log with the KV-cache line).
Then we build: one chart (M3 TTFT+RAM curves + L4 clean decode curve) + 800-word post.
