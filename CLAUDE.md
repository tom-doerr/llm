# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# DGX Spark 200G RDMA Setup

## Key Facts

- **RoCE over Ethernet** via ConnectX-7 QSFP ports (NOT InfiniBand mode)
- **GPUDirect RDMA NOT supported** on GB10 (dma-buf/nvidia-peermem don't work)
- **NCCL IB works with GDR disabled** - ~22 GB/s proven; NGC container may need plugin workarounds
- **200 Gb/s = 25 GB/s = ~23.28 GiB/s** line-rate before overheads

## Hardware Quirk: Each Port Has Two "Halves"

Each 200G QSFP port = **two logical interfaces** (need IPs on BOTH for full 200 Gb/s):
- Port 0: `enp1s0f0np0` + `enP2p1s0f0np0`
- Port 1: `enp1s0f1np1` + `enP2p1s0f1np1`

**Interface naming:** Capital P in `enP2p1s0f1np1` = different PCI domain. Not a typo.

Use `ibdev2netdev` to check status, `ethtool <iface>` to verify 200000Mb/s link.

## Cabling

- Start with **one QSFP DAC/AOC cable**
- **Same port to same port**: port 1↔1, port 2↔2 (cross-connecting fails)
- One cable achieves full bandwidth

## IP Setup (RoCE requires L3 connectivity)

### Option A: NVIDIA's netplan link-local (recommended)
```bash
sudo wget -O /etc/netplan/40-cx7.yaml \
  https://github.com/NVIDIA/dgx-spark-playbooks/raw/main/nvidia/connect-two-sparks/assets/cx7-netplan.yaml
sudo chmod 600 /etc/netplan/40-cx7.yaml
sudo netplan apply
```

### Option B: Static IPs on BOTH halves (for full 200 Gb/s)
```bash
# Node A - both halves of port 1
sudo ip addr add 192.168.177.11/24 dev enp1s0f1np1
sudo ip addr add 192.168.177.12/24 dev enP2p1s0f1np1

# Node B - both halves
sudo ip addr add 192.168.177.21/24 dev enp1s0f1np1
sudo ip addr add 192.168.177.22/24 dev enP2p1s0f1np1
```
Set MTU 9000 and disable IPv6 for stable RoCE GID indexing.

## Verification Commands

```bash
# 1. Check which interfaces are up
ibdev2netdev

# 2. Confirm link-layer is Ethernet (RoCE)
ibv_devinfo -v | grep "Link Layer"

# 3. Ping test
ping -c 3 192.168.100.11  # from spark-1
```

## RDMA Bandwidth Test

```bash
# Install tools
sudo apt-get install -y rdma-core ibverbs-utils perftest

# Server (spark-1)
sudo ib_write_bw -R --report_gbits -d rocep1s0f1 -q 4 -s 8388608 -D 10

# Client (spark-2)
sudo ib_write_bw -R --report_gbits -d rocep1s0f1 -q 4 -s 8388608 -D 10 192.168.100.10
```

Flags: `-R` (rdma_cm for RoCE), `-q 4` (4 QPs), `-s 8M` (message size), `-D 10` (10s duration)

### Full 200G Test (both interfaces in parallel)
```bash
# Server (spark-2) - run both
sudo ib_write_bw -R --report_gbits -d rocep1s0f1 -p 18515 &
sudo ib_write_bw -R --report_gbits -d roceP2p1s0f1 -p 18516 &

# Client (spark-3) - run both
sudo ib_write_bw -R --report_gbits -d rocep1s0f1 -p 18515 192.168.100.10 &
sudo ib_write_bw -R --report_gbits -d roceP2p1s0f1 -p 18516 192.168.101.10 &
```

## NCCL/MPI Environment Variables

```bash
export UCX_NET_DEVICES=enp1s0f1np1
export NCCL_SOCKET_IFNAME=enp1s0f1np1
export OMPI_MCA_btl_tcp_if_include=enp1s0f1np1
```

**NCCL IB (Jan 2026):** GPUDirect RDMA not supported on GB10 (dma-buf/nvidia-peermem don't work).
NCCL *can* run IB with GDR disabled (~22 GB/s / 176 Gb/s proven). Our NGC container crashes due to
DMABUF plugin issue. **Quick fix:** `NCCL_NET=Socket`. **Better fix:** see below.

### NCCL IB with GDR Disabled (Jan 2026)

**Key:** Set env vars via `docker run -e`, NOT later via `docker exec`. Ray workers inherit
container env but NOT driver shell env. Also need device access + memlock.

```bash
docker run ... \
  --device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK \
  -e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0 \
  -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0 \
  -e NCCL_IB_HCA="=rocep1s0f1:1,roceP2p1s0f1:1" \
  ray start ...
```

## Socket vs RDMA Limits (Jan 2026)

| Transport | Practical | Notes |
|-----------|-----------|-------|
| Socket | 4-6 GB/s | PCIe+CPU bound |
| RDMA (GDR off) | ~22 GB/s | Proven on Spark |

**vLLM measured:** **79% throughput gain** with RDMA over Socket (299 vs 167 tok/s at c=256).

**Latency matters:** RDMA's ~1-2μs latency (vs Socket ~1-2ms) helps tensor parallelism all-reduce ops.

**Single vs Dual-rail:** Dual-rail adds CPU staging overhead with GDR disabled.
- Single: `NCCL_IB_HCA='=rocep1s0f1:1'`
- Dual: `NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1'`

**Current config:** Dual-rail (Jan 2026). `NCCL_IB_HCA='=rocep1s0f1:1,roceP2p1s0f1:1'`
**Benchmark (old single-rail):** Single-rail 299 tok/s, dual-rail 239 tok/s at c=256.

**Sources:** [NCCL env vars](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [vLLM distributed troubleshooting](https://docs.vllm.ai/en/stable/serving/distributed_troubleshooting/), [Spark 22GB/s](https://forums.developer.nvidia.com/t/dgx-spark-nccl-test-10gb-s-not-200-gbps-25-gb-s/350077)

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Link UP but ping fails | Use ONE cable, same port↔port |
| `ib_write_bw` socket error | Wrong interface/GID, use `-R -x <gid>` |
| ~100-112G per interface | Expected - use BOTH interfaces for full 200G |
| GPUDirect RDMA fails | Not supported on Spark, use host buffers |

## Current Setup: spark-2 ↔ spark-3

| Interface | spark-2 IP | spark-3 IP | RDMA device |
|-----------|------------|------------|-------------|
| enp1s0f1np1 | 192.168.100.10 | 192.168.100.11 | rocep1s0f1 |
| enP2p1s0f1np1 | 192.168.101.10 | 192.168.101.11 | roceP2p1s0f1 |

**Tested:** ~106 Gb/s per path, ~212 Gb/s aggregate (Jan 2026)

**Config:** `/etc/netplan/40-cx7.yaml` on both machines (persistent)

## 10GbE Link: spark-1 ↔ spark-2

Interface `enP7s7` on both machines.

| Host | IPv4 | IPv6 link-local |
|------|------|-----------------|
| spark-1 | 192.168.102.10/24 | fe80::55d3:2800:283f:4655%enP7s7 |
| spark-2 | 192.168.102.11/24 | fe80::4b2e:d2df:bb09:5f59%enP7s7 |

**Config:** `/etc/netplan/50-10gbe.yaml` on spark-2 (persistent)

**Tested:** ~1 GB/s rsync throughput (Jan 2026)

## 200G QSFP Links: spark-1 ↔ spark-2 and spark-1 ↔ spark-3

Both links UP at 200G but **no IPs assigned** (port 0: `enp1s0f0np0`).
Needed for PP=3 Qwen3.5 deployment across all 3 Sparks.

## Multi-Node Inference (Qwen3-VL-235B-AWQ)

### SGLang Multi-Node TP=2 (Jan 2026)

**STATUS: SGLang multi-node AWQ NOT WORKING** - Hangs after NCCL init, model never loads. Use vLLM instead.

Container: `nvcr.io/nvidia/sglang:25.11-py3`

**Env vars (both nodes):**
```bash
export MN_IF_NAME=enp1s0f1np1
export NCCL_SOCKET_IFNAME=$MN_IF_NAME
export GLOO_SOCKET_IFNAME=$MN_IF_NAME
export UCX_NET_DEVICES=$MN_IF_NAME
export SGLANG_VLM_CACHE_SIZE_MB=8192
```

**spark-2 (rank 0):**
```bash
python3 -m sglang.launch_server \
  --model-path QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ \
  --tp 2 --nnodes 2 --node-rank 0 --dist-init-addr 192.168.100.10:20000 \
  --mem-fraction-static 0.75 --context-length 4096 --max-running-requests 16 \
  --attention-backend flashinfer --host 0.0.0.0 --port 30000 --trust-remote-code
```

**spark-3 (rank 1):** Same but `--node-rank 1`

**AWQ note:** Just use AWQ model path directly, no special flags needed

**Test endpoint:**
```bash
curl http://192.168.100.10:30000/generate -H "Content-Type: application/json" \
  -d '{"text":"Hello","sampling_params":{"max_new_tokens":64}}'
```

**SGLang issues (Jan 2026):**
- Hangs after NCCL init - model load never starts
- NCCL 16 channels via RDMA work, but AWQ loader hangs
- Use vLLM instead (proven working)

**Model:** `HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ` (117GB)

**Before launch (BOTH nodes):**
```bash
sudo swapoff -a
docker rm -f vllm-head vllm-worker 2>/dev/null
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'  # Recommended for unified memory
```

## Qwen3-VL-235B on 2× Spark: Key Facts

- **Must use quantized** (AWQ or NVFP4) - BF16/FP8 won't fit
- **TP=2 required** across two Sparks (one GPU each)
- **Cap context to 4K** - 16K causes NVIDIA driver OOM → hard crash
- **Drop OS caches** before launch: `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'`
- KV scales linearly: 16K = 4× memory of 4K

### vLLM + Ray Multi-Node Setup (Jan 2026)

Container: `nvcr.io/nvidia/vllm:25.11-py3`

**VERIFIED WORKING (Jan 2026):** ~3 min model load, ~5s TTFT

**Startup script:** `./start-vllm-multinode.sh` (deploys Ray + vLLM across spark-2/spark-3, dual-rail RDMA, mounts custom vLLM with async encoder)

Docs: https://build.nvidia.com/spark/vllm/stacked-sparks

**CRITICAL env vars (both nodes):**
```bash
export MN_IF_NAME=enp1s0f1np1
export NCCL_SOCKET_IFNAME=$MN_IF_NAME
export GLOO_SOCKET_IFNAME=$MN_IF_NAME
export UCX_NET_DEVICES=$MN_IF_NAME
export RAY_memory_monitor_refresh_ms=0
```

**Socket vs RoCE:** vLLM guide uses socket transport (~100 Gb/s). For full ~200 Gb/s, need NCCL NET/IB (RoCE) with IPs on BOTH halves of the port. GPUDirect unsupported but host-staged RoCE works.

**Max throughput path:** TRT-LLM + NVFP4 (23,477 tok/s prefill, 11.73 tok/s decode). See [build.nvidia.com/spark/trt-llm/stacked-sparks](https://build.nvidia.com/spark/trt-llm/stacked-sparks)

**cuDNN Fix (REQUIRED):** GB10 lacks cuDNN conv3d kernels for sm_121. Mount import hook:
```bash
-v /home/tom/sitecustomize.py:/usr/lib/python3.12/sitecustomize.py:ro
```

**Attention Backend Fix (REQUIRED for VLM):** Vision encoder profiling hangs without this:
```bash
-e VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

**HF_HUB_OFFLINE=1** required when DNS is broken (common on Spark WiFi issues).

vLLM serve (auto-configured 256K context, Feb 2026):
```bash
vllm serve QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ \
  --tensor-parallel-size 2 --trust-remote-code \
  --quantization awq --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 4096 \
  --scheduling-policy priority --mm-encoder-tp-mode data \
  --limit-mm-per-prompt '{"video": 0}' --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --distributed-executor-backend ray --host 0.0.0.0 --port 8000
```

**Key:** `--gpu-memory-utilization 0.70` (0.65 crashes - only 8GB KV, 0.75 crashes during capture).

**Eager mode (REQUIRED for multi-node):** `--enforce-eager` disables CUDA graphs. CUDA graphs **hurt** multi-node perf on Spark: 2x worse at c=1 (8 vs 17 tok/s), ~20% worse at c=256 (200 vs 249 tok/s). Likely due to Ray compiled DAG + piecewise graph overhead.

**Tool calling:** `--enable-auto-tool-choice --tool-call-parser hermes` enables function/tool calling.

**Video disabled:** `--limit-mm-per-prompt '{"video": 0}'` saves ~4GB KV (17GB vs 13GB per node). Encoder cache 16K tokens vs 153K with video.

**VLM encoder profiling:** ~5 min with video disabled, ~1 hour with video enabled.

**VLLM_USE_RAY_COMPILED_DAG=0:** Doesn't work - vLLM V1 forces it to 1 for multi-node.

**Auto-config (Jan 2026):** vLLM auto-detects available memory and configures 256K context with ~547K token KV cache (34K blocks × 16). No `--max-model-len` needed.

**Memory:** 97GB/119GB used. Enough for ~2 concurrent 256K requests or many shorter ones.

**Benchmark (Jan 2026 - CUDA graphs, 0.70 mem util, dual-rail RDMA, video disabled):**

| Concurrency | dec tok/s | p50 |
|-------------|-----------|-----|
| 1 | 3.3 | 9.8s |
| 64 | 117 | 14.4s |
| 256 | **157** | 28.3s |

**Single request:** ~7 tok/s streaming, **TTFT ~9s**. Script: `benchmark_vllm.py --sweep`

**Thinking variant (Feb 2026 - enforce-eager, 4096 batch tokens):** ~324 tok/s at c=14 (decode-only, ~23 steps/s).

**Instruct + async encoder (Feb 2026):**

| Concurrency | dec tok/s | p50 |
|-------------|-----------|-----|
| 1 | 17.1 | 1.87s |
| 64 | 250.1 | 7.64s |
| 256 | **248.9** | 18.95s |

**Results saved to:** `benchmark_results/<timestamp>_sweep.json`

**Prefill benchmark:** `benchmark_vllm.py --prefill`
| Tokens | Enc tok/s |
|--------|-----------|
| 1K | 1,270 |
| 8K | 1,010 |
| 32K | 525 |
| 64K | 319 |
| 128K | 177 |

**Image benchmark:** `benchmark_vllm.py --image`

| Resolution | c=32 tok/s |
|------------|------------|
| 256×256 | 371 |
| 512×512 | 862 |
| 1024×1024 | 950 |
| 2048×2048 | 1,320 |

**Note:** `--mm-encoder-tp-mode data` enabled for ~2x image throughput. Encoder profiling takes ~1 hour on first startup.

**vllm bench serve results (Feb 2026 - Instruct, enforce-eager, async encoder):**

Text-only:
| Config | Out tok/s | Peak | TTFT p50 | TPOT p50 |
|--------|----------|------|---------|---------|
| c=1, 128/128 | 10 | 22 | 120ms | 46ms |
| c=50, 128/128 | 175 | 300 | 3.6s | 254ms |
| c=100, 512/256 | 216 | 400 | 16.4s | 394ms |
| 2rps, 256/256 | 55 | 90 | ~0ms | 126ms |

Multimodal (random-mm, synthetic images):
| Config | Out tok/s | Peak | TTFT p50 | TPOT p50 |
|--------|----------|------|---------|---------|
| c=30, 256/512px | 56 | 174 | 34.7s | 247ms |
| c=20, 720p | 13 | 133 | 101s | 641ms |
| 1rps, 256/512px | 42 | 120 | 17.8s | 221ms |

**Results:** `benchmark_results/vllm-bench/*.json`

**vllm bench serve usage:** `docker exec vllm-head vllm bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url http://localhost:8000 --model <model> --dataset-name random-mm --trust-remote-code` (termplotlib+gnuplot installed for plots)

**Video support:** Tested up to 541MB / 30 min (1.2GB / 1hr crashes server). Processing time ~4 min regardless of length (frame sampling). Send as base64 data URL:
```python
{"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}}
```

**Encoder cache:** vLLM automatically caches image encoder outputs by content hash (blake3). No explicit ID needed - same image bytes = cache hit. Cache is in-memory on worker node.

**Async encoder (custom branch):** ViT runs on a separate CUDA stream overlapping with CPU input prep. See `vllm/` clone below.

**Metrics note:** `prompt_tokens_total` includes cache hits. Real prefill compute = queries - hits.
**Chunked prefill:** Enabled by default in vLLM V1. Tune via `--max-num-batched-tokens`.

**Priority scheduling:** Use `extra_body={"priority": N}` in API calls. Lower N = higher priority (0 = highest).

**Port:** 8000 (default). Use same port for vLLM/SGLang for consistent client code.

**Access from spark-1:** Use 10GbE IP `192.168.102.11:8000` (200G IPs only work between spark-2/spark-3)
```bash
curl http://192.168.102.11:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

**Deps:** `pip install qwen-vl-utils==0.0.14` (required for Qwen-VL)

### Head vs Worker Node Resource Usage (Jan 2026)

**spark-2 (head)** uses more CPU/RAM than **spark-3 (worker)**:

| State | spark-2 (head) | spark-3 (worker) |
|-------|----------------|------------------|
| Idle | ~280% CPU (load avg ~3) | ~100% CPU |
| Under load | ~1000% CPU (load avg ~13) | ~350% CPU |
| RAM | 112GB used, <2GB free | 106GB used, ~6GB free |

**Why head is heavier:** EngineCore (scheduling, KV mgmt), tokenization (CPU-only), vision preprocessing (image decode/resize before GPU), Ray GCS server.

**`--mm-encoder-tp-mode data`:** GPU data parallel for vision encoder. **ENABLED** - encoder profiling takes ~1 hour on first startup but succeeds. ~2x image throughput.

**Reduce CPU load:** Resize images client-side before API calls (<100KB).

### Ray Compiled DAG CPU Overhead (Jan 2026)

**Root cause of ~1600% CPU:** EngineCore busy-polling + Ray compiled DAG channels.

**Fix:** `VLLM_SLEEP_WHEN_IDLE=1` - drops idle CPU from ~1600% to ~280% (load avg 18→3).

**Multi-node limitation:** vLLM V1 forces compiled DAG on. `VLLM_USE_RAY_COMPILED_DAG=0` ignored.

**Result:** CPU idle 8%→78%, load avg 18→3. Small latency cost on wake-up (acceptable).

**Gotcha:** Stale/disconnected client connections still show as `num_requests_running` in metrics.
When all real work is done but stale requests remain, engine sleeps → 0 tok/s despite N "running".
A new request wakes the engine back up. Not a real problem, just misleading metrics.

**CPU tuning (in script):**
- `OMP_NUM_THREADS=1` - Enabled, reduces threading overhead
- `RAY_DEDUP_LOGS=1` - Optional, reduces log overhead
- `--mm-processor-cache-gb 1` - Optional, reduces image cache

### Shared CPU/GPU Power Budget

DGX Spark has a **shared power limit** for CPU and GPU. High CPU load steals power from GPU, reducing inference throughput. Minimize CPU overhead during inference.

### GPU Idle Power (NCCL Polling)

Multi-node TP causes **~96% GPU util even when idle** due to NCCL busy-polling.
Both nodes draw 60-85W idle. Inherent to keeping RDMA connection "hot".
**No fix** - accept power cost or stop vLLM when not in use.

### Head vs Worker Clock Speeds

Worker (spark-3) runs higher clocks: ~2400 MHz, 85W (pure tensor compute).
Head (spark-2) runs lower: ~2100 MHz, 60W (mixed scheduling + compute).
This is normal - pure matmul workloads boost higher.

### Verify RoCE/IB is Active (not Socket)

Check RDMA counters are incrementing during inference:
```bash
# Single rail
cat /sys/class/infiniband/rocep1s0f1/ports/1/counters/port_xmit_data
# Both rails (for dual-rail config)
cat /sys/class/infiniband/roceP2p1s0f1/ports/1/counters/port_xmit_data
```
If counters increase, IB is working. For detailed logs: `./start-vllm-multinode.sh --debug`

### Scheduler Tuning

**`max_num_batched_tokens`:** Set to 4096 (Feb 2026). Higher values (16384, 32768) caused NVIDIA driver OOM under load on unified memory.
Cached tokens count against budget but cost no compute — higher helps concurrency but risks OOM.

**`max_num_partial_prefills`:** Default 1. **Dead code in vLLM V1 0.11.0** — the V1 scheduler
does NOT reference this setting. Mixed prefill (prefill + decode in same step) works by default.
The scheduler schedules all RUNNING requests first (1 decode token each), then fills remaining
budget with WAITING request prefills. No artificial serialization.

**Note:** Earlier vLLM versions forced V0 fallback when >1. In 0.11.0 V1, this is irrelevant.

**`--gpu-memory-utilization`:** 0.70 minimum. 0.65 hangs (no KV cache room).

### TP vs PP Mode

**TP (default):** `./start-vllm-multinode.sh` - Tensor parallel, splits layers across GPUs
**PP mode:** `./start-vllm-multinode.sh --pp` - Pipeline parallel, each GPU runs different layers

Both modes work. Heavy prefill queues can make throughput appear zero initially - wait for ramp-up.

### Auto-Restart on Failure (Feb 2026)

Two layers: Docker `--restart=on-failure:10` on both containers + vLLM retry loop (10 retries, 15s delay) in head entrypoint.

**Entrypoints:** `vllm-head-entrypoint.sh` (Ray head + vLLM retry loop), `vllm-worker-entrypoint.sh` (Ray worker with retry-connect loop)

**Recovery:** vLLM crash → head retries vLLM. Container crash → Docker restarts → worker retries Ray join → head waits for worker → vLLM starts.

## Model Cache (Jan 2026)

**spark-2** (~700GB): AWQ Instruct/Thinking (116+117G), NVFP4 variants (127G each), BF16 Thinking (214G)

**spark-3**: AWQ Instruct + Thinking (116+117G)

**SSH:** spark-2 ↔ spark-3 keys configured.

**Working (Jan 2026):** vLLM TP=2 with sitecustomize.py cuDNN disable + FP8 KV cache + HF_HUB_OFFLINE=1

**Model load:** ~3 min (42 shards) + ~5 min encoder profiling. **TTFT:** ~5s.

**KV Cache:** 25.77 GiB per node, 256K max context, ~51GB total across TP=2

## Qwen3.5-122B-A10B-FP8 (Feb 2026)

**Status:** RUNNING on vLLM TP=2, spark-2 + spark-3.
**Model:** `Qwen/Qwen3.5-122B-A10B-FP8` | **Container:** `vllm/vllm-openai:qwen3_5-cu130`
**Script:** `./start-vllm-multinode.sh` | **API:** `http://192.168.102.11:8000/v1/chat/completions`

**Config fix:** `rope_theta: 10000000` added to `text_config` (missing from HF, defaults to wrong 10000).
**MoE:** `VLLM_TEST_FORCE_FP8_MARLIN=1` — CUTLASS crashes on sm_121a.
**Memory:** 59.1 GiB/node, 0.70 util, FP8 KV. **TTFT:** ~6s. **Load:** ~80s worker, ~474s head.
**Multimodal:** Images enabled (`--limit-mm-per-prompt '{"video": 0}'`). Video disabled.
**Encoder cache:** 128K tokens via `VLLM_ENCODER_CACHE_TOKENS=131072` + `vllm_scheduler_patched.py`.
**Reasoning:** `--reasoning-parser qwen3` separates `<think>` into `reasoning_content` field. Per-request toggle via `extra_body={"chat_template_kwargs":{"enable_thinking": false}}`.
**NVFP4 broken:** `alpertor/Qwen3.5-122B-A10B-NVFP4` produces garbage (missing E2M1 PTX).

**Image benchmark (img_tok/s, 128K encoder cache, 0.70 util):**

| Resolution | c=1 | c=8 | c=16 | c=32 |
|-----------|------|------|-------|-------|
| 256×256 | 108 | 94 | 355 | 541 |
| 512×512 | 161 | 333 | 443 | 575 |
| 1024×1024 | 193 | 399 | 603 | **910** |
| 2048×2048 | 128 | crash | crash | crash |

Peak: **910 tok/s** at 1024×1024 c=32. 2048×2048 c>=8 crashes (Ray timeout).

## Qwen3.5-35B-A3B-FP8 (Feb 2026)

**Status:** RUNNING on spark-2 single-node, port 8001.
**Model:** `Qwen/Qwen3.5-35B-A3B-FP8` (35B total, 3B active per token)
**Script:** `./start-vllm-fast.sh` | **API:** `http://192.168.102.11:8001/v1`
**Memory:** 34.71 GiB, 0.50 util. Runs alongside 122B on port 8000.
**Config fix:** `rope_theta: 10000000` added (same bug as 122B).

## Qwen3.5-397B-A17B llama.cpp Deployment (Feb 2026)

**Status:** RUNNING. Q2_K via llama.cpp RPC across spark-2 + spark-3.

**Script:** `./start-llamacpp-multinode.sh [model_path]`
**Env vars:** `NP=16` (parallel slots), `CTX_SIZE=` (auto-fit), `MMPROJ=<path>` (vision)
**Containers:** `llamacpp-head` (spark-2), `llamacpp-rpc` (spark-3)
**Image:** NGC vllm:26.01-py3 base + mounted `~/llama.cpp/build/bin/`
**API:** `http://192.168.102.11:8000/v1/chat/completions`
**Open WebUI:** http://localhost:7070 (model auto-discovered via OpenAI API)
**Metrics:** `--metrics` flag, Prometheus scrapes `:8000/metrics` (`llamacpp:` prefix)
**Build:** `CMAKE_CUDA_ARCHITECTURES=native` → sm_121a (GB10)

**Performance (single):** ~43 tok/s prefill, ~15.6 tok/s decode (64.7ms/tok)
**Model split:** 67 GB spark-2 (CUDA0) + 71 GB spark-3 (RPC0)
**Architecture:** Layer-split (pipeline parallel via RPC), NOT tensor parallel

**Parallel (np=16):** Text 33.5 tok/s at c=16. Image+decode 13.8 tok/s at c=16.

**Vision:** mmproj-BF16.gguf for image input. Max tokens: 4,113 (>=2048px).
Encoding: 512px=2s, 1024px=5.4s, 2048px=24s. Not parallelized across requests.
Video: not supported natively — send frames as multi-image.

**Slot sizing:** np=128→2K/slot (too small for images). np=16→~16K/slot (good).

## Embeddings Server (spark-3, Feb 2026)

BGE embeddings running alongside vLLM worker on spark-3.

**Script:** `./start-embeddings-spark3.sh`
**Model:** BAAI/bge-base-en-v1.5 (768 dim)
**Port:** 8001 | **API:** http://spark-3:8001/v1/embeddings

**Settings:** `--gpu-memory-utilization 0.02 --max-num-seqs 1 --enforce-eager` (minimal footprint)

**Note:** ModernBERT not compatible with vLLM `--task embed`.

## Instruct vs Thinking Variants

**Current (Feb 2026):** Thinking variant (`QuantTrio/Qwen3-VL-235B-A22B-Thinking-AWQ`)

**Qwen3.5-397B-A17B (Feb 2026):** Q2_K GGUF via llama.cpp (working).
vLLM GGUF had GemmaRMSNorm + MXFP4 nibble bugs → garbage output.
Chat template includes `<think>` block (thinking model).

**Qwen3.5 thinking (Feb 2026):** Controlled via `enable_thinking` parameter (not `/think`/`/no_think` like Qwen3).
- **Default ON:** Model generates `<think>...</think>` before responding.
- **Per-request toggle:** `extra_body={"chat_template_kwargs":{"enable_thinking": false}}`
- **vLLM `--reasoning-parser qwen3`:** Separates thinking into `reasoning_content` field. Without it, thinking is inline in `content`.
- **Recommended temps:** 0.6 (thinking), 0.7 (non-thinking). TopP=0.95, TopK=20.
- **Multi-turn:** Chat template auto-strips thinking from history.

## Model Loading on Spark (Slow mmap Issue)

Default vLLM uses mmap (lazy) which is **pathologically slow on Spark** (~8-9 min).

**Fix 1 - Disable mmap (recommended):**
```bash
--safetensors-load-strategy eager
```
Reduces load to ~1-2 min. Uses +20GB RAM.

**Fix 2 - GPU Direct Storage:**
```bash
pip install -U fastsafetensors
--load-format fastsafetensors
```
Can be dramatically faster. May OOM in TP scenarios.

**Fix 3 - Ensure NVMe not overlay:** `df -Th ~/.cache/huggingface` should show ext4.

**Fix 4 - Increase read-ahead:** `echo 8192 > /sys/block/nvme0n1/queue/read_ahead_kb`

**Why slow:** mmap causes page faults, Spark has 50x slowdown for small H2D copies.

## Grafana Dashboard

**Dashboard:** `grafana/vllm-dashboard.json` (UID: `vllm-spark`)
**URL:** http://localhost:3000/d/vllm-spark
**Title:** "Inference Server (spark-2/spark-3)" — unified for vLLM + llama.cpp

**Metrics:** Queries use `or` to show both `vllm:` and `llamacpp:` prefixes.
Whichever server runs on :8000 gets scraped automatically.

**Exporters:** :8000 (vLLM or llama.cpp), node_exporter :9100, dcgm :9400

**Config files:** `prometheus.yml`, `systemd/node-exporter.service`

**Export:** `curl -s -u admin:admin123 http://localhost:3000/api/dashboards/uid/vllm-spark | jq '.dashboard' > grafana/vllm-dashboard.json`

**Import:** See `grafana/README.md`

## Notes

**OOM killers:** Inactive on spark-2/spark-3 (unlike spark-1). Only kernel OOM kicks in.

**SGLang vs vLLM (Jan 2026):** vLLM is more stable for multi-node AWQ. SGLang hangs after NCCL init.

**`--async-scheduling` (Jan 2026):** NOT compatible with multi-node - forces mp executor which expects multiple local GPUs.

**Cost vs OpenRouter:** ~$0.53/hour savings ($382/month). Break-even ~31 months at current load.

## Helicone (Jan 2026)

**Status:** Dashboard works, ai-gateway missing (no ARM64 build).

**Container:** `helicone/helicone-all-in-one:latest`
**Ports:** Web :3001, Jawn API :8585, Kong :8100
**Config:** `helicone/docker/.env`

**Workaround:** Use header-based logging instead of proxy since ai-gateway not available on ARM64.

## Arize Phoenix (Jan 2026)

**Service:** `~/.config/systemd/user/phoenix.service`
**Port:** 6006
**Data:** `~/.local/share/phoenix/`

`systemctl --user {status|restart|stop} phoenix`

## Langfuse (Jan 2026)

**Dir:** `~/llm/langfuse/`
**Port:** 3100
**Services:** langfuse-web, langfuse-worker, postgres, clickhouse, redis

`docker compose -f ~/llm/langfuse/docker-compose.yml {ps|up -d|down}`

## vLLM Local Clone (Feb 2026)

**Dir:** `~/llm/vllm/` | **Branch:** `custom` (based on `main` at `05339a7b2`)

**Async ViT encoder patch:** Overlaps image encoding with CPU input preparation using a separate CUDA stream.

Files changed:
- `vllm/v1/worker/gpu_model_runner.py` — async launch before `_prepare_inputs`, sync before `_preprocess`
- `vllm/v1/worker/gpu/mm/encoder_runner.py` — cache-check optimization

**Deploy:** Mount into NGC container: `-v ~/llm/vllm/vllm:/usr/lib/python3/dist-packages/vllm`

**CRITICAL:** This mount is REQUIRED. NGC 25.11 ships transformers 5.3.0.dev0 which removed
`all_special_tokens_extended` attribute. The bundled vLLM 0.11.0 references it in
`transformers_utils/tokenizer.py:99`. The local clone fixes this. Without the mount, vLLM
crashes with `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`.

**Guards:** Skips async for LoRA, encoder-decoder, mm_processor_stats. Falls back to sync on error.

**Status:** Deployed and tested on spark-2/spark-3 (Feb 2026). No errors under concurrent image load.
