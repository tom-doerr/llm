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

**Current config:** Single-rail (Jan 2026). Faster than dual-rail with GDR disabled.
**Benchmark:** Single-rail 299 tok/s, dual-rail 239 tok/s at c=256 (~20% slower).

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

**Startup script:** `./start-vllm-multinode.sh` (deploys Ray + vLLM across spark-2/spark-3, dual-rail RDMA)

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

vLLM serve (auto-configured 256K context, Jan 2026):
```bash
vllm serve QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ \
  --tensor-parallel-size 2 --trust-remote-code \
  --quantization awq --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 \
  --max-num-batched-tokens 2048 \
  --scheduling-policy priority --mm-encoder-tp-mode data \
  --distributed-executor-backend ray --host 0.0.0.0 --port 8000
```

**Key:** `--gpu-memory-utilization 0.70` enables CUDA graphs (0.75 crashes during capture). Graphs reduce CPU ~2.5x.

**VLM encoder profiling:** Takes ~1 hour on multi-node TP=2 with `--mm-encoder-tp-mode data`. Not a hang - just slow. Wait for it.

**Auto-config (Jan 2026):** vLLM auto-detects available memory and configures 256K context with ~547K token KV cache (34K blocks × 16). No `--max-model-len` needed.

**Memory:** 97GB/119GB used. Enough for ~2 concurrent 256K requests or many shorter ones.

**Benchmark (Jan 2026 - CUDA graphs, 0.70 mem util, single-rail RDMA):**

| Concurrency | dec tok/s | p50 |
|-------------|-----------|-----|
| 1 | 3.3 | 9.8s |
| 64 | 117 | 14.4s |
| 256 | **157** | 28.3s |

**Single request:** ~7 tok/s streaming, **TTFT ~9s**. Script: `benchmark_vllm.py --sweep`

**Results saved to:** `benchmark_results/<timestamp>_sweep.json` with vLLM config and latency p50/p95/p99

**Prefill benchmark:** `benchmark_vllm.py --prefill`
| Tokens | Enc tok/s |
|--------|-----------|
| 1K | 1,064 |
| 8K | 909 |
| 32K | 459 |
| 64K | 288 |

**Image benchmark:** `benchmark_vllm.py --image`

| Resolution | c=32 tok/s |
|------------|------------|
| 256×256 | 240 |
| 512×512 | 608 |

**Note:** `--mm-encoder-tp-mode data` enabled for ~2x image throughput. Encoder profiling takes ~1 hour on first startup.

**Encoder cache:** vLLM automatically caches image encoder outputs by content hash (blake3). No explicit ID needed - same image bytes = cache hit. Cache is in-memory on worker node.

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

**`max_num_batched_tokens`:** Default 8192 (≥70GB GPU). Lower = smoother streaming, higher = better throughput.
Peak decode ~151 tok/s at c=64 with 32768, reverted to 8192 for lower latency.

### TP vs PP Mode

**TP (default):** `./start-vllm-multinode.sh` - Tensor parallel, splits layers across GPUs
**PP mode:** `./start-vllm-multinode.sh --pp` - Pipeline parallel, each GPU runs different layers

Both modes work. Heavy prefill queues can make throughput appear zero initially - wait for ramp-up.

## Model Cache (Jan 2026)

**spark-2** (~700GB): AWQ Instruct/Thinking (116+117G), NVFP4 variants (127G each), BF16 Thinking (214G)

**spark-3**: AWQ Instruct only (116G)

**SSH:** spark-2 ↔ spark-3 keys configured.

**Working (Jan 2026):** vLLM TP=2 with sitecustomize.py cuDNN disable + FP8 KV cache + HF_HUB_OFFLINE=1

**Model load:** ~3 min (42 shards) + ~5 min encoder profiling. **TTFT:** ~5s.

**KV Cache:** 25.77 GiB per node, 256K max context, ~51GB total across TP=2

## Instruct vs Thinking Variants

**Instruct:** Direct answers, 15-25% faster, no `<think>` tags.
**Thinking:** Always reasons in `<think>` blocks, +11% math accuracy, slower.

**Thinking budget:** `thinking_budget=N` limits reasoning tokens (not yet in vLLM).

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

**Panels (20):** Requests, KV Cache, Throughput stats, Token Throughput, E2E/Prefill Latency, Decode vs Prefill, Prompt/Completion Length (p1/p50/p95/p99), GPU Util/Power/Temp, CPU/RAM %, Network RDMA

**Exporters:** vLLM :8000, node_exporter :9100 (spark-1/2), dcgm-exporter :9400 (spark-2)

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
