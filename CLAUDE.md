# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# DGX Spark 200G RDMA Setup

## Key Facts

- **RoCE over Ethernet** via ConnectX-7 QSFP ports (NOT InfiniBand mode)
- **GPUDirect RDMA NOT supported** on Spark - use `cudaHostAlloc` + `ib_reg_mr` instead
- **200 Gbit/s = 25 GB/s = ~23.28 GiB/s** line-rate before overheads

## Hardware Quirk: 4 Interfaces, 2 Physical Ports

GB10 PCIe limitation requires ConnectX-7 **multi-host mode**. You'll see:
- `enp1s0f0np0`, `enp1s0f1np1`
- `enP2p1s0f0np0`, `enP2p1s0f1np1`

Use `ibdev2netdev` to find which is actually **(Up)**.

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

### Option B: Static IPs (non-persistent)
```bash
# Find active interface first
ibdev2netdev

# Node A (spark-1)
sudo ip addr add 192.168.100.10/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up

# Node B (spark-2)
sudo ip addr add 192.168.100.11/24 dev enp1s0f1np1
sudo ip link set enp1s0f1np1 up
```

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

**NCCL IB (Jan 2026): FAILS on Spark RoCE.** `NCCL_NET=IB` + `NCCL_IB_HCA` causes `ncclCommInitRank` error. Use socket-based NCCL.

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

**Startup script:** `./start-vllm-multinode.sh` (deploys Ray + vLLM across spark-2/spark-3)

Docs: https://build.nvidia.com/spark/vllm/stacked-sparks

**CRITICAL env vars (both nodes):**
```bash
export MN_IF_NAME=enp1s0f1np1
export NCCL_SOCKET_IFNAME=$MN_IF_NAME
export GLOO_SOCKET_IFNAME=$MN_IF_NAME
export UCX_NET_DEVICES=$MN_IF_NAME
export RAY_memory_monitor_refresh_ms=0
```

**Single interface used:** Peak TP=2 bandwidth ~9 Gbit/s. NCCL socket mode only uses ONE interface ([#278](https://github.com/NVIDIA/nccl/issues/278)). Dual interface would reduce latency (full-duplex) but requires IB mode.

**NCCL IB mode (experimental):** `--ib` flag mounts `/dev/infiniband` and sets `NCCL_NET=IB`. Both HCAs detected but fails on AllReduce - GPUDirect RDMA not supported on Spark. Disabling GDR causes silent crash. **Status: not working.**

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
  --tensor-parallel-size 2 --trust-remote-code --enforce-eager \
  --quantization awq --gpu-memory-utilization 0.75 --kv-cache-dtype fp8 \
  --limit-mm-per-prompt.video 0 --host 0.0.0.0 --port 8000
```

**Key:** `--quantization awq` avoids marlin repack OOM, `--kv-cache-dtype fp8` halves KV mem

**Auto-config (Jan 2026):** vLLM auto-detects available memory and configures 256K context with ~547K token KV cache (34K blocks × 16). No `--max-model-len` needed.

**Memory:** 97GB/119GB used. Enough for ~2 concurrent 256K requests or many shorter ones.

**Benchmark (Jan 2026):** Peak decode ~288 tok/s at c=256 (long generation). Short prompts plateau at c=64. Script: `benchmark_vllm.py --sweep -t 256`

**Port:** 8000 (default). Use same port for vLLM/SGLang for consistent client code.

**Access from spark-1:** Use 10GbE IP `192.168.102.11:8000` (200G IPs only work between spark-2/spark-3)
```bash
curl http://192.168.102.11:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ","messages":[{"role":"user","content":"Hi"}],"max_tokens":64}'
```

**Deps:** `pip install qwen-vl-utils==0.0.14` (required for Qwen-VL)

### Head vs Worker Node Resource Usage (Jan 2026)

**spark-2 (head)** uses more CPU/RAM than **spark-3 (worker)**:

| Resource | spark-2 (head) | spark-3 (worker) |
|----------|----------------|------------------|
| CPU | ~1200% (EngineCore) + 134% (Ray) | ~246% (Ray worker) |
| RAM | 118GB used, <1GB free | 109GB used, 10GB free |

**Why head is heavier:** EngineCore (scheduling, KV mgmt), tokenization (CPU-only), vision preprocessing (image decode/resize before GPU), Ray GCS server.

**`--mm-encoder-tp-mode data`:** GPU data parallel for vision encoder. Disabled - causes hangs in multi-node encoder profiling.

**Reduce CPU load:** Resize images client-side before API calls (<100KB).

### Ray Compiled DAG CPU Overhead (Jan 2026)

**Root cause of ~1200% CPU:** 12× `worker.channel_` threads busy-polling Ray shared memory channels.

**Fix (single-node only):** Set `VLLM_USE_RAY_COMPILED_DAG=0` at container startup.

**Multi-node limitation:** vLLM v1 **forces compiled DAG on** for multi-node TP. Setting `VLLM_USE_RAY_COMPILED_DAG=0` gets overwritten to `1`. The CPU overhead is unavoidable for multi-node setups.

**Tradeoff:** Benchmarks show ~2.5x higher throughput with DAG disabled (single-node only).

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

## Notes

**OOM killers:** Inactive on spark-2/spark-3 (unlike spark-1). Only kernel OOM kicks in.

**SGLang vs vLLM (Jan 2026):** vLLM is more stable for multi-node AWQ. SGLang hangs after NCCL init.

**Cost vs OpenRouter:** ~$0.53/hour savings ($382/month). Break-even ~31 months at current load.
