#!/usr/bin/env bash
# Start vLLM multi-node on two DGX Sparks using dual-rail RoCE (RDMA).
# Autodetects port group, requires IPv4 on BOTH rails for RoCE GIDs.
# Usage: ./start-vllm-multinode-dualrail.sh [--debug] [--port f0|f1] [--dry-run]
set -Eeuo pipefail

HEAD_HOST=${HEAD_HOST:-spark-2}
WORKER_HOST=${WORKER_HOST:-spark-3}
MODEL=${MODEL:-QuantTrio/Qwen3-VL-235B-A22B-Instruct-AWQ}
CONTAINER=${CONTAINER:-nvcr.io/nvidia/vllm:25.11-py3}
RAY_PORT=${RAY_PORT:-6379}
VLLM_PORT=${VLLM_PORT:-8000}
HF_CACHE_DIR=${HF_CACHE_DIR:-/home/tom/.cache/huggingface}
SITE_CUSTOMIZE=${SITE_CUSTOMIZE:-/home/tom/llm/sitecustomize.py}
DEBUG_NCCL=${DEBUG_NCCL:-0}
DRY_RUN=0
FORCE_PORT=${FORCE_PORT:-""}
HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# Arg parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --debug) DEBUG_NCCL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --port) FORCE_PORT="${2:-}"; shift 2 ;;
    -h|--help) echo "Usage: $0 [--debug] [--port f0|f1] [--dry-run]"; exit 0 ;;
    *) echo "Unknown: $1"; exit 2 ;;
  esac
done

log() { echo "[$(date +'%F %T')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
ssh_run() {
  local host="$1"; shift
  [[ "$DRY_RUN" == "1" ]] && { echo "+ ssh $host \"$*\""; return 0; }
  ssh -o BatchMode=yes "$host" "$*"
}
# Always-run SSH for validation (even in dry-run)
ssh_check() { ssh -o BatchMode=yes "$1" "$2"; }
get_ipv4() { ssh_check "$1" "ip -4 -o addr show dev $2 2>/dev/null | awk '{print \$4}' | cut -d/ -f1 | head -n1"; }
operstate() { ssh_check "$1" "cat /sys/class/net/$2/operstate 2>/dev/null || echo unknown"; }
ping_check() { ssh_check "$HEAD_HOST" "ping -I $1 -c 1 -W 1 $2 >/dev/null 2>&1"; }

# Port group definitions (f0 and f1)
P0_IF_A="enp1s0f0np0"; P0_IF_B="enP2p1s0f0np0"
P0_RDMA_A="rocep1s0f0"; P0_RDMA_B="roceP2p1s0f0"
P1_IF_A="enp1s0f1np1"; P1_IF_B="enP2p1s0f1np1"
P1_RDMA_A="rocep1s0f1"; P1_RDMA_B="roceP2p1s0f1"

select_group() {
  case "$1" in
    f0) OOB_IF=$P0_IF_A; RAIL_A=$P0_IF_A; RAIL_B=$P0_IF_B; RDMA_A=$P0_RDMA_A; RDMA_B=$P0_RDMA_B ;;
    f1) OOB_IF=$P1_IF_A; RAIL_A=$P1_IF_A; RAIL_B=$P1_IF_B; RDMA_A=$P1_RDMA_A; RDMA_B=$P1_RDMA_B ;;
    *) die "Invalid port group '$1'" ;;
  esac
}

pick_best_group() {
  local candidates=(f0 f1)
  [[ -n "$FORCE_PORT" ]] && candidates=("$FORCE_PORT")
  for p in "${candidates[@]}"; do
    select_group "$p"
    local h_st_a h_st_b w_st_a w_st_b
    h_st_a=$(operstate "$HEAD_HOST" "$RAIL_A" | tr -d '\r')
    h_st_b=$(operstate "$HEAD_HOST" "$RAIL_B" | tr -d '\r')
    w_st_a=$(operstate "$WORKER_HOST" "$RAIL_A" | tr -d '\r')
    w_st_b=$(operstate "$WORKER_HOST" "$RAIL_B" | tr -d '\r')
    # Require both rails up on both nodes
    [[ "$h_st_a" != "up" || "$h_st_b" != "up" || "$w_st_a" != "up" || "$w_st_b" != "up" ]] && {
      log "Skip $p: not all rails up"; continue; }
    local h_ip_a h_ip_b w_ip_a w_ip_b
    h_ip_a=$(get_ipv4 "$HEAD_HOST" "$RAIL_A" | tr -d '\r')
    h_ip_b=$(get_ipv4 "$HEAD_HOST" "$RAIL_B" | tr -d '\r')
    w_ip_a=$(get_ipv4 "$WORKER_HOST" "$RAIL_A" | tr -d '\r')
    w_ip_b=$(get_ipv4 "$WORKER_HOST" "$RAIL_B" | tr -d '\r')
    [[ -z "$h_ip_a" || -z "$h_ip_b" || -z "$w_ip_a" || -z "$w_ip_b" ]] && {
      log "Skip $p: missing IPv4 on one/both rails"; continue; }
    ping_check "$RAIL_A" "$w_ip_a" || { log "Skip $p: ping A failed"; continue; }
    ping_check "$RAIL_B" "$w_ip_b" || { log "Skip $p: ping B failed"; continue; }
    # Found valid group
    PORT_GROUP="$p"; HEAD_IP="$h_ip_a"; WORKER_IP="$w_ip_a"
    HEAD_IP_B="$h_ip_b"; WORKER_IP_B="$w_ip_b"
    return 0
  done
  die "No dual-rail link found. Ensure QSFP cable + IPv4 on both rails."
}

# Main
log "Autodetecting dual-rail link..."
pick_best_group
log "Selected: $PORT_GROUP, Control=$OOB_IF, Head=$HEAD_IP, Worker=$WORKER_IP"
log "Rail B: Head=$HEAD_IP_B, Worker=$WORKER_IP_B"
NCCL_HCA="=${RDMA_A}:1,${RDMA_B}:1"
log "NCCL_IB_HCA=$NCCL_HCA"

ENV="-e NCCL_NET_PLUGIN=none -e NCCL_DMABUF_ENABLE=0"
ENV="$ENV -e NCCL_NET_GDR_LEVEL=LOC -e NCCL_NET_GDR_C2C=0"
ENV="$ENV -e NCCL_IB_HCA='$NCCL_HCA'"
ENV="$ENV -e NCCL_SOCKET_IFNAME=$OOB_IF -e GLOO_SOCKET_IFNAME=$OOB_IF"
ENV="$ENV -e RAY_memory_monitor_refresh_ms=0 -e HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
[[ "$DEBUG_NCCL" == "1" ]] && {
  ENV="$ENV -e NCCL_DEBUG=INFO -e NCCL_DEBUG_SUBSYS=INIT,NET"
  ENV="$ENV -e NCCL_DEBUG_FILE=/tmp/nccl.%h.%p.log"
  log "NCCL debug enabled"
}
ENV_HEAD="$ENV -e VLLM_HOST_IP=$HEAD_IP"
ENV_WORKER="$ENV -e VLLM_HOST_IP=$WORKER_IP"

VOLS="-v $HF_CACHE_DIR:/root/.cache/huggingface"
[[ -f "$SITE_CUSTOMIZE" ]] && VOLS="$VOLS -v $SITE_CUSTOMIZE:/usr/lib/python3.12/sitecustomize.py:ro"
RDMA="--device=/dev/infiniband --ulimit memlock=-1 --cap-add=IPC_LOCK"

log "=== Cleanup ==="
ssh_run "$HEAD_HOST" "docker rm -f vllm-head 2>/dev/null || true"
ssh_run "$WORKER_HOST" "docker rm -f vllm-worker 2>/dev/null || true"

log "=== Starting Ray head on $HEAD_HOST ==="
ssh_run "$HEAD_HOST" "docker run -d --name vllm-head --gpus all --shm-size 16g \\
  --network host --ipc host $RDMA $ENV_HEAD $VOLS $CONTAINER \\
  ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_IP --block"
log "Waiting for Ray head..."
[[ "$DRY_RUN" != "1" ]] && sleep 10

log "=== Starting Ray worker on $WORKER_HOST ==="
ssh_run "$WORKER_HOST" "docker run -d --name vllm-worker --gpus all --shm-size 16g \\
  --network host --ipc host $RDMA $ENV_WORKER $VOLS $CONTAINER \\
  ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP --block"
log "Waiting for worker to join..."
[[ "$DRY_RUN" != "1" ]] && sleep 5

log "=== Starting vLLM server ==="
VLLM_ARGS="--tensor-parallel-size 2 --trust-remote-code --enforce-eager"
VLLM_ARGS="$VLLM_ARGS --quantization awq --gpu-memory-utilization 0.75"
VLLM_ARGS="$VLLM_ARGS --kv-cache-dtype fp8 --limit-mm-per-prompt.video 0"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0 --port $VLLM_PORT"

ssh_run "$HEAD_HOST" "docker exec -d -e RAY_ADDRESS=$HEAD_IP:$RAY_PORT \\
  -e VLLM_ATTENTION_BACKEND=TRITON_ATTN vllm-head vllm serve $MODEL $VLLM_ARGS"

log "=== Done ==="
log "API: http://$HEAD_IP:$VLLM_PORT/v1/chat/completions"
log "Logs: ssh $HEAD_HOST 'docker logs -f vllm-head'"
