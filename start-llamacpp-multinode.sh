#!/bin/bash
# Start llama.cpp multi-node: spark-2 (head) + spark-3 (RPC)
set -e
MODEL="${1:-/models/models--unsloth--Qwen3.5-397B-A17B-GGUF/snapshots/abcbf9b1686cbdec98d678f3db51f04abb7a7ca2/Q2_K/Qwen3.5-397B-A17B-Q2_K-00001-of-00004.gguf}"
MMPROJ="${MMPROJ:-}" CTX="${CTX_SIZE:-}" NP="${NP:-128}"
IMG="nvcr.io/nvidia/vllm:26.01-py3" WIP="192.168.100.11" RPC=50052
LCPP="/home/tom/llama.cpp/build/bin"
VOLS="-v /home/tom/.cache/huggingface/hub:/models:ro -v $LCPP:$LCPP:ro"
RUN="--gpus all --network host --ipc host --restart=on-failure:10"

echo "=== Cleanup ==="
ssh spark-2 "docker rm -f llamacpp-head 2>/dev/null||true"
ssh spark-3 "docker rm -f llamacpp-rpc 2>/dev/null||true"

echo "=== Starting RPC on spark-3 ==="
ssh spark-3 "docker run -d --name llamacpp-rpc $RUN $VOLS \
    -e LD_LIBRARY_PATH=$LCPP --entrypoint $LCPP/rpc-server $IMG -H $WIP -p $RPC"
sleep 3

echo "=== Starting head on spark-2 ==="
MP=""; [ -n "$MMPROJ" ] && MP="--mmproj $MMPROJ"
CTX_ARG=""; [ -n "$CTX" ] && CTX_ARG="-c $CTX"
ssh spark-2 "docker run -d --name llamacpp-head $RUN $VOLS \
    -e LD_LIBRARY_PATH=$LCPP --entrypoint $LCPP/llama-server \
    $IMG -m $MODEL $MP --rpc $WIP:$RPC \
    -ngl 999 $CTX_ARG -np $NP --metrics --host 0.0.0.0 --port 8000"

echo "=== Started ==="
echo "API: http://192.168.102.11:8000/v1/chat/completions"
