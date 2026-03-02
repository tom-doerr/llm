#!/bin/bash
set -u
API="http://192.168.110.2:8000"
SCRIPT="/home/tom/llm/start-vllm-multinode.sh"
NEW="vllm/vllm-openai:cu130-nightly"
SWITCHED=0

source /home/tom/llm/vllm-monitor-lib.sh
echo "$(date): Monitor started (1hr interval)"

while true; do
    echo ""; echo "$(date): Health check..."
    if check_health; then
        echo "$(date): Server healthy"
    else
        echo "$(date): UNHEALTHY"
        if [[ $SWITCHED -eq 0 ]]; then
            switch_to_nightly
            SWITCHED=1
        else
            echo "$(date): Redeploying nightly..."
            bash "$SCRIPT"
            sleep 900
        fi
    fi
    echo "$(date): Next check in 3600s"
    sleep 3600
done
