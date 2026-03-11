#!/bin/bash
# Write GPU metrics for node_exporter textfile collector
DIR="$HOME/.local/share/node_exporter/textfile"
mkdir -p "$DIR"
while true; do
    nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.sm \
        --format=csv,noheader,nounits 2>/dev/null | \
    awk -F', ' '{
        print "node_gpu_utilization_percent " $1
        print "node_gpu_power_watts " $2
        print "node_gpu_temperature_celsius " $3
        print "node_gpu_clock_mhz " $4
    }' > "$DIR/gpu.prom.$$"
    mv "$DIR/gpu.prom.$$" "$DIR/gpu.prom"
    sleep 5
done
