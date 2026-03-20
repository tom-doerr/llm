#!/bin/bash
# Write GPU metrics for node_exporter textfile collector
DIR="$HOME/.local/share/node_exporter/textfile"
mkdir -p "$DIR"
gpu_extra() {
  nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.sm \
    --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{
    print "node_gpu_utilization_percent "$1; print "node_gpu_power_watts "$2
    print "node_gpu_temperature_celsius "$3; print "node_gpu_clock_mhz "$4}'
  nvidia-smi -q 2>/dev/null | awk \
    '/SW Power Capping *:/{print "node_gpu_throttle_sw_power_us "$(NF-1)}
     /SW Thermal.*us/{print "node_gpu_throttle_sw_thermal_us "$(NF-1)}
     /HW Thermal.*us/{print "node_gpu_throttle_hw_thermal_us "$(NF-1)}'
  # P-state (0=active, 8=idle)
  nvidia-smi --query-gpu=pstate --format=csv,noheader,nounits 2>/dev/null | \
    awk '{gsub(/P/,""); print "node_gpu_pstate "$1}'
  # Zram physical RAM usage (hidden memory consumer on UMA)
  awk '/nr_zspages/{printf "node_zram_backing_bytes %.0f\n",$2*4096}' /proc/vmstat
  # RDMA error counters
  for d in /sys/class/infiniband/*/ports/1/counters; do
    dev=$(echo "$d" | awk -F/ '{print $5}')
    for f in port_rcv_errors port_xmit_discards; do
      v=$(cat "$d/$f" 2>/dev/null)
      [ -n "$v" ] && echo "node_rdma_${f}{device=\"$dev\"} $v"
    done
  done
  for z in /sys/class/thermal/thermal_zone*/; do
    i=${z##*zone};i=${i%/};t=$(cat "${z}temp" 2>/dev/null)
    [ -n "$t" ] && echo "node_soc_thermal_celsius{zone=\"$i\"} $((t/1000))"
  done
}
while true; do
    gpu_extra > "$DIR/gpu.prom.$$"
    mv "$DIR/gpu.prom.$$" "$DIR/gpu.prom"
    sleep 5
done
