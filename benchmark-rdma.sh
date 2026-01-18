#!/bin/bash
# Benchmark 200G RDMA connection between spark-2 and spark-3

SPARK2_IP1="192.168.100.10"
SPARK2_IP2="192.168.101.10"
DEV1="rocep1s0f1"
DEV2="roceP2p1s0f1"
PORT1=18515
PORT2=18516
DUR=10

BW="-R --report_gbits -q 4 -s 8388608 -D $DUR"

echo "=== RDMA Bandwidth Test: spark-2 ↔ spark-3 ==="
echo "Cleanup..."
ssh spark-2 "pkill -f ib_write_bw 2>/dev/null || true"
ssh spark-3 "pkill -f ib_write_bw 2>/dev/null || true"
sleep 1

echo ""
echo "=== Path 1: $DEV1 ($SPARK2_IP1) ==="
ssh spark-2 "ib_write_bw $BW -d $DEV1 -p $PORT1" &
sleep 2
ssh spark-3 "ib_write_bw $BW -d $DEV1 -p $PORT1 $SPARK2_IP1"
wait

echo ""
echo "=== Path 2: $DEV2 ($SPARK2_IP2) ==="
ssh spark-2 "ib_write_bw $BW -d $DEV2 -p $PORT2" &
sleep 2
ssh spark-3 "ib_write_bw $BW -d $DEV2 -p $PORT2 $SPARK2_IP2"
wait

echo ""
echo "=== Expected Results ==="
echo "Per path: ~106 Gb/s"
echo "Aggregate: ~212 Gb/s (run both paths manually for aggregate test)"
