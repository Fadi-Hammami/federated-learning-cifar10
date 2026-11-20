#!/usr/bin/env bash
# Launch 1 server + 2 client processes on the same machine (Linux / macOS / Git Bash).
# Usage: bash cross_process/run_local.sh [partition] [rounds] [tag]
PARTITION=${1:-iid}; ROUNDS=${2:-20}; TAG=${3:-CP_$PARTITION}
WORLD=3
PY=$(command -v python || command -v python3)
cd "$(dirname "$0")/.."
mkdir -p results
$PY cross_process/server.py --world_size $WORLD --rounds $ROUNDS --partition $PARTITION --tag $TAG > results/log_${TAG}_server.txt 2>&1 &
sleep 2
$PY cross_process/client.py --world_size $WORLD --rank 1 --partition $PARTITION > results/log_${TAG}_client1.txt 2>&1 &
$PY cross_process/client.py --world_size $WORLD --rank 2 --partition $PARTITION > results/log_${TAG}_client2.txt 2>&1 &
wait
echo "done -> results/$TAG.json  (logs in results/log_${TAG}_*.txt)"
