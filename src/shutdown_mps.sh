#!/bin/bash
if [ -z "$1" ]
then
    echo "Number of GPUs not given! The sript will use nvidia-smi to detect number of GPUs "
    NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM=$1
fi

NUM=$((NUM-1))

pkill proxy # for killing proxy servers remaining in the system
for i in `seq 0 $NUM`
do
sudo nvidia-smi -i $i -c DEFAULT
done

try_stop_mps() {
    local pipe_dir="$1"
    if [ -n "$pipe_dir" ]; then
        CUDA_MPS_PIPE_DIRECTORY="$pipe_dir" echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
    else
        echo quit | nvidia-cuda-mps-control >/dev/null 2>&1
    fi
}

# Try common MPS pipe directories in order.
TARGET_UID=${MPS_UID:-${SUDO_UID:-$(id -u)}}
try_stop_mps "${CUDA_MPS_PIPE_DIRECTORY}"
try_stop_mps "/tmp/nvidia-mps-${TARGET_UID}"
try_stop_mps "/tmp/nvidia-mps"

sleep 0.5

# Fallback: terminate lingering MPS server if it is still running.
if pgrep -x nvidia-cuda-mps-server >/dev/null 2>&1; then
    pkill -x nvidia-cuda-mps-server >/dev/null 2>&1
fi

echo "MPS shutdown complete"
