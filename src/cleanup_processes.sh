#!/bin/bash

# GPU inference process cleanup script
# Used to clean up tritonserver and related processes
# System configuration constants
SUDO_PASSWORD="${SUDO_PASSWORD:-wzk123456}"  # Can be overridden by environment variable

echo "========================================="
echo "  GPU Inference Process Cleanup Tool"
echo "========================================="

# Show current GPU processes
echo "Current GPU processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "No GPU processes or nvidia-smi unavailable"

echo ""
echo "Starting process cleanup..."

# 1. Force kill all tritonserver processes
echo "1. Cleaning tritonserver processes..."
triton_pids=$(pgrep -f tritonserver)
if [ -n "$triton_pids" ]; then
    echo "  Found tritonserver processes: $triton_pids"
    pkill -9 -f tritonserver
    sleep 2
    echo "  tritonserver processes cleaned"
else
    echo "  No tritonserver processes found"
fi

# 2. Clean nvidia-cuda-mps-server processes
echo "2. Cleaning nvidia-cuda-mps-server processes..."
mps_pids=$(pgrep -f nvidia-cuda-mps-server)
if [ -n "$mps_pids" ]; then
    echo "  Found MPS processes: $mps_pids"
    pkill -9 -f nvidia-cuda-mps-server
    sleep 1
    echo "  MPS processes cleaned"
else
    echo "  No MPS processes found"
fi

# 3. Reset GPU state
echo "3. Resetting GPU state..."
echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -rmc 2>/dev/null && echo "  GPU compute mode reset" || echo "  GPU compute mode reset failed"
echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -rgc 2>/dev/null && echo "  GPU clocks reset" || echo "  GPU clocks reset failed"

# 4. Restore GPU power limits to default
echo "4. Restoring GPU power limits..."
# Detect GPU count
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Detected $NUM_GPUS GPU(s)"

for i in $(seq 0 $((NUM_GPUS-1))); do
    # Get GPU default power limit (usually max power)
    DEFAULT_POWER=$(nvidia-smi -i $i --query-gpu=power.max_limit --format=csv,noheader,nounits | tr -d ' ')
    CURRENT_POWER=$(nvidia-smi -i $i --query-gpu=power.limit --format=csv,noheader,nounits | tr -d ' ')

    echo "  GPU $i: current power limit ${CURRENT_POWER}W, default power limit ${DEFAULT_POWER}W"

    if [ "$CURRENT_POWER" != "$DEFAULT_POWER" ]; then
        echo "  Restoring GPU $i power limit to ${DEFAULT_POWER}W..."
        echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -pl $DEFAULT_POWER 2>/dev/null && \
            echo "    GPU $i power limit restored" || \
            echo "    GPU $i power limit restore failed"
    else
        echo "    GPU $i power limit is already default"
    fi

    # Force reset GPU to lowest performance state
    echo "  Resetting GPU $i performance state..."

    # Method 0: enable persistence mode (important)
    echo "    Enabling persistence mode..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -pm 1 2>/dev/null && echo "    Persistence mode enabled" || echo "    Persistence mode enable failed"

    # Method 1: reset application clocks
    echo "    Resetting application clocks..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -rac 2>/dev/null && echo "    Application clocks reset" || echo "    Application clocks reset failed"

    # Method 2: set lowest clock frequency
    echo "    Setting lowest clock frequency..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -lgc 210,210 2>/dev/null && echo "    GPU clocks set to minimum" || echo "    GPU clock set failed"
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -lmc 405,405 2>/dev/null && echo "    Memory clocks set to minimum" || echo "    Memory clock set failed"

    # Method 3: disable auto boost
    echo "    Disabling auto boost..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -acp 0 2>/dev/null && echo "    Auto boost disabled" || echo "    Auto boost disable failed"

    # Method 4: force enter low-power state (P12)
    echo "    Forcing low-power state..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -pm 1 2>/dev/null  # Enable persistence mode
    sleep 1
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -pm 0 2>/dev/null  # Disable persistence mode to force state switch

    # Method 5: reset all settings
    echo "    Resetting all GPU settings..."
    echo "$SUDO_PASSWORD" | sudo -S nvidia-smi -i $i -r 2>/dev/null && echo "    GPU reset" || echo "    GPU reset failed"
done

# 5. Run shutdown_mps.sh script
echo "5. Running shutdown_mps.sh script..."
if [ -f "./shutdown_mps.sh" ]; then
    echo "$SUDO_PASSWORD" | sudo -S ./shutdown_mps.sh 2>/dev/null && \
        echo "  shutdown_mps.sh completed" || \
        echo "  shutdown_mps.sh failed"
else
    echo "  shutdown_mps.sh file not found"
fi

# 6. Wait for GPU state to stabilize and check results
echo ""
echo "6. Waiting for GPU state to stabilize..."
echo "  Waiting 2 seconds for GPU state to stabilize..."
sleep 2

echo ""
echo "Cleanup complete, current GPU state:"
nvidia-smi --query-gpu=index,name,power.limit,power.draw,clocks.gr,clocks.mem,pstate --format=csv,noheader
echo ""
echo "Current GPU processes:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || echo "No GPU processes"

echo ""
echo "Detailed power info:"
CURRENT_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | tr -d ' ')
PSTATE=$(nvidia-smi --query-gpu=pstate --format=csv,noheader | tr -d ' ')
echo "  Current power draw: ${CURRENT_POWER}W"
echo "  Performance state: ${PSTATE}"

if (( $(echo "$CURRENT_POWER > 50" | bc -l) )); then
    echo "  WARNING: GPU power draw is still high (${CURRENT_POWER}W)"
    echo "  Suggestion: you may need to reboot or wait longer"
else
    echo "  GPU power draw is down to normal idle level"
fi

echo ""
echo "========================================="
echo "  Cleanup completed"
echo "========================================="
