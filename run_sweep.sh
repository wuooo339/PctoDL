#!/bin/bash
# MPS Frequency Sweep Script
MODELS=("mnasnet" "vgg19" "densenet201" "efficientnet_v2_m" "maxvit_t" "mobilenet_v2" "resnet50" "static")
# MODELS=("static")

MEM_FREQS=(1215)
declare -a CONFIGS
CONFIGS+=("25 25 25 25" "8 8 8 8")
CONFIGS+=("25 25 25 25" "16 16 16 16")
CONFIGS+=("25 25 25 25" "32 32 32 32")
CONFIGS+=("50 50" "8 8")
CONFIGS+=("50 50" "16 16")
CONFIGS+=("50 50" "32 32")
CONFIGS+=("10 10 10 10 10 10 10 10 10 10" "8 8 8 8 8 8 8 8 8 8")
CONFIGS+=("10 10 10 10 10 10 10 10 10 10" "16 16 16 16 16 16 16 16 16 16")
CONFIGS+=("10 10 10 10 10 10 10 10 10 10" "32 32 32 32 32 32 32 32 32 32")
DATASET="ImageNet"
F_START=210
F_END=1410
F_STEP=150

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "MPS Frequency Sweep Test"
echo "=========================================="
echo "Models: ${MODELS[@]}"
echo "Memory Freqs: ${MEM_FREQS[@]}"
echo "SM Freq Range: ${F_START}-${F_END} MHz (step=${F_STEP})"
echo "Number of configs: $((${#CONFIGS[@]} / 2))"
echo "=========================================="

config_count=$((${#CONFIGS[@]} / 2))

for ((config_idx=0; config_idx<config_count; config_idx++)); do
    P_LIST="${CONFIGS[$((config_idx * 2))]}"
    B_LIST="${CONFIGS[$((config_idx * 2 + 1))]}"

    echo ""
    echo "=========================================="
    echo "Config $((config_idx + 1)): P=[$P_LIST], B=[$B_LIST]"
    echo "=========================================="

    for model in "${MODELS[@]}"; do
        echo ""
        echo "--- Model: $model ---"

        for mem_freq in "${MEM_FREQS[@]}"; do
            echo "  mem_freq=$mem_freq MHz"

            python src/profile_mps.py \
                --mem-freq "$mem_freq" \
                --p $P_LIST \
                --b $B_LIST \
                --dataset "$DATASET" \
                --sweep \
                --f-start "$F_START" \
                --f-end "$F_END" \
                --f-step "$F_STEP" \
                --models "$model" \
                --measure-seconds 7 \
                --warmup-seconds 8

            echo "  Result saved to: ${model}_sweep_results.csv"

            # 清理残留进程
            ./src/cleanup_processes.sh
        done
    done
done

echo ""
echo "=========================================="
echo "All sweeps completed!"
echo "=========================================="

./src/cleanup_processes.sh
