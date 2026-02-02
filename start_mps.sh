#!/bin/bash
if [ -z "$1" ]; then
    NUM=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    NUM=$1
fi

NUM=$((NUM-1))
GPU_ARRAY=($(seq 0 $NUM))
GPU_STR=""
for id in "${GPU_ARRAY[@]}"; do
    GPU_STR="${GPU_STR}${id}"
    if [ $id -ne $NUM ]; then
        GPU_STR="${GPU_STR},"
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_STR

for id in "${GPU_ARRAY[@]}"; do
    nvidia-smi -i $id -c EXCLUSIVE_PROCESS
done

TARGET_UID=${MPS_UID:-${SUDO_UID:-$(id -u)}}
echo "Configuring MPS for UID: $TARGET_UID"
echo "Pipe Dir: $CUDA_MPS_PIPE_DIRECTORY"

nvidia-cuda-mps-control -d

echo "Waiting for pipe directory..."
for i in {1..50}; do
    if [ -d "$CUDA_MPS_PIPE_DIRECTORY" ]; then
        echo "Directory found. Changing ownership to $TARGET_UID..."
        # 修改管道目录权限，确保 Server 进程 (用户身份) 能在里面创建文件
        chown -R $TARGET_UID "$CUDA_MPS_PIPE_DIRECTORY"
        # 同时修改日志目录权限
        if [ -d "$CUDA_MPS_LOG_DIRECTORY" ]; then
            chown -R $TARGET_UID "$CUDA_MPS_LOG_DIRECTORY"
        fi
        break
    fi
    sleep 0.1
done

echo "start_server -uid $TARGET_UID" | nvidia-cuda-mps-control 

echo "MPS initiation complete"