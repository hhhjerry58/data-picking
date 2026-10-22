#!/usr/bin/env bash
# Source before using Python so all runtime caches land on the artifact mount.
export HF_HOME=/mnt/tianzhu_hot/data-picking/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_DISABLE_XET=1
export HF_HUB_DISABLE_TELEMETRY=1
export TORCH_HOME=/mnt/tianzhu_hot/data-picking/cache/torch
export TORCHINDUCTOR_CACHE_DIR=/mnt/tianzhu_hot/data-picking/cache/torchinductor
export TRITON_CACHE_DIR=/mnt/tianzhu_hot/data-picking/cache/triton
export CUDA_CACHE_PATH=/mnt/tianzhu_hot/data-picking/cache/cuda
export MPLCONFIGDIR=/mnt/tianzhu_hot/data-picking/cache/matplotlib
export XDG_CACHE_HOME=/mnt/tianzhu_hot/data-picking/cache/xdg
# blobfuse2 cannot unlink open temporary files. POSIX scratch is ephemeral RAM;
# every retained dataset, cache, checkpoint, log and result stays on the mount.
export TMPDIR=/dev/shm/data-picking-runtime
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$XDG_CACHE_HOME" "$TMPDIR"
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
