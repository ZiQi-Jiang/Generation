#!/usr/bin/env bash
set -eo pipefail

cd /data/Generation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

num_gpus="${NGPUS:-8}"
data_path="$PWD/data/imagenet/images"


# JiT：t=1.0 是干净图像；layer_idx 从 0 开始，11 是最后一个 block。
# for t in 1.0 0.9 0.8 0.7 0.6 0.5; do
#     for layer_idx in 0 3 7 11; do
#         accelerate launch \
#             --multi_gpu \
#             --num_processes "$num_gpus" \
#             --mixed_precision no \
#             linear_probing/knn_dit.py \
#             --exp jit/20261112_024237 \
#             --epoch 80 \
#             --data_path "$data_path" \
#             --t "$t" \
#             --layer_idx "$layer_idx"
#     done
# done


# DINO：已有结果；如果只想补 JiT，可以跳过这一条。
accelerate launch \
    --multi_gpu \
    --num_processes "$num_gpus" \
    --mixed_precision bf16 \
    linear_probing/knn_dino.py \
    --exp dino/20261112_024302 \
    --epoch 40 \
    --data_path "$data_path"