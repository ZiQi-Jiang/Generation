#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

timestamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
epochs=800

accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    train_dino.py \
    --timestamp "$timestamp" \
    --dino_type vit_s_16 \
    --epochs "$epochs" \
    --warmup_epochs 10 \
    --image_size 224 \
    --blr 0.0005 \
    --ema_decay 0.996 \
    --batch_size 128 \
    --local_views 10 \
    --global_views 2 \
    --data_backend dali \
    --data_path ./data/imagenet/images/train



# Evaluate DINO
accelerate launch \
    --multi_gpu \
    --num_processes 8 \
    --mixed_precision bf16 \
    linear_probing/knn_dino.py \
    --exp "$timestamp" \
    --epoch "$epochs" \
    --data_path ./data/imagenet/images


# Official
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no \
#   linear_probing/knn_dino.py \
#   --dino_type vit_s_16