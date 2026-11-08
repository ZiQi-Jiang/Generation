#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm


accelerate launch  --mixed_precision=bf16 --num_processes 8\
  train_dino.py \
  --dino_type 'vit_s_16' \
  --epochs 800 \
  --warmup_epochs 10 \
  --image_size 224 \
  --blr 0.0005 \
  --ema_decay 0.996 \
  --batch_size 128 \
  --local_views 10 \
  --global_views 2 \
  --data_backend dali \

