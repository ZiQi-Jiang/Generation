#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm

  
accelerate launch  --mixed_precision=bf16 --num_processes 8\
  train_jit.py \
  --dit_type 'JiT-S/16_no_incontext' \
  --epochs 80 \
  --ema_decay 0.9999 \
  --batch_size 128 \
  --data_path ./data/imagenet/images/train

# accelerate launch  --mixed_precision=bf16 \
#   train_jit.py \
#   --dit_type 'JiT-S/16_no_incontext' \
#   --epochs 40 \
#   --ema_decay 0.9999 \
#   --batch_size 128 \
#   --freq_lam 1.0\

# accelerate launch  --mixed_precision=bf16 \
#   train_jit.py \
#   --dit_type 'JiT-S/16_no_incontext' \
#   --epochs 40 \
#   --ema_decay 0.9999 \
#   --batch_size 128 \
#   --freq_lam 10.0 \


# accelerate launch  --mixed_precision=bf16 \
#   train_jit.py \
#   --dit_type 'JiT-S/16_no_incontext' \
#   --epochs 40 \
#   --ema_decay 0.9999 \
#   --batch_size 128 \
#   --freq_lam 100.0 \

# accelerate launch  --mixed_precision=bf16 \
#   train_jit.py \
#   --dit_type 'JiT-S/16_no_incontext' \
#   --epochs 40 \
#   --ema_decay 0.9999 \
#   --batch_size 128 \
#   --freq_lam 1000.0 \
