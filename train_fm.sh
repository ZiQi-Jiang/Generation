#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sslfm
conda activate sslfm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

timestamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
epochs=80
accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
  train_jit.py \
  --dit_type 'JiT-B/16_no_incontext' \
  --timestamp jit/"$timestamp" \
  --epochs "$epochs" \
  --ema_decay 0.9999 \
  --global_batch_size 1024 \
  --data_backend dali \
  --drop_class 1.0 \
  --data_path ./data/imagenet/images/train


for epoch in 40 80; do 
  # FID evaluation
  accelerate launch \
      --multi_gpu \
      --num_processes 8 \
      --mixed_precision bf16 \
      sample_jit.py \
      --exp jit/"$timestamp" \
      --epoch "$epoch" \
      --num_samples 50000 \
      --sample_steps 50 \
      --batch_size 64
  for t in 1.0 0.9 0.8 0.7 0.6 0.5; do
    for layer_idx in 0 3 7 11; do
      # kNN evaluation
      accelerate launch \
          --multi_gpu \
          --num_processes 8 \
          --mixed_precision no \
          linear_probing/knn_dit.py \
          --exp jit/"$timestamp" \
          --epoch "$epoch" \
          --data_path ./data/imagenet/images \
          --t "$t" \
          --layer_idx "$layer_idx"
    done
  done
done

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
