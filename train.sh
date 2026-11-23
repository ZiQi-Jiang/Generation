############ This stage: Explo


#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sslfm
conda activate sslfm
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

timestamp=$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)
epochs=200

accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
  stage_01/train_pit.py \
  --dit_type 'PiT-B/16' \
  --dino_image_size 256 \
  --timestamp pixeldit/cond/"$timestamp" \
  --epochs "$epochs" \
  --data_backend dali \
  --drop_class 0.1 \
  --data_path ./data/imagenet/images/train

accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
  stage_01/train_pit.py \
  --dit_type 'PiT-B/16' \
  --dino_image_size 256 \
  --timestamp pixeldit/uncond/"$timestamp" \
  --epochs "$epochs" \
  --data_backend dali \
  --drop_class 1.0 \
  --data_path ./data/imagenet/images/train

accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
  stage_01/train_ddt.py \
  --dit_type 'DDT-B/16' \
  --dino_image_size 256 \
  --timestamp ddt/cond/"$timestamp" \
  --epochs "$epochs" \
  --data_backend dali \
  --drop_class 0.1 \
  --data_path ./data/imagenet/images/train

accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
  stage_01/train_ddt.py \
  --dit_type 'DDT-B/16' \
  --dino_image_size 256 \
  --timestamp ddt/uncond/"$timestamp" \
  --epochs "$epochs" \
  --data_backend dali \
  --drop_class 1.0 \
  --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v2.py \
#   --dit_type 'JiT-B/16' \
#   --dino_image_size 256 \
#   --timestamp simple_jit_dino_v2/cond/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train

# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v2.py \
#   --dit_type 'JiT-B/16' \
#   --dino_image_size 256 \
#   --timestamp simple_jit_dino_v2/cond_res/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train \
#   --dino_residual

# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v2.py \
#   --dit_type 'JiT-B/16' \
#   --dino_image_size 256 \
#   --timestamp simple_jit_dino_v2/uncond/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 1.0 \
#   --data_path ./data/imagenet/images/train

# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v2.py \
#   --dit_type 'JiT-B/16' \
#   --dino_image_size 256 \
#   --timestamp simple_jit_dino_v2/uncond_res/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 1.0 \
#   --data_path ./data/imagenet/images/train \
#   --dino_residual

# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v1.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp simple_jit_dino_v1/cond/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_02/simple_jit_dino_v1.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp simple_jit_dino_v1/uncond/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 1.0 \
#   --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_01/train_jit_dino.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp jit_dino/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_01/train_jit_repa.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp jit_repa/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#   stage_01/train_jit_sra.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp jit_sra/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train


# accelerate launch  --mixed_precision=bf16 --multi_gpu --num_processes 8\
#    stage_01/train_jit_selfflow.py \
#   --dit_type 'JiT-B/16' \
#   --timestamp jit_selfflow/"$timestamp" \
#   --epochs "$epochs" \
#   --data_backend dali \
#   --drop_class 0.1 \
#   --data_path ./data/imagenet/images/train
