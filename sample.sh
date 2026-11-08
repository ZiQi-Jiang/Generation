#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm


# freq_lam = 0
accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20260930_231320 --epoch 40 --batch_size 128

# freq_lam = 1
accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_014426 --epoch 40 --batch_size 128

# freq_lam = 10
accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_041741 --epoch 40 --batch_size 128

# freq_lam = 100
accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_064928 --epoch 40 --batch_size 128

# freq_lam = 1000
accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_092218 --epoch 40 --batch_size 128
