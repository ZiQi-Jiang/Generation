

#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"


for epoch in 40 80 120 160 200; 
do

for exp in \
    jit/20261116_040804 \
    jit_repa/20261116_233214 \
    jit_sra/20261116_224116 \
    jit_dino/20261116_232659
do
    echo "Evaluating: $exp, epoch $epoch"
    accelerate launch \
        --multi_gpu \
        --num_processes 8 \
        --mixed_precision bf16 \
        stage_01/sample_jit.py \
        --exp "$exp" \
        --epoch "$epoch" \
        --num_samples 10000 \
        --sample_steps 50 \
        --batch_size 64 \
        --sample_seed 0 \
        --cfg_scale 1.0 \
        --t_eps 0.05
    done
done

