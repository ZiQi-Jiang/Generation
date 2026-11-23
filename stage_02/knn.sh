#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

num_gpus=8
epochs=(40 80 120 160)
times=(1.0 0.9 0.8 0.7 0.6 0.5)
layers=(0 3 7 11)

    for epoch in "${epochs[@]}"; do
        for t in "${times[@]}"; do
            for layer_idx in "${layers[@]}"; do
                for exp in \
                    jit/20261116_040804 \
                    jit_dino/20261116_232659 \
                    jit_repa/20261116_233214 \
                    jit_sra/20261116_224116 
                do
                echo "kNN $exp epoch=$epoch t=$t layer_idx=$layer_idx"
                accelerate launch \
                    --multi_gpu \
                    --num_processes "$num_gpus" \
                    --mixed_precision no \
                    linear_probing/knn_dit.py \
                    --exp "$exp" \
                    --epoch "$epoch" \
                    --t "$t" \
                    --layer_idx "$layer_idx" \
                    --data_path "$PWD/data/imagenet/images"
            done
        done
    done
done