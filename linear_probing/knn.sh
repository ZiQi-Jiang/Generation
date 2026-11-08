#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate negfm


export PYTHONPATH=/data/Generation:${PYTHONPATH:-}


for model_type in SiT SRA SelfFlow; do
    for t in 0.9 0.8 0.7 0.6; do
        for layer_idx in 3 7 11 15 17 23 27; do
            accelerate launch \
                --multi_gpu \
                --num_processes 8 \
                --mixed_precision no \
                knn.py \
                --model_type "$model_type" \
                --t "$t" \
                --layer_idx "$layer_idx"
        done
    done
done

for t in 0.6 0.5; do
    for layer_idx in 3 7 11 15 17 23 27 31; do
        accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --model_type JiT-H/16 --t "$t" --layer_idx "$layer_idx"
    done
done
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --model_type JiT-H/16 --t 1.0 --layer_idx 3 7 11 15 17 23 27 31
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --model_type SiT --t 1.0 --layer_idx 3 
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --model_type SRA --t 1.0 --layer_idx 3 
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --model_type SelfFlow --t 1.0 --layer_idx 3 

