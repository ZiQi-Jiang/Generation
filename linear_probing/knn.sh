#!/usr/bin/env bash
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sslfm
conda activate sslfm

cd "$(dirname "$0")"


export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH:-}"

for dit_type in SiT SRA SelfFlow; do
    for t in 0.9 0.8 0.7 0.6; do
        for layer_idx in 3 7 11 15 17 23 27; do
            accelerate launch \
                --multi_gpu \
                --num_processes 8 \
                --mixed_precision no \
                knn_dit.py \
                --dit_type "$dit_type" \
                --t "$t" \
                --layer_idx "$layer_idx"
        done
    done
done

for t in 0.6 0.5; do
    for layer_idx in 3 7 11 15 17 23 27 31; do
        accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn_dit.py --dit_type JiT-H/16 --t "$t" --layer_idx "$layer_idx"
    done
done
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --dit_type JiT-H/16 --t 1.0 --layer_idx 3 7 11 15 17 23 27 31
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --dit_type SiT --t 1.0 --layer_idx 3 
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --dit_type SRA --t 1.0 --layer_idx 3 
# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no knn.py --dit_type SelfFlow --t 1.0 --layer_idx 3 

