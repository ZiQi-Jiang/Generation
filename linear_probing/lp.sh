#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate negfm


export PYTHONPATH=/data/Generation:${PYTHONPATH:-}

# for t in 0.9 0.8 0.7; do
#     accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train_classifier.py --model_type JiT-H/16 --t "$t" --layer_idx 3 7 11 15 17 23 27 31
# done



# accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train_classifier.py --model_type JiT-H/16 --t 1.0 --layer_idx 3 7 11 15 17 23 27 31
accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train_classifier.py --model_type SiT --t 1.0 --layer_idx 3 7 11 15 19 23 27
accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train_classifier.py --model_type SRA --t 1.0 --layer_idx 3 7 11 15 19 23 27
accelerate launch --multi_gpu --num_processes 8 --mixed_precision no train_classifier.py --model_type SelfFlow --t 1.0 --layer_idx 3 7 11 15 19 23 27

