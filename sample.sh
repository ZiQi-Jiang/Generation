# #!/usr/bin/env bash
# set -e

# cd "$(dirname "$0")"

# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate sslfm


# # freq_lam = 0
# accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20260930_231320 --epoch 40 --batch_size 128

# # freq_lam = 1
# accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_014426 --epoch 40 --batch_size 128

# # freq_lam = 10
# accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_041741 --epoch 40 --batch_size 128

# # freq_lam = 100
# accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_064928 --epoch 40 --batch_size 128

# # freq_lam = 1000
# accelerate launch --mixed_precision=bf16 sample_jit.py --exp 20261001_092218 --epoch 40 --batch_size 128


#!/usr/bin/env bash
set -e

cd /data/Generation

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sslfm

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for exp in \
    jit/20261111_021514 \
    jit/20261112_024237 \
    jit_dino/20261112_182345
do
    echo "Evaluating: $exp, epoch 80"

    accelerate launch \
        --multi_gpu \
        --num_processes 8 \
        --mixed_precision bf16 \
        sample_jit.py \
        --exp "$exp" \
        --epoch 80 \
        --num_samples 50000 \
        --sample_steps 50 \
        --batch_size 64 \
        --sample_seed 0 \
        --cfg_scale 1.0 \
        --t_eps 0.05
done