# Incorporate SSL into Flow Matching


## Environment

The environment uses Python 3.10, PyTorch 2.7.1+cu128,
torchvision 0.22.1+cu128, and NVIDIA DALI 2.4.0.


```bash
conda create -n sslfm python=3.10 -y
conda activate sslfm

pip install -r requirements.txt
```

## Data and checkpoints

Download 
```bash
bash download_data.sh
```

```
wget -O data/jit_in256_stats.npz https://raw.githubusercontent.com/LTH14/JiT/main/fid_stats/jit_in256_stats.npz
```

ImageNet must use the ImageFolder layout:

```text
#  train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......
#  val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......
```




Run training scripts from the repository root. For example:

```bash
conda activate sslfm

accelerate launch --multi_gpu --num_processes 8 \
    --mixed_precision bf16 train_jit.py \
    --dit_type JiT-B/16 \
    --data_path ./data/imagenet/images/train \
    --image_size 256 \
    --global_batch_size 1024 \
    --epochs 40
```

Training configuration is defined in `config.py`.
`--global_batch_size` is the total training batch size across all GPUs.

For cached SiT training, `--use_latent` selects the LMDB dataset.
Its location is configured through `latent_path` in `config.py`.

Results are organised as follows:
```text
exp/<timestamp>/
├── config.json
├── log.txt
├── epoch_080.pth
├── knn.jsonl
└── samples_epoch_080/
    ├── sample_log.txt
    └── images/
```

