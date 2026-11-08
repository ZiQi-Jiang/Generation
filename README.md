# Incorporate SSL into Flow Matching


## Environment

The current environment uses Python 3.10, PyTorch 2.1.0,
torchvision 0.16.0, and CUDA 12.1 wheels.

Create the environment and install dependencies from the repository root:

```bash
conda create -n sslfm python=3.10 -y
conda activate sslfm

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Data and checkpoints

Download 
```bash
bash download_data.sh
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
    --batch_size 32 \
    --epochs 40
```

Training configuration is defined in `config.py`.
`--batch_size` is the local batch size per process.
Training outputs are saved under `exp/<timestamp>/`.

For cached SiT training, `--use_latent` selects the LMDB dataset.
Its location is configured through `latent_path` in `config.py`.

## Linear probing

The current linear probing script uses absolute paths under `/data/Generation`.
Update the dataset path in `train_classifier.py` and the environment,
repository path, and GPU count in `lp.sh` when moving to another machine.

Run the local launcher from its directory:

```bash
conda activate negfm
cd /data/Generation/linear_probing
bash lp.sh
```

The same experiment can be launched directly:

```bash
export PYTHONPATH=/data/Generation:${PYTHONPATH:-}

accelerate launch --multi_gpu --num_processes 8 \
    --mixed_precision no train_classifier.py \
    --model_type SelfFlow \
    --t 1.0 \
    --layer_idx 3 7 11 15 19 23 27
```

Replace `SelfFlow` with `SiT` or `SRA` to evaluate the other latent models.

