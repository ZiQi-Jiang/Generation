import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from time import time
from models.jit_clean import JiT_models
from models.pit import PiT_models
from models.sit import SiT_models
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor, Normalize,RandomResizedCrop
import torch.utils.data as tud
from torchvision.datasets import ImageFolder
from models.utils import center_crop_arr
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from config import DefaultConfig
from config import DefaultConfig, SimpleLogger
from models.dino_vit import ViT
from torchvision.transforms import Resize, CenterCrop, InterpolationMode
import sys
class StopForwardException(Exception):
    pass


@torch.no_grad()
def evaluate(all_features, all_labels, val_loader, accelerator, model,  cfg, ):
    model.eval()
    total_samples = 0
    total_correct = 0
    total_correct5 = 0
    # for x, target in val_loader:
    for x, target in tqdm(val_loader, desc='Evaludate kNN', disable = not accelerator.is_main_process):
        x1 = x.to(torch.float32).div_(255)
        x1 = Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tokens = model.forward_feature(x1)
            features = model.norm(tokens[:,0])
            features = F.normalize(features.float(), p=2, dim=1)


        similarities, indices = (features @ all_features.T).topk(cfg.k, dim=1)
        neighbor_labels = all_labels[indices] #[B, k]
        weights = (similarities / cfg.temperature).exp() # [B,k]
        scores = torch.zeros(x1.shape[0], cfg.num_classes, device=accelerator.device)
        # scores = torch.scatter(1, neighbor_labels, weights) #[B, k]
        scores.scatter_add_(1, neighbor_labels, weights)
        predictions = scores.topk(5, dim=1).indices
        correct = predictions[:, 0].eq(target)
        correct5 = predictions.eq(target[:, None]).any(dim=1)

        correct, correct5 = accelerator.gather_for_metrics(( correct, correct5))
        total_correct += correct.sum().item()
        total_correct5 += correct5.sum().item()
        total_samples += correct.shape[0]

    acc = total_correct / total_samples
    acc5 = total_correct5 / total_samples
    accelerator.print(
        f'Epoch={cfg.epoch} '
        f'Top1 Acc: {acc:.4f}, Top5 Acc: {acc5:.4f}'
    )
    return acc, acc5


def main(cfg):
     # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(dataloader_config=dataloader_config)
    PROJECT_DIR = cfg.project_dir
    # if cfg.exp and accelerator.is_main_process:
        # logger = SimpleLogger(f'{PROJECT_DIR}/exp/{cfg.exp}/knn.jsonl')
        # sys.stdout, sys.stderr = logger, logger
    set_seed(0)
    transform = Compose([
        Resize(256, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(224),
        PILToTensor(), # Faster than ToTensor
    ])
    train_dataset = ImageFolder(f'{cfg.data_path}/train', transform=transform)
    val_dataset = ImageFolder(f'{cfg.data_path}/val', transform=transform)

    # TODO:
    train_loader = tud.DataLoader(train_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    val_loader = tud.DataLoader(val_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    if cfg.dino_type == 'vit_s_16':
        model = ViT(img_size=224, patch_size=16, num_classes=0, dim=384, depth=12, num_heads=6, mlp_ratio=4, drop_path_rate=0.1, qkv_bias=True, qk_norm=False)
        if cfg.exp:
            state_dict = torch.load(cfg.checkpoint_path, map_location='cpu',weights_only=False)
            weights = state_dict['dino_ema'] if 'dino_ema' in state_dict else state_dict['ema']
        else:
            model.head = nn.Identity()
            state_dict = torch.load(cfg.checkpoint_path, map_location='cpu',weights_only=True)
            weights = {key.replace('patch_embed.proj', 'patch_emb.conv').replace('mlp.', 'ffn.'): value for key,value in state_dict.items()}

        model.load_state_dict(weights, strict=True)
        model = model.eval().requires_grad_(False).to(accelerator.device)
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    all_labels, all_features = [], []
    # for x1, labels in train_loader:
    for x1, labels in tqdm(train_loader, desc='extract train features', disable = not accelerator.is_main_process):
        with accelerator.autocast():
            with torch.no_grad():
                x1 = x1.to(torch.float32).div_(255)
                x1 = Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x1)

                tokens = model.forward_feature(x1)
                features = model.norm(tokens[:,0])
                features = F.normalize(features.float(), p=2, dim=1)
                features, labels = accelerator.gather_for_metrics((features, labels))
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())

    all_features, all_labels = torch.cat(all_features, dim=0).to(accelerator.device), torch.cat(all_labels, dim=0).to(accelerator.device)
    acc, acc5 = evaluate(all_features, all_labels, val_loader, accelerator, model, cfg)
    if accelerator.is_main_process:
        record = {
            "dino_type": cfg.dino_type,
            "k": cfg.k,
            "temperature": cfg.temperature,
            "top1": acc,
            "top5": acc5,
            "image_size": cfg.dino_image_size,
            "num_train": all_labels.shape[0],
            'exp': cfg.exp,
            'base model epoch': cfg.epoch,
            
        }
        results_path = f'{PROJECT_DIR}/exp/{cfg.exp}/knn.jsonl' if cfg.exp else Path('knn_results.jsonl')
        with open(results_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saving result: {record}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear Probing Training.")
    parser.add_argument('--dino_type', type=str, default='vit_s_16')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--data_path',type=str, default=str(Path(__file__).resolve().parents[1]/ 'data/imagenet/images'), )
    parser.add_argument('--exp', type=str, default=None)

    cfg = parser.parse_args()
    cfg.project_dir = Path(__file__).resolve().parents[1]
    cfg.dino_image_size =  224
    cfg.num_classes = 1000
    cfg.local_batch_size = 128
    cfg.k = 20
    cfg.temperature = 0.07

    if cfg.exp: # from our trained model, instead of official checkpoint
        train_cfg = DefaultConfig.from_json(f'{cfg.project_dir}/exp/{cfg.exp}/config.json')
        cfg.dino_type = train_cfg.dino_type
        cfg.dino_image_size = train_cfg.dino_image_size
        cfg.num_classes = train_cfg.num_classes
        cfg.checkpoint_path = f'{cfg.project_dir}/exp/{cfg.exp}/epoch_{cfg.epoch:03d}.pth'
    else:
        cfg.checkpoint_path = f'{cfg.project_dir}/data/official_ckpts/dino_vits16.pt'  
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


    # torch._dynamo.config.disable = True
    main(cfg)

