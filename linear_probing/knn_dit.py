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
def evaluate(all_features, all_labels, val_loader, accelerator, model, hook_cache, cfg, vae=None):
    model.eval()
    total_samples = 0
    total_correct = 0
    total_correct5 = 0
    # for x, target in val_loader:
    for x, target in tqdm(val_loader, desc='Evaludate kNN', disable = not accelerator.is_main_process):
        x1 = x.to(torch.float32).div_(255)
        x1 = 2.0 * x1 - 1.0  
        if vae is not None:
            x1 = vae.encode(x1).latent_dist.sample().mul_(0.18215)  # [B_local, C, H/8, W/8]
        b = x1.shape[0]
        t = torch.full((b, ), cfg.t, dtype=torch.float32, device=accelerator.device)
        y = torch.full((b, ), cfg.num_classes, dtype=torch.long, device=accelerator.device)

        noise = torch.randn_like(x1) 
        tb = t[:, None, None,None] #[B,1,1,1]
        xt = (1-tb) * noise + tb * x1
        hook_cache.clear()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            try: 
                model_t = 1.0 - t if cfg.dit_type == 'SRA' else t
                _ = model(xt, model_t, y)
            except StopForwardException:
                pass
        similarities, indices = (hook_cache[0] @ all_features.T).topk(cfg.k, dim=1)
        neighbor_labels = all_labels[indices] #[B, k]
        weights = (similarities / cfg.temperature).exp() # [B,k]
        scores = torch.zeros(b, cfg.num_classes, device=accelerator.device)
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
        f't={cfg.t}, layer_idx={cfg.layer_idx}, '
        f'Top1 Acc: {acc:.4f}, Top5 Acc: {acc5:.4f}'
    )
    return acc, acc5


def main(cfg):
     # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(mixed_precision="no", dataloader_config=dataloader_config)
    PROJECT_DIR = cfg.project_dir
    # if cfg.exp and accelerator.is_main_process:
        # logger = SimpleLogger(f'{PROJECT_DIR}/exp/{cfg.exp}/knn.jsonl')
        # sys.stdout, sys.stderr = logger, logger
    set_seed(0)

    train_transform = Compose([
        Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        PILToTensor(), # Faster than ToTensor
    ])
    val_transform = Compose([
        Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        PILToTensor(), 
    ])
    train_dataset = ImageFolder(f'{cfg.data_path}/train', transform=train_transform)
    val_dataset = ImageFolder(f'{cfg.data_path}/val', transform=val_transform)

    # TODO:
    train_loader = tud.DataLoader(train_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False, persistent_workers=True)
    val_loader = tud.DataLoader(val_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # register hook
    hook_cache = []
    layer_idx, last_layer, vae = cfg.layer_idx, cfg.layer_idx, None
    # 
    if  cfg.dit_type in JiT_models.keys(): 
        model = JiT_models[cfg.dit_type](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
        if cfg.exp:
            state_dict = torch.load(cfg.checkpoint_path, map_location='cpu',weights_only=False)
            weights = state_dict['ema']
            cfg.epoch = state_dict['epoch']
        else:
            state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/jit_h_16.pt', map_location='cpu',weights_only=False)
            weights = state_dict['model_ema1']
            weights = {key.removeprefix('net.'): value for key,value in weights.items() }
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        model = model.to(accelerator.device)
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                tokens = output
                if layer_idx >= model.in_context_start:
                    tokens = output[:, model.in_context_len:, :]
                tokens = tokens.float().mean(dim=1)  # [B, N, D] -> [B, D]
                hook_cache.append( F.normalize(tokens, p=2, dim=1) ) # [B, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        
        model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
    
    elif cfg.dit_type in PiT_models.keys():
        model = PiT_models['PiT-XL/16'](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
        state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/pixeldit.pt', map_location='cpu',weights_only=False)
        weights = {}
        for key, value in state_dict['state_dict'].items():
            if key.startswith('ema_denoiser'):
                key = key.removeprefix('ema_denoiser.')
                key = key.replace('pixel_embedder.proj.', 'x_embedder.')
                key = key.replace('s_embedder.proj.', 's_embedder.')
                weights[key] = value

        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        model = model.to(accelerator.device)

        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if layer_idx < model.patch_depth: # output is [B, N, D]
                    tokens = output.float().mean(dim=1)  # [B, N, D] -> [B, D]

                else: # output is [B*h*w, p1*p2, c]
                    h, w = cfg.image_size // model.patch_size, cfg.image_size // model.patch_size
                    p1, p2 = model.patch_size, model.patch_size
                    output = rearrange(output, '(B h w) (p1 p2) c -> B (h w) (p1 p2 c)', h=h, w=w, p1=p1, p2=p2)
                    tokens = output.float().mean(dim=1)  # [B, N, D] -> [B, D]
                hook_cache.append( F.normalize(tokens, p=2, dim=1))  # [B, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        if layer_idx < model.patch_depth:
            model.patch_blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        else:
            model.pixel_blocks[layer_idx - model.patch_depth].register_forward_hook(make_hook(layer_idx))
    

    elif cfg.dit_type in ['SiT', 'SRA', 'SelfFlow']:
        model = SiT_models['SiT-XL/2'](input_size=cfg.image_size//8, in_channels=4, num_classes=cfg.num_classes, mlp_ratio=4, learn_sigma=cfg.dit_type != 'SRA')
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(accelerator.device)
        if cfg.dit_type == 'SiT':
            weights = torch.load(f'{PROJECT_DIR}/data/official_ckpts/sit.pt', map_location='cpu', weights_only=True, mmap=True)

        elif cfg.dit_type == 'SRA':
            weights = torch.load(f'{PROJECT_DIR}/data/official_ckpts/sra.pt', map_location='cpu', weights_only=True, mmap=True)
            weights = {key: value for key, value in weights.items() if not key.startswith('ap_head.')}

        elif cfg.dit_type == 'SelfFlow':
            state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/self_flow.pt', map_location='cpu', weights_only=True, mmap=True)
            weights = state_dict['model']
            weights = {key: value for key, value in weights.items() if not key.startswith('projector.')}
            key = 'x_embedder.proj.weight'
            weights[key] = weights[key].reshape(1152, 4, 2, 2)

        model.load_state_dict(weights, strict=True)
        model  = model.eval().requires_grad_(False).to(accelerator.device)
        vae = vae.eval().requires_grad_(False).to(accelerator.device)
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                tokens = output.float().mean(dim=1)  # [B, N, D] -> [B, D]
                hook_cache.append( F.normalize(tokens, p=2, dim=1) ) # [B, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        
        model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        
    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)
    all_labels, all_features = [], []
    # for x1, labels in train_loader:
    for x1, labels in tqdm(train_loader, desc='extract train features', disable = not accelerator.is_main_process):
        with torch.no_grad():
            x1 = x1.to(torch.float32).div_(255)
            x1 = 2.0 * x1 - 1.0  

            if vae is not None:
                x1 = vae.encode(x1).latent_dist.sample().mul_(0.18215)  # [B_local, C, H/8, W/8]

            b = x1.shape[0]
            noise = torch.randn_like(x1) 
            t = torch.full((b, ), cfg.t, dtype=torch.float32, device=accelerator.device)
            y = torch.full((b, ), cfg.num_classes, dtype=torch.long, device=accelerator.device)
            tb = t[:, None, None,None] #[B,1,1,1]
            xt = (1-tb) * noise + tb * x1
            hook_cache.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                try:
                    model_t = 1.0 - t if cfg.dit_type == 'SRA' else t
                    _ = model(xt, model_t, y)
                except StopForwardException:
                    pass
            features, labels = accelerator.gather_for_metrics((hook_cache[0], labels))
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features, all_labels = torch.cat(all_features, dim=0).to(accelerator.device), torch.cat(all_labels, dim=0).to(accelerator.device)
    acc, acc5 = evaluate(all_features, all_labels, val_loader, accelerator, model, hook_cache, cfg, vae)
    if accelerator.is_main_process:
        record = {
            "dit_type": cfg.dit_type,
            "t": cfg.t,
            "layer_idx": cfg.layer_idx,
            "k": cfg.k,
            "temperature": cfg.temperature,
            "top1": acc,
            "top5": acc5,
            "image_size": cfg.image_size,
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
    parser.add_argument('--dit_type', type=str, default=None )
    parser.add_argument('--t', type=float, default=None)
    parser.add_argument('--layer_idx', type=int, default=None)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--data_path',type=str, default=str(Path(__file__).resolve().parents[1]/ 'data/imagenet/images'), )


    cfg = parser.parse_args()
    cfg.project_dir = Path(__file__).resolve().parents[1]
    cfg.image_size =  256
    cfg.num_classes = 1000
    cfg.local_batch_size = 128
    cfg.k = 20
    cfg.temperature = 0.07

    if cfg.exp: # from our trained model, instead of official checkpoint
        train_cfg = DefaultConfig.from_json(f'{cfg.project_dir}/exp/{cfg.exp}/config.json')
        cfg.dit_type = train_cfg.dit_type 
        cfg.image_size = train_cfg.image_size
        cfg.num_classes = train_cfg.num_classes
        cfg.checkpoint_path = f'{cfg.project_dir}/exp/{cfg.exp}/epoch_{cfg.epoch:03d}.pth'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


    # torch._dynamo.config.disable = True
    main(cfg)

