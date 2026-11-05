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
from einops import rearrange
class StopForwardException(Exception):
    pass


@torch.no_grad()
def evaluate(classifiers, val_loader, accelerator, model, hook_cache, cfg, vae=None):
    model.eval()
    [classifier.eval() for classifier in classifiers]
    total_samples = 0
    total_loss = torch.zeros(len(cfg.layer_idx), dtype=torch.float64, device=accelerator.device)
    total_correct = torch.zeros(len(cfg.layer_idx), dtype=torch.int64, device=accelerator.device)
    total_correct5 = torch.zeros(len(cfg.layer_idx), dtype=torch.int64, device = accelerator.device)
    for x, target in val_loader:
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
                model_t = 1.0 - t if cfg.model_type == 'SRA' else t
                _ = model(xt, model_t, y)
            except StopForwardException:
                pass
        
        output = torch.stack([classifier(hook_cache[layer_idx]) for classifier, layer_idx in zip(classifiers, cfg.layer_idx)], dim=1) #[B, len(cfg.layer_idx), num_classes]
        
        loss = F.cross_entropy(output.float().transpose(1,2), target[:,None].expand(-1, len(cfg.layer_idx)), reduction='none')#[B, len(cfg.layer_idx)]

        correct = output.argmax(dim=-1).eq(target[:,None]) #[B, len(cfg.layer_idx)]
        correct5 = output.topk(5,dim=-1).indices.eq(target[:,None, None]).any(dim=-1) 

        loss, correct, correct5 = accelerator.gather_for_metrics((loss, correct, correct5))
        total_loss += loss.double().sum(dim=0)
        total_correct += correct.sum(dim=0)
        total_correct5 += correct5.sum(dim=0)
        total_samples += correct.shape[0]
    loss_per_image = (total_loss / total_samples).tolist()
    correct_counts = total_correct.tolist()
    correct5_counts = total_correct5.tolist()

    acc = [count / total_samples for count in correct_counts]
    acc5 = [count / total_samples for count in correct5_counts]
    for i, layer_idx in enumerate(cfg.layer_idx):
        accelerator.print( f't={cfg.t}, layer_idx={layer_idx}, loss per image: {loss_per_image[i]:.4f}, Top1 Acc: [{total_correct[i]}|{total_samples}]({acc[i]:.4f}), Top5 Acc: [{total_correct5[i]}|{total_samples}]({acc5[i]:.4f}) ')
    return acc

def main(cfg):
     # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(mixed_precision="no", dataloader_config=dataloader_config)
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    set_seed(0)

    train_transform = Compose([
        # Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        RandomResizedCrop(cfg.image_size),
        RandomHorizontalFlip(),
        # ToTensor(), 
        PILToTensor(), # Faster than ToTensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    val_transform = Compose([
        Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        PILToTensor(), 
    ])
    train_dataset = ImageFolder(f'{cfg.data_path}/train', transform=train_transform)
    val_dataset = ImageFolder(f'{cfg.data_path}/val', transform=val_transform)

    # TODO:
    train_loader = tud.DataLoader(train_dataset, batch_size=cfg.local_batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = tud.DataLoader(val_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

    # register hook
    hook_cache = {}
    last_layer = max(cfg.layer_idx)
    # 
    if  cfg.model_type in JiT_models.keys(): 
        model = JiT_models['JiT-H/16'](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
        state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/jit_h_16.pt', map_location='cpu')
        weights = state_dict['model_ema1']
        weights = {key.removeprefix('net.'): value for key,value in weights.items() }
        model.load_state_dict(weights, strict=True)
        model.eval().requires_grad_(False)
        model = model.to(accelerator.device)
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                tokens = output
                if layer_idx >= model.in_context_start:
                    tokens = tokens[:, model.in_context_len:, :]
                hook_cache[layer_idx] = tokens.float().mean(dim=1)  # [B, N, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        
        for layer_idx in cfg.layer_idx:
            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
    
    if cfg.model_type in PiT_models.keys():
        model = PiT_models['PiT-XL/16'](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
        state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/pixeldit.pt', map_location='cpu')
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
                    hook_cache[layer_idx] = output.float().mean(dim=1)  # [B, N, D] -> [B, D]
                else: # output is [B*h*w, p1*p2, c]
                    h, w = cfg.image_size // model.patch_size, cfg.image_size // model.patch_size
                    p1, p2 = model.patch_size, model.patch_size
                    output = rearrange(output, '(B h w) (p1 p2) c -> B (h w) (p1 p2 c)', h=h, w=w, p1=p1, p2=p2)
                    hook_cache[layer_idx] = output.float().mean(dim=1)  # [B, N, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        for layer_idx in cfg.layer_idx:
            if layer_idx < model.patch_depth:
                model.patch_blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
            else:
                model.pixel_blocks[layer_idx - model.patch_depth].register_forward_hook(make_hook(layer_idx))
    

    if cfg.model_type in ['SiT', 'SRA', 'SelfFlow']:
        model = SiT_models['SiT-XL/2'](input_size=cfg.image_size//8, in_channels=4, num_classes=cfg.num_classes, mlp_ratio=4, learn_sigma=cfg.model_type != 'SRA')
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(accelerator.device)
        if cfg.model_type == 'SiT':
            weights = torch.load(f'{PROJECT_DIR}/data/official_ckpts/sit.pt', map_location='cpu', weights_only=True, mmap=True)

        elif cfg.model_type == 'SRA':
            weights = torch.load(f'{PROJECT_DIR}/data/official_ckpts/sra.pt', map_location='cpu', weights_only=True, mmap=True)
            weights = {key: value for key, value in weights.items() if not key.startswith('ap_head.')}

        elif cfg.model_type == 'SelfFlow':
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
                hook_cache[layer_idx] = output.float().mean(dim=1)  # [B, N, D] -> [B, D]
                if layer_idx == last_layer:
                    raise StopForwardException
            return hook_fn
        
        for layer_idx in cfg.layer_idx:
            model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        
    num_classes = 1000
    feature_dim = model.hidden_size

    lr = cfg.blr * cfg.local_batch_size * accelerator.num_processes / 256
    classifiers, optimizers, schedulers = [], [], []
    feature_dims = {}
    for layer_idx in cfg.layer_idx:
        if cfg.model_type in JiT_models.keys():
            feature_dim = model.hidden_size
        elif cfg.model_type in PiT_models.keys():
            feature_dim = model.hidden_size if layer_idx < model.patch_depth else model.patch_size * model.patch_size * model.pixel_hidden_dim
        classifier = nn.Sequential(nn.BatchNorm1d(feature_dim, affine=False),nn.Linear(feature_dim, num_classes))
        nn.init.normal_(classifier[1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(classifier[1].bias)

        feature_dims[layer_idx] = feature_dim
        optimizer = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        classifier, optimizer = accelerator.prepare(classifier, optimizer)
        classifiers.append(classifier)
        optimizers.append(optimizer)
        schedulers.append(scheduler)

    train_loader, val_loader = accelerator.prepare(train_loader, val_loader)

    best_vals,train_steps, train_start_time = [0.0 for _ in cfg.layer_idx], 0, time()

    for epoch in range(1, cfg.epochs + 1):
        [classifier.train() for classifier in classifiers]
        train_loss = torch.zeros(len(cfg.layer_idx), device=accelerator.device)
        train_total = 0

        for x1, labels in train_loader:
            with torch.no_grad():
                x1 = x1.to(torch.float32).div_(255)
                x1 = 2.0 * x1 - 1.0  

                if cfg.model_type in ['SiT', 'SRA', 'SelfFlow']:
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
                        model_t = 1.0 - t if cfg.model_type == 'SRA' else t
                        _ = model(xt, model_t, y)
                    except StopForwardException:
                        pass

            for i, layer_idx in enumerate(cfg.layer_idx):
                
                logits = classifiers[i](hook_cache[layer_idx])
                loss = F.cross_entropy(logits, labels)
                optimizers[i].zero_grad()
                accelerator.backward(loss)
                optimizers[i].step()
                train_loss[i] += loss.detach() * labels.shape[0]
            train_total += labels.shape[0]
            train_steps +=1
            if train_steps % 100 == 0 and accelerator.is_main_process:
                avg_loss = (train_loss / train_total).tolist()
                elapsed = time() - train_start_time
                for i, layer_idx in enumerate(cfg.layer_idx):
                    accelerator.print(f"Epoch: {epoch:03d}, Step {train_steps:06d}|{len(train_loader)}, t:{cfg.t}, layer_idx:{layer_idx:02d}, Avg Loss Per Image: {avg_loss[i]:.4f}, LR:{optimizers[i].param_groups[0]['lr']:.6f}, " +f"Elapsed: {elapsed:.1f}s,")
        [scheduler.step() for scheduler in schedulers]

        if epoch >= cfg.epochs * 0.8:
            val_top1s = evaluate(classifiers, val_loader, accelerator, model, hook_cache, cfg, vae)
            best_vals = [max(best_val, val_top1) for best_val, val_top1 in zip(best_vals, val_top1s)]

    if accelerator.is_main_process:
        file_path = 'results.jsonl'
        with open(file_path, mode='a', encoding='utf-8') as f:
            for i, layer_idx in enumerate(cfg.layer_idx):
                record = {
                        "model_type": cfg.model_type,
                        "t": cfg.t,
                        "layer_idx": layer_idx,
                        "best_val_top1": float(best_vals[i]),   
                        "epochs": cfg.epochs,
                        "local_batch_size": cfg.local_batch_size,
                        "blr": cfg.blr,
                        "image_size": cfg.image_size,
                        'feature_dim': feature_dims[layer_idx],
                    }
                print(f"Saving result: {record}")
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear Probing Training.")
    parser.add_argument('--model_type', type=str, default=None )
    parser.add_argument('--t', type=float, default=None)
    parser.add_argument('--layer_idx', type=int, nargs='+', default=None)

    cfg = parser.parse_args()
    cfg.local_batch_size = 128
    cfg.data_path = '/data/Generation/data/imagenet/images'
    cfg.image_size =  256
    cfg.num_classes = 1000
    cfg.blr = 0.001
    cfg.epochs = 100
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


    # torch._dynamo.config.disable = True
    main(cfg)

