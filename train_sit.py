import torch
import torch.nn.functional as F
import sys
import os
from config import DefaultConfig, SimpleLogger
from models import SiT_models, update_ema
from copy import deepcopy
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, Normalize
from torchvision.datasets import ImageFolder
from diffusers.models import AutoencoderKL
import torch.utils.data as tud
import random
from collections import Counter
from time import time
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
from einops import rearrange,repeat
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list
import numpy as np
from PIL import Image
from sample import sample
from evaluator import Evaluator, evalutate
from types import SimpleNamespace



def compute_xt_distance(x0, x1, t):
    """
    x0, x1: [B, C, H, W]
    t: [B,1,1,1]
    Return: D, [B, B]; 
    D[i,j] represents the distance between i-th sample and j-th sample at time t[i].
    """
    x0_flat = rearrange(x0, 'b c h w -> b (c h w)')
    x1_flat = rearrange(x1, 'b c h w -> b (c h w)') 
    t = rearrange(t, 'b -> b 1')
    x_sum_flat = x0_flat + x1_flat
    
    D0 = torch.cdist(x0_flat, x0_flat, p=2).pow(2) 
    D1 = torch.cdist(x1_flat, x1_flat, p=2).pow(2)
    D_sum = torch.cdist(x_sum_flat, x_sum_flat, p=2).pow(2)
    
    
    c0 = (1 - t) * (1 - 2 * t)
    c1 = t * (2 * t - 1)
    c_sum = t * (1 - t)
    
    D = c0 * D0 + c1 * D1 + c_sum * D_sum

    return D**0.5

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main():
    # set up config
    accelerator = Accelerator()
    cfg = DefaultConfig()
    cfg.parse_args() 
    timestamp_list = [cfg.timestamp]
    broadcast_object_list(timestamp_list, from_process=0)
    cfg.timestamp = timestamp_list[0]
    cfg.results_dir = f'./exp/{cfg.timestamp}/'

    set_seed(cfg.seed, device_specific=True)
    # set up logger
    if accelerator.is_main_process:
        os.makedirs(cfg.results_dir, exist_ok=True)
        cfg.save()
        logger = SimpleLogger(os.path.join(cfg.results_dir, "log.txt"))
        sys.stdout, sys.stderr = logger, logger

    accelerator.print(cfg)

    # prepare data
    # Normalize is different for vae and dino. So we move it to the training loop.
    transform = Compose([
        Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        RandomHorizontalFlip(),
        ToTensor(), 
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(cfg.data_path, transform=transform)

    dataloader = tud.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    #prepare model
    assert cfg.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = cfg.image_size // 8
   
    if cfg.use_repa:
        with accelerator.main_process_first():
            repa_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
        del repa_encoder.head
        patch_resolution = 16 * (cfg.image_size // 256)
        repa_encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                repa_encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
        repa_encoder.head = torch.nn.Identity()
        repa_encoder = repa_encoder.to(accelerator.device)
        repa_encoder.eval()
    model = SiT_models[cfg.model_type](input_size=latent_size,num_classes=cfg.num_classes, use_repa=cfg.use_repa)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{cfg.vae}").to(accelerator.device).eval()
    ema = deepcopy(model).to(accelerator.device)
    for param in ema.parameters():
        param.requires_grad_(False)
    
    accelerator.register_for_checkpointing(ema) # By default, Accelerator doesn't save EMA, so register it. 
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    dataloader, model, optimizer = accelerator.prepare(dataloader, model,  optimizer)

    accelerator.print(f"Num proccess: {accelerator.num_processes}; Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Dataset size: {len(dataset)} images")

        
    num_params = sum(p.numel() for p in model.parameters())
    accelerator.print(f"Model {cfg.model_type} has {num_params/1e6:.2f} M parameters.")
    accelerator.print(
        f"Parameter dtypes -> model: {next(model.parameters()).dtype}, vae: {next(vae.parameters()).dtype}"
    )

    # start training
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Initialize EMA with the same weights as the model


    model.train() # important, train(). => class dropout.
    ema.eval()
    start_epoch,train_steps, log_steps, running_loss = 1,0,0,0
    total_steps = len(dataloader) * cfg.epochs
    train_start_time = time()

        
    for epoch in range(start_epoch, cfg.epochs+1):
        accelerator.print(f"[Exp Id: {cfg.timestamp}] | Start Epoch {epoch}/{cfg.epochs}...")
   
        for raw_x, y in dataloader: #raw_x: [0-1]
            with torch.no_grad(): # vae should be fp32.
                # vae input: [-1,1].
                x1 = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=False)(raw_x)
        
                if cfg.use_repa:
                    repa_x1 = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(raw_x)
                    repa_x1 = F.interpolate(repa_x1, 224*(cfg.image_size//256), mode='bicubic')
                    repa_x1 = repa_encoder.forward_features(repa_x1)['x_norm_patchtokens']
                
            with accelerator.autocast():
                x0 = torch.randn_like(x1) 


                t = torch.rand((x1.shape[0],)).to(accelerator.device) # [B_local, ]
                t = rearrange(t, 'b -> b 1 1 1')
                xt = x0 * (1 - t) + x1 * t
                target_v = x1 - x0 # [B_local, C, H/8, W/8]

                if cfg.use_repa:
                    pred_v, h = model(xt, t.flatten(), y)
                    h, repa_x1 = F.normalize(h, dim=-1), F.normalize(repa_x1, dim=-1) #[B, T, D]
                    repa_loss = - (h * repa_x1).sum(dim=-1).mean()
                else:
                    pred_v = model(xt, t.flatten(), y) #[B, C, H ,W]
                    repa_loss = 0
                fm_loss = torch.mean((pred_v - target_v) ** 2) # over all dimensions

                loss = fm_loss + 0.5 * repa_loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            if cfg.use_repa:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema, accelerator.unwrap_model(model))

            train_steps += 1
            log_steps += 1
            running_loss += fm_loss.detach()  # only compare the standard FM loss.

            if train_steps % cfg.log_every == 0 :
                running_loss = accelerator.reduce(running_loss.detach(), reduction="mean")
                avg_loss = running_loss / log_steps
                elapsed = time() - train_start_time
                avg_step_time = elapsed / train_steps
                accelerator.print(
                    f"Step {train_steps}, Avg Loss: {avg_loss:.4f}, Avg_sample_t:{t.mean().item():.4f} " +
                    f"Elapsed: {elapsed:.1f}s, Avg step time: {avg_step_time:.3f}s",
                    
                    flush=True,
                )
                running_loss = 0
                log_steps = 0
    
            
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.epochs:
            accelerator.wait_for_everyone()
            
            to_save = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": cfg,
            }

            checkpoint_path = os.path.join(cfg.results_dir, f"epoch_{epoch:03d}.pth")
            if accelerator.is_main_process:
                torch.save(to_save, checkpoint_path)
            # accelerator.save_state(checkpoint_path)
            accelerator.print(f"✅ Saved checkpoint at: {checkpoint_path}")
            

if __name__ == '__main__':
    main()


#  accelerate launch --num_processes=3 train.py --batch_size 64  --epochs 500 --num_classes 10 --seed 0 --model_type 'SiT-S/2' --flow_type 'standard';