import torch
import torch.nn.functional as F
import sys
import os
from config import DefaultConfig, SimpleLogger
from copy import deepcopy
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor, Normalize
from torchvision.datasets import ImageFolder
import torch.utils.data as tud
from time import time
from einops import rearrange
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, broadcast_object_list
from models.jit_clean import JiT_models
import numpy as np
from PIL import Image
import timm
from collections import OrderedDict
from torch.func import jvp
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
        
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

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

    w,h = pil_image.size
    crop_x = (w - image_size) // 2
    crop_y = (h - image_size) // 2
    # return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])
    return pil_image.crop((crop_x, crop_y, crop_x+image_size, crop_y+image_size))

def main():
    # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(mixed_precision="bf16", dataloader_config=dataloader_config)
    
    cfg = DefaultConfig()
    cfg.parse_args() 
    timestamp_list = [cfg.timestamp]
    broadcast_object_list(timestamp_list, from_process=0)
    cfg.timestamp = timestamp_list[0]
    cfg.results_dir = f'./exp/{cfg.timestamp}/'

    set_seed(cfg.seed, device_specific=True)
    # for accelerationf

    torch.set_float32_matmul_precision('high') 
    # torch.backends.cudnn.benchmark = True
    #TODO
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
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
        # ToTensor(), 
        PILToTensor(), # Faster than ToTensor
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    dataset = ImageFolder(cfg.data_path, transform=transform)

    # TODO:
    dataloader = tud.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    # TODO: proj_drop 
    model = JiT_models[cfg.model_type](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
    ema = deepcopy(model).to(accelerator.device)
    for param in ema.parameters():
        param.requires_grad_(False)
    
    # accelerator.register_for_checkpointing(ema) # By default, Accelerator doesn't save EMA, so register it. 
    
    # optimizer
    effective_bs = cfg.batch_size * accelerator.num_processes
    lr = cfg.blr * effective_bs / 256
    accelerator.print(f'Actual learning rate:{lr}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0)
    accelerator.print(optimizer)

    # linear warmup, then constant
    warmup_steps = cfg.warmup_epochs * len(dataloader)
    lr_lambda = lambda cur_step : cur_step / warmup_steps if cur_step < warmup_steps else 1.0 
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    # model = torch.compile(model)
    dataloader, model, optimizer, scheduler= accelerator.prepare(dataloader, model,  optimizer,scheduler)

    accelerator.print(f"Num proccess: {accelerator.num_processes}; Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Dataset size: {len(dataset)} images")

    # num_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model {cfg.model_type} has trainable {num_params/1e6:.6f} M parameters.")
    accelerator.print(
        f"Parameter dtypes -> model: {next(model.parameters()).dtype}"
    )

    # start training
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Initialize EMA with the same weights as the model

    model.train() # important, train(). => class dropout.
    ema.eval()
    start_epoch,train_steps, log_steps, running_loss, running_reg_loss, running_fm_loss = 1,0,0,0,0,0
    train_start_time = time()
    rng = torch.Generator(accelerator.device).manual_seed(42) # for negative sampling.



    for epoch in range(start_epoch, cfg.epochs+1):
        accelerator.print(f"[Exp Id: {cfg.timestamp}] | Start Epoch {epoch}/{cfg.epochs}...")

        for x1, y in dataloader: 
            # x1: (B,C,H,W)
            # y: (B,)
            
            # 1. scale x to [-1, 1]
            x1 = x1.to(accelerator.device, non_blocking=True).to(torch.float32).div_(255)
            x1 = 2.0 * x1 - 1.0  

            # dropout labels, here we use constant dropout ratio as 0.1
            drop_ids = torch.rand(y.shape[0], device=y.device) < 0.1
            y =  torch.where(drop_ids, cfg.num_classes, y)

            with accelerator.autocast():
                # logit-normal time sampling
                z = torch.randn(x1.shape[0], device=x1.device) * cfg.P_std + cfg.P_mean
                t = torch.sigmoid(z)
                # t = torch.rand((x1.shape[0],)).to(accelerator.device) # [B_local, ]
                t= rearrange(t, 'b -> b 1 1 1')

                # flow matching:predict x0, regress v.
                x0 = torch.randn_like(x1) * cfg.noise_scale
                xt = x0 * (1 - t) + x1 * t
                v_target = (x1 - xt) / (1-t).clamp_min(0.05)

                x1_pred = model(xt, t.flatten(), y)
                v_pred = (x1_pred - xt) / (1 - t).clamp_min(0.05)
                fm_loss = (v_target - v_pred)**2 # (B, C, H, W)
                fm_loss = fm_loss.mean(dim=[1,2,3]).mean()


                loss = fm_loss 

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            update_ema(ema, accelerator.unwrap_model(model), decay=cfg.ema_decay)
            train_steps += 1
            log_steps += 1
            running_fm_loss += fm_loss.detach()
            # running_reg_loss += reg_loss.detach()
            running_loss += loss.detach()  # only compare the standard FM loss.
            if train_steps % cfg.log_step_freq == 0 :
                running_loss = accelerator.reduce(running_loss.detach(), reduction="mean")
                running_fm_loss = accelerator.reduce(running_fm_loss.detach(), reduction="mean")
                avg_loss = running_loss / log_steps
                avg_fm_loss = running_fm_loss / log_steps
                elapsed = time() - train_start_time
                accelerator.print(
                    f"Step {train_steps}, Avg Loss: {avg_loss:.4f}, Avg FM Loss:{avg_fm_loss:.4f}, Avg_sample_t:{t.mean().item():.4f} " +
                    f"Elapsed: {elapsed:.1f}s",
                    
                    flush=True,
                )
                running_loss = 0
                running_fm_loss = 0
                running_reg_loss = 0
                log_steps = 0
    
            
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.epochs:
            accelerator.wait_for_everyone()
            
            to_save = {
                "model": accelerator.unwrap_model(model).state_dict(),
                "ema": ema.state_dict(),
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