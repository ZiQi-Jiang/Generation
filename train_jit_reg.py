import torch
import torch.nn.functional as F
import sys
import os
from config import DefaultConfig, SimpleLogger
from copy import deepcopy
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor,Normalize
from torchvision.datasets import ImageFolder
import torch.utils.data as tud
from time import time
from einops import rearrange, repeat
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, broadcast_object_list
from models.jit_reg import JiT_models
from models.utils import dino_cross_entropy
import numpy as np
from PIL import Image
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def calculate_pairwise_distance(h):
    # h: [B, TD]
    # memory-efficient
    with torch.autocast(device_type=h.device.type, enabled=False):
        z = h.float()
        norm_sq = z.square().sum(dim=1,keepdim=True) #[B, 1]
        dist = ((norm_sq + norm_sq.T) - 2.0 * z @ z.T).clamp_min(0.0)
        dist = dist / h.shape[1] # normalize by feature dimension
        dist[range(dist.shape[0]), range(dist.shape[0])] = 0.0
    return dist

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

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def main():
    # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(mixed_precision="bf16", dataloader_config=dataloader_config)
    
    cfg = DefaultConfig()
    cfg.parse_args() 
    timestamp_list = [cfg.timestamp]
    broadcast_object_list(timestamp_list, from_process=0)
    cfg.timestamp = timestamp_list[0]
    cfg.results_dir = f'./exp/{cfg.timestamp}/' # type: ignore

    set_seed(cfg.seed, device_specific=True)
    # for acceleration

    torch.set_float32_matmul_precision('high') 
    # torch.backends.cudnn.benchmark = True
    # #TODO
    # torch.backends.cudnn.benchmark = True
    # torch._dynamo.config.cache_size_limit = 128
    # torch._dynamo.config.optimize_ddp = False
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
    dataloader = tud.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)

    # TODO: proj_drop 
    model = JiT_models[cfg.dit_type](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4, reg_type = cfg.reg_type)
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
    dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model,  optimizer, scheduler)

    accelerator.print(f"Num proccess: {accelerator.num_processes}; Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Dataset size: {len(dataset)} images")

    # num_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model {cfg.dit_type} has trainable {num_params/1e6:.6f} M parameters.")
    accelerator.print(
        f"Parameter dtypes -> model: {next(model.parameters()).dtype}"
    )

    # start training
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Initialize EMA with the same weights as the model

    if cfg.reg_type == 'repa':
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

    model.train() # important, train(). => class dropout.
    ema.eval() # self.training = False, for sra.
    start_epoch,train_steps, log_steps, running_fm_loss, running_reg_loss = 1,0,0,0,0
    train_start_time = time()
    # if cfg.resume_exp is not None: # resume training
    #     accelerator.print(f"Resuming from experiment: {cfg.resume_exp}, epoch: {cfg.resume_epoch:03d}")
    #     ckpt_path = os.path.join('exp',cfg.resume_exp, f'epochs_{cfg.resume_epoch:03d}')
    #     accelerator.load_state(ckpt_path) 
    #     start_epoch = cfg.resume_epoch + 1
    rng = torch.Generator(accelerator.device).manual_seed(42) # for negative sampling.

    for epoch in range(start_epoch, cfg.epochs+1):
        accelerator.print(f"[Exp Id: {cfg.timestamp}] | Start Epoch {epoch}/{cfg.epochs}...")

        for raw_x, raw_y in dataloader: 
            # x1: (B,C,H,W)
            # y: (B,)
            
            # 1. scale x to [-1, 1]
            raw_x = raw_x.to(accelerator.device, non_blocking=True).to(torch.float32).div_(255) # (0,1)
            x1 = 2.0 * raw_x - 1.0

            # dropout labels, here we use constant dropout ratio as 0.1
            drop_ids = torch.rand(raw_y.shape[0], device=raw_y.device) < 0.1
            y =  torch.where(drop_ids, cfg.num_classes, raw_y) # label dropout for cfg.

            if cfg.reg_type == 'repa':
                repa_x1 = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(raw_x)
                repa_x1 = F.interpolate(repa_x1, 224*(cfg.image_size//256), mode='bicubic')
                with torch.inference_mode(): # 5s -> 2.8 (per 20 iters)
                    with accelerator.autocast():
                        repa_x1 = repa_encoder.forward_features(repa_x1)['x_norm_patchtokens'] #(B, 256, 768)

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
                if cfg.reg_type == 'repa': # none is false, 'repa'/'irepa' is true
                    x1_pred, h = model(xt, t.flatten(), y, cfg.reg_depth)
                    h, repa_x1 = F.normalize(h, dim=-1), F.normalize(repa_x1, dim=-1) #[B, T, D]
                    reg_loss = - (h * repa_x1).sum(dim=-1).mean()
                    reg_lam = cfg.reg_lam
                elif cfg.reg_type == 'disperse':
                    x1_pred, h = model(xt, t.flatten(), y, cfg.reg_depth)
                    h = h.reshape((h.shape[0],-1)) # flatten [B, TD]

                    # 1. official,  requires larger memory
                    # diff = F.pdist(h).pow(2)/h.shape[1] # pairwise distance, [B*(B-1)/2]
                    # diff = torch.concat((diff, diff, torch.zeros(h.shape[0], device=accelerator.device)))  # match JAX implementation of full BxB matrix

                    # 2. equivalent to pdist, but requires less memory
                    dist = calculate_pairwise_distance(h) #[B, B]
                    reg_loss = torch.log(torch.exp(-dist).mean()) # calculate loss
                    reg_lam = cfg.reg_lam
                elif cfg.reg_type == 'ig':
                    x1_pred, x1_h = model(xt, t.flatten(), y, cfg.reg_depth)# (B, C, H W) each
                    v_h = (x1_h - xt ) / (1-t).clamp_min(0.05)
                    reg_loss = (v_target - v_h)**2
                    reg_loss = reg_loss.mean()
                    reg_lam = cfg.reg_lam
                elif cfg.reg_type == 'sra':
                    x1_pred, x1_h_stu = model(xt, t.flatten(), y, cfg.reg_depth)

                    t_tea =torch.clamp( t + (cfg.t_max * torch.rand(t.shape,device=t.device, dtype=t.dtype, generator=rng)), 0, 1)
                    xt_tea = x0 * (1 - t_tea) + x1 * t_tea
                    with torch.inference_mode():
                        _, x1_h_tea = ema(xt_tea, t_tea.flatten(), y, cfg.reg_depth_tea)

                    reg_loss = 1 - F.cosine_similarity(x1_h_tea, x1_h_stu, dim=-1)
                    reg_loss = reg_loss.mean()
                    reg_lam = cfg.reg_lam
                elif cfg.reg_type == 'sd-dit':
                    x1_pred, visible_h_stu, ids_keep = model(xt, t.flatten(), y, depth=cfg.reg_depth) #[B,N,D], [B,N*(1-mask_ratio), k], [B, N*(1-mask_ratio)]

                    t_tea = torch.ones(t.shape, device=t.device, dtype=t.dtype)

                    xt_tea = x0* (1-t_tea) + x1 * t_tea # for teacher, almost no noise. 
                    with torch.no_grad():
                        _, x1_h_tea = ema(xt_tea, t_tea.flatten(), y, depth=cfg.reg_depth_tea) # same depth as student.
                        ids_tea_keep = repeat(ids_keep, 'b n -> b n k', k=ema.K)
                        visible_h_tea = torch.gather(x1_h_tea, dim=1, index=ids_tea_keep)

                    tau_stu, tau_tea =0.1,  0.09 + (train_steps * accelerator.num_processes/warmup_steps)*(0.099 - 0.09) if epoch <=cfg.warmup_epochs else 0.099

                    reg_loss = dino_cross_entropy(visible_h_tea, visible_h_stu, ema.center_patch, tau_stu, tau_tea)
                    reg_lam = cfg.reg_lam

                    batch_center = x1_h_tea.mean(dim=[0,1], keepdim=True, dtype=torch.float32)
                    batch_center = accelerator.reduce(batch_center, reduction='mean')
                    ema.center_patch = ema.m_p * ema.center_patch + (1-ema.m_p) * batch_center
                else: # no reg.
                    x1_pred = model(xt, t.flatten(), y)
                    reg_loss = torch.zeros((), device=accelerator.device, dtype=x1_pred.dtype)
                    reg_lam = 0.0
                v_pred = (x1_pred - xt) / (1 - t).clamp_min(0.05)
                fm_loss = (v_target - v_pred)**2 # (B, C, H, W)
                fm_loss = fm_loss.mean(dim=[1,2,3]).mean()
                loss = fm_loss + reg_lam * reg_loss
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            update_ema(ema, accelerator.unwrap_model(model), decay=cfg.ema_decay)
            train_steps += 1
            log_steps += 1
            running_fm_loss += fm_loss.detach() 
            running_reg_loss += reg_loss.detach()  
            if train_steps % cfg.log_step_freq == 0 :
                running_fm_loss = accelerator.reduce(running_fm_loss.detach(), reduction="mean")
                running_reg_loss = accelerator.reduce(running_reg_loss.detach(), reduction="mean")
                avg_fm_loss = running_fm_loss / log_steps
                avg_reg_loss = running_reg_loss / log_steps
                elapsed = time() - train_start_time
                accelerator.print(
                    f"Step {train_steps}, Avg_fm_Loss: {avg_fm_loss:.4f}, Avg_reg_loss: {avg_reg_loss:.4f}, Avg_sample_t:{t.mean().item():.4f} " +
                    f"Elapsed: {elapsed:.1f}s",
                    
                    flush=True,
                )
                running_fm_loss = 0
                running_reg_loss = 0
                log_steps = 0
    
            
        if epoch % cfg.save_epoch_freq == 0:
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


#  accelerate launch --num_processes=3 train.py --batch_size 64  --epochs 500 --num_classes 10 --seed 0 --dit_type 'SiT-S/2' --flow_type 'standard';