import torch
import torch.nn.functional as F
import sys
import os
import torch.utils.data as tud
from config import DefaultConfig, SimpleLogger
from torchvision.datasets import ImageFolder
import torch.utils.data as tud
from time import time
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.utils import set_seed, broadcast_object_list
from models.dino_vit import ViT
import numpy as np
from datasets.utils import DINOCropAugmentation

def dino_loss(stu_logits, tea_logits, center, cfg, tea_temp):
    stu_log_prob = F.log_softmax(stu_logits.float() / 0.1, dim=-1).chunk(cfg.global_views + cfg.local_views)
    tea_prob = F.softmax((tea_logits.detach().float() - center) / tea_temp, dim=-1).chunk(cfg.global_views) #[2B, K] -> [[B,K], [B,K]]
    loss, n_terms= 0, 0
    for i, q in enumerate(tea_prob):
        for j, log_p in enumerate(stu_log_prob):
            if i == j:
                continue
            else:
                loss += -(q * log_p).sum(dim=-1).mean()
                n_terms +=1
    return loss / n_terms




@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
        
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


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

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # set up logger
    if accelerator.is_main_process:
        os.makedirs(cfg.results_dir, exist_ok=True)
        cfg.save()
        logger = SimpleLogger(os.path.join(cfg.results_dir, "log.txt"))
        sys.stdout, sys.stderr = logger, logger

    accelerator.print(cfg)

    # prepare data
    # Normalize is different for vae and dino. So we move it to the training loop.
    dataset = ImageFolder(cfg.data_path)
    if cfg.data_backend == 'pil':
        dataset.transform =  DINOCropAugmentation(cfg)
        dataloader = tud.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True) 
    else:
        from datasets.dali_loader import DinoDaliLoader
        dataloader = DinoDaliLoader(dataset, cfg, accelerator)
    if cfg.dino_type == 'vit_s_16':
        model = ViT(img_size=224, patch_size=16, num_classes=0, dim=384, depth=12, num_heads=6, mlp_ratio=4, drop_path_rate=0.1, qkv_bias=True, qk_norm=False)
        ema = ViT(img_size=224, patch_size=16, num_classes=0, dim=384, depth=12, num_heads=6, mlp_ratio=4, drop_path_rate=0.0, qkv_bias=True, qk_norm=False)
        ema = ema.to(accelerator.device).requires_grad_(False)
    
    # accelerator.register_for_checkpointing(ema) # By default, Accelerator doesn't save EMA, so register it. 
    
    # optimizer
    effective_bs = cfg.batch_size * accelerator.num_processes
    lr = cfg.blr * effective_bs / 256
    accelerator.print(f'Actual learning rate:{lr}')

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.ndim==1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.AdamW([{'params':decay, 'weight_decay':cfg.weight_decay}, {'params':no_decay, 'weight_decay':0.0}], lr=lr, betas=(0.9, 0.999))
    accelerator.print(optimizer)

    # model = torch.compile(model)
    model, optimizer= accelerator.prepare(model,  optimizer)
    if cfg.data_backend == 'pil':
        dataloader = accelerator.prepare(dataloader)
    cosine_value = lambda start,end, step, total_steps: end + 0.5 * (start-end) * (1 + np.cos(np.pi*step/total_steps))
    total_steps, warmup_steps = cfg.epochs * len(dataloader), cfg.warmup_epochs * len(dataloader)
    accelerator.print(f"Num proccess: {accelerator.num_processes}; Mixed precision: {accelerator.mixed_precision}")
    accelerator.print(f"Dataset size: {len(dataset)} images")

    # num_params = sum(p.numel() for p in model.parameters())
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Model {cfg.dino_type} has trainable {num_params/1e6:.6f} M parameters.")
    accelerator.print(
        f"Parameter dtypes -> model: {next(model.parameters()).dtype}"
    )

    # start training
    update_ema(ema, accelerator.unwrap_model(model), decay=0)  # Initialize EMA with the same weights as the model

    model.train() # important, train(). => class dropout.
    ema.eval()
    start_epoch,train_steps, log_steps, running_loss= 1,0,0,0,
    train_start_time = time()
    center = torch.zeros(1, 65536,device=accelerator.device,)
    for epoch in range(start_epoch, cfg.epochs+1):
        accelerator.print(f"[Exp Id: {cfg.timestamp}] | Start Epoch {epoch}/{cfg.epochs}...")
        tea_temp = np.interp(epoch-1, [0, cfg.warmup_teacher_temp_epochs -1], [cfg.warmup_teacher_temp, cfg.teacher_temp])
        for x, _ in dataloader: 

            if train_steps < warmup_steps:
                lr_now = lr * train_steps / (warmup_steps -1)
            else:
                lr_now = cosine_value(lr, cfg.min_lr, train_steps-warmup_steps, total_steps-warmup_steps)
            for group in optimizer.param_groups: 
                group['lr'] = lr_now
            optimizer.param_groups[0]['weight_decay'] = cosine_value(cfg.weight_decay, cfg.weight_decay_end, train_steps, total_steps)
            ema_decay = cosine_value(cfg.ema_decay, 1.0, train_steps, total_steps)
            global_views = torch.cat(x[:cfg.global_views],dim=0) #[2B, 3, 224, 224]
            local_views =  torch.cat(x[cfg.global_views:], dim=0) #[10B, 3, 96, 96]
            with accelerator.autocast():
                with torch.no_grad():
                    tea_logits = ema(global_views) #[2B, K]
                stu_global = model(global_views)
                stu_local = model(local_views)
            stu_logits = torch.cat([stu_global, stu_local],dim=0)  #[10B, K]

            # Loss Calculation
            loss = dino_loss(stu_logits, tea_logits, center, cfg, tea_temp)

            # update center
            with torch.no_grad():
                batch_center = accelerator.reduce(tea_logits.float().mean(dim=0, keepdim=True), reduction='mean')
                center = center* 0.9 + batch_center * 0.1

            optimizer.zero_grad()
            accelerator.backward(loss)
            if epoch <= cfg.freeze_last_layer:
                head = accelerator.unwrap_model(model).head
                head.prototypes.grad = None
                head.prototype_scale.grad = None
            optimizer.step()
            update_ema(ema, accelerator.unwrap_model(model), decay=ema_decay)
            train_steps += 1
            log_steps += 1
            # running_reg_loss += reg_loss.detach()
            running_loss += loss.detach()  # only compare the standard FM loss.
            if train_steps % cfg.log_step_freq == 0 :
                running_loss = accelerator.reduce(running_loss.detach(), reduction="mean")
                avg_loss = running_loss / log_steps
                elapsed = time() - train_start_time
                accelerator.print(
                    f"Step {train_steps}, Avg Loss: {avg_loss:.4f}," +
                    f"Elapsed: {elapsed:.1f}s",
                    
                    flush=True,
                )
                running_loss = 0
                log_steps = 0
    
            
        if epoch % cfg.save_epoch_freq == 0 or epoch == cfg.epochs:
            accelerator.wait_for_everyone()
            
            to_save = {
                "stu": accelerator.unwrap_model(model).state_dict(),
                "ema": ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "center": center,
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