# used to sample images for metrics.
from config import DefaultConfig,  SimpleLogger
# from models.pit import PiT_models
from models.official_pit import PiT_models
import os, sys, argparse,datetime
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, broadcast_object_list
from einops import repeat
from time import time
from torchvision.utils import save_image
from tqdm import tqdm
import math
import numpy as np
import torch.distributed as dist
from PIL import Image
import torch_fidelity
from einops import repeat
from evaluator import Evaluator, evalutate
import cv2

@torch.no_grad()
def velocity_with_cfg(model, xt, y, t, sample_args, num_classes=1000):
    # xt: (B, 3, H, W)
    # y: (B,), xt: (B, 1, 1, 1)
    # conditional
    v_cond = model(xt, t.flatten(), y)
    #unconditionl
    v_uncond = model(xt, t.flatten(), torch.full_like(y, num_classes))
    low, high = sample_args.cfg_min, sample_args.cfg_max
    interval_mask = (t < high) & ((low == 0) | (t > low))
    cfg_scale_interval = torch.where(interval_mask, sample_args.cfg_scale, 1.0)
    v = v_uncond + cfg_scale_interval*(v_cond - v_uncond)
    return v

@torch.no_grad()
def sample(model,  accelerator, sample_dir, sample_args,train_cfg):
    samples_per_gpu = math.ceil(sample_args.num_samples / accelerator.num_processes)
    accelerator.print(f"Generating  {sample_args.num_samples} samples in total \n{accelerator.num_processes} processes,{samples_per_gpu} samples per GPU.")
    accelerator.print(f"Mixed precision: {accelerator.mixed_precision}")
    start_idx = accelerator.process_index * samples_per_gpu
    num_class = train_cfg.num_classes
    # JiT style
    class_labels_global = np.arange(0, num_class).repeat(sample_args.num_samples // num_class) #(50000,)
    class_labels_global = np.hstack([class_labels_global, np.zeros(50000)]) # assure the last step 
    num_steps = sample_args.num_samples // (sample_args.batch_size * accelerator.num_processes) + 1
    for i in range(num_steps + 1):
        accelerator.print(f'Generatino Step: {i}/{num_steps}')
        start_idx = accelerator.num_processes * sample_args.batch_size * i + sample_args.batch_size * accelerator.process_index
        end_idx = start_idx + sample_args.batch_size

        # with accelerator.autocast():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            start_time = time()
            # dit style
            # y = torch.randint(0, train_cfg.num_classes, (sample_args.batch_size,),device=accelerator.device)

            # jit style
            y = class_labels_global[start_idx:end_idx]
            y = torch.Tensor(y).long().cuda()
            z = torch.randn(sample_args.batch_size, 3, train_cfg.image_size, train_cfg.image_size, device=accelerator.device) * train_cfg.noise_scale
            t = torch.linspace(0,1.0, sample_args.sample_steps+1, device=accelerator.device) # ode: 0 to 1, no last step
            t = repeat(t, 'n -> n b 1 1 1', b = sample_args.batch_size)
            x = z # init
            v_prev = None
            for idx in tqdm(range(sample_args.sample_steps)):
                t_cur, t_next = t[idx], t[idx+1] # (b,1,1,1)
                v = velocity_with_cfg(accelerator.unwrap_model(model), x, y, t_cur, sample_args)

                if sample_args.solver == 'FlowDPM':
                    if v_prev is None:
                        x = x + (t_next - t_cur) * v
                    else:
                        x = x + (t_next - t_cur) * (1.5 * v - 0.5 * v_prev)
                    v_prev = v
                elif sample_args.solver == 'Euler':
                    x = x + (t_next - t_cur) * v
                elif sample_args.solver == 'Heun':
                    if t_next == 1:
                        x = x + (t_next - t_cur) * v
                    else:
                        x_next_euler = x + v * (t_next - t_cur)
                        v_next = velocity_with_cfg(model, x_next_euler, y, t_next, sample_args)
                        x = x + (t_next - t_cur) * 0.5*(v_next + v)
                
        samples = (x + 1)/2
        samples = samples.detach().cpu()

        # store image
        for b_id in range(samples.size(0)):
            global_id = i * sample_args.batch_size * accelerator.num_processes + accelerator.process_index * sample_args.batch_size + b_id
            if global_id >= sample_args.num_samples:
                break
            img = np.round(np.clip(samples[b_id].numpy().transpose([1,2,0])*255, 0, 255)) # tensor-> numpy -> (H,W,C) -> [0,255]
            img = img.astype(np.uint8)[:,:,::-1] # rgb -> bgr for opencv
            cv2.imwrite(f"{sample_dir}/images/{global_id:06d}.png", img)
        accelerator.print(f"[Path:{sample_dir}] | Generated {end_idx}/{sample_args.num_samples} samples so far.\n Sampling took {time() - start_time:.2f} seconds for the current batch.")

    accelerator.wait_for_everyone()
    accelerator.print(f"Finished generating {samples_per_gpu* accelerator.num_processes} samples. Saved to {sample_dir}/")
    # if accelerator.is_main_process:
    #     create_npz_from_sample_folder(sample_dir, num=sample_args.num_samples)
    # accelerator.print(f"Finished creating .npz file from samples")

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/images/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}/generated_samples.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(sample_args):
    accelerator = Accelerator(mixed_precision="bf16")
    time_list = [datetime.datetime.now().strftime("%H%M%S")]
    broadcast_object_list(time_list, from_process=0)
    
    sample_dir = os.path.join('samples', time_list[0])
    if accelerator.is_main_process:
        os.makedirs(sample_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_dir, "images"), exist_ok=True)
    sys.stdout = SimpleLogger(os.path.join(sample_dir, "log.txt"))

    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.optimize_ddp = False
    torch.set_float32_matmul_precision('high') # important for accleration

    train_cfg = DefaultConfig()
    if sample_args.exp : # load by experiments
        accelerator.print(f"Loading from experiment with timestamp: {sample_args.exp}, epoch: {sample_args.epoch:03d}, num_classes:{train_cfg.num_classes}")
        cfg_path = os.path.join('exp',sample_args.exp, 'config.json')
        train_cfg = DefaultConfig.from_json(cfg_path)
        ckpt_path = os.path.join('exp',sample_args.exp, f'epoch_{sample_args.epoch:03d}.pth')
        state_dict = torch.load(ckpt_path, map_location='cpu')
    else: # load by hands.
        accelerator.print("Using default configuration for SiT-XL/2 trained on 1000 classes.")
        train_cfg.model_type = 'JiT-B/16'
        train_cfg.image_size = 256
        train_cfg.num_classes = 1000
        train_cfg.mlp_ratio = 4.0
        train_cfg.in_channels = 3
        ckpt_path = './data/checkpoints/jit_b16_e600.pth'
        state_dict = torch.load(ckpt_path, map_location='cpu')
    # print train cfg and sampling sample_args
    accelerator.print(train_cfg)
    sample_args_str = "\n".join([f"{key}: {value}" for key, value in vars(sample_args).items()])
    accelerator.print(f"===== Sampling Settings =====\n{sample_args_str}\n===========================")
    
    # load model and checkpoint
    model = PiT_models[train_cfg.model_type](input_size=train_cfg.image_size,num_classes=train_cfg.num_classes, in_channels=3, mlp_ratio=4, reg_type = train_cfg.reg_type, reg_depth = train_cfg.reg_depth, pit_adaln_post_modulation=train_cfg.adaln_post_modulation)
    # model = JiT_models[train_cfg.model_type](input_size=train_cfg.image_size, in_channels=3, num_classes=train_cfg.num_classes, mlp_ratio=4)

    model.load_state_dict(state_dict['ema'])
    # new_state_dict = {}
    # for k, v in state_dict['model_ema1'].items():
    #     if k.startswith('net.'):
    #         name = k[4:]
    #     else:
    #         name = k
    #     new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    model.eval()
    model = accelerator.prepare(model)

    # start sampling sample_args.num_samples samples and save to sample_dir.
    set_seed(sample_args.sample_seed, device_specific=True)
    sample(model, accelerator, sample_dir, sample_args, train_cfg)

    # evaluate FID,IS, sFID, Precision/Recall. 
    if train_cfg.image_size == 256:
        fid_statistics_file = './data/jit_in256_stats.npz'
    elif train_cfg.image_size == 512:
        fid_statistics_file = './data/jit_in512_stats.npz' 
    else:
            raise NotImplementedError

    if accelerator.is_main_process:
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=f'{sample_dir}/images',
            input2=None,
            fid_statistics_file=fid_statistics_file,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        accelerator.print(f"FID: {metrics_dict['frechet_inception_distance']:.4f}, Inception Score: {metrics_dict['inception_score_mean']:.4f}")
        # evaluator = Evaluator(accelerator.device)
        # evalutate(evaluator = evaluator, ref_batch_path='./data/VIRTUAL_imagenet256_labeled.npz', sample_batch_path=f'{sample_dir}/generated_samples.npz')

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from the trained model")
    parser.add_argument('--exp', type=str, default=None, help='Timestamp of the experiment to load (default, using JiT-B/16 trained on 1000 classes)')
    parser.add_argument('--epoch', type=int, default=200, help='Checkpoint step to load')
    parser.add_argument('--sample_seed', type=int, default=0, help='Random seed for sampling')
    parser.add_argument('--cfg_scale', type=float, default=1.0,  help='Classifier-free guidance scale')
    parser.add_argument('--cfg_min', type=float, default=0.1, help="CFG interval min")
    parser.add_argument('--cfg_max', type = float, default=1.0, help = " CFG interval max")
    parser.add_argument('--t_eps', type = float, default=0.05)
    parser.add_argument('--sample_steps', type=int, default=50, help='Number of sampling steps')
    # parser.add_argument('--last_step',type=float, default=0.04, help='last sample stepsize, only used in SDE EM sampler.')
    parser.add_argument('--num_samples',type=int, default=50_000, help='generate for metrics; None to use the default 8 samples.')
    parser.add_argument('--batch_size',type=int, default=256)
    parser.add_argument('--solver', type=str, default='Euler')

    sample_args = parser.parse_args()
    main(sample_args)

# python sample.py --exp 015900 --ckpt_step 30000 --cfg_scale 1.5
# accelerate launch sample.py --exp 015900 --ckpt_step 30000 --cfg_scale 1.5 --num_samples 50000 --batch_size 16