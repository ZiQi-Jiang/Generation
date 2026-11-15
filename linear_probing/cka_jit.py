import argparse
from pathlib import Path
import torch
import torch.nn as nn
from time import time
from models.jit_clean import JiT_models
from accelerate import Accelerator, DataLoaderConfiguration
from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor, Normalize,RandomResizedCrop
import torch.utils.data as tud
from torchvision.datasets import ImageFolder
from models.utils import center_crop_arr
from accelerate.utils import set_seed
from pathlib import Path
from tqdm.auto import tqdm
from einops import rearrange
from config import DefaultConfig
from config import DefaultConfig, SimpleLogger
from models.dino_vit import ViT
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def compute_cka(A, B):
    # A: [N, D]; B: [N, D]
    A = A - A.mean(dim=0, keepdim=True) #[N, D]
    B = B - B.mean(dim=0, keepdim=True) #[N, D]

    numerator = torch.square(A.T @ B).sum(dtype=torch.float64)
    norm_a_sq = (A.T @ A).square().sum(dtype=torch.float64)
    norm_b_sq = (B.T @ B).square().sum(dtype=torch.float64)
    return (numerator / (norm_a_sq * norm_b_sq).sqrt()).item()

def compute_cknna(A, B, k=10):
    def neighbors(X,k):
        X = X.float()
        K = X @ X.T #[N,N]
        mean = K.mean(dim=1, keepdim=True)
        K.fill_diagonal_(-torch.inf)
        values, indices = K.topk(k, dim=1)
        return indices, values - mean

    idx_a, val_a = neighbors(A,k)
    idx_b, val_b = neighbors(B,k)
    common = idx_a[:, :, None] == idx_b[:, None, :]

    numerator = (common * val_a[:, :, None] * val_b[:, None, :]).sum(
        dtype=torch.float64
    )
    norm_a_sq = val_a.square().sum(dtype=torch.float64)
    norm_b_sq = val_b.square().sum(dtype=torch.float64)
    return (numerator / (norm_a_sq * norm_b_sq).sqrt()).item()

def main(cfg):
     # set up config
    dataloader_config = DataLoaderConfiguration(non_blocking=True)
    accelerator = Accelerator(mixed_precision="no", dataloader_config=dataloader_config)
    PROJECT_DIR = cfg.project_dir
    # if cfg.exp and accelerator.is_main_process:
        # logger = SimpleLogger(f'{PROJECT_DIR}/exp/{cfg.exp}/knn.jsonl')
        # sys.stdout, sys.stderr = logger, logger
    set_seed(0, device_specific=True)

  
    val_transform = Compose([
        Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
        PILToTensor(), 
    ])
    val_dataset = ImageFolder(f'{cfg.data_path}/val', transform=val_transform)
    generator = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(val_dataset), generator=generator)[:10_000]
    val_dataset = tud.Subset(val_dataset, indices.tolist())
    # TODO:
    val_loader = tud.DataLoader(val_dataset, batch_size=cfg.local_batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
    
    dino_model = ViT(img_size=224, patch_size=16, num_classes=0, dim=384, depth=12, num_heads=6, mlp_ratio=4, drop_path_rate=0.1, qkv_bias=True, qk_norm=False)
    dino_model.head = nn.Identity()
    state_dict = torch.load(f'{cfg.project_dir}/data/official_ckpts/dino_vits16.pt', map_location='cpu',weights_only=True)
    weights = {key.replace('patch_embed.proj', 'patch_emb.conv').replace('mlp.', 'ffn.'): value for key,value in state_dict.items()}
    dino_model.load_state_dict(weights)
    dino_model = dino_model.eval().requires_grad_(False).to(accelerator.device)

    model = JiT_models[cfg.dit_type](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
    if cfg.exp:
        state_dict = torch.load(cfg.checkpoint_path, map_location='cpu',weights_only=False)
        weights = state_dict['jit_ema'] if 'jit_ema' in state_dict else state_dict['ema']
        cfg.epoch = state_dict['epoch']
    else:
        state_dict = torch.load(f'{PROJECT_DIR}/data/official_ckpts/jit_h_16.pt', map_location='cpu',weights_only=False)
        weights = state_dict['model_ema1']
        weights = {key.removeprefix('net.'): value for key,value in weights.items() }
    

    model.load_state_dict(weights, strict=True)
    model  = model.eval().requires_grad_(False).to(accelerator.device)
    blocks = model.blocks
    batch_features = {}
    features_by_layer = [[] for _ in blocks]
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            tokens = output
            if cfg.dit_type in JiT_models and layer_idx >= model.in_context_start:
                tokens = output[:, model.in_context_len:, :]
            tokens = tokens.float().mean(dim=1)  # [B, N, D] -> [B, D]
            batch_features[layer_idx]=tokens # [B, D] -> [B, D]
        return hook_fn

    for layer_idx, block in enumerate(model.blocks):
        model.blocks[layer_idx].register_forward_hook(make_hook(layer_idx))
        
    val_loader = accelerator.prepare( val_loader)
    # for x1, labels in train_loader:
    all_dino_features = []
    for x1, labels in tqdm(val_loader, desc='extract val features', disable = not accelerator.is_main_process):
        with torch.no_grad():
            x1 = x1.to(torch.float32).div_(255)
            dino_x = Normalize( (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(x1)
            dino_features = dino_model(dino_x)
            gathered_dino_features = accelerator.gather_for_metrics(dino_features)

            x1 = 2.0 * x1 - 1.0  
            b = x1.shape[0]
            noise = torch.randn_like(x1) 
            t = torch.full((b, ), cfg.t, dtype=torch.float32, device=accelerator.device)
            y = torch.full((b, ), cfg.num_classes, dtype=torch.long, device=accelerator.device)
            tb = t[:, None, None,None] #[B,1,1,1]
            xt = (1-tb) * noise + tb * x1
            batch_features.clear()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _ = model(xt, t, y)
            gathered = accelerator.gather_for_metrics(tuple(batch_features[i] for i in range(len(blocks))))
  
            if accelerator.is_main_process:
                all_dino_features.append(gathered_dino_features)
                for chunks, features in zip(features_by_layer, gathered):
                    chunks.append(features.cpu())

    # inner layer CKA
    if accelerator.is_main_process:
        all_dino_features = torch.cat(all_dino_features,dim=0) 
        score = compute_cka if cfg.cka_type == 'cka' else compute_cknna
        features = [torch.cat(chunks, dim=0) for chunks in features_by_layer] #[[B,D], [B,D],..]
        n_layers = len(features)
        matrix = torch.zeros(n_layers,n_layers)
        dino_matrix = torch.zeros(1,n_layers)
        for i in range(n_layers):
            for j in range(i, n_layers):
                matrix[i,j] = matrix[j,i] = score(features[i], features[j])
            dino_matrix[0,i] = score(features[i], all_dino_features)
        
        results_path = f'{PROJECT_DIR}/exp/{cfg.exp}/{cfg.cka_type}_t{cfg.t}_e{cfg.epoch}.png' if cfg.exp else f'official_{cfg.cka_type}_{cfg.t:g}.png'

        #plot
        plot_cka(matrix, cfg, results_path)

        #TODO, DINO

    # dit with dion CKA


def plot_cka(matrix,cfg, png_path):
    fig, ax = plt.subplots(figsize=(5, 4))
    vmin = -1 if cfg.cka_type == "cknna" else 0
    im = ax.imshow(matrix.numpy(), origin="lower", cmap="viridis", vmin=vmin, vmax=1)
    ax.set(xlabel="Layer j", ylabel="Layer i", title=f"{cfg.dit_type} | t={cfg.t:g}")
    fig.colorbar(im, ax=ax, label=cfg.cka_type.upper())
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Linear Probing Training.")
    parser.add_argument('--dit_type', type=str, default=None )
    parser.add_argument('--t', type=float, default=None)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--data_path',type=str, default=str(Path(__file__).resolve().parents[1]/ 'data/imagenet/images'), )
    parser.add_argument('--cka_type',type=str, default=None)


    cfg = parser.parse_args()
    cfg.project_dir = Path(__file__).resolve().parents[1]
    cfg.image_size =  256
    cfg.num_classes = 1000
    cfg.local_batch_size = 128

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

