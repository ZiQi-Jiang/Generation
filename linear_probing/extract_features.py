# import torch.nn as nn
# import torch
# from models.jit_clean import JiT_models
# import argparse
# from accelerate import Accelerator, DataLoaderConfiguration
# from torchvision.transforms import Compose, Lambda, RandomHorizontalFlip, ToTensor, PILToTensor, Normalize
# import torch.utils.data as tud
# from torchvision.datasets import ImageFolder
# from models.utils import center_crop_arr
# class FeatureExtractor(nn.Module):
#     def __init__(self, model, layer_idx):
#         super().__init__()
#         self.model = model
#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#         self.layer_idx = layer_idx
#         self.features = None # store features

#         self.model.blocks[layer_idx].register_forward_hook(self._hook_fn)


#     def _hook_fn(self, module, input, output): # do sth after module's forward
#         self.features = output


    


# def main(cfg):

#     # set up config
#     dataloader_config = DataLoaderConfiguration(non_blocking=True)
#     accelerator = Accelerator(mixed_precision="no", dataloader_config=dataloader_config)

#     transform = Compose([
#         Lambda(lambda pil_image: center_crop_arr(pil_image, cfg.image_size)),
#         RandomHorizontalFlip(),
#         # ToTensor(), 
#         PILToTensor(), # Faster than ToTensor
#         # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
#     ])
#     train_dataset = ImageFolder(f'{cfg.data_path}/train', transform=transform)
#     val_dataset = ImageFolder(f'{cfg.data_path}/val', transform=transform)

#     # TODO:
#     train_dataloader = tud.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)
#     val_dataloader = tud.DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

#     # register hook
#     hook_cache = {}
#     def hook_fn(module, input, output):
#         hook_cache['feat'] = output
#     # 
#     if  cfg.model_type in JiT_models.keys(): 
#         model = JiT_models['JiT-H/16'](input_size=cfg.image_size, in_channels=3, num_classes=cfg.num_classes, mlp_ratio=4)
#         state_dict = torch.load(f'official_ckpts/jit_h_16.pt', map_location='cpu')
#         weights = state_dict['model_ema1']
#         weights = {key.removeprefix('net.'): value for key,value in weights.item() }
#         model.load_state_dict(weights, strict=True)
#         model.eval()

#         model.blocks[cfg.layer_idx].register_forward_hook(hook_fn)


#     train_dataloader, val_dataloader, model = accelerator.prepare(train_dataloader, val_dataloader, model)

#     train_features, val_features = [], []
#     train_labels, val_labels = [], []
    
#     for raw_x, raw_y in train_dataloader:
#         with accelerator.auto_cast():
#             with torch.no_grad():
#                 x = raw_x.to(accelerator.device).float()  / 127.5 - 1.0
#                 t = torch.full((cfg.batch_size, ), cfg.t, dtype=torch.flaot32, device=accelerator.device)
#                 y = torch.full((cfg.batch_size), cfg.num_classes, dtype=torch.long, device=accelerator.device)

#                 noise = torch.randn_like(x) 
#                 tb = t[:, None, None,None] #[B,1,1,1]
#                 xt = (1-tb) * noise + tb * x


#                 _ = model(xt, t, y)

#                 tokens = hook_cache['feat'].mean(dim=1) #[B, N, D] -> [B, D]
#                 if cfg.layer_idx >= model.in_context_start:
#                     tokens = tokens[:, model.in_context_len:, :]
#                     feats = tokens.float().mean(dim=1)
#                 train_features.append(feats.detach().cpu())
#                 train_labels.append(raw_y.cpu())

#     for raw_x, raw_y in val_dataloader:
#         with accelerator.autocast():
#             with torch.no_grad():
#                 x = raw_x.to(accelerator.device).float()  / 127.5 - 1.0
#                 t = torch.full((cfg.batch_size, ), cfg.t, dtype=torch.flaot32, device=accelerator.device)
#                 y = torch.full((cfg.batch_size), cfg.num_classes, dtype=torch.long, device=accelerator.device)

#                 noise = torch.randn_like(x) 
#                 tb = t[:, None, None,None] #[B,1,1,1]
#                 xt = (1-tb) * noise + tb * x


#                 _ = model(xt, t, y)

#                 tokens = hook_cache['feat'].mean(dim=1) #[B, N, D] -> [B, D]
#                 if cfg.layer_idx >= model.in_context_start:
#                     tokens = tokens[:, model.in_context_len:, :]
#                     feats = tokens.float().mean(dim=1)
#                 val_features.append(feats.detach().cpu())
#                 val_labels.append(raw_y.cpu())

#     train_features = torch.cat(train_features, dim=0)
#     train_labels = torch.cat(train_labels, dim=0)
#     val_features = torch.cat(val_features, dim=0)
#     val_labels = torch.cat(val_labels, dim=0)


#     accelerator.print(f'Extraction Done! : Features:{train_features.shape}, Labels:{train_labels.shape}')
#     save_states = {'train_features': train_features, 
#                    "train_labels": train_labels, 
#                    "val_features": val_features,
#                    "val_labels": val_labels,
#                    'model_type':cfg.model_type, 
#                    't':cfg.t, 
#                    'layer_index': cfg.layer_index}
#     torch.save(save_states, f'{cfg.model_type}_t{cfg.t}_l{cfg.layer_index}.pt')




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Sample from the trained model")
#     parser.add_argument('--model_type', type=str, default=None )
#     parser.add_argument('--t', type=float, default=None)
#     parser.add_argument('--layer_idx', type=float, default=None)
#     cfg = parser.parse_args()
#     cfg.batch_size = 1024
#     cfg.data_path = './data/imagenet/images'
#     cfg.image_size =  256
#     cfg.num_classes = 1000
#     main(cfg)





