import datetime
import torch
import json
import os
import random
import sys
import numpy as np
import argparse


class SimpleLogger:
    """A simple logger that prints to both console and file."""
    def __init__(self, log_path='training.log'):
        self.terminal = sys.stdout
        self.log_file = open(log_path, "a")  # Use 'a' mode for appending

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return False
    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        self.close()

class DefaultConfig:
    def __init__(self):
        # =================================================================
        # Basic Training Parameters
        # =================================================================
        self.global_batch_size = 1024
        self.epochs = 40
        self.num_classes = 1000
        self.seed = 0
        self.dit_type = 'JiT-B/16' 
        self.image_size = 256 
        self.noise_scale = self.image_size / 256 * 1.0
        self.blr = 5e-4
        self.warmup_epochs = 10
        self.data_backend = 'pil' # pil / dali  
        # for logit-normal timestep sampling
        self.P_mean = -0.8
        self.P_std =  0.8
        self.ema_decay = 0.9999
        # saving and logging
        beijing_tz = datetime.timezone(datetime.timedelta(hours=8))  # UTC+8, Beijing time
        self.timestamp = datetime.datetime.now(beijing_tz).strftime("%Y%m%d_%H%M%S")
        self.latent_path = './data/imagenet/train_vae_latents_lmdb'
        self.data_path = './data/imagenet/images/train'
        self.discription= ''
        
        # other helper parameters (only changed here)
        self.log_step_freq = 100
        self.save_epoch_freq = 50
        self.use_latent = False # For SiT acceleration

        # our project
        self.t_level = 'image'
        self.reg_type = 'none' # 'repa' 'irepa', 'disperse', 'ig'
        self.reg_lam = 0.5
        self.reg_depth = 4
        self.adaln_post_modulation = False

        # for dino
        self.dino_type = 'vit_s_16'
        self.global_views = 2
        self.local_views = 10
        self.global_crops_scale =  (0.25, 1.0)
        self.local_crops_scale = (0.05, 0.25)
        self.warmup_teacher_temp = 0.04
        self.teacher_temp = 0.07
        self.warmup_teacher_temp_epochs = 30
        self.weight_decay = 0.04
        self.weight_decay_end = 0.4
        self.min_lr = 1e-5
        self.freeze_last_layer = 1
        # for sra
        self.t_max = 0.2
        self.reg_depth_tea = 8

        # for frequency
        # self.jacobian = 'repa' #['repa', 'freq']
        # self.freq_w_type = 'rgb' #['rgb', channel]

    def save(self, file_path=None):
        """Save configuration to JSON file"""
        if file_path is None:
            # Ensure save directory exists
            os.makedirs(self.results_dir, exist_ok=True)
            file_path = os.path.join(self.results_dir, 'config.json')
        else:
            os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)

        # Use __dict__ to get all instance attributes
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4, ensure_ascii=False)
        print(f"✅ Config saved to {file_path}")

    @classmethod
    def from_json(cls, file_path):
        """Load configuration from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            attrs = json.load(f)
        
        # Create a default instance
        config = cls()
        # Update instance with loaded attributes
        config.__dict__.update(attrs)
        
        print(f"✅ Config loaded from {file_path}")
        return config
    
    def parse_args(self, args=None):
        """Parse command line arguments and update configuration"""
        parser = argparse.ArgumentParser(description='Train model with configurable parameters')
        
        parser.add_argument('--global_batch_size', type=int, help='Batch size for training')
        parser.add_argument('--epochs', type=int, help='Number of training epochs')
        parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset')
        parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        parser.add_argument('--dit_type', type=str, help='Type of model to train')
        parser.add_argument('--image_size', type=int, help='Size of input images')
        parser.add_argument('--data_path', type=str, help='Path to the training data')
        parser.add_argument('--data_backend', type=str, choices=['pil', 'dali'])
        parser.add_argument('--vae', type=str, help='VAE model to use (ema or mse)')
        parser.add_argument('--use_latent', action='store_true')
        parser.add_argument('--timestamp', type=str)
        parser.add_argument('--t_level', type=str)
        parser.add_argument('--reg_depth', type=int)
        parser.add_argument('--reg_depth_tea', type=int)
        parser.add_argument('--reg_type', type=str, help='Regularization type for representation alignment.')
        parser.add_argument('--reg_lam', type=float, help='Weight for regularization loss.')
        parser.add_argument('--freq_lam', type=float, help='Weight for regularization loss.')
        parser.add_argument('--P_mean', type=float, help='Mean for logit-normal timestep sampling')
        parser.add_argument('--P_std', type=float, help='Standard deviation for logit-normal timestep sampling')
        parser.add_argument('--ema_decay', type=float)
        parser.add_argument('--blr', type=float, help='Base learning rate')
        parser.add_argument('--adaln_post_modulation', action='store_true' )

        parser.add_argument('--discription', type=str, help='Description of the experiment')
        parser.add_argument('--dino_type', type=str)
        parser.add_argument('--warmup_epochs', type=int)
        parser.add_argument('--local_views', type=int)
        parser.add_argument('--global_views', type=int)

        # for CoFlow
        parser.add_argument('--flow_type',type=str)
        #for sampling
        parsed_args = parser.parse_args(args=args)
        
        # Only update user-specified parameters (non-None parameters)
        for key, value in vars(parsed_args).items():
            if value is not None:
                setattr(self, key, value)
        return self
    
    def __str__(self):
        """
        Return a formatted string to print all class configurations.
        """
        settings = "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
        return f"===== Training Config Settings =====\n{settings}\n==========================="