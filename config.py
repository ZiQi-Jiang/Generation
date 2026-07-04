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

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        self.close()

def set_seed(seed):
    """
    Sets the seed for various random number generators to ensure reproducibility.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class DefaultConfig:
    def __init__(self):
        # =================================================================
        # Basic Training Parameters
        # =================================================================
        self.batch_size = 64
        self.epochs = 80
        self.num_classes = 1000
        self.seed = 0
        self.model_type = 'SiT-S/2' 
        self.image_size = 256 
        self.vae = 'ema' # ['ema', 'mse'] 
        # saving and logging
        self.timestamp = datetime.datetime.now().strftime("%H%M%S") # 
        self.data_path = './data/imagenet/images/train'

        # =================================================================
        # Negative Flow Matching Parameters
        # =================================================================  
        self.flow_type = 'standard' #['standard', 'delta', 'neg'(ours)]
        self.use_repa = False # Ture: only use Dinov2-vit-b. 
        self.lam = 0.2 # weight used in deltaFM and NegFM.
        self.neg_type = 'res xt' # [ 'res xt', 'res x1', 'res vt'] --- only used when flow_type is 'neg'.
        self.neg_policy = 'random' # ['random', 'nearest', 'vote']

        
        # other helper parameters (only changed here)
        self.log_every = 100
        self.save_every = 50_000
        self.resume_exp = None # timestamp to resume training; None for training from scratch.
        self.resume_step = None # checkpoint step to load when resuming training.
        self.align_x0 = False # whether use the same x0(noise) for samples and corresponding negatives.


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
        
        parser.add_argument('--batch_size', type=int, help='Batch size for training')
        parser.add_argument('--epochs', type=int, help='Number of training epochs')
        parser.add_argument('--num_classes', type=int, help='Number of classes in the dataset')
        parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
        parser.add_argument('--model_type', type=str, help='Type of model to train')
        parser.add_argument('--image_size', type=int, help='Size of input images')
        parser.add_argument('--data_path', type=str, help='Path to the training data')
        parser.add_argument('--vae', type=str, help='VAE model to use (ema or mse)')
        parser.add_argument('--flow_type', type=str, help='Type of flow matching to use')
        parser.add_argument('--use_repa', action='store_true', help='Whether to use Dinov2-vit-b for representation')
        parser.add_argument('--lam', type=float, help='Weight for deltaFM and NegFM')
        parser.add_argument('--neg_policy', type=str, help='Policy for selecting negatives (random, nearest, vote)')
        parser.add_argument('--neg_type', type=str, help='Type of negative flow to use (only for NegFM)')
        parser.add_argument('--resume_exp', type=str, help='Timestamp of the experiment to resume')
        parser.add_argument('--resume_step', type=int, help='Checkpoint step to load')

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