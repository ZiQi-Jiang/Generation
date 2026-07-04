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
        self.batch_size = 128
        self.epochs = 200
        self.num_classes = 1000
        self.seed = 0
        self.model_type = 'JiT-B/16' 
        self.image_size = 256 
        self.noise_scale = self.image_size / 256 * 1.0
        self.blr = 5e-5
        self.warmup_epochs = 5

        # for logit-normal timestep sampling
        self.P_mean = -0.8
        self.P_std =  0.8

        # saving and logging
        self.timestamp = datetime.datetime.now().strftime("%H%M%S") # 
        self.data_path = './data/imagenet/images/train'

   

        
        # other helper parameters (only changed here)
        self.log_step_freq = 20
        self.save_epoch_freq = 50
        self.resume_exp = None # timestamp to resume training; None for training from scratch.
        self.resume_epoch = None # checkpoint step to load when resuming training.


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