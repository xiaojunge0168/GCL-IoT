"""
Helper functions for training and evaluation
"""

import torch
import numpy as np
import random
import os
from typing import Optional, Tuple, Dict, Any
import json
import yaml


def set_seed(seed: int = 42, deterministic: bool = False):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed
        deterministic: Whether to enable CUDA deterministic mode
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(model, optimizer, epoch, metrics, save_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, checkpoint_path: str):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    return model, optimizer, epoch, metrics


def compute_heterophily(labels: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Compute graph heterophily ratio (Eq. 3)"""
    num_heterophilic = 0
    for i in range(edge_index.size(1)):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        if labels[src] != labels[dst]:
            num_heterophilic += 1
    return num_heterophilic / edge_index.size(1)


def compute_node_heterophily(labels: torch.Tensor, 
                              neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Compute per-node heterophily (Eq. 2)"""
    heter = torch.zeros(labels.size(0))
    for i, neighbors in enumerate(neighbor_indices):
        if neighbors[0] >= 0:  # Valid neighbor
            valid_neighbors = neighbors[neighbors >= 0]
            if len(valid_neighbors) > 0:
                diff = (labels[valid_neighbors] != labels[i]).float()
                heter[i] = diff.mean()
    return heter


def moving_average(values: list, window_size: int = 10) -> list:
    """Compute moving average for smoothing curves"""
    if len(values) < window_size:
        return values
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid').tolist()


class EarlyStopping:
    """Early stopping callback"""
    
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
            return False
        
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False


class AverageMeter:
    """Compute and store average and current values"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count