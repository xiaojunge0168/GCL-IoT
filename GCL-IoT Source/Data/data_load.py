"""
Dataset utilities for loading and managing IoT intrusion detection datasets
"""

import os
import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from typing import Optional, List, Tuple
import pickle


class IoTGraphDataset(Dataset):
    """
    PyTorch Geometric Dataset for IoT intrusion detection
    
    Handles loading, caching, and splitting of graph data
    """
    
    def __init__(self, 
                 root: str,
                 dataset_name: str,
                 preprocessor=None,
                 transform=None,
                 pre_transform=None):
        """
        Args:
            root: Root directory for data storage
            dataset_name: Name of dataset ('ton_iot' or 'im_iot')
            preprocessor: Preprocessor instance
            transform: Optional transform
            pre_transform: Optional pre-transform
        """
        self.dataset_name = dataset_name
        self.preprocessor = preprocessor
        self.root = root
        super(IoTGraphDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self) -> List[str]:
        """Raw data file names"""
        if self.dataset_name == 'ton_iot':
            return ['ton_iot.csv']
        elif self.dataset_name == 'im_iot':
            return ['im_iot.csv']
        else:
            return []
    
    @property
    def processed_file_names(self) -> List[str]:
        """Processed data file names"""
        return ['data.pt']
    
    def download(self):
        """Download dataset (placeholder - actual download handled separately)"""
        pass
    
    def process(self):
        """Process raw data into graph format"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not provided")
        
        # Process data
        data_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        data = self.preprocessor.process(data_path, self.dataset_name)
        
        # Save processed data
        torch.save(data, os.path.join(self.processed_dir, 'data.pt'))
        
    def len(self) -> int:
        return 1
    
    def get(self, idx: int) -> Data:
        data = torch.load(os.path.join(self.processed_dir, 'data.pt'))
        return data


def load_iot_data(dataset_name: str, 
                  data_path: str,
                  feature_dim: int = 78,
                  temporal_split: bool = True) -> Data:
    """
    Convenience function to load IoT intrusion detection data
    
    Args:
        dataset_name: 'ton_iot' or 'im_iot'
        data_path: Path to raw data file
        feature_dim: Feature dimension
        temporal_split: Whether to use temporal splitting
        
    Returns:
        PyG Data object
    """
    from .preprocessing import IoTDataPreprocessor
    
    preprocessor = IoTDataPreprocessor(
        feature_dim=feature_dim,
        temporal_split=temporal_split
    )
    
    data = preprocessor.process(data_path, dataset_name)
    
    return data


def get_dataset_statistics(data: Data) -> dict:
    """
    Compute dataset statistics
    
    Returns:
        dict with: num_nodes, num_edges, num_features, 
                   num_classes, class_distribution, heterophily_ratio
    """
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    num_features = data.x.size(1)
    num_classes = len(torch.unique(data.y))
    
    # Class distribution
    benign_count = (data.y == 0).sum().item()
    malicious_count = (data.y == 1).sum().item()
    total = num_nodes
    
    # Heterophily ratio
    heter_count = 0
    for i in range(num_edges):
        src = data.edge_index[0, i]
        dst = data.edge_index[1, i]
        if data.y[src] != data.y[dst]:
            heter_count += 1
    heter_ratio = heter_count / num_edges
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'num_features': num_features,
        'num_classes': num_classes,
        'benign_count': benign_count,
        'malicious_count': malicious_count,
        'class_imbalance_ratio': benign_count / max(malicious_count, 1),
        'heterophily_ratio': heter_ratio,
        'train_size': data.train_mask.sum().item(),
        'val_size': data.val_mask.sum().item(),
        'test_size': data.test_mask.sum().item()
    }