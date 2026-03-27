"""
Data preprocessing for TON-IoT and IM-IoT datasets
Transforms raw network flows into graph format
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class IoTDataPreprocessor:
    """
    Preprocessor for IoT intrusion detection datasets
    
    Steps:
    1. Extract source/destination IP addresses as nodes
    2. Build edges based on communication flows
    3. Aggregate features per node
    4. Normalize features
    5. Create graph object with train/val/test splits
    """
    
    def __init__(self, 
                 feature_dim: int = 78,
                 normalize: bool = True,
                 temporal_split: bool = True,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2):
        """
        Args:
            feature_dim: Output feature dimension (after aggregation)
            normalize: Whether to apply z-score normalization
            temporal_split: Whether to split by time (True) or randomly (False)
            train_ratio: Training set proportion
            val_ratio: Validation set proportion
            test_ratio: Test set proportion
        """
        self.feature_dim = feature_dim
        self.normalize = normalize
        self.temporal_split = temporal_split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_ton_iot(self, data_path: str) -> pd.DataFrame:
        """
        Load TON-IoT dataset
        
        Expected columns:
        - src_ip: Source IP address
        - dst_ip: Destination IP address
        - timestamp: Flow timestamp
        - label: 0=benign, 1=malicious (or attack type)
        - [various flow features]
        """
        df = pd.read_csv(data_path)
        
        # Standard column mapping for TON-IoT
        column_mapping = {
            'src_ip': 'src_ip',
            'dst_ip': 'dst_ip',
            'ts': 'timestamp',
            'label': 'label'
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
    
    def load_im_iot(self, data_path: str) -> pd.DataFrame:
        """
        Load IM-IoT dataset (collected in-house)
        """
        df = pd.read_csv(data_path)
        return df
    
    def extract_node_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Extract features per node (IP address)
        
        Features include:
        - Statistical: avg packet size, flow duration, byte counts
        - Protocol distribution: TCP/UDP/ICMP ratios
        - Temporal: flow frequency, inter-arrival times
        - Connection behavior: degree, number of unique peers
        """
        # Group flows by source IP
        node_features = {}
        
        for ip in pd.concat([df['src_ip'], df['dst_ip']]).unique():
            # Get all flows involving this IP
            src_flows = df[df['src_ip'] == ip]
            dst_flows = df[df['dst_ip'] == ip]
            
            features = []
            
            # Statistical features
            if 'bytes' in df.columns:
                total_bytes = src_flows['bytes'].sum() + dst_flows['bytes'].sum()
                features.append(total_bytes)
                features.append(src_flows['bytes'].mean() if len(src_flows) > 0 else 0)
                features.append(dst_flows['bytes'].mean() if len(dst_flows) > 0 else 0)
            
            # Packet count features
            if 'packets' in df.columns:
                total_packets = src_flows['packets'].sum() + dst_flows['packets'].sum()
                features.append(total_packets)
                features.append(src_flows['packets'].mean() if len(src_flows) > 0 else 0)
            
            # Duration features
            if 'duration' in df.columns:
                avg_duration = src_flows['duration'].mean() if len(src_flows) > 0 else 0
                features.append(avg_duration)
            
            # Protocol distribution
            protocol_features = self._extract_protocol_features(df, ip)
            features.extend(protocol_features)
            
            # Connection behavior
            unique_peers = set(src_flows['dst_ip'].tolist() + dst_flows['src_ip'].tolist())
            features.append(len(unique_peers))
            features.append(len(src_flows) + len(dst_flows))  # total flows
            
            # Pad or truncate to feature_dim
            if len(features) < self.feature_dim:
                features.extend([0] * (self.feature_dim - len(features)))
            else:
                features = features[:self.feature_dim]
            
            node_features[ip] = np.array(features)
        
        # Convert to matrix
        ip_list = list(node_features.keys())
        feature_matrix = np.array([node_features[ip] for ip in ip_list])
        ip_to_idx = {ip: i for i, ip in enumerate(ip_list)}
        
        return feature_matrix, ip_to_idx
    
    def _extract_protocol_features(self, df: pd.DataFrame, ip: str) -> list:
        """Extract protocol distribution features"""
        features = []
        
        # Protocol counts
        protocol_counts = {}
        if 'proto' in df.columns:
            for proto in ['tcp', 'udp', 'icmp']:
                count = len(df[(df['src_ip'] == ip) & (df['proto'].str.lower() == proto)])
                count += len(df[(df['dst_ip'] == ip) & (df['proto'].str.lower() == proto)])
                features.append(count)
        
        # If no protocol info, return zeros
        while len(features) < 3:
            features.append(0)
        
        return features[:3]
    
    def build_edges(self, df: pd.DataFrame, ip_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Build edge list from communication flows
        Undirected edges, aggregated per unique IP pair
        """
        edges = set()
        
        for _, row in df.iterrows():
            src_ip = row['src_ip']
            dst_ip = row['dst_ip']
            
            if src_ip in ip_to_idx and dst_ip in ip_to_idx:
                src_idx = ip_to_idx[src_ip]
                dst_idx = ip_to_idx[dst_ip]
                
                if src_idx != dst_idx:
                    edges.add((src_idx, dst_idx))
                    edges.add((dst_idx, src_idx))  # Undirected
        
        edge_list = np.array(list(edges)).T
        return edge_list
    
    def create_labels(self, df: pd.DataFrame, ip_to_idx: Dict[str, int]) -> np.ndarray:
        """
        Create node labels (binary: 0=benign, 1=malicious)
        Aggregates multi-class attack labels to binary
        """
        num_nodes = len(ip_to_idx)
        labels = np.zeros(num_nodes, dtype=np.int64)
        
        # Determine malicious IPs
        malicious_ips = set()
        
        for _, row in df.iterrows():
            label = row.get('label', 0)
            # Convert multi-class to binary: any attack = malicious
            is_malicious = label != 0 and str(label).lower() not in ['benign', 'normal']
            
            if is_malicious:
                malicious_ips.add(row['src_ip'])
                malicious_ips.add(row['dst_ip'])
        
        for ip, idx in ip_to_idx.items():
            if ip in malicious_ips:
                labels[idx] = 1
        
        return labels
    
    def compute_heterophily_ratio(self, labels: np.ndarray, edge_index: np.ndarray) -> float:
        """Compute graph heterophily ratio (Eq. 3)"""
        num_heterophilic = 0
        
        for i in range(edge_index.shape[1]):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            if labels[src] != labels[dst]:
                num_heterophilic += 1
        
        return num_heterophilic / edge_index.shape[1]
    
    def create_splits(self, 
                      num_nodes: int, 
                      timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create train/validation/test masks
        
        Args:
            num_nodes: Total number of nodes
            timestamps: Optional timestamps for temporal split
            
        Returns:
            train_mask, val_mask, test_mask: Boolean arrays
        """
        if self.temporal_split and timestamps is not None:
            # Sort nodes by timestamp and split temporally
            sorted_indices = np.argsort(timestamps)
            num_train = int(num_nodes * self.train_ratio)
            num_val = int(num_nodes * self.val_ratio)
            num_test = num_nodes - num_train - num_val
            
            train_mask = np.zeros(num_nodes, dtype=bool)
            val_mask = np.zeros(num_nodes, dtype=bool)
            test_mask = np.zeros(num_nodes, dtype=bool)
            
            train_mask[sorted_indices[:num_train]] = True
            val_mask[sorted_indices[num_train:num_train + num_val]] = True
            test_mask[sorted_indices[num_train + num_val:]] = True
        else:
            # Random stratified split (by label)
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            
            num_train = int(num_nodes * self.train_ratio)
            num_val = int(num_nodes * self.val_ratio)
            
            train_mask = np.zeros(num_nodes, dtype=bool)
            val_mask = np.zeros(num_nodes, dtype=bool)
            test_mask = np.zeros(num_nodes, dtype=bool)
            
            train_mask[indices[:num_train]] = True
            val_mask[indices[num_train:num_train + num_val]] = True
            test_mask[indices[num_train + num_val:]] = True
        
        return train_mask, val_mask, test_mask
    
    def process(self, 
                data_path: str, 
                dataset_name: str = 'ton_iot') -> Data:
        """
        Main preprocessing pipeline
        
        Args:
            data_path: Path to raw data file
            dataset_name: 'ton_iot' or 'im_iot'
            
        Returns:
            torch_geometric.data.Data object with:
            - x: Node features
            - edge_index: Graph edges
            - y: Node labels
            - train_mask, val_mask, test_mask: Split masks
        """
        # Load data
        if dataset_name == 'ton_iot':
            df = self.load_ton_iot(data_path)
        elif dataset_name == 'im_iot':
            df = self.load_im_iot(data_path)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Extract node features
        x, ip_to_idx = self.extract_node_features(df)
        
        # Build edges
        edge_index = self.build_edges(df, ip_to_idx)
        
        # Create labels
        y = self.create_labels(df, ip_to_idx)
        
        # Normalize features
        if self.normalize:
            x = self.scaler.fit_transform(x)
        
        # Extract timestamps for splitting (if available)
        timestamps = None
        if 'timestamp' in df.columns:
            timestamps = np.zeros(len(ip_to_idx))
            # Aggregate timestamp per node (use first occurrence)
            for ip, idx in ip_to_idx.items():
                node_flows = df[(df['src_ip'] == ip) | (df['dst_ip'] == ip)]
                if len(node_flows) > 0:
                    timestamps[idx] = node_flows['timestamp'].min()
        
        # Create splits
        train_mask, val_mask, test_mask = self.create_splits(len(ip_to_idx), timestamps)
        
        # Convert to torch tensors
        x = torch.tensor(x, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
        train_mask = torch.tensor(train_mask, dtype=torch.bool)
        val_mask = torch.tensor(val_mask, dtype=torch.bool)
        test_mask = torch.tensor(test_mask, dtype=torch.bool)
        
        # Compute heterophily ratio for analysis
        heter_ratio = self.compute_heterophily_ratio(y.numpy(), edge_index.numpy())
        print(f"Dataset {dataset_name} heterophily ratio: {heter_ratio:.4f}")
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        
        return data