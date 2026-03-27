"""
GCL-IoT: Reinforced Graph Contrastive Learning with Homophily-Heterophily Decoupling
Main model implementation combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from typing import Optional, Tuple, List

from .encoders import LowFrequencyEncoder, HighFrequencyEncoder
from .reinforcement import ReinforcedNeighborSelector
from .augmentation import FeatureSubstitutionAugmentation


class GCLIoT(nn.Module):
    """
    GCL-IoT: IoT Intrusion Detection Framework

    Components:
    - Reinforced Neighbor Selection (RL-based)
    - Dual-Frequency Encoders (Low-pass and High-pass)
    - Graph Augmentation via Feature Substitution
    - Contrastive Learning with Homophily-Heterophily Decoupling
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 out_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 temperature: float = 0.5,
                 alpha: float = 0.5,
                 top_k: int = 7,
                 device: str = 'cuda'):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            out_dim: Output embedding dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            temperature: Contrastive learning temperature (τ)
            alpha: High-pass filter intensity
            top_k: Number of top similar neighbors for contrastive learning
            device: Device to run on
        """
        super(GCLIoT, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.temperature = temperature
        self.alpha = alpha
        self.top_k = top_k
        self.device = device

        # Reinforced neighbor selection module
        self.neighbor_selector = ReinforcedNeighborSelector(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            device=device
        )

        # Dual-frequency encoders
        self.low_encoder = LowFrequencyEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        self.high_encoder = HighFrequencyEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            alpha=alpha
        )

        # Graph augmentation module (heuristic, counterfactual-inspired)
        self.augmentation = FeatureSubstitutionAugmentation(device=device)

        # Classification head (after concatenating embeddings)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary classification
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, GCNConv):
                nn.init.xavier_uniform_(module.lin.weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through GCL-IoT

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            z_homo: Homophily-focused embeddings [num_nodes, out_dim]
            z_heter: Heterophily-focused embeddings [num_nodes, out_dim]
            logits: Classification logits [num_nodes, 2]
        """
        # Get top-k neighbors from reinforced selection
        top_k_neighbors = self.neighbor_selector.get_top_k_neighbors(x, edge_index)

        # Generate augmented graph views (heuristic augmentation)
        x_aug, edge_index_aug_homo, edge_index_aug_heter = self.augmentation(
            x, edge_index, top_k_neighbors
        )

        # Dual-frequency encoding
        z_homo = self.low_encoder(x, edge_index_aug_homo)
        z_heter = self.high_encoder(x, edge_index_aug_heter)

        # Concatenate for classification
        z_concat = torch.cat([z_homo, z_heter], dim=1)
        logits = self.classifier(z_concat)

        return z_homo, z_heter, logits

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode nodes into concatenated embeddings"""
        z_homo, z_heter, _ = self.forward(x, edge_index)
        return torch.cat([z_homo, z_heter], dim=1)

    def compute_contrastive_loss(self,
                                 z_homo: torch.Tensor,
                                 z_heter: torch.Tensor,
                                 top_k_neighbors: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between homophily and heterophily encoders

        Args:
            z_homo: Homophily-focused embeddings [num_nodes, out_dim]
            z_heter: Heterophily-focused embeddings [num_nodes, out_dim]
            top_k_neighbors: Indices of top-k similar nodes [num_nodes, k]

        Returns:
            contrastive_loss: InfoNCE loss
        """
        num_nodes = z_homo.shape[0]

        # Normalize embeddings for cosine similarity
        z_homo_norm = F.normalize(z_homo, p=2, dim=1)
        z_heter_norm = F.normalize(z_heter, p=2, dim=1)

        # Positive pairs: same node across encoders
        pos_sim = (z_homo_norm * z_heter_norm).sum(dim=1)  # [num_nodes]

        # Negative pairs: different nodes
        # Compute similarity matrix
        sim_matrix = z_homo_norm @ z_heter_norm.T  # [num_nodes, num_nodes]

        # Mask out positive pairs
        mask = torch.eye(num_nodes, device=z_homo.device)
        sim_matrix = sim_matrix / self.temperature

        # InfoNCE loss
        exp_sim = torch.exp(sim_matrix)
        pos_exp = exp_sim.diag()

        # Sum over all negatives (including positives)
        sum_exp = exp_sim.sum(dim=1)

        loss = -torch.log(pos_exp / sum_exp)

        return loss.mean()

    def compute_classification_loss(self,
                                    logits: torch.Tensor,
                                    labels: torch.Tensor,
                                    train_mask: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for classification"""
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        return loss

    def get_top_k_neighbors(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get top-k similar neighbors for contrastive learning"""
        return self.neighbor_selector.get_top_k_neighbors(x, edge_index)