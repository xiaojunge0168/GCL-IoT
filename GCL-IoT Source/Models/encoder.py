"""
Dual-Frequency Encoders for Homophily and Heterophily Pattern Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops, degree
import numpy as np


class LowFrequencyEncoder(nn.Module):
    """
    Low-frequency encoder (GCN) for capturing homophily patterns
    Acts as a low-pass filter, smoothing node representations
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5):
        super(LowFrequencyEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # GCN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input layer
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through low-frequency encoder

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            z: Node embeddings [num_nodes, out_dim]
        """
        # Add self-loops for GCN
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # GCN layers
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer (no activation)
        x = self.convs[self.num_layers - 1](x, edge_index)

        return x


class HighFrequencyEncoder(nn.Module):
    """
    High-frequency encoder for capturing heterophily patterns
    Implements high-pass filter: (I - αA) * X

    This encoder amplifies differences between nodes and their neighbors,
    making it effective for detecting heterophilic connections.
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_layers: int = 3,
                 dropout: float = 0.5,
                 alpha: float = 0.5):
        super(HighFrequencyEncoder, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout
        self.alpha = alpha

        # Linear projection layers (since we do custom aggregation)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # Input projection
        self.layers.append(nn.Linear(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Output layer
        self.layers.append(nn.Linear(hidden_dim, out_dim))

    def _compute_high_pass_aggregation(self,
                                       x: torch.Tensor,
                                       edge_index: torch.Tensor) -> torch.Tensor:
        """
        Compute high-pass filtered features: (I - αA_norm) * X

        Args:
            x: Node features [num_nodes, feat_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            high_pass: High-pass filtered features [num_nodes, feat_dim]
        """
        num_nodes = x.size(0)
        device = x.device

        # Build normalized adjacency matrix
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)

        # Compute normalized adjacency values
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Sparse matrix multiplication for neighbor aggregation
        # First, aggregate neighbor features: A_norm * X
        neighbor_agg = torch.zeros_like(x)
        for i in range(num_nodes):
            # Get neighbors of node i
            mask = (row == i)
            neighbor_indices = col[mask]
            neighbor_weights = norm[mask]

            if len(neighbor_indices) > 0:
                neighbor_agg[i] = (x[neighbor_indices] * neighbor_weights.unsqueeze(1)).sum(dim=0)

        # High-pass filter: x - alpha * A_norm * x
        high_pass = x - self.alpha * neighbor_agg

        return high_pass

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through high-frequency encoder

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edges [2, num_edges]

        Returns:
            z: Node embeddings [num_nodes, out_dim]
        """
        # Apply high-pass filtering at each layer
        h = x

        for i in range(self.num_layers):
            # High-pass filtering
            h = self._compute_high_pass_aggregation(h, edge_index)

            # Linear projection
            h = self.layers[i](h)

            if i < self.num_layers - 1:
                h = self.bns[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h