"""
Reinforced Neighbor Selection using Proximal Policy Optimization (PPO)
Identifies top-k optimal neighbors for each node to mitigate class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class PolicyNetwork(nn.Module):
    """PPO Policy Network for neighbor selection"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return action logits and value estimate"""
        h = F.relu(self.fc1(state))
        h = F.relu(self.fc2(h))

        logits = self.actor(h)
        value = self.critic(h)

        return logits, value


class ReinforcedNeighborSelector:
    """
    Reinforcement Learning-based neighbor selection
    Identifies top-k similar neighbors for central nodes

    Uses Coordinated Proximal Policy Optimization (CoPPO) to learn
    optimal selection policy
    """

    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 max_k: int = 15,
                 initial_k: int = 3,
                 lr: float = 1e-4,
                 gamma: float = 0.99,
                 clip_epsilon: float = 0.2,
                 device: str = 'cuda'):
        """
        Args:
            in_dim: Input feature dimension
            hidden_dim: Hidden dimension for policy network
            max_k: Maximum number of neighbors to select
            initial_k: Initial k value
            lr: Learning rate for policy network
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            device: Device to run on
        """
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.max_k = max_k
        self.initial_k = initial_k
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.device = device

        # State dimension: node feature + neighbor features + historical selections
        self.state_dim = in_dim + in_dim + max_k * in_dim  # Simplified
        self.action_dim = max_k  # One action per candidate neighbor

        self.policy = PolicyNetwork(self.state_dim, self.action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Store trajectories for training
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []

    def get_top_k_neighbors(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            num_epochs: int = 50) -> torch.Tensor:
        """
        Identify top-k similar neighbors for each node

        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Graph edges [2, num_edges]
            num_epochs: Number of RL epochs for training

        Returns:
            top_k_neighbors: Indices of top-k similar nodes [num_nodes, k]
        """
        num_nodes = x.size(0)

        # Compute initial similarity between nodes
        x_norm = F.normalize(x, p=2, dim=1)
        similarity = x_norm @ x_norm.T  # [num_nodes, num_nodes]

        # Build neighbor list from edge_index
        neighbor_list = self._build_neighbor_list(edge_index, num_nodes)

        # For each node, learn optimal k
        top_k_neighbors = []

        for node_idx in range(num_nodes):
            # Get candidate neighbors (all neighbors plus self)
            candidates = neighbor_list[node_idx] + [node_idx]
            candidates = list(set(candidates))  # Remove duplicates

            # Compute similarity scores
            sim_scores = similarity[node_idx, candidates]

            # Sort by similarity
            sorted_indices = torch.argsort(sim_scores, descending=True)
            sorted_candidates = [candidates[i] for i in sorted_indices]

            # RL training for this node
            best_k = self._learn_optimal_k(node_idx, x, sorted_candidates, num_epochs)

            # Select top-k neighbors
            top_k = sorted_candidates[:best_k]
            top_k_neighbors.append(top_k)

        # Pad to uniform size
        max_k = max(len(nbrs) for nbrs in top_k_neighbors)
        padded_neighbors = torch.full((num_nodes, max_k), -1, dtype=torch.long, device=self.device)
        for i, nbrs in enumerate(top_k_neighbors):
            padded_neighbors[i, :len(nbrs)] = torch.tensor(nbrs, device=self.device)

        return padded_neighbors

    def _build_neighbor_list(self, edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        """Build adjacency list from edge_index"""
        neighbor_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            if src != dst:
                neighbor_list[src].append(dst)
                neighbor_list[dst].append(src)
        return neighbor_list

    def _compute_state(self,
                       node_idx: int,
                       x: torch.Tensor,
                       selected_neighbors: List[int]) -> torch.Tensor:
        """
        Compute state representation for RL agent

        State includes:
        - Node's own features
        - Aggregated features of selected neighbors
        - Historical selection information
        """
        node_feat = x[node_idx]

        if len(selected_neighbors) > 0:
            neighbor_feats = x[selected_neighbors]
            agg_neighbor_feat = neighbor_feats.mean(dim=0)
        else:
            agg_neighbor_feat = torch.zeros_like(node_feat)

        # Simple state representation
        state = torch.cat([node_feat, agg_neighbor_feat])

        return state

    def _compute_reward(self,
                        node_idx: int,
                        selected_neighbors: List[int],
                        x: torch.Tensor) -> float:
        """
        Compute reward for neighbor selection

        Reward = +1 if average distance decreases, -1 otherwise
        Based on Eq. 4-5 in the paper
        """
        if len(selected_neighbors) == 0:
            return -1.0

        node_feat = x[node_idx]
        neighbor_feats = x[selected_neighbors]

        # Compute average cosine distance (Eq. 4)
        similarities = F.cosine_similarity(node_feat.unsqueeze(0), neighbor_feats)
        avg_distance = 1 - similarities.mean().item()

        # Reward based on distance change (Eq. 5)
        if hasattr(self, 'prev_distance'):
            if avg_distance <= self.prev_distance:
                reward = 1.0
            else:
                reward = -1.0
        else:
            reward = 0.0

        self.prev_distance = avg_distance
        return reward

    def _learn_optimal_k(self,
                         node_idx: int,
                         x: torch.Tensor,
                         candidates: List[int],
                         num_epochs: int) -> int:
        """Learn optimal k using PPO"""
        # Reset trajectory for this node
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []

        current_k = self.initial_k
        selected = candidates[:current_k]

        for epoch in range(num_epochs):
            state = self._compute_state(node_idx, x, selected)
            self.states.append(state)

            # Get action from policy
            with torch.no_grad():
                logits, _ = self.policy(state.unsqueeze(0))
                action_probs = F.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample().item()
                log_prob = action_dist.log_prob(torch.tensor(action))
                self.old_log_probs.append(log_prob)

            # Adjust k based on action
            delta = action - self.action_dim // 2
            new_k = max(1, min(self.max_k, current_k + delta))

            # Update selected neighbors
            selected = candidates[:new_k]
            current_k = new_k
            self.actions.append(action)

            # Compute reward
            reward = self._compute_reward(node_idx, selected, x)
            self.rewards.append(reward)

        # Train policy on collected trajectories
        self._update_policy()

        return current_k

    def _update_policy(self):
        """Update policy network using PPO"""
        if len(self.states) == 0:
            return

        states = torch.stack(self.states)
        actions = torch.tensor(self.actions, device=self.device)
        rewards = torch.tensor(self.rewards, device=self.device, dtype=torch.float)
        old_log_probs = torch.stack(self.old_log_probs)

        # Compute returns
        returns = torch.zeros_like(rewards)
        running_return = 0
        for i in reversed(range(len(rewards))):
            running_return = rewards[i] + self.gamma * running_return
            returns[i] = running_return

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(5):  # Multiple epochs
            logits, values = self.policy(states)
            action_probs = F.softmax(logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions)

            # Compute ratio
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Compute surrogate loss
            advantages = returns - values.squeeze()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()