"""
Multi-Head State Encoder

Encodes different state modalities (position, rotation, velocity, goal) with separate
MLP heads before fusion. This allows the model to learn specialized representations
for each type of information.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class MultiHeadStateEncoder(nn.Module):
    """
    Encodes state vector with separate heads for each modality.

    Input structure (for M history frames):
    - Position: M x 3 values
    - Rotation: M x 3 values
    - Velocity: M x 2 values
    - Goal: 4 values (dx, dy, distance, angle)
    - Previous action: 4 values
    """

    def __init__(
        self,
        history_length: int = 1,
        include_goal: bool = True,
        hidden_dim: int = 64,
        output_dim: int = 64
    ):
        super().__init__()
        self.history_length = history_length
        self.include_goal = include_goal

        # Input dimensions per modality
        self.pos_dim = 3 * history_length
        self.rot_dim = 3 * history_length
        self.vel_dim = 2 * history_length
        self.goal_dim = 4 if include_goal else 0
        self.action_dim = 4

        # Separate encoders for each modality
        self.pos_encoder = nn.Sequential(
            nn.Linear(self.pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.rot_encoder = nn.Sequential(
            nn.Linear(self.rot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.vel_encoder = nn.Sequential(
            nn.Linear(self.vel_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        if include_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(self.goal_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2)
            )

        # Fusion layer
        fusion_input_dim = (hidden_dim // 2) * 3 + hidden_dim // 4  # pos, rot, vel, action
        if include_goal:
            fusion_input_dim += hidden_dim // 2

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.ReLU()
        )

        self.output_dim = output_dim

    def get_input_dim(self) -> int:
        """Total input dimension expected"""
        return self.pos_dim + self.rot_dim + self.vel_dim + self.goal_dim + self.action_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, state_dim) tensor with concatenated state

        Returns:
            (batch, output_dim) encoded state features
        """
        # Split state into modalities
        idx = 0
        pos = state[:, idx:idx + self.pos_dim]
        idx += self.pos_dim

        rot = state[:, idx:idx + self.rot_dim]
        idx += self.rot_dim

        vel = state[:, idx:idx + self.vel_dim]
        idx += self.vel_dim

        if self.include_goal:
            goal = state[:, idx:idx + self.goal_dim]
            idx += self.goal_dim

        action = state[:, idx:idx + self.action_dim]

        # Encode each modality
        pos_feat = self.pos_encoder(pos)
        rot_feat = self.rot_encoder(rot)
        vel_feat = self.vel_encoder(vel)
        action_feat = self.action_encoder(action)

        # Fuse
        if self.include_goal:
            goal_feat = self.goal_encoder(goal)
            fused = torch.cat([pos_feat, rot_feat, vel_feat, goal_feat, action_feat], dim=1)
        else:
            fused = torch.cat([pos_feat, rot_feat, vel_feat, action_feat], dim=1)

        return self.fusion(fused)


class SingleHeadStateEncoder(nn.Module):
    """Simple MLP encoder for all state features concatenated."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.output_dim = output_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)
