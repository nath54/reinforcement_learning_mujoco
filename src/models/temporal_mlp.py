"""
Temporal MLP Encoder Module

Uses 1D convolutions over the history dimension to capture temporal patterns,
followed by MLP for final encoding. More efficient than transformers for
simple temporal dependencies.
"""

import torch
import torch.nn as nn


#
class TemporalMLP(nn.Module):
    """
    Temporal encoder using 1D convolutions over history frames.

    Input: (batch, history_length, features_per_frame)
    Output: (batch, output_dim)
    """

    #
    def __init__(
        self,
        history_length: int,
        features_per_frame: int,
        goal_dim: int = 0,
        action_dim: int = 4,
        hidden_dim: int = 64,
        output_dim: int = 64
    ) -> None:

        # Initialize parent
        super().__init__()

        # Store parameters
        self.history_length: int = history_length
        self.features_per_frame: int = features_per_frame
        self.goal_dim: int = goal_dim
        self.action_dim: int = action_dim

        # 1D conv over time dimension
        # Input: (batch, features_per_frame, history_length)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(features_per_frame, hidden_dim, kernel_size=min(3, history_length), padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=min(3, history_length), padding=1),
            nn.ReLU(),
        )

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Process goal and action separately if present
        #
        extra_dim: int = 0

        if goal_dim > 0:
            self.goal_encoder = nn.Sequential(
                nn.Linear(goal_dim, hidden_dim // 2),
                nn.ReLU()
            )
            extra_dim += hidden_dim // 2

        if action_dim > 0:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, hidden_dim // 4),
                nn.ReLU()
            )
            extra_dim += hidden_dim // 4

        # Final MLP
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_dim + extra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim: int = output_dim

    #
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            state: (batch, total_dim) flattened state
                   Structure: [history_features, goal, action]

        Returns:
            (batch, output_dim) encoded state
        """

        # Get batch size
        batch_size: int = state.shape[0]

        # Split state
        history_dim: int = self.history_length * self.features_per_frame
        history_flat = state[:, :history_dim]

        idx: int = history_dim
        #
        goal: torch.Tensor
        #
        if self.goal_dim > 0:
            goal = state[:, idx:idx + self.goal_dim]
            idx += self.goal_dim

        action: torch.Tensor = state[:, idx:idx + self.action_dim]

        # Reshape history for conv: (batch, features, time)
        history: torch.Tensor = history_flat.view(batch_size, self.history_length, self.features_per_frame)
        history = history.permute(0, 2, 1)  # (batch, features, time)

        # Apply temporal conv
        temporal_features: torch.Tensor = self.temporal_conv(history)  # (batch, hidden, time)
        temporal_features = self.pool(temporal_features).squeeze(-1)  # (batch, hidden)

        # Encode goal and action
        features_list: list[torch.Tensor] = [temporal_features]

        if self.goal_dim > 0:
            goal_features: torch.Tensor = self.goal_encoder(goal)
            features_list.append(goal_features)

        if self.action_dim > 0:
            action_features: torch.Tensor = self.action_encoder(action)
            features_list.append(action_features)

        # Fuse and final MLP
        fused: torch.Tensor = torch.cat(features_list, dim=1)

        return self.final_mlp(fused)
