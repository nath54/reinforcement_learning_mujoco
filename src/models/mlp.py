"""
MLP Model Module

Simple Multi-Layer Perceptron for policy networks.
"""

import torch
import torch.nn as nn


class PolicyMLP(nn.Module):
    """
    Simple MLP policy network.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_sizes: list[int]
    ) -> None:

        #
        super().__init__()

        #
        layers: list[nn.Module] = []
        last: int = input_dim

        # Build layers
        #
        h: int
        #
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h

        layers.append(nn.Linear(last, output_dim))

        # Note: No softmax here, strictly logits for Categorical
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        """

        return self.net(x)
