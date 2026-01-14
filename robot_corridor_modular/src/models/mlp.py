import torch
import torch.nn as nn
from typing import Tuple

class PolicyMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: list[int]):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, output_dim))
        # Note: No softmax here, strictly logits for Categorical
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)