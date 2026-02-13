import torch

import numpy as np
from numpy.typing import NDArray


class Memory:
    """Memory buffer for PPO training"""

    def __init__(self) -> None:
        self.actions: list[torch.Tensor] = []
        self.states: list[torch.Tensor] = []
        self.logprobs: list[NDArray[np.float64]] = []
        self.rewards: list[NDArray[np.float64]] = []
        self.is_terminals: list[NDArray[np.float64]] = []

    def clear_memory(self) -> None:
        """Clear all stored memories"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
