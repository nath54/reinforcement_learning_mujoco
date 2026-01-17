import torch
import numpy as np
from typing import List
import numpy.typing as npt

class Memory:
    """Memory buffer for PPO training"""

    def __init__(self) -> None:
        self.actions: List[torch.Tensor] = []
        self.states: List[torch.Tensor] = []
        self.logprobs: List[npt.NDArray[np.float64]] = []
        self.rewards: List[npt.NDArray[np.float64]] = []
        self.is_terminals: List[npt.NDArray[np.float64]] = []

    def clear_memory(self) -> None:
        """Clear all stored memories"""
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]