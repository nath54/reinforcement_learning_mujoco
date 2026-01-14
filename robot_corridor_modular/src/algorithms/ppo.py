import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, cast

from src.core.types import ModelConfig
from src.models.factory import create_policy_network

class ActorCritic(nn.Module):
    def __init__(self, config: ModelConfig, state_dim: int, action_dim: int):
        super().__init__()
        # In this modular version, we assume the 'state_dim' passed includes flattened vision
        # The factory creates the specific 'Policy' backbone
        self.actor = create_policy_network(config, state_dim, action_dim)
        self.critic = create_policy_network(config, state_dim, 1)
        self.control_mode = config.control_mode

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        if self.control_mode == "discrete_direction":
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return action, dist.log_prob(action)
        else:
            # Continuous implementation omitted for brevity, similar structure
            raise NotImplementedError("Continuous not fully ported in this snippet")

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(state)
        values = self.critic(state)
        
        if self.control_mode == "discrete_direction":
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(action.squeeze() if action.ndim > 1 else action)
            return log_probs, values, dist.entropy()
        return torch.tensor(0), values, torch.tensor(0)

class PPOAgent:
    def __init__(self, config: ModelConfig, state_dim: int, action_dim: int, lr: float):
        self.policy = ActorCritic(config, state_dim, action_dim).to("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.device = next(self.policy.parameters()).device
    
    def select_action(self, state: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        with torch.no_grad():
            t_state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            action, log_prob = self.policy.act(t_state)
        return action.cpu().numpy(), log_prob.cpu().numpy()
    
    # Update logic (standard PPO) omitted for brevity, strictly follows original logic