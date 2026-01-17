import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.typing import NDArray
from math import log
from typing import Tuple, cast

from src.core.types import ModelConfig
from src.models.factory import create_policy_network
from src.utils.memory import Memory

class ActorCritic(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        vision_shape: tuple[int, int],
        state_vector_dim: int = 13,
        action_std_init: float = 0.5,
        action_std_min: float = 0.01,
        action_std_max: float = 1.0,
        actor_hidden_gain: float = 1.414,
        actor_output_gain: float = 0.01,
        control_mode: str = "continuous_wheels"
    ) -> None:
        super(ActorCritic, self).__init__()

        self.action_dim = action_dim
        self.control_mode = control_mode
        self.action_std_min = action_std_min
        self.action_std_max = action_std_max

        if self.control_mode != "discrete_direction":
            self.action_log_std = nn.Parameter(torch.full((action_dim,), log(action_std_init)))

        self.vision_shape = vision_shape
        self.state_vector_dim = state_vector_dim
        self.vision_size = vision_shape[0] * vision_shape[1]

        # CNN for Vision
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, vision_shape[0], vision_shape[1])
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # Encoder for State Vector
        self.state_encoder = nn.Sequential(
            nn.Linear(state_vector_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # Fusion Dimension
        fusion_dim = cnn_out_size + 64

        # Actor Head
        self.actor = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm((256,)),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm((128,)),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )

        # Initialize actor weights
        actor_layers = [layer for layer in self.actor if isinstance(layer, nn.Linear)]
        for i, layer in enumerate(actor_layers):
            is_output_layer = (i == len(actor_layers) - 1)
            gain = actor_output_gain if is_output_layer else actor_hidden_gain
            nn.init.orthogonal_(layer.weight, gain=gain)
            nn.init.constant_(layer.bias, 0.0)

        # Critic Head
        self.critic = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def _process_input(self, state: torch.Tensor):
        """Process input: split vision and state, encode both, fuse"""
        vision_flat = state[:, :self.vision_size]
        state_vec = state[:, self.vision_size:]

        vision_img = vision_flat.view(-1, 1, self.vision_shape[0], self.vision_shape[1])

        vision_features = self.cnn(vision_img)
        state_features = self.state_encoder(state_vec)

        return torch.cat((vision_features, state_features), dim=1)

    def act(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Select action"""
        features = self._process_input(state)

        if self.control_mode == "discrete_direction":
            action_logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=action_logits)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
        else:
            action_mean = self.actor(features)
            action_std = torch.exp(self.action_log_std).clamp(
                min=self.action_std_min, max=self.action_std_max
            ).to(state.device)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action_logprob = dist.log_prob(action).sum(axis=-1)

        return action.detach(), action_logprob.detach()

    def get_action_mean(self, state: torch.Tensor) -> torch.Tensor:
        """Get action mean (for deterministic play)"""
        features = self._process_input(state)
        return self.actor(features)

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions"""
        features = self._process_input(state)

        if self.control_mode == "discrete_direction":
            action_logits = self.actor(features)
            dist = torch.distributions.Categorical(logits=action_logits)
            action_logprobs = dist.log_prob(action.squeeze(-1) if action.ndim > 1 else action)
            dist_entropy = dist.entropy()
        else:
            action_mean = self.actor(features)
            action_std = torch.exp(self.action_log_std).clamp(
                min=self.action_std_min, max=self.action_std_max
            ).to(state.device)
            dist = torch.distributions.Normal(action_mean, action_std)
            action_logprobs = dist.log_prob(action).sum(axis=-1)
            dist_entropy = dist.entropy().sum(axis=-1)

        state_values = self.critic(features)

        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        vision_shape: tuple[int, int],
        lr: float = 0.0003,
        gamma: float = 0.99,
        K_epochs: int = 4,
        eps_clip: float = 0.2,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.02,
        value_loss_coeff: float = 0.5,
        grad_clip_max_norm: float = 0.5,
        action_std_init: float = 0.5,
        action_std_min: float = 0.01,
        action_std_max: float = 1.0,
        actor_hidden_gain: float = 1.414,
        actor_output_gain: float = 0.01,
        control_mode: str = "continuous_wheels"
    ) -> None:

        self.gamma: float = gamma
        self.eps_clip: float = eps_clip
        self.K_epochs: int = K_epochs
        self.gae_lambda: float = gae_lambda
        self.control_mode: str = control_mode
        self.entropy_coeff: float = entropy_coeff
        self.value_loss_coeff: float = value_loss_coeff
        self.grad_clip_max_norm: float = grad_clip_max_norm

        # Detect device
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPOAgent using device: {self.device}")

        self.action_dim: int = action_dim

        self.policy: ActorCritic = ActorCritic(
            state_dim, action_dim, vision_shape,
            action_std_init=action_std_init,
            action_std_min=action_std_min,
            action_std_max=action_std_max,
            actor_hidden_gain=actor_hidden_gain,
            actor_output_gain=actor_output_gain,
            control_mode=control_mode
        ).float().to(self.device)

        self.optimizer: optim.Adam = optim.Adam(self.policy.parameters(), lr=lr)

        self.policy_old: ActorCritic = ActorCritic(
            state_dim, action_dim, vision_shape,
            action_std_init=action_std_init,
            action_std_min=action_std_min,
            action_std_max=action_std_max,
            actor_hidden_gain=actor_hidden_gain,
            actor_output_gain=actor_output_gain,
            control_mode=control_mode
        ).float().to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss: nn.MSELoss = nn.MSELoss()

    def select_action(self, state: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Select action from policy"""
        with torch.no_grad():
            state_tensor: torch.Tensor = torch.FloatTensor(state).to(self.device)

            is_single_obs = (state.ndim == 1)
            if is_single_obs:
                state_tensor = state_tensor.unsqueeze(0)

            action_tensor, action_logprob_tensor = self.policy_old.act(state_tensor)

        action = action_tensor.cpu().numpy()
        action_logprob = action_logprob_tensor.cpu().numpy()

        if is_single_obs:
            return action.flatten(), action_logprob.flatten()

        return action, action_logprob

    def get_action_statistics(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Get action statistics (mean) for deterministic play"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            if state.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)

            mean_val = self.policy.get_action_mean(state_tensor)

        return mean_val.cpu().numpy().flatten()

    def update(self, memory: Memory, next_state: NDArray[np.float64], next_done: NDArray[np.float64]) -> None:
        """Update policy using PPO"""

        # Convert memory to tensors
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.tensor(np.array(memory.logprobs), dtype=torch.float32).to(self.device).detach()

        # Flatten for training
        old_states_flat = old_states.view(-1, self.policy.vision_size + self.policy.state_vector_dim)

        if self.control_mode == "discrete_direction":
            old_actions_flat = old_actions.view(-1, 1)
        else:
            old_actions_flat = old_actions.view(-1, self.action_dim)

        old_logprobs_flat = old_logprobs.view(-1)

        # Get values
        with torch.no_grad():
            values = self.policy.critic(self.policy._process_input(old_states_flat)).view(len(memory.states), -1)

            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            if next_state.ndim == 1:
                next_state_tensor = next_state_tensor.unsqueeze(0)

            next_value = self.policy.critic(self.policy._process_input(next_state_tensor))

        # Calculate GAE
        rewards = np.array(memory.rewards)
        is_terminals = np.array(memory.is_terminals)

        rewards_t = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        is_terminals_t = torch.tensor(is_terminals, dtype=torch.float32).to(self.device)
        next_done_t = torch.tensor(next_done, dtype=torch.float32).to(self.device)

        returns = torch.zeros_like(rewards_t).to(self.device)
        advantages = torch.zeros_like(rewards_t).to(self.device)

        gae = 0
        num_steps = len(memory.rewards)

        for t in range(num_steps - 1, -1, -1):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - next_done_t
                next_val = next_value.squeeze()
            else:
                next_non_terminal = 1.0 - is_terminals_t[t + 1]
                next_val = values[t + 1].squeeze()

            delta = rewards_t[t] + self.gamma * next_val * next_non_terminal - values[t].squeeze()
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages[t] = gae
            returns[t] = advantages[t] + values[t].squeeze()

        # Flatten
        returns = returns.view(-1)
        advantages = advantages.view(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Optimize policy
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_flat, old_actions_flat)
            state_values = torch.squeeze(state_values)

            ratios = torch.exp(logprobs - old_logprobs_flat)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = (
                -torch.min(surr1, surr2)
                + self.value_loss_coeff * self.MseLoss(state_values, returns)
                - self.entropy_coeff * dist_entropy
            )

            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_max_norm)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())