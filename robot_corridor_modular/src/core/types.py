from dataclasses import dataclass, field
import random
from typing import Any, Optional
import numpy as np
import numpy.typing as npt

@dataclass
class Vec3:
    x: float
    y: float
    z: float

@dataclass
class Point2d:
    x: float
    y: float

    def dist(self, p: "Point2d") -> float:
        return (p.x - self.x) ** 2 + (p.y - self.y) ** 2

@dataclass
class Rect2d:
    corner_top_left: Point2d
    corner_bottom_right: Point2d
    height: float = 0.0

    @property
    def corner_top_right(self) -> Point2d:
        return Point2d(self.corner_bottom_right.x, self.corner_top_left.y)

    @property
    def corner_bottom_left(self) -> Point2d:
        return Point2d(self.corner_top_left.x, self.corner_bottom_right.y)

@dataclass
class ValType:
    value: float | tuple[float, float]

    def get_value(self) -> float:
        return random.uniform(*self.value) if isinstance(self.value, tuple) else self.value

    def get_max_value(self) -> float:
        return max(*self.value) if isinstance(self.value, tuple) else self.value

@dataclass
class ModelInput:
    vision: npt.NDArray[np.float64]
    state_vector: npt.NDArray[np.float64]

# --- Config Dataclasses ---

@dataclass
class SimulationConfig:
    corridor_length: float = 100.0
    corridor_width: float = 3.0
    robot_view_range: float = 4.0
    max_steps: int = 30000
    env_precision: float = 0.2
    warmup_steps: int = 1000
    action_repeat: int = 10
    num_envs: int = 6
    obstacles_mode: str = "sinusoidal"
    obstacles_mode_param: dict[str, Any] = field(default_factory=dict)

@dataclass
class RobotConfig:
    xml_path: str = "four_wheels_robot.xml"
    action_smoothing_k: int = 5
    control_mode: str = "discrete_direction"
    max_speed: float = 10.0

@dataclass
class RewardConfig:
    type: str = "velocity"
    goal: float = 100.0
    collision: float = -0.05
    straight_line_penalty: float = 0.1
    straight_line_dist: float = 40.0
    pacer_speed: float = 0.004
    use_velocity_reward: bool = True
    velocity_reward_scale: float = 0.1
    stuck_penalty: float = -0.1
    stuck_x_velocity_threshold: float = 0.05
    backward_escape_bonus: float = 0.02

@dataclass
class TrainingConfig:
    agent_type: str = "ppo"
    max_episodes: int = 20000
    model_path: str = "exp4_obstacles.pth"
    load_weights_from: Optional[str] = None
    learning_rate: float = 0.0001
    gamma: float = 0.99
    k_epochs: int = 3
    eps_clip: float = 0.2
    update_timestep: int = 4000
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    grad_clip_max_norm: float = 0.5
    reward_scale: float = 1.0

@dataclass
class ModelConfig:
    type: str = "mlp"
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    n_heads: int = 4
    n_layers: int = 2
    embedding_dim: int = 128
    dropout: float = 0.1
    state_vector_dim: int = 13
    action_std_init: float = 0.5
    action_std_min: float = 0.01
    action_std_max: float = 1.0
    actor_hidden_gain: float = 1.414
    actor_output_gain: float = 0.01
    control_mode: str = "discrete_direction"

@dataclass
class GlobalConfig:
    simulation: SimulationConfig
    robot: RobotConfig
    rewards: RewardConfig
    training: TrainingConfig
    model: ModelConfig