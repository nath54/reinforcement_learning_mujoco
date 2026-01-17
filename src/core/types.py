from dataclasses import dataclass, field
import random
from typing import Any, Optional
import numpy as np
from numpy.typing import NDArray

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
        return ((p.x - self.x) ** 2 + (p.y - self.y) ** 2) ** 0.5

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

    def point_inside(self, p: Point2d) -> bool:
        """Check if point is inside rectangle"""
        return (
            p.x >= self.corner_top_left.x and
            p.x <= self.corner_bottom_right.x and
            p.y >= self.corner_top_left.y and
            p.y <= self.corner_bottom_right.y
        )

    def rect_collision(self, r: "Rect2d") -> bool:
        """Check if two rectangles collide"""
        return (
            self.point_inside(r.corner_top_left) or
            self.point_inside(r.corner_top_right) or
            self.point_inside(r.corner_bottom_left) or
            self.point_inside(r.corner_bottom_right)
        ) or (
            r.point_inside(self.corner_top_left) or
            r.point_inside(self.corner_top_right) or
            r.point_inside(self.corner_bottom_left) or
            r.point_inside(self.corner_bottom_right)
        )

@dataclass
class ValType:
    """Represents either a fixed value or a random range"""
    value: float | tuple[float, float]

    def get_value(self) -> float:
        return random.uniform(*self.value) if isinstance(self.value, tuple) else self.value

    def get_max_value(self) -> float:
        return max(*self.value) if isinstance(self.value, tuple) else self.value

@dataclass
class ModelInput:
    vision: NDArray[np.float64]
    state_vector: NDArray[np.float64]

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

    # Physics settings (defaults match original script logic)
    gravity: str = "0 0 -0.20"     # Original low gravity
    dt: float = 0.01               # Timestep
    solver: str = "Newton"         # Physics solver
    iterations: int = 500          # Solver iterations
    ground_friction: str = "1 0.005 0.0001" # Sliding, Torsional, Rolling friction

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
    use_true_velocity: bool = False  # If True, use actual velocity instead of position
    forward_progress_scale: float = 0.0  # Bonus for forward position change (delta X)

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
    # Early stopping
    early_stopping_enabled: bool = False
    early_stopping_consecutive_successes: int = 50
    early_stopping_success_threshold: float = 90.0  # Reward > this = success

@dataclass
class ModelConfig:
    type: str = "mlp"  # "mlp", "temporal_mlp", "ft_transformer"
    input_mode: str = "single_head"  # "single_head" or "multi_head"
    hidden_sizes: list[int] = field(default_factory=lambda: [256, 256])
    n_heads: int = 4
    n_layers: int = 2
    embedding_dim: int = 64
    dropout: float = 0.1
    # State configuration
    state_history_length: int = 1  # M frames of history (1 = current only)
    include_goal: bool = False  # Include goal-relative coordinates
    state_vector_dim: int = 13  # Will be computed if include_goal/history changes
    # Action configuration
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