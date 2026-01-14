from src.core.interfaces import RewardStrategyProtocol
from src.core.types import Vec3, RewardConfig
import numpy as np
import numpy.typing as npt

class VelocityReward(RewardStrategyProtocol):
    def __init__(self, config: RewardConfig):
        self.cfg = config

    def compute(self, pos: Vec3, velocity: Vec3, action: npt.NDArray[np.float64], step_count: int, is_stuck: bool, is_backward: bool) -> float:
        reward = 0.0
        
        # Main component
        if self.cfg.use_velocity_reward:
             reward += pos.x * self.cfg.velocity_reward_scale
        else:
             reward += pos.x - (self.cfg.pacer_speed * step_count)
             
        # Penalties
        if is_stuck:
            reward += self.cfg.stuck_penalty
            if is_backward:
                reward += self.cfg.backward_escape_bonus
                
        return reward