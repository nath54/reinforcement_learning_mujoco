"""
Reward strategy module

This module implements different reward strategies for the environment.
"""


import numpy as np
from numpy.typing import NDArray

from src.core.interfaces import RewardStrategyProtocol
from src.core.types import Vec3, RewardConfig


#
class VelocityReward(RewardStrategyProtocol):
    """
    Velocity reward strategy
    """

    #
    def __init__(self, config: RewardConfig):

        #
        self.cfg = config
        self.prev_x: float = 0.0  # Track previous X position for progress bonus

    #
    def reset(self) -> None:
        """
        Reset state for new episode
        """

        # Reset previous X position
        self.prev_x = 0.0

    #
    def compute(
        self,
        pos: Vec3,
        velocity: Vec3,
        action: NDArray[np.float64],
        step_count: int,
        is_stuck: bool,
        is_backward: bool
    ) -> float:

        """
        Compute reward based on current state
        """

        reward: float = 0.0

        # Main component
        if self.cfg.use_velocity_reward:

            # Use velocity or position based on config
            if getattr(self.cfg, 'use_true_velocity', False):
                # Use actual X velocity (reward forward movement)
                reward += velocity.x * self.cfg.velocity_reward_scale

            else:
                # Legacy: use position (accumulates over time)
                reward += pos.x * self.cfg.velocity_reward_scale

        # If not using velocity reward, use position-based reward
        else:
            reward += pos.x - (self.cfg.pacer_speed * step_count)

        # Forward progress bonus (delta X)
        forward_progress_scale = getattr(self.cfg, 'forward_progress_scale', 0.0)

        # If forward progress scale is positive, add forward progress bonus
        if forward_progress_scale > 0.0:
            delta_x = pos.x - self.prev_x
            reward += delta_x * forward_progress_scale
            self.prev_x = pos.x

        # Stuck and backward penalties
        if is_stuck:
            reward += self.cfg.stuck_penalty

            # If backward, add escape bonus (encourages escape from stuck state)
            if is_backward:
                reward += self.cfg.backward_escape_bonus

        # Return computed reward
        return reward