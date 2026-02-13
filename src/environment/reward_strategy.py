"""
Reward strategy module

This module implements reward strategies for goal-based navigation.
The primary reward is based on distance reduction toward the goal.
"""

import numpy as np
from numpy.typing import NDArray

from src.core.interfaces import RewardStrategyProtocol
from src.core.types import Vec3, RewardConfig


#
class GoalDistanceReward(RewardStrategyProtocol):
    """
    Goal distance-based reward strategy.

    Rewards the agent for reducing distance to the goal.
    Works for both corridor and flat_world scenes.
    """

    #
    def __init__(self, config: RewardConfig):

        #
        self.cfg = config
        self.prev_distance: float = 0.0  # Track previous distance to goal
        self.initial_distance: float = 0.0  # Distance at episode start

    #
    def reset(
        self, goal_position: Vec3 | None = None, robot_position: Vec3 | None = None
    ) -> None:
        """
        Reset state for new episode.

        Args:
            goal_position: Goal position (used to calculate initial distance)
            robot_position: Robot starting position
        """

        # Calculate initial distance if both positions are provided
        if goal_position is not None and robot_position is not None:
            self.initial_distance = np.sqrt(
                (goal_position.x - robot_position.x) ** 2
                + (goal_position.y - robot_position.y) ** 2
            )
        else:
            self.initial_distance = 100.0  # Default fallback

        self.prev_distance = self.initial_distance

    #
    def compute(
        self,
        pos: Vec3,
        velocity: Vec3,
        goal_position: Vec3,
        action: NDArray[np.float64],
        step_count: int,
        is_stuck: bool,
        is_backward: bool,
    ) -> float:
        """
        Compute reward based on distance reduction toward goal.

        Args:
            pos: Current robot position
            velocity: Current robot velocity
            goal_position: Goal position
            action: Current action
            step_count: Current step count
            is_stuck: Whether robot is stuck
            is_backward: Whether robot is moving backward
        """

        reward: float = 0.0

        # Calculate current distance to goal
        current_distance: float = np.sqrt(
            (goal_position.x - pos.x) ** 2 + (goal_position.y - pos.y) ** 2
        )

        # Distance reduction reward (positive when getting closer)
        distance_delta: float = self.prev_distance - current_distance
        velocity_reward_scale: float = getattr(self.cfg, "velocity_reward_scale", 1.0)
        reward += distance_delta * velocity_reward_scale

        # Optional: forward progress bonus (scaled distance improvement)
        forward_progress_scale: float = getattr(self.cfg, "forward_progress_scale", 0.0)
        if forward_progress_scale > 0.0:
            reward += distance_delta * forward_progress_scale

        # Update previous distance
        self.prev_distance = current_distance

        # Stuck and backward penalties
        if is_stuck:
            reward += self.cfg.stuck_penalty

            # If backward, add escape bonus (encourages escape from stuck state)
            if is_backward:
                reward += self.cfg.backward_escape_bonus

        # Return computed reward
        return reward


# Legacy alias for backwards compatibility
VelocityReward = GoalDistanceReward
