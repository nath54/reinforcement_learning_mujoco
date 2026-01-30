"""
Environment Wrapper Module

Wraps the MuJoCo simulation into a Gymnasium-compatible environment.
Handles scene building, physics stepping, and observation generation.
"""

from typing import Any
from collections import deque

import mujoco

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces

from src.core.types import GlobalConfig, Vec3
from src.simulation.generator import SceneBuilder
from src.simulation.physics import Physics
from src.simulation.sensors import EfficientCollisionSystemBetweenEnvAndAgent
from src.environment.reward_strategy import VelocityReward


# Helper function for quaternion conversion
def quaternion_to_euler(
    w: float,
    x: float,
    y: float,
    z: float
) -> Vec3:

    """
    Convert quaternion to euler angles (roll, pitch, yaw)
    """

    t0: float = +2.0 * (w * x + y * z)
    t1: float = +1.0 - 2.0 * (x * x + y * y)
    roll_x: float = np.arctan2(t0, t1)

    t2: float = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2

    pitch_y: float = np.arcsin(t2)

    t3: float = +2.0 * (w * z + x * y)
    t4: float = +1.0 - 2.0 * (y * y + z * z)

    yaw_z: float = np.arctan2(t3, t4)

    return Vec3(roll_x, pitch_y, yaw_z)


# Main Environment Class
class SimulationEnv(gym.Env):
    """
    Gymnasium environment for the robot corridor task.
    """

    #
    def __init__(self, config: GlobalConfig, scene: SceneBuilder | None = None) -> None:

        # Store config
        self.config = config

        # Use provided scene OR create new one (backward compatible)
        if scene is None:
            self.scene = SceneBuilder(config)
            self.scene.build()
        else:
            self.scene = scene

        # Store MuJoCo model and data
        self.model = self.scene.mujoco_model
        self.data = self.scene.mujoco_data

        # Physics
        self.physics = Physics(self.model, self.data)

        # Collision/Vision System
        self.collision_system = EfficientCollisionSystemBetweenEnvAndAgent(
            environment_obstacles=self.scene.environment_rects,
            env_bounds=self.scene.env_bounds,
            env_precision=config.simulation.env_precision
        )

        # Reward Strategy
        self.reward_strategy = VelocityReward(config.rewards)

        # Goal position (from scene builder)
        self.goal_position: Vec3 | None = self.scene.goal_position

        # Determine if goal info is included in state
        self.include_goal: bool = config.model.include_goal

        # State & Observation Spaces
        view_range_grid: int = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width: int = 2 * view_range_grid
        vision_height: int = 2 * view_range_grid
        self.vision_size: int = vision_width * vision_height

        # Base state vector: 13 dimensions
        # Goal-relative coords add 4 dimensions if include_goal
        goal_dims: int = 4 if self.include_goal else 0
        self.state_dim: int = self.vision_size + 13 + goal_dims

        # Define Observation Space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        # Define Action Space based on control mode
        if config.robot.control_mode == "discrete_direction":
            # 0: Forward, 1: Backward, 2: Left, 3: Right
            self.action_space = spaces.Discrete(4)
            self.action_dim = 4
        elif config.robot.control_mode == "continuous_vector":
            # [0]: Speed, [1]: Rotation
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            self.action_dim = 2
        else: # "continuous_wheels"
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
            self.action_dim = 4

        # Action smoothing
        self.action_history: deque = deque(maxlen=config.robot.action_smoothing_k)

        # Previous action and step count
        self.previous_action = np.zeros(4, dtype=np.float64)
        self.current_step_count: int = 0
        self.previous_x_pos: float = 0.0

    #
    def reset(
        self,
        seed: Any = None,
        options: Any = None
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:

        """
        Reset environment to initial state
        """

        # Reset super class
        super().reset(seed=seed)

        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)

        # Reset Physics
        self.physics.reset()

        # Lift robot
        robot_id: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_id != -1:
            self.data.xpos[robot_id][2] = 0.2

        # Warmup
        self.physics.robot_wheels_speed[:] = 0
        for _ in range(self.config.simulation.warmup_steps):
            mujoco.mj_step(self.model, self.data)

        # Reset previous action and step count
        self.previous_action = np.zeros(4, dtype=np.float64)
        self.current_step_count = 0
        self.action_history.clear()
        self.previous_x_pos = self.data.xpos[robot_id][0]

        # Randomize goal for flat_world (if configured)
        if self.config.simulation.scene_type == "flat_world":
            self.goal_position = self.scene.reset_goal_position()

        # Reset reward strategy with goal and robot position
        robot_pos: Vec3 = Vec3(
            self.data.xpos[robot_id][0],
            self.data.xpos[robot_id][1],
            self.data.xpos[robot_id][2]
        )
        if hasattr(self.reward_strategy, 'reset'):
            self.reward_strategy.reset(goal_position=self.goal_position, robot_position=robot_pos)

        return self.get_observation(), {}

    #
    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
             action: Action to execute

        Returns:
             tuple containing (observation, reward, terminated, truncated, info)
        """

        # Initialize target speeds
        target_speeds: NDArray[np.float64] = np.zeros(4, dtype=np.float64)

        # Maximum speed
        max_speed: float = self.config.robot.max_speed

        # 1. Process Actions
        if self.config.robot.control_mode == "discrete_direction":

            # Get action index
            action_idx: int = int(action) if np.isscalar(action) else int(action[0])

            # Set target speeds based on action
            if action_idx == 0: target_speeds[:] = max_speed
            elif action_idx == 1: target_speeds[:] = -max_speed
            elif action_idx == 2: target_speeds = np.array([-max_speed, max_speed, -max_speed, max_speed])
            elif action_idx == 3: target_speeds = np.array([max_speed, -max_speed, max_speed, -max_speed])

            # Set wheel speeds
            self.physics.set_wheel_speeds_directly(target_speeds)

            # Map discrete to simulated continuous for state vector
            simulated = np.zeros(4)
            if action_idx == 0: simulated[:] = 1.0
            elif action_idx == 1: simulated[:] = -1.0
            elif action_idx == 2: simulated = np.array([-1.0, 1.0, -1.0, 1.0])
            elif action_idx == 3: simulated = np.array([1.0, -1.0, 1.0, -1.0])

            # Update previous action
            self.previous_action = simulated

        # Continuous vector control
        elif self.config.robot.control_mode == "continuous_vector":

            # Clip action & rotation
            speed: float = np.clip(action[0], -1.0, 1.0)
            rotation: float = np.clip(action[1], -1.0, 1.0)

            # Calculate left and right speeds
            left_speed = speed + rotation
            right_speed = speed - rotation

            # Create raw speeds array & clip
            raw_speeds = np.array([left_speed, right_speed, left_speed, right_speed])
            raw_speeds = np.clip(raw_speeds, -1.0, 1.0)

            # Smoothing
            self.action_history.append(raw_speeds)
            avg_action = np.mean(self.action_history, axis=0)

            # Set wheel speeds
            target_speeds = avg_action * max_speed
            self.physics.set_wheel_speeds_directly(target_speeds)

            # Update previous action
            self.previous_action = raw_speeds

        # Continuous wheels control
        else:

            # Clip action
            action = np.clip(action, -1.0, 1.0)

            # Smoothing
            self.action_history.append(action)
            avg_action = np.mean(self.action_history, axis=0)

            # Set wheel speeds
            target_speeds = avg_action * max_speed

            # Update previous action
            self.physics.set_wheel_speeds_directly(target_speeds)
            self.previous_action = action

        # 2. Physics Steps & Reward Calculation
        total_reward: float = 0.0
        terminated: bool = False
        truncated: bool = False

        # Physics steps
        for _ in range(self.config.simulation.action_repeat):

            # Uses air_drag and deceleration logic
            self.physics.apply_additionnal_physics()

            # Step physics
            mujoco.mj_step(self.model, self.data)

            # Update step count
            self.current_step_count += 1

            # Extract Data for Reward
            rid: int = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            pos: Vec3 = Vec3(self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2])
            vel_x: float = self.data.cvel[rid][0]

            # Stuck logic check
            is_stuck: bool = abs(vel_x) < self.config.rewards.stuck_x_velocity_threshold

            # Backward logic check
            is_going_backward: bool = False
            if self.config.robot.control_mode == "discrete_direction":
                #
                action_idx = int(action) if np.isscalar(action) else int(action[0])
                #
                if action_idx == 1: is_going_backward = True
            #
            elif np.mean(self.previous_action) < -0.3:
                #
                is_going_backward = True

            # Compute Reward using goal distance strategy
            step_reward: float = self.reward_strategy.compute(
                pos, Vec3(vel_x, 0, 0), self.goal_position, self.previous_action,
                self.current_step_count, is_stuck, is_going_backward
            )

            # Unified goal-based termination (both flat_world and corridor)
            if self.goal_position is not None:
                dist_to_goal: float = np.sqrt(
                    (pos.x - self.goal_position.x)**2 + (pos.y - self.goal_position.y)**2
                )
                if dist_to_goal < self.config.simulation.goal_radius:
                    terminated = True
                    step_reward += self.config.rewards.goal

            # Termination if robot falls
            if pos.z < -5.0:
                terminated = True

            # Truncation for flat_world only (corridor has walls)
            if self.config.simulation.scene_type == "flat_world":
                arena_x: float = self.config.simulation.corridor_length
                arena_y: float = self.config.simulation.corridor_width
                if abs(pos.x) > arena_x + 5.0 or abs(pos.y) > arena_y + 5.0:
                    truncated = True

            # Scale reward
            step_reward *= self.config.training.reward_scale

            # Add bonus for termination
            if terminated: step_reward += 0.5
            if truncated: step_reward -= 0.5

            # Add step reward to total
            total_reward += step_reward

            # Break if terminated or truncated
            if terminated or truncated: break

        # Enforce max_steps truncation
        if self.current_step_count >= self.config.simulation.max_steps:
            truncated = True

        # 3. Observation & Info
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        pos_arr = [self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2]]

        # Return observation, reward, terminated, truncated, info
        return self.get_observation(), total_reward, terminated, truncated, {"robot_pos": pos_arr}

    #
    def get_observation(self) -> NDArray[np.float64]:
        """
        Generate current observation vector
        """

        # If no collision system, return zero observation
        if self.collision_system is None:
            return np.zeros(self.state_dim)

        # Get robot position
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        pos = Vec3(self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2])

        # Get robot rotation
        quat = self.data.xquat[rid]
        rot = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3]) # w, x, y, z

        # Get robot velocity
        vel = Vec3(self.data.cvel[rid][0], self.data.cvel[rid][1], self.data.cvel[rid][2])

        # Get robot vision and state (pass goal if include_goal is enabled)
        goal_for_sensor: Vec3 | None = self.goal_position if self.include_goal else None
        model_input = self.collision_system.get_robot_vision_and_state(
            pos, rot, vel, self.previous_action, self.config.simulation.robot_view_range,
            goal_position=goal_for_sensor,
            vision_position_offset=self.config.simulation.vision_position_offset,
            vision_encoding_mode=self.config.simulation.vision_encoding_mode
        )

        # Flatten vision and concatenate
        return np.concatenate((model_input.vision.flatten(), model_input.state_vector)).astype(np.float64)
