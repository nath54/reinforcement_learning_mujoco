import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.typing import NDArray
import mujoco
from collections import deque
from typing import Any, cast

from src.core.types import GlobalConfig, Vec3
from src.simulation.generator import SceneBuilder
from src.simulation.physics import Physics
from src.simulation.sensors import EfficientCollisionSystemBetweenEnvAndAgent
from src.environment.reward_strategy import VelocityReward

def quaternion_to_euler(w: float, x: float, y: float, z: float) -> Vec3:
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    return Vec3(roll_x, pitch_y, yaw_z)

class CorridorEnv(gym.Env):
    def __init__(self, config: GlobalConfig) -> None:
        self.config = config

        # Build Scene
        self.scene = SceneBuilder(config)
        self.scene.build()
        self.model = self.scene.mujoco_model
        self.data = self.scene.mujoco_data

        self.physics = Physics(self.model, self.data)

        # Collision/Vision System
        self.collision_system = EfficientCollisionSystemBetweenEnvAndAgent(
            environment_obstacles=self.scene.environment_rects,
            env_bounds=self.scene.env_bounds,
            env_precision=config.simulation.env_precision
        )

        self.reward_strategy = VelocityReward(config.rewards)

        # State & Observation Spaces
        view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
        vision_width = 2 * view_range_grid
        vision_height = 2 * view_range_grid
        self.vision_size = vision_width * vision_height
        self.state_dim = self.vision_size + 13 # 13 for state vector

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

        self.previous_action = np.zeros(4, dtype=np.float64)
        self.current_step_count = 0
        self.previous_x_pos = 0.0

    def reset(self, seed: Any = None, options: Any = None) -> tuple[NDArray[np.float64], dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.physics.reset()

        # Lift robot
        robot_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        if robot_id != -1:
            self.data.xpos[robot_id][2] = 0.2

        # Warmup
        self.physics.robot_wheels_speed[:] = 0
        for _ in range(self.config.simulation.warmup_steps):
            mujoco.mj_step(self.model, self.data)

        self.previous_action = np.zeros(4, dtype=np.float64)
        self.current_step_count = 0
        self.action_history.clear()
        self.previous_x_pos = self.data.xpos[robot_id][0]

        # Reset reward strategy state (for forward progress tracking)
        if hasattr(self.reward_strategy, 'reset'):
            self.reward_strategy.reset()

        return self.get_observation(), {}

    def step(self, action: NDArray[np.float64]) -> tuple[NDArray[np.float64], float, bool, bool, dict[str, Any]]:
        target_speeds: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        max_speed = self.config.robot.max_speed

        # 1. Process Actions
        if self.config.robot.control_mode == "discrete_direction":
            action_idx = int(action) if np.isscalar(action) else int(action[0])
            if action_idx == 0: target_speeds[:] = max_speed
            elif action_idx == 1: target_speeds[:] = -max_speed
            elif action_idx == 2: target_speeds = np.array([-max_speed, max_speed, -max_speed, max_speed])
            elif action_idx == 3: target_speeds = np.array([max_speed, -max_speed, max_speed, -max_speed])

            self.physics.set_wheel_speeds_directly(target_speeds)

            # Map discrete to simulated continuous for state vector
            simulated = np.zeros(4)
            if action_idx == 0: simulated[:] = 1.0
            elif action_idx == 1: simulated[:] = -1.0
            elif action_idx == 2: simulated = np.array([-1.0, 1.0, -1.0, 1.0])
            elif action_idx == 3: simulated = np.array([1.0, -1.0, 1.0, -1.0])
            self.previous_action = simulated

        elif self.config.robot.control_mode == "continuous_vector":
            speed = np.clip(action[0], -1.0, 1.0)
            rotation = np.clip(action[1], -1.0, 1.0)

            left_speed = speed + rotation
            right_speed = speed - rotation
            raw_speeds = np.array([left_speed, right_speed, left_speed, right_speed])
            raw_speeds = np.clip(raw_speeds, -1.0, 1.0)

            # Smoothing
            self.action_history.append(raw_speeds)
            avg_action = np.mean(self.action_history, axis=0)

            target_speeds = avg_action * max_speed
            self.physics.set_wheel_speeds_directly(target_speeds)
            self.previous_action = raw_speeds

        else: # "continuous_wheels"
            action = np.clip(action, -1.0, 1.0)
            self.action_history.append(action)
            avg_action = np.mean(self.action_history, axis=0)
            target_speeds = avg_action * max_speed

            self.physics.set_wheel_speeds_directly(target_speeds)
            self.previous_action = action

        # 2. Physics Steps & Reward Calculation
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(self.config.simulation.action_repeat):
            self.physics.apply_additionnal_physics() # Uses air_drag and deceleration logic
            mujoco.mj_step(self.model, self.data)
            self.current_step_count += 1

            # Extract Data for Reward
            rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            pos = Vec3(self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2])
            vel_x = self.data.cvel[rid][0]

            # Stuck logic
            is_stuck = abs(vel_x) < self.config.rewards.stuck_x_velocity_threshold

            # Backward logic check
            is_going_backward = False
            if self.config.robot.control_mode == "discrete_direction":
                action_idx = int(action) if np.isscalar(action) else int(action[0])
                if action_idx == 1: is_going_backward = True
            elif np.mean(self.previous_action) < -0.3:
                is_going_backward = True

            # Compute Reward using strategy
            step_reward = self.reward_strategy.compute(
                pos, Vec3(vel_x, 0, 0), self.previous_action,
                self.current_step_count, is_stuck, is_going_backward
            )

            # Check conditions
            if pos.x < -self.config.simulation.corridor_length: truncated = True
            if abs(pos.y) > self.config.simulation.corridor_width + 10.0: truncated = True
            if pos.z < -5.0: terminated = True
            if pos.x > self.config.simulation.corridor_length:
                terminated = True
                step_reward += self.config.rewards.goal

            step_reward *= self.config.training.reward_scale

            if terminated: step_reward += 0.5
            if truncated: step_reward -= 0.5

            total_reward += step_reward
            if terminated or truncated: break

        # Enforce max_steps truncation
        if self.current_step_count >= self.config.simulation.max_steps:
            truncated = True

        # 3. Observation & Info
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        pos_arr = [self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2]]

        return self.get_observation(), total_reward, terminated, truncated, {"robot_pos": pos_arr}

    def get_observation(self) -> NDArray[np.float64]:
        if self.collision_system is None:
            return np.zeros(self.state_dim)

        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        pos = Vec3(self.data.xpos[rid][0], self.data.xpos[rid][1], self.data.xpos[rid][2])

        quat = self.data.xquat[rid]
        rot = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3]) # w, x, y, z

        vel = Vec3(self.data.cvel[rid][0], self.data.cvel[rid][1], self.data.cvel[rid][2])

        model_input = self.collision_system.get_robot_vision_and_state(
            pos, rot, vel, self.previous_action, self.config.simulation.robot_view_range
        )
        # Flatten vision and concatenate
        return np.concatenate((model_input.vision.flatten(), model_input.state_vector)).astype(np.float64)