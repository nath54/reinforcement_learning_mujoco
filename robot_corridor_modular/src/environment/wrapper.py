import gymnasium as gym
from gymnasium import spaces
import numpy as np
import numpy.typing as npt
import mujoco
from collections import deque
from typing import Tuple, Any, cast

from src.core.types import GlobalConfig, Vec3
from src.simulation.generator import SceneBuilder
from src.simulation.physics import Physics
from src.simulation.sensors import EfficientCollisionSystemBetweenEnvAndAgent
from src.environment.reward_strategy import VelocityReward

def quaternion_to_euler(w: float, x: float, y: float, z: float) -> Vec3:
    # Simplified Euler conversion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(t0, t1)
    return Vec3(roll, 0.0, 0.0) # Placeholder for full math

class CorridorEnv(gym.Env):
    def __init__(self, config: GlobalConfig):
        self.config = config
        
        # Build Scene
        self.scene = SceneBuilder(config)
        self.scene.build()
        self.model = self.scene.mujoco_model
        self.data = self.scene.mujoco_data
        
        self.physics = Physics(self.model, self.data)
        self.sensors = EfficientCollisionSystemBetweenEnvAndAgent(
            self.scene.environment_rects, self.scene.env_bounds, config.simulation.env_precision
        )
        self.reward_strategy = VelocityReward(config.rewards)
        
        # Spaces
        # Vision size
        v_grid = int(config.simulation.robot_view_range / config.simulation.env_precision) * 2
        input_dim = (v_grid * v_grid) + 13
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(input_dim,), dtype=np.float32)
        
        if config.robot.control_mode == "discrete_direction":
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32) # vector mode default
            
        self.prev_action = np.zeros(4)
        self.step_count = 0
        self.action_history = deque(maxlen=config.robot.action_smoothing_k)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.physics.reset()
        
        # Lift robot
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        self.data.xpos[rid][2] = 0.2
        
        # Warmup
        for _ in range(self.config.simulation.warmup_steps):
            mujoco.mj_step(self.model, self.data)
            
        self.step_count = 0
        self.prev_action = np.zeros(4)
        return self._get_obs(), {}

    def step(self, action):
        # Action Processing
        target_speeds = np.zeros(4)
        max_s = self.config.robot.max_speed
        
        # Map Discrete/Vector to Wheels
        if self.config.robot.control_mode == "discrete_direction":
            idx = int(action) if np.isscalar(action) else int(action[0])
            if idx == 0: target_speeds[:] = max_s
            elif idx == 1: target_speeds[:] = -max_s
            elif idx == 2: target_speeds = np.array([-max_s, max_s, -max_s, max_s])
            elif idx == 3: target_speeds = np.array([max_s, -max_s, max_s, -max_s])
            
            # Update prev action for state
            self.prev_action = np.zeros(4); self.prev_action[idx] = 1.0 # One-hot-ish
            
        elif self.config.robot.control_mode == "continuous_vector":
            speed, rot = action[0], action[1]
            l, r = speed + rot, speed - rot
            target_speeds = np.array([l, r, l, r]) * max_s
            self.prev_action = target_speeds / max_s

        # Sim Loop
        total_reward = 0.0
        terminated = False; truncated = False
        
        self.physics.set_wheel_speeds_directly(target_speeds)
        
        for _ in range(self.config.simulation.action_repeat):
            self.physics.apply_air_resistance()
            mujoco.mj_step(self.model, self.data)
            self.step_count += 1
            
            # Info extraction
            rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
            pos = Vec3(*self.data.xpos[rid])
            vel = Vec3(*self.data.cvel[rid][:3])
            
            # Stuck check
            is_stuck = abs(vel.x) < self.config.rewards.stuck_x_velocity_threshold
            is_back = (target_speeds[0] < 0)
            
            reward = self.reward_strategy.compute(pos, vel, self.prev_action, self.step_count, is_stuck, is_back)
            
            # Check bounds
            if pos.x > self.config.simulation.corridor_length: 
                terminated = True; reward += self.config.rewards.goal
            if pos.y > self.config.simulation.corridor_width + 5: truncated = True
            
            total_reward += reward
            if terminated or truncated: break
            
        return self._get_obs(), total_reward * self.config.training.reward_scale, terminated, truncated, {}

    def _get_obs(self):
        rid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        pos = Vec3(*self.data.xpos[rid])
        quat = self.data.xquat[rid]
        rot = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])
        vel = Vec3(*self.data.cvel[rid][:3])
        
        inp = self.sensors.get_observation(pos, rot, vel, self.prev_action, self.config.simulation.robot_view_range)
        return np.concatenate((inp.vision, inp.state_vector)).astype(np.float32)