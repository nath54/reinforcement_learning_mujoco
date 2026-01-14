import numpy as np
import numpy.typing as npt
import mujoco
from typing import List, Tuple, Any
from src.core.types import Rect2d, Vec3, ModelInput

class EfficientCollisionSystemBetweenEnvAndAgent:
    def __init__(self, obstacles: List[Rect2d], bounds: Rect2d, precision: float):
        self.precision = precision
        self.bounds = bounds
        
        w = int((bounds.corner_bottom_right.x - bounds.corner_top_left.x) / precision)
        h = int((bounds.corner_bottom_right.y - bounds.corner_top_left.y) / precision)
        self.env_matrix = np.zeros((w, h), dtype=np.float64)
        
        for r in obstacles:
            sx = int((r.corner_top_left.x - bounds.corner_top_left.x) / precision)
            ex = int((r.corner_bottom_right.x - bounds.corner_top_left.x) / precision)
            sy = int((r.corner_top_left.y - bounds.corner_top_left.y) / precision)
            ey = int((r.corner_bottom_right.y - bounds.corner_top_left.y) / precision)
            
            sx = max(0, sx); ex = min(w, ex)
            sy = max(0, sy); ey = min(h, ey)
            self.env_matrix[sx:ex, sy:ey] = r.height

    def get_observation(self, pos: Vec3, rot: Vec3, vel: Vec3, prev_action: npt.NDArray[np.float64], view_range: float) -> ModelInput:
        # 1. Vision
        gx = int((pos.x - self.bounds.corner_top_left.x) / self.precision)
        gy = int((pos.y - self.bounds.corner_top_left.y) / self.precision)
        gr = int(view_range / self.precision)
        
        sx, ex = gx - gr, gx + gr
        sy, ey = gy - gr, gy + gr
        
        vision = np.zeros((ex-sx, ey-sy), dtype=np.float64)
        
        # Overlap logic
        mx, my = self.env_matrix.shape
        isx, iex = max(0, sx), min(mx, ex)
        isy, iey = max(0, sy), min(my, ey)
        
        if isx < iex and isy < iey:
            vision[isx-sx:iex-sx, isy-sy:iey-sy] = self.env_matrix[isx:iex, isy:iey]
            
        vision = np.sign(vision).flatten()
        
        # 2. State Vector
        state = np.array([
            pos.x / 100.0, pos.y / 10.0, pos.z,
            rot.x, rot.y, rot.z,
            vel.x / 10.0, vel.y / 10.0, vel.z / 10.0,
            *prev_action
        ], dtype=np.float64)
        
        return ModelInput(vision, state)

class Camera:
    def __init__(self):
        self.mode = "free"
        self.robot_id = -1

    def update(self, cam: Any, model: mujoco.MjModel):
        if self.robot_id == -1:
            self.robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        
        if self.mode == "follow":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            cam.distance = 5.0
            cam.elevation = -20.0
            cam.azimuth = 0
        elif self.mode == "top_down":
             cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
             cam.trackbodyid = self.robot_id
             cam.distance = 15.0
             cam.elevation = -90.0
        else:
             cam.type = mujoco.mjtCamera.mjCAMERA_FREE

class TrackRobot:
    def __init__(self, data: mujoco.MjData):
        self.data = data
        self.history: List[List[float]] = []

    def update(self):
        self.history.append([self.data.qpos[0], self.data.qpos[1], self.data.qpos[2]])