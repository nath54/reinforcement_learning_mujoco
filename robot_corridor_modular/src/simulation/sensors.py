import numpy as np
import numpy.typing as npt
import mujoco
from typing import List, Tuple, Any, cast
from src.core.types import Rect2d, Vec3, ModelInput, Point2d

class EfficientCollisionSystemBetweenEnvAndAgent:
    def __init__(
        self,
        environment_obstacles: List[Rect2d],
        env_bounds: Rect2d,
        env_precision: float = 0.1
    ) -> None:
        self.env_precision = env_precision
        self.env_bounds = env_bounds

        w = int((env_bounds.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision)
        h = int((env_bounds.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)

        self.env_matrix: npt.NDArray[np.float64] = np.zeros((w, h), dtype=np.float64)

        for rect in environment_obstacles:
            start_x = int((rect.corner_top_left.x - env_bounds.corner_top_left.x) / env_precision)
            end_x = int((rect.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision)
            start_y = int((rect.corner_top_left.y - env_bounds.corner_top_left.y) / env_precision)
            end_y = int((rect.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)

            start_x = max(0, start_x)
            end_x = min(w, end_x)
            start_y = max(0, start_y)
            end_y = min(h, end_y)

            self.env_matrix[start_x:end_x, start_y:end_y] = rect.height

    def get_robot_vision_and_state(
        self,
        robot_pos: Vec3,
        robot_rot: Vec3,
        robot_speed: Vec3,
        previous_action: npt.NDArray[np.float64],
        robot_view_range: float,
    ) -> ModelInput:
        # 1. Grid Coords
        robot_grid_x = int((robot_pos.x - self.env_bounds.corner_top_left.x) / self.env_precision)
        robot_grid_y = int((robot_pos.y - self.env_bounds.corner_top_left.y) / self.env_precision)
        view_range_grid = int(robot_view_range / self.env_precision)

        start_x = robot_grid_x - view_range_grid
        end_x = robot_grid_x + view_range_grid
        start_y = robot_grid_y - view_range_grid
        end_y = robot_grid_y + view_range_grid

        # 2. Extract Submatrix
        vision_w = end_x - start_x
        vision_h = end_y - start_y
        vision_matrix = np.zeros((vision_w, vision_h), dtype=np.float64)

        env_max_x, env_max_y = self.env_matrix.shape
        inter_start_x = max(0, start_x)
        inter_end_x = min(env_max_x, end_x)
        inter_start_y = max(0, start_y)
        inter_end_y = min(env_max_y, end_y)

        if inter_start_x < inter_end_x and inter_start_y < inter_end_y:
            paste_start_x = inter_start_x - start_x
            paste_end_x = paste_start_x + (inter_end_x - inter_start_x)
            paste_start_y = inter_start_y - start_y
            paste_end_y = paste_start_y + (inter_end_y - inter_start_y)

            vision_matrix[paste_start_x:paste_end_x, paste_start_y:paste_end_y] = \
                self.env_matrix[inter_start_x:inter_end_x, inter_start_y:inter_end_y]

        vision_matrix = np.sign(vision_matrix)

        # 3. State Vector (Normalized)
        state_vector = np.array([
            robot_pos.x / 100.0,
            robot_pos.y / 10.0,
            robot_pos.z,
            robot_rot.x,
            robot_rot.y,
            robot_rot.z,
            robot_speed.x / 10.0,
            robot_speed.y / 10.0,
            robot_speed.z / 10.0,
            previous_action[0], previous_action[1], previous_action[2], previous_action[3]
        ], dtype=np.float64)

        return ModelInput(vision_matrix, state_vector)

class Camera:
    def __init__(self) -> None:
        self.current_mode: str = "free"
        self.robot_id: int = -1

    def set_mode(self, mode: str) -> None:
        if mode in ["free", "follow_robot", "top_down"]:
            self.current_mode = mode

    def update_viewer_camera(
        self,
        cam: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData
    ) -> None:
        if self.robot_id == -1:
            self.robot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot")

        if self.current_mode == "free":
            if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        elif self.current_mode == "follow_robot":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            cam.distance = 5.0
            cam.elevation = -16
            cam.azimuth = 1.01171875

        elif self.current_mode == "top_down":
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            cam.distance = 10.0
            cam.azimuth = 90.0
            cam.elevation = -89.9

    def get_camera_data(self, cam: Any) -> Tuple[npt.NDArray[np.float64], float, float, float, int, Any]:
        return (cam.lookat, cam.distance, cam.azimuth, cam.elevation, cam.trackbodyid, cam.type)