"""
Sensors Module

This module handles collision detection and visual sensor simulation
for the robot in the environment.
"""

from typing import Any, cast

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.core.types import Rect2d, Vec3, ModelInput

# Efficient Collision System
class EfficientCollisionSystemBetweenEnvAndAgent:
    """
    Handles efficient collision detection between the agent and static environment obstacles.
    Uses a grid-based approach (Occupancy Grid) for fast lookups.
    """

    #
    def __init__(
        self,
        environment_obstacles: list[Rect2d],
        env_bounds: Rect2d,
        env_precision: float = 0.1
    ) -> None:

        """
        Initialize the collision system.

        Args:
           environment_obstacles: List of rectangular obstacles in the environment
           env_bounds: The bounding box of the entire environment
           env_precision: Resolution of the grid (meters per cell)
        """

        #
        self.env_precision = env_precision
        self.env_bounds = env_bounds

        # Calculate grid dimensions
        w: int = int((env_bounds.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision)
        h: int = int((env_bounds.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)

        # Initialize environment matrix (0 = empty, value = height)
        self.env_matrix: NDArray[np.float64] = np.zeros((w, h), dtype=np.float64)

        # Rasterize obstacles into the grid
        #
        rect: Rect2d
        #
        for rect in environment_obstacles:
            start_x: int = int((rect.corner_top_left.x - env_bounds.corner_top_left.x) / env_precision)
            end_x: int = int((rect.corner_bottom_right.x - env_bounds.corner_top_left.x) / env_precision)
            start_y: int = int((rect.corner_top_left.y - env_bounds.corner_top_left.y) / env_precision)
            end_y: int = int((rect.corner_bottom_right.y - env_bounds.corner_top_left.y) / env_precision)

            # Clamp to grid bounds
            start_x = max(0, start_x)
            end_x = min(w, end_x)
            start_y = max(0, start_y)
            end_y = min(h, end_y)

            # Fill grid
            self.env_matrix[start_x:end_x, start_y:end_y] = rect.height

    #
    def get_robot_vision_and_state(
        self,
        robot_pos: Vec3,
        robot_rot: Vec3,
        robot_speed: Vec3,
        previous_action: NDArray[np.float64],
        robot_view_range: float,
        goal_position: Vec3 | None = None,
    ) -> ModelInput:

        """
        Extracts the local vision grid around the robot and the state vector.

        Args:
            robot_pos: Robot position
            robot_rot: Robot rotation (euler angles)
            robot_speed: Robot velocity
            previous_action: Previous action taken
            robot_view_range: Vision range in meters
            goal_position: Optional goal position for goal-relative coordinates
        """

        # 1. Grid Coords
        robot_grid_x: int = int((robot_pos.x - self.env_bounds.corner_top_left.x) / self.env_precision)
        robot_grid_y: int = int((robot_pos.y - self.env_bounds.corner_top_left.y) / self.env_precision)
        view_range_grid: int = int(robot_view_range / self.env_precision)

        start_x: int = robot_grid_x - view_range_grid
        end_x: int = robot_grid_x + view_range_grid
        start_y: int = robot_grid_y - view_range_grid
        end_y: int = robot_grid_y + view_range_grid

        # 2. Extract Submatrix
        vision_w: int = end_x - start_x
        vision_h: int = end_y - start_y
        vision_matrix: NDArray[np.float64] = np.zeros((vision_w, vision_h), dtype=np.float64)

        # Handle boundary conditions (cropping)
        env_max_x, env_max_y = self.env_matrix.shape
        inter_start_x: int = max(0, start_x)
        inter_end_x: int = min(env_max_x, end_x)
        inter_start_y: int = max(0, start_y)
        inter_end_y: int = min(env_max_y, end_y)

        if inter_start_x < inter_end_x and inter_start_y < inter_end_y:
            paste_start_x: int = inter_start_x - start_x
            paste_end_x: int = paste_start_x + (inter_end_x - inter_start_x)
            paste_start_y: int = inter_start_y - start_y
            paste_end_y: int = paste_start_y + (inter_end_y - inter_start_y)

            vision_matrix[paste_start_x:paste_end_x, paste_start_y:paste_end_y] = \
                self.env_matrix[inter_start_x:inter_end_x, inter_start_y:inter_end_y]

        # Binarize/Normalize vision (just checking for non-zero height implies obstacle)
        # Using sign() to just show obstacle presence vs absence
        vision_matrix = np.sign(vision_matrix)

        # 3. State Vector (Normalized) - Base 13 dimensions
        base_state: list[float] = [
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
        ]

        # 4. Goal-relative coordinates (4 dimensions if goal provided)
        if goal_position is not None:
            # Compute goal-relative values
            dx: float = (goal_position.x - robot_pos.x) / 100.0  # Normalized
            dy: float = (goal_position.y - robot_pos.y) / 100.0  # Normalized
            distance: float = np.sqrt(dx**2 + dy**2)
            # Angle to goal relative to robot yaw (robot_rot.z is yaw)
            angle_to_goal: float = np.arctan2(dy, dx) - robot_rot.z

            # Normalize angle to [-pi, pi]
            angle_to_goal = np.arctan2(np.sin(angle_to_goal), np.cos(angle_to_goal))

            # Add goal info to state
            base_state.extend([dx, dy, distance, angle_to_goal / np.pi])  # angle normalized to [-1, 1]

        state_vector: NDArray[np.float64] = np.array(base_state, dtype=np.float64)

        # Return ModelInput
        return ModelInput(vision_matrix, state_vector)


# Camera sensor wrapper
class Camera:
    """
    Wrapper for MuJoCo camera control in the viewer.
    """

    #
    def __init__(self) -> None:
        #
        self.current_mode: str = "free"
        self.robot_id: int = -1

    #
    def set_mode(self, mode: str) -> None:
        """
        Set the camera mode (free, follow_robot, top_down)
        """

        if mode in ["free", "follow_robot", "top_down"]:
            self.current_mode = mode

    #
    def update_viewer_camera(
        self,
        cam: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData
    ) -> None:

        """
        Update the viewer camera position/target based on the current mode.
        """

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

    #
    def get_camera_data(self, cam: Any) -> tuple[NDArray[np.float64], float, float, float, int, Any]:
        """
        Return raw camera data for debugging
        """

        return (cam.lookat, cam.distance, cam.azimuth, cam.elevation, cam.trackbodyid, cam.type)