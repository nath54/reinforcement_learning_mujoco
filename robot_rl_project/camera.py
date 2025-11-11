"""
Camera management for different viewing modes.
"""
from typing import Any
import numpy as np
from numpy.typing import NDArray
import mujoco


class Camera:
    """Manages camera modes and positioning."""
    
    def __init__(self) -> None:
        # Modes: "free", "follow_robot", "top_down"
        self.current_mode: str = "free"
        
        # Store robot ID to avoid looking it up every frame
        self.robot_id: int = -1
    
    def set_mode(self, mode: str) -> None:
        """Set camera mode."""
        if mode in ["free", "follow_robot", "top_down"]:
            self.current_mode = mode
    
    def update_viewer_camera(
        self,
        cam: Any,
        model: mujoco.MjModel,
        data: mujoco.MjData
    ) -> None:
        """Update viewer camera based on current mode."""
        # Get robot ID once
        if self.robot_id == -1:
            self.robot_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            if self.robot_id == -1:
                print("Warning: Could not find robot body named 'robot'")
                return
        
        # Handle camera logic based on mode
        if self.current_mode == "free":
            # Set camera to free mode and do nothing else
            # This lets the user control it manually with the mouse
            if cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
                cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        
        elif self.current_mode == "follow_robot":
            # Use MuJoCo's built-in tracking camera
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            # You can set preferred distance and elevation
            cam.distance = 5.0
            cam.elevation = -16
            cam.azimuth = 1.01171875
        
        elif self.current_mode == "top_down":
            # Use tracking camera, but point straight down
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = self.robot_id
            cam.distance = 10.0
            cam.azimuth = 90.0
            cam.elevation = -89.9
    
    def get_camera_data(self, cam: Any) -> tuple[NDArray[np.float64], float, float, float, int, Any]:
        """Get camera data for debugging."""
        return (cam.lookat, cam.distance, cam.azimuth, cam.elevation, cam.trackbodyid, cam.type)
