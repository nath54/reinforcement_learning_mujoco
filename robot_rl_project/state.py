"""
State management for the simulation.
"""
from typing import Any
import numpy as np
import mujoco

from physics import Physics
from camera import Camera
from controls import Controls
from track import TrackRobot
from robot import Robot
from vision_sensor import VisionSensor


class State:
    """Manages the complete simulation state."""
    
    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        physics: Physics,
        camera: Camera,
        controls: Controls,
        viewer_instance: Any,
        robot_track: TrackRobot,
        robot: Robot,
        vision_sensor: VisionSensor
    ) -> None:
        self.mj_model: mujoco.MjModel = mj_model
        self.mj_data: mujoco.MjData = mj_data
        self.physics: Physics = physics
        self.camera: Camera = camera
        self.controls: Controls = controls
        self.viewer_instance: Any = viewer_instance
        self.robot_track: TrackRobot = robot_track
        self.robot: Robot = robot
        self.vision_sensor: VisionSensor = vision_sensor
        self.quit: bool = False
        
        self.init_state()
    
    def init_state(self) -> None:
        """Initialize the state with camera parameters."""
        # Configure camera parameters with your good settings
        self.viewer_instance.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.viewer_instance.cam.fixedcamid = -1
        
        # Set good camera position and orientation parameters
        self.viewer_instance.cam.azimuth = 1.01171875
        self.viewer_instance.cam.elevation = -16.6640625
        self.viewer_instance.cam.lookat = np.array(
            [1.55633679e-04, -4.88295545e-02, 1.05485916e+00]
        )
