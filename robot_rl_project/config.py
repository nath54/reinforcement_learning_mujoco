"""
Configuration constants and parameters for the robot RL simulation.
"""
import os
from utils import ValType

# Set environment variable to indicate mujoco to use GPU rendering
os.environ["MUJOCO_GL"] = "egl"

# File paths
CTRL_SAVE_PATH: str = "saved_control.json"

# Render settings
RENDER_WIDTH: int = 1440
RENDER_HEIGHT: int = 1024

# Corridor generation parameters
GENERATE_CORRIDOR_PARAM: dict = {
    "corridor_length": ValType(100.0),
    "corridor_width": ValType(3.0),
    "obstacles_mode": "sinusoidal",
    "obstacles_mode_param": {
        "obstacle_sep": ValType(4.0),
        "obstacle_size_x": ValType(0.4),
        "obstacle_size_y": ValType(0.4),
        "obstacle_size_z": ValType(0.2),
    }
}
