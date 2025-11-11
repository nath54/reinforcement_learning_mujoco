"""
Robot RL Project - Modular MuJoCo Simulation with Vision Sensors

A modular implementation of a robot simulation with raycasting vision sensors
for reinforcement learning applications.
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

# Core modules
from .vision_sensor import VisionSensor
from .config import *
from .utils import Vec3, ValType

__all__ = [
    'VisionSensor',
    'Vec3',
    'ValType',
]
