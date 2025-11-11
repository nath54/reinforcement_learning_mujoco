"""
Robot tracking for logging position and velocity.
"""
from typing import Any
import mujoco
from matplotlib import pyplot as plt


class TrackRobot:
    """Tracks robot position and velocity over time."""
    
    def __init__(self, mujoco_data_scene: mujoco.MjData) -> None:
        self.mjdata: mujoco.MjData = mujoco_data_scene
        
        self.robot_posx_track: list[float] = []
        self.robot_posy_track: list[float] = []
        self.robot_posz_track: list[float] = []
        
        self.robot_velx_track: list[float] = []
        self.robot_vely_track: list[float] = []
        self.robot_velz_track: list[float] = []
    
    def track(self) -> None:
        """Record current robot state."""
        self.robot_posx_track.append(self.mjdata.qpos[0])
        self.robot_posy_track.append(self.mjdata.qpos[1])
        self.robot_posz_track.append(self.mjdata.qpos[2])
        
        self.robot_velx_track.append(self.mjdata.qvel[0])
        self.robot_vely_track.append(self.mjdata.qvel[1])
        self.robot_velz_track.append(self.mjdata.qvel[2])
    
    def plot_tracking(self) -> None:
        """Plot the tracking data."""
        time_range: list[int] = list(range(len(self.robot_posx_track)))
        
        plt.plot(time_range, self.robot_posx_track, label="pos x")  # type: ignore
        plt.plot(time_range, self.robot_posy_track, label="pos y")  # type: ignore
        plt.plot(time_range, self.robot_posz_track, label="pos z")  # type: ignore
        plt.plot(time_range, self.robot_velx_track, label="vel x")  # type: ignore
        plt.plot(time_range, self.robot_vely_track, label="vel y")  # type: ignore
        plt.plot(time_range, self.robot_velz_track, label="vel z")  # type: ignore
        plt.yscale("symlog")
        plt.legend()  # type: ignore
        plt.show()  # type: ignore
