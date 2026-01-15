import mujoco
from matplotlib import pyplot as plt
from typing import List

class TrackRobot:
    """Track robot position and velocity for plotting"""

    def __init__(self, mujoco_data_scene: mujoco.MjData) -> None:
        self.mjdata: mujoco.MjData = mujoco_data_scene

        self.robot_posx_track: List[float] = []
        self.robot_posy_track: List[float] = []
        self.robot_posz_track: List[float] = []

        self.robot_velx_track: List[float] = []
        self.robot_vely_track: List[float] = []
        self.robot_velz_track: List[float] = []

    def track(self) -> None:
        """Record current robot state"""
        self.robot_posx_track.append(self.mjdata.qpos[0])
        self.robot_posy_track.append(self.mjdata.qpos[1])
        self.robot_posz_track.append(self.mjdata.qpos[2])

        self.robot_velx_track.append(self.mjdata.qvel[0])
        self.robot_vely_track.append(self.mjdata.qvel[1])
        self.robot_velz_track.append(self.mjdata.qvel[2])

    def plot_tracking(self) -> None:
        """Plot tracked data"""
        time_range: List[int] = list(range(len(self.robot_posx_track)))

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        plt.plot(time_range, self.robot_posx_track, label="pos x")
        plt.plot(time_range, self.robot_posy_track, label="pos y")
        plt.plot(time_range, self.robot_posz_track, label="pos z")
        plt.legend()
        plt.title("Robot Position")
        plt.ylabel("Position (m)")

        plt.subplot(2, 1, 2)
        plt.plot(time_range, self.robot_velx_track, label="vel x")
        plt.plot(time_range, self.robot_vely_track, label="vel y")
        plt.plot(time_range, self.robot_velz_track, label="vel z")
        plt.legend()
        plt.title("Robot Velocity")
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Steps")

        plt.tight_layout()
        plt.show()