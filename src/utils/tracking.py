"""
Tracking utilities for robot corridor environment
"""

import mujoco

from matplotlib import pyplot as plt


#
class TrackRobot:
    """
    Track robot position and velocity for plotting
    """

    #
    def __init__(self, mujoco_data_scene: mujoco.MjData) -> None:

        # mjdata is the physics state of the simulation from mujoco
        self.mjdata: mujoco.MjData = mujoco_data_scene

        # Robot position tracking
        self.robot_posx_track: list[float] = []
        self.robot_posy_track: list[float] = []
        self.robot_posz_track: list[float] = []

        # Robot velocity tracking
        self.robot_velx_track: list[float] = []
        self.robot_vely_track: list[float] = []
        self.robot_velz_track: list[float] = []

    #
    def track(self) -> None:
        """
        Record current robot state
        """

        # Record robot position
        self.robot_posx_track.append(self.mjdata.qpos[0])
        self.robot_posy_track.append(self.mjdata.qpos[1])
        self.robot_posz_track.append(self.mjdata.qpos[2])

        # Record robot velocity
        self.robot_velx_track.append(self.mjdata.qvel[0])
        self.robot_vely_track.append(self.mjdata.qvel[1])
        self.robot_velz_track.append(self.mjdata.qvel[2])

    #
    def plot_tracking(self) -> None:
        """
        Plot tracked data
        """

        # Create time range
        time_range: list[int] = list(range(len(self.robot_posx_track)))

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot position
        plt.subplot(2, 1, 1)
        plt.plot(time_range, self.robot_posx_track, label="pos x")
        plt.plot(time_range, self.robot_posy_track, label="pos y")
        plt.plot(time_range, self.robot_posz_track, label="pos z")

        # Add legend and title for the position plot
        plt.legend()
        plt.title("Robot Position")
        plt.ylabel("Position (m)")

        # Plot velocity
        plt.subplot(2, 1, 2)
        plt.plot(time_range, self.robot_velx_track, label="vel x")
        plt.plot(time_range, self.robot_vely_track, label="vel y")
        plt.plot(time_range, self.robot_velz_track, label="vel z")

        # Add legend and title for the velocity plot
        plt.legend()
        plt.title("Robot Velocity")
        plt.ylabel("Velocity (m/s)")
        plt.xlabel("Steps")

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
