"""
Tracking utilities for robot corridor environment
"""

import numpy as np
import mujoco

from matplotlib import pyplot as plt

from src.core.types import Vec3


#
class TrackRobot:
    """
    Track robot position, velocity, and distance to goal for plotting
    """

    #
    def __init__(
        self, mujoco_data_scene: mujoco.MjData, goal_position: Vec3 | None = None
    ) -> None:

        # mjdata is the physics state of the simulation from mujoco
        self.mjdata: mujoco.MjData = mujoco_data_scene

        # Goal position for distance tracking
        self.goal_position: Vec3 | None = goal_position

        # Robot position tracking
        self.robot_posx_track: list[float] = []
        self.robot_posy_track: list[float] = []
        self.robot_posz_track: list[float] = []

        # Robot velocity tracking
        self.robot_velx_track: list[float] = []
        self.robot_vely_track: list[float] = []
        self.robot_velz_track: list[float] = []

        # Distance to goal tracking
        self.distance_to_goal_track: list[float] = []

        # Print interval for distance display
        self.print_interval: int = 100

    #
    def set_goal_position(self, goal_position: Vec3) -> None:
        """
        Set or update the goal position
        """

        self.goal_position = goal_position

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

        # Record distance to goal
        if self.goal_position is not None:
            dist: float = np.sqrt(
                (self.mjdata.qpos[0] - self.goal_position.x) ** 2
                + (self.mjdata.qpos[1] - self.goal_position.y) ** 2
            )
            self.distance_to_goal_track.append(dist)

            # Print distance periodically
            # if len(self.distance_to_goal_track) % self.print_interval == 0:
            #     print(f"Distance to goal: {dist:.2f}m", end="\r")

    #
    def get_current_distance(self) -> float | None:
        """
        Get current distance to goal
        """

        if self.distance_to_goal_track:
            return self.distance_to_goal_track[-1]
        return None

    #
    def plot_tracking(self) -> None:
        """
        Plot tracked data including distance to goal
        """

        # Create time range
        time_range: list[int] = list(range(len(self.robot_posx_track)))

        # Determine subplot count
        has_goal: bool = len(self.distance_to_goal_track) > 0
        n_subplots: int = 3 if has_goal else 2

        # Create figure
        plt.figure(figsize=(12, 4 * n_subplots))

        # Plot position
        plt.subplot(n_subplots, 1, 1)
        plt.plot(time_range, self.robot_posx_track, label="pos x")
        plt.plot(time_range, self.robot_posy_track, label="pos y")
        plt.plot(time_range, self.robot_posz_track, label="pos z")

        # Add legend and title for the position plot
        plt.legend()
        plt.title("Robot Position")
        plt.ylabel("Position (m)")

        # Plot velocity
        plt.subplot(n_subplots, 1, 2)
        plt.plot(time_range, self.robot_velx_track, label="vel x")
        plt.plot(time_range, self.robot_vely_track, label="vel y")
        plt.plot(time_range, self.robot_velz_track, label="vel z")

        # Add legend and title for the velocity plot
        plt.legend()
        plt.title("Robot Velocity")
        plt.ylabel("Velocity (m/s)")

        # Plot distance to goal if available
        if has_goal:
            plt.subplot(n_subplots, 1, 3)
            plt.plot(
                time_range, self.distance_to_goal_track, label="distance", color="red"
            )
            plt.axhline(y=0, color="green", linestyle="--", label="goal")
            plt.legend()
            plt.title("Distance to Goal")
            plt.ylabel("Distance (m)")
            plt.xlabel("Steps")
        else:
            plt.xlabel("Steps")

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()
