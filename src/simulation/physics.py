"""
Physics module.

Handles all physics related logic.
"""

import mujoco

import numpy as np
from numpy.typing import NDArray

from src.core.types import Vec3


#
class Physics:

    #
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        air_drag_coefficient: float = 0.5,
        robot_deceleration_force: float = 0.01
    ) -> None:

        #
        self.model_scene = model
        self.data_scene = data
        self.air_drag_coefficient = air_drag_coefficient

        # Exact logic from original
        self.robot_deceleration_force = robot_deceleration_force
        self.robot_deceleration_factor = 1.0 - self.robot_deceleration_force

        # Initialize wheels speed
        self.robot_wheels_speed: NDArray[np.float64] = np.zeros((4,), dtype=np.float64)

    #
    def reset(self) -> None:
        """
        Reset physics state
        """

        self.data_scene.qvel[:] = 0
        self.data_scene.qacc[:] = 0
        self.robot_wheels_speed[:] = 0

    #
    def apply_air_resistance(self) -> None:
        """
        Applies air drag to all bodies based on their velocity.
        """

        for i in range(self.model_scene.nbody):
            if i == 0:  # Skip world
                continue

            vel = self.data_scene.cvel[i][:3]
            speed = np.linalg.norm(vel)
            if speed < 1e-6:
                continue

            F_drag = -self.air_drag_coefficient * speed * vel
            qfrc_target = np.zeros(self.model_scene.nv)

            mujoco.mj_applyFT(
                m=self.model_scene,
                d=self.data_scene,
                force=F_drag,
                torque=np.zeros(3),
                point=self.data_scene.xpos[i],
                body=i,
                qfrc_target=qfrc_target
            )

    #
    def apply_robot_deceleration(self) -> None:
        """
        Apply deceleration to robot wheels
        """

        self.robot_wheels_speed[:] *= self.robot_deceleration_factor

    #
    def set_robot_wheel_speeds(self) -> None:
        """
        Set actuator values from wheel speeds
        """

        self.data_scene.ctrl = self.robot_wheels_speed

    #
    def apply_additionnal_physics(self) -> None:
        """
        Apply all additional physics (air resistance, deceleration, wheel speeds)
        """

        self.apply_air_resistance()
        self.apply_robot_deceleration()
        self.set_robot_wheel_speeds()

    #
    def apply_control(
        self,
        acceleration_factor: float = 0.0,
        rotation_factor: float = 0.0,
        acceleration_force: float = 0.15,
        rotation_force: float = 15000000,
        decceleration_factor: float = 1.0,
        max_front_wheel_speeds: float = 200.0,
        max_back_wheel_speeds: float = 100.0,
    ) -> None:

        """
        Apply control inputs to robot wheels.
        Exact logic from original:
        - Accel > 0: Front wheels get 100% force, Rear get 20%.
        - Accel < 0: Front get 20%, Rear get 100% (braking/reverse bias).
        """

        if acceleration_factor > 0:
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force * 0.2
        else:
            self.robot_wheels_speed[0] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[1] += acceleration_factor * acceleration_force * 0.2
            self.robot_wheels_speed[2] += acceleration_factor * acceleration_force
            self.robot_wheels_speed[3] += acceleration_factor * acceleration_force

        # Rotation logic
        self.robot_wheels_speed[0] -= rotation_factor * rotation_force
        self.robot_wheels_speed[1] += rotation_factor * rotation_force
        self.robot_wheels_speed[2] -= rotation_factor * rotation_force
        self.robot_wheels_speed[3] += rotation_factor * rotation_force

        # Manual deceleration (e.g. space bar)
        if decceleration_factor < 1.0:
            self.robot_wheels_speed[:] *= decceleration_factor

        # Clamp values
        if abs(acceleration_factor) > 0.00001:
            self.robot_wheels_speed[0:2] = np.clip(
                self.robot_wheels_speed[0:2], -max_front_wheel_speeds, max_front_wheel_speeds
            )
            self.robot_wheels_speed[2:4] = np.clip(
                self.robot_wheels_speed[2:4], -max_back_wheel_speeds, max_back_wheel_speeds
            )

        # Apply to MuJoCo
        self.data_scene.ctrl = self.robot_wheels_speed

    #
    def set_wheel_speeds_directly(self, speeds: NDArray[np.float64]) -> None:
        """
        Set wheel speeds directly
        """

        self.robot_wheels_speed[:] = speeds
        self.data_scene.ctrl = self.robot_wheels_speed
