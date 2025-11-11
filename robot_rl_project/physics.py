"""
Physics simulation management including air resistance and robot control.
"""
import numpy as np
from numpy.typing import NDArray
import mujoco


class Physics:
    """Manages physics simulation and robot movement."""
    
    def __init__(
        self,
        mujoco_model_scene: mujoco.MjModel,
        mujoco_data_scene: mujoco.MjData,
        air_drag_coefficient: float = 0.5,
        robot_deceleration_force: float = 0.01
    ) -> None:
        self.model_scene: mujoco.MjModel = mujoco_model_scene
        self.data_scene: mujoco.MjData = mujoco_data_scene
        
        self.air_drag_coefficient: float = air_drag_coefficient
        
        self.robot_deceleration_force: float = robot_deceleration_force
        self.robot_deceleration_factor: float = 1.0 - self.robot_deceleration_force
        
        # Initialize simulation with zero velocities for stability
        self.data_scene.qvel[:] = 0
        self.data_scene.qacc[:] = 0
        
        # Initialize wheels speed
        self.robot_wheels_speed: NDArray[np.float64] = np.zeros((4,), dtype=np.float64)
    
    def apply_air_resistance(self) -> None:
        """
        Applies air drag to all bodies based on their velocity.
        F_drag = -k * v^2 * sign(v)
        """
        for i in range(self.model_scene.nbody):
            # Skip world body
            if i == 0:
                continue
            
            # Get linear velocity of body i
            vel = self.data_scene.cvel[i][:3]
            speed = np.linalg.norm(vel)
            
            if speed < 1e-6:
                continue
            
            # Simple quadratic drag model: F = -k * v * |v|
            F_drag = -self.air_drag_coefficient * speed * vel
            
            # Allocate full-sized qfrc_target vector (size nv)
            qfrc_target = np.zeros(self.model_scene.nv)
            
            # Apply the force at the center of mass
            mujoco.mj_applyFT(
                m=self.model_scene,
                d=self.data_scene,
                force=F_drag,
                torque=np.zeros(3),
                point=self.data_scene.xpos[i],
                body=i,
                qfrc_target=qfrc_target
            )
    
    def apply_additionnal_physics(self) -> None:
        """Apply all additional physics effects."""
        self.apply_air_resistance()
        self.apply_robot_deceleration()
        self.set_robot_wheel_speeds()
    
    def apply_robot_ctrl_movement(
        self,
        acceleration_factor: float = 0.0,
        rotation_factor: float = 0.0,
        acceleration_force: float = 0.15,
        rotation_force: float = 0.05,
        decceleration_factor: float = 1.0,
        max_front_wheel_speeds: float = 200.0,
        max_back_wheel_speeds: float = 100,
    ) -> None:
        """Apply control movements to the robot."""
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
        
        self.robot_wheels_speed[0] -= rotation_factor * rotation_force
        self.robot_wheels_speed[1] += rotation_factor * rotation_force
        self.robot_wheels_speed[2] -= rotation_factor * rotation_force
        self.robot_wheels_speed[3] += rotation_factor * rotation_force
        
        if decceleration_factor < 1.0:
            self.robot_wheels_speed[:] *= decceleration_factor
        
        # Clamp values
        self.robot_wheels_speed[0:2] = np.clip(self.robot_wheels_speed[0:2], -max_front_wheel_speeds, max_front_wheel_speeds)
        self.robot_wheels_speed[2:5] = np.clip(self.robot_wheels_speed[2:5], -max_back_wheel_speeds, max_back_wheel_speeds)
    
    def apply_robot_deceleration(self) -> None:
        """Apply deceleration to robot wheels."""
        self.robot_wheels_speed[:] *= self.robot_deceleration_factor
    
    def set_robot_wheel_speeds(self) -> None:
        """Set the actuator values."""
        self.data_scene.ctrl = self.robot_wheels_speed
