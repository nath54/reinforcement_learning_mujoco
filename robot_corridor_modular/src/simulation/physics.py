import mujoco
import numpy as np
import numpy.typing as npt
from src.core.types import Vec3

class Physics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, air_drag_coeff: float = 0.5, deceleration: float = 0.01):
        self.model = model
        self.data = data
        self.air_drag_coeff = air_drag_coeff
        self.deceleration_factor = 1.0 - deceleration
        self.robot_wheels_speed: npt.NDArray[np.float64] = np.zeros(4, dtype=np.float64)

    def reset(self):
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        self.robot_wheels_speed[:] = 0

    def apply_air_resistance(self):
        for i in range(self.model.nbody):
            if i == 0: continue
            vel = self.data.cvel[i][:3]
            speed = np.linalg.norm(vel)
            if speed < 1e-6: continue
            
            f_drag = -self.air_drag_coeff * speed * vel
            qfrc = np.zeros(self.model.nv)
            mujoco.mj_applyFT(self.model, self.data, f_drag, np.zeros(3), self.data.xpos[i], i, qfrc)

    def apply_control(self, acceleration: float, rotation: float, max_speed: float):
        # Differential drive logic mapped to 4 wheels
        # Fl, Fr, Rl, Rr
        
        # Simple logic: Accel adds to all, Rotation adds to Left, subs from Right
        acc_force = 0.15 * acceleration
        rot_force = 0.05 * rotation
        
        # Apply forces to internal speed state
        if acceleration != 0:
            self.robot_wheels_speed += acc_force
        else:
            self.robot_wheels_speed *= self.deceleration_factor
            
        # Rotation differential
        self.robot_wheels_speed[0] -= rot_force # FL
        self.robot_wheels_speed[1] += rot_force # FR
        self.robot_wheels_speed[2] -= rot_force # RL
        self.robot_wheels_speed[3] += rot_force # RR

        # Clip
        np.clip(self.robot_wheels_speed, -max_speed, max_speed, out=self.robot_wheels_speed)
        
        # Apply to Mujoco
        self.data.ctrl[:] = self.robot_wheels_speed

    def set_wheel_speeds_directly(self, speeds: npt.NDArray[np.float64]):
        self.robot_wheels_speed[:] = speeds
        self.data.ctrl[:] = self.robot_wheels_speed