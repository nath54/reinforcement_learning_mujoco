import json
import mujoco
from typing import Any, Set, Dict, List, Optional
from src.simulation.physics import Physics
from src.simulation.sensors import Camera

CTRL_SAVE_PATH: str = "saved_control.json"

class Controls:
    def __init__(self, physics: Physics, camera: Camera, render_mode: bool = False) -> None:
        self.physics = physics
        self.camera = camera
        self.render_mode = render_mode

        self.quit_requested: bool = False
        self.display_camera_info: bool = False

        # History recording
        self.key_pressed: Set[int] = set()
        self.controls_history: Dict[str, List[int]] = {}
        self.current_frame: int = 0

        # Logic to clear conflicting keys
        self.easy_control: Set[int] = {32, 262, 263, 264, 265}

    def new_frame(self, cam: Any) -> None:
        self.current_frame += 1

    def apply_controls_each_frame_render_mode(self) -> None:
        """Replay recorded controls."""
        if str(self.current_frame) in self.controls_history:
            for k in self.controls_history[str(self.current_frame)]:
                self.key_callback(keycode=k, render_mode=True)

    def apply_controls_each_frame(self) -> None:
        if self.render_mode:
            self.apply_controls_each_frame_render_mode()

        # Robot Movements
        # Forward (Up Arrow)
        if 265 in self.key_pressed:
            self.physics.apply_control(acceleration_factor=1.0)
        # Turn Left (Left Arrow)
        elif 263 in self.key_pressed:
            self.physics.apply_control(rotation_factor=1.0)
        # Turn Right (Right Arrow)
        elif 262 in self.key_pressed:
            self.physics.apply_control(rotation_factor=-1.0)
        # Backward (Down Arrow)
        elif 264 in self.key_pressed:
            self.physics.apply_control(acceleration_factor=-1.0)
        # Space Key (Stop)
        elif 32 in self.key_pressed:
            self.physics.apply_control(decceleration_factor=0.2)

    def key_callback(self, keycode: int, render_mode: bool = False) -> None:
        if not render_mode:
            # Record history
            if str(self.current_frame) not in self.controls_history:
                self.controls_history[str(self.current_frame)] = [keycode]
            else:
                self.controls_history[str(self.current_frame)].append(keycode)

            # Commands
            if keycode == ord('c') or keycode == ord('C'):
                self.display_camera_info = True
            elif keycode == ord('s') or keycode == ord('S'):
                with open(CTRL_SAVE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.controls_history, f)
                print(f"Saved control history at path : `{CTRL_SAVE_PATH}` !")
            elif keycode == ord('q') or keycode == ord('Q') or keycode == 256: # ESC
                self.quit_requested = True

        if self.render_mode and not render_mode:
            return

        # Camera Modes
        if keycode == ord('1'):
            self.camera.set_mode("free")
        elif keycode == ord('2'):
            self.camera.set_mode("follow_robot")
        elif keycode == ord('3'):
            self.camera.set_mode("top_down")
        else:
            # Input handling
            if keycode in self.key_pressed:
                self.key_pressed.remove(keycode)
            else:
                self.key_pressed.add(keycode)

            # Easy control exclusive logic
            if keycode in self.easy_control:
                for kk in list(self.key_pressed):
                    if kk != keycode and kk in self.easy_control:
                        self.key_pressed.remove(kk)