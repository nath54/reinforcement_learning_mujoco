"""
Control handling for keyboard input and robot commands.
"""
from typing import Any
import json

from config import CTRL_SAVE_PATH
from physics import Physics
from camera import Camera


class Controls:
    """Manages keyboard controls and robot commands."""
    
    def __init__(
        self,
        physics: Physics,
        camera: Camera,
        render_mode: bool = False
    ) -> None:
        self.physics: Physics = physics
        self.camera: Camera = camera
        
        self.render_mode: bool = render_mode
        
        self.quit_requested: bool = False
        self.display_camera_info: bool = False
        
        self.key_pressed: set[int] = set()
        
        self.controls_history: dict[str, list[int]] = {}
        
        self.current_frame: int = 0
        
        self.easy_control: set[int] = {32, 262, 263, 264, 265}
    
    def new_frame(self, cam: Any) -> None:
        """Start a new frame."""
        self.current_frame += 1
    
    def apply_controls_each_frame_render_mode(self) -> None:
        """Apply controls from history in render mode."""
        if str(self.current_frame) in self.controls_history:
            for k in self.controls_history[str(self.current_frame)]:
                self.key_callback(keycode=k, render_mode=True)
    
    def apply_controls_each_frame(self) -> None:
        """Apply controls each frame."""
        if self.render_mode:
            self.apply_controls_each_frame_render_mode()
        
        # Robot Movements
        # Forward
        if 265 in self.key_pressed:
            self.physics.apply_robot_ctrl_movement(acceleration_factor=1.0)
        # Turn Left
        elif 263 in self.key_pressed:
            self.physics.apply_robot_ctrl_movement(rotation_factor=1.0)
        # Turn Right
        elif 262 in self.key_pressed:
            self.physics.apply_robot_ctrl_movement(rotation_factor=-1.0)
        # Backward
        elif 264 in self.key_pressed:
            self.physics.apply_robot_ctrl_movement(acceleration_factor=-1.0)
        # Space Key, robot stops
        elif 32 in self.key_pressed:
            self.physics.apply_robot_ctrl_movement(decceleration_factor=0.2)
    
    def key_callback(self, keycode: int, render_mode: bool = False) -> None:
        """Handle keyboard input."""
        if not render_mode:
            if not str(self.current_frame) in self.controls_history:
                self.controls_history[str(self.current_frame)] = [keycode]
            else:
                self.controls_history[str(self.current_frame)].append(keycode)
            
            # Display Camera Informations
            if keycode == ord('c') or keycode == ord('C'):
                self.display_camera_info = True
            
            # Save the control history
            elif keycode == ord('s') or keycode == ord('S'):
                with open(CTRL_SAVE_PATH, "w", encoding="utf-8") as f:
                    json.dump(self.controls_history, f)
                print(f"Saved control history at path : `{CTRL_SAVE_PATH}` !")
            
            # Quit, with 'Q' or 'Esc' Keys
            elif keycode == ord('q') or keycode == ord('Q'):
                self.quit_requested = True
            elif keycode == 256:
                self.quit_requested = True
        
        if self.render_mode and not render_mode:
            return
        
        # Camera Mode Switching
        if keycode == ord('1'):
            self.camera.set_mode("free")
        elif keycode == ord('2'):
            self.camera.set_mode("follow_robot")
        elif keycode == ord('3'):
            self.camera.set_mode("top_down")
        else:
            if keycode in self.key_pressed:
                self.key_pressed.remove(keycode)
            else:
                self.key_pressed.add(keycode)
            
            if keycode in self.easy_control:
                for kk in list(self.key_pressed):
                    if kk != keycode and kk in self.easy_control:
                        self.key_pressed.remove(kk)
