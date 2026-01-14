from typing import Any, Set
import json
from src.simulation.physics import Physics
from src.simulation.sensors import Camera

class Controls:
    def __init__(self, physics: Physics, camera: Camera):
        self.physics = physics
        self.camera = camera
        self.pressed_keys: Set[int] = set()
        self.quit = False
    
    def key_callback(self, keycode: int):
        if keycode == 256 or keycode == ord('q'): # ESC or q
            self.quit = True
        elif keycode == ord('1'): self.camera.mode = "free"
        elif keycode == ord('2'): self.camera.mode = "follow"
        elif keycode == ord('3'): self.camera.mode = "top_down"
        else:
            if keycode in self.pressed_keys: self.pressed_keys.remove(keycode)
            else: self.pressed_keys.add(keycode)

    def apply(self):
        # 265: Up, 264: Down, 263: Left, 262: Right
        acc, rot = 0.0, 0.0
        if 265 in self.pressed_keys: acc = 1.0
        elif 264 in self.pressed_keys: acc = -1.0
        
        if 263 in self.pressed_keys: rot = 1.0
        elif 262 in self.pressed_keys: rot = -1.0
        
        self.physics.apply_control(acc, rot, 20.0)