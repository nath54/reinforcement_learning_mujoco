"""
Main entry point for the robot RL simulation.
"""
from typing import Any, cast
import os
import json
import argparse
import numpy as np
from numpy.typing import NDArray
import mujoco
from mujoco import viewer as viewer_
from tqdm import tqdm
import mediapy as media

from config import CTRL_SAVE_PATH, RENDER_WIDTH, RENDER_HEIGHT
from scene import RootWorldScene
from physics import Physics
from camera import Camera
from controls import Controls
from track import TrackRobot
from vision_sensor import VisionSensor
from state import State

# Fix to remove pylance type hinting errors with mujoco.viewer stubs errors
viewer: Any = cast(Any, viewer_)


class Main:
    """Main simulation controller."""
    
    @staticmethod
    def state_step(state: State) -> State:
        """Execute one simulation step."""
        # Get vision sensor readings (array of distances)
        # Shape: (num_rays,), dtype: np.float32
        # Values: -1.0 if no collision, distance if collision detected
        vision_distances = state.vision_sensor.get_sensor_readings()
        
        # Print debug info less frequently to improve performance
        if state.mj_data.time % 0.5 < state.mj_model.opt.timestep:
            # Count active sensors (distance >= 0)
            active_rays = np.sum(vision_distances >= 0)
            if active_rays > 0:
                print(f"Vision: {active_rays}/{len(vision_distances)} rays detecting obstacles")
                # Show distances for rays that hit something
                hit_indices = np.where(vision_distances >= 0)[0]
                print(f"  Hit distances: {vision_distances[hit_indices]}")
            else:
                print(f"Vision: Clear (no obstacles detected)")
        
        state.robot_track.track()
        
        state.controls.new_frame(state.viewer_instance.cam)
        
        state.controls.apply_controls_each_frame()
        
        # Print camera info when requested
        if state.controls.display_camera_info:
            print(state.viewer_instance.cam)
            robot_id: int = mujoco.mj_name2id(
                state.mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            pos: float | NDArray[np.float32] = state.mj_data.xpos[robot_id]
            quat: float | NDArray[np.float32] = state.mj_data.xquat[robot_id]
            print(f"Robot position: {pos}")
            print(f"Robot orientation (quat): {quat}")
            state.controls.display_camera_info = False
        
        state.physics.apply_additionnal_physics()
        
        mujoco.mj_step(
            m=state.mj_model,
            d=state.mj_data,
            nstep=1
        )
        
        state.camera.update_viewer_camera(
            cam=state.viewer_instance.cam,
            model=state.mj_model,
            data=state.mj_data
        )
        
        state.viewer_instance.sync()
        
        return state
    
    @staticmethod
    def main(render_mode: bool = False) -> None:
        """Main simulation loop with interactive viewer."""
        root_scene: RootWorldScene = RootWorldScene(
            sensor_boxes_enabled=True,
            sensor_box_size=0.3,
            sensor_layers=2,
            boxes_per_layer=8
        )
        
        root_scene.construct_scene(
            floor_type="standard",
            robot_height=1.0
        )
        
        # Initialize Vision Sensor (raycasting)
        # num_rays parameter replaces sensor_layers * boxes_per_layer
        num_rays = 16
        vision_sensor: VisionSensor = VisionSensor(
            model=root_scene.mujoco_model,
            data=root_scene.mujoco_data,
            robot_body_name="robot",
            num_rays=num_rays,
            max_distance=5.0,
            height_offset=0.0,
            ray_pattern="circle"
        )
        
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )
        
        camera: Camera = Camera()
        
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=render_mode
        )
        
        robot_track: TrackRobot = TrackRobot(mujoco_data_scene=root_scene.mujoco_data)
        
        if render_mode:
            if not os.path.exists(CTRL_SAVE_PATH):
                raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")
            
            with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
                controls.controls_history = json.load(f)
        
        print("")
        print("Controls:")
        print("  'up arrow': Robot goes forward (key control)")
        print("  'down arrow': Robot goes backward (key control)")
        print("  'left arrow': Robot goes left (key control)")
        print("  'right arrow': Robot goes right (key control)")
        print("  '1': Free camera (mouse control)")
        print("  '2': Follow robot (3rd person)")
        print("  '3': Top-down camera")
        print("  'c' or 'C': Print camera parameters")
        print("  'q' or 'Q': Quit")
        print("  ESC: Quit")
        print("")
        
        viewer_instance: Any
        
        with viewer.launch_passive(
                root_scene.mujoco_model,
                root_scene.mujoco_data,
                key_callback=controls.key_callback
            ) as viewer_instance:
            
            state: State = State(
                mj_model=root_scene.mujoco_model,
                mj_data=root_scene.mujoco_data,
                physics=physics,
                camera=camera,
                controls=controls,
                viewer_instance=viewer_instance,
                robot_track=robot_track,
                robot=root_scene.robot,
                vision_sensor=vision_sensor
            )
            
            # Mainloop
            while viewer_instance.is_running() and not state.controls.quit_requested:
                state = Main.state_step(state)
        
        robot_track.plot_tracking()
    
    @staticmethod
    def main_video_render() -> None:
        """Render simulation to video file."""
        root_scene: RootWorldScene = RootWorldScene()
        root_scene.construct_scene(
            floor_type="standard",
            robot_height=1.0
        )
        
        physics: Physics = Physics(
            mujoco_model_scene=root_scene.mujoco_model,
            mujoco_data_scene=root_scene.mujoco_data
        )
        
        camera: Camera = Camera()
        
        controls: Controls = Controls(
            physics=physics,
            camera=camera,
            render_mode=True
        )
        
        if not os.path.exists(CTRL_SAVE_PATH):
            raise UserWarning(f"Error: there is no saved control files at path `{CTRL_SAVE_PATH}` !")
        
        with open(CTRL_SAVE_PATH, "r", encoding="utf-8") as f:
            controls.controls_history = json.load(f)
        
        # Create a camera
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        
        # Camera parameters
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        
        # Set good camera position and orientation parameters
        cam.azimuth = 1.01171875
        cam.elevation = -16.6640625
        cam.lookat = np.array(
            [1.55633679e-04, -4.88295545e-02, 1.05485916e+00]
        )
        
        # Set up parameters
        framerate: int = 60
        skip_factor: int = framerate
        n_frames: int = max([int(f_id) for f_id in controls.controls_history.keys()])
        
        # Simulate and display video
        mujoco.mj_resetData(root_scene.mujoco_model, root_scene.mujoco_data)
        
        with media.VideoWriter('rendered_video.mp4', shape=(RENDER_HEIGHT, RENDER_WIDTH), fps=framerate) as writer:
            with mujoco.Renderer(root_scene.mujoco_model, RENDER_HEIGHT, RENDER_WIDTH) as renderer:
                for frame_idx in tqdm(range(min(2000000, n_frames))):
                    controls.new_frame(cam)
                    controls.apply_controls_each_frame()
                    
                    # Quit if requested
                    if controls.quit_requested:
                        break
                    
                    physics.apply_additionnal_physics()
                    
                    mujoco.mj_step(
                        m=root_scene.mujoco_model,
                        d=root_scene.mujoco_data,
                        nstep=1
                    )
                    
                    camera.update_viewer_camera(
                        cam=cam,
                        model=root_scene.mujoco_model,
                        data=root_scene.mujoco_data
                    )
                    
                    if frame_idx % skip_factor == 0:
                        renderer.update_scene(root_scene.mujoco_data, cam)
                        pixels: NDArray[np.int8] = renderer.render()
                        writer.add_image(pixels)


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--render_mode', action="store_true", default=False)
    parser.add_argument('--render_video', action="store_true", default=False)
    args: argparse.Namespace = parser.parse_args()
    
    if args.render_video:
        Main.main_video_render()
    else:
        Main.main(render_mode=args.render_mode)
