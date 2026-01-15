import argparse
import os
import time
import json
from typing import Any, cast

import numpy as np
import mujoco
from mujoco import viewer as viewer_

from src.core.config_loader import load_config
from src.core.types import GlobalConfig
from src.simulation.generator import SceneBuilder
from src.simulation.physics import Physics
from src.simulation.sensors import Camera
from src.simulation.controls import Controls
from src.utils.tracking import TrackRobot

viewer: Any = cast(Any, viewer_)


def interactive(config: GlobalConfig, render_mode: bool = False) -> None:
    """Interactive keyboard control mode"""
    print("Starting interactive mode...")

    scene = SceneBuilder(config)
    scene.build()

    physics = Physics(scene.mujoco_model, scene.mujoco_data)
    camera = Camera()
    controls = Controls(physics, camera, render_mode=render_mode)
    robot_track = TrackRobot(scene.mujoco_data)

    if render_mode:
        if not os.path.exists("saved_control.json"):
            raise FileNotFoundError("No saved control file found!")
        with open("saved_control.json", "r") as f:
            controls.controls_history = json.load(f)

    print("\nControls:")
    print("  Arrow Up: Forward")
    print("  Arrow Down: Backward")
    print("  Arrow Left: Turn Left")
    print("  Arrow Right: Turn Right")
    print("  Space: Stop")
    print("  '1': Free camera")
    print("  '2': Follow robot")
    print("  '3': Top-down camera")
    print("  'C': Print camera info")
    print("  'S': Save control history")
    print("  'Q' or ESC: Quit\n")

    with viewer.launch_passive(
        scene.mujoco_model,
        scene.mujoco_data,
        key_callback=controls.key_callback
    ) as viewer_instance:

        # Initialize camera
        viewer_instance.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer_instance.cam.azimuth = 1.01171875
        viewer_instance.cam.elevation = -16.6640625
        viewer_instance.cam.lookat = np.array([1.55633679e-04, -4.88295545e-02, 1.05485916e+00])

        target_dt = 1.0 / 400.0

        while viewer_instance.is_running() and not controls.quit_requested:
            start_time = time.time()

            controls.new_frame(viewer_instance.cam)
            controls.apply_controls_each_frame()

            if controls.display_camera_info:
                print(viewer_instance.cam)
                robot_id = mujoco.mj_name2id(scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
                pos = scene.mujoco_data.xpos[robot_id]
                print(f"Robot position: {pos}")
                controls.display_camera_info = False

            physics.apply_additionnal_physics()
            mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)

            camera.update_viewer_camera(viewer_instance.cam, scene.mujoco_model, scene.mujoco_data)
            viewer_instance.sync()
            robot_track.track()

            elapsed = time.time() - start_time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    robot_track.plot_tracking()


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Corridor RL - Interactive Mode")
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    parser.add_argument('--render_mode', action='store_true', help='Replay saved controls')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    interactive(cfg, args.render_mode)


if __name__ == "__main__":
    main()
