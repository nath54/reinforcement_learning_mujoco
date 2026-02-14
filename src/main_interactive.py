"""
Play in interactive mode

There are two ways to use this mode:
    1. Play in real-time and optionally save controls with 'S' key
    2. Replay saved controls

Usage:
    python -m src.main_interactive --config config/main.yaml

Or:
    python -m src.main --interactive --config config/main.yaml
"""

from typing import Any, cast

import os
import time
import json
import argparse

import numpy as np
from numpy.typing import NDArray

import mujoco
from mujoco import viewer as viewer_

from .core.config_loader import load_config
from .core.types import GlobalConfig
from .simulation.generator import SceneBuilder
from .simulation.physics import Physics
from .simulation.sensors import Camera
from .simulation.controls import Controls
from .utils.tracking import TrackRobot

# Type cast to Any to avoid mypy errors
viewer: Any = cast(Any, viewer_)


# Interactive mode
def interactive(config: GlobalConfig, render_mode: bool = False) -> None:
    """
    Interactive keyboard control mode
    """

    # Print starting message
    print("Starting interactive mode...")

    # Setup scene
    scene: SceneBuilder = SceneBuilder(config)
    scene.build()

    # Setup physics
    physics: Physics = Physics(scene.mujoco_model, scene.mujoco_data)

    # Setup camera and controls
    camera: Camera = Camera()
    controls: Controls = Controls(physics, camera, render_mode=render_mode)

    # Setup robot tracking with goal position
    robot_track: TrackRobot = TrackRobot(
        scene.mujoco_data, goal_position=scene.goal_position
    )

    # Load saved controls if in render mode
    if render_mode:
        # Check if saved control file exists
        if not os.path.exists("saved_control.json"):
            raise FileNotFoundError("No saved control file found!")

        # Load saved controls
        with open("saved_control.json", "r") as f:
            controls.controls_history = json.load(f)

    # Print controls
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

    # Launch viewer
    with viewer.launch_passive(
        scene.mujoco_model, scene.mujoco_data, key_callback=controls.key_callback
    ) as viewer_instance:
        # Initialize camera
        viewer_instance.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer_instance.cam.azimuth = 1.01171875
        viewer_instance.cam.elevation = -16.6640625
        viewer_instance.cam.lookat = np.array(
            [1.55633679e-04, -4.88295545e-02, 1.05485916e00]
        )

        # Target time between two frames (dt for delta time)
        target_dt: float = 1.0 / 400.0

        # Main loop
        while viewer_instance.is_running() and not controls.quit_requested:
            # Get start time
            start_time: float = time.time()

            # Indicates to the controls that a new frame is starting
            controls.new_frame(viewer_instance.cam)

            # Apply controls for each frame
            controls.apply_controls_each_frame()

            # Display camera info if requested
            if controls.display_camera_info:
                # Display camera info
                print(viewer_instance.cam)
                # Get the robot id to get its position from mujoco data
                robot_id: int = mujoco.mj_name2id(
                    scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
                )
                # Get the robot position
                pos: NDArray[np.float64] = scene.mujoco_data.xpos[robot_id]
                # Print the robot position
                print(f"Robot position: {pos}")
                # Reset display camera info
                controls.display_camera_info = False

            # Apply additionnal physics
            physics.apply_additionnal_physics()

            # Step the physics
            mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)

            # Update the camera
            camera.update_viewer_camera(
                viewer_instance.cam, scene.mujoco_model, scene.mujoco_data
            )

            # Sync the viewer
            viewer_instance.sync()

            # Track the robot (includes distance to goal)
            robot_track.track()

            # Detect endless fall and reset
            robot_id = mujoco.mj_name2id(
                scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
            )
            if scene.mujoco_data.xpos[robot_id][2] < config.simulation.endless_fall_threshold:
                print(f"\nEndless fall detected (Z < {config.simulation.endless_fall_threshold}). Resetting simulation...")
                mujoco.mj_resetData(scene.mujoco_model, scene.mujoco_data)
                physics.reset()
                scene.mujoco_data.xpos[robot_id][2] = 0.2 # Warmup lift

            # Get elapsed time
            elapsed: float = time.time() - start_time

            # If the elapsed time is less than the target dt, sleep for the remaining time
            if elapsed < target_dt:
                time.sleep(target_dt - elapsed)

    # Plot the robot tracking
    robot_track.plot_tracking()


# Main function
def main() -> None:

    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Robot Corridor RL - Interactive Mode"
    )
    #
    parser.add_argument(
        "--config", type=str, default="config/main_custom_xml.yaml", help="Config file path"
    )
    parser.add_argument(
        "--render_mode", action="store_true", help="Replay saved controls"
    )
    #
    args = parser.parse_args()

    # Load configuration
    cfg: GlobalConfig = load_config(args.config)

    # Run interactive mode
    interactive(cfg, args.render_mode)


# This script can also be directly run from the command line instead of using src.main
if __name__ == "__main__":
    #
    main()
