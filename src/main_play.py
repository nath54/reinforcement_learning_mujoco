"""
Play with trained model in the same scene than interactive mode

Usage:
    python -m src.main_play --config config/main.yaml --model_path path/to/model.pt --live_vision

Or:
    python -m src.main --play --config config/main.yaml --model_path path/to/model.pt --live_vision
"""

from typing import Any, cast

import os
import time
import argparse

import numpy as np
from numpy.typing import NDArray

import torch

import mujoco
from mujoco import viewer as viewer_

from .core.config_loader import load_config
from .core.types import GlobalConfig
from .environment.wrapper import CorridorEnv
from .algorithms.ppo import PPOAgent
from .simulation.generator import SceneBuilder
from .simulation.physics import Physics
from .simulation.sensors import Camera
from .simulation.controls import Controls
from .utils.tracking import TrackRobot

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not found. Live vision disabled.")

# Type cast to Any to avoid mypy errors
viewer: Any = cast(Any, viewer_)


# Play with trained model in interactive mode
def play(
    config: GlobalConfig,
    model_path: str,
    live_vision: bool = False
) -> None:

    """
    Play with trained model in interactive mode
    """

    print(f"Playing with model: {model_path}")

    # Setup scene
    scene: SceneBuilder = SceneBuilder(config)
    scene.build()

    # Setup physics
    physics: Physics = Physics(scene.mujoco_model, scene.mujoco_data)

    # Setup camera and controls
    camera: Camera = Camera()
    controls: Controls = Controls(physics, camera, render_mode=False)

    # Setup robot tracking
    robot_track: TrackRobot = TrackRobot(scene.mujoco_data)

    # Setup agent
    view_range_grid: int = int(config.simulation.robot_view_range / config.simulation.env_precision)
    vision_width: int = 2 * view_range_grid
    vision_height: int = 2 * view_range_grid
    vision_size: int = vision_width * vision_height
    state_dim: int = vision_size + 13

    # Setup action space
    #
    action_dim: int
    #
    if config.robot.control_mode == "discrete_direction":
        action_dim = 4
    elif config.robot.control_mode == "continuous_vector":
        action_dim = 2
    else:
        action_dim = 4

    # Setup agent
    agent: PPOAgent = PPOAgent(
        state_dim, action_dim, (vision_width, vision_height),
        lr=config.training.learning_rate,
        gamma=config.training.gamma,
        K_epochs=config.training.k_epochs,
        eps_clip=config.training.eps_clip,
        gae_lambda=config.training.gae_lambda,
        entropy_coeff=config.training.entropy_coeff,
        value_loss_coeff=config.training.value_loss_coeff,
        grad_clip_max_norm=config.training.grad_clip_max_norm,
        action_std_init=config.model.action_std_init,
        action_std_min=config.model.action_std_min,
        action_std_max=config.model.action_std_max,
        actor_hidden_gain=config.model.actor_hidden_gain,
        actor_output_gain=config.model.actor_output_gain,
        control_mode=config.robot.control_mode
    )

    # Load model
    try:
        agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
        agent.policy.eval()
        #
        print("Model loaded successfully")
    #
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup environment for observation
    env: CorridorEnv = CorridorEnv(config)
    env.reset()

    # Print controls
    print("\nControls:")
    print("  '1': Free camera")
    print("  '2': Follow robot")
    print("  '3': Top-down camera")
    print("  'Q' or ESC: Quit\n")

    # Launch viewer
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

        # Reset robot position
        robot_id = mujoco.mj_name2id(scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
        scene.mujoco_data.xpos[robot_id][2] = 0.2

        # Initialize variables
        previous_action: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        crt_step: int = 0

        # Main loop
        while viewer_instance.is_running() and not controls.quit_requested:

            # Get observation
            obs: NDArray[np.float64] = env.get_observation()

            # Visualize if requested
            if live_vision and cv2 is not None:

                # Get vision data
                vision_data: NDArray[np.float64] = obs[:vision_size]

                # Reshape and resize vision data
                vision_img: NDArray[np.float64] = vision_data.reshape((vision_height, vision_width))
                vision_img_disp: NDArray[np.float64] = cv2.resize(vision_img, (400, 400), interpolation=cv2.INTER_NEAREST)

                # Normalize and convert to BGR
                vision_img_disp = (vision_img_disp - vision_img_disp.min()) / (vision_img_disp.max() - vision_img_disp.min() + 1e-6)
                vision_img_disp = (vision_img_disp * 255).astype(np.uint8)
                vision_img_color: NDArray[np.uint8] = cv2.cvtColor(vision_img_disp, cv2.COLOR_GRAY2BGR)

                # Get actions probabilities from the agent
                raw_action_stats: NDArray[np.float64] = agent.get_action_statistics(obs)

                # Display action statistics:

                # Initialize y offset which is used for the new lines
                y_offset: int = 30

                # Display title of the action statistics
                cv2.putText(vision_img_color, "Agent Output:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display action statistics for each action (green for positive, red for negative)
                #
                i: int
                val: float
                #
                for i, val in enumerate(raw_action_stats):

                    # Text content
                    text: str = f"Out[{i}]: {val:.3f}"

                    # Text color
                    color: Tuple[int, int, int] = (0, 0, 255) if val < 0 else (255, 0, 0)

                    # Display the text
                    cv2.putText(vision_img_color, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                    # Calculate the length of the bar
                    bar_len: int = int(abs(val) * 50)

                    # Display the bar
                    cv2.rectangle(vision_img_color, (150, y_offset - 10), (150 + bar_len, y_offset), color, -1)

                    # Move to the next line
                    y_offset += 20

                # Display the robot vision
                cv2.imshow("Robot Vision", vision_img_color)

                # Wait for a key press
                cv2.waitKey(1)

            #
            action: float | NDArray[np.float64]

            # The warmup steps let the robot to stabilize before starting the training
            if crt_step >= config.simulation.warmup_steps:

                # Get action statistics from the agent
                raw_stats: NDArray[np.float64] = agent.get_action_statistics(obs)

                # If the control mode is discrete, get the action index with the highest probability
                if config.robot.control_mode == "discrete_direction":
                    #
                    action = float(np.argmax(raw_stats))

                # If the control mode is continuous, get the action from the agent
                else:
                    #
                    action = raw_stats

            # If the warmup steps are not reached, set the action to 0 (no action)
            else:

                #
                if config.robot.control_mode == "discrete_direction":
                    action = 0

                #
                else:
                    action = np.zeros(action_dim, dtype=np.float64)

                # Print warmup progress
                if crt_step % 100 == 0:
                    print(f"Warmup: {crt_step}/{config.simulation.warmup_steps}", end="\r")

            # Ensure the robot wheels speed is set to 0 during the warmup steps
            if crt_step < config.simulation.warmup_steps:
                physics.robot_wheels_speed[:] = 0

            # If we are past the warmup steps, apply the action
            else:

                # Initialize target speeds
                target_speeds: NDArray[np.float64] = np.zeros(4, dtype=np.float64)

                # Get the maximum speed
                max_speed: float = config.robot.max_speed

                # If the control mode is discrete, set the target speeds directly
                if config.robot.control_mode == "discrete_direction":
                    #
                    action_idx: int = int(action)
                    #
                    if action_idx == 0:
                        target_speeds[:] = max_speed
                    #
                    elif action_idx == 1:
                        target_speeds[:] = -max_speed
                    #
                    elif action_idx == 2:
                        target_speeds = np.array([-max_speed, max_speed, -max_speed, max_speed])
                    #
                    elif action_idx == 3:
                        target_speeds = np.array([max_speed, -max_speed, max_speed, -max_speed])
                    #
                    physics.set_wheel_speeds_directly(target_speeds)

                # If the control mode is continuous, set the target speeds based on the action
                elif config.robot.control_mode == "continuous_vector":

                    # action[0] is the speed
                    speed: NDArray[np.float64] = np.clip(action[0], -1.0, 1.0)

                    # action[1] is the rotation
                    rotation: NDArray[np.float64] = np.clip(action[1], -1.0, 1.0)

                    # Compute the left and right speeds
                    left_speed: NDArray[np.float64] = speed + rotation
                    right_speed: NDArray[np.float64] = speed - rotation

                    # Construct the raw wheel speeds
                    raw_wheel_speeds: NDArray[np.float64] = np.array([left_speed, right_speed, left_speed, right_speed])
                    raw_wheel_speeds = np.clip(raw_wheel_speeds, -1.0, 1.0)

                    # Scale the raw wheel speeds to the maximum speed
                    target_speeds = raw_wheel_speeds * max_speed

                    # Set the wheel speeds
                    physics.set_wheel_speeds_directly(target_speeds)

                # If the control mode is continuous, set the target speeds based on the action
                else:
                    # Clip the action to the range [-1, 1]
                    action = np.clip(action, -1.0, 1.0)

                    # Scale the action to the maximum speed
                    target_speeds = action * max_speed

                    # Set the wheel speeds
                    physics.set_wheel_speeds_directly(target_speeds)

            # Number of physics steps to perform
            n_repeats: int = config.simulation.action_repeat if crt_step >= config.simulation.warmup_steps else 1

            # Perform the physics steps
            for _ in range(n_repeats):

                # Apply additional physics
                physics.apply_additionnal_physics()

                # Step the physics
                mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)

            # Update camera
            camera.update_viewer_camera(viewer_instance.cam, scene.mujoco_model, scene.mujoco_data)

            # Sync viewer
            viewer_instance.sync()

            # Track robot
            robot_track.track()

            # Increment step counter
            crt_step += 1

            # Timing
            time.sleep(1.0 / 400.0)

    # Plot tracking
    robot_track.plot_tracking()

    # Close live vision
    if live_vision and cv2 is not None:
        cv2.destroyAllWindows()


# Main function
def main() -> None:

    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Robot Corridor RL - Play Mode")
    #
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for play mode')
    parser.add_argument('--live_vision', action='store_true', help='Show live vision window')
    #
    args: argparse.Namespace = parser.parse_args()

    # Load configuration
    cfg: Config = load_config(args.config)

    # Get model path
    model_path: str = args.model_path if args.model_path else cfg.training.model_path

    # Play
    play(cfg, model_path, args.live_vision)


# This script can also be directly run from the command line instead of using src.main
if __name__ == "__main__":
    #
    main()
