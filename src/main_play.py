"""
Play with trained model in the same scene than interactive mode

Usage:
    python -m src.main_play --config config/main.yaml --model_path path/to/model.pt --live_vision

Or:
    python -m src.main --play --config config/main.yaml --model_path path/to/model.pt --live_vision
"""

from typing import Any, Tuple, cast

import time
import argparse

import numpy as np
from numpy.typing import NDArray

import torch

import mujoco
from mujoco import viewer as viewer_

from .core.config_loader import load_config
from .core.types import GlobalConfig
from .environment.wrapper import SimulationEnv
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
def play(config: GlobalConfig, model_path: str, live_vision: bool = False) -> None:
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

    # Setup robot tracking with goal position
    robot_track: TrackRobot = TrackRobot(
        scene.mujoco_data, goal_position=scene.goal_position
    )

    # Setup agent dimensions
    view_range_grid: int = int(
        config.simulation.robot_view_range / config.simulation.env_precision
    )
    vision_width: int = 2 * view_range_grid
    vision_height: int = 2 * view_range_grid
    vision_size: int = vision_width * vision_height

    # Base state vector: 13 dimensions
    # Goal-relative coords add 4 dimensions if include_goal is True
    goal_dims: int = 4 if config.model.include_goal else 0
    state_vector_dim: int = 13 + goal_dims
    state_dim: int = vision_size + state_vector_dim

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
        state_dim,
        action_dim,
        (vision_width, vision_height),
        state_vector_dim=state_vector_dim,
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
        control_mode=config.robot.control_mode,
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

    # Setup environment for observation (use same scene to share MuJoCo data)
    env: SimulationEnv = SimulationEnv(config, scene=scene)
    env.reset()

    # Print controls
    print("\nControls:")
    print("  '1': Free camera")
    print("  '2': Follow robot")
    print("  '3': Top-down camera")
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

        # Reset robot position
        robot_id = mujoco.mj_name2id(
            scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot"
        )
        scene.mujoco_data.xpos[robot_id][2] = 0.2

        # Initialize variables
        previous_action: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        crt_step: int = 0

        # Reward tracking (for live_vision display)
        episode_reward: float = 0.0
        step_reward: float = 0.0

        # Goal tracking (for auto-reset)
        goals_reached: int = 0

        # Main loop
        while viewer_instance.is_running() and not controls.quit_requested:
            # Get observation
            obs: NDArray[np.float64] = env.get_observation()

            # Visualize if requested
            if live_vision and cv2 is not None:
                # Get vision data
                vision_data: NDArray[np.float64] = obs[:vision_size]

                # Get state vector
                state_vector: NDArray[np.float64] = obs[vision_size:]

                # Reshape and resize vision data
                vision_img: NDArray[np.float64] = vision_data.reshape(
                    (vision_height, vision_width)
                )
                vision_img_disp: NDArray[np.float64] = cv2.resize(
                    vision_img, (400, 400), interpolation=cv2.INTER_NEAREST
                )

                # Normalize and convert to BGR
                vision_img_disp = (vision_img_disp - vision_img_disp.min()) / (
                    vision_img_disp.max() - vision_img_disp.min() + 1e-6
                )
                vision_img_disp = (vision_img_disp * 255).astype(np.uint8)
                vision_img_color: NDArray[np.uint8] = cv2.cvtColor(
                    vision_img_disp, cv2.COLOR_GRAY2BGR
                )

                # Get actions probabilities from the agent
                raw_action_stats: NDArray[np.float64] = agent.get_action_statistics(obs)

                # Display action statistics and input info:

                # Initialize y offset which is used for the new lines
                y_offset: int = 20

                # Display agent input statistics
                cv2.putText(
                    vision_img_color,
                    "Agent Input:",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                y_offset += 15

                # Vision statistics
                cv2.putText(
                    vision_img_color,
                    f"Vision: min={vision_data.min():.2f} max={vision_data.max():.2f}",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (200, 200, 200),
                    1,
                )
                y_offset += 15

                # State vector info (first 13 components are: pos(3), rot(3), vel(3), prev_action(4))
                if len(state_vector) >= 13:
                    cv2.putText(
                        vision_img_color,
                        f"Pos: [{state_vector[0]:.2f}, {state_vector[1]:.2f}, {state_vector[2]:.2f}]",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                    )
                    y_offset += 15
                    cv2.putText(
                        vision_img_color,
                        f"Vel: [{state_vector[6]:.2f}, {state_vector[7]:.2f}, {state_vector[8]:.2f}]",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                    )
                    y_offset += 15

                # Goal and reward info
                if config.model.include_goal and scene.goal_position is not None:
                    y_offset += 5  # Add spacing

                    # Display absolute goal position
                    cv2.putText(
                        vision_img_color,
                        "Goal (absolute):",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    y_offset += 15
                    cv2.putText(
                        vision_img_color,
                        f"  [{scene.goal_position.x:.2f}, {scene.goal_position.y:.2f}, {scene.goal_position.z:.2f}]",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (200, 200, 200),
                        1,
                    )
                    y_offset += 15

                    # Display goal-relative coordinates from agent input (part of state vector)
                    if len(state_vector) >= 17:
                        dx = state_vector[13]  # dx (normalized)
                        dy = state_vector[14]  # dy (normalized)
                        distance = state_vector[15]  # distance
                        angle = (
                            state_vector[16] * np.pi
                        )  # angle in radians (was normalized)

                        cv2.putText(
                            vision_img_color,
                            "Goal (relative from input):",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 0),
                            1,
                        )
                        y_offset += 15
                        cv2.putText(
                            vision_img_color,
                            f"  dx={dx:.2f} dy={dy:.2f} d={distance:.2f}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                        y_offset += 15
                        cv2.putText(
                            vision_img_color,
                            f"  angle={angle:.2f} rad ({np.degrees(angle):.1f}deg)",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (200, 200, 200),
                            1,
                        )
                        y_offset += 15

                    y_offset += 5  # Add spacing

                    # Display reward info (using environment's reward strategy for modularity)
                    cv2.putText(
                        vision_img_color,
                        "Reward:",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 0),
                        1,
                    )
                    y_offset += 15
                    reward_color: Tuple[int, int, int] = (
                        (0, 255, 0) if step_reward >= 0 else (0, 0, 255)
                    )
                    cv2.putText(
                        vision_img_color,
                        f"  step={step_reward:+.3f} total={episode_reward:+.2f}",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        reward_color,
                        1,
                    )
                    y_offset += 15

                y_offset += 5  # Add spacing

                # Display title of the action statistics
                cv2.putText(
                    vision_img_color,
                    "Agent Output:",
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
                y_offset += 15

                # Display action statistics for each action (green for positive, red for negative)
                #
                i: int
                val: float
                #
                for i, val in enumerate(raw_action_stats):
                    # Text content
                    text: str = f"Out[{i}]: {val:.3f}"

                    # Text color
                    color: Tuple[int, int, int] = (
                        (0, 0, 255) if val < 0 else (255, 0, 0)
                    )

                    # Display the text
                    cv2.putText(
                        vision_img_color,
                        text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

                    # Calculate the length of the bar
                    bar_len: int = int(abs(val) * 50)

                    # Display the bar
                    cv2.rectangle(
                        vision_img_color,
                        (150, y_offset - 10),
                        (150 + bar_len, y_offset),
                        color,
                        -1,
                    )

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
                    print(
                        f"Warmup: {crt_step}/{config.simulation.warmup_steps}", end="\r"
                    )

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
                        target_speeds = np.array(
                            [-max_speed, max_speed, -max_speed, max_speed]
                        )
                    #
                    elif action_idx == 3:
                        target_speeds = np.array(
                            [max_speed, -max_speed, max_speed, -max_speed]
                        )
                    #
                    physics.set_wheel_speeds_directly(target_speeds)

                    # Update environment's previous_action for reward calculation
                    env.previous_action = target_speeds / max_speed

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
                    raw_wheel_speeds: NDArray[np.float64] = np.array(
                        [left_speed, right_speed, left_speed, right_speed]
                    )
                    raw_wheel_speeds = np.clip(raw_wheel_speeds, -1.0, 1.0)

                    # Scale the raw wheel speeds to the maximum speed
                    target_speeds = raw_wheel_speeds * max_speed

                    # Set the wheel speeds
                    physics.set_wheel_speeds_directly(target_speeds)

                    # Update environment's previous_action for reward calculation
                    env.previous_action = raw_wheel_speeds

                # If the control mode is continuous, set the target speeds based on the action
                else:
                    # Clip the action to the range [-1, 1]
                    action = np.clip(action, -1.0, 1.0)

                    # Scale the action to the maximum speed
                    target_speeds = action * max_speed

                    # Set the wheel speeds
                    physics.set_wheel_speeds_directly(target_speeds)

                    # Update environment's previous_action for reward calculation
                    # Convert target_speeds to normalized action (same as environment does)
                    env.previous_action = target_speeds / max_speed

            # Number of physics steps to perform
            n_repeats: int = (
                config.simulation.action_repeat
                if crt_step >= config.simulation.warmup_steps
                else 1
            )

            # Perform the physics steps
            for _ in range(n_repeats):
                # Apply additional physics
                physics.apply_additionnal_physics()

                # Step the physics
                mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)

            # Update camera
            camera.update_viewer_camera(
                viewer_instance.cam, scene.mujoco_model, scene.mujoco_data
            )

            # Sync viewer
            viewer_instance.sync()

            # Track robot (includes distance to goal)
            robot_track.track()

            # Calculate reward using environment's reward strategy (modular - respects reward config changes)
            if (
                crt_step >= config.simulation.warmup_steps
                and scene.goal_position is not None
            ):
                # Get current robot state for reward calculation
                robot_pos_vec = scene.mujoco_data.xpos[robot_id]
                robot_vel_vec = scene.mujoco_data.cvel[robot_id][:3]

                # Convert to Vec3 for reward calculation
                from src.core.types import Vec3

                current_pos = Vec3(robot_pos_vec[0], robot_pos_vec[1], robot_pos_vec[2])
                current_vel = Vec3(robot_vel_vec[0], robot_vel_vec[1], robot_vel_vec[2])

                # Determine if stuck or backward (simplified checks for play mode)
                velocity_magnitude = np.linalg.norm(robot_vel_vec[:2])  # XY velocity
                is_stuck = velocity_magnitude < 0.01
                is_backward = (
                    False  # Could be enhanced with more sophisticated detection
                )

                # Calculate step reward using environment's reward strategy (modular!)
                # Use env.previous_action which is already tracked by the environment
                step_reward = env.reward_strategy.compute(
                    current_pos,
                    current_vel,
                    scene.goal_position,
                    env.previous_action,
                    crt_step,
                    is_stuck,
                    is_backward,
                )

                # Accumulate episode reward
                episode_reward += step_reward
            else:
                # No reward during warmup or without goal
                step_reward = 0.0

            # Increment step counter
            crt_step += 1

            # Check if goal reached (auto-reset with scene regeneration)
            if scene.goal_position is not None:
                robot_pos_vec = scene.mujoco_data.xpos[robot_id]

                # Calculate distance to goal
                dist_to_goal = np.sqrt(
                    (scene.goal_position.x - robot_pos_vec[0]) ** 2
                    + (scene.goal_position.y - robot_pos_vec[1]) ** 2
                )

                # If goal reached, regenerate scene and reset
                if dist_to_goal < config.simulation.goal_radius:
                    goals_reached += 1

                    # Calculate total physical steps (agent steps * action_repeat)
                    physical_steps = crt_step * config.simulation.action_repeat

                    print(
                        f"\n\033[1;32mGoal reached: {goals_reached} times | Steps: {crt_step} agent, {physical_steps} physics | Reward: {episode_reward:.2f}\033[m"
                    )

                    # Reset environment (this will randomize goal for flat_world and reset physics)
                    env.reset()

                    # Reset tracking variables
                    episode_reward = 0.0
                    step_reward = 0.0
                    crt_step = 0

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
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Robot Corridor RL - Play Mode"
    )
    #
    parser.add_argument(
        "--config", type=str, default="config/main.yaml", help="Config file path"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Model path for play mode"
    )
    parser.add_argument(
        "--live_vision", action="store_true", help="Show live vision window"
    )
    #
    args: argparse.Namespace = parser.parse_args()

    # Load configuration
    cfg: GlobalConfig = load_config(args.config)

    # Get model path
    model_path: str = args.model_path if args.model_path else cfg.training.model_path

    # Play
    play(cfg, model_path, args.live_vision)


# This script can also be directly run from the command line instead of using src.main
if __name__ == "__main__":
    #
    main()
