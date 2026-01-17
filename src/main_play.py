import argparse
import os
import time
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
import mujoco
from mujoco import viewer as viewer_

from src.core.config_loader import load_config
from src.core.types import GlobalConfig
from src.environment.wrapper import CorridorEnv
from src.algorithms.ppo import PPOAgent
from src.simulation.generator import SceneBuilder
from src.simulation.physics import Physics
from src.simulation.sensors import Camera
from src.simulation.controls import Controls
from src.utils.tracking import TrackRobot

try:
    import cv2
except ImportError:
    cv2 = None
    print("Warning: opencv-python not found. Live vision disabled.")

viewer: Any = cast(Any, viewer_)


def play(config: GlobalConfig, model_path: str, live_vision: bool = False) -> None:
    """Play with trained model in interactive mode"""
    print(f"Playing with model: {model_path}")

    # Setup scene
    scene = SceneBuilder(config)
    scene.build()

    physics = Physics(scene.mujoco_model, scene.mujoco_data)
    camera = Camera()
    controls = Controls(physics, camera, render_mode=False)
    robot_track = TrackRobot(scene.mujoco_data)

    # Setup agent
    view_range_grid = int(config.simulation.robot_view_range / config.simulation.env_precision)
    vision_width = 2 * view_range_grid
    vision_height = 2 * view_range_grid
    vision_size = vision_width * vision_height
    state_dim = vision_size + 13

    if config.robot.control_mode == "discrete_direction":
        action_dim = 4
    elif config.robot.control_mode == "continuous_vector":
        action_dim = 2
    else:
        action_dim = 4

    agent = PPOAgent(
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
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Setup environment for observation
    env = CorridorEnv(config)
    env.reset()

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

        previous_action = np.zeros(4, dtype=np.float64)
        crt_step = 0

        while viewer_instance.is_running() and not controls.quit_requested:
            # Get observation
            obs = env.get_observation()

            # Visualize if requested
            if live_vision and cv2 is not None:
                vision_data = obs[:vision_size]
                vision_img = vision_data.reshape((vision_height, vision_width))
                vision_img_disp = cv2.resize(vision_img, (400, 400), interpolation=cv2.INTER_NEAREST)
                vision_img_disp = (vision_img_disp - vision_img_disp.min()) / (vision_img_disp.max() - vision_img_disp.min() + 1e-6)
                vision_img_disp = (vision_img_disp * 255).astype(np.uint8)
                vision_img_color = cv2.cvtColor(vision_img_disp, cv2.COLOR_GRAY2BGR)

                raw_action_stats = agent.get_action_statistics(obs)
                y_offset = 30
                cv2.putText(vision_img_color, "Agent Output:", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                for i, val in enumerate(raw_action_stats):
                    text = f"Out[{i}]: {val:.3f}"
                    color = (0, 0, 255) if val < 0 else (255, 0, 0)
                    cv2.putText(vision_img_color, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    bar_len = int(abs(val) * 50)
                    cv2.rectangle(vision_img_color, (150, y_offset - 10), (150 + bar_len, y_offset), color, -1)
                    y_offset += 20

                cv2.imshow("Robot Vision", vision_img_color)
                cv2.waitKey(1)

            # Get action
            if crt_step >= config.simulation.warmup_steps:
                raw_stats = agent.get_action_statistics(obs)
                if config.robot.control_mode == "discrete_direction":
                    action = float(np.argmax(raw_stats))
                else:
                    action = raw_stats
            else:
                if config.robot.control_mode == "discrete_direction":
                    action = 0
                else:
                    action = np.zeros(action_dim, dtype=np.float64)
                if crt_step % 100 == 0:
                    print(f"Warmup: {crt_step}/{config.simulation.warmup_steps}", end="\r")

            # Apply action
            if crt_step < config.simulation.warmup_steps:
                physics.robot_wheels_speed[:] = 0
            else:
                target_speeds: npt.NDArray[np.float64] = np.zeros(4, dtype=np.float64)
                max_speed = config.robot.max_speed

                if config.robot.control_mode == "discrete_direction":
                    action_idx = int(action)
                    if action_idx == 0:
                        target_speeds[:] = max_speed
                    elif action_idx == 1:
                        target_speeds[:] = -max_speed
                    elif action_idx == 2:
                        target_speeds = np.array([-max_speed, max_speed, -max_speed, max_speed])
                    elif action_idx == 3:
                        target_speeds = np.array([max_speed, -max_speed, max_speed, -max_speed])
                    physics.set_wheel_speeds_directly(target_speeds)
                elif config.robot.control_mode == "continuous_vector":
                    speed = np.clip(action[0], -1.0, 1.0)
                    rotation = np.clip(action[1], -1.0, 1.0)
                    left_speed = speed + rotation
                    right_speed = speed - rotation
                    raw_wheel_speeds = np.array([left_speed, right_speed, left_speed, right_speed])
                    raw_wheel_speeds = np.clip(raw_wheel_speeds, -1.0, 1.0)
                    target_speeds = raw_wheel_speeds * max_speed
                    physics.set_wheel_speeds_directly(target_speeds)
                else:
                    action = np.clip(action, -1.0, 1.0)
                    target_speeds = action * max_speed
                    physics.set_wheel_speeds_directly(target_speeds)

            # Step physics
            n_repeats = config.simulation.action_repeat if crt_step >= config.simulation.warmup_steps else 1
            for _ in range(n_repeats):
                physics.apply_additionnal_physics()
                mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)

            # Update camera
            camera.update_viewer_camera(viewer_instance.cam, scene.mujoco_model, scene.mujoco_data)
            viewer_instance.sync()
            robot_track.track()
            crt_step += 1

            # Timing
            time.sleep(1.0 / 400.0)

    robot_track.plot_tracking()
    if live_vision and cv2 is not None:
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Corridor RL - Play Mode")
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    parser.add_argument('--model_path', type=str, default=None, help='Model path for play mode')
    parser.add_argument('--live_vision', action='store_true', help='Show live vision window')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    model_path = args.model_path if args.model_path else cfg.training.model_path
    play(cfg, model_path, args.live_vision)


if __name__ == "__main__":
    main()
