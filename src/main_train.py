"""
Train the model from a configuration file

Usage:
    python -m src.main_train --config config/main.yaml

Or:
    python -m src.main --train --config config/main.yaml
"""

from .core.config_loader import load_config
from .core.types import GlobalConfig
from .environment.wrapper import SimulationEnv
from .algorithms.ppo import PPOAgent
from .utils.memory import Memory
from .utils.parallel_env import SubprocVecEnv

from typing import Any, Optional

import torch

import os
import argparse
import datetime
import multiprocessing as mp

import numpy as np
from numpy.typing import NDArray

from tqdm import tqdm
from functools import partial


# Factory function for creating environments
def make_env(config: GlobalConfig) -> SimulationEnv:
    """
    Factory function for creating environments
    """

    return SimulationEnv(config)


# Train the PPO agent with parallel environments
def train(config: GlobalConfig, exp_dir_override: Optional[str] = None) -> None:
    """
    Train the PPO agent with parallel environments

    Args:
        config: Global configuration
        exp_dir_override: Override output directory (for pipeline mode)
    """

    print("Starting training...")

    # Create experiment directory
    #
    exp_dir: str
    #
    if exp_dir_override:
        #
        exp_dir = exp_dir_override
    else:
        #
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #
        exp_dir = os.path.join("trainings_exp", timestamp)

    os.makedirs(exp_dir, exist_ok=True)

    print(f"Experiment data: {exp_dir}")

    # Paths
    current_model_path: str = os.path.join(exp_dir, "model_latest.pth")
    best_model_path: str = os.path.join(exp_dir, "best_model.pth")
    training_log_path: str = os.path.join(exp_dir, "rewards.txt")

    # Number of environments
    num_envs: int = config.simulation.num_envs

    # Create parallel environments

    print(f"Using {num_envs} parallel environments")

    envs: SubprocVecEnv = SubprocVecEnv(
        [
            partial(make_env, config=config)
            for _ in range(num_envs)
        ]
    )

    # Calculate dimensions
    view_range_grid: int = int(config.simulation.robot_view_range / config.simulation.env_precision)
    vision_width: int = 2 * view_range_grid
    vision_height: int = 2 * view_range_grid
    vision_size: int = vision_width * vision_height

    # Base state vector: 13 dimensions (position, rotation, velocity, action)
    # Goal-relative coords add 4 dimensions if include_goal is True
    goal_dims: int = 4 if config.model.include_goal else 0
    state_vector_dim: int = 13 + goal_dims
    state_dim: int = vision_size + state_vector_dim

    if config.robot.control_mode == "discrete_direction":
        action_dim: int = 4
    elif config.robot.control_mode == "continuous_vector":
        action_dim: int = 2
    else:
        action_dim: int = 4

    # Create agent
    agent: PPOAgent = PPOAgent(
        state_dim, action_dim, (vision_width, vision_height),
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
        control_mode=config.robot.control_mode
    )

    # Initialize memory
    memory: Memory = Memory()

    # Load weights if specified
    if config.training.load_weights_from \
        and os.path.exists(config.training.load_weights_from):
        #
        print(f"Loading weights from {config.training.load_weights_from}")
        #
        try:
            #
            agent.policy.load_state_dict(torch.load(config.training.load_weights_from, map_location=agent.device))
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            #
            print("Weights loaded successfully")
        #
        except Exception as e:
            #
            print(f"Error loading weights: {e}")

    # Training variables
    episode_rewards: list[float] = [0.0] * num_envs
    episode_steps: list[int] = [0] * num_envs
    i_episode: int = 0
    best_reward: float = -float('inf')

    #
    avg_rewards_list: list[float] = []

    # Early stopping tracking
    consecutive_successes: int = 0
    success_threshold: float = getattr(config.training, 'early_stopping_success_threshold', 90.0)
    required_successes: int = getattr(config.training, 'early_stopping_consecutive_successes', 50)
    early_stopping_enabled: bool = getattr(config.training, 'early_stopping_enabled', False)

    state: NDArray[np.float32] = envs.reset()

    # Training loop with progress bar
    pbar: tqdm.tqdm = tqdm(total=config.training.max_episodes, desc="Training")

    while i_episode < config.training.max_episodes:
        steps_per_update: int = config.training.update_timestep // num_envs
        interval_completed_rewards: list[float] = []

        #
        action: NDArray[np.float32]
        action_logprob: NDArray[np.float32]
        next_state: NDArray[np.float32]
        reward: NDArray[np.float32]
        terminated: NDArray[np.bool_]
        truncated: NDArray[np.bool_]
        infos: dict[str, Any]
        done: NDArray[np.bool_]
        idx: int
        t: int
        #
        for t in range(steps_per_update):

            # Select action
            action, action_logprob = agent.select_action(state)

            # Step
            next_state, reward, terminated, truncated, infos = envs.step(action)
            done = np.logical_or(terminated, truncated)

            # Save to memory
            memory.states.append(torch.FloatTensor(state))
            memory.actions.append(torch.FloatTensor(action))
            memory.logprobs.append(action_logprob)
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

            # Update tracking
            for idx in range(num_envs):

                # Update episode tracking
                episode_rewards[idx] += reward[idx]
                episode_steps[idx] += 1

                # Episode completed
                if done[idx]:
                    #
                    ep_reward = float(episode_rewards[idx])
                    ep_steps = episode_steps[idx]
                    interval_completed_rewards.append(ep_reward)

                    # Print episode info to terminal
                    print(f"\n[Episode {i_episode + 1}] Reward: {ep_reward:.2f} | Steps: {ep_steps}")

                    # Log immediately (Ctrl+C safe)
                    try:
                        with open(training_log_path, 'a') as f:
                            f.write(f"{ep_reward}\n")
                            f.flush()
                    #
                    except Exception as e:
                        print(f"Error writing log: {e}")

                    # Reset episode tracking
                    episode_rewards[idx] = 0.0
                    episode_steps[idx] = 0
                    i_episode += 1
                    pbar.update(1)

            # Update state
            state = next_state

        # Update PPO
        agent.update(memory, state, done)
        memory.clear_memory()

        # Logging
        if len(interval_completed_rewards) > 0:
            avg_reward: float = sum(interval_completed_rewards) / len(interval_completed_rewards)
            min_reward: float = min(interval_completed_rewards)
            max_reward: float = max(interval_completed_rewards)

            #
            avg_rewards_list.append(avg_reward)

            #
            last_5_rewards_avgs = np.mean(avg_rewards_list[-5:])
            last_10_rewards_avgs = np.mean(avg_rewards_list[-10:])

            # Print interval summary
            print(f"\n--- Update Summary ---")
            print(f"Episodes in interval: {len(interval_completed_rewards)}")
            print(f"Avg reward: \033[1m{avg_reward:.2f}\033[0m | Min: \033[2m{min_reward:.2f}\033[0m | Max: \033[2m{max_reward:.2f}\033[0m | Best: {best_reward:.2f}")
            print(f"Last 5 avg: \033[1m{last_5_rewards_avgs:.2f}\033[0m | Last 10 avg: \033[1m{last_10_rewards_avgs:.2f}\033[0m")
            print(f"----------------------")

            # Update progress bar
            pbar.set_postfix({
                'avg': f'{avg_reward:.2f}',
                'min': f'{min_reward:.2f}',
                'max': f'{max_reward:.2f}',
                'best': f'{best_reward:.2f}'
            })

            # Save best model
            if avg_reward > best_reward:
                #
                best_reward = avg_reward
                torch.save(agent.policy.state_dict(), best_model_path)
                #
                print(f"New best model saved! (reward: {best_reward:.2f})")

            # Save latest periodically
            if i_episode % 10 == 0:
                torch.save(agent.policy.state_dict(), current_model_path)

            # Early stopping
            if early_stopping_enabled:
                #
                if avg_reward >= success_threshold:
                    consecutive_successes += 1
                    print(f"Success! ({consecutive_successes}/{required_successes} consecutive)")
                else:
                    consecutive_successes = 0

                # Early stopping check
                if consecutive_successes >= required_successes:
                    print(f"\nEarly stopping triggered! {required_successes} consecutive successes achieved.")
                    break

    # Close environment
    pbar.close()
    envs.close()

    # Training complete
    print("Training complete!")


# Main training function
def main() -> None:

    # Parse arguments
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Robot Corridor RL Training")
    #
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    #
    args: argparse.Namespace = parser.parse_args()

    # Load configuration
    cfg: Config = load_config(args.config)

    # Set multiprocessing method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Run training
    train(cfg)


# This script can also be directly run from the command line instead of using src.main
if __name__ == "__main__":
    #
    main()
