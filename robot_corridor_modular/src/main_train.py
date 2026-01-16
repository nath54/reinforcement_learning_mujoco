import argparse
import os
import datetime
import multiprocessing as mp
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.core.config_loader import load_config
from src.core.types import GlobalConfig
from src.environment.wrapper import CorridorEnv
from src.algorithms.ppo import PPOAgent
from src.utils.memory import Memory
from src.utils.parallel_env import SubprocVecEnv


def make_env(config: GlobalConfig) -> CorridorEnv:
    """Factory function for creating environments"""
    return CorridorEnv(config)


def train(config: GlobalConfig, exp_dir_override: Optional[str] = None) -> None:
    """Train the PPO agent with parallel environments
    
    Args:
        config: Global configuration
        exp_dir_override: Override output directory (for pipeline mode)
    """
    print("Starting training...")

    # Create experiment directory
    if exp_dir_override:
        exp_dir = exp_dir_override
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_dir = os.path.join("trainings_exp", timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"Experiment data: {exp_dir}")

    # Paths
    current_model_path = os.path.join(exp_dir, "model_latest.pth")
    best_model_path = os.path.join(exp_dir, "best_model.pth")
    training_log_path = os.path.join(exp_dir, "rewards.txt")

    # Create parallel environments
    num_envs = config.simulation.num_envs
    print(f"Using {num_envs} parallel environments")

    envs = SubprocVecEnv([partial(make_env, config=config) for _ in range(num_envs)])

    # Calculate dimensions
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

    # Create agent
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

    memory = Memory()

    # Load weights if specified
    if config.training.load_weights_from and os.path.exists(config.training.load_weights_from):
        print(f"Loading weights from {config.training.load_weights_from}")
        try:
            agent.policy.load_state_dict(torch.load(config.training.load_weights_from, map_location=agent.device))
            agent.policy_old.load_state_dict(agent.policy.state_dict())
            print("Weights loaded successfully")
        except Exception as e:
            print(f"Error loading weights: {e}")

    # Training variables
    episode_rewards = [0.0] * num_envs
    episode_steps = [0] * num_envs
    i_episode = 0
    best_reward = -float('inf')
    
    # Early stopping tracking
    consecutive_successes = 0
    success_threshold = getattr(config.training, 'early_stopping_success_threshold', 90.0)
    required_successes = getattr(config.training, 'early_stopping_consecutive_successes', 50)
    early_stopping_enabled = getattr(config.training, 'early_stopping_enabled', False)

    state = envs.reset()

    # Training loop with progress bar
    pbar = tqdm(total=config.training.max_episodes, desc="Training")

    while i_episode < config.training.max_episodes:
        steps_per_update = config.training.update_timestep // num_envs
        interval_completed_rewards = []

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
                episode_rewards[idx] += reward[idx]
                episode_steps[idx] += 1

                if done[idx]:
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
                    except Exception as e:
                        print(f"Error writing log: {e}")

                    episode_rewards[idx] = 0.0
                    episode_steps[idx] = 0
                    i_episode += 1
                    pbar.update(1)

            state = next_state

        # Update PPO
        agent.update(memory, state, done)
        memory.clear_memory()

        # Logging
        if len(interval_completed_rewards) > 0:
            avg_reward = sum(interval_completed_rewards) / len(interval_completed_rewards)
            min_reward = min(interval_completed_rewards)
            max_reward = max(interval_completed_rewards)

            # Print interval summary
            print(f"\n--- Update Summary ---")
            print(f"Episodes in interval: {len(interval_completed_rewards)}")
            print(f"Avg reward: {avg_reward:.2f} | Min: {min_reward:.2f} | Max: {max_reward:.2f} | Best: {best_reward:.2f}")
            print(f"----------------------")

            pbar.set_postfix({
                'avg': f'{avg_reward:.2f}',
                'min': f'{min_reward:.2f}',
                'max': f'{max_reward:.2f}',
                'best': f'{best_reward:.2f}'
            })

            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(agent.policy.state_dict(), best_model_path)
                print(f"New best model saved! (reward: {best_reward:.2f})")

            # Save latest periodically
            if i_episode % 10 == 0:
                torch.save(agent.policy.state_dict(), current_model_path)
            
            # Early stopping check
            if early_stopping_enabled:
                if avg_reward >= success_threshold:
                    consecutive_successes += 1
                    print(f"Success! ({consecutive_successes}/{required_successes} consecutive)")
                else:
                    consecutive_successes = 0
                
                if consecutive_successes >= required_successes:
                    print(f"\nEarly stopping triggered! {required_successes} consecutive successes achieved.")
                    break

    pbar.close()
    envs.close()
    print("Training complete!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robot Corridor RL Training")
    parser.add_argument('--config', type=str, default='config/main.yaml', help='Config file path')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    train(cfg)


if __name__ == "__main__":
    main()
