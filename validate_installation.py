#!/usr/bin/env python3
"""
Validation script for robot_corridor_modular installation.
Run this to verify everything is set up correctly.
"""

import sys
import os
from pathlib import Path

def print_status(test_name: str, passed: bool, message: str = "") -> None:
    """Print test status"""
    status = "✓" if passed else "✗"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    msg = f" - {message}" if message else ""
    print(f"{color}{status}{reset} {test_name}{msg}")
    return passed

def main():
    print("=" * 60)
    print("Robot Corridor Modular - Installation Validation")
    print("=" * 60)
    print()

    all_passed = True

    # 1. Check directory structure
    print("📁 Checking directory structure...")
    required_dirs = [
        "src/core",
        "src/simulation",
        "src/environment",
        "src/models",
        "src/algorithms",
        "src/utils",
        "config",
        "config/agents",
        "config/rewards",
    ]
    for dir_path in required_dirs:
        exists = Path(dir_path).exists()
        all_passed &= print_status(f"  {dir_path}", exists)

    print()

    # 2. Check required files
    print("📄 Checking required files...")
    required_files = [
        "config/main.yaml",
        "src/main.py",
        "src/core/config_loader.py",
        "src/core/types.py",
        "src/simulation/physics.py",
        "src/simulation/generator.py",
        "src/environment/wrapper.py",
        "src/algorithms/ppo.py",
        "xml/four_wheels_robot.xml",
    ]
    for file_path in required_files:
        exists = Path(file_path).exists()
        all_passed &= print_status(f"  {file_path}", exists)

    print()

    # 3. Check Python dependencies
    print("🐍 Checking Python dependencies...")
    dependencies = {
        "mujoco": "MuJoCo physics engine",
        "torch": "PyTorch deep learning",
        "numpy": "Numerical computing",
        "gymnasium": "RL environment interface",
        "yaml": "YAML config parser",
        "matplotlib": "Plotting library",
        "tqdm": "Progress bars",
    }

    for module, description in dependencies.items():
        try:
            __import__(module)
            print_status(f"  {module}", True, description)
        except ImportError:
            all_passed &= print_status(f"  {module}", False, f"MISSING: {description}")

    # Optional dependencies
    try:
        import cv2
        print_status("  cv2 (optional)", True, "OpenCV for live vision")
    except ImportError:
        print_status("  cv2 (optional)", True, "Not installed (live vision disabled)")

    print()

    # 4. Test config loading
    print("⚙️  Testing configuration loading...")
    try:
        from src.core.config_loader import load_config
        cfg = load_config('config/main.yaml')
        print_status("  Config loading", True)

        # Check config structure
        has_simulation = hasattr(cfg, 'simulation')
        has_robot = hasattr(cfg, 'robot')
        has_rewards = hasattr(cfg, 'rewards')
        has_training = hasattr(cfg, 'training')
        has_model = hasattr(cfg, 'model')

        all_passed &= print_status("  Config structure",
                                   has_simulation and has_robot and has_rewards and has_training and has_model)

    except Exception as e:
        all_passed &= print_status("  Config loading", False, str(e))

    print()

    # 5. Test environment creation
    print("🏃 Testing environment creation...")
    try:
        from src.core.config_loader import load_config
        from src.environment.wrapper import CorridorEnv

        cfg = load_config('config/main.yaml')
        env = CorridorEnv(cfg)
        print_status("  Environment creation", True)

        # Test reset
        obs, info = env.reset()
        print_status("  Environment reset", True, f"obs shape: {obs.shape}")

        # Test step
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print_status("  Environment step", True, f"reward: {reward:.2f}")

    except Exception as e:
        all_passed &= print_status("  Environment test", False, str(e))

    print()

    # 6. Test PPO agent
    print("🤖 Testing PPO agent creation...")
    try:
        from src.algorithms.ppo import PPOAgent

        # Use dimensions from environment test
        state_dim = obs.shape[0]
        action_dim = 4  # discrete_direction
        vision_width = 40
        vision_height = 40

        agent = PPOAgent(state_dim, action_dim, (vision_width, vision_height))
        print_status("  Agent creation", True, f"device: {agent.device}")

        # Test action selection
        action, logprob = agent.select_action(obs)
        print_status("  Action selection", True, f"action: {action}")

    except Exception as e:
        all_passed &= print_status("  Agent test", False, str(e))

    print()

    # Summary
    print("=" * 60)
    if all_passed:
        print("✓ All checks passed! Installation is valid.")
        print()
        print("Next steps:")
        print("  1. Train: python -m src.main --train")
        print("  2. Play: python -m src.main --interactive")
        print("  3. See QUICKSTART.md for more options")
    else:
        print("✗ Some checks failed.")
        print()
        print("Common fixes:")
        print("  • Install missing dependencies: pip install -r requirements.txt")
        print("  • Copy four_wheels_robot.xml to project xml/ directory")
        print("  • Run from project root directory")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()