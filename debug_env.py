
import sys
import os
import numpy as np
import gymnasium as gym

# Add current directory to path
sys.path.append(os.getcwd())

from robot_corridor.exo04_rl_robots_corridor_CERISARA_Nathan_optimized import SimConfig, CorridorEnv, quaternion_to_euler

def test_env():
    print("Loading config...")
    config = SimConfig()
    
    print("Creating env...")
    env = CorridorEnv(config=config)
    
    print("Resetting env...")
    try:
        obs, info = env.reset()
        print("Reset successful.")
        print(f"Observation shape: {obs.shape}")
    except Exception as e:
        print(f"Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Stepping env...")
    try:
        action = np.zeros(4)
        obs, reward, terminated, truncated, info = env.step(action)
        print("Step successful.")
    except Exception as e:
        print(f"Step failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_env()
