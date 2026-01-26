#!/usr/bin/env python3
"""
Validation script to test that the environment shares MuJoCo data correctly in play mode
"""

import sys
import numpy as np
from src.core.config_loader import load_config
from src.simulation.generator import SceneBuilder
from src.environment.wrapper import CorridorEnv
import mujoco

def test_shared_scene():
    """Test that environment can use a shared scene"""

    print("Testing scene sharing fix...")

    # Load config
    config = load_config('config/main.yaml')

    # Create scene
    print("1. Creating scene...")
    scene = SceneBuilder(config)
    scene.build()

    # Create environment with shared scene
    print("2. Creating environment with shared scene...")
    env = CorridorEnv(config, scene=scene)
    env.reset()

    # Initialize step counter
    crt_step = 0

    # Get initial observation
    obs1 = env.get_observation()
    print(f"3. Initial observation shape: {obs1.shape}")
    print(f"   Vision min/max: {obs1[:400].min():.3f} / {obs1[:400].max():.3f}")
    print(f"   Position: {obs1[400:403]}")

    # Step the physics multiple times
    print("4. Stepping physics 500 times to pass warmup...")

    # Get robot ID
    robot_id = mujoco.mj_name2id(scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")

    for i in range(500):
        # After warmup, apply some movement
        if i >= config.simulation.warmup_steps:
            # Set wheel speeds to move forward
            max_speed = config.robot.max_speed
            target_speeds = np.array([max_speed, max_speed, max_speed, max_speed])
            env.physics.set_wheel_speeds_directly(target_speeds)
        else:
            # During warmup, keep wheels at 0
            env.physics.robot_wheels_speed[:] = 0

        # Apply physics
        env.physics.apply_additionnal_physics()

        # Step MuJoCo
        mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)
        crt_step += 1

    # Get new observation
    obs2 = env.get_observation()

    # Also get actual robot position from MuJoCo data
    actual_pos = scene.mujoco_data.xpos[robot_id]
    actual_vel = scene.mujoco_data.cvel[robot_id][:3]

    print(f"5. After {crt_step} steps:")
    print(f"   Vision min/max: {obs2[:400].min():.3f} / {obs2[:400].max():.3f}")
    print(f"   Observation position: {obs2[400:403]}")
    print(f"   Actual MuJoCo position: {actual_pos}")
    print(f"   Actual MuJoCo velocity: {actual_vel}")

    # Check if observations changed
    obs_changed = not np.array_equal(obs1, obs2)
    position_changed = not np.array_equal(obs1[400:403], obs2[400:403])
    mujoco_pos_changed = not np.allclose(actual_pos, [0, 0, 0], atol=0.01)

    print(f"\n6. Results:")
    print(f"   Observation changed: {obs_changed} ✓" if obs_changed else f"   Observation changed: {obs_changed} ✗")
    print(f"   Observation position changed: {position_changed} ✓" if position_changed else f"   Observation position changed: {position_changed} ✗")
    print(f"   MuJoCo position changed: {mujoco_pos_changed} ✓" if mujoco_pos_changed else f"   MuJoCo position changed: {mujoco_pos_changed} ✗")

    # Test backward compatibility
    print("\n7. Testing backward compatibility...")
    env_standalone = CorridorEnv(config)
    obs_standalone = env_standalone.get_observation()
    print(f"   Standalone environment works: ✓")
    print(f"   Observation shape: {obs_standalone.shape}")

    if obs_changed and mujoco_pos_changed:
        print("\n✅ SUCCESS: Environment correctly shares MuJoCo data with scene!")
        print("   The observation updates correctly as physics steps.")
        return 0
    else:
        print("\n❌ FAILURE: Issue detected!")
        if not obs_changed:
            print("   - Observations are completely static")
        if not mujoco_pos_changed:
            print("   - Robot position didn't change in MuJoCo")
        return 1

if __name__ == "__main__":
    sys.exit(test_shared_scene())
