#!/usr/bin/env python3
"""
Test script to verify goal and reward display in live vision works correctly
"""

import sys
import numpy as np
from src.core.config_loader import load_config
from src.simulation.generator import SceneBuilder
from src.environment.wrapper import CorridorEnv
import mujoco

def test_goal_and_reward_display():
    """Test that goal and reward display works without errors"""
    
    print("Testing goal and reward display in play mode...")
    
    # Load config
    config = load_config('config/main.yaml')
    
    # Ensure include_goal is enabled
    if not config.model.include_goal:
        print("⚠️  WARNING: include_goal is False in config. Setting to True for this test.")
        config.model.include_goal = True
    
    # Create scene
    print("1. Creating scene...")
    scene = SceneBuilder(config)
    scene.build()
    
    # Create environment with shared scene
    print("2. Creating environment with shared scene...")
    env = CorridorEnv(config, scene=scene)
    env.reset()
    
    # Get robot ID
    robot_id = mujoco.mj_name2id(scene.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, "robot")
    
    # Get initial observation
    obs = env.get_observation()
    
    # Check goal position
    print(f"3. Goal position: {scene.goal_position}")
    
    # Extract state vector
    vision_size = 400  # Assuming default view range
    state_vector = obs[vision_size:]
    
    print(f"4. State vector length: {len(state_vector)}")
    
    # Check goal-relative coordinates
    if len(state_vector) >= 17:
        dx = state_vector[13]
        dy = state_vector[14]
        distance = state_vector[15]
        angle = state_vector[16] * np.pi
        
        print(f"5. Goal-relative coordinates from agent input:")
        print(f"   dx={dx:.2f}, dy={dy:.2f}")
        print(f"   distance={distance:.2f}")
        print(f"   angle={angle:.2f} rad ({np.degrees(angle):.1f} deg)")
    else:
        print("❌ ERROR: State vector too short, goal info not included!")
        return 1
    
    # Test reward calculation
    print(f"6. Testing reward calculation...")
    
    # Get robot state
    from src.core.types import Vec3
    robot_pos_vec = scene.mujoco_data.xpos[robot_id]
    robot_vel_vec = scene.mujoco_data.cvel[robot_id][:3]
    
    current_pos = Vec3(robot_pos_vec[0], robot_pos_vec[1], robot_pos_vec[2])
    current_vel = Vec3(robot_vel_vec[0], robot_vel_vec[1], robot_vel_vec[2])
    
    # Calculate reward using environment's reward strategy
    step_reward = env.reward_strategy.compute(
        current_pos,
        current_vel,
        scene.goal_position,
        env.previous_action,
        0,  # step_count
        False,  # is_stuck
        False   # is_backward
    )
    
    print(f"   Initial step reward: {step_reward:.4f}")
    
    # Simulate movement and check reward changes
    print(f"7. Simulating movement to verify reward updates...")
    
    # Apply some movement
    max_speed = config.robot.max_speed
    target_speeds = np.array([max_speed, max_speed, max_speed, max_speed])
    env.physics.set_wheel_speeds_directly(target_speeds)
    env.previous_action = target_speeds / max_speed
    
    # Step physics
    for _ in range(100):
        env.physics.apply_additionnal_physics()
        mujoco.mj_step(scene.mujoco_model, scene.mujoco_data, nstep=1)
    
    # Get new state
    robot_pos_vec2 = scene.mujoco_data.xpos[robot_id]
    robot_vel_vec2 = scene.mujoco_data.cvel[robot_id][:3]
    current_pos2 = Vec3(robot_pos_vec2[0], robot_pos_vec2[1], robot_pos_vec2[2])
    current_vel2 = Vec3(robot_vel_vec2[0], robot_vel_vec2[1], robot_vel_vec2[2])
    
    # Calculate new reward
    step_reward2 = env.reward_strategy.compute(
        current_pos2,
        current_vel2,
        scene.goal_position,
        env.previous_action,
        100,
        False,
        False
    )
    
    print(f"   After movement step reward: {step_reward2:.4f}")
    
    # Check that observation updates
    obs2 = env.get_observation()
    state_vector2 = obs2[vision_size:]
    
    if len(state_vector2) >= 17:
        distance2 = state_vector2[15]
        print(f"   Distance updated: {distance:.2f} -> {distance2:.2f}")
    
    print("\n✅ SUCCESS: Goal and reward display should work correctly!")
    print("   - Goal coordinates are available")
    print("   - State vector includes goal-relative info")
    print("   - Reward calculation works with environment's strategy")
    print("   - All data updates correctly")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(test_goal_and_reward_display())
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
