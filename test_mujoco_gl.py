#!/usr/bin/env python3
"""
MuJoCo Getting Started Example
===============================

This script demonstrates basic MuJoCo usage:
1. Loading a simple model
2. Running simulation steps
3. Basic visualization with mujoco-python-viewer

Run this script to test your MuJoCo installation!
"""

import mujoco
import mujoco.viewer
import numpy as np
import time

def create_simple_model():
    """Create a simple MuJoCo model with a falling box"""
    xml = """
    <mujoco model="falling_box">
        <option timestep="0.002"/>
        
        <worldbody>
            <!-- Ground plane -->
            <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.9 0.8 1"/>
            
            <!-- Falling box -->
            <body name="box" pos="0 0 2">
                <freejoint/>
                <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0.4 0.4 1" mass="1"/>
            </body>
            
            <!-- Another box -->
            <body name="box2" pos="1 0 3">
                <freejoint/>
                <geom name="box2_geom" type="box" size="0.15 0.15 0.15" rgba="0.4 0.4 1 1" mass="0.5"/>
            </body>
        </worldbody>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)

def main():
    print("🤖 MuJoCo Example - Falling Boxes")
    print("=" * 40)
    
    # Create model and data
    model = create_simple_model()
    data = mujoco.MjData(model)
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Bodies: {model.nbody}")
    print(f"   - Geoms: {model.ngeom}")
    print(f"   - DOFs: {model.nv}")
    print(f"   - Timestep: {model.opt.timestep}")
    
    # Run simulation in viewer
    print("\n🎮 Starting interactive viewer...")
    print("   - Use mouse to rotate/zoom")
    print("   - Press SPACE to pause/resume")
    print("   - Press ESC to exit")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Run simulation
        start_time = time.time()
        step_count = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # Physics step
            mujoco.mj_step(model, data)
            step_count += 1
            
            # Update viewer
            viewer.sync()
            
            # Print info every 2 seconds
            if step_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"⏱️  Time: {elapsed:.1f}s, Steps: {step_count}, Sim time: {data.time:.2f}s")
            
            # Target real-time rendering
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Simulation stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nIf you see import errors, make sure MuJoCo is properly installed:")
        print("  pip install mujoco mujoco-python-viewer")