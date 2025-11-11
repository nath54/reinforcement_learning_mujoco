# Robot RL Project - Modular Structure

A modular refactoring of the robot reinforcement learning simulation with MuJoCo and raycasting vision sensors.

## Project Structure

```
robot_rl_project/
├── main.py              # Entry point
├── config.py            # Configuration constants
├── utils.py             # Utility classes (Vec3, ValType)
├── corridor.py          # Corridor generation
├── robot.py             # Robot XML management
├── scene.py             # World scene construction
├── track.py             # Robot tracking
├── physics.py           # Physics simulation
├── camera.py            # Camera management
├── controls.py          # Keyboard controls
├── state.py             # State management
└── vision_sensor.py     # Raycasting vision sensor
```

## Features

- **Modular Design**: Each component is separated into its own file for better maintainability
- **Raycasting Vision Sensor**: Efficient obstacle detection using MuJoCo's raycasting API
- **No Old Collision Sensor**: The old box-based collision detection has been removed
- **Clean Dependencies**: Clear import structure between modules

## Usage

```bash
# Run interactive simulation
python main.py

# Run with saved controls
python main.py --render_mode

# Render to video
python main.py --render_video
```

## Controls

- **Arrow Keys**: Robot movement (forward, backward, left, right)
- **Space**: Brake
- **1/2/3**: Camera modes (free, follow, top-down)
- **C**: Display camera info
- **S**: Save control history
- **Q/ESC**: Quit

## Vision Sensor

The vision sensor uses raycasting to detect obstacles:
- Returns `np.float32` array of shape `(num_rays,)`
- Value is `-1.0` if no collision detected
- Value is distance (meters) if collision detected

Configuration in `main.py`:
```python
vision_sensor = VisionSensor(
    model=root_scene.mujoco_model,
    data=root_scene.mujoco_data,
    robot_body_name="robot",
    num_rays=16,
    max_distance=5.0,
    height_offset=0.0,
    ray_pattern="circle"
)
```
