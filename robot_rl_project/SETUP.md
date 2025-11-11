# Robot RL Project - Setup Instructions

## Project Location
The modular project is located at: `/workspace/code/robot_rl_project/`

## Required Files (Not Included)
You need to copy these XML files to the project directory:
1. `corridor_3x100.xml` - The corridor scene definition
2. `four_wheels_robot.xml` - The robot model definition

These files are referenced in your original script and should be in the same directory as `main.py`.

## How to Run

1. **Copy XML files to project directory**:
   ```bash
   cd /path/to/your/xml/files
   cp corridor_3x100.xml /workspace/code/robot_rl_project/
   cp four_wheels_robot.xml /workspace/code/robot_rl_project/
   ```

2. **Run the simulation**:
   ```bash
   cd /workspace/code/robot_rl_project
   python main.py
   ```

3. **Other run modes**:
   ```bash
   # Run with saved controls
   python main.py --render_mode
   
   # Render to video
   python main.py --render_video
   ```

## What's Different from Original

### ✅ Removed
- Old `CollisionSensor` class (box-based detection)
- All commented-out collision sensor code
- Monolithic 2000+ line file

### ✅ Added
- Clean modular structure (13 separate files)
- Clear dependencies between modules
- Documentation and README
- Easier to maintain and extend

### ✅ Kept
- VisionSensor with raycasting (working solution)
- All physics, camera, controls functionality
- Video rendering capability
- Robot tracking and plotting

## Module Overview

```
main.py              # Entry point - run this!
├── config.py        # Constants (RENDER_WIDTH, etc.)
├── utils.py         # Vec3, ValType helper classes
├── scene.py         # Build the complete MuJoCo scene
│   ├── corridor.py  # Generate corridor with obstacles
│   └── robot.py     # Load and enhance robot model
├── physics.py       # Physics simulation & robot control
├── camera.py        # Camera modes (free, follow, top-down)
├── controls.py      # Keyboard input handling
├── track.py         # Track robot position/velocity
├── state.py         # Simulation state management
└── vision_sensor.py # Raycasting vision sensor
```

## Vision Sensor Configuration

In `main.py`, you can adjust the vision sensor parameters:

```python
vision_sensor = VisionSensor(
    model=root_scene.mujoco_model,
    data=root_scene.mujoco_data,
    robot_body_name="robot",
    num_rays=16,           # Number of rays (adjust as needed)
    max_distance=5.0,      # Detection range in meters
    height_offset=0.0,     # Height above robot center
    ray_pattern="circle"   # "circle" or "hemisphere"
)
```

## Troubleshooting

**Q: ModuleNotFoundError: No module named 'mujoco'**  
A: Install MuJoCo: `pip install mujoco`

**Q: File not found: corridor_3x100.xml**  
A: Copy the XML files to the project directory (see step 1 above)

**Q: How do I modify the corridor or robot?**  
A: Edit `corridor.py` for corridor generation, `robot.py` for robot appearance, or modify the XML files directly.

**Q: How do I use the vision sensor in my RL training?**  
A: See the example in `main.py` line ~117-130. The sensor returns a numpy array of distances that you can use as observations for your RL agent.
