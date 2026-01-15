# Robot Corridor Modular - Complete Implementation

## Overview
This is a fully modular, type-hinted, and YAML-configurable implementation of the robot corridor RL training environment. All features from the original monolithic script have been preserved and organized into a clean, maintainable structure.

## Directory Structure

```
robot_corridor_modular/
├── config/                          # All YAML configurations
│   ├── main.yaml                    # Main config (points to sub-configs)
│   ├── agents/                      # Model architectures
│   │   ├── policy_mlp_small.yaml
│   │   └── policy_transformer_large.yaml
│   ├── environments/                # Map and physics configs
│   │   ├── corridor_standard.yaml
│   │   └── corridor_hard.yaml
│   └── rewards/                     # Reward function configs
│       ├── standard_reward.yaml
│       └── velocity_focused.yaml
│
├── src/
│   ├── __init__.py
│   ├── core/                        # LEVEL 0: No external dependencies
│   │   ├── __init__.py
│   │   ├── types.py                 # Dataclasses, Enums (Vec3, Rect2d, configs)
│   │   ├── interfaces.py            # Protocols/ABC
│   │   └── config_loader.py         # Smart YAML loader
│   │
│   ├── simulation/                  # LEVEL 1: MuJoCo & Physics
│   │   ├── __init__.py
│   │   ├── generator.py             # Procedural corridor generation
│   │   ├── robot.py                 # Robot XML management
│   │   ├── physics.py               # Physics engine (forces, drag, wheels)
│   │   ├── sensors.py               # Vision and Collision systems
│   │   └── controls.py              # Keyboard controls
│   │
│   ├── environment/                 # LEVEL 2: Gym Wrapper & Logic
│   │   ├── __init__.py
│   │   ├── wrapper.py               # CorridorEnv (Gym)
│   │   └── reward_strategy.py       # Interchangeable reward strategies
│   │
│   ├── models/                      # LEVEL 3: Neural Networks (PyTorch)
│   │   ├── __init__.py
│   │   ├── factory.py               # Dynamic model creation
│   │   ├── mlp.py                   # MLP implementation
│   │   └── transformer.py           # Transformer implementation
│   │
│   ├── algorithms/                  # LEVEL 4: RL Algorithms
│   │   ├── __init__.py
│   │   └── ppo.py                   # PPO agent with ActorCritic
│   │
│   ├── utils/                       # Utilities
│   │   ├── __init__.py
│   │   ├── memory.py                # PPO memory buffer
│   │   ├── parallel_env.py          # Parallel environment wrapper
│   │   └── tracking.py              # Robot trajectory tracking/plotting
│   │
│   └── main.py                      # Entry point with all modes
│
└── four_wheels_robot.xml            # Robot URDF/XML
```

## Complete Feature List

### ✅ All Original Features Preserved

1. **Multiple Control Modes**:
   - `discrete_direction`: 4 discrete actions (Forward, Backward, Left, Right)
   - `continuous_vector`: 2D continuous (Speed, Rotation)
   - `continuous_wheels`: 4D continuous (individual wheel control)

2. **Obstacle Generation Modes**:
   - `sinusoidal`: Wave pattern obstacles
   - `double_sinusoidal`: Double wave pattern (NEW - was missing)
   - `random`: Random placement
   - `none`: No obstacles

3. **Training Features**:
   - Parallel environments with multiprocessing
   - GAE (Generalized Advantage Estimation)
   - Gradient clipping
   - Entropy regularization
   - Action smoothing
   - Warmup steps
   - Checkpoint saving (best + latest)
   - Crash-safe logging (immediate file writes)

4. **Reward Components**:
   - Position-based or velocity-based rewards
   - Goal reaching bonus
   - Stuck detection penalty
   - Backward escape bonus
   - Collision detection
   - Straight line penalty

5. **Modes of Operation**:
   - `--train`: Multi-environment parallel training
   - `--play`: Play with trained model (with optional live vision)
   - `--interactive`: Keyboard control mode
   - `--render_mode`: Replay saved controls

6. **Vision System**:
   - Grid-based obstacle detection
   - Efficient collision system
   - Configurable view range
   - CNN-based vision processing

7. **Camera Modes**:
   - Free camera (mouse control)
   - Follow robot (3rd person)
   - Top-down view

8. **Physics Simulation**:
   - Air resistance
   - Robot deceleration
   - Differential drive control
   - Configurable wheel speeds

9. **Visualization**:
   - Live vision window (OpenCV)
   - Agent output visualization
   - Trajectory plotting (matplotlib)

## Usage

### Training
```bash
python -m src.main --train --config config/main.yaml
```

### Play with Trained Model
```bash
python -m src.main --play --model_path path/to/model.pth
```

### Play with Live Vision
```bash
python -m src.main --play --model_path path/to/model.pth --live_vision
```

### Interactive Keyboard Control
```bash
python -m src.main --interactive
```

### Replay Saved Controls
```bash
python -m src.main --interactive --render_mode
```

## Configuration

All aspects are configurable via YAML:

### Main Config (`config/main.yaml`)
```yaml
simulation:
  max_steps: 30000
  env_precision: 0.2

model:
  config_file: "agents/policy_mlp_small.yaml"

rewards:
  config_file: "rewards/velocity_focused.yaml"

training:
  lr: 0.0003
```

### Model Config Example (`config/agents/policy_mlp_small.yaml`)
```yaml
type: "mlp"
hidden_sizes: [128, 64]
state_vector_dim: 13
action_std_init: 0.5
control_mode: "discrete_direction"
```

### Environment Config Example (`config/environments/corridor_standard.yaml`)
```yaml
corridor_length: 100.0
corridor_width: 3.0
robot_view_range: 4.0
obstacles_mode: "sinusoidal"
obstacles_mode_param:
  obstacle_sep: 5.0
  obstacle_size_x: 0.4
```

## Key Improvements Over Original

1. **Modularity**: Clean separation of concerns across layers
2. **Type Safety**: Full type hints throughout
3. **Configurability**: Everything is YAML-configurable
4. **Maintainability**: Easy to modify/extend individual components
5. **Testing**: Each module can be tested independently
6. **Documentation**: Clear structure and purpose for each file

## Dependencies

```
mujoco>=3.0.0
torch>=2.0.0
numpy>=1.24.0
gymnasium>=0.29.0
PyYAML>=6.0
matplotlib>=3.7.0
tqdm>=4.65.0
opencv-python>=4.8.0  # Optional, for live vision
```

## Notes

- All physics behavior matches the original exactly
- Action smoothing, GAE, and reward calculation are identical
- The modular structure makes it easy to swap components (e.g., different models, reward functions)
- Progress bars added for better training feedback
- All file I/O is crash-safe (immediate flush)
