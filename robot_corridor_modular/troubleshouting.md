# Troubleshooting Guide

## Common Issues and Solutions

### 1. Config Loading Errors

#### Error: `TypeError: TrainingConfig.__init__() got an unexpected keyword argument 'lr'`

**Problem**: Config file uses wrong parameter name.

**Solution**: Check that your YAML config uses the exact parameter names from the dataclasses:
- Use `learning_rate` not `lr`
- Use `k_epochs` not `K_epochs`
- Check `src/core/types.py` for the exact parameter names

**Quick Fix**:
```yaml
training:
  learning_rate: 0.0003  # Correct
  # lr: 0.0003          # Wrong!
```

#### Error: `FileNotFoundError: Config file config/main.yaml not found`

**Problem**: Running from wrong directory or incorrect path.

**Solution**: 
1. Make sure you're in the project root: `cd robot_corridor_modular`
2. Check the file exists: `ls config/main.yaml`
3. Use absolute path if needed: `--config /full/path/to/config/main.yaml`

### 2. Robot XML Issues

#### Error: `Warning: Robot XML four_wheels_robot.xml not found.`

**Problem**: Robot XML file not in the expected location.

**Solution**:
1. Copy `four_wheels_robot.xml` to project root
2. Or update `robot.xml_path` in config to point to correct location:
```yaml
robot:
  xml_path: "/absolute/path/to/four_wheels_robot.xml"
```

### 3. Training Issues

#### Error: Worker initialization failed

**Problem**: Environment creation fails in subprocess.

**Solution**:
1. Test environment locally first:
```python
from src.core.config_loader import load_config
from src.environment.wrapper import CorridorEnv

cfg = load_config('config/main.yaml')
env = CorridorEnv(cfg)
obs, info = env.reset()
print("Environment works!", obs.shape)
```

2. Check multiprocessing is set to 'spawn':
```python
import multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

#### Training is very slow

**Problem**: Too many environments or CPU bottleneck.

**Solution**:
1. Reduce `num_envs` in config (try 4 or 6)
2. Check GPU is being used: `PPOAgent using device: cuda`
3. Increase `update_timestep` to update less frequently
4. Reduce `action_repeat` (but this affects physics accuracy)

#### Agent not learning / rewards not improving

**Problem**: Hyperparameters or reward function issues.

**Solution**:
1. Check reward scale: values should be roughly -10 to +10 per episode
2. Increase `entropy_coeff` (e.g., 0.02) for more exploration
3. Decrease `learning_rate` if loss is unstable
4. Check `stuck_penalty` isn't too harsh
5. Visualize training: plot rewards from `trainings_exp/*/rewards.txt`

### 4. Play Mode Issues

#### Error: `Model loaded successfully` but robot doesn't move

**Problem**: Warmup period or model not trained enough.

**Solution**:
1. Wait for warmup to complete (check console: "Warmup: X/1000")
2. Train longer or load a better model
3. Try continuous control mode instead of discrete
4. Check model architecture matches training config

#### Live vision window shows black screen

**Problem**: OpenCV not installed or vision data all zeros.

**Solution**:
1. Install OpenCV: `pip install opencv-python`
2. Check if obstacles are in view range
3. Verify `robot_view_range` in config is not too small

### 5. Interactive Mode Issues

#### Robot not responding to keyboard

**Problem**: Window focus or key mapping issue.

**Solution**:
1. Click on the MuJoCo viewer window to give it focus
2. Check key codes (arrow keys: 262-265, space: 32)
3. Try pressing keys slowly one at a time
4. Check console for "Saved control history" when pressing 'S'

#### Camera not moving

**Problem**: Camera mode or mouse control issue.

**Solution**:
1. Press '1' for free camera mode
2. Hold Ctrl + drag mouse to rotate view
3. Press '2' to follow robot
4. Press '3' for top-down view

### 6. Import Errors

#### Error: `ModuleNotFoundError: No module named 'src'`

**Problem**: Running script incorrectly.

**Solution**:
Run as module from project root:
```bash
python -m src.main --train  # Correct
# python src/main.py        # Wrong!
```

#### Error: `ImportError: cannot import name 'X' from 'src.Y'`

**Problem**: Missing file or incorrect import.

**Solution**:
1. Check all `__init__.py` files exist
2. Verify file structure matches imports
3. Check for circular imports
4. Try: `python -c "import src; print('OK')"`

### 7. Performance Issues

#### Training crashes with out of memory

**Problem**: Too many environments or batch too large.

**Solution**:
1. Reduce `num_envs` (try 2-4)
2. Reduce `update_timestep` (try 2000)
3. Use smaller model (try MLP with [64, 32])
4. Check VRAM usage: `nvidia-smi`

#### Physics simulation unstable

**Problem**: Timestep or warmup issues.

**Solution**:
1. Increase `warmup_steps` (try 2000)
2. Check physics in `generator.py`: timestep='0.01'
3. Reduce robot `max_speed`
4. Check for NaN values in observations

### 8. Config Merging Issues

#### External config not loading

**Problem**: Path or merge logic issue.

**Solution**:
1. Use relative paths from config directory:
```yaml
model:
  config_file: "agents/policy_mlp_small.yaml"  # Correct
  # config_file: "config/agents/policy_mlp_small.yaml"  # Wrong!
```

2. Check file exists relative to config directory
3. Inline config takes precedence over file

### 9. Debugging Tips

#### Enable verbose logging

Add to top of `src/main.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### Test individual components

```python
# Test config loading
from src.core.config_loader import load_config
cfg = load_config('config/main.yaml')
print(cfg)

# Test scene building
from src.simulation.generator import SceneBuilder
scene = SceneBuilder(cfg)
scene.build()
print("Scene built successfully")

# Test environment
from src.environment.wrapper import CorridorEnv
env = CorridorEnv(cfg)
obs, info = env.reset()
print("Env works!", obs.shape)

# Test PPO
from src.algorithms.ppo import PPOAgent
agent = PPOAgent(obs.shape[0], 4, (40, 40))
action, logprob = agent.select_action(obs)
print("Agent works!", action)
```

#### Check GPU usage

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0)}")
```

### 10. Common YAML Mistakes

1. **Indentation**: Use spaces, not tabs
2. **Quotes**: Use quotes for strings with special chars
3. **Null values**: Use `null` not `None`
4. **Lists**: Use `[1, 2, 3]` or:
```yaml
list:
  - 1
  - 2
  - 3
```

### Getting Help

If issues persist:
1. Check the full error traceback
2. Print intermediate values
3. Test with minimal config
4. Compare with working example configs
5. Check if issue exists in original non-modular version