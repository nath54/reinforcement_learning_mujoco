# Quick Start Guide

## Installation

```bash
# 1. Clone/navigate to project
cd robot_corridor_modular

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## File Structure Check

Make sure you have:
```
robot_corridor_modular/
├── config/
│   ├── main.yaml
│   ├── main_robot_trained.yaml
│   └── ... (other config files)
├── configs_pipelines/
|   ├── curriculum_v1
│   │   ├── pipeline.yaml
│   │   └── stage_0x.yaml (x = 1 to 5, the different stages of the curriculum)
|   ├── curriculum_v2_hd
│   │   ├── pipeline.yaml
│   │   └── stage_0x.yaml (x = 1 to 5, the different stages of the curriculum)
├── src/
│   ├── main.py
│   └── ... (all modules)
├── xml/
│   ├── four_wheels_robot.xml  # IMPORTANT: Robot XML file
│   └── ... (other xml files)
```

## Quick Test

```bash
python validate_installation.py
```

## Usage

### 1. Interactive Mode (Manual Control)

**Best for**: Testing the environment, understanding the task

```bash
python -m src.main --interactive
```

**Controls**:
- `↑` Forward
- `↓` Backward
- `←` Turn left
- `→` Turn right
- `Space` Stop
- `1/2/3` Camera modes
- `S` Save control sequence
- `Q` Quit

### 2. Training Mode

**Best for**: Training a new agent

Single config training:

```bash
python -m src.main --train --config config/main.yaml
```

Pipeline training:

```bash
python -m src.main --pipeline --train --config configs_pipelines/curriculum_v2_hd/pipeline.yaml
```

**What happens**:
- Creates experiment folder in `trainings_exp/TIMESTAMP/`
- Saves best model as `best_model.pth`
- Saves latest model as `model_latest.pth`
- Logs rewards to `rewards.txt`
- Shows progress bar with stats

**Tip**: Start with small `max_episodes` (e.g., 100) to test, then increase.

### 3. Play Mode (Watch Trained Agent)

**Best for**: Evaluating trained agent

```bash
python -m src.main --play --model_path trainings_exp/TIMESTAMP/best_model.pth
```

**With live vision** to examine what the agent sees:
```bash
python -m src.main --play --model_path path/to/model.pth --live_vision
```

## Configuration Quick Reference

Edit `config/main.yaml`:

```yaml
# Environment difficulty
simulation:
  corridor_length: 100.0    # Shorter = easier
  obstacles_mode: "none"    # Start with no obstacles

# Robot behavior
robot:
  control_mode: "discrete_direction"  # Easier than continuous
  max_speed: 10.0                     # Lower = more stable

# Training speed
training:
  num_envs: 6               # More = faster (if you have CPU cores)
  update_timestep: 4000     # Lower = more frequent updates
  max_episodes: 1000        # Total episodes to train
```

## Example Workflow

### Beginner: Simple corridor, discrete control

1. Edit config:
```yaml
simulation:
  corridor_length: 50.0
  obstacles_mode: "none"
robot:
  control_mode: "discrete_direction"
training:
  max_episodes: 500
```

2. Train:
```bash
python -m src.main --train
```

3. Watch:
```bash
python -m src.main --play --model_path trainings_exp/*/best_model.pth
```

### Intermediate: Add obstacles

```yaml
simulation:
  corridor_length: 100.0
  obstacles_mode: "sinusoidal"
training:
  max_episodes: 2000
```

### Advanced: Continuous control, hard corridor

```yaml
simulation:
  corridor_length: 200.0
  corridor_width: 2.5
  obstacles_mode: "random"
robot:
  control_mode: "continuous_vector"
training:
  max_episodes: 10000
```

## Monitoring Training

### View rewards in real-time

```bash
# In another terminal
tail -f trainings_exp/TIMESTAMP/rewards.txt
```

### Plot rewards

```python
import matplotlib.pyplot as plt

with open('trainings_exp/TIMESTAMP/rewards.txt') as f:
    rewards = [float(line) for line in f]

plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
```

## Common Issues

**"Config file not found"**: Run from project root, not `src/`

**"Robot XML not found"**: Copy `four_wheels_robot.xml` to the `xml/` folder

**Training very slow**: Reduce `num_envs` in config

**Agent not learning**: Check reward scale, try discrete control first

## Next Steps

1. **Experiment with rewards**: Edit `config/rewards/velocity_focused.yaml`
2. **Try different models**: Switch between MLP and Transformer
3. **Customize obstacles**: Create new obstacle patterns in `simulation/generator.py`
4. **Add features**: Extend the modular structure with new components

## Tips

- Start simple (no obstacles, discrete control)
- Train for a few hundred episodes first
- Use `--live_vision` to understand what agent sees
- Save good models (they're small, ~7MB)
- Compare different configs by training multiple times
