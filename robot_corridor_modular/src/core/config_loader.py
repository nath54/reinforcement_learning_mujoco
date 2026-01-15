import yaml
import os
from pathlib import Path
from typing import Any, Dict
from src.core.types import (
    GlobalConfig, SimulationConfig, RobotConfig, RewardConfig, TrainingConfig, ModelConfig
)

def load_config(main_config_path: str) -> GlobalConfig:
    if not os.path.exists(main_config_path):
        raise FileNotFoundError(f"Config file {main_config_path} not found")

    with open(main_config_path, 'r') as f:
        main_cfg_dict = yaml.safe_load(f)

    base_path = Path(main_config_path).parent

    def _merge_sub_config(section_key: str, path_key: str = 'config_file'):
        """Merges a sub-config file pointed to by path_key into the section."""
        section = main_cfg_dict.get(section_key, {})
        if path_key in section:
            sub_file_path = base_path / section[path_key]
            if sub_file_path.exists():
                with open(sub_file_path, 'r') as f:
                    sub_cfg = yaml.safe_load(f)
                    # Update section with sub-config (sub-config takes precedence over defaults, but main overrides)
                    # Strategy: sub_config -> update with existing main section data
                    full_section = sub_cfg.copy()
                    full_section.update(section)
                    main_cfg_dict[section_key] = full_section
            else:
                print(f"Warning: Sub-config {sub_file_path} not found.")

    # Load pointers
    _merge_sub_config('model')
    _merge_sub_config('rewards')
    # Simulation/Environment config is often standalone or pointed to.
    # Let's assume we might have an 'environment' key in simulation or similar,
    # but based on the prompt structure, simulation params are in env config.
    # We will try to load a 'simulation' subconfig if 'config_file' exists there.
    if 'simulation' in main_cfg_dict and 'config_file' in main_cfg_dict['simulation']:
         _merge_sub_config('simulation')

    # Clean up 'config_file' keys to avoid kwargs errors in dataclasses
    for key in main_cfg_dict:
        if isinstance(main_cfg_dict[key], dict) and 'config_file' in main_cfg_dict[key]:
            del main_cfg_dict[key]['config_file']

    return GlobalConfig(
        simulation=SimulationConfig(**main_cfg_dict.get('simulation', {})),
        robot=RobotConfig(**main_cfg_dict.get('robot', {})),
        rewards=RewardConfig(**main_cfg_dict.get('rewards', {})),
        training=TrainingConfig(**main_cfg_dict.get('training', {})),
        model=ModelConfig(**main_cfg_dict.get('model', {}))
    )