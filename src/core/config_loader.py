"""
Configuration Loader Module

This module handles loading and merging of YAML configuration files
into strongly-typed dataclasses.
"""


from typing import Any

import os
import yaml

from pathlib import Path

from src.core.types import (
    GlobalConfig, SimulationConfig, RobotConfig, RewardConfig, TrainingConfig, ModelConfig
)


# Load the global configuration from a YAML file
def load_config(main_config_path: str) -> GlobalConfig:
    """
    Load the global configuration from a YAML file.

    Args:
        main_config_path: Path to the main configuration file

    Returns:
        GlobalConfig object populated with configuration data
    """

    # Check if file exists
    if not os.path.exists(main_config_path):
        raise FileNotFoundError(f"Config file {main_config_path} not found")

    # Load main config
    #
    main_cfg_dict: dict[str, Any]
    #
    with open(main_config_path, 'r') as f:
        main_cfg_dict = yaml.safe_load(f)

    # Base path for relative imports
    base_path: Path = Path(main_config_path).parent

    # Helper function to merge sub-configs
    def _merge_sub_config(
        section_key: str,
        path_key: str = 'config_file'
    ) -> None:

        """
        Merges a sub-config file pointed to by path_key into the section.
        """

        #
        section: dict[str, Any] = main_cfg_dict.get(section_key, {})
        #
        if path_key in section:
            #
            sub_file_path: Path = base_path / section[path_key]
            #
            if sub_file_path.exists():
                with open(sub_file_path, 'r') as f:
                    #
                    sub_cfg: dict[str, Any] = yaml.safe_load(f)
                    #
                    # Update section with sub-config (sub-config takes precedence over defaults, but main overrides)
                    # Strategy: sub_config -> update with existing main section data
                    full_section: dict[str, Any] = sub_cfg.copy()
                    full_section.update(section)
                    main_cfg_dict[section_key] = full_section
            else:
                print(f"Warning: Sub-config {sub_file_path} not found.")

    # Load pointers
    _merge_sub_config('model')
    _merge_sub_config('rewards')
    _merge_sub_config('robot')
    _merge_sub_config('training')

    # Simulation/Environment config
    if 'simulation' in main_cfg_dict and 'config_file' in main_cfg_dict['simulation']:
         _merge_sub_config('simulation')

    # Clean up 'config_file' keys to avoid kwargs errors in dataclasses
    #
    key: str
    #
    for key in main_cfg_dict:
        if isinstance(main_cfg_dict[key], dict) and 'config_file' in main_cfg_dict[key]:
            del main_cfg_dict[key]['config_file']

    # Return GlobalConfig
    return GlobalConfig(
        simulation=SimulationConfig(**main_cfg_dict.get('simulation', {})),
        robot=RobotConfig(**main_cfg_dict.get('robot', {})),
        rewards=RewardConfig(**main_cfg_dict.get('rewards', {})),
        training=TrainingConfig(**main_cfg_dict.get('training', {})),
        model=ModelConfig(**main_cfg_dict.get('model', {}))
    )
