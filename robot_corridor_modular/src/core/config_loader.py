import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass

@dataclass
class GlobalConfig:
    simulation: Dict[str, Any]
    model: Dict[str, Any]
    training: Dict[str, Any]
    rewards: Dict[str, Any]

def load_config(main_config_path: str) -> GlobalConfig:
    with open(main_config_path, 'r') as f:
        main_cfg = yaml.safe_load(f)
    
    base_path = Path(main_config_path).parent

    # Chargement dynamique des sous-fichiers
    # Dans main.yaml, on aurait: agent_config: "agents/policy_mlp_small.yaml"
    def _load_sub(key: str, sub_path_key: str):
        sub_file = main_cfg.get(key, {}).get(sub_path_key)
        if sub_file:
            with open(base_path / sub_file, 'r') as f:
                # Merge ou remplace la config
                main_cfg[key].update(yaml.safe_load(f))

    _load_sub('model', 'config_file')
    _load_sub('rewards', 'config_file')
    
    return GlobalConfig(**main_cfg)