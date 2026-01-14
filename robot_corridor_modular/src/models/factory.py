import torch.nn as nn
from src.models.mlp import PolicyMLP
from src.models.transformer import PolicyTransformer

def create_policy_network(config: dict, input_dim: int, output_dim: int) -> nn.Module:
    # Le YAML contient une clé "type": "mlp" ou "transformer"
    net_type = config.get("type", "mlp")
    
    if net_type == "mlp":
        return PolicyMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_sizes=config.get("hidden_sizes", [256, 256])
        )
    elif net_type == "transformer":
        return PolicyTransformer(
            input_dim=input_dim,
            output_dim=output_dim,
            n_heads=config.get("n_heads", 4),
            n_layers=config.get("n_layers", 2)
        )
    else:
        raise ValueError(f"Unknown network type: {net_type}")