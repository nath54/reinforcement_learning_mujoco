import torch.nn as nn
from src.core.types import ModelConfig
from src.models.mlp import PolicyMLP
from src.models.transformer import PolicyTransformer

def create_policy_network(config: ModelConfig, input_dim: int, output_dim: int) -> nn.Module:
    if config.type == "mlp":
        return PolicyMLP(input_dim, output_dim, config.hidden_sizes)
    elif config.type == "transformer":
        return PolicyTransformer(input_dim, output_dim, config.n_heads, config.n_layers, config.embedding_dim, config.dropout)
    else:
        raise ValueError(f"Unknown model type: {config.type}")