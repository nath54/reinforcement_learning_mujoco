import torch.nn as nn
from src.core.types import ModelConfig
from src.models.mlp import PolicyMLP
from src.models.transformer import PolicyTransformer
from src.models.state_transformer import StateTransformer
from src.models.temporal_mlp import TemporalMLP
from src.models.multi_head_encoder import MultiHeadStateEncoder, SingleHeadStateEncoder


def create_policy_network(config: ModelConfig, input_dim: int, output_dim: int) -> nn.Module:
    """Create a policy network head (actor output layer)."""
    if config.type == "mlp":
        return PolicyMLP(input_dim, output_dim, config.hidden_sizes)
    elif config.type == "transformer":
        return PolicyTransformer(input_dim, output_dim, config.n_heads, config.n_layers, config.embedding_dim, config.dropout)
    else:
        raise ValueError(f"Unknown model type: {config.type}")


def create_state_encoder(config: ModelConfig, state_vector_dim: int) -> nn.Module:
    """
    Create a state encoder based on config.

    Args:
        config: Model configuration
        state_vector_dim: Total dimension of state vector

    Returns:
        Encoder module with .output_dim attribute
    """
    if config.input_mode == "multi_head":
        return MultiHeadStateEncoder(
            history_length=config.state_history_length,
            include_goal=config.include_goal,
            hidden_dim=config.embedding_dim,
            output_dim=config.embedding_dim
        )
    elif config.type == "ft_transformer":
        return StateTransformer(
            num_features=state_vector_dim,
            embed_dim=config.embedding_dim,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=config.dropout,
            output_dim=config.embedding_dim
        )
    elif config.type == "temporal_mlp":
        # Calculate features per frame (position:3, rotation:3, velocity:2 = 8)
        features_per_frame = 8
        goal_dim = 4 if config.include_goal else 0
        return TemporalMLP(
            history_length=config.state_history_length,
            features_per_frame=features_per_frame,
            goal_dim=goal_dim,
            action_dim=4,
            hidden_dim=config.embedding_dim,
            output_dim=config.embedding_dim
        )
    else:
        # Default: single head MLP encoder
        return SingleHeadStateEncoder(
            input_dim=state_vector_dim,
            hidden_dim=config.embedding_dim,
            output_dim=config.embedding_dim
        )