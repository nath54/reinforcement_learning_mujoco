"""
FT-Transformer Style State Encoder

Feature Tokenizer Transformer for state vector encoding.
Embeds each feature (or feature group) as a token, then applies transformer attention.
"""

from typing import Optional

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Tokenizes numerical features into embeddings.
    Each feature gets its own embedding that combines the feature value with a learned embedding.
    """

    #
    def __init__(self, num_features: int, embed_dim: int) -> None:

        # Initialize parent
        super().__init__()

        # Store parameters
        self.num_features: int = num_features
        self.embed_dim: int = embed_dim

        # Learned embeddings for each feature position
        self.feature_embeddings = nn.Parameter(torch.randn(num_features, embed_dim))

        # Linear projection for feature values
        self.value_projections: nn.ModuleList = nn.ModuleList(
            [nn.Linear(1, embed_dim) for _ in range(num_features)]
        )

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features) numerical features

        Returns:
            (batch, num_features, embed_dim) token embeddings
        """

        # Tokenize each feature
        tokens: list[torch.Tensor] = []

        #
        i: int
        #
        for i in range(self.num_features):
            # Project feature value
            feat_val: torch.Tensor = x[:, i : i + 1]  # (batch, 1)
            projected: torch.Tensor = self.value_projections[i](
                feat_val
            )  # (batch, embed_dim)

            # Add feature embedding
            token: torch.Tensor = projected + self.feature_embeddings[i]
            tokens.append(token)

        # Stack tokens: (batch, num_features, embed_dim)
        stacked_tokens = torch.stack(tokens, dim=1)

        # Layer norm
        return self.layer_norm(stacked_tokens)


#
class TransformerBlock(nn.Module):
    """
    Standard transformer encoder block.
    """

    #
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1) -> None:

        # Initialize parent
        super().__init__()

        #
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

        #
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        #
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim) input tokens

        Returns:
            (batch, seq_len, embed_dim) encoded tokens
        """

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)

        return x


#
class StateTransformer(nn.Module):
    """
    FT-Transformer style encoder for state vectors.

    1. Tokenizes each feature with FeatureTokenizer
    2. Adds a [CLS] token for aggregation
    3. Applies transformer layers
    4. Returns [CLS] token as final representation
    """

    #
    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ) -> None:

        # Initialize parent
        super().__init__()

        # Store parameters
        self.num_features: int = num_features
        self.embed_dim: int = embed_dim

        # Feature tokenizer
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer layers
        self.transformer_layers: nn.ModuleList = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output projection
        self.output_dim: int = output_dim or embed_dim
        if self.output_dim != embed_dim:
            self.output_proj = nn.Linear(embed_dim, self.output_dim)
        else:
            self.output_proj = nn.Identity()

    #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features) state vector

        Returns:
            (batch, output_dim) encoded state
        """

        # Get batch size
        batch_size: int = x.shape[0]

        # Tokenize features
        tokens: torch.Tensor = self.tokenizer(x)  # (batch, num_features, embed_dim)

        # Prepend [CLS] token
        cls_tokens: torch.Tensor = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat(
            [cls_tokens, tokens], dim=1
        )  # (batch, 1 + num_features, embed_dim)

        # Apply transformer layers
        #
        layer: nn.Module
        #
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Extract [CLS] token
        cls_output: torch.Tensor = tokens[:, 0]  # (batch, embed_dim)

        # Project to output dimension
        return self.output_proj(cls_output)
