import torch
import torch.nn as nn

class PolicyTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, n_heads: int, n_layers: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(embedding_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim) -> (batch, 1, input_dim) for seq len 1
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.head(x)