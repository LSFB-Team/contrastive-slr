from typing import Optional

import torch
from torch import nn, Tensor
from einops import repeat


class PositionalEmbedding(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            sequence_length: int,
            dropout: float = 0.2,
    ):
        super().__init__()

        self.projection = nn.Sequential(nn.Linear(in_channels, out_channels))
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, out_channels))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_channels))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, l, c = x.shape

        x = self.projection(x)
        x = x + self.pos_encoding
        cls_tokens = repeat(self.cls_token, "() l c -> b l c", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.dropout(x)

        return x


class PoseViT(nn.Module):
    def __init__(
        self,
        in_channels: int = 150,
        out_channels: int = 1024,
        sequence_length: int = 64,
        n_layers: int = 8,
        n_heads: int = 4,
        pool: Optional[str] = "clf_token",
    ):
        super().__init__()
        self.embedding = PositionalEmbedding(in_channels, out_channels, sequence_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_channels,
            nhead=n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_latent = nn.Identity()
        self.pool = pool

    def forward(self, x, src_mask: Optional[Tensor] = None):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=src_mask)
        if self.pool is not None:
            x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        x = self.to_latent(x)
        return x
