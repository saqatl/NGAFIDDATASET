import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    3-layer 1D CNN that downsamples the time dimension:
        4096 -> 2048 -> 1024 -> 256
    while projecting channels from `in_channels` up to `d_model`.
    """

    def __init__(self, in_channels: int = 23, d_model: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_model, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T) -> (B, d_model, T//16)
        return self.layers(x)


class CNNTransformerClassifier(nn.Module):
    """
    CNN feature extractor  +  Transformer encoder  +  linear classification head.

    Input shape:  (batch, time_steps=4096, channels=23)
    Output shape: (batch, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 23,
        num_classes: int = 11,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels, d_model)

        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) — time-major input
        x = x.permute(0, 2, 1)                     # (B, C, T) for Conv1d
        x = self.cnn(x)                             # (B, d_model, T')
        x = x.permute(0, 2, 1)                     # (B, T', d_model)

        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.pos_drop(x)

        x = self.transformer_encoder(x)             # (B, T', d_model)
        x = x.mean(dim=1)                           # global average pooling
        x = self.head(x)                            # (B, num_classes)
        return x
