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
        return self.layers(x)


def _make_head(d_model: int, num_classes: int, dropout: float = 0.3) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_model, d_model // 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model // 2, num_classes),
    )


class CNNTransformerClassifier(nn.Module):
    """
    CNN feature extractor  +  Transformer encoder  +  task-specific head(s).

    task="binary"      → single head, 2 classes  (before / after maintenance)
    task="multiclass"  → single head, num_classes (maintenance issue type)
    task="combined"    → two heads, joint binary + multiclass
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
        task: str = "multiclass",
    ):
        super().__init__()
        self.task = task
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
        self.norm = nn.LayerNorm(d_model)

        head_dropout = 0.3
        if task == "binary":
            self.head = _make_head(d_model, 2, head_dropout)
        elif task == "multiclass":
            self.head = _make_head(d_model, num_classes, head_dropout)
        elif task == "combined":
            self.binary_head = _make_head(d_model, 2, head_dropout)
            self.multi_head = _make_head(d_model, num_classes, head_dropout)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)                     # (B, C, T)
        x = self.cnn(x)                             # (B, d_model, T')
        x = x.permute(0, 2, 1)                     # (B, T', d_model)

        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.pos_drop(x)
        x = self.transformer_encoder(x)
        features = self.norm(x.mean(dim=1))         # (B, d_model)

        if self.task == "combined":
            return self.binary_head(features), self.multi_head(features)
        return self.head(features)
