import math
import torch
import torch.nn as nn

from src.config import Config


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=cfg.LSTM_DROPOUT if cfg.LSTM_NUM_LAYERS > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.LSTM_HIDDEN),
            nn.Linear(cfg.LSTM_HIDDEN, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)  # h: (layers, B, H)
        last = h[-1]
        return self.head(last)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


class CNNTransformerClassifier(nn.Module):
    """
    Main model: Conv1D feature extractor + TransformerEncoder + pooling + head.
    Input: (B, L, F)
    """
    def __init__(self, input_dim: int, num_classes: int, cfg: Config):
        super().__init__()
        d_model = cfg.TF_D_MODEL

        layers = []
        in_ch = input_dim
        out_ch = cfg.CNN_CHANNELS
        k = cfg.CNN_KERNEL
        for _ in range(cfg.CNN_LAYERS):
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=k // 2))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(out_ch))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.proj = nn.Linear(cfg.CNN_CHANNELS, d_model)
        self.pos = PositionalEncoding(d_model, dropout=cfg.TF_DROPOUT, max_len=max(2048, cfg.SEQ_LEN + 1))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.TF_NHEAD,
            dim_feedforward=4 * d_model,
            dropout=cfg.TF_DROPOUT,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.TF_LAYERS)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F) -> (B, F, L)
        x = x.transpose(1, 2)
        x = self.cnn(x)  # (B, C, L)
        x = x.transpose(1, 2)  # (B, L, C)
        x = self.proj(x)       # (B, L, D)
        x = self.pos(x)
        x = self.encoder(x)    # (B, L, D)

        # mean pooling over time
        x = x.mean(dim=1)      # (B, D)
        return self.head(x)


def build_model(input_dim: int, num_classes: int, cfg: Config) -> nn.Module:
    if cfg.MODEL_TYPE == "baseline":
        return LSTMClassifier(input_dim=input_dim, num_classes=num_classes, cfg=cfg)
    return CNNTransformerClassifier(input_dim=input_dim, num_classes=num_classes, cfg=cfg)
