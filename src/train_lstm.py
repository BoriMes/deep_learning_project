from __future__ import annotations
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.utils import setup_logger
from src.dataset import BullFlagDataset, build_label_map

logger = setup_logger("train")


@dataclass(frozen=True)
class TrainConfig:
    # Paths in container
    DATASET_DIR: Path = Path("/app/output/dataset")
    INDEX_CSV: Path = Path("/app/output/dataset/index.csv")
    MODEL_DIR: Path = Path("/app/output/models")
    RUN_LOG: Path = Path("/app/output/run_artifacts")  # optional extra outputs

    # Training
    SEQ_LEN: int = 256
    BATCH_SIZE: int = 64
    EPOCHS: int = 10
    LR: float = 1e-3
    WEIGHT_DECAY: float = 1e-4

    # Split
    SEED: int = 42
    TRAIN_FRAC: float = 0.8
    VAL_FRAC: float = 0.1  # rest is test

    # Model (small)
    HIDDEN: int = 32
    NUM_LAYERS: int = 1
    DROPOUT: float = 0.0


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden: int, num_layers: int, num_classes: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, (h, c) = self.lstm(x)      # h: (num_layers, B, hidden)
        last = h[-1]                    # (B, hidden)
        return self.head(last)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.size(0))
    return loss_sum / max(total, 1), correct / max(total, 1)


def make_splits(n: int, seed: int, train_frac: float, val_frac: float):
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def main():
    cfg = TrainConfig()
    cfg.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cfg.RUN_LOG.mkdir(parents=True, exist_ok=True)

    if not cfg.INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing index.csv at {cfg.INDEX_CSV}. Run preprocessing first.")

    # Repro
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load labels from index.csv to build consistent mapping
    import pandas as pd
    df = pd.read_csv(cfg.INDEX_CSV)
    labels = df["target"].astype(str).tolist()
    label_map = build_label_map(labels)
    num_classes = len(label_map.to_id)

    # Build dataset
    ds = BullFlagDataset(cfg.INDEX_CSV, seq_len=cfg.SEQ_LEN, label_map=label_map)

    # Splits
    train_idx, val_idx, test_idx = make_splits(len(ds), cfg.SEED, cfg.TRAIN_FRAC, cfg.VAL_FRAC)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=cfg.BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)

    # Infer input dim from one batch
    x0, y0 = ds[0]
    input_dim = x0.shape[-1]

    model = LSTMClassifier(
        input_dim=input_dim,
        hidden=cfg.HIDDEN,
        num_layers=cfg.NUM_LAYERS,
        num_classes=num_classes,
        dropout=cfg.DROPOUT,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    ce = nn.CrossEntropyLoss()

    # ---- Logging requirements (hyperparams + model summary) ----
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=== TRAIN START ===")
    logger.info(f"device={device}")
    logger.info(f"dataset_size={len(ds)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    logger.info(f"SEQ_LEN={cfg.SEQ_LEN} BATCH_SIZE={cfg.BATCH_SIZE} EPOCHS={cfg.EPOCHS} LR={cfg.LR} WD={cfg.WEIGHT_DECAY}")
    logger.info(f"model=LSTMClassifier input_dim={input_dim} hidden={cfg.HIDDEN} layers={cfg.NUM_LAYERS} classes={num_classes} dropout={cfg.DROPOUT}")
    logger.info(f"params_total={n_params} params_trainable={n_trainable}")
    logger.info(f"labels={label_map.to_id}")

    # ---- Train loop ----
    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total, correct = 0, 0
        loss_sum = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * y.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(y.size(0))

        train_loss = loss_sum / max(total, 1)
        train_acc = correct / max(total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device)

        logger.info(f"epoch={epoch}/{cfg.EPOCHS} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # ---- Final evaluation ----
    test_loss, test_acc = evaluate(model, test_loader, device)
    logger.info(f"FINAL test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    # Save model + label map
    model_path = cfg.MODEL_DIR / "baseline_lstm.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "label_map": label_map.to_id,
            "seq_len": cfg.SEQ_LEN,
            "input_dim": input_dim,
            "cfg": cfg.__dict__,
        },
        model_path,
    )
    logger.info(f"Saved model: {model_path}")

    # ---- Tiny inference demo on 5 test samples ----
    model.eval()
    shown = 0
    for i in test_idx[:5]:
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())
        true = int(y.item())
        logger.info(f"inference sample={i} true={label_map.to_name[true]} pred={label_map.to_name[pred]}")
        shown += 1

    logger.info("=== TRAIN END ===")


if __name__ == "__main__":
    main()
