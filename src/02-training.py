from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from src.config import Config
from src.dataset import BullFlagDataset, build_label_map
from src.models import build_model
from src.utils import setup_logger, ensure_dirs, save_json, save_train_curves

logger = setup_logger("train")


def make_splits_group_aware(
    df: pd.DataFrame,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> Tuple[List[int], List[int], List[int]]:
    """Stratified random split by target label (keeps class proportions across splits)."""
    if "target" not in df.columns:
        raise ValueError("Missing 'target' column in index.csv for stratified split.")

    rnd = random.Random(seed)

    by_cls: dict[str, List[int]] = {}
    for i, y in enumerate(df["target"].astype(str).tolist()):
        by_cls.setdefault(y, []).append(i)

    for idxs in by_cls.values():
        rnd.shuffle(idxs)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, idxs in by_cls.items():
        n = len(idxs)
        if n == 0:
            continue

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        # small-class safety: try to keep at least 1 in val and 1 in test when possible
        if n >= 3:
            if n_val == 0:
                n_val = 1
            if (n - (n_train + n_val)) == 0:
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val = max(1, n_val - 1)

        train_idx.extend(idxs[:n_train])
        val_idx.extend(idxs[n_train : n_train + n_val])
        test_idx.extend(idxs[n_train + n_val :])

    rnd.shuffle(train_idx)
    rnd.shuffle(val_idx)
    rnd.shuffle(test_idx)

    return train_idx, val_idx, test_idx


@torch.no_grad()
def eval_loss_acc(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0

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


def _cfg_snapshot(cfg: Config) -> dict:
    """Torch checkpoints + JSON like primitives/strings better than raw Paths."""
    snap = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            snap[k] = v
        else:
            snap[k] = str(v)
    return snap


def main():
    cfg = Config()
    ensure_dirs(cfg.OUT_DIR, cfg.MODELS_DIR, cfg.METRICS_DIR, cfg.FIGURES_DIR)

    if not cfg.INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {cfg.INDEX_CSV}. Run preprocessing first.")

    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg.INDEX_CSV)
    labels = df["target"].astype(str).tolist()
    label_map = build_label_map(labels)
    num_classes = len(label_map.to_id)

    ds = BullFlagDataset(cfg.INDEX_CSV, cfg=cfg, label_map=label_map)
    train_idx, val_idx, test_idx = make_splits_group_aware(df, cfg.SEED, cfg.TRAIN_FRAC, cfg.VAL_FRAC)

    # --- class counts from TRAIN split only (for sampler weighting) ---
    y_train = df.iloc[train_idx]["target"].astype(str).tolist()
    counts = np.zeros(num_classes, dtype=np.int64)
    for name in y_train:
        counts[label_map.to_id[name]] += 1

    # --- WeightedRandomSampler to balance classes in training batches ---
    sample_weights = []
    for i in train_idx:
        name = str(df.iloc[i]["target"])
        cid = label_map.to_id[name]
        sample_weights.append(1.0 / max(int(counts[cid]), 1))

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        Subset(ds, train_idx),
        batch_size=cfg.BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(Subset(ds, val_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)

    x0, _ = ds[0]
    input_dim = int(x0.shape[-1])

    model = build_model(input_dim=input_dim, num_classes=num_classes, cfg=cfg).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

    # --- plain CE loss: sampler already balances classes in training batches ---
    ce = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("=== TRAIN START ===")
    logger.info(f"model_type={cfg.MODEL_TYPE} overfit_one_batch={cfg.OVERFIT_ONE_BATCH}")
    logger.info(f"device={device}")
    logger.info(f"dataset_size={len(ds)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")
    logger.info(
        f"SEQ_LEN={cfg.SEQ_LEN} BATCH_SIZE={cfg.BATCH_SIZE} EPOCHS={cfg.EPOCHS} LR={cfg.LR} WD={cfg.WEIGHT_DECAY}"
    )
    logger.info(f"features: log_return={cfg.USE_LOG_RETURN} ohlc_rel={cfg.USE_OHLC_REL_FEATURES} pad={cfg.PAD_MODE}")
    logger.info(f"params_total={n_params} params_trainable={n_trainable}")
    logger.info(f"label_map={label_map.to_id}")

    logger.info("Split label dist (train):")
    logger.info(df.iloc[train_idx]["target"].value_counts().to_string())
    logger.info("Split label dist (val):")
    logger.info(df.iloc[val_idx]["target"].value_counts().to_string())
    logger.info("Split label dist (test):")
    logger.info(df.iloc[test_idx]["target"].value_counts().to_string())

    cfg_snap = _cfg_snapshot(cfg)
    save_json(cfg.METRICS_DIR / "train_config.json", cfg_snap)

    train_losses: List[float] = []
    train_accs: List[float] = []
    val_losses: List[float] = []
    val_accs: List[float] = []

    fixed_batch = None
    if cfg.OVERFIT_ONE_BATCH:
        fixed_batch = next(iter(train_loader))

    for epoch in range(1, cfg.EPOCHS + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        batches = [fixed_batch] if fixed_batch is not None else train_loader

        for x, y in batches:
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

        v_loss, v_acc = eval_loss_acc(model, val_loader, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(v_loss)
        val_accs.append(v_acc)

        logger.info(
            f"epoch={epoch}/{cfg.EPOCHS} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={v_loss:.4f} val_acc={v_acc:.4f}"
        )

    ckpt = {
        "state_dict": model.state_dict(),
        "model_type": cfg.MODEL_TYPE,
        "label_map": label_map.to_id,
        "class_names": [label_map.to_name[i] for i in range(num_classes)],
        "seq_len": int(cfg.SEQ_LEN),
        "input_dim": input_dim,
        "config": cfg_snap,
    }

    torch.save(ckpt, cfg.MODEL_PATH)
    logger.info(f"Saved model: {cfg.MODEL_PATH}")

    hist = {"train_loss": train_losses, "train_acc": train_accs, "val_loss": val_losses, "val_acc": val_accs}
    save_json(cfg.METRICS_DIR / "train_history.json", hist)

    if cfg.SAVE_TRAIN_CURVES:
        p = save_train_curves(train_losses, train_accs, cfg.OUT_DIR)
        logger.info(f"Saved train curves: {p}")

    t_loss, t_acc = eval_loss_acc(model, test_loader, device)
    logger.info(f"FINAL quick_test_loss={t_loss:.4f} quick_test_acc={t_acc:.4f}")
    logger.info("=== TRAIN END ===")


if __name__ == "__main__":
    main()
