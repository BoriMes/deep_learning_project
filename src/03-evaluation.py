from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import importlib.util
from pathlib import Path

from src.config import Config
from src.dataset import BullFlagDataset, LabelMap, build_label_map
from src.models import build_model
from src.utils import (
    setup_logger,
    ensure_dirs,
    save_json,
    confusion_matrix,
    classification_metrics_from_cm,
    plot_confusion_matrix,
)

logger = setup_logger("eval")

module_path = Path(__file__).resolve().parent / "02-training.py"
spec = importlib.util.spec_from_file_location("training_02", module_path)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(mod)

make_splits_group_aware = mod.make_splits_group_aware

def load_checkpoint(cfg: Config) -> dict:
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {cfg.MODEL_PATH}. Run training first.")
    return torch.load(cfg.MODEL_PATH, map_location="cpu")


def _label_map_from_checkpoint(ckpt: dict) -> LabelMap | None:
    lm = ckpt.get("label_map")
    if not isinstance(lm, dict) or not lm:
        return None

    to_id = {str(k): int(v) for k, v in lm.items()}
    ids = sorted(set(to_id.values()))
    if ids != list(range(len(ids))):
        raise ValueError(
            "Checkpoint label_map has non-contiguous ids. "
            f"Expected 0..{len(ids)-1}, got {ids}."
        )

    to_name = {i: name for name, i in to_id.items()}
    return LabelMap(to_id=to_id, to_name=to_name)


@torch.no_grad()
def collect_preds(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        y_pred.append(pred)
        y_true.append(y.numpy().astype(np.int64))

    if not y_true:
        raise ValueError("No samples in evaluation loader (test split is empty).")

    return {"y_true": np.concatenate(y_true), "y_pred": np.concatenate(y_pred)}


def main():
    cfg = Config()
    ensure_dirs(cfg.OUT_DIR, cfg.METRICS_DIR, cfg.FIGURES_DIR)

    if not cfg.INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {cfg.INDEX_CSV}. Run preprocessing first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(cfg.INDEX_CSV)

    ckpt = load_checkpoint(cfg)

    label_map = _label_map_from_checkpoint(ckpt)
    if label_map is None:
        labels = df["target"].astype(str).tolist()
        label_map = build_label_map(labels)

    num_classes = len(label_map.to_id)
    class_names = [label_map.to_name[i] for i in range(num_classes)]

    ds = BullFlagDataset(cfg.INDEX_CSV, cfg=cfg, label_map=label_map)

    _, _, test_idx = make_splits_group_aware(df, cfg.SEED, cfg.TRAIN_FRAC, cfg.VAL_FRAC)
    test_loader = DataLoader(Subset(ds, test_idx), batch_size=cfg.BATCH_SIZE, shuffle=False)

    model_type = ckpt.get("model_type", cfg.MODEL_TYPE)
    input_dim = int(ckpt.get("input_dim", ds[0][0].shape[-1]))

    # Ensure cfg is aligned with checkpoint model type (and file naming)
    cfg.MODEL_TYPE = str(model_type)  # type: ignore

    model = build_model(input_dim=input_dim, num_classes=num_classes, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)

    logger.info("=== EVAL START ===")
    logger.info(f"model_path={cfg.MODEL_PATH}")
    logger.info(f"model_type={model_type} device={device} test_size={len(test_idx)}")

    out = collect_preds(model, test_loader, device)
    y_true, y_pred = out["y_true"], out["y_pred"]

    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_class, summary = classification_metrics_from_cm(cm)

    per_class_named = {class_names[i]: per_class[str(i)] for i in range(num_classes)}
    metrics = {
        "summary": summary,
        "per_class": per_class_named,
        "num_classes": num_classes,
        "class_names": class_names,
        "model_type": str(model_type),
        "model_path": str(cfg.MODEL_PATH),
    }

    out_path = cfg.METRICS_DIR / cfg.METRICS_FILENAME
    save_json(out_path, metrics)
    logger.info(f"Saved metrics: {out_path}")
    logger.info(f"accuracy={summary['accuracy']:.4f} macro_f1={summary['macro_f1']:.4f}")

    if cfg.SAVE_CONFUSION_MATRIX:
        fig_path = cfg.FIGURES_DIR / cfg.CONFUSION_FILENAME
        plot_confusion_matrix(cm, class_names, fig_path)
        logger.info(f"Saved confusion matrix: {fig_path}")

    logger.info("=== EVAL END ===")


if __name__ == "__main__":
    main()
