import logging
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


_TS_RE_EPOCH_MS = re.compile(r"^\d{13}$")
_TS_RE_DT1 = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}$")
_TS_RE_DT2 = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")


def parse_timestamp(x) -> pd.Timestamp:
    """
    STRICT:
      - 13-digit epoch ms
      - YYYY-MM-DD HH:MM
      - YYYY-MM-DD HH:MM:SS
    Output: tz-naive Timestamp floored to minute.
    """
    s = str(x).strip()
    if _TS_RE_EPOCH_MS.match(s):
        ts = pd.to_datetime(int(s), unit="ms", utc=True).tz_convert(None)
    elif _TS_RE_DT1.match(s):
        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M")
    elif _TS_RE_DT2.match(s):
        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S")
    else:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Unsupported timestamp format: {s}")

    return ts.floor("min").tz_localize(None)


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: dict) -> None:
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def save_train_curves(losses: List[float], accs: List[float], out_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(losses) + 1))
    fig, ax1 = plt.subplots()

    ax1.plot(epochs, losses)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train loss")

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs)
    ax2.set_ylabel("Train accuracy")

    fig.tight_layout()
    out_path = out_dir / "train_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics_from_cm(cm: np.ndarray) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """
    Returns:
      per_class: {str(i): {"precision":..., "recall":..., "f1":..., "support":...}}
      summary: {"accuracy":..., "macro_f1":...}
    """
    eps = 1e-12
    num_classes = cm.shape[0]
    per_class: Dict[str, Dict[str, float]] = {}

    correct = float(np.trace(cm))
    total = float(np.sum(cm))
    acc = correct / (total + eps)

    f1s = []
    for i in range(num_classes):
        tp = float(cm[i, i])
        fp = float(np.sum(cm[:, i]) - tp)
        fn = float(np.sum(cm[i, :]) - tp)
        support = float(np.sum(cm[i, :]))

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)
        f1 = (2 * prec * rec) / (prec + rec + eps)

        per_class[str(i)] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "support": float(support),
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    summary = {"accuracy": float(acc), "macro_f1": macro_f1}
    return per_class, summary


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    im = ax.imshow(cm)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
