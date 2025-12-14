from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


def _env_path(name: str, default: str) -> Path:
    v = os.getenv(name, default)
    return Path(v)


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass
class Config:
    # -----------------------------
    # Container paths (env overridable)
    # -----------------------------
    DATA_DIR: Path = _env_path("DATA_DIR", "/app/data")
    OUT_DIR: Path = _env_path("OUT_DIR", "/app/output")
    ZIP_NAME: str = _env_str("ZIP_NAME", "bullflagdetector.zip")
    DATASET_NAME: str = _env_str("DATASET_NAME", "dataset")

    # -----------------------------
    # Preprocessing
    # -----------------------------
    INCLUDE_POLE: bool = True
    POLE_LOOKBACK: int = 100
    MIN_SEG_LEN: int = 20
    DEDUP_IOU: float = 0.80

    # -----------------------------
    # Experiment / model selection (env overridable)
    # -----------------------------
    # Allowed: baseline | main
    MODEL_TYPE: Literal["baseline", "main"] = os.getenv("MODEL_TYPE", "baseline")  # type: ignore
    OVERFIT_ONE_BATCH: bool = False

    # -----------------------------
    # Dataset / features
    # -----------------------------
    SEQ_LEN: int = 256
    USE_LOG_RETURN: bool = True
    USE_OHLC_REL_FEATURES: bool = True
    PAD_MODE: Literal["zeros", "repeat_last"] = "repeat_last"

    # -----------------------------
    # Split
    # -----------------------------
    SEED: int = 42
    TRAIN_FRAC: float = 0.80
    VAL_FRAC: float = 0.10

    # -----------------------------
    # Training
    # -----------------------------
    BATCH_SIZE: int = 64
    EPOCHS: int = int(os.getenv("EPOCHS", "10"))
    LR: float = float(os.getenv("LR", "1e-3"))
    WEIGHT_DECAY: float = float(os.getenv("WEIGHT_DECAY", "1e-4"))

    # -----------------------------
    # Baseline (LSTM)
    # -----------------------------
    LSTM_HIDDEN: int = 32
    LSTM_NUM_LAYERS: int = 1
    LSTM_DROPOUT: float = 0.0

    # -----------------------------
    # Main model (CNN + Transformer)
    # -----------------------------
    CNN_CHANNELS: int = 64
    CNN_KERNEL: int = 5
    CNN_LAYERS: int = 2
    TF_D_MODEL: int = 64
    TF_NHEAD: int = 4
    TF_LAYERS: int = 2
    TF_DROPOUT: float = 0.1

    # -----------------------------
    # Output control
    # -----------------------------
    SAVE_TRAIN_CURVES: bool = True
    SAVE_CONFUSION_MATRIX: bool = True

    # -----------------------------
    # Derived paths
    # -----------------------------
    @property
    def DATASET_DIR(self) -> Path:
        return self.OUT_DIR / self.DATASET_NAME

    @property
    def INDEX_CSV(self) -> Path:
        return self.DATASET_DIR / "index.csv"

    @property
    def SEGMENTS_DIR(self) -> Path:
        return self.DATASET_DIR / "segments"

    @property
    def MODELS_DIR(self) -> Path:
        return self.OUT_DIR / "models"

    @property
    def METRICS_DIR(self) -> Path:
        return self.OUT_DIR / "metrics"

    @property
    def FIGURES_DIR(self) -> Path:
        return self.OUT_DIR / "figures"

    @property
    def WORK_DIR(self) -> Path:
        return self.OUT_DIR / "work"

    @property
    def MODEL_FILENAME(self) -> str:
        return f"{self.MODEL_TYPE}.pt"

    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / self.MODEL_FILENAME

    @property
    def METRICS_FILENAME(self) -> str:
        return f"test_metrics_{self.MODEL_TYPE}.json"

    @property
    def INFERENCE_FILENAME(self) -> str:
        return f"inference_demo_{self.MODEL_TYPE}.csv"

    @property
    def CONFUSION_FILENAME(self) -> str:
        return f"confusion_matrix_{self.MODEL_TYPE}.png"
