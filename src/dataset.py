from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config


@dataclass(frozen=True)
class LabelMap:
    to_id: Dict[str, int]
    to_name: Dict[int, str]


def build_label_map(labels: List[str]) -> LabelMap:
    uniq = sorted(set(str(l) for l in labels))
    to_id = {name: i for i, name in enumerate(uniq)}
    to_name = {i: name for name, i in to_id.items()}
    return LabelMap(to_id=to_id, to_name=to_name)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def load_segment_features(csv_path: Path, cfg: Config) -> np.ndarray:
    """Load a single segment CSV and return a (SEQ_LEN, feat_dim) float32 array."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"{csv_path}: missing close")

    close = df["close"].map(_safe_float).astype(np.float32).values
    close = np.clip(close, 1e-8, np.inf)

    feats: List[np.ndarray] = []

    if cfg.USE_LOG_RETURN:
        logp = np.log(close)
        r = np.diff(logp, prepend=logp[0]).astype(np.float32)
        feats.append(r)

    if cfg.USE_OHLC_REL_FEATURES and all(c in df.columns for c in ["open", "high", "low"]):
        o = df["open"].map(_safe_float).astype(np.float32).values
        h = df["high"].map(_safe_float).astype(np.float32).values
        l = df["low"].map(_safe_float).astype(np.float32).values

        oc = (o - close) / close
        hc = (h - close) / close
        lc = (l - close) / close
        feats += [oc.astype(np.float32), hc.astype(np.float32), lc.astype(np.float32)]

    if not feats:
        raise ValueError("No features selected (check Config.USE_LOG_RETURN / USE_OHLC_REL_FEATURES).")

    x = np.stack(feats, axis=1)  # (T, F)

    # pad / truncate to SEQ_LEN (left-pad)
    T, F = x.shape
    L = int(cfg.SEQ_LEN)

    if T >= L:
        x = x[-L:, :]
    else:
        pad_len = L - T
        if cfg.PAD_MODE == "zeros":
            pad = np.zeros((pad_len, F), dtype=np.float32)
        else:  # repeat_last
            if T == 0:
                pad = np.zeros((pad_len, F), dtype=np.float32)
            else:
                last = x[-1:, :]
                pad = np.repeat(last, repeats=pad_len, axis=0).astype(np.float32)
        x = np.concatenate([pad, x], axis=0)

    return x.astype(np.float32)


class BullFlagDataset(Dataset):
    def __init__(self, index_csv: Path, cfg: Config, label_map: LabelMap):
        self.cfg = cfg
        self.df = pd.read_csv(index_csv)

        if "segment_path" not in self.df.columns or "target" not in self.df.columns:
            raise ValueError(f"{index_csv}: must contain segment_path and target columns")

        self.label_map = label_map
        self.paths = [Path(p) for p in self.df["segment_path"].astype(str).tolist()]
        self.targets = [label_map.to_id[str(t)] for t in self.df["target"].astype(str).tolist()]

        self.groups = None
        if "source_csv" in self.df.columns:
            self.groups = self.df["source_csv"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = load_segment_features(self.paths[i], cfg=self.cfg)  # (L, F)
        y = self.targets[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
