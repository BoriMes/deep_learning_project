from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LabelMap:
    to_id: Dict[str, int]
    to_name: Dict[int, str]


def build_label_map(labels: List[str]) -> LabelMap:
    uniq = sorted(set(labels))
    to_id = {name: i for i, name in enumerate(uniq)}
    to_name = {i: name for name, i in to_id.items()}
    return LabelMap(to_id=to_id, to_name=to_name)


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def load_segment_features(
    csv_path: Path,
    seq_len: int,
) -> np.ndarray:
    """
    Returns (seq_len, feat_dim) float32.
    Feature: log returns of close + (optional) normalized OHLC spreads.
    Robust to extra columns.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "close" not in df.columns:
        raise ValueError(f"{csv_path}: missing close")

    # Use close series
    close = df["close"].map(_safe_float).astype(np.float32).values
    close = np.clip(close, 1e-8, np.inf)

    # log returns (length n-1), pad one at start
    logp = np.log(close)
    r = np.diff(logp, prepend=logp[0]).astype(np.float32)  # same length as close

    # Optional extra simple signals if OHLC present
    feats = [r]

    if all(c in df.columns for c in ["open", "high", "low"]):
        o = df["open"].map(_safe_float).astype(np.float32).values
        h = df["high"].map(_safe_float).astype(np.float32).values
        l = df["low"].map(_safe_float).astype(np.float32).values

        # scale-invariant spreads relative to close
        oc = (o - close) / close
        hc = (h - close) / close
        lc = (l - close) / close
        feats += [oc.astype(np.float32), hc.astype(np.float32), lc.astype(np.float32)]

    x = np.stack(feats, axis=1)  # (T, F)

    # pad / truncate to seq_len
    T, F = x.shape
    if T >= seq_len:
        x = x[-seq_len:, :]
    else:
        pad = np.zeros((seq_len - T, F), dtype=np.float32)
        x = np.concatenate([pad, x], axis=0)

    return x.astype(np.float32)


class BullFlagDataset(Dataset):
    def __init__(
        self,
        index_csv: Path,
        seq_len: int,
        label_map: LabelMap,
    ):
        self.index_csv = index_csv
        self.seq_len = seq_len
        self.df = pd.read_csv(index_csv)
        if "segment_path" not in self.df.columns or "target" not in self.df.columns:
            raise ValueError(f"{index_csv}: must contain segment_path and target columns")

        self.label_map = label_map
        self.paths = [Path(p) for p in self.df["segment_path"].astype(str).tolist()]
        self.targets = [label_map.to_id[t] for t in self.df["target"].astype(str).tolist()]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = load_segment_features(self.paths[i], seq_len=self.seq_len)  # (L, F)
        y = self.targets[i]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
