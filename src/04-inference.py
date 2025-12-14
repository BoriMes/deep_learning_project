from __future__ import annotations

import pandas as pd
import torch

from src.config import Config
from src.dataset import BullFlagDataset, LabelMap, build_label_map
from src.models import build_model
from src.utils import setup_logger, ensure_dirs

logger = setup_logger("infer")


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
def main():
    cfg = Config()
    ensure_dirs(cfg.OUT_DIR, cfg.METRICS_DIR)

    if not cfg.INDEX_CSV.exists():
        raise FileNotFoundError(f"Missing {cfg.INDEX_CSV}. Run preprocessing first.")
    if not cfg.MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {cfg.MODEL_PATH}. Run training first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(cfg.INDEX_CSV)
    ckpt = torch.load(cfg.MODEL_PATH, map_location="cpu")

    label_map = _label_map_from_checkpoint(ckpt)
    if label_map is None:
        labels = df["target"].astype(str).tolist()
        label_map = build_label_map(labels)

    num_classes = len(label_map.to_id)

    ds = BullFlagDataset(cfg.INDEX_CSV, cfg=cfg, label_map=label_map)

    model_type = ckpt.get("model_type", cfg.MODEL_TYPE)
    input_dim = int(ckpt.get("input_dim", ds[0][0].shape[-1]))

    cfg.MODEL_TYPE = str(model_type)  # type: ignore

    model = build_model(input_dim=input_dim, num_classes=num_classes, cfg=cfg).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    N = min(30, len(ds))
    rows = []

    for i in range(N):
        x, y = ds[i]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        p = int(logits.argmax(dim=1).item())

        rows.append(
            {
                "sample_id": int(i),
                "true": label_map.to_name[int(y.item())],
                "pred": label_map.to_name[p],
                "model_type": str(model_type),
            }
        )

    out_path = cfg.METRICS_DIR / cfg.INFERENCE_FILENAME
    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info(f"Saved inference demo: {out_path}")


if __name__ == "__main__":
    main()
