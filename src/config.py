from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # Container paths
    DATA_DIR: Path = Path("/app/data")     # ide mountolja a zipet
    OUT_DIR: Path = Path("/app/output")   # ide írja a datasetet

    ZIP_NAME: str = "bullflagdetector.zip"

    # Preprocessing
    INCLUDE_POLE: bool = True
    POLE_LOOKBACK: int = 100
    MIN_SEG_LEN: int = 20
    DEDUP_IOU: float = 0.80

    # Output dataset
    DATASET_NAME: str = "dataset"
