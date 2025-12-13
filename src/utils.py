import logging
import os
import re
import sys
from typing import Any, Optional

import pandas as pd


def setup_logger(name: str = "app", level: Optional[str] = None) -> logging.Logger:
    """
    Logger that prints to stdout so Docker can capture it.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    lvl = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, lvl, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


# --- strict timestamp parsing (minute resolution) ---

_MS_RE = re.compile(r"^\d{13}$")
_DT_RE_HM = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
_DT_RE_HMS = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")


def parse_timestamp(x: Any) -> pd.Timestamp:
    """
    STRICT parser used by preprocessing.

    Accept only:
      - 13-digit epoch milliseconds (e.g. 1707829200000)
      - 'YYYY-MM-DD HH:MM'
      - 'YYYY-MM-DD HH:MM:SS'

    Output:
      - tz-naive pandas Timestamp
      - floored to minute

    Anything else raises ValueError (so preprocessing can skip it deterministically).
    """
    s = str(x).strip()

    # 13-digit epoch milliseconds
    if _MS_RE.match(s):
        # parse as UTC then make tz-naive
        ts = pd.to_datetime(int(s), unit="ms", utc=True).tz_convert(None)
        return ts.floor("min")

    # datetime string with seconds
    if _DT_RE_HMS.match(s):
        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="raise")
        return ts.floor("min")

    # datetime string without seconds
    if _DT_RE_HM.match(s):
        ts = pd.to_datetime(s, format="%Y-%m-%d %H:%M", errors="raise")
        return ts.floor("min")

    raise ValueError(f"Unsupported timestamp format: {s}")
