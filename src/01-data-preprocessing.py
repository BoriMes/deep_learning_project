import re
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils import setup_logger, parse_timestamp
from src.config import Config

logger = setup_logger("preprocess")
OUT_DT_FMT = "%Y-%m-%d %H:%M"

# Progress logging frequency for long loops
PROGRESS_EVERY = 200


def norm_label(lbl: str) -> str:
    return re.sub(r"\s+", " ", str(lbl).strip())


def polarity(lbl: str) -> Optional[str]:
    u = lbl.upper()
    if "BULL" in u:
        return "BULL"
    if "BEAR" in u:
        return "BEAR"
    return None


def read_ohlc_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "timestamp" not in df.columns:
        raise ValueError(f"{csv_path}: missing 'timestamp' column")
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"{csv_path}: missing '{c}' column")

    df["timestamp"] = df["timestamp"].apply(parse_timestamp)
    df = df.set_index("timestamp").sort_index()

    if not df.index.is_unique:
        before = len(df)
        df = df[~df.index.duplicated(keep="last")]
        after = len(df)
        logger.warning(f"Dropped duplicate timestamps in {csv_path.name}: {before - after} rows")

    return df


def iter_labels_from_json(json_path: Path) -> List[Dict[str, Any]]:
    text = json_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("Empty JSON file")

    data = json.loads(text)
    if not isinstance(data, list):
        data = [data]

    out: List[Dict[str, Any]] = []
    for task in data:
        file_upload = task.get("file_upload") or Path(task.get("data", {}).get("csv", "")).name
        task_id = task.get("id")

        for ann in task.get("annotations", []):
            ann_id = ann.get("id")
            for res in ann.get("result", []):
                v = res.get("value", {})
                start_s = v.get("start")
                end_s = v.get("end")
                labels = v.get("timeserieslabels", [])

                if not (start_s and end_s and labels):
                    continue

                start_dt = parse_timestamp(start_s)
                end_dt = parse_timestamp(end_s)

                out.append(
                    {
                        "file_upload": str(file_upload),
                        "task_id": task_id,
                        "annotation_id": ann_id,
                        "start_dt": start_dt,
                        "end_dt": end_dt,
                        "label": norm_label(labels[0]),
                        "json_path": str(json_path),
                    }
                )
    return out


def build_csv_index(all_csvs: List[Path]) -> Dict[str, Path]:
    return {p.name: p for p in all_csvs}


def fuzzy_match_csv(file_upload: str, by_name: Dict[str, Path]) -> Optional[Path]:
    if file_upload in by_name:
        return by_name[file_upload]

    name = Path(file_upload).name
    if name in by_name:
        return by_name[name]

    name2 = re.sub(r"^[0-9a-fA-F]+-", "", name)
    if name2 in by_name:
        return by_name[name2]

    for k, p in by_name.items():
        if name2 in k or k in name2:
            return p

    return None


def standardize_pole_start(
    ohlc: pd.DataFrame,
    flag_start_dt: pd.Timestamp,
    label: str,
    lookback: int,
) -> Optional[pd.Timestamp]:
    if not ohlc.index.is_monotonic_increasing:
        ohlc = ohlc.sort_index()
    if not ohlc.index.is_unique:
        ohlc = ohlc[~ohlc.index.duplicated(keep="last")]

    pos_arr = ohlc.index.get_indexer([flag_start_dt], method="nearest")
    pos = int(pos_arr[0])
    if pos < 0 or pos >= len(ohlc):
        return None

    flag_bar = ohlc.iloc[pos]
    pol = polarity(label)
    if pol is None:
        return None

    best_ts = None
    best_slope = float("-inf")

    for i in range(1, lookback + 1):
        cpos = pos - i
        if cpos < 0:
            break

        cand = ohlc.iloc[cpos]

        if pol == "BULL":
            anchor = float(flag_bar["high"])
            cand_price = float(cand["low"])
            change = anchor - cand_price
        else:
            anchor = float(flag_bar["low"])
            cand_price = float(cand["high"])
            change = cand_price - anchor

        if change <= 0:
            continue

        slope = change / i
        if slope > best_slope:
            best_slope = slope
            best_ts = ohlc.index[cpos]

    return best_ts


def iou_1d(a0: pd.Timestamp, a1: pd.Timestamp, b0: pd.Timestamp, b1: pd.Timestamp) -> float:
    if a0 > a1:
        a0, a1 = a1, a0
    if b0 > b1:
        b0, b1 = b1, b0

    inter_start = max(a0, b0)
    inter_end = min(a1, b1)
    if inter_start >= inter_end:
        return 0.0

    inter = (inter_end - inter_start).total_seconds()
    union = (max(a1, b1) - min(a0, b0)).total_seconds()
    return float(inter / union) if union > 0 else 0.0


def dedup_labels(labels: List[Dict[str, Any]], iou_threshold: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for x in labels:
        by_file.setdefault(x["file_upload"], []).append(x)

    for _, lst in by_file.items():
        lst = sorted(lst, key=lambda z: (z["segment_start_dt"], z["segment_end_dt"]))
        kept: List[Dict[str, Any]] = []
        for x in lst:
            ok = True
            for k in kept:
                if x["label"] == k["label"]:
                    if (
                        iou_1d(
                            x["segment_start_dt"],
                            x["segment_end_dt"],
                            k["segment_start_dt"],
                            k["segment_end_dt"],
                        )
                        >= iou_threshold
                    ):
                        ok = False
                        break
            if ok:
                kept.append(x)
        out.extend(kept)

    return out


def cut_segment(ohlc: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp, min_len: int) -> Optional[pd.DataFrame]:
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt
    seg = ohlc.loc[(ohlc.index >= start_dt) & (ohlc.index <= end_dt)].copy()
    if len(seg) < min_len:
        return None
    return seg


def main():
    cfg = Config()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = cfg.DATA_DIR / cfg.ZIP_NAME

    workdir = cfg.WORK_DIR
    extract_dir = workdir / "extracted"

    logger.info("=== DATA PREPROCESSING START ===")
    logger.info(f"data_dir={cfg.DATA_DIR}")
    logger.info(f"out_dir={cfg.OUT_DIR}")
    logger.info(f"zip_path={zip_path}")
    logger.info(f"include_pole={cfg.INCLUDE_POLE} pole_lookback={cfg.POLE_LOOKBACK}")
    logger.info(f"min_seg_len={cfg.MIN_SEG_LEN} dedup_iou={cfg.DEDUP_IOU}")

    # --- choose input mode: zip or extracted folder ---
    if zip_path.exists():
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Mode: ZIP -> extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    else:
        extract_dir = cfg.DATA_DIR
        logger.info("Mode: EXTRACTED FOLDER (no zip found)")

        # sanity check: do we see any expected files?
        json_paths_probe = list(extract_dir.rglob("*.json"))
        csv_paths_probe = list(extract_dir.rglob("*.csv"))
        if not json_paths_probe and not csv_paths_probe:
            raise FileNotFoundError(
                f"No .json/.csv found under {extract_dir}.\n"
                f"Expected a decompressed dataset folder mounted to /app/data, "
                f"or a zip file named {cfg.ZIP_NAME}."
            )

    # --- proceed with normal discovery under extract_dir ---
    json_paths = sorted(extract_dir.rglob("*.json"))
    csv_paths = sorted(extract_dir.rglob("*.csv"))
    logger.info(f"Found JSON files: {len(json_paths)}")
    logger.info(f"Found CSV files:  {len(csv_paths)}")

    by_name = build_csv_index(csv_paths)

    raw_labels: List[Dict[str, Any]] = []
    json_failed = 0

    logger.info("Parsing JSON label files...")
    for i, jp in enumerate(json_paths, start=1):
        try:
            raw_labels.extend(iter_labels_from_json(jp))
        except Exception as e:
            json_failed += 1
            logger.warning(f"JSON parse failed: {jp} ({e})")

        if i % PROGRESS_EVERY == 0 or i == len(json_paths):
            logger.info(
                f"  JSON progress: {i}/{len(json_paths)} files | spans_so_far={len(raw_labels)} | failed={json_failed}"
            )

    logger.info(f"Raw label spans found: {len(raw_labels)}")
    if json_failed:
        logger.warning(f"JSON files failed: {json_failed}")

    ohlc_cache: Dict[Path, pd.DataFrame] = {}
    csv_failed_once: set[Path] = set()

    enriched: List[Dict[str, Any]] = []
    missing_csv = 0
    csv_read_fail = 0
    no_pole = 0
    fallback_pole = 0

    logger.info("Computing pole start + segment bounds...")
    for i, lab in enumerate(raw_labels, start=1):
        csv_path = fuzzy_match_csv(lab["file_upload"], by_name)
        if csv_path is None:
            missing_csv += 1
            if i % PROGRESS_EVERY == 0 or i == len(raw_labels):
                logger.info(
                    f"  Enrich progress: {i}/{len(raw_labels)} | enriched={len(enriched)} "
                    f"missing_csv={missing_csv} csv_read_fail={csv_read_fail} "
                    f"no_pole={no_pole} fallback_pole={fallback_pole} cache_csvs={len(ohlc_cache)}"
                )
            continue

        if csv_path not in ohlc_cache:
            try:
                ohlc_cache[csv_path] = read_ohlc_csv(csv_path)
            except Exception as e:
                csv_read_fail += 1
                if csv_path not in csv_failed_once:
                    logger.warning(f"CSV read failed: {csv_path} ({e})")
                    csv_failed_once.add(csv_path)
                if i % PROGRESS_EVERY == 0 or i == len(raw_labels):
                    logger.info(
                        f"  Enrich progress: {i}/{len(raw_labels)} | enriched={len(enriched)} "
                        f"missing_csv={missing_csv} csv_read_fail={csv_read_fail} "
                        f"no_pole={no_pole} fallback_pole={fallback_pole} cache_csvs={len(ohlc_cache)}"
                    )
                continue

        ohlc = ohlc_cache[csv_path]
        start_dt = lab["start_dt"]
        end_dt = lab["end_dt"]
        label = lab["label"]

        pole_start = start_dt
        if cfg.INCLUDE_POLE:
            pole_start = standardize_pole_start(ohlc, start_dt, label, lookback=cfg.POLE_LOOKBACK)
            if pole_start is None:
                no_pole += 1
                pole_start = start_dt
                fallback_pole += 1

        enriched.append(
            {
                **lab,
                "csv_path": str(csv_path),
                "pole_start_dt": pole_start,
                "segment_start_dt": pole_start,
                "segment_end_dt": end_dt,
            }
        )

        if i % PROGRESS_EVERY == 0 or i == len(raw_labels):
            logger.info(
                f"  Enrich progress: {i}/{len(raw_labels)} | enriched={len(enriched)} "
                f"missing_csv={missing_csv} csv_read_fail={csv_read_fail} "
                f"no_pole={no_pole} fallback_pole={fallback_pole} cache_csvs={len(ohlc_cache)}"
            )

    logger.info(f"Enriched labels (with poles): {len(enriched)}")
    logger.warning(
        f"Skipped: missing_csv={missing_csv}, csv_read_fail={csv_read_fail}, no_pole={no_pole} | "
        f"fallback_pole_used={fallback_pole}"
    )

    before = len(enriched)
    logger.info("Deduplicating labels...")
    deduped = dedup_labels(enriched, iou_threshold=cfg.DEDUP_IOU)
    logger.info(f"Dedup: {before} -> {len(deduped)} (iou_thr={cfg.DEDUP_IOU})")

    outdir = cfg.DATASET_DIR
    seg_dir = outdir / "segments"
    if outdir.exists():
        shutil.rmtree(outdir)
    seg_dir.mkdir(parents=True, exist_ok=True)

    index_rows: List[Dict[str, Any]] = []
    sample_id = 0
    too_short = 0

    logger.info("Saving segment CSVs...")
    for i, x in enumerate(deduped, start=1):
        csv_path = Path(x["csv_path"])
        ohlc = ohlc_cache[csv_path]

        seg = cut_segment(ohlc, x["segment_start_dt"], x["segment_end_dt"], min_len=cfg.MIN_SEG_LEN)
        if seg is None:
            too_short += 1
        else:
            out_name = f"sample_{sample_id:06d}.csv"
            out_path = seg_dir / out_name

            seg_out = seg.reset_index().rename(columns={"index": "timestamp"})
            seg_out["timestamp"] = seg_out["timestamp"].dt.strftime(OUT_DT_FMT)
            seg_out["target"] = x["label"]
            seg_out.to_csv(out_path, index=False)

            index_rows.append(
                {
                    "sample_id": sample_id,
                    "segment_path": str(out_path.as_posix()),
                    "target": x["label"],
                    "source_csv": x["csv_path"],
                    "file_upload": x["file_upload"],
                    "pole_start": x["pole_start_dt"].strftime(OUT_DT_FMT),
                    "flag_start": x["start_dt"].strftime(OUT_DT_FMT),
                    "flag_end": x["end_dt"].strftime(OUT_DT_FMT),
                    "length": int(len(seg)),
                    "task_id": x.get("task_id"),
                    "annotation_id": x.get("annotation_id"),
                    "json_path": x.get("json_path"),
                }
            )
            sample_id += 1

        if i % PROGRESS_EVERY == 0 or i == len(deduped):
            logger.info(f"  Save progress: {i}/{len(deduped)} | saved={sample_id} too_short={too_short}")

    df_index = pd.DataFrame(index_rows)
    out_index = outdir / "index.csv"
    df_index.to_csv(out_index, index=False)

    logger.info(f"Saved samples: {sample_id}")
    logger.warning(f"too_short_after_cut: {too_short}")
    logger.info(f"Wrote index: {out_index}")

    if len(df_index) > 0:
        logger.info("Label distribution:")
        vc = df_index["target"].value_counts()
        for k, v in vc.items():
            logger.info(f"  {k}: {v}")

    logger.info("=== DATA PREPROCESSING END ===")


if __name__ == "__main__":
    main()
