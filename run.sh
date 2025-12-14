#!/usr/bin/env bash
set -euo pipefail

DOWNLOAD_LINK="https://bmeedu-my.sharepoint.com/:u:/g/personal/gyires-toth_balint_vik_bme_hu/IQAlEFc87da4SLpRVTCs81KwAS3DG4Ft8JPtUKQe9vV5eng?download=1"

DATA_DIR="${DATA_DIR:-/app/data}"
OUT_DIR="${OUT_DIR:-/app/output}"
ZIP_NAME="${ZIP_NAME:-bullflagdetector.zip}"
ZIP_PATH="${DATA_DIR}/${ZIP_NAME}"

echo "=== PIPELINE START ==="
echo "DATA_DIR=${DATA_DIR}"
echo "OUT_DIR=${OUT_DIR}"
echo "ZIP_PATH=${ZIP_PATH}"

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

# If /app/data has no extracted json/csv AND no zip, download zip
HAS_ANY_DATA="$(find "${DATA_DIR}" -type f \( -name "*.json" -o -name "*.csv" \) -print -quit 2>/dev/null || true)"

if [[ ! -f "${ZIP_PATH}" && -z "${HAS_ANY_DATA}" ]]; then
  echo "No dataset found in ${DATA_DIR}. Downloading ${ZIP_NAME} ..."
  export DOWNLOAD_LINK ZIP_PATH
  python - <<'PY'
import os
import requests

url = os.environ["DOWNLOAD_LINK"]
out = os.environ["ZIP_PATH"]

r = requests.get(url, stream=True, timeout=300)
r.raise_for_status()

tmp = out + ".part"
written = 0
with open(tmp, "wb") as f:
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if not chunk:
            continue
        f.write(chunk)
        written += len(chunk)
        if written % (50 * 1024 * 1024) < 1024 * 1024:  # roughly every ~50MB
            print(f"  downloaded {written/1024/1024:.1f} MB", flush=True)

os.replace(tmp, out)
print(f"Downloaded -> {out}", flush=True)
PY
fi

echo "=== RUN PREPROCESS ==="
DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" ZIP_NAME="${ZIP_NAME}" python -m src.01-data-preprocessing

for MT in baseline main; do
  echo "=== RUN TRAIN (MODEL_TYPE=$MT) ==="
  MODEL_TYPE="$MT" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" ZIP_NAME="${ZIP_NAME}" python -m src.02-training

  echo "=== RUN EVAL (MODEL_TYPE=$MT) ==="
  MODEL_TYPE="$MT" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" ZIP_NAME="${ZIP_NAME}" python -m src.03-evaluation

  echo "=== RUN INFERENCE (MODEL_TYPE=$MT) ==="
  MODEL_TYPE="$MT" DATA_DIR="${DATA_DIR}" OUT_DIR="${OUT_DIR}" ZIP_NAME="${ZIP_NAME}" python -m src.04-inference
done

echo "=== PIPELINE END ==="
