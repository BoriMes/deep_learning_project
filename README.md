# Deep Learning Class (VITMMA19) – Bull-Flag Detector

This repository implements a full deep-learning pipeline for **bull/bear flag detection** on labeled OHLC time series:
- data preprocessing (segment dataset creation),
- model training,
- evaluation,
- inference demo,
all runnable inside a Docker container.

---

## Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: _[Fill in]_
- **Aiming for +1 Mark**: _[Yes/No]_

---

## Solution Description

### Problem
Given OHLC time series and labeled time intervals (bull/bear flags), the goal is to learn a classifier that predicts the label of an extracted segment.

### Data representation
Each labeled interval is converted into a **fixed-length** multivariate sequence. The dataset builder:
- extracts the interval from the matching OHLC CSV,
- optionally includes a “pole” lookback window,
- enforces a minimum segment length,
- deduplicates overlapping samples,
- pads/truncates to a fixed sequence length.

### Models
Two model types are supported:
- **`baseline`**: LSTM classifier (reference model)
- **`main`**: CNN feature extractor + TransformerEncoder + classification head

### Training
- train/val/test split (configurable fractions)
- optional class balancing via sampling
- model checkpoint saved to `output/models/`

### Evaluation
Evaluation is performed on the test split and includes:
- **accuracy**
- **macro F1**
- **confusion matrix** (saved as an image when enabled)

---

## Data Preparation

### Expected input location
Mount your local `./data` folder into the container at `/app/data`.

The preprocessing step searches for label JSON(s) and OHLC CSV(s) in one of these formats:
1) **ZIP mode (default)**: `/app/data/bullflagdetector.zip`
2) **Folder mode**: recursive search under `/app/data`

### OHLC CSV requirements
Each OHLC CSV must contain (case-insensitive):
- `timestamp`, `open`, `high`, `low`, `close`

Supported `timestamp` formats:
- 13-digit epoch milliseconds
- `YYYY-MM-DD HH:MM`
- `YYYY-MM-DD HH:MM:SS`

### Label JSON requirements (high-level)
The preprocessing expects labeled time intervals containing:
- `start` timestamp
- `end` timestamp
- a label that can be mapped to **BULL** or **BEAR** (case-insensitive match)

Each labeled interval must be matchable to a source OHLC CSV and must fall within its available time range.

---

## Logging Requirements (for submission)

When running the container, **all stdout/stderr** should be captured into:

- `log/run.log`

The pipeline logs include:
- configuration snapshot (hyperparameters, feature flags),
- confirmation of data loading and dataset sizes,
- model type and parameter counts,
- final quick test metrics after training,
- test evaluation metrics (**accuracy**, **macro F1**) and confusion matrix path,
- inference demo output path.

> Ensure the `log/` directory exists on the host before running.

---

## Docker Instructions

This project is containerized using Docker.

### Build
Run in the repository root:
```bash
docker build --progress=plain -t bullflag-pipeline .

### Run
Run in the repository root:
```bash
docker run --rm \
  -v "$(pwd -W)/data:/app/data" \
  -v "$(pwd -W)/output:/app/output" \
  bullflag-pipeline 2>&1 | tee log/run.log


