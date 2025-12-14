# BULL FLAG DETECTOR PROJECT


## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: Mészáros Bori Anna
- **Aiming for +1 Mark**: No

### Solution Description

This project tackles the automation of technical analysis in financial markets by detecting bullish vs. bearish flag-type formations in OHLC time series. Since manual pattern spotting is subjective, slow, and hard to scale across many instruments, we framed the task as a supervised classification problem: given a fixed-length price segment, the model predicts the corresponding pattern class.

To approach this, we built a hybrid deep learning classifier that combines 1D Convolutional Neural Networks (CNNs) with Transformer encoders. The CNN layers focus on learning local geometric cues (e.g., abrupt “pole”-like moves and consolidation structure), while the Transformer’s multi-head attention captures longer-range temporal dependencies across the segment. For reference, we also implemented a baseline LSTM.

A key limitation of the project is label quality: the annotations were not produced by domain experts, so the ground truth is inherently noisy and sometimes inconsistent. As a result, model performance is strongly influenced by the classic GIGO effect (“garbage in, garbage out”)—even a well-designed architecture cannot reliably learn sharp decision boundaries if the training labels do not accurately represent the underlying patterns.

#### Data Preparation
Pipeline summary:
- Data is acquired either by mounting the dataset to /app/data or by automatically downloading bullflagdetector.zip from SharePoint if missing.
- Preprocessing converts raw OHLC CSVs and JSON annotation spans into a training-ready dataset consisting of:
  - output/dataset/index.csv (metadata + label per sample)
  - output/dataset/segments/*.csv (cut segments with target label)
- Two models are trained:
  - Baseline model (reference)
  - Main model (stronger architecture)
- Training uses:
  - stratified train/validation/test split by label (target)
  - class-weighted CrossEntropyLoss to reduce majority-class collapse
- Evaluation runs on the test set and produces:
  - metrics JSON files (accuracy, macro F1, per-class metrics)
  - confusion matrix images
- Inference demo produces CSV outputs with example predictions.

All artifacts (dataset, models, metrics, figures, intermediate extraction) are written under output/ (mounted to /app/output in Docker).

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build --progress=plain -t bullflag-pipeline .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm -v "$(pwd -W)/data:/app/data" -v "$(pwd -W)/output:/app/output" bullflag-pipeline 2>&1 | tee log/run.log
```

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Scripts for downloading, loading, cleaning, and preprocessing the raw data.
    - `02-training.py`: The main script for defining the model and executing the training loop.
    - `03-evaluation.py`: Scripts for evaluating the trained model on test data and generating metrics.
    - `04-inference.py`: Script for running the model on new, unseen data to generate predictions.
    - `config.py`: Configuration file containing hyperparameters (e.g., epochs) and paths.
    - `utils.py`: Helper functions and utilities used across different scripts.
    - `dataset.py`: Helper function.
    - `models.py`: Helper function.
    -  `__innit__.py`: Helper function.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `versions': Experimenting, sanity checking.

- **`log/`**: Contains log files.
    - `run.log`: Example log file showing the output of a successful training run.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.


