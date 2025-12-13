#!/usr/bin/env bash
set -e

echo "=== RUN START ==="
#python -u src/data_preprocessing.py
python -u src/train_lstm.py
echo "=== RUN END ==="