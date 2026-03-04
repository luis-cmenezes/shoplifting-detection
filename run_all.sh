#!/bin/bash
export PYTHONUNBUFFERED=1

LOG_DIR="logs"
mkdir -p $LOG_DIR

echo "Running all scripts sequentially for full training..."

echo "[BASH] Downloading data..."
uv run scripts/download_data.py --config scripts/config.yaml \
  > $LOG_DIR/download.log 2>&1

echo "[BASH] Preprocessing I3D data..."
uv run scripts/preprocess_i3d.py --config scripts/config.yaml \
  > $LOG_DIR/preprocess_i3d.log 2>&1

echo "[BASH] Preprocessing Timesformer data..."
uv run scripts/preprocess_timesformer.py --config scripts/config.yaml \
  > $LOG_DIR/preprocess_timesformer.log 2>&1

echo "[BASH] Training I3D model..."
uv run scripts/train_i3d.py --config scripts/config.yaml \
  > $LOG_DIR/train_i3d.log 2>&1

echo "[BASH] Training Timesformer model..."
uv run scripts/train_timesformer.py --config scripts/config.yaml \
  > $LOG_DIR/train_timesformer.log 2>&1