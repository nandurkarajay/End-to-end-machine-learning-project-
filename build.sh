#!/usr/bin/env bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running training pipeline..."
python src/components/data_ingestion.py

echo "Build complete."
