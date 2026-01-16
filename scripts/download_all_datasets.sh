#!/usr/bin/env bash
set -euo pipefail

python -m src.data.download --config config/data_sources.yaml --output data/raw
