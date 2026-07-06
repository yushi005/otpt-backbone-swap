#!/usr/bin/env bash
# Pilot run: BioMedCLIP (PubMedBERT + ViT-B/16) + DermaMNIST (224x224) via O-TPT.
# Uses the same TTA protocol as test_tpt_otpt_fg.sh; DermaMNIST is downloaded
# automatically via the medmnist package on first run.
#
# Usage:
#   bash scripts/run_pilot_biomedclip.sh [DATA_ROOT] [GPU_ID]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${1:-$REPO_ROOT/data}"
GPU_ID="${2:-0}"

mkdir -p "$DATA_ROOT" "$REPO_ROOT/results"

cd "$REPO_ROOT/otpt-base"
python otpt_classification.py "$DATA_ROOT" \
    --test_sets dermamnist \
    --arch biomedclip \
    -b 64 \
    --tta_steps 1 \
    --lr 5e-3 \
    --n_ctx 4 \
    --ctx_init a_photo_of_a \
    --lambda_term 18 \
    --run_type tpt_otpt \
    --gpu "$GPU_ID" \
    --seed 0 \
    --tpt \
    --print-freq 200 \
    --workers 4 \
    --csv_log "$REPO_ROOT/results/pilot_biomedclip_dermamnist.csv"
