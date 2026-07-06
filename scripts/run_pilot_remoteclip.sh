#!/usr/bin/env bash
# Pilot run: RemoteCLIP (ViT-B/32) + EuroSAT via O-TPT (orthogonality loss).
# Args match otpt-base/scripts/test_tpt_otpt_fg.sh: --tta_steps 1, --lr 5e-3,
# --n_ctx 4, --ctx_init a_photo_of_a, --lambda_term 18, --run_type tpt_otpt.
# EuroSAT test images are auto-downloaded by torchvision under $DATA_ROOT.
#
# Usage:
#   bash scripts/run_pilot_remoteclip.sh [DATA_ROOT] [GPU_ID]
#     DATA_ROOT — where to put/find downloaded datasets (default: ./data)
#     GPU_ID    — CUDA index (default: 0)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${1:-$REPO_ROOT/data}"
GPU_ID="${2:-0}"

mkdir -p "$DATA_ROOT" "$REPO_ROOT/results"

cd "$REPO_ROOT/otpt-base"
python otpt_classification.py "$DATA_ROOT" \
    --test_sets eurosat_tv \
    --arch remoteclip \
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
    --csv_log "$REPO_ROOT/results/pilot_remoteclip_eurosat.csv"
