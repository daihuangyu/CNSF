#!/usr/bin/env bash

set -euo pipefail

nproc_per_node="${NPROC_PER_NODE:-4}"
exec torchrun \
    --standalone \
    --nproc_per_node="${nproc_per_node}" \
    scripts/train_cnsf.py \
    --config configs/training/cnsf.yaml \
    --output-dir outputs/cnsf \
    "$@"
