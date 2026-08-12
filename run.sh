#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
conda run --no-capture-output -n synthrl-stage1 python train.py "$@"
