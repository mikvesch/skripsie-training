#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
./setup_data.sh
conda run --no-capture-output -n synthrl-stage1 python finetune.py --config-name stage3 "$@"
