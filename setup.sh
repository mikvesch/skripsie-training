#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
conda env create --file environment.yml
echo "Environment created. Run: ./run.sh"
