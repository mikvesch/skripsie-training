#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Run seeds 1 and 3 sequentially by default. Optional Hydra overrides are
# forwarded to both runs, for example: ./run_stage1_seeds.sh train.num_workers=4
for seed in 1 3; do
    echo "Starting Stage 1 seed ${seed}"
    ./run.sh "seed=${seed}" "run_name=stage1/seed${seed}" "$@"
done
