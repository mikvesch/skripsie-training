#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

dataset_dir="datasets/surge"
expected_sha256="fb6d0e75513a9969975fda6543aa1baee5a74e59bf90d597683dcc6076d1098d"
default_url="https://github.com/mikvesch/skripsie-training/releases/download/data-v1/surge-dataset-v1.tar.gz"
dataset_url="${SURGE_DATASET_URL:-$default_url}"

if [[ -d "$dataset_dir/wav" && -d "$dataset_dir/spectrogram" ]]; then
    echo "Surge dataset is already present at $dataset_dir"
    exit 0
fi

archive="$(mktemp --tmpdir surge-dataset-v1.XXXXXX.tar.gz)"
trap 'rm -f "$archive"' EXIT

echo "Downloading the Surge dataset (about 1.5 GB)..."
curl --fail --location --retry 3 --output "$archive" "$dataset_url"

actual_sha256="$(sha256sum "$archive" | cut -d ' ' -f 1)"
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
    echo "Surge dataset checksum mismatch" >&2
    echo "Expected: $expected_sha256" >&2
    echo "Actual:   $actual_sha256" >&2
    exit 1
fi

mkdir -p datasets
tar -xzf "$archive" -C datasets
echo "Surge dataset installed at $dataset_dir"
