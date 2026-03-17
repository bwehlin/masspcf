#!/bin/bash
set -euo pipefail

# Build runner images for CUDA 12 and CUDA 13.
# Usage: ./build.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Building CUDA 12 runner image ==="
docker build \
    --build-arg CUDA_VERSION=12.8.0 \
    -t masspcf-runner:cuda12 \
    "$SCRIPT_DIR"

echo "=== Building CUDA 13 runner image ==="
docker build \
    --build-arg CUDA_VERSION=13.1.1 \
    -t masspcf-runner:cuda13 \
    "$SCRIPT_DIR"

echo ""
echo "Done. Images available:"
echo "  masspcf-runner:cuda12"
echo "  masspcf-runner:cuda13"
