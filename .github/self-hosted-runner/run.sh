#!/bin/bash
set -euo pipefail

# Spawn ephemeral GPU runners for CUDA 12 and CUDA 13.
# Each runner picks up one job then exits; the loop respawns it.
#
# Prerequisites:
#   - Run build.sh first
#   - gh CLI authenticated (used to fetch registration tokens)
#
# Usage: ./run.sh              (auto-detects repo from git remote)
#    or: ./run.sh cuda12       (single version only)
#    or: REPO=owner/repo ./run.sh

if [[ -z "${REPO:-}" ]]; then
    REPO="$(git remote get-url origin 2>/dev/null | sed -E 's#(https://github\.com/|git@github\.com:)##; s#\.git$##')"
fi
: "${REPO:?Could not detect repo. Set REPO=owner/repo}"

VERSIONS=("cuda12" "cuda13")
if [[ $# -ge 1 ]]; then
    VERSIONS=("$1")
fi

get_token() {
    gh api -X POST "repos/${REPO}/actions/runners/registration-token" --jq .token
}

run_runner() {
    local version="$1"
    echo "[${version}] Waiting for jobs..."
    while true; do
        TOKEN="$(get_token)"
        docker run --rm --gpus all \
            -e REPO_URL="https://github.com/${REPO}" \
            -e TOKEN="$TOKEN" \
            -e RUNNER_NAME="$(hostname)-${version}" \
            -e LABELS="self-hosted,gpu,${version}" \
            "masspcf-runner:${version}"
        echo "[${version}] Job finished. Respawning..."
        sleep 1
    done
}

# Run all requested versions in parallel
for version in "${VERSIONS[@]}"; do
    run_runner "$version" &
done

wait
