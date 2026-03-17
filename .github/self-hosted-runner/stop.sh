#!/bin/bash
set -euo pipefail

# Stop all running masspcf-runner containers and clean up.
# Usage: ./stop.sh

# Kill run.sh first so it doesn't respawn containers
if pkill -f "run\.sh"; then
    echo "Killed run.sh background processes."
fi

echo "Stopping masspcf-runner containers..."
CONTAINERS=$(docker ps -q --filter "ancestor=masspcf-runner:cuda12")

if [[ -z "$CONTAINERS" ]]; then
    echo "No running masspcf-runner containers found."
else
    docker stop $CONTAINERS
    echo "Stopped."
fi
