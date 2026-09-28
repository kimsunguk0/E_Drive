#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
docker build -t "$IMAGE" -f "$REL/docker/Dockerfile" "$REL/docker"
