#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --directory "${ROOT_DIR}" --no-sync python -m vipe_pipeline.cli.reconstruct "$@"