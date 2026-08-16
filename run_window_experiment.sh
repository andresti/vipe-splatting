#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="${1:-}"
FRAME_START="${2:-}"
FRAME_END="${3:-}"
INPUT_PATH="${4:-zavod70}"

if [[ -z "${RUN_NAME}" || -z "${FRAME_START}" || -z "${FRAME_END}" ]]; then
  echo "Usage: bash run_window_experiment.sh RUN_NAME FRAME_START FRAME_END [IMAGE_DIR]" >&2
  exit 1
fi

if [[ ! "${RUN_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_NAME may contain only letters, numbers, dots, underscores, and hyphens." >&2
  exit 1
fi

if [[ ! "${FRAME_START}" =~ ^[0-9]+$ || ! "${FRAME_END}" =~ ^[0-9]+$ || "${FRAME_END}" -le "${FRAME_START}" ]]; then
  echo "FRAME_START and FRAME_END must be integers with FRAME_END greater than FRAME_START." >&2
  exit 1
fi

if [[ ! -d "${INPUT_PATH}" ]]; then
  echo "Image directory not found: ${INPUT_PATH}" >&2
  exit 1
fi

RUN_DIR="${ROOT_DIR}/vipe_smoke_test_out/windows/${RUN_NAME}"
if [[ -e "${RUN_DIR}" ]]; then
  echo "Experiment directory already exists; choose a new RUN_NAME: ${RUN_DIR}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
OUTPUT_DIR="${RUN_DIR}/results"

uv run --directory "${ROOT_DIR}" --no-sync python -m vipe_pipeline.cli.run_vipe \
  "${INPUT_PATH}" \
  --pipeline "${PIPELINE:-static_vda}" \
  --buffer "${SLAM_BUFFER:-128}" \
  --image-max-edge "${IMAGE_MAX_EDGE:-640}" \
  --frame-start "${FRAME_START}" \
  --frame-end "${FRAME_END}" \
  --save-slam-map \
  --output "${OUTPUT_DIR}" 2>&1 | tee "${RUN_DIR}/run.log"

ARTIFACT_NAME="$(basename "${INPUT_PATH%/}")"
uv run --directory "${ROOT_DIR}" --no-sync python -m vipe_pipeline.cli.evaluate_trajectory \
  "${OUTPUT_DIR}/pose/${ARTIFACT_NAME}.npz" \
  "${INPUT_PATH}" \
  --frame-start "${FRAME_START}" \
  --output "${RUN_DIR}/gps_comparison.png" \
  --metrics-output "${RUN_DIR}/metrics.json"

echo "Experiment complete: ${RUN_DIR}"
