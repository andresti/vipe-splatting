#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMPDIR="${TMPDIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/vipe-trajectory-pipeline/tmp}"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

require_command uv
require_command nvidia-smi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
  elif [[ -x /usr/local/cuda-12.8/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda-12.8
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    CUDA_HOME=/usr/local/cuda
  else
    echo "CUDA Toolkit 12.8 was not found." >&2
    echo "Install it or set CUDA_HOME to its installation directory." >&2
    exit 1
  fi
fi

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA 12.8 compiler not found at ${CUDA_HOME}/bin/nvcc." >&2
  echo "Install CUDA Toolkit 12.8 or set CUDA_HOME to its installation directory." >&2
  exit 1
fi

if ! "${CUDA_HOME}/bin/nvcc" --version | grep -q 'release 12\.8'; then
  echo "ViPE requires CUDA Toolkit 12.8; found:" >&2
  "${CUDA_HOME}/bin/nvcc" --version | tail -1 >&2
  exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver check failed." >&2
  exit 1
fi

mkdir -p "${TMPDIR}"
cd "${ROOT_DIR}"

export CUDA_HOME TMPDIR
export PATH="${CUDA_HOME}/bin:${PATH}"

echo "[1/3] Creating the Python 3.11 uv environment"
uv sync --frozen --python 3.11

echo "[2/3] Verifying the pinned CUDA runtime and ViPE extension"
if ! uv run --no-sync python -c 'import torch, torchvision, vipe_ext'; then
  echo "The existing environment has an inconsistent CUDA runtime; recreating it."
  rm -rf "${ROOT_DIR}/.venv"
  uv sync --frozen --python 3.11
fi

uv run --no-sync python -c 'import torch, torchvision, vipe_ext; print(f"torch={torch.__version__} cuda={torch.version.cuda}"); print(f"torchvision={torchvision.__version__}"); print("vipe_ext=OK")'

echo "[3/3] Setup complete"
echo "Run a window with: bash run_window_experiment.sh RUN_NAME FRAME_START FRAME_END IMAGE_DIR"