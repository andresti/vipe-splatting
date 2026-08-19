# Experiments

This directory contains non-primary workflows and analysis artifacts.

## Contents

- `scripts/`: experiment harnesses and report plot generators.
- `training/assets/`: generated figures referenced by `TRAINING.md`.

## Canonical Commands

Run a bounded ViPE window experiment:

```bash
bash experiments/scripts/run_window_experiment.sh RUN_NAME FRAME_START FRAME_END [IMAGE_DIR]
```

Regenerate training report plots:

```bash
uv run python experiments/scripts/build_training_report_plots.py
```

Use only the canonical paths in this directory.
