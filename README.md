# ViPE trajectory pipeline

This project runs NVIDIA ViPE on videos or ordered image directories and builds one complete camera-to-world trajectory from reliable overlapping windows. Window selection and pose stitching do not use GPS. Optional GPS fusion corrects translation drift while preserving ViPE camera orientations.

The reference environment is Ubuntu 24.04, Python 3.11, CUDA Toolkit 12.8, and an NVIDIA GPU with at least 8 GB VRAM.

## Setup

Prerequisites:

- NVIDIA driver and CUDA Toolkit 12.8
- [`uv`](https://docs.astral.sh/uv/)
- Linux or WSL with Bash
- About 20 GB of free space for packages, builds, and model weights

```bash
bash setup.sh
```

For a nonstandard CUDA installation or temporary directory:

```bash
CUDA_HOME=/opt/cuda-12.8 TMPDIR=/path/with/free/space bash setup.sh
```

The locked environment uses ViPE 1.2.0, Torch 2.9.0, torchvision 0.24.0, and CUDA 12.8. The setup script creates `.venv`, builds ViPE's native extension against the runtime Torch version, and verifies that it imports.

## Run ViPE

Process an image directory:

```bash
uv run --no-sync python -m vipe_pipeline.cli.run_vipe zavod70 \
	--frame-start 0 --frame-end 30 \
	--output vipe_smoke_test_out/windows/example_000_030/results
```

MP4 input is also supported:

```bash
uv run --no-sync python -m vipe_pipeline.cli.run_vipe /path/to/video.mp4 \
	--output /path/to/output
```

Image filenames are sorted lexicographically. Use zero-padded names. Directory images are resized to a 640-pixel maximum edge by default.

For downstream reconstruction, export ViPE's native dense-disparity SLAM map from a geometrically reliable segment:

```bash
uv run --no-sync python -m vipe_pipeline.cli.run_vipe zavod70 \
	--frame-start 70 --frame-end 126 \
	--pipeline static_vda --buffer 128 --image-max-edge 640 \
	--save-slam-map \
	--output vipe_smoke_test_out/gaussian/example_vipe_map_070_126
```

The map, poses, intrinsics, and RGB must come from the same ViPE run so they share one coordinate frame.

Record an isolated Zavod70 window experiment:

```bash
bash run_window_experiment.sh example_060_126 60 126 zavod70
```

`FRAME_START` is inclusive and `FRAME_END` is exclusive. Each experiment stores its ViPE artifacts, log, GPS evaluation, and plot under `vipe_smoke_test_out/windows/<run-name>`. Existing output directories are protected from accidental overwrite.

The equivalent direct command with all commonly adjusted options is:

```bash
uv run --no-sync python -m vipe_pipeline.cli.run_vipe zavod70 \
	--pipeline static_vda --buffer 128 --image-max-edge 640 \
	--frame-start 70 --frame-end 126 \
	--output vipe_smoke_test_out/windows/example_070_126/results
```

## Gaussian Splatting

Train Gaussian splats directly from the ViPE SLAM map, camera poses, intrinsics, and RGB:

```bash
uv run --no-sync python -m vipe_pipeline.cli.train_gaussians \
	vipe_smoke_test_out/gaussian/example_vipe_map_070_126 \
	--artifact zavod70 \
	--output-dir vipe_smoke_test_out/gaussian/example_model_070_126 \
	--max-gaussians 100000 --iterations 2000 --render-width 320
```

ViPE provides all scene geometry and calibrated cameras. `gsplat` is used only as the differentiable Gaussian rasterizer and standard PLY exporter; this pipeline does not use COLMAP or replace ViPE camera estimation. The first training run compiles gsplat's CUDA extension and can take a few extra minutes.

`--max-gaussians` limits the ViPE map points used for initialization. During training, gsplat's AbsGS strategy duplicates or splits Gaussians with strong image-plane gradients and prunes low-opacity or oversized Gaussians. The final count can therefore exceed the seed count. The refinement schedule is configurable with `--refine-start`, `--refine-stop`, `--refine-every`, and `--grow-gradient`.

Outputs:

- `model.pt`: reusable Gaussian parameters and ViPE camera metadata
- `model.ply`: standard 3D Gaussian Splatting PLY for external viewers
- `trajectory.mp4`: H.264 render along the ViPE camera path
- `metrics.json`: held-out PSNR, training loss, scene dimensions, and peak CUDA allocation

The validated adaptive 56-frame Zavod70 run uses 100,000 ViPE map seeds and 2,000 optimization steps at 320×240. It finishes with 137,442 Gaussians after four growth/pruning passes, improves held-out PSNR from 6.65 dB to 13.47 dB, and renders all 56 trajectory frames at 15 FPS. Peak Torch CUDA allocation is 78 MB after gsplat extension setup.

## Build Full Poses

Select a contiguous chain from completed windows using only ViPE kinematics and overlap consistency:

```bash
uv run --no-sync python -m vipe_pipeline.cli.select_windows zavod70 \
	--target-end 126 \
	--output-dir vipe_smoke_test_out/selection/example_internal
```

The selector writes `selection.json` and a ready-to-run `stitch_command.txt`. For the validated Zavod70 runs, the selected chain is:

```bash
uv run --no-sync python -m vipe_pipeline.cli.stitch_full_poses zavod70 \
	--window frames_000_030:0:30 \
	--window frames_020_037:20:37 \
	--window frames_030_054:30:54 \
	--window frames_050_070:50:70 \
	--window frames_062_126:62:126 \
	--output-dir vipe_smoke_test_out/full_pose/example_internal
```

The stitcher aligns overlapping positions with Sim(3), composes the corresponding world rotations, and averages duplicate orientations on SO(3). Its primary output is `poses.npz`:

- `data`: GPS-independent $N \times 4 \times 4$ camera-to-world poses
- `inds`: source frame indices
- `gps_aligned_data`: poses after one global similarity alignment, for evaluation and visualization only

The validated overlap-only result has 4.940 m GPS-aligned RMSE across all 126 frames. GPS does not influence window selection or the stitched pose chain; this metric is post-hoc evaluation. Use `--skip-gps-evaluation` to stitch a dataset without EXIF GPS.

## Reconstruct A Full Sequence

Window experiments now save their native ViPE SLAM maps by default. After automatic window selection, one resumable command exports any missing selected maps, validates map reruns against the poses that were selected, stitches cameras and maps into one coordinate frame, and trains the Gaussian model:

```bash
uv run --no-sync python -m vipe_pipeline.cli.reconstruct_full_sequence \
	/path/to/images \
	/path/to/selection.json \
	--source-runs-dir /path/to/window/runs \
	--output-dir /path/to/full_sequence \
	--max-gaussians 200000 --iterations 4000 --refine-stop 2000
```

The image directory name must match `dataset_name` in `selection.json`. Completed map, stitch, and Gaussian stages are reused on subsequent invocations. The first invocation saves `configuration.json`; later invocations must use the same inputs and training settings. If a legacy selected window does not already contain a map, the command reruns it up to `--map-attempts` times and accepts it only when its similarity-aligned positions and orientations reproduce the selected source poses. Rejected attempts are retained for diagnosis.

GPS is not required. Pass `--evaluate-gps` only when the images contain EXIF GPS and post-hoc trajectory metrics are wanted. The validated full Zavod70 reconstruction covers all 126 frames, starts from 200,000 merged ViPE map points, finishes with 378,699 Gaussians after 4,000 optimization steps, and reaches 12.80 dB held-out PSNR at 320×240.

## Optional GPS Fusion

Correct low-frequency translation drift while preserving the stitched orientations:

```bash
uv run --no-sync python -m vipe_pipeline.cli.fuse_gps_full_poses \
	vipe_smoke_test_out/full_pose/automatic_internal_v1/poses.npz \
	zavod70 \
	--output-dir vipe_smoke_test_out/full_pose_fused/example_conservative
```

Defaults use every fifth GPS record, 0.5 m ViPE motion sigma, 3 m horizontal GPS sigma, and 6 m vertical GPS sigma. The output contains fused poses, overlap-only poses, source indices, GPS positions, an anchor mask, metrics, and a comparison plot.

GPS-assisted metrics are agreement with observations used by the fusion objective, not independent ground-truth accuracy. The conservative validated configuration has 3.655 m RMSE and changes camera orientations by 0 degrees.

## Inspect Results

Evaluate an individual ViPE window:

```bash
uv run --no-sync python -m vipe_pipeline.cli.evaluate_trajectory \
	vipe_smoke_test_out/windows/frames_070_126/results/pose/zavod70.npz \
	zavod70 --frame-start 70 \
	--output /tmp/frames_070_126.png
```

View a complete pose artifact with camera frustums:

```bash
uv run --no-sync python -m vipe_pipeline.cli.view_trajectory \
	vipe_smoke_test_out/full_pose/automatic_internal_v1/poses.npz \
	--port 20544 --frustum-step 5
```

## Code Layout

- `vipe_pipeline/core/`: GPS loading, trajectory geometry, metrics, fusion, window artifacts, and shared validation
- `vipe_pipeline/cli/`: ViPE runner, Gaussian training/rendering, selection, stitching, fusion, evaluation, and visualization commands
- `run_window_experiment.sh`: guarded window experiment runner
- `EXPERIMENTS.md`: append-only record of all experiments, failures, settings, and measured outcomes
- `vipe_smoke_test_out/`: preserved generated artifacts and validation runs

The maintained trajectory path is selector to full-pose stitcher to optional full-pose fusion. Earlier position-only experiments remain documented and preserved in `EXPERIMENTS.md` and `vipe_smoke_test_out`, but their superseded implementations are no longer part of the supported code.