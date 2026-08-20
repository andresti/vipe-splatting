# ViPE reconstruction pipeline

This project runs NVIDIA ViPE on videos or ordered image directories, evaluates the recovered camera trajectory, and trains a Gaussian scene directly from ViPE cameras and geometry. The preferred path is one full-sequence ViPE run; window stitching is retained separately as a recovery option.

The reference environment is Ubuntu 24.04, Python 3.11, CUDA Toolkit 12.8, and an NVIDIA GPU with at least 8 GB VRAM.

## Demo

- [Watch the end-to-end pipeline demo](demo/end_to_end_demo.mp4)
- [Watch the trajectory render only](demo/trajectory.mp4)

The [Gaussian Splatting training report](TRAINING.md) explains the optimization process, experiment comparisons, evaluation graphs, and final configuration choice.

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

Setup also downloads the 126-image Zavod70 dataset from the provided public Google Drive folder when `zavod70/` is absent. It verifies the JPEG count before moving the completed download into place and never modifies an existing dataset directory. Set `DATASET_DIR` or `DATASET_URL` to override the destination or source.

Google Drive public-link throttling or rate limiting can occasionally block automated downloads. If this happens, pre-download and extract the dataset into `zavod70/` (or your `DATASET_DIR`) so it contains the 126 JPEG files, then rerun `bash setup.sh`; the script detects that existing directory, reuses the pre-downloaded files, and skips downloading.

## Run The Reconstruction

Run full-sequence ViPE, train the Gaussian model, and render the camera-path video with one command:

```bash
bash scripts/run_reconstruction.sh
```

The default input is `zavod70/` and the default output is `output/zavod70/`. Pass another image directory and output directory explicitly:

```bash
bash scripts/run_reconstruction.sh /path/to/images --output-dir /path/to/output
```

Completed ViPE or Gaussian stages are reused only when their saved configuration matches the requested settings. An incomplete stage stops with an error instead of being overwritten. All reconstruction parameters are CLI options and are listed by:

```bash
bash scripts/run_reconstruction.sh --help
```

The no-argument command uses the validated settings. The equivalent explicit command is:

```bash
bash scripts/run_reconstruction.sh zavod70 \
	--output-dir output/zavod70 \
	--buffer 256 --image-max-edge 512 \
	--max-gaussians 200000 --iterations 4000 \
	--render-width 512 --video-fps 5 --refine-stop 1000
```

## Run Stages Manually

### ViPE

Process a complete image directory and export the native SLAM map:

```bash
uv run python -m vipe_pipeline.cli.run_vipe zavod70 \
	--save-slam-map \
	--output output/zavod70/vipe
```

Directory input defaults to a 512-pixel maximum edge and buffer 256. A 4:3 sequence is therefore processed at 512x384 to preserve aspect ratio while keeping ViPE and Gaussian training within the validated memory and runtime envelope. Image filenames are sorted lexicographically, so use zero-padded names.

The map, poses, intrinsics, and RGB must come from the same ViPE run so they share one coordinate frame.

### Gaussian Splatting

Train Gaussian splats directly from the ViPE SLAM map, camera poses, intrinsics, and RGB:

```bash
uv run python -m vipe_pipeline.cli.train_gaussians \
	output/zavod70/vipe \
	--artifact zavod70 \
	--output-dir output/zavod70/gaussians \
	--max-gaussians 200000 --iterations 4000 --refine-stop 1000
```

ViPE provides all scene geometry and calibrated cameras. `gsplat` is used only as the differentiable Gaussian rasterizer and standard PLY exporter, while ViPE camera estimation remains unchanged. The first training run compiles gsplat's CUDA extension and can take a few extra minutes.

`--max-gaussians` limits the ViPE map points used for initialization. During training, gsplat's AbsGS strategy duplicates or splits Gaussians with strong image-plane gradients and prunes low-opacity or oversized Gaussians. The final count can therefore exceed the seed count. The refinement schedule is configurable with `--refine-start`, `--refine-stop`, `--refine-every`, and `--grow-gradient`.

Outputs:

- `model.pt`: reusable Gaussian parameters and ViPE camera metadata
- `model.ply`: standard 3D Gaussian Splatting PLY for external viewers
- `trajectory.mp4`: H.264 render along the ViPE camera path
- `metrics.json`: held-out PSNR, training loss, scene dimensions, and peak CUDA allocation

Gaussian training and trajectory videos default to 512x384 for 4:3 input and 5 FPS. Higher-resolution input paths introduced unstable full-sequence convergence; stitching reliable overlapping windows was used as the recovery path, and the workflow was then standardized on direct 512x384 processing for reproducibility on the reference 8 GB GPU setup.

## Cross-Checks

Evaluate a ViPE trajectory against image EXIF GPS after one global similarity alignment:

```bash
uv run python -m vipe_pipeline.tools.evaluate_trajectory \
	output/zavod70/vipe/pose/zavod70.npz \
	zavod70 --frame-start 0 \
	--output /tmp/zavod70_gps.png \
	--metrics-output /tmp/zavod70_metrics.json
```

GPS evaluation is a quality check and does not modify ViPE cameras or geometry. Use one global GPS similarity only when metric scale or geographic orientation is required.

View any complete pose artifact with camera frustums:

```bash
uv run python -m vipe_pipeline.tools.view_trajectory \
	output/zavod70/vipe/pose/zavod70.npz \
	--port 20544 --frustum-step 5
```

## Window-Recovery Fallback

Stitching was introduced after a direct full-sequence run at max-edge 640 completed but had poor geometric accuracy against GPS, and overlapping-window runs then showed that late windows could be accurate while earlier/middle windows remained unstable. The fallback tooling was added to recover a full trajectory by selecting and stitching reliable segments when direct full-sequence convergence is unstable.

If a full-sequence ViPE run fails, record bounded diagnostic windows:

```bash
bash experiments/scripts/run_window_experiment.sh example_060_126 60 126 zavod70
```

The image-directory argument defaults to the `zavod70/` dataset prepared by `setup.sh`. Set `OUTPUT_ROOT` to change the default `output/` destination.

Select, stitch, and reconstruct overlapping windows with the separate fallback commands:

```bash
uv run python -m vipe_pipeline.fallback.select_windows zavod70 \
	--target-end 126 --output-dir /tmp/selection

uv run python -m vipe_pipeline.fallback.reconstruct_full_sequence \
	zavod70 /tmp/selection/selection.json \
	--output-dir /tmp/windowed_reconstruction
```

## Code Layout

- `vipe_pipeline/cli/`: primary ViPE and Gaussian commands
- `vipe_pipeline/core/`: shared geometry, data loading, rendering, and validation
- `vipe_pipeline/tools/`: optional GPS evaluation/fusion and visualization
- `vipe_pipeline/fallback/`: overlapping-window selection, stitching, and reconstruction recovery
- `scripts/run_reconstruction.sh`: primary end-to-end reconstruction command
- `experiments/scripts/run_window_experiment.sh`: guarded diagnostic window runner
- `experiments/training/assets/`: generated figures used by `TRAINING.md`
- `output/`: generated outputs from maintained workflows