# Gaussian Splatting Training Report

This document describes how the ViPE reconstruction is converted into a trained Gaussian scene, summarizes the main alternatives tested, and explains the selected configuration.

## Selected Configuration

The final model uses one coherent full-sequence ViPE run and the following Gaussian settings:

| Setting | Value |
| --- | ---: |
| Input frames | 126 |
| Training / holdout views | 110 / 16 |
| Render resolution | 512x384 |
| Initial ViPE points | 200,000 |
| Training iterations | 4,000 |
| Refinement interval | Steps 500-1,000, every 100 steps |
| AbsGS growth threshold | 0.0015 |
| Final Gaussians | 314,319 |
| Final holdout PSNR | 12.475 dB |
| Final 100-step mean loss | 0.26343 |
| Peak Torch CUDA allocation | 170.29 MiB |

The end-to-end command is:

```bash
bash scripts/run_reconstruction.sh zavod70 \
	--output-dir output/zavod70 \
	--buffer 256 --image-max-edge 512 \
	--max-gaussians 200000 --iterations 4000 \
	--render-width 512 --video-fps 5 --refine-stop 1000
```

## Training Process

### 1. Load One Coherent ViPE Reconstruction

ViPE exports four artifacts from the same run:

- Camera-to-world poses, which locate and orient each camera.
- Camera intrinsics $(f_x, f_y, c_x, c_y)$, which define the 3D-to-pixel projection.
- RGB frames used as optimization targets.
- A colored native SLAM point map used to initialize the Gaussian geometry.

Keeping these artifacts together is essential because they share one coordinate frame. A camera trajectory from one run cannot safely be combined with an unchanged map from another run.

### 2. Prepare Cameras and Images

The RGB frames are area-resampled to 512x384. Focal lengths and principal-point coordinates are scaled by the same width and height ratios, preserving the camera projection at the training resolution.

The selected direct ViPE run processed all 126 frames at 512x384, retained 123 keyframes, and exported approximately 320,000 finite colored map points. A deterministic seed selects at most 200,000 points for Gaussian initialization.

### 3. Initialize Gaussians

Each sampled ViPE point initializes one Gaussian:

- Position comes directly from the SLAM map.
- Color comes from the ViPE point color.
- Initial scale is based on nearest-neighbor point spacing.
- Rotation starts as the identity quaternion.
- Opacity starts low and is optimized during training.

This preserves ViPE as the source of cameras and scene geometry. `gsplat` supplies differentiable rasterization, adaptive Gaussian operations, and PLY export.

### 4. Split Training and Evaluation Views

Every eighth frame is held out: indices `0, 8, 16, ..., 120`. The remaining 110 frames train the Gaussian model, while 16 interleaved frames measure novel-view interpolation throughout the trajectory.

ViPE itself processed all 126 images when estimating cameras and geometry. The reported number is therefore **held-out Gaussian-rendering PSNR**, not a fully independent end-to-end test score.

### 5. Optimize Appearance and Geometry

It helps to think of each Gaussian as a small, soft, colored 3D blob. Thousands of these blobs overlap to form the scene. Training repeatedly asks: **how should the blobs change so that a rendered view looks more like the real photograph?**

One optimization step works as follows:

1. Select one of the 110 training photographs and its ViPE camera.
2. Render the current Gaussian scene from that camera position.
3. Compare every rendered pixel with the real photograph.
4. Measure how different the two images are with a loss value. Lower is better.
5. Backpropagation calculates how each Gaussian contributed to that error.
6. Adam makes a small adjustment to the Gaussian parameters.

The adjustable parameters have direct visual meanings:

- **Position:** where the Gaussian is in 3D space.
- **Scale and rotation:** the size, shape, and orientation of the blob.
- **Color:** what color it contributes to the rendered image.
- **Opacity:** how solid or transparent it is.

For example, if a roof edge appears in the wrong image location, nearby Gaussians may move or change shape. If an area is too dark, their colors or opacities may change. The next iteration selects another camera, renders again, and repeats the same process. Seeing the scene from many viewpoints prevents the model from matching only one photograph.

The comparison combines direct pixel error with a structural image-similarity term:

$$
\mathcal{L} = 0.8\,\mathcal{L}_{1} + 0.2\,(1-\operatorname{SSIM})
$$

$\mathcal{L}_{1}$ measures average pixel differences, while SSIM rewards similar local structure such as edges and contrast. Separate Adam optimizers update positions, scales, rotations, opacities, and colors using the gradients from this loss.

The number of Gaussians can also change. Between steps 500 and 1,000, the AbsGS strategy:

- Duplicates or splits Gaussians where more detail is needed.
- Removes Gaussians that remain nearly transparent or become excessively large.

After step 1,000, no more Gaussians are added or removed. The remaining 3,000 steps refine the fixed set, allowing its positions, shapes, colors, and opacities to settle. The 16 holdout photographs are never selected for these updates; they are used only for evaluation.

### 6. Evaluate and Export

PSNR compares renders with the original held-out RGB frames:

$$
\operatorname{PSNR} = -10\log_{10}(\operatorname{MSE})
$$

The 6.01 dB initialization score is measured immediately after converting the ViPE points into Gaussians, before any optimization step. The selected model reaches 12.47 dB after 4,000 steps.

Training exports a reusable Torch checkpoint, a standard Gaussian PLY, `metrics.json`, and an H.264 trajectory render.

## Main Alternatives Tested

### Window Stitching vs. One Full ViPE Run

Overlapping windows were initially needed because some full runs entered poor local minima. Stitching recovered all 126 cameras, but each window carried its own map and coordinate frame. Sim(3) alignment introduced additional geometry uncertainty and reached 12.80 dB under the comparison render settings.

Repeated direct 512x384 ViPE runs later produced stable complete trajectories. A single run avoids map stitching, preserves camera-to-geometry consistency, and reached 13.72 dB under the same comparison render settings. The direct run became the primary path; window stitching remains a fallback.

This is a like-for-like recovery-path comparison: both models use all 126 frames, the same 110/16 view split, 320x240 rendering, 200,000 initial points, 4,000 iterations, refinement through step 2,000, and growth threshold 0.0015. The difference is whether the cameras and geometry come from stitched windows or one direct ViPE run.

### GPS-Assisted Pose Fusion Diagnostics

GPS-assisted non-rigid pose fusion is retained as a diagnostic tool for drifting trajectories, not as a primary reconstruction path.

Run:

```bash
uv run python -m vipe_pipeline.tools.fuse_gps_full_poses \
	/path/to/stitched/poses.npz \
	zavod70 --output-dir /tmp/gps_fusion
```

The diagnostic output includes:

- `poses.npz`: fused camera poses plus overlap-only baseline poses.
- `metrics.json`: overlap-only vs GPS-assisted trajectory metrics.
- `comparison.png`: horizontal path overlay, altitude traces, and per-step correction magnitude.

![GPS-assisted pose fusion diagnostic comparison](output/fused/gps_assisted_conservative_v1/comparison.png)

The plot overlays GPS, overlap-only, and GPS-assisted trajectories and shows how much per-step motion correction was applied.

This tool is for analysis and visualization only. 

### Render Resolution

Increasing wrapper input resolution did not increase ViPE geometry density because DROID still operates on a 64x48 disparity grid for this 4:3 sequence. Direct 512 input was also more reliable than the tested 1024 path, while native 4000x3000 processing exhausted available host resources.

Gaussian render resolution was tested separately using full-sequence models. Every model uses all 126 frames, the same 110/16 split, 200,000 initial points, 4,000 iterations, refinement through step 2,000, and growth threshold 0.0015.

Higher-resolution PSNR is expected to be lower because fine pixel errors are no longer hidden by downsampling. In these tests, increasing output resolution raised Gaussian count and memory use without fixing the underlying geometry errors. The final 512x384 choice was kept as the practical deployment setting for detail vs. resource use.

Two full-sequence 512x384 comparisons also tested bounded gray-world exposure and white-balance normalization with the original and selected refinement schedules. It reduced frame-to-frame color-statistic variation, but changed original-RGB holdout PSNR by less than 0.01 dB in either comparison and produced no compelling visual improvement. The normalization code was removed rather than carrying an ineffective option.

## 512x384 Tuning Decision

The controlled final sweep kept the same direct ViPE map, camera split, 200,000 initial points, 4,000 iterations, and random seed.

Stopping refinement at step 1,000 was the useful change. Relative to refinement through step 2,000, it:

- Improved holdout PSNR from 12.335 to 12.475 dB.
- Reduced the model from 453,871 to 314,319 Gaussians.
- Reduced peak Torch CUDA allocation from 212.95 to 170.29 MiB.
- Gave the final topology 3,000 iterations to settle.

Raising the growth threshold to 0.002 reduced the model further to 277,642 Gaussians with nearly identical 12.472 dB PSNR. The 0.0015 threshold was retained as the quality-oriented choice because it achieved the highest measured score while remaining substantially smaller than the original model.

## Interpretation

The final tuning is an efficiency improvement with a modest quality gain, not a dramatic reconstruction breakthrough. Visible blur, floaters, and duplicated structures are primarily limited by camera accuracy, sparse ViPE geometry, occlusion, and scene coverage. Improving those inputs seems more promising than simply adding Gaussians or iterations.