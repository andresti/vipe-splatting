import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as functional
from gsplat import rasterization
from gsplat.exporter import export_splats
from scipy.spatial import cKDTree
from vipe.slam.interface import SLAMMap
from vipe.utils.io import read_rgb_artifacts

from vipe_pipeline.core.windows import WindowSpec


SH_C0 = 0.28209479177387814


@dataclass
class VipeGaussianDataset:
	images: torch.Tensor
	camera_to_world: torch.Tensor
	intrinsics: torch.Tensor
	frame_indices: torch.Tensor
	points: torch.Tensor
	point_colors: torch.Tensor
	height: int
	width: int


def _load_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
	if not path.exists():
		raise ValueError(f"ViPE artifact does not exist: {path}")
	artifact = np.load(path)
	return artifact["data"], artifact["inds"]


def _prepare_camera_artifacts(
	rgb_frames: list[torch.Tensor],
	intrinsic_data: np.ndarray,
	render_width: int,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
	source_height, source_width = rgb_frames[0].shape[:2]
	render_height = round(source_height * render_width / source_width)
	images = torch.stack(rgb_frames).permute(0, 3, 1, 2)
	images = functional.interpolate(images, size=(render_height, render_width), mode="area")
	images = (images.permute(0, 2, 3, 1).clamp(0, 1) * 255).round().to(torch.uint8)

	intrinsics = torch.zeros((len(intrinsic_data), 3, 3), dtype=torch.float32)
	intrinsics[:, 0, 0] = torch.from_numpy(intrinsic_data[:, 0]) * render_width / source_width
	intrinsics[:, 1, 1] = torch.from_numpy(intrinsic_data[:, 1]) * render_height / source_height
	intrinsics[:, 0, 2] = torch.from_numpy(intrinsic_data[:, 2]) * render_width / source_width
	intrinsics[:, 1, 2] = torch.from_numpy(intrinsic_data[:, 3]) * render_height / source_height
	intrinsics[:, 2, 2] = 1
	return images, intrinsics, render_height, render_width


def _load_slam_points(artifact_dir: Path, artifact_name: str) -> tuple[torch.Tensor, torch.Tensor]:
	map_path = artifact_dir / "vipe" / f"{artifact_name}_slam_map.pt"
	if not map_path.exists():
		raise ValueError(f"ViPE SLAM map does not exist: {map_path}; rerun ViPE with --save-slam-map")
	slam_map = SLAMMap.load(map_path, device=torch.device("cpu"))
	points, point_colors = slam_map.get_dense_disp_full_pcd()
	valid = torch.isfinite(points).all(dim=1) & torch.isfinite(point_colors).all(dim=1)
	points = points[valid].float()
	point_colors = point_colors[valid].float().clamp(0, 1)
	if not len(points):
		raise ValueError(f"ViPE SLAM map contains no finite points: {map_path}")
	return points, point_colors


def _sample_points(
	points: torch.Tensor,
	point_colors: torch.Tensor,
	max_gaussians: int,
	seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
	if len(points) > max_gaussians:
		generator = torch.Generator().manual_seed(seed)
		selection = torch.randperm(len(points), generator=generator)[:max_gaussians]
		points = points[selection]
		point_colors = point_colors[selection]
	return points, point_colors


def load_vipe_gaussian_dataset(
	artifact_dir: Path,
	artifact_name: str,
	render_width: int,
	max_gaussians: int,
	seed: int,
) -> VipeGaussianDataset:
	pose_data, pose_indices = _load_npz(artifact_dir / "pose" / f"{artifact_name}.npz")
	intrinsic_data, intrinsic_indices = _load_npz(artifact_dir / "intrinsics" / f"{artifact_name}.npz")
	if pose_data.shape != (len(pose_indices), 4, 4):
		raise ValueError("ViPE poses must have shape N x 4 x 4")
	if intrinsic_data.shape != (len(intrinsic_indices), 4):
		raise ValueError("ViPE intrinsics must have shape N x 4")
	if not np.array_equal(pose_indices, intrinsic_indices):
		raise ValueError("ViPE pose and intrinsic frame indices do not match")

	rgb_path = artifact_dir / "rgb" / f"{artifact_name}.mp4"
	if not rgb_path.exists():
		raise ValueError(f"ViPE RGB artifact does not exist: {rgb_path}")
	rgb_frames = [rgb for _, rgb in read_rgb_artifacts(rgb_path)]
	if len(rgb_frames) != len(pose_data):
		raise ValueError(f"ViPE RGB frame count {len(rgb_frames)} does not match pose count {len(pose_data)}")
	images, intrinsics, render_height, render_width = _prepare_camera_artifacts(
		rgb_frames,
		intrinsic_data,
		render_width,
	)
	points, point_colors = _load_slam_points(artifact_dir, artifact_name)
	points, point_colors = _sample_points(points, point_colors, max_gaussians, seed)

	return VipeGaussianDataset(
		images=images,
		camera_to_world=torch.from_numpy(pose_data).float(),
		intrinsics=intrinsics,
		frame_indices=torch.from_numpy(pose_indices).long(),
		points=points,
		point_colors=point_colors,
		height=render_height,
		width=render_width,
	)


def _load_stitched_window(
	runs_dir: Path,
	artifact_name: str,
	window: WindowSpec,
	transform: dict,
	render_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
	name, start, end = window
	if transform["frame_start"] != start or transform["frame_end"] != end:
		raise ValueError(f"stitched transform frame range does not match window: {name}")
	artifact_dir = runs_dir / name / "results"
	intrinsic_data, _ = _load_npz(artifact_dir / "intrinsics" / f"{artifact_name}.npz")
	if intrinsic_data.shape != (end - start, 4):
		raise ValueError(f"{name} intrinsics have shape {intrinsic_data.shape}, expected {(end - start, 4)}")
	rgb_path = artifact_dir / "rgb" / f"{artifact_name}.mp4"
	if not rgb_path.exists():
		raise ValueError(f"ViPE RGB artifact does not exist: {rgb_path}")
	rgb_frames = [rgb for _, rgb in read_rgb_artifacts(rgb_path)]
	if len(rgb_frames) != end - start:
		raise ValueError(f"{name} RGB frame count {len(rgb_frames)} does not match window length {end - start}")
	images, intrinsics, render_height, _ = _prepare_camera_artifacts(
		rgb_frames,
		intrinsic_data,
		render_width,
	)
	points, point_colors = _load_slam_points(artifact_dir, artifact_name)
	scale = float(transform["scale"])
	row_rotation = torch.tensor(transform["row_rotation"], dtype=torch.float32)
	translation = torch.tensor(transform["translation"], dtype=torch.float32)
	return images, intrinsics, scale * points @ row_rotation + translation, point_colors, render_height


def load_stitched_vipe_gaussian_dataset(
	stitch_dir: Path,
	runs_dir: Path,
	artifact_name: str,
	windows: list[WindowSpec],
	render_width: int,
	max_gaussians: int,
	seed: int,
) -> VipeGaussianDataset:
	pose_data, pose_indices = _load_npz(stitch_dir / "poses.npz")
	if pose_data.shape != (len(pose_indices), 4, 4):
		raise ValueError("stitched ViPE poses must have shape N x 4 x 4")
	transform_path = stitch_dir / "window_transforms.json"
	if not transform_path.exists():
		raise ValueError(f"window transform artifact does not exist: {transform_path}")
	transform_records = json.loads(transform_path.read_text(encoding="utf-8"))
	transforms = {record["run_name"]: record for record in transform_records}

	images_by_frame: dict[int, torch.Tensor] = {}
	intrinsics_by_frame: dict[int, torch.Tensor] = {}
	point_chunks = []
	color_chunks = []
	render_height = 0
	for name, start, end in windows:
		if name not in transforms:
			raise ValueError(f"stitched transform is missing window: {name}")
		window_images, window_intrinsics, window_points, window_colors, window_height = _load_stitched_window(
			runs_dir,
			artifact_name,
			(name, start, end),
			transforms[name],
			render_width,
		)
		if render_height not in (0, window_height):
			raise ValueError("stitched windows do not share one RGB aspect ratio")
		render_height = window_height
		for local_index, frame in enumerate(range(start, end)):
			images_by_frame.setdefault(frame, window_images[local_index])
			intrinsics_by_frame.setdefault(frame, window_intrinsics[local_index])

		point_chunks.append(window_points)
		color_chunks.append(window_colors)

	frame_indices = [int(frame) for frame in pose_indices]
	missing_frames = [frame for frame in frame_indices if frame not in images_by_frame]
	if missing_frames:
		raise ValueError(f"stitched windows do not provide RGB and intrinsics for frames: {missing_frames}")
	points = torch.cat(point_chunks)
	point_colors = torch.cat(color_chunks)
	points, point_colors = _sample_points(points, point_colors, max_gaussians, seed)
	return VipeGaussianDataset(
		images=torch.stack([images_by_frame[frame] for frame in frame_indices]),
		camera_to_world=torch.from_numpy(pose_data).float(),
		intrinsics=torch.stack([intrinsics_by_frame[frame] for frame in frame_indices]),
		frame_indices=torch.tensor(frame_indices, dtype=torch.long),
		points=points,
		point_colors=point_colors,
		height=render_height,
		width=render_width,
	)


def initialize_gaussians(
	dataset: VipeGaussianDataset,
	device: torch.device,
	initial_scale: float,
) -> torch.nn.ParameterDict:
	points = dataset.points.numpy()
	nearest_distances = cKDTree(points).query(points, k=2, workers=-1)[0][:, 1]
	positive_distances = nearest_distances[np.isfinite(nearest_distances) & (nearest_distances > 0)]
	if not len(positive_distances):
		raise ValueError("ViPE map points do not have usable nearest-neighbour distances")
	lower, upper = np.quantile(positive_distances, (0.05, 0.95))
	nearest_distances = np.clip(nearest_distances, lower, upper)
	log_scales = torch.from_numpy(np.log(nearest_distances * initial_scale)).float().unsqueeze(1).repeat(1, 3)
	quaternions = torch.zeros((len(points), 4), dtype=torch.float32)
	quaternions[:, 0] = 1
	opacity_logits = torch.full((len(points),), -2.1972246, dtype=torch.float32)
	color_logits = torch.logit(dataset.point_colors.clamp(1e-4, 1 - 1e-4))
	return torch.nn.ParameterDict(
		{
			"means": torch.nn.Parameter(dataset.points.to(device)),
			"scales": torch.nn.Parameter(log_scales.to(device)),
			"quats": torch.nn.Parameter(quaternions.to(device)),
			"opacities": torch.nn.Parameter(opacity_logits.to(device)),
			"colors": torch.nn.Parameter(color_logits.to(device)),
		}
	)


def render_gaussians(
	gaussians: torch.nn.ParameterDict,
	camera_to_world: torch.Tensor,
	intrinsics: torch.Tensor,
	width: int,
	height: int,
	absgrad: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
	view_matrix = torch.linalg.inv(camera_to_world).unsqueeze(0)
	rendered, alpha, info = rasterization(
		means=gaussians["means"],
		quats=functional.normalize(gaussians["quats"], dim=-1),
		scales=torch.exp(gaussians["scales"]),
		opacities=torch.sigmoid(gaussians["opacities"]),
		colors=torch.sigmoid(gaussians["colors"]),
		viewmats=view_matrix,
		Ks=intrinsics.unsqueeze(0),
		width=width,
		height=height,
		packed=True,
		absgrad=absgrad,
		backgrounds=torch.ones(3, device=camera_to_world.device),
	)
	return rendered[0], alpha[0], info


def photometric_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
	l1_loss = functional.l1_loss(rendered, target)
	first = rendered.permute(2, 0, 1).unsqueeze(0)
	second = target.permute(2, 0, 1).unsqueeze(0)
	first_mean = functional.avg_pool2d(first, 11, stride=1, padding=5)
	second_mean = functional.avg_pool2d(second, 11, stride=1, padding=5)
	first_variance = functional.avg_pool2d(first * first, 11, stride=1, padding=5) - first_mean.square()
	second_variance = functional.avg_pool2d(second * second, 11, stride=1, padding=5) - second_mean.square()
	covariance = functional.avg_pool2d(first * second, 11, stride=1, padding=5) - first_mean * second_mean
	ssim = ((2 * first_mean * second_mean + 0.01**2) * (2 * covariance + 0.03**2)) / (
		(first_mean.square() + second_mean.square() + 0.01**2)
		* (first_variance + second_variance + 0.03**2)
	)
	return 0.8 * l1_loss + 0.2 * (1 - ssim.mean())


def save_gaussian_checkpoint(
	path: Path,
	gaussians: torch.nn.ParameterDict,
	dataset: VipeGaussianDataset,
) -> None:
	torch.save(
		{
			"gaussians": {name: value.detach().cpu() for name, value in gaussians.items()},
			"camera_to_world": dataset.camera_to_world,
			"intrinsics": dataset.intrinsics,
			"frame_indices": dataset.frame_indices,
			"height": dataset.height,
			"width": dataset.width,
		},
		path,
	)


def export_gaussian_ply(path: Path, gaussians: torch.nn.ParameterDict) -> None:
	colors = torch.sigmoid(gaussians["colors"].detach())
	sh0 = ((colors - 0.5) / SH_C0).unsqueeze(1)
	sh_rest = torch.empty((len(colors), 0, 3), device=colors.device)
	export_splats(
		means=gaussians["means"].detach(),
		scales=gaussians["scales"].detach(),
		quats=functional.normalize(gaussians["quats"].detach(), dim=-1),
		opacities=gaussians["opacities"].detach(),
		sh0=sh0,
		shN=sh_rest,
		format="ply",
		save_to=str(path),
	)


@torch.no_grad()
def render_camera_path_video(
	path: Path,
	gaussians: torch.nn.ParameterDict,
	dataset: VipeGaussianDataset,
	device: torch.device,
	fps: int,
) -> None:
	with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
		for camera_to_world, intrinsics in zip(dataset.camera_to_world, dataset.intrinsics):
			rendered, _, _ = render_gaussians(
				gaussians,
				camera_to_world.to(device),
				intrinsics.to(device),
				dataset.width,
				dataset.height,
			)
			writer.append_data((rendered.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))