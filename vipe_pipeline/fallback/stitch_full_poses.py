import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from vipe_pipeline.core.trajectory import load_gps_positions, similarity_transform, trajectory_metrics
from vipe_pipeline.core.windows import load_window_poses, parse_window


ROTATION_COMPOSE_EINSUM = "nij,njk->nik"


def mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
	return Rotation.from_matrix(np.asarray(rotations)).mean().as_matrix()


def current_positions(samples: dict[int, list[np.ndarray]], frames: list[int]) -> np.ndarray:
	return np.asarray([np.mean(samples[frame], axis=0) for frame in frames])


def current_rotations(samples: dict[int, list[np.ndarray]], frames: list[int]) -> np.ndarray:
	return np.asarray([mean_rotation(samples[frame]) for frame in frames])


def main() -> None:
	parser = argparse.ArgumentParser(description="Stitch complete ViPE camera-to-world poses")
	parser.add_argument("image_dir", type=Path)
	parser.add_argument("--window", action="append", type=parse_window, required=True)
	parser.add_argument("--runs-dir", type=Path, default=Path("output/windows"))
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--skip-gps-evaluation", action="store_true")
	args = parser.parse_args()
	try:
		stitch_full_poses(
			image_dir=args.image_dir,
			windows=args.window,
			runs_dir=args.runs_dir,
			output_dir=args.output_dir,
			skip_gps_evaluation=args.skip_gps_evaluation,
		)
	except ValueError as error:
		parser.error(str(error))


def stitch_full_poses(
	image_dir: Path,
	windows: list[tuple[str, int, int]],
	runs_dir: Path = Path("output/windows"),
	output_dir: Path = Path("."),
	skip_gps_evaluation: bool = False,
) -> None:
	if output_dir.exists():
		raise ValueError(f"refusing to overwrite existing output: {output_dir}")
	first_poses = load_window_poses(runs_dir, image_dir.name, windows[0])
	_, first_start, first_end = windows[0]
	position_samples = {
		frame: [pose[:3, 3]]
		for frame, pose in zip(range(first_start, first_end), first_poses)
	}
	rotation_samples = {
		frame: [pose[:3, :3]]
		for frame, pose in zip(range(first_start, first_end), first_poses)
	}
	window_transforms = [
		{
			"run_name": windows[0][0],
			"frame_start": first_start,
			"frame_end": first_end,
			"scale": 1.0,
			"row_rotation": np.eye(3).tolist(),
			"translation": np.zeros(3).tolist(),
		}
	]
	overlap_metrics = []

	for window in windows[1:]:
		name, start, end = window
		poses = load_window_poses(runs_dir, image_dir.name, window)
		overlap = sorted(set(position_samples).intersection(range(start, end)))
		if len(overlap) < 3:
			raise ValueError(f"{name} has only {len(overlap)} overlapping frames; at least 3 are required")

		source_positions = np.asarray([poses[frame - start, :3, 3] for frame in overlap])
		target_positions = current_positions(position_samples, overlap)
		scale, row_rotation, translation = similarity_transform(source_positions, target_positions)
		transformed_positions = scale * poses[:, :3, 3] @ row_rotation + translation
		world_rotation = row_rotation.T
		transformed_rotations = np.einsum("ij,njk->nik", world_rotation, poses[:, :3, :3])
		window_transforms.append(
			{
				"run_name": name,
				"frame_start": start,
				"frame_end": end,
				"scale": scale,
				"row_rotation": row_rotation.tolist(),
				"translation": translation.tolist(),
			}
		)

		aligned_overlap_positions = transformed_positions[[frame - start for frame in overlap]]
		position_residuals = np.linalg.norm(aligned_overlap_positions - target_positions, axis=1)
		target_rotations = current_rotations(rotation_samples, overlap)
		aligned_overlap_rotations = transformed_rotations[[frame - start for frame in overlap]]
		relative_rotations = np.einsum(
			ROTATION_COMPOSE_EINSUM,
			target_rotations.transpose(0, 2, 1),
			aligned_overlap_rotations,
		)
		angular_residuals = np.rad2deg(Rotation.from_matrix(relative_rotations).magnitude())
		overlap_metrics.append(
			{
				"run_name": name,
				"frame_start": overlap[0],
				"frame_end": overlap[-1] + 1,
				"frame_count": len(overlap),
				"relative_scale": scale,
				"position_rmse_stitched_units": float(np.sqrt(np.mean(np.square(position_residuals)))),
				"orientation_rmse_deg": float(np.sqrt(np.mean(np.square(angular_residuals)))),
				"orientation_median_deg": float(np.median(angular_residuals)),
				"orientation_max_deg": float(angular_residuals.max()),
			}
		)
		for frame, position, rotation in zip(range(start, end), transformed_positions, transformed_rotations):
			position_samples.setdefault(frame, []).append(position)
			rotation_samples.setdefault(frame, []).append(rotation)

	frame_indices = np.asarray(sorted(position_samples))
	if not np.array_equal(frame_indices, np.arange(frame_indices[-1] + 1)):
		raise ValueError("stitched windows do not provide contiguous coverage beginning at frame 0")
	stitched_positions = current_positions(position_samples, frame_indices.tolist())
	stitched_rotations = current_rotations(rotation_samples, frame_indices.tolist())
	stitched_poses = np.repeat(np.eye(4)[None], len(frame_indices), axis=0)
	stitched_poses[:, :3, :3] = stitched_rotations
	stitched_poses[:, :3, 3] = stitched_positions

	orthogonality = np.linalg.norm(
		np.einsum(ROTATION_COMPOSE_EINSUM, stitched_rotations.transpose(0, 2, 1), stitched_rotations) - np.eye(3),
		axis=(1, 2),
	)
	determinants = np.linalg.det(stitched_rotations)
	angular_steps = np.rad2deg(
		Rotation.from_matrix(
			np.einsum(
				ROTATION_COMPOSE_EINSUM,
				stitched_rotations[:-1].transpose(0, 2, 1),
				stitched_rotations[1:],
			)
		).magnitude()
	)
	metrics = {
		"frame_count": len(frame_indices),
		"pose_convention": "camera_to_world",
		"rotation_determinant_min": float(determinants.min()),
		"rotation_determinant_max": float(determinants.max()),
		"rotation_orthogonality_max": float(orthogonality.max()),
		"angular_step_median_deg": float(np.median(angular_steps)),
		"angular_step_p95_deg": float(np.percentile(angular_steps, 95)),
		"angular_step_max_deg": float(angular_steps.max()),
		"overlaps": overlap_metrics,
	}
	pose_artifact = {"data": stitched_poses, "inds": frame_indices}
	if not skip_gps_evaluation:
		gps_positions = load_gps_positions(image_dir)[frame_indices]
		final_scale, final_row_rotation, final_translation = similarity_transform(stitched_positions, gps_positions)
		aligned_positions = final_scale * stitched_positions @ final_row_rotation + final_translation
		aligned_rotations = np.einsum("ij,njk->nik", final_row_rotation.T, stitched_rotations)
		aligned_poses = stitched_poses.copy()
		aligned_poses[:, :3, :3] = aligned_rotations
		aligned_poses[:, :3, 3] = aligned_positions
		for overlap in overlap_metrics:
			overlap["position_rmse_m"] = overlap.pop("position_rmse_stitched_units") * final_scale
		metrics.update({"scale": final_scale, **trajectory_metrics(aligned_positions, gps_positions, frame_indices)})
		pose_artifact["gps_aligned_data"] = aligned_poses

	output_dir.mkdir(parents=True)
	np.savez_compressed(output_dir / "poses.npz", **pose_artifact)
	(output_dir / "metrics.json").write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")
	(output_dir / "window_transforms.json").write_text(
		f"{json.dumps(window_transforms, indent=2)}\n",
		encoding="utf-8",
	)

	figure, axes = plt.subplots(1, 2, figsize=(12, 5))
	if skip_gps_evaluation:
		axes[0].plot(stitched_positions[:, 0], stitched_positions[:, 1], label="Full-pose stitch")
	else:
		axes[0].plot(gps_positions[:, 0], gps_positions[:, 1], label="GPS")
		axes[0].plot(aligned_positions[:, 0], aligned_positions[:, 1], label="Full-pose stitch")
	axes[0].set(xlabel="East (m)", ylabel="North (m)", title="Stitched camera centers")
	axes[0].set_aspect("equal")
	axes[0].grid(alpha=0.3)
	axes[0].legend()
	axes[1].plot(frame_indices[1:], angular_steps)
	axes[1].set(xlabel="Source frame", ylabel="Rotation step (degrees)", title="Frame-to-frame orientation change")
	axes[1].grid(alpha=0.3)
	figure.tight_layout()
	figure.savefig(output_dir / "comparison.png", dpi=160)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()