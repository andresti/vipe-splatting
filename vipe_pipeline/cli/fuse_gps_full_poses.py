import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from vipe_pipeline.core.cli import positive_float, require_new_output
from vipe_pipeline.core.fusion import fuse_positions
from vipe_pipeline.core.trajectory import align_similarity, load_gps_positions, similarity_transform, trajectory_metrics


ROTATION_COMPOSE_EINSUM = "nij,njk->nik"
SOURCE_FRAME_LABEL = "Source frame"


def main() -> None:
	parser = argparse.ArgumentParser(description="Fuse GPS translations while preserving stitched camera orientations")
	parser.add_argument("poses", type=Path)
	parser.add_argument("image_dir", type=Path)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--motion-sigma", type=positive_float, default=0.5)
	parser.add_argument("--gps-horizontal-sigma", type=positive_float, default=3.0)
	parser.add_argument("--gps-vertical-sigma", type=positive_float, default=6.0)
	parser.add_argument("--gps-stride", type=int, default=5)
	args = parser.parse_args()
	require_new_output(parser, args.output_dir)
	if args.gps_stride < 1:
		parser.error("--gps-stride must be at least 1")

	pose_data = np.load(args.poses)
	poses = pose_data["data"].astype(float)
	frame_indices = pose_data["inds"].astype(int)
	if poses.shape != (len(frame_indices), 4, 4):
		parser.error("poses must contain data with shape N x 4 x 4 and matching inds")
	all_gps_positions = load_gps_positions(args.image_dir)
	if frame_indices.min() < 0 or frame_indices.max() >= len(all_gps_positions):
		parser.error("pose frame indices exceed the available GPS records")
	gps_positions = all_gps_positions[frame_indices]

	stitched_positions = poses[:, :3, 3]
	stitched_rotations = poses[:, :3, :3]
	metric_vipe_positions, initial_scale = align_similarity(stitched_positions, gps_positions)
	_, initial_row_rotation, _ = similarity_transform(stitched_positions, gps_positions)
	metric_vipe_rotations = np.einsum("ij,njk->nik", initial_row_rotation.T, stitched_rotations)

	anchor_mask = frame_indices % args.gps_stride == 0
	anchor_mask[-1] = True
	fused_positions = fuse_positions(
		metric_vipe_positions,
		gps_positions,
		anchor_mask,
		args.motion_sigma,
		args.gps_horizontal_sigma,
		args.gps_vertical_sigma,
	)
	step_corrections = np.linalg.norm(
		np.diff(fused_positions, axis=0) - np.diff(metric_vipe_positions, axis=0),
		axis=1,
	)

	overlap_only_poses = np.repeat(np.eye(4)[None], len(poses), axis=0)
	overlap_only_poses[:, :3, :3] = metric_vipe_rotations
	overlap_only_poses[:, :3, 3] = metric_vipe_positions
	fused_poses = overlap_only_poses.copy()
	fused_poses[:, :3, 3] = fused_positions

	orientation_changes = np.rad2deg(
		Rotation.from_matrix(
			np.einsum(
				ROTATION_COMPOSE_EINSUM,
				overlap_only_poses[:, :3, :3].transpose(0, 2, 1),
				fused_poses[:, :3, :3],
			)
		).magnitude()
	)
	determinants = np.linalg.det(fused_poses[:, :3, :3])
	orthogonality = np.linalg.norm(
		np.einsum(
			ROTATION_COMPOSE_EINSUM,
			fused_poses[:, :3, :3].transpose(0, 2, 1),
			fused_poses[:, :3, :3],
		) - np.eye(3),
		axis=(1, 2),
	)
	metrics = {
		"frame_count": len(frame_indices),
		"pose_convention": "camera_to_world",
		"gps_anchor_count": int(anchor_mask.sum()),
		"parameters": {
			"motion_sigma_m": args.motion_sigma,
			"gps_horizontal_sigma_m": args.gps_horizontal_sigma,
			"gps_vertical_sigma_m": args.gps_vertical_sigma,
			"gps_stride": args.gps_stride,
			"initial_similarity_scale": initial_scale,
		},
		"overlap_only": trajectory_metrics(metric_vipe_positions, gps_positions, frame_indices),
		"gps_assisted": trajectory_metrics(fused_positions, gps_positions, frame_indices),
		"gps_assisted_anchors": trajectory_metrics(
			fused_positions[anchor_mask],
			gps_positions[anchor_mask],
			frame_indices[anchor_mask],
		),
		"gps_assisted_unanchored": trajectory_metrics(
			fused_positions[~anchor_mask],
			gps_positions[~anchor_mask],
			frame_indices[~anchor_mask],
		),
		"motion_change": {
			"median_step_vector_change_m": float(np.median(step_corrections)),
			"max_step_vector_change_m": float(step_corrections.max()),
		},
		"orientation_preservation": {
			"maximum_change_deg": float(orientation_changes.max()),
			"rotation_determinant_min": float(determinants.min()),
			"rotation_determinant_max": float(determinants.max()),
			"rotation_orthogonality_max": float(orthogonality.max()),
		},
	}

	args.output_dir.mkdir(parents=True)
	np.savez_compressed(
		args.output_dir / "poses.npz",
		data=fused_poses,
		inds=frame_indices,
		overlap_only_data=overlap_only_poses,
		gps_positions=gps_positions,
		gps_anchor_mask=anchor_mask,
	)
	(args.output_dir / "metrics.json").write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")

	figure, axes = plt.subplots(2, 2, figsize=(13, 10))
	axes[0, 0].plot(gps_positions[:, 0], gps_positions[:, 1], label="GPS")
	axes[0, 0].plot(metric_vipe_positions[:, 0], metric_vipe_positions[:, 1], label="Overlap-only")
	axes[0, 0].plot(fused_positions[:, 0], fused_positions[:, 1], label="GPS-assisted full pose")
	axes[0, 0].scatter(gps_positions[anchor_mask, 0], gps_positions[anchor_mask, 1], s=12, label="GPS anchors")
	axes[0, 0].set(xlabel="East (m)", ylabel="North (m)", title="Horizontal trajectory")
	axes[0, 0].set_aspect("equal")
	axes[0, 0].grid(alpha=0.3)
	axes[0, 0].legend()
	axes[0, 1].plot(frame_indices, gps_positions[:, 2], label="GPS altitude")
	axes[0, 1].plot(frame_indices, metric_vipe_positions[:, 2], label="Overlap-only")
	axes[0, 1].plot(frame_indices, fused_positions[:, 2], label="GPS-assisted")
	axes[0, 1].set(xlabel=SOURCE_FRAME_LABEL, ylabel="Height (m)", title="Vertical trajectory")
	axes[0, 1].grid(alpha=0.3)
	axes[0, 1].legend()
	axes[1, 0].plot(frame_indices[1:], step_corrections)
	axes[1, 0].set(xlabel=SOURCE_FRAME_LABEL, ylabel="Step-vector change (m)", title="Correction to ViPE motion")
	axes[1, 0].grid(alpha=0.3)
	angular_steps = np.rad2deg(
		Rotation.from_matrix(
			np.einsum(
				ROTATION_COMPOSE_EINSUM,
				metric_vipe_rotations[:-1].transpose(0, 2, 1),
				metric_vipe_rotations[1:],
			)
		).magnitude()
	)
	axes[1, 1].plot(frame_indices[1:], angular_steps)
	axes[1, 1].set(xlabel=SOURCE_FRAME_LABEL, ylabel="Rotation step (degrees)", title="Preserved orientation changes")
	axes[1, 1].grid(alpha=0.3)
	figure.tight_layout()
	figure.savefig(args.output_dir / "comparison.png", dpi=160)
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()