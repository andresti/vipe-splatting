import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from vipe_pipeline.cli.run_vipe import run_vipe
from vipe_pipeline.cli.train_gaussians import TrainGaussiansConfig, train_gaussians
from vipe_pipeline.core.cli import positive_float
from vipe_pipeline.core.trajectory import similarity_transform
from vipe_pipeline.core.windows import WindowSpec
from vipe_pipeline.fallback.stitch_full_poses import stitch_full_poses


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("value must be greater than zero")
	return parsed


def load_selection(path: Path, image_dir: Path) -> tuple[str, list[WindowSpec]]:
	if not path.exists():
		raise ValueError(f"selection artifact does not exist: {path}")
	selection = json.loads(path.read_text(encoding="utf-8"))
	dataset_name = selection.get("dataset_name")
	if dataset_name != image_dir.name:
		raise ValueError(
			f"selection dataset {dataset_name!r} does not match image directory name {image_dir.name!r}"
		)
	selected = selection.get("selected_windows")
	if not isinstance(selected, list) or not selected:
		raise ValueError("selection artifact does not contain selected windows")
	windows = [
		(record["run_name"], int(record["frame_start"]), int(record["frame_end"]))
		for record in selected
	]
	return dataset_name, windows


def expected_window_artifacts(results_dir: Path, dataset_name: str) -> list[Path]:
	return [
		results_dir / "pose" / f"{dataset_name}.npz",
		results_dir / "intrinsics" / f"{dataset_name}.npz",
		results_dir / "rgb" / f"{dataset_name}.mp4",
		results_dir / "vipe" / f"{dataset_name}_slam_map.pt",
	]


def pose_consistency(
	source_path: Path,
	candidate_path: Path,
) -> tuple[float, float]:
	source = np.load(source_path)["data"].astype(float)
	candidate = np.load(candidate_path)["data"].astype(float)
	if source.shape != candidate.shape or source.ndim != 3 or source.shape[1:] != (4, 4):
		return float("inf"), float("inf")
	scale, row_rotation, translation = similarity_transform(candidate[:, :3, 3], source[:, :3, 3])
	aligned_positions = scale * candidate[:, :3, 3] @ row_rotation + translation
	position_rmse = np.sqrt(np.mean(np.sum(np.square(aligned_positions - source[:, :3, 3]), axis=1)))
	extent = np.linalg.norm(np.ptp(source[:, :3, 3], axis=0))
	normalized_position_rmse = float(position_rmse / max(extent, np.finfo(float).eps))
	aligned_rotations = np.einsum("ij,njk->nik", row_rotation.T, candidate[:, :3, :3])
	relative_rotations = np.einsum(
		"nij,njk->nik",
		source[:, :3, :3].transpose(0, 2, 1),
		aligned_rotations,
	)
	angles = np.rad2deg(Rotation.from_matrix(relative_rotations).magnitude())
	return normalized_position_rmse, float(np.sqrt(np.mean(np.square(angles))))


def window_is_consistent(
	source_pose: Path,
	results_dir: Path,
	dataset_name: str,
	maximum_position_rmse: float,
	maximum_rotation_rmse: float,
) -> bool:
	artifacts = expected_window_artifacts(results_dir, dataset_name)
	if not all(path.exists() for path in artifacts):
		return False
	position_rmse, rotation_rmse = pose_consistency(source_pose, artifacts[0])
	print(
		f"Consistency {results_dir.parent.name}: normalized_position_rmse={position_rmse:.6f} "
		f"rotation_rmse_deg={rotation_rmse:.3f}",
		flush=True,
	)
	return position_rmse <= maximum_position_rmse and rotation_rmse <= maximum_rotation_rmse


def reuse_existing_window(
	name: str,
	source_runs_dir: Path,
	maps_dir: Path,
	attempts_dir: Path,
	source_pose: Path,
	dataset_name: str,
	args: argparse.Namespace,
) -> bool:
	accepted_dir = maps_dir / name
	source_dir = source_runs_dir / name
	if not accepted_dir.exists() and all(
		path.exists() for path in expected_window_artifacts(source_dir / "results", dataset_name)
	):
		maps_dir.mkdir(parents=True, exist_ok=True)
		accepted_dir.symlink_to(source_dir.resolve(), target_is_directory=True)
	if not accepted_dir.exists():
		return False
	if window_is_consistent(
		source_pose,
		accepted_dir / "results",
		dataset_name,
		args.maximum_rerun_position_rmse,
		args.maximum_rerun_rotation_rmse,
	):
		print(f"Reusing map window: {name}", flush=True)
		return True
	rejected_dir = attempts_dir / f"{name}_rejected_existing"
	if rejected_dir.exists():
		raise ValueError(f"cannot preserve divergent existing window; destination exists: {rejected_dir}")
	accepted_dir.rename(rejected_dir)
	return False


def export_window_map(
	image_dir: Path,
	window: WindowSpec,
	source_runs_dir: Path,
	maps_dir: Path,
	attempts_dir: Path,
	dataset_name: str,
	args: argparse.Namespace,
) -> None:
	name, start, end = window
	source_pose = source_runs_dir / name / "results" / "pose" / f"{dataset_name}.npz"
	if not source_pose.exists():
		raise ValueError(f"selected source pose does not exist: {source_pose}")
	accepted_dir = maps_dir / name
	if reuse_existing_window(
		name,
		source_runs_dir,
		maps_dir,
		attempts_dir,
		source_pose,
		dataset_name,
		args,
	):
		return

	for attempt in range(1, args.map_attempts + 1):
		attempt_dir = attempts_dir / f"{name}_{attempt:02d}"
		results_dir = attempt_dir / "results"
		if attempt_dir.exists():
			if not all(path.exists() for path in expected_window_artifacts(results_dir, dataset_name)):
				raise ValueError(f"incomplete map attempt blocks resume: {attempt_dir}")
		else:
			run_vipe(
				input_path=image_dir,
				output_path=results_dir,
				pipeline=args.pipeline,
				buffer=args.buffer,
				image_max_edge=args.image_max_edge,
				save_slam_map=True,
				frame_start=start,
				frame_end=end,
			)
		if window_is_consistent(
			source_pose,
			results_dir,
			dataset_name,
			args.maximum_rerun_position_rmse,
			args.maximum_rerun_rotation_rmse,
		):
			maps_dir.mkdir(parents=True, exist_ok=True)
			attempt_dir.rename(accepted_dir)
			return
	print(f"No consistent map rerun found for {name} after {args.map_attempts} attempts", file=sys.stderr)
	raise RuntimeError(f"map export did not reproduce selected trajectory: {name}")


def complete_stage(path: Path, expected_files: list[str]) -> bool:
	if not path.exists():
		return False
	missing = [name for name in expected_files if not (path / name).exists()]
	if missing:
		raise ValueError(f"incomplete stage at {path}; missing: {', '.join(missing)}")
	return True


def validate_resume_configuration(
	args: argparse.Namespace,
	dataset_name: str,
	windows: list[WindowSpec],
) -> None:
	configuration = {
		"image_dir": str(args.image_dir.resolve()),
		"selection_json": str(args.selection_json.resolve()),
		"dataset_name": dataset_name,
		"windows": windows,
		"pipeline": args.pipeline,
		"buffer": args.buffer,
		"image_max_edge": args.image_max_edge,
		"iterations": args.iterations,
		"max_gaussians": args.max_gaussians,
		"render_width": args.render_width,
		"holdout_stride": args.holdout_stride,
		"video_fps": args.video_fps,
		"initial_scale": args.initial_scale,
		"learning_rate_scale": args.learning_rate_scale,
		"refine_start": args.refine_start,
		"refine_stop": args.refine_stop,
		"refine_every": args.refine_every,
		"grow_gradient": args.grow_gradient,
		"evaluate_gps": args.evaluate_gps,
		"seed": args.seed,
	}
	configuration_path = args.output_dir / "configuration.json"
	if configuration_path.exists():
		existing = json.loads(configuration_path.read_text(encoding="utf-8"))
		if existing != configuration:
			raise ValueError(f"resume configuration does not match existing output: {configuration_path}")
		return
	args.output_dir.mkdir(parents=True, exist_ok=True)
	configuration_path.write_text(f"{json.dumps(configuration, indent=2)}\n", encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Reconstruct a full image sequence from an automatic ViPE window selection")
	parser.add_argument("image_dir", type=Path)
	parser.add_argument("selection_json", type=Path)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--source-runs-dir", type=Path, default=Path("output/windows"))
	parser.add_argument("--pipeline", default="static_vda")
	parser.add_argument("--buffer", type=positive_int, default=128)
	parser.add_argument("--image-max-edge", type=positive_int, default=512)
	parser.add_argument("--map-attempts", type=positive_int, default=3)
	parser.add_argument("--maximum-rerun-position-rmse", type=positive_float, default=0.02)
	parser.add_argument("--maximum-rerun-rotation-rmse", type=positive_float, default=5.0)
	parser.add_argument("--iterations", type=positive_int, default=2000)
	parser.add_argument("--max-gaussians", type=positive_int, default=100000)
	parser.add_argument("--render-width", type=positive_int, default=512)
	parser.add_argument("--holdout-stride", type=positive_int, default=8)
	parser.add_argument("--video-fps", type=positive_int, default=5)
	parser.add_argument("--initial-scale", type=positive_float, default=0.35)
	parser.add_argument("--learning-rate-scale", type=positive_float, default=1.0)
	parser.add_argument("--refine-start", type=positive_int, default=500)
	parser.add_argument("--refine-stop", type=positive_int, default=1000)
	parser.add_argument("--refine-every", type=positive_int, default=100)
	parser.add_argument("--grow-gradient", type=positive_float, default=0.0015)
	parser.add_argument("--evaluate-gps", action="store_true")
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	if not args.image_dir.is_dir():
		parser.error(f"image directory does not exist: {args.image_dir}")
	try:
		dataset_name, windows = load_selection(args.selection_json, args.image_dir)
		validate_resume_configuration(args, dataset_name, windows)
		maps_dir = args.output_dir / "windows"
		attempts_dir = args.output_dir / "map_attempts"
		attempts_dir.mkdir(parents=True, exist_ok=True)
		for window in windows:
			export_window_map(
				args.image_dir,
				window,
				args.source_runs_dir,
				maps_dir,
				attempts_dir,
				dataset_name,
				args,
			)

		stitch_dir = args.output_dir / "stitch"
		if not complete_stage(stitch_dir, ["poses.npz", "window_transforms.json", "metrics.json"]):
			stitch_full_poses(
				image_dir=args.image_dir,
				windows=windows,
				runs_dir=maps_dir,
				output_dir=stitch_dir,
				skip_gps_evaluation=not args.evaluate_gps,
			)
		else:
			print(f"Reusing stitch: {stitch_dir}", flush=True)

		gaussian_dir = args.output_dir / "gaussians"
		if not complete_stage(gaussian_dir, ["model.pt", "model.ply", "trajectory.mp4", "metrics.json"]):
			train_gaussians(
				TrainGaussiansConfig(
					vipe_output=stitch_dir,
					artifact=dataset_name,
					window=windows,
					runs_dir=maps_dir,
					output_dir=gaussian_dir,
					iterations=args.iterations,
					max_gaussians=args.max_gaussians,
					render_width=args.render_width,
					holdout_stride=args.holdout_stride,
					video_fps=args.video_fps,
					initial_scale=args.initial_scale,
					learning_rate_scale=args.learning_rate_scale,
					refine_start=args.refine_start,
					refine_stop=args.refine_stop,
					refine_every=args.refine_every,
					grow_gradient=args.grow_gradient,
					seed=args.seed,
				)
			)
		else:
			print(f"Reusing Gaussian reconstruction: {gaussian_dir}", flush=True)
	except (KeyError, TypeError, ValueError, RuntimeError) as error:
		parser.error(str(error))


if __name__ == "__main__":
	main()