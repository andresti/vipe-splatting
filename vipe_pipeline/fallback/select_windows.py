import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vipe_pipeline.core.cli import require_new_output
from vipe_pipeline.core.trajectory import similarity_transform


RUN_PATTERN = re.compile(r"^frames_(\d+)_(\d+)$")


@dataclass
class Window:
	name: str
	start: int
	end: int
	positions: np.ndarray
	step_ratio: float
	acceleration_ratio: float
	window_cost: float


@dataclass(frozen=True)
class SelectWindowsConfig:
	dataset_name: str
	runs_dir: Path = Path("output/windows")
	output_dir: Path = Path(".")
	target_start: int = 0
	target_end: int = -1
	minimum_overlap: int = 3
	maximum_step_ratio: float = 2.0
	maximum_acceleration_ratio: float = 8.0
	window_penalty: float = 0.25


def robust_max_ratio(values: np.ndarray) -> float:
	median = float(np.median(values))
	return float(values.max() / max(median, np.finfo(float).eps))


def load_window(run_dir: Path, dataset_name: str, window_penalty: float) -> Window | None:
	match = RUN_PATTERN.fullmatch(run_dir.name)
	if match is None:
		return None
	start, end = (int(value) for value in match.groups())
	pose_path = run_dir / "results" / "pose" / f"{dataset_name}.npz"
	if not pose_path.exists():
		return None
	positions = np.load(pose_path)["data"][:, :3, 3].astype(float)
	if len(positions) != end - start or len(positions) < 3:
		return None
	steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
	accelerations = np.linalg.norm(np.diff(positions, n=2, axis=0), axis=1)
	step_ratio = robust_max_ratio(steps)
	acceleration_ratio = robust_max_ratio(accelerations)
	window_cost = window_penalty + 0.25 * max(0.0, step_ratio - 1) + 0.05 * max(0.0, acceleration_ratio - 1)
	return Window(run_dir.name, start, end, positions, step_ratio, acceleration_ratio, window_cost)


def overlap_score(previous: Window, following: Window, minimum_overlap: int) -> dict[str, float | int] | None:
	overlap_start = max(previous.start, following.start)
	overlap_end = min(previous.end, following.end)
	if overlap_end - overlap_start < minimum_overlap or following.end <= previous.end:
		return None
	previous_overlap = previous.positions[overlap_start - previous.start : overlap_end - previous.start]
	following_overlap = following.positions[overlap_start - following.start : overlap_end - following.start]
	scale, rotation, translation = similarity_transform(following_overlap, previous_overlap)
	aligned = scale * following_overlap @ rotation + translation
	residuals = np.linalg.norm(aligned - previous_overlap, axis=1)
	previous_steps = np.linalg.norm(np.diff(previous_overlap, axis=0), axis=1)
	normalization = max(float(np.median(previous_steps)), np.finfo(float).eps)
	normalized_rmse = float(np.sqrt(np.mean(np.square(residuals))) / normalization)
	frame_count = overlap_end - overlap_start
	return {
		"frame_start": overlap_start,
		"frame_end": overlap_end,
		"frame_count": frame_count,
		"relative_scale": scale,
		"normalized_rmse": normalized_rmse,
		"edge_cost": normalized_rmse + 1 / frame_count,
	}


def update_best_transition(
	window: Window,
	previous: Window,
	minimum_overlap: int,
	best: dict[str, tuple[float, list[Window], list[dict[str, object]]]],
) -> None:
	if previous.name not in best:
		return
	edge = overlap_score(previous, window, minimum_overlap)
	if edge is None:
		return
	previous_cost, previous_chain, previous_edges = best[previous.name]
	candidate_cost = previous_cost + float(edge["edge_cost"]) + window.window_cost
	current = best.get(window.name)
	if current is None or candidate_cost < current[0]:
		best[window.name] = (
			candidate_cost,
			previous_chain + [window],
			previous_edges + [{"from": previous.name, "to": window.name, **edge}],
		)


def select_chain(
	windows: list[Window],
	target_start: int,
	target_end: int,
	minimum_overlap: int,
) -> tuple[list[Window], list[dict[str, float | int | str]], float]:
	ordered = sorted(windows, key=lambda window: (window.end, window.start))
	best: dict[str, tuple[float, list[Window], list[dict[str, object]]]] = {}
	for window in ordered:
		if window.start == target_start:
			best[window.name] = (window.window_cost, [window], [])
		for previous in ordered:
			update_best_transition(window, previous, minimum_overlap, best)

	complete = [result for name, result in best.items() if next(window for window in ordered if window.name == name).end >= target_end]
	if not complete:
		raise ValueError(f"no accepted window chain covers frames {target_start}:{target_end}")
	best_complete = min(complete, key=lambda result: result[0])
	return best_complete[1], best_complete[2], best_complete[0]


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("dataset_name")
	parser.add_argument("--runs-dir", type=Path, default=Path("output/windows"))
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--target-start", type=int, default=0)
	parser.add_argument("--target-end", type=int, required=True)
	parser.add_argument("--minimum-overlap", type=int, default=3)
	parser.add_argument("--maximum-step-ratio", type=float, default=2.0)
	parser.add_argument("--maximum-acceleration-ratio", type=float, default=8.0)
	parser.add_argument("--window-penalty", type=float, default=0.25)
	args = parser.parse_args()
	try:
		selection, command = select_windows(
			SelectWindowsConfig(
				dataset_name=args.dataset_name,
				runs_dir=args.runs_dir,
				output_dir=args.output_dir,
				target_start=args.target_start,
				target_end=args.target_end,
				minimum_overlap=args.minimum_overlap,
				maximum_step_ratio=args.maximum_step_ratio,
				maximum_acceleration_ratio=args.maximum_acceleration_ratio,
				window_penalty=args.window_penalty,
			)
		)
	except ValueError as error:
		parser.error(str(error))

	print(json.dumps(selection, indent=2))
	print(f"\nStitch command:\n{command}")


def select_windows(config: SelectWindowsConfig) -> tuple[dict[str, object], str]:
	if config.output_dir.exists():
		raise ValueError(f"refusing to overwrite existing output: {config.output_dir}")
	if not config.runs_dir.is_dir():
		raise ValueError(f"runs directory does not exist: {config.runs_dir}")
	if config.target_start < 0 or config.target_end <= config.target_start:
		raise ValueError("target range must satisfy 0 <= START < END")
	if config.minimum_overlap < 3:
		raise ValueError("--minimum-overlap must be at least 3")

	loaded = [load_window(run_dir, config.dataset_name, config.window_penalty) for run_dir in config.runs_dir.iterdir()]
	windows = [window for window in loaded if window is not None]
	accepted = [
		window
		for window in windows
		if window.step_ratio <= config.maximum_step_ratio
		and window.acceleration_ratio <= config.maximum_acceleration_ratio
	]
	chain, edges, total_cost = select_chain(accepted, config.target_start, config.target_end, config.minimum_overlap)

	def window_record(window: Window) -> dict[str, object]:
		rejection_reasons = []
		if window.step_ratio > config.maximum_step_ratio:
			rejection_reasons.append("step_ratio")
		if window.acceleration_ratio > config.maximum_acceleration_ratio:
			rejection_reasons.append("acceleration_ratio")
		return {
			"run_name": window.name,
			"frame_start": window.start,
			"frame_end": window.end,
			"frame_count": window.end - window.start,
			"step_ratio": window.step_ratio,
			"acceleration_ratio": window.acceleration_ratio,
			"window_cost": window.window_cost,
			"accepted": window in accepted,
			"rejection_reasons": rejection_reasons,
		}

	selection: dict[str, object] = {
		"selection_uses_gps": False,
		"dataset_name": config.dataset_name,
		"target_start": config.target_start,
		"target_end": config.target_end,
		"parameters": {
			"minimum_overlap": config.minimum_overlap,
			"maximum_step_ratio": config.maximum_step_ratio,
			"maximum_acceleration_ratio": config.maximum_acceleration_ratio,
			"window_penalty": config.window_penalty,
		},
		"candidate_count": len(windows),
		"accepted_count": len(accepted),
		"selected_total_cost": total_cost,
		"selected_windows": [window_record(window) for window in chain],
		"selected_edges": edges,
		"all_candidates": [window_record(window) for window in sorted(windows, key=lambda item: (item.start, item.end))],
	}
	config.output_dir.mkdir(parents=True)
	(config.output_dir / "selection.json").write_text(f"{json.dumps(selection, indent=2)}\n", encoding="utf-8")
	stitch_arguments = " ".join(f"--window {window.name}:{window.start}:{window.end}" for window in chain)
	command = (
		f"uv run python -m vipe_pipeline.fallback.stitch_full_poses <IMAGE_DIR> "
		f"--runs-dir {config.runs_dir} {stitch_arguments} --output-dir <OUTPUT_DIR>"
	)
	(config.output_dir / "stitch_command.txt").write_text(f"{command}\n", encoding="utf-8")
	return selection, command


if __name__ == "__main__":
	main()