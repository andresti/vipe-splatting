import argparse
from pathlib import Path

import numpy as np


WindowSpec = tuple[str, int, int]


def parse_window(value: str) -> WindowSpec:
	try:
		name, start_text, end_text = value.split(":")
		start = int(start_text)
		end = int(end_text)
	except ValueError as error:
		raise argparse.ArgumentTypeError("window must be RUN_NAME:START:END") from error
	if start < 0 or end <= start:
		raise argparse.ArgumentTypeError("window must satisfy 0 <= START < END")
	return name, start, end


def load_window_poses(runs_dir: Path, dataset_name: str, window: WindowSpec) -> np.ndarray:
	name, start, end = window
	pose_path = runs_dir / name / "results" / "pose" / f"{dataset_name}.npz"
	if not pose_path.exists():
		raise ValueError(f"pose artifact does not exist: {pose_path}")
	poses = np.load(pose_path)["data"].astype(float)
	if poses.shape != (end - start, 4, 4):
		raise ValueError(f"{name} has pose shape {poses.shape}, expected {(end - start, 4, 4)}")
	return poses