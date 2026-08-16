import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from vipe_pipeline.core.trajectory import align_similarity, load_gps_positions, trajectory_metrics


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument("pose", type=Path)
	parser.add_argument("image_dir", type=Path)
	parser.add_argument("--frame-start", type=int, required=True)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--metrics-output", type=Path)
	args = parser.parse_args()
	for output_path in (args.output, args.metrics_output):
		if output_path is not None and output_path.exists():
			parser.error(f"refusing to overwrite existing output: {output_path}")

	pose_data = np.load(args.pose)
	vipe_positions = pose_data["data"][:, :3, 3]
	all_gps_positions = load_gps_positions(args.image_dir)
	frame_end = args.frame_start + len(vipe_positions)
	if args.frame_start < 0 or frame_end > len(all_gps_positions):
		parser.error(f"pose range {args.frame_start}:{frame_end} exceeds available GPS frames")
	gps_positions = all_gps_positions[args.frame_start:frame_end]

	aligned_positions, scale = align_similarity(vipe_positions, gps_positions)
	position_metrics = trajectory_metrics(
		aligned_positions,
		gps_positions,
		np.arange(args.frame_start, frame_end),
	)
	metrics = {
		"frame_start": args.frame_start,
		"frame_end": frame_end,
		"frame_count": len(vipe_positions),
		"scale": scale,
		"rmse_m": position_metrics["rmse_m"],
		"median_error_m": position_metrics["median_error_m"],
		"max_error_m": position_metrics["max_error_m"],
		"step_length_correlation": position_metrics["step_length_correlation"],
		"largest_error_source_frame": position_metrics["largest_error_source_frame"],
	}
	metrics_json = json.dumps(metrics, indent=2)
	print(metrics_json)
	if args.metrics_output is not None:
		args.metrics_output.write_text(f"{metrics_json}\n", encoding="utf-8")

	if args.output is not None:
		figure, axes = plt.subplots(1, 2, figsize=(12, 5))
		axes[0].plot(gps_positions[:, 0], gps_positions[:, 1], "-o", markersize=2, label="GPS")
		axes[0].plot(aligned_positions[:, 0], aligned_positions[:, 1], "-o", markersize=2, label="ViPE aligned")
		axes[0].set(xlabel="East (m)", ylabel="North (m)", title=f"Frames {args.frame_start}-{frame_end - 1}")
		axes[0].set_aspect("equal")
		axes[0].grid(alpha=0.3)
		axes[0].legend()
		axes[1].plot(range(args.frame_start, frame_end), gps_positions[:, 2], label="GPS altitude")
		axes[1].plot(range(args.frame_start, frame_end), aligned_positions[:, 2], label="ViPE aligned altitude")
		axes[1].set(xlabel="Source frame", ylabel="Height (m)", title="Vertical comparison")
		axes[1].grid(alpha=0.3)
		axes[1].legend()
		figure.tight_layout()
		figure.savefig(args.output, dpi=160)


if __name__ == "__main__":
	main()
