import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from vipe_pipeline.core.cli import positive_float


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("value must be greater than zero")
	return parsed


def stage_complete(paths: list[Path]) -> bool:
	return all(path.is_file() for path in paths)


def run_module(module: str, arguments: list[str], log_path: Path) -> None:
	command = [sys.executable, "-m", module, *arguments]
	print(f"Running: {shlex.join(command)}", flush=True)
	with log_path.open("w", encoding="utf-8") as log_file:
		process = subprocess.Popen(
			command,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
		)
		if process.stdout is None:
			raise RuntimeError(f"failed to capture output from {module}")
		for line in process.stdout:
			print(line, end="")
			log_file.write(line)
		return_code = process.wait()
	if return_code:
		raise subprocess.CalledProcessError(return_code, command)


def validate_resume_configuration(output_dir: Path, configuration: dict[str, object]) -> None:
	configuration_path = output_dir / "configuration.json"
	if configuration_path.exists():
		existing = json.loads(configuration_path.read_text(encoding="utf-8"))
		if existing != configuration:
			raise ValueError(f"resume configuration does not match existing output: {configuration_path}")
		return
	if output_dir.exists() and any(output_dir.iterdir()):
		raise ValueError(f"output directory is not empty and has no configuration: {output_dir}")
	output_dir.mkdir(parents=True, exist_ok=True)
	configuration_path.write_text(f"{json.dumps(configuration, indent=2)}\n", encoding="utf-8")


def main() -> None:
	parser = argparse.ArgumentParser(description="Run full-sequence ViPE and train a Gaussian reconstruction")
	parser.add_argument("image_dir", nargs="?", type=Path, default=Path("zavod70"))
	parser.add_argument("--output-dir", type=Path)
	parser.add_argument("--pipeline", default="static_vda")
	parser.add_argument("--buffer", type=positive_int, default=256)
	parser.add_argument("--image-max-edge", type=positive_int, default=512)
	parser.add_argument("--max-gaussians", type=positive_int, default=200000)
	parser.add_argument("--iterations", type=positive_int, default=4000)
	parser.add_argument("--render-width", type=positive_int, default=512)
	parser.add_argument("--holdout-stride", type=positive_int, default=8)
	parser.add_argument("--video-fps", type=positive_int, default=5)
	parser.add_argument("--initial-scale", type=positive_float, default=0.35)
	parser.add_argument("--learning-rate-scale", type=positive_float, default=1.0)
	parser.add_argument("--refine-start", type=positive_int, default=500)
	parser.add_argument("--refine-stop", type=positive_int, default=2000)
	parser.add_argument("--refine-every", type=positive_int, default=100)
	parser.add_argument("--grow-gradient", type=positive_float, default=0.0015)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()

	image_dir = args.image_dir.resolve()
	if not image_dir.is_dir():
		parser.error(f"image directory does not exist: {image_dir}; run bash setup.sh for the default dataset")
	artifact_name = image_dir.name
	output_dir = (args.output_dir or Path("output") / artifact_name).resolve()
	vipe_dir = output_dir / "vipe"
	gaussian_dir = output_dir / "gaussians"
	configuration = {
		"image_dir": str(image_dir),
		"artifact": artifact_name,
		"pipeline": args.pipeline,
		"buffer": args.buffer,
		"image_max_edge": args.image_max_edge,
		"max_gaussians": args.max_gaussians,
		"iterations": args.iterations,
		"render_width": args.render_width,
		"holdout_stride": args.holdout_stride,
		"video_fps": args.video_fps,
		"initial_scale": args.initial_scale,
		"learning_rate_scale": args.learning_rate_scale,
		"refine_start": args.refine_start,
		"refine_stop": args.refine_stop,
		"refine_every": args.refine_every,
		"grow_gradient": args.grow_gradient,
		"seed": args.seed,
	}

	try:
		validate_resume_configuration(output_dir, configuration)
		vipe_artifacts = [
			vipe_dir / "pose" / f"{artifact_name}.npz",
			vipe_dir / "intrinsics" / f"{artifact_name}.npz",
			vipe_dir / "rgb" / f"{artifact_name}.mp4",
			vipe_dir / "vipe" / f"{artifact_name}_slam_map.pt",
		]
		if stage_complete(vipe_artifacts):
			print(f"Reusing complete ViPE stage: {vipe_dir}", flush=True)
		elif vipe_dir.exists():
			raise ValueError(f"incomplete ViPE stage blocks resume: {vipe_dir}")
		else:
			run_module(
				"vipe_pipeline.cli.run_vipe",
				[
					str(image_dir),
					"--pipeline", args.pipeline,
					"--buffer", str(args.buffer),
					"--image-max-edge", str(args.image_max_edge),
					"--save-slam-map",
					"--output", str(vipe_dir),
				],
				output_dir / "vipe.log",
			)

		gaussian_artifacts = [
			gaussian_dir / "model.pt",
			gaussian_dir / "model.ply",
			gaussian_dir / "trajectory.mp4",
			gaussian_dir / "metrics.json",
		]
		if stage_complete(gaussian_artifacts):
			print(f"Reusing complete Gaussian stage: {gaussian_dir}", flush=True)
		elif gaussian_dir.exists():
			raise ValueError(f"incomplete Gaussian stage blocks resume: {gaussian_dir}")
		else:
			run_module(
				"vipe_pipeline.cli.train_gaussians",
				[
					str(vipe_dir),
					"--artifact", artifact_name,
					"--output-dir", str(gaussian_dir),
					"--max-gaussians", str(args.max_gaussians),
					"--iterations", str(args.iterations),
					"--render-width", str(args.render_width),
					"--holdout-stride", str(args.holdout_stride),
					"--video-fps", str(args.video_fps),
					"--initial-scale", str(args.initial_scale),
					"--learning-rate-scale", str(args.learning_rate_scale),
					"--refine-start", str(args.refine_start),
					"--refine-stop", str(args.refine_stop),
					"--refine-every", str(args.refine_every),
					"--grow-gradient", str(args.grow_gradient),
					"--seed", str(args.seed),
				],
				output_dir / "gaussians.log",
			)
	except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
		parser.error(str(error))

	print(f"Reconstruction complete: {output_dir}")
	print(f"Model: {gaussian_dir / 'model.ply'}")
	print(f"Video: {gaussian_dir / 'trajectory.mp4'}")


if __name__ == "__main__":
	main()