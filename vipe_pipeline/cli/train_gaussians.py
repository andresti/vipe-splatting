import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from gsplat import DefaultStrategy

from vipe_pipeline.core.cli import positive_float
from vipe_pipeline.core.gaussian import (
	VipeGaussianDataset,
	export_gaussian_ply,
	initialize_gaussians,
	load_stitched_vipe_gaussian_dataset,
	load_vipe_gaussian_dataset,
	photometric_loss,
	render_camera_path_video,
	render_gaussians,
	save_gaussian_checkpoint,
)
from vipe_pipeline.core.windows import parse_window


def positive_int(value: str) -> int:
	parsed = int(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("value must be greater than zero")
	return parsed


@torch.no_grad()
def evaluate(
	gaussians: torch.nn.ParameterDict,
	dataset: VipeGaussianDataset,
	indices: list[int],
	device: torch.device,
) -> float:
	mean_squared_errors = []
	for index in indices:
		rendered, _, _ = render_gaussians(
			gaussians,
			dataset.camera_to_world[index].to(device),
			dataset.intrinsics[index].to(device),
			dataset.width,
			dataset.height,
		)
		target = dataset.images[index].to(device).float() / 255
		mean_squared_errors.append(torch.mean((rendered - target).square()).clamp_min(1e-10))
	return float((-10 * torch.log10(torch.stack(mean_squared_errors).mean())).item())


@dataclass(frozen=True)
class TrainGaussiansConfig:
	vipe_output: Path
	artifact: str
	output_dir: Path
	window: list[tuple[str, int, int]] | None = None
	runs_dir: Path = Path("output/windows")
	iterations: int = 2000
	max_gaussians: int = 60000
	render_width: int = 512
	holdout_stride: int = 8
	video_fps: int = 5
	initial_scale: float = 0.35
	learning_rate_scale: float = 1.0
	refine_start: int = 500
	refine_stop: int = 1000
	refine_every: int = 100
	grow_gradient: float = 0.0015
	seed: int = 42


def main() -> None:
	parser = argparse.ArgumentParser(description="Train Gaussian splats from ViPE cameras and SLAM geometry")
	parser.add_argument("vipe_output", type=Path)
	parser.add_argument("--artifact", required=True, help="ViPE artifact name, usually the input directory or video stem")
	parser.add_argument("--window", action="append", type=parse_window, help="stitched window as RUN_NAME:START:END")
	parser.add_argument("--runs-dir", type=Path, default=Path("output/windows"))
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--iterations", type=positive_int, default=2000)
	parser.add_argument(
		"--max-gaussians",
		type=positive_int,
		default=60000,
		help="maximum ViPE map points used to seed training; adaptive refinement may exceed this count",
	)
	parser.add_argument("--render-width", type=positive_int, default=512)
	parser.add_argument("--holdout-stride", type=positive_int, default=8)
	parser.add_argument("--video-fps", type=positive_int, default=5)
	parser.add_argument("--initial-scale", type=positive_float, default=0.35)
	parser.add_argument("--learning-rate-scale", type=positive_float, default=1.0)
	parser.add_argument("--refine-start", type=positive_int, default=500)
	parser.add_argument("--refine-stop", type=positive_int, default=1000)
	parser.add_argument("--refine-every", type=positive_int, default=100)
	parser.add_argument("--grow-gradient", type=positive_float, default=0.0015)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	try:
		train_gaussians(
			TrainGaussiansConfig(
				vipe_output=args.vipe_output,
				artifact=args.artifact,
				output_dir=args.output_dir,
				window=args.window,
				runs_dir=args.runs_dir,
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
	except ValueError as error:
		parser.error(str(error))


def train_gaussians(config: TrainGaussiansConfig) -> None:
	if config.output_dir.exists():
		raise ValueError(f"refusing to overwrite existing output: {config.output_dir}")
	if not torch.cuda.is_available():
		raise ValueError("Gaussian training requires a CUDA GPU")

	torch.manual_seed(config.seed)
	device = torch.device("cuda")
	if config.window:
		dataset = load_stitched_vipe_gaussian_dataset(
			config.vipe_output,
			config.runs_dir,
			config.artifact,
			config.window,
			config.render_width,
			config.max_gaussians,
			config.seed,
		)
	else:
		dataset = load_vipe_gaussian_dataset(
			config.vipe_output,
			config.artifact,
			config.render_width,
			config.max_gaussians,
			config.seed,
		)
	gaussians = initialize_gaussians(dataset, device, config.initial_scale)
	learning_rates = {
		"means": 1.6e-4,
		"scales": 5e-3,
		"quats": 1e-3,
		"opacities": 5e-2,
		"colors": 2.5e-3,
	}
	optimizers = {
		name: torch.optim.Adam(
			[{"params": [gaussians[name]], "lr": learning_rate * config.learning_rate_scale}],
			lr=0.0,
			weight_decay=0.0,
		)
		for name, learning_rate in learning_rates.items()
	}
	strategy = DefaultStrategy(
		refine_start_iter=config.refine_start,
		refine_stop_iter=min(config.refine_stop, config.iterations),
		refine_every=config.refine_every,
		reset_every=config.iterations + 1,
		grow_grad2d=config.grow_gradient,
		absgrad=True,
		verbose=True,
	)
	strategy.check_sanity(gaussians, optimizers)
	scene_extent = torch.linalg.vector_norm(dataset.points.max(dim=0).values - dataset.points.min(dim=0).values).item()
	strategy_state = strategy.initialize_state(scene_scale=scene_extent)

	holdout_indices = list(range(0, len(dataset.images), config.holdout_stride))
	holdout_set = set(holdout_indices)
	train_indices = [index for index in range(len(dataset.images)) if index not in holdout_set]
	if not train_indices or not holdout_indices:
		raise ValueError("holdout split must leave at least one training and one evaluation frame")
	initial_psnr = evaluate(gaussians, dataset, holdout_indices, device)
	loss_history = []
	for iteration in range(1, config.iterations + 1):
		index = train_indices[torch.randint(len(train_indices), ()).item()]
		rendered, _, info = render_gaussians(
			gaussians,
			dataset.camera_to_world[index].to(device),
			dataset.intrinsics[index].to(device),
			dataset.width,
			dataset.height,
			absgrad=True,
		)
		target = dataset.images[index].to(device).float() / 255
		loss = photometric_loss(rendered, target)
		for optimizer in optimizers.values():
			optimizer.zero_grad(set_to_none=True)
		strategy.step_pre_backward(gaussians, optimizers, strategy_state, iteration, info)
		loss.backward()
		for optimizer in optimizers.values():
			optimizer.step()
		strategy.step_post_backward(
			gaussians,
			optimizers,
			strategy_state,
			iteration,
			info,
			packed=True,
		)
		loss_history.append(float(loss.detach()))
		if iteration == 1 or iteration % 100 == 0 or iteration == config.iterations:
			window_loss = sum(loss_history[-100:]) / min(100, len(loss_history))
			print(f"iteration={iteration} loss={window_loss:.6f}")

	final_psnr = evaluate(gaussians, dataset, holdout_indices, device)
	config.output_dir.mkdir(parents=True)
	save_gaussian_checkpoint(config.output_dir / "model.pt", gaussians, dataset)
	export_gaussian_ply(config.output_dir / "model.ply", gaussians)
	render_camera_path_video(config.output_dir / "trajectory.mp4", gaussians, dataset, device, config.video_fps)
	metrics = {
		"source": "stitched ViPE SLAMMap windows" if config.window else "single ViPE SLAMMap window",
		"vipe_output": str(config.vipe_output),
		"artifact": config.artifact,
		"frame_count": len(dataset.images),
		"training_frame_count": len(train_indices),
		"holdout_frame_count": len(holdout_indices),
		"initial_gaussian_count": min(len(dataset.points), config.max_gaussians),
		"gaussian_count": len(gaussians["means"]),
		"iterations": config.iterations,
		"initial_scale": config.initial_scale,
		"learning_rate_scale": config.learning_rate_scale,
		"refine_start": config.refine_start,
		"refine_stop": min(config.refine_stop, config.iterations),
		"refine_every": config.refine_every,
		"grow_gradient": config.grow_gradient,
		"render_width": dataset.width,
		"render_height": dataset.height,
		"initial_holdout_psnr_db": initial_psnr,
		"final_holdout_psnr_db": final_psnr,
		"final_training_loss": sum(loss_history[-100:]) / min(100, len(loss_history)),
		"peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
	}
	(config.output_dir / "metrics.json").write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()