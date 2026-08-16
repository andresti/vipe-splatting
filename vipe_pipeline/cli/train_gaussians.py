import argparse
import json
from pathlib import Path

import torch
from gsplat import DefaultStrategy

from vipe_pipeline.core.cli import positive_float, require_new_output
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Train Gaussian splats from ViPE cameras and SLAM geometry")
	parser.add_argument("vipe_output", type=Path)
	parser.add_argument("--artifact", required=True, help="ViPE artifact name, usually the input directory or video stem")
	parser.add_argument("--window", action="append", type=parse_window, help="stitched window as RUN_NAME:START:END")
	parser.add_argument("--runs-dir", type=Path, default=Path("vipe_smoke_test_out/windows"))
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
	require_new_output(parser, args.output_dir)
	if not torch.cuda.is_available():
		parser.error("Gaussian training requires a CUDA GPU")

	torch.manual_seed(args.seed)
	device = torch.device("cuda")
	try:
		if args.window:
			dataset = load_stitched_vipe_gaussian_dataset(
				args.vipe_output,
				args.runs_dir,
				args.artifact,
				args.window,
				args.render_width,
				args.max_gaussians,
				args.seed,
			)
		else:
			dataset = load_vipe_gaussian_dataset(
				args.vipe_output,
				args.artifact,
				args.render_width,
				args.max_gaussians,
				args.seed,
			)
	except ValueError as error:
		parser.error(str(error))
	gaussians = initialize_gaussians(dataset, device, args.initial_scale)
	learning_rates = {
		"means": 1.6e-4,
		"scales": 5e-3,
		"quats": 1e-3,
		"opacities": 5e-2,
		"colors": 2.5e-3,
	}
	optimizers = {
		name: torch.optim.Adam(
			[{"params": [gaussians[name]], "lr": learning_rate * args.learning_rate_scale}],
			lr=0.0,
			weight_decay=0.0,
		)
		for name, learning_rate in learning_rates.items()
	}
	strategy = DefaultStrategy(
		refine_start_iter=args.refine_start,
		refine_stop_iter=min(args.refine_stop, args.iterations),
		refine_every=args.refine_every,
		reset_every=args.iterations + 1,
		grow_grad2d=args.grow_gradient,
		absgrad=True,
		verbose=True,
	)
	strategy.check_sanity(gaussians, optimizers)
	scene_extent = torch.linalg.vector_norm(dataset.points.max(dim=0).values - dataset.points.min(dim=0).values).item()
	strategy_state = strategy.initialize_state(scene_scale=scene_extent)

	holdout_indices = list(range(0, len(dataset.images), args.holdout_stride))
	holdout_set = set(holdout_indices)
	train_indices = [index for index in range(len(dataset.images)) if index not in holdout_set]
	if not train_indices or not holdout_indices:
		parser.error("holdout split must leave at least one training and one evaluation frame")
	initial_psnr = evaluate(gaussians, dataset, holdout_indices, device)
	loss_history = []
	for iteration in range(1, args.iterations + 1):
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
		if iteration == 1 or iteration % 100 == 0 or iteration == args.iterations:
			window_loss = sum(loss_history[-100:]) / min(100, len(loss_history))
			print(f"iteration={iteration} loss={window_loss:.6f}")

	final_psnr = evaluate(gaussians, dataset, holdout_indices, device)
	args.output_dir.mkdir(parents=True)
	save_gaussian_checkpoint(args.output_dir / "model.pt", gaussians, dataset)
	export_gaussian_ply(args.output_dir / "model.ply", gaussians)
	render_camera_path_video(args.output_dir / "trajectory.mp4", gaussians, dataset, device, args.video_fps)
	metrics = {
		"source": "stitched ViPE SLAMMap windows" if args.window else "single ViPE SLAMMap window",
		"vipe_output": str(args.vipe_output),
		"artifact": args.artifact,
		"frame_count": len(dataset.images),
		"training_frame_count": len(train_indices),
		"holdout_frame_count": len(holdout_indices),
		"initial_gaussian_count": min(len(dataset.points), args.max_gaussians),
		"gaussian_count": len(gaussians["means"]),
		"iterations": args.iterations,
		"initial_scale": args.initial_scale,
		"learning_rate_scale": args.learning_rate_scale,
		"refine_start": args.refine_start,
		"refine_stop": min(args.refine_stop, args.iterations),
		"refine_every": args.refine_every,
		"grow_gradient": args.grow_gradient,
		"render_width": dataset.width,
		"render_height": dataset.height,
		"initial_holdout_psnr_db": initial_psnr,
		"final_holdout_psnr_db": final_psnr,
		"final_training_loss": sum(loss_history[-100:]) / min(100, len(loss_history)),
		"peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
	}
	(args.output_dir / "metrics.json").write_text(f"{json.dumps(metrics, indent=2)}\n", encoding="utf-8")
	print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
	main()