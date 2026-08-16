import argparse
import json
from pathlib import Path

import torch

from vipe_pipeline.core.cli import positive_float, require_new_output
from vipe_pipeline.core.gaussian import (
	VipeGaussianDataset,
	export_gaussian_ply,
	initialize_gaussians,
	load_vipe_gaussian_dataset,
	photometric_loss,
	render_camera_path_video,
	render_gaussians,
	save_gaussian_checkpoint,
)


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
		rendered, _ = render_gaussians(
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
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--iterations", type=positive_int, default=2000)
	parser.add_argument("--max-gaussians", type=positive_int, default=60000)
	parser.add_argument("--render-width", type=positive_int, default=320)
	parser.add_argument("--holdout-stride", type=positive_int, default=8)
	parser.add_argument("--video-fps", type=positive_int, default=15)
	parser.add_argument("--initial-scale", type=positive_float, default=0.35)
	parser.add_argument("--learning-rate-scale", type=positive_float, default=1.0)
	parser.add_argument("--seed", type=int, default=42)
	args = parser.parse_args()
	require_new_output(parser, args.output_dir)
	if not torch.cuda.is_available():
		parser.error("Gaussian training requires a CUDA GPU")

	torch.manual_seed(args.seed)
	device = torch.device("cuda")
	try:
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
	optimizer = torch.optim.Adam(
		[
			{"params": [gaussians["means"]], "lr": 1.6e-4 * args.learning_rate_scale},
			{"params": [gaussians["log_scales"]], "lr": 5e-3 * args.learning_rate_scale},
			{"params": [gaussians["quaternions"]], "lr": 1e-3 * args.learning_rate_scale},
			{"params": [gaussians["opacity_logits"]], "lr": 5e-2 * args.learning_rate_scale},
			{"params": [gaussians["color_logits"]], "lr": 2.5e-3 * args.learning_rate_scale},
		],
		lr=0.0,
		weight_decay=0.0,
	)

	holdout_indices = list(range(0, len(dataset.images), args.holdout_stride))
	holdout_set = set(holdout_indices)
	train_indices = [index for index in range(len(dataset.images)) if index not in holdout_set]
	if not train_indices or not holdout_indices:
		parser.error("holdout split must leave at least one training and one evaluation frame")
	initial_psnr = evaluate(gaussians, dataset, holdout_indices, device)
	loss_history = []
	for iteration in range(1, args.iterations + 1):
		index = train_indices[torch.randint(len(train_indices), ()).item()]
		rendered, _ = render_gaussians(
			gaussians,
			dataset.camera_to_world[index].to(device),
			dataset.intrinsics[index].to(device),
			dataset.width,
			dataset.height,
		)
		target = dataset.images[index].to(device).float() / 255
		loss = photometric_loss(rendered, target)
		optimizer.zero_grad(set_to_none=True)
		loss.backward()
		optimizer.step()
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
		"source": "ViPE SLAMMap dense disparity points, camera poses, intrinsics, and RGB",
		"vipe_output": str(args.vipe_output),
		"artifact": args.artifact,
		"frame_count": len(dataset.images),
		"training_frame_count": len(train_indices),
		"holdout_frame_count": len(holdout_indices),
		"gaussian_count": len(gaussians["means"]),
		"iterations": args.iterations,
		"initial_scale": args.initial_scale,
		"learning_rate_scale": args.learning_rate_scale,
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