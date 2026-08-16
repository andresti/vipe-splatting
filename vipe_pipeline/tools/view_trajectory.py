import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf


def trajectory_colors(count: int) -> np.ndarray:
	fraction = np.linspace(0.0, 1.0, count)
	return np.column_stack(
		(
			255 * (1.0 - fraction),
			80 + 100 * np.sin(np.pi * fraction),
			255 * fraction,
		)
	).astype(np.uint8)


def main() -> None:
	parser = argparse.ArgumentParser(description="View all ViPE camera poses and the complete trajectory")
	parser.add_argument("pose", type=Path)
	parser.add_argument("--port", type=int, default=20543)
	parser.add_argument("--frame-start", type=int, default=0)
	parser.add_argument("--frustum-step", type=int, default=5)
	args = parser.parse_args()
	if args.frustum_step <= 0:
		parser.error("--frustum-step must be greater than zero")

	poses = np.load(args.pose)["data"]
	positions = poses[:, :3, 3]
	colors = trajectory_colors(len(positions))
	span = np.ptp(positions, axis=0)
	scene_size = max(float(span.max()), 1.0)
	center = (positions.min(axis=0) + positions.max(axis=0)) / 2
	frustum_scale = scene_size * 0.035

	server = viser.ViserServer(host="0.0.0.0", port=args.port, label="ViPE trajectory", verbose=False)
	segments = np.stack((positions[:-1], positions[1:]), axis=1)
	segment_colors = np.stack((colors[:-1], colors[1:]), axis=1)
	server.scene.add_line_segments("/trajectory", segments, segment_colors, line_width=5.0)
	server.scene.add_point_cloud(
		"/camera_centers",
		positions,
		colors,
		point_size=scene_size * 0.012,
		point_shape="circle",
	)
	server.scene.add_grid(
		"/reference_grid",
		width=scene_size * 1.5,
		height=scene_size * 1.5,
		cell_size=max(scene_size / 20, 0.1),
		section_size=max(scene_size / 5, 0.5),
		position=(float(center[0]), float(center[1]), float(positions[:, 2].min())),
	)

	frustum_indices = list(range(0, len(poses), args.frustum_step))
	if frustum_indices[-1] != len(poses) - 1:
		frustum_indices.append(len(poses) - 1)
	for index in frustum_indices:
		server.scene.add_camera_frustum(
			f"/cameras/{args.frame_start + index}",
			fov=np.deg2rad(60.0),
			aspect=4 / 3,
			scale=frustum_scale,
			line_width=3.0,
			color=tuple(int(value) for value in colors[index]),
			wxyz=tf.SO3.from_matrix(poses[index, :3, :3]).wxyz,
			position=positions[index],
		)

	server.scene.add_label(
		"/labels/start",
		f"Start: frame {args.frame_start}",
		position=positions[0],
		font_screen_scale=1.2,
	)
	server.scene.add_label(
		"/labels/end",
		f"End: frame {args.frame_start + len(poses) - 1}",
		position=positions[-1],
		font_screen_scale=1.2,
	)

	@server.on_client_connect
	async def _(client: viser.ClientHandle) -> None:
		await asyncio.sleep(0.5)
		client.camera.look_at = center
		client.camera.position = center + np.array((scene_size * 1.2, scene_size * 1.2, scene_size * 1.5))
		client.camera.up_direction = np.array((0.0, 0.0, 1.0))
		client.camera.fov = np.deg2rad(55.0)

	print(f"Loaded {len(poses)} poses, {len(segments)} path segments, and {len(frustum_indices)} frustums")
	print(f"Trajectory span: {span}")
	print(f"Open http://127.0.0.1:{args.port}")
	try:
		while True:
			time.sleep(10)
	except KeyboardInterrupt:
		server.stop()


if __name__ == "__main__":
	main()
