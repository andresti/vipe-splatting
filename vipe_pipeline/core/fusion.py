import numpy as np


def fuse_positions(
	vipe_positions: np.ndarray,
	gps_positions: np.ndarray,
	anchor_mask: np.ndarray,
	motion_sigma: float,
	gps_horizontal_sigma: float,
	gps_vertical_sigma: float,
) -> np.ndarray:
	"""Correct low-frequency position drift while preserving ViPE increments."""
	frame_count = len(vipe_positions)
	diagonal = np.full(frame_count, 2.0)
	diagonal[[0, -1]] = 1.0
	laplacian = (
		np.diag(diagonal)
		+ np.diag(np.full(frame_count - 1, -1.0), 1)
		+ np.diag(np.full(frame_count - 1, -1.0), -1)
	)
	motion_weight = 1 / motion_sigma**2
	fused = np.empty_like(vipe_positions)
	for axis, gps_sigma in enumerate((gps_horizontal_sigma, gps_horizontal_sigma, gps_vertical_sigma)):
		gps_weights = anchor_mask.astype(float) / gps_sigma**2
		normal_matrix = motion_weight * laplacian + np.diag(gps_weights)
		right_hand_side = motion_weight * laplacian @ vipe_positions[:, axis] + gps_weights * gps_positions[:, axis]
		fused[:, axis] = np.linalg.solve(normal_matrix, right_hand_side)
	return fused