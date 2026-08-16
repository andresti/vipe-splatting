from pathlib import Path

import numpy as np
from PIL import ExifTags, Image


IMAGE_SUFFIXES = {".jpg", ".jpeg"}


def load_gps_positions(image_dir: Path) -> np.ndarray:
	"""Load EXIF GPS coordinates as local east, north, altitude positions in metres."""
	gps_tag = {value: key for key, value in ExifTags.TAGS.items()}["GPSInfo"]
	image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
	if not image_paths:
		raise ValueError(f"no JPEG images found in {image_dir}")

	coordinates = []
	for path in image_paths:
		gps = Image.open(path).getexif().get_ifd(gps_tag)
		if not gps:
			raise ValueError(f"image has no EXIF GPS data: {path}")
		try:
			latitude = sum(float(value) / divisor for value, divisor in zip(gps[2], (1, 60, 3600)))
			longitude = sum(float(value) / divisor for value, divisor in zip(gps[4], (1, 60, 3600)))
			latitude *= -1 if gps[1] == "S" else 1
			longitude *= -1 if gps[3] == "W" else 1
			altitude = float(gps[6]) * (-1 if gps.get(5, 0) == 1 else 1)
		except (KeyError, TypeError, ValueError) as error:
			raise ValueError(f"image has incomplete EXIF GPS data: {path}") from error
		coordinates.append((latitude, longitude, altitude))

	coordinates_array = np.asarray(coordinates, dtype=float)
	latitude_origin = np.deg2rad(coordinates_array[0, 0])
	return np.column_stack(
		(
			(coordinates_array[:, 1] - coordinates_array[0, 1]) * 111320 * np.cos(latitude_origin),
			(coordinates_array[:, 0] - coordinates_array[0, 0]) * 110540,
			coordinates_array[:, 2] - coordinates_array[0, 2],
		)
	)


def similarity_transform(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
	"""Return scale, row-vector rotation, and translation aligning source to target."""
	if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
		raise ValueError("source and target must have matching N x 3 shapes")
	source_centered = source - source.mean(axis=0)
	target_centered = target - target.mean(axis=0)
	denominator = np.square(source_centered).sum()
	if denominator <= np.finfo(float).eps:
		raise ValueError("source positions do not span a usable trajectory")
	left, singular_values, right_transpose = np.linalg.svd(source_centered.T @ target_centered)
	rotation = left @ right_transpose
	if np.linalg.det(rotation) < 0:
		left[:, -1] *= -1
		rotation = left @ right_transpose
	scale = singular_values.sum() / denominator
	translation = target.mean(axis=0) - scale * source.mean(axis=0) @ rotation
	return float(scale), rotation, translation


def align_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
	"""Align positions while preserving the operation order used by historical metrics."""
	source_centered = source - source.mean(axis=0)
	target_centered = target - target.mean(axis=0)
	left, singular_values, right_transpose = np.linalg.svd(source_centered.T @ target_centered)
	rotation = left @ right_transpose
	if np.linalg.det(rotation) < 0:
		left[:, -1] *= -1
		rotation = left @ right_transpose
	scale = singular_values.sum() / np.square(source_centered).sum()
	aligned = scale * source_centered @ rotation + target.mean(axis=0)
	return aligned, float(scale)


def step_length_correlation(first: np.ndarray, second: np.ndarray) -> float | None:
	if len(first) < 2 or len(second) < 2:
		return None
	first_steps = np.linalg.norm(np.diff(first, axis=0), axis=1)
	second_steps = np.linalg.norm(np.diff(second, axis=0), axis=1)
	if np.std(first_steps) <= np.finfo(float).eps or np.std(second_steps) <= np.finfo(float).eps:
		return None
	return float(np.corrcoef(first_steps, second_steps)[0, 1])


def trajectory_metrics(
	positions: np.ndarray,
	reference: np.ndarray,
	frame_indices: np.ndarray | None = None,
) -> dict[str, float | int | None]:
	if positions.shape != reference.shape:
		raise ValueError("positions and reference must have matching shapes")
	if frame_indices is None:
		frame_indices = np.arange(len(positions))
	difference = positions - reference
	errors = np.linalg.norm(difference, axis=1)
	horizontal_errors = np.linalg.norm(difference[:, :2], axis=1)
	vertical_errors = np.abs(difference[:, 2])
	return {
		"rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
		"median_error_m": float(np.median(errors)),
		"max_error_m": float(errors.max()),
		"horizontal_rmse_m": float(np.sqrt(np.mean(np.square(horizontal_errors)))),
		"vertical_rmse_m": float(np.sqrt(np.mean(np.square(vertical_errors)))),
		"step_length_correlation": step_length_correlation(reference, positions),
		"largest_error_source_frame": int(frame_indices[errors.argmax()]),
	}