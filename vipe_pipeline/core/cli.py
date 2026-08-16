import argparse
from pathlib import Path


def positive_float(value: str) -> float:
	parsed = float(value)
	if parsed <= 0:
		raise argparse.ArgumentTypeError("value must be greater than zero")
	return parsed


def require_new_output(parser: argparse.ArgumentParser, output_path: Path) -> None:
	if output_path.exists():
		parser.error(f"refusing to overwrite existing output: {output_path}")