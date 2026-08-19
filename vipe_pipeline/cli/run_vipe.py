import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from vipe.config.parse import parse_typed_config
from vipe.pipeline import make_pipeline
from vipe.streams.base import ProcessedVideoStream, StreamProcessor, VideoFrame
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.utils.logging import configure_logging


class ResizeLongestEdgeProcessor(StreamProcessor):
	n_passes_required = 1

	def __init__(self, max_edge: int) -> None:
		self.max_edge = max_edge

	def update_frame_size(self, previous_frame_size: tuple[int, int]) -> tuple[int, int]:
		height, width = previous_frame_size
		scale = min(1.0, self.max_edge / max(height, width))
		return round(height * scale), round(width * scale)

	def __call__(self, frame_idx: int, frame: VideoFrame) -> VideoFrame:
		return frame.resize(self.update_frame_size(frame.size()))


def main() -> None:
	parser = argparse.ArgumentParser(description="Run a bounded ViPE pipeline")
	parser.add_argument("input", type=Path, help="Directory of image frames")
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--pipeline", default="static_vda")
	parser.add_argument("--buffer", type=int, default=256)
	parser.add_argument("--image-max-edge", type=int, default=512)
	parser.add_argument("--save-slam-map", action="store_true")
	parser.add_argument("--frame-start", type=int, default=0)
	parser.add_argument("--frame-end", type=int, default=-1, help="exclusive end index; -1 processes to the end")
	args = parser.parse_args()
	try:
		run_vipe(
			input_path=args.input,
			output_path=args.output,
			pipeline=args.pipeline,
			buffer=args.buffer,
			image_max_edge=args.image_max_edge,
			save_slam_map=args.save_slam_map,
			frame_start=args.frame_start,
			frame_end=args.frame_end,
		)
	except ValueError as error:
		parser.error(str(error))


def run_vipe(
	input_path: Path,
	output_path: Path,
	pipeline: str = "static_vda",
	buffer: int = 256,
	image_max_edge: int = 512,
	save_slam_map: bool = False,
	frame_start: int = 0,
	frame_end: int = -1,
) -> None:
	if image_max_edge <= 0:
		raise ValueError("--image-max-edge must be greater than zero")
	if frame_start < 0:
		raise ValueError("--frame-start must be zero or greater")
	if frame_end != -1 and frame_end <= frame_start:
		raise ValueError("--frame-end must be greater than --frame-start or -1")
	if not input_path.exists():
		raise ValueError(f"input does not exist: {input_path}")
	if output_path.exists():
		raise ValueError(f"refusing to overwrite existing output: {output_path}")
	seek_range = range(frame_start, frame_end)
	if input_path.is_dir():
		raw_stream = FrameDirStream(input_path, seek_range=seek_range)
		input_processors = [ResizeLongestEdgeProcessor(image_max_edge)]
	else:
		raise ValueError("input must be a directory of image frames")

	configure_logging()
	config = parse_typed_config(
		"default",
		hydra_args=[
			f"pipeline={pipeline}",
			f"pipeline.slam.buffer={buffer}",
			f"pipeline.output.path={output_path}",
			"pipeline.post.depth_align_model=null",
			"pipeline.output.save_artifacts=true",
			f"pipeline.output.save_slam_map={str(save_slam_map).lower()}",
			"pipeline.output.save_viz=false",
		],
	)
	video_stream = ProcessedVideoStream(raw_stream, input_processors).cache(desc="Reading input stream")
	make_pipeline(config.pipeline).run(video_stream)


if __name__ == "__main__":
	main()