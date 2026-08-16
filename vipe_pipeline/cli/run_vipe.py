import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from vipe.config.parse import parse_typed_config
from vipe.pipeline import make_pipeline
from vipe.streams.base import ProcessedVideoStream, StreamProcessor, VideoFrame
from vipe.streams.frame_dir_stream import FrameDirStream
from vipe.streams.raw_mp4_stream import RawMp4Stream
from vipe.utils.logging import configure_logging

from vipe_pipeline.core.cli import require_new_output


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
	parser = argparse.ArgumentParser(description="Run a bounded NVIDIA ViPE pipeline")
	parser.add_argument("input", type=Path, help="MP4 file or directory of image frames")
	parser.add_argument("--output", type=Path, required=True)
	parser.add_argument("--pipeline", default="static_vda")
	parser.add_argument("--buffer", type=int, default=128)
	parser.add_argument("--image-max-edge", type=int, default=640)
	parser.add_argument(
		"--depth-align-model",
		help="ViPE post-SLAM depth recipe, for example adaptive_unidepth-l_vda; omitted for pose-only output",
	)
	parser.add_argument("--save-slam-map", action="store_true")
	parser.add_argument("--frame-start", type=int, default=0)
	parser.add_argument("--frame-end", type=int, default=-1, help="exclusive end index; -1 processes to the end")
	args = parser.parse_args()
	if args.image_max_edge <= 0:
		parser.error("--image-max-edge must be greater than zero")
	if args.frame_start < 0:
		parser.error("--frame-start must be zero or greater")
	if args.frame_end != -1 and args.frame_end <= args.frame_start:
		parser.error("--frame-end must be greater than --frame-start or -1")
	if not args.input.exists():
		parser.error(f"input does not exist: {args.input}")
	require_new_output(parser, args.output)
	seek_range = range(args.frame_start, args.frame_end)
	if args.input.is_dir():
		raw_stream = FrameDirStream(args.input, seek_range=seek_range)
		input_processors = [ResizeLongestEdgeProcessor(args.image_max_edge)]
	elif args.input.suffix.lower() == ".mp4":
		raw_stream = RawMp4Stream(args.input, seek_range=seek_range)
		input_processors = []
	else:
		parser.error("input must be an MP4 file or a directory of image frames")

	configure_logging()
	depth_align_model = args.depth_align_model if args.depth_align_model is not None else "null"
	config = parse_typed_config(
		"default",
		hydra_args=[
			f"pipeline={args.pipeline}",
			f"pipeline.slam.buffer={args.buffer}",
			f"pipeline.output.path={args.output}",
			f"pipeline.post.depth_align_model={depth_align_model}",
			"pipeline.output.save_artifacts=true",
			f"pipeline.output.save_slam_map={str(args.save_slam_map).lower()}",
			"pipeline.output.save_viz=false",
		],
	)
	video_stream = ProcessedVideoStream(raw_stream, input_processors).cache(desc="Reading input stream")
	make_pipeline(config.pipeline).run(video_stream)


if __name__ == "__main__":
	main()