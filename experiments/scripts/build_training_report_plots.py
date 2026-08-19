from pathlib import Path

import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "experiments" / "training" / "assets"

BACKGROUND = "#f7f7f4"
INK = "#20262e"
MUTED = "#66717f"
ACCENT = "#087f8c"
SECONDARY = "#d17b0f"


def configure() -> None:
	plt.rcParams.update(
		{
			"figure.facecolor": BACKGROUND,
			"axes.facecolor": BACKGROUND,
			"axes.edgecolor": MUTED,
			"axes.labelcolor": INK,
			"axes.titlecolor": INK,
			"font.family": "DejaVu Sans",
			"font.size": 10,
			"text.color": INK,
			"xtick.color": MUTED,
			"ytick.color": MUTED,
		}
	)


def save(figure: plt.Figure, name: str) -> None:
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
	plt.close(figure)


def plot_selected_improvement() -> None:
	labels = ["ViPE initialization", "After 4,000\noptimization steps"]
	values = [6.0109, 12.4749]
	figure, axis = plt.subplots(figsize=(7.2, 4.2))
	bars = axis.bar(labels, values, color=[MUTED, ACCENT], width=0.55)
	axis.bar_label(bars, labels=[f"{value:.2f} dB" for value in values], padding=5, fontweight="bold")
	axis.set_ylim(0, 14)
	axis.set_ylabel("Interleaved holdout PSNR (dB)")
	axis.set_title("Selected 512x384 model: initialization to final reconstruction", loc="left", fontweight="bold")
	axis.grid(axis="y", alpha=0.22)
	axis.spines[["top", "right"]].set_visible(False)
	save(figure, "selected_psnr_improvement.png")


def plot_stitching_comparison() -> None:
	labels = ["Stitched full sequence", "Direct full sequence"]
	values = [12.8029, 13.7235]
	colors = [SECONDARY, ACCENT]
	figure, axis = plt.subplots(figsize=(7.2, 4.5))
	bars = axis.bar(labels, values, color=colors, width=0.72)
	axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=4, fontsize=9)
	axis.set_ylim(12.0, 14.1)
	axis.set_ylabel("Holdout PSNR (dB)")
	axis.set_title("Full-sequence recovery paths at 320x240", loc="left", fontweight="bold")
	axis.text(
		0,
		-0.20,
		"Both runs use all 126 frames, 110 training views, 16 holdouts, and 4,000 iterations.",
		transform=axis.transAxes,
		color=MUTED,
		fontsize=9,
	)
	axis.grid(axis="y", alpha=0.22)
	axis.spines[["top", "right"]].set_visible(False)
	save(figure, "stitching_comparison.png")


def plot_resolution_comparison() -> None:
	series = [
		(
			"Direct ViPE map",
			["320x240", "512x384"],
			[13.7235, 12.3352],
			[158.52, 212.95],
		),
		(
			"Stitched ViPE map",
			["320x240", "640x480", "1280x960"],
			[12.8029, 11.5482, 10.8519],
			[160.32, 253.01, 557.83],
		),
	]
	figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
	for axis, (title, labels, psnr_values, memory_values) in zip(axes, series, strict=True):
		bars = axis.bar(labels, psnr_values, color=ACCENT, width=0.62)
		axis.bar_label(bars, labels=[f"{value:.2f} dB" for value in psnr_values], padding=4, fontsize=9)
		for bar, memory in zip(bars, memory_values, strict=True):
			axis.text(
				bar.get_x() + bar.get_width() / 2,
				10.15,
				f"{memory:.0f} MiB",
				ha="center",
				va="bottom",
				fontsize=8.5,
				color=MUTED,
			)
		axis.set_title(title, fontweight="bold")
		axis.set_ylim(10.0, 14.2)
		axis.grid(axis="y", alpha=0.22)
		axis.spines[["top", "right"]].set_visible(False)
	axes[0].set_ylabel("Holdout PSNR (dB)")
	figure.suptitle("Full-sequence Gaussian render-resolution experiments", x=0.06, ha="left", fontweight="bold")
	figure.text(
		0.06,
		-0.01,
		"All runs use 126 frames, the same 110/16 split, 200k seeds, 4k iterations, and refinement through step 2000. Labels inside bars show peak CUDA allocation.",
		color=MUTED,
		fontsize=8.7,
	)
	save(figure, "resolution_comparison.png")


def plot_512_tradeoff() -> None:
	variants = [
		("Baseline\nstop 2000", 453871, 12.3352, 212.95, (8, -30)),
		("Gray-world\nstop 2000", 454603, 12.3423, 213.05, (8, 8)),
		("Selected\nstop 1000", 314319, 12.4749, 170.29, (8, 18)),
		("Gray-world\nstop 1000", 314063, 12.4683, 169.10, (8, -38)),
		("Threshold 0.002\nstop 1000", 277642, 12.4722, 159.45, (8, 8)),
	]
	figure, axis = plt.subplots(figsize=(8.6, 5.2))
	for label, count, psnr, memory, offset in variants:
		selected = label.startswith("Selected")
		axis.scatter(
			count / 1000,
			psnr,
			s=135 if selected else 85,
			color=ACCENT if selected else MUTED,
			edgecolor=INK if selected else "none",
			zorder=3,
		)
		axis.annotate(
			f"{label}\n{memory:.0f} MiB",
			(count / 1000, psnr),
			xytext=offset,
			textcoords="offset points",
			fontsize=8.5,
			color=INK,
		)
	axis.set_xlim(250, 480)
	axis.set_ylim(12.30, 12.51)
	axis.set_xlabel("Final Gaussian count (thousands)")
	axis.set_ylabel("Original-RGB holdout PSNR (dB)")
	axis.set_title("512x384 tuning: quality, model size, and peak CUDA allocation", loc="left", fontweight="bold")
	axis.grid(alpha=0.22)
	axis.spines[["top", "right"]].set_visible(False)
	save(figure, "tuning_tradeoff.png")


def main() -> None:
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	(OUTPUT_DIR / "approach_comparison.png").unlink(missing_ok=True)
	configure()
	plot_selected_improvement()
	plot_stitching_comparison()
	plot_resolution_comparison()
	plot_512_tradeoff()
	print(f"Wrote training report plots to {OUTPUT_DIR}")


if __name__ == "__main__":
	main()