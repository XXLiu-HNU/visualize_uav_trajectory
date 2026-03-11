import argparse

from .config import DEFAULT_DIR, DEFAULT_FILE, DEFAULT_PARAMS, IMPROVED_LINE_STYLES
from .core import RenderParams, create_comparison_image, overlay_drone_trajectory, save_result_image
from .gui import run_gui


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize UAV trajectory from video.")
    parser.add_argument("--video", help="Path to the input video.")
    parser.add_argument("--output", default=f"{DEFAULT_DIR}/{DEFAULT_FILE}", help="Path to the output image.")
    parser.add_argument("--method", choices=["legacy", "improved"], default="improved", help="Trajectory generation method.")
    parser.add_argument("--compare", action="store_true", help="Export a side-by-side legacy/improved comparison image.")
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_PARAMS["sample_interval"])
    parser.add_argument("--diff-threshold", type=int, default=DEFAULT_PARAMS["diff_threshold"])
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_PARAMS["kernel_size"])
    parser.add_argument("--start-time", type=float, default=DEFAULT_PARAMS["start_time"])
    parser.add_argument("--end-time", type=float, default=DEFAULT_PARAMS["end_time"])
    parser.add_argument("--alpha-start", type=float, default=DEFAULT_PARAMS["alpha_start"])
    parser.add_argument("--alpha-end", type=float, default=DEFAULT_PARAMS["alpha_end"])
    parser.add_argument("--line-style", choices=list(IMPROVED_LINE_STYLES.keys()), default=DEFAULT_PARAMS["line_style"])
    parser.add_argument("--line-thickness-scale", type=float, default=DEFAULT_PARAMS["line_thickness_scale"])
    parser.add_argument("--glow-strength", type=float, default=DEFAULT_PARAMS["glow_strength"])
    return parser


def params_from_args(args):
    return RenderParams(
        sample_interval=args.sample_interval,
        diff_threshold=args.diff_threshold,
        kernel_size=args.kernel_size,
        start_time=args.start_time,
        end_time=args.end_time,
        alpha_start=args.alpha_start,
        alpha_end=args.alpha_end,
        line_style=args.line_style,
        line_thickness_scale=args.line_thickness_scale,
        glow_strength=args.glow_strength,
    )


def run_cli(args):
    params = params_from_args(args)
    if args.compare:
        legacy_image = overlay_drone_trajectory(args.video, params, method="legacy")
        improved_image = overlay_drone_trajectory(args.video, params, method="improved")
        if legacy_image is None or improved_image is None:
            raise SystemExit(1)
        save_result_image(create_comparison_image(legacy_image, improved_image), args.output)
        print(f"结果已保存到: {args.output}")
        return

    result = overlay_drone_trajectory(args.video, params, method=args.method)
    if result is None:
        raise SystemExit(1)
    save_result_image(result, args.output)
    print(f"结果已保存到: {args.output}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.video:
        run_cli(args)
        return
    run_gui()

