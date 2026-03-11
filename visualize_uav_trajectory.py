import argparse
import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Entry, Button, OptionMenu
from PIL import Image, ImageTk

# 常量定义
DEFAULT_DIR = os.path.join(os.getcwd(), "images")
DEFAULT_FILE = "default.png"
DISPLAY_HEIGHT_RATIO = 1.8
DEFAULT_PARAMS = {
    "sample_interval": 10,
    "diff_threshold": 30,
    "kernel_size": 15,
    "start_time": 0.0,
    "end_time": 10.0,
    "alpha_start": 0.2,
    "alpha_end": 1.0,
}
MAX_BACKGROUND_FRAMES = 30
MIN_CONTOUR_AREA = 12
MIN_TRACK_POINTS = 8
EXPECTED_TRACK_Y_RATIO = 0.49
IMPROVED_LINE_STYLES = {
    "white_orange": {
        "label": "亮白+橙晕",
        "main": (245, 245, 245),
        "glow": (0, 150, 255),
    },
    "bright_yellow": {
        "label": "亮黄",
        "main": (0, 255, 255),
        "glow": (0, 210, 255),
    },
    "bright_pink": {
        "label": "亮粉",
        "main": (255, 160, 255),
        "glow": (255, 60, 255),
    },
    "bright_cyan": {
        "label": "亮青",
        "main": (255, 255, 0),
        "glow": (255, 180, 0),
    },
}


def get_video_segment_info(video_path, start_time, end_time):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}！请检查路径。")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total_frames / fps if fps > 0 else 0.0
    cap.release()

    start_time = max(0.0, start_time)
    effective_end = duration if end_time <= 0 else min(end_time, duration)
    if effective_end <= start_time:
        raise ValueError("结束时间必须大于起始时间。")

    return fps, duration, start_time, effective_end


def collect_segment_frames(cap, start_time, end_time):
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time >= end_time:
            break
        frames.append(frame)

    return frames


def create_background_from_frames(frames):
    if not frames:
        return None

    background_stack = np.stack(frames[:MAX_BACKGROUND_FRAMES], axis=0)
    background = np.median(background_stack, axis=0).astype(np.uint8)
    return background


def normalize_progress(index, total_steps):
    if total_steps <= 1:
        return 1.0
    return index / (total_steps - 1)


def alpha_from_progress(alpha_start, alpha_end, progress):
    return alpha_start + (alpha_end - alpha_start) * progress


def ensure_odd(value):
    return value if value % 2 == 1 else value + 1


def overlay_drone_trajectory_legacy(
    video_path,
    sample_interval,
    diff_threshold,
    kernel_size,
    start_time,
    end_time,
    alpha_start,
    alpha_end,
    notifier=None,
):
    reporter = notifier or print
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        reporter(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    _, _, start_time, end_time = get_video_segment_info(video_path, start_time, end_time)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    ret, first_frame = cap.read()
    if not ret:
        reporter("无法读取视频的第一帧！")
        cap.release()
        return None

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background_image = first_frame.copy().astype(np.float32)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    sampled_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time >= end_time:
            break

        if sampled_index % sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, first_gray)
            _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            dilated_mask = cv2.dilate(diff_mask, kernel, iterations=1)
            motion_only = cv2.bitwise_and(frame, frame, mask=dilated_mask).astype(np.float32)

            progress = normalize_progress(current_time - start_time, end_time - start_time)
            alpha = alpha_from_progress(alpha_start, alpha_end, progress)
            mask = dilated_mask == 255
            background_image[mask] = background_image[mask] * (1 - alpha) + motion_only[mask] * alpha

        sampled_index += 1

    cap.release()
    cv2.destroyAllWindows()
    return np.clip(background_image, 0, 255).astype(np.uint8)


def detect_motion_mask(background_gray, gray_frame, diff_threshold, kernel_size):
    blur_size = ensure_odd(max(3, kernel_size // 2))
    background_blur = cv2.GaussianBlur(background_gray, (blur_size, blur_size), 0)
    frame_blur = cv2.GaussianBlur(gray_frame, (blur_size, blur_size), 0)
    frame_diff = cv2.absdiff(frame_blur, background_blur)
    _, diff_mask = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    morph_size = max(3, kernel_size // 3)
    morph_kernel = np.ones((morph_size, morph_size), np.uint8)
    cleaned = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, morph_kernel, iterations=2)
    return cleaned


def detect_colored_motion_mask(previous_frame, current_frame, diff_threshold, kernel_size):
    previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    blur_size = ensure_odd(max(5, kernel_size // 2))
    diff = cv2.absdiff(
        cv2.GaussianBlur(current_gray, (blur_size, blur_size), 0),
        cv2.GaussianBlur(previous_gray, (blur_size, blur_size), 0),
    )
    motion_threshold = max(12, diff_threshold - 10)
    _, motion_mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)

    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    saturation_mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    color_motion_mask = cv2.bitwise_and(motion_mask, saturation_mask)

    morph_size = max(3, kernel_size // 4)
    morph_kernel = np.ones((morph_size, morph_size), np.uint8)
    cleaned = cv2.morphologyEx(color_motion_mask, cv2.MORPH_OPEN, morph_kernel, iterations=1)
    cleaned = cv2.dilate(cleaned, morph_kernel, iterations=2)
    return cleaned


def score_tracking_candidate(candidate, previous_center, previous_dx):
    area = candidate["area"]
    center_x, center_y = candidate["center"]

    if previous_center is None:
        return area - center_x * 0.04 - abs(center_y - candidate["expected_y"]) * 2.5

    dx = center_x - previous_center[0]
    dy = center_y - previous_center[1]
    forward_penalty = max(0, -dx) * 10
    speed_penalty = abs(dx - previous_dx) * 1.8
    vertical_penalty = abs(dy) * 4
    return area * 2.2 - forward_penalty - speed_penalty - vertical_penalty


def find_tracking_candidate(mask, frame_shape, previous_center, previous_dx):
    height, width = frame_shape[:2]
    expected_y = int(height * EXPECTED_TRACK_Y_RATIO)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 8 or area > 1200:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2

        if previous_center is None:
            if x > width * 0.35 or y < height * 0.43 or y > height * 0.8:
                continue
        else:
            search_left = previous_center[0] - 30
            search_right = previous_center[0] + max(140, previous_dx * 3)
            search_top = previous_center[1] - 24
            search_bottom = previous_center[1] + 24
            if not (search_left <= center_x <= search_right and search_top <= center_y <= search_bottom):
                continue

        candidate = {
            "bbox": (x, y, w, h),
            "center": (center_x, center_y),
            "area": area,
            "expected_y": expected_y if previous_center is None else previous_center[1],
        }
        candidate["score"] = score_tracking_candidate(candidate, previous_center, previous_dx)
        candidates.append(candidate)

    if not candidates:
        return None

    return max(candidates, key=lambda candidate: candidate["score"])


def trim_unstable_points(points):
    if len(points) < 4:
        return points

    for index in range(len(points) - 3):
        x_values = [point[1] for point in points[index:index + 4]]
        if x_values == sorted(x_values) and x_values[-1] - x_values[0] > 15:
            return points[index:]

    return points


def interpolate_missing_points(points, sample_interval):
    if len(points) < 2:
        return points

    interpolated = [points[0]]
    for previous_point, current_point in zip(points, points[1:]):
        previous_frame, previous_x, previous_y = previous_point
        current_frame, current_x, current_y = current_point
        gap = current_frame - previous_frame
        if gap > sample_interval:
            steps = gap // sample_interval
            for step in range(1, steps):
                ratio = step / steps
                interpolated.append(
                    (
                        previous_frame + step * sample_interval,
                        int(previous_x + (current_x - previous_x) * ratio),
                        int(previous_y + (current_y - previous_y) * ratio),
                    )
                )
        interpolated.append(current_point)
    return interpolated


def smooth_trajectory_points(points):
    if len(points) < 3:
        return points

    smoothed = []
    last_x = None
    for index, (frame_index, center_x, center_y) in enumerate(points):
        window = points[max(0, index - 2):min(len(points), index + 3)]
        median_x = int(np.median([point[1] for point in window]))
        median_y = int(np.median([point[2] for point in window]))
        if last_x is not None:
            median_x = max(last_x + 1, median_x)
        smoothed.append((frame_index, median_x, median_y))
        last_x = median_x
    return smoothed


def densify_trajectory_points(points, segments=6):
    if len(points) < 2:
        return points

    dense_points = []
    for index in range(len(points) - 1):
        frame_index, start_x, start_y = points[index]
        next_frame_index, end_x, end_y = points[index + 1]
        prev_x, prev_y = (points[index - 1][1], points[index - 1][2]) if index > 0 else (start_x, start_y)
        next2_x, next2_y = (
            (points[index + 2][1], points[index + 2][2]) if index + 2 < len(points) else (end_x, end_y)
        )

        if index == 0:
            dense_points.append((frame_index, start_x, start_y))

        for step in range(1, segments + 1):
            t = step / segments
            t2 = t * t
            t3 = t2 * t

            x = 0.5 * (
                (2 * start_x)
                + (-prev_x + end_x) * t
                + (2 * prev_x - 5 * start_x + 4 * end_x - next2_x) * t2
                + (-prev_x + 3 * start_x - 3 * end_x + next2_x) * t3
            )
            y = 0.5 * (
                (2 * start_y)
                + (-prev_y + end_y) * t
                + (2 * prev_y - 5 * start_y + 4 * end_y - next2_y) * t2
                + (-prev_y + 3 * start_y - 3 * end_y + next2_y) * t3
            )
            interpolated_frame = int(frame_index + (next_frame_index - frame_index) * t)
            dense_points.append((interpolated_frame, int(x), int(y)))

    return dense_points


def track_drone_trajectory(frames, sample_interval, diff_threshold, kernel_size):
    tracked_points = []
    previous_center = None
    previous_dx = 25

    for frame_index in range(sample_interval, len(frames), sample_interval):
        previous_frame = frames[frame_index - sample_interval]
        current_frame = frames[frame_index]
        motion_mask = detect_colored_motion_mask(previous_frame, current_frame, diff_threshold, kernel_size)
        candidate = find_tracking_candidate(motion_mask, current_frame.shape, previous_center, previous_dx)
        if candidate is None:
            continue

        center_x, center_y = candidate["center"]
        if previous_center is not None:
            previous_dx = max(5, center_x - previous_center[0])
        previous_center = (center_x, center_y)
        tracked_points.append((frame_index, center_x, center_y))

    tracked_points = trim_unstable_points(tracked_points)
    tracked_points = interpolate_missing_points(tracked_points, sample_interval)
    tracked_points = smooth_trajectory_points(tracked_points)
    return tracked_points


def overlay_drone_trajectory_improved(
    video_path,
    sample_interval,
    diff_threshold,
    kernel_size,
    start_time,
    end_time,
    alpha_start,
    alpha_end,
    line_style="white_orange",
    notifier=None,
):
    reporter = notifier or print
    fps, _, start_time, end_time = get_video_segment_info(video_path, start_time, end_time)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        reporter(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    frames = collect_segment_frames(cap, start_time, end_time)
    cap.release()
    if not frames:
        reporter("指定时间段内没有可处理的视频帧。")
        return None

    legacy_base = overlay_drone_trajectory_legacy(
        video_path,
        sample_interval,
        diff_threshold,
        kernel_size,
        start_time,
        end_time,
        alpha_start,
        alpha_end,
        notifier=notifier,
    )
    if legacy_base is None:
        return None

    background = create_background_from_frames(frames)
    overlay_image = legacy_base.astype(np.float32)
    tracked_points = track_drone_trajectory(frames, sample_interval, diff_threshold, kernel_size)
    if len(tracked_points) < MIN_TRACK_POINTS:
        reporter("改进方法未能稳定跟踪到足够多的轨迹点。")
        return None
    dense_points = densify_trajectory_points(tracked_points, segments=6)

    total_points = max(1, len(dense_points))
    glow_layer = np.zeros_like(overlay_image)
    line_points = [(point[1], point[2]) for point in dense_points]

    style = IMPROVED_LINE_STYLES.get(line_style, IMPROVED_LINE_STYLES["white_orange"])
    cyan_line_color = style["main"]
    glow_line_color = style["glow"]

    for index, (center_x, center_y) in enumerate(line_points):
        progress = normalize_progress(index, total_points)
        alpha = alpha_from_progress(alpha_start, alpha_end, progress)
        point_color = cyan_line_color

        if index >= 1:
            previous_point = line_points[index - 1]
            glow_thickness = max(10, kernel_size + 2)
            thickness = max(7, kernel_size // 2 + 2)
            cv2.line(
                glow_layer,
                previous_point,
                (center_x, center_y),
                glow_line_color,
                glow_thickness,
                lineType=cv2.LINE_AA,
            )
            line_color = tuple(int(channel * max(0.88, alpha)) for channel in point_color)
            cv2.line(
                overlay_image,
                previous_point,
                (center_x, center_y),
                line_color,
                thickness,
                lineType=cv2.LINE_AA,
            )

    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=3.2, sigmaY=3.2)
    overlay_image = cv2.addWeighted(overlay_image, 1.0, glow_layer, 0.3, 0)

    result = cv2.addWeighted(background.astype(np.float32), 0.15, overlay_image, 0.85, 0)
    summary = result.astype(np.uint8)
    summary = cv2.putText(
        summary,
        f"hybrid tracked_points={len(tracked_points)}  fps={fps:.1f}",
        (20, summary.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return summary


def overlay_drone_trajectory(
    video_path,
    sample_interval,
    diff_threshold,
    kernel_size,
    start_time,
    end_time,
    alpha_start,
    alpha_end,
    method="improved",
    line_style="white_orange",
    notifier=None,
):
    if method == "legacy":
        return overlay_drone_trajectory_legacy(
            video_path,
            sample_interval,
            diff_threshold,
            kernel_size,
            start_time,
            end_time,
            alpha_start,
            alpha_end,
            notifier=notifier,
        )

    return overlay_drone_trajectory_improved(
        video_path,
        sample_interval,
        diff_threshold,
        kernel_size,
        start_time,
        end_time,
        alpha_start,
        alpha_end,
        line_style=line_style,
        notifier=notifier,
    )


def render_image_for_display(result_image, default_height):
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(result_image_rgb)
    display_height = int(default_height / DISPLAY_HEIGHT_RATIO)
    aspect_ratio = img.width / img.height
    display_width = int(display_height * aspect_ratio)
    display_img = img.resize((display_width, display_height), Image.LANCZOS)
    return ImageTk.PhotoImage(image=display_img)


def parse_ui_values():
    try:
        return {
            "method": method_var.get(),
            "line_style": line_style_var.get(),
            "sample_interval": int(sample_interval_entry.get()),
            "diff_threshold": int(diff_threshold_entry.get()),
            "kernel_size": int(kernel_size_entry.get()),
            "start_time": float(start_time_entry.get()),
            "end_time": float(end_time_entry.get()),
            "alpha_start": float(alpha_start_entry.get()),
            "alpha_end": float(alpha_end_entry.get()),
        }
    except ValueError:
        set_status("请输入有效的数值！", is_error=True)
        return None


def update_image():
    params = parse_ui_values()
    if params is None:
        return

    if not video_path:
        set_status("请先选择视频文件。", is_error=True)
        return

    set_status("处理中...")
    result_image = overlay_drone_trajectory(video_path, notifier=show_gui_message, **params)
    if result_image is not None:
        imgtk = render_image_for_display(result_image, default_height)
        result_label.config(image=imgtk)
        result_label.image = imgtk

        global saved_image
        saved_image = result_image
        set_status("处理完成。")


def save_image():
    if saved_image is not None:
        if not os.path.exists(DEFAULT_DIR):
            os.makedirs(DEFAULT_DIR)
        file_path = filedialog.asksaveasfilename(
            initialdir=DEFAULT_DIR,
            initialfile=DEFAULT_FILE,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if file_path:
            cv2.imwrite(file_path, saved_image)
            set_status(f"图像已保存到: {file_path}")


def select_file():
    global video_path
    file_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
    if file_path:
        video_path = file_path
        file_label.config(text=f"已选择文件: {file_path}")
        set_status("视频已选择。")


def set_status(message, is_error=False):
    if "status_label" in globals():
        status_label.config(text=message, fg="crimson" if is_error else "black")


def show_gui_message(message):
    set_status(message, is_error=True)


def create_comparison_image(legacy_image, improved_image):
    separator_width = 24
    height = max(legacy_image.shape[0], improved_image.shape[0])

    def pad_image(image):
        if image.shape[0] == height:
            return image
        pad_rows = height - image.shape[0]
        return cv2.copyMakeBorder(image, 0, pad_rows, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    legacy_padded = pad_image(legacy_image)
    improved_padded = pad_image(improved_image)
    separator = np.full((height, separator_width, 3), 20, dtype=np.uint8)
    comparison = np.hstack([legacy_padded, separator, improved_padded])

    cv2.putText(comparison, "Legacy: motion overlay", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    right_offset = legacy_padded.shape[1] + separator_width + 20
    cv2.putText(comparison, "Improved: hybrid overlay + track", (right_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return comparison


def save_cli_result(result_image, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, result_image)
    print(f"结果已保存到: {output_path}")


def run_cli(args):
    params = {
        "sample_interval": args.sample_interval,
        "diff_threshold": args.diff_threshold,
        "kernel_size": args.kernel_size,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "alpha_start": args.alpha_start,
        "alpha_end": args.alpha_end,
        "line_style": args.line_style,
    }

    if args.compare:
        legacy_image = overlay_drone_trajectory(args.video, method="legacy", **params)
        improved_image = overlay_drone_trajectory(args.video, method="improved", **params)
        if legacy_image is None or improved_image is None:
            raise SystemExit(1)
        comparison = create_comparison_image(legacy_image, improved_image)
        save_cli_result(comparison, args.output)
        return

    result = overlay_drone_trajectory(args.video, method=args.method, **params)
    if result is None:
        raise SystemExit(1)
    save_cli_result(result, args.output)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Visualize UAV trajectory from video.")
    parser.add_argument("--video", help="Path to the input video.")
    parser.add_argument("--output", default=os.path.join(DEFAULT_DIR, DEFAULT_FILE), help="Path to the output image.")
    parser.add_argument("--method", choices=["legacy", "improved"], default="improved", help="Trajectory generation method.")
    parser.add_argument("--compare", action="store_true", help="Export a side-by-side legacy/improved comparison image.")
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_PARAMS["sample_interval"])
    parser.add_argument("--diff-threshold", type=int, default=DEFAULT_PARAMS["diff_threshold"])
    parser.add_argument("--kernel-size", type=int, default=DEFAULT_PARAMS["kernel_size"])
    parser.add_argument("--start-time", type=float, default=DEFAULT_PARAMS["start_time"])
    parser.add_argument("--end-time", type=float, default=DEFAULT_PARAMS["end_time"])
    parser.add_argument("--alpha-start", type=float, default=DEFAULT_PARAMS["alpha_start"])
    parser.add_argument("--alpha-end", type=float, default=DEFAULT_PARAMS["alpha_end"])
    parser.add_argument("--line-style", choices=list(IMPROVED_LINE_STYLES.keys()), default="white_orange")
    return parser


def run_gui():
    global video_path
    global saved_image
    global root
    global default_height
    global file_label
    global sample_interval_entry
    global diff_threshold_entry
    global kernel_size_entry
    global start_time_entry
    global end_time_entry
    global alpha_start_entry
    global alpha_end_entry
    global result_label
    global method_var
    global line_style_var
    global status_label

    video_path = ""
    saved_image = None

    root = tk.Tk()
    root.title("无人机轨迹参数调整")
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    default_width = int(screen_width * 0.5)
    default_height = int(screen_height * 0.7)
    root.geometry(f"{default_width}x{default_height}")

    file_label = Label(root, text="未选择文件", wraplength=default_width)
    file_label.pack()
    file_button = Button(root, text="选择视频文件", command=select_file)
    file_button.pack()

    Label(root, text="模式").pack()
    method_var = tk.StringVar(value="legacy")
    method_menu = OptionMenu(root, method_var, "legacy", "improved")
    method_menu.pack()

    Label(root, text="改进线条颜色").pack()
    line_style_var = tk.StringVar(value="white_orange")
    line_style_menu = OptionMenu(root, line_style_var, *IMPROVED_LINE_STYLES.keys())
    line_style_menu.pack()

    Label(root, text="采样间隔").pack()
    sample_interval_entry = Entry(root)
    sample_interval_entry.pack()
    sample_interval_entry.insert(0, str(DEFAULT_PARAMS["sample_interval"]))

    Label(root, text="差分阈值").pack()
    diff_threshold_entry = Entry(root)
    diff_threshold_entry.pack()
    diff_threshold_entry.insert(0, str(DEFAULT_PARAMS["diff_threshold"]))

    Label(root, text="膨胀核大小").pack()
    kernel_size_entry = Entry(root)
    kernel_size_entry.pack()
    kernel_size_entry.insert(0, str(DEFAULT_PARAMS["kernel_size"]))

    Label(root, text="视频起始时间（秒）").pack()
    start_time_entry = Entry(root)
    start_time_entry.pack()
    start_time_entry.insert(0, str(DEFAULT_PARAMS["start_time"]))

    Label(root, text="视频结束时间（秒）").pack()
    end_time_entry = Entry(root)
    end_time_entry.pack()
    end_time_entry.insert(0, str(DEFAULT_PARAMS["end_time"]))

    Label(root, text="透明度起始值").pack()
    alpha_start_entry = Entry(root)
    alpha_start_entry.pack()
    alpha_start_entry.insert(0, str(DEFAULT_PARAMS["alpha_start"]))

    Label(root, text="透明度最终值").pack()
    alpha_end_entry = Entry(root)
    alpha_end_entry.pack()
    alpha_end_entry.insert(0, str(DEFAULT_PARAMS["alpha_end"]))

    update_button = Button(root, text="更新图像", command=update_image)
    update_button.pack()
    save_button = Button(root, text="保存图片", command=save_image)
    save_button.pack()

    status_label = Label(root, text="请选择视频并设置参数。", wraplength=default_width)
    status_label.pack()

    result_label = Label(root)
    result_label.pack()

    root.mainloop()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.video:
        run_cli(args)
        return
    run_gui()


if __name__ == "__main__":
    main()
