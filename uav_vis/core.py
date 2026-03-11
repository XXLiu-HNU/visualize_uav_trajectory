from dataclasses import dataclass
import os

import cv2
import numpy as np

from .config import (
    DEFAULT_DIR,
    DEFAULT_FILE,
    DEFAULT_PARAMS,
    EXPECTED_TRACK_Y_RATIO,
    IMPROVED_LINE_STYLES,
    MAX_BACKGROUND_FRAMES,
    MIN_TRACK_POINTS,
)


@dataclass
class RenderParams:
    sample_interval: int = DEFAULT_PARAMS["sample_interval"]
    diff_threshold: int = DEFAULT_PARAMS["diff_threshold"]
    kernel_size: int = DEFAULT_PARAMS["kernel_size"]
    start_time: float = DEFAULT_PARAMS["start_time"]
    end_time: float = DEFAULT_PARAMS["end_time"]
    alpha_start: float = DEFAULT_PARAMS["alpha_start"]
    alpha_end: float = DEFAULT_PARAMS["alpha_end"]
    line_style: str = DEFAULT_PARAMS["line_style"]
    line_thickness_scale: float = DEFAULT_PARAMS["line_thickness_scale"]
    glow_strength: float = DEFAULT_PARAMS["glow_strength"]


def ensure_odd(value):
    return value if value % 2 == 1 else value + 1


def normalize_progress(index, total_steps):
    if total_steps <= 1:
        return 1.0
    return index / (total_steps - 1)


def alpha_from_progress(alpha_start, alpha_end, progress):
    return alpha_start + (alpha_end - alpha_start) * progress


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
    return np.median(background_stack, axis=0).astype(np.uint8)


def overlay_drone_trajectory_legacy(video_path, params: RenderParams, notifier=None):
    reporter = notifier or print
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        reporter(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    _, _, start_time, end_time = get_video_segment_info(video_path, params.start_time, params.end_time)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    ret, first_frame = cap.read()
    if not ret:
        reporter("无法读取视频的第一帧！")
        cap.release()
        return None

    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    background_image = first_frame.copy().astype(np.float32)
    kernel = np.ones((params.kernel_size, params.kernel_size), np.uint8)
    sampled_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if current_time >= end_time:
            break

        if sampled_index % params.sample_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(gray_frame, first_gray)
            _, diff_mask = cv2.threshold(frame_diff, params.diff_threshold, 255, cv2.THRESH_BINARY)
            dilated_mask = cv2.dilate(diff_mask, kernel, iterations=1)
            motion_only = cv2.bitwise_and(frame, frame, mask=dilated_mask).astype(np.float32)

            progress = normalize_progress(current_time - start_time, end_time - start_time)
            alpha = alpha_from_progress(params.alpha_start, params.alpha_end, progress)
            mask = dilated_mask == 255
            background_image[mask] = background_image[mask] * (1 - alpha) + motion_only[mask] * alpha

        sampled_index += 1

    cap.release()
    return np.clip(background_image, 0, 255).astype(np.uint8)


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
    return area * 2.2 - max(0, -dx) * 10 - abs(dx - previous_dx) * 1.8 - abs(dy) * 4


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

    return max(candidates, key=lambda candidate: candidate["score"]) if candidates else None


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


def track_drone_trajectory(frames, params: RenderParams):
    tracked_points = []
    previous_center = None
    previous_dx = 25
    for frame_index in range(params.sample_interval, len(frames), params.sample_interval):
        previous_frame = frames[frame_index - params.sample_interval]
        current_frame = frames[frame_index]
        motion_mask = detect_colored_motion_mask(previous_frame, current_frame, params.diff_threshold, params.kernel_size)
        candidate = find_tracking_candidate(motion_mask, current_frame.shape, previous_center, previous_dx)
        if candidate is None:
            continue
        center_x, center_y = candidate["center"]
        if previous_center is not None:
            previous_dx = max(5, center_x - previous_center[0])
        previous_center = (center_x, center_y)
        tracked_points.append((frame_index, center_x, center_y))
    tracked_points = trim_unstable_points(tracked_points)
    tracked_points = interpolate_missing_points(tracked_points, params.sample_interval)
    return smooth_trajectory_points(tracked_points)


def get_improved_failure_message():
    return "改进模式未能稳定跟踪到足够多的轨迹点。可尝试减小采样间隔、缩短时间段，或切换回原始模式。"


def overlay_drone_trajectory_improved(video_path, params: RenderParams, notifier=None):
    reporter = notifier or print
    fps, _, start_time, end_time = get_video_segment_info(video_path, params.start_time, params.end_time)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        reporter(f"无法打开视频文件: {video_path}！请检查路径。")
        return None

    frames = collect_segment_frames(cap, start_time, end_time)
    cap.release()
    if not frames:
        reporter("指定时间段内没有可处理的视频帧。")
        return None

    legacy_base = overlay_drone_trajectory_legacy(video_path, params, notifier=notifier)
    if legacy_base is None:
        return None

    background = create_background_from_frames(frames)
    overlay_image = legacy_base.astype(np.float32)
    tracked_points = track_drone_trajectory(frames, params)
    if len(tracked_points) < MIN_TRACK_POINTS:
        reporter(get_improved_failure_message())
        return None
    dense_points = densify_trajectory_points(tracked_points, segments=6)

    total_points = max(1, len(dense_points))
    glow_layer = np.zeros_like(overlay_image)
    track_layer = np.zeros_like(overlay_image)
    line_points = [(point[1], point[2]) for point in dense_points]
    style = IMPROVED_LINE_STYLES.get(params.line_style, IMPROVED_LINE_STYLES["white_orange"])

    for index, (center_x, center_y) in enumerate(line_points):
        if index >= 1:
            previous_point = line_points[index - 1]
            glow_thickness = max(10, int((params.kernel_size + 2) * params.line_thickness_scale))
            thickness = max(5, int((params.kernel_size // 2 + 2) * params.line_thickness_scale))
            cv2.line(glow_layer, previous_point, (center_x, center_y), style["glow"], glow_thickness, lineType=cv2.LINE_AA)
            cv2.line(track_layer, previous_point, (center_x, center_y), style["main"], thickness, lineType=cv2.LINE_AA)

    glow_layer = cv2.GaussianBlur(glow_layer, (0, 0), sigmaX=3.2, sigmaY=3.2)
    overlay_image = cv2.addWeighted(overlay_image, 1.0, glow_layer, params.glow_strength, 0)
    track_mask = np.any(track_layer > 0, axis=2)
    if np.any(track_mask):
        overlay_image[track_mask] = overlay_image[track_mask] * 0.1 + track_layer[track_mask] * 0.9
    result = cv2.addWeighted(background.astype(np.float32), 0.15, overlay_image, 0.85, 0)
    summary = result.astype(np.uint8)
    cv2.putText(
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


def overlay_drone_trajectory(video_path, params: RenderParams, method="improved", notifier=None):
    if method == "legacy":
        return overlay_drone_trajectory_legacy(video_path, params, notifier=notifier)
    return overlay_drone_trajectory_improved(video_path, params, notifier=notifier)


def create_comparison_image(legacy_image, improved_image):
    separator_width = 24
    height = max(legacy_image.shape[0], improved_image.shape[0])

    def pad_image(image):
        if image.shape[0] == height:
            return image
        return cv2.copyMakeBorder(image, 0, height - image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))

    legacy_padded = pad_image(legacy_image)
    improved_padded = pad_image(improved_image)
    separator = np.full((height, separator_width, 3), 20, dtype=np.uint8)
    comparison = np.hstack([legacy_padded, separator, improved_padded])
    cv2.putText(comparison, "Legacy: motion overlay", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    right_offset = legacy_padded.shape[1] + separator_width + 20
    cv2.putText(comparison, "Improved: hybrid overlay + track", (right_offset, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return comparison


def save_result_image(result_image, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, result_image)
