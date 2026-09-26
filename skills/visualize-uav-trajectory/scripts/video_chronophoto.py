#!/usr/bin/env python3
"""Portable source-pixel video chronophotography helper."""
import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


def fail(message):
    raise ValueError(message)


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        fail(f"{name} must be a finite number")
    return float(value)


def write_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write image: {path.name}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if not path.is_file() or path.stat().st_size == 0:
        raise OSError(f"could not write JSON: {path.name}")


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        fail(f"cannot open video: {path.name}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0 or frames <= 0:
        cap.release()
        fail("video has unusable fps, dimensions, or frame count")
    return cap, {"fps": fps, "width": width, "height": height, "frame_count": frames,
                 "duration": (frames - 1) / fps}


def frame_at(cap, fps, frame_count, time, label):
    index = int(round(time * fps))
    if index < 0 or index >= frame_count:
        fail(f"{label} maps outside the decodable video")
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok or frame is None:
        fail(f"cannot decode {label} frame {index}")
    return frame, index


def normalized_polygon(value, name):
    if not isinstance(value, list) or len(value) < 3:
        fail(f"{name} must have at least 3 points")
    points = []
    for i, point in enumerate(value):
        if not isinstance(point, list) or len(point) != 2:
            fail(f"{name}[{i}] must be [x, y]")
        x, y = finite_number(point[0], f"{name}[{i}][0]"), finite_number(point[1], f"{name}[{i}][1]")
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            fail(f"{name} points must be within [0, 1]")
        points.append([x, y])
    area = abs(sum(points[i][0] * points[(i + 1) % len(points)][1] -
                   points[(i + 1) % len(points)][0] * points[i][1] for i in range(len(points)))) / 2
    if area <= 1e-9:
        fail(f"{name} must have non-zero area")
    return points


def parse_keyframe(value, i, width, height):
    if not isinstance(value, dict):
        fail(f"keyframes[{i}] must be an object")
    allowed = {"time", "box", "opacity", "manual_polygon", "include_polygons", "exclude_polygons"}
    unknown = set(value) - allowed
    if unknown or "time" not in value or "box" not in value:
        fail(f"keyframes[{i}] has invalid schema")
    time = finite_number(value["time"], f"keyframes[{i}].time")
    box = value["box"]
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, int) for x in box):
        fail(f"keyframes[{i}].box must contain four integers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height and x2 - x1 >= 3 and y2 - y1 >= 3):
        fail(f"keyframes[{i}].box is invalid or smaller than 3 pixels")
    item = {"time": time, "box": box}
    if "opacity" in value:
        opacity = finite_number(value["opacity"], f"keyframes[{i}].opacity")
        if not 0 <= opacity <= 1:
            fail(f"keyframes[{i}].opacity must be in [0, 1]")
        item["opacity"] = opacity
    for key in ("manual_polygon",):
        if key in value:
            item[key] = normalized_polygon(value[key], f"keyframes[{i}].{key}")
    for key in ("include_polygons", "exclude_polygons"):
        if key in value:
            if not isinstance(value[key], list):
                fail(f"keyframes[{i}].{key} must be a list")
            item[key] = [normalized_polygon(polygon, f"keyframes[{i}].{key}[{j}]")
                         for j, polygon in enumerate(value[key])]
    return item


def validate_config(value, info):
    if not isinstance(value, dict) or set(value) != {"terminal_time", "keyframes"}:
        fail("config schema requires exactly terminal_time and keyframes")
    terminal = finite_number(value["terminal_time"], "terminal_time")
    if terminal <= 0 or terminal > info["duration"]:
        fail("terminal_time is outside the video time range")
    if not isinstance(value["keyframes"], list) or not value["keyframes"]:
        fail("keyframes must be a non-empty list")
    entries = [parse_keyframe(item, i, info["width"], info["height"])
               for i, item in enumerate(value["keyframes"])]
    previous_time, indices = -1.0, set()
    terminal_index = int(round(terminal * info["fps"]))
    for i, entry in enumerate(entries):
        if not 0 <= entry["time"] < terminal or entry["time"] <= previous_time:
            fail("keyframe times must be strictly increasing and before terminal_time")
        index = int(round(entry["time"] * info["fps"]))
        if index in indices:
            fail("keyframes map to the same frame")
        if index >= terminal_index:
            fail("keyframe actual frame must precede terminal frame")
        indices.add(index)
        previous_time = entry["time"]
    return {"terminal_time": terminal, "keyframes": entries}


def polygon_mask(shape, polygons):
    height, width = shape
    result = np.zeros(shape, np.uint8)
    for polygon in polygons:
        points = np.rint(np.asarray(polygon) * [width - 1, height - 1]).astype(np.int32)
        cv2.fillPoly(result, [points], 255)
    return result


def segmentation(crop, entry):
    if "manual_polygon" in entry:
        mask = polygon_mask(crop.shape[:2], [entry["manual_polygon"]])
        method = "manual_polygon"
    else:
        height, width = crop.shape[:2]
        labels = np.zeros((height, width), np.uint8)
        rect = (1, 1, width - 2, height - 2)
        try:
            cv2.setRNGSeed(0)
            cv2.grabCut(crop, labels, rect, np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_RECT)
        except cv2.error as error:
            fail(f"GrabCut failed; provide manual_polygon: {error}")
        mask = np.where((labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        method = "grabcut"
    if entry.get("include_polygons"):
        mask[polygon_mask(crop.shape[:2], entry["include_polygons"]) > 0] = 255
    if entry.get("exclude_polygons"):
        mask[polygon_mask(crop.shape[:2], entry["exclude_polygons"]) > 0] = 0
    if not np.any(mask):
        fail("foreground mask is empty")
    return mask, method


def thumbnail(image, time, index, max_width=320):
    """Make one labelled, bounded-memory preview without touching source pixels."""
    height, width = image.shape[:2]
    if width > max_width:
        height = max(1, round(height * max_width / width))
        image = cv2.resize(image, (max_width, height), interpolation=cv2.INTER_AREA)
    preview = image.copy()
    label = f"t={time:.3f}s  frame={index}"
    scale = max(.35, min(.7, preview.shape[1] / 700))
    thickness = 1 if scale < .6 else 2
    (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(preview, (0, 0), (label_width + 10, label_height + baseline + 10), (255, 255, 255), -1)
    cv2.putText(preview, label, (5, label_height + 5), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness, cv2.LINE_AA)
    return preview


def contact_sheet(images):
    if not images:
        fail("no sample frames to make contact sheet")
    width = max(image.shape[1] for image in images)
    height = max(image.shape[0] for image in images)
    blank = np.zeros((height, width, 3), np.uint8)
    rows = []
    for start in range(0, len(images), 3):
        row = images[start:start + 3] + [blank] * (3 - len(images[start:start + 3]))
        rows.append(np.hstack([np.pad(image, ((0, height - image.shape[0]), (0, width - image.shape[1]), (0, 0)))
                               for image in row]))
    return np.vstack(rows)


def inspect(args):
    try:
        times = [finite_number(float(x), "times") for x in args.times.split(",") if x.strip()]
    except ValueError:
        fail("--times must contain finite comma-separated numbers")
    if not times:
        fail("--times needs at least one comma-separated time")
    cap, info = video_info(args.video)
    try:
        samples, previews = [], []
        for time in times:
            if time < 0 or time > info["duration"]:
                fail("requested inspect time is outside the video time range")
            frame, index = frame_at(cap, info["fps"], info["frame_count"], time, "inspect time")
            write_image(args.output / f"frame_{index:06d}.png", frame)
            previews.append(thumbnail(frame, time, index))
            samples.append({"time_requested": time, "frame_index": index, "time_actual": index / info["fps"]})
        write_image(args.output / "contact_sheet.jpg", contact_sheet(previews))
        write_json(args.output / "metadata.json", {**info, "video": args.video.name, "samples": samples})
    finally:
        cap.release()


def render(args):
    cap, info = video_info(args.video)
    try:
        try:
            raw_config = json.loads(args.config.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            fail(f"cannot read config: {error}")
        config = validate_config(raw_config, info)
        terminal, terminal_index = frame_at(cap, info["fps"], info["frame_count"], config["terminal_time"], "terminal")
        composite = terminal.astype(np.float32)
        roi = np.zeros(terminal.shape[:2], bool)
        records, audits = [], []
        first_time = config["keyframes"][0]["time"]
        span = config["terminal_time"] - first_time
        for entry in config["keyframes"]:
            frame, index = frame_at(cap, info["fps"], info["frame_count"], entry["time"], "keyframe")
            x1, y1, x2, y2 = entry["box"]
            crop = frame[y1:y2, x1:x2]
            mask, method = segmentation(crop, entry)
            opacity = entry.get("opacity", .25 + .47 * ((entry["time"] - first_time) / span))
            alpha = (mask.astype(np.float32) / 255 * opacity)[..., None]
            region = composite[y1:y2, x1:x2]
            composite[y1:y2, x1:x2] = region * (1 - alpha) + crop * alpha
            roi[y1:y2, x1:x2] = True
            write_image(args.output / f"mask_{index:06d}.png", mask)
            audits.append(thumbnail(np.hstack([crop, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]), entry["time"], index))
            records.append({"time_requested": entry["time"], "frame_index": index,
                            "time_actual": index / info["fps"], "box": entry["box"],
                            "opacity": opacity, "mask_pixels": int(np.count_nonzero(mask)), "method": method})
        result = np.clip(composite, 0, 255).astype(np.uint8)
        outside_changed = int(np.count_nonzero(np.any(result != terminal, axis=2) & ~roi))
        if outside_changed:
            raise RuntimeError("ROI-outside background verification failed")
        write_image(args.output / "composite.png", result)
        write_image(args.output / "terminal.png", terminal)
        write_image(args.output / "mask_audit.jpg", contact_sheet(audits))
        manifest = {"video": args.video.name, "fps": info["fps"],
                    "original_resolution": [info["width"], info["height"]],
                    "terminal_time_requested": config["terminal_time"], "terminal_frame_index": terminal_index,
                    "terminal_time_actual": terminal_index / info["fps"], "config": config,
                    "keyframes": records,
                    "method": "terminal-frame background with source-pixel ROI masks",
                    "verification": {"roi_outside_identical": True, "outside_changed_pixels": outside_changed}}
        write_json(args.output / "manifest.json", manifest)
    finally:
        cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "render"):
        command = commands.add_parser(name)
        command.add_argument("--video", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "inspect":
            command.add_argument("--times", required=True)
        else:
            command.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        if not args.output.is_dir():
            fail("output is not a directory")
        (inspect if args.command == "inspect" else render)(args)
    except (ValueError, OSError, RuntimeError, cv2.error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
