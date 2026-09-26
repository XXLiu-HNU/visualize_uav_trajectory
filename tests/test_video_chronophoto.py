import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/visualize-uav-trajectory/scripts/video_chronophoto.py"


def load_module():
    spec = importlib.util.spec_from_file_location("video_chronophoto", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VideoChronophotoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        self.video = self.work / "two_objects.avi"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (80, 60))
        self.assertTrue(writer.isOpened())
        for index in range(76):
            frame = np.full((60, 80, 3), (20, 40, 60), np.uint8)
            cv2.rectangle(frame, (8 + index // 3, 20), (16 + index // 3, 30), (0, 0, 255), -1)
            cv2.rectangle(frame, (46, 38), (52, 44), (0, 255, 0), -1)  # disconnected payload
            writer.write(frame)
        writer.release()

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              text=True, capture_output=True)

    def config(self, **first):
        keyframe = {"time": 0.0, "box": [4, 16, 56, 48],
                    "manual_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        keyframe.update(first)
        return {"terminal_time": 2.0, "keyframes": [keyframe]}

    def test_render_manual_mask_preserves_disconnected_payload_and_background(self):
        config = self.work / "config.json"
        config.write_text(json.dumps(self.config()))
        output = self.work / "render"
        result = self.run_cli("render", "--video", self.video, "--config", config, "--output", output)
        self.assertEqual(result.returncode, 0, result.stderr)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["keyframes"][0]["method"], "manual_polygon")
        self.assertGreater(manifest["keyframes"][0]["mask_pixels"], 1000)
        self.assertTrue(manifest["verification"]["roi_outside_identical"])
        mask = cv2.imread(str(output / "mask_000000.png"), cv2.IMREAD_GRAYSCALE)
        self.assertGreater(mask[24, 45], 0, "manual polygon must retain disconnected payload")
        terminal = cv2.imread(str(output / "terminal.png"))
        composite = cv2.imread(str(output / "composite.png"))
        self.assertTrue(np.array_equal(terminal[0:15], composite[0:15]))

    def test_inspect_exports_metadata_sheet_and_requested_frames(self):
        output = self.work / "inspect"
        result = self.run_cli("inspect", "--video", self.video, "--output", output, "--times", "0,0.5,2")
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["fps"], 25.0)
        self.assertEqual([item["frame_index"] for item in metadata["samples"]], [0, 12, 50])
        self.assertTrue((output / "contact_sheet.jpg").is_file())
        self.assertTrue((output / "frame_000012.png").is_file())
        self.assertNotIn(str(self.work), (output / "metadata.json").read_text())

    def test_invalid_configs_are_rejected_with_clear_errors(self):
        cases = [
            (self.config(time=float("nan")), "finite"),
            (self.config(box=[-1, 2, 9, 9]), "box"),
            ({"terminal_time": 2.0, "keyframes": []}, "keyframes"),
            ({"terminal_time": 2.0, "keyframes": [
                {"time": 0.0, "box": [4, 4, 12, 12]},
                {"time": 0.01, "box": [4, 4, 12, 12]}]}, "same frame"),
        ]
        for number, (payload, phrase) in enumerate(cases):
            config = self.work / f"bad_{number}.json"
            config.write_text(json.dumps(payload))
            result = self.run_cli("render", "--video", self.video, "--config", config,
                                  "--output", self.work / f"out_{number}")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(phrase, result.stderr.lower())

    def test_thumbnail_is_small_and_carries_time_frame_label(self):
        module = load_module()
        preview = module.thumbnail(np.full((480, 640, 3), (20, 40, 60), np.uint8), 1.25, 31)
        self.assertLess(preview.shape[1], 640)
        self.assertTrue(np.any(np.all(preview > 220, axis=2)), "preview needs a readable label")

    def test_manual_polygon_does_not_invoke_grabcut(self):
        module = load_module()
        entry = {"manual_polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}
        with mock.patch.object(module.cv2, "grabCut", side_effect=AssertionError("must not run")):
            mask, method = module.segmentation(np.zeros((20, 20, 3), np.uint8), entry)
        self.assertEqual(method, "manual_polygon")
        self.assertGreater(np.count_nonzero(mask), 300)

    def test_include_and_exclude_keep_disconnected_manual_payload(self):
        module = load_module()
        entry = {
            "manual_polygon": [[0, 0], [.35, 0], [.35, 1], [0, 1]],
            "include_polygons": [[[.75, .25], [1, .25], [1, .75], [.75, .75]]],
            "exclude_polygons": [[[.15, .25], [.25, .25], [.25, .75], [.15, .75]]],
        }
        mask, _ = module.segmentation(np.zeros((20, 20, 3), np.uint8), entry)
        self.assertGreater(mask[10, 2], 0)
        self.assertEqual(mask[10, 4], 0)
        self.assertGreater(mask[10, 18], 0, "include polygon must retain remote payload")


if __name__ == "__main__":
    unittest.main()
