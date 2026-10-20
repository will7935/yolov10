"""Verify shared, custom and manually drawn ROI entry paths without a model."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from industrial_inspection.tools import predict_slot_groups as groups
from industrial_inspection.tools import predict_video as video
from industrial_inspection.tools.slot_roi import DEFAULT_SLOT_ROI_FILE, load_roi, save_roi


class RoiConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="roi_configuration_")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.source = self.root / "new_recording.avi"
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"MJPG"), 10., (640, 480))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((480, 640, 3), dtype=np.uint8))
        writer.release()
        self.weights = self.root / "unused.pt"
        self.weights.touch()
        self.custom = np.array([[.1, .1], [.9, .1], [.9, .9], [.1, .9]], dtype=np.float32)

    def args(self, options=()):
        with (patch.object(groups, "SOURCE_VIDEO", self.source),
              patch.object(groups, "MODEL_WEIGHTS", self.weights)):
            return groups.build_args(list(options))

    def test_default_keeps_first_roi_despite_a_different_video_sidecar(self):
        save_roi(self.source.with_name(self.source.stem + "_slot_roi.json"),
                 self.custom, (640, 480), self.source, 0)
        with patch.object(video, "select_roi", side_effect=AssertionError("Unexpected ROI dialog")):
            context = groups.prepare_context(self.args())
        self.assertEqual(Path(context["roi_file"]), DEFAULT_SLOT_ROI_FILE)
        np.testing.assert_array_equal(context["polygon"], load_roi(DEFAULT_SLOT_ROI_FILE, (640, 480)))

    def test_group_cli_reads_custom_configuration_and_retains_runtime_options(self):
        path = self.root / "custom.json"
        save_roi(path, self.custom, (640, 480), self.source, 0)
        args = self.args(["--roi-file", str(path), "--no-reselect-roi", "--no-ui",
                          "--no-show", "--device", "cpu", "--max-frames", "1"])
        self.assertFalse(args.ui)
        self.assertFalse(args.show)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.max_frames, 1)
        with patch.object(video, "select_roi", side_effect=AssertionError("Unexpected ROI dialog")):
            context = groups.prepare_context(args)
        np.testing.assert_array_equal(context["polygon"], self.custom)
        self.assertEqual(Path(context["roi_file"]), path)

    def test_group_cli_manual_mode_saves_and_reloads_the_drawn_polygon(self):
        path = self.root / "drawn.json"
        args = self.args(["--roi-file", str(path), "--reselect-roi", "--roi-select-frame", "0"])
        shared_before = DEFAULT_SLOT_ROI_FILE.read_bytes()

        def draw(source, target, frame):
            save_roi(target, self.custom, (640, 480), source, frame)

        with patch.object(video, "select_roi", side_effect=draw) as select:
            context = groups.prepare_context(args)
        select.assert_called_once_with(self.source, path, 0)
        np.testing.assert_array_equal(context["polygon"], self.custom)
        self.assertEqual(DEFAULT_SLOT_ROI_FILE.read_bytes(), shared_before)
        args.reselect_roi = False
        with patch.object(video, "select_roi", side_effect=AssertionError("Unexpected ROI dialog")):
            np.testing.assert_array_equal(groups.prepare_context(args)["polygon"], self.custom)

    def test_group_script_roi_settings_are_used_when_no_cli_override_is_given(self):
        path = self.root / "script_roi.json"
        with (patch.object(groups, "SLOT_ROI_FILE", path),
              patch.object(groups, "ROI_RESELECT", True),
              patch.object(groups, "ROI_SELECT_FRAME", 0)):
            args = self.args()
        self.assertEqual(args.roi_file, path)
        self.assertTrue(args.reselect_roi)

    def test_cancelled_manual_selection_leaves_configuration_unchanged(self):
        path = self.root / "existing.json"
        save_roi(path, self.custom, (640, 480), self.source, 0)
        before = path.read_bytes()
        args = self.args(["--roi-file", str(path), "--reselect-roi"])
        with (patch.object(video, "select_roi", side_effect=RuntimeError("ROI selection cancelled")),
              self.assertRaisesRegex(RuntimeError, "cancelled")):
            groups.prepare_context(args)
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
