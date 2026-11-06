"""Regression-suite dispatch tests without loading a detector or video."""
from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from industrial_inspection.tools import predict_slot_groups as prediction


class RegressionDispatchTests(TestCase):
    def args(self, root):
        return SimpleNamespace(source=root / "primary.avi", output_dir=None,
                               reselect_roi=True, roi_file=root / "primary.json",
                               ui=True, show=True, max_frames=0)

    def test_each_regression_video_uses_its_own_roi_and_headless_output(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, roi = root / "raw.avi", root / "raw_roi.json"
            source.write_bytes(b"video")
            roi.write_text("{}", encoding="utf-8")
            seen = []
            primary = self.args(root)
            primary.settled_unload_trial = True

            def prepare(args):
                seen.append(("prepare", args.source, args.roi_file, args.ui,
                             args.show, args.reselect_roi))
                return {"polygon": [], "size": (640, 480), "start_frame": 0,
                        "s4_hand_region": None}

            def run_once(args, **kwargs):
                self.assertTrue(args.settled_unload_trial)
                seen.append(("run", args.source, args.output_dir, args.ui,
                             args.show, kwargs.get("on_frame")))
                return {"status": "end_of_stream", "output_dir": str(args.output_dir),
                        "processed_frames": 1, "group_process_reports": {}}

            with patch.object(prediction, "REGRESSION_TEST_VIDEOS",
                              (("测试", source, roi),)), \
                 patch.object(prediction, "S4_HAND_REGION_ENABLED", False), \
                 patch.object(prediction, "REGRESSION_OUTPUT_ROOT", root / "outputs"), \
                 patch.object(prediction, "prepare_context", prepare), \
                 patch.object(prediction, "_run_prediction_once", run_once):
                results = prediction._run_regression_tests(
                    primary, {"status": "end_of_stream"})

            self.assertEqual(results[0]["status"], "end_of_stream")
            self.assertEqual(seen[0][0], "prepare")
            self.assertEqual(seen[0][1:3], (source, roi))
            self.assertEqual(seen[0][3:], (False, False, False))
            self.assertEqual(seen[1][0], "run")
            self.assertEqual(seen[1][1], source)
            self.assertEqual(seen[1][3:], (False, False, None))

    def test_primary_result_is_reused_only_with_the_same_source_and_roi(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, roi = root / "primary.avi", root / "primary_roi.json"
            source.write_bytes(b"video")
            roi.write_text("{}", encoding="utf-8")
            args = self.args(root)
            args.roi_file = roi
            with patch.object(prediction, "REGRESSION_TEST_VIDEOS",
                              (("同一视频", source, roi),)):
                results = prediction._run_regression_tests(
                    args, {"status": "end_of_stream", "output_dir": str(root / "primary")})
            self.assertTrue(results[0]["reused_primary_result"])
            self.assertEqual(results[0]["output_dir"], str(root / "primary"))

    def test_one_video_failure_does_not_skip_the_next_video(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, roi = root / "raw.avi", root / "raw_roi.json"
            source.write_bytes(b"video")
            roi.write_text("{}", encoding="utf-8")
            with patch.object(prediction, "REGRESSION_TEST_VIDEOS",
                              (("一", source, roi), ("二", source, roi))), \
                 patch.object(prediction, "S4_HAND_REGION_ENABLED", False), \
                 patch.object(prediction, "prepare_context", return_value={}), \
                 patch.object(prediction, "_run_prediction_once", side_effect=[
                     RuntimeError("inference failed"),
                     {"status": "hatch_closed", "output_dir": "second", "processed_frames": 2}]):
                results = prediction._run_regression_tests(
                    self.args(root), {"status": "end_of_stream"})
            self.assertEqual([result["status"] for result in results], ["failed", "hatch_closed"])
            self.assertEqual(results[0]["error"], "inference failed")

    def test_stopping_primary_does_not_start_regression(self):
        with TemporaryDirectory() as folder, \
             patch.object(prediction, "_run_prediction_once") as run_once:
            results = prediction._run_regression_tests(
                self.args(Path(folder)), {"status": "user_stopped"})
        self.assertEqual(results, [])
        run_once.assert_not_called()

    def test_wrapper_saves_regression_index_without_modifying_primary_results(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.args(root)
            args.run_regression_tests = True
            primary = {"status": "end_of_stream", "output_dir": str(root), "processed_frames": 1}
            results = [{"name": "测试", "status": "hatch_closed", "output_dir": "regression"}]
            context, on_frame = object(), object()
            with patch.object(prediction, "_run_prediction_once", return_value=primary) as run_once, \
                 patch.object(prediction, "_run_regression_tests", return_value=results):
                summary = prediction.run_prediction(args, on_frame=on_frame, roi_polygon=context)
            self.assertEqual(summary["processed_frames"], 1)
            self.assertEqual(summary["regression_tests"], results)
            self.assertEqual(json.loads((root / "regression_tests.json").read_text(encoding="utf-8")), results)
            self.assertEqual(json.loads((root / "summary.json").read_text(encoding="utf-8")), summary)
            self.assertIs(run_once.call_args.kwargs["on_frame"], on_frame)
            self.assertIs(run_once.call_args.kwargs["roi_polygon"], context)

    def test_legacy_programmatic_call_does_not_launch_raw_video_regression(self):
        with TemporaryDirectory() as folder, \
             patch.object(prediction, "_run_prediction_once", return_value={"status": "end_of_stream"}), \
             patch.object(prediction, "_run_regression_tests") as run_regression:
            prediction.run_prediction(self.args(Path(folder)))
        run_regression.assert_not_called()

    def test_script_args_enable_regression_by_default_and_allow_disabling(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.args(root)
            args.source.write_bytes(b"video")
            args.weights = root / "model.pt"
            args.weights.write_bytes(b"weights")
            args.conf, args.audit_conf = .1, .001
            args.stride, args.imgsz, args.device = 1, 640, "cpu"
            with patch.object(prediction.base, "parse_args", return_value=args) as parse_args:
                self.assertTrue(prediction.build_args([]).run_regression_tests)
                trial = prediction.build_args(["--no-regression-tests", "--settled-unload-numbering"])
                self.assertFalse(trial.run_regression_tests)
                self.assertTrue(trial.settled_unload_trial)
                self.assertNotIn("--no-regression-tests", parse_args.call_args.args[0])
                self.assertNotIn("--settled-unload-numbering", parse_args.call_args.args[0])

    def test_settled_numbering_trial_applies_to_regression_as_well_as_primary(self):
        polygon = ((.1, .1), (.9, .1), (.9, .9), (.1, .9))
        cases = (
            ("camera_B_20260924_021033.avi", True, True, .20),
            ("camera_B_20260923_143845.avi", True, True, .20),
            ("camera_B_20260924_013432.avi", True, False, .20),
            ("camera_B_20260924_015247.avi", True, False, .20),
            ("new_video.avi", False, False, None),
            ("new_video.avi", True, False, .20),
        )
        with patch.multiple(prediction, PROCESS_PHASE="unload", AUTO_GROUP_SWITCH=True, EXIT_MIN_SLOTS=1):
            for name, trial, regression, expected in cases:
                with self.subTest(name=name, trial=trial, regression=regression):
                    args = SimpleNamespace(source=Path(name), settled_unload_trial=trial,
                                           regression_run=regression)
                    sequence = prediction.make_sequence(polygon, (640, 480), args=args)
                    self.assertEqual(sequence.tracker.unload_entry_numbering_seconds, expected)


if __name__ == "__main__":
    import unittest
    unittest.main()
