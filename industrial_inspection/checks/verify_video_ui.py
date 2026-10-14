"""Regression checks for the UI7V local-video worker lifecycle.

Run with: python -m unittest discover -s industrial_inspection/checks -p verify_video_ui.py
"""

import argparse
import io
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
try:
    from PyQt5.QtWidgets import QApplication
    from industrial_inspection.tools.predict_video_ui import VideoMainWindow
except ModuleNotFoundError as exc:
    if not exc.name.startswith("PyQt5"):
        raise
    QApplication = None


@unittest.skipIf(QApplication is None, "UI checks require optional PyQt5")
class VideoWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.app.setQuitOnLastWindowClosed(False)

    def setUp(self):
        self.window = None
        self.args = argparse.Namespace(source=Path("test.avi"), weights=Path("test.pt"),
                                       output_dir=None, show=True)

    def tearDown(self):
        if self.window is not None:
            if self.window.thread is not None:
                self.window.stop_monitoring()
                self.wait_for(lambda: self.window.thread is None)
            self.window.close()
        self.app.processEvents()

    def wait_for(self, condition, timeout=5):
        deadline = time.monotonic() + timeout
        while not condition():
            self.app.processEvents()
            if time.monotonic() > deadline:
                self.fail("Qt worker did not reach the expected state")
            time.sleep(0.005)
        self.app.processEvents()

    def make_window(self, predict, prepare_roi=lambda args: None):
        self.window = VideoMainWindow(self.args, predict, prepare_roi)
        self.window.show()
        self.app.processEvents()
        return self.window

    @staticmethod
    def summary(status, count):
        return {"status": status, "processed_frames": count, "output_dir": "test-output"}

    def test_display_complete_and_restart_without_legacy_engine(self):
        runs = []

        def predict(args, *, on_frame, on_log, should_stop, roi_polygon):
            self.assertFalse(args.show)
            runs.append(1)
            on_log("test inference started")
            image = np.zeros((90, 160, 3), dtype=np.uint8)
            image[:, :] = (0, 0, 255)
            on_frame(image, {"status": "processing", "result": "hand: 1"})
            return self.summary("frame_limit", 1)

        window = self.make_window(predict)
        for _ in range(2):
            window.start_btn.click()
            self.wait_for(lambda: window.thread is None)
            self.assertTrue(window.start_btn.isEnabled())
            self.assertFalse(window.stop_btn.isEnabled())
            self.assertIn("已处理 1 帧", window.status_label.text())
            pixmap = window.image_label.pixmap()
            self.assertFalse(pixmap.isNull())
            self.assertEqual(pixmap.toImage().pixelColor(0, 0).red(), 255)
        self.assertEqual(len(runs), 2)
        self.assertIn("test inference started", window.log_text.toPlainText())

    def test_stop_and_close_finish_worker_before_window_is_destroyed(self):
        def predict(args, *, on_frame, on_log, should_stop, roi_polygon):
            count = 0
            while not should_stop():
                count += 1
                on_frame(np.zeros((90, 160, 3), dtype=np.uint8),
                         {"status": "processing", "result": "none"})
            return self.summary("user_stopped", count)

        window = self.make_window(predict)
        window.start_btn.click()
        self.wait_for(lambda: window.image_label.pixmap() is not None)
        window.stop_btn.click()
        self.wait_for(lambda: window.thread is None)
        self.assertIn("检测已停止", window.status_label.text())
        window.start_btn.click()
        window.close()
        self.wait_for(lambda: window.thread is None and not window.isVisible())
        self.assertTrue(window.close_when_finished)

    def test_error_reenables_start_and_is_reported(self):
        def predict(*args, **kwargs):
            raise RuntimeError("test model load failure")

        window = self.make_window(predict)
        with patch("sys.stderr", new_callable=io.StringIO):
            window.start_btn.click()
            self.wait_for(lambda: window.thread is None)
        self.assertTrue(window.start_btn.isEnabled())
        self.assertEqual(window.status_label.text(), "检测失败")
        self.assertIn("test model load failure", window.err_text.toPlainText())

    def test_cancelled_roi_does_not_start_inference(self):
        calls = []

        def prepare_roi(args):
            raise RuntimeError("ROI selection cancelled")

        window = self.make_window(lambda *args, **kwargs: calls.append(1), prepare_roi)
        with patch("industrial_inspection.tools.predict_video_ui.QMessageBox.critical"):
            window.start_btn.click()
        self.assertIsNone(window.thread)
        self.assertTrue(window.start_btn.isEnabled())
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
