"""Connect local-video prediction to the reusable UI7V window."""

from __future__ import annotations

import argparse
import sys
import threading
import traceback

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMessageBox

from industrial_inspection.tools.UI7V import MainWindow


class VideoPredictionThread(QThread):
    frame_signal = pyqtSignal(object, dict)
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, args, predict, roi_polygon, parent=None):
        super().__init__(parent)
        self.args = argparse.Namespace(**vars(args))
        # 图像由 Qt 主线程显示，后台线程不再创建 OpenCV 预览窗口。
        self.args.show = False
        self.predict = predict
        self.roi_polygon = roi_polygon
        self.stop_event = threading.Event()
        self.frame_displayed = threading.Event()

    def publish_frame(self, image, info):
        self.frame_displayed.clear()
        self.frame_signal.emit(image, info)
        # 等主线程完成显示，防止大图像在 Qt 消息队列中持续积压。
        # 这里不丢弃任何需要推理或写入结果的视频帧。
        while not self.frame_displayed.wait(0.1):
            if self.stop_event.is_set():
                break

    def run(self):
        try:
            summary = self.predict(self.args, on_frame=self.publish_frame,
                                   on_log=self.log_signal.emit,
                                   should_stop=self.stop_event.is_set,
                                   roi_polygon=self.roi_polygon)
            self.result_signal.emit(summary)
        except Exception as exc:
            traceback.print_exc()
            self.error_signal.emit(str(exc))

    def stop(self):
        self.stop_event.set()
        self.frame_displayed.set()


class VideoMainWindow(MainWindow):
    def __init__(self, args, predict, prepare_roi):
        # 复用原布局、配色、日志和 update_image / update_status 显示方法。
        super().__init__(initialize_system=False)
        self.args = args
        self.group_completion_log_only = bool(getattr(args, "group_completion_log_only", False))
        self.predict = predict
        self.prepare_roi = prepare_roi
        self.thread = None
        self.close_when_finished = False
        self.setWindowTitle("智能晶圆外延工艺检测系统 - 本地视频检测")
        self.image_label.setText("本地视频画面")
        self.start_btn.setText("▶️ 开始检测")
        self.stop_btn.setText("⏹️ 停止检测")
        self.status_label.setWordWrap(True)
        self.log_text.document().setMaximumBlockCount(1000)
        self.err_text.document().setMaximumBlockCount(1000)
        if self.group_completion_log_only:
            self.log_text.setFontFamily("Consolas")
        else:
            self.add_log(f"视频: {args.source.resolve()}")
            self.add_log(f"模型: {args.weights.resolve()}")
            self.add_log("点击开始检测；每次运行保存到独立结果目录。")

    def start_monitoring(self):
        if self.thread is not None:
            return
        if self.args.output_dir is not None and self.args.output_dir.exists():
            QMessageBox.warning(self, "输出目录已存在", "请使用尚不存在的 --output-dir，或省略该参数自动创建新目录。")
            return
        self.start_btn.setEnabled(False)
        self.status_label.setText("准备槽位 ROI...")
        try:
            # 交互圈选在主线程完成，读取已有 ROI 时不会弹出窗口。
            roi_polygon = self.prepare_roi(self.args)
        except Exception as exc:
            self.status_label.setText("初始化失败")
            if not self.group_completion_log_only:
                self.add_log(f"ROI 初始化失败: {exc}")
            self.err_text.append(str(exc))
            self.start_btn.setEnabled(True)
            QMessageBox.critical(self, "初始化失败", str(exc))
            return
        if self.group_completion_log_only:
            self.log_text.clear()
        self.thread = VideoPredictionThread(self.args, self.predict, roi_polygon, self)
        self.thread.frame_signal.connect(self.display_frame)
        self.thread.log_signal.connect(self.add_log)
        self.thread.result_signal.connect(self.show_result)
        self.thread.error_signal.connect(self.show_error)
        self.thread.finished.connect(self.prediction_finished)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("正在加载模型...")
        if not self.group_completion_log_only:
            self.add_log("视频检测已开启")
        self.thread.start()

    def display_frame(self, image, info):
        try:
            self.update_image(image)
            if self.thread is not None and not self.thread.stop_event.is_set():
                self.update_status(info)
        finally:
            if self.thread is not None:
                self.thread.frame_displayed.set()

    def stop_monitoring(self):
        if self.thread is None:
            return
        self.thread.stop()
        self.stop_btn.setEnabled(False)
        self.status_label.setText("正在停止并保存结果...")
        if not self.group_completion_log_only:
            self.add_log("已请求停止，当前帧处理完成后保存结果。")

    def show_result(self, summary):
        reasons = {"end_of_stream": "视频检测完成", "frame_limit": "已达到处理帧数上限",
                   "user_stopped": "检测已停止", "interrupted": "检测已中断"}
        self.status_label.setText(f"{reasons.get(summary['status'], summary['status'])}\n"
                                  f"已处理 {summary['processed_frames']} 帧")
        if not self.group_completion_log_only:
            self.add_log(f"结果已保存: {summary['output_dir']}")

    def show_error(self, message):
        self.status_label.setText("检测失败")
        self.err_text.append(message)
        if not self.group_completion_log_only:
            self.add_log(f"检测失败: {message}")

    def prediction_finished(self):
        thread = self.thread
        self.thread = None
        thread.deleteLater()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.close_when_finished:
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            # 等待后台正常释放视频写入器，避免未完成的 MP4 或 QThread 被销毁。
            self.close_when_finished = True
            self.stop_monitoring()
            event.ignore()
            return
        event.accept()


def launch_ui(args, predict, prepare_roi):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = VideoMainWindow(args, predict, prepare_roi)
    window.show()
    return app.exec_()
