import sys
import cv2
import os
import csv
import time
import threading
import numpy as np
from datetime import datetime
from collections import deque
from PIL import ImageFont, ImageDraw, Image

# PyQt5 导入
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QTextEdit, QGroupBox,
                             QPushButton, QFrame, QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

# ==========================================
# 异步抓图器：负责清空缓冲区，只保留最新一帧
# ==========================================
class FrameGrabber(threading.Thread):
    def __init__(self, cam_id):
        super().__init__()
        self.cam_id = cam_id
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.is_connected = False
        self.daemon = True

    def run(self):
        while not self.stopped:
            src = int(self.cam_id) if str(self.cam_id).isdigit() else self.cam_id
            # 显式使用 FFMPEG 并配置优化参数
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.is_connected = False
                time.sleep(2)
                continue

            self.is_connected = True
            while not self.stopped:
                ret, frame = cap.read()
                if not ret:
                    self.is_connected = False
                    break

                with self.lock:
                    self.frame = frame  # 异步覆盖：永远只有最新一帧

            cap.release()

    def get_latest_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True


# ==========================================
# 核心检测逻辑（保留你的所有检测算法）
# ==========================================
class WaferSystem(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self, model_path, pinmodel_path, output_folder, conf_thresh=0.7, wafer_fixed_box=(350, 200, 550, 300)):
        super().__init__()
        # 仅原检测引擎需要这些依赖；其他视频检测器可独立复用下面的界面。
        from ultralytics import YOLO
        from ABCDE import ImageDetector
        from haxi import compare_images, phash, find_local_difference

        self.compare_images = compare_images
        self.phash = phash
        self.find_local_difference = find_local_difference
        self.output_folder = output_folder
        self.conf_thresh = conf_thresh
        self.wafer_fixed_box = wafer_fixed_box

        self.waferframe = deque(maxlen=5)
        self.xiha = deque(maxlen=10)
        self.min_area = 100
        self.max_area = 500
        self.threshold_value = 4
        self.diff_distance_thresh = 2
        self.diff_contour_thresh = 8

        self.preview_size = (200, 150)
        self.text_font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_scale = 0.8
        self.text_thickness = 2

        self.detector = ImageDetector(model_path=model_path, conf_thresh=conf_thresh, history_length=10,
                                      min_detection_frames=5, min_stable_frames=10, pin_threshold=5)
        self.PINmodel = YOLO(pinmodel_path)

        self.detection_history = deque(maxlen=10)
        self.pin_counter = 0
        self.PIN_THRESHOLD = 5
        self.normal_cycle_count = 0
        self.frame_id = 0
        self.is_recording = False
        self.cycle_logs = []
        self.cycle_start_time = None
        self.current_stats = {'has_wafer': False, 'run_frames': 0, 'anomaly_frames': 0, 'has_pin_error': False,
                              'cycle_count': 0}

        self.class_names = {0: 'Status A', 1: 'Status B', 2: 'Status C', 3: 'Unloaded', 4: 'Wafer'}
        self.colors = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255), 3: (255, 255, 0), 4: (255, 0, 255)}

    def process_frame(self, frame):
        self.frame_id += 1
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        frame = np.ascontiguousarray(frame)
        annotated = frame.copy()
        result_text = "Normal"
        is_abnormal = False

        # 基础检测
        final_cls, final_conf, best_box, raw_class_probs = self._detect_with_filter(frame)
        current_classes = {final_cls} if final_cls is not None else set()

        if not current_classes:
            self.detection_history.clear()
        else:
            self.detection_history.append(current_classes)
            self.detector._update_variable2(current_classes)

        status, pin_logic = self.detector._analyze_operation_sequence(self.frame_id, f"frame_{self.frame_id}",
                                                                      self.detection_history)

        # 绘制检测框
        if final_cls is not None and best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            c = self.colors.get(final_cls, (200, 200, 200))
            cv2.rectangle(annotated, (x1, y1), (x2, y2), c, 2)
            cv2.putText(annotated, f"{self.class_names.get(final_cls)} {final_conf:.2f}", (x1, y1 - 10), self.text_font,
                        0.6, c, 2)

        # PIN 检测
        if pin_logic == 'Pin':
            if self._check_pin(frame, annotated):
                result_text = "PIN Abnormal";
                is_abnormal = True
                if self.is_recording: self.current_stats['has_pin_error'] = True

        # Wafer 区域差异检测
        if status == "wafer Runing":
            fx1, fy1, fx2, fy2 = self.wafer_fixed_box
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 255), 3)
            try:
                crop_frame = frame[fy1:fy2, fx1:fx2]
                crop_image = Image.fromarray(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB))
                self.waferframe.append(crop_image)
                self.xiha.append(self.phash(crop_image))

                if len(self.waferframe) > 1:
                    hamming_distance = self.compare_images(self.xiha)
                    img1 = cv2.resize(np.array(self.waferframe[-2]), (crop_frame.shape[1], crop_frame.shape[0]))
                    img2 = np.array(self.waferframe[-1])
                    contours = self.find_local_difference(img1, img2, threshold_value=self.threshold_value)
                    valid_contours = [c for c in contours if self.min_area <= cv2.contourArea(c) <= self.max_area]

                    for contour in valid_contours:
                        contour[:, :, 0] += fx1;
                        contour[:, :, 1] += fy1
                        cv2.drawContours(annotated, [contour], -1, (0, 0, 255), 2)

                    if hamming_distance > self.diff_distance_thresh or len(valid_contours) > self.diff_contour_thresh:
                        result_text = "Wafer Shake";
                        is_abnormal = True
                        if self.is_recording: self.current_stats['anomaly_frames'] += 1
            except:
                pass

        # 叠加显示
        cv2.putText(annotated, f"Process Status: {status}", (20, 60), self.text_font, self.text_scale, (255, 255, 255),
                    self.text_thickness)
        cv2.putText(annotated, f"Detection Result: {result_text}", (20, 100), self.text_font, self.text_scale,
                    (0, 0, 255) if is_abnormal else (0, 255, 0), self.text_thickness)
        cv2.putText(annotated, f"Frame ID (Processed): {self.frame_id}", (20, 140), self.text_font, self.text_scale,
                    (255, 255, 255), self.text_thickness)

        frame_log = {
            "时间": ts_str, "是否异常": "是" if is_abnormal else "否", "异常类型": result_text,
            "目标类别": self.class_names.get(final_cls, "None") if final_cls is not None else "None",
            "Status A信度": round(raw_class_probs[0], 4), "Status B信度": round(raw_class_probs[1], 4),
            "Status C信度": round(raw_class_probs[2], 4), "Unloaded信度": round(raw_class_probs[3], 4),
            "Wafer信度": round(raw_class_probs[4], 4)
        }
        self._manage_recording(status, frame_log)
        return annotated, {"status": status, "result": result_text, "recording": self.is_recording,
                           "frame_id": self.frame_id}

    def _detect_with_filter(self, frame):
        try:
            results, _ = self.detector.detect_image(frame)
            raw_class_probs = [0.0] * 5
            raw_box = None
            if results and results[0].boxes:
                raw_box = results[0].boxes[0]
                for box in results[0].boxes:
                    c, conf = int(box.cls), float(box.conf[0])
                    if 0 <= c < 5: raw_class_probs[c] = max(raw_class_probs[c], conf)
            final_cls = np.argmax(raw_class_probs) if max(raw_class_probs) >= self.conf_thresh else None
            return final_cls, (raw_class_probs[final_cls] if final_cls is not None else 0), raw_box, raw_class_probs
        except:
            return None, 0.0, None, [0.0] * 5

    def _check_pin(self, frame, img):
        try:
            res = self.PINmodel(frame, conf=0.1, verbose=False)
            found = any(self.PINmodel.names[int(box.cls)] == 'Pin' for r in res for box in r.boxes)
            if found: self.pin_counter += 1
            print('self.pin_counter',self.pin_counter)
            return self.pin_counter >= self.PIN_THRESHOLD
        except:
            return False

    def _manage_recording(self, status, frame_log):
        stable = self.detector.variable2
        is_idle = (3 in (stable[-1] if stable else set())) if stable else True
        if not self.is_recording and not is_idle:
            self.is_recording, self.cycle_start_time, self.cycle_logs = True, datetime.now(), []
            self.normal_cycle_count += 1
            self.current_stats = {'has_wafer': False, 'run_frames': 0, 'anomaly_frames': 0, 'has_pin_error': False,
                                  'cycle_count': self.normal_cycle_count}
            self.log_signal.emit(f"▶️ 开始周期 #{self.normal_cycle_count}")
        if self.is_recording:
            self.cycle_logs.append(frame_log)
            if frame_log["目标类别"] == "Wafer": self.current_stats['has_wafer'] = True
            if status == "wafer Runing": self.current_stats['run_frames'] += 1
        if self.is_recording and is_idle:
            self.is_recording = False
            self.log_signal.emit("⏹️ 周期结束，保存日志...")
            self.csv_worker = CSVWriterWorker(self.cycle_logs.copy(), self.cycle_start_time, datetime.now(),
                                              self.output_folder, self.current_stats.copy())
            self.csv_worker.finished_signal.connect(lambda s: self.log_signal.emit(s))
            self.csv_worker.start()


class CSVWriterWorker(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, frame_logs, start_dt, end_dt, output_folder, run_stats):
        super().__init__()
        self.frame_logs, self.start_dt, self.end_dt, self.output_folder, self.stats = frame_logs, start_dt, end_dt, output_folder, run_stats

    def run(self):
        try:
            if (self.end_dt - self.start_dt).total_seconds() < 0.1: return
            daily_folder = os.path.join(self.output_folder, self.start_dt.strftime("%Y-%m-%d"))
            os.makedirs(daily_folder, exist_ok=True)
            csv_path = os.path.join(daily_folder,
                                    f"{self.stats['cycle_count']}_{self.start_dt.strftime('%H%M%S')}-{self.end_dt.strftime('%H%M%S')}.csv")
            with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=['时间', '是否异常', '异常类型', '目标类别', 'Status A信度',
                                                       'Status B信度', 'Status C信度', 'Unloaded信度', 'Wafer信度'])
                writer.writeheader()
                writer.writerows(self.frame_logs)
            self.finished_signal.emit(f"✅ CSV保存成功: {csv_path}")
        except Exception as e:
            self.finished_signal.emit(f"❌ 保存失败: {e}")


# ==========================================
# 消费者线程：获取最新帧并触发逻辑
# ==========================================
class CameraThread(QThread):
    image_signal = pyqtSignal(np.ndarray)
    info_signal = pyqtSignal(dict)

    def __init__(self, system, cam_id):
        super().__init__()
        self.system = system
        self.grabber = FrameGrabber(cam_id)
        self.running = True

    def run(self):
        self.grabber.start()
        while self.running:
            frame = self.grabber.get_latest_frame()
            if frame is None:
                time.sleep(0.01);
                continue

            try:
                out_img, info = self.system.process_frame(frame)
                self.image_signal.emit(out_img)
                self.info_signal.emit(info)
            except Exception as e:
                print(f"Error: {e}")

            # AI 处理频率控制 (此处 0.03s 约为 30FPS，可根据需要调大如 0.1s 节省算力)
            time.sleep(0.03)

    def stop(self):
        self.running = False
        self.grabber.stop()
        self.wait()


# ==========================================
# 主窗口：保留原始配色
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self, initialize_system=True):
        super().__init__()
        # 参数配置
        self.MODEL_PATH = r"G:\code\wferdetect\weight\noraml1.pt"
        self.PIN_MODEL_PATH = r"G:\code\wferdetect\weight\pin0831.pt"
        self.OUTPUT_DIR = r"G:\code\wferdetect\data\output"
        self.CAMERA_ID = "rtsp://192.168.1.168/play/0"
        self.FIXED_BOX = (1000, 520, 1700, 720)
        self.setFixedSize(1600, 900)
        self.init_ui()
        if not initialize_system:
            return
        try:
            self.system = WaferSystem(self.MODEL_PATH, self.PIN_MODEL_PATH, self.OUTPUT_DIR,
                                      wafer_fixed_box=self.FIXED_BOX)
            self.system.log_signal.connect(self.add_log)
            self.add_log("异步检测引擎就绪")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化失败: {e}")

    def init_ui(self):
        self.setWindowTitle("智能晶圆外延工艺检测系统")
        self.setStyleSheet("background-color: #ecf0f1;")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板配色（恢复原始样式）
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        title = QLabel("智能晶圆外延工艺检测系统")
        title.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: #2c3e50;
            padding: 10px; background-color: #ecf0f1; margin-bottom: 20px;
        """)
        title.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(title)

        self.status_label = QLabel("等待启动...")
        self.status_label.setStyleSheet("""
            background-color: #e8f4fc; border: 2px solid #3498db;
            border-radius: 10px; padding: 20px; font-size: 18px; font-weight: bold; color: #2c3e50;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.status_label)

        # 日志组配色
        log_group = QGroupBox("📝 运行日志")
        log_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 2px solid #34495e; border-radius: 8px; margin-top: 10px; }")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit();
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "QTextEdit { background-color: #2c3e50; color: #ecf0f1; font-family: Consolas; font-size: 11px; }")
        log_layout.addWidget(self.log_text)
        left_layout.addWidget(log_group)

        # 异常事件配色
        err_group = QGroupBox("异常事件")
        err_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #2980b9; border: 1px solid #3498db; border-radius: 5px; margin-top: 10px; }")
        err_layout = QVBoxLayout(err_group)
        self.err_text = QTextEdit();
        self.err_text.setReadOnly(True)
        self.err_text.setStyleSheet("QTextEdit { background-color: #2c3e50; color: #ecf0f1; font-size: 12px; }")
        err_layout.addWidget(self.err_text)
        left_layout.addWidget(err_group)

        # 按钮配色
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶️ 开始监控")
        self.start_btn.setStyleSheet(
            "QPushButton { background-color: #3498db; color: white; padding: 12px; font-weight: bold; border-radius: 6px; } QPushButton:hover { background-color: #2980b9; }")
        self.start_btn.clicked.connect(self.start_monitoring)

        self.stop_btn = QPushButton("⏹️ 停止监控")
        self.stop_btn.setStyleSheet(
            "QPushButton { background-color: #95a5a6; color: white; padding: 12px; font-weight: bold; border-radius: 6px; }")
        self.stop_btn.clicked.connect(self.stop_monitoring);
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.start_btn);
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)
        main_layout.addWidget(left_panel, 1)

        # 视频区域
        self.image_label = QLabel("摄像头画面")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; border: 3px solid #34495e;")
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        main_layout.addWidget(self.image_label, 3)

    def start_monitoring(self):
        self.thread = CameraThread(self.system, self.CAMERA_ID)
        self.thread.image_signal.connect(self.update_image)
        self.thread.info_signal.connect(self.update_status)
        self.thread.start()
        self.start_btn.setEnabled(False);
        self.stop_btn.setEnabled(True)
        self.add_log("监控已开启")

    def stop_monitoring(self):
        if hasattr(self, 'thread'): self.thread.stop()
        self.start_btn.setEnabled(True);
        self.stop_btn.setEnabled(False)
        self.status_label.setText("监控已停止")
        self.add_log("监控已手动关闭")

    def update_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        # 拷贝数据，避免后台线程下一帧覆盖 NumPy 缓冲区。
        qt_img = QImage(img_rgb.data, w, h, img_rgb.strides[0], QImage.Format_RGB888).copy()
        self.image_label.setPixmap(
            QPixmap.fromImage(qt_img).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, info):
        self.status_label.setText(f"状态: {info['status']}\n结果: {info['result']}")
        if "Abnormal" in info['result'] or "Shake" in info['result']:
            self.err_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {info['result']}")

    def add_log(self, msg):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow();
    window.show()
    sys.exit(app.exec_())
