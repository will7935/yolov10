"""在 PyCharm 中直接运行此文件；只需修改下方配置，无需填写运行参数。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

# ==================== 在这里修改运行配置 ====================
MODE = "replay"  # replay：回放检测缓存；live：使用 YOLO 检测视频或摄像头
VIDEO = r"I:\youyan\test_dataset\测试视频4\camera_B_20260924_161009.avi"  # 摄像头可填 "0"
CACHE = PROJECT_ROOT / "industrial_inspection/video_results/carrier_box_exp42_20261021/exp42_final_detections.jsonl"
WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp4.2\weights\best.pt")
CONF = 0.35
IMGSZ = 1280
DEVICE = "0"  # live 模式使用；无 CUDA 时可改为 "cpu"
OUTPUT_DIR = SCRIPT_DIR / "outputs/pycharm_runs"  # 每次运行自动创建带时间戳的子目录
MAX_FRAMES = 0  # 0：全部；试跑时可改为 30 等正整数
RENDER_VIDEO = True  # 输出 annotated.mp4；False 时只输出记录、事件和汇总
CIRCLE_CHECK = False  # 可选同圆约90°几何提示，不参与切组判定
# ==========================================================


def _project_path(value: str | Path) -> Path:
    """配置中的相对路径统一相对于仓库根目录，避免依赖 PyCharm 工作目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def main() -> int:
    if MODE not in ("replay", "live"):
        raise ValueError('MODE 只能填写 "replay" 或 "live"')

    # 直接运行脚本时也能导入包，不要求把仓库设置为 Sources Root。
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from industrial_inspection.slot_process_v2.run import main as run_pipeline

    output = _project_path(OUTPUT_DIR) / f"{MODE}_{datetime.now().astimezone():%Y%m%d_%H%M%S_%f}"
    video = str(VIDEO)
    if not video.isdecimal():
        video = str(_project_path(video))
    argv = [
        MODE,
        "--video", video,
        "--output-dir", str(output),
        "--weights", str(_project_path(WEIGHTS)),
        "--conf", str(CONF),
        "--imgsz", str(IMGSZ),
        "--max-frames", str(MAX_FRAMES),
    ]
    if MODE == "replay":
        argv.extend(["--cache", str(_project_path(CACHE))])
    else:
        argv.extend(["--device", str(DEVICE)])
    if CIRCLE_CHECK:
        argv.append("--circle-check")
    if not RENDER_VIDEO:
        argv.append("--no-render")

    print(f"运行模式：{MODE}\n视频来源：{video}\n输出目录：{output}", flush=True)
    return run_pipeline(argv)


if __name__ == "__main__":
    raise SystemExit(main())
