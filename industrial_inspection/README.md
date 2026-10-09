# 工业检测：YOLOv10 检测训练

## 训练结果（2026-10-09）

已完成 49 个 epoch；连续 30 轮验证 fitness 无改善触发早停。
使用验证集选出的第 19 轮 `weights/best.pt`，而不是最后一轮模型。

验证集：mAP@0.5=87.88%，mAP@0.5:0.95=49.22%。
测试集（142 张）：mAP@0.5=90.27%，mAP@0.5:0.95=59.01%，
平均 precision=92.47%，平均 recall=86.81%。P/R 是框架在评估工作点统计的值，
不等同于所有类别在固定 0.25 阈值下的业务准确率。

| 类别 | 测试 AP@0.5 | 测试 recall |
| --- | --- | --- |
| hand | 92.92% | 89.73% |
| hatch_handle | 99.06% | 98.75% |
| water_gun | 99.16% | 98.08% |
| wafer_slot_empty | 98.14% | 97.07% |
| wafer_slot_filled | 62.07% | 50.41% |

已填充槽存在明显漏检，当前结果不能直接支持可靠计数。
例如 `frame_100779` 中标注了 7 个 filled 槽，默认置信度 0.25 的样例预测仅检出 1 个。
建议下一轮增加舱门关闭、玻璃反光等情况下的 filled 槽训练样本，并复核可见性和标注一致性。

模型：`I:/youyan/yolo_dataset/runs/yolov10s_detect_grouped/weights/best.pt`。
完整指标：同一运行目录的 `training_summary.json`。
测试图表：`I:/youyan/yolo_dataset/runs/yolov10s_detect_grouped_test2`。
预测图片：`I:/youyan/yolo_dataset/runs/yolov10s_detect_grouped_predictions2`。
初次图表生成缺少 seaborn；已安装 seaborn 0.13.2 并用相同权重和配置补绘。
两次测试指标一致，修复图表时没有调整模型或超参数。
原始 `_test` / `_predictions` 目录保留，最终完整交付使用上述带 `2` 的目录。
可进一步通过 `confusion_counts.json` 检查固定 confidence=0.25、IoU=0.45 下的混淆计数。

## 本次数据

原始图片和 X-AnyLabeling / LabelMe JSON 混放于 `I:/youyan/yolo_dataset/images`。
实际读取到 970 对图片与 JSON，图片尺寸与 JSON 一致。

检测类别顺序直接使用 `configs/detect_classes.yaml`：

| ID | 类别 | 中文 | 原始框数量 |
| --- | --- | --- | --- |
| 0 | hand | 手 | 889 |
| 1 | hatch_handle | 舱门把手 | 639 |
| 2 | water_gun | 水枪 | 391 |
| 3 | wafer_slot_empty | 空晶圆槽 | 2354 |
| 4 | wafer_slot_filled | 已填充晶圆槽 | 3848 |

`segment_classes.yaml` 的 `hatch` 含 168 个多边形，仅在转换报告中记录，本次检测不包含该类别。
原始 JSON 和图片保留不变；生成的 TXT 为归一化的 `class xc yc width height`。

最终使用 `I:/youyan/yolo_dataset/yolo_detect_grouped/dataset.yaml`。
按帧编号间隔大于 1000 推断三段序列，整段分配，不随机拆分相邻帧：

| 集合 | 图片数 | 帧编号范围 |
| --- | --- | --- |
| train | 628 | 200000–200627 |
| val | 200 | 00003–01397 |
| test | 142 | 100046–101317 |

每个集合都覆盖 5 类，检查了图片完整性、维度、框坐标、类别和跨集合相同文件内容。
帧号分组是基于文件名的推断，正式验收应使用已确认独立拍摄的视频。
如果有来源视频 ID，应通过 `--group-regex` 显式分组。
早期 `yolo_detect` 为随机划分的准备目录，已被本次 `yolo_detect_grouped` 取代。

## 转换

在 PowerShell 的项目根目录执行（输出目录必须是新的或为空）：

```powershell
& D:\Anaconda\envs\youyan311\python.exe industrial_inspection\tools\convert_labelme_to_yolo.py `
  --source I:\youyan\yolo_dataset\images `
  --output I:\youyan\yolo_dataset\yolo_detect_grouped `
  --classes industrial_inspection\configs\detect_classes.yaml `
  --ignored-classes industrial_inspection\configs\segment_classes.yaml
```

完整的划分清单、各类统计保存在数据集的 `preparation_report.json`。
未知类别、错误 shape 类型或无效框会报错，避免静默训练错误数据。

## 训练

使用现有 `youyan311` 环境、RTX 3090 和官方 THU-MIG YOLOv10s 权重。
权重下载来源：`https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt`。
启动脚本只用于信任的官方或自行训练的本地 checkpoint。
它兼容该旧仓库在 PyTorch 2.6+ 下的 checkpoint 加载方式；无需更换现有 PyTorch。

```powershell
& D:\Anaconda\envs\youyan311\python.exe -u industrial_inspection\tools\train_detection.py
```

默认配置：640 输入、batch 16、最多 150 epochs、30 epochs 无改善提前停止。
采用 AdamW、初始学习率 0.001、余弦退火；使用较温和的色彩和位置增强，禁用水平翻转。
Windows 下 workers=0，避免多进程加载造成的启动问题。
实际执行配置保存在运行目录的 `args.yaml`。

训练完成后脚本自动加载验证集选择的 `best.pt`，在 test 上评估一次，生成测试图表、
12 张测试样例的预测图以及 `training_summary.json`。
不要使用测试集分数反复调整超参数；改参数时以 val 为依据，最终验收另留独立视频。

输出根目录为 `I:/youyan/yolo_dataset/runs`。
初次正式运行目录为 `yolov10s_detect_grouped`；新训练若目录已存在会自动创建后缀目录。
当前运行日志为项目内 `industrial_inspection/training_resume.log`；
恢复前的日志保存在 `industrial_inspection/training.log`。

独立加载 YOLOv10 的 `best.pt` / `last.pt` 时应使用 `from ultralytics import YOLOv10`；
YOLOv8 权重应使用 `YOLO`。本仓库的通用 `YOLO` 类仅通过文件名识别 YOLOv10。
训练脚本已经显式使用 `YOLOv10`，视频脚本已按 checkpoint 实际模型架构自动选择加载类。
绘制混淆矩阵需要 seaborn（本次已在 `youyan311` 中安装 0.13.2）。

仅评估已完成的权重并生成样例：

```powershell
& D:\Anaconda\envs\youyan311\python.exe -u industrial_inspection\tools\train_detection.py `
  --evaluate-only I:\youyan\yolo_dataset\runs\yolov10s_detect_grouped\weights\best.pt
```

中断后恢复同一次训练（未完成且包含 optimizer 的 `last.pt`）：

```powershell
& D:\Anaconda\envs\youyan311\python.exe -u industrial_inspection\tools\train_detection.py `
  --resume I:\youyan\yolo_dataset\runs\yolov10s_detect_grouped\weights\last.pt
```

本模型识别手、把手、水枪和槽位空满状态。数量、放置顺序、舱门开关以及晶圆正反，
需要后续时序算法、其他标注或传感信息。

## 视频检测

推荐在编辑器中直接运行，无需终端。在 `tools/predict_video.py` 顶部修改“用户配置”：

```python
SOURCE_VIDEO = Path(r"I:\youyan\camera_B_20260918_112330.avi")
MODEL_WEIGHTS = Path(r"I:\youyan\yolo_dataset\runs\yolov10s_detect_grouped\weights\best.pt")
OUTPUT_ROOT = Path(r"I:\youyan\video_results")
MAX_FRAMES = 0        # 0 处理整段；300 先试跑 300 帧
SHOW_PREVIEW = True  # 显示检测窗口
```

在 PyCharm 中将项目解释器选为 `D:\Anaconda\envs\youyan311\python.exe`，
右键 `predict_video.py` → Run。运行配置的脚本参数留空。
配置中的输入路径必须指向具体视频文件，输出根目录可已存在；每次运行自动创建新的子目录。
无需手动设置工作目录。`CONFIDENCE`、`IMAGE_SIZE`、`DEVICE`、`FRAME_STRIDE` 也可在配置区修改。

可选的命令行用法仍保留，命令行参数会覆盖配置区默认值：

在 PowerShell 中执行下面命令。脚本默认加载本次训练的 `best.pt`，使用 GPU 0、
640 输入、confidence=0.25，并逐帧处理整个视频：

```powershell
Set-Location E:\GitHub\yolov10
& D:\Anaconda\envs\youyan311\python.exe industrial_inspection\tools\predict_video.py `
  --source "I:\youyan\camera_B_20260918_112330.avi"
```

将 `--source` 替换为需要检测的视频路径。默认结果保存在项目的
`industrial_inspection/video_results/<视频名>_<时间>/`，每次运行创建新目录。
也可通过 `--output-dir "I:/youyan/video_results/run01"` 指定一个尚不存在的目录。

输出文件：

- `annotated.mp4`：原始分辨率的检测框视频，不保留源视频音轨。
- `detections.jsonl`：每个已处理帧一行，含源帧号、时间、类别、置信度及原图像素坐标 `xyxy`。
  无检测的帧也会记录，源帧号从 0 开始；时间按源帧号 / FPS 计算，适用于固定帧率视频。
- `frame_counts.csv`：每帧各类检测框数量，方便查看检测变化；这不是去重后的晶圆数量或放入次数。
- `summary.json`：模型和视频路径、FPS、处理参数、处理帧数及结束状态。

常用可选参数：

| 参数 | 用途 |
| --- | --- |
| `--max-frames 300` | 仅处理 300 帧试跑；默认 0 表示整段 |
| `--show` | 显示实时预览，按 Q / Escape 停止 |
| `--conf 0.35` | 提高检测置信度门槛；降低可增加检出，也可能增加误报 |
| `--device cpu` | 使用 CPU，默认有 CUDA 时使用 GPU 0 |
| `--stride 2` | 每两帧处理一帧，输出 FPS 同时减半以保持播放时长近似一致 |
| `--weights ".../best.pt"` | 替换使用的可信本地 YOLOv10 权重 |

放置顺序、短时手部动作等时序任务建议保持默认 `--stride 1`。
槽位填充仍有明显漏检，需要复核视频结果；此脚本只执行检测和可视化。

已用 `camera_B_20260918_112330.avi` 处理 20 帧（stride=2）验证：
输出 MP4 可完整解码、帧率为源帧率的一半、源帧号为 0,2,...,38，JSON 和 CSV 数量一致。
这项检查验证视频处理流程，不代表整段视频检测精度验收。
试跑文件在 `industrial_inspection/video_results/smoke_video_01/`。

## 槽位重复框修正（2026-10-09）

检查当前视频配置发现 `MODEL_WEIGHTS` 已切到 YOLOv8n exp3.2，但原脚本仍固定使用
`YOLOv10` 加载，导致 YOLOv8 输出走不执行常规 NMS 的后处理。
已修复为自动检测 checkpoint 实际架构，支持两种权重文件名均为 `best.pt` 的情况。
保留用户的视频、权重、输出路径及 IMAGE_SIZE 设置；无需手动切换加载类。

正确加载也可能产生同一槽位的 empty / filled 两个高度重叠框：
YOLOv8 默认 NMS 按类别分别执行；YOLOv10 原生后处理不执行传统 NMS。
新的槽位专用去重只合并两个槽位类别，不对手、把手、水枪执行跨类抑制：

```python
NMS_IOU = 0.70           # YOLOv8 同类去重；YOLOv10 不使用此参数
SLOT_DEDUPLICATE = True
SLOT_DUPLICATE_IOU = 0.85
```

按置信度排序，对 IoU > 0.85 的槽位框保留高分者。该阈值保守，
不会解决所有偏移重复框，也不能确保保留者的空满状态正确。
将阈值大幅降低可能误删相邻的真实槽位，不建议为了让画面干净而直接压低。
可设 `SLOT_DEDUPLICATE=False` 查看去重前结果。
`detections.jsonl` 额外记录 `raw_box_count` 和 `suppressed_slots`，
`summary.json` 记录实际加载类、去重配置及抑制数量，方便复核。
视频、CSV、JSON 的最终检测结果同步使用去重后的框。

验证了 YOLOv8 第 1500 帧：9 个全部目标框变为 7 个，删除 2 个槽位冲突重复框；
YOLOv10 第 3500 帧：8 个变为 7 个，删除 1 个槽位同类重复框。
还验证了高度重叠的手不被槽位抑制、相邻槽位保留和空结果可处理。
前 20 帧视频试跑通过；需要重新运行整段视频才能获得修正后的完整视频。

## 后续模型改进优先级

1. 人工复核重复框、空满冲突和满槽漏检帧，一物一框；与模型预测比较时关注真实物理槽位，
   不把所有框重叠都当作重复。不可见或状态无法区分的槽位不要凭经验强行标 empty/filled。
2. 补充满槽、舱门关闭后透过玻璃观察、反光、手部遮挡和操作变化的独立视频片段。
   从不同状态抽帧，避免连续相似帧占据训练集；模型预标注必须人工逐框复核。
3. YOLOv10 当前在 640 训练：若试验 1280，应重新训练并在独立验证片段上验证，
   不应将现有 640 权重的推理分辨率直接改成 1280 期待精度提升。
   exp3.2 已在 1280 训练，可以用该分辨率先做视频比较。
4. 面向计数，优先拆成稳定槽位定位（统一 wafer_slot 类或固定 ROI）与局部 empty/filled/unknown
   分类，每个 slot_id 只输出一个状态，加入遮挡判断和多帧确认。
   槽位/门体会移动时需要跟随定位或门体关键点标定，不能一直复用静态像素坐标。
5. 留出经人工完整复核的新视频作验收集，评价满槽召回、空满误判、重复框数、
   数量完全正确率和状态变化延迟。不能将新视频的模型预标注直接当作准确率真值。
