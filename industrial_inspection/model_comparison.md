# exp3.2 与 YOLOv10s 对比（2026-10-09）

目前建议优先试用 exp3.2，但未完成共同独立测试集上的精度验收，不能确认泛化能力一定更强。

## 模型与数据

| 项目 | exp3.2 | 本次 YOLOv10s |
| --- | --- | --- |
| 架构 | YOLOv8n | YOLOv10s |
| 训练输入 | 1280 | 640 |
| 训练图片 | 776 | 628 |
| 验证图片 | 194 | 200 |
| 训练轮数 | 100 | 49（早停） |
| 最佳权重 | C:/Users/win10/xanylabeling_data/trainer/ultralytics/runs/detect/exp3.2/weights/best.pt | I:/youyan/yolo_dataset/runs/yolov10s_detect_grouped/weights/best.pt |

类别顺序完全相同：hand、hatch_handle、water_gun、wafer_slot_empty、wafer_slot_filled。
架构、输入分辨率、样本及数据划分不同，结果差异不能仅归因于 YOLO 版本。

通过图片内容 SHA-256 对照确认，YOLOv10s 的 142 张测试图片，
有 113 张曾用于 exp3.2 训练、29 张用于 exp3.2 验证。
所以下面的共同图片评估是诊断，不能作为公平的独立泛化排名。

## 共同 142 张图片的诊断结果

用本仓库同一评估框架、GPU 0、batch=8、confidence=0.001、IoU=0.7、max_det=300。
通过原图 pixel/JSON 坐标计算检测指标。

| 模型 / 输入 | mAP@0.5 | mAP@0.5:0.95 | filled recall |
| --- | --- | --- | --- |
| exp3.2 / 640 | 99.15% | 79.07% | 97.70% |
| YOLOv10s / 640 | 90.27% | 59.01% | 50.41% |
| exp3.2 / 1280 | 99.45% | 86.78% | 97.89% |
| YOLOv10s / 1280 | 53.76% | 26.08% | 29.27% |

这里的 filled recall 是框架选定的评估工作点值，不是固定 confidence=0.25 的视频业务召回。
exp3.2 见过这些图片，分数偏乐观；没有独立真值不能把这里的 97.89% 当作新视频准确率。
YOLOv10s 在 640 训练，直接将其推理改成 1280 并不能保证提高精度，本次实测退化。

## 新视频抽查

抽查 `I:/youyan/test_dataset/camera_B_20260918_144258.avi` 的
233、1000、1500、2500、3500、4500、6500、7500 帧。
两模型均用 confidence=0.25，各自分别在 640、1280 推理。
未更改视频、权重及标签；生成的对比图左侧为 exp3.2，右侧为 YOLOv10s。

- 640 下，手、把手和水枪的样例检出总体相近。
- 第 1500 帧 exp3.2 检出 3 个 empty 与 4 个 filled，YOLOv10s 检出 0 个 empty 与 4 个 filled。
- 第 2500 帧 exp3.2 检出 6 个 empty 与 2 个 filled，YOLOv10s 检出 3 个 empty 与 2 个 filled。
- 第 6500、7500 帧，两者均检出 1 个 empty、6 个 filled。
- 舱门关闭的第 233、1000 帧，两者 640 下都没有检出槽位；需要根据可见性和业务要求决定是否允许输出 unknown。
- YOLOv10s 在 1280 下出现新增误检和漏检，exp3.2 在 1280 下的这些样例更合理。

这些是抽样目视观察和框数量差异，没有人工复核真值，因此“更多框”不自动等于“更准确”。
目录中已有的部分新视频 JSON 含模型 score 且 checked=false；未经确认人工复核前不能用作独立真值。

## 使用建议

当前视频检测优先试用 exp3.2，1280 作为首选输入，640 可用来比较速度与细节损失。
使用 exp3.2 时需要 `from ultralytics import YOLO` 和 `YOLO(weights)`；
当前 predict_video.py 显式使用 YOLOv10，所以不能仅替换模型路径而保持加载类不变。
本次对比没有修改用户当前的视频脚本配置或切换默认模型。

正式选择前，人工完整复核另一段独立视频的代表性帧，标注所有可见目标，
再用同一组图片、同样的置信度和匹配规则比较：filled recall、误检、空满误判、数量完全正确率。
可分别按推荐分辨率测试，再按相同输入分辨率测试，区分配置优势与架构差异。

## 文件

- 原始指标和样例框：`industrial_inspection/video_results/model_comparison/comparison.json`。
- 同分辨率对比图：`video_comparison_640_1.jpg`、`video_comparison_640_2.jpg`，1280 同名。
- 每个样例的大图：`<模型>_<输入>_frame_<帧号>.jpg`。
- 完整过程日志：`industrial_inspection/model_comparison.log`。
- 重跑入口：`industrial_inspection/tools/compare_detection_models.py`。

脚本用于本地诊断；它的报告明确注明共同图片被 exp3.2 使用的限制。
