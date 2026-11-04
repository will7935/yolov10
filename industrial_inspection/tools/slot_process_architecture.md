# 工艺检测配置与模块边界

算法参数统一修改 `slot_process_config.py`，保存后重启预测。视频路径、模型、
通用 `--conf` 和审计下限仍由 `predict_slot_groups.py` / 批处理入口设置。
单视频、UI和批处理共用算法配置。旧入口仍暴露原参数名，兼容已有调用和测试。

## 置信度按职责设置

| 参数 | 作用 | 不直接控制的环节 |
| --- | --- | --- |
| `SLOT_STATE_CONFIDENCE` | 槽位空满观测准入 | 手部、把手、ByteTrack高分门槛 |
| `HAND_DETECTION_CONFIDENCE` | 手部遮挡和S4接触证据准入 | 槽框准入、把手准入 |
| `HATCH_HANDLE_CONFIDENCE` | 把手准入与开关门监测 | 手部、槽框准入 |
| `BYTE_TRACK_HIGH_CONFIDENCE` | 槽位轨迹建立及高分关联 | 空满证据门槛 |
| `BYTE_TRACK_LOW_CONFIDENCE_OVERRIDE` | ByteTrack低分关联 | 空满确认帧数 |
| 入口 `--conf` | 通用显示门槛、其他类别准入，以及上述未指定参数的兼容默认值 | 已独立配置的职责门槛 |

新参数默认 `None`，完整保留旧的 `--conf` 默认行为。需要独立调节某个职责时，
把对应 `None` 改为数值，例如只调整手部证据：

```python
HAND_DETECTION_CONFIDENCE = 0.05
```

默认跟踪低分门槛仍为 `min(BYTE_TRACK_LOW_CONFIDENCE, 跟踪高分门槛/2)`。
填写 `BYTE_TRACK_LOW_CONFIDENCE_OVERRIDE` 后使用该数值，必须小于跟踪高分门槛。
审计下限必须不高于任何使用门槛，否则相应证据在模型推理时已经被过滤，程序会报错。

门槛职责分离不代表结果完全独立：改变手部证据仍会合理地改变遮挡判定，
改变跟踪准入仍可能影响后续槽位身份。本次只消除共用参数的隐式覆盖，不改变这些证据依赖。

## 补充算法的权限

| 模块 | 输入与输出 | 开关/参数 |
| --- | --- | --- |
| `slot_detection_policy.py` | 检测行 → 各类业务证据及准入原因 | 上述独立置信度 |
| `s4_hand_contact.py` | 槽位几何、手框 → 当前帧接触证据及拒绝原因；不修改槽位、不计时、不产生事件 | `LOAD_HAND_CONTACT_*`、`S4_HAND_REGION_*` |
| `slot_local_redetection.py` | 漏检位置、局部图 → 补充观测；不修改身份和状态 | `LOCAL_REDETECT_*` |
| `slot_group_tracker.py` | 关联、接触累计、多帧空满确认 → 上/下料事件 | `MATCH_DISTANCE_RATIO`、`MIN_CONFIRM_FRAMES` 等 |
| `roi_group_sequence.py` | 轨迹、ROI出口证据 → 组切换、历史基线恢复 | `EXIT_*`、`NEXT_GROUP_*` 等 |

三点初始化的开关仍为 `THREE_POINT_GROUP_INITIALIZATION`。几何跟随和S4恢复关联规则
保持原实现，本次不放宽关联门限，也不修改工艺顺序与完成判定。

S4门槛仍为连续2个有效处理帧、最低秒数0。接触只建立待核验证据，YOLO直接状态变化优先，
持续无法观测时仍使用原有等待及基线推断路径。其他槽位不会因此增加手部事件。
`LOAD_HAND_CONTACT_COMPETING_SLOT_ENABLED=False` 仅关闭竞争面积抑制；
覆盖其他槽中心的歧义检查独立保留，其拒绝原因为 `other_slot_center_covered`。

## 查看实际生效配置与原因

- `summary.json` 的 `algorithm_configuration` 保存算法配置；
  `effective_detection_thresholds` 保存解析后的实际置信度。
- `detection_confidences.csv` 新增 `business_confidence_threshold`、`business_admitted`、
  `business_admission_reason`。它们表示进入业务流程前的置信度/轨迹ID准入，之后仍可能被ROI、
  旧组隔离、关联或遮挡规则拒绝。原显示字段含义保持不变。
- `slot_states.jsonl` 和 `slot_states.csv` 新增 `s4_contact_diagnostic`。
  `decision_reason` 解释当前帧几何证据；`contact_accepted` 表示是否可用于接触累计；
  `accumulation_status` 区分收集、待确认、离手核验、中断清零和事件确认。
- `s4_region_diagnostic.status=eligible` 继续表示顺序门控允许，兼容旧结果读取程序。
  必须结合 `contact_accepted`、`contact_count` 和 `evidence_ready` 判断是否形成有效接触。

已有输出不会被更新；重新预测才会生成上述字段。
