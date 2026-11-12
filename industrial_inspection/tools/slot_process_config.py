"""工艺算法配置：按职责分组，供单视频和批处理入口共用。

修改本文件后重新启动预测。视频路径、模型和通用显示门槛仍在 predict_slot_groups.py。
None 的置信度沿用入口 --conf，保留整理前行为；填具体数值后仅覆盖对应职责。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 检测证据与跟踪置信度；审计下限 AUDIT_CONFIDENCE 须不高于任一使用门槛。
SLOT_STATE_CONFIDENCE = None       # 空满状态证据；使用跟踪后的置信度
HAND_DETECTION_CONFIDENCE = None   # 手框证据；用于遮挡与S4接触，不控制槽框/把手
HATCH_HANDLE_CONFIDENCE = None     # 把手证据；只控制开关门监测
BYTE_TRACK_HIGH_CONFIDENCE = None # 槽位轨迹建立/高分关联，不控制空满状态确认门槛
BYTE_TRACK_LOW_CONFIDENCE_OVERRIDE = None # None保留旧自动下限；数值则作为独立低分关联门槛

# 分组、舱门与工艺顺序
CURRENT_GROUP_ID = 1          # 视频起始实际组号，1..5；之后按确认的换组次数递增
PROCESS_PHASE = "unload"     # 起始工序；默认下料五组后自动上料，纯上料片段可设"load"
AUTO_LOAD_AFTER_UNLOAD = True # 下料G5退出后自动回到G1开始上料；上料G5退出后结束
HATCH_CONTROL_ENABLED = True # 开门后启动现有上下料流程；关门结束并核验本轮20槽
# 本相机把手框中心的归一化位置；换机位时须重新标定，不能把漏检当作关门。
HATCH_OPEN_POLYGON = ((.39,.01),(.495,.01),(.495,.18),(.39,.18))
HATCH_CLOSED_POLYGON = ((.495,.18),(.72,.18),(.72,.49),(.495,.49))
HATCH_CONFIRM_SECONDS = .3
HATCH_MIN_CONFIRM_FRAMES = 3
HATCH_MAX_GAP_SECONDS = .25
# 先用关门位置建立基准，再按归一化中心距离判断。
# 已确认关门后，漏检开门的等待时间从最后一次关门位置观测起算。
HATCH_DISTANCE_MODE = True
HATCH_CLOSED_DISTANCE_MAX = .08
HATCH_OPEN_DISTANCE_MIN = .14
HATCH_MISSING_OPEN_SECONDS = 5.0
HATCH_REFERENCE_UPDATE_ALPHA = .05
CLOCKWISE_ORDER = (1,3,4,2)    # 按实物确认后填写；允许从任意槽开始
DISPLAY_ABSOLUTE_SLOT_NUMBERS = False  # True：显示1–20；False：每组显示S1–S4
GROUP_TRACKING_ENABLED = True
AUTO_GROUP_SWITCH = True      # 单点真实越界即切到下一组并逐点记录；False保持单组模式
EXIT_CONFIRM_SECONDS = .1     # 仅多点换组模式使用；单点换组立即关闭旧组
EXIT_BOUNDARY_MARGIN_PX = 4   # 边界内/外容差，避免中心在线上抖动被反复计数
EXIT_MIN_SLOTS = 1            # 任意一个已编号槽实际越出ROI即触发换组
PARTIAL_EXIT_MIN_SLOTS = 1    # 单点换组模式下同样只需要一个越界槽
EXIT_MIN_MOTION_RATIO = .5    # 多点换组规则的组中心位移比例；单点换组时不参与触发
MOTION_PAUSE_RATIO = .18      # 多点换组规则的共同外移门限；单点换组时不参与触发
NEXT_GROUP_STABLE_SECONDS = .1 # 收齐可靠轨迹且几何停稳后一次性建立S号映射；此前仅按ByteTrack ID记录
NEXT_GROUP_STABILITY_RATIO = .1 # 停稳期间各槽偏离起始位置不得超过最小槽间距×此比例
NEXT_GROUP_MATCH_DISTANCE_PX = 40 # 新点/旧组残留点逐帧匹配的最大距离（像素）
NEXT_GROUP_ENTRY_FRACTION = .6 # 沿EXIT_DIRECTION从入口到出口的前60%允许建立新轨迹
REJECT_TRANSITION_ROI_DETECTIONS = True # 换组时隔离有稳定旧组证据的未绑定ID，并启用短时空间隔离
TRANSITION_QUARANTINE_MIN_FRAMES = 5 # 未绑定ID在旧组ROI内累计真实出现至少此帧数，才持续隔离
OUTSIDE_MAPPING_QUARANTINE_SECONDS = 3.0 # 新组映射后短时间内，ROI外新ID不参与当前组判断
SLOT2_BELOW_SLOT3 = True      # 左上S1、顺时针1→3→4→2，y辅助核验S2通常在S3下方；编号后不重排
MACHINE_CENTER_NORMALIZED = (637 / 640, 266 / 480) # 机心点选坐标；四槽证据充分时最近点为S4，None关闭
EXIT_DIRECTION = (0, -1)     # 本视频已核实旧组从右下ROI向左上退出；None可恢复自动方向判断

# 每轮五组按顺序计数；自动模式运行五组下料再五组上料。
START_FRAME = None            # None：从第0帧开始；数字：指定当前组开始的源帧号
END_FRAME = None              # 不包含该帧；例如1800表示到1799帧，None表示视频结尾
INITIALIZATION_SECONDS = .1   # 候选四槽已分别可靠看见，至少三槽持续支持稳定几何后启用编号
MIN_INITIALIZATION_FRAMES = 2 # 同时满足最少帧数；编号后不再按x重新排序
INITIAL_GROUP_STABLE_SECONDS = .3  # G1正式建立S号前的独立几何稳定时间
INITIALIZATION_HISTORY_SECONDS = 5.0 # 基线及可靠临时轨迹有效期；未获多帧支持的点按IDENTITY_LOST_SECONDS清理
THREE_POINT_GROUP_INITIALIZATION = True  # 三个真实槽位可推算缺失S4后建立组几何
S4_USE_PREVIOUS_GROUP_LAYOUTS = True  # 下料逐点换组时，优先采用一致的前组实测四槽布局推算S4
# G1下料在首个可信下料事件前只作为候选；若候选先离开ROI则自动释放，不占用G1。
REQUIRE_FIRST_UNLOAD_EVENT_FOR_G1 = True


# 槽位关联、空满状态确认
MATCH_DISTANCE_RATIO = .46    # 匹配门限=最近两槽间距×此比例；放宽以容纳空/满框中心偏移，必须小于0.5
CONFIRM_SECONDS = 0.0          # 槽位空/满状态不再设置时间门槛，只按MIN_CONFIRM_FRAMES确认
MIN_CONFIRM_FRAMES = 4
IDENTITY_LOST_SECONDS = 1.2   # 连续无匹配超时，锁定为unknown，避免下一组继承旧编号
CONFLICT_CONF_MARGIN = .15    # 同位置空/满分数差小于该值时，输出unknown
HAND_OVERLAP_RATIO = .2       # 手遮挡槽框达到此比例，或覆盖中心时，暂停该槽状态确认

# S4手部接触补偿：只提供操作证据，由槽位状态机确认事件
LOAD_HAND_CONTACT_ENABLED = True       # 仅S4的手接触可补足本工序的unknown空满变化
# 以下接触门槛仅用于S4，上料和下料均适用，与MIN_CONFIRM_FRAMES的空满视觉确认门槛独立。
# 有效接触须先通过手框置信度、接触位置及歧义判定；未累计满2帧前中断会清零重计。
# 连续2帧有效接触后仅进入待确认，手离开后优先用YOLO确认；持续漏检时按已有空满基准推断变化。
LOAD_HAND_CONTACT_SECONDS = 0.0       # 单位：秒；0.0表示取消最低接触时长，仅检查下方连续帧数
LOAD_HAND_CONTACT_MIN_FRAMES = 2      # 单位：处理帧；连续2帧有效接触即可，不要求额外持续0.1秒
LOAD_HAND_CONTACT_OVERLAP_RATIO = .03  # 手框覆盖S4检测框面积比例；中心覆盖仍优先
LOAD_HAND_CONTACT_COMPETING_SLOT_ENABLED = False  # 启用竞争槽位抑制：其他槽位明显占优时不触发S4手接触
LOAD_HAND_CONTACT_COMPETING_SLOT_ID = 2          # 竞争槽位编号；当前使用S2作为与S4比较的槽位
LOAD_HAND_CONTACT_COMPETING_MARGIN = .10         # 竞争槽位的最低重叠率；S2重叠率低于此值不抑制S4
LOAD_HAND_CONTACT_COMPETING_RATIO = 1.0          # 竞争倍数；S2重叠率至少是S4的该倍数才抑制S4
S4_CONTACT_PREDICTION_SECONDS = .8               # S4漏检时临时接触框的最长有效时长
S4_PREDICTION_CONTACT_OVERLAP_RATIO = .15         # 仅靠预测框建立待视觉核验时的最小覆盖比例
S4_HAND_REGION_ENABLED = True  # 手框进入手动选定的区域时，作为S4接触线索
S4_HAND_REGION_RESELECT = False  # True：每次启动重新选择；False：优先读取已保存区域
S4_HAND_REGION_FILE = ROOT / "industrial_inspection" / "configs" / "s4_hand_region.json"
ROI_TRACKING_MARGIN_PX = 40   # 已编号槽的ROI外跟踪距离
ROI_EDGE_EVIDENCE_PX = 8.0   # 640×480下的ROI外视觉取证距离，0表示关闭

# 局部补检：只补观测，不修改身份和工艺结论
LOCAL_REDETECT_ENABLED = False # 已编号槽被整图漏检时，用同一模型局部补检
LOCAL_REDETECT_CROP_SIZE = (180, 140) # 原图裁剪宽、高（像素），保留槽周围背景
LOCAL_REDETECT_IMAGE_SIZE = 320 # 局部图送入YOLO时的尺寸；分别沿用槽位/手部证据门槛
ROTATION_INTERVALS = []       # 可选源视频秒数[(开始,结束),...]；期间跟踪位置、暂停空满确认
SHOW_SLOT_TEXT = True         # 正式编号后在点旁标注S1-S4；初始化阶段在ByteTrack框上标状态
SHOW_ALL_DETECTION_BOXES = True  # 显示审计下限以上的槽位真实检测框，不改变跟踪筛选
SHOW_NON_SLOT_DETECTION_BOXES = True  # 显示手等非槽位类别的检测框
# 工序状态面板的位置；可选top_left或top_right。
STATUS_PANEL_POSITION = "top_right"

# ByteTrack轨迹、置信度平滑与重新识别
BYTE_TRACK_SLOTS_ENABLED = True     # 用ByteTrack跟踪空槽/满槽框，并映射到正式S号
BYTE_TRACK_LOW_CONFIDENCE = .10     # 旧自动下限的上限：min(本值,高分门槛/2)；独立下限用顶部OVERRIDE
BYTE_TRACK_BUFFER_SECONDS = 3.0
TRACK_CONFIDENCE_SMOOTHING_ENABLED = False
TRACK_CONFIDENCE_SMOOTHING_WINDOW = 1  # K-n到K共n+1帧；5帧约等于30FPS下0.17秒
TRACK_CONFIDENCE_PREASSOCIATION_IOU = .30  # 预关联只用于找历史，不直接替代ByteTrack匹配
REIDENTIFICATION_ENABLED = True
REIDENTIFICATION_MAX_GAP_SECONDS = 15.0
REIDENTIFICATION_MAX_DISTANCE_PX = 40.0


def snapshot_algorithm_config(namespace):
    """保存入口实际使用的参数，包括兼容旧调用方在入口覆写的值。"""
    def serializable(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (tuple, list)):
            return [serializable(item) for item in value]
        return value

    return {name: serializable(namespace.get(name, default))
            for name, default in globals().items() if name.isupper() and name != "ROOT"}
