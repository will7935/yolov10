"""运行此文件：按ROI越界自动换组，输出五组槽号、空满及顺时针结果。"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import cv2
import numpy as np
import torch

from industrial_inspection.tools import predict_video as base
from industrial_inspection.tools.detection_display import draw_detection_points
from industrial_inspection.tools.slot_group_tracker import SLOT_STATES, SlotGroupTracker
from industrial_inspection.tools.slot_track_display import slot_track_label, slot_track_record
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_roi import draw_roi, pixel_polygon
from industrial_inspection.tools.slot_byte_tracker import SlotByteTracker
from industrial_inspection.tools.hatch_loading_inspection import (
    HatchHandleMonitor, HatchLoadingInspection, format_fullness_report)


# =============================================================================
# 在这里改配置，直接在 PyCharm 运行，无需终端；原 predict_video.py 不受影响。
# 视频和 ROI 沿用旧配置；本入口使用 exp3.3 槽位模型。
# =============================================================================
SOURCE_VIDEO = base.SOURCE_VIDEO
MODEL_WEIGHTS = Path(r"C:\Users\win10\xanylabeling_data\trainer\ultralytics\runs\detect\exp3.3\weights\best.pt")
OUTPUT_ROOT = base.OUTPUT_ROOT
CONFIDENCE = base.CONFIDENCE
AUDIT_CONFIDENCE = getattr(base, "AUDIT_CONFIDENCE", .001)  # CSV审计下限，不改变显示/状态门槛
IMAGE_SIZE = base.IMAGE_SIZE
DEVICE = base.DEVICE
UI_ENABLED = True             # True：沿用 UI7V；False：OpenCV 预览/批处理
SHOW_PREVIEW = True           # 仅 UI_ENABLED=False 时生效；Q退出、空格暂停
MAX_FRAMES = 0                # 处理整段视频，避免在自动上料前被试跑帧数截断
FRAME_STRIDE = 1
SLOT_ROI_FILE = base.SLOT_ROI_FILE  # 默认第一次ROI；可改成需要读取或保存的JSON配置路径
ROI_RESELECT = base.ROI_RESELECT   # True：手动划分并保存；False：读取配置，也可在界面切换
ROI_SELECT_FRAME = base.ROI_SELECT_FRAME

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
# 把手位置会因视频机位和门体姿态改变。先用稳定关门位置建立基准，
# 再按归一化中心距离判断；长时间完全漏检表示把手已离开画面。
HATCH_DISTANCE_MODE = True
HATCH_CLOSED_DISTANCE_MAX = .08
HATCH_OPEN_DISTANCE_MIN = .14
HATCH_MISSING_OPEN_SECONDS = 1.0
HATCH_REFERENCE_UPDATE_ALPHA = .05
CLOCKWISE_ORDER = (1,3,4,2)    # 按实物确认后填写；允许从任意槽开始
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
REJECT_TRANSITION_ROI_DETECTIONS = True # 旧组切换瞬间ROI内的轨迹ID不参与下一组映射
SLOT2_BELOW_SLOT3 = True      # 左上S1、顺时针1→3→4→2，y辅助核验S2通常在S3下方；编号后不重排
EXIT_DIRECTION = (0, -1)     # 本视频已核实旧组从右下ROI向左上退出；None可恢复自动方向判断

# 每轮五组按顺序计数；自动模式运行五组下料再五组上料。
START_FRAME = None            # None：从第0帧开始；数字：指定当前组开始的源帧号
END_FRAME = None              # 不包含该帧；例如1800表示到1799帧，None表示视频结尾
INITIALIZATION_SECONDS = .1   # 候选四槽已分别可靠看见，至少三槽持续支持稳定几何后启用编号
MIN_INITIALIZATION_FRAMES = 2 # 同时满足最少帧数；编号后不再按x重新排序
INITIAL_GROUP_STABLE_SECONDS = .3  # G1正式建立S号前的独立几何稳定时间
INITIALIZATION_HISTORY_SECONDS = 5.0 # 基线及可靠临时轨迹有效期；未获多帧支持的点按IDENTITY_LOST_SECONDS清理
THREE_POINT_GROUP_INITIALIZATION = True  # 三个真实槽位可推算缺失S4后建立组几何

MATCH_DISTANCE_RATIO = .46    # 匹配门限=最近两槽间距×此比例；放宽以容纳空/满框中心偏移，必须小于0.5
CONFIRM_SECONDS = 0.0          # 槽位空/满状态不再设置时间门槛，只按MIN_CONFIRM_FRAMES确认
MIN_CONFIRM_FRAMES = 4
IDENTITY_LOST_SECONDS = 1.2   # 连续无匹配超时，锁定为unknown，避免下一组继承旧编号
CONFLICT_CONF_MARGIN = .15    # 同位置空/满分数差小于该值时，输出unknown
HAND_OVERLAP_RATIO = .2       # 手遮挡槽框达到此比例，或覆盖中心时，暂停该槽状态确认
LOAD_HAND_CONTACT_ENABLED = True       # 仅S4的手接触可补足本工序的unknown空满变化
LOAD_HAND_CONTACT_SECONDS = .1        # 手在同一槽位的最短连续接触时间
LOAD_HAND_CONTACT_MIN_FRAMES = 3       # 接触至少需要的连续检测帧数
LOAD_HAND_CONTACT_OVERLAP_RATIO = .03  # 手框覆盖S4检测框面积比例；中心覆盖仍优先
LOAD_HAND_CONTACT_COMPETING_SLOT_ENABLED = True  # 启用竞争槽位抑制：其他槽位明显占优时不触发S4手接触
LOAD_HAND_CONTACT_COMPETING_SLOT_ID = 2          # 竞争槽位编号；当前使用S2作为与S4比较的槽位
LOAD_HAND_CONTACT_COMPETING_MARGIN = .10         # 竞争槽位的最低重叠率；S2重叠率低于此值不抑制S4
LOAD_HAND_CONTACT_COMPETING_RATIO = 1.0          # 竞争倍数；S2重叠率至少是S4的该倍数才抑制S4
ROI_TRACKING_MARGIN_PX = 40   # ROI外只跟踪当前已编号槽；区域外框不确认空满、不初始化
LOCAL_REDETECT_ENABLED = True # 已编号槽被整图漏检时，用同一模型局部补检
LOCAL_REDETECT_CROP_SIZE = (180, 140) # 原图裁剪宽、高（像素），保留槽周围背景
LOCAL_REDETECT_IMAGE_SIZE = 320 # 局部图送入YOLO时的尺寸；置信度沿用整图配置
ROTATION_INTERVALS = []       # 可选源视频秒数[(开始,结束),...]；期间跟踪位置、暂停空满确认
DRAW_SLOT_ROI = base.DRAW_SLOT_ROI
SHOW_SLOT_TEXT = True         # 正式编号后在点旁标注S1-S4；初始化阶段在ByteTrack框上标状态
SHOW_ALL_DETECTION_BOXES = True  # 显示审计下限以上的槽位真实检测框，不改变跟踪筛选
SHOW_NON_SLOT_DETECTION_BOXES = False  # 隐藏手等非槽位类别的检测框
POINT_RADIUS = base.POINT_RADIUS
BYTE_TRACK_SLOTS_ENABLED = True     # 用ByteTrack跟踪空槽/满槽框，并映射到正式S号
BYTE_TRACK_LOW_CONFIDENCE = .10     # 低分框只参与ByteTrack二阶段关联，不进入工艺状态机
BYTE_TRACK_BUFFER_SECONDS = 3.0
TRACK_CONFIDENCE_SMOOTHING_ENABLED = False
TRACK_CONFIDENCE_SMOOTHING_WINDOW = 1  # K-n到K共n+1帧；5帧约等于30FPS下0.17秒
TRACK_CONFIDENCE_PREASSOCIATION_IOU = .30  # 预关联只用于找历史，不直接替代ByteTrack匹配
REIDENTIFICATION_ENABLED = True
REIDENTIFICATION_MAX_GAP_SECONDS = 15.0
REIDENTIFICATION_MAX_DISTANCE_PX = 40.0
# =============================================================================
STATE_TEXT = {"empty": "空槽","filled": "已填充","unknown": "未知"}
GROUP_TEXT = {"initializing": "等待四槽初始化","active": "操作中","exiting": "正在退出",
              "entering": "新组逐槽跟踪中",
              "waiting_next": "等待下一组","stabilizing_next": "下一组停稳确认中",
              "sequence_complete": "本工序结束(G5)","identity_uncertain": "组身份待确认"}


def build_args(argv=None):
    defaults = ["--source",str(SOURCE_VIDEO),"--weights",str(MODEL_WEIGHTS),
        "--conf",str(CONFIDENCE),"--audit-conf",str(AUDIT_CONFIDENCE),
        "--imgsz",str(IMAGE_SIZE),"--device",str(DEVICE),
        "--ui" if UI_ENABLED else "--no-ui", "--show" if SHOW_PREVIEW else "--no-show",
        "--max-frames",str(MAX_FRAMES),"--stride",str(FRAME_STRIDE),
        "--roi-file",str(SLOT_ROI_FILE or base.DEFAULT_SLOT_ROI_FILE),
        "--reselect-roi" if ROI_RESELECT else "--no-reselect-roi",
        "--roi-select-frame",str(ROI_SELECT_FRAME)]
    args = base.parse_args(defaults + list(argv or []))
    args.slot_roi = True
    args.group_completion_log_only = True
    if not GROUP_TRACKING_ENABLED:
        raise ValueError("本脚本专用于四槽跟踪；普通检测请运行 predict_video.py")
    if not args.source.is_file() or not args.weights.is_file():
        raise FileNotFoundError("请检查本脚本配置的视频和权重路径")
    audit_conf = getattr(args, "audit_conf", AUDIT_CONFIDENCE)
    if (not 0 < args.conf < 1 or not 0 < audit_conf <= args.conf
            or args.stride < 1 or args.max_frames < 0 or args.imgsz < 32):
        raise ValueError("置信度、audit-conf、stride、max_frames 或 imgsz 配置错误")
    if START_FRAME is not None and START_FRAME < 0 or END_FRAME is not None and END_FRAME < 1:
        raise ValueError("起止帧号必须为有效非负帧号，结束帧大于0")
    for start,end in ROTATION_INTERVALS:
        if not 0 <= start < end:
            raise ValueError("ROTATION_INTERVALS中每个区间必须满足0<=开始<结束")
    if ROI_TRACKING_MARGIN_PX < 0:
        raise ValueError("ROI缓冲宽度不能为负")
    if TRACK_CONFIDENCE_SMOOTHING_WINDOW < 1:
        raise ValueError("置信度滑动窗口必须至少为1帧")
    if (REIDENTIFICATION_MAX_GAP_SECONDS <= 0
            or REIDENTIFICATION_MAX_DISTANCE_PX <= 0):
        raise ValueError("重新识别的最大时间和最大距离必须为正数")
    if (len(LOCAL_REDETECT_CROP_SIZE) != 2 or any(v < 32 for v in LOCAL_REDETECT_CROP_SIZE)
            or LOCAL_REDETECT_IMAGE_SIZE < 32):
        raise ValueError("局部补检裁剪宽高和推理尺寸须至少为32像素")
    if args.device == "auto":
        args.device = "0" if torch.cuda.is_available() else "cpu"
    return args


def prepare_context(args):
    """GUI主线程只准备ROI及视频信息，不再点选或读取四槽中心。"""
    polygon = base.prepare_slot_roi(args)
    cap = cv2.VideoCapture(str(args.source))
    try:
        ok,frame = cap.read()
        if not ok:
            raise RuntimeError("无法读取视频")
        h,w = frame.shape[:2]
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    start_frame = 0 if START_FRAME is None else START_FRAME
    if not 0 <= start_frame < total or END_FRAME is not None and END_FRAME <= start_frame:
        raise ValueError("开始帧超出视频范围，或结束帧不大于开始帧")
    make_sequence(polygon,(w,h))  # 提前验证配置，推理线程中才开始收集YOLO中心。
    return {"polygon": polygon,"size": (w,h),"start_frame": start_frame,
            "roi_file": str(Path(args.roi_file or base.DEFAULT_SLOT_ROI_FILE).resolve())}


def make_tracker():
    return SlotGroupTracker(group_id=CURRENT_GROUP_ID,phase=PROCESS_PHASE,
        clockwise=CLOCKWISE_ORDER,match_distance_ratio=MATCH_DISTANCE_RATIO,confirm_seconds=CONFIRM_SECONDS,
        minimum_frames=MIN_CONFIRM_FRAMES,lost_seconds=IDENTITY_LOST_SECONDS,
        conflict_margin=CONFLICT_CONF_MARGIN,hand_overlap=HAND_OVERLAP_RATIO,
        initialization_seconds=INITIALIZATION_SECONDS,initialization_frames=MIN_INITIALIZATION_FRAMES,
        geometry_initialization_seconds=INITIAL_GROUP_STABLE_SECONDS,
        initialization_history_seconds=INITIALIZATION_HISTORY_SECONDS,
        load_hand_contact_enabled=LOAD_HAND_CONTACT_ENABLED,
        load_hand_contact_seconds=LOAD_HAND_CONTACT_SECONDS,
        load_hand_contact_min_frames=LOAD_HAND_CONTACT_MIN_FRAMES,
        load_hand_contact_overlap_ratio=LOAD_HAND_CONTACT_OVERLAP_RATIO,
        load_hand_contact_competing_slot_enabled=LOAD_HAND_CONTACT_COMPETING_SLOT_ENABLED,
        load_hand_contact_competing_slot_id=LOAD_HAND_CONTACT_COMPETING_SLOT_ID,
        load_hand_contact_competing_margin=LOAD_HAND_CONTACT_COMPETING_MARGIN,
        load_hand_contact_competing_ratio=LOAD_HAND_CONTACT_COMPETING_RATIO,
        slot2_below_slot3=SLOT2_BELOW_SLOT3,
        allow_three_point_initialization=THREE_POINT_GROUP_INITIALIZATION)


def make_sequence(polygon,size):
    return RoiGroupSequence(make_tracker(),pixel_polygon(polygon,size),enabled=AUTO_GROUP_SWITCH,
        buffer_px=ROI_TRACKING_MARGIN_PX,boundary_margin_px=EXIT_BOUNDARY_MARGIN_PX,
        exit_seconds=EXIT_CONFIRM_SECONDS,exit_min_slots=EXIT_MIN_SLOTS,
        partial_exit_slots=PARTIAL_EXIT_MIN_SLOTS,minimum_motion_ratio=EXIT_MIN_MOTION_RATIO,
        motion_pause_ratio=MOTION_PAUSE_RATIO,entry_seconds=NEXT_GROUP_STABLE_SECONDS,
        entry_stability_ratio=NEXT_GROUP_STABILITY_RATIO,entry_match_distance_px=NEXT_GROUP_MATCH_DISTANCE_PX,
        exit_direction=EXIT_DIRECTION,auto_load_after_unload=AUTO_LOAD_AFTER_UNLOAD,
        entry_birth_fraction=NEXT_GROUP_ENTRY_FRACTION,
        reject_transition_roi_detections=REJECT_TRANSITION_ROI_DETECTIONS,
        reidentification_enabled=REIDENTIFICATION_ENABLED,
        reidentification_max_gap_seconds=REIDENTIFICATION_MAX_GAP_SECONDS,
        reidentification_max_distance_px=REIDENTIFICATION_MAX_DISTANCE_PX)


def recover_missing_slots(model, frame, objects, sequence, args):
    """仅为操作中已编号的漏检槽补框；坐标还原后仍走原来的关联、遮挡和连续确认。"""
    tracker = sequence.tracker
    if (not LOCAL_REDETECT_ENABLED or sequence.state != "active" or tracker.points is None
            or tracker.identity_lost):
        return objects, []
    gate = tracker._point_tracking_gate or tracker._spacing()*tracker.match_distance_ratio
    matches = tracker._associate(tracker._candidates(objects), gate)
    points = tracker.reference_points.copy()
    height,width = frame.shape[:2]
    crop_width,crop_height = LOCAL_REDETECT_CROP_SIZE
    additions,attempts = [],[]
    added_hand_boxes = set()
    for i,center in enumerate(points):
        if i in matches or sequence._distance(center) < sequence.margin:
            continue
        left,top = round(center[0])-crop_width//2,round(center[1])-crop_height//2
        x1,y1 = max(0,left),max(0,top)
        x2,y2 = min(width,left+crop_width),min(height,top+crop_height)
        if x2-x1 < 32 or y2-y1 < 32:
            continue
        result = model.predict(frame[y1:y2,x1:x2],imgsz=LOCAL_REDETECT_IMAGE_SIZE,
            conf=args.conf,device=args.device,iou=args.nms_iou,agnostic_nms=False,
            verbose=False,save=False)[0]
        boxes = result.boxes.cpu()
        accepted,hands = [],[]
        for cls,score,xy in zip(boxes.cls.tolist(),boxes.conf.tolist(),boxes.xyxy.tolist()):
            name = model.names[int(cls)]
            if name not in SLOT_STATES and name != "hand":
                continue
            box = [xy[0]+x1,xy[1]+y1,xy[2]+x1,xy[3]+y1]
            obj = {"class_id": int(cls),"class_name": name,"confidence": float(score),
                   "xyxy": box,"detection_source": "local_crop"}
            if name == "hand":
                hands.append(obj)
                continue
            point = np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            distances = np.linalg.norm(points-point,axis=1)
            # 只给本次漏检槽补框，邻槽和ROI外的框不能借补检参与编号或换组。
            if (distances[i] <= gate and np.argmin(distances) == i
                    and sequence._distance(point) >= 0):
                accepted.append(obj)
        additions.extend(accepted)
        # 槽框和手框可能同时被局部模型检出，也可能只有手框：后一种
        # 情况只提供 S4 的遮挡/接触信息，不能证明已上料。只保留靠近当前补检槽
        # 的手框，并跨裁剪去重，避免一个手框重复参与接触计数。
        relevant_hands = hands if accepted or tracker.phase == "load" else []
        for hand in relevant_hands:
            a = hand["xyxy"]
            hand_center = np.array([(a[0]+a[2])/2,(a[1]+a[3])/2])
            if np.linalg.norm(hand_center-center) <= gate or (
                    a[0] <= center[0] <= a[2] and a[1] <= center[1] <= a[3]):
                key = tuple(round(float(v), 1) for v in a)
                if key not in added_hand_boxes:
                    added_hand_boxes.add(key)
                    additions.append(hand)
        attempts.append({"group_id": tracker.group_id,"slot_id": i+1,"crop_xyxy": [x1,y1,x2,y2],
                         "accepted_detections": accepted})
    return objects+additions, attempts


def process_completed(snapshot):
    target = "empty" if snapshot.get("phase") == "unload" else "filled"
    slots = snapshot.get("slots", [])
    return len(slots) >= 4 and all(
        slot.get("last_confirmed_state") == target for slot in slots[:4])


def operation_status_text(snapshot):
    """Return the user-facing progress text for the current process phase."""
    action = "下料" if snapshot.get("phase") == "unload" else "上料"
    if process_completed(snapshot):
        return f"{action}完成"
    terminal_states = {"exiting", "waiting_next", "sequence_complete"}
    if snapshot.get("group_status") in terminal_states:
        return f"{action}未完成"
    return f"{action}中"


def build_process_report(summary, phase=None):
    """按正式事件和最后确认状态汇总五组；没有取放证据的初始空/满槽不算完成。"""
    phase = summary["phase"] if phase is None else phase
    snapshots = {g["group_id"]: g["snapshot"] for g in summary["closed_groups"]
                 if g["snapshot"]["phase"] == phase}
    current = summary["final_group_snapshot"]
    if summary.get("inspection_processed_frames",summary["processed_frames"]) and current["phase"] == phase and current["group_id"] not in snapshots:
        snapshots[current["group_id"]] = current
    target = "empty" if phase == "unload" else "filled"
    groups = []
    for group_id in range(1,summary["group_count"]+1):
        snapshot = snapshots.get(group_id)
        sequence = [] if snapshot is None else list(snapshot["confirmed_event_sequence"])
        slots = {} if snapshot is None else {s["slot_id"]: s for s in snapshot["slots"]}
        completed = ([None]*4 if snapshot is None else [int(slot_id in sequence
            and slots.get(slot_id,{}).get("last_confirmed_state") == target) for slot_id in range(1,5)])
        groups.append({"group_id": group_id,"observed": snapshot is not None,
            "slot_completed": completed,"completed": snapshot is not None and all(completed),
            "sequence": sequence,"order_result": None if snapshot is None else snapshot["order_result"]})
    return {"phase": phase,"slot_ids": [1,2,3,4],"matrix": [g["slot_completed"] for g in groups],
            "groups": groups,"end_status": summary["status"]}


def format_process_report(report):
    action = "下料" if report["phase"] == "unload" else "上料"
    lines = [f"游星片槽位{action}完成矩阵（1=已确认，0=未确认，-=未检测到游星片）",
             "游星片    S1  S2  S3  S4"]
    for group in report["groups"]:
        cells = "   ".join("-" if cell is None else str(cell) for cell in group["slot_completed"])
        lines.append(f"G{group['group_id']}       {cells}")
    lines.append(f"游星片{action}顺序（检测记录）：")
    for group in report["groups"]:
        sequence = " → ".join(map(str,group["sequence"])) or "未记录"
        lines.append(f"G{group['group_id']}：{sequence}")
    if report["end_status"] != "end_of_stream":
        lines.append("以上为本次已处理视频范围内的检测结果。")
    return "\n".join(lines)


def overlay_status(frame,snapshot,show_text=True,inspection=None):
    """左上角仅显示当前组是否完成工序；槽位状态保留在各中心点旁。"""
    colors = {"empty": (0,165,255),"filled": (40,220,80),"unknown": (160,160,160)}
    h,w = frame.shape[:2]
    panel_width,panel_height = min(w,420 if inspection is not None else 310),min(h,32)
    panel = frame[:panel_height,:panel_width]
    panel[:] = cv2.addWeighted(panel,.2,np.full_like(panel,20),.8,0)
    complete = process_completed(snapshot)
    header = operation_status_text(snapshot)
    if inspection is not None:
        fullness = inspection['fullness']
        if inspection['status'] == 'waiting_open':
            header = 'Hatch: waiting for open'
        elif inspection['status'] == 'completed':
            header = "上料完成" if fullness['all_filled'] else "上料未完成"
        elif snapshot['phase'] == 'load':
            header = "上料中"
    cv2.putText(frame,header,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.52,
                (40,220,80) if complete else (0,165,255),1,cv2.LINE_AA)
    for slot in snapshot["slots"]:
        display_state = slot["state"]
        # Before S numbering is finalized, a current detector box may already
        # have a confirmed visual state while the group is still stabilizing.
        # Show that confirmed state as a solid point; a missing current box
        # remains hollow so an old reference point is not presented as fresh
        # evidence.
        if (snapshot.get("slot_numbering_status") == "provisional"
                and display_state == "unknown"
                and slot.get("geometry_observed")
                and slot.get("last_confirmed_state") in colors):
            display_state = slot["last_confirmed_state"]
        color = colors[display_state]
        if slot["center_xy"] is None:
            continue
        x,y = slot["center_xy"]
        if not 0 <= x < w or not 0 <= y < h:
            continue
        point = (round(x),round(y))
        if display_state == "unknown":
            cv2.circle(frame,point,POINT_RADIUS+2,color,1,cv2.LINE_AA)
        else:
            cv2.circle(frame,point,POINT_RADIUS+1,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(frame,point,POINT_RADIUS,color,-1,cv2.LINE_AA)
        # Provisional points are identified by their ByteTrack boxes. Showing
        # the internal temporary array index here would look like a stable ID.
        if show_text and snapshot.get("slot_numbering_status") == "final":
            label = f"G{slot['group_id']}-S{slot['slot_id']} {slot['state']}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.38,1)
            left = max(0,min(w-tw-3,point[0]+7))
            top = max(th+3,min(h-3,point[1]-7))
            cv2.putText(frame,label,(left,top),cv2.FONT_HERSHEY_SIMPLEX,.38,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(frame,label,(left,top),cv2.FONT_HERSHEY_SIMPLEX,.38,color,1,cv2.LINE_AA)
    if snapshot["identity_status"] == "lost":
        cv2.putText(frame,"Identity lost: restart a verified group clip",(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,0,255),1,cv2.LINE_AA)
    elif snapshot.get("pending_initialization") is not None:
        init = snapshot["pending_initialization"]
        label = f"Next G{snapshot['pending_group_id']}: identity pending | observed actions: {init['buffered_event_count']}"
        cv2.putText(frame,label,(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(160,210,255),1,cv2.LINE_AA)
    elif snapshot["identity_status"] == "waiting_initialization":
        init = snapshot["initialization"]
        label = f"Identity pending: seen {len(init.get('geometry_seen_track_ids', []))}/4 | observed actions: {init['buffered_event_count']}"
        cv2.putText(frame,label,(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,210,255),1,cv2.LINE_AA)
    return frame


def draw_slot_tracks(frame, detections, snapshot, roi_contour):
    """Draw slot tracks only when their detector center is inside the ROI."""
    for item in detections:
        box = item["xyxy"]
        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if cv2.pointPolygonTest(roi_contour, center, False) < 0:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in item["xyxy"])
        track_id = item.get("track_id")
        color = (255, 170, 0) if track_id is not None else (150, 150, 150)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = slot_track_label(item, snapshot)
        cv2.putText(frame, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .4, color, 1, cv2.LINE_AA)
    return frame


def draw_all_detection_boxes(frame, detections, snapshot, sequence):
    """Draw selected real model detections, including low-confidence/ROI-outside rows."""
    from ultralytics.utils.plotting import colors

    for item in detections:
        box = item.get("xyxy")
        if box is None or len(box) != 4:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        is_slot = item.get("class_name") in SLOT_STATES
        if not is_slot and not SHOW_NON_SLOT_DETECTION_BOXES:
            continue
        track_id = item.get("track_id")
        retired = is_slot and sequence.is_retired_track(track_id)
        if is_slot:
            if retired:
                color = (120, 120, 120)
            else:
                color = (255, 170, 0) if track_id is not None else (150, 150, 150)
            label = slot_track_label(item, snapshot)
            if retired:
                label += " OLD"
        else:
            color = colors(int(item.get("class_id", 0)), True)
            label = (f"{item.get('class_name', 'det')} "
                     f"{float(item.get('confidence', 0.0)):.2f}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.putText(frame, label, (max(0, x1), max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
    return frame


def draw_predicted_slot_boxes(frame, slots, roi_contour, *, color=(0, 220, 255), label_prefix="S"):
    """Show geometry-backed virtual slot tracks as dashed boxes inside the ROI."""
    for slot in slots:
        box = slot.get("tracking_box_xyxy")
        center = slot.get("center_xy")
        if box is None or center is None or cv2.pointPolygonTest(roi_contour, tuple(center), False) < 0:
            continue
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        for x in range(x1, x2, 12):
            cv2.line(frame, (x, y1), (min(x+6, x2), y1), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x, y2), (min(x+6, x2), y2), color, 1, cv2.LINE_AA)
        for y in range(y1, y2, 12):
            cv2.line(frame, (x1, y), (x1, min(y+6, y2)), color, 1, cv2.LINE_AA)
            cv2.line(frame, (x2, y), (x2, min(y+6, y2)), color, 1, cv2.LINE_AA)
        slot_label = f"{label_prefix}{slot['slot_id']}" if label_prefix else "slot"
        label = f"G{slot['group_id']}-{slot_label} PRED"
        cv2.putText(frame, label, (x1, max(14, y1-4)), cv2.FONT_HERSHEY_SIMPLEX,
                    .38, color, 1, cv2.LINE_AA)
    return frame


def prepare_business_detections(tracked_raw, confidence_threshold):
    """Select detections that are allowed to enter the slot state machine.

    A detector row without a ByteTrack ID remains available to the audit and
    drawing layers, but it is deliberately excluded from business logic.  A
    high score cannot substitute for a track identity: admitting such a row
    would let an unstable box create a temporary slot or confirm a state.
    Non-slot detections keep their existing behavior, and tracked slot rows
    use the confidence produced by the tracker (raw or smoothed).
    """
    business = []
    for item in tracked_raw:
        tracking_confidence = float(item.get("tracking_confidence", item["confidence"]))
        if tracking_confidence < confidence_threshold:
            continue
        if (item.get("class_name") in SLOT_STATES
                and item.get("track_id") is None):
            continue
        working = dict(item)
        working["confidence"] = tracking_confidence
        business.append(working)
    return business


def run_prediction(args,*,on_frame=None,on_log=None,should_stop=None,roi_polygon=None):
    """接口兼容现有UI7V：roi_polygon参数携带ROI及视频信息。"""
    context = prepare_context(args) if roi_polygon is None else roi_polygon
    polygon = context["polygon"]
    def log(message):
        print(message,flush=True)
        if on_log is not None:
            on_log(message)
    group_completion = {}
    def _action_text(phase):
        return "下料" if phase == "unload" else "上料"

    def log_group_in_progress(phase, group_id):
        key = (phase, group_id)
        if group_completion.get(key) is None:
            action = _action_text(phase)
            log(f"游星片{group_id}：{action}中")
            group_completion[key] = "in_progress"

    def log_group_completion(snapshot, *, terminal=False):
        # 新流程的上料日志由占用核验发布，避免首次已满但无取放事件时
        # 同时输出“已放满”和旧事件统计中的“上料未完成”。
        if HATCH_CONTROL_ENABLED and snapshot['phase'] == 'load':
            return
        group_id, complete = snapshot["group_id"], process_completed(snapshot)
        key = (snapshot["phase"], group_id)
        current = group_completion.get(key)
        action = _action_text(snapshot["phase"])
        if complete:
            if current != "completed":
                log(f"游星片{group_id}：{action}完成")
                group_completion[key] = "completed"
        elif not terminal and current is None:
            log(f"游星片{group_id}：{action}中")
            group_completion[key] = "in_progress"
        elif terminal and current not in ("completed", "incomplete"):
            log(f"游星片{group_id}：{action}未完成")
            group_completion[key] = "incomplete"
    original_load = torch.load
    def trusted_load(*a,**kw):
        kw.setdefault("weights_only",False)
        return original_load(*a,**kw)
    torch.load = trusted_load
    try:
        model,architecture = base.load_detection_model(args.weights)
    finally:
        torch.load = original_load
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    slot_classes = [int(class_id) for class_id, name in names.items() if name in SLOT_STATES]
    audit_confidence = float(getattr(args, "audit_conf", AUDIT_CONFIDENCE))
    if not 0 < audit_confidence <= args.conf:
        raise ValueError("检测框审计置信度必须满足 0 < audit-conf <= conf")
    sequence = make_sequence(polygon,context["size"])
    tracker = sequence.tracker
    inspection = HatchLoadingInspection(HatchHandleMonitor(HATCH_OPEN_POLYGON, HATCH_CLOSED_POLYGON,
        confirm_seconds=HATCH_CONFIRM_SECONDS, minimum_frames=HATCH_MIN_CONFIRM_FRAMES,
        maximum_gap_seconds=HATCH_MAX_GAP_SECONDS, confidence=args.conf,
        distance_mode=HATCH_DISTANCE_MODE, closed_distance_max=HATCH_CLOSED_DISTANCE_MAX,
        open_distance_min=HATCH_OPEN_DISTANCE_MIN,
        missing_open_seconds=HATCH_MISSING_OPEN_SECONDS,
        reference_update_alpha=HATCH_REFERENCE_UPDATE_ALPHA)) if HATCH_CONTROL_ENABLED else None
    sequence_finished = False
    output = args.output_dir or Path(OUTPUT_ROOT)/(args.source.stem+"_slots_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    cap,writer = cv2.VideoCapture(str(args.source)),None
    processed,index,status = 0,context["start_frame"],"failed"
    inspection_processed = 0
    started = time.perf_counter()
    summary = {"source": str(args.source.resolve()),"weights": str(args.weights.resolve()),
        "output_dir": str(output.resolve()),"output_video": str((output/"annotated.mp4").resolve()),
        "architecture": architecture,"conf": args.conf,"audit_conf": audit_confidence,
        "imgsz": args.imgsz,"stride": args.stride,
        "detection_confidence_csv": "detection_confidences.csv",
        "slot_tracking": {"enabled": False, "backend": None,
            "class_names": sorted(SLOT_STATES), "identity_mapping": "ByteTrack ID -> one-time geometry-assigned S1-S4",
            "low_confidence": BYTE_TRACK_LOW_CONFIDENCE,
            "high_confidence": args.conf, "buffer_seconds": BYTE_TRACK_BUFFER_SECONDS,
            "role": "slot_identity_association_and_mapping",
            "continuous_across_groups": True, "state_before_slot_mapping": True,
            "untracked_slot_box_policy": "audit_and_display_only",
            "untracked_slot_admission": "never_without_bytetrack_id",
            "untracked_slot_effects": [],
            "per_detection_csv": "slot_tracks.csv",
            "confidence_smoothing": {
                "enabled": TRACK_CONFIDENCE_SMOOTHING_ENABLED,
                "window_size": TRACK_CONFIDENCE_SMOOTHING_WINDOW,
                "definition": "causal_mean_of_raw_confidence_from_K-n_through_K",
                "association_input": "preassociated_smoothed_confidence",
                "raw_confidence_retained_as": "confidence",
                "state_input": "tracking_confidence",
                "preassociation_iou": TRACK_CONFIDENCE_PREASSOCIATION_IOU,
                "preassociation_roi": "slot_roi_center_inside"},
            "reidentification": {
                "enabled": REIDENTIFICATION_ENABLED,
                "max_gap_seconds": REIDENTIFICATION_MAX_GAP_SECONDS,
                "max_distance_px": REIDENTIFICATION_MAX_DISTANCE_PX,
                "identity_field": "logical_track_id",
                "method": "stationary_slot_geometry"}},
        "group_id": CURRENT_GROUP_ID,"group_count": 5,"phase": PROCESS_PHASE,"clockwise_order": CLOCKWISE_ORDER,
        "auto_load_after_unload": sequence.auto_load_after_unload,
        "transition_roi_quarantine": {
            "enabled": REJECT_TRANSITION_ROI_DETECTIONS,
            "id_scope": "entire_next_group",
            "spatial_duration_seconds": max(IDENTITY_LOST_SECONDS, NEXT_GROUP_STABLE_SECONDS, .5),
            "distance_px": NEXT_GROUP_MATCH_DISTANCE_PX,
            "description": "old-group-exit ROI track IDs are excluded from the next group; ID-less boxes use temporary spatial quarantine"},
        "closed_group_inferred_s4_reassociation": {
            "enabled": False,
            "description": "an inferred S4 is frozen at group handoff and cannot bind a new ByteTrack ID"},
        "hatch_control_config": {"enabled": HATCH_CONTROL_ENABLED,
            "open_polygon_normalized": HATCH_OPEN_POLYGON, "closed_polygon_normalized": HATCH_CLOSED_POLYGON,
            "confirm_seconds": HATCH_CONFIRM_SECONDS, "minimum_frames": HATCH_MIN_CONFIRM_FRAMES,
            "maximum_gap_seconds": HATCH_MAX_GAP_SECONDS,
            "distance_mode": HATCH_DISTANCE_MODE,
            "closed_distance_max": HATCH_CLOSED_DISTANCE_MAX,
            "open_distance_min": HATCH_OPEN_DISTANCE_MIN,
            "missing_open_seconds": HATCH_MISSING_OPEN_SECONDS,
            "reference_update_alpha": HATCH_REFERENCE_UPDATE_ALPHA},
        "automatic_grouping_implemented": True,"automatic_grouping_enabled": AUTO_GROUP_SWITCH,
        "group_assignment": "roi_exit_sequence" if AUTO_GROUP_SWITCH else "configured",
        "group_switch_config": {"exit_confirm_seconds": EXIT_CONFIRM_SECONDS,
            "single_exit_immediate_entry": EXIT_MIN_SLOTS == 1,
            "provisional_tracks_can_trigger_exit": False,
            "old_group_exclusion": "persistent_id_then_real_identity_recovery; inferred_s4_reassociation_disabled",
            "retired_track_handoff": "unique_geometry_supported_candidate_when_current_group_has_two_or_three_fresh_slots",
            "entry_geometry_selection": "unique_supported_quadrilateral_not_top_confidence",
            "boundary_margin_px": EXIT_BOUNDARY_MARGIN_PX,"exit_min_slots": EXIT_MIN_SLOTS,
            "partial_exit_min_slots": PARTIAL_EXIT_MIN_SLOTS,"min_motion_ratio": EXIT_MIN_MOTION_RATIO,
            "motion_pause_ratio": MOTION_PAUSE_RATIO,"entry_seconds": NEXT_GROUP_STABLE_SECONDS,
            "entry_stability_ratio": NEXT_GROUP_STABILITY_RATIO,
            "entry_match_distance_px": NEXT_GROUP_MATCH_DISTANCE_PX,"exit_direction": EXIT_DIRECTION,
            "entry_birth_fraction": NEXT_GROUP_ENTRY_FRACTION},
        "center_source": "yolo_box_centers","initialization_seconds": INITIALIZATION_SECONDS,
        "initial_group_stable_seconds": INITIAL_GROUP_STABLE_SECONDS,
        "min_initialization_frames": MIN_INITIALIZATION_FRAMES,
        "initialization_history_seconds": INITIALIZATION_HISTORY_SECONDS,
        "initialization_event_retention": "until_numbering_or_track_invalidation",
        "entry_track_confirmation": {
            "seconds": max(CONFIRM_SECONDS, INITIALIZATION_SECONDS),
            "minimum_frames": max(2, MIN_CONFIRM_FRAMES, MIN_INITIALIZATION_FRAMES),
            "unsupported_timeout_seconds": min(IDENTITY_LOST_SECONDS, INITIALIZATION_HISTORY_SECONDS),
            "hidden_track_requires_confirmed_identity": True,
            "stale_association_rejects_opposing_motion": True},
        "initialization_stability_ratio": tracker.initialization_stability_ratio,
        "numbering_method": "stable_reference_clockwise_geometry_with_y_constraint" if SLOT2_BELOW_SLOT3 else "stable_x_then_track",
        "slot2_below_slot3": SLOT2_BELOW_SLOT3,
        "three_point_group_initialization": THREE_POINT_GROUP_INITIALIZATION,
        "slot_roi_file": context.get("roi_file"),
        "roi_polygon_normalized": polygon.tolist(),"roi_tracking_margin_px": ROI_TRACKING_MARGIN_PX,
        "confirm_seconds": CONFIRM_SECONDS,"min_confirm_frames": MIN_CONFIRM_FRAMES,
        "load_hand_contact_config": {"enabled": LOAD_HAND_CONTACT_ENABLED,
            "contact_seconds": LOAD_HAND_CONTACT_SECONDS,
            "min_frames": LOAD_HAND_CONTACT_MIN_FRAMES,
            "overlap_ratio": LOAD_HAND_CONTACT_OVERLAP_RATIO, "target_slot_ids": [4],
            "competing_slot": {
                "enabled": LOAD_HAND_CONTACT_COMPETING_SLOT_ENABLED,
                "slot_id": LOAD_HAND_CONTACT_COMPETING_SLOT_ID,
                "margin": LOAD_HAND_CONTACT_COMPETING_MARGIN,
                "ratio": LOAD_HAND_CONTACT_COMPETING_RATIO,
                "min_frames": LOAD_HAND_CONTACT_MIN_FRAMES,
                "behavior": "suppress_S4_hand_contact_when_competing_slot_is_clearly_dominant"},
            "fallback_transitions": {"unload": ["unknown->empty", "filled->unknown"],
                                      "load": ["unknown->filled", "empty->unknown"]},
            "yolo_state_has_priority": True,"infer_after_unknown_grace": True},
        "match_distance_ratio": MATCH_DISTANCE_RATIO,"identity_lost_seconds": IDENTITY_LOST_SECONDS,
        "local_redetection_config": {"enabled": LOCAL_REDETECT_ENABLED,
            "crop_size_wh": list(LOCAL_REDETECT_CROP_SIZE),"imgsz": LOCAL_REDETECT_IMAGE_SIZE,
            "conf": args.conf,"match_distance": "same_as_slot_tracker"},
        "rotation_intervals": ROTATION_INTERVALS,"start_frame": index,"end_frame_exclusive": END_FRAME,
        "timestamp_method": "source_frame/source_fps","audio_included": False}
    try:
        if not cap.isOpened():
            raise RuntimeError("无法打开源视频")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("视频帧率无效")
        summary.update(source_fps=fps,output_fps=fps/args.stride)
        byte_tracker = None
        if BYTE_TRACK_SLOTS_ENABLED and slot_classes:
            low_conf = min(BYTE_TRACK_LOW_CONFIDENCE, args.conf * .5)
            # low_conf = BYTE_TRACK_LOW_CONFIDENCE
            byte_tracker = SlotByteTracker(fps / args.stride, high_thresh=args.conf,
                low_thresh=low_conf, buffer_seconds=BYTE_TRACK_BUFFER_SECONDS,
                confidence_smoothing_enabled=TRACK_CONFIDENCE_SMOOTHING_ENABLED,
                confidence_smoothing_window=TRACK_CONFIDENCE_SMOOTHING_WINDOW,
                confidence_preassociation_iou=TRACK_CONFIDENCE_PREASSOCIATION_IOU,
                confidence_preassociation_roi=sequence.contour)
            summary["slot_tracking"].update(enabled=True, low_confidence=low_conf,
                high_confidence=args.conf, backend="ultralytics.ByteTrack", class_ids=slot_classes)
        elif BYTE_TRACK_SLOTS_ENABLED:
            summary["slot_tracking"]["reason"] = "model_has_no_wafer_slot_empty_or_filled_class"
        track_summary = {}
        track_counts = {"slot_boxes": 0, "tracked_slot_boxes": 0,
                        "untracked_slot_boxes": 0, "multi_high_conf_frames": 0}
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        with ExitStack() as stack:
            records = stack.enter_context((output/"slot_states.jsonl").open("w",encoding="utf-8"))
            events = stack.enter_context((output/"slot_events.jsonl").open("w",encoding="utf-8"))
            group_events = stack.enter_context((output/"group_events.jsonl").open("w",encoding="utf-8"))
            inspection_events = stack.enter_context((output/"inspection_events.jsonl").open("w",encoding="utf-8"))
            detection_csv = stack.enter_context(
                (output/"detection_confidences.csv").open("w", encoding="utf-8-sig", newline=""))
            detection_fields = ["source_frame", "time_seconds", "detection_index", "class_id", "class_name",
                                "confidence", "smoothed_confidence", "smoothing_window_count",
                                "smoothing_ready", "smoothing_status", "smoothing_source_track_id",
                                "smoothing_preassociation_iou", "smoothing_roi_eligible",
                                "raw_meets_display_confidence",
                                "x1", "y1", "x2", "y2", "track_id", "logical_track_id",
                                "reidentified_from_track_id", "reid_gap_seconds", "reid_method",
                                "group_id", "phase", "slot_point",
                                "meets_display_confidence", "displayed", "display_reason", "display_style",
                                "detection_source"]
            detection_rows = csv.DictWriter(detection_csv, fieldnames=detection_fields, extrasaction="ignore")
            detection_rows.writeheader()
            slot_track_records = stack.enter_context((output/"slot_tracks.jsonl").open("w",encoding="utf-8"))
            track_csv = stack.enter_context((output/'slot_tracks.csv').open('w', encoding='utf-8-sig', newline=''))
            track_fields = ['source_frame', 'time_seconds', 'track_id', 'logical_track_id',
                            'reidentified_from_track_id', 'reid_gap_seconds', 'reid_method',
                            'group_id', 'phase',
                            'slot_id', 'formal_slot_id', 'mapping_status', 'mapping_reason',
                            'slot_state', 'observation', 'last_confirmed_state', 'state_reason',
                            'confirmation_source', 'class_name', 'confidence', 'raw_confidence',
                            'smoothed_confidence', 'smoothing_window_count', 'smoothing_ready',
                            'smoothing_status', 'smoothing_source_track_id',
                            'smoothing_preassociation_iou', 'smoothing_roi_eligible', 'xyxy']
            track_rows = csv.DictWriter(track_csv, fieldnames=track_fields, extrasaction='ignore')
            track_rows.writeheader()
            def publish_inspection_events(items):
                for item in items:
                    inspection_events.write(json.dumps(item,ensure_ascii=False)+"\n")
                    if item.get('inspection_started'):
                        log('舱门把手已打开：开始检测，按下料→上料流程执行。' if tracker.phase == 'unload'
                            else '舱门把手已打开：开始上料检测。')
                        log_group_in_progress(tracker.phase, tracker.group_id)
                    elif item['type'] == 'group_fullness':
                        log(f"游星片{item['group_id']}：{item['result']}（{item['filled_count']}/4）")
                    elif item['type'] == 'loading_fullness_final':
                        log("上料完成" if item.get('all_filled') else "上料未完成")
            csv_file = stack.enter_context((output/"slot_states.csv").open("w",encoding="utf-8-sig",newline=""))
            fields = ["source_frame","time_seconds","group_id","slot_id","formal_slot_id","internal_slot_index",
                      "byte_track_id","logical_track_id","reidentified_from_track_id","reid_gap_seconds",
                      "reid_method","phase","state","observation",
                      "last_confirmed_state","confidence","center_x","center_y","reason","order_result","group_status",
                      "slot_numbering_status","group_transform","group_transform_definition",
                      "transform_valid_this_frame","transform_fit_error","transform_support_slot_ids",
                      "transform_support_byte_track_ids"]
            rows = csv.DictWriter(csv_file,fieldnames=fields)
            rows.writeheader()
            while True:
                if should_stop is not None and should_stop():
                    status = "user_stopped"
                    break
                if END_FRAME is not None and index >= END_FRAME:
                    status = "end_frame_limit"
                    break
                ok,frame = cap.read()
                if not ok:
                    status = "end_of_stream"
                    break
                source_frame = index
                index += 1
                if (source_frame-context["start_frame"]) % args.stride:
                    continue
                h,w = frame.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(output/"annotated.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps/args.stride,(w,h))
                    if not writer.isOpened():
                        raise RuntimeError("无法创建输出MP4")
                # 推理使用独立的审计下限；ByteTrack适配器会先用上一帧
                # 的轨迹框预关联，再把滑动平均置信度送入当前帧跟踪。
                result = model.predict(frame,imgsz=args.imgsz,conf=audit_confidence,device=args.device,
                    iou=args.nms_iou,agnostic_nms=False,verbose=False,save=False)[0]
                boxes = result.boxes.cpu()
                all_raw = [{"class_id": int(c),"class_name": names[int(c)],"confidence": float(score),"xyxy": xy}
                    for c,score,xy in zip(boxes.cls.tolist(),boxes.conf.tolist(),boxes.xyxy.tolist())]
                tracked_raw = (byte_tracker.update(all_raw, frame_index=processed + 1)
                                if byte_tracker else all_raw)
                # Non-slot detections do not enter SlotByteTracker. Keep the
                # same audit schema for them and use their raw score.
                for item in tracked_raw:
                    item.setdefault("raw_confidence", float(item["confidence"]))
                    item.setdefault("tracking_confidence", float(item["confidence"]))
                    item.setdefault("smoothed_confidence", float(item["confidence"]))
                    item.setdefault("smoothing_window_count", 0)
                    item.setdefault("smoothing_ready", False)
                    item.setdefault("smoothing_status", "not_slot")
                    item.setdefault("smoothing_source_track_id", None)
                    item.setdefault("smoothing_preassociation_iou", None)
                    item.setdefault("smoothing_roi_eligible", None)

                # ``raw`` is the input to the business sequence. The adapter
                # has already supplied the pre-associated moving-average score
                # as tracking_confidence; retain the model score for auditing.
                # Keep the complete tracked_raw list for audit/video output;
                # only the explicitly admitted rows reach the business logic.
                raw = prepare_business_detections(tracked_raw, args.conf)
                tracking_confidence = (summary["slot_tracking"]["low_confidence"]
                                       if byte_tracker else args.conf)
                slot_track_detections = [item for item in tracked_raw
                                         if item["class_name"] in SLOT_STATES
                                         and item["tracking_confidence"] >= tracking_confidence]
                high_conf_slots = [item for item in slot_track_detections
                                   if item["tracking_confidence"] >= args.conf]
                track_counts["slot_boxes"] += len(slot_track_detections)
                track_counts["tracked_slot_boxes"] += sum(item.get("track_id") is not None
                                                             for item in slot_track_detections)
                track_counts["untracked_slot_boxes"] += sum(item.get("track_id") is None
                                                               for item in slot_track_detections)
                track_counts["multi_high_conf_frames"] += len(high_conf_slots) >= 2
                # Keep retired detections out of local re-detection and hand
                # contact inspection, but pass them to the sequence as
                # candidates. During a new-group entry the sequence may
                # explicitly hand off one such ID when it is the only way to
                # complete the current group's three/four-slot geometry.
                sequence_raw = [item for item in raw
                                if not (item.get("class_name") in SLOT_STATES
                                        and sequence.is_retired_track(item.get("track_id")))]
                retired_raw = [item for item in raw
                               if item.get("class_name") in SLOT_STATES
                               and sequence.is_retired_track(item.get("track_id"))]
                timestamp = source_frame/fps
                enabled = not any(start <= timestamp < end for start,end in ROTATION_INTERVALS)
                if inspection is not None:
                    publish_inspection_events(inspection.observe_handle(sequence_raw,timestamp,(w,h)))
                if inspection is None or inspection.active:
                    objects,local_attempts = (recover_missing_slots(model,frame,sequence_raw,sequence,args)
                                              if enabled else (sequence_raw,[]))
                    # The group sequence applies its own conservative handoff
                    # gate. Retired boxes remain available for that gate but
                    # never enter the local crop detector.
                    objects.extend(retired_raw)
                    snapshot = sequence.update(objects,timestamp,observations_enabled=enabled)
                    inspection_processed += 1
                else:
                    local_attempts = []
                    if inspection.closed_at is not None and not sequence_finished:
                        snapshot = sequence.finish()
                        sequence_finished = True
                    else:
                        snapshot = {**sequence.snapshot(),"new_events":[],"new_group_events":[]}
                tracker = sequence.tracker
                # ByteTrack runs continuously for the whole video. G/S is a
                # separate business mapping; switching G must not renumber BT.
                track_states = snapshot.setdefault('track_states', [])
                known_state_ids = {item.get('byte_track_id') for item in track_states}
                for item in slot_track_detections:
                    track_id = item.get('track_id')
                    if (track_id is not None and sequence.is_retired_track(track_id)
                            and track_id not in known_state_ids):
                        previous = sequence.track_store.describe(track_id)
                        if previous is not None:
                            track_states.append({**previous,
                                'state': 'unknown', 'observation': 'unknown',
                                'reason': 'previous_group_track',
                                'mapping_reason': 'previous_group_track'})
                            known_state_ids.add(track_id)
                known_records = {item['byte_track_id'] for item in track_states}
                for item in slot_track_detections:
                    if item['tracking_confidence'] < args.conf and item.get('track_id') not in known_records:
                        previous = sequence.track_store.describe(item.get('track_id'))
                        if previous is not None:
                            snapshot.setdefault('track_states', []).append({**previous,
                                'state': 'unknown', 'observation': 'unknown',
                                'reason': 'below_state_confidence_threshold',
                                'mapping_reason': 'below_state_confidence_threshold'})
                            known_records.add(item['track_id'])
                for detection_index, item in enumerate(tracked_raw):
                    is_slot = item["class_name"] in SLOT_STATES
                    slot_record = slot_track_record(item, snapshot) if is_slot else {}
                    slot_id = slot_record.get("formal_slot_id") or slot_record.get("slot_id")
                    slot_point = (f"S{slot_id}" if slot_id is not None else
                                  "待编号" if slot_record.get("mapping_status") == "provisional" else "")
                    tracking_score = float(item.get("tracking_confidence", item["confidence"]))
                    raw_score = float(item.get("raw_confidence", item["confidence"]))
                    if tracking_score < args.conf:
                        displayed, display_reason = False, "below_confidence_threshold"
                    elif not is_slot:
                        displayed, display_reason = True, "drawn_detection_point"
                    elif sequence.is_retired_track(item.get("track_id")):
                        displayed, display_reason = False, "previous_group_track"
                    elif sequence.is_transition_quarantined_track(item.get("track_id")):
                        displayed, display_reason = False, "transition_roi_quarantine"
                    else:
                        box = item["xyxy"]
                        center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        displayed = cv2.pointPolygonTest(sequence.contour, center, False) >= 0
                        display_reason = "drawn_slot_box" if displayed else "outside_slot_roi"
                    audit_row = base.detection_confidence_row(
                        source_frame=source_frame, time_seconds=round(timestamp, 6),
                        index=detection_index, detection=item,
                        confidence_threshold=args.conf, displayed=displayed,
                        display_reason=display_reason,
                        display_style="slot_box" if is_slot else "detection_point")
                    audit_row.update(
                        confidence=round(raw_score, 6),
                        smoothed_confidence=round(tracking_score, 6),
                        smoothing_window_count=item.get("smoothing_window_count", 0),
                        smoothing_ready=item.get("smoothing_ready", False),
                        smoothing_status=item.get("smoothing_status", "disabled"),
                        smoothing_source_track_id=item.get("smoothing_source_track_id"),
                        smoothing_preassociation_iou=item.get("smoothing_preassociation_iou"),
                        smoothing_roi_eligible=item.get("smoothing_roi_eligible"),
                        raw_meets_display_confidence=raw_score >= args.conf,
                        meets_display_confidence=tracking_score >= args.conf)
                    audit_row.update(track_id=item.get("track_id"),
                                     logical_track_id=slot_record.get("logical_track_id", item.get("track_id")),
                                     reidentified_from_track_id=slot_record.get("reidentified_from_track_id"),
                                     reid_gap_seconds=slot_record.get("reid_gap_seconds"),
                                     reid_method=slot_record.get("reid_method"),
                                     group_id=slot_record.get("group_id"),
                                     phase=(slot_record.get("phase") or tracker.phase) if is_slot else "",
                                     slot_point=slot_point,
                                     detection_source=item.get("detection_source", "full_frame"))
                    detection_rows.writerow(audit_row)
                detection_records = [slot_track_record(item, snapshot) for item in slot_track_detections]
                # Track statistics represent business-group membership, not
                # merely which group's frame happened to contain a raw box.
                # Old-group tails remain in slot_tracks.jsonl for diagnostics,
                # but unassigned/previous_group_track detections must not
                # create a second G entry for the same ByteTrack ID.
                for item, detection_record in zip(slot_track_detections, detection_records):
                    track_id = item.get("track_id")
                    assigned_group = detection_record.get("group_id")
                    mapping_status = detection_record.get("mapping_status")
                    if (track_id is None or assigned_group is None
                            or mapping_status not in ("provisional", "final")):
                        continue
                    assigned_phase = detection_record.get("phase") or tracker.phase
                    metric_key = (assigned_group, assigned_phase, track_id)
                    metric = track_summary.setdefault(metric_key, {"group_id": assigned_group,
                        "phase": assigned_phase, "byte_track_id": track_id, "detections": 0,
                        "first_time_seconds": source_frame / fps,
                        "last_time_seconds": None, "high_confidence_detections": 0,
                        "low_confidence_detections": 0})
                    metric["detections"] += 1
                    metric["last_time_seconds"] = source_frame / fps
                    confidence_key = ("high_confidence_detections" if item["tracking_confidence"] >= args.conf
                                      else "low_confidence_detections")
                    metric[confidence_key] += 1
                for detection_record in detection_records:
                    track_rows.writerow({**detection_record, 'source_frame': source_frame,
                        'time_seconds': round(timestamp, 6),
                        'phase': detection_record.get('phase') or tracker.phase,
                        'xyxy': json.dumps(detection_record.get('xyxy'))})
                slot_track_records.write(json.dumps({"source_frame": source_frame,
                    "time_seconds": timestamp,"group_id": tracker.group_id,"phase": tracker.phase,
                    "slot_numbering_status": snapshot.get("slot_numbering_status"),
                    "group_transform": snapshot.get("group_transform"),
                    "group_transform_definition": snapshot.get("group_transform_definition"),
                    "transform_valid_this_frame": snapshot.get("transform_valid_this_frame"),
                    "transform_fit_error": snapshot.get("transform_fit_error"),
                    "transform_support_slot_ids": snapshot.get("transform_support_slot_ids", []),
                    "transform_support_byte_track_ids": snapshot.get("transform_support_byte_track_ids", []),
                    "detections": detection_records,
                    "mapping": [{key: slot.get(key) for key in (
                        "group_id", "slot_id", "formal_slot_id", "byte_track_id", "logical_track_id",
                        "reidentified_from_track_id", "reid_gap_seconds", "reid_method",
                        "state", "observation",
                        "last_confirmed_state", "confidence", "reason", "confirmation_source")}
                        for slot in snapshot.get("slots", [])]},ensure_ascii=False)+"\n")
                if inspection is not None:
                    publish_inspection_events(inspection.update_fullness(sequence,timestamp))
                record = {"source_frame": source_frame,"time_seconds": timestamp,**snapshot,
                          "local_redetection": local_attempts,
                          "inspection": inspection.snapshot() if inspection is not None else None}
                records.write(json.dumps(record,ensure_ascii=False)+"\n")
                for event in snapshot["new_events"]:
                    events.write(json.dumps(event,ensure_ascii=False)+"\n")
                for event in snapshot["new_group_events"]:
                    group_events.write(json.dumps(event,ensure_ascii=False)+"\n")
                    if event["type"] == "group_started":
                        log_group_in_progress(event["phase"], event["group_id"])
                    if event["type"] in ("group_closed", "group_updated"):
                        log_group_completion(event["snapshot"], terminal=True)
                if inspection is None or inspection.started_at is not None:
                    log_group_completion(snapshot)
                for slot in snapshot["slots"]:
                    center = slot["center_xy"] or (None,None)
                    row = {key: slot.get(key) for key in (
                        "group_id","slot_id","formal_slot_id","internal_slot_index","byte_track_id",
                        "logical_track_id","reidentified_from_track_id","reid_gap_seconds","reid_method","state",
                        "observation","last_confirmed_state","confidence","reason")}
                    row.update({"source_frame": source_frame,"time_seconds": round(timestamp,6),
                        "phase": tracker.phase,"center_x": center[0],"center_y": center[1],
                        "order_result": snapshot["order_result"],"group_status": snapshot["group_status"],
                        "slot_numbering_status": snapshot["slot_numbering_status"],
                        "group_transform": json.dumps(snapshot.get("group_transform"), ensure_ascii=False),
                        "group_transform_definition": snapshot.get("group_transform_definition"),
                        "transform_valid_this_frame": snapshot.get("transform_valid_this_frame"),
                        "transform_fit_error": snapshot.get("transform_fit_error"),
                        "transform_support_slot_ids": json.dumps(
                            snapshot.get("transform_support_slot_ids", []), ensure_ascii=False),
                        "transform_support_byte_track_ids": json.dumps(
                            snapshot.get("transform_support_byte_track_ids", []), ensure_ascii=False)})
                    rows.writerow(row)
                plotted = frame.copy()
                if SHOW_ALL_DETECTION_BOXES:
                    draw_all_detection_boxes(plotted, tracked_raw, snapshot, sequence)
                else:
                    others = [o for o in raw if o["class_name"] not in SLOT_STATES]
                    plotted = draw_detection_points(frame, others, radius=POINT_RADIUS)
                    if slot_track_detections:
                        # Retired IDs remain in the raw track log for diagnosis,
                        # but must not visually occupy a slot in the current group.
                        display_slots = [item for item in high_conf_slots
                                         if not sequence.is_retired_track(item.get('track_id'))]
                        draw_slot_tracks(plotted, display_slots, snapshot, sequence.contour)
                draw_predicted_slot_boxes(plotted,
                    [slot for slot in snapshot.get("slots", []) if slot.get("position_is_prediction")],
                    sequence.contour,
                    label_prefix="S" if snapshot.get("slot_numbering_status") == "final" else None)
                if DRAW_SLOT_ROI:
                    draw_roi(plotted,polygon)
                overlay_status(plotted,snapshot,SHOW_SLOT_TEXT,
                               inspection.snapshot() if inspection is not None else None)
                writer.write(plotted)
                processed += 1
                if snapshot["slot_numbering_status"] == "provisional":
                    text = "；".join(
                        f"组{s['group_id']}-BT{s['byte_track_id'] if s['byte_track_id'] is not None else '待关联'}:"
                        f"{STATE_TEXT[s['state'] if s['state'] != 'unknown' else s['last_confirmed_state']]}"
                        f"{'*' if s['state'] == 'unknown' and s['last_confirmed_state'] != 'unknown' else ''}"
                        for s in snapshot["slots"])
                else:
                    text = "；".join(f"组{s['group_id']}-S{s['slot_id']}:{STATE_TEXT[s['state']]}"
                                    for s in snapshot["slots"])
                status_text = (f"组{tracker.group_id} · {operation_status_text(snapshot)} · "
                               f"{timestamp:.2f}s · {GROUP_TEXT[snapshot['group_status']]} · "
                               f"顺序{snapshot['order_result']}")
                if inspection is not None:
                    if inspection.started_at is None:
                        status_text = text = '等待舱门把手打开'
                    elif inspection.closed_at is not None:
                        report = inspection.report
                        status_text = text = ("上料完成" if report['all_filled']
                                              else "上料未完成")
                    else:
                        status_text += f" · 已确认放满{inspection.report['confirmed_filled_count']}/20"
                if on_frame is not None:
                    on_frame(plotted,{"status": status_text,
                        "result": text,"frame_id": processed,"source_frame": source_frame,"time_seconds": timestamp,
                        "slots": snapshot["slots"],"group_id": tracker.group_id,"phase": tracker.phase,
                        "group_status": snapshot["group_status"],"pending_group_id": snapshot["pending_group_id"],
                        "pending_phase": snapshot["pending_phase"],
                        "pending_initialization": snapshot["pending_initialization"],
                        "slot_numbering_status": snapshot["slot_numbering_status"],
                        "track_states": snapshot.get('track_states', []),
                        "inspection": inspection.snapshot() if inspection is not None else None})
                if inspection is not None and inspection.closed_at is not None:
                    status = "hatch_closed"
                    break
                if args.show:
                    cv2.imshow("Slot groups: Q quit / Space pause",plotted)
                    key = cv2.waitKey(1)&0xff
                    if key == ord(' '):
                        while True:
                            key = cv2.waitKey(50)&0xff
                            if key in (ord(' '),ord('q'),27):
                                break
                    if key in (ord('q'),27) or cv2.getWindowProperty("Slot groups: Q quit / Space pause",cv2.WND_PROP_VISIBLE) < 1:
                        status = "preview_stopped"
                        break
                if args.max_frames and processed >= args.max_frames:
                    status = "frame_limit"
                    break
            final = (sequence.finish() if not sequence_finished else
                     {**sequence.snapshot(),"new_events":[],"new_group_events":[]})
            for event in final["new_events"]:
                events.write(json.dumps(event,ensure_ascii=False)+"\n")
            for event in final["new_group_events"]:
                group_events.write(json.dumps(event,ensure_ascii=False)+"\n")
                if event["type"] == "group_started":
                    log_group_in_progress(event["phase"], event["group_id"])
                if event["type"] in ("group_closed", "group_updated"):
                    log_group_completion(event["snapshot"], terminal=True)
            if processed and (inspection is None or inspection.started_at is not None):
                log_group_completion(final, terminal=True)
            if inspection is not None and processed:
                publish_inspection_events(inspection.update_fullness(sequence,timestamp))
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        summary.update(status=status,processed_frames=processed,elapsed_seconds=round(time.perf_counter()-started,3),
                       inspection_processed_frames=inspection_processed,
                       final_group_snapshot=sequence.snapshot(),events=sequence.all_events,
                       closed_groups=sequence.closed_groups)
        if "track_counts" in locals():
            summary["slot_tracking"].update(track_counts=track_counts,
                track_count=len(track_summary),
                reidentification_events=list(sequence.track_store.reidentification_events),
                tracks={f"{phase}:G{group_id}:BT{track_id}": values
                        for (group_id, phase, track_id), values in sorted(track_summary.items())})
        summary["group_process_report"] = build_process_report(summary)
        phases = [summary["phase"]]
        if sequence.auto_load_after_unload and summary["phase"] == "unload":
            phases.append("load")
        summary["group_process_reports"] = {phase:build_process_report(summary,phase) for phase in phases}
        summary["current_phase"] = sequence.tracker.phase
        if inspection is not None:
            inspection.update_fullness(sequence, timestamp if processed else 0.)
            summary['inspection'] = inspection.snapshot()
            summary['inspection_events'] = inspection.events
            summary['loading_fullness_report'] = inspection.report
            (output/'loading_fullness_report.json').write_text(
                json.dumps(inspection.report,ensure_ascii=False,indent=2),encoding='utf-8')
            (output/'loading_fullness_report.txt').write_text(format_fullness_report(inspection.report)+'\n',encoding='utf-8')
        report_text = "\n\n".join(format_process_report(report)
                                 for report in summary["group_process_reports"].values())
        if inspection is not None:
            report_text += '\n\n'+format_fullness_report(inspection.report)
        (output/"group_process_report.json").write_text(
            json.dumps(summary["group_process_reports"],ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"group_process_report.txt").write_text(report_text+"\n",encoding="utf-8")
        (output/"group_summaries.json").write_text(json.dumps(sequence.closed_groups,ensure_ascii=False,indent=2),encoding="utf-8")
        (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    if not processed and status != "user_stopped":
        raise RuntimeError("没有处理任何帧")
    log(report_text)
    return summary


def main():
    args = build_args(sys.argv[1:])
    if args.ui:
        from industrial_inspection.tools.predict_video_ui import launch_ui
        launch_ui(args,run_prediction,prepare_context)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
