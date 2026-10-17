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
from industrial_inspection.tools.roi_group_sequence import RoiGroupSequence
from industrial_inspection.tools.slot_roi import draw_roi, pixel_polygon


# =============================================================================
# 在这里改配置，直接在 PyCharm 运行，无需终端；原 predict_video.py 不受影响。
# 视频、模型和 ROI 默认沿用原脚本，也可以改为自己的 Path(r"...")。
# =============================================================================
SOURCE_VIDEO = base.SOURCE_VIDEO
MODEL_WEIGHTS = base.MODEL_WEIGHTS
OUTPUT_ROOT = base.OUTPUT_ROOT
CONFIDENCE = base.CONFIDENCE
IMAGE_SIZE = base.IMAGE_SIZE
DEVICE = base.DEVICE
UI_ENABLED = True             # True：沿用 UI7V；False：OpenCV 预览/批处理
SHOW_PREVIEW = True           # 仅 UI_ENABLED=False 时生效；Q退出、空格暂停
MAX_FRAMES = 0                # 处理整段视频，避免在自动上料前被试跑帧数截断
FRAME_STRIDE = 1

CURRENT_GROUP_ID = 1          # 视频起始实际组号，1..5；之后按确认的换组次数递增
PROCESS_PHASE = "unload"     # 起始工序；默认下料五组后自动上料，纯上料片段可设"load"
AUTO_LOAD_AFTER_UNLOAD = True # 下料G5退出后自动回到G1开始上料；上料G5退出后结束
CLOCKWISE_ORDER = (1,3,4,2)    # 按实物确认后填写；允许从任意槽开始
GROUP_TRACKING_ENABLED = True
AUTO_GROUP_SWITCH = True      # 单点真实越界即切到下一组并逐点记录；False保持单组模式
EXIT_CONFIRM_SECONDS = .1     # 仅多点换组模式使用；单点换组立即关闭旧组
EXIT_BOUNDARY_MARGIN_PX = 4   # 边界内/外容差，避免中心在线上抖动被反复计数
EXIT_MIN_SLOTS = 1            # 任意一个已编号槽实际越出ROI即触发换组
PARTIAL_EXIT_MIN_SLOTS = 1    # 单点换组模式下同样只需要一个越界槽
EXIT_MIN_MOTION_RATIO = .5    # 多点换组规则的组中心位移比例；单点换组时不参与触发
MOTION_PAUSE_RATIO = .18      # 多点换组规则的共同外移门限；单点换组时不参与触发
NEXT_GROUP_STABLE_SECONDS = .3 # 仅多点换组模式使用；单点换组收齐四条轨迹即启用正式槽号
NEXT_GROUP_STABILITY_RATIO = .12 # 停稳期间各槽偏离起始位置不得超过最小槽间距×此比例
NEXT_GROUP_MATCH_DISTANCE_PX = 40 # 新点/旧组残留点逐帧匹配的最大距离（像素）
EXIT_DIRECTION = (-1, -1)     # 本视频已核实旧组从右下ROI向左上退出；None可恢复自动方向判断

# 每轮五组按顺序计数；自动模式运行五组下料再五组上料。
START_FRAME = None            # None：从第0帧开始；数字：指定当前组开始的源帧号
END_FRAME = None              # 不包含该帧；例如1800表示到1799帧，None表示视频结尾
INITIALIZATION_SECONDS = .3   # 候选四槽已分别可靠看见，至少三槽持续支持稳定几何后启用编号
MIN_INITIALIZATION_FRAMES = 2 # 同时满足最少帧数；编号后不再按x重新排序
INITIALIZATION_HISTORY_SECONDS = 5.0 # 确认前仅保留窗口内有原/新状态证据的取放；超出窗口不补报

MATCH_DISTANCE_RATIO = .46    # 匹配门限=最近两槽间距×此比例；放宽以容纳空/满框中心偏移，必须小于0.5
CONFIRM_SECONDS = .2          # 空/满持续达到该时间且满足帧数，才确认状态
MIN_CONFIRM_FRAMES = 3
IDENTITY_LOST_SECONDS = 1.2   # 连续无匹配超时，锁定为unknown，避免下一组继承旧编号
CONFLICT_CONF_MARGIN = .15    # 同位置空/满分数差小于该值时，输出unknown
HAND_OVERLAP_RATIO = .2       # 手遮挡槽框达到此比例，或覆盖中心时，暂停该槽状态确认
LOAD_HAND_CONTACT_ENABLED = True       # 上料时记录手接触供复核，不据此判断空满或操作顺序
LOAD_HAND_CONTACT_SECONDS = .15        # 手在同一槽位的最短连续接触时间
LOAD_HAND_CONTACT_MIN_FRAMES = 2       # 接触至少需要的连续检测帧数
ROI_TRACKING_MARGIN_PX = 40   # ROI外只跟踪当前已编号槽；区域外框不确认空满、不初始化
LOCAL_REDETECT_ENABLED = True # 已编号槽被整图漏检时，用同一模型局部补检
LOCAL_REDETECT_CROP_SIZE = (180, 140) # 原图裁剪宽、高（像素），保留槽周围背景
LOCAL_REDETECT_IMAGE_SIZE = 320 # 局部图送入YOLO时的尺寸；置信度沿用整图配置
ROTATION_INTERVALS = []       # 可选源视频秒数[(开始,结束),...]；期间跟踪位置、暂停空满确认
DRAW_SLOT_ROI = base.DRAW_SLOT_ROI
SHOW_SLOT_TEXT = True         # 在点旁标注G1-S2 empty/filled/unknown
POINT_RADIUS = base.POINT_RADIUS
# =============================================================================
STATE_TEXT = {"empty": "空槽","filled": "已填充","unknown": "未知"}
GROUP_TEXT = {"initializing": "等待四槽初始化","active": "操作中","exiting": "正在退出",
              "entering": "新组逐槽跟踪中",
              "waiting_next": "等待下一组","stabilizing_next": "下一组停稳确认中",
              "sequence_complete": "本工序结束(G5)","identity_uncertain": "组身份待确认"}


def build_args():
    args = base.parse_args(["--source",str(SOURCE_VIDEO),"--weights",str(MODEL_WEIGHTS)])
    args.source,args.weights = Path(SOURCE_VIDEO),Path(MODEL_WEIGHTS)
    args.conf,args.imgsz,args.device = CONFIDENCE,IMAGE_SIZE,DEVICE
    args.ui,args.show,args.max_frames,args.stride = UI_ENABLED,SHOW_PREVIEW,MAX_FRAMES,FRAME_STRIDE
    args.output_dir = None
    args.slot_roi = True
    args.group_completion_log_only = True
    if not GROUP_TRACKING_ENABLED:
        raise ValueError("本脚本专用于四槽跟踪；普通检测请运行 predict_video.py")
    if not args.source.is_file() or not args.weights.is_file():
        raise FileNotFoundError("请检查本脚本配置的视频和权重路径")
    if not 0 < args.conf < 1 or args.stride < 1 or args.max_frames < 0 or args.imgsz < 32:
        raise ValueError("置信度、stride、max_frames 或 imgsz 配置错误")
    if START_FRAME is not None and START_FRAME < 0 or END_FRAME is not None and END_FRAME < 1:
        raise ValueError("起止帧号必须为有效非负帧号，结束帧大于0")
    for start,end in ROTATION_INTERVALS:
        if not 0 <= start < end:
            raise ValueError("ROTATION_INTERVALS中每个区间必须满足0<=开始<结束")
    if ROI_TRACKING_MARGIN_PX < 0:
        raise ValueError("ROI缓冲宽度不能为负")
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
    return {"polygon": polygon,"size": (w,h),"start_frame": start_frame}


def make_tracker():
    return SlotGroupTracker(group_id=CURRENT_GROUP_ID,phase=PROCESS_PHASE,
        clockwise=CLOCKWISE_ORDER,match_distance_ratio=MATCH_DISTANCE_RATIO,confirm_seconds=CONFIRM_SECONDS,
        minimum_frames=MIN_CONFIRM_FRAMES,lost_seconds=IDENTITY_LOST_SECONDS,
        conflict_margin=CONFLICT_CONF_MARGIN,hand_overlap=HAND_OVERLAP_RATIO,
        initialization_seconds=INITIALIZATION_SECONDS,initialization_frames=MIN_INITIALIZATION_FRAMES,
        initialization_history_seconds=INITIALIZATION_HISTORY_SECONDS,
        load_hand_contact_enabled=LOAD_HAND_CONTACT_ENABLED,
        load_hand_contact_seconds=LOAD_HAND_CONTACT_SECONDS,
        load_hand_contact_min_frames=LOAD_HAND_CONTACT_MIN_FRAMES)


def make_sequence(polygon,size):
    return RoiGroupSequence(make_tracker(),pixel_polygon(polygon,size),enabled=AUTO_GROUP_SWITCH,
        buffer_px=ROI_TRACKING_MARGIN_PX,boundary_margin_px=EXIT_BOUNDARY_MARGIN_PX,
        exit_seconds=EXIT_CONFIRM_SECONDS,exit_min_slots=EXIT_MIN_SLOTS,
        partial_exit_slots=PARTIAL_EXIT_MIN_SLOTS,minimum_motion_ratio=EXIT_MIN_MOTION_RATIO,
        motion_pause_ratio=MOTION_PAUSE_RATIO,entry_seconds=NEXT_GROUP_STABLE_SECONDS,
        entry_stability_ratio=NEXT_GROUP_STABILITY_RATIO,entry_match_distance_px=NEXT_GROUP_MATCH_DISTANCE_PX,
        exit_direction=EXIT_DIRECTION,auto_load_after_unload=AUTO_LOAD_AFTER_UNLOAD)


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
    return (set(snapshot.get("confirmed_event_sequence",[])) == {1,2,3,4}
            and snapshot.get("all_four_target_previously_verified",False))


def build_process_report(summary, phase=None):
    """按正式事件和最后确认状态汇总五组；没有取放证据的初始空/满槽不算完成。"""
    phase = summary["phase"] if phase is None else phase
    snapshots = {g["group_id"]: g["snapshot"] for g in summary["closed_groups"]
                 if g["snapshot"]["phase"] == phase}
    current = summary["final_group_snapshot"]
    if summary["processed_frames"] and current["phase"] == phase and current["group_id"] not in snapshots:
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


def overlay_status(frame,snapshot,show_text=True):
    """左上角仅显示当前组是否完成工序；槽位状态保留在各中心点旁。"""
    colors = {"empty": (0,165,255),"filled": (40,220,80),"unknown": (160,160,160)}
    h,w = frame.shape[:2]
    panel_width,panel_height = min(w,310),min(h,32)
    panel = frame[:panel_height,:panel_width]
    panel[:] = cv2.addWeighted(panel,.2,np.full_like(panel,20),.8,0)
    complete = process_completed(snapshot)
    action = "Unload" if snapshot["phase"] == "unload" else "Load"
    header = f"G{snapshot['group_id']} {action}: {'completed' if complete else 'not completed'}"
    cv2.putText(frame,header,(7,22),cv2.FONT_HERSHEY_SIMPLEX,.52,
                (40,220,80) if complete else (0,165,255),1,cv2.LINE_AA)
    for slot in snapshot["slots"]:
        color = colors[slot["state"]]
        prefix = "T" if snapshot.get("slot_numbering_status") == "provisional" else "S"
        label = f"G{slot['group_id']}-{prefix}{slot['slot_id']} {slot['state']}"
        if slot["center_xy"] is None:
            continue
        x,y = slot["center_xy"]
        if not 0 <= x < w or not 0 <= y < h:
            continue
        point = (round(x),round(y))
        if slot["state"] == "unknown":
            cv2.circle(frame,point,POINT_RADIUS+2,color,1,cv2.LINE_AA)
        else:
            cv2.circle(frame,point,POINT_RADIUS+1,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(frame,point,POINT_RADIUS,color,-1,cv2.LINE_AA)
        if show_text:
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
        label = f"Identity pending: seen {len(init['geometry_seen_slot_ids'])}/4 | observed actions: {init['buffered_event_count']}"
        cv2.putText(frame,label,(8,h-12),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,210,255),1,cv2.LINE_AA)
    return frame


def run_prediction(args,*,on_frame=None,on_log=None,should_stop=None,roi_polygon=None):
    """接口兼容现有UI7V：roi_polygon参数携带ROI及视频信息。"""
    context = prepare_context(args) if roi_polygon is None else roi_polygon
    polygon = context["polygon"]
    def log(message):
        print(message,flush=True)
        if on_log is not None:
            on_log(message)
    group_completion = {}
    def log_group_completion(snapshot):
        group_id,complete = snapshot["group_id"],process_completed(snapshot)
        key = (snapshot["phase"], group_id)
        if group_completion.get(key) is None or group_completion[key] != complete:
            action = "下料" if snapshot["phase"] == "unload" else "上料"
            log(f"游星片{group_id}：{action}{'完成' if complete else '未完成'}")
            group_completion[key] = complete
    original_load = torch.load
    def trusted_load(*a,**kw):
        kw.setdefault("weights_only",False)
        return original_load(*a,**kw)
    torch.load = trusted_load
    try:
        model,architecture = base.load_detection_model(args.weights)
    finally:
        torch.load = original_load
    sequence = make_sequence(polygon,context["size"])
    tracker = sequence.tracker
    output = args.output_dir or Path(OUTPUT_ROOT)/(args.source.stem+"_slots_"+datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True,exist_ok=False)
    cap,writer = cv2.VideoCapture(str(args.source)),None
    processed,index,status = 0,context["start_frame"],"failed"
    started = time.perf_counter()
    summary = {"source": str(args.source.resolve()),"weights": str(args.weights.resolve()),
        "output_dir": str(output.resolve()),"output_video": str((output/"annotated.mp4").resolve()),
        "architecture": architecture,"conf": args.conf,"imgsz": args.imgsz,"stride": args.stride,
        "group_id": CURRENT_GROUP_ID,"group_count": 5,"phase": PROCESS_PHASE,"clockwise_order": CLOCKWISE_ORDER,
        "auto_load_after_unload": sequence.auto_load_after_unload,
        "automatic_grouping_implemented": True,"automatic_grouping_enabled": AUTO_GROUP_SWITCH,
        "group_assignment": "roi_exit_sequence" if AUTO_GROUP_SWITCH else "configured",
        "group_switch_config": {"exit_confirm_seconds": EXIT_CONFIRM_SECONDS,
            "single_exit_immediate_entry": EXIT_MIN_SLOTS == 1,
            "boundary_margin_px": EXIT_BOUNDARY_MARGIN_PX,"exit_min_slots": EXIT_MIN_SLOTS,
            "partial_exit_min_slots": PARTIAL_EXIT_MIN_SLOTS,"min_motion_ratio": EXIT_MIN_MOTION_RATIO,
            "motion_pause_ratio": MOTION_PAUSE_RATIO,"entry_seconds": NEXT_GROUP_STABLE_SECONDS,
            "entry_stability_ratio": NEXT_GROUP_STABILITY_RATIO,
            "entry_match_distance_px": NEXT_GROUP_MATCH_DISTANCE_PX,"exit_direction": EXIT_DIRECTION},
        "center_source": "yolo_box_centers","initialization_seconds": INITIALIZATION_SECONDS,
        "min_initialization_frames": MIN_INITIALIZATION_FRAMES,
        "initialization_history_seconds": INITIALIZATION_HISTORY_SECONDS,
        "initialization_stability_ratio": tracker.initialization_stability_ratio,
        "numbering_method": "initial_x_ascending_y_tiebreak_then_track",
        "roi_polygon_normalized": polygon.tolist(),"roi_tracking_margin_px": ROI_TRACKING_MARGIN_PX,
        "confirm_seconds": CONFIRM_SECONDS,"min_confirm_frames": MIN_CONFIRM_FRAMES,
        "load_hand_contact_config": {"enabled": LOAD_HAND_CONTACT_ENABLED,
            "contact_seconds": LOAD_HAND_CONTACT_SECONDS,
            "min_frames": LOAD_HAND_CONTACT_MIN_FRAMES,"infer_from_contact": False},
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
        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
        with ExitStack() as stack:
            records = stack.enter_context((output/"slot_states.jsonl").open("w",encoding="utf-8"))
            events = stack.enter_context((output/"slot_events.jsonl").open("w",encoding="utf-8"))
            group_events = stack.enter_context((output/"group_events.jsonl").open("w",encoding="utf-8"))
            csv_file = stack.enter_context((output/"slot_states.csv").open("w",encoding="utf-8-sig",newline=""))
            fields = ["source_frame","time_seconds","group_id","slot_id","phase","state","observation",
                      "last_confirmed_state","confidence","center_x","center_y","reason","order_result","group_status",
                      "slot_numbering_status"]
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
                result = model.predict(frame,imgsz=args.imgsz,conf=args.conf,device=args.device,
                    iou=args.nms_iou,agnostic_nms=False,verbose=False,save=False)[0]
                boxes = result.boxes.cpu()
                raw = [{"class_id": int(c),"class_name": model.names[int(c)],"confidence": float(score),"xyxy": xy}
                    for c,score,xy in zip(boxes.cls.tolist(),boxes.conf.tolist(),boxes.xyxy.tolist())]
                timestamp = source_frame/fps
                enabled = not any(start <= timestamp < end for start,end in ROTATION_INTERVALS)
                objects,local_attempts = recover_missing_slots(model,frame,raw,sequence,args) if enabled else (raw,[])
                snapshot = sequence.update(objects,timestamp,observations_enabled=enabled)
                tracker = sequence.tracker
                record = {"source_frame": source_frame,"time_seconds": timestamp,**snapshot,
                          "local_redetection": local_attempts}
                records.write(json.dumps(record,ensure_ascii=False)+"\n")
                for event in snapshot["new_events"]:
                    events.write(json.dumps(event,ensure_ascii=False)+"\n")
                for event in snapshot["new_group_events"]:
                    group_events.write(json.dumps(event,ensure_ascii=False)+"\n")
                    if event["type"] in ("group_closed", "group_updated"):
                        log_group_completion(event["snapshot"])
                log_group_completion(snapshot)
                for slot in snapshot["slots"]:
                    center = slot["center_xy"] or (None,None)
                    rows.writerow({"source_frame": source_frame,"time_seconds": round(timestamp,6),
                        **{key: slot[key] for key in ("group_id","slot_id","state","observation","last_confirmed_state","confidence","reason")},
                        "phase": tracker.phase,"center_x": center[0],"center_y": center[1],"order_result": snapshot["order_result"],
                        "group_status": snapshot["group_status"],"slot_numbering_status": snapshot["slot_numbering_status"]})
                others = [o for o in raw if o["class_name"] not in SLOT_STATES]
                plotted = draw_detection_points(frame,others,radius=POINT_RADIUS)
                if DRAW_SLOT_ROI:
                    draw_roi(plotted,polygon)
                overlay_status(plotted,snapshot,SHOW_SLOT_TEXT)
                writer.write(plotted)
                processed += 1
                slot_label = "临时点" if snapshot["slot_numbering_status"] == "provisional" else "槽"
                text = "；".join(f"组{s['group_id']}-{slot_label}{s['slot_id']}:{STATE_TEXT[s['state']]}" for s in snapshot["slots"])
                if on_frame is not None:
                    identity = GROUP_TEXT[snapshot["group_status"]]
                    on_frame(plotted,{"status": f"组{tracker.group_id} {tracker.phase} · {timestamp:.2f}s · {identity} · 顺序{snapshot['order_result']}",
                        "result": text,"frame_id": processed,"source_frame": source_frame,"time_seconds": timestamp,
                        "slots": snapshot["slots"],"group_id": tracker.group_id,"phase": tracker.phase,
                        "group_status": snapshot["group_status"],"pending_group_id": snapshot["pending_group_id"],
                        "pending_phase": snapshot["pending_phase"],
                        "pending_initialization": snapshot["pending_initialization"],
                        "slot_numbering_status": snapshot["slot_numbering_status"]})
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
            final = sequence.finish()
            for event in final["new_events"]:
                events.write(json.dumps(event,ensure_ascii=False)+"\n")
            for event in final["new_group_events"]:
                group_events.write(json.dumps(event,ensure_ascii=False)+"\n")
                log_group_completion(event["snapshot"])
            if processed:
                log_group_completion(final)
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
        summary.update(status=status,processed_frames=processed,elapsed_seconds=round(time.perf_counter()-started,3),
                       final_group_snapshot=sequence.snapshot(),events=sequence.all_events,
                       closed_groups=sequence.closed_groups)
        summary["group_process_report"] = build_process_report(summary)
        phases = [summary["phase"]]
        if sequence.auto_load_after_unload and summary["phase"] == "unload":
            phases.append("load")
        summary["group_process_reports"] = {phase:build_process_report(summary,phase) for phase in phases}
        summary["current_phase"] = sequence.tracker.phase
        report_text = "\n\n".join(format_process_report(report)
                                 for report in summary["group_process_reports"].values())
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
    args = build_args()
    if args.ui:
        from industrial_inspection.tools.predict_video_ui import launch_ui
        launch_ui(args,run_prediction,prepare_context)
    else:
        run_prediction(args)


if __name__ == "__main__":
    main()
