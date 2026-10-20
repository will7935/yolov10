"""把手开关确认和五组二十槽满槽统计；漏检不代表关门。"""
from __future__ import annotations

from copy import deepcopy
import math

import cv2
import numpy as np


class HatchHandleMonitor:
    """固定相机中用已标定的把手中心区域判断开关，连续证据确认后才切换。"""

    def __init__(self, open_polygon, closed_polygon, *, confirm_seconds=.3,
                 minimum_frames=3, maximum_gap_seconds=.25, confidence=.35,
                 distance_mode=True, closed_distance_max=.08,
                 open_distance_min=.14, missing_open_seconds=1.0,
                 reference_update_alpha=.05):
        self.open_polygon = self._polygon(open_polygon)
        self.closed_polygon = self._polygon(closed_polygon)
        if cv2.intersectConvexConvex(self.open_polygon, self.closed_polygon)[0] > 1e-8:
            raise ValueError("把手开、关位置区域不能重叠")
        if (not all(math.isfinite(v) for v in (confirm_seconds, maximum_gap_seconds, confidence,
                                              closed_distance_max, open_distance_min,
                                              missing_open_seconds, reference_update_alpha))
                or confirm_seconds < 0 or maximum_gap_seconds <= 0
                or minimum_frames < 1 or int(minimum_frames) != minimum_frames or not 0 < confidence <= 1
                or closed_distance_max < 0 or open_distance_min <= closed_distance_max
                or missing_open_seconds <= 0 or not 0 <= reference_update_alpha <= 1):
            raise ValueError("把手确认时间、帧数、间隔或置信度无效")
        self.confirm_seconds, self.minimum_frames = confirm_seconds, minimum_frames
        self.maximum_gap_seconds, self.confidence = maximum_gap_seconds, confidence
        self.distance_mode = bool(distance_mode)
        self.closed_distance_max, self.open_distance_min = closed_distance_max, open_distance_min
        self.missing_open_seconds = missing_open_seconds
        self.reference_update_alpha = reference_update_alpha
        self.state = self.observation = "unknown"
        self.candidate, self.candidate_since, self.candidate_count = None, None, 0
        self.candidate_last_seen = self.last_time = None
        self.closed_reference = None
        self.current_center = None
        self.distance_from_closed = None
        self.last_handle_seen = None

    @staticmethod
    def _polygon(value):
        polygon = np.asarray(value, dtype=np.float32)
        if (polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3
                or not np.isfinite(polygon).all() or np.any(polygon < 0) or np.any(polygon > 1)
                or not cv2.isContourConvex(polygon) or cv2.contourArea(polygon) <= 0):
            raise ValueError("把手区域须为0..1归一化的凸多边形")
        return polygon

    def update(self, objects, now, size):
        if not math.isfinite(now) or self.last_time is not None and now <= self.last_time:
            raise ValueError("把手检测时间须有限且严格递增")
        if len(size) != 2 or not all(math.isfinite(v) and v > 0 for v in size):
            raise ValueError("画面宽高须为正有限数")
        self.last_time = float(now)
        support = set()
        centers = []
        for obj in objects:
            if obj['class_name'] != 'hatch_handle' or obj['confidence'] < self.confidence:
                continue
            box = np.asarray(obj['xyxy'], dtype=float)
            if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
                continue
            center = tuple((box[:2]+box[2:])/2/size)
            centers.append(np.asarray(center, dtype=float))
            for state, polygon in (('open', self.open_polygon), ('closed', self.closed_polygon)):
                if cv2.pointPolygonTest(polygon, center, False) >= 0:
                    support.add(state)
        self.current_center = np.mean(centers, axis=0).tolist() if centers else None
        if centers:
            self.last_handle_seen = now
        self.distance_from_closed = None
        if len(support) > 1:
            # 同帧出现开、关两侧证据时优先视为冲突；不能让中心平均值
            # 恰好落在距离迟滞带内而绕过冲突保护。
            self.observation = 'unknown'
        elif self.distance_mode and centers:
            center = np.mean(centers, axis=0)
            if self.closed_reference is not None:
                self.distance_from_closed = float(np.linalg.norm(center-self.closed_reference))
                if self.distance_from_closed <= self.closed_distance_max:
                    self.observation = 'closed'
                elif self.distance_from_closed >= self.open_distance_min:
                    self.observation = 'open'
                else:
                    # 迟滞带内不猜测，避免把手经过中间位置时来回切换。
                    self.observation = 'unknown'
            elif 'closed' in support:
                # 只用确认前的关门区域建立初始基准；正式判断使用距离。
                self.closed_reference = center.copy()
                self.observation = 'closed'
            elif 'open' in support:
                self.observation = 'open'
            else:
                self.observation = 'unknown'
        else:
            self.observation = next(iter(support)) if len(support) == 1 else 'unknown'
        if (not centers and self.distance_mode and self.state == 'closed'
                and self.last_handle_seen is not None
                and now-self.last_handle_seen+1e-9 >= self.missing_open_seconds):
            # 已确认关门后，把手持续消失是开门时离开检测区域的证据；
            # 初始未知状态不会因漏检直接启动整段流程。
            self.observation = 'open'
        if (self.candidate_last_seen is not None
                and now-self.candidate_last_seen > self.maximum_gap_seconds+1e-9):
            self.candidate, self.candidate_since, self.candidate_count = None, None, 0
        if self.observation == 'unknown':
            # 同时支持开、关是冲突，不能把两边证据拼接成一次确认。
            if support:
                self.candidate, self.candidate_since, self.candidate_count = None, None, 0
            return []
        if self.candidate != self.observation:
            self.candidate, self.candidate_since, self.candidate_count = self.observation, now, 0
        self.candidate_count += 1
        self.candidate_last_seen = now
        if (self.candidate_count < self.minimum_frames
                or now-self.candidate_since+1e-9 < self.confirm_seconds or self.state == self.candidate):
            return []
        previous, self.state = self.state, self.candidate
        if self.state == 'closed' and self.current_center is not None:
            point = np.asarray(self.current_center, dtype=float)
            if self.closed_reference is None:
                self.closed_reference = point
            else:
                self.closed_reference = ((1-self.reference_update_alpha)*self.closed_reference
                                         + self.reference_update_alpha*point)
        return [{'type': 'hatch_opened' if self.state == 'open' else 'hatch_closed',
                 'previous_state': previous, 'state': self.state,
                 'observed_since': float(self.candidate_since), 'at': float(now)}]

    def snapshot(self):
        return {'state': self.state, 'observation': self.observation,
                'candidate': self.candidate, 'candidate_frames': self.candidate_count,
                'distance_mode': self.distance_mode,
                'current_center': self.current_center,
                'closed_reference': None if self.closed_reference is None else self.closed_reference.tolist(),
                'distance_from_closed_reference': self.distance_from_closed,
                'seconds_without_handle': None if self.last_handle_seen is None or self.last_time is None
                    else max(0., self.last_time-self.last_handle_seen)}


def build_loading_fullness_report(sequence, *, started, finalized=False):
    """只统计本轮上料的正式物理槽身份，首次看见已满也可确认占用。"""
    snapshots = {g['group_id']: g['snapshot'] for g in sequence.closed_groups
                 if g['phase'] == 'load'} if started else {}
    current = sequence.snapshot()
    if started and current['phase'] == 'load' and current['group_id'] not in snapshots:
        snapshots[current['group_id']] = current
    groups = []
    for group_id in range(1, 6):
        snapshot = snapshots.get(group_id)
        formal = snapshot is not None and snapshot.get('slot_numbering_status') == 'final'
        slots = {s['slot_id']: s for s in snapshot['slots']} if formal else {}
        filled = [i for i in range(1, 5) if slots.get(i, {}).get('last_confirmed_state') == 'filled']
        empty = [i for i in range(1, 5) if slots.get(i, {}).get('last_confirmed_state') == 'empty']
        unknown = [i for i in range(1, 5) if i not in filled and i not in empty]
        groups.append({'group_id': group_id, 'observed': snapshot is not None,
                       'slot_numbering_status': snapshot.get('slot_numbering_status') if snapshot else None,
                       'filled_slot_ids': filled, 'empty_slot_ids': empty, 'unknown_slot_ids': unknown,
                       'filled_count': len(filled), 'all_filled': len(filled) == 4,
                       'result': '已放满' if len(filled) == 4 else '未放满'})
    count = sum(g['filled_count'] for g in groups)
    return {'phase': 'load', 'started': bool(started), 'finalized': bool(finalized and started),
            'required_slot_count': 20, 'confirmed_filled_count': count, 'all_filled': count == 20,
            'result': '已放满' if count == 20 else '未放满', 'groups': groups}


class HatchLoadingInspection:
    """一次开门至关门的工序；保留原下料到上料流程，关门冻结本轮统计。"""

    def __init__(self, monitor):
        self.monitor = monitor
        self.started_at = self.closed_at = None
        self._group_results = {}
        self._final_reported = False
        self.report = None
        self.events = []

    @property
    def active(self):
        return self.started_at is not None and self.closed_at is None

    def observe_handle(self, objects, now, size):
        events = self.monitor.update(objects, now, size)
        for event in events:
            if event['state'] == 'open' and self.started_at is None:
                self.started_at = now
                event['inspection_started'] = True
            elif event['state'] == 'closed' and self.active:
                self.closed_at = now
                event['inspection_finished'] = True
        self.events.extend(deepcopy(events))
        return events

    def update_fullness(self, sequence, now):
        self.report = build_loading_fullness_report(sequence, started=self.started_at is not None,
                                                  finalized=self.closed_at is not None)
        self.report.update(started_at=self.started_at, closed_at=self.closed_at)
        events = []
        closed = {g['group_id'] for g in sequence.closed_groups if g['phase'] == 'load'}
        verifying = sequence.departing_load_tracker
        pending_id = verifying.group_id if verifying is not None else None
        if self.started_at is not None:
            for group in self.report['groups']:
                group_id = group['group_id']
                ready = (group['all_filled'] or self.closed_at is not None
                         or group_id in closed and group_id != pending_id or group_id in self._group_results)
                key = tuple(group['filled_slot_ids'])
                if ready and self._group_results.get(group_id) != key:
                    self._group_results[group_id] = key
                    events.append({'type': 'group_fullness', 'at': float(now), **deepcopy(group)})
        if self.closed_at is not None and not self._final_reported:
            events.append({'type': 'loading_fullness_final', 'at': float(now), **deepcopy(self.report)})
            self._final_reported = True
        self.events.extend(deepcopy(events))
        return events

    def snapshot(self):
        return {'status': 'completed' if self.closed_at is not None else
                'detecting' if self.active else 'waiting_open',
                'started_at': self.started_at, 'closed_at': self.closed_at,
                'handle': self.monitor.snapshot(), 'fullness': deepcopy(self.report)}


def format_fullness_report(report):
    lines = ['上料满槽核验' + ('（关门最终结果）' if report['finalized'] else '（尚未确认关门，阶段统计）')]
    lines.extend(f"游星片{g['group_id']}：{g['result']}（{g['filled_count']}/4）" for g in report['groups'])
    lines.append(f"{'最终结果' if report['finalized'] else '当前统计'}：{report['result']}"
                 f"（{report['confirmed_filled_count']}/20）")
    return '\n'.join(lines)
