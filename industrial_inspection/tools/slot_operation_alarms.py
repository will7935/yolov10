"""Turn confirmed slot evidence into order and group completion alarms."""

from collections import deque

import numpy as np

from industrial_inspection.tools.slot_group_tracker import sort_events_by_time_upper, slot_event_order_result


class SlotMotionAlarmReview:
    def __init__(self):
        self.groups = {}
        self.current = None

    @property
    def paused(self):
        if self.current is None:
            return True
        record = self.groups[self.current]
        return not record['stationary'] or record['motion_candidate'] is not None

    @staticmethod
    def _moving_pair(previous, current, threshold):
        vectors = [center-previous[tid] for tid, center in current.items() if tid in previous]
        vectors = [v for v in vectors if np.linalg.norm(v) >= threshold]
        return any(.5 <= np.linalg.norm(a)/np.linalg.norm(b) <= 2.
                   and np.dot(a, b) >= .8*np.linalg.norm(a)*np.linalg.norm(b)
                   for i, a in enumerate(vectors) for b in vectors[i+1:])

    def observe_group(self, snapshot, now):
        key = (snapshot['phase'], snapshot['group_id'])
        if self.current is not None and key != self.current:
            previous = self.groups[self.current]
            if previous['stationary']:
                previous['ranges'][-1]['end'] = now
                previous['stationary'] = False
        self.current = key
        record = self.groups.setdefault(key, {'stationary': False, 'history': deque(),
                                              'ranges': [], 'motion_candidate': None})
        points = {s['byte_track_id']: np.asarray(s['center_xy'], dtype=float)
                  for s in snapshot['slots'] if s.get('byte_track_id') is not None
                  and s.get('center_xy') is not None and not s.get('geometry_inferred')
                  and s.get('last_seen_at') == now and s.get('observation') in ('empty', 'filled')}
        boxes = [s['tracking_box_xyxy'] for s in snapshot['slots']
                 if s.get('byte_track_id') in points and s.get('tracking_box_xyxy') is not None]
        observations = {s['byte_track_id']: s['observation'] for s in snapshot['slots']
                        if s.get('byte_track_id') in points}
        scale = float(np.median([b[3]-b[1] for b in boxes]))/50. if boxes else 1.
        history = record['history']
        history.append((now, points, observations))
        while history and now-history[0][0] > .35:
            history.popleft()
        if len(history) < 2 or now-history[0][0] < .25-1e-9:
            return
        start, previous, prior_observations = history[0]
        consistent = {tid: center for tid, center in points.items()
                      if observations[tid] == prior_observations.get(tid)}
        if record['stationary']:
            moving = self._moving_pair(previous, consistent, max(4., 6.*scale))
            if moving:
                if record['motion_candidate'] is None:
                    record['motion_candidate'] = (now, start)
                if now-record['motion_candidate'][0] >= .1-1e-9:
                    record['ranges'][-1]['end'] = max(record['ranges'][-1]['start'],
                                                     record['motion_candidate'][1])
                    record['stationary'] = False
                    record['motion_candidate'] = None
                    history.clear()
                    history.append((now, points, observations))
                    return
            else:
                record['motion_candidate'] = None
        if not record['stationary']:
            common = set.intersection(*(set(p) for _, p, _ in history))
            stable = [tid for tid in common if max(np.linalg.norm(p[tid]-points[tid])
                      for _, p, _ in history) <= max(2., 3.*scale)]
            if len(stable) >= 2:
                record['stationary'] = True
                record['ranges'].append({'start': start, 'end': None})

    def review(self, snapshot, events):
        record = self.groups.get((snapshot['phase'], snapshot['group_id']))
        if record is None:
            return list(events), []
        retained, ignored = [], []
        for event in events:
            at = event.get('time_upper', event.get('confirmed_at'))
            valid = any(r['start']-1e-6 <= at and (r['end'] is None or at <= r['end']+1e-6)
                        for r in record['ranges'])
            if valid:
                retained.append(event)
            else:
                ignored.append(dict(event, alarm_review_reason='outside_stationary_operation_window'))
        return retained, ignored


def _slot_label(slot_id, numbering):
    number = numbering.get(slot_id)
    return f"{number}（S{slot_id}）" if number is not None else f"S{slot_id}"


def group_operation_alarms(snapshot, events, numbering, now, *, close_reason=None,
                           clockwise=(1, 3, 4, 2), motion_review=None):
    phase, group_id = snapshot["phase"], snapshot["group_id"]
    action = "下料" if phase == "unload" else "上料"
    formal = snapshot["slot_numbering_status"] == "final"
    identity = {"at": float(now), "phase": phase, "group_id": group_id}
    alarms = []
    effective_events = sort_events_by_time_upper(events)
    ignored_motion = []
    if motion_review is not None:
        effective_events, ignored_motion = motion_review.review(snapshot, effective_events)
    result = (slot_event_order_result(effective_events, tuple(clockwise))
              if ignored_motion else snapshot["order_result"])
    snapshot["alarm_order_result"] = result
    snapshot["alarm_ignored_motion_events"] = ignored_motion
    snapshot["order_alarm_paused_for_motion"] = bool(motion_review is not None and motion_review.paused)
    if (formal and result == "fail"
            and (motion_review is None or not motion_review.paused or close_reason == 'hatch_closed')
            and (close_reason is not None or not snapshot.get("s4_inferred_load_pending"))):
        ordered = sort_events_by_time_upper(effective_events)
        sequence = [event["slot_id"] for event in ordered]
        if any(not event["expected_action"] for event in ordered):
            reason, detail = "unexpected_action", "出现与当前工序相反的动作"
        elif len(set(sequence)) < len(sequence):
            reason, detail = "repeated_slot", "同一槽位重复操作"
        else:
            reason, detail = "not_clockwise", "无法按顺时针环序解释"
        headline = f"报警 G{group_id} {action}顺序异常"
        original = " → ".join(f"S{slot_id}" for slot_id in sequence)
        continuous_sequence = [numbering.get(i) for i in sequence]
        continuous = " → ".join(str(number) if number is not None else "待编号"
                                for number in continuous_sequence) if numbering else "起点待确认"
        alarms.append({**identity, "type": "slot_order_alarm", "reason": reason,
            "slot_sequence": sequence, "continuous_sequence": continuous_sequence,
            "headline": headline, "message": (f"{headline}：{detail}\n原槽号顺序 {original}"
                                              f"\n跨组顺时针编号顺序 {continuous}")})
    if close_reason is None:
        return alarms

    target = "empty" if phase == "unload" else "filled"
    states = {slot["slot_id"]: slot["last_confirmed_state"] for slot in snapshot["slots"]} if formal else {}
    missing = []
    details = []
    for slot_id in range(1, 5):
        state = states.get(slot_id, "unknown")
        if state == target:
            continue
        missing.append({"slot_id": slot_id, "continuous_slot_id": numbering.get(slot_id),
                        "last_confirmed_state": state})
        state_text = {"filled": "仍满", "empty": "仍空"}.get(state, "未确认")
        details.append(f"{_slot_label(slot_id, numbering)} {state_text}")
    if missing:
        confirmed_count = 4-len(missing)
        headline = f"报警 G{group_id} {action}未完成（{confirmed_count}/4）"
        alarms.append({**identity, "type": "group_incomplete_alarm", "reason": close_reason,
            "target_state": target, "confirmed_count": confirmed_count, "slots": missing,
            "headline": headline, "message": f"{headline}：{'，'.join(details)}"})
    return alarms
