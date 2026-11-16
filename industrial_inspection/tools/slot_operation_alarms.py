"""Turn confirmed slot evidence into order and group completion alarms."""

from industrial_inspection.tools.slot_group_tracker import sort_events_by_time_upper


def _slot_label(slot_id, numbering):
    number = numbering.get(slot_id)
    return f"{number}（S{slot_id}）" if number is not None else f"S{slot_id}"


def group_operation_alarms(snapshot, events, numbering, now, *, close_reason=None):
    phase, group_id = snapshot["phase"], snapshot["group_id"]
    action = "下料" if phase == "unload" else "上料"
    formal = snapshot["slot_numbering_status"] == "final"
    identity = {"at": float(now), "phase": phase, "group_id": group_id}
    alarms = []
    if (formal and snapshot["order_result"] == "fail"
            and (close_reason is not None or not snapshot.get("s4_inferred_load_pending"))):
        ordered = sort_events_by_time_upper(events)
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
