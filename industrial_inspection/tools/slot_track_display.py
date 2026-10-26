"""ByteTrack-ID-based display and per-detection state records for slot tracks."""
from __future__ import annotations

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES


def slot_track_record(item, snapshot):
    """Attach the tracked slot's maintained status to a ByteTrack detection."""
    track_id = item.get("track_id")
    slot = next((slot for slot in snapshot.get("slots", [])
                 if track_id is not None and slot.get("byte_track_id") == track_id), None)
    record = dict(item)
    independent = next((track for track in snapshot.get('track_states', [])
                        if track_id is not None and track.get('byte_track_id') == track_id), None)
    if slot is None:
        record.update(slot_id=None, slot_state="unknown", last_confirmed_state="unknown",
                      formal_slot_id=None,
                      observation=SLOT_STATES.get(item.get("class_name"), "unknown"),
                      state_reason="unmapped_track", confirmation_source=None, group_id=None)
        if independent is not None:
            record.update(slot_id=independent.get('slot_id'), formal_slot_id=independent.get('formal_slot_id'),
                          slot_state=independent['state'], last_confirmed_state=independent['last_confirmed_state'],
                          observation=independent['observation'], state_reason=independent['reason'],
                          confirmation_source=independent.get('confirmation_source'),
                          group_id=independent.get('group_id'), phase=independent.get('phase'))
    else:
        formal = (slot["slot_id"] if snapshot.get("slot_numbering_status") == "final" else None)
        record.update(slot_id=formal, formal_slot_id=formal, slot_state=slot["state"],
                      last_confirmed_state=slot["last_confirmed_state"],
                      observation=slot["observation"], state_reason=slot["reason"],
                      confirmation_source=slot.get("confirmation_source"), group_id=slot["group_id"])
        if (formal is None and independent is not None
                and slot.get('confirmation_source') != 'hand_contact'):
            # Waiting for a coherent quartet must not hide a track's otherwise
            # confirmed visual state (e.g. overflow pauses the group observer).
            record.update(slot_state=independent['state'], observation=independent['observation'],
                          last_confirmed_state=independent['last_confirmed_state'],
                          state_reason=independent['reason'],
                          confirmation_source=independent.get('confirmation_source'))
    record['mapping_reason'] = (independent.get('mapping_reason') if independent is not None
                                else 'bytetrack_id_pending' if track_id is None else
                                'unmapped_track' if slot is None else
                                'mapped' if record['formal_slot_id'] is not None else 'waiting_slot_numbering')
    record['mapping_status'] = ('final' if record['formal_slot_id'] is not None else
                                'provisional' if record['group_id'] is not None else 'unassigned')
    return record


def slot_track_label(item, snapshot):
    """Format a detection label from its ByteTrack identity and maintained state."""
    record = slot_track_record(item, snapshot)
    state_label = record["slot_state"]
    if state_label == "unknown" and record["last_confirmed_state"] != "unknown":
        state_label = f"{record['last_confirmed_state']}*"
    slot_label = ""
    if record['mapping_status'] == 'final':
        slot_label = f"G{record['group_id']}-S{record['slot_id']} "
    else:
        slot_label = (f"G{record['group_id']} " if record['group_id'] is not None else '') + 'S:pending '
    detected = SLOT_STATES.get(item.get("class_name"), "unknown")
    return (f"BT:{item.get('track_id') if item.get('track_id') is not None else '-'} "
            f"{slot_label}{state_label} det:{detected} {item['confidence']:.2f}")
