"""Record YOLO state evidence by persistent ByteTrack ID before S numbering."""
from __future__ import annotations

from copy import deepcopy

import cv2

from industrial_inspection.tools.slot_group_tracker import SLOT_STATES, SlotState


class SlotTrackStateStore:
    """A track's visual history is independent of its group/slot assignment."""

    def __init__(self, tracker, contour, *, reidentification_enabled=False,
                 reidentification_max_gap_seconds=15.0,
                 reidentification_max_distance_px=40.0):
        self.rules = tracker
        self.contour = contour
        self.tracks = {}
        self.closed = set()
        self.rejected = {}
        self.reidentification_enabled = bool(reidentification_enabled)
        self.reidentification_max_gap_seconds = float(reidentification_max_gap_seconds)
        self.reidentification_max_distance_px = float(reidentification_max_distance_px)
        self.aliases = {}
        self.reidentification_events = []
        self._seen_this_frame = set()

    def __deepcopy__(self, memo):
        return self

    def observe(self, objects, now, enabled=True, reentry_allowed=None):
        hands = [o for o in objects if o['class_name'] == 'hand']
        candidates = self.rules._candidates(objects)
        accepted = {obj.get('track_id') for obj in candidates}
        self.rejected = {obj['track_id']: 'duplicate_slot_detection' for obj in objects
                         if obj['class_name'] in SLOT_STATES and obj.get('track_id') is not None
                         and obj['track_id'] not in accepted}
        seen = set()
        for obj in candidates:
            track_id = obj.get('track_id')
            if track_id is None:
                continue
            track_id = int(track_id)
            seen.add(track_id)
            inside = cv2.pointPolygonTest(self.contour, tuple(obj['center']), False) >= 0
            record = self.tracks.get(track_id)
            if (record is not None and record['owner'] in self.closed and inside
                    and now-record['last_inside'] > self.rules.lost_seconds
                    and (reentry_allowed is None or reentry_allowed(obj['center']))):
                record = None
            if record is None:
                record = dict(state=SlotState(0), previous=SlotState(0), events=[],
                              imported=set(), owner=None, slot_id=None,
                              first_seen=float(now), last_seen=float(now),
                              last_inside=float(now), reliable_since=None, reliable_count=0,
                              logical_track_id=track_id, reidentified_from_track_id=None,
                              reid_gap_seconds=None, reid_method=None)
                self.tracks[track_id] = record
            if now-record['last_seen'] > self.rules.lost_seconds and record['owner'] is None:
                record['state'] = SlotState(0)
                record['events'].clear()
                record['reliable_since'], record['reliable_count'] = None, 0
            record['previous'] = deepcopy(record['state'])
            gap = now-record['last_seen']
            record.update(last_seen=float(now), center=list(obj['center']), xyxy=list(obj['xyxy']))
            if inside:
                record['last_inside'] = float(now)
            state, reason = 'unknown', 'not_detected'
            if not enabled:
                reason = 'inspection_paused'
            elif not inside:
                reason = 'outside_roi'
            elif obj['conflict']:
                reason = 'empty_filled_conflict'
            elif self.rules._occluded(obj, hands):
                reason = 'hand_occlusion'
            else:
                state, reason = SLOT_STATES[obj['class_name']], 'detected'
            if state != 'unknown':
                if record['reliable_since'] is None or gap > .25:
                    record['reliable_since'], record['reliable_count'] = float(now), 0
                record['reliable_count'] += 1
            else:
                record['reliable_since'], record['reliable_count'] = None, 0
            event = record['state'].observe(state, now, obj['confidence'], reason,
                                           self.rules.confirm_seconds, self.rules.minimum_frames)
            if event is not None:
                record['events'].append({**event, 'byte_track_id': track_id})
        for track_id, record in self.tracks.items():
            if track_id not in seen:
                record['previous'] = deepcopy(record['state'])
                record['state'].observe('unknown', now, None, 'not_detected',
                                        self.rules.confirm_seconds, self.rules.minimum_frames)
                record['reliable_since'], record['reliable_count'] = None, 0
        self._seen_this_frame = seen

    def owner(self, track_id):
        record = self.tracks.get(track_id)
        return None if record is None else record['owner']

    def bind(self, track_id, identity, slot_id=None, *, center=None, now=None):
        record = self.tracks.get(track_id)
        if record is None or record['owner'] not in (None, identity):
            return False
        record['owner'] = identity
        if slot_id is not None:
            record['slot_id'] = slot_id
            self._try_reidentify(record, int(track_id), identity, int(slot_id), center, now)
        return True

    def _try_reidentify(self, record, track_id, identity, slot_id, center, now):
        """Attach a new ByteTrack ID to a prior logical ID using slot geometry."""
        if (not self.reidentification_enabled or center is None or now is None
                or record.get('reidentified_from_track_id') is not None):
            return False
        try:
            center = [float(center[0]), float(center[1])]
            now = float(now)
        except (TypeError, ValueError, IndexError):
            return False
        candidates = []
        for old_id, old in self.tracks.items():
            if int(old_id) == track_id or int(old_id) in self._seen_this_frame:
                continue
            if old.get('owner') != identity or old.get('slot_id') != slot_id:
                continue
            old_center = old.get('center')
            old_seen = old.get('last_seen')
            if old_center is None or old_seen is None:
                continue
            gap = now - float(old_seen)
            if gap <= 0.0 or gap > self.reidentification_max_gap_seconds:
                continue
            distance = ((center[0] - float(old_center[0])) ** 2
                        + (center[1] - float(old_center[1])) ** 2) ** .5
            if distance <= self.reidentification_max_distance_px:
                candidates.append((distance, -float(old_seen), int(old_id), gap))
        if not candidates:
            return False
        candidates.sort()
        distance, _, old_id, gap = candidates[0]
        # A second old record at almost the same location makes the identity
        # ambiguous; leave the new ByteTrack ID untouched in that case.
        if len(candidates) > 1 and candidates[1][0] - distance < 3.0:
            return False
        old = self.tracks[old_id]
        logical_id = int(old.get('logical_track_id') or old_id)
        record.update(logical_track_id=logical_id,
                      reidentified_from_track_id=old_id,
                      reid_gap_seconds=round(gap, 6),
                      reid_method='stationary_slot_geometry')
        self.aliases[track_id] = logical_id
        event = {'track_id': track_id, 'logical_track_id': logical_id,
                 'reidentified_from_track_id': old_id, 'group_id': identity[1],
                 'phase': identity[0], 'slot_id': slot_id,
                 'gap_seconds': round(gap, 6), 'distance_px': round(distance, 3),
                 'method': 'stationary_slot_geometry', 'at': now}
        self.reidentification_events.append(event)
        return True

    def logical_id(self, track_id):
        if track_id is None:
            return None
        record = self.tracks.get(int(track_id))
        return int(record.get('logical_track_id') or track_id) if record is not None else int(track_id)

    def release_candidate(self, track_id, identity):
        record = self.tracks.get(track_id)
        if record and record['owner'] == identity and record['slot_id'] is None:
            record['owner'] = None
            record['imported'].clear()

    def restore(self, slot, track_id, identity, now):
        record = self.tracks.get(track_id)
        if record is None or not self.bind(track_id, identity):
            return []
        imported = getattr(slot, '_restored_track_ids', set())
        if track_id in imported:
            return []
        imported.add(track_id)
        slot._restored_track_ids = imported
        # A late-mapped slot may already have current-frame evidence.  That
        # must not prevent importing older state-change events from the track;
        # only restore the slot's visual state when it has no local evidence.
        if slot.confirmed == 'unknown' and not slot.candidate_count:
            prior = record['previous']
            for key in ('confirmed', 'evidence_time', 'confirmed_since', 'confirmed_at',
                        'candidate', 'candidate_since', 'candidate_count', 'last_seen',
                        'confirmation_source'):
                setattr(slot, key, deepcopy(getattr(prior, key)))
        events = []
        for index, event in enumerate(record['events']):
            if event['confirmed_at'] >= now or index in record['imported']:
                continue
            record['imported'].add(index)
            events.append({**event, 'slot_id': slot.slot_id, 'group_id': identity[1],
                           'phase': identity[0], 'expected_action': event['action'] == identity[0],
                           'reported_at': float(now), 'recovered_before_initialization': True,
                           'reason': 'track_history_before_slot_mapping'})
        return events

    def describe(self, track_id):
        record = self.tracks.get(track_id)
        if record is None:
            return None
        state, owner = record['state'], record['owner']
        return dict(byte_track_id=track_id,
                    logical_track_id=record.get('logical_track_id', track_id),
                    reidentified_from_track_id=record.get('reidentified_from_track_id'),
                    reid_gap_seconds=record.get('reid_gap_seconds'),
                    reid_method=record.get('reid_method'),
                    state=state.state, observation=state.observation,
                    last_confirmed_state=state.confirmed, reason=state.reason,
                    confirmation_source=state.confirmation_source,
                    group_id=None if owner is None else owner[1],
                    phase=None if owner is None else owner[0], slot_id=record['slot_id'],
                    formal_slot_id=record['slot_id'], first_seen=record['first_seen'],
                    last_seen=record['last_seen'], center_xy=record.get('center'),
                    visual_event_count=len(record['events']))

    def visible_records(self, objects):
        return [record for track_id in dict.fromkeys(o.get('track_id') for o in objects
                if o['class_name'] in SLOT_STATES) if track_id is not None
                for record in [self.describe(track_id)] if record is not None]
