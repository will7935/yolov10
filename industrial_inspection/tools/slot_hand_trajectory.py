"""Review a hand passing a predicted slot before visually confirming its target."""
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class _HandPath:
    samples: deque = field(default_factory=lambda: deque(maxlen=16))


class LoadHandTrajectory:
    """Trajectory supplies an estimated action time, never a filled state."""
    def __init__(self, history_seconds=.5, review_seconds=12.):
        self.history_seconds = history_seconds
        self.review_seconds = review_seconds
        self.paths = []
        self.pending = {}
        self.passed = {}
        self.reviews = deque(maxlen=16)

    @staticmethod
    def _overlap(box, hand):
        low, high = np.maximum(box[:2], hand[:2]), np.minimum(box[2:], hand[2:])
        area = max(1., (box[2]-box[0])*(box[3]-box[1]))
        return float(np.prod(np.maximum(0., high-low))/area)

    def _cancel(self, slot_id, now, reason):
        item = self.pending.pop(slot_id)
        if not any(p['passed_slot_id'] == item['passed_slot_id'] for p in self.pending.values()):
            self.passed.pop(item['passed_slot_id'], None)
        self.reviews.append(dict(slot_id=slot_id, status=reason, at=float(now),
                                 action_time_upper=item['time_upper']))

    def observe(self, hands, targets, inferred_slots, now, enabled=True):
        targets = {int(sid): target for sid, target in targets.items()}
        for slot_id, item in list(self.pending.items()):
            if now-item['time_upper'] > self.review_seconds:
                self._cancel(slot_id, now, 'expired')
        self.passed = {slot: end for slot, end in self.passed.items() if end >= now}
        if not enabled:
            self.paths.clear()
            return
        if len(targets) < 2:
            self.paths.clear()
            return
        centers = np.array([t['center'] for t in targets.values()])
        spacing = float(np.min(np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
                               [np.triu_indices(len(centers), 1)]))
        if spacing < 2:
            return
        self.paths = [p for p in self.paths if now-p.samples[-1]['at'] <= .25]
        used = set()
        for hand in sorted(hands, key=lambda h: h['confidence'], reverse=True):
            box = np.asarray(hand['xyxy'], dtype=float)
            center = box.reshape(2, 2).mean(0)
            options = [(float(np.linalg.norm(center-p.samples[-1]['center'])), i, p)
                       for i, p in enumerate(self.paths) if i not in used]
            nearest = min(options, key=lambda x: x[0]) if options else None
            if nearest is None or nearest[0] > max(40., spacing*1.5):
                path = _HandPath()
                self.paths.append(path)
                index = len(self.paths)-1
            else:
                _, index, path = nearest
            used.add(index)
            path.samples.append({'at': float(now), 'center': center, 'box': box,
                'distances': {sid: float(np.linalg.norm(center-t['center'])) for sid, t in targets.items()},
                'overlaps': {sid: self._overlap(np.asarray(t['box']), box) for sid, t in targets.items()}})
            while path.samples and now-path.samples[0]['at'] > self.history_seconds:
                path.samples.popleft()
            if len(path.samples) < 4:
                continue
            samples = list(path.samples)
            for origin in map(int, inferred_slots):
                if origin not in targets or any(origin not in p['distances'] for p in samples):
                    continue
                distances = np.array([p['distances'][origin] for p in samples])
                closest = int(np.argmin(distances))
                # A visible dwell around the first slot is evidence of its own operation.
                near = [p for p in samples if p['distances'][origin] <= .55*spacing]
                dwelling = any(b['at']-a['at'] >= .1 and all(
                    np.linalg.norm(p['center']-a['center']) <= .12*spacing
                    for p in near if a['at'] <= p['at'] <= b['at'])
                    for i, a in enumerate(near) for b in near[i+1:])
                if dwelling:
                    self.passed.pop(origin, None)
                    continue
                if (closest == 0 or closest == len(samples)-1 or distances[closest] > .55*spacing
                        or distances[-1]-distances[closest] < .12*spacing):
                    continue
                velocity = samples[-1]['center']-samples[-2]['center']
                if np.linalg.norm(velocity) < .12*spacing:
                    continue
                alternatives = []
                for sid, target in targets.items():
                    if sid == origin or not target['bound'] or any(sid not in p['distances'] for p in samples):
                        continue
                    ds = np.array([p['distances'][sid] for p in samples])
                    direction = np.asarray(target['center'])-center
                    alignment = float(velocity @ direction/max(1e-9, np.linalg.norm(velocity)*np.linalg.norm(direction)))
                    if (ds[0]-ds[-1] < .5*spacing or ds[-2]-ds[-1] < .1*spacing
                            or ds[-1] > .65*spacing or alignment < .65
                            or samples[-1]['overlaps'][sid] < .03
                            or np.count_nonzero(np.diff(ds) < 0) < len(ds)-2):
                        continue
                    alternatives.append(sid)
                if len(alternatives) != 1:
                    continue
                target_id = alternatives[0]
                self.passed[origin] = now+self.review_seconds
                if origin in self.pending:
                    self._cancel(origin, now, 'passed_toward_another_slot')
                if target_id not in self.pending:
                    self.pending[target_id] = {
                        'slot_id': target_id, 'passed_slot_id': origin,
                        'time_lower': samples[0]['at'], 'time_upper': float(now),
                        'baseline_at': float(targets[target_id]['baseline_at']),
                        'observed_contact_frames': sum(p['overlaps'][target_id] >= .03 for p in samples),
                        'trajectory_sample_count': len(samples), 'empty_frames': 0,
                        'status': 'waiting_for_visual',
                        'hand_centers': [p['center'].round(2).tolist() for p in samples]}

    def blocks_inference(self, slot_id):
        return slot_id in self.passed

    def observe_empty(self, slot_id, now, required_frames):
        item = self.pending.get(slot_id)
        if item is not None and now-item['time_upper'] > .2:
            item['empty_frames'] += 1
            if item['empty_frames'] >= max(2, required_frames):
                self._cancel(slot_id, now, 'cancelled_by_visual_empty')

    def break_empty_run(self, slot_id):
        if slot_id in self.pending:
            self.pending[slot_id]['empty_frames'] = 0

    def confirm(self, event, now):
        if (event.get('action') != 'load' or event.get('from') != 'empty'
                or event.get('to') != 'filled' or event.get('inferred')):
            return event
        self.passed.pop(event['slot_id'], None)
        # A real operation at the passed slot must never be deleted or renamed.
        for sid, item in list(self.pending.items()):
            if item['passed_slot_id'] == event['slot_id'] and event['time_upper'] <= item['time_upper']:
                self._cancel(sid, now, 'passed_slot_visually_loaded_first')
        item = self.pending.get(event['slot_id'])
        if (item is None or event.get('byte_track_id') is None
                or now-item['time_upper'] > self.review_seconds
                or event['time_upper'] < item['time_upper']
                or (event.get('time_lower') is not None and event['time_lower'] > item['time_upper'])):
            return event
        self.pending.pop(event['slot_id'])
        self.reviews.append(dict(slot_id=event['slot_id'], status='visually_confirmed', at=float(now),
                                 action_time_upper=item['time_upper']))
        return dict(event, reason='visual_after_hand_trajectory', action_time_inferred=True,
                    visual_time_lower=event.get('visual_time_lower', event.get('time_lower')),
                    visual_time_upper=event.get('visual_time_upper', event['time_upper']),
                    time_lower=max(item['baseline_at'], item['time_lower']), time_upper=item['time_upper'],
                    hand_trajectory=dict(item))

    def snapshot(self):
        return {'pending': [dict(item) for item in self.pending.values()],
                'passed_slot_ids': sorted(self.passed), 'reviews': list(self.reviews)}
