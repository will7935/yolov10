"""Local state evidence for tracks first reviewed before formal unload mapping."""
import cv2
import numpy as np


STATES = {'wafer_slot_empty': 'empty', 'wafer_slot_filled': 'filled'}


def state_observation(obj):
    review = obj.get('unload_state_review')
    if review is not None:
        return review['observation'], review.get('confidence'), review['reason']
    return STATES[obj['class_name']], obj['confidence'], 'detected'


class PremappingUnloadReview:
    def __init__(self, loader, threshold, *, crop_size=(180, 140), imgsz=320,
                 device='0', iou=.7, conflict_margin=.15, hand_threshold=None):
        self.loader, self.model = loader, None
        self.threshold, self.crop_size = threshold, crop_size
        self.imgsz, self.device, self.iou = imgsz, device, iou
        self.conflict_margin = conflict_margin
        self.hand_threshold = threshold if hand_threshold is None else hand_threshold
        self.enrolled = {}

    def apply(self, frame, objects, sequence, now):
        tracker = sequence.pending_tracker or sequence.tracker
        if tracker.phase != 'unload':
            self.enrolled.clear()
            return objects, []
        store = sequence.track_store
        for tid, entry in list(self.enrolled.items()):
            record = store.tracks.get(tid)
            if (record is not None and (record['owner'] in store.closed
                    or entry['record'] is not None and entry['record'] is not record)
                    or record is None and now-entry['last_seen'] > tracker.lost_seconds
                    or record is not None and record['owner'] is None
                    and now-entry['last_seen'] > tracker.lost_seconds):
                del self.enrolled[tid]
        candidates = tracker._candidates(objects)
        hands = [o for o in objects if o['class_name'] == 'hand']
        reviews, attempts = {}, []
        for obj in candidates:
            tid = obj.get('track_id')
            if tid is None:
                # Existing primary-model missing-slot recovery has no ByteTrack ID.
                # Review its state only when it uniquely fits an enrolled formal slot.
                if obj.get('detection_source') != 'local_crop' or tracker.points is None:
                    continue
                distances = np.linalg.norm(tracker.reference_points-np.asarray(obj['center']), axis=1)
                index = int(np.argmin(distances))
                tid = tracker.byte_track_ids[index]
                if tid not in self.enrolled or distances[index] > min(40., .45*tracker._spacing()):
                    continue
            tid = int(tid)
            record = store.tracks.get(tid)
            owner = None if record is None else record['owner']
            if owner in store.closed or owner is not None and owner != (tracker.phase, tracker.group_id):
                continue
            point = np.asarray(obj['center'])
            # Existing ROI/entry filters remain responsible for admitting geometry.
            if cv2.pointPolygonTest(sequence.contour, tuple(point), False) < 0:
                continue
            if tid not in self.enrolled:
                premapping = owner is None or tracker.points is None
                if not premapping or obj['class_name'] != 'wafer_slot_empty':
                    continue
                self.enrolled[tid] = {'record': record, 'last_seen': now, 'started_at': now}
            entry = self.enrolled[tid]
            if entry['record'] is None:
                entry['record'] = record
            entry['last_seen'] = now
            # A primary full observation keeps the existing confirmation rules.
            if obj['class_name'] != 'wafer_slot_empty':
                continue
            base = {'track_id': tid, 'started_at': entry['started_at'],
                    'primary_class': obj['class_name'], 'primary_confidence': obj['confidence'],
                    'group_id': tracker.group_id, 'observation': 'unknown', 'confidence': None}
            if obj['conflict'] or tracker._occluded(obj, hands):
                review = dict(base, reason='local_review_occluded_or_conflicting')
            else:
                review = self._detect(frame, obj, candidates, base, tracker)
            reviews[(obj.get('track_id'), tuple(obj['xyxy']))] = review
            attempts.append(review)
        return [dict(obj, unload_state_review=reviews[(obj.get('track_id'), tuple(obj['xyxy']))])
                if obj['class_name'] in STATES and (obj.get('track_id'), tuple(obj['xyxy'])) in reviews
                else obj for obj in objects], attempts

    def _detect(self, frame, obj, candidates, base, tracker):
        # Import lazily: the geometry tracker imports state_observation above.
        from industrial_inspection.tools.slot_local_redetection import _crop_bounds
        from industrial_inspection.tools.slot_group_tracker import box_iou
        center = np.asarray(obj['center'])
        peers = [np.asarray(c['center']) for c in candidates
                 if c.get('track_id') != obj.get('track_id')]
        spacing = min((float(np.linalg.norm(p-center)) for p in peers), default=90.)
        gate = min(40., .45*spacing)
        bounds = _crop_bounds(center, self.crop_size, frame.shape[:2])
        x1, y1, x2, y2 = bounds
        if x2-x1 < 32 or y2-y1 < 32:
            return dict(base, reason='local_review_crop_unavailable', crop_xyxy=list(bounds))
        if self.model is None:
            self.model = self.loader()
        result = self.model.predict(frame[y1:y2, x1:x2], imgsz=self.imgsz,
            conf=min(self.threshold, self.hand_threshold), device=self.device, iou=self.iou,
            agnostic_nms=False, verbose=False, save=False)[0]
        boxes = result.boxes.cpu()
        matched, local_hands = [], []
        for cls, score, raw in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
            label = self.model.names[int(cls)]
            box = np.asarray(raw)+[x1, y1, x1, y1]
            if label == 'hand' and score >= self.hand_threshold:
                local_hands.append({'class_name': label, 'xyxy': box.tolist()})
            if label not in STATES or score < self.threshold:
                continue
            p = box.reshape(2, 2).mean(0)
            if np.linalg.norm(p-center) <= gate:
                matched.append({'class_name': label, 'confidence': float(score),
                                'xyxy': box.tolist(), 'center': p.tolist()})
        matched.sort(key=lambda d: d['confidence'], reverse=True)
        details = dict(base, crop_xyxy=list(bounds), candidates=matched)
        if not matched:
            return dict(details, reason='local_review_not_detected')
        best = matched[0]
        if any(c['class_name'] != best['class_name'] and
               best['confidence']-c['confidence'] < self.conflict_margin for c in matched[1:]):
            return dict(details, reason='local_review_class_conflict')
        # Reject duplicate/poorly aligned crops instead of borrowing a neighbour's state.
        if box_iou(obj['xyxy'], best['xyxy']) < .25:
            return dict(details, reason='local_review_position_mismatch')
        if tracker._occluded(best, local_hands):
            return dict(details, reason='local_review_hand_occlusion')
        state = STATES[best['class_name']]
        return dict(details, observation=state, confidence=(min(base['primary_confidence'], best['confidence'])
                    if state == 'empty' else best['confidence']),
                    reason='dual_model_empty' if state == 'empty' else 'local_model_filled')
