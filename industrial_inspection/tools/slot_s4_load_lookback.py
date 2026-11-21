"""Cache hand observations before S4 can be numbered, and review them against the measured pose."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from industrial_inspection.tools.s4_hand_contact import HandContactPolicy, evaluate_s4_contact


@dataclass(frozen=True)
class Anchor:
    at: float
    center: tuple
    size: tuple


@dataclass(frozen=True)
class S4LoadContact:
    first_at: float
    last_at: float
    frames: int
    expires_at: float
    require_bound_visual: bool = False
    from_reach_crop: bool = False


class S4LoadLookback:
    def __init__(self, contour, gate):
        self.contour = contour
        self.gate = gate
        self.clear()

    def clear(self):
        self.anchors = {}
        self.frames = deque()
        self.reach_hands = {}
        self.hint_frames = 0
        self.hint_since = self.hint_at = None

    def _inside(self, point):
        return cv2.pointPolygonTest(self.contour, tuple(map(float, point)), False) >= 0

    def record(self, tracker, now, hands, tracks, live_reference, quarantined_ids, *, reach_hand=None):
        if self.frames and now-self.frames[-1][0] > .25:
            self.clear()
        hold = min(tracker.lost_seconds, tracker.s4_contact_prediction_seconds)
        self.anchors = {track: anchor for track, anchor in self.anchors.items()
                        if track in tracks and now-anchor.at <= hold}
        for track_id, record in tracks.items():
            if record["last_seen"] != now or record["reliable_since"] is None:
                continue
            x1, y1, x2, y2 = record["xyxy"]
            self.anchors[track_id] = Anchor(now, tuple(record["center"]), (x2-x1, y2-y1))
        hands = [hand for hand in hands if self._inside(np.asarray(hand["xyxy"]).reshape(2, 2).mean(axis=0))]
        while self.frames and now-self.frames[0][0] > tracker.initialization_history_seconds:
            expired_at, _, _ = self.frames.popleft()
            self.reach_hands.pop(expired_at, None)
        self.frames.append((now, self.anchors.copy(), hands))
        if reach_hand is not None:
            self.reach_hands[now] = reach_hand

        pairs = {}
        if live_reference is not None:
            for track_id, anchor in self.anchors.items():
                distances = np.linalg.norm(live_reference-np.asarray(anchor.center), axis=1)
                nearest = np.argsort(distances)
                slot = int(nearest[0])
                if (slot == 3 or distances[slot] > self.gate*.5
                        or distances[nearest[1]]-distances[slot] <= self.gate*.25 or slot in pairs):
                    pairs.clear()
                    break
                pairs[slot] = (track_id, anchor)
        if (any(track in quarantined_ids for track, _ in pairs.values())
                and self._contact(tracker, now, live_reference,
                    pairs={i: anchor for i, (_, anchor) in pairs.items()}, hands=hands)):
            if not self.hint_frames:
                self.hint_since = now
            self.hint_frames += 1
            if self._meets_contact_rule(tracker, self.hint_since, now, self.hint_frames):
                self.hint_at = now
        else:
            self.hint_frames = 0
        return self.hint_at is not None and now-self.hint_at <= tracker.initialization_history_seconds

    def _contact(self, tracker, now, reference, pairs, hands):
        if not hands or len(pairs) < 2 or not any(anchor.at == now for anchor in pairs.values()):
            return False
        offsets = np.asarray([np.asarray(anchor.center)-reference[i] for i, anchor in pairs.items()])
        shift = offsets.mean(axis=0)
        # The few early observations can confirm a common shift, not a rotation.
        if np.max(np.linalg.norm(offsets-shift, axis=1)) > self.gate*.175:
            return False
        layout = reference+shift
        if not self._inside(layout[3]):
            return False
        sizes = [None]*4
        for i, anchor in pairs.items():
            sizes[i] = anchor.size
        sizes[3] = np.median([anchor.size for anchor in pairs.values()], axis=0)
        policy = HandContactPolicy(tracker.s4_prediction_contact_overlap_ratio,
            tracker.load_hand_contact_competing_slot_enabled, tracker.load_hand_contact_competing_slot_id-1,
            tracker.load_hand_contact_competing_margin, tracker.load_hand_contact_competing_ratio)
        evidence = evaluate_s4_contact(layout, sizes, 3, hands, policy)
        return evidence.accepted and evidence.overlap_ratio >= policy.overlap_ratio

    @staticmethod
    def _meets_contact_rule(tracker, first, last, frames):
        return (frames >= tracker.load_hand_contact_min_frames
                and last-first+1e-9 >= tracker.load_hand_contact_seconds)

    def resolve_reach(self, tracker, now, empty_at):
        """Match an unnumbered far-side reach to a missing S4 using the final S1/S3 IDs."""
        if not self.reach_hands:
            return None
        points = tracker.reference_points
        target = points[3]
        spacing = np.min(np.linalg.norm(points[:3]-target, axis=1))
        if not self._inside(target) or target[0]-points[:3, 0].max() < .6*spacing:
            return None
        first_id, third_id = tracker.byte_track_ids[0], tracker.byte_track_ids[2]
        target_id = tracker.byte_track_ids[3]
        contact = None
        first = last = None
        frames = 0
        for at, anchors, _ in self.frames:
            hand = self.reach_hands.get(at)
            visible = target_id in anchors and anchors[target_id].at == at
            if (at >= empty_at and not visible and hand is not None
                    and first_id in anchors and third_id in anchors
                    and self._reaches_s4(at, (anchors[first_id], anchors[third_id]), points, spacing, hand)):
                first = at if first is None else first
                last, frames = at, frames+1
                continue
            if first is not None and self._meets_contact_rule(tracker, first, last, frames):
                if contact is not None:
                    return None
                contact = S4LoadContact(first, last, frames, now+tracker.initialization_history_seconds,
                                        require_bound_visual=True, from_reach_crop=True)
            first = last = None
            frames = 0
        return contact

    def _reaches_s4(self, at, anchors, points, spacing, hand):
        if max(anchor.at for anchor in anchors) != at:
            return False
        offsets = np.array([anchor.center for anchor in anchors])-points[[0, 2]]
        # Both tracks must support the formal pose, with at least one fresh center.
        tolerance = .6*self.gate
        if (np.max(np.linalg.norm(offsets, axis=1)) > tolerance
                or np.linalg.norm(offsets[0]-offsets[1]) > tolerance):
            return False
        x1, y1, x2, y2 = hand["xyxy"]
        # The right side of the detected glove approximates the reaching end.
        reach = np.array([x2-.05*(x2-x1), (y1+y2)/2])
        distances = np.linalg.norm(points-reach, axis=1)
        return (self._inside(reach) and distances[3] <= .45*spacing
                and distances[:3].min()-distances[3] >= .15*spacing)

    def resolve(self, tracker, now, empty_at, *, require_bound_visual=False):
        """Review only contacts that ended before formal mapping and fit its track IDs."""
        contact = None
        first = last = None
        frames = 0
        reference = tracker.reference_points
        for at, anchors, hands in self.frames:
            pairs = {i: anchors[track] for i, track in enumerate(tracker.byte_track_ids[:3])
                     if track in anchors}
            if at >= empty_at and self._contact(tracker, at, reference, pairs, hands):
                first = at if first is None else first
                last, frames = at, frames+1
                continue
            if first is not None and self._meets_contact_rule(tracker, first, last, frames):
                if contact is not None:
                    return None
                contact = S4LoadContact(first, last, frames, now+tracker.initialization_history_seconds,
                                        require_bound_visual)
            first = last = None
            frames = 0
        return contact
