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


class S4LoadLookback:
    def __init__(self, contour, gate):
        self.contour = contour
        self.gate = gate
        self.clear()

    def clear(self):
        self.anchors = {}
        self.frames = deque()
        self.hint_frames = 0
        self.hint_since = self.hint_at = None

    def _inside(self, point):
        return cv2.pointPolygonTest(self.contour, tuple(map(float, point)), False) >= 0

    def record(self, tracker, now, hands, tracks, live_reference, quarantined_ids):
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
            self.frames.popleft()
        self.frames.append((now, self.anchors.copy(), hands))

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

    def resolve(self, tracker, now, empty_at):
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
                contact = S4LoadContact(first, last, frames, now+tracker.initialization_history_seconds)
            first = last = None
            frames = 0
        return contact
