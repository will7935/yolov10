import unittest
from types import SimpleNamespace
import numpy as np

from industrial_inspection.tools.slot_group_tracker import SlotGroupTracker
from industrial_inspection.tools.slot_track_state import SlotTrackStateStore
from industrial_inspection.tools.slot_unload_state_review import PremappingUnloadReview, state_observation
from industrial_inspection.tools.test_group_prediction_pipeline import SyntheticModel


def detection(state='empty', track_id=44):
    return dict(class_name='wafer_slot_'+state, confidence=.9, track_id=track_id,
                xyxy=[80., 80., 120., 120.])


def local(state='filled', confidence=.8, box=None):
    return dict(class_name='wafer_slot_'+state, confidence=confidence,
                xyxy=box or [70., 50., 110., 90.])


def environment():
    tracker = SlotGroupTracker(phase='unload', minimum_frames=4, confirm_seconds=0.)
    contour = np.array([[0, 0], [640, 0], [640, 480], [0, 480]], dtype=np.float32)
    store = SlotTrackStateStore(tracker, contour)
    return SimpleNamespace(tracker=tracker, pending_tracker=None, track_store=store, contour=contour)


class UnloadReviewTests(unittest.TestCase):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def apply(self, predictions, primary='empty'):
        model = SyntheticModel([predictions])
        review = PremappingUnloadReview(lambda: model, .1)
        seq = environment()
        objects, attempts = review.apply(self.frame, [detection(primary)], seq, 0.)
        return objects[0], attempts

    def test_both_empty_provides_empty_evidence(self):
        obj, _ = self.apply([local('empty')])
        self.assertEqual(state_observation(obj), ('empty', .8, 'dual_model_empty'))

    def test_positive_local_full_changes_state_evidence_without_changing_geometry(self):
        obj, _ = self.apply([local()])
        self.assertEqual(state_observation(obj), ('filled', .8, 'local_model_filled'))
        for key in ('xyxy', 'class_name', 'confidence', 'track_id'):
            self.assertEqual(obj[key], detection()[key])

    def test_missing_and_conflicting_local_predictions_are_unknown(self):
        for predictions in ([], [local('empty', .8), local('filled', .75)]):
            obj, _ = self.apply(predictions)
            self.assertEqual(state_observation(obj)[0], 'unknown')

    def test_neighbour_cannot_supply_full_evidence(self):
        obj, _ = self.apply([local(box=[130., 50., 170., 90.])])
        self.assertEqual(state_observation(obj)[0], 'unknown')

    def test_local_hand_occlusion_is_not_state_evidence(self):
        obj, _ = self.apply([local(), dict(class_name='hand', confidence=.9,
                                         xyxy=[70., 50., 110., 90.])])
        self.assertEqual(state_observation(obj)[0], 'unknown')

    def test_primary_full_does_not_load_auxiliary_model(self):
        review = PremappingUnloadReview(lambda: self.fail('unexpected model load'), .1)
        objects, attempts = review.apply(self.frame, [detection('filled')], environment(), 0.)
        self.assertEqual(objects, [detection('filled')])
        self.assertFalse(attempts)

    def test_review_continues_after_mapping_but_does_not_enroll_new_formal_tracks(self):
        seq = environment()
        model = SyntheticModel([[local()], [local('empty')]])
        review = PremappingUnloadReview(lambda: model, .1)
        objects, _ = review.apply(self.frame, [detection()], seq, 0.)
        seq.track_store.observe(objects, 0.)
        seq.tracker.points = np.array([[100.,100.],[200.,100.],[200.,200.],[100.,200.]])
        seq.track_store.tracks[44]['owner'] = ('unload', 1)
        objects, attempts = review.apply(self.frame, [detection()], seq, .1)
        self.assertEqual(state_observation(objects[0])[0], 'empty')
        self.assertEqual(len(attempts), 1)
        seq.track_store.tracks[45] = dict(owner=('unload', 1))
        _, attempts = review.apply(self.frame, [detection(track_id=45)], seq, .2)
        self.assertFalse(attempts)

    def test_closed_group_and_loading_do_not_inherit_review(self):
        seq = environment()
        review = PremappingUnloadReview(lambda: SyntheticModel([[local()]]), .1)
        objects, _ = review.apply(self.frame, [detection()], seq, 0.)
        seq.track_store.observe(objects, 0.)
        seq.track_store.tracks[44]['owner'] = ('unload', 1)
        seq.track_store.closed.add(('unload', 1))
        _, attempts = review.apply(self.frame, [detection()], seq, .1)
        self.assertFalse(attempts)
        self.assertFalse(review.enrolled)
        seq.tracker.phase = 'load'
        _, attempts = review.apply(self.frame, [detection()], seq, .2)
        self.assertFalse(attempts)

    def test_primary_local_recovery_of_enrolled_formal_slot_still_requires_review(self):
        seq = environment()
        model = SyntheticModel([[local()], []])
        review = PremappingUnloadReview(lambda: model, .1)
        objects, _ = review.apply(self.frame, [detection()], seq, 0.)
        seq.track_store.observe(objects, 0.)
        seq.tracker.points = seq.tracker.reference_points = np.array(
            [[100.,100.],[200.,100.],[200.,200.],[100.,200.]])
        seq.tracker.byte_track_ids = [44, 45, 46, 47]
        seq.track_store.tracks[44]['owner'] = ('unload', 1)
        recovered = dict(detection(track_id=None), detection_source='local_crop')
        del recovered['track_id']
        objects, attempts = review.apply(self.frame, [recovered], seq, .1)
        self.assertEqual(state_observation(objects[0])[0], 'unknown')
        self.assertNotIn('track_id', objects[0])
        self.assertEqual(attempts[0]['track_id'], 44)

    def test_track_history_preserves_baseline_through_unknown_then_records_one_unload(self):
        seq = environment()
        model = SyntheticModel([[local()]]*4+[[]]*4+[[local('empty')]]*4)
        review = PremappingUnloadReview(lambda: model, .1)
        for i in range(12):
            objects, _ = review.apply(self.frame, [detection()], seq, i*.1)
            seq.track_store.observe(objects, i*.1)
            if 3 <= i <= 7:
                self.assertEqual(seq.track_store.tracks[44]['state'].confirmed, 'filled')
                self.assertEqual(seq.track_store.tracks[44]['events'], [])
        events = seq.track_store.tracks[44]['events']
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['action'], 'unload')
        self.assertEqual(events[0]['unload_state_review']['reason'], 'dual_model_empty')
        self.assertAlmostEqual(events[0]['time_upper'], .8)

    def test_initial_verified_empty_is_baseline_and_never_fabricates_an_unload(self):
        seq = environment()
        model = SyntheticModel([[local('empty')]]*4)
        review = PremappingUnloadReview(lambda: model, .1)
        for i in range(4):
            objects, _ = review.apply(self.frame, [detection()], seq, i*.1)
            seq.track_store.observe(objects, i*.1)
        self.assertEqual(seq.track_store.tracks[44]['state'].confirmed, 'empty')
        self.assertFalse(seq.track_store.tracks[44]['events'])


if __name__ == '__main__':
    unittest.main()
