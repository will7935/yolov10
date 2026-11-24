"""Fixed-angle diagnostic rotation remains separate from process detection."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from industrial_inspection.tools import debug_slot_perspective as debug


def row(index, phase, points, *, predicted=()):
    return {"source_frame": index, "time_seconds": index/10, "phase": phase,
        "group_id": 1, "group_status": "active", "slot_numbering_status": "final",
        "slots": [{"slot_id": i+1, "byte_track_id": i+10, "center_xy": point,
                   "observation": "filled", "position_is_prediction": i in predicted,
                   "geometry_inferred": i in predicted} for i, point in enumerate(points)]}


class RotationPreviewTests(unittest.TestCase):
    def test_rotation_defaults_to_unload_to_load_counterclockwise(self):
        args, _ = debug.parse_args(['--source', 'source.avi', '--output-dir', 'output',
            '--points', '0', '0', '95', '0', '95', '95', '0', '95', '--size', '96', '96',
            '--rotation-degrees', '50'])
        self.assertEqual(args.rotation_direction, 'counterclockwise')

    def test_clockwise_rotation_and_inverse_projection_preserve_slot_order_and_radius(self):
        h = np.array([[1.1, .2, -10.], [-.1, 1.3, 8.], [.001, .0002, 1.]])
        points = np.array([[20., 20.], [20., 50.], [50., 20.], [50., 50.]])
        original = points.copy()
        top, pivot, rotated, raw = debug.rotate_slot_centers(points, (80., 60.), h, 50.)
        np.testing.assert_allclose(debug.map_plane_points(raw, h), rotated, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(rotated-pivot, axis=1), np.linalg.norm(top-pivot, axis=1))
        _, _, restored, _ = debug.rotate_slot_centers(raw, (80., 60.), h, -50.)
        np.testing.assert_allclose(restored, top, atol=1e-10)
        np.testing.assert_array_equal(points, original)
        _, _, quarter, _ = debug.rotate_slot_centers([[20., 10.]], (10., 10.), np.eye(3), 90.)
        np.testing.assert_allclose(quarter, [[10., 20.]])

    def test_reference_selection_waits_for_four_stable_measured_slots(self):
        points = [[20., 20.], [20., 50.], [50., 20.], [50., 50.]]
        rows = [row(0, 'unload', points, predicted=(3,)),
                *[row(i, 'unload', points) for i in (1, 2, 3)], row(4, 'load', points, predicted=(2, 3))]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'slot_states.jsonl'
            path.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            reference, targets = debug.rotation_test_rows(path)
            self.assertEqual(reference['source_frame'], 2)
            self.assertEqual(targets[0]['source_frame'], 4)
            with self.assertRaisesRegex(ValueError, 'four measured'):
                debug.rotation_test_rows(path, 0)

    def test_explicit_log_from_different_video_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/'slot_states.jsonl'
            path.write_text('', encoding='utf-8')
            (root/'summary.json').write_text(json.dumps({'source': str(root/'other.avi')}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'different source'):
                debug.find_rotation_slot_states(root/'input.avi', path)

    def test_rotation_export_leaves_video_and_log_unchanged_and_supports_original_preview_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10., (96, 96))
            self.assertTrue(writer.isOpened())
            for _ in range(5):
                writer.write(np.zeros((96, 96, 3), np.uint8))
            writer.release()
            states = root/'slot_states.jsonl'
            points = [[20., 20.], [20., 50.], [50., 20.], [50., 50.]]
            states.write_text('\n'.join(json.dumps(row(i, 'unload' if i < 4 else 'load', points,
                predicted=(2, 3) if i == 4 else ())) for i in range(5)), encoding='utf-8')
            video_bytes, log_bytes = source.read_bytes(), states.read_bytes()
            args = argparse.Namespace(source=source, frames=[0], calibration=None,
                points=[0., 0., 95., 0., 95., 95., 0., 95.], size=[96, 96],
                output_dir=root/'rotation', slot_states=states, rotation_degrees=50.,
                rotation_direction='both', machine_center=(80., 50.),
                rotation_reference_frame=None, rotation_target_frames=None)
            debug.export_rotation_test(args)
            report = json.loads((args.output_dir/'rotation_test.json').read_text(encoding='utf-8'))
            self.assertEqual([p['clockwise_degrees'] for p in report['predictions']], [50., -50.])
            self.assertFalse(report['translation_correction'])
            self.assertEqual(len(list(args.output_dir.glob('*.png'))), 3)
            for prediction in report['predictions']:
                image = cv2.imdecode(np.fromfile(args.output_dir/prediction['comparisons'][0]['image'], np.uint8), 1)
                self.assertIsNotNone(image)
            self.assertEqual(source.read_bytes(), video_bytes)
            self.assertEqual(states.read_bytes(), log_bytes)
            args.output_dir = root/'ordinary'
            debug.export_previews(args)
            self.assertTrue((args.output_dir/'top_down_frame_000000.png').is_file())


if __name__ == '__main__':
    unittest.main()
