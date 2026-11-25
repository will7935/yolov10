"""Preserve physical slot numbers across a calibrated unload-to-load rotation."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


class SlotPerspectiveNumbering:
    def __init__(self, config, image_size):
        self.config = config
        self.image_size = np.asarray(image_size, dtype=float)
        self.calibration_size = np.asarray(config["image_size"], dtype=float)
        self.homography = np.asarray(config["homography"], dtype=float)
        self.rotation_degrees = float(config["unload_to_load_clockwise_degrees"])
        self.angle_tolerance = float(config.get("angle_tolerance_degrees", 25))
        self.maximum_error_ratio = float(config.get("maximum_error_ratio", .45))
        if (self.image_size.shape != (2,) or self.calibration_size.shape != (2,)
                or not np.isfinite([*self.image_size, *self.calibration_size]).all()
                or min(*self.image_size, *self.calibration_size) <= 0
                or self.homography.shape != (3, 3) or not np.isfinite(self.homography).all()
                or abs(np.linalg.det(self.homography)) < 1e-12
                or not np.isfinite([self.rotation_degrees, self.angle_tolerance,
                                    self.maximum_error_ratio]).all()
                or not 0 < self.angle_tolerance < 45 or not 0 < self.maximum_error_ratio < 1):
            raise ValueError("Invalid slot perspective numbering calibration")

    @classmethod
    def for_source(cls, filename, source, image_size):
        """Apply the configured camera-plane calibration to every input video."""
        if not filename or source is None:
            return None
        with Path(filename).open(encoding="utf-8") as stream:
            config = json.load(stream)
        return cls(config, image_size)

    def project(self, points):
        points = np.asarray(points, dtype=float)*self.calibration_size/self.image_size
        homogeneous = np.column_stack((points, np.ones(len(points)))) @ self.homography.T
        if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            raise ValueError("Slot position lies outside the calibrated perspective plane")
        return homogeneous[:, :2]/homogeneous[:, 2:]

    @staticmethod
    def fit_rotation(reference, current):
        x = reference-reference.mean(axis=0)
        y = current-current.mean(axis=0)
        u, singular, vh = np.linalg.svd(x.T @ y)
        signs = np.array([1., np.linalg.det(u @ vh)])
        rotation = (u*signs) @ vh
        scale = float(np.sum(singular*signs)/np.sum(x*x))
        residual = float(np.sqrt(np.mean(np.sum((x @ rotation*scale-y)**2, axis=1))))
        angle = float(np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0])))
        return angle, scale, residual

    def rotate_about_machine(self, reference, machine_center, clockwise_degrees):
        """Carry physical S1-S4 through a pure rotation in the calibrated plane."""
        points = self.project(reference)
        pivot, = self.project([machine_center])
        angle = np.radians(clockwise_degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        predicted = (points-pivot) @ rotation.T+pivot
        homogeneous = np.column_stack((predicted, np.ones(len(points)))) @ np.linalg.inv(self.homography).T
        if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            raise ValueError("Orbit prediction crosses the calibrated plane horizon")
        return homogeneous[:, :2]/homogeneous[:, 2:]*self.image_size/self.calibration_size

    def match(self, reference, current, *, inferred_indices=(), same_load_pose=False):
        expected = 0. if same_load_pose else self.rotation_degrees
        diagnostic = {"status": "waiting_for_reference", "expected_clockwise_degrees": expected,
                      "reference": "g1_load" if same_load_pose else "same_group_unload"}
        if reference is None:
            return None, diagnostic
        if np.shape(reference) != (4, 2) or np.shape(current) != (4, 2):
            return None, dict(diagnostic, status="invalid_geometry")
        try:
            reference, current = self.project(reference), self.project(current)
        except ValueError:
            return None, dict(diagnostic, status="invalid_projection")
        offsets = [points-points.mean(axis=0) for points in (reference, current)]
        rings = [np.argsort(np.arctan2(delta[:, 1], delta[:, 0])) for delta in offsets]
        if any(not cv2.isContourConvex(points[ring].astype(np.float32))
               for points, ring in zip((reference, current), rings)):
            return None, dict(diagnostic, status="invalid_geometry")
        spacing = float(np.min(np.linalg.norm(reference[:, None]-reference[None, :], axis=2)
                               [np.triu_indices(4, 1)]))
        if spacing < 1e-6:
            return None, dict(diagnostic, status="invalid_geometry")
        inferred = set(inferred_indices)
        alternatives = []
        for shift in range(4):
            order = np.empty(4, dtype=int)
            order[rings[0]] = np.roll(rings[1], shift)
            measured = [i for i in range(4) if order[i] not in inferred]
            if len(measured) < 3:
                return None, dict(diagnostic, status="insufficient_measured_support")
            angle, scale, residual = self.fit_rotation(reference[measured], current[order[measured]])
            angle_error = abs((angle-expected+180) % 360-180)
            error_ratio = residual/(spacing*max(scale, 1e-9))
            alternatives.append((angle_error, order, dict(diagnostic,
                clockwise_degrees=angle, angle_error_degrees=angle_error, scale=scale,
                error_ratio=error_ratio, measured_support=len(measured))))
        angle_error, order, diagnostic = min(alternatives, key=lambda item: item[0])
        if angle_error > self.angle_tolerance:
            return None, dict(diagnostic, status="rotation_mismatch")
        if not .5 <= diagnostic["scale"] <= 2 or diagnostic["error_ratio"] > self.maximum_error_ratio:
            return None, dict(diagnostic, status="layout_mismatch")
        return order, dict(diagnostic, status="matched", point_indices_in_slot_order=order.tolist())

    def align_load_layout(self, reference, current, anchor_indices):
        """Express a later measured layout in the original loading pose."""
        anchors = list(anchor_indices)
        if len(anchors) < 2:
            return None, {"status": "insufficient_measured_support"}
        try:
            reference, current = self.project(reference), self.project(current)
        except ValueError:
            return None, {"status": "invalid_projection"}
        spacing = float(np.min(np.linalg.norm(reference[anchors, None]-reference[None, anchors], axis=2)
                               [np.triu_indices(len(anchors), 1)]))
        if spacing < 1e-6:
            return None, {"status": "invalid_geometry"}
        angle, scale, residual = self.fit_rotation(reference[anchors], current[anchors])
        error = residual/(spacing*max(scale, 1e-9))
        info = {"status": "aligned", "clockwise_degrees": angle, "scale": scale,
                "error_ratio": error, "measured_support": len(anchors)}
        if not .5 <= scale <= 2 or error > self.maximum_error_ratio:
            return None, dict(info, status="layout_mismatch")
        radians = np.radians(angle)
        rotation = np.array([[np.cos(radians), np.sin(radians)],
                             [-np.sin(radians), np.cos(radians)]])
        aligned = ((current-current[anchors].mean(0)) @ rotation.T/scale
                   + reference[anchors].mean(0))
        homogeneous = np.column_stack((aligned, np.ones(4))) @ np.linalg.inv(self.homography).T
        if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
            return None, dict(info, status="invalid_projection")
        return homogeneous[:, :2]/homogeneous[:, 2:]*self.image_size/self.calibration_size, info

    def missing_load_point(self, reference, observed, roi, *, same_load_pose=True,
                           preferred_slot_ids=None):
        """Select a corner from measured support and the reference phase's rotation."""
        from industrial_inspection.tools.slot_group_tracker import infer_fourth_parallelogram_candidates

        info = {"status": "waiting_for_reference",
                "reference": "g1_load" if same_load_pose else "same_group_unload",
                "expected_clockwise_degrees": 0. if same_load_pose else self.rotation_degrees,
                "measured_support": 3}
        if reference is None:
            return None, info
        observed = np.asarray(observed, dtype=float)
        accepted, reviewed = [], []
        for point in infer_fourth_parallelogram_candidates(observed):
            if roi is not None and cv2.pointPolygonTest(roi, tuple(point), False) < 0:
                reviewed.append({"center_xy": point.tolist(), "status": "outside_roi"})
                continue
            order, match = self.match(reference, np.vstack((observed, point)),
                                     inferred_indices=(3,), same_load_pose=same_load_pose)
            reviewed.append({**match, "center_xy": point.tolist()})
            if order is not None:
                # Rotation is an admission gate in match(), not a competing score:
                # a smaller angle must not outweigh a better measured triangle.
                score = match["error_ratio"]
                accepted.append((score, point, order, match))
        info["candidates"] = reviewed
        if not accepted:
            return None, dict(info, status="no_compatible_corner")
        accepted.sort(key=lambda item: (item[0], item[3]["angle_error_degrees"]))
        chosen = accepted[0]
        selection = "lowest_shape_error"
        if len(accepted) > 1 and accepted[1][0]-accepted[0][0] < .05:
            compatible = [item for item in accepted if item[0]-accepted[0][0] < .05
                          and preferred_slot_ids is not None
                          and (np.argsort(item[2])+1)[:3].tolist() == list(preferred_slot_ids)]
            if len(compatible) != 1:
                return None, dict(info, status="ambiguous_corner", candidate_count=len(accepted))
            chosen, selection = compatible[0], "previous_verified_roles"
        _, point, order, match = chosen
        roles = np.argsort(order)+1
        return point, dict(info, status="matched", candidate_count=len(accepted),
            slot_ids_in_observation_order=roles[:3].tolist(), missing_slot_id=int(roles[3]),
            center_xy=point.tolist(), clockwise_degrees=match["clockwise_degrees"],
            error_ratio=match["error_ratio"], scale=match["scale"], selection=selection)

    def snapshot(self):
        return {**self.config, "runtime_image_size": self.image_size.tolist()}

    def match_pair(self, reference, layout, direction, roi, *, same_load_pose=False):
        """Fit two measured centers; use the preview only to reject incompatible geometry."""
        expected = 0. if same_load_pose else self.rotation_degrees
        info = {"status": "waiting_for_reference", "measured_support": 2,
                "reference": "g1_load" if same_load_pose else "same_group_unload",
                "expected_clockwise_degrees": expected}
        if reference is None:
            return None, info
        try:
            source, current = self.project(reference), self.project(layout)
        except (ValueError, TypeError):
            return None, dict(info, status="invalid_projection")
        if source.shape != (4, 2) or current.shape != (4, 2):
            return None, dict(info, status="invalid_geometry")
        rings = [np.argsort(np.arctan2((p-p.mean(0))[:, 1], (p-p.mean(0))[:, 0]))
                 for p in (source, current)]
        if any(not cv2.isContourConvex(p[ring].astype(np.float32))
               for p, ring in zip((source, current), rings)):
            return None, dict(info, status="invalid_geometry")
        spacing = float(np.min(np.linalg.norm(source[:, None]-source[None, :], axis=2)
                               [np.triu_indices(4, 1)]))
        if spacing < 1e-6 or np.linalg.norm(current[0]-current[1]) < 1e-6:
            return None, dict(info, status="invalid_geometry")
        edge = np.asarray(layout[1])-layout[0]
        normal = np.array([-edge[1], edge[0]])
        normal *= np.sign(normal @ direction)
        accepted, rejected = [], {"angle": 0, "scale": 0, "side": 0, "roi": 0, "geometry": 0}
        for shift in range(4):
            order = np.empty(4, dtype=int)
            order[rings[0]] = np.roll(rings[1], shift)
            roles = np.argsort(order)
            measured = source[roles[:2]]
            angle, scale, _ = self.fit_rotation(measured, current[:2])
            error = abs((angle-expected+180) % 360-180)
            if error > self.angle_tolerance:
                rejected["angle"] += 1
                continue
            if not .5 <= scale <= 2:
                rejected["scale"] += 1
                continue
            radians = np.radians(angle)
            rotation = np.array([[np.cos(radians), np.sin(radians)],
                                 [-np.sin(radians), np.cos(radians)]])
            projected = (source-measured.mean(0)) @ rotation*scale+current[:2].mean(0)
            homogeneous = np.column_stack((projected, np.ones(4))) @ np.linalg.inv(self.homography).T
            if np.any(np.abs(homogeneous[:, 2]) < 1e-9):
                rejected["geometry"] += 1
                continue
            pixels = homogeneous[:, :2]/homogeneous[:, 2:]*self.image_size/self.calibration_size
            if np.any((pixels[roles[2:]]-layout[0]) @ normal <= 0):
                rejected["side"] += 1
                continue
            if any(cv2.pointPolygonTest(roi, tuple(point), False) < 0 for point in pixels[roles[2:]]):
                rejected["roi"] += 1
                continue
            preview_error = float(np.sqrt(np.mean(np.sum(
                (projected[roles[2:]]-current[2:])**2, axis=1)))/(spacing*scale))
            if preview_error > self.maximum_error_ratio:
                rejected["geometry"] += 1
                continue
            accepted.append((tuple(int(i) for i in roles), dict(info,
                clockwise_degrees=angle, angle_error_degrees=error, scale=scale,
                preview_error_ratio=preview_error)))
        info = dict(info, candidate_count=len(accepted), rejected_candidates=rejected)
        if len(accepted) != 1:
            return None, dict(info, status="ambiguous" if accepted else "no_compatible_pair")
        roles, measured_info = accepted[0]
        return roles, dict(measured_info, status="matched", candidate_count=1,
                           rejected_candidates=rejected, slot_ids_in_preview_order=[i+1 for i in roles])
