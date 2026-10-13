"""Draw compact center markers without changing detection geometry."""
from __future__ import annotations

import cv2


def _overlap_area(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))


def draw_detection_points(frame, objects, *, show_labels=True, show_confidence=True,
                          radius=4, font_scale=.4, label_names=None):
    """Draw at box centers; place short labels nearby with bounded coordinates.

    Objects and the original frame are left unchanged. Nearby labels are placed
    greedily to reduce overlap; crowded scenes may still have overlapping text.
    """
    from ultralytics.utils.plotting import colors

    plotted = frame.copy()
    height, width = plotted.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    placed = []
    markers = []
    for obj in sorted(objects, key=lambda o: o["confidence"], reverse=True):
        x1, y1, x2, y2 = obj["xyxy"]
        center = (min(width-1, max(0, round((x1+x2)/2))),
                  min(height-1, max(0, round((y1+y2)/2))))
        color = colors(obj["class_id"], True)
        markers.append((center, color))
        if not show_labels:
            continue
        name = (label_names or {}).get(obj["class_name"], obj["class_name"])
        text = f"{name} {obj['confidence']:.2f}" if show_confidence else name
        scale = font_scale
        (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
        # Fit unusually long labels or small frames without drawing beyond edges.
        for _ in range(5):
            if tw+8 <= width and th+baseline+6 <= height:
                break
            scale *= min(width/max(tw+8, 1), height/max(th+baseline+6, 1)) * .9
            (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
        bw, bh = min(width, tw+8), min(height, th+baseline+6)
        cx, cy = center
        candidates = []
        for offset in (0, -bh, bh, -2*bh, 2*bh):
            for left in (cx+radius+5, cx-radius-5-bw):
                top = min(height-bh, max(0, cy-bh+offset))
                left = min(width-bw, max(0, left))
                rect = (left, top, left+bw, top+bh)
                penalty = sum(_overlap_area(rect, other) for other in placed)
                candidates.append((penalty, abs(offset), rect))
        _, _, rect = min(candidates, key=lambda item: (item[0], item[1]))
        placed.append(rect)
        left, top, right, bottom = rect
        endpoint = (min(right-1, max(left, cx)), min(bottom-1, max(top, cy)))
        cv2.line(plotted, center, endpoint, color, 1, cv2.LINE_AA)
        cv2.rectangle(plotted, (left, top), (right-1, bottom-1), (25,25,25), -1)
        cv2.putText(plotted, text, (left+4, top+3+th), font, scale, color, 1, cv2.LINE_AA)
    # Draw markers last so label backgrounds do not cover centers.
    for center, color in markers:
        cv2.circle(plotted, center, radius+1, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(plotted, center, radius, color, -1, cv2.LINE_AA)
    return plotted
