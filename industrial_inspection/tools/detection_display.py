"""Draw detection center points and their color legend."""
from __future__ import annotations

import cv2


def draw_detection_points(frame, objects, *, radius=4):
    """Draw only colored points at box centers, leaving inputs unchanged."""
    from ultralytics.utils.plotting import colors

    plotted = frame.copy()
    height, width = plotted.shape[:2]
    for obj in objects:
        x1, y1, x2, y2 = obj["xyxy"]
        center = (min(width-1, max(0, round((x1+x2)/2))),
                  min(height-1, max(0, round((y1+y2)/2))))
        color = colors(obj["class_id"], True)
        cv2.circle(plotted, center, radius+1, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(plotted, center, radius, color, -1, cv2.LINE_AA)
    return plotted


def draw_detection_legend(frame, names, *, radius=4, font_scale=.4, label_names=None):
    """Overlay all model classes at the top left using the marker palette.

    Include classes even when absent in the current frame so the legend stays
    consistent throughout the video. Modify the annotated frame in place.
    """
    from ultralytics.utils.plotting import colors

    if not names:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    entries = [(int(class_id), (label_names or {}).get(name, name))
               for class_id, name in sorted(names.items(), key=lambda item: int(item[0]))]
    sizes = [cv2.getTextSize(label, font, font_scale, 1) for _, label in entries]
    text_height = max(size[0][1] for size in sizes)
    baseline = max(size[1] for size in sizes)
    row_height = max(text_height + baseline, 2 * radius + 2) + 8
    margin, padding = 8, 8
    text_x = margin + padding + 2 * radius + 10
    right = min(frame.shape[1], text_x + max(size[0][0] for size in sizes) + padding)
    bottom = min(frame.shape[0], margin + 2 * padding + row_height * len(entries))
    panel = frame[margin:bottom, margin:right]
    if panel.size == 0:
        return
    panel[:] = cv2.addWeighted(panel, .25, panel * 0 + 25, .75, 0)
    for index, (class_id, label) in enumerate(entries):
        top = margin + padding + index * row_height
        center = (margin + padding + radius + 1, top + (text_height + baseline) // 2)
        color = colors(class_id, True)
        cv2.circle(frame, center, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (text_x, top + text_height), font, font_scale,
                    (240, 240, 240), 1, cv2.LINE_AA)
