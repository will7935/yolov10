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


def draw_detection_legend(frame, names, *, radius=4, font_scale=.4, label_names=None,
                          position="top_right"):
    """Overlay all model classes at the top of the frame using the marker palette.

    Include classes even when absent in the current frame so the legend stays
    consistent throughout the video. Modify the annotated frame in place.

    ``position`` can be ``top_left`` or ``top_right``. The default keeps the
    legend away from the main inspection area in the generated video.
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
    max_text_width = max(size[0][0] for size in sizes)
    panel_width = 2 * padding + 2 * radius + 10 + max_text_width
    panel_height = 2 * padding + row_height * len(entries)
    if position == "top_left":
        left = margin
    elif position == "top_right":
        left = max(0, frame.shape[1] - margin - panel_width)
    else:
        raise ValueError("position must be 'top_left' or 'top_right'")
    top = margin
    right = min(frame.shape[1], left + panel_width)
    bottom = min(frame.shape[0], top + panel_height)
    panel = frame[top:bottom, left:right]
    if panel.size == 0:
        return
    panel[:] = cv2.addWeighted(panel, .25, panel * 0 + 25, .75, 0)
    text_x = left + padding + 2 * radius + 10
    for index, (class_id, label) in enumerate(entries):
        row_top = top + padding + index * row_height
        center = (left + padding + radius + 1,
                  row_top + (text_height + baseline) // 2)
        color = colors(class_id, True)
        cv2.circle(frame, center, radius + 1, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)
        cv2.putText(frame, label, (text_x, row_top + text_height), font, font_scale,
                    (240, 240, 240), 1, cv2.LINE_AA)
