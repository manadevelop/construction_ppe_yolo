from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def xyxy_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_inside(box, region) -> bool:
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = region
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def head_region(person_box):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1, x2, y1 + 0.42 * h]


def torso_region(person_box):
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    return [x1, y1 + 0.25 * h, x2, y1 + 0.82 * h]
