"""Target path generation utilities.
We reuse the composite path idea: arc + line + arc + line.
The target moves one step per outer iteration; we treat step index as progress.
"""
from __future__ import annotations
from math import pi, cos, sin
from typing import Tuple
import numpy as np

from .config import TOTAL_STEPS, SEGMENT_RATIOS, ARC_RADIUS, REF_X0, REF_Y0

# Pre-compute segment lengths (integers) from ratios
r1, r2, r3, r4 = SEGMENT_RATIOS
s1 = int(TOTAL_STEPS * r1)
s2 = int(TOTAL_STEPS * r2)
s3 = int(TOTAL_STEPS * r3)
used = s1 + s2 + s3
s4 = TOTAL_STEPS - used
SEGMENTS = (s1, s2, s3, s4)
BOUNDS = (s1, s1 + s2, s1 + s2 + s3)  # boundaries between segment types


def path_point(step: int) -> np.ndarray:
    step = min(max(step, 0), TOTAL_STEPS - 1)
    s1, s2, s3, s4 = SEGMENTS
    b1, b2, b3 = BOUNDS

    if step < b1:  # arc 1 (semi circle)
        t = step / s1
        angle = pi + pi * t
        x = REF_X0 - ARC_RADIUS * cos(angle)
        y = REF_Y0 - ARC_RADIUS * sin(angle)
    elif step < b2:  # line down
        t = (step - b1) / s2
        x = REF_X0 - ARC_RADIUS + 1.0  # vertical line x constant
        y = REF_Y0 - (ARC_RADIUS + 1.0) * t
    elif step < b3:  # arc 2
        t = (step - b2) / s3
        angle = pi + pi * t
        x = REF_X0 + ARC_RADIUS * cos(angle)
        y = (REF_Y0 - (ARC_RADIUS + 1.0)) + ARC_RADIUS * sin(angle)
    else:  # final line up
        t = (step - b3) / s4
        x = REF_X0 + ARC_RADIUS + 0.0
        y = (REF_Y0 - (ARC_RADIUS + 1.0)) + (ARC_RADIUS + 1.0) * t

    return np.array([x, y])


def path_tangent(step: int) -> float:
    p = path_point(step)
    p_next = path_point(min(step + 1, TOTAL_STEPS - 1))
    v = p_next - p
    return np.arctan2(v[1], v[0])
