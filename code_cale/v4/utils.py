"""Utility helpers for geometric calculations and orientation handling."""
from __future__ import annotations
from math import atan2, pi, cos, sin, radians
import numpy as np


def normalize_angle(a: float) -> float:
    return (a + pi) % (2 * pi) - pi


def offset_point_from_tangent(base: np.ndarray, tangent_angle: float, offset_dist: float, offset_angle_rel: float) -> np.ndarray:
    """Return an offset point located at offset_dist rotated by offset_angle_rel from tangent direction."""
    total_angle = tangent_angle + offset_angle_rel
    return base + offset_dist * np.array([cos(total_angle), sin(total_angle)])


def desired_facing_from_target(target_heading: float, relative_facing_angle: float) -> float:
    return normalize_angle(target_heading + relative_facing_angle)


def offset_point_local_frame(base: np.ndarray, target_heading: float, dx_fwd: float, dy_left: float, forward_sign: float = 1.0) -> np.ndarray:
    """Offset using target local frame where +X=fwd, +Y=left.

    forward_sign can be set to -1.0 for diagnostics if path direction assumptions invert.
    """
    fx = cos(target_heading)
    fy = sin(target_heading)
    lx = -sin(target_heading)  # left vector
    ly = cos(target_heading)
    return base + np.array([(forward_sign * dx_fwd) * fx + dy_left * lx,
                            (forward_sign * dx_fwd) * fy + dy_left * ly])
