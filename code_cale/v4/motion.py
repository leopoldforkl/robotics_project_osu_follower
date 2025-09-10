"""Simple holonomic (mecanum) robot kinematics.
State vector: [x, y, theta]
Control: planar velocity (vx, vy). Heading (theta) is separately controlled toward a desired orientation.
"""
from __future__ import annotations
from typing import List
from math import atan2

from .config import DT

def integrate_position(state: List[float], vx: float, vy: float) -> List[float]:
    x, y, theta = state
    x += vx * DT
    y += vy * DT
    return [x, y, theta]


def heading_from_velocity(vx: float, vy: float) -> float:
    return atan2(vy, vx)
