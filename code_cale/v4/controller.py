"""Hybrid DWA + simple GA controller for holonomic follower (v4).
Simplified from v3 for clarity; we sample velocity commands, evolve them briefly,
score trajectories, and choose the best smoothed command.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import atan2, hypot, sin, cos, pi
from typing import List, Tuple
import numpy as np

from .config import (
    MAX_SPEED, VEL_RES, PREDICT_TIME, DT,
    POPULATION_SIZE, GENERATIONS, PARENTS, MUTATION_STD,
    SMOOTHING_ALPHA,
    W_GOAL_DIST, W_OFFSET_DIST, W_PATH_LENGTH, W_FACING, W_CURVATURE, W_JERK,
    CURVATURE_DEADZONE, JERK_DEADZONE
)
from .motion import integrate_position

@dataclass
class Candidate:
    vx: float
    vy: float
    traj: List[List[float]]
    cost: float

# ---------------- Trajectory rollout ---------------- #

def rollout(state: List[float], vx: float, vy: float) -> List[List[float]]:
    t = 0.0
    s = list(state)
    traj = [s]
    while t < PREDICT_TIME:
        s = integrate_position(s, vx, vy)
        traj.append(s)
        t += DT
    return traj

# ---------------- Cost Function ---------------- #

def trajectory_cost(traj: List[List[float]], goal: np.ndarray, offset_point: np.ndarray, desired_facing: float) -> float:
    end = np.array(traj[-1][:2])
    start = np.array(traj[0][:2])

    # Primary objective: stay near offset anchor; secondary: keep reasonable proximity to true target
    d_offset = np.linalg.norm(offset_point - end)
    d_goal = np.linalg.norm(goal - end)
    # Early discard if moving farther from offset compared to start (allows aggressive pruning)
    if np.linalg.norm(offset_point - start) + 0.05 < d_offset:
        return 1e6

    # path length
    path_len = 0.0
    for i in range(len(traj) - 1):
        p0 = np.array(traj[i][:2])
        p1 = np.array(traj[i+1][:2])
        path_len += np.linalg.norm(p1 - p0)

    # curvature (sum of heading changes) with deadzone
    curvature = 0.0
    for i in range(1, len(traj) - 1):
        v1 = np.array(traj[i][:2]) - np.array(traj[i-1][:2])
        v2 = np.array(traj[i+1][:2]) - np.array(traj[i][:2])
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            continue
        a1 = atan2(v1[1], v1[0])
        a2 = atan2(v2[1], v2[0])
        da = np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1))
        if abs(da) > CURVATURE_DEADZONE:
            curvature += abs(da)

    # jerk (velocity change magnitudes) with deadzone
    jerk = 0.0
    for i in range(1, len(traj) - 1):
        v_prev = np.array(traj[i][:2]) - np.array(traj[i-1][:2])
        v_next = np.array(traj[i+1][:2]) - np.array(traj[i][:2])
        dv = np.linalg.norm(v_next - v_prev)
        if dv > JERK_DEADZONE:
            jerk += dv

    # facing alignment: we treat desired_facing as an absolute heading target for end displacement vector
    move_vec = end - start
    if np.linalg.norm(move_vec) < 1e-6:
        facing_pen = 0.0
    else:
        move_angle = atan2(move_vec[1], move_vec[0])
        diff = np.arctan2(np.sin(move_angle - desired_facing), np.cos(move_angle - desired_facing))
        facing_pen = abs(diff)

    cost = (W_OFFSET_DIST * d_offset +
            W_GOAL_DIST * d_goal +
            W_PATH_LENGTH * path_len +
            W_FACING * facing_pen +
            W_CURVATURE * curvature +
            W_JERK * jerk)
    return cost

# ---------------- Population / GA ---------------- #

def initial_population(state: List[float], goal: np.ndarray) -> List[Candidate]:
    vector = goal - np.array(state[:2])
    base_angle = atan2(vector[1], vector[0])
    pop: List[Candidate] = []
    for _ in range(POPULATION_SIZE):
        ang = base_angle + np.random.uniform(-pi/3, pi/3)
        speed = np.random.uniform(0.2 * MAX_SPEED, MAX_SPEED)
        vx = speed * np.cos(ang)
        vy = speed * np.sin(ang)
        traj = rollout(state, vx, vy)
        pop.append(Candidate(vx, vy, traj, 0.0))
    return pop

def evaluate_population(pop: List[Candidate], goal: np.ndarray, offset_point: np.ndarray, desired_facing: float):
    for c in pop:
        c.cost = trajectory_cost(c.traj, goal, offset_point, desired_facing)


def select_elite(pop: List[Candidate]) -> List[Candidate]:
    return sorted(pop, key=lambda c: c.cost)[:PARENTS]

def reproduce(elite: List[Candidate], state: List[float]) -> List[Candidate]:
    new: List[Candidate] = []
    while len(new) < POPULATION_SIZE:
        p1 = elite[np.random.randint(0, len(elite))]
        p2 = elite[np.random.randint(0, len(elite))]
        vx = p1.vx if np.random.rand() < 0.5 else p2.vx
        vy = p1.vy if np.random.rand() < 0.5 else p2.vy
        vx += np.random.randn() * MUTATION_STD
        vy += np.random.randn() * MUTATION_STD
        speed = hypot(vx, vy)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            vx *= scale
            vy *= scale
        traj = rollout(state, vx, vy)
        new.append(Candidate(vx, vy, traj, 0.0))
    return new

# ---------------- Public Control Function ---------------- #

def hybrid_control(state: List[float], goal: np.ndarray, offset_point: np.ndarray, desired_facing: float, prev_cmd: Tuple[float,float]) -> Tuple[Tuple[float,float], List[List[float]]]:
    pop = initial_population(state, goal)
    evaluate_population(pop, goal, offset_point, desired_facing)
    for _ in range(GENERATIONS):
        elite = select_elite(pop)
        pop = reproduce(elite, state)
        evaluate_population(pop, goal, offset_point, desired_facing)
    best = min(pop, key=lambda c: c.cost)
    # Smooth velocity command
    smoothed_vx = SMOOTHING_ALPHA * prev_cmd[0] + (1 - SMOOTHING_ALPHA) * best.vx
    smoothed_vy = SMOOTHING_ALPHA * prev_cmd[1] + (1 - SMOOTHING_ALPHA) * best.vy
    return (smoothed_vx, smoothed_vy), best.traj
