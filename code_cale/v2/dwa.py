# === REWRITTEN dwa.py FOR OMNIDIRECTIONAL ROBOT ===

import numpy as np

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2.config import *
from v2.robot_model import motion


def predict_trajectory(state, vx, vy):
    traj = []
    temp_state = list(state)
    time = 0.0
    while time <= PREDICT_TIME:
        temp_state = motion(temp_state, vx, vy, DT)
        traj.append(temp_state)
        time += DT
    return traj


def dwa_control(state, goal):
    best_traj = []
    min_cost = float("inf")
    best_u = [0.0, 0.0]

    #distance_to_goal = np.linalg.norm(goal - state[:2])

    # Search full (vx, vy) space
    for vx in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
        for vy in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
            speed = np.hypot(vx, vy)
            if speed > MAX_SPEED:
                continue

            traj = predict_trajectory(state, vx, vy)
            end_pose = traj[-1]

            # Position cost
            position_error = np.linalg.norm(goal - np.array(end_pose[:2]))
            to_goal_cost = POSITION_ERROR_COST_GAIN * position_error

            # Speed reward
            speed_score = speed / MAX_SPEED
            speed_cost = SPEED_COST_GAIN * (1 - speed_score)

            total_cost = to_goal_cost + speed_cost  # angle irrelevant for omnidirectional

            if total_cost < min_cost:
                min_cost = total_cost
                best_u = [vx, vy]
                best_traj = traj

    return best_u, best_traj
