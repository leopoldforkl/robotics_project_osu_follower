import numpy as np
from v1.config import *
from v1.robot_model import motion


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

    for vx in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
        for vy in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
            speed = np.hypot(vx, vy)
            if speed > MAX_SPEED:
                continue

            traj = predict_trajectory(state, vx, vy)
            end_pose = traj[-1]

            position_error = np.linalg.norm(goal - np.array(end_pose[:2]))
            to_goal_cost = POSITION_ERROR_COST_GAIN * position_error

            speed_score = speed / MAX_SPEED
            speed_cost = SPEED_COST_GAIN * (1 - speed_score)

            total_cost = to_goal_cost + speed_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_u = [vx, vy]
                best_traj = traj

    return best_u, best_traj
