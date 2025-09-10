import numpy as np
from v1.config import *

def get_displaced_goal(step, offset_dist, offset_angle_rad, lookahead_steps=0):
    projected_step = min(step + lookahead_steps, ARC_TOTAL_STEPS)
    ratio = projected_step / ARC_TOTAL_STEPS
    angle = ARC_START_ANGLE + ratio * (ARC_END_ANGLE - ARC_START_ANGLE)

    cx, cy = ARC_CENTER
    radius = ARC_RADIUS

    base_x = cx + radius * np.cos(angle)
    base_y = cy + radius * np.sin(angle)

    tangent_theta = angle + np.pi / 2.0
    total_angle = tangent_theta + offset_angle_rad

    goal_x = base_x + offset_dist * np.cos(total_angle)
    goal_y = base_y + offset_dist * np.sin(total_angle)

    return [goal_x, goal_y]

def get_guide_point(step):
    ratio = min(step / ARC_TOTAL_STEPS, 1.0)
    angle = ARC_START_ANGLE + ratio * (ARC_END_ANGLE - ARC_START_ANGLE)
    cx, cy = ARC_CENTER
    radius = ARC_RADIUS
    x = cx + radius * np.cos(angle)
    y = cy + radius * np.sin(angle)
    return [x, y]

def calc_relative_angle_cost(final_pose, goal, desired_angle_rad):
    dx = goal[0] - final_pose[0]
    dy = goal[1] - final_pose[1]
    angle_to_goal = np.arctan2(dy, dx)
    robot_yaw = final_pose[2]
    relative_angle = angle_to_goal - robot_yaw
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    angle_error = relative_angle - desired_angle_rad
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi
    return abs(angle_error)

def calc_speed_cost(v):
    return MAX_SPEED - v

def compute_angle_error(robot_pose, goal, desired_angle_rad):
    dx = goal[0] - robot_pose[0]
    dy = goal[1] - robot_pose[1]
    angle_to_goal = np.arctan2(dy, dx)
    robot_yaw = robot_pose[2]
    relative_angle = angle_to_goal - robot_yaw
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
    error = relative_angle - desired_angle_rad
    error = (error + np.pi) % (2 * np.pi) - np.pi
    return error
