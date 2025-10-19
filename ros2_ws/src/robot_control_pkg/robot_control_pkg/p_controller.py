#!/usr/bin/env python3

import numpy as np
import math


def compute_cmd_vel(target_position, prev_target_position, prev_cmd_vel, dt_ms, controller_params):
    """
    Compute command velocities based on target position relative to robot.
    
    Args:
        target_position (np.ndarray): 3x1 numpy array [x, y, z] of target position
                                    in robot_sim frame coordinates
        prev_target_position (np.ndarray): 3x1 numpy array [x, y, z] of target position
                                         from previous timestep
        prev_cmd_vel (np.ndarray): 6x1 numpy array [vx, vy, vz, wx, wy, wz] of previous
                                 command velocities
        dt_ms (float): Time difference between timesteps in milliseconds
        controller_params (dict): Dictionary containing controller parameters
    
    Returns:
        np.ndarray: 6x1 numpy array [vx, vy, vz, wx, wy, wz] command velocities
    """
    
    # Extract parameters
    Kp_linear = controller_params['kp_linear']
    Kp_angular = controller_params['kp_angular']
    max_linear_vel = controller_params['max_linear_velocity']
    max_angular_vel = controller_params['max_angular_velocity']
    x_offset = controller_params['x_offset']
    y_offset = controller_params['y_offset']
    angle_offset = controller_params['angle_offset']
    vx = Kp_linear * (target_position[0] - x_offset)
    vy = Kp_linear * (target_position[1] - y_offset)

    # Angular control:
    Kp_angular = controller_params['kp_angular']
    angle_offset = controller_params['angle_offset']
    
    # Convert dt to seconds
    dt = dt_ms / 1000.0
    
    # Extract previous velocities
    prev_vx, prev_vy, _, _, _, prev_wz = prev_cmd_vel
    
    # Estimate robot's motion in world frame
    robot_rotation = prev_wz * dt
    robot_dx = prev_vx * dt
    robot_dy = prev_vy * dt

    current_robot_heading = robot_rotation #radians

    # Get Target heading
    prev_target_world_x = prev_target_position[0]
    prev_target_world_y = prev_target_position[1]

    cos_rot = math.cos(robot_rotation)
    sin_rot = math.sin(robot_rotation)

    current_target_world_x = robot_dx + (prev_target_position[0] * cos_rot - prev_target_position[1] * sin_rot)
    current_target_world_y = robot_dy + (prev_target_position[0] * sin_rot + prev_target_position[1] * cos_rot)

    target_velocity_vector = np.array([
        current_target_world_x - prev_target_world_x,
        current_target_world_y - prev_target_world_y
    ])

    #normalize the vector if its magnitude is larger than 0.001:
    target_speed = np.linalg.norm(target_velocity_vector)
    if target_speed > 0.001:
        target_velocity_vector /= target_speed
    else:
        target_velocity_vector = np.array([0.0, 0.0])

    robot_heading_vector = np.array([
        math.cos(current_robot_heading),
        math.sin(current_robot_heading)
    ])

    # Do the math:
    # Compute 2D cross and dot products
    cross = robot_heading_vector[0]*target_velocity_vector[1] - robot_heading_vector[1]*target_velocity_vector[0]
    dot = np.dot(robot_heading_vector, target_velocity_vector)

    # Compute signed angle in radians (-pi to pi)
    angle_error = np.arctan2(cross, dot)


    # use p controller to compute angular velocity
    wz = Kp_angular * (angle_error - angle_offset)

    # Apply velocity limits using parameters
    if wz > max_angular_vel:
        wz = max_angular_vel
    elif wz < -max_angular_vel:
        wz = -max_angular_vel
    
    # limit linear velocities using parameters
    if vx > max_linear_vel:
        vx = max_linear_vel
    elif vx < -max_linear_vel:
        vx = -max_linear_vel
    if vy > max_linear_vel:
        vy = max_linear_vel
    elif vy < -max_linear_vel:
        vy = -max_linear_vel

    cmd_vel = np.array([vx, vy, 0.0, 0.0, 0.0, wz])
    
    return cmd_vel
