#!/usr/bin/env python3

import numpy as np
import math


def compute_cmd_vel(target_position, prev_target_position, prev_cmd_vel, dt_ms):
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
    
    Returns:
        np.ndarray: 6x1 numpy array [vx, vy, vz, wx, wy, wz] command velocities
    """
    
    # Simple proportional controller gains
    Kp_linear = 2.0
    Kp_angular = 1.0
    x, y, z = target_position
    vx = Kp_linear * x
    vy = Kp_linear * y
    wz = 0.0 # Kp_angular * math.atan2(y, x)

    cmd_vel = np.array([vx, vy, 0.0, 0.0, 0.0, wz])
    
    return cmd_vel
