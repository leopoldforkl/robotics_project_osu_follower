import numpy as np
import matplotlib.pyplot as plt
import copy

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import *
from dwa import dwa_control_v3, motion
from utils import get_zero_path_point  # Or your racetrack generator

import itertools

# --- Basic Setup ---
total_steps = 400
dt = DT
target_path = [get_zero_path_point(step, total_steps) for step in range(total_steps)]

# --- Run Simulation with Given Parameters ---
def run_simulation(params, total_steps=400):
    # Patch config dynamically
    globals()['POPULATION_SIZE'] = params['population_size']
    globals()['NUM_SELECTED'] = params['num_selected']
    globals()['BASE_MUTATION_SCALE_V'] = params['mutation_scale_v']
    globals()['BASE_MUTATION_SCALE_W'] = params['mutation_scale_w']
    globals()['TRAJECTORY_PENALTY_GAIN'] = params['trajectory_penalty_gain']
    globals()['TRAJECTORY_PENALTY_EXPONENT'] = params['trajectory_penalty_exponent']
    globals()['POSITION_ERROR_COST_GAIN'] = params['position_error_gain']
    globals()['RELATIVE_ANGLE_COST_GAIN'] = params['relative_angle_gain']
    globals()['SPEED_COST_GAIN'] = params['speed_cost_gain']
    globals()['ANGLE_SPIKE_THRESHOLD_DEG'] = params['spike_threshold_deg']
    globals()['ANGLE_SPIKE_PENALTY_GAIN'] = params['spike_penalty_gain']
    globals()['ANGLE_DEADZONE_BONUS_GAIN'] = params['deadzone_bonus_gain']

    # Initial position
    state = [target_path[0][0] - 1.0, target_path[0][1], np.pi/2]
    prev_population = None
    distance_errors = []
    angle_errors = []

    for step in range(total_steps):
        goal = target_path[step % total_steps]
        [v, w], cost, prev_population = dwa_control_v3(
            state, goal, prev_population,
            step_count=step,
            total_steps=total_steps,
            desired_relative_angle_rad=0.0  # Tracking ahead of target
        )
        state = motion(state, v, w, dt)

        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        dist_error = np.hypot(dx, dy)

        target_heading = np.arctan2(dy, dx)
        angle_error = np.abs(np.arctan2(np.sin(target_heading - state[2]), np.cos(target_heading - state[2])))

        distance_errors.append(dist_error)
        angle_errors.append(np.degrees(angle_error))

    avg_dist = np.mean(distance_errors)
    avg_angle = np.mean(angle_errors)
    score = avg_dist + 0.1 * avg_angle  # Weighted combination

    return score, avg_dist, avg_angle

# --- Define Sweep Grid ---
param_grid = {
    'population_size': [20, 30],
    'num_selected': [5, 10],
    'mutation_scale_v': [0.1, 0.2],
    'mutation_scale_w': [0.1, 0.2],
    'trajectory_penalty_gain': [0.2, 0.3],
    'trajectory_penalty_exponent': [1.2, 1.5],
    'position_error_gain': [1.0, 2.0],
    'relative_angle_gain': [2.0, 4.0, 6.0],
    'speed_cost_gain': [0.0, 0.2, 0.5],
    'spike_threshold_deg': [20, 30],
    'spike_penalty_gain': [10.0, 20.0],
    'deadzone_bonus_gain': [2.0, 5.0]
}

keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Total Combinations: {len(param_combinations)}")

# --- Sweep All Combinations ---
results = []
for idx, params in enumerate(param_combinations):
    print(f"Testing {idx+1}/{len(param_combinations)}...")
    score, avg_dist_error, avg_angle_error = run_simulation(params)
    results.append((params, score, avg_dist_error, avg_angle_error))

# --- Sort and Show Top Results ---
results.sort(key=lambda x: x[1])  # Sort by overall score

print("\nTop 10 Best Parameter Sets:")
for i in range(min(10, len(results))):
    best_params, best_score, best_dist, best_angle = results[i]
    print(f"\nRank {i+1}")
    for key, val in best_params.items():
        print(f"  {key}: {val}")
    print(f"  Avg Distance Error: {best_dist:.3f} m")
    print(f"  Avg Angle Error: {best_angle:.3f} deg")

# --- Optional: Save Results to File
import json
with open('sweep_results.json', 'w') as f:
    json.dump(results, f, indent=2)
