import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v1.dwa import dwa_control as dwa_v1
from v1.robot_model import motion as motion_v1
from v2.dwa import dwa_control as dwa_v2
from v2.robot_model import motion as motion_v2
from v3.dwa import hybrid_dwa_ga_control as dwa_v3
from v3.robot_model import motion as motion_v3
from v2.config import *
from v3.config import *

#NOISE_LEVELS = np.linspace(0.0, 0.02, 9)
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
#NOISE_LEVELS = [0.0, 0.05]
VIEW_ANGLES_DEG = np.linspace(0, 360, 8, endpoint=False)
#VIEW_ANGLES_DEG = [135]
TRIALS = 5
from concurrent.futures import ProcessPoolExecutor

RESULTS_DIR = "noisy_comparison_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def get_zero_path_point(step, total_steps):
    radius = 5.0
    center_x = 6.0
    top_center_y = 9.0
    bottom_center_y = 3.0
    arc_steps = total_steps // 4
    segment = step // arc_steps
    step_in_seg = step % arc_steps
    t = step_in_seg / arc_steps

    if segment == 0:
        angle = np.pi + np.pi * t
        x = center_x - radius * np.cos(angle)
        y = top_center_y - radius * np.sin(angle)
    elif segment == 1:
        x = center_x - radius
        y = top_center_y - (top_center_y - bottom_center_y) * t
    elif segment == 2:
        angle = np.pi + np.pi * t
        x = center_x + radius * np.cos(angle)
        y = bottom_center_y + radius * np.sin(angle)
    else:
        x = center_x + radius
        y = bottom_center_y + (top_center_y - bottom_center_y) * t
    return np.array([x, y])

def get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps=0, noise=None):
    total_steps = ARC_TOTAL_STEPS
    current_step = min(step + lookahead_steps, total_steps - 1)
    curr = get_zero_path_point(current_step, total_steps)
    next_pt = get_zero_path_point(min(current_step + 1, total_steps - 1), total_steps)
    #if noise is not None:
    #    curr += noise[current_step]
    #    next_pt += noise[min(current_step + 1, total_steps - 1)]
    heading = np.arctan2(next_pt[1] - curr[1], next_pt[0] - curr[0])
    angle = heading + offset_angle_rad
    dx = offset_dist * np.cos(angle)
    dy = offset_dist * np.sin(angle)
    return np.array([curr[0] + dx, curr[1] + dy]), heading

def run_dwa(dwa_control, motion, offset_dist, offset_angle_rad, noise, tag=None):
    goal0, _ = get_displaced_zero_goal(0, offset_dist, offset_angle_rad, noise=noise)
    state = [goal0[0], goal0[1], 0.0]
    prev_velocity = (0.0, 0.0) if tag == "v3" else None

    path = []
    target_path = []
    distance_errors = []
    angle_errors = []

    for step in range(ARC_TOTAL_STEPS):
        target_pt = get_zero_path_point(step, ARC_TOTAL_STEPS) + (noise[step] if noise is not None else 0)
        target_path.append(target_pt)

        goal, heading = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, noise=noise)
        distance_error = np.linalg.norm(goal - np.array(state[:2]))
        vector_to_robot = np.array(state[:2]) - target_pt
        actual_angle = np.arctan2(vector_to_robot[1], vector_to_robot[0])
        target_heading = np.arctan2(goal[1] - target_pt[1], goal[0] - target_pt[0])
        angle_error = (actual_angle - target_heading + np.pi) % (2 * np.pi) - np.pi

        lookahead_steps = int(np.clip(
            LOOKAHEAD_DISTANCE_GAIN * distance_error + LOOKAHEAD_ANGLE_GAIN * abs(angle_error),
            LOOKAHEAD_MIN,
            LOOKAHEAD_MAX
        ))
        goal, _ = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps)

        if tag == "v3":
            u, _ = dwa_control(state, goal, heading, prev_velocity=prev_velocity)
            prev_velocity = u
        else:
            u, _ = dwa_control(state, goal)

        state = motion(state, *u, DT)
        state[0] += noise[step][0]
        state[1] += noise[step][1]

        path.append(state[:2].copy())
        true_target_pt = get_zero_path_point(step, ARC_TOTAL_STEPS)
        true_goal, true_heading = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps, noise=noise)

        # Use actual final (noisy) position
        final_pos = np.array(state[:2])
        distance_error = np.linalg.norm(true_goal - final_pos)

        vector_to_robot = final_pos - true_target_pt
        actual_angle = np.arctan2(vector_to_robot[1], vector_to_robot[0])
        target_heading = np.arctan2(true_goal[1] - true_target_pt[1], true_goal[0] - true_target_pt[0])
        angle_error = (actual_angle - target_heading + np.pi) % (2 * np.pi) - np.pi

        distance_errors.append(distance_error)
        angle_errors.append(np.abs(angle_error))

    path = np.array(path)
    target_path = np.array(target_path)
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return np.mean(distance_errors), np.mean(np.rad2deg(angle_errors)), path, target_path, path_length
    #return np.mean(0), np.mean(np.rad2deg(0)), [0, 1, 2], [0, 1, 2], 0


def run_trial(args):
    std, seed, offset_dist, angle_rad = args
    from v1.dwa import dwa_control as dwa_v1
    from v1.robot_model import motion as motion_v1
    from v2.dwa import dwa_control as dwa_v2
    from v2.robot_model import motion as motion_v2
    from v3.dwa import hybrid_dwa_ga_control as dwa_v3
    from v3.robot_model import motion as motion_v3
    from v2.config import ARC_TOTAL_STEPS, DT
    from noise_compare import run_dwa, get_zero_path_point, get_displaced_zero_goal
    import numpy as np

    rng = np.random.default_rng(seed)
    process_noise = rng.normal(0, std, size=(ARC_TOTAL_STEPS, 2))
    d1, a1, _, _, l1 = run_dwa(dwa_v1, motion_v1, offset_dist, angle_rad, process_noise)
    d2, a2, _, _, l2 = run_dwa(dwa_v2, motion_v2, offset_dist, angle_rad, process_noise)
    d3, a3, _, _, l3 = run_dwa(dwa_v3, motion_v3, offset_dist, angle_rad, process_noise, tag="v3")
        
    return (d1, a1, l1), (d2, a2, l2), (d3, a3, l3)

def main(offset_dist=2.0):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    results = {
        "v1": {"distance": {}, "angle": {}, "length": {}},
        "v2": {"distance": {}, "angle": {}, "length": {}},
        "v3": {"distance": {}, "angle": {}, "length": {}}
    }

    for version in results:
        for metric in results[version]:
            for angle_deg in VIEW_ANGLES_DEG:
                results[version][metric][angle_deg] = {}

    for angle_deg in VIEW_ANGLES_DEG:
        angle_rad = np.deg2rad(angle_deg)
        errs_v1, errs_v2, errs_v3 = [], [], []
        angle_v1, angle_v2, angle_v3 = [], [], []
        length_v1, length_v2, length_v3 = [], [], []

        os.makedirs(f"{RESULTS_DIR}/degree_{int(angle_deg)}", exist_ok=True)
        rng = np.random.default_rng(seed=42)
        
        for std_index, std in enumerate(NOISE_LEVELS):
            tasks = [(std, 1000 + trial, offset_dist, angle_rad) for trial in range(TRIALS)]
            with ProcessPoolExecutor() as executor:
                for (d1, a1, l1), (d2, a2, l2), (d3, a3, l3) in executor.map(run_trial, tasks):
                    
                    results["v1"]["distance"][angle_deg][std] = d1
                    results["v1"]["angle"][angle_deg][std] = a1
                    results["v1"]["length"][angle_deg][std] = l1
                    results["v2"]["distance"][angle_deg][std] = d2
                    results["v2"]["angle"][angle_deg][std] = a2
                    results["v2"]["length"][angle_deg][std] = l2
                    results["v3"]["distance"][angle_deg][std] = d3
                    results["v3"]["angle"][angle_deg][std] = a3
                    results["v3"]["length"][angle_deg][std] = l3
    
            print(f"Finished Noise Level {std} for Angle: {angle_deg}")
     
    # === Export nested results to CSV ===
    import pandas as pd
    records = []

    for version in results:
        for metric in results[version]:
            for angle in results[version][metric]:
                for noise in results[version][metric][angle]:
                    records.append({
                        "version": version,
                        "metric": metric,
                        "angle": angle,
                        "noise": noise,
                        "value": results[version][metric][angle][noise]
                    })

    df = pd.DataFrame(records)
    df.to_csv(f"{RESULTS_DIR}/noise_results_summary.csv", index=False)
 

    # === Plot results per angle ===
    for i, angle_deg in enumerate(VIEW_ANGLES_DEG):
        label_angle = int(angle_deg)

        for metric, ylabel, filename in [
            ("length", "Path Length (m)", "length"),
            ("distance", "Mean Distance Error (m)", "distance"),
            ("angle", "Mean Angle Error (deg)", "angle")
        ]:
            plt.figure()
            plt.plot(NOISE_LEVELS, [results["v1"][metric][angle_deg][n] for n in NOISE_LEVELS], label="Naive DWA", color='red', marker='o')
            plt.plot(NOISE_LEVELS, [results["v2"][metric][angle_deg][n] for n in NOISE_LEVELS], label="Orientation-Aware DWA", color='blue', marker='o')
            plt.plot(NOISE_LEVELS, [results["v3"][metric][angle_deg][n] for n in NOISE_LEVELS], label="Hybrid DWA-GA", color='green', marker='o')
            plt.title(f"{ylabel} at {label_angle}° Offset", fontsize=24)
            plt.xlabel("Noise Std Dev", fontsize=20)
            plt.ylabel(ylabel, fontsize=20)
            plt.legend(fontsize=14)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{RESULTS_DIR}/angle_{label_angle}_{filename}.png", bbox_inches='tight')
            plt.close()

    # === Summary plots across all angles ===
    for metric, ylabel, filename in [
        ("length", "Path Length (m)", "length"),
        ("distance", "Mean Distance Error (m)", "distance"),
        ("angle", "Mean Angle Error (deg)", "angle")
    ]:
        plt.figure()
        for version, name, color in [
            ("v1", "Naive DWA", "red"),
            ("v2", "Orientation-Aware DWA", "blue"),
            ("v3", "Hybrid DWA-GA", "green")
        ]:
            y_vals = []
            for std in NOISE_LEVELS:
                # Collect values across angles for this noise level
                values = [
                    results[version][metric][angle][std]
                    for angle in results[version][metric]
                    if std in results[version][metric][angle]
                ]
                if values:
                    y_vals.append(np.mean(values))
                else:
                    y_vals.append(np.nan)  # handle missing values gracefully

            plt.plot(NOISE_LEVELS, y_vals, label=name, color=color, marker='o')

        plt.title(f"{ylabel} vs Noise", fontsize=24)
        plt.xlabel("Noise Std Dev", fontsize=20)
        plt.ylabel(ylabel, fontsize=20)
        plt.legend(fontsize=18)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/{filename}_summary.png")
        plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset_dist", type=float, default=2.0)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist)