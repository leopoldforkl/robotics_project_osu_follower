import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# === compare_dwa_versions.py WITH ANGLE + NOISE SWEEP, DISTANCE + ANGLE ERROR, PATH + LENGTH ===

import numpy as np
import matplotlib.pyplot as plt
from v1.dwa import dwa_control as dwa_v1
from v1.robot_model import motion as motion_v1
from v2.dwa import dwa_control as dwa_v2
from v2.robot_model import motion as motion_v2
from v2.config import *
import os

np.random.seed(42)

NOISE_LEVELS = np.linspace(0.0, 0.02, 9)
VIEW_ANGLES_DEG = np.linspace(0, 360, 8, endpoint=False)
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
    if noise is not None:
        curr += noise[current_step]
        next_pt += noise[min(current_step + 1, total_steps - 1)]
    heading = np.arctan2(next_pt[1] - curr[1], next_pt[0] - curr[0])
    angle = heading + offset_angle_rad
    dx = offset_dist * np.cos(angle)
    dy = offset_dist * np.sin(angle)
    return np.array([curr[0] + dx, curr[1] + dy]), heading

def compute_angle_error(robot_pose, target_heading, offset_angle_rad):
    vector_to_robot = np.array([robot_pose[0], robot_pose[1]])
    expected_heading = target_heading + offset_angle_rad
    angle_diff = np.arctan2(np.sin(expected_heading), np.cos(expected_heading))
    robot_heading = np.arctan2(vector_to_robot[1], vector_to_robot[0])
    return (robot_heading - angle_diff + np.pi) % (2 * np.pi) - np.pi

def run_dwa(dwa_control, motion, offset_dist, offset_angle_rad, noise):
    state = [0.0, 0.0, 0.0]
    goal, _ = get_displaced_zero_goal(0, offset_dist, offset_angle_rad, noise=noise)
    state[0], state[1] = goal[0], goal[1]
    distance_errors = []
    angle_errors = []
    path = [state[:2].copy()]
    target_path = []

    for step in range(ARC_TOTAL_STEPS):
        target_pt = get_zero_path_point(step, ARC_TOTAL_STEPS) + (noise[step] if noise is not None else 0)
        target_path.append(target_pt)

        goal, heading = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, noise=noise)
        distance_error = np.linalg.norm(goal - np.array(state[:2]))
        target_heading = np.arctan2(goal[1] - target_pt[1], goal[0] - target_pt[0])
        vector_to_robot = np.array([state[0] - target_pt[0], state[1] - target_pt[1]])
        actual_angle = np.arctan2(vector_to_robot[1], vector_to_robot[0])
        expected_angle = target_heading
        angle_error = (actual_angle - expected_angle + np.pi) % (2 * np.pi) - np.pi

        lookahead_steps = int(np.clip(
            LOOKAHEAD_DISTANCE_GAIN * distance_error,
            LOOKAHEAD_MIN,
            LOOKAHEAD_MAX
        ))
        goal, _ = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps, noise=noise)
        u, _ = dwa_control(state, goal)
        state = motion(state, *u, DT)

        distance_errors.append(distance_error)
        angle_errors.append(np.abs(angle_error))
        path.append(state[:2].copy())

    path = np.array(path)
    target_path = np.array(target_path)
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return np.mean(distance_errors), np.mean(np.rad2deg(angle_errors)), path, target_path, path_length

def main(offset_dist=2.0):
    for angle_deg in VIEW_ANGLES_DEG:
        angle_rad = np.deg2rad(angle_deg)
        errs_v1, errs_v2 = [], []
        angle_v1, angle_v2 = [], []
        length_v1, length_v2 = [], []
        
        os.makedirs(f"{RESULTS_DIR}/degree_{int(angle_deg)}", exist_ok=True)

        for std in NOISE_LEVELS:
            noise = np.random.normal(0, std, size=(ARC_TOTAL_STEPS, 2))
            d1, a1, p1, t1, l1 = run_dwa(dwa_v1, motion_v1, offset_dist, angle_rad, noise)
            d2, a2, p2, t2, l2 = run_dwa(dwa_v2, motion_v2, offset_dist, angle_rad, noise)

            print(d1,a1)
            print(d2,a2)
            errs_v1.append(d1)
            errs_v2.append(d2)
            angle_v1.append(a1)
            angle_v2.append(a2)
            length_v1.append(l1)
            length_v2.append(l2)

            # Path plot
            plt.figure()
            plt.plot(p1[:, 0], p1[:, 1], "r-", label="DWA v1")
            plt.plot(p2[:, 0], p2[:, 1], "b-", label="DWA v2")
            plt.plot(t1[:, 0], t1[:, 1], "k--", label="Target Path")
            plt.axis("equal")
            plt.title(f"Paths at angle {angle_deg:.0f}°, noise {std:.2f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"degree_{int(angle_deg)}/path_angle_{int(angle_deg)}_noise_{std:.2f}.png"))
            plt.close()

            # Distance error over time
            plt.figure()
            plt.plot(np.abs(np.diff(p1, axis=0)).sum(axis=1), label="v1 step length")
            plt.plot(np.abs(np.diff(p2, axis=0)).sum(axis=1), label="v2 step length")
            plt.title(f"Step Path Lengths at angle {angle_deg:.0f}°, noise {std:.2f}")
            plt.xlabel("Timestep")
            plt.ylabel("Step Distance")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"degree_{int(angle_deg)}/step_path_lengths_{int(angle_deg)}_noise_{std:.2f}.png"))
            plt.close()

            # Error over time
            plt.figure()
            plt.plot(np.linspace(0, 1, len(errs_v1)), errs_v1, "r-", label="Distance Error v1")
            plt.plot(np.linspace(0, 1, len(errs_v2)), errs_v2, "b-", label="Distance Error v2")
            plt.title(f"Distance Error over Time at angle {angle_deg:.0f}°, noise {std:.2f}")
            plt.xlabel("Normalized Time")
            plt.ylabel("Error")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"degree_{int(angle_deg)}/distance_error_overtime_angle_{int(angle_deg)}_noise_{std:.2f}.png"))
            plt.close()


            plt.figure()
            plt.plot(np.linspace(0, 1, len(angle_v1)), angle_v1, "r--", label="Angle Error v1")
            plt.plot(np.linspace(0, 1, len(angle_v2)), angle_v2, "b--", label="Angle Error v2")
            plt.title(f"Angle Error over Time at angle {angle_deg:.0f}°, noise {std:.2f}")
            plt.xlabel("Normalized Time")
            plt.ylabel("Error")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"degree_{int(angle_deg)}/angle_error_overtime_angle_{int(angle_deg)}_noise_{std:.2f}.png"))
            plt.close()

            print(f"Finished Noise Level {std}")

        # Distance Error
        plt.figure()
        plt.plot(NOISE_LEVELS, errs_v1, "r-o", label="DWA v1")
        plt.plot(NOISE_LEVELS, errs_v2, "b-o", label="DWA v2")
        plt.xlabel("Noise Std Dev")
        plt.ylabel("Mean Distance Error")
        plt.title(f"Distance Error at {angle_deg:.0f}° Offset")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"angle_{int(angle_deg)}_distance.png"))
        plt.close()

        # Angle Error
        plt.figure()
        plt.plot(NOISE_LEVELS, angle_v1, "r-o", label="DWA v1")
        plt.plot(NOISE_LEVELS, angle_v2, "b-o", label="DWA v2")
        plt.xlabel("Noise Std Dev")
        plt.ylabel("Mean Angle Error (deg)")
        plt.title(f"Angle Error at {angle_deg:.0f}° Offset")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"angle_{int(angle_deg)}_angle.png"))
        plt.close()

        # Path Length
        plt.figure()
        plt.plot(NOISE_LEVELS, length_v1, "r-o", label="DWA v1")
        plt.plot(NOISE_LEVELS, length_v2, "b-o", label="DWA v2")
        plt.xlabel("Noise Std Dev")
        plt.ylabel("Path Length")
        plt.title(f"Path Length at {angle_deg:.0f}° Offset")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"angle_{int(angle_deg)}_length.png"))
        plt.close()
        print(f"Saved plots for {angle_deg:.0f}°")
        print(f"Finished Angle {angle_deg}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset_dist", type=float, default=2.0)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist)
