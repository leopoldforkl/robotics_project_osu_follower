# === compare_dwa_versions_gif.py ===
# Runs all 3 planners side-by-side with synchronized motion and saves to a single GIF

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
import sys
import matplotlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v1.dwa import dwa_control as dwa_v1
from v1.robot_model import motion as motion_v1
from v2.dwa import dwa_control as dwa_v2
from v2.robot_model import motion as motion_v2
from v3.dwa import hybrid_dwa_ga_control as dwa_v3
from v3.robot_model import motion as motion_v3
from v3.config import *

def get_zero_path_point(step, total_steps):
    arc1_steps = int(total_steps * 0.35)
    line1_steps = int(total_steps * 0.15)
    arc2_steps = int(total_steps * 0.35)
    line2_steps = total_steps - (arc1_steps + line1_steps + arc2_steps)

    segment_bounds = [arc1_steps, arc1_steps + line1_steps,
                      arc1_steps + line1_steps + arc2_steps]

    if step < segment_bounds[0]:
        t = step / arc1_steps
        angle = np.pi + np.pi * t
        x = 6.0 - 5.0 * np.cos(angle)
        y = 9.0 - 5.0 * np.sin(angle)
    elif step < segment_bounds[1]:
        t = (step - segment_bounds[0]) / line1_steps
        x = 1.0
        y = 9.0 - 6.0 * t
    elif step < segment_bounds[2]:
        t = (step - segment_bounds[1]) / arc2_steps
        angle = np.pi + np.pi * t
        x = 6.0 + 5.0 * np.cos(angle)
        y = 3.0 + 5.0 * np.sin(angle)
    else:
        t = (step - segment_bounds[2]) / line2_steps
        x = 11.0
        y = 3.0 + 6.0 * t

    return np.array([x, y])

def get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps=0):
    total_steps = ARC_TOTAL_STEPS
    current_step = min(step + lookahead_steps, total_steps - 1)
    curr = get_zero_path_point(current_step, total_steps)
    next_pt = get_zero_path_point(min(current_step + 1, total_steps - 1), total_steps)
    heading = np.arctan2(next_pt[1] - curr[1], next_pt[0] - curr[0])
    goal_heading = heading + offset_angle_rad
    dx = offset_dist * np.cos(goal_heading)
    dy = offset_dist * np.sin(goal_heading)
    return np.array([curr[0] + dx, curr[1] + dy]), heading

def angle_error(a, b):
    return np.arctan2(np.sin(a - b), np.cos(a - b))

def compute_angle_err(robot_state, target_pos, target_heading, desired_offset_rad):
    robot_pos = np.array(robot_state[:2])
    offset_vector = robot_pos - target_pos
    actual_offset_angle = np.arctan2(offset_vector[1], offset_vector[0]) - target_heading
    actual_offset_angle = (actual_offset_angle + np.pi) % (2 * np.pi) - np.pi
    return angle_error(actual_offset_angle, desired_offset_rad)

def main(offset_dist=1.5, offset_angle_deg=-45):
    offset_angle_rad = np.deg2rad(offset_angle_deg)

    matplotlib.use('TkAgg')
    # (Window state now set earlier)
    # fig already created above
    try:
        fig.canvas.manager.window.state('zoomed')
    except:
        try:
            fig.canvas.manager.resize(*fig.canvas.manager.window.maxsize())
        except:
            pass

    goal0, heading0 = get_displaced_zero_goal(0, offset_dist, offset_angle_rad)
    state_v1 = [*goal0, heading0]
    state_v2 = state_v1.copy()
    state_v3 = state_v1.copy()

    traj_v1, traj_v2, traj_v3 = [state_v1], [state_v2], [state_v3]
    errors_v1, errors_v2, errors_v3 = [], [], []
    angle_v1, angle_v2, angle_v3 = [], [], []
    guide_trail = []
    prev_velocity_v3 = (0.0, 0.0)

    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.0, 1.0, 1.0])
    axs = [[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])],
        [fig.add_subplot(gs[1, :])],
        [fig.add_subplot(gs[2, :])]]
    planner_labels = ["DWA", "Orientation-Aware DWA", "Hybrid DWA-GA"]
    gif_frames = []
    base_size = None

    max_error = offset_dist * 2
    max_angle_err = 0.5  # in radians

    for step in range(ARC_TOTAL_STEPS):
        current_guide = get_zero_path_point(step, ARC_TOTAL_STEPS)
        guide_trail.append(current_guide)

        distance_error = np.linalg.norm(current_guide - np.array(state_v1[:2]))
        lookahead_steps = int(np.clip(
            LOOKAHEAD_DISTANCE_GAIN * distance_error,
            LOOKAHEAD_MIN,
            LOOKAHEAD_MAX
        ))

        goal, target_heading = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps)

        u1, _ = dwa_v1(state_v1, goal)
        u2, _ = dwa_v2(state_v2, goal)
        u3, _ = dwa_v3(state_v3, goal, target_heading, prev_velocity=prev_velocity_v3)

        prev_velocity_v3 = u3[:2]
        state_v1 = motion_v1(state_v1, *u1, DT)
        state_v2 = motion_v2(state_v2, *u2, DT)
        state_v3 = motion_v3(state_v3, *u3, DT)

        traj_v1.append(state_v1)
        traj_v2.append(state_v2)
        traj_v3.append(state_v3)

        e1 = np.linalg.norm(goal - np.array(state_v1[:2]))
        e2 = np.linalg.norm(goal - np.array(state_v2[:2]))
        e3 = np.linalg.norm(goal - np.array(state_v3[:2]))
        errors_v1.append(e1)
        errors_v2.append(e2)
        errors_v3.append(e3)

        a1 = abs(compute_angle_err(state_v1, current_guide, target_heading, offset_angle_rad))
        a2 = abs(compute_angle_err(state_v2, current_guide, target_heading, offset_angle_rad))
        a3 = abs(compute_angle_err(state_v3, current_guide, target_heading, offset_angle_rad))
        angle_v1.append(a1)
        angle_v2.append(a2)
        angle_v3.append(a3)

        max_error = max(max_error, e1, e2, e3)
        max_angle_err = max(max_angle_err, a1, a2, a3)

        trajs = [np.array(traj_v1), np.array(traj_v2), np.array(traj_v3)]
        guide_np = np.array(guide_trail)

        for i, (ax, traj, label) in enumerate(zip(axs[0], trajs, planner_labels)):
            ax.clear()
            color = ['blue', 'red', 'green'][i]
            ax.plot(traj[:, 0], traj[:, 1], label=label, linewidth=2.5, color=color)
            ax.plot(goal[0], goal[1], "yo", markersize=8, label="Goal")
            ax.plot(traj[-1, 0], traj[-1, 1], "mo", markersize=6, label="Robot")
            ax.plot(guide_np[:, 0], guide_np[:, 1], "k--", linewidth=2.5)
            ax.plot(current_guide[0], current_guide[1], 'ks', markersize=6, label="Target")

            for j in range(0, len(guide_np), 25):
                dx = guide_np[j, 0] - traj[j, 0]
                dy = guide_np[j, 1] - traj[j, 1]
                ax.quiver(traj[j, 0], traj[j, 1], dx, dy, angles='xy', scale_units='xy', scale=1.0, width=0.008, headwidth=4, headlength=6)

            ax.set_xlim(-6, 18)
            ax.set_ylim(-6, 18)
            ax.set_aspect('equal')
            ax.set_title(label, fontsize=20)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.legend(loc='upper right', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        axs[1][0].clear()
        axs[1][0].plot(errors_v1, color='blue', label='DWA')
        axs[1][0].plot(errors_v2, color='red', label='Orientation-Aware DWA')
        axs[1][0].plot(errors_v3, color='green', label='Hybrid DWA-GA')
        axs[1][0].set_ylim(0, max_error * 1.1)
        axs[1][0].set_title("Distance Error", fontsize=14)
        axs[1][0].set_xlabel("Step")
        axs[1][0].set_ylabel("Error (m)")
        axs[1][0].legend()

        axs[2][0].clear()
        axs[2][0].plot(np.rad2deg(angle_v1), color='blue', label='DWA')
        axs[2][0].plot(np.rad2deg(angle_v2), color='red', label='Orientation-Aware DWA')
        axs[2][0].plot(np.rad2deg(angle_v3), color='green', label='Hybrid DWA-GA')
        axs[2][0].set_ylim(0, np.rad2deg(max_angle_err * 1.1))
        axs[2][0].set_title("Angle Error", fontsize=14)
        axs[2][0].set_xlabel("Step")
        axs[2][0].set_ylabel("Error (deg)")
        axs[2][0].legend()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if base_size is None:
            base_size = img.size
        img = img.resize(base_size)
        gif_frames.append(img)

        plt.pause(0.01)

    gif_frames[0].save(
        "compare_all_planners.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0
    )

    plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare All DWA Planners in Real Time")
    parser.add_argument("--offset_dist", type=float, default=1.0)
    parser.add_argument("--offset_angle_deg", type=float, default=-45)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist, offset_angle_deg=args.offset_angle_deg)
