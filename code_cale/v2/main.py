# === UPDATED main.py FOR OMNIDIRECTIONAL DWA TRACKING with GIF and Visual Improvements ===

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import io
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v2.dwa import dwa_control
from v2.robot_model import motion
from v3.config import *

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

def get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps=0):
    total_steps = ARC_TOTAL_STEPS
    current_step = min(step + lookahead_steps, total_steps - 1)
    curr = get_zero_path_point(current_step, total_steps)
    next_pt = get_zero_path_point(min(current_step + 1, total_steps - 1), total_steps)
    heading = np.arctan2(next_pt[1] - curr[1], next_pt[0] - curr[0])
    angle = heading + offset_angle_rad
    dx = offset_dist * np.cos(angle)
    dy = offset_dist * np.sin(angle)
    return np.array([curr[0] + dx, curr[1] + dy])

def main(offset_dist=2.0, offset_angle_deg=90.0):
    offset_angle_rad = np.deg2rad(offset_angle_deg)
    initial_goal = get_displaced_zero_goal(0, offset_dist, offset_angle_rad)
    state = [initial_goal[0], initial_goal[1], 0.0]
    trajectory = [state]
    guide_trail = []
    distance_errors = []

    fig, ax = plt.subplots(figsize=(6, 6))
    mng = plt.get_current_fig_manager()
    try:
        mng.full_screen_toggle()
    except AttributeError:
        try:
            mng.window.showMaximized()
        except AttributeError:
            pass

    gif_frames = []
    base_size = None

    for step in range(ARC_TOTAL_STEPS):
        current_guide = get_zero_path_point(step, ARC_TOTAL_STEPS)
        goal = get_displaced_zero_goal(step, offset_dist, offset_angle_rad)

        distance_error = np.linalg.norm(goal - np.array(state[:2]))
        lookahead_steps = int(np.clip(
            LOOKAHEAD_DISTANCE_GAIN * distance_error,
            LOOKAHEAD_MIN,
            LOOKAHEAD_MAX
        ))

        goal = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps)
        u, traj = dwa_control(state, goal)
        state = motion(state, *u, DT)

        trajectory.append(state)
        distance_errors.append(distance_error)
        guide_trail.append(current_guide)

        # Plotting
        ax.clear()
        traj_np = np.array(trajectory)
        guide_np = np.array(guide_trail)

        ax.plot(traj_np[:, 0], traj_np[:, 1], "-b", label="Robot Path", linewidth=2.5)
        ax.plot(goal[0], goal[1], "ro", label="Goal", markersize=8)
        ax.plot(state[0], state[1], "mo", label="Robot", markersize=8)
        ax.plot(guide_np[:, 0], guide_np[:, 1], "k--", linewidth=2.5, label="Target Path")

        for i in range(0, len(guide_np), 10):
            dx = guide_np[i, 0] - traj_np[i, 0]
            dy = guide_np[i, 1] - traj_np[i, 1]
            ax.quiver(traj_np[i, 0], traj_np[i, 1], dx, dy, angles='xy', scale_units='xy', scale=1, color='blue', width=0.003)

        ax.set_xlim(-3, 15)
        ax.set_ylim(-3, 15)
        ax.set_aspect('equal')
        ax.set_title("Omnidirectional DWA Tracking", fontsize=24)
        ax.set_xlabel("X [m]", fontsize=18)
        ax.set_ylabel("Y [m]", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16)

        # Save current frame to GIF
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if base_size is None:
            base_size = img.size
        img = img.resize(base_size)
        gif_frames.append(img)

        plt.pause(0.01)

    # Save GIF
    gif_frames[0].save(
        "omnidirectional_tracking.gif",
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0
    )

    print("\nAverage Distance Error:", np.mean(distance_errors))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Omnidirectional DWA Simulation")
    parser.add_argument("--offset_dist", type=float, default=2.0)
    parser.add_argument("--offset_angle_deg", type=float, default=90.0)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist, offset_angle_deg=args.offset_angle_deg)
