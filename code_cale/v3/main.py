# === FINAL main.py: Hybrid DWA-GA Tracking with Live View + GIF Saving ===

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import argparse
import sys
import os
import io

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dwa import hybrid_dwa_ga_control
from v3.robot_model import motion
from v3.config import *
from v3.utils import calc_relative_angle_cost

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
    angle_errors = []

    prev_velocity = (0.0, 0.0)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Maximize or fullscreen window
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
        next_pt = get_zero_path_point(min(step + 1, ARC_TOTAL_STEPS - 1), ARC_TOTAL_STEPS)
        goal_heading = np.arctan2(next_pt[1] - current_guide[1], next_pt[0] - current_guide[0])
        u, traj = hybrid_dwa_ga_control(state, goal, goal_heading, prev_velocity=prev_velocity)
        state = motion(state, *u, DT)
        prev_velocity = u[:2]

        trajectory.append(state)
        distance_errors.append(distance_error)
        angle_error = calc_relative_angle_cost(state, goal, DESIRED_RELATIVE_ANGLE)
        angle_errors.append(np.rad2deg(angle_error))
        guide_trail.append(current_guide)

        # Plot
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

        ax.set_xlim(-6, 18)
        ax.set_ylim(-6, 18)
        ax.set_aspect('equal')
        ax.set_title("Hybrid DWA-GA Tracking", fontsize=24)
        ax.set_xlabel("X [m]", fontsize=18)
        ax.set_ylabel("Y [m]", fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.legend(fontsize=16)

        # Save current frame as consistent RGB image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=200)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        if base_size is None:
            base_size = img.size
        img = img.resize(base_size)
        gif_frames.append(img)

        plt.pause(0.001)


    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(__file__), '../output')
    os.makedirs(output_dir, exist_ok=True)

    # Save GIF
    gif_path = os.path.join(output_dir, "tracking_animation.gif")
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=50,
        loop=0
    )
    plt.close(fig)

    # Save waypoints
    waypoints_path = os.path.join(output_dir, "waypoints.txt")
    with open(waypoints_path, "w") as f:
        f.write("# Arrow 1\n")
        for i in range(len(guide_trail)):
            x, y = guide_trail[i]
            if i == 0 and len(guide_trail) > 1:
                dx = guide_trail[1][0] - guide_trail[0][0]
                dy = guide_trail[1][1] - guide_trail[0][1]
                theta = np.arctan2(dy, dx)
            elif i > 0:
                dx = guide_trail[i][0] - guide_trail[i-1][0]
                dy = guide_trail[i][1] - guide_trail[i-1][1]
                theta = np.arctan2(dy, dx)
            else:
                theta = 0.0
            f.write(f"{x:.3f}, {y:.3f}, {theta:.3f}\n")

        f.write("# Arrow 2\n")
        for pt in trajectory:
            f.write(f"{pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}\n")

    print(f"\nAverage Distance Error: {np.mean(distance_errors)}")
    print(f"Average Angle Error (deg): {np.mean(angle_errors)}")
    print(f"GIF saved to: {gif_path}")
    print(f"Waypoints saved to: {waypoints_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid DWA-GA Simulation")
    parser.add_argument("--offset_dist", type=float, default=2.0)
    parser.add_argument("--offset_angle_deg", type=float, default=90.0)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist, offset_angle_deg=args.offset_angle_deg)
