import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import imageio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v1.dwa import dwa_control as dwa_v1
from v1.robot_model import motion as motion_v1
from v1.utils import get_displaced_goal as get_goal_v1

from v3.dwa import hybrid_dwa_ga_control as dwa_v3
from v3.robot_model import motion as motion_v3
from v3.config import *
from v3.utils import get_displaced_goal as get_goal_v3

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

def run_dwa(dwa_control, motion, offset_dist, offset_angle_rad, tag):
    trajectory = []
    state = get_displaced_zero_goal(0, offset_dist, offset_angle_rad).tolist() + [0.0]
    prev_velocity = [0.0, 0.0]

    for step in range(ARC_TOTAL_STEPS + 1):
        goal = get_displaced_zero_goal(step, offset_dist, offset_angle_rad)
        target_pos = get_displaced_zero_goal(step, 0.0, 0.0)

        if tag == 'v3':
            goal_heading = np.arctan2(goal[1] - target_pos[1], goal[0] - target_pos[0])
            u, _ = dwa_control(state, goal, goal_heading, prev_velocity=prev_velocity)
            prev_velocity = u
        else:
            u, _ = dwa_control(state, goal)

        state = motion(state, *u, DT)
        trajectory.append(state[:2])

    return np.array(trajectory)

def make_side_by_side_gif(traj_v1, traj_v3, guide, filename="compare_dwa/side_by_side.gif"):
    os.makedirs("compare_dwa/frames", exist_ok=True)
    images = []
    for i in range(min(len(traj_v1), len(traj_v3))):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        for ax, traj, label, color in zip([ax1, ax2], [traj_v1, traj_v3], ["Naive DWA", "Hybrid DWA-GA"], ["red", "green"]):
            ax.plot(guide[:, 0], guide[:, 1], 'k--', label='Target Path')
            ax.plot(traj[:i+1, 0], traj[:i+1, 1], color=color, label=label)
            ax.scatter(traj[i, 0], traj[i, 1], c=color, s=60)
            ax.scatter(guide[i, 0], guide[i, 1], c='blue', s=40, label='Target')
            ax.set_title(label)
            ax.set_xlim(0, 12)
            ax.set_ylim(0, 12)
            ax.set_aspect('equal')
            ax.legend()

        plt.tight_layout()
        frame_path = f"compare_dwa/frames/frame_{i:04d}.png"
        plt.savefig(frame_path)
        images.append(imageio.imread(frame_path))
        plt.close()

    imageio.mimsave(filename, images, fps=10)

def main(offset_dist=2.0, offset_angle_deg=90):
    offset_angle_rad = np.deg2rad(offset_angle_deg)

    print(f"\nRunning planners at {offset_angle_deg} degrees...")
    traj_v1 = run_dwa(dwa_v1, motion_v1, offset_dist, offset_angle_rad, tag='v1')
    traj_v3 = run_dwa(dwa_v3, motion_v3, offset_dist, offset_angle_rad, tag='v3')
    guide = np.array([get_displaced_zero_goal(i, 0.0, 0.0) for i in range(ARC_TOTAL_STEPS + 1)])

    print("Generating GIF...")
    os.makedirs("compare_dwa", exist_ok=True)
    make_side_by_side_gif(traj_v1, traj_v3, guide)
    print("GIF saved to compare_dwa/side_by_side.gif")

if __name__ == "__main__":
    main()
