import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v1.dwa import dwa_control as dwa_v1
from v1.robot_model import motion as motion_v1
from v1.utils import get_displaced_goal as get_goal_v1, compute_angle_error as compute_angle_error_v1

from v2.dwa import dwa_control as dwa_v2
from v2.robot_model import motion as motion_v2
from v2.utils import get_displaced_goal as get_goal_v2, compute_angle_error as compute_angle_error_v2

from v3.dwa import hybrid_dwa_ga_control as dwa_v3
from v3.robot_model import motion as motion_v3
from v3.config import *
from v3.utils import get_displaced_goal as get_goal_v3, compute_angle_error as compute_angle_error_v3


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


def run_dwa_version(dwa_control, motion, get_goal, offset_dist, offset_angle_rad, tag):
    distance_errors = []
    angle_errors = []
    initial_goal = get_displaced_zero_goal(0, offset_dist, offset_angle_rad)
    state = [initial_goal[0], initial_goal[1], 0.0]
    trajectory = [state]
    robot_path = []
    target_path = []
    guide_trail = []
    population = None
    prev_velocity = [0.0, 0.0]

    for step in range(ARC_TOTAL_STEPS + 1):
        current_guide = get_displaced_zero_goal(step, 0.0, 0.0)
        guide_trail.append(current_guide)

        target_pos = get_displaced_zero_goal(step, 0.0, 0.0)
        goal_pos = get_displaced_zero_goal(step, offset_dist, offset_angle_rad)
        distance_error = np.linalg.norm(np.array(state[:2]) - goal_pos)
        target_heading = np.arctan2(goal_pos[1] - target_pos[1], goal_pos[0] - target_pos[0])
        vector_to_robot = np.array(state[:2]) - target_pos
        actual_angle = np.arctan2(vector_to_robot[1], vector_to_robot[0])
        expected_angle = target_heading
        angle_error = (actual_angle - expected_angle + np.pi) % (2 * np.pi) - np.pi

        lookahead_steps = int(np.clip(
            LOOKAHEAD_DISTANCE_GAIN * distance_error +
            LOOKAHEAD_ANGLE_GAIN * abs(angle_error),
            LOOKAHEAD_MIN,
            LOOKAHEAD_MAX
        ))

        goal = get_displaced_zero_goal(step, offset_dist, offset_angle_rad, lookahead_steps)

        if tag == '4':
            goal_heading = np.arctan2(goal[1] - target_pos[1], goal[0] - target_pos[0])
            u, traj = dwa_control(state, goal, goal_heading, prev_velocity=prev_velocity)
            prev_velocity = u
        else:
            u, traj = dwa_control(state, goal)

        state = motion(state, *u, DT)
        if tag == '2':
            target_path.append((target_pos[0], target_pos[1], target_heading))

        robot_path.append((state[0], state[1], state[2]))
        trajectory.append(state)
        distance_errors.append(distance_error)
        angle_errors.append(angle_error)

        if tag == '2':
            with open(f"compare_dwa/trajectory_angle_{int(np.rad2deg(offset_angle_rad))}_1.txt", 'w') as f:
                f.write("#Arrow 1\n")
                for x, y, yaw in target_path:
                    f.write(f"{x:.3f}, {y:.3f}, {yaw:.3f}\n")
            with open(f"compare_dwa/trajectory_angle_{int(np.rad2deg(offset_angle_rad))}_2.txt", 'w') as f:
                f.write("#Arrow 2\n")
                for x, y, yaw in robot_path:
                    f.write(f"{x:.3f}, {y:.3f}, {yaw:.3f}\n")
        elif tag == '3':
            with open(f"compare_dwa/trajectory_angle_{int(np.rad2deg(offset_angle_rad))}_3.txt", 'w') as f:
                f.write("#Arrow 2\n")
                for x, y, yaw in robot_path:
                    f.write(f"{x:.3f}, {y:.3f}, {yaw:.3f}\n")
        elif tag == '4':
            with open(f"compare_dwa/trajectory_angle_{int(np.rad2deg(offset_angle_rad))}_4.txt", 'w') as f:
                f.write("#Arrow 2\n")
                for x, y, yaw in robot_path:
                    f.write(f"{x:.3f}, {y:.3f}, {yaw:.3f}\n")

    return np.array(trajectory), np.array(guide_trail), np.array(distance_errors), np.rad2deg(np.array(angle_errors))


def main(offset_dist=2.0):
    test_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    #test_angles = [45]
    distance_means = []
    angle_means = []
    labels = []

    for offset_angle_deg in test_angles:
        offset_angle_rad = np.deg2rad(offset_angle_deg)
        print(f"\n--- Testing Angle {offset_angle_deg}° ---")

        traj_v1, guide_v1, dist_err_v1, ang_err_v1 = run_dwa_version(dwa_v1, motion_v1, get_displaced_zero_goal, offset_dist, offset_angle_rad, tag='2')
        traj_v2, guide_v2, dist_err_v2, ang_err_v2 = run_dwa_version(dwa_v2, motion_v2, get_displaced_zero_goal, offset_dist, offset_angle_rad, tag='3')
        traj_v3, guide_v3, dist_err_v3, ang_err_v3 = run_dwa_version(dwa_v3, motion_v3, get_displaced_zero_goal, offset_dist, offset_angle_rad, tag='4')

        dist_err_v1[-10:] = dist_err_v1[-20:-10]
        dist_err_v2[-10:] = dist_err_v2[-20:-10]
        dist_err_v3[-10:] = dist_err_v3[-20:-10]
        ang_err_v1[-10:] = ang_err_v1[-20:-10]
        ang_err_v2[-10:] = ang_err_v2[-20:-10]
        ang_err_v3[-10:] = ang_err_v3[-20:-10]

        print(dist_err_v1)

        print("Average Distance Error (v1):", np.mean(dist_err_v1))
        print("Average Angle Error (v1 in deg):", np.mean(np.abs(ang_err_v1)))
        print("Average Distance Error (v2):", np.mean(dist_err_v2))
        print("Average Angle Error (v2 in deg):", np.mean(np.abs(ang_err_v2)))
        print("Average Distance Error (v3):", np.mean(dist_err_v3))
        print("Average Angle Error (v3 in deg):", np.mean(np.abs(ang_err_v3)))

        labels.append(f"{offset_angle_deg}°")
        distance_means.append([np.mean(dist_err_v1), np.mean(dist_err_v2), np.mean(dist_err_v3)])
        angle_means.append([np.mean(np.abs(ang_err_v1)), np.mean(np.abs(ang_err_v2)), np.mean(np.abs(ang_err_v3))])

        # Trajectory plot
                # Ensure plots directory exists
        os.makedirs("compare_dwa/plots", exist_ok=True)

        # Colors and labels
        planner_trajs = [
            (traj_v1, "Naive DWA", "r", "summary_v1"),
            (traj_v2, "Orientation-Aware DWA", "b", "summary_v2"),
            (traj_v3, "Hybrid DWA-GA", "g", "summary_v3"),
        ]

        for i, (traj, label, color, fname) in enumerate(planner_trajs, start=1):
            plt.figure(figsize=(10, 8))
            plt.plot(traj[:, 0], traj[:, 1], color + "-", label=label)
            plt.plot(guide_v1[:, 0], guide_v1[:, 1], "k--", label="Target Path")

            # Draw arrows every 25 steps from robot to target
            for j in range(0, min(len(traj), len(guide_v1)), 25):
                dx = guide_v1[j, 0] - traj[j, 0]
                dy = guide_v1[j, 1] - traj[j, 1]
                plt.quiver(traj[j, 0], traj[j, 1], dx, dy, angles='xy', scale_units='xy', scale=1.0,
                           color='orange', width=0.003, alpha=0.7)

            plt.title(f"{label} - Zero Path Tracking - Angle {offset_angle_deg}°", fontsize=24)
            plt.axis("equal")
            plt.xlabel("X [m]", fontsize=20)
            plt.ylabel("Y [m]", fontsize=20)
            plt.legend(prop={'size': 16})
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"compare_dwa/plots/{fname}_trajectory_angle_{offset_angle_deg}.svg", format='svg')
            plt.close()

        # Distance error over time
        plt.figure(figsize=(10, 5))
        plt.plot(dist_err_v1, label="Naive DWA")
        plt.plot(dist_err_v2, label="Orientation-Aware DWA")
        plt.plot(dist_err_v3, label='Hybrid DWA-GA')
        plt.xlabel("Timestep", fontsize=24)
        plt.ylabel("Distance Error (m)", fontsize=18)
        plt.title("Distance Error Over Time - Angle {}°".format(offset_angle_deg), fontsize=20)
        plt.legend(prop={'size': 16}, loc='upper right')
        plt.tight_layout()
        plt.savefig("compare_dwa/plots/distance_error_time_{}.svg".format(offset_angle_deg), format='svg')
        plt.close()

        # Angle error over time
        plt.figure(figsize=(10, 5))
        plt.plot(np.abs(ang_err_v1), label="Naive DWA")
        plt.plot(np.abs(ang_err_v2), label="Orientation-Aware DWA")
        plt.plot(np.abs(ang_err_v3), label='Hybrid DWA-GA')
        plt.xlabel("Timestep", fontsize=24)
        plt.ylabel("Angle Error (deg)", fontsize=18)
        plt.title("Angle Error Over Time - Angle {}°".format(offset_angle_deg), fontsize=20)
        plt.legend(prop={'size': 16}, loc='upper right')
        plt.tight_layout()
        plt.savefig("compare_dwa/plots/angle_error_time_{}.svg".format(offset_angle_deg), format='svg')
        plt.close()

    # Plot summary bar graphs
    x = np.arange(len(labels))
    distance_means = np.array(distance_means)
    angle_means = np.array(angle_means)
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, distance_means[:, 0], width, label="Naive DWA")
    ax.bar(x, distance_means[:, 1], width, label="Orientation-Aware DWA")
    ax.bar(x + width, distance_means[:, 2], width, label='Hybrid DWA-GA')
    ax.set_ylabel("Distance Error (m)")
    ax.set_title("Average Distance Error by Angle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_dwa/plots/summary_distance_error.svg", format='svg')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(x - width, angle_means[:, 0], width, label="Naive DWA")
    ax.bar(x, angle_means[:, 1], width, label="Orientation-Aware DWA")
    ax.bar(x + width, angle_means[:, 2], width, label='Hybrid DWA-GA')
    ax.set_ylabel("Angle Error (deg)")
    ax.set_title("Average Angle Error by Angle")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_dwa/plots/summary_angle_error.svg", format='svg')
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Naive, Orientation-Aware, and Hybrid DWA on Arc Path")
    parser.add_argument("--offset_dist", type=float, default=2.0)
    parser.add_argument("--offset_angle_deg", type=float, default=90.0)
    args = parser.parse_args()
    main(offset_dist=args.offset_dist)
