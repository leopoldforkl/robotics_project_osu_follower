"""v4 Simulation entry point.
Robot follows a moving target along a composite path, maintaining an offset and desired facing.
Outputs a live plot and (optionally) an animated GIF.
"""
from __future__ import annotations
import os
from math import radians, cos, sin, atan2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v4.config import (
    DT, TOTAL_STEPS, TARGET_STEP_SIZE,
    OFFSET_LOCAL_X, OFFSET_LOCAL_Y, FACING_OFFSET_DEG,
    ARROW_LEN_FACING, ARROW_LEN_VELOCITY, ARROW_LEN_TARGET,
    PLOT_XLIM, PLOT_YLIM, SAVE_GIF, GIF_NAME, OUTPUT_DIR, GIF_DPI, GIF_FRAME_MS, GIF_FRAME_PAUSE,
    MAX_YAW_RATE
)
from v4.path_target import path_point, path_tangent
from v4.utils import offset_point_local_frame, desired_facing_from_target, normalize_angle
from v4.motion import integrate_position
from v4.controller import hybrid_control


def run():
    np.random.seed(42)

    facing_offset_rad = radians(FACING_OFFSET_DEG)

    # Initial target + derived offset + facing
    target_pos = path_point(0)
    target_heading = path_tangent(0)
    prev_target_pos = target_pos.copy()
    offset_pos = offset_point_local_frame(target_pos, target_heading, OFFSET_LOCAL_X, OFFSET_LOCAL_Y)
    desired_facing = desired_facing_from_target(target_heading, facing_offset_rad)

    # Robot state: [x, y, theta]
    state = [offset_pos[0], offset_pos[1], desired_facing]
    prev_cmd = (0.0, 0.0)

    # Recording
    robot_traj = [state.copy()]
    target_traj = [target_pos.copy()]
    offset_traj = [offset_pos.copy()]

    if SAVE_GIF:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    gif_frames = []
    base_size = None

    fig, ax = plt.subplots(figsize=(6,6))

    progress = 0.0
    for step in range(TOTAL_STEPS):
        # --- Target update with controllable speed ---
        progress = min(progress, TOTAL_STEPS - 2)  # clamp to allow lookahead for heading
        idx = int(progress)
        frac = progress - idx
        p0 = path_point(idx)
        p1 = path_point(min(idx + 1, TOTAL_STEPS - 1))
        target_pos = p0 + frac * (p1 - p0)
        # Compute heading from actual motion (previous -> current)
        move_vec = target_pos - prev_target_pos
        mv_norm = np.linalg.norm(move_vec)
        if mv_norm > 1e-9:
            target_heading = atan2(move_vec[1], move_vec[0])
        else:
            target_heading = path_tangent(idx)
        prev_target_pos = target_pos.copy()
        target_traj.append(target_pos.copy())
        progress += TARGET_STEP_SIZE

        # --- Desired offset + facing ---
        offset_pos = offset_point_local_frame(target_pos, target_heading, OFFSET_LOCAL_X, OFFSET_LOCAL_Y)
        desired_facing = desired_facing_from_target(target_heading, facing_offset_rad)
        offset_traj.append(offset_pos.copy())

        # --- Control (choose planar velocity) ---
        # Pass offset as primary goal (second arg kept for cost context)
        (vx, vy), pred_traj = hybrid_control(state, offset_pos, offset_pos, desired_facing, prev_cmd)
        prev_cmd = (vx, vy)

        # Integrate planar position
        state = integrate_position(state, vx, vy)

        # Update heading toward desired_facing with slew limit
        angle_err = normalize_angle(desired_facing - state[2])
        max_delta = MAX_YAW_RATE * DT
        delta = np.clip(angle_err, -max_delta, max_delta)
        state[2] = normalize_angle(state[2] + delta)

        robot_traj.append(state.copy())

        # --- Plot ---
        ax.clear()
        r_np = np.array(robot_traj)
        t_np = np.array(target_traj)
        o_np = np.array(offset_traj)

        ax.plot(r_np[:,0], r_np[:,1], '-b', linewidth=2.0, label='Robot Path')
        ax.plot(t_np[:,0], t_np[:,1], 'k--', linewidth=2.0, label='Target Path')
        ax.plot(o_np[:,0], o_np[:,1], ':g', linewidth=1.5, label='Offset Path')

        # Current positions
        ax.plot(target_pos[0], target_pos[1], 'ro', label='Target')
        ax.plot(state[0], state[1], 'mo', label='Robot')

        # Target heading arrow
        ax.arrow(target_pos[0], target_pos[1],
                 ARROW_LEN_TARGET*cos(target_heading), ARROW_LEN_TARGET*sin(target_heading),
                 head_width=0.18, head_length=0.18, fc='red', ec='red', length_includes_head=True)

        # Robot facing arrow
        ax.arrow(state[0], state[1],
                 ARROW_LEN_FACING*cos(state[2]), ARROW_LEN_FACING*sin(state[2]),
                 head_width=0.18, head_length=0.18, fc='green', ec='green', length_includes_head=True)

        # Robot velocity arrow (movement); scale to arrow length
        speed = np.hypot(vx, vy)
        if speed > 1e-6:
            scale = ARROW_LEN_VELOCITY / speed
            ax.arrow(state[0], state[1], vx*scale, vy*scale,
                     head_width=0.14, head_length=0.14, fc='orange', ec='orange', length_includes_head=True)

        ax.set_xlim(*PLOT_XLIM)
        ax.set_ylim(*PLOT_YLIM)
        ax.set_aspect('equal')
        ax.set_title('v4 Hybrid Follower')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.legend(loc='upper right')

        if SAVE_GIF:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=GIF_DPI)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')
            if base_size is None:
                base_size = img.size
            img = img.resize(base_size)
            gif_frames.append(img)

        plt.pause(GIF_FRAME_PAUSE)

    if SAVE_GIF and gif_frames:
        out_path = os.path.join(OUTPUT_DIR, GIF_NAME)
        gif_frames[0].save(out_path, save_all=True, append_images=gif_frames[1:], duration=GIF_FRAME_MS, loop=0)
        print(f"Saved GIF to {out_path}")

    plt.show()


if __name__ == '__main__':
    run()
