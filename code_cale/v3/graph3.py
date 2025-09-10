import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

OFFSET_ANGLE=270

# --- Global Parameters ---
def generate_faster_target_path(radius=5.0, steps=100, speed_scale=1.5, offset_magnitude=2.0, offset_angle_deg=OFFSET_ANGLE):
    theta = np.linspace(0, np.pi, steps)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta * speed_scale)
    target_path = np.stack([x, y], axis=1)

    # Compute motion vectors (forward direction of the target)
    motion_vecs = np.gradient(target_path, axis=0)
    motion_unit = motion_vecs / np.linalg.norm(motion_vecs, axis=1, keepdims=True)

    # Rotate motion vector by offset angle (e.g. 270 degrees behind)
    offset_angle_rad = np.deg2rad(offset_angle_deg)
    rot_matrix = np.array([
        [np.cos(offset_angle_rad), -np.sin(offset_angle_rad)],
        [np.sin(offset_angle_rad),  np.cos(offset_angle_rad)],
    ])
    rotated_offsets = motion_unit @ rot_matrix.T

    offset_path = target_path + rotated_offsets * offset_magnitude

    return target_path, offset_path

def simulate_tight_hybrid_motion(offset_path, max_speed=0.8, alpha=0.85):
    path, pos, velocity = [], offset_path[0].copy(), np.zeros(2)
    for goal in offset_path:
        direction = goal - pos
        direction /= np.linalg.norm(direction) + 1e-6
        desired_velocity = max_speed * direction
        velocity = alpha * velocity + (1 - alpha) * desired_velocity
        pos += velocity
        path.append(pos.copy())
    return np.array(path)

def simulate_orientation_aware_dwa(target_path, desired_angle_deg=OFFSET_ANGLE, offset_magnitude=2.0, max_accel=0.015, speed_ratio=1.5, goal_update_interval=3):
    path = []
    velocity = np.zeros(2)
    desired_angle_rad = np.radians(desired_angle_deg)

    initial_direction = target_path[1] - target_path[0]
    if np.linalg.norm(initial_direction) < 1e-6:
        initial_direction = np.array([1.0, 0.0])
    else:
        initial_direction = initial_direction / np.linalg.norm(initial_direction)

    rot_matrix = np.array([
        [np.cos(desired_angle_rad), -np.sin(desired_angle_rad)],
        [np.sin(desired_angle_rad),  np.cos(desired_angle_rad)]
    ])
    offset_vec = rot_matrix @ initial_direction * offset_magnitude
    pos = target_path[0] + offset_vec

    path = [pos.copy()]
    current_offset_goal = None

    for i in range(1, len(target_path)):
        motion_vec = target_path[i] - target_path[i - 1]
        motion_norm = np.linalg.norm(motion_vec)
        if motion_norm < 1e-6:
            forward_dir = np.array([1.0, 0.0])
        else:
            forward_dir = motion_vec / motion_norm

        offset_vec = rot_matrix @ forward_dir * offset_magnitude
        new_goal = target_path[i] + offset_vec

        if i % goal_update_interval == 0 or current_offset_goal is None:
            current_offset_goal = new_goal

        direction = current_offset_goal - pos
        direction /= (np.linalg.norm(direction) + 1e-6)

        target_speed = speed_ratio * motion_norm
        target_velocity = target_speed * direction

        delta_v = target_velocity - velocity
        delta_v_norm = np.linalg.norm(delta_v)
        if delta_v_norm > max_accel:
            delta_v = delta_v / delta_v_norm * max_accel

        velocity += delta_v
        pos += velocity
        path.append(pos.copy())

    return np.array(path)

def simulate_realistic_dwa(offset_path, max_accel=0.015, speed_ratio=1.5):
    path, pos, velocity = [], offset_path[0].copy(), np.zeros(2)
    for i in range(1, len(offset_path)):
        goal, prev_goal = offset_path[i], offset_path[i - 1]
        target_velocity_vector = goal - prev_goal
        target_speed = np.linalg.norm(target_velocity_vector)
        max_speed = speed_ratio * target_speed

        direction = goal - pos
        direction /= np.linalg.norm(direction) + 1e-6
        target_velocity = max_speed * direction
        delta_v = target_velocity - velocity
        delta_v = delta_v / (np.linalg.norm(delta_v) + 1e-6) * min(max_accel, np.linalg.norm(delta_v))
        velocity += delta_v
        pos += velocity
        path.append(pos.copy())
    path.insert(0, offset_path[0].copy())
    return np.array(path)

# --- Run Simulations ---
target_path, offset_path = generate_faster_target_path()
hybrid_path = simulate_tight_hybrid_motion(offset_path)
orient_path = simulate_orientation_aware_dwa(target_path)
dwa_path = simulate_realistic_dwa(offset_path)

# --- Plot Animation ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(np.min(target_path[:, 0]) - 4, np.max(target_path[:, 0]) + 4)
ax.set_ylim(np.min(target_path[:, 1]) - 4, np.max(target_path[:, 1]) + 4)
ax.set_title('Trajectory Comparison', fontsize=24)
ax.set_aspect('equal')

lines = {
    'target': ax.plot([], [], 'k-', label='Target')[0],
    'offset': ax.plot([], [], 'm--', label='Ideal Offset')[0],
    'hybrid': ax.plot([], [], 'g-', label='Hybrid')[0],
    'orient': ax.plot([], [], 'r--', label='Orientation-Aware DWA')[0],
    'dwa': ax.plot([], [], 'b--', label='DWA')[0]
}
dots = {
    'target': ax.plot([], [], 'ko', markersize=6)[0],
    'offset': ax.plot([], [], 'mo', markersize=6)[0],
    'hybrid': ax.plot([], [], 'go', markersize=4)[0],
    'orient': ax.plot([], [], 'ro', markersize=4)[0],
    'dwa': ax.plot([], [], 'bo', markersize=4)[0]
}
ax.legend(fontsize=16)

def init():
    for line in lines.values(): line.set_data([], [])
    for dot in dots.values(): dot.set_data([], [])
    return list(lines.values()) + list(dots.values())

def update(i):
    lines['target'].set_data(target_path[:i, 0], target_path[:i, 1])
    lines['offset'].set_data(offset_path[:i, 0], offset_path[:i, 1])
    lines['hybrid'].set_data(hybrid_path[:i, 0], hybrid_path[:i, 1])
    lines['orient'].set_data(orient_path[:i, 0], orient_path[:i, 1])
    lines['dwa'].set_data(dwa_path[:i, 0], dwa_path[:i, 1])

    dots['target'].set_data([target_path[i, 0]], [target_path[i, 1]])
    dots['offset'].set_data([offset_path[i, 0]], [offset_path[i, 1]])
    dots['hybrid'].set_data([hybrid_path[i, 0]], [hybrid_path[i, 1]])
    dots['orient'].set_data([orient_path[i, 0]], [orient_path[i, 1]])
    dots['dwa'].set_data([dwa_path[i, 0]], [dwa_path[i, 1]])

    return list(lines.values()) + list(dots.values())

ani = animation.FuncAnimation(fig, update, frames=len(target_path), init_func=init, blit=True, interval=100)
ani.save("hybrid_vs_orientation_aware_dwa.gif", writer="pillow", fps=10)
