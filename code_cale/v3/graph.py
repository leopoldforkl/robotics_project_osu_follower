import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Global Parameters ---
def generate_faster_target_path(radius=5.0, steps=100, speed_scale=1.5, offset_magnitude=2.0):
    theta = np.linspace(0, np.pi, steps)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta * speed_scale)
    target_path = np.stack([x, y], axis=1)

    offset_angle = 0 * np.pi / 180
    dx = offset_magnitude * np.cos(theta + offset_angle)
    dy = offset_magnitude * np.sin(theta + offset_angle)
    offset_path = np.stack([x + dx, y + dy], axis=1)

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

# --- Run Simulation ---
target_path, offset_path = generate_faster_target_path()
hybrid_path = simulate_tight_hybrid_motion(offset_path)
#dwa_path = simulate_realistic_dwa(offset_path)

# --- Plot Animation ---
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(np.min(target_path[:, 0]) - 4, np.max(target_path[:, 0]) + 4)
ax.set_ylim(np.min(target_path[:, 1]) - 4, np.max(target_path[:, 1]) + 4)
ax.set_title('Trajectory Comparison')
ax.set_aspect('equal')

lines = {
    'target': ax.plot([], [], 'r-', label='Target')[0],
    'offset': ax.plot([], [], 'g--', label='Ideal Offset')[0],
    'hybrid': ax.plot([], [], 'b-', label='Hybrid')[0],
    #'dwa': ax.plot([], [], 'm--', label='DWA')[0]
}
dots = {
    'target': ax.plot([], [], 'ro', markersize=10)[0],
    'offset': ax.plot([], [], 'go', markersize=10)[0],
    'hybrid': ax.plot([], [], 'bo', markersize=5)[0],
    #'dwa': ax.plot([], [], 'mo', markersize=10)[0]
}
ax.legend()

def init():
    for line in lines.values(): line.set_data([], [])
    for dot in dots.values(): dot.set_data([], [])
    return list(lines.values()) + list(dots.values())

def update(i):
    lines['target'].set_data(target_path[:i, 0], target_path[:i, 1])
    lines['offset'].set_data(offset_path[:i, 0], offset_path[:i, 1])
    lines['hybrid'].set_data(hybrid_path[:i, 0], hybrid_path[:i, 1])
    #lines['dwa'].set_data(dwa_path[:i, 0], dwa_path[:i, 1])

    dots['target'].set_data([target_path[i, 0]], [target_path[i, 1]])
    dots['offset'].set_data([offset_path[i, 0]], [offset_path[i, 1]])
    dots['hybrid'].set_data([hybrid_path[i, 0]], [hybrid_path[i, 1]])
    #dots['dwa'].set_data([dwa_path[i, 0]], [dwa_path[i, 1]])

    return list(lines.values()) + list(dots.values())

ani = animation.FuncAnimation(fig, update, frames=len(target_path), init_func=init, blit=True, interval=100)
ani.save("trajectory_only_large.gif", writer="pillow", fps=10)
