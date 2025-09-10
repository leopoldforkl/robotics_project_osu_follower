import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from config import MAX_SPEED, GOAL_OFFSET_DIST, GOAL_OFFSET_ANGLE, DT
from dwa import predict_trajectory

plt.rcParams['figure.dpi'] = 300

# === Genetic Algorithm Config ===
POP_SIZE = 50
GENERATIONS = 20
MUTATION_STD = 0.5
SEED = 42
np.random.seed(SEED)

# === Setup ===
initial_state = [0.0, 0.0, 0.0]

# Simulate target motion with slight curve
target_start = np.array([5.0, 5.0])
target_dir = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4)])  # Diagonal
target_future = target_start + target_dir * 0.5

# Goal offset from future target position
angle_to_goal = np.arctan2(target_dir[1], target_dir[0]) + GOAL_OFFSET_ANGLE
goal_position = target_future + GOAL_OFFSET_DIST * np.array([np.cos(angle_to_goal), np.sin(angle_to_goal)])

# === Normalized goal vector for visualization ===
raw_goal_vec = goal_position - np.array(initial_state[:2])
norm = np.linalg.norm(raw_goal_vec)
goal_velocity = (raw_goal_vec / norm) * MAX_SPEED if norm > 1e-6 else np.array([0.0, 0.0])

# === Cost Function ===
def evaluate(vx, vy):
    traj = predict_trajectory(initial_state, vx, vy)
    if traj is None or len(traj) == 0:
        return np.inf
    final_pos = traj[-1][:2]
    pos_err = np.linalg.norm(final_pos - goal_position)
    speed = np.linalg.norm([vx, vy])
    cost = pos_err + 0.1 * (MAX_SPEED - speed)  # Prefer fast accurate solutions
    return cost

# === Genetic Algorithm ===
def run_ga():
    population = [(np.random.uniform(-MAX_SPEED, MAX_SPEED),
                   np.random.uniform(-MAX_SPEED, MAX_SPEED)) for _ in range(POP_SIZE)]
    evaluated = [(vx, vy, evaluate(vx, vy)) for vx, vy in population]

    for _ in range(GENERATIONS):
        evaluated.sort(key=lambda x: x[2])
        top = evaluated[:20]
        children = []

        for _ in range(POP_SIZE):
            parent = top[np.random.randint(len(top))]
            vx = parent[0] + np.random.normal(0, MUTATION_STD)
            vy = parent[1] + np.random.normal(0, MUTATION_STD)
            children.append((vx, vy))

        evaluated = [(vx, vy, evaluate(vx, vy)) for vx, vy in children]

    return population, evaluated

# === Plotting Function ===
def plot_population(ax, population, title, color_by_cost=False):
    if color_by_cost:
        vx, vy, cost = zip(*population)
        scatter = ax.scatter(vx, vy, c=cost, cmap=cm.viridis, s=60, edgecolor='k', label='Sampled Velocities')
    else:
        vx, vy = zip(*[(p[0], p[1]) for p in population])
        scatter = ax.scatter(vx, vy, color="blue", s=60)

    ax.scatter(0, 0, color="black", s=50, label="Current Velocity")  # size 50

    # Optional: mark goal direction (no arrow)
    ax.plot(goal_velocity[0], goal_velocity[1], 'rx', markersize=15, label='Goal Direction')  # size 15

    # Mark best velocity sample
    if color_by_cost:
        best = min(population, key=lambda x: x[2])
        ax.plot(best[0], best[1], 'bo', markersize=13, label='Best Velocity')  # size 13

    # Add dashed speed circle
    speed_circle = Circle((0, 0), MAX_SPEED, fill=False, linestyle='--',
                          edgecolor='gray', linewidth=1.5, label='Speed Limit')
    ax.add_patch(speed_circle)

    # Zoom out
    margin = 1.5
    x_max = MAX_SPEED + margin
    y_max = MAX_SPEED + margin
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)

    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("vx [m/s]")
    ax.set_ylabel("vy [m/s]")
    ax.legend(prop={'size':20})

    return scatter



# === Run and Plot ===
init_population, final_population = run_ga()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = plot_population(ax1, [(vx, vy) for vx, vy in init_population], "Initial Population")
scatter2 = plot_population(ax2, final_population, "Final Population", color_by_cost=True)

# Add colorbar beside final plot
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
cbar = plt.colorbar(scatter2, cax=cax)
cbar.set_label("Trajectory Cost")

plt.tight_layout()
plt.show()

