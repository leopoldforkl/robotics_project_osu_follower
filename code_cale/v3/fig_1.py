import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams['figure.dpi'] = 300

def plot_robot_box(ax, x, y, size=0.3, color='blue'):
    box = Rectangle((x - size/2, y - size/2), size, size,
                    edgecolor=color, facecolor='none', lw=1)
    ax.add_patch(box)

def plot_arrow_to_target(ax, x, y, dx, dy, color='blue'):
    ax.arrow(x, y, dx, dy,
             head_width=0.2, head_length=0.3, fc='orange', ec='orange', length_includes_head=True)

def load_paths_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    arrow1 = []
    arrow2 = []
    current = None
    for line in lines:
        line = line.strip()
        if line.startswith("# Arrow 1"):
            current = arrow1
        elif line.startswith("# Arrow 2"):
            current = arrow2
        elif line and not line.startswith("#"):
            x, y, theta = map(float, line.split(","))
            current.append([x, y, theta])
    return np.array(arrow1), np.array(arrow2)

def main():
    # Settings
    arrow_stride = 5         # Arrows from robot to target
    square_stride = 5        # Plot square every N robot steps
    path_stride = 1          # Plot every Nth point for path
    truncate_ratio = 0.35    # Fraction of trajectory to show

    # Load paths
    target_path, robot_path = load_paths_from_file("waypoints.txt")
    max_steps = int(min(len(robot_path), len(target_path)) * truncate_ratio)

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)

    # 1. Plot target path (subsampled)
    ax.plot(target_path[:max_steps:path_stride, 0],
            target_path[:max_steps:path_stride, 1],
            'k.', markersize=10, label="Target Path")

    # 2. Plot robot path as blue dots and overlay squares
    for i in range(0, max_steps, path_stride):
        x, y, _ = robot_path[i]
        if i == 0:
            ax.plot(x, y, 'b.', markersize=6, label='Robot Trajectory')
        ax.plot(x, y, 'b.', markersize=6)
        if i % square_stride == 0:
            plot_robot_box(ax, x, y, color='blue')

    # 3. Plot arrows at stride intervals
    for i in range(0, max_steps, arrow_stride):
        x, y, _ = robot_path[i]
        tx, ty = target_path[i][:2]
        dx = tx - x
        dy = ty - y
        plot_arrow_to_target(ax, x, y, dx, dy, color='blue')

    # 4. Start and goal markers
    ax.plot(robot_path[0, 0], robot_path[0, 1], 'gs', markersize=10, label='Start')
    ax.plot(robot_path[max_steps-1, 0], robot_path[max_steps-1, 1], 'ro', markersize=10, label='End')

    ax.set_aspect('equal')
    ax.set_title("Robot Path with Directional Arrows to Target")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(prop={'size': 14})

    plt.savefig("fig_1_output.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
