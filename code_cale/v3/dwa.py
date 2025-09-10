
import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from v3.robot_model import motion
from v3.config import *

from scipy.interpolate import CubicSpline

def predict_trajectory(state, vx, vy):
    traj = []
    temp_state = list(state)
    time = 0.0
    while time <= PREDICT_TIME:
        temp_state = motion(temp_state, vx, vy, DT)
        traj.append(temp_state)
        time += DT
    return traj

def smooth_trajectory(traj):
    traj_np = np.array(traj)
    if len(traj_np) > 3:
        t_steps = np.arange(len(traj_np))
        x_spline = CubicSpline(t_steps, traj_np[:, 0])
        y_spline = CubicSpline(t_steps, traj_np[:, 1])
        smooth_t = np.linspace(0, len(traj_np) - 1, len(traj_np))
        smooth_x = x_spline(smooth_t)
        smooth_y = y_spline(smooth_t)
        return [[x, y, 0.0] for x, y in zip(smooth_x, smooth_y)]
    return traj

def evaluate_trajectory(traj, goal, goal_heading):
    traj_np = np.array(traj)
    if len(traj_np) > 3:
        t_steps = np.arange(len(traj_np))
        x_spline = CubicSpline(t_steps, traj_np[:, 0])
        y_spline = CubicSpline(t_steps, traj_np[:, 1])
        smooth_t = np.linspace(0, len(traj_np) - 1, len(traj_np))
        smooth_x = x_spline(smooth_t)
        smooth_y = y_spline(smooth_t)
        traj = [[x, y, 0.0] for x, y in zip(smooth_x, smooth_y)]

    start_pos = np.array(traj[0][:2])
    end_pos = np.array(traj[-1][:2])

    start_dist = np.linalg.norm(goal - start_pos)
    end_dist = np.linalg.norm(goal - end_pos)

    if end_dist > start_dist:
        return float("inf")

    trajectory_length = sum(
        np.linalg.norm(np.array(traj[i+1][:2]) - np.array(traj[i][:2]))
        for i in range(len(traj) - 1)
    )

    # Position error with deadzone
    if end_dist < POSITION_DEADZONE_RADIUS:
        position_cost = 0.0
    else:
        position_cost = POSITION_ERROR_COST_GAIN * end_dist

    # Relative angle cost with deadzone
    vector_to_robot = end_pos - goal
    actual_angle = np.arctan2(vector_to_robot[1], vector_to_robot[0])
    desired_angle = goal_heading + RELATIVE_ANGLE_GOAL_RAD
    angle_diff = np.abs(np.arctan2(np.sin(actual_angle - desired_angle), np.cos(actual_angle - desired_angle)))
    if angle_diff < ANGLE_DEADZONE_RAD:
        angle_cost = 0.0
    else:
        angle_cost = RELATIVE_ANGLE_COST_GAIN * angle_diff

    # Curvature penalty with deadzone
    curvature_penalty = 0.0
    for i in range(1, len(traj) - 1):
        v1 = np.array(traj[i][:2]) - np.array(traj[i-1][:2])
        v2 = np.array(traj[i+1][:2]) - np.array(traj[i][:2])
        a1 = np.arctan2(v1[1], v1[0])
        a2 = np.arctan2(v2[1], v2[0])
        angle_change = np.abs(np.arctan2(np.sin(a2 - a1), np.cos(a2 - a1)))
        if angle_change > CURVATURE_DEADZONE_RAD:
            curvature_penalty += angle_change

    # Jerk penalty with deadzone
    jerk_penalty = 0.0
    for i in range(1, len(traj) - 1):
        v_prev = np.array(traj[i][:2]) - np.array(traj[i-1][:2])
        v_next = np.array(traj[i+1][:2]) - np.array(traj[i][:2])
        delta_v = np.linalg.norm(v_next - v_prev)
        if delta_v > JERK_DEADZONE:
            jerk_penalty += delta_v

    cost = position_cost
    cost += TRAJECTORY_PENALTY_GAIN * trajectory_length
    cost += angle_cost
    cost += CURVATURE_PENALTY_GAIN * curvature_penalty
    cost += JERK_PENALTY_GAIN * jerk_penalty

    return cost


def select_and_breed(state, population, goal, goal_heading, num_selected, mutation_rate):
    scored = [(vx, vy, traj, evaluate_trajectory(traj, goal, goal_heading)) for (vx, vy, traj) in population]
    scored.sort(key=lambda x: x[3])
    parents = scored[:num_selected]

    new_population = []
    for _ in range(len(population)):
        p1 = parents[np.random.randint(0, len(parents))]
        p2 = parents[np.random.randint(0, len(parents))]

        new_vx = p1[0] if np.random.rand() < 0.5 else p2[0]
        new_vy = p1[1] if np.random.rand() < 0.5 else p2[1]

        new_vx += np.random.randn() * mutation_rate
        new_vy += np.random.randn() * mutation_rate

        speed = np.hypot(new_vx, new_vy)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            new_vx *= scale
            new_vy *= scale

        new_vx = np.clip(new_vx, -MAX_SPEED, MAX_SPEED)
        new_vy = np.clip(new_vy, -MAX_SPEED, MAX_SPEED)

        traj = predict_trajectory(state, new_vx, new_vy)
        new_population.append((new_vx, new_vy, traj))

    return new_population

def hybrid_dwa_ga_control(state, goal, goal_heading, generations=5, population_size=80, prev_velocity=(0.0, 0.0)):
    candidates = []
    for vx in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
        for vy in np.arange(-MAX_SPEED, MAX_SPEED + VELOCITY_RESOLUTION, VELOCITY_RESOLUTION):
            if np.hypot(vx, vy) > MAX_SPEED:
                continue
            traj = predict_trajectory(state, vx, vy)
            candidates.append((vx, vy, traj))

    goal_vector = goal - np.array(state[:2])
    goal_angle = np.arctan2(goal_vector[1], goal_vector[0])

    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    dist_to_goal = np.hypot(dx, dy)
    
    DEADZONE_RADIUS = 0.0  # meters

    if dist_to_goal < DEADZONE_RADIUS:
        return [0.0, 0.0], [state] 
        #print(dist_to_goal)
    
    population = []
    for _ in range(population_size):
        angle = goal_angle + np.random.uniform(-np.pi / 4, np.pi / 4)  # ±45° cone
        speed = np.random.uniform(0.5 * MAX_SPEED, MAX_SPEED)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        traj = predict_trajectory(state, vx, vy)
        population.append((vx, vy, traj))


    for _ in range(generations):
        population = select_and_breed(state, population, goal, goal_heading, num_selected=5, mutation_rate=0.1)

    best = min(population, key=lambda x: evaluate_trajectory(x[2], goal, goal_heading))
    # Smooth the velocity with the previous command
    smoothed_vx = ALPHA * prev_velocity[0] + (1 - ALPHA) * best[0]
    smoothed_vy = ALPHA * prev_velocity[1] + (1 - ALPHA) * best[1]
    return [smoothed_vx, smoothed_vy], smooth_trajectory(best[2])

def hybrid_dwa_ga_predict_full_trajectory(
    state, goal, goal_heading, generations=5, population_size=80
):
    goal_vector = goal - np.array(state[:2])
    goal_angle = np.arctan2(goal_vector[1], goal_vector[0])

    population = []
    for _ in range(population_size):
        angle = goal_angle + np.random.uniform(-np.pi/4, np.pi/4)
        speed = np.random.uniform(0.5 * MAX_SPEED, MAX_SPEED)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        traj = predict_trajectory(state, vx, vy)
        population.append((vx, vy, traj))

    intermediate_goals = []

    for _ in range(generations):
        population = select_and_breed(
            state, population, goal, goal_heading,
            num_selected=5, mutation_rate=0.1
        )
        best = min(
            population,
            key=lambda x: evaluate_trajectory(x[2], goal, goal_heading)
        )
        # store endpoint of best trajectory for this generation
        intermediate_goals.append(best[2][-1])

    # Final best trajectory
    best = min(
        population,
        key=lambda x: evaluate_trajectory(x[2], goal, goal_heading)
    )
    return best[2], intermediate_goals


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Parameters
    dt = 0.1
    visual_dt = 0.5  # visual exaggeration of target motion from t=-1 to t=0
    follow_distance = 2.0
    relative_angle_rad = np.deg2rad(-45)

    # Target motion
    target_vel = np.array([0.2, 0.0])
    target_pos_t0 = np.array([2.0, 0.0])  # t = 0
    target_pos_tm1 = target_pos_t0 - visual_dt * target_vel  # t = -1

    # Target heading
    target_heading = np.arctan2(target_vel[1], target_vel[0])

    # Project target forward in time (simulate "lookahead")
    target_lookahead = target_pos_t0 + 1.5 * follow_distance * target_vel

    # Compute robot goal at -45° from projected target
    offset_dx = follow_distance * np.cos(target_heading + relative_angle_rad)
    offset_dy = follow_distance * np.sin(target_heading + relative_angle_rad)
    robot_goal = target_lookahead + np.array([offset_dx, offset_dy])

    # Set robot at proper offset from current target (t = 0)
    robot_pos_t0 = target_pos_t0 + np.array([offset_dx, offset_dy])
    robot_pos_tm1 = target_pos_tm1 + np.array([offset_dx, offset_dy])

    # Predict robot trajectory from current state
    robot_state = [robot_pos_t0[0], robot_pos_t0[1], 0.0]
    predicted_traj, _ = hybrid_dwa_ga_predict_full_trajectory(
        robot_state, robot_goal, target_heading
    )

    traj_x = [p[0] for p in predicted_traj]
    traj_y = [p[1] for p in predicted_traj]

    # Plotting
    plt.figure(figsize=(8, 6))

    # Target positions
    plt.plot(*target_pos_tm1, 'o', color='gray', label='Target t=-1')
    plt.plot(*target_pos_t0, 'r*', markersize=12, label='Target t=0')
    plt.arrow(target_pos_tm1[0], target_pos_tm1[1],
              target_pos_t0[0] - target_pos_tm1[0],
              target_pos_t0[1] - target_pos_tm1[1],
              head_width=0.1, color='red', alpha=0.6)

    # Robot positions
    plt.plot(*robot_pos_tm1, 'o', color='gray', label='Robot t=-1')
    plt.plot(*robot_pos_t0, 'bo', label='Robot t=0')
    plt.arrow(robot_pos_t0[0], robot_pos_t0[1],
              traj_x[1] - traj_x[0], traj_y[1] - traj_y[0],
              head_width=0.1, color='blue', alpha=0.6)

    # Predicted trajectory
    plt.plot(traj_x, traj_y, 'b--o', alpha=0.7, label='Predicted Trajectory')

    # Visual markers
    plt.plot(robot_goal[0], robot_goal[1], 'c*', markersize=10, label='Projected Robot Goal')
    plt.plot(target_lookahead[0], target_lookahead[1], 'r^', markersize=8, label='Projected Target')

    plt.title('Hybrid DWA-GA Snapshot: t=-1 and t=0 with Prediction')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


