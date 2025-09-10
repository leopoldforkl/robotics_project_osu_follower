import numpy as np

# === Robot Dynamics ===
MAX_SPEED = 5.0                 # Max linear speed [m/s]
MIN_SPEED = -5.0                # Allow reversing
MAX_ACCEL = 2.0                 # Faster acceleration
MAX_YAW_RATE = np.deg2rad(90)   # Fast rotation
MAX_YAW_ACCEL = np.deg2rad(90)  # Rapid angular acceleration
DT = 0.07                        # Time step for motion updates

# === Prediction Horizon ===
#PREDICT_TIME = 2.5              # How far to simulate ahead [s]
PREDICT_TIME = DT*10
# === Dynamic Speed Scaling ===
BASE_MAX_SPEED = 0.5            # Minimal forward drive when close
MAX_SPEED_CAP = 10.0            # Upper limit of scaling
SPEED_SCALE_GAIN = 1.5          # Faster ramp-up based on distance

# === DWA Sampling Resolution ===
VELOCITY_RESOLUTION = 0.1      # Granularity of velocity samples
YAWRATE_RESOLUTION = np.deg2rad(1.0)

# === Cost Function Weights ===
POSITION_ERROR_COST_GAIN = 0.75     # High priority: catch up
RELATIVE_ANGLE_COST_GAIN = 2.0      # Moderate angular alignment
RELATIVE_ANGLE_GOAL_RAD = np.pi / 2
SPEED_COST_GAIN = 2.0              # Prefer faster speeds (used negatively in scoring)

# === Arc Path Geometry ===
ARC_CENTER = [5.0, 5.0]
ARC_RADIUS = 10.0
ARC_TOTAL_STEPS = 500
ARC_START_ANGLE = 0.0
ARC_END_ANGLE = np.pi

# === Goal Offset Control ===
GOAL_OFFSET_DIST = 1.0
GOAL_OFFSET_ANGLE = np.deg2rad(90)  # Default angle

# === Lookahead Adjustment ===
LOOKAHEAD_DISTANCE_GAIN = 4.0
LOOKAHEAD_ANGLE_GAIN = 3.0
LOOKAHEAD_MIN = 2
LOOKAHEAD_MAX = 12

TRAJECTORY_PENALTY_GAIN = 0.2 
DIRECTIONAL_PENALTY_GAIN = 2.0 

# === Angular Reference ===
DESIRED_RELATIVE_ANGLE = 0.0  # Robot should be offset but aligned in heading

CURVATURE_PENALTY_GAIN = 3.0  # Start small and increase if needed

JERK_PENALTY_GAIN = 2.0

# Smoothing Factor
# Higher means smoother movements, but lower means more reactive
ALPHA = 0.8

# Deadzone Variables
POSITION_DEADZONE_RADIUS = 0.0           # meters
ANGLE_DEADZONE_RAD = np.deg2rad(10)      # radians
CURVATURE_DEADZONE_RAD = np.deg2rad(2)   # small heading jitter
JERK_DEADZONE = 0.05                     # m/s² threshold for delta velocity

