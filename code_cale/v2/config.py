import numpy as np

# === Robot Dynamics ===
MAX_SPEED = 6.0                 # Max linear speed [m/s]
MIN_SPEED = -1.0                # Allow reversing
MAX_ACCEL = 2.0                 # Faster acceleration
MAX_YAW_RATE = np.deg2rad(90)   # Fast rotation
MAX_YAW_ACCEL = np.deg2rad(90)  # Rapid angular acceleration
DT = 0.1                        # Time step for motion updates

# === Prediction Horizon ===
PREDICT_TIME = 1.5              # How far to simulate ahead [s]

# === Dynamic Speed Scaling ===
BASE_MAX_SPEED = 0.5            # Minimal forward drive when close
MAX_SPEED_CAP = 10.0            # Upper limit of scaling
SPEED_SCALE_GAIN = 1.5          # Faster ramp-up based on distance

# === DWA Sampling Resolution ===
VELOCITY_RESOLUTION = 0.05      # Granularity of velocity samples
YAWRATE_RESOLUTION = np.deg2rad(1.0)

# === Cost Function Weights ===
POSITION_ERROR_COST_GAIN = 50.0     # High priority: catch up
RELATIVE_ANGLE_COST_GAIN = 0.5      # Moderate angular alignment
SPEED_COST_GAIN = 25.0              # Prefer faster speeds (used negatively in scoring)

# === Arc Path Geometry ===
ARC_CENTER = [5.0, 5.0]
ARC_RADIUS = 10.0
ARC_TOTAL_STEPS = 1000
ARC_START_ANGLE = 0.0
ARC_END_ANGLE = np.pi

# === Goal Offset Control ===
GOAL_OFFSET_DIST = 2.0
GOAL_OFFSET_ANGLE = np.deg2rad(90)  # Default angle

# === Lookahead Adjustment ===
LOOKAHEAD_DISTANCE_GAIN = 4.0
LOOKAHEAD_ANGLE_GAIN = 3.0
LOOKAHEAD_MIN = 0
LOOKAHEAD_MAX = 12

# === Angular Reference ===
DESIRED_RELATIVE_ANGLE = 0.0  # Robot should be offset but aligned in heading
