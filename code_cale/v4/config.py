"""Configuration parameters for v4 hybrid DWA + GA omnidirectional follower.
All tunable constants live here to keep algorithm code clean.
"""
from math import pi

# --- Simulation Timing ---
DT: float = 0.07                 # [s] integration time step
PREDICT_TIME: float = DT * 10    # horizon for trajectory rollout
TOTAL_STEPS: int = 150           # number of outer simulation steps
TARGET_STEP_SIZE: float = 0.6    # fraction of a discrete path step advanced per simulation tick (<=1 slows target)

# --- Robot Physical / Limits (Mecanum holonomic) ---
MAX_SPEED: float = 5.0           # [m/s] magnitude of planar velocity
MAX_ACCEL: float = 2.0           # [m/s^2] (not yet enforced strictly)
MAX_YAW_RATE: float = pi / 2     # [rad/s] orientation tracking slew cap

# --- GA / Sampling Parameters ---
VEL_RES: float = 0.3             # sampling resolution for candidate velocities
POPULATION_SIZE: int = 60        # GA population
GENERATIONS: int = 5             # GA refinement iterations
PARENTS: int = 6                 # number of elites kept each generation
MUTATION_STD: float = 0.4        # Gaussian noise applied to vx, vy
SMOOTHING_ALPHA: float = 0.75    # exponential smoothing of chosen command

# --- Cost Weights ---
W_GOAL_DIST: float = 0.6
W_OFFSET_DIST: float = 1.2       # encourage staying near offset point (now dominant)
W_PATH_LENGTH: float = 0.15
W_FACING: float = 1.0            # heading alignment weight
W_CURVATURE: float = 0.4
W_JERK: float = 0.3

CURVATURE_DEADZONE: float = 0.02  # rad threshold
JERK_DEADZONE: float = 0.05       # m/s per step threshold

# --- Target Path Specification ---
# We define a composite path (two arcs + lines) similar to v3.
SEGMENT_RATIOS = (0.35, 0.15, 0.35, 0.15)  # must sum <= 1.0; last segment auto-fills remainder
ARC_RADIUS: float = 5.0

# Base reference scaling (used to shape path)
REF_X0: float = 6.0
REF_Y0: float = 9.0

# --- Follower Offset & Facing (LOCAL TARGET FRAME) ---
# Target local frame: +X = target forward direction (tangent), +Y = left normal.
# Robot desired offset expressed in this frame: (forward, left)
OFFSET_LOCAL_X: float = 2.5   # meters forward of target
OFFSET_LOCAL_Y: float = 1.5   # meters left of target

# Robot desired facing = target heading + FACING_OFFSET_DEG (anticlockwise)
FACING_OFFSET_DEG: float = 225.0

# --- Visualization ---
ARROW_LEN_FACING: float = 0.9
ARROW_LEN_VELOCITY: float = 0.9
ARROW_LEN_TARGET: float = 0.9
PLOT_XLIM = (-6, 18)
PLOT_YLIM = (-6, 18)
SAVE_GIF: bool = True
GIF_NAME: str = "tracking_animation_v4.gif"
OUTPUT_DIR: str = "code_cale/output"
GIF_DPI: int = 160
GIF_FRAME_PAUSE: float = 0.001
GIF_FRAME_MS: int = 50

# --- Random Seed for Repro ---
SEED: int | None = 42
