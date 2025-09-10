# === REWRITTEN robot_model.py FOR OMNIDIRECTIONAL ROBOT ===

import numpy as np

# State: [x, y, theta] (theta will be retained but unused)
def motion(state, vx, vy, dt):
    x, y, theta = state
    x += vx * dt
    y += vy * dt
    return [x, y, theta]
