"""
Constants ported from src/consts.hpp for the escape room environment.

These values define the scene caps, physics parameters, observation layout,
and reward scales that must match the original Madrona environment for
behavioral parity.
"""

# Scene caps (parity with consts.hpp)
NUM_ROOMS: int = 3
NUM_AGENTS: int = 2
MAX_ENTITIES_PER_ROOM: int = 6

# World / entity size parameters
WORLD_LENGTH: float = 40.0  # Y-axis length
WORLD_WIDTH: float = 20.0   # X-axis width
WALL_WIDTH: float = 1.0
BUTTON_WIDTH: float = 1.3
AGENT_RADIUS: float = 1.0
ROOM_LENGTH: float = WORLD_LENGTH / NUM_ROOMS

# Door dimensions (from level_gen.cpp)
DOOR_WIDTH: float = WORLD_WIDTH / 3.0

# Episode length in control steps
EPISODE_LEN: int = 200

# Control timestep (seconds)
DELTA_T: float = 0.04

# Physics substeps per control step
NUM_PHYSICS_SUBSTEPS: int = 4

# Lidar
NUM_LIDAR_SAMPLES: int = 30
LIDAR_MAX_RANGE: float = 200.0  # Original uses 200.f as max ray length

# Door animation speed (units per second)
DOOR_SPEED: float = 30.0
DOOR_CLOSED_Z: float = 0.0
DOOR_OPEN_Z: float = -4.5

# Rewards
REWARD_PER_DIST: float = 0.05    # Progress along +Y
SLACK_REWARD: float = -0.005     # Penalty for no progress
PARTNER_BONUS_MULT: float = 1.25 # Multiplier when partners are close
PARTNER_CLOSE_THRESHOLD: float = 2.0  # Max distance for partner bonus

# Movement (continuous action scaling, replaces discrete buckets)
MOVE_MAX_FORCE: float = 1000.0   # Max linear force magnitude
TURN_MAX_TORQUE: float = 320.0   # Max yaw torque

# Grab
GRAB_RAY_LENGTH: float = 2.0     # Max distance for grab raycast
GRAB_OFFSET_FWD: float = 1.25    # Forward offset from agent center for grab anchor
GRAB_OFFSET_UP: float = 0.5      # Up offset from agent center for grab anchor

# Entity types (matching EntityType enum from types.hpp)
class EntityType:
    NONE = 0
    BUTTON = 1
    CUBE = 2
    WALL = 3
    AGENT = 4
    DOOR = 5
    NUM_TYPES = 6

# Room types (matching RoomType enum from level_gen.cpp)
class RoomType:
    SINGLE_BUTTON = 0
    DOUBLE_BUTTON = 1
    CUBE_BLOCKING = 2
    CUBE_BUTTONS = 3
    NUM_TYPES = 4

# Observation normalization
def dist_obs(v: float) -> float:
    """Normalize distance by world length."""
    return v / WORLD_LENGTH

def angle_obs(v: float) -> float:
    """Normalize angle by pi."""
    import math
    return v / math.pi

# Action dimensions per agent (continuous)
# (move_x, move_y, yaw_rate, grab)
ACTION_DIM_PER_AGENT: int = 4
TOTAL_ACTION_DIM: int = NUM_AGENTS * ACTION_DIM_PER_AGENT

# Agent spawn positions (relative to room 0)
AGENT_SPAWN_Y_MIN: float = AGENT_RADIUS * 1.1
AGENT_SPAWN_Y_MAX: float = 2.0
AGENT_SPAWN_X_SPREAD: float = WORLD_WIDTH / 4.0
