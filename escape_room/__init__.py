"""
Escape Room environment for mjlab + mjwarp + rsl-rl.

Multi-agent cooperative puzzle environment with procedural rooms,
buttons/doors, grab mechanics, and progress-based rewards.
"""

from escape_room.consts import (
    NUM_ROOMS,
    NUM_AGENTS,
    MAX_ENTITIES_PER_ROOM,
    EPISODE_LEN,
    DELTA_T,
    NUM_PHYSICS_SUBSTEPS,
    TOTAL_ACTION_DIM,
)

__all__ = [
    "NUM_ROOMS",
    "NUM_AGENTS",
    "MAX_ENTITIES_PER_ROOM",
    "EPISODE_LEN",
    "DELTA_T",
    "NUM_PHYSICS_SUBSTEPS",
    "TOTAL_ACTION_DIM",
]
