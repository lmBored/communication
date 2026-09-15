"""
Action terms for the escape room environment.

Maps policy actions to EscapeRoomState methods.
"""

import torch
from typing import Tuple

from escape_room.consts import (
    NUM_AGENTS,
    ACTION_DIM_PER_AGENT,
    TOTAL_ACTION_DIM,
)


def action_term(
    actions: torch.Tensor,
    game_state,
    mujoco_model,
    mujoco_data,
) -> None:
    """
    Apply actions to the game state.

    Args:
        actions: [num_envs, total_action_dim] - flattened policy actions
        game_state: EscapeRoomState instance
        mujoco_model: MuJoCo model
        mujoco_data: MuJoCo data
    """
    # Reshape to [num_envs, num_agents, action_dim]
    num_envs = actions.shape[0]
    actions = actions.view(num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)

    game_state.apply_movement(actions, mujoco_model, mujoco_data)
    game_state.apply_grab(actions, mujoco_model, mujoco_data)


def get_action_dim() -> int:
    """Get total action dimension."""
    return TOTAL_ACTION_DIM


def get_action_bounds() -> Tuple[torch.Tensor, torch.Tensor]:
    """Get action bounds for normalization."""
    low = torch.full((TOTAL_ACTION_DIM,), -1.0)
    high = torch.full((TOTAL_ACTION_DIM,), 1.0)
    return low, high
