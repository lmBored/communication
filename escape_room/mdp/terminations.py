"""
Termination terms for the escape room environment.

Implements timeout-based termination matching the original stepTrackerSystem.
"""

import torch

from escape_room.consts import NUM_AGENTS


def compute_termination(
    game_state,
) -> torch.Tensor:
    """
    Compute termination flags for all agents.

    Args:
        game_state: EscapeRoomState instance

    Returns:
        dones: [num_envs, num_agents] - boolean tensor
    """
    return game_state.dones.clone()


def compute_timeouts(
    game_state,
) -> torch.Tensor:
    """
    Compute timeout flags (separate from dones for rsl-rl).

    Returns:
        timeouts: [num_envs, num_agents] - boolean tensor
    """
    return torch.zeros_like(game_state.dones, dtype=torch.bool)
