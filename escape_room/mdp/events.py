"""
Reset event terms for the escape room environment.

Handles episode reset and level reconfiguration.
"""

import numpy as np


def reset_event(
    game_state,
    env_ids: list,
    seed: int = None,
) -> None:
    """
    Reset specified environments and regenerate levels.

    Args:
        game_state: EscapeRoomState instance
        env_ids: List of environment indices to reset
        seed: Optional seed for reproducibility
    """
    for env_idx in env_ids:
        # Reset step counters
        game_state.steps_remaining[env_idx, :] = 200  # EPISODE_LEN
        game_state.progress_max_y[env_idx, :] = 0.0
        game_state.rewards[env_idx, :] = 0.0
        game_state.dones[env_idx, :] = False

        # Reset doors
        game_state.door_open[env_idx, :] = False
        game_state.door_z[env_idx, :] = 0.0  # DOOR_CLOSED_Z

        # Reset buttons
        game_state.button_pressed[env_idx, :, :] = False

        # Reset grab
        game_state.grab_target[env_idx, :] = -1
        game_state.is_grabbing[env_idx, :] = False

    # If resetting all environments, regenerate levels
    if len(env_ids) == game_state.num_envs:
        if seed is not None:
            game_state.rng = np.random.default_rng(seed)
        game_state._generate_levels()


def full_reset(
    game_state,
    seed: int = None,
) -> None:
    """Reset all environments."""
    env_ids = list(range(game_state.num_envs))
    reset_event(game_state, env_ids, seed)
