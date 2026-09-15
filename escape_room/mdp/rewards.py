"""
Reward terms for the escape room environment.

Implements progress-based rewards with slack penalty and partner bonus,
matching the original rewardSystem and bonusRewardSystem from sim.cpp.
"""

import torch

from escape_room.consts import (
    NUM_AGENTS,
    REWARD_PER_DIST,
    SLACK_REWARD,
    PARTNER_BONUS_MULT,
    PARTNER_CLOSE_THRESHOLD,
    WORLD_LENGTH,
)


def compute_reward(
    game_state,
    agent_positions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rewards for all agents.

    Args:
        game_state: EscapeRoomState instance
        agent_positions: [num_envs, num_agents, 3] - agent positions

    Returns:
        rewards: [num_envs, num_agents]
    """
    num_envs = agent_positions.shape[0]
    rewards = torch.zeros(
        (num_envs, NUM_AGENTS), device=game_state.device, dtype=torch.float32
    )

    # Clamp Y position to prevent crazy rewards
    agent_y = torch.clamp(agent_positions[:, :, 1], max=WORLD_LENGTH * 2)

    for agent_idx in range(NUM_AGENTS):
        old_max_y = game_state.progress_max_y[:, agent_idx]
        new_y = agent_y[:, agent_idx]

        # Progress delta
        progress_delta = new_y - old_max_y

        # Slack reward (default)
        rewards[:, agent_idx] = SLACK_REWARD

        # Progress reward (if agent moved forward)
        mask = progress_delta > 0
        rewards[mask, agent_idx] = progress_delta[mask] * REWARD_PER_DIST
        game_state.progress_max_y[mask, agent_idx] = new_y[mask, agent_idx]

        # Partner bonus
        for env_idx in range(num_envs):
            if rewards[env_idx, agent_idx] > 0:
                other_idx = 1 - agent_idx
                y_diff = abs(
                    game_state.progress_max_y[env_idx, agent_idx]
                    - game_state.progress_max_y[env_idx, other_idx]
                )
                if y_diff <= PARTNER_CLOSE_THRESHOLD:
                    rewards[env_idx, agent_idx] *= PARTNER_BONUS_MULT

    return rewards
