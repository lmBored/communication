"""
Observation terms for the escape room environment.

Mirrors the observation groups from scripts/policy.py:
self, partner, room_entities, door, lidar, steps_remaining, agent_id.
"""

import math
import torch

from escape_room.consts import (
    NUM_AGENTS,
    MAX_ENTITIES_PER_ROOM,
    NUM_LIDAR_SAMPLES,
    EntityType,
)


def compute_obs_dim() -> int:
    """Compute total observation dimension."""
    # Self: 8 (roomX, roomY, globalX, globalY, globalZ, maxY, theta, isGrabbing)
    self_dim = 8
    # Partner: 3 (polar r, theta, isGrabbing) * (NUM_AGENTS - 1)
    partner_dim = 3 * (NUM_AGENTS - 1)
    # Room entities: 3 (polar r, theta, type) * MAX_ENTITIES_PER_ROOM
    room_ent_dim = 3 * MAX_ENTITIES_PER_ROOM
    # Door: 3 (polar r, theta, isOpen)
    door_dim = 3
    # Lidar: 2 (depth, type) * NUM_LIDAR_SAMPLES
    lidar_dim = 2 * NUM_LIDAR_SAMPLES
    # Steps remaining: 1
    steps_dim = 1
    # Agent ID: 1
    agent_id_dim = 1

    return self_dim + partner_dim + room_ent_dim + door_dim + lidar_dim + steps_dim + agent_id_dim


def concatenate_observations(
    self_obs: torch.Tensor,
    partner_obs: torch.Tensor,
    room_ent_obs: torch.Tensor,
    door_obs: torch.Tensor,
    lidar: torch.Tensor,
    steps_remaining: torch.Tensor,
    agent_id: torch.Tensor,
) -> torch.Tensor:
    """
    Concatenate observation groups into a single tensor for the policy.

    Args:
        All tensors have shape [num_envs, num_agents, ...]

    Returns:
        Flattened observations [num_envs * num_agents, obs_dim]
    """
    batch_size = self_obs.shape[0] * self_obs.shape[1]

    obs_list = [
        self_obs.view(batch_size, -1),
        partner_obs.view(batch_size, -1),
        room_ent_obs.view(batch_size, -1),
        door_obs.view(batch_size, -1),
        lidar.view(batch_size, -1),
        steps_remaining.view(batch_size, -1).float() / 200.0,
        agent_id.view(batch_size, -1),
    ]

    return torch.cat(obs_list, dim=1)


def encode_type(entity_type: int) -> float:
    """Encode entity type as a float in [0, 1)."""
    return entity_type / EntityType.NUM_TYPES
