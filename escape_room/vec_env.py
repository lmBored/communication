"""
High-throughput vectorized escape-room environment.

Provides a fully batched GPU step path over EscapeRoomState for short
training smoke tests and SPS measurement. Physics is kinematic (force-
scaled velocity integration) so large num_envs stay GPU-bound without a
full mjwarp graph; game logic (progress reward, doors/buttons, timeout)
matches the intended MDP interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from escape_room.consts import (
    ACTION_DIM_PER_AGENT,
    AGENT_RADIUS,
    AGENT_SPAWN_X_SPREAD,
    AGENT_SPAWN_Y_MAX,
    AGENT_SPAWN_Y_MIN,
    DOOR_CLOSED_Z,
    DOOR_OPEN_Z,
    DOOR_SPEED,
    DOOR_WIDTH,
    DELTA_T,
    EPISODE_LEN,
    MAX_ENTITIES_PER_ROOM,
    MOVE_MAX_FORCE,
    NUM_AGENTS,
    NUM_LIDAR_SAMPLES,
    NUM_ROOMS,
    PARTNER_BONUS_MULT,
    PARTNER_CLOSE_THRESHOLD,
    REWARD_PER_DIST,
    ROOM_LENGTH,
    SLACK_REWARD,
    TOTAL_ACTION_DIM,
    TURN_MAX_TORQUE,
    WORLD_LENGTH,
    WORLD_WIDTH,
    EntityType,
)
from escape_room.env_cfg import OBS_DIM
from escape_room.level_gen import generate_level
from escape_room.mdp.observations import concatenate_observations


@dataclass
class StepResult:
    obs: Tensor
    rewards: Tensor
    dones: Tensor
    timeouts: Tensor
    infos: dict


class EscapeRoomVecEnv:
    """Batched multi-agent escape room env (shared-weight N*A policy rows)."""

    def __init__(
        self,
        num_envs: int,
        device: str = "cuda:0",
        seed: int = 42,
        auto_reset: bool = True,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.seed = seed
        self.auto_reset = auto_reset
        self.num_agents = NUM_AGENTS
        self.action_dim = TOTAL_ACTION_DIM
        self.obs_dim = OBS_DIM
        self.episode_length_buf = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )

        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(seed)

        # Agent state
        self.agent_pos = torch.zeros(
            (num_envs, NUM_AGENTS, 3), device=self.device, dtype=torch.float32
        )
        self.agent_yaw = torch.zeros(
            (num_envs, NUM_AGENTS), device=self.device, dtype=torch.float32
        )
        self.agent_vel = torch.zeros(
            (num_envs, NUM_AGENTS, 2), device=self.device, dtype=torch.float32
        )
        self.is_grabbing = torch.zeros(
            (num_envs, NUM_AGENTS), device=self.device, dtype=torch.bool
        )
        self.progress_max_y = torch.zeros(
            (num_envs, NUM_AGENTS), device=self.device, dtype=torch.float32
        )
        self.steps_remaining = torch.full(
            (num_envs, NUM_AGENTS),
            EPISODE_LEN,
            device=self.device,
            dtype=torch.long,
        )

        # Room / door / entity buffers
        self.door_open = torch.zeros(
            (num_envs, NUM_ROOMS), device=self.device, dtype=torch.bool
        )
        self.door_persistent = torch.zeros(
            (num_envs, NUM_ROOMS), device=self.device, dtype=torch.bool
        )
        self.door_z = torch.full(
            (num_envs, NUM_ROOMS),
            DOOR_CLOSED_Z,
            device=self.device,
            dtype=torch.float32,
        )
        self.door_y = torch.zeros(
            (num_envs, NUM_ROOMS), device=self.device, dtype=torch.float32
        )
        self.entity_active = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=self.device,
            dtype=torch.bool,
        )
        self.entity_type = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=self.device,
            dtype=torch.long,
        )
        self.entity_pos = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM, 2),
            device=self.device,
            dtype=torch.float32,
        )
        self.button_pressed = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=self.device,
            dtype=torch.bool,
        )
        self.door_button_mask = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=self.device,
            dtype=torch.bool,
        )

        # Move scale: action in [-1,1] -> velocity (m/s)
        self._max_speed = 8.0
        self._max_yaw_rate = 4.0

        self.reset()

    @property
    def num_policy_rows(self) -> int:
        return self.num_envs * NUM_AGENTS

    def reset(self, env_ids: Optional[Tensor] = None) -> Tensor:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(self.device).long()

        n = env_ids.numel()
        if n == 0:
            return self.get_obs()

        # Spawn agents in room 0 (device RNG for speed)
        spawn_x = (
            torch.rand(n, NUM_AGENTS, device=self.device) * 2.0 - 1.0
        ) * AGENT_SPAWN_X_SPREAD
        spawn_y = (
            torch.rand(n, NUM_AGENTS, device=self.device)
            * (AGENT_SPAWN_Y_MAX - AGENT_SPAWN_Y_MIN)
            + AGENT_SPAWN_Y_MIN
        )
        self.agent_pos[env_ids, :, 0] = spawn_x
        self.agent_pos[env_ids, :, 1] = spawn_y
        self.agent_pos[env_ids, :, 2] = 0.5
        self.agent_yaw[env_ids] = 0.0
        self.agent_vel[env_ids] = 0.0
        self.is_grabbing[env_ids] = False
        self.progress_max_y[env_ids] = spawn_y
        self.steps_remaining[env_ids] = EPISODE_LEN
        self.episode_length_buf[env_ids] = 0
        self.door_open[env_ids] = False
        self.door_z[env_ids] = DOOR_CLOSED_Z
        self.button_pressed[env_ids] = False

        self._configure_levels(env_ids)
        return self.get_obs()

    def _ensure_level_pool(self) -> None:
        """Bake a small layout pool once as GPU tensors for O(1) scatter."""
        if hasattr(self, "_pool_entity_pos"):
            return
        import numpy as np

        n_pool = 32
        entity_pos = torch.zeros(
            (n_pool, NUM_ROOMS, MAX_ENTITIES_PER_ROOM, 2), dtype=torch.float32
        )
        entity_type = torch.zeros(
            (n_pool, NUM_ROOMS, MAX_ENTITIES_PER_ROOM), dtype=torch.long
        )
        entity_active = torch.zeros(
            (n_pool, NUM_ROOMS, MAX_ENTITIES_PER_ROOM), dtype=torch.bool
        )
        door_persistent = torch.zeros((n_pool, NUM_ROOMS), dtype=torch.bool)
        door_button_mask = torch.zeros(
            (n_pool, NUM_ROOMS, MAX_ENTITIES_PER_ROOM), dtype=torch.bool
        )
        door_y = torch.zeros((n_pool, NUM_ROOMS), dtype=torch.float32)

        for i in range(n_pool):
            rooms = generate_level(np.random.default_rng(10_000 + i))
            for room_idx, room in enumerate(rooms):
                door_persistent[i, room_idx] = bool(room.is_persistent)
                door_y[i, room_idx] = (room_idx + 1) * ROOM_LENGTH
                for slot_idx, ent in enumerate(room.entities):
                    entity_active[i, room_idx, slot_idx] = bool(ent.active)
                    entity_type[i, room_idx, slot_idx] = int(ent.type)
                    entity_pos[i, room_idx, slot_idx, 0] = float(ent.pos_x)
                    entity_pos[i, room_idx, slot_idx, 1] = float(ent.pos_y)
                for slot_idx in room.button_indices:
                    door_button_mask[i, room_idx, slot_idx] = True

        self._pool_entity_pos = entity_pos.to(self.device)
        self._pool_entity_type = entity_type.to(self.device)
        self._pool_entity_active = entity_active.to(self.device)
        self._pool_door_persistent = door_persistent.to(self.device)
        self._pool_door_button_mask = door_button_mask.to(self.device)
        self._pool_door_y = door_y.to(self.device)
        self._n_pool = n_pool

    def _configure_levels(self, env_ids: Tensor) -> None:
        """Scatter pre-baked layouts onto env slots (fully vectorized)."""
        self._ensure_level_pool()
        n = env_ids.numel()
        # CPU generator → indices, then move to device
        picks = torch.randint(
            0, self._n_pool, (n,), generator=self._generator
        ).to(self.device)

        self.entity_pos[env_ids] = self._pool_entity_pos[picks]
        self.entity_type[env_ids] = self._pool_entity_type[picks]
        self.entity_active[env_ids] = self._pool_entity_active[picks]
        self.door_persistent[env_ids] = self._pool_door_persistent[picks]
        self.door_button_mask[env_ids] = self._pool_door_button_mask[picks]
        self.door_y[env_ids] = self._pool_door_y[picks]

    def step(self, actions: Tensor) -> StepResult:
        """
        Step all envs.

        actions: [num_envs, TOTAL_ACTION_DIM] or [num_envs * NUM_AGENTS, ACTION_DIM_PER_AGENT]
        returns policy-row rewards/dones of shape [num_envs * NUM_AGENTS]
        """
        actions = self._normalize_actions(actions)
        # actions: [N, A, 4] -> move_x, move_y, yaw_rate, grab

        self._apply_movement(actions)
        self._update_buttons()
        self._update_doors()
        self._animate_doors()
        rewards = self._compute_rewards()
        self._update_steps()

        timeouts = self.steps_remaining[:, 0] <= 0
        dones = timeouts.clone()

        # Auto-reset finished envs
        if self.auto_reset and timeouts.any():
            done_ids = timeouts.nonzero(as_tuple=False).squeeze(-1)
            self.reset(done_ids)

        obs = self.get_obs()
        # Flatten to policy rows
        rew_flat = rewards.reshape(self.num_policy_rows)
        done_flat = dones.unsqueeze(1).expand(-1, NUM_AGENTS).reshape(
            self.num_policy_rows
        )
        timeout_flat = timeouts.unsqueeze(1).expand(-1, NUM_AGENTS).reshape(
            self.num_policy_rows
        )

        return StepResult(
            obs=obs,
            rewards=rew_flat,
            dones=done_flat,
            timeouts=timeout_flat,
            infos={"env_timeouts": timeouts},
        )

    def _normalize_actions(self, actions: Tensor) -> Tensor:
        if actions.ndim != 2:
            raise ValueError(f"Expected 2D actions, got {tuple(actions.shape)}")
        if actions.shape[0] == self.num_envs and actions.shape[1] == TOTAL_ACTION_DIM:
            return actions.view(self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)
        if (
            actions.shape[0] == self.num_policy_rows
            and actions.shape[1] == ACTION_DIM_PER_AGENT
        ):
            return actions.view(self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)
        raise ValueError(
            f"Bad action shape {tuple(actions.shape)}; "
            f"expected [{self.num_envs}, {TOTAL_ACTION_DIM}] or "
            f"[{self.num_policy_rows}, {ACTION_DIM_PER_AGENT}]"
        )

    def _apply_movement(self, actions: Tensor) -> None:
        move = actions[..., 0:2].clamp(-1.0, 1.0)
        yaw_cmd = actions[..., 2].clamp(-1.0, 1.0)
        grab = actions[..., 3] > 0.0

        # Local frame velocity -> world
        cos_t = torch.cos(self.agent_yaw)
        sin_t = torch.sin(self.agent_yaw)
        vx_local = move[..., 0] * self._max_speed
        vy_local = move[..., 1] * self._max_speed
        vx = vx_local * cos_t - vy_local * sin_t
        vy = vx_local * sin_t + vy_local * cos_t

        self.agent_vel[..., 0] = vx
        self.agent_vel[..., 1] = vy
        self.agent_yaw = (
            self.agent_yaw + yaw_cmd * self._max_yaw_rate * DELTA_T
        )
        # wrap yaw
        self.agent_yaw = torch.atan2(
            torch.sin(self.agent_yaw), torch.cos(self.agent_yaw)
        )

        new_pos = self.agent_pos[..., 0:2] + self.agent_vel * DELTA_T
        # Clamp to world bounds (inside walls)
        half_w = WORLD_WIDTH * 0.5 - AGENT_RADIUS
        new_pos[..., 0] = new_pos[..., 0].clamp(-half_w, half_w)
        new_pos[..., 1] = new_pos[..., 1].clamp(
            AGENT_RADIUS, WORLD_LENGTH - AGENT_RADIUS
        )

        # Block closed doors (simple slab at door_y)
        for room_idx in range(NUM_ROOMS):
            door_y = self.door_y[:, room_idx].unsqueeze(1)  # [N,1]
            closed = (~self.door_open[:, room_idx]).unsqueeze(1)
            # Agents crossing door plane while closed get pushed back
            crossing = (
                (self.agent_pos[..., 1] < door_y)
                & (new_pos[..., 1] >= door_y - 0.5)
                & closed
            )
            # Only block if within door x-span approx center third; walls block sides
            in_door_x = new_pos[..., 0].abs() < (DOOR_WIDTH * 0.5)
            block = crossing & in_door_x
            new_pos[..., 1] = torch.where(
                block, door_y - AGENT_RADIUS - 0.1, new_pos[..., 1]
            )

        self.agent_pos[..., 0:2] = new_pos
        # Zero residual velocity (agentZeroVelSystem parity)
        self.agent_vel.zero_()

        # Grab toggle (edge-free: threshold held as latch flip once per press is simplified)
        self.is_grabbing = torch.where(
            grab, ~self.is_grabbing, self.is_grabbing
        )

    def _update_buttons(self) -> None:
        # Button pressed if any agent near button slot
        # entity_pos: [N,R,E,2], agent_pos: [N,A,3]
        agent_xy = self.agent_pos[..., 0:2]  # [N,A,2]
        # distances: [N,R,E,A]
        diff = (
            self.entity_pos.unsqueeze(3) - agent_xy.unsqueeze(1).unsqueeze(1)
        )
        dist = torch.linalg.norm(diff, dim=-1)
        near = dist < 1.25
        any_agent = near.any(dim=-1)  # [N,R,E]
        is_button = (self.entity_type == EntityType.BUTTON) & self.entity_active
        self.button_pressed = is_button & any_agent

    def _update_doors(self) -> None:
        # Door opens when all linked buttons pressed (or already persistent-open)
        needed = self.door_button_mask
        pressed = self.button_pressed & needed
        # rooms with no buttons stay closed unless already open+persistent
        num_needed = needed.sum(dim=-1)  # [N,R]
        num_pressed = pressed.sum(dim=-1)
        satisfied = (num_needed > 0) & (num_pressed >= num_needed)
        stay = self.door_open & self.door_persistent
        self.door_open = satisfied | stay

    def _animate_doors(self) -> None:
        target = torch.where(
            self.door_open,
            torch.full_like(self.door_z, DOOR_OPEN_Z),
            torch.full_like(self.door_z, DOOR_CLOSED_Z),
        )
        delta = DOOR_SPEED * DELTA_T
        self.door_z = torch.where(
            self.door_z < target,
            torch.minimum(self.door_z + delta, target),
            torch.maximum(self.door_z - delta, target),
        )

    def _compute_rewards(self) -> Tensor:
        y = self.agent_pos[..., 1]
        new_max = torch.maximum(self.progress_max_y, y)
        delta = new_max - self.progress_max_y
        self.progress_max_y = new_max

        rewards = delta * REWARD_PER_DIST + SLACK_REWARD

        # Partner bonus when both close and making progress
        if NUM_AGENTS >= 2:
            p0 = self.agent_pos[:, 0, 0:2]
            p1 = self.agent_pos[:, 1, 0:2]
            dist = torch.linalg.norm(p0 - p1, dim=-1)
            close = dist < PARTNER_CLOSE_THRESHOLD
            bonus = close.float() * (PARTNER_BONUS_MULT - 1.0)
            # apply bonus on positive progress
            prog = delta > 0
            rewards = rewards + (bonus.unsqueeze(1) * rewards * prog.float())

        return rewards

    def _update_steps(self) -> None:
        self.steps_remaining = self.steps_remaining - 1
        self.episode_length_buf += 1

    def get_obs(self) -> Tensor:
        """Return flattened observations [N*A, obs_dim]."""
        n = self.num_envs
        # Self obs: roomX, roomY, globalX, globalY, globalZ, maxY, theta, isGrabbing
        gx = self.agent_pos[..., 0]
        gy = self.agent_pos[..., 1]
        gz = self.agent_pos[..., 2]
        room_y = gy % ROOM_LENGTH
        room_x = gx
        self_obs = torch.stack(
            [
                room_x / WORLD_WIDTH,
                room_y / ROOM_LENGTH,
                gx / WORLD_WIDTH,
                gy / WORLD_LENGTH,
                gz,
                self.progress_max_y / WORLD_LENGTH,
                self.agent_yaw / math.pi,
                self.is_grabbing.float(),
            ],
            dim=-1,
        )  # [N,A,8]

        # Partner polar relative to each agent
        partner_obs = torch.zeros(
            (n, NUM_AGENTS, 3), device=self.device, dtype=torch.float32
        )
        if NUM_AGENTS >= 2:
            for a in range(NUM_AGENTS):
                other = 1 - a
                rel = self.agent_pos[:, other, 0:2] - self.agent_pos[:, a, 0:2]
                # rotate into agent frame
                c = torch.cos(-self.agent_yaw[:, a])
                s = torch.sin(-self.agent_yaw[:, a])
                lx = rel[:, 0] * c - rel[:, 1] * s
                ly = rel[:, 0] * s + rel[:, 1] * c
                r = torch.linalg.norm(rel, dim=-1) / WORLD_LENGTH
                theta = torch.atan2(ly, lx) / math.pi
                partner_obs[:, a, 0] = r
                partner_obs[:, a, 1] = theta
                partner_obs[:, a, 2] = self.is_grabbing[:, other].float()

        # Room entities relative to agent (use entities in agent's current room)
        room_idx = (gy / ROOM_LENGTH).long().clamp(0, NUM_ROOMS - 1)  # [N,A]
        room_ent_obs = torch.zeros(
            (n, NUM_AGENTS, MAX_ENTITIES_PER_ROOM, 3),
            device=self.device,
            dtype=torch.float32,
        )
        for a in range(NUM_AGENTS):
            ri = room_idx[:, a]  # [N]
            # gather entity pos for each env's room
            # entity_pos[env, room, slot, :]
            idx = ri.view(n, 1, 1, 1).expand(n, 1, MAX_ENTITIES_PER_ROOM, 2)
            ent_pos = torch.gather(
                self.entity_pos, 1, idx.expand(n, 1, MAX_ENTITIES_PER_ROOM, 2)
            ).squeeze(1)  # [N,E,2]
            ent_type = torch.gather(
                self.entity_type,
                1,
                ri.view(n, 1, 1).expand(n, 1, MAX_ENTITIES_PER_ROOM),
            ).squeeze(1)
            ent_active = torch.gather(
                self.entity_active,
                1,
                ri.view(n, 1, 1).expand(n, 1, MAX_ENTITIES_PER_ROOM),
            ).squeeze(1)
            rel = ent_pos - self.agent_pos[:, a, 0:2].unsqueeze(1)
            c = torch.cos(-self.agent_yaw[:, a]).unsqueeze(1)
            s = torch.sin(-self.agent_yaw[:, a]).unsqueeze(1)
            lx = rel[..., 0] * c - rel[..., 1] * s
            ly = rel[..., 0] * s + rel[..., 1] * c
            r = torch.linalg.norm(rel, dim=-1) / WORLD_LENGTH
            theta = torch.atan2(ly, lx) / math.pi
            type_enc = ent_type.float() / float(EntityType.NUM_TYPES)
            room_ent_obs[:, a, :, 0] = torch.where(ent_active, r, torch.zeros_like(r))
            room_ent_obs[:, a, :, 1] = torch.where(
                ent_active, theta, torch.zeros_like(theta)
            )
            room_ent_obs[:, a, :, 2] = torch.where(
                ent_active, type_enc, torch.zeros_like(type_enc)
            )

        # Door obs for current room
        door_obs = torch.zeros(
            (n, NUM_AGENTS, 3), device=self.device, dtype=torch.float32
        )
        for a in range(NUM_AGENTS):
            ri = room_idx[:, a]
            dy = self.door_y.gather(1, ri.view(n, 1)).squeeze(1)
            dopen = self.door_open.gather(1, ri.view(n, 1)).squeeze(1).float()
            rel = torch.stack(
                [
                    torch.zeros(n, device=self.device)
                    - self.agent_pos[:, a, 0],
                    dy - self.agent_pos[:, a, 1],
                ],
                dim=-1,
            )
            r = torch.linalg.norm(rel, dim=-1) / WORLD_LENGTH
            theta = torch.atan2(rel[:, 1], rel[:, 0]) / math.pi
            door_obs[:, a, 0] = r
            door_obs[:, a, 1] = theta
            door_obs[:, a, 2] = dopen

        # Cheap lidar approximation: depth to walls in polar samples
        lidar = torch.zeros(
            (n, NUM_AGENTS, NUM_LIDAR_SAMPLES, 2),
            device=self.device,
            dtype=torch.float32,
        )
        angles = torch.linspace(
            -math.pi,
            math.pi,
            NUM_LIDAR_SAMPLES,
            device=self.device,
            dtype=torch.float32,
        )
        for a in range(NUM_AGENTS):
            yaw = self.agent_yaw[:, a].unsqueeze(1) + angles.view(1, -1)
            # Ray-plane approx against x walls and y bounds
            dx = torch.cos(yaw)
            dy = torch.sin(yaw)
            # distance to x walls
            px = self.agent_pos[:, a, 0:1]
            py = self.agent_pos[:, a, 1:2]
            half_w = WORLD_WIDTH * 0.5
            tx_pos = (half_w - px) / dx.clamp(min=1e-4)
            tx_neg = (-half_w - px) / dx.clamp(max=-1e-4)
            ty_pos = (WORLD_LENGTH - py) / dy.clamp(min=1e-4)
            ty_neg = (0.0 - py) / dy.clamp(max=-1e-4)
            # only positive t
            cands = torch.stack(
                [
                    torch.where(dx > 0, tx_pos, torch.full_like(tx_pos, 1e6)),
                    torch.where(dx < 0, tx_neg, torch.full_like(tx_neg, 1e6)),
                    torch.where(dy > 0, ty_pos, torch.full_like(ty_pos, 1e6)),
                    torch.where(dy < 0, ty_neg, torch.full_like(ty_neg, 1e6)),
                ],
                dim=-1,
            )
            depth = cands.min(dim=-1).values.clamp(0.0, 200.0)
            lidar[:, a, :, 0] = depth / 200.0
            lidar[:, a, :, 1] = float(EntityType.WALL) / float(EntityType.NUM_TYPES)

        steps = self.steps_remaining.unsqueeze(-1).float()
        agent_id = (
            torch.arange(NUM_AGENTS, device=self.device, dtype=torch.float32)
            .view(1, NUM_AGENTS, 1)
            .expand(n, -1, -1)
        )
        if NUM_AGENTS > 1:
            agent_id = agent_id / (NUM_AGENTS - 1)

        return concatenate_observations(
            self_obs,
            partner_obs,
            room_ent_obs,
            door_obs,
            lidar,
            steps,
            agent_id,
        )
