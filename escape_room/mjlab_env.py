"""mjlab manager terms for the MuJoCo-Warp escape-room environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from escape_room.consts import (
    ACTION_DIM_PER_AGENT,
    AGENT_RADIUS,
    AGENT_SPAWN_X_SPREAD,
    AGENT_SPAWN_Y_MAX,
    AGENT_SPAWN_Y_MIN,
    BUTTON_WIDTH,
    DOOR_CLOSED_Z,
    DOOR_OPEN_Z,
    DOOR_SPEED,
    EPISODE_LEN,
    GRAB_OFFSET_FWD,
    GRAB_RAY_LENGTH,
    LIDAR_MAX_RANGE,
    MAX_ENTITIES_PER_ROOM,
    NUM_AGENTS,
    NUM_LIDAR_SAMPLES,
    NUM_ROOMS,
    PARTNER_BONUS_MULT,
    PARTNER_CLOSE_THRESHOLD,
    REWARD_PER_DIST,
    ROOM_LENGTH,
    SLACK_REWARD,
    TOTAL_ACTION_DIM,
    WORLD_LENGTH,
    WORLD_WIDTH,
    EntityType,
)
from escape_room.level_gen import generate_level
from escape_room.scene import (
    BUTTON_ENTITY_NAMES,
    CUBE_ENTITY_NAMES,
    DOOR_ENTITY_NAMES,
)


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _quat_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    quat = torch.zeros((*yaw.shape, 4), device=yaw.device)
    quat[..., 0] = torch.cos(0.5 * yaw)
    quat[..., 3] = torch.sin(0.5 * yaw)
    return quat


@dataclass(kw_only=True)
class EscapeRoomActionCfg(ActionTermCfg):
    """Configuration for the combined two-agent game action term."""

    max_speed: float = 8.0
    max_yaw_rate: float = 4.0

    def build(self, env):
        return EscapeRoomAction(self, env)


class EscapeRoomAction(ActionTerm):
    """Own game state and apply controls to the real MuJoCo-Warp bodies."""

    cfg: EscapeRoomActionCfg

    def __init__(self, cfg: EscapeRoomActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(
            self.num_envs, TOTAL_ACTION_DIM, device=self.device
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._agents = [env.scene[f"agent_{idx}"] for idx in range(NUM_AGENTS)]
        self._doors = [env.scene[name] for name in DOOR_ENTITY_NAMES]
        self._buttons = [env.scene[name] for name in BUTTON_ENTITY_NAMES]
        self._cubes = [env.scene[name] for name in CUBE_ENTITY_NAMES]

        self.entity_active = torch.zeros(
            self.num_envs,
            NUM_ROOMS,
            MAX_ENTITIES_PER_ROOM,
            dtype=torch.bool,
            device=self.device,
        )
        self.entity_type = torch.zeros_like(self.entity_active, dtype=torch.long)
        self.entity_pos = torch.zeros(
            self.num_envs,
            NUM_ROOMS,
            MAX_ENTITIES_PER_ROOM,
            3,
            device=self.device,
        )
        self.button_pressed = torch.zeros_like(self.entity_active)
        self.door_button_mask = torch.zeros_like(self.entity_active)
        self.door_open = torch.zeros(
            self.num_envs, NUM_ROOMS, dtype=torch.bool, device=self.device
        )
        self.door_persistent = torch.zeros_like(self.door_open)
        self.door_pos = torch.zeros(
            self.num_envs, NUM_ROOMS, 3, device=self.device
        )
        self.door_pos[..., 2] = DOOR_CLOSED_Z
        self.held_cube = torch.full(
            (self.num_envs, NUM_AGENTS), -1, dtype=torch.long, device=self.device
        )
        self._grab_was_down = torch.zeros(
            self.num_envs, NUM_AGENTS, dtype=torch.bool, device=self.device
        )
        self.progress_max_y = torch.zeros(
            self.num_envs, NUM_AGENTS, device=self.device
        )
        self.progress_delta = torch.zeros_like(self.progress_max_y)
        self._build_level_pool()

    @property
    def action_dim(self) -> int:
        return TOTAL_ACTION_DIM

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def _build_level_pool(self) -> None:
        pool_size = 64
        pos = torch.zeros(
            pool_size, NUM_ROOMS, MAX_ENTITIES_PER_ROOM, 3, dtype=torch.float32
        )
        types = torch.zeros(
            pool_size, NUM_ROOMS, MAX_ENTITIES_PER_ROOM, dtype=torch.long
        )
        active = torch.zeros_like(types, dtype=torch.bool)
        button_mask = torch.zeros_like(active)
        persistent = torch.zeros(pool_size, NUM_ROOMS, dtype=torch.bool)
        doors = torch.zeros(pool_size, NUM_ROOMS, 3, dtype=torch.float32)
        for pool_idx in range(pool_size):
            rooms = generate_level(random.Random(0xE5CA9E + pool_idx))
            for room_idx, room in enumerate(rooms):
                doors[pool_idx, room_idx] = torch.tensor(
                    [0.0, room.door_y, DOOR_CLOSED_Z]
                )
                persistent[pool_idx, room_idx] = room.is_persistent
                for slot_idx, slot in enumerate(room.entities):
                    pos[pool_idx, room_idx, slot_idx] = torch.tensor(
                        [slot.pos_x, slot.pos_y, slot.pos_z]
                    )
                    types[pool_idx, room_idx, slot_idx] = int(slot.type)
                    active[pool_idx, room_idx, slot_idx] = slot.active
                for slot_idx in room.button_indices:
                    button_mask[pool_idx, room_idx, slot_idx] = True
        self._pool_pos = pos.to(self.device)
        self._pool_type = types.to(self.device)
        self._pool_active = active.to(self.device)
        self._pool_button_mask = button_mask.to(self.device)
        self._pool_persistent = persistent.to(self.device)
        self._pool_doors = doors.to(self.device)

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        return env_ids.to(device=self.device, dtype=torch.long)

    def _write_pose(self, entity, env_ids: torch.Tensor, pos: torch.Tensor) -> None:
        pose = torch.zeros((env_ids.numel(), 7), device=self.device)
        pose[:, :3] = pos
        pose[:, 3] = 1.0
        if entity.data.is_fixed_base:
            entity.data.write_mocap_pose(pose, env_ids)
        else:
            entity.data.write_root_pose(pose, env_ids)
            entity.data.write_root_velocity(
                torch.zeros((env_ids.numel(), 6), device=self.device), env_ids
            )

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        picks = torch.randint(self._pool_pos.shape[0], (ids.numel(),), device=self.device)
        self.entity_pos[ids] = self._pool_pos[picks]
        self.entity_type[ids] = self._pool_type[picks]
        self.entity_active[ids] = self._pool_active[picks]
        self.door_button_mask[ids] = self._pool_button_mask[picks]
        self.door_persistent[ids] = self._pool_persistent[picks]
        self.door_pos[ids] = self._pool_doors[picks]
        self.button_pressed[ids] = False
        self.door_open[ids] = False
        self.held_cube[ids] = -1
        self._grab_was_down[ids] = False
        self.progress_delta[ids] = 0.0

        spawn_x = (
            2.0 * torch.rand(ids.numel(), NUM_AGENTS, device=self.device) - 1.0
        ) * AGENT_SPAWN_X_SPREAD
        spawn_y = (
            torch.rand(ids.numel(), NUM_AGENTS, device=self.device)
            * (AGENT_SPAWN_Y_MAX - AGENT_SPAWN_Y_MIN)
            + AGENT_SPAWN_Y_MIN
        )
        self.progress_max_y[ids] = spawn_y
        for agent_idx, agent in enumerate(self._agents):
            pos = torch.stack(
                (spawn_x[:, agent_idx], spawn_y[:, agent_idx], torch.full_like(spawn_y[:, agent_idx], 0.5)),
                dim=-1,
            )
            self._write_pose(agent, ids, pos)

        for room_idx, door in enumerate(self._doors):
            pos = self.door_pos[ids, room_idx].clone()
            pos[:, 2] += 0.875
            self._write_pose(door, ids, pos)

        for room_idx in range(NUM_ROOMS):
            for slot_idx in range(2):
                entity_idx = room_idx * 2 + slot_idx
                pos = self.entity_pos[ids, room_idx, slot_idx].clone()
                enabled = self.entity_active[ids, room_idx, slot_idx]
                pos[:, 2] = torch.where(enabled, torch.full_like(pos[:, 2], 0.10), -10.0)
                self._write_pose(self._buttons[entity_idx], ids, pos)
            for slot_idx in range(2, 6):
                entity_idx = room_idx * 4 + slot_idx - 2
                pos = self.entity_pos[ids, room_idx, slot_idx].clone()
                enabled = self.entity_active[ids, room_idx, slot_idx]
                pos[:, 2] = torch.where(enabled, torch.full_like(pos[:, 2], 0.80), -10.0)
                self._write_pose(self._cubes[entity_idx], ids, pos)

    def _agent_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        pose = torch.stack([agent.data.root_link_pose_w for agent in self._agents], dim=1)
        return pose[..., :3], _yaw_from_quat(pose[..., 3:7])

    def _cube_positions(self) -> torch.Tensor:
        return torch.stack([cube.data.root_link_pos_w for cube in self._cubes], dim=1)

    def _sync_cube_slots(self) -> torch.Tensor:
        cube_pos = self._cube_positions()
        for room_idx in range(NUM_ROOMS):
            self.entity_pos[:, room_idx, 2:6] = cube_pos[:, room_idx * 4 : (room_idx + 1) * 4]
        return cube_pos

    def _update_buttons_and_doors(self) -> None:
        agent_pos, _ = self._agent_pose()
        cube_pos = self._sync_cube_slots()
        cube_active = self.entity_active[:, :, 2:6].reshape(self.num_envs, -1)
        self.button_pressed.zero_()
        radius_sq = (BUTTON_WIDTH * 0.5 + AGENT_RADIUS * 0.5) ** 2
        for room_idx in range(NUM_ROOMS):
            for slot_idx in range(2):
                active = self.entity_active[:, room_idx, slot_idx]
                button = self.entity_pos[:, room_idx, slot_idx, :2]
                agent_dist_sq = ((agent_pos[..., :2] - button[:, None]) ** 2).sum(-1)
                cube_dist_sq = ((cube_pos[..., :2] - button[:, None]) ** 2).sum(-1)
                pressed = (agent_dist_sq <= radius_sq).any(-1) | (
                    (cube_dist_sq <= radius_sq) & cube_active
                ).any(-1)
                self.button_pressed[:, room_idx, slot_idx] = active & pressed

        required = self.door_button_mask
        satisfied = (~required) | self.button_pressed
        requested = satisfied.all(-1) & required.any(-1)
        self.door_open = requested | (self.door_open & self.door_persistent)
        dz = DOOR_SPEED * self._env.step_dt
        target = torch.where(
            self.door_open,
            torch.full_like(self.door_pos[..., 2], DOOR_OPEN_Z),
            torch.full_like(self.door_pos[..., 2], DOOR_CLOSED_Z),
        )
        self.door_pos[..., 2] += (target - self.door_pos[..., 2]).clamp(-dz, dz)

    def _update_grab(self, grab_down: torch.Tensor) -> None:
        rising = grab_down & ~self._grab_was_down
        self._grab_was_down.copy_(grab_down)
        if not rising.any():
            return
        agent_pos, yaw = self._agent_pose()
        cube_pos = self._cube_positions()
        cube_active = self.entity_active[:, :, 2:6].reshape(self.num_envs, -1)
        for agent_idx in range(NUM_AGENTS):
            pressed_ids = rising[:, agent_idx].nonzero(as_tuple=False).squeeze(-1)
            if pressed_ids.numel() == 0:
                continue
            holding = self.held_cube[pressed_ids, agent_idx] >= 0
            release_ids = pressed_ids[holding]
            self.held_cube[release_ids, agent_idx] = -1
            acquire_ids = pressed_ids[~holding]
            if acquire_ids.numel() == 0:
                continue
            forward = torch.stack((-torch.sin(yaw[acquire_ids, agent_idx]), torch.cos(yaw[acquire_ids, agent_idx])), dim=-1)
            origin = agent_pos[acquire_ids, agent_idx, :2] + forward * AGENT_RADIUS
            rel = cube_pos[acquire_ids, :, :2] - origin[:, None]
            along = (rel * forward[:, None]).sum(-1)
            perp_sq = (rel.square().sum(-1) - along.square()).clamp_min(0.0)
            already_held = (
                torch.arange(len(self._cubes), device=self.device)[None, :, None]
                == self.held_cube[:, None, :]
            ).any(-1)
            valid = (
                (along >= 0.0)
                & (along <= GRAB_RAY_LENGTH)
                & (perp_sq <= 0.9**2)
                & cube_active[acquire_ids]
                & ~already_held[acquire_ids]
            )
            distance = torch.where(valid, along, torch.full_like(along, float("inf")))
            nearest_distance, nearest = distance.min(-1)
            hit = torch.isfinite(nearest_distance)
            self.held_cube[acquire_ids[hit], agent_idx] = nearest[hit]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(actions.clamp(-1.0, 1.0))
        shaped = self._processed_actions.view(self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)
        self._update_buttons_and_doors()
        self._update_grab(shaped[..., 3] > 0.0)

    def _apply_agent_controls(self) -> None:
        shaped = self._processed_actions.view(self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)
        _, yaw = self._agent_pose()
        move = shaped[..., :2]
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        vx = self.cfg.max_speed * (move[..., 0] * cos_yaw - move[..., 1] * sin_yaw)
        vy = self.cfg.max_speed * (move[..., 0] * sin_yaw + move[..., 1] * cos_yaw)
        yaw_rate = shaped[..., 2] * self.cfg.max_yaw_rate
        for agent_idx, agent in enumerate(self._agents):
            velocity = torch.zeros((self.num_envs, 6), device=self.device)
            velocity[:, 0] = vx[:, agent_idx]
            velocity[:, 1] = vy[:, agent_idx]
            velocity[:, 5] = yaw_rate[:, agent_idx]
            agent.data.write_root_velocity(velocity)

    def _apply_attachments(self) -> None:
        agent_pos, yaw = self._agent_pose()
        for agent_idx in range(NUM_AGENTS):
            ids = (self.held_cube[:, agent_idx] >= 0).nonzero(as_tuple=False).squeeze(-1)
            if ids.numel() == 0:
                continue
            cube_ids = self.held_cube[ids, agent_idx]
            forward = torch.stack((-torch.sin(yaw[ids, agent_idx]), torch.cos(yaw[ids, agent_idx])), dim=-1)
            pos = agent_pos[ids, agent_idx].clone()
            pos[:, :2] += forward * GRAB_OFFSET_FWD
            pos[:, 2] = 0.85
            quat = _quat_from_yaw(yaw[ids, agent_idx])
            pose = torch.cat((pos, quat), dim=-1)
            velocity = torch.zeros((ids.numel(), 6), device=self.device)
            shaped = self._processed_actions.view(self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT)
            velocity[:, :2] = shaped[ids, agent_idx, :2] * self.cfg.max_speed
            for cube_idx, cube in enumerate(self._cubes):
                selected = cube_ids == cube_idx
                if selected.any():
                    selected_ids = ids[selected]
                    cube.data.write_root_pose(pose[selected], selected_ids)
                    cube.data.write_root_velocity(velocity[selected], selected_ids)

    def _apply_doors(self) -> None:
        ids = self._ids(None)
        for room_idx, door in enumerate(self._doors):
            pose = torch.zeros((self.num_envs, 7), device=self.device)
            pose[:, :3] = self.door_pos[:, room_idx]
            pose[:, 2] += 0.875
            pose[:, 3] = 1.0
            door.data.write_mocap_pose(pose, ids)

    def apply_actions(self) -> None:
        self._apply_agent_controls()
        self._apply_attachments()
        self._apply_doors()

    def observation(self) -> torch.Tensor:
        agent_pos, yaw = self._agent_pose()
        self._sync_cube_slots()
        room_idx = torch.clamp((agent_pos[..., 1] / ROOM_LENGTH).long(), 0, NUM_ROOMS - 1)
        env_idx = torch.arange(self.num_envs, device=self.device)[:, None]
        current_entities = self.entity_pos[env_idx, room_idx]
        current_types = self.entity_type[env_idx, room_idx]
        current_active = self.entity_active[env_idx, room_idx]
        current_doors = self.door_pos[env_idx, room_idx]
        current_open = self.door_open[env_idx, room_idx]

        per_agent = []
        for agent_idx in range(NUM_AGENTS):
            pos = agent_pos[:, agent_idx]
            self_obs = torch.stack(
                (
                    pos[:, 0] / (WORLD_WIDTH * 0.5),
                    (pos[:, 1] % ROOM_LENGTH) / ROOM_LENGTH,
                    pos[:, 0] / (WORLD_WIDTH * 0.5),
                    pos[:, 1] / WORLD_LENGTH,
                    pos[:, 2],
                    self.progress_max_y[:, agent_idx] / WORLD_LENGTH,
                    yaw[:, agent_idx] / math.pi,
                    (self.held_cube[:, agent_idx] >= 0).float(),
                ),
                dim=-1,
            )
            partner_idx = 1 - agent_idx
            partner_rel = agent_pos[:, partner_idx, :2] - pos[:, :2]
            partner_obs = torch.stack(
                (
                    torch.linalg.vector_norm(partner_rel, dim=-1) / WORLD_LENGTH,
                    (torch.atan2(partner_rel[:, 1], partner_rel[:, 0]) - yaw[:, agent_idx]) / math.pi,
                    (self.held_cube[:, partner_idx] >= 0).float(),
                ),
                dim=-1,
            )
            rel = current_entities[:, agent_idx, :, :2] - pos[:, None, :2]
            distances = torch.linalg.vector_norm(rel, dim=-1) / WORLD_LENGTH
            angles = (torch.atan2(rel[..., 1], rel[..., 0]) - yaw[:, agent_idx, None]) / math.pi
            types = current_types[:, agent_idx].float() / float(EntityType.NUM_TYPES)
            room_obs = torch.stack((distances, angles, types), dim=-1)
            room_obs = torch.where(current_active[:, agent_idx, :, None], room_obs, torch.zeros_like(room_obs)).flatten(1)
            door_rel = current_doors[:, agent_idx, :2] - pos[:, :2]
            door_obs = torch.stack(
                (
                    torch.linalg.vector_norm(door_rel, dim=-1) / WORLD_LENGTH,
                    (torch.atan2(door_rel[:, 1], door_rel[:, 0]) - yaw[:, agent_idx]) / math.pi,
                    current_open[:, agent_idx].float(),
                ),
                dim=-1,
            )
            lidar = self._lidar(pos[:, :2], yaw[:, agent_idx])
            steps = (EPISODE_LEN - self._env.episode_length_buf).float().unsqueeze(-1) / EPISODE_LEN
            ident = torch.full_like(steps, float(agent_idx))
            per_agent.append(torch.cat((self_obs, partner_obs, room_obs, door_obs, lidar, steps, ident), dim=-1))
        return torch.cat(per_agent, dim=-1)

    def _lidar(self, origin: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        beam = torch.linspace(-math.pi, math.pi, NUM_LIDAR_SAMPLES + 1, device=self.device)[:-1]
        angle = yaw[:, None] + beam[None]
        direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
        eps = 1.0e-6
        tx_pos = ((WORLD_WIDTH * 0.5) - origin[:, 0, None]) / direction[..., 0].clamp_min(eps)
        tx_neg = ((-WORLD_WIDTH * 0.5) - origin[:, 0, None]) / direction[..., 0].clamp_max(-eps)
        ty_pos = (WORLD_LENGTH - origin[:, 1, None]) / direction[..., 1].clamp_min(eps)
        ty_neg = (0.0 - origin[:, 1, None]) / direction[..., 1].clamp_max(-eps)
        candidates = torch.stack((tx_pos, tx_neg, ty_pos, ty_neg), dim=-1)
        candidates = torch.where(candidates > 0.0, candidates, torch.full_like(candidates, float("inf")))
        depth = candidates.min(-1).values.clamp(max=LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE
        hit_type = torch.full_like(depth, float(EntityType.WALL) / float(EntityType.NUM_TYPES))
        return torch.stack((depth, hit_type), dim=-1).flatten(1)

    def consume_progress_reward(self) -> torch.Tensor:
        agent_pos, _ = self._agent_pose()
        y = agent_pos[..., 1]
        self.progress_delta = (y - self.progress_max_y).clamp_min(0.0)
        self.progress_max_y = torch.maximum(self.progress_max_y, y)
        return self.progress_delta.mean(-1)


def game_term(env) -> EscapeRoomAction:
    return env.action_manager.get_term("game")


def policy_observation(env) -> torch.Tensor:
    return game_term(env).observation()


def progress_reward(env) -> torch.Tensor:
    return game_term(env).consume_progress_reward()


def slack_reward(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def partner_bonus(env) -> torch.Tensor:
    game = game_term(env)
    pos, _ = game._agent_pose()
    close = torch.linalg.vector_norm(pos[:, 0, :2] - pos[:, 1, :2], dim=-1) < PARTNER_CLOSE_THRESHOLD
    return close.float() * game.progress_delta.mean(-1)


PROGRESS_REWARD_WEIGHT = REWARD_PER_DIST
SLACK_REWARD_WEIGHT = SLACK_REWARD
PARTNER_REWARD_WEIGHT = REWARD_PER_DIST * PARTNER_BONUS_MULT