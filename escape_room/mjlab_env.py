"""mjlab manager terms for the MuJoCo-Warp escape-room environment."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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


def _lidar_impl(
    origin: torch.Tensor, yaw: torch.Tensor, lidar_beam: torch.Tensor
) -> torch.Tensor:
    angle = yaw[..., None] + lidar_beam
    direction = torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)
    eps = 1.0e-6
    tx_pos = ((WORLD_WIDTH * 0.5) - origin[..., 0, None]) / direction[
        ..., 0
    ].clamp_min(eps)
    tx_neg = ((-WORLD_WIDTH * 0.5) - origin[..., 0, None]) / direction[
        ..., 0
    ].clamp_max(-eps)
    ty_pos = (WORLD_LENGTH - origin[..., 1, None]) / direction[..., 1].clamp_min(eps)
    ty_neg = (0.0 - origin[..., 1, None]) / direction[..., 1].clamp_max(-eps)
    candidates = torch.stack((tx_pos, tx_neg, ty_pos, ty_neg), dim=-1)
    candidates = torch.where(
        candidates > 0.0, candidates, torch.full_like(candidates, float("inf"))
    )
    depth = candidates.min(-1).values.clamp(max=LIDAR_MAX_RANGE) / LIDAR_MAX_RANGE
    hit_type = torch.full_like(
        depth, float(EntityType.WALL) / float(EntityType.NUM_TYPES)
    )
    return torch.stack((depth, hit_type), dim=-1).flatten(2)


def _observation_impl(
    agent_pos: torch.Tensor,
    yaw: torch.Tensor,
    entity_pos: torch.Tensor,
    entity_type: torch.Tensor,
    entity_active: torch.Tensor,
    door_pos: torch.Tensor,
    door_open: torch.Tensor,
    progress_max_y: torch.Tensor,
    held_cube: torch.Tensor,
    steps_left: torch.Tensor,
    env_ids: torch.Tensor,
    partner_ids: torch.Tensor,
    agent_ids: torch.Tensor,
    lidar_beam: torch.Tensor,
) -> torch.Tensor:
    """Pure observation math, kept side-effect free so it can be compiled."""
    num_envs = agent_pos.shape[0]
    room_idx = torch.clamp((agent_pos[..., 1] / ROOM_LENGTH).long(), 0, NUM_ROOMS - 1)
    env_idx = env_ids[:, None]
    current_entities = entity_pos[env_idx, room_idx]
    current_types = entity_type[env_idx, room_idx]
    current_active = entity_active[env_idx, room_idx]
    current_doors = door_pos[env_idx, room_idx]
    current_open = door_open[env_idx, room_idx]

    self_obs = torch.stack(
        (
            agent_pos[..., 0] / (WORLD_WIDTH * 0.5),
            (agent_pos[..., 1] % ROOM_LENGTH) / ROOM_LENGTH,
            agent_pos[..., 0] / (WORLD_WIDTH * 0.5),
            agent_pos[..., 1] / WORLD_LENGTH,
            agent_pos[..., 2],
            progress_max_y / WORLD_LENGTH,
            yaw / math.pi,
            (held_cube >= 0).float(),
        ),
        dim=-1,
    )
    partner_rel = agent_pos[:, partner_ids, :2] - agent_pos[..., :2]
    partner_obs = torch.stack(
        (
            torch.linalg.vector_norm(partner_rel, dim=-1) / WORLD_LENGTH,
            (torch.atan2(partner_rel[..., 1], partner_rel[..., 0]) - yaw) / math.pi,
            (held_cube[:, partner_ids] >= 0).float(),
        ),
        dim=-1,
    )
    rel = current_entities[..., :2] - agent_pos[..., None, :2]
    room_obs = torch.stack(
        (
            torch.linalg.vector_norm(rel, dim=-1) / WORLD_LENGTH,
            (torch.atan2(rel[..., 1], rel[..., 0]) - yaw[..., None]) / math.pi,
            current_types.float() / float(EntityType.NUM_TYPES),
        ),
        dim=-1,
    )
    room_obs = torch.where(
        current_active[..., None], room_obs, torch.zeros_like(room_obs)
    ).flatten(2)
    door_rel = current_doors[..., :2] - agent_pos[..., :2]
    door_obs = torch.stack(
        (
            torch.linalg.vector_norm(door_rel, dim=-1) / WORLD_LENGTH,
            (torch.atan2(door_rel[..., 1], door_rel[..., 0]) - yaw) / math.pi,
            current_open.float(),
        ),
        dim=-1,
    )
    lidar = _lidar_impl(agent_pos[..., :2], yaw, lidar_beam)
    steps = steps_left.view(num_envs, 1, 1).expand(-1, NUM_AGENTS, -1) / EPISODE_LEN
    ident = agent_ids.expand(num_envs, -1, -1)
    return torch.cat(
        (self_obs, partner_obs, room_obs, door_obs, lidar, steps, ident), dim=-1
    ).flatten(1)


@dataclass(kw_only=True)
class EscapeRoomActionCfg(ActionTermCfg):
    """Configuration for the combined two-agent game action term."""

    max_speed: float = 8.0
    max_yaw_rate: float = 4.0
    compile_game: bool = False
    """Compile the observation math with ``torch.compile`` (static shapes)."""

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
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._cube_ids = torch.arange(len(self._cubes), device=self.device)
        self._partner_ids = torch.arange(NUM_AGENTS - 1, -1, -1, device=self.device)
        self._agent_ids = torch.arange(
            NUM_AGENTS, dtype=torch.float32, device=self.device
        ).view(1, NUM_AGENTS, 1)
        self._lidar_beam = torch.linspace(
            -math.pi, math.pi, NUM_LIDAR_SAMPLES + 1, device=self.device
        )[:-1].view(1, 1, NUM_LIDAR_SAMPLES)
        self.progress_max_y = torch.zeros(
            self.num_envs, NUM_AGENTS, device=self.device
        )
        self.progress_delta = torch.zeros_like(self.progress_max_y)
        self._mean_progress_delta = torch.zeros(self.num_envs, device=self.device)
        self._slack = torch.ones(self.num_envs, device=self.device)
        self._pose_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._cube_pos_cache: torch.Tensor | None = None
        self._cube_slots_synced = False
        self._build_index_tables()
        self._build_level_pool()
        self._observation_fn = _observation_impl
        if cfg.compile_game:
            self._observation_fn = torch.compile(_observation_impl, dynamic=False)

    def _build_index_tables(self) -> None:
        """Cache flat indices so every body is read/written in one batched op.

        mjlab entities are thin views over the shared MuJoCo-Warp state, so a
        per-entity loop costs one kernel launch per agent/cube/door. Gathering
        the addresses once collapses those loops into a handful of launches.
        """
        self._sim_data = self._agents[0].data.data
        self._num_cubes = len(self._cubes)

        def free_joint_adr(entities, attr: str) -> torch.Tensor:
            return torch.stack(
                [getattr(e.data.indexing, attr) for e in entities]
            ).to(device=self.device, dtype=torch.long)

        def mocap_ids(entities) -> torch.Tensor:
            return torch.tensor(
                [e.data.indexing.mocap_id for e in entities],
                device=self.device,
                dtype=torch.long,
            )

        agent_q = free_joint_adr(self._agents, "free_joint_q_adr")
        self._agent_q_flat = agent_q.reshape(-1)
        self._agent_quat_flat = agent_q[:, 3:7].reshape(-1)
        self._agent_v_flat = free_joint_adr(
            self._agents, "free_joint_v_adr"
        ).reshape(-1)

        self._cube_q_flat = free_joint_adr(
            self._cubes, "free_joint_q_adr"
        ).reshape(-1)
        self._cube_v_flat = free_joint_adr(
            self._cubes, "free_joint_v_adr"
        ).reshape(-1)

        self._door_mocap_ids = mocap_ids(self._doors)
        self._button_mocap_ids = mocap_ids(self._buttons)
        self._door_z_offset = torch.tensor(
            [0.0, 0.0, 0.875], device=self.device
        ).view(1, 1, 3)
        self._identity_quat = torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )

    def _invalidate_cache(self) -> None:
        self._pose_cache = None
        self._cube_pos_cache = None
        self._cube_slots_synced = False

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
            return self._env_ids
        return env_ids.to(device=self.device, dtype=torch.long)

    def _write_free_bodies(
        self,
        q_flat: torch.Tensor,
        v_flat: torch.Tensor,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
    ) -> None:
        """Write identity-oriented poses and zero velocities for free bodies."""
        count = pos.shape[1]
        pose = torch.zeros((pos.shape[0], count, 7), device=self.device)
        pose[..., :3] = pos
        pose[..., 3] = 1.0
        rows = env_ids[:, None]
        self._sim_data.qpos[rows, q_flat] = pose.reshape(pos.shape[0], -1)
        self._sim_data.qvel[rows, v_flat] = torch.zeros(
            (pos.shape[0], v_flat.numel()), device=self.device
        )

    def _write_mocap_bodies(
        self, mocap_ids: torch.Tensor, env_ids: torch.Tensor, pos: torch.Tensor
    ) -> None:
        rows = env_ids[:, None]
        self._sim_data.mocap_pos[rows, mocap_ids] = pos
        self._sim_data.mocap_quat[rows, mocap_ids] = self._identity_quat.expand(
            pos.shape[0], pos.shape[1], 4
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

        count = ids.numel()
        spawn_x = (
            2.0 * torch.rand(count, NUM_AGENTS, device=self.device) - 1.0
        ) * AGENT_SPAWN_X_SPREAD
        spawn_y = (
            torch.rand(count, NUM_AGENTS, device=self.device)
            * (AGENT_SPAWN_Y_MAX - AGENT_SPAWN_Y_MIN)
            + AGENT_SPAWN_Y_MIN
        )
        self.progress_max_y[ids] = spawn_y
        agent_pos = torch.stack(
            (spawn_x, spawn_y, torch.full_like(spawn_y, 0.5)), dim=-1
        )
        self._write_free_bodies(
            self._agent_q_flat, self._agent_v_flat, ids, agent_pos
        )

        self._write_mocap_bodies(
            self._door_mocap_ids, ids, self.door_pos[ids] + self._door_z_offset
        )

        slots = self.entity_pos[ids]
        active = self.entity_active[ids]
        button_pos = slots[:, :, 0:2].reshape(count, -1, 3).clone()
        button_pos[..., 2] = torch.where(
            active[:, :, 0:2].reshape(count, -1), 0.10, -10.0
        )
        self._write_mocap_bodies(self._button_mocap_ids, ids, button_pos)

        cube_pos = slots[:, :, 2:6].reshape(count, -1, 3).clone()
        cube_pos[..., 2] = torch.where(
            active[:, :, 2:6].reshape(count, -1), 0.80, -10.0
        )
        self._write_free_bodies(
            self._cube_q_flat, self._cube_v_flat, ids, cube_pos
        )
        self._invalidate_cache()

    def _agent_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pose_cache is None:
            pose = self._sim_data.qpos[:, self._agent_q_flat].view(
                self.num_envs, NUM_AGENTS, 7
            )
            pos = pose[..., :3]
            yaw = _yaw_from_quat(pose[..., 3:7])
            self._pose_cache = (pos, yaw)
        return self._pose_cache

    def _cube_positions(self) -> torch.Tensor:
        if self._cube_pos_cache is None:
            self._cube_pos_cache = self._sim_data.qpos[:, self._cube_q_flat].view(
                self.num_envs, self._num_cubes, 7
            )[..., :3]
        return self._cube_pos_cache

    def _sync_cube_slots(self) -> torch.Tensor:
        cube_pos = self._cube_positions()
        if not self._cube_slots_synced:
            self.entity_pos[:, :, 2:6] = cube_pos.view(
                self.num_envs, NUM_ROOMS, -1, 3
            )
            self._cube_slots_synced = True
        return cube_pos

    def _update_buttons_and_doors(self) -> None:
        agent_pos, _ = self._agent_pose()
        cube_pos = self._sync_cube_slots()
        radius_sq = (BUTTON_WIDTH * 0.5 + AGENT_RADIUS * 0.5) ** 2
        button_xy = self.entity_pos[:, :, 0:2, :2].unsqueeze(-2)
        agent_xy = agent_pos[..., :2].view(self.num_envs, 1, 1, NUM_AGENTS, 2)
        cube_xy = cube_pos[..., :2].view(self.num_envs, 1, 1, self._num_cubes, 2)
        cube_active = self.entity_active[:, :, 2:6].reshape(
            self.num_envs, 1, 1, -1
        )
        agent_hit = ((agent_xy - button_xy).square().sum(-1) <= radius_sq).any(-1)
        cube_hit = (
            ((cube_xy - button_xy).square().sum(-1) <= radius_sq) & cube_active
        ).any(-1)
        self.button_pressed[:, :, 0:2] = self.entity_active[:, :, 0:2] & (
            agent_hit | cube_hit
        )

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
        agent_pos, yaw = self._agent_pose()
        cube_pos = self._cube_positions()
        cube_active = self.entity_active[:, :, 2:6].reshape(self.num_envs, 1, -1)
        held = self.held_cube
        was_holding = held >= 0
        current = torch.where(
            rising & was_holding, torch.full_like(held, -1), held
        )
        acquire = rising & ~was_holding
        forward = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        origin = agent_pos[..., :2] + forward * AGENT_RADIUS
        rel = cube_pos[:, None, :, :2] - origin[:, :, None, :]
        along = (rel * forward[:, :, None, :]).sum(-1)
        perp_sq = (rel.square().sum(-1) - along.square()).clamp_min(0.0)
        already_held = (
            self._cube_ids.view(1, -1, 1) == held[:, None, :]
        ).any(-1).unsqueeze(1)
        valid = (
            acquire[..., None]
            & (along >= 0.0)
            & (along <= GRAB_RAY_LENGTH)
            & (perp_sq <= 0.9**2)
            & cube_active
            & ~already_held
        )
        distance = torch.where(valid, along, torch.full_like(along, float("inf")))
        nearest_distance, nearest = distance.min(-1)
        hit = torch.isfinite(nearest_distance)
        acquired = torch.where(hit, nearest, current)
        # Preserve the sequential semantics of the original per-agent loop: if
        # both agents target the same cube on the same step, agent 0 wins.
        conflict = hit[:, 0] & hit[:, 1] & (acquired[:, 0] == acquired[:, 1])
        acquired[:, 1] = torch.where(conflict, current[:, 1], acquired[:, 1])
        self.held_cube.copy_(acquired)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._invalidate_cache()
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
        zeros = torch.zeros_like(vx)
        lin_vel = torch.stack((vx, vy, zeros), dim=-1)
        ang_vel_w = torch.stack((zeros, zeros, yaw_rate), dim=-1)
        quat_w = self._sim_data.qpos[:, self._agent_quat_flat].view(
            self.num_envs, NUM_AGENTS, 4
        )
        ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
        self._sim_data.qvel[:, self._agent_v_flat] = torch.cat(
            (lin_vel, ang_vel_b), dim=-1
        ).view(self.num_envs, -1)

    def _apply_attachments(self) -> None:
        agent_pos, yaw = self._agent_pose()
        forward = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        pos = agent_pos.clone()
        pos[..., :2] += forward * GRAB_OFFSET_FWD
        pos[..., 2] = 0.85
        pose = torch.cat((pos, _quat_from_yaw(yaw)), dim=-1)
        shaped = self._processed_actions.view(
            self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT
        )
        velocity = torch.zeros((self.num_envs, NUM_AGENTS, 6), device=self.device)
        velocity[..., :2] = shaped[..., :2] * self.cfg.max_speed

        held_by = self.held_cube[:, :, None] == self._cube_ids.view(1, 1, -1)
        held_any = held_by.any(1, keepdim=True).transpose(1, 2)
        by_agent_0 = held_by[:, 0].unsqueeze(-1)
        selected_pose = torch.where(by_agent_0, pose[:, 0:1], pose[:, 1:2])
        selected_velocity = torch.where(by_agent_0, velocity[:, 0:1], velocity[:, 1:2])

        data = self._sim_data
        current_pose = data.qpos[:, self._cube_q_flat].view(
            self.num_envs, self._num_cubes, 7
        )
        data.qpos[:, self._cube_q_flat] = torch.where(
            held_any, selected_pose, current_pose
        ).view(self.num_envs, -1)
        current_velocity = data.qvel[:, self._cube_v_flat].view(
            self.num_envs, self._num_cubes, 6
        )
        data.qvel[:, self._cube_v_flat] = torch.where(
            held_any, selected_velocity, current_velocity
        ).view(self.num_envs, -1)

    def _apply_doors(self) -> None:
        self._sim_data.mocap_pos[:, self._door_mocap_ids] = (
            self.door_pos + self._door_z_offset
        )

    def apply_actions(self) -> None:
        self._invalidate_cache()
        self._apply_agent_controls()
        self._apply_attachments()
        self._apply_doors()
        self._invalidate_cache()

    def observation(self) -> torch.Tensor:
        self._invalidate_cache()
        agent_pos, yaw = self._agent_pose()
        self._sync_cube_slots()
        steps_left = (EPISODE_LEN - self._env.episode_length_buf).float()
        return self._observation_fn(
            agent_pos,
            yaw,
            self.entity_pos,
            self.entity_type,
            self.entity_active,
            self.door_pos,
            self.door_open,
            self.progress_max_y,
            self.held_cube,
            steps_left,
            self._env_ids,
            self._partner_ids,
            self._agent_ids,
            self._lidar_beam,
        )

    def consume_progress_reward(self) -> torch.Tensor:
        agent_pos, _ = self._agent_pose()
        y = agent_pos[..., 1]
        self.progress_delta = (y - self.progress_max_y).clamp_min(0.0)
        self.progress_max_y = torch.maximum(self.progress_max_y, y)
        self._mean_progress_delta = self.progress_delta.mean(-1)
        return self._mean_progress_delta

    def combined_reward(self) -> torch.Tensor:
        mean_progress = self.consume_progress_reward()
        pos, _ = self._agent_pose()
        partner_dist_sq = (pos[:, 0, :2] - pos[:, 1, :2]).square().sum(-1)
        partner_scale = (
            partner_dist_sq < PARTNER_CLOSE_THRESHOLD * PARTNER_CLOSE_THRESHOLD
        ).float()
        return (
            mean_progress
            * (PROGRESS_REWARD_WEIGHT + partner_scale * PARTNER_REWARD_WEIGHT)
            + SLACK_REWARD_WEIGHT
        )


def game_term(env) -> EscapeRoomAction:
    return env.action_manager.get_term("game")


def policy_observation(env) -> torch.Tensor:
    return game_term(env).observation()


def progress_reward(env) -> torch.Tensor:
    return game_term(env).consume_progress_reward()


def slack_reward(env) -> torch.Tensor:
    return game_term(env)._slack


def partner_bonus(env) -> torch.Tensor:
    game = game_term(env)
    pos, _ = game._agent_pose()
    close = torch.linalg.vector_norm(pos[:, 0, :2] - pos[:, 1, :2], dim=-1) < PARTNER_CLOSE_THRESHOLD
    return close.float() * game._mean_progress_delta


def combined_reward(env) -> torch.Tensor:
    return game_term(env).combined_reward()


PROGRESS_REWARD_WEIGHT = REWARD_PER_DIST
SLACK_REWARD_WEIGHT = SLACK_REWARD
PARTNER_REWARD_WEIGHT = REWARD_PER_DIST * PARTNER_BONUS_MULT