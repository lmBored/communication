"""Fused MuJoCo-Warp kernels for the escape-room game systems.

The eager PyTorch implementation in :mod:`escape_room.mjlab_env` needs on the
order of two hundred small kernel launches per control step for movement,
buttons, doors, grab, attachments, rewards and observations.  All of that math
is per-world with static shapes, so it collapses into four Warp kernels that
read and write the MuJoCo-Warp state in place, right next to the physics
kernels, and write observations straight into a persistent buffer.
"""

from __future__ import annotations

import math

import torch
import warp as wp

from escape_room.consts import (
    AGENT_RADIUS,
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
    PARTNER_CLOSE_THRESHOLD,
    ROOM_LENGTH,
    WORLD_LENGTH,
    WORLD_WIDTH,
    EntityType,
)

_A = NUM_AGENTS
_R = NUM_ROOMS
_S = MAX_ENTITIES_PER_ROOM
_L = NUM_LIDAR_SAMPLES
_BUTTON_SLOTS = 2

# Observation stride per agent: self 8, partner 3, room 3*slots, door 3,
# lidar 2*samples, steps 1, agent id 1.
OBS_PER_AGENT = 8 + 3 * (_A - 1) + 3 * _S + 3 + 2 * _L + 1 + 1

_GRAB_PERP_SQ = 0.9 * 0.9
_CARRY_Z = 0.85
_DOOR_Z_OFFSET = 0.875
_PI = math.pi
_TYPE_SCALE = 1.0 / float(EntityType.NUM_TYPES)
_WALL_TYPE = float(EntityType.WALL) * _TYPE_SCALE
_INV_HALF_WIDTH = 1.0 / (WORLD_WIDTH * 0.5)
_INV_WORLD_LENGTH = 1.0 / WORLD_LENGTH
_INV_ROOM_LENGTH = 1.0 / ROOM_LENGTH
_INV_PI = 1.0 / math.pi
_INV_EPISODE_LEN = 1.0 / float(EPISODE_LEN)
_INV_LIDAR_RANGE = 1.0 / LIDAR_MAX_RANGE
_LIDAR_STEP = 2.0 * math.pi / float(_L)
_PARTNER_THRESHOLD_SQ = PARTNER_CLOSE_THRESHOLD * PARTNER_CLOSE_THRESHOLD


@wp.func
def _yaw_from_qpos(qpos: wp.array2d(dtype=wp.float32), w: int, adr: int) -> float:
    qw = qpos[w, adr + 3]
    qx = qpos[w, adr + 4]
    qy = qpos[w, adr + 5]
    qz = qpos[w, adr + 6]
    return wp.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


@wp.kernel
def _pre_step_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    agent_q_adr: wp.array(dtype=wp.int32),
    cube_q_adr: wp.array(dtype=wp.int32),
    cube_entity: wp.array(dtype=wp.int32),
    actions: wp.array2d(dtype=wp.float32),
    entity_pos: wp.array2d(dtype=wp.vec3),
    entity_active: wp.array2d(dtype=wp.bool),
    button_pressed: wp.array2d(dtype=wp.bool),
    door_button_mask: wp.array2d(dtype=wp.bool),
    door_persistent: wp.array2d(dtype=wp.bool),
    door_open: wp.array2d(dtype=wp.bool),
    door_pos: wp.array2d(dtype=wp.vec3),
    held_cube: wp.array2d(dtype=wp.int32),
    grab_was_down: wp.array2d(dtype=wp.bool),
    num_cubes: int,
    button_radius_sq: float,
    door_dz: float,
):
    """Publish cube poses, update buttons and doors, resolve grab toggles."""
    w = wp.tid()

    for c in range(num_cubes):
        adr = cube_q_adr[c]
        entity_pos[w, cube_entity[c]] = wp.vec3(
            qpos[w, adr + 0], qpos[w, adr + 1], qpos[w, adr + 2]
        )

    for r in range(_R):
        for j in range(_BUTTON_SLOTS):
            idx = r * _S + j
            pressed = bool(False)
            if entity_active[w, idx]:
                button = entity_pos[w, idx]
                for a in range(_A):
                    adr = agent_q_adr[a]
                    dx = qpos[w, adr + 0] - button[0]
                    dy = qpos[w, adr + 1] - button[1]
                    if dx * dx + dy * dy <= button_radius_sq:
                        pressed = True
                for c in range(num_cubes):
                    if entity_active[w, cube_entity[c]]:
                        adr = cube_q_adr[c]
                        dx = qpos[w, adr + 0] - button[0]
                        dy = qpos[w, adr + 1] - button[1]
                        if dx * dx + dy * dy <= button_radius_sq:
                            pressed = True
            button_pressed[w, idx] = pressed

    for r in range(_R):
        satisfied = bool(True)
        required = bool(False)
        for s in range(_S):
            idx = r * _S + s
            if door_button_mask[w, idx]:
                required = True
                if not button_pressed[w, idx]:
                    satisfied = False
        is_open = (satisfied and required) or (
            door_open[w, r] and door_persistent[w, r]
        )
        door_open[w, r] = is_open
        target = DOOR_CLOSED_Z
        if is_open:
            target = DOOR_OPEN_Z
        pose = door_pos[w, r]
        z = pose[2] + wp.clamp(target - pose[2], -door_dz, door_dz)
        door_pos[w, r] = wp.vec3(pose[0], pose[1], z)

    held_0 = held_cube[w, 0]
    held_1 = held_cube[w, 1]
    acquired_0 = int(held_0)
    acquired_1 = int(held_1)
    current_1 = int(held_1)
    hit_0 = bool(False)
    hit_1 = bool(False)

    for a in range(_A):
        grab_down = actions[w, a * 4 + 3] > 0.0
        rising = grab_down and not grab_was_down[w, a]
        grab_was_down[w, a] = grab_down
        held = held_cube[w, a]
        current = int(held)
        if rising and held >= 0:
            current = -1
        best = int(-1)
        best_along = float(0.0)
        if rising and held < 0:
            adr = agent_q_adr[a]
            yaw = _yaw_from_qpos(qpos, w, adr)
            fx = -wp.sin(yaw)
            fy = wp.cos(yaw)
            ox = qpos[w, adr + 0] + fx * AGENT_RADIUS
            oy = qpos[w, adr + 1] + fy * AGENT_RADIUS
            for c in range(num_cubes):
                if c != held_0 and c != held_1 and entity_active[w, cube_entity[c]]:
                    cadr = cube_q_adr[c]
                    rx = qpos[w, cadr + 0] - ox
                    ry = qpos[w, cadr + 1] - oy
                    along = rx * fx + ry * fy
                    perp_sq = wp.max(rx * rx + ry * ry - along * along, 0.0)
                    if (
                        along >= 0.0
                        and along <= GRAB_RAY_LENGTH
                        and perp_sq <= _GRAB_PERP_SQ
                    ):
                        if best < 0 or along < best_along:
                            best = c
                            best_along = along
        if a == 0:
            acquired_0 = current
            if best >= 0:
                acquired_0 = best
                hit_0 = True
        else:
            current_1 = current
            acquired_1 = current
            if best >= 0:
                acquired_1 = best
                hit_1 = True

    # Preserve the sequential per-agent semantics: agent 0 wins a contested cube.
    if hit_0 and hit_1 and acquired_0 == acquired_1:
        acquired_1 = current_1
    held_cube[w, 0] = acquired_0
    held_cube[w, 1] = acquired_1


@wp.kernel
def _apply_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    mocap_pos: wp.array2d(dtype=wp.vec3),
    agent_q_adr: wp.array(dtype=wp.int32),
    agent_v_adr: wp.array(dtype=wp.int32),
    cube_q_adr: wp.array(dtype=wp.int32),
    cube_v_adr: wp.array(dtype=wp.int32),
    door_mocap: wp.array(dtype=wp.int32),
    actions: wp.array2d(dtype=wp.float32),
    held_cube: wp.array2d(dtype=wp.int32),
    door_pos: wp.array2d(dtype=wp.vec3),
    max_speed: float,
    max_yaw_rate: float,
):
    """Drive agents, carry held cubes and slide the kinematic doors."""
    w = wp.tid()

    for a in range(_A):
        qadr = agent_q_adr[a]
        vadr = agent_v_adr[a]
        yaw = _yaw_from_qpos(qpos, w, qadr)
        move_x = wp.clamp(actions[w, a * 4 + 0], -1.0, 1.0)
        move_y = wp.clamp(actions[w, a * 4 + 1], -1.0, 1.0)
        yaw_rate = wp.clamp(actions[w, a * 4 + 2], -1.0, 1.0) * max_yaw_rate
        cos_yaw = wp.cos(yaw)
        sin_yaw = wp.sin(yaw)
        quat = wp.quat(
            qpos[w, qadr + 4],
            qpos[w, qadr + 5],
            qpos[w, qadr + 6],
            qpos[w, qadr + 3],
        )
        ang = wp.quat_rotate_inv(quat, wp.vec3(0.0, 0.0, yaw_rate))
        qvel[w, vadr + 0] = max_speed * (move_x * cos_yaw - move_y * sin_yaw)
        qvel[w, vadr + 1] = max_speed * (move_x * sin_yaw + move_y * cos_yaw)
        qvel[w, vadr + 2] = 0.0
        qvel[w, vadr + 3] = ang[0]
        qvel[w, vadr + 4] = ang[1]
        qvel[w, vadr + 5] = ang[2]

    for a in range(_A):
        cube = held_cube[w, a]
        if cube >= 0:
            qadr = agent_q_adr[a]
            yaw = _yaw_from_qpos(qpos, w, qadr)
            cq = cube_q_adr[cube]
            cv = cube_v_adr[cube]
            qpos[w, cq + 0] = qpos[w, qadr + 0] - wp.sin(yaw) * GRAB_OFFSET_FWD
            qpos[w, cq + 1] = qpos[w, qadr + 1] + wp.cos(yaw) * GRAB_OFFSET_FWD
            qpos[w, cq + 2] = _CARRY_Z
            qpos[w, cq + 3] = wp.cos(0.5 * yaw)
            qpos[w, cq + 4] = 0.0
            qpos[w, cq + 5] = 0.0
            qpos[w, cq + 6] = wp.sin(0.5 * yaw)
            qvel[w, cv + 0] = (
                wp.clamp(actions[w, a * 4 + 0], -1.0, 1.0) * max_speed
            )
            qvel[w, cv + 1] = (
                wp.clamp(actions[w, a * 4 + 1], -1.0, 1.0) * max_speed
            )
            qvel[w, cv + 2] = 0.0
            qvel[w, cv + 3] = 0.0
            qvel[w, cv + 4] = 0.0
            qvel[w, cv + 5] = 0.0

    for r in range(_R):
        pose = door_pos[w, r]
        mocap_pos[w, door_mocap[r]] = wp.vec3(
            pose[0], pose[1], pose[2] + _DOOR_Z_OFFSET
        )


@wp.kernel
def _reward_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    agent_q_adr: wp.array(dtype=wp.int32),
    progress_max_y: wp.array2d(dtype=wp.float32),
    progress_delta: wp.array2d(dtype=wp.float32),
    reward: wp.array(dtype=wp.float32),
    progress_weight: float,
    partner_weight: float,
    slack: float,
):
    """Progress, partner bonus and slack in a single pass over the agents."""
    w = wp.tid()
    total = float(0.0)
    for a in range(_A):
        y = qpos[w, agent_q_adr[a] + 1]
        delta = wp.max(y - progress_max_y[w, a], 0.0)
        progress_delta[w, a] = delta
        progress_max_y[w, a] = wp.max(progress_max_y[w, a], y)
        total += delta
    mean = total / float(_A)
    adr_0 = agent_q_adr[0]
    adr_1 = agent_q_adr[1]
    dx = qpos[w, adr_0 + 0] - qpos[w, adr_1 + 0]
    dy = qpos[w, adr_0 + 1] - qpos[w, adr_1 + 1]
    partner = float(0.0)
    if dx * dx + dy * dy < _PARTNER_THRESHOLD_SQ:
        partner = 1.0
    reward[w] = mean * (progress_weight + partner * partner_weight) + slack


@wp.kernel
def _observation_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    agent_q_adr: wp.array(dtype=wp.int32),
    entity_pos: wp.array2d(dtype=wp.vec3),
    entity_type: wp.array2d(dtype=wp.int32),
    entity_active: wp.array2d(dtype=wp.bool),
    door_pos: wp.array2d(dtype=wp.vec3),
    door_open: wp.array2d(dtype=wp.bool),
    progress_max_y: wp.array2d(dtype=wp.float32),
    held_cube: wp.array2d(dtype=wp.int32),
    episode_length: wp.array(dtype=wp.int32),
    obs: wp.array2d(dtype=wp.float32),
):
    """Write the full per-agent observation vector into the policy buffer."""
    w, a = wp.tid()
    adr = agent_q_adr[a]
    x = qpos[w, adr + 0]
    y = qpos[w, adr + 1]
    z = qpos[w, adr + 2]
    yaw = _yaw_from_qpos(qpos, w, adr)
    base = a * OBS_PER_AGENT

    room_y = float(wp.mod(y, ROOM_LENGTH))
    if room_y < 0.0:
        room_y += ROOM_LENGTH
    obs[w, base + 0] = x * _INV_HALF_WIDTH
    obs[w, base + 1] = room_y * _INV_ROOM_LENGTH
    obs[w, base + 2] = x * _INV_HALF_WIDTH
    obs[w, base + 3] = y * _INV_WORLD_LENGTH
    obs[w, base + 4] = z
    obs[w, base + 5] = progress_max_y[w, a] * _INV_WORLD_LENGTH
    obs[w, base + 6] = yaw * _INV_PI
    held = held_cube[w, a]
    if held >= 0:
        obs[w, base + 7] = 1.0
    else:
        obs[w, base + 7] = 0.0

    partner = (a + 1) % _A
    padr = agent_q_adr[partner]
    prx = qpos[w, padr + 0] - x
    pry = qpos[w, padr + 1] - y
    obs[w, base + 8] = wp.sqrt(prx * prx + pry * pry) * _INV_WORLD_LENGTH
    obs[w, base + 9] = (wp.atan2(pry, prx) - yaw) * _INV_PI
    if held_cube[w, partner] >= 0:
        obs[w, base + 10] = 1.0
    else:
        obs[w, base + 10] = 0.0

    room = wp.clamp(int(y * _INV_ROOM_LENGTH), 0, _R - 1)
    for s in range(_S):
        idx = room * _S + s
        out = base + 11 + 3 * s
        if entity_active[w, idx]:
            slot = entity_pos[w, idx]
            rx = slot[0] - x
            ry = slot[1] - y
            obs[w, out + 0] = wp.sqrt(rx * rx + ry * ry) * _INV_WORLD_LENGTH
            obs[w, out + 1] = (wp.atan2(ry, rx) - yaw) * _INV_PI
            obs[w, out + 2] = float(entity_type[w, idx]) * _TYPE_SCALE
        else:
            obs[w, out + 0] = 0.0
            obs[w, out + 1] = 0.0
            obs[w, out + 2] = 0.0

    door = door_pos[w, room]
    drx = door[0] - x
    dry = door[1] - y
    out = base + 11 + 3 * _S
    obs[w, out + 0] = wp.sqrt(drx * drx + dry * dry) * _INV_WORLD_LENGTH
    obs[w, out + 1] = (wp.atan2(dry, drx) - yaw) * _INV_PI
    if door_open[w, room]:
        obs[w, out + 2] = 1.0
    else:
        obs[w, out + 2] = 0.0

    lidar = out + 3
    eps = 1.0e-6
    for k in range(_L):
        angle = yaw - _PI + float(k) * _LIDAR_STEP
        dir_x = wp.cos(angle)
        dir_y = wp.sin(angle)
        depth = float(LIDAR_MAX_RANGE)
        t = float((WORLD_WIDTH * 0.5 - x) / wp.max(dir_x, eps))
        if t > 0.0 and t < depth:
            depth = t
        t = (-WORLD_WIDTH * 0.5 - x) / wp.min(dir_x, -eps)
        if t > 0.0 and t < depth:
            depth = t
        t = (WORLD_LENGTH - y) / wp.max(dir_y, eps)
        if t > 0.0 and t < depth:
            depth = t
        t = (0.0 - y) / wp.min(dir_y, -eps)
        if t > 0.0 and t < depth:
            depth = t
        obs[w, lidar + 2 * k + 0] = depth * _INV_LIDAR_RANGE
        obs[w, lidar + 2 * k + 1] = _WALL_TYPE

    steps = float(EPISODE_LEN - episode_length[w])
    obs[w, base + OBS_PER_AGENT - 2] = steps * _INV_EPISODE_LEN
    obs[w, base + OBS_PER_AGENT - 1] = float(a)


class WarpGame:
    """Launch the fused game kernels over the shared MuJoCo-Warp state."""

    def __init__(self, term) -> None:
        self._term = term
        self._device = wp.device_from_torch(torch.device(term.device))
        # Torch tensors are only referenced, never owned, by Warp arrays, so
        # every wrapped tensor has to stay alive for the life of the term.
        self._retain: list[torch.Tensor] = []
        data = term._sim_data
        self._qpos = data.qpos.wp_array
        self._qvel = data.qvel.wp_array
        self._mocap_pos = data.mocap_pos.wp_array
        self._num_cubes = term._num_cubes
        self._num_envs = term.num_envs

        self._agent_q_adr = self._int32(term._agent_q_flat.view(_A, 7)[:, 0])
        self._agent_v_adr = self._int32(term._agent_v_flat.view(_A, 6)[:, 0])
        cube_q = term._cube_q_flat
        cube_v = term._cube_v_flat
        if self._num_cubes:
            cube_q = cube_q.view(self._num_cubes, 7)[:, 0]
            cube_v = cube_v.view(self._num_cubes, 6)[:, 0]
        self._cube_q_adr = self._int32(cube_q)
        self._cube_v_adr = self._int32(cube_v)
        self._cube_entity = self._int32(term._cube_entity_flat)
        self._door_mocap = self._int32(term._door_mocap_ids)

        self.obs = torch.zeros(
            term.num_envs, _A * OBS_PER_AGENT, device=term.device
        )
        self.reward = torch.zeros(term.num_envs, device=term.device)
        self._obs_wp = self._wrap(self.obs)
        self._reward_wp = self._wrap(self.reward)
        self._actions = self._wrap(term._processed_actions)
        self._entity_pos = self._wrap(term._entity_pos_flat, dtype=wp.vec3)
        self._entity_type = self._wrap(term.entity_type.view(term.num_envs, -1))
        self._entity_active = self._wrap(term._entity_active_flat)
        self._button_pressed = self._wrap(
            term.button_pressed.view(term.num_envs, -1)
        )
        self._door_button_mask = self._wrap(
            term.door_button_mask.view(term.num_envs, -1)
        )
        self._door_persistent = self._wrap(term.door_persistent)
        self._door_open = self._wrap(term.door_open)
        self._door_pos = self._wrap(term.door_pos, dtype=wp.vec3)
        self._held_cube = self._wrap(term.held_cube)
        self._grab_was_down = self._wrap(term._grab_was_down)
        self._progress_max_y = self._wrap(term.progress_max_y)
        self._progress_delta = self._wrap(term.progress_delta)
        self._episode_length_torch = torch.zeros(
            term.num_envs, dtype=torch.int32, device=term.device
        )
        self._episode_length = self._wrap(self._episode_length_torch)

        self._button_radius_sq = (BUTTON_WIDTH * 0.5 + AGENT_RADIUS * 0.5) ** 2
        self._door_dz = DOOR_SPEED * term._env.step_dt

    def _wrap(self, tensor: torch.Tensor, dtype=None) -> wp.array:
        self._retain.append(tensor)
        if dtype is None:
            return wp.from_torch(tensor)
        return wp.from_torch(tensor, dtype=dtype)

    def _int32(self, tensor: torch.Tensor) -> wp.array:
        return self._wrap(tensor.to(torch.int32).contiguous())

    def pre_step(self) -> None:
        wp.launch(
            _pre_step_kernel,
            dim=self._num_envs,
            inputs=[
                self._qpos,
                self._agent_q_adr,
                self._cube_q_adr,
                self._cube_entity,
                self._actions,
                self._entity_pos,
                self._entity_active,
                self._button_pressed,
                self._door_button_mask,
                self._door_persistent,
                self._door_open,
                self._door_pos,
                self._held_cube,
                self._grab_was_down,
                self._num_cubes,
                self._button_radius_sq,
                self._door_dz,
            ],
            device=self._device,
        )

    def apply(self, max_speed: float, max_yaw_rate: float) -> None:
        wp.launch(
            _apply_kernel,
            dim=self._num_envs,
            inputs=[
                self._qpos,
                self._qvel,
                self._mocap_pos,
                self._agent_q_adr,
                self._agent_v_adr,
                self._cube_q_adr,
                self._cube_v_adr,
                self._door_mocap,
                self._actions,
                self._held_cube,
                self._door_pos,
                max_speed,
                max_yaw_rate,
            ],
            device=self._device,
        )

    def compute_reward(
        self, progress_weight: float, partner_weight: float, slack: float
    ) -> torch.Tensor:
        wp.launch(
            _reward_kernel,
            dim=self._num_envs,
            inputs=[
                self._qpos,
                self._agent_q_adr,
                self._progress_max_y,
                self._progress_delta,
                self._reward_wp,
                progress_weight,
                partner_weight,
                slack,
            ],
            device=self._device,
        )
        return self.reward

    def observe(self) -> torch.Tensor:
        self._episode_length_torch.copy_(self._term._env.episode_length_buf)
        wp.launch(
            _observation_kernel,
            dim=(self._num_envs, _A),
            inputs=[
                self._qpos,
                self._agent_q_adr,
                self._entity_pos,
                self._entity_type,
                self._entity_active,
                self._door_pos,
                self._door_open,
                self._progress_max_y,
                self._held_cube,
                self._episode_length,
                self._obs_wp,
            ],
            device=self._device,
        )
        return self.obs
