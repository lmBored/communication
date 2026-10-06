"""Manager terms for the two-way communication scenario.

The action term owns every piece of game state: the three arrow directions, the
door colour, both agents' poses, and the outcome flags. Managers reach it only
through the module-level thunks at the bottom of this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import requires_model_fields

from escape_room.twowaycomm.scene import (
    ARENA_NAME,
    ARROW_POSITIONS,
    ARROW_SIGN_NAMES,
    DECISION_CENTER_X,
    DEFAULT_ARROW_LAYOUT,
    DOOR_TINT_GEOM_NAMES,
    NUM_COLORS,
    RECEIVER_NAME,
    RECEIVER_SPAWN,
    SENDER_NAME,
    SENDER_ROOM_CENTER,
    SENDER_ROOM_HALF_X,
    SENDER_ROOM_HALF_Y,
    SENDER_SPAWNS,
    TERMINAL_THRESHOLD_Y,
    COLOR_RGBA,
    arrow_quaternions,
)

LEFT = -1
RIGHT = 1

SENDER_CAMERA_SENSOR = "sender_camera"
RECEIVER_CAMERA_SENSOR = "receiver_camera"

NUM_AGENTS = 2
SENDER_INDEX = 0
RECEIVER_INDEX = 1

ACTION_DIM_PER_AGENT = 3
ACTION_DIM = NUM_AGENTS * ACTION_DIM_PER_AGENT

# Shared 14-wide observation layout. Both agents publish the same slots; only
# the content differs, which is what forces information across the channel.
POSE_FEATURE_DIM = 7
AGENT_OBS_DIM = POSE_FEATURE_DIM + NUM_COLORS + NUM_COLORS + 1
SENDER_OBS_DIM = AGENT_OBS_DIM
RECEIVER_OBS_DIM = AGENT_OBS_DIM
CRITIC_OBS_DIM = NUM_AGENTS * AGENT_OBS_DIM

ARROW_SLICE = slice(POSE_FEATURE_DIM, POSE_FEATURE_DIM + NUM_COLORS)
COLOR_SLICE = slice(POSE_FEATURE_DIM + NUM_COLORS, POSE_FEATURE_DIM + 2 * NUM_COLORS)
TIME_INDEX = AGENT_OBS_DIM - 1

# Per-room pose normalization: each agent's position is expressed in its own
# room's frame so both land in a comparable range despite sharing one slot.
# The receiver constants are frozen to the one-way scenario's so indices 0-6 are
# numerically identical there.
RECEIVER_ORIGIN_X, RECEIVER_SCALE_X = DECISION_CENTER_X, 5.0
RECEIVER_ORIGIN_Y, RECEIVER_SCALE_Y = 0.0, 9.0
SENDER_ORIGIN_X, SENDER_SCALE_X = SENDER_ROOM_CENTER[0], SENDER_ROOM_HALF_X
SENDER_ORIGIN_Y, SENDER_SCALE_Y = SENDER_ROOM_CENTER[1], SENDER_ROOM_HALF_Y


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(kw_only=True)
class TwoWayCommActionCfg(ActionTermCfg):
    """Configuration for the six controls shared by the two agents."""

    max_speed: float = 8.0
    max_yaw_rate: float = 4.0
    arrow_layout: str = DEFAULT_ARROW_LAYOUT
    obs_mode: str = "vector"
    sender_name: str = SENDER_NAME
    receiver_name: str = RECEIVER_NAME

    def build(self, env):
        return TwoWayCommAction(self, env)


class TwoWayCommAction(ActionTerm):
    """Own the puzzle state and drive both mobile agents."""

    cfg: TwoWayCommActionCfg

    def __init__(self, cfg: TwoWayCommActionCfg, env):
        super().__init__(cfg, env)
        layout = cfg.arrow_layout
        if layout not in ARROW_POSITIONS:
            raise ValueError(f"unknown arrow layout {layout!r}")

        self._raw_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._processed_flat = torch.zeros_like(self._raw_actions)
        # A view, so clamping into the flat buffer updates the per-agent shape.
        self._processed_actions = self._processed_flat.view(
            self.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT
        )
        self._env_ids = torch.arange(self.num_envs, device=self.device)

        sender = env.scene[cfg.sender_name]
        receiver = env.scene[cfg.receiver_name]
        self._sim_data = receiver.data.data
        self._q = torch.stack(
            (
                sender.data.indexing.free_joint_q_adr,
                receiver.data.indexing.free_joint_q_adr,
            )
        ).to(device=self.device, dtype=torch.long)
        self._v = torch.stack(
            (
                sender.data.indexing.free_joint_v_adr,
                receiver.data.indexing.free_joint_v_adr,
            )
        ).to(device=self.device, dtype=torch.long)
        if self._q.shape != (NUM_AGENTS, 7) or self._v.shape != (NUM_AGENTS, 6):
            raise RuntimeError(
                "both agents must have a free joint; got q/v address shapes "
                f"{tuple(self._q.shape)}/{tuple(self._v.shape)}"
            )

        mocap_ids = []
        for name in ARROW_SIGN_NAMES:
            mocap_id = env.scene[name].data.indexing.mocap_id
            if mocap_id is None:
                raise RuntimeError(f"arrow sign {name!r} is not a mocap body")
            mocap_ids.append(mocap_id)
        self._arrow_mocap_ids = torch.tensor(
            mocap_ids, device=self.device, dtype=torch.long
        )

        arena = env.scene[ARENA_NAME]
        local_ids, _ = arena.find_geoms(list(DOOR_TINT_GEOM_NAMES), preserve_order=True)
        self._door_geom_ids = arena.data.indexing.geom_ids[local_ids].to(
            device=self.device, dtype=torch.long
        )
        self._assert_paintable_doors(env)

        self._arrow_pos = torch.tensor(ARROW_POSITIONS[layout], device=self.device)
        self._arrow_quats = torch.tensor(arrow_quaternions(layout), device=self.device)
        self._door_rgba = torch.tensor(COLOR_RGBA, device=self.device)
        self._spawns = torch.tensor(
            (SENDER_SPAWNS[layout], RECEIVER_SPAWN), device=self.device
        )
        self._colour_axis = torch.arange(NUM_COLORS, device=self.device)
        self._zeros_colors = torch.zeros(self.num_envs, NUM_COLORS, device=self.device)
        self._color_eye = torch.eye(NUM_COLORS, device=self.device)

        # Zeros, not torch.empty: the reset-mode event term reads door_color
        # before the action term's first reset() and would index out of range.
        self.arrow_direction = torch.full(
            (self.num_envs, NUM_COLORS), RIGHT, dtype=torch.long, device=self.device
        )
        self.door_color = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.correct_entry = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wrong_entry = torch.zeros_like(self.correct_entry)
        self.timeout = torch.zeros_like(self.correct_entry)

        self._write_spawn_poses(self._env_ids)
        self._write_signs(self._env_ids)
        self.write_door_colors(self._env_ids)

    def _assert_paintable_doors(self, env) -> None:
        """Fail loudly when per-world door colour cannot actually be written."""
        geom_matid = env.sim.mj_model.geom_matid[self._door_geom_ids.cpu().numpy()]
        if (geom_matid != -1).any():
            raise RuntimeError(
                "doorway tint geoms must not carry a material; geom_rgba is "
                "ignored for geoms whose matid is set"
            )
        rgba = env.sim.model.geom_rgba
        if self.num_envs > 1:
            broadcast = rgba.shape[0] != self.num_envs or rgba.stride(0) == 0
            if broadcast or "geom_rgba" not in env.sim.expanded_fields:
                raise RuntimeError(
                    "per-world geom_rgba is unavailable, so every world would "
                    "share one door colour; register the 'door_color' reset "
                    "event, which carries @requires_model_fields('geom_rgba')"
                )

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    def _ids(self, env_ids) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return self._env_ids
        return env_ids.to(device=self.device, dtype=torch.long)

    # Sampling.

    def _balanced_directions(self, count: int) -> torch.Tensor:
        half = count // 2
        directions = torch.empty(count, dtype=torch.long, device=self.device)
        directions[:half] = LEFT
        directions[half : 2 * half] = RIGHT
        if count % 2:
            directions[-1] = torch.where(
                torch.rand((), device=self.device) < 0.5,
                torch.tensor(LEFT, device=self.device),
                torch.tensor(RIGHT, device=self.device),
            )
        return directions[torch.randperm(count, device=self.device)]

    def _balanced_colors(self, count: int) -> torch.Tensor:
        """Exactly ``count // 3`` of each colour plus a distinct random remainder."""
        base = count // NUM_COLORS
        colors = self._colour_axis.repeat_interleave(base)
        remainder = count - colors.numel()
        if remainder:
            extra = torch.randperm(NUM_COLORS, device=self.device)[:remainder]
            colors = torch.cat((colors, extra))
        return colors[torch.randperm(count, device=self.device)]

    # World writes.

    def _write_signs(self, ids: torch.Tensor) -> None:
        count = ids.numel()
        selection = (self.arrow_direction[ids] == RIGHT).long()
        quats = self._arrow_quats[self._colour_axis, selection]
        rows = ids[:, None]
        self._sim_data.mocap_quat[rows, self._arrow_mocap_ids] = quats
        # Rewritten every reset because reset_scene_to_default teleports mocap
        # bodies to their initial state rather than their XML position.
        self._sim_data.mocap_pos[rows, self._arrow_mocap_ids] = self._arrow_pos.expand(
            count, -1, -1
        )

    def write_door_colors(self, env_ids=None) -> None:
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        rgba = self._door_rgba[self.door_color[ids]]
        self._env.sim.model.geom_rgba[ids[:, None], self._door_geom_ids] = rgba[
            :, None, :
        ].expand(-1, self._door_geom_ids.numel(), -1)

    def _write_spawn_poses(self, ids: torch.Tensor) -> None:
        count = ids.numel()
        pose = torch.zeros(count, NUM_AGENTS, 7, device=self.device)
        pose[..., :3] = self._spawns
        pose[..., 3] = 1.0
        rows = ids[:, None, None]
        self._sim_data.qpos[rows, self._q] = pose
        self._sim_data.qvel[rows, self._v] = 0.0

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        count = ids.numel()
        for colour in range(NUM_COLORS):
            self.arrow_direction[ids, colour] = self._balanced_directions(count)
        self.door_color[ids] = self._balanced_colors(count)
        self.correct_entry[ids] = False
        self.wrong_entry[ids] = False
        self.timeout[ids] = False
        self._raw_actions[ids] = 0.0
        self._processed_flat[ids] = 0.0
        self._write_spawn_poses(ids)
        self._write_signs(ids)
        # Authoritative paint. The reset-mode event term runs before this and
        # repaints the previous episode's colour, which nothing observes.
        self.write_door_colors(ids)

    # Setters used by tests and evaluation.

    def set_arrow_directions(self, directions: torch.Tensor, env_ids=None) -> None:
        ids = self._ids(env_ids)
        directions = directions.to(device=self.device, dtype=torch.long)
        if directions.ndim == 1:
            directions = directions.expand(ids.numel(), -1)
        if directions.shape != (ids.numel(), NUM_COLORS) or not torch.all(
            (directions == LEFT) | (directions == RIGHT)
        ):
            raise ValueError(
                "directions must be LEFT/RIGHT with shape (num_colors,) or "
                "(num_worlds, num_colors)"
            )
        self.arrow_direction[ids] = directions
        self._write_signs(ids)

    def set_door_color(self, color: torch.Tensor, env_ids=None) -> None:
        ids = self._ids(env_ids)
        color = color.to(device=self.device, dtype=torch.long).flatten()
        if color.numel() == 1:
            color = color.expand(ids.numel())
        if color.shape != ids.shape or not torch.all(
            (color >= 0) & (color < NUM_COLORS)
        ):
            raise ValueError("color must hold one index in [0, 3) per world")
        self.door_color[ids] = color
        self.write_door_colors(ids)

    def _set_pose(self, agent_index: int, position: torch.Tensor, env_ids) -> None:
        ids = self._ids(env_ids)
        position = position.to(device=self.device, dtype=torch.float32)
        if position.ndim == 1:
            position = position.expand(ids.numel(), -1)
        if position.shape != (ids.numel(), 3):
            raise ValueError("position must have shape (3,) or (num_worlds, 3)")
        pose = torch.zeros(ids.numel(), 7, device=self.device)
        pose[:, :3] = position
        pose[:, 3] = 1.0
        rows = ids[:, None]
        self._sim_data.qpos[rows, self._q[agent_index]] = pose
        self._sim_data.qvel[rows, self._v[agent_index]] = 0.0

    def set_sender_pose(self, position: torch.Tensor, env_ids=None) -> None:
        self._set_pose(SENDER_INDEX, position, env_ids)

    def set_receiver_pose(self, position: torch.Tensor, env_ids=None) -> None:
        self._set_pose(RECEIVER_INDEX, position, env_ids)

    def sender_pose(self) -> torch.Tensor:
        return self._sim_data.qpos[:, self._q[SENDER_INDEX]]

    def receiver_pose(self) -> torch.Tensor:
        return self._sim_data.qpos[:, self._q[RECEIVER_INDEX]]

    def sender_velocity(self) -> torch.Tensor:
        return self._sim_data.qvel[:, self._v[SENDER_INDEX]]

    def receiver_velocity(self) -> torch.Tensor:
        return self._sim_data.qvel[:, self._v[RECEIVER_INDEX]]

    def target_direction(self) -> torch.Tensor:
        """Direction of the arrow whose colour matches the doors."""
        return self.arrow_direction.gather(1, self.door_color[:, None]).squeeze(1)

    def sender_alignment(self) -> torch.Tensor:
        """How squarely the sender faces the arrow that matches the door colour.

        Returns a per-world value in [0, 1]. Optional shaping: it is the only
        thing that makes the sender's own controls affect reward, and therefore
        the only thing that gives the back-channel a job in vector mode, where
        the sender can already read every arrow from its observation.
        """
        pose = self.sender_pose()
        yaw = _yaw_from_quat(pose[:, 3:7])
        forward = torch.stack((-torch.sin(yaw), torch.cos(yaw)), dim=-1)
        target = self._arrow_pos[self.door_color][:, :2]
        delta = target - pose[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
        return ((delta / distance[:, None]) * forward).sum(-1).clamp_min(0.0)

    # Control.

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        torch.clamp(actions, -1.0, 1.0, out=self._processed_flat)

    def apply_actions(self) -> None:
        pose = self._sim_data.qpos[:, self._q]
        yaw = _yaw_from_quat(pose[..., 3:7])
        forward_action, strafe_action, yaw_action = self._processed_actions.unbind(-1)
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        velocity = torch.zeros(
            self.num_envs, NUM_AGENTS, 6, device=self.device
        )
        velocity[..., 0] = self.cfg.max_speed * (
            -forward_action * sin_yaw + strafe_action * cos_yaw
        )
        velocity[..., 1] = self.cfg.max_speed * (
            forward_action * cos_yaw + strafe_action * sin_yaw
        )
        velocity[..., 5] = yaw_action * self.cfg.max_yaw_rate
        self._sim_data.qvel[:, self._v] = velocity

    # Outcome.

    def _update_outcome(self) -> None:
        position = self.receiver_pose()[:, :3]
        entered = position[:, 1] >= TERMINAL_THRESHOLD_Y
        entered_right = position[:, 0] > DECISION_CENTER_X
        chose_right = self.target_direction() == RIGHT
        self.correct_entry.copy_(entered & (entered_right == chose_right))
        self.wrong_entry.copy_(entered & (entered_right != chose_right))

    def _update_timeout(self) -> None:
        self._update_outcome()
        elapsed = self._env.episode_length_buf >= self._env.max_episode_length
        self.timeout.copy_(elapsed & ~self.correct_entry & ~self.wrong_entry)

    # Observations.

    def _remaining(self) -> torch.Tensor:
        steps_left = (
            self._env.max_episode_length - self._env.episode_length_buf
        ).clamp_min(0)
        return steps_left.float() / self._env.max_episode_length

    def _pose_features(
        self,
        agent_index: int,
        origin_x: float,
        scale_x: float,
        origin_y: float,
        scale_y: float,
    ) -> torch.Tensor:
        pose = self._sim_data.qpos[:, self._q[agent_index]]
        velocity = self._sim_data.qvel[:, self._v[agent_index]]
        position = pose[:, :3]
        yaw = _yaw_from_quat(pose[:, 3:7])
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        right = torch.stack((cos_yaw, sin_yaw), dim=-1)
        forward = torch.stack((-sin_yaw, cos_yaw), dim=-1)
        local_forward_velocity = (velocity[:, :2] * forward).sum(-1) / self.cfg.max_speed
        local_strafe_velocity = (velocity[:, :2] * right).sum(-1) / self.cfg.max_speed
        local_position = torch.stack(
            (
                (position[:, 0] - origin_x) / scale_x,
                (position[:, 1] - origin_y) / scale_y,
            ),
            dim=-1,
        )
        return torch.cat(
            (
                local_position,
                sin_yaw[:, None],
                cos_yaw[:, None],
                local_forward_velocity[:, None],
                local_strafe_velocity[:, None],
                (velocity[:, 5] / self.cfg.max_yaw_rate)[:, None],
            ),
            dim=-1,
        )

    def sender_observation(self) -> torch.Tensor:
        """Arrow directions visible; door colour zeroed out.

        In pixel mode the arrow block is blanked too: the sender must read the
        arrows off its camera instead.
        """
        arrows = (
            self._zeros_colors
            if self.cfg.obs_mode == "pixel"
            else self.arrow_direction.float()
        )
        return torch.cat(
            (
                self._pose_features(
                    SENDER_INDEX,
                    SENDER_ORIGIN_X,
                    SENDER_SCALE_X,
                    SENDER_ORIGIN_Y,
                    SENDER_SCALE_Y,
                ),
                arrows,
                self._zeros_colors,
                self._remaining()[:, None],
            ),
            dim=-1,
        )

    def receiver_observation(self) -> torch.Tensor:
        """Door colour visible; arrow directions zeroed out.

        In pixel mode the colour block is blanked too and must be read off the
        receiver's camera.
        """
        colors = (
            self._zeros_colors
            if self.cfg.obs_mode == "pixel"
            else self._color_eye[self.door_color]
        )
        return torch.cat(
            (
                self._pose_features(
                    RECEIVER_INDEX,
                    RECEIVER_ORIGIN_X,
                    RECEIVER_SCALE_X,
                    RECEIVER_ORIGIN_Y,
                    RECEIVER_SCALE_Y,
                ),
                self._zeros_colors,
                colors,
                self._remaining()[:, None],
            ),
            dim=-1,
        )

    def privileged_observation(self) -> torch.Tensor:
        """Full state for the centralized critic, regardless of obs_mode."""
        return torch.cat(
            (
                self._pose_features(
                    SENDER_INDEX,
                    SENDER_ORIGIN_X,
                    SENDER_SCALE_X,
                    SENDER_ORIGIN_Y,
                    SENDER_SCALE_Y,
                ),
                self.arrow_direction.float(),
                self._color_eye[self.door_color],
                self._remaining()[:, None],
                self._pose_features(
                    RECEIVER_INDEX,
                    RECEIVER_ORIGIN_X,
                    RECEIVER_SCALE_X,
                    RECEIVER_ORIGIN_Y,
                    RECEIVER_SCALE_Y,
                ),
                self.arrow_direction.float(),
                self._color_eye[self.door_color],
                self._remaining()[:, None],
            ),
            dim=-1,
        )


def twowaycomm_term(env) -> TwoWayCommAction:
    return env.action_manager.get_term("agents")


def sender_observation(env) -> torch.Tensor:
    return twowaycomm_term(env).sender_observation()


def receiver_observation(env) -> torch.Tensor:
    return twowaycomm_term(env).receiver_observation()


def critic_observation(env) -> torch.Tensor:
    # The critic is centralized and discarded at inference, so it keeps the
    # private halves even when the actors can only see pixels.
    return twowaycomm_term(env).privileged_observation()


def _camera_rgb(env, sensor_name: str) -> torch.Tensor:
    """CNN-shaped RGB: (batch, channels, height, width) in [0, 1]."""
    rgb = env.scene[sensor_name].data.rgb
    if rgb is None:
        raise RuntimeError(f"camera sensor {sensor_name!r} has no RGB data")
    return rgb.permute(0, 3, 1, 2).float() / 255.0


def sender_image_observation(env) -> torch.Tensor:
    return _camera_rgb(env, SENDER_CAMERA_SENSOR)


def receiver_image_observation(env) -> torch.Tensor:
    return _camera_rgb(env, RECEIVER_CAMERA_SENSOR)


def correct_entry(env) -> torch.Tensor:
    term = twowaycomm_term(env)
    term._update_outcome()
    return term.correct_entry


def wrong_entry(env) -> torch.Tensor:
    term = twowaycomm_term(env)
    term._update_outcome()
    return term.wrong_entry


def step_reward_rate(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device) / env.step_dt


def correct_reward_rate(env) -> torch.Tensor:
    return correct_entry(env).float() / env.step_dt


def wrong_reward_rate(env) -> torch.Tensor:
    return wrong_entry(env).float() / env.step_dt


def sender_alignment_reward_rate(env) -> torch.Tensor:
    return twowaycomm_term(env).sender_alignment() / env.step_dt


def timeout_reward_rate(env) -> torch.Tensor:
    term = twowaycomm_term(env)
    term._update_timeout()
    return term.timeout.float() / env.step_dt


def correct_metric(env) -> torch.Tensor:
    return twowaycomm_term(env).correct_entry.float()


def wrong_metric(env) -> torch.Tensor:
    return twowaycomm_term(env).wrong_entry.float()


def timeout_metric(env) -> torch.Tensor:
    return twowaycomm_term(env).timeout.float()


@requires_model_fields("geom_rgba")
def paint_door_colors(env, env_ids) -> None:
    """Reset-mode hook whose decorator is what makes door colour per-world.

    ``ManagerBasedRlEnv.load_managers`` collects ``model_fields`` from the event
    terms and calls ``sim.expand_model_fields``, turning the shared
    ``(1, ngeom, 4)`` colour array into a real ``(num_envs, ngeom, 4)`` one.
    Reset events fire before ``ActionManager.reset``, so this repaints the
    previous episode's colour; the action term repaints the new one immediately
    afterwards, still before ``sim.forward()`` and the first observation.
    """
    twowaycomm_term(env).write_door_colors(env_ids)
