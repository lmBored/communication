"""Manager terms for the two-agent communication scenario."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from escape_room.communication.scene import (
    CLUE_SIGN_NAME,
    CLUE_SIGN_POSITION,
    DECISION_CENTER_X,
    DOOR_CENTERS,
    RECEIVER_SPAWN,
    TERMINAL_THRESHOLD_Y,
)

LEFT = -1
RIGHT = 1
ACTION_DIM = 3
SENDER_OBS_DIM = 2
RECEIVER_OBS_DIM = 12


def _yaw_from_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


@dataclass(kw_only=True)
class CommunicationActionCfg(ActionTermCfg):
    """Configuration for the receiver's three physical controls."""

    max_speed: float = 8.0
    max_yaw_rate: float = 4.0

    def build(self, env):
        return CommunicationAction(self, env)


class CommunicationAction(ActionTerm):
    """Own the private clue and control only the mobile receiver."""

    cfg: CommunicationActionCfg

    def __init__(self, cfg: CommunicationActionCfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self._receiver = env.scene["receiver"]
        self._sim_data = self._receiver.data.data
        self._receiver_q = self._receiver.data.indexing.free_joint_q_adr.to(
            device=self.device, dtype=torch.long
        )
        self._receiver_v = self._receiver.data.indexing.free_joint_v_adr.to(
            device=self.device, dtype=torch.long
        )
        sign = env.scene[CLUE_SIGN_NAME]
        assert sign.data.indexing.mocap_id is not None
        self.sign_mocap_id = sign.data.indexing.mocap_id

        self.clue_direction = torch.empty(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.correct_entry = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.wrong_entry = torch.zeros_like(self.correct_entry)
        self.timeout = torch.zeros_like(self.correct_entry)
        self._door_centers = torch.tensor(DOOR_CENTERS, device=self.device)
        self._receiver_spawn = torch.tensor(RECEIVER_SPAWN, device=self.device)
        self._sign_position = torch.tensor(CLUE_SIGN_POSITION, device=self.device)

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

    def _write_sign(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        self._sim_data.mocap_pos[env_ids, self.sign_mocap_id] = (
            self._sign_position.expand(count, -1)
        )
        quat = torch.zeros(count, 4, device=self.device)
        right = self.clue_direction[env_ids] == RIGHT
        quat[right, 0] = 1.0
        quat[~right, 3] = 1.0
        self._sim_data.mocap_quat[env_ids, self.sign_mocap_id] = quat

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        self.clue_direction[ids] = self._balanced_directions(ids.numel())
        self.correct_entry[ids] = False
        self.wrong_entry[ids] = False
        self.timeout[ids] = False
        self._raw_actions[ids] = 0.0
        self._processed_actions[ids] = 0.0
        self.set_receiver_pose(self._receiver_spawn, ids)
        self._write_sign(ids)

    def set_clue_direction(self, direction: torch.Tensor, env_ids=None) -> None:
        ids = self._ids(env_ids)
        direction = direction.to(device=self.device, dtype=torch.long).flatten()
        if direction.numel() == 1:
            direction = direction.expand(ids.numel())
        if direction.shape != ids.shape or not torch.all(
            (direction == LEFT) | (direction == RIGHT)
        ):
            raise ValueError("direction must contain one LEFT/RIGHT value per world")
        self.clue_direction[ids] = direction
        self._write_sign(ids)

    def set_receiver_pose(self, position: torch.Tensor, env_ids=None) -> None:
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
        self._sim_data.qpos[rows, self._receiver_q] = pose
        self._sim_data.qvel[rows, self._receiver_v] = 0.0

    def receiver_pose(self) -> torch.Tensor:
        return self._sim_data.qpos[:, self._receiver_q]

    def receiver_velocity(self) -> torch.Tensor:
        return self._sim_data.qvel[:, self._receiver_v]

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        torch.clamp(actions, -1.0, 1.0, out=self._processed_actions)

    def apply_actions(self) -> None:
        pose = self.receiver_pose()
        yaw = _yaw_from_quat(pose[:, 3:7])
        forward_action, strafe_action, yaw_action = self._processed_actions.unbind(-1)
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        vx = self.cfg.max_speed * (-forward_action * sin_yaw + strafe_action * cos_yaw)
        vy = self.cfg.max_speed * (forward_action * cos_yaw + strafe_action * sin_yaw)
        velocity = torch.zeros(self.num_envs, 6, device=self.device)
        velocity[:, 0] = vx
        velocity[:, 1] = vy
        velocity[:, 5] = yaw_action * self.cfg.max_yaw_rate
        self._sim_data.qvel[:, self._receiver_v] = velocity

    def _update_outcome(self) -> None:
        position = self.receiver_pose()[:, :3]
        entered = position[:, 1] >= TERMINAL_THRESHOLD_Y
        entered_right = position[:, 0] > DECISION_CENTER_X
        chose_right = self.clue_direction == RIGHT
        self.correct_entry.copy_(entered & (entered_right == chose_right))
        self.wrong_entry.copy_(entered & (entered_right != chose_right))

    def _update_timeout(self) -> None:
        self._update_outcome()
        elapsed = self._env.episode_length_buf >= self._env.max_episode_length
        self.timeout.copy_(elapsed & ~self.correct_entry & ~self.wrong_entry)

    def sender_observation(self) -> torch.Tensor:
        steps_left = (
            self._env.max_episode_length - self._env.episode_length_buf
        ).clamp_min(0)
        remaining = steps_left.float() / self._env.max_episode_length
        return torch.stack((self.clue_direction.float(), remaining), dim=-1)

    def receiver_observation(self) -> torch.Tensor:
        pose = self.receiver_pose()
        position = pose[:, :3]
        yaw = _yaw_from_quat(pose[:, 3:7])
        sin_yaw = torch.sin(yaw)
        cos_yaw = torch.cos(yaw)
        velocity = self.receiver_velocity()
        right = torch.stack((cos_yaw, sin_yaw), dim=-1)
        forward = torch.stack((-sin_yaw, cos_yaw), dim=-1)
        local_forward_velocity = (velocity[:, :2] * forward).sum(-1) / self.cfg.max_speed
        local_strafe_velocity = (velocity[:, :2] * right).sum(-1) / self.cfg.max_speed

        relative = self._door_centers[None, :, :] - position[:, None, :2]
        door_right = (relative * right[:, None, :]).sum(-1) / 10.0
        door_forward = (relative * forward[:, None, :]).sum(-1) / 10.0
        doors = torch.stack((door_right, door_forward), dim=-1).flatten(1)
        steps_left = (
            self._env.max_episode_length - self._env.episode_length_buf
        ).clamp_min(0)
        remaining = steps_left.float() / self._env.max_episode_length
        local_position = torch.stack(
            ((position[:, 0] - DECISION_CENTER_X) / 5.0, position[:, 1] / 9.0),
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
                doors,
                remaining[:, None],
            ),
            dim=-1,
        )


def communication_term(env) -> CommunicationAction:
    return env.action_manager.get_term("receiver")


def sender_observation(env) -> torch.Tensor:
    return communication_term(env).sender_observation()


def receiver_observation(env) -> torch.Tensor:
    return communication_term(env).receiver_observation()


def critic_observation(env) -> torch.Tensor:
    term = communication_term(env)
    return torch.cat((term.sender_observation(), term.receiver_observation()), dim=-1)


def correct_entry(env) -> torch.Tensor:
    term = communication_term(env)
    term._update_outcome()
    return term.correct_entry


def wrong_entry(env) -> torch.Tensor:
    term = communication_term(env)
    term._update_outcome()
    return term.wrong_entry


def step_reward_rate(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device) / env.step_dt


def correct_reward_rate(env) -> torch.Tensor:
    return correct_entry(env).float() / env.step_dt


def wrong_reward_rate(env) -> torch.Tensor:
    return wrong_entry(env).float() / env.step_dt


def timeout_reward_rate(env) -> torch.Tensor:
    term = communication_term(env)
    term._update_timeout()
    return term.timeout.float() / env.step_dt


def correct_metric(env) -> torch.Tensor:
    return communication_term(env).correct_entry.float()


def wrong_metric(env) -> torch.Tensor:
    return communication_term(env).wrong_entry.float()


def timeout_metric(env) -> torch.Tensor:
    return communication_term(env).timeout.float()