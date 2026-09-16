import math

import pytest
import torch

from escape_room.communication.action import LEFT, RIGHT
from escape_room.communication.env import CommunicationRlEnv, make_env
from escape_room.communication.env_cfg import (
    ACTION_DIM,
    CONTROL_DT,
    EPISODE_STEPS,
    RECEIVER_OBS_DIM,
    SENDER_OBS_DIM,
    communication_env_cfg,
)
from escape_room.communication.scene import (
    CAMERA_FOV_DEGREES,
    CLUE_SIGN_NAME,
    DECISION_CENTER_X,
    DOOR_CENTERS,
    ENTRY_Y,
    RECEIVER_CAMERA_NAME,
    RECEIVER_SPAWN,
    SENDER_CAMERA_NAME,
    SENDER_SPAWN,
)


def test_communication_config_keeps_private_observations_separate():
    cfg = communication_env_cfg(num_envs=4)

    assert tuple(cfg.observations) == ("sender", "receiver", "critic")
    assert cfg.observations["sender"].nan_policy == "error"
    assert cfg.observations["receiver"].nan_policy == "error"
    assert cfg.actions["receiver"].entity_name == "receiver"
    assert cfg.episode_length_s == pytest.approx(EPISODE_STEPS * CONTROL_DT)
    assert cfg.scale_rewards_by_dt


def test_spawn_geometry_separates_rooms_and_puts_both_doors_in_view():
    assert SENDER_SPAWN[0] < DECISION_CENTER_X - 5.0
    camera_half_angle = math.radians(CAMERA_FOV_DEGREES * 0.5)

    for door_x, door_y in DOOR_CENTERS:
        offset_x = door_x - RECEIVER_SPAWN[0]
        offset_y = door_y - RECEIVER_SPAWN[1]
        assert offset_y > 0.0
        assert abs(math.atan2(offset_x, offset_y)) < camera_half_angle
        assert door_y == ENTRY_Y


def test_reset_exposes_cameras_private_observations_and_balanced_clues():
    env = make_env(num_envs=8, device="cpu", seed=7)
    try:
        assert isinstance(env, CommunicationRlEnv)
        obs, _ = env.reset()
        term = env.action_manager.get_term("receiver")

        assert obs["sender"].shape == (8, SENDER_OBS_DIM)
        assert obs["receiver"].shape == (8, RECEIVER_OBS_DIM)
        assert obs["critic"].shape == (8, SENDER_OBS_DIM + RECEIVER_OBS_DIM)
        assert set(term.clue_direction.tolist()) == {LEFT, RIGHT}
        assert torch.count_nonzero(term.clue_direction == LEFT) == 4
        assert torch.count_nonzero(term.clue_direction == RIGHT) == 4
        assert torch.equal(obs["sender"][:, 0].long(), term.clue_direction)
        receiver_before = obs["receiver"].clone()
        term.set_clue_direction(-term.clue_direction)
        assert torch.allclose(term.receiver_observation(), receiver_before)
        assert torch.equal(term.sender_observation()[:, 0].long(), term.clue_direction)
        assert env.scene["sender"].find_cameras(SENDER_CAMERA_NAME)[1]
        assert env.scene["receiver"].find_cameras(RECEIVER_CAMERA_NAME)[1]
        assert env.scene[CLUE_SIGN_NAME].data.indexing.mocap_id is not None
    finally:
        env.close()


def test_clue_sign_rotation_and_sender_pose_match_private_clue():
    env = make_env(num_envs=6, device="cpu", seed=3)
    try:
        env.reset()
        term = env.action_manager.get_term("receiver")
        sign_quat = env.sim.data.mocap_quat[:, term.sign_mocap_id]

        assert torch.allclose(
            env.scene["sender"].data.body_link_pos_w[:, 1],
            torch.tensor(SENDER_SPAWN).expand(6, -1),
        )
        assert torch.allclose(sign_quat[:, 0], (term.clue_direction == RIGHT).float())
        assert torch.allclose(sign_quat[:, 3].abs(), (term.clue_direction == LEFT).float())
    finally:
        env.close()


def test_receiver_action_mapping_and_fixed_sender():
    env = make_env(num_envs=2, device="cpu", seed=5)
    try:
        env.reset()
        sender_pose = env.scene["sender"].data.root_link_pose_w.clone()
        term = env.action_manager.get_term("receiver")
        term.process_actions(torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]))
        term.apply_actions()

        velocity = term.receiver_velocity()
        assert velocity[0, 1] == pytest.approx(term.cfg.max_speed)
        assert velocity[0, 5] == pytest.approx(0.5 * term.cfg.max_yaw_rate)
        assert velocity[1, 0] == pytest.approx(term.cfg.max_speed)
        assert velocity[1, 5] == pytest.approx(-0.5 * term.cfg.max_yaw_rate)

        env.step(torch.zeros(2, ACTION_DIM))
        assert torch.allclose(env.scene["sender"].data.root_link_pose_w, sender_pose)
    finally:
        env.close()


def test_partial_reset_changes_only_requested_worlds():
    env = make_env(num_envs=4, device="cpu", seed=13)
    try:
        env.reset()
        term = env.action_manager.get_term("receiver")
        before_clue = term.clue_direction.clone()
        before_pose = term.receiver_pose().clone()
        term.set_receiver_pose(torch.tensor([5.0, 5.0, 0.5]), env_ids=torch.tensor([2]))

        env.reset(env_ids=torch.tensor([1, 3]))

        assert term.clue_direction[0] == before_clue[0]
        assert term.clue_direction[2] == before_clue[2]
        assert torch.allclose(term.receiver_pose()[0], before_pose[0])
        assert torch.allclose(term.receiver_pose()[2, :2], torch.tensor([5.0, 5.0]))
        assert torch.allclose(
            term.receiver_pose()[torch.tensor([1, 3]), :3],
            torch.tensor(RECEIVER_SPAWN).expand(2, -1),
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    ("target", "entry_x", "expected_reward", "correct", "wrong"),
    [
        (LEFT, DOOR_CENTERS[0][0], 0.99, True, False),
        (RIGHT, DOOR_CENTERS[0][0], -10.01, False, True),
    ],
)
def test_terminal_entry_rewards_and_outcomes(
    target, entry_x, expected_reward, correct, wrong
):
    env = make_env(num_envs=1, device="cpu", seed=19, auto_reset=False)
    try:
        env.reset()
        term = env.action_manager.get_term("receiver")
        term.set_clue_direction(torch.tensor([target]))
        term.set_receiver_pose(torch.tensor([entry_x, ENTRY_Y + 0.75, 0.5]))
        env.sim.forward()

        _, reward, terminated, truncated, _ = env.step(torch.zeros(1, ACTION_DIM))

        assert reward.item() == pytest.approx(expected_reward)
        assert terminated.item()
        assert not truncated.item()
        assert term.correct_entry.item() is correct
        assert term.wrong_entry.item() is wrong
    finally:
        env.close()


def test_timeout_has_large_exact_penalty_and_distinct_outcome():
    env = make_env(num_envs=1, device="cpu", seed=23, auto_reset=False)
    try:
        env.reset()
        term = env.action_manager.get_term("receiver")
        env.episode_length_buf[:] = env.max_episode_length - 1

        _, reward, terminated, truncated, _ = env.step(torch.zeros(1, ACTION_DIM))

        assert reward.item() == pytest.approx(-20.01)
        assert not terminated.item()
        assert truncated.item()
        assert term.timeout.item()
        assert not term.correct_entry.item()
        assert not term.wrong_entry.item()
    finally:
        env.close()