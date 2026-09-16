import pytest
import torch

from escape_room.consts import PARTNER_BONUS_MULT, REWARD_PER_DIST, SLACK_REWARD
from escape_room.env import make_env
from escape_room.mjlab_env import game_term


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]])


def test_inactive_cube_slots_do_not_fill_contact_buffer():
    env = make_env(num_envs=1, device="cpu", seed=7, play=True)
    try:
        env.reset()
        env.step(torch.zeros(1, 8))

        assert int(env.sim.data.nacon[0]) < 32
    finally:
        env.close()


def test_grab_acquire_move_release_and_reset():
    env = make_env(num_envs=1, device="cpu", seed=7, play=True)
    try:
        env.reset()
        game = game_term(env)
        env_id = torch.tensor([0], dtype=torch.long)
        for agent in game._agents:
            agent.data.write_root_pose(_pose(0.0, 1.0, 0.5), env_id)
            agent.data.write_root_velocity(torch.zeros(1, 6), env_id)
        game._cubes[0].data.write_root_pose(_pose(0.0, 2.5, 0.8), env_id)
        game._cubes[0].data.write_root_velocity(torch.zeros(1, 6), env_id)
        game.entity_active.zero_()
        env.sim.forward()

        grab_both = torch.zeros(1, 8)
        grab_both[0, 3] = 1.0
        grab_both[0, 7] = 1.0
        game.process_actions(grab_both)
        assert game.held_cube.tolist() == [[-1, -1]]

        game.process_actions(torch.zeros(1, 8))
        game.entity_active[0, 0, 2] = True
        game.process_actions(grab_both)
        assert game.held_cube.tolist() == [[0, -1]]

        game.apply_actions()
        env.sim.forward()
        carried = game._cubes[0].data.root_link_pos_w[0]
        assert torch.allclose(carried[:2], torch.tensor([0.0, 2.25]), atol=1.0e-4)

        game.process_actions(torch.zeros(1, 8))
        release = torch.zeros(1, 8)
        release[0, 3] = 1.0
        game.process_actions(release)
        assert game.held_cube.tolist() == [[-1, -1]]

        game.held_cube[0, 0] = 0
        env.reset(env_ids=env_id)
        assert game.held_cube.tolist() == [[-1, -1]]
    finally:
        env.close()


def test_combined_reward_preserves_progress_slack_and_partner_formula():
    env = make_env(num_envs=1, device="cpu", seed=7, play=True)
    try:
        env.reset()
        game = game_term(env)
        env_id = torch.tensor([0], dtype=torch.long)
        game._agents[0].data.write_root_pose(_pose(0.0, 2.0, 0.5), env_id)
        game._agents[1].data.write_root_pose(_pose(0.0, 3.0, 0.5), env_id)
        game.progress_max_y[0] = torch.tensor([1.0, 2.0])
        game._invalidate_cache()

        reward = env.reward_manager.compute(dt=env.step_dt)

        expected = (
            REWARD_PER_DIST
            + REWARD_PER_DIST * PARTNER_BONUS_MULT
            + SLACK_REWARD
        )
        assert reward.item() == pytest.approx(expected)
    finally:
        env.close()