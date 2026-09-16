"""The fused Warp kernels must reproduce the eager PyTorch game systems."""

import torch

from escape_room.consts import (
    PARTNER_BONUS_MULT,
    REWARD_PER_DIST,
    SLACK_REWARD,
)
from escape_room.env import make_env
from escape_room.mjlab_env import game_term
from escape_room.warp_game import WarpGame


def _warm_state(env, steps: int = 12) -> torch.Tensor:
    """Drive the env with reproducible actions into a non-trivial state."""
    torch.manual_seed(7)
    actions = 2.0 * torch.rand(steps, env.num_envs, 8) - 1.0
    for step in range(steps):
        env.step(actions[step])
    return actions[-1]


def test_warp_observation_and_reward_match_torch_backend():
    env = make_env(num_envs=3, device="cpu", seed=5, game_backend="torch")
    try:
        env.reset()
        _warm_state(env)
        game = game_term(env)
        warp = WarpGame(game)

        torch_obs = game.observation().clone()
        warp_obs = warp.observe().clone()
        assert torch.allclose(torch_obs, warp_obs, atol=1.0e-5)

        max_y = game.progress_max_y.clone()
        torch_reward = game.combined_reward().clone()
        game.progress_max_y.copy_(max_y)
        warp_reward = warp.compute_reward(
            REWARD_PER_DIST, REWARD_PER_DIST * PARTNER_BONUS_MULT, SLACK_REWARD
        ).clone()
        assert torch.allclose(torch_reward, warp_reward, atol=1.0e-6)
        assert torch.allclose(game.progress_max_y, max_y.maximum(game.progress_max_y))
    finally:
        env.close()


def test_warp_buttons_doors_and_grab_match_torch_backend():
    env = make_env(num_envs=3, device="cpu", seed=5, game_backend="torch")
    try:
        env.reset()
        actions = _warm_state(env)
        game = game_term(env)
        warp = WarpGame(game)

        grab = actions.clone()
        grab[:, 3] = 1.0
        grab[:, 7] = 1.0
        snapshot = {
            name: getattr(game, name).clone()
            for name in (
                "button_pressed",
                "door_open",
                "door_pos",
                "held_cube",
                "_grab_was_down",
                "entity_pos",
            )
        }

        game.process_actions(grab)
        expected = {
            name: getattr(game, name).clone() for name in snapshot
        }

        for name, value in snapshot.items():
            getattr(game, name).copy_(value)
        game._invalidate_cache()
        torch.clamp(grab, -1.0, 1.0, out=game._processed_actions)
        warp.pre_step()

        assert torch.equal(game.button_pressed, expected["button_pressed"])
        assert torch.equal(game.door_open, expected["door_open"])
        assert torch.equal(game.held_cube, expected["held_cube"])
        assert torch.equal(game._grab_was_down, expected["_grab_was_down"])
        assert torch.allclose(game.door_pos, expected["door_pos"], atol=1.0e-6)
        assert torch.allclose(game.entity_pos, expected["entity_pos"], atol=1.0e-6)
    finally:
        env.close()


def test_warp_backend_runs_the_full_environment_step():
    env = make_env(num_envs=2, device="cpu", seed=5, game_backend="warp")
    try:
        obs, _ = env.reset()
        assert obs["actor"].shape == (2, 188)
        for _ in range(5):
            obs, reward, _, _, _ = env.step(torch.zeros(2, 8))
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(reward).all()
    finally:
        env.close()
