import pytest
import torch

from escape_room.env import EscapeRoomRlEnv, make_env
from escape_room.env_cfg import CONTROL_DT, escape_room_env_cfg, escape_room_ppo_runner_cfg


def test_fast_training_config_reuses_actor_observations_for_critic():
    cfg = escape_room_env_cfg(num_envs=4)
    runner_cfg = escape_room_ppo_runner_cfg()

    assert tuple(cfg.observations) == ("actor",)
    assert cfg.observations["actor"].nan_policy == "disabled"
    assert runner_cfg.obs_groups == {"actor": ("actor",), "critic": ("actor",)}
    assert runner_cfg.actor.distribution_cfg["std_type"] == "log"
    assert cfg.decimation == 1
    assert cfg.sim.mujoco.timestep == pytest.approx(CONTROL_DT)
    assert cfg.sim.nconmax == 96
    assert cfg.sim.njmax == 384
    assert cfg.sim.mujoco.iterations == 10
    assert cfg.sim.mujoco.ls_iterations == 5


def test_physics_substeps_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        escape_room_env_cfg(physics_substeps=0)


def test_fast_training_step_returns_finite_observations_and_synchronous_timeout():
    env = make_env(num_envs=2, device="cpu", seed=11)
    try:
        assert isinstance(env, EscapeRoomRlEnv)
        obs, _ = env.reset()
        assert torch.isfinite(obs["actor"]).all()

        action = torch.zeros(2, 8)
        for _ in range(env.max_episode_length - 1):
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not truncated.any()
        obs, reward, terminated, truncated, _ = env.step(action)

        assert obs["actor"].shape == (2, 188)
        assert torch.isfinite(obs["actor"]).all()
        assert torch.isfinite(reward).all()
        assert not terminated.any()
        assert truncated.all()
    finally:
        env.close()