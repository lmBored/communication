import pytest

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