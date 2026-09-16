"""mjlab configuration for the two-agent communication environment."""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from escape_room.communication.action import (
    ACTION_DIM,
    RECEIVER_OBS_DIM,
    SENDER_OBS_DIM,
    CommunicationActionCfg,
    correct_entry,
    correct_metric,
    correct_reward_rate,
    critic_observation,
    receiver_observation,
    sender_observation,
    step_reward_rate,
    timeout_metric,
    timeout_reward_rate,
    wrong_entry,
    wrong_metric,
    wrong_reward_rate,
)
from escape_room.communication.scene import make_scene_entities

ENV_ID = "EscapeRoomCommunication-v0"
CONTROL_DT = 0.04
EPISODE_STEPS = 100
DEFAULT_NUM_ENVS = 4096
DEFAULT_SEED = 42
DEFAULT_PHYSICS_SUBSTEPS = 1
DEFAULT_NCONMAX = 24
DEFAULT_NJMAX = 96


@dataclass
class DirectDialModelCfg(RslRlModelCfg):
    sender_hidden_dims: tuple[int, ...] = (64, 64)
    message_dim: int = 2


def communication_env_cfg(
    play: bool = False,
    num_envs: int = DEFAULT_NUM_ENVS,
    seed: int = DEFAULT_SEED,
    physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
    nconmax: int | None = DEFAULT_NCONMAX,
    njmax: int | None = DEFAULT_NJMAX,
) -> ManagerBasedRlEnvCfg:
    """Build the independently selectable communication task."""
    if physics_substeps < 1:
        raise ValueError("physics_substeps must be at least 1")
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=make_scene_entities(),
            num_envs=num_envs,
            env_spacing=0.0,
        ),
        observations={
            "sender": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=sender_observation)},
                concatenate_terms=True,
                nan_policy="error",
            ),
            "receiver": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=receiver_observation)},
                concatenate_terms=True,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=critic_observation)},
                concatenate_terms=True,
                nan_policy="error",
            ),
        },
        actions={
            "receiver": CommunicationActionCfg(entity_name="receiver"),
        },
        rewards={
            "step": RewardTermCfg(func=step_reward_rate, weight=-0.01),
            "correct": RewardTermCfg(func=correct_reward_rate, weight=1.0),
            "wrong": RewardTermCfg(func=wrong_reward_rate, weight=-10.0),
            "timeout": RewardTermCfg(func=timeout_reward_rate, weight=-20.0),
        },
        terminations={
            "correct": TerminationTermCfg(func=correct_entry),
            "wrong": TerminationTermCfg(func=wrong_entry),
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        metrics={
            "correct": MetricsTermCfg(func=correct_metric, reduce="last"),
            "wrong": MetricsTermCfg(func=wrong_metric, reduce="last"),
            "timeout": MetricsTermCfg(func=timeout_metric, reduce="last"),
        },
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(timestep=CONTROL_DT / physics_substeps),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(-1.0, 2.0, 0.0),
            distance=25.0,
            elevation=-65.0,
            azimuth=90.0,
            width=960,
            height=540,
        ),
        decimation=physics_substeps,
        episode_length_s=EPISODE_STEPS * CONTROL_DT,
        seed=seed,
        is_finite_horizon=True,
        scale_rewards_by_dt=True,
    )
    if play:
        cfg.scene.num_envs = 1
    return cfg


def communication_ppo_runner_cfg(message_dim: int = 2) -> RslRlOnPolicyRunnerCfg:
    """Configure a decentralized Direct DIAL actor and centralized critic."""
    return RslRlOnPolicyRunnerCfg(
        actor=DirectDialModelCfg(
            class_name="escape_room.communication.model:DirectDialActor",
            hidden_dims=(256, 256, 256),
            sender_hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
            message_dim=message_dim,
        ),
        critic=RslRlModelCfg(
            class_name="MLPModel",
            hidden_dims=(256, 256, 256),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3.0e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
        ),
        experiment_name="communication",
        logger="tensorboard",
        obs_groups={
            "actor": ("sender", "receiver"),
            "critic": ("critic",),
        },
        clip_actions=1.0,
        num_steps_per_env=16,
        max_iterations=1000,
        save_interval=50,
    )


def get_obs_dims() -> tuple[int, int]:
    return SENDER_OBS_DIM, RECEIVER_OBS_DIM


def get_action_dim() -> int:
    return ACTION_DIM
