"""mjlab environment and rsl-rl configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from escape_room.consts import DELTA_T, EPISODE_LEN, TOTAL_ACTION_DIM
from escape_room.mjlab_env import (
    PARTNER_REWARD_WEIGHT,
    PROGRESS_REWARD_WEIGHT,
    SLACK_REWARD_WEIGHT,
    EscapeRoomActionCfg,
    partner_bonus,
    policy_observation,
    progress_reward,
    slack_reward,
)
from escape_room.mdp.observations import compute_obs_dim
from escape_room.scene import make_scene_entities


# Observation dimension
OBS_DIM = compute_obs_dim() * 2

# Action dimension
ACTION_DIM = TOTAL_ACTION_DIM

# Environment ID for gym registration
ENV_ID = "EscapeRoom-v0"

# Physics configuration
PHYSICS_DT = 0.005  # MuJoCo timestep
CONTROL_DT = DELTA_T  # Control timestep (0.04)
DECIMATION = int(CONTROL_DT / PHYSICS_DT)  # 8 substeps per control step

# Conservative default for a 40 GiB A100.  This scene has 24 independently
# batched entities, so the kinematic environment's old 4096-world default does
# not fit the real MuJoCo-Warp contact and CUDA-graph buffers.
DEFAULT_NUM_ENVS = 512

# Seed
DEFAULT_SEED = 42


def get_obs_dim() -> int:
    """Get observation dimension."""
    return OBS_DIM


def get_action_dim() -> int:
    """Get action dimension."""
    return ACTION_DIM


def escape_room_env_cfg(
    play: bool = False,
    num_envs: int = DEFAULT_NUM_ENVS,
    seed: int = DEFAULT_SEED,
) -> ManagerBasedRlEnvCfg:
    """Build the real MuJoCo-Warp manager environment configuration."""
    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=make_scene_entities(),
            num_envs=num_envs,
            env_spacing=0.0,
        ),
        observations={
            "actor": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=policy_observation)},
                concatenate_terms=True,
                nan_policy="error",
            ),
            "critic": ObservationGroupCfg(
                terms={"state": ObservationTermCfg(func=policy_observation)},
                concatenate_terms=True,
                nan_policy="error",
            ),
        },
        actions={"game": EscapeRoomActionCfg(entity_name="agent_0")},
        rewards={
            "progress": RewardTermCfg(
                func=progress_reward, weight=PROGRESS_REWARD_WEIGHT
            ),
            "slack": RewardTermCfg(func=slack_reward, weight=SLACK_REWARD_WEIGHT),
            "partner": RewardTermCfg(
                func=partner_bonus, weight=PARTNER_REWARD_WEIGHT
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        sim=SimulationCfg(mujoco=MujocoCfg(timestep=PHYSICS_DT)),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 20.0, 0.0),
            distance=46.0,
            elevation=-58.0,
            azimuth=90.0,
            width=960,
            height=720,
        ),
        decimation=DECIMATION,
        episode_length_s=EPISODE_LEN * CONTROL_DT,
        seed=seed,
        scale_rewards_by_dt=False,
    )
    if play:
        cfg.scene.num_envs = 1
        cfg.episode_length_s = EPISODE_LEN * CONTROL_DT
    return cfg


def escape_room_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """rsl-rl PPO configuration used by ``scripts/train.py``."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 256, 256),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
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
        experiment_name="escape_room",
        logger="tensorboard",
        clip_actions=1.0,
        num_steps_per_env=16,
        max_iterations=1000,
        save_interval=50,
    )


def get_env_config(num_envs: int = DEFAULT_NUM_ENVS, seed: int = DEFAULT_SEED):
    """Compatibility alias returning the runnable mjlab configuration."""
    return escape_room_env_cfg(num_envs=num_envs, seed=seed)
