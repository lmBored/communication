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
    EscapeRoomActionCfg,
    combined_reward,
    policy_observation,
)
from escape_room.mdp.observations import compute_obs_dim
from escape_room.scene import make_scene_entities


# Observation dimension
OBS_DIM = compute_obs_dim() * 2

# Action dimension
ACTION_DIM = TOTAL_ACTION_DIM

# Environment ID for gym registration
ENV_ID = "EscapeRoom-v0"

# Physics configuration. A single 0.04 s step matches the original environment
# and remains stable because maximum controlled displacement is below an agent
# radius. More substeps are available for integration-sensitive experiments.
CONTROL_DT = DELTA_T  # Control timestep (0.04)
DEFAULT_PHYSICS_SUBSTEPS = 1
PHYSICS_DT = CONTROL_DT / DEFAULT_PHYSICS_SUBSTEPS
DECIMATION = DEFAULT_PHYSICS_SUBSTEPS
DEFAULT_NCONMAX = 96
DEFAULT_NJMAX = 384
DEFAULT_SOLVER_ITERATIONS = 10
DEFAULT_SOLVER_LS_ITERATIONS = 5

# Fuse the observation math with ``torch.compile``. Disabled by default until a
# measured win is confirmed on the target GPU; enable with ``--compile-game``.
DEFAULT_COMPILE_GAME = False

# Profiled default for a 40 GiB A100. The specialized synchronous training step
# leaves about 8 GiB free with 32768 worlds and a 16-step rollout. Larger world
# counts or a 32-step rollout leave too little headroom for robust submissions.
DEFAULT_NUM_ENVS = 32768

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
    physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
    nconmax: int | None = DEFAULT_NCONMAX,
    njmax: int | None = DEFAULT_NJMAX,
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS,
    solver_ls_iterations: int = DEFAULT_SOLVER_LS_ITERATIONS,
    broadphase: str | None = None,
    compile_game: bool = DEFAULT_COMPILE_GAME,
) -> ManagerBasedRlEnvCfg:
    """Build the real MuJoCo-Warp manager environment configuration."""
    if physics_substeps < 1:
        raise ValueError("physics_substeps must be at least 1")
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
                nan_policy="disabled",
            ),
        },
        actions={
            "game": EscapeRoomActionCfg(
                entity_name="agent_0", compile_game=compile_game
            )
        },
        rewards={
            "total": RewardTermCfg(func=combined_reward, weight=1.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=time_out, time_out=True),
        },
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            broadphase=broadphase,
            mujoco=MujocoCfg(
                timestep=CONTROL_DT / physics_substeps,
                iterations=solver_iterations,
                ls_iterations=solver_ls_iterations,
            ),
        ),
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 20.0, 0.0),
            distance=46.0,
            elevation=-58.0,
            azimuth=90.0,
            width=960,
            height=720,
        ),
        decimation=physics_substeps,
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
                "std_type": "log",
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
        obs_groups={"actor": ("actor",), "critic": ("actor",)},
        clip_actions=1.0,
        num_steps_per_env=16,
        max_iterations=1000,
        save_interval=50,
    )


def get_env_config(num_envs: int = DEFAULT_NUM_ENVS, seed: int = DEFAULT_SEED):
    """Compatibility alias returning the runnable mjlab configuration."""
    return escape_room_env_cfg(num_envs=num_envs, seed=seed)
