"""mjlab configuration for the two-way communication environment."""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import time_out
from mjlab.envs.mdp.events import reset_scene_to_default
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from escape_room.twowaycomm.action import (
    ACTION_DIM,
    ACTION_DIM_PER_AGENT,
    CRITIC_OBS_DIM,
    RECEIVER_CAMERA_SENSOR,
    RECEIVER_OBS_DIM,
    SENDER_CAMERA_SENSOR,
    SENDER_OBS_DIM,
    TwoWayCommActionCfg,
    correct_entry,
    correct_metric,
    correct_reward_rate,
    critic_observation,
    paint_door_colors,
    receiver_image_observation,
    receiver_observation,
    sender_alignment_reward_rate,
    sender_image_observation,
    sender_observation,
    step_reward_rate,
    timeout_metric,
    timeout_reward_rate,
    wrong_entry,
    wrong_metric,
    wrong_reward_rate,
)
from escape_room.twowaycomm.scene import (
    ARROW_LAYOUTS,
    DEFAULT_ARROW_LAYOUT,
    RECEIVER_CAMERA_NAME,
    RECEIVER_NAME,
    SENDER_CAMERA_NAME,
    SENDER_NAME,
    make_scene_entities,
)

ENV_ID = "EscapeRoomTwoWayComm-v0"
CONTROL_DT = 0.04
EPISODE_STEPS = 100
DEFAULT_NUM_ENVS = 4096
DEFAULT_SEED = 42
DEFAULT_PHYSICS_SUBSTEPS = 1
# Twice the one-way scenario's 24/96: this scene has two mobile agents instead
# of one, each generating floor and wall contacts. Reasoned, not yet measured —
# run scripts/sim_bench.py and record the result in docs/sps.md.
DEFAULT_NCONMAX = 48
DEFAULT_NJMAX = 192

REWARD_SHARING_MODES: tuple[str, ...] = ("shared", "receiver_only")
DEFAULT_REWARD_SHARING = "shared"
OBS_MODES: tuple[str, ...] = ("vector", "pixel")
DEFAULT_OBS_MODE = "vector"
# Rendering costs num_envs * W * H rays every step, so resolution is the knob
# that decides whether pixel mode fits. The two agents need very different
# resolutions: the receiver only has to tell three door colours apart, which is
# a large flat region, while the sender has to read the direction of an arrow
# that subtends ~5.6% of the frame width. Measured on the rendered frames, the
# arrow glyph is ~1.8 px at 32x32 (unreadable) and ~14 px at 256x256.
DEFAULT_CAMERA_WIDTH = 64
DEFAULT_CAMERA_HEIGHT = 64
DEFAULT_SENDER_CAMERA_WIDTH = 192
DEFAULT_SENDER_CAMERA_HEIGHT = 192


@dataclass
class TwoWayDialModelCfg(RslRlModelCfg):
    """Actor configuration; the extra fields reach TwoWayDialActor as kwargs."""

    sender_hidden_dims: tuple[int, ...] = (256, 256, 256)
    sender_message_hidden_dims: tuple[int, ...] = (64, 64)
    receiver_message_hidden_dims: tuple[int, ...] = (64, 64)
    message_dim: int = 2
    sender_message_dim: int | None = None
    receiver_message_dim: int | None = None
    channel_mode: str = "same_step"
    delayed_message_feedback: bool = True
    message_noise_std: float = 0.0


def twowaycomm_env_cfg(
    play: bool = False,
    num_envs: int = DEFAULT_NUM_ENVS,
    seed: int = DEFAULT_SEED,
    physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
    nconmax: int | None = DEFAULT_NCONMAX,
    njmax: int | None = DEFAULT_NJMAX,
    arrow_layout: str = DEFAULT_ARROW_LAYOUT,
    reward_sharing: str = DEFAULT_REWARD_SHARING,
    sender_shaping_weight: float = 0.0,
    obs_mode: str = DEFAULT_OBS_MODE,
    camera_width: int = DEFAULT_CAMERA_WIDTH,
    camera_height: int = DEFAULT_CAMERA_HEIGHT,
    sender_camera_width: int | None = None,
    sender_camera_height: int | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the two-way communication task.

    ``reward_sharing="shared"`` (the default) leaves the outcome reward exactly
    as the one-way scenario defines it: one scalar per world per step, from
    which both agents' branches are optimized. ``"receiver_only"`` additionally
    forces every sender-originated shaping term off, so nothing but the
    receiver's door entry can produce reward.
    """
    if physics_substeps < 1:
        raise ValueError("physics_substeps must be at least 1")
    if arrow_layout not in ARROW_LAYOUTS:
        raise ValueError(
            f"unknown arrow layout {arrow_layout!r}; choose from {ARROW_LAYOUTS}"
        )
    if reward_sharing not in REWARD_SHARING_MODES:
        raise ValueError(
            f"unknown reward_sharing {reward_sharing!r}; "
            f"choose from {REWARD_SHARING_MODES}"
        )
    if sender_shaping_weight < 0.0:
        raise ValueError("sender_shaping_weight must be non-negative")
    if obs_mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {OBS_MODES}")

    rewards = {
        "step": RewardTermCfg(func=step_reward_rate, weight=-0.01),
        "correct": RewardTermCfg(func=correct_reward_rate, weight=1.0),
        "wrong": RewardTermCfg(func=wrong_reward_rate, weight=-10.0),
        "timeout": RewardTermCfg(func=timeout_reward_rate, weight=-20.0),
    }
    # Off by default, which keeps the reward function identical to the one-way
    # scenario. Turning it on is what makes the sender's own motion matter, and
    # therefore what gives the back-channel something to be useful for.
    if reward_sharing == "shared" and sender_shaping_weight > 0.0:
        rewards["sender_alignment"] = RewardTermCfg(
            func=sender_alignment_reward_rate, weight=sender_shaping_weight
        )

    cfg = ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities=make_scene_entities(arrow_layout),
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
            "agents": TwoWayCommActionCfg(
                entity_name=RECEIVER_NAME,
                arrow_layout=arrow_layout,
                obs_mode=obs_mode,
            ),
        },
        # Passing events at all overrides the dataclass default, so the stock
        # reset has to be re-listed or entities stop returning to their
        # initial state.
        events={
            "reset_scene_to_default": EventTermCfg(
                func=reset_scene_to_default, mode="reset"
            ),
            "door_color": EventTermCfg(func=paint_door_colors, mode="reset"),
        },
        rewards=rewards,
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
            lookat=(-3.5, 2.0, 0.0),
            distance=36.0,
            elevation=-75.0,
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
    if obs_mode == "pixel":
        # camera_width/height size the receiver; the sender gets its own, larger
        # default because it has to resolve an arrow rather than a flat colour.
        _add_first_person_cameras(
            cfg,
            sender_size=(
                sender_camera_width or DEFAULT_SENDER_CAMERA_WIDTH,
                sender_camera_height or DEFAULT_SENDER_CAMERA_HEIGHT,
            ),
            receiver_size=(camera_width, camera_height),
        )
    if play:
        cfg.scene.num_envs = 1
    return cfg


def _add_first_person_cameras(
    cfg: ManagerBasedRlEnvCfg,
    sender_size: tuple[int, int],
    receiver_size: tuple[int, int],
) -> None:
    """Register both first-person cameras and publish them as image groups.

    Registering any camera sensor makes ``sim.sense()`` render every world on
    every step, so this is only done for pixel mode; in vector mode the scene
    has no sensors and sensing costs nothing. Resolution is per camera, which
    matters here because the sender needs far more of it than the receiver.
    """
    from mjlab.sensor import CameraSensorCfg

    # mujoco_warp requires these three settings to agree across every camera;
    # resolution and data types may differ.
    shared = dict(
        data_types=("rgb",),
        use_textures=True,
        use_shadows=False,
        enabled_geom_groups=(0, 1, 2),
    )
    cfg.scene.sensors = tuple(cfg.scene.sensors or ()) + (
        CameraSensorCfg(
            name=SENDER_CAMERA_SENSOR,
            camera_name=f"{SENDER_NAME}/{SENDER_CAMERA_NAME}",
            width=sender_size[0],
            height=sender_size[1],
            **shared,
        ),
        CameraSensorCfg(
            name=RECEIVER_CAMERA_SENSOR,
            camera_name=f"{RECEIVER_NAME}/{RECEIVER_CAMERA_NAME}",
            width=receiver_size[0],
            height=receiver_size[1],
            **shared,
        ),
    )
    cfg.observations["sender_image"] = ObservationGroupCfg(
        terms={"rgb": ObservationTermCfg(func=sender_image_observation)},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )
    cfg.observations["receiver_image"] = ObservationGroupCfg(
        terms={"rgb": ObservationTermCfg(func=receiver_image_observation)},
        concatenate_terms=True,
        enable_corruption=False,
        nan_policy="error",
    )


def twowaycomm_ppo_runner_cfg(
    message_dim: int = 2,
    sender_message_dim: int | None = None,
    receiver_message_dim: int | None = None,
    channel_mode: str = "same_step",
    delayed_message_feedback: bool = True,
    obs_mode: str = DEFAULT_OBS_MODE,
) -> RslRlOnPolicyRunnerCfg:
    """Configure the two-way DIAL actor and a centralized critic."""
    if obs_mode not in OBS_MODES:
        raise ValueError(f"unknown obs_mode {obs_mode!r}; choose from {OBS_MODES}")
    actor_groups = ("sender", "receiver")
    cnn_cfg = None
    if obs_mode == "pixel":
        from escape_room.twowaycomm.model import DEFAULT_CNN_CFG

        actor_groups = ("sender", "receiver", "sender_image", "receiver_image")
        cnn_cfg = dict(DEFAULT_CNN_CFG)
    return RslRlOnPolicyRunnerCfg(
        actor=TwoWayDialModelCfg(
            class_name="escape_room.twowaycomm.model:TwoWayDialActor",
            hidden_dims=(256, 256, 256),
            sender_hidden_dims=(256, 256, 256),
            sender_message_hidden_dims=(64, 64),
            receiver_message_hidden_dims=(64, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "log",
            },
            message_dim=message_dim,
            sender_message_dim=sender_message_dim,
            receiver_message_dim=receiver_message_dim,
            channel_mode=channel_mode,
            delayed_message_feedback=delayed_message_feedback,
            cnn_cfg=cnn_cfg,
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
        experiment_name="twowaycomm",
        logger="tensorboard",
        obs_groups={
            "actor": actor_groups,
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
