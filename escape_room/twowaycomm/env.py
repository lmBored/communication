"""Construction and registration for the two-way communication environment."""

from mjlab.envs import ManagerBasedRlEnv

from escape_room.twowaycomm.env_cfg import (
    DEFAULT_ARROW_LAYOUT,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_OBS_MODE,
    DEFAULT_NCONMAX,
    DEFAULT_NJMAX,
    DEFAULT_NUM_ENVS,
    DEFAULT_PHYSICS_SUBSTEPS,
    DEFAULT_REWARD_SHARING,
    DEFAULT_SEED,
    ENV_ID,
    twowaycomm_env_cfg,
)


class TwoWayCommRlEnv(ManagerBasedRlEnv):
    """Independent two-agent bidirectional communication task."""


def make_env(
    num_envs: int = DEFAULT_NUM_ENVS,
    device: str = "cuda:0",
    seed: int = DEFAULT_SEED,
    render_mode: str | None = None,
    play: bool = False,
    viewer_width: int | None = None,
    viewer_height: int | None = None,
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
    auto_reset: bool = True,
) -> TwoWayCommRlEnv:
    """Construct the two-way communication scenario.

    The first seven parameters match ``communication.env.make_env`` because
    ``play.py`` passes exactly that keyword set for every task.
    """
    cfg = twowaycomm_env_cfg(
        play=play,
        num_envs=num_envs,
        seed=seed,
        physics_substeps=physics_substeps,
        nconmax=nconmax,
        njmax=njmax,
        arrow_layout=arrow_layout,
        reward_sharing=reward_sharing,
        sender_shaping_weight=sender_shaping_weight,
        obs_mode=obs_mode,
        camera_width=camera_width,
        camera_height=camera_height,
        sender_camera_width=sender_camera_width,
        sender_camera_height=sender_camera_height,
    )
    cfg.auto_reset = auto_reset
    if viewer_width is not None:
        cfg.viewer.width = viewer_width
    if viewer_height is not None:
        cfg.viewer.height = viewer_height
    return TwoWayCommRlEnv(cfg=cfg, device=device, render_mode=render_mode)


def register_env() -> None:
    """Register the scenario with Gymnasium under its own environment ID."""
    try:
        import gymnasium as gym

        if ENV_ID not in gym.registry:
            gym.register(
                id=ENV_ID,
                entry_point="escape_room.twowaycomm.env:TwoWayCommRlEnv",
                kwargs={"cfg": twowaycomm_env_cfg()},
                disable_env_checker=True,
            )
    except ImportError:
        pass


register_env()
