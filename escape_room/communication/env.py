"""Construction and registration for the communication environment."""

from mjlab.envs import ManagerBasedRlEnv

from escape_room.communication.env_cfg import (
    DEFAULT_NCONMAX,
    DEFAULT_NJMAX,
    DEFAULT_NUM_ENVS,
    DEFAULT_PHYSICS_SUBSTEPS,
    DEFAULT_SEED,
    ENV_ID,
    communication_env_cfg,
)


class CommunicationRlEnv(ManagerBasedRlEnv):
    """Independent two-agent communication task."""


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
    auto_reset: bool = True,
) -> CommunicationRlEnv:
    """Construct the communication scenario."""
    cfg = communication_env_cfg(
        play=play,
        num_envs=num_envs,
        seed=seed,
        physics_substeps=physics_substeps,
        nconmax=nconmax,
        njmax=njmax,
    )
    cfg.auto_reset = auto_reset
    if viewer_width is not None:
        cfg.viewer.width = viewer_width
    if viewer_height is not None:
        cfg.viewer.height = viewer_height
    return CommunicationRlEnv(cfg=cfg, device=device, render_mode=render_mode)


def register_env() -> None:
    """Register the scenario with Gymnasium under its own environment ID."""
    try:
        import gymnasium as gym

        if ENV_ID not in gym.registry:
            gym.register(
                id=ENV_ID,
                entry_point="escape_room.communication.env:CommunicationRlEnv",
                kwargs={"cfg": communication_env_cfg()},
                disable_env_checker=True,
            )
    except ImportError:
        pass


register_env()