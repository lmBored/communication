"""Construction and registration helpers for the mjlab environment."""

from escape_room.env_cfg import DEFAULT_NUM_ENVS, ENV_ID, escape_room_env_cfg


def make_env(
    num_envs: int = DEFAULT_NUM_ENVS,
    device: str = "cuda:0",
    seed: int = 42,
    render_mode: str | None = None,
    play: bool = False,
    viewer_width: int | None = None,
    viewer_height: int | None = None,
):
    """Construct the primary MuJoCo-Warp environment."""
    from mjlab.envs import ManagerBasedRlEnv

    cfg = escape_room_env_cfg(play=play, num_envs=num_envs, seed=seed)
    if viewer_width is not None:
        cfg.viewer.width = viewer_width
    if viewer_height is not None:
        cfg.viewer.height = viewer_height
    return ManagerBasedRlEnv(cfg=cfg, device=device, render_mode=render_mode)


def register_env():
    """Register the escape room environment with Gymnasium."""
    try:
        import gymnasium as gym

        if ENV_ID not in gym.registry:
            gym.register(
                id=ENV_ID,
                entry_point="mjlab.envs:ManagerBasedRlEnv",
                kwargs={"cfg": escape_room_env_cfg()},
                disable_env_checker=True,
            )
    except ImportError:
        pass


# Auto-register on import
register_env()
