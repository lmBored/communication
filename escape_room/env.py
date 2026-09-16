"""Construction and registration helpers for the mjlab environment."""

import torch
from mjlab.envs import ManagerBasedRlEnv

from escape_room.consts import DEFAULT_CUBE_LAYOUT, DEFAULT_GAME_BACKEND
from escape_room.env_cfg import (
    DEFAULT_NCONMAX,
    DEFAULT_NJMAX,
    DEFAULT_NUM_ENVS,
    DEFAULT_PHYSICS_SUBSTEPS,
    DEFAULT_SOLVER_ITERATIONS,
    DEFAULT_SOLVER_LS_ITERATIONS,
    ENV_ID,
    escape_room_env_cfg,
)


class EscapeRoomRlEnv(ManagerBasedRlEnv):
    """Manager environment optimized for direct root-state observations.

    The escape room has no MuJoCo sensor observations, and all dynamic objects
    read by the game layer are root free bodies. Their post-integration world
    poses are therefore available directly in qpos, so the extra forward and
    sense graphs in the generic manager step are unnecessary during training.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._game_term = self.action_manager.get_term("game")
        self._fast_env_ids = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self.reset_time_outs = self.termination_manager.time_outs
        self.reset_terminated = self.termination_manager.terminated
        self.reset_buf = self.reset_time_outs

    def step(self, action: torch.Tensor):
        if not self.cfg.auto_reset and torch.any(self._manual_reset_pending):
            pending_ids = self._manual_reset_pending.nonzero(
                as_tuple=False
            ).squeeze(-1)
            raise RuntimeError(
                f"Environments {pending_ids.cpu().tolist()} must be reset via "
                "reset(env_ids=...) before calling step() again when auto_reset=False."
            )

        self.extras["log"] = dict()
        self._game_term.process_actions(action)

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._game_term.apply_actions()
            self.sim.step()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        time_out = self.common_step_counter % self.max_episode_length == 0
        self.reset_time_outs.fill_(time_out)
        self.reset_terminated.zero_()
        self.reset_buf = self.reset_time_outs
        self.termination_manager._term_dones["time_out"].fill_(time_out)
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        reset_env_ids = self._fast_env_ids if time_out else None
        if self.cfg.auto_reset and reset_env_ids is not None:
            self._reset_idx(reset_env_ids)

        self.obs_buf = {"actor": self._game_term.observation()}
        self.observation_manager._obs_buffer = self.obs_buf

        if not self.cfg.auto_reset and reset_env_ids is not None:
            self._manual_reset_pending[reset_env_ids] = True

        return (
            self.obs_buf,
            self.reward_buf,
            self.reset_terminated,
            self.reset_time_outs,
            self.extras,
        )


def make_env(
    num_envs: int = DEFAULT_NUM_ENVS,
    device: str = "cuda:0",
    seed: int = 42,
    render_mode: str | None = None,
    play: bool = False,
    viewer_width: int | None = None,
    viewer_height: int | None = None,
    physics_substeps: int = DEFAULT_PHYSICS_SUBSTEPS,
    nconmax: int | None = DEFAULT_NCONMAX,
    njmax: int | None = DEFAULT_NJMAX,
    solver_iterations: int = DEFAULT_SOLVER_ITERATIONS,
    solver_ls_iterations: int = DEFAULT_SOLVER_LS_ITERATIONS,
    broadphase: str | None = None,
    cube_layout: str = DEFAULT_CUBE_LAYOUT,
    game_backend: str = DEFAULT_GAME_BACKEND,
    fast_step: bool | None = None,
):
    """Construct the primary MuJoCo-Warp environment."""
    cfg = escape_room_env_cfg(
        play=play,
        num_envs=num_envs,
        seed=seed,
        physics_substeps=physics_substeps,
        nconmax=nconmax,
        njmax=njmax,
        solver_iterations=solver_iterations,
        solver_ls_iterations=solver_ls_iterations,
        broadphase=broadphase,
        cube_layout=cube_layout,
        game_backend=game_backend,
    )
    if viewer_width is not None:
        cfg.viewer.width = viewer_width
    if viewer_height is not None:
        cfg.viewer.height = viewer_height
    if fast_step is None:
        fast_step = not play and render_mode is None
    env_cls = EscapeRoomRlEnv if fast_step else ManagerBasedRlEnv
    return env_cls(cfg=cfg, device=device, render_mode=render_mode)


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
