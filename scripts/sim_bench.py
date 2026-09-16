#!/usr/bin/env python3
"""
Rollout-only simulation microbenchmark (random actions, no PPO optimization).

The figures printed here are simulation throughput only; they are NOT trainer
SPS. Use `python scripts/train.py` for end-to-end training throughput.

Usage:
    python scripts/sim_bench.py --backend mjlab --num-envs 8192 --num-steps 200
    python scripts/sim_bench.py --backend cartpole --num-envs 8192 --num-steps 200
    python scripts/sim_bench.py --backend kinematic --num-envs 8192 --num-steps 200
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running from repo root without install
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from escape_room.consts import (  # noqa: E402
    ACTION_DIM_PER_AGENT,
    CUBE_SLOT_LAYOUTS,
    DEFAULT_CUBE_LAYOUT,
    DEFAULT_GAME_BACKEND,
    GAME_BACKENDS,
    NUM_AGENTS,
)
from escape_room.env_cfg import (  # noqa: E402
    DEFAULT_NCONMAX,
    DEFAULT_NJMAX,
    DEFAULT_NUM_ENVS,
    DEFAULT_PHYSICS_SUBSTEPS,
    DEFAULT_SOLVER_ITERATIONS,
    DEFAULT_SOLVER_LS_ITERATIONS,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout-only mjwarp SPS benchmark")
    p.add_argument(
        "--backend",
        type=str,
        default="mjlab",
        choices=["mjlab", "cartpole", "kinematic"],
        help=(
            "mjlab = escape room; cartpole = canonical simple mjlab task; "
            "kinematic = tensor-only fallback"
        ),
    )
    p.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20, help="Steps before timing")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument(
        "--physics-substeps",
        type=int,
        default=DEFAULT_PHYSICS_SUBSTEPS,
        help="MuJoCo steps per 0.04 s control step",
    )
    p.add_argument(
        "--step-mode",
        choices=["full", "physics", "reset", "phases"],
        default="full",
        help=(
            "full environment step, raw mjwarp physics graphs only, "
            "all-world resets (to size the reset cost against a step), or "
            "per-phase timings of one full step"
        ),
    )
    p.add_argument("--nconmax", type=int, default=DEFAULT_NCONMAX)
    p.add_argument("--njmax", type=int, default=DEFAULT_NJMAX)
    p.add_argument(
        "--solver-iterations", type=int, default=DEFAULT_SOLVER_ITERATIONS
    )
    p.add_argument(
        "--solver-ls-iterations", type=int, default=DEFAULT_SOLVER_LS_ITERATIONS
    )
    p.add_argument(
        "--broadphase",
        choices=["nxn", "sap_tile", "sap_segmented"],
        default=None,
    )
    p.add_argument(
        "--base-step",
        action="store_true",
        help="use mjlab's generic post-step forward/sense path",
    )
    p.add_argument(
        "--cube-layout",
        choices=sorted(CUBE_SLOT_LAYOUTS),
        default=DEFAULT_CUBE_LAYOUT,
        help="which entity slots get physical cube bodies",
    )
    p.add_argument(
        "--game-backend",
        choices=GAME_BACKENDS,
        default=DEFAULT_GAME_BACKEND,
        help="fused Warp game kernels or the eager PyTorch fallback",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="timed repetitions; the median SPS is reported as the result",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def profile_phases(env, actions: torch.Tensor, num_steps: int, device: str) -> None:
    """Attribute one full step to game, physics, reward, observation and rest.

    Each phase is synchronized, which inflates the total slightly but shows
    where the gap between raw physics and a full environment step sits.
    """
    game = env.action_manager.get_term("game")
    decimation = env.cfg.decimation
    totals = {
        "process_actions": 0.0,
        "apply_actions": 0.0,
        "physics": 0.0,
        "reward": 0.0,
        "observation": 0.0,
        "managers": 0.0,
    }

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(num_steps):
            sync()
            mark = time.perf_counter()
            game.process_actions(actions)
            sync()
            totals["process_actions"] += time.perf_counter() - mark

            for _ in range(decimation):
                mark = time.perf_counter()
                game.apply_actions()
                sync()
                totals["apply_actions"] += time.perf_counter() - mark
                mark = time.perf_counter()
                env.sim.step()
                sync()
                totals["physics"] += time.perf_counter() - mark

            mark = time.perf_counter()
            env.episode_length_buf += 1
            env.reward_manager.compute(dt=env.step_dt)
            sync()
            totals["reward"] += time.perf_counter() - mark

            mark = time.perf_counter()
            game.observation()
            sync()
            totals["observation"] += time.perf_counter() - mark

            mark = time.perf_counter()
            env.termination_manager.compute()
            sync()
            totals["managers"] += time.perf_counter() - mark

    # nacon/nefc count the whole batch, so compare them against the per-world
    # capacity times the world count before tightening the buffers.
    worlds = env.num_envs
    print(
        f"phase=contacts used_nacon={int(env.sim.data.nacon.max())} "
        f"nacon_capacity={env.cfg.sim.nconmax * worlds} "
        f"used_nefc={int(env.sim.data.nefc.max())} "
        f"nefc_capacity={env.cfg.sim.njmax * worlds}"
    )
    total = sum(totals.values())
    for name, seconds in totals.items():
        print(
            f"phase={name} ms_per_step={seconds / num_steps * 1000.0:.3f} "
            f"share_pct={seconds / total * 100.0:.1f}"
        )
    print(f"phase=total ms_per_step={total / num_steps * 1000.0:.3f}")


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"
    torch.manual_seed(args.seed)

    agent_count = NUM_AGENTS
    if args.backend == "mjlab":
        from escape_room.env import make_env

        env = make_env(
            num_envs=args.num_envs,
            device=device,
            seed=args.seed,
            physics_substeps=args.physics_substeps,
            nconmax=args.nconmax,
            njmax=args.njmax,
            solver_iterations=args.solver_iterations,
            solver_ls_iterations=args.solver_ls_iterations,
            broadphase=args.broadphase,
            cube_layout=args.cube_layout,
            game_backend=args.game_backend,
            fast_step=not args.base_step,
        )
        action_shape = (args.num_envs, env.action_manager.total_action_dim)
        env.reset()
        substeps = args.physics_substeps
    elif args.backend == "cartpole":
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.cartpole.cartpole_env_cfg import cartpole_balance_env_cfg

        cfg = cartpole_balance_env_cfg()
        cfg.scene.num_envs = args.num_envs
        cfg.seed = args.seed
        env = ManagerBasedRlEnv(cfg=cfg, device=device)
        action_shape = (args.num_envs, env.action_manager.total_action_dim)
        env.reset()
        substeps = cfg.decimation
        agent_count = 1
    else:
        from escape_room.vec_env import EscapeRoomVecEnv

        env = EscapeRoomVecEnv(
            num_envs=args.num_envs, device=device, seed=args.seed
        )
        action_shape = (env.num_policy_rows, ACTION_DIM_PER_AGENT)
        env.reset()
        substeps = 1

    if args.step_mode != "full" and args.backend == "kinematic":
        raise ValueError(f"--step-mode {args.step_mode} requires a real mjwarp backend")

    if args.step_mode == "phases":
        if args.backend != "mjlab":
            raise ValueError("--step-mode phases requires the mjlab backend")
        phase_actions = 2.0 * torch.rand(action_shape, device=device) - 1.0
        profile_phases(env, phase_actions, args.num_steps, device)
        env.close()
        return

    if args.step_mode == "physics":
        def step(actions: torch.Tensor) -> None:
            for _ in range(substeps):
                env.sim.step()
    elif args.step_mode == "reset":
        reset_ids = torch.arange(args.num_envs, dtype=torch.long, device=device)

        def step(actions: torch.Tensor) -> None:
            env._reset_idx(reset_ids)
    else:
        def step(actions: torch.Tensor) -> None:
            env.step(actions)

    actions = 2.0 * torch.rand(action_shape, device=device) - 1.0

    def rollout(num_steps: int) -> None:
        with torch.inference_mode():
            for _ in range(num_steps):
                step(actions)

    rollout(args.warmup_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    env_steps = args.num_envs * args.num_steps
    samples: list[float] = []
    for repeat in range(max(args.repeats, 1)):
        start = time.perf_counter()
        rollout(args.num_steps)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        samples.append(env_steps / elapsed)
        print(f"repeat={repeat} wall_s={elapsed:.3f} env_SPS={samples[-1]:,.0f}")

    samples.sort()
    env_sps = samples[len(samples) // 2]
    print(
        f"backend={args.backend} num_envs={args.num_envs} "
        f"num_steps={args.num_steps} step_mode={args.step_mode} "
        f"physics_substeps={substeps} nconmax={args.nconmax} "
        f"njmax={args.njmax} solver_iterations={args.solver_iterations} "
        f"solver_ls_iterations={args.solver_ls_iterations} "
        f"broadphase={args.broadphase} "
        f"cube_layout={args.cube_layout} "
        f"game_backend={args.game_backend} "
        f"base_step={args.base_step} "
        f"repeats={len(samples)} "
        f"spread_pct={(samples[-1] - samples[0]) / env_sps * 100.0:.2f}"
    )
    print(
        f"rollout_only_env_SPS={env_sps:,.0f} "
        f"rollout_only_agent_SPS={env_sps * agent_count:,.0f} "
        "(median of repeats; simulation only; not trainer SPS)"
    )
    close = getattr(env, "close", None)
    if close is not None:
        close()


if __name__ == "__main__":
    main()
