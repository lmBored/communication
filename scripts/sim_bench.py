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

from escape_room.consts import ACTION_DIM_PER_AGENT, NUM_AGENTS  # noqa: E402
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
        choices=["full", "physics"],
        default="full",
        help="full environment step or raw mjwarp physics graphs only",
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
        "--compile-game",
        action="store_true",
        help="fuse the observation math with torch.compile",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


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
            compile_game=args.compile_game,
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

    if args.step_mode == "physics" and args.backend == "kinematic":
        raise ValueError("--step-mode physics requires a real mjwarp backend")

    if args.step_mode == "physics":
        def step(actions: torch.Tensor) -> None:
            for _ in range(substeps):
                env.sim.step()
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

    start = time.perf_counter()
    rollout(args.num_steps)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    env_steps = args.num_envs * args.num_steps
    print(
        f"backend={args.backend} num_envs={args.num_envs} "
        f"num_steps={args.num_steps} step_mode={args.step_mode} "
        f"physics_substeps={substeps} nconmax={args.nconmax} "
        f"njmax={args.njmax} solver_iterations={args.solver_iterations} "
        f"solver_ls_iterations={args.solver_ls_iterations} "
        f"broadphase={args.broadphase} "
        f"compile_game={args.compile_game} "
        f"wall_s={elapsed:.3f}"
    )
    print(
        f"rollout_only_env_SPS={env_steps / elapsed:,.0f} "
        f"rollout_only_agent_SPS={env_steps * agent_count / elapsed:,.0f} "
        "(simulation only; not trainer SPS)"
    )
    close = getattr(env, "close", None)
    if close is not None:
        close()


if __name__ == "__main__":
    main()
