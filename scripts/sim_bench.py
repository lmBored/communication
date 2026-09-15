#!/usr/bin/env python3
"""
Rollout-only simulation microbenchmark (random actions, no PPO optimization).

The figures printed here are simulation throughput only; they are NOT trainer
SPS. Use `python scripts/train.py` for end-to-end training throughput.

Usage:
    python scripts/sim_bench.py --backend mjlab --num-envs 4096 --num-steps 200
    python scripts/sim_bench.py --backend kinematic --num-envs 4096 --num-steps 200
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rollout-only escape room SPS benchmark")
    p.add_argument(
        "--backend",
        type=str,
        default="mjlab",
        choices=["mjlab", "kinematic"],
        help="mjlab = real MuJoCo Warp physics; kinematic = tensor-only fallback",
    )
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--num-steps", type=int, default=200)
    p.add_argument("--warmup-steps", type=int, default=20, help="Steps before timing")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU")
        device = "cpu"
    torch.manual_seed(args.seed)

    if args.backend == "mjlab":
        from escape_room.env import make_env

        env = make_env(num_envs=args.num_envs, device=device, seed=args.seed)
        action_shape = (args.num_envs, env.action_manager.total_action_dim)
        env.reset()

        def step(actions: torch.Tensor) -> None:
            env.step(actions)
    else:
        from escape_room.vec_env import EscapeRoomVecEnv

        env = EscapeRoomVecEnv(
            num_envs=args.num_envs, device=device, seed=args.seed
        )
        action_shape = (env.num_policy_rows, ACTION_DIM_PER_AGENT)
        env.reset()

        def step(actions: torch.Tensor) -> None:
            env.step(actions)

    def rollout(num_steps: int) -> None:
        with torch.inference_mode():
            for _ in range(num_steps):
                step(2.0 * torch.rand(action_shape, device=device) - 1.0)

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
        f"num_steps={args.num_steps} wall_s={elapsed:.3f}"
    )
    print(
        f"rollout_only_env_SPS={env_steps / elapsed:,.0f} "
        f"rollout_only_agent_SPS={env_steps * NUM_AGENTS / elapsed:,.0f} "
        "(simulation only; not trainer SPS)"
    )
    close = getattr(env, "close", None)
    if close is not None:
        close()


if __name__ == "__main__":
    main()
