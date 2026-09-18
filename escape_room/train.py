"""
rsl-rl PPO training for the MuJoCo-Warp escape room environment.

Usage:
    escape-room-train --num-envs 1024 --num-updates 50 --ckpt-dir ./ckpts
"""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch

from escape_room.consts import (
    CUBE_SLOT_LAYOUTS,
    DEFAULT_CUBE_LAYOUT,
    DEFAULT_GAME_BACKEND,
    GAME_BACKENDS,
    NUM_AGENTS,
)
from escape_room.env_cfg import DEFAULT_NUM_ENVS


def _print_cuda_memory(device: str, label: str) -> None:
    """Report total free memory, including allocations made outside PyTorch."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    gib = 1024**3
    print(
        f"cuda_memory[{label}]: free={free / gib:.2f}GiB "
        f"total={total / gib:.2f}GiB torch_allocated={allocated / gib:.2f}GiB "
        f"torch_reserved={reserved / gib:.2f}GiB"
    )


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "warp" in message)


# Sharing the actor/critic observation preprocessing (one normalizer, no
# single-group concatenation) was implemented and measured on an A100 at 32768
# worlds: 1,111,260 against 1,117,644 trainer env SPS, i.e. inside the 0.8%
# run-to-run spread. It was therefore dropped rather than kept as a patch over
# rsl-rl internals.


def _task_banner(args: argparse.Namespace) -> str:
    """Report only the flags the selected task actually reads."""
    if args.task == "escape-room":
        return f"cube_layout={args.cube_layout} game_backend={args.game_backend} "
    if args.task == "communication":
        return f"message_dim={args.message_dim} "
    if args.task == "twowaycomm":
        sender = args.sender_message_dim or args.message_dim
        receiver = args.receiver_message_dim or args.message_dim
        return (
            f"sender_message_dim={sender} receiver_message_dim={receiver} "
            f"channel_mode={args.channel_mode} "
            f"delayed_feedback={not args.no_delayed_feedback} "
            f"arrow_layout={args.arrow_layout} obs_mode={args.obs_mode} "
            f"reward_sharing={args.reward_sharing} "
            f"sender_shaping_weight={args.sender_shaping_weight} "
        )
    return ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train escape room agents with mjlab (MuJoCo Warp) + rsl-rl"
    )
    p.add_argument(
        "--task",
        choices=["escape-room", "communication", "twowaycomm", "cartpole"],
        default="escape-room",
        help="environment to train; cartpole provides a simple-stack baseline",
    )
    p.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    p.add_argument("--num-updates", type=int, default=50)
    p.add_argument("--steps-per-update", type=int, default=16)
    p.add_argument(
        "--physics-substeps",
        type=int,
        default=1,
        help="MuJoCo steps per 0.04 s control step (default: 1)",
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
        "--check-nans",
        action="store_true",
        help="enable synchronization-heavy per-transition rsl-rl NaN checks",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-epochs", type=int, default=2)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ckpt-dir", type=str, default="./ckpts")
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--num-channels", type=int, default=256)
    p.add_argument(
        "--message-dim",
        type=int,
        default=2,
        help="continuous sender-to-receiver message width for communication",
    )
    p.add_argument(
        "--sender-message-dim",
        type=int,
        default=None,
        help="twowaycomm sender channel width; defaults to --message-dim. "
        "Narrowing it stops the sender from broadcasting every arrow direction",
    )
    p.add_argument(
        "--receiver-message-dim",
        type=int,
        default=None,
        help="twowaycomm receiver-to-sender channel width; defaults to --message-dim",
    )
    p.add_argument(
        "--channel-mode",
        choices=["same_step", "delayed"],
        default="same_step",
        help="twowaycomm channel timing; delayed enables a query/response protocol",
    )
    p.add_argument(
        "--no-delayed-feedback",
        action="store_true",
        help="with --channel-mode delayed, keep the partner message out of the "
        "message encoder (a one-step-lagged control, not a dialogue)",
    )
    p.add_argument(
        "--arrow-layout",
        choices=["front", "scattered"],
        default="front",
        help="twowaycomm arrow placement; scattered forces the sender to turn",
    )
    p.add_argument(
        "--obs-mode",
        choices=["vector", "pixel"],
        default="vector",
        help="twowaycomm observation form",
    )
    p.add_argument(
        "--reward-sharing",
        choices=["shared", "receiver_only"],
        default="shared",
        help="twowaycomm reward routing; receiver_only disables sender shaping",
    )
    p.add_argument(
        "--sender-shaping-weight",
        type=float,
        default=0.0,
        help="twowaycomm reward for facing the matching-colour arrow (0 disables, "
        "which keeps the reward function identical to communication)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Train with mjlab's MuJoCo-Warp environment and rsl-rl runner."""
    args = parse_args(argv)

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.cartpole import cartpole_balance_env_cfg, cartpole_ppo_runner_cfg
    from mjlab.utils.torch import configure_torch_backends

    from escape_room.env import make_env as make_escape_room_env
    from escape_room.env_cfg import escape_room_ppo_runner_cfg

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("mjwarp training requires an NVIDIA CUDA device")
    configure_torch_backends()
    torch.manual_seed(args.seed)

    if args.task == "escape-room":
        runner_cfg = escape_room_ppo_runner_cfg()
        env_cfg = None
        agent_count = NUM_AGENTS
    elif args.task == "communication":
        from escape_room.communication.env_cfg import communication_ppo_runner_cfg

        runner_cfg = communication_ppo_runner_cfg(message_dim=args.message_dim)
        env_cfg = None
        agent_count = 2
    elif args.task == "twowaycomm":
        from escape_room.twowaycomm.env_cfg import twowaycomm_ppo_runner_cfg

        runner_cfg = twowaycomm_ppo_runner_cfg(
            message_dim=args.message_dim,
            sender_message_dim=args.sender_message_dim,
            receiver_message_dim=args.receiver_message_dim,
            channel_mode=args.channel_mode,
            delayed_message_feedback=not args.no_delayed_feedback,
            obs_mode=args.obs_mode,
        )
        env_cfg = None
        agent_count = 2
    else:
        runner_cfg = cartpole_ppo_runner_cfg()
        env_cfg = cartpole_balance_env_cfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = args.seed
        agent_count = 1
    runner_cfg.seed = args.seed
    runner_cfg.logger = "tensorboard"
    runner_cfg.max_iterations = args.num_updates
    runner_cfg.num_steps_per_env = args.steps_per_update
    runner_cfg.save_interval = args.save_interval
    runner_cfg.actor.hidden_dims = (args.num_channels,) * 3
    runner_cfg.critic.hidden_dims = (args.num_channels,) * 3
    runner_cfg.algorithm.learning_rate = args.lr
    runner_cfg.algorithm.gamma = args.gamma
    runner_cfg.algorithm.lam = args.gae_lambda
    runner_cfg.algorithm.clip_param = args.clip_range
    runner_cfg.algorithm.entropy_coef = args.entropy_coef
    runner_cfg.algorithm.value_loss_coef = args.value_coef
    runner_cfg.algorithm.max_grad_norm = args.max_grad_norm
    runner_cfg.algorithm.num_learning_epochs = args.num_epochs
    runner_cfg.algorithm.num_mini_batches = args.num_minibatches

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = None
    try:
        _print_cuda_memory(device, "before_env")
        if args.task == "escape-room":
            env = make_escape_room_env(
                num_envs=args.num_envs,
                device=device,
                seed=args.seed,
                physics_substeps=args.physics_substeps,
                cube_layout=args.cube_layout,
                game_backend=args.game_backend,
                fast_step=not args.base_step,
            )
        elif args.task == "communication":
            from escape_room.communication.env import make_env as make_communication_env

            env = make_communication_env(
                num_envs=args.num_envs,
                device=device,
                seed=args.seed,
                physics_substeps=args.physics_substeps,
            )
        elif args.task == "twowaycomm":
            from escape_room.twowaycomm.env import make_env as make_twowaycomm_env

            env = make_twowaycomm_env(
                num_envs=args.num_envs,
                device=device,
                seed=args.seed,
                physics_substeps=args.physics_substeps,
                arrow_layout=args.arrow_layout,
                reward_sharing=args.reward_sharing,
                sender_shaping_weight=args.sender_shaping_weight,
                obs_mode=args.obs_mode,
            )
        else:
            env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        _print_cuda_memory(device, "after_env")
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner_dict = asdict(runner_cfg)
        runner_dict["check_for_nan"] = args.check_nans
        runner = MjlabOnPolicyRunner(
            wrapped_env,
            runner_dict,
            log_dir=str(ckpt_dir),
            device=device,
        )
        _print_cuda_memory(device, "after_runner")

        print(
            f"mjlab/mjwarp+rsl-rl training: task={args.task} envs={args.num_envs} "
            f"steps/update={args.steps_per_update} updates={args.num_updates} "
            f"physics_substeps={args.physics_substeps} "
            + _task_banner(args)
            + f"base_step={args.base_step} "
            f"check_nans={args.check_nans}"
        )
        start = time.perf_counter()
        runner.learn(
            num_learning_iterations=args.num_updates,
            init_at_random_ep_len=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_steps = args.num_envs * args.steps_per_update * args.num_updates
        print(
            f"trainer_env_SPS={total_steps / elapsed:,.0f} "
            f"trainer_agent_SPS={total_steps * agent_count / elapsed:,.0f} "
            f"wall_s={elapsed:.3f} (rollout + PPO optimization + logging)"
        )
        final_checkpoint = ckpt_dir / "model_final.pt"
        runner.save(str(final_checkpoint))
        print(f"final_checkpoint={final_checkpoint}")
        _print_cuda_memory(device, "finished")
    except RuntimeError as error:
        if _is_cuda_oom(error):
            _print_cuda_memory(device, "oom")
            raise RuntimeError(
                f"MuJoCo-Warp exhausted GPU memory for task={args.task}, "
                f"--num-envs={args.num_envs} and "
                f"--steps-per-update={args.steps_per_update}. Reduce --num-envs "
                f"first (the 40 GiB A100 default is {DEFAULT_NUM_ENVS}); reducing "
                "--steps-per-update also shrinks rsl-rl rollout storage."
            ) from error
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
