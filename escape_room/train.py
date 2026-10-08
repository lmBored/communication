"""
rsl-rl PPO training for the MuJoCo-Warp escape room environment.

Usage:
    escape-room-train --num-envs 1024 --num-updates 50 --ckpt-dir ./ckpts
"""

from __future__ import annotations

import argparse
import copy
import json
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


def _resume_update_offset(saved_iteration: int, saved_cli: dict, num_envs: int,
                          steps_per_update: int) -> tuple[int, int]:
    """Convert completed checkpoint transitions to the new rollout batch size."""
    old_batch = saved_cli["num_envs"] * saved_cli["steps_per_update"]
    new_batch = num_envs * steps_per_update
    completed_steps = (saved_iteration + 1) * old_batch
    if saved_iteration < 0 or min(old_batch, new_batch) <= 0:
        raise ValueError("Invalid checkpoint iteration or rollout batch size")
    if completed_steps % new_batch:
        raise ValueError("Checkpoint transitions do not divide the new rollout batch size")
    return completed_steps // new_batch, completed_steps


# Sharing the actor/critic observation preprocessing (one normalizer, no
# single-group concatenation) was implemented and measured on an A100 at 32768
# worlds: 1,111,260 against 1,117,644 trainer env SPS, i.e. inside the 0.8%
# run-to-run spread. It was therefore dropped rather than kept as a patch over
# rsl-rl internals.


# The PPO hyperparameter flags default to None so that MAPPO keeps its own
# task defaults unless a flag is given explicitly. PPO keeps the historical
# behaviour: these values always override the task's runner configuration.
PPO_FLAG_DEFAULTS: dict[str, float | int] = {
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    "num_epochs": 2,
    "num_minibatches": 4,
}
# Flag name -> runner_cfg.algorithm attribute.
_ALGORITHM_FLAGS: dict[str, str] = {
    "lr": "learning_rate",
    "gamma": "gamma",
    "gae_lambda": "lam",
    "clip_range": "clip_param",
    "entropy_coef": "entropy_coef",
    "value_coef": "value_loss_coef",
    "max_grad_norm": "max_grad_norm",
    "num_epochs": "num_learning_epochs",
    "num_minibatches": "num_mini_batches",
}
_MAPPO_ONLY_FLAGS: dict[str, str] = {
    "critic_lr": "critic_learning_rate",
    "value_loss": "value_loss",
    "max_kl": "max_kl",
}


def _validate_args(args: argparse.Namespace) -> None:
    """Reject flag combinations that would silently do nothing."""
    if args.critic_obs_mode is not None and (
        args.task != "twowaycomm" or args.obs_mode != "pixel" or args.algorithm != "mappo"
    ):
        raise SystemExit("--critic-obs-mode requires twowaycomm pixel actors with MAPPO")
    if args.task != "twowaycomm":
        if args.resume_checkpoint:
            raise SystemExit("--resume-checkpoint currently requires --task twowaycomm")
        if args.algorithm != "ppo":
            raise SystemExit("--algorithm mappo is only implemented for twowaycomm")
        if args.message_unit != "tanh" or args.message_noise_std is not None:
            raise SystemExit("--message-unit/--message-noise-std need twowaycomm")
    if args.algorithm != "mappo":
        given = [
            flag
            for flag, value in (
                ("--critic-lr", args.critic_lr),
                ("--value-loss", args.value_loss),
                ("--max-kl", args.max_kl),
                ("--no-value-norm", args.no_value_norm or None),
            )
            if value is not None
        ]
        if given:
            raise SystemExit(f"{', '.join(given)} only apply to --algorithm mappo")
        if args.reward_sharing == "individual":
            raise SystemExit(
                "--reward-sharing individual only differs from shared for a "
                "per-agent learner; use it with --algorithm mappo"
            )
    if args.message_noise_std is not None and args.message_noise_std < 0.0:
        raise SystemExit("--message-noise-std must be non-negative")


def _apply_algorithm_flags(runner_cfg, args: argparse.Namespace) -> None:
    algorithm_cfg = runner_cfg.algorithm
    for flag, attribute in _ALGORITHM_FLAGS.items():
        value = getattr(args, flag)
        if value is None and args.algorithm == "ppo":
            value = PPO_FLAG_DEFAULTS[flag]
        if value is not None:
            setattr(algorithm_cfg, attribute, value)
    if args.lr_schedule is not None:
        algorithm_cfg.schedule = args.lr_schedule
    if args.algorithm == "mappo":
        for flag, attribute in _MAPPO_ONLY_FLAGS.items():
            value = getattr(args, flag)
            if value is not None:
                setattr(algorithm_cfg, attribute, value)
        if args.no_value_norm:
            algorithm_cfg.use_value_norm = False


def _logger_cfg(args: argparse.Namespace, ckpt_dir: Path) -> str | dict:
    if args.logger == "tensorboard":
        return "tensorboard"
    return {
        "class_name": "escape_room.wandb_writer:EscapeRoomWandbWriter",
        "project_name": args.wandb_project,
        "run_name": args.run_name or ckpt_dir.name,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "tags": list(args.wandb_tags),
        # Every flag, including the defaults that were not typed, so a run can
        # be reproduced from its W&B page alone.
        "config": {"cli_args": vars(args)},
    }


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
            f"algorithm={args.algorithm} message_unit={args.message_unit} "
            f"message_noise_std={args.message_noise_std} "
            f"sender_message_dim={sender} receiver_message_dim={receiver} "
            f"channel_mode={args.channel_mode} "
            f"delayed_feedback={not args.no_delayed_feedback} "
            f"arrow_layout={args.arrow_layout} obs_mode={args.obs_mode} "
            f"pixel_size={args.pixel_size if args.obs_mode == 'pixel' else None} "
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
        "--release-cuda-cache", action="store_true",
        help="release unused PyTorch cache after each update so Warp can allocate simulation buffers",
    )
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
    # PPO fills unset flags from PPO_FLAG_DEFAULTS; MAPPO keeps its own.
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--gae-lambda", type=float, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--value-coef", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--num-minibatches", type=int, default=None)
    p.add_argument(
        "--lr-schedule",
        choices=["adaptive", "fixed"],
        default=None,
        help="learning-rate schedule (task default: PPO adaptive, MAPPO fixed)",
    )
    p.add_argument(
        "--algorithm",
        choices=["ppo", "mappo"],
        default="ppo",
        help="twowaycomm learner: joint-ratio PPO, or per-agent MAPPO with a "
        "per-agent centralized critic",
    )
    p.add_argument(
        "--critic-lr",
        type=float,
        default=None,
        help="MAPPO critic learning rate (default: same as --lr)",
    )
    p.add_argument(
        "--value-loss",
        choices=["huber", "mse"],
        default=None,
        help="MAPPO value loss (default: huber, delta 10)",
    )
    p.add_argument(
        "--max-kl",
        type=float,
        default=None,
        help="MAPPO KL early stopping: skip the rest of an update once either "
        "agent's KL from the rollout policy exceeds this (default: off)",
    )
    p.add_argument(
        "--no-value-norm",
        action="store_true",
        help="disable MAPPO's running value normalization",
    )
    p.add_argument(
        "--logger",
        choices=["tensorboard", "wandb"],
        default="tensorboard",
        help="wandb also writes TensorBoard event files",
    )
    p.add_argument("--wandb-project", type=str, default="escape-room")
    p.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (default: $WANDB_USERNAME, else your default entity)",
    )
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=())
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="W&B run name (default: the --ckpt-dir directory name)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--ckpt-dir", type=str, default="./ckpts")
    p.add_argument("--resume-checkpoint", type=str, default=None,
                   help="restore twowaycomm models/optimizers; --num-updates is the total target at the new batch size")
    p.add_argument("--save-interval", type=int, default=25)
    p.add_argument("--num-channels", type=int, default=256)
    p.add_argument("--critic-obs-mode", choices=["pixel", "vector"], default=None,
                   help="pixel MAPPO critic input; vector keeps the GRU but reads privileged state")
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
        "--message-unit",
        choices=["tanh", "dru", "dru_st"],
        default="tanh",
        help="twowaycomm channel unit: continuous tanh, DIAL's DRU (noisy "
        "sigmoid in training, hard bit 1{m>0} in evaluation), or dru_st "
        "(hard bit in training too, sigmoid gradient straight through)",
    )
    p.add_argument(
        "--message-noise-std",
        type=float,
        default=None,
        help="twowaycomm channel training noise (default: 0 for tanh, 2 for dru)",
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
    p.add_argument("--pixel-size", type=int, choices=[32, 64, 192], default=64,
                   help="twowaycomm pixel mode: shared sender/receiver image size")
    p.add_argument(
        "--reward-sharing",
        choices=["shared", "receiver_only", "individual"],
        default="shared",
        help="twowaycomm reward routing; receiver_only disables sender shaping, "
        "individual credits it to the sender alone (needs --algorithm mappo)",
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
    _validate_args(args)

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
            algorithm=args.algorithm,
            message_unit=args.message_unit,
            message_noise_std=args.message_noise_std,
            critic_obs_mode=args.critic_obs_mode,
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
    runner_cfg.max_iterations = args.num_updates
    runner_cfg.num_steps_per_env = args.steps_per_update
    runner_cfg.save_interval = args.save_interval
    runner_cfg.actor.hidden_dims = (args.num_channels,) * 3
    runner_cfg.critic.hidden_dims = (args.num_channels,) * 3
    _apply_algorithm_flags(runner_cfg, args)

    ckpt_dir = Path(args.ckpt_dir).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        runner_cfg.run_name = args.run_name

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
                pixel_size=args.pixel_size if args.obs_mode == "pixel" else None,
            )
        else:
            env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        _print_cuda_memory(device, "after_env")
        wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner_dict = asdict(runner_cfg)
        runner_dict["check_for_nan"] = args.check_nans
        runner_dict["logger"] = _logger_cfg(args, ckpt_dir)
        metadata = {
            "version": 1, "cli_args": vars(args),
            "environment": {"obs_mode": args.obs_mode,
                            "pixel_size": args.pixel_size if args.obs_mode == "pixel" else None,
                            "arrow_layout": args.arrow_layout,
                            "reward_sharing": args.reward_sharing,
                            "sender_shaping_weight": args.sender_shaping_weight},
            "runner": copy.deepcopy(runner_dict),
        }
        (ckpt_dir / "training_config.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        class ConfiguredRunner(MjlabOnPolicyRunner):
            def save(self, path, infos=None):
                if args.task == "twowaycomm":
                    infos = {**(infos or {}), "twowaycomm": metadata}
                return super().save(path, infos)
        runner = ConfiguredRunner(
            wrapped_env,
            runner_dict,
            log_dir=str(ckpt_dir),
            device=device,
        )
        remaining_updates = args.num_updates
        if args.resume_checkpoint:
            infos = runner.load(args.resume_checkpoint, map_location=device)
            saved_cli = infos["twowaycomm"]["cli_args"]
            for key in ("task", "algorithm", "obs_mode", "pixel_size", "arrow_layout",
                        "channel_mode", "message_unit", "message_dim", "sender_message_dim",
                        "receiver_message_dim", "no_delayed_feedback", "reward_sharing"):
                if saved_cli.get(key) != getattr(args, key):
                    raise SystemExit(f"Resume configuration mismatch: {key}")
            if saved_cli.get("critic_obs_mode") != args.critic_obs_mode:
                raise SystemExit("Resume configuration mismatch: critic_obs_mode")
            saved_iteration = runner.current_learning_iteration
            offset, prior_steps = _resume_update_offset(saved_iteration, saved_cli,
                                                       args.num_envs, args.steps_per_update)
            remaining_updates -= offset
            if remaining_updates <= 0:
                raise SystemExit("The checkpoint already reaches --num-updates at this batch size")
            runner.current_learning_iteration = offset
            metadata["resume"] = {"checkpoint": str(Path(args.resume_checkpoint).resolve()),
                "saved_iteration": saved_iteration, "previous_num_envs": saved_cli["num_envs"],
                "environment_steps_at_resume": prior_steps, "start_update": offset,
                "remaining_updates": remaining_updates,
                "started_wall_time": time.time(),
                "previous_resume": infos["twowaycomm"].get("resume"),
                "rollout_state": "fresh environments and recurrent state; models, optimizers and normalizers restored"}
            (ckpt_dir / "training_config.json").write_text(
                json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            print(f"resumed_environment_steps={prior_steps} start_update={offset} "
                  f"remaining_updates={remaining_updates}")
        if args.release_cuda_cache and device.startswith("cuda"):
            original_update = runner.alg.update

            def update_and_release_cache():
                result = original_update()
                torch.cuda.empty_cache()
                return result

            runner.alg.update = update_and_release_cache
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
            num_learning_iterations=remaining_updates,
            init_at_random_ep_len=False,
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total_steps = args.num_envs * args.steps_per_update * remaining_updates
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
