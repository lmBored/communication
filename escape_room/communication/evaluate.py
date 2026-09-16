"""Balanced evaluation, message probing, and channel ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from escape_room.communication.action import LEFT, RIGHT, communication_term

Intervention = Literal["normal", "zero", "shuffled"]
Probe = dict[str, Any]


@dataclass
class RolloutData:
    messages: torch.Tensor
    directions: torch.Tensor
    correct: torch.Tensor
    wrong: torch.Tensor
    timeout: torch.Tensor

    def metrics(self) -> dict[str, float | int]:
        episodes = self.directions.numel()
        return {
            "episodes": episodes,
            "correct": int(self.correct.sum()),
            "wrong": int(self.wrong.sum()),
            "timeout": int(self.timeout.sum()),
            "correct_rate": float(self.correct.float().mean()),
            "wrong_rate": float(self.wrong.float().mean()),
            "timeout_rate": float(self.timeout.float().mean()),
        }


def fit_message_probe(messages: torch.Tensor, directions: torch.Tensor) -> Probe:
    """Fit the diagnostic nearest-centroid LEFT/RIGHT message probe."""
    messages = torch.as_tensor(messages, dtype=torch.float32).cpu()
    directions = torch.as_tensor(directions, dtype=torch.long).flatten().cpu()
    if messages.ndim != 2 or messages.shape[0] != directions.numel():
        raise ValueError("messages and directions must have matching sample counts")
    centroids = {}
    for direction in (LEFT, RIGHT):
        selected = messages[directions == direction]
        if selected.shape[0] == 0:
            raise ValueError("probe calibration requires both LEFT and RIGHT messages")
        centroids[str(direction)] = selected.mean(0).tolist()
    return {
        "version": 1,
        "method": "nearest-centroid",
        "message_dim": messages.shape[1],
        "labels": {str(LEFT): "LEFT", str(RIGHT): "RIGHT"},
        "centroids": centroids,
    }


def decode_messages(messages: torch.Tensor, probe: Probe) -> torch.Tensor:
    """Decode messages with a fitted nearest-centroid probe."""
    messages = torch.as_tensor(messages, dtype=torch.float32)
    if messages.ndim == 1:
        messages = messages[None, :]
    labels = torch.tensor((LEFT, RIGHT), device=messages.device)
    centroids = torch.tensor(
        [probe["centroids"][str(LEFT)], probe["centroids"][str(RIGHT)]],
        dtype=messages.dtype,
        device=messages.device,
    )
    if messages.shape[-1] != centroids.shape[-1]:
        raise ValueError("message dimension does not match the probe")
    distances = (messages[:, None, :] - centroids[None, :, :]).square().sum(-1)
    return labels[distances.argmin(-1)]


def probe_accuracy(
    messages: torch.Tensor, directions: torch.Tensor, probe: Probe
) -> float:
    directions = torch.as_tensor(directions, dtype=torch.long).flatten()
    predicted = decode_messages(messages, probe).cpu()
    return float((predicted == directions.cpu()).float().mean())


def evaluate_probe(
    messages: torch.Tensor,
    directions: torch.Tensor,
    calibration_fraction: float = 0.5,
    seed: int = 0,
) -> tuple[Probe, float]:
    """Fit on a balanced calibration split and score held-out episodes."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must lie strictly between 0 and 1")
    messages = torch.as_tensor(messages, dtype=torch.float32).cpu()
    directions = torch.as_tensor(directions, dtype=torch.long).flatten().cpu()
    generator = torch.Generator().manual_seed(seed)
    calibration: list[torch.Tensor] = []
    held_out: list[torch.Tensor] = []
    for direction in (LEFT, RIGHT):
        indices = torch.where(directions == direction)[0]
        if indices.numel() < 2:
            raise ValueError("at least two episodes per direction are required")
        indices = indices[torch.randperm(indices.numel(), generator=generator)]
        calibration_count = round(indices.numel() * calibration_fraction)
        calibration_count = min(max(calibration_count, 1), indices.numel() - 1)
        calibration.append(indices[:calibration_count])
        held_out.append(indices[calibration_count:])
    calibration_indices = torch.cat(calibration)
    held_out_indices = torch.cat(held_out)
    probe = fit_message_probe(
        messages[calibration_indices], directions[calibration_indices]
    )
    return probe, probe_accuracy(
        messages[held_out_indices], directions[held_out_indices], probe
    )


def apply_message_intervention(
    messages: torch.Tensor,
    intervention: Intervention,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply the normal, zero, or cross-world shuffled message channel."""
    if intervention == "normal":
        return messages
    if intervention == "zero":
        return torch.zeros_like(messages)
    if intervention != "shuffled":
        raise ValueError(f"unknown message intervention: {intervention}")
    if messages.shape[0] < 2:
        raise ValueError("shuffled evaluation requires at least two worlds")

    count = messages.shape[0]
    original = torch.arange(count)
    permutation = original
    for _ in range(32):
        permutation = torch.randperm(count, generator=generator)
        if torch.all(permutation != original):
            break
    else:
        permutation = torch.roll(original, 1)
    return messages[permutation.to(messages.device)]


def default_probe_path(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    return checkpoint.with_suffix(".message_probe.json")


def save_message_probe(probe: Probe, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")


def load_message_probe(path: str | Path) -> Probe:
    return json.loads(Path(path).read_text())


def checkpoint_actor_architecture(path: str | Path) -> dict[str, Any]:
    """Infer Direct DIAL layer widths from an actor checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor_state_dict", {})
    sender: list[tuple[int, torch.Tensor]] = []
    receiver: list[tuple[int, torch.Tensor]] = []
    for key, value in actor_state.items():
        parts = key.split(".")
        if (
            len(parts) == 3
            and parts[1].isdigit()
            and parts[2] == "weight"
        ):
            if parts[0] == "sender_encoder":
                sender.append((int(parts[1]), value))
            elif parts[0] == "receiver_encoder":
                receiver.append((int(parts[1]), value))
    if not sender or not receiver:
        raise ValueError(f"checkpoint has no Direct DIAL actor architecture: {path}")
    sender.sort(key=lambda item: item[0])
    receiver.sort(key=lambda item: item[0])
    return {
        "message_dim": int(sender[-1][1].shape[0]),
        "sender_hidden_dims": tuple(int(value.shape[0]) for _, value in sender[:-1]),
        "hidden_dims": tuple(int(value.shape[0]) for _, value in receiver),
    }


def checkpoint_message_dim(path: str | Path) -> int:
    return int(checkpoint_actor_architecture(path)["message_dim"])


def _balanced_directions(count: int, device: str) -> torch.Tensor:
    directions = torch.empty(count, dtype=torch.long, device=device)
    directions[: count // 2] = LEFT
    directions[count // 2 :] = RIGHT
    return directions


def run_balanced_rollout(
    wrapped_env,
    actor,
    intervention: Intervention,
    directions: torch.Tensor,
    generator: torch.Generator,
) -> RolloutData:
    """Run exactly one balanced episode in every vectorized world."""
    env = wrapped_env.unwrapped
    obs, _ = wrapped_env.reset()
    term = communication_term(env)
    term.set_clue_direction(directions)
    env.sim.forward()
    obs = env.observation_manager.compute(update_history=True)
    initial_messages = actor.encode_message(obs["sender"]).detach().cpu()
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    correct = torch.zeros_like(active)
    wrong = torch.zeros_like(active)
    timeout = torch.zeros_like(active)

    for _ in range(env.max_episode_length):
        with torch.inference_mode():
            messages = actor.encode_message(obs["sender"])
            delivered = apply_message_intervention(
                messages, intervention, generator=generator
            )
            actions = actor.action_from_message(obs["receiver"], delivered)
            _, _, dones, _ = wrapped_env.step(actions)
        done = dones.bool()
        finished = done & active
        correct[finished] = term.correct_entry[finished]
        wrong[finished] = term.wrong_entry[finished]
        timeout[finished] = term.timeout[finished]
        active[finished] = False
        if not active.any():
            break
        if done.any():
            env.reset(env_ids=torch.where(done)[0])
        obs = wrapped_env.get_observations()
    if active.any():
        raise RuntimeError("evaluation worlds did not finish within the horizon")
    return RolloutData(
        messages=initial_messages,
        directions=directions.cpu(),
        correct=correct.cpu(),
        wrong=wrong.cpu(),
        timeout=timeout.cpu(),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="communication checkpoint")
    parser.add_argument("--episodes", type=int, default=4096)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--message-dim", type=int, default=None)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--message-probe", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.episodes < 4 or args.episodes % 2:
        raise SystemExit("--episodes must be an even number of at least 4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from escape_room.communication.env import make_env
    from escape_room.communication.env_cfg import communication_ppo_runner_cfg

    architecture = checkpoint_actor_architecture(args.ckpt)
    message_dim = args.message_dim or architecture["message_dim"]
    runner_cfg = communication_ppo_runner_cfg(message_dim=message_dim)
    runner_cfg.actor.hidden_dims = architecture["hidden_dims"]
    runner_cfg.actor.sender_hidden_dims = architecture["sender_hidden_dims"]
    env = make_env(
        num_envs=args.episodes,
        device=args.device,
        seed=args.seed,
        auto_reset=False,
    )
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
    try:
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg), device=args.device)
        runner.load(
            str(Path(args.ckpt)),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        actor = runner.get_inference_policy(device=args.device)
        directions = _balanced_directions(args.episodes, args.device)
        generator = torch.Generator().manual_seed(args.seed)
        normal = run_balanced_rollout(
            wrapped, actor, "normal", directions, generator
        )
        zero = run_balanced_rollout(wrapped, actor, "zero", directions, generator)
        shuffled = run_balanced_rollout(
            wrapped, actor, "shuffled", directions, generator
        )
        probe, held_out_probe_accuracy = evaluate_probe(
            normal.messages,
            normal.directions,
            calibration_fraction=args.calibration_fraction,
            seed=args.seed,
        )
        probe_path = Path(args.message_probe or default_probe_path(args.ckpt))
        save_message_probe(probe, probe_path)
        summary = {
            "checkpoint": str(Path(args.ckpt).resolve()),
            "message_dim": message_dim,
            "probe_path": str(probe_path.resolve()),
            "held_out_probe_accuracy": held_out_probe_accuracy,
            "normal": normal.metrics(),
            "zero_message": zero.metrics(),
            "shuffled_message": shuffled.metrics(),
        }
        output = Path(args.output) if args.output else Path(args.ckpt).with_suffix(
            ".evaluation.json"
        )
        output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary, indent=2, sort_keys=True))
    finally:
        wrapped.close()


if __name__ == "__main__":
    main()