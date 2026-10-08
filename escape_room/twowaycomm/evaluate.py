"""Held-out evaluation, message probes, and per-channel ablations.

The question this module exists to answer is not "does the policy solve the
task" but "which channel is actually carrying the information". Both channels
are probed and ablated independently, so a back-channel that turns out to be
vestigial shows up as a clean null result rather than as a mystery.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from escape_room.twowaycomm.action import LEFT, RIGHT, twowaycomm_term
from escape_room.twowaycomm.scene import COLOR_NAMES, NUM_COLORS

Intervention = Literal["normal", "zero", "shuffled"]
# Where an intervention applies to a channel's message:
#   "all"      - everywhere the partner reads it: its action head AND, in the
#                delayed channel with feedback, its message encoder.
#   "head"     - only the partner's action head.
#   "feedback" - only the partner's message encoder (delayed + feedback only).
# A query->response protocol travels through "feedback": the receiver's message
# changes what the sender SAYS, not what the sender does, and the sender's own
# actions never affect the reward. A head-only ablation of the receiver channel
# therefore reads zero by construction, whatever the protocol.
InterventionPath = Literal["all", "head", "feedback"]
ChannelPlan = tuple[Intervention, InterventionPath]
Probe = dict[str, Any]

NUM_ARROWS = NUM_COLORS
NUM_CONDITIONS = (2**NUM_ARROWS) * NUM_COLORS  # 8 arrow patterns x 3 door colours
DEFAULT_EPISODES = 4800  # 200 per condition; 4096 does not divide by 24

_INTACT: ChannelPlan = ("normal", "all")
# (sender channel, receiver channel). Unsuffixed arms cut every path.
ABLATION_ARMS: dict[str, tuple[ChannelPlan, ChannelPlan]] = {
    "normal": (_INTACT, _INTACT),
    "zero_sender": (("zero", "all"), _INTACT),
    "zero_receiver": (_INTACT, ("zero", "all")),
    "zero_both": (("zero", "all"), ("zero", "all")),
    "shuffle_sender": (("shuffled", "all"), _INTACT),
    "shuffle_receiver": (_INTACT, ("shuffled", "all")),
    # Path decomposition. Before the "all" arms existed, zero_receiver meant
    # zero_receiver_head_only, which is why it always read no effect.
    "shuffle_sender_head_only": (("shuffled", "head"), _INTACT),
    "shuffle_receiver_head_only": (_INTACT, ("shuffled", "head")),
    "shuffle_sender_feedback_only": (("shuffled", "feedback"), _INTACT),
    "shuffle_receiver_feedback_only": (_INTACT, ("shuffled", "feedback")),
}
FEEDBACK_ARMS = frozenset(
    arm for arm, plans in ABLATION_ARMS.items() if any(p[1] == "feedback" for p in plans)
)


def ablation_arms(has_feedback: bool) -> dict[str, tuple[ChannelPlan, ChannelPlan]]:
    """Arms that are meaningful for this actor.

    Without a feedback path (same-step, or delayed with feedback disabled) the
    feedback-only arms would be identical to ``normal``, and ``all`` reduces
    to ``head``.
    """
    if has_feedback:
        return dict(ABLATION_ARMS)
    return {arm: plans for arm, plans in ABLATION_ARMS.items() if arm not in FEEDBACK_ARMS}


@dataclass
class Conditions:
    """One balanced episode condition per world."""

    directions: torch.Tensor  # [E, 3] in {-1, +1}
    colors: torch.Tensor  # [E] in {0, 1, 2}

    @property
    def target_direction(self) -> torch.Tensor:
        return self.directions.gather(1, self.colors[:, None]).squeeze(1)

    @property
    def condition_id(self) -> torch.Tensor:
        bits = ((self.directions == RIGHT).long() * (2 ** torch.arange(NUM_ARROWS, device=self.directions.device))).sum(
            -1
        )
        return self.colors * (2**NUM_ARROWS) + bits


def balanced_conditions(episodes: int, device: str, seed: int) -> Conditions:
    """Every one of the 24 joint conditions exactly ``episodes // 24`` times.

    The shuffle decorrelates condition from world index, and the exact balance
    makes chance-level door choice exactly 1/2 and chance-level colour 1/3.
    """
    if episodes % NUM_CONDITIONS:
        raise ValueError(
            f"--episodes must be a multiple of {NUM_CONDITIONS}, got {episodes}"
        )
    generator = torch.Generator().manual_seed(seed)
    identifiers = torch.arange(episodes) % NUM_CONDITIONS
    identifiers = identifiers[torch.randperm(episodes, generator=generator)]
    colors = identifiers // (2**NUM_ARROWS)
    bits = identifiers % (2**NUM_ARROWS)
    directions = torch.stack(
        [
            torch.where((bits >> arrow) & 1 == 1, RIGHT, LEFT)
            for arrow in range(NUM_ARROWS)
        ],
        dim=-1,
    )
    return Conditions(
        directions=directions.to(device=device, dtype=torch.long),
        colors=colors.to(device=device, dtype=torch.long),
    )


# Probes.


def fit_nearest_centroid(
    messages: torch.Tensor, labels: torch.Tensor, names: dict[int, str] | None = None
) -> Probe:
    """Per-class mean vector. Generalizes the one-way scenario's 2-class probe."""
    if messages.ndim != 2:
        raise ValueError("messages must be two-dimensional")
    if messages.shape[0] != labels.shape[0]:
        raise ValueError("messages and labels must describe the same samples")
    centroids: dict[str, list[float]] = {}
    for label in sorted(int(value) for value in labels.unique()):
        selected = messages[labels == label]
        if selected.numel() == 0:
            raise ValueError(f"no samples for label {label}")
        centroids[str(label)] = selected.float().mean(dim=0).tolist()
    if len(centroids) < 2:
        raise ValueError("a probe needs at least two classes")
    return {
        "message_dim": int(messages.shape[-1]),
        "labels": {
            key: (names or {}).get(int(key), str(key)) for key in centroids
        },
        "centroids": centroids,
    }


def decode_messages(messages: torch.Tensor, head: Probe) -> torch.Tensor:
    """Nearest-centroid decode over however many classes the head holds."""
    if messages.ndim == 1:
        messages = messages.unsqueeze(0)
    labels = sorted(int(key) for key in head["centroids"])
    centroids = torch.tensor(
        [head["centroids"][str(label)] for label in labels],
        dtype=messages.dtype,
        device=messages.device,
    )
    distances = torch.cdist(messages.float(), centroids.float()).square()
    chosen = distances.argmin(dim=-1)
    return torch.tensor(labels, device=messages.device)[chosen]


def probe_accuracy(
    messages: torch.Tensor, labels: torch.Tensor, head: Probe
) -> float:
    predicted = decode_messages(messages, head)
    return float((predicted == labels.to(predicted.device)).float().mean())


def evaluate_probe(
    messages: torch.Tensor,
    labels: torch.Tensor,
    calibration_fraction: float = 0.5,
    seed: int = 0,
    names: dict[int, str] | None = None,
) -> tuple[Probe, float]:
    """Fit on a stratified calibration split and score the held-out remainder."""
    generator = torch.Generator().manual_seed(seed)
    calibration: list[torch.Tensor] = []
    held_out: list[torch.Tensor] = []
    for label in labels.unique():
        indices = torch.nonzero(labels == label, as_tuple=False).squeeze(-1)
        if indices.numel() < 2:
            raise ValueError(f"label {int(label)} needs at least two samples")
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        count = int(round(indices.numel() * calibration_fraction))
        count = max(1, min(count, indices.numel() - 1))
        calibration.append(shuffled[:count])
        held_out.append(shuffled[count:])
    calibration_ids = torch.cat(calibration)
    held_out_ids = torch.cat(held_out)
    head = fit_nearest_centroid(
        messages[calibration_ids], labels[calibration_ids], names
    )
    return head, probe_accuracy(messages[held_out_ids], labels[held_out_ids], head)


# Interventions.


def derangement(
    count: int, generator: torch.Generator | None = None, device=None
) -> torch.Tensor:
    """A permutation that moves every index (falls back to a roll by one)."""
    if count < 2:
        raise ValueError("shuffling needs at least two worlds")
    identity = torch.arange(count, device=device)
    for _ in range(32):
        source = generator.device if generator is not None else "cpu"
        permutation = torch.randperm(count, generator=generator, device=source).to(device)
        if torch.all(permutation != identity):
            return permutation
    return torch.roll(identity, 1)


def apply_message_intervention(
    messages: torch.Tensor,
    intervention: Intervention,
    generator: torch.Generator | None = None,
    permutation: torch.Tensor | None = None,
) -> torch.Tensor:
    """``permutation`` fixes which world a shuffled message comes from; the
    rollout passes one per channel so a world hears one consistent stranger
    for the whole episode instead of a different one every step."""
    if intervention == "normal":
        return messages
    if intervention == "zero":
        return torch.zeros_like(messages)
    if intervention == "shuffled" and permutation is not None:
        return messages[permutation]
    if intervention == "shuffled":
        count = messages.shape[0]
        if count < 2:
            raise ValueError("shuffling needs at least two worlds")
        for _ in range(32):
            permutation = torch.randperm(
                count, generator=generator, device=messages.device
            )
            if torch.all(permutation != torch.arange(count, device=messages.device)):
                return messages[permutation]
        return torch.roll(messages, 1, dims=0)
    raise ValueError(f"unknown message intervention: {intervention}")


# Checkpoints.


def default_probe_path(checkpoint: str | Path) -> Path:
    return Path(checkpoint).with_suffix(".message_probe.json")


def save_message_probe(probe: Probe, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")


def load_message_probe(path: str | Path) -> Probe:
    return json.loads(Path(path).read_text())


def probe_head(probe: Probe, channel: str, target: str) -> Probe:
    return probe["channels"][channel]["targets"][target]


_ENCODER_PREFIXES = (
    "sender_message_encoder",
    "receiver_message_encoder",
    "sender_latent_encoder",
    "receiver_latent_encoder",
)


def channel_effects(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Accuracy lost when each channel is replaced by a stranger's message."""
    normal = results["normal"]["correct_rate"]
    effects = {}
    for channel, label in (("sender", "forward"), ("receiver", "back")):
        for suffix in ("", "_head_only", "_feedback_only"):
            arm = f"shuffle_{channel}{suffix}"
            if arm in results:
                effects[f"{label}_channel_effect{suffix}"] = (
                    normal - results[arm]["correct_rate"]
                )
    return effects


def checkpoint_actor_architecture(path: str | Path) -> dict[str, Any]:
    """Infer the actor's widths and channel mode so strict loading succeeds.

    The one-way scenario's inference function looks for ``sender_encoder`` and
    raises when it is missing, so it correctly rejects these checkpoints rather
    than silently returning the wrong widths.
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("actor_state_dict", {})
    metadata = (checkpoint.get("infos") or {}).get("twowaycomm", {})
    has_pixels = any(key.startswith("sender_cnn.") for key in state)
    if has_pixels and not metadata:
        sidecar = Path(path).parent / "training_config.json"
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        else:
            raise ValueError("pixel checkpoint requires saved image-size metadata")
    layers: dict[str, list[tuple[int, torch.Tensor]]] = {
        prefix: [] for prefix in _ENCODER_PREFIXES
    }
    for key, value in state.items():
        parts = key.split(".")
        if (
            len(parts) == 3
            and parts[0] in layers
            and parts[1].isdigit()
            and parts[2] == "weight"
        ):
            layers[parts[0]].append((int(parts[1]), value))
    if not all(layers.values()):
        raise ValueError(f"checkpoint has no two-way DIAL actor architecture: {path}")
    for entries in layers.values():
        entries.sort(key=lambda item: item[0])

    code = int(state.get("channel_mode_code", torch.tensor(0)).reshape(-1)[0])
    # Absent from checkpoints that predate the DRU, all of which are tanh.
    unit_code = int(state.get("message_unit_code", torch.tensor(0)).reshape(-1)[0])
    from escape_room.twowaycomm.model import CHANNEL_MODES, MESSAGE_UNITS

    sender_message_in = int(layers["sender_message_encoder"][0][1].shape[1])
    sender_obs_dim = int(layers["sender_latent_encoder"][0][1].shape[1])
    return {
        "obs_mode": "pixel" if has_pixels else "vector",
        "environment": metadata.get("environment", {"obs_mode": "vector", "arrow_layout": "front"}),
        "cnn_cfg": metadata.get("runner", {}).get("actor", {}).get("cnn_cfg"),
        "critic_obs_mode": metadata.get("cli_args", {}).get("critic_obs_mode"),
        "sender_message_dim": int(layers["sender_message_encoder"][-1][1].shape[0]),
        "receiver_message_dim": int(layers["receiver_message_encoder"][-1][1].shape[0]),
        "sender_message_hidden_dims": tuple(
            int(value.shape[0]) for _, value in layers["sender_message_encoder"][:-1]
        ),
        "receiver_message_hidden_dims": tuple(
            int(value.shape[0]) for _, value in layers["receiver_message_encoder"][:-1]
        ),
        "sender_hidden_dims": tuple(
            int(value.shape[0]) for _, value in layers["sender_latent_encoder"]
        ),
        "hidden_dims": tuple(
            int(value.shape[0]) for _, value in layers["receiver_latent_encoder"]
        ),
        "channel_mode": CHANNEL_MODES[code],
        "message_unit": MESSAGE_UNITS[unit_code],
        # With feedback the message encoder reads the partner message too, so
        # its input is wider than the latent encoder's by exactly that width.
        "delayed_message_feedback": sender_message_in > sender_obs_dim,
    }


# Rollout.


@dataclass
class RolloutData:
    sender_messages: torch.Tensor  # [E, steps, sender_dim]
    receiver_messages: torch.Tensor  # [E, steps, receiver_dim]
    correct: torch.Tensor
    wrong: torch.Tensor
    timeout: torch.Tensor
    conditions: Conditions

    def metrics(self) -> dict[str, float]:
        episodes = int(self.correct.numel())
        rates = {
            "correct_rate": float(self.correct.float().mean()),
            "wrong_rate": float(self.wrong.float().mean()),
            "timeout_rate": float(self.timeout.float().mean()),
        }
        # With six ablation arms it is easy to over-read a few percent.
        confidence = 1.96 * (
            rates["correct_rate"] * (1.0 - rates["correct_rate"]) / episodes
        ) ** 0.5
        by_color = [
            float(self.correct[self.conditions.colors == colour].float().mean())
            for colour in range(NUM_COLORS)
        ]
        return {
            "episodes": episodes,
            "correct": int(self.correct.sum()),
            "wrong": int(self.wrong.sum()),
            "timeout": int(self.timeout.sum()),
            **rates,
            "correct_rate_ci95": confidence,
            "correct_rate_by_color": by_color,
        }


def route_messages(
    emitted: tuple[torch.Tensor, torch.Tensor],
    plans: tuple[ChannelPlan, ChannelPlan],
    permutations: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
    generator: torch.Generator | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    """Return ``(to_encoders, to_heads)``: what each partner's message encoder
    (feedback) and action head receive, given the per-channel ablation plans.

    The intervention is applied once per channel, so an ``all`` plan feeds the
    same altered message to both paths.
    """
    to_encoders, to_heads = [], []
    for message, (intervention, path), permutation in zip(emitted, plans, permutations):
        altered = apply_message_intervention(message, intervention, generator, permutation)
        to_heads.append(altered if path in ("all", "head") else message)
        to_encoders.append(altered if path in ("all", "feedback") else message)
    return (to_encoders[0], to_encoders[1]), (to_heads[0], to_heads[1])


def run_balanced_rollout(
    wrapped_env,
    actor,
    conditions: Conditions,
    plan: tuple[ChannelPlan, ChannelPlan] = (_INTACT, _INTACT),
    generator: torch.Generator | None = None,
    capture_steps: int = 3,
) -> RolloutData:
    """One episode per world under a fixed condition set and ablation plan.

    The logged messages are what each agent emitted, before any intervention.
    """
    env = wrapped_env.unwrapped
    device = env.device
    wrapped_env.reset()
    term = twowaycomm_term(env)
    term.set_arrow_directions(conditions.directions)
    term.set_door_color(conditions.colors)
    env.sim.forward()
    env.sim.sense()
    obs = env.observation_manager.compute(update_history=True)
    actor.reset()

    worlds = conditions.colors.numel()
    sender_width = actor.sender_message_dim
    receiver_width = actor.receiver_message_dim
    sender_log = torch.zeros(worlds, capture_steps, sender_width)
    receiver_log = torch.zeros(worlds, capture_steps, receiver_width)
    # One fixed stranger per channel for the whole episode.
    permutations = tuple(
        derangement(worlds, generator, device) if channel[0] == "shuffled" else None
        for channel in plan
    )
    zeros = (
        torch.zeros(worlds, sender_width, device=device),
        torch.zeros(worlds, receiver_width, device=device),
    )
    # Delayed channel: what the encoders (feedback) and heads read at step t
    # is what was routed at t-1.
    heard_by_encoders = zeros
    heard_by_heads = zeros
    correct = torch.zeros(worlds, dtype=torch.bool, device=device)
    wrong = torch.zeros_like(correct)
    timeout = torch.zeros_like(correct)
    active = torch.ones_like(correct)

    with torch.inference_mode():
        for step in range(int(env.max_episode_length)):
            sender_message, receiver_message = actor.encode_messages(
                obs, heard_by_encoders
            )
            if step < capture_steps:
                sender_log[:, step] = sender_message.detach().cpu()
                receiver_log[:, step] = receiver_message.detach().cpu()
            to_encoders, to_heads = route_messages(
                (sender_message, receiver_message), plan, permutations, generator
            )
            # Same-step heads read this step's messages; delayed heads read last
            # step's. Same-step encoders never read the partner at all.
            delivered = heard_by_heads if actor.is_recurrent else to_heads
            actions = actor.action_from_messages(obs, delivered[0], delivered[1])
            obs, _, dones, _ = wrapped_env.step(actions)
            heard_by_encoders, heard_by_heads = to_encoders, to_heads

            done_mask = dones.bool()
            if torch.any(done_mask & active):
                latch = done_mask & active
                correct[latch] = term.correct_entry[latch]
                wrong[latch] = term.wrong_entry[latch]
                timeout[latch] = term.timeout[latch]
                active &= ~latch
            if not bool(active.any()):
                break
            if torch.any(done_mask):
                reset_ids = torch.nonzero(done_mask, as_tuple=False).squeeze(-1)
                env.reset(env_ids=reset_ids)
                zeroed = done_mask[:, None]
                heard_by_encoders = tuple(
                    torch.where(zeroed, torch.zeros_like(m), m) for m in heard_by_encoders
                )
                heard_by_heads = tuple(
                    torch.where(zeroed, torch.zeros_like(m), m) for m in heard_by_heads
                )
                obs = wrapped_env.get_observations()

    if bool(active.any()):
        raise RuntimeError("evaluation worlds did not finish within the horizon")
    return RolloutData(
        sender_messages=sender_log,
        receiver_messages=receiver_log,
        correct=correct.cpu(),
        wrong=wrong.cpu(),
        timeout=timeout.cpu(),
        conditions=Conditions(
            directions=conditions.directions.cpu(), colors=conditions.colors.cpu()
        ),
    )


def build_probe(
    rollout: RolloutData,
    channel_mode: str,
    calibration_fraction: float,
    seed: int,
    probe_step: int | None = None,
) -> tuple[Probe, dict[str, float]]:
    """Fit one probe per channel and report held-out accuracy for each target."""
    # At t=0 a delayed sender has heard nothing, so a query/response code can
    # only appear from the second step onwards.
    step = probe_step if probe_step is not None else (1 if channel_mode == "delayed" else 0)
    step = min(step, rollout.sender_messages.shape[1] - 1)
    sender = rollout.sender_messages[:, step]
    receiver = rollout.receiver_messages[:, step]
    conditions = rollout.conditions

    sender_targets: dict[str, Probe] = {}
    accuracies: dict[str, float] = {}
    for arrow in range(NUM_ARROWS):
        head, accuracy = evaluate_probe(
            sender,
            conditions.directions[:, arrow],
            calibration_fraction,
            seed,
            {LEFT: "LEFT", RIGHT: "RIGHT"},
        )
        sender_targets[f"direction_{arrow}"] = head
        accuracies[f"sender/direction_{arrow}"] = accuracy
    head, accuracy = evaluate_probe(
        sender,
        conditions.target_direction,
        calibration_fraction,
        seed,
        {LEFT: "LEFT", RIGHT: "RIGHT"},
    )
    sender_targets["queried_direction"] = head
    accuracies["sender/queried_direction"] = accuracy

    colour_head, colour_accuracy = evaluate_probe(
        receiver,
        conditions.colors,
        calibration_fraction,
        seed,
        {index: name.upper() for index, name in enumerate(COLOR_NAMES)},
    )
    accuracies["receiver/door_color"] = colour_accuracy

    probe = {
        "version": 2,
        "method": "nearest-centroid",
        "channel_mode": channel_mode,
        "channels": {
            "sender": {
                "message_dim": int(sender.shape[-1]),
                "probe_step": step,
                "targets": sender_targets,
            },
            "receiver": {
                "message_dim": int(receiver.shape[-1]),
                "probe_step": step,
                "targets": {"door_color": colour_head},
            },
        },
    }
    return probe, accuracies


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a twowaycomm checkpoint and probe both channels"
    )
    parser.add_argument("--ckpt", required=True, help="twowaycomm checkpoint")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arrow-layout", choices=["front", "scattered"], default=None,
                        help="default: restore the checkpoint's layout")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="evaluation batch size; pixel default 96, vector default episodes")
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--probe-step", type=int, default=None)
    parser.add_argument("--message-probe", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--arms", nargs="+", choices=list(ABLATION_ARMS), default=None,
                        help="optional subset of intervention arms; must include normal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.arms is not None and "normal" not in args.arms:
        raise SystemExit("--arms must include normal")
    if args.episodes % NUM_CONDITIONS:
        raise SystemExit(
            f"--episodes must be a multiple of {NUM_CONDITIONS} so every "
            "arrow/colour combination appears equally often"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu")

    from dataclasses import asdict

    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

    from escape_room.twowaycomm.env import make_env
    from escape_room.twowaycomm.env_cfg import twowaycomm_ppo_runner_cfg

    architecture = checkpoint_actor_architecture(args.ckpt)
    num_envs = args.num_envs or (min(96, args.episodes) if architecture["obs_mode"] == "pixel" else args.episodes)
    if num_envs < 24 or args.episodes % num_envs:
        raise SystemExit("--num-envs must be >=24 and divide --episodes")
    runner_cfg = twowaycomm_ppo_runner_cfg(
        obs_mode=architecture["obs_mode"],
        algorithm="mappo" if architecture["obs_mode"] == "pixel" else "ppo",
        sender_message_dim=architecture["sender_message_dim"],
        receiver_message_dim=architecture["receiver_message_dim"],
        channel_mode=architecture["channel_mode"],
        delayed_message_feedback=architecture["delayed_message_feedback"],
        message_unit=architecture["message_unit"],
        critic_obs_mode=architecture.get("critic_obs_mode"),
    )
    runner_cfg.actor.hidden_dims = architecture["hidden_dims"]
    runner_cfg.actor.sender_hidden_dims = architecture["sender_hidden_dims"]
    runner_cfg.actor.sender_message_hidden_dims = architecture[
        "sender_message_hidden_dims"
    ]
    runner_cfg.actor.receiver_message_hidden_dims = architecture[
        "receiver_message_hidden_dims"
    ]
    if architecture["cnn_cfg"] is not None:
        runner_cfg.actor.cnn_cfg = architecture["cnn_cfg"]

    env_config = dict(architecture["environment"])
    if args.arrow_layout is not None:
        env_config["arrow_layout"] = args.arrow_layout
    env = make_env(
        num_envs=num_envs,
        device=args.device,
        seed=args.seed,
        **env_config,
        auto_reset=False,
    )
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg), device=args.device)
        runner.load(
            str(Path(args.ckpt)),
            load_cfg={"actor": True},
            strict=True,
            map_location=args.device,
        )
        actor = runner.get_inference_policy(device=args.device)
        conditions = balanced_conditions(args.episodes, args.device, args.seed)

        results: dict[str, Any] = {}
        probe: Probe | None = None
        accuracies: dict[str, float] = {}
        has_feedback = actor.is_recurrent and actor.delayed_message_feedback
        for arm, plan in ablation_arms(has_feedback).items():
            if args.arms is not None and arm not in args.arms:
                continue
            generator = torch.Generator(device=args.device).manual_seed(args.seed)
            chunks = []
            for start in range(0, args.episodes, num_envs):
                batch_conditions = Conditions(conditions.directions[start:start+num_envs],
                                              conditions.colors[start:start+num_envs])
                chunks.append(run_balanced_rollout(wrapped, actor, batch_conditions, plan, generator))
            rollout = RolloutData(
                **{key: torch.cat([getattr(chunk, key) for chunk in chunks])
                   for key in ("sender_messages", "receiver_messages", "correct", "wrong", "timeout")},
                conditions=Conditions(conditions.directions.cpu(), conditions.colors.cpu()),
            )
            results[arm] = rollout.metrics()
            if arm == "normal":
                probe, accuracies = build_probe(
                    rollout,
                    architecture["channel_mode"],
                    args.calibration_fraction,
                    args.seed,
                    args.probe_step,
                )
    finally:
        env.close()

    assert probe is not None
    probe_path = Path(args.message_probe or default_probe_path(args.ckpt))
    save_message_probe(probe, probe_path)

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "observation_config": env_config,
        "evaluation_batch_size": num_envs,
        "channel_mode": architecture["channel_mode"],
        # A DRU is evaluated with hard bits, so "zero" is a valid all-zeros
        # message rather than silence; "shuffled" is the cleaner control.
        "message_unit": architecture["message_unit"],
        "delayed_message_feedback": architecture["delayed_message_feedback"],
        "sender_message_dim": architecture["sender_message_dim"],
        "receiver_message_dim": architecture["receiver_message_dim"],
        "probe_path": str(probe_path.resolve()),
        "held_out_probe_accuracy": accuracies,
        "arms": results,
        # The headline numbers: how much each channel is worth, cut on every
        # path. Shuffle rather than zero, because for a DRU the all-zeros code
        # is a valid message, not silence.
        **channel_effects(results),
    }
    output_path = Path(args.output or Path(args.ckpt).with_suffix(".evaluation.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
