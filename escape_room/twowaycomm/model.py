"""DIAL-inspired two-way differentiable actor for the twowaycomm scenario.

Both agents emit a bounded message and consume the partner's. The channel is an
internal activation: it never enters an observation tensor, the rollout buffer,
or the action space, so the only thing that trains it is the task reward flowing
back through the partner's action head.

Two channel modes are supported:

``same_step``
    Both messages are computed from the current observations and delivered
    within the same control step. There is no circular dependency because each
    message depends only on its own agent's observation.

``delayed``
    Each agent consumes the message its partner emitted one step earlier. With
    ``delayed_message_feedback`` the incoming message also reaches the message
    encoder, which is what makes a query->response protocol possible: the
    receiver can announce the door colour at ``t`` and the sender can reply with
    that colour's arrow direction at ``t+1``. The pair of messages is carried as
    the model's recurrent hidden state so rsl-rl stores and replays it with
    trajectory-aligned minibatches.

Two message units are supported:

``tanh``
    A continuous ``tanh`` channel, identical in training and execution.

``dru``
    DIAL's discretise/regularise unit (Foerster et al., 2016). Training emits
    ``sigmoid(m + sigma * eps)``; the noise pushes the encoder towards saturated
    logits, so that execution can emit the hard bit ``1{m > 0}`` without
    changing what the partner receives.

``dru_st``
    A straight-through DRU. Training already emits the hard bit
    ``1{m + sigma * eps > 0}``, so the partner trains on exactly what it will
    receive at execution, and the backward pass uses the sigmoid's gradient.
    Nothing rewards saturated logits, which keeps that gradient alive; the
    plain DRU's saturation is what freezes its code early.

Training noise ``eps`` is part of the policy input. An algorithm that wants a
correct PPO ratio samples it once per step and supplies it under
``MESSAGE_NOISE_KEY`` in the observation TensorDict, so the rollout storage
records it and the update replays exactly the same draw. Without the key the
actor samples fresh noise itself.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import CNN, MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

SENDER_IMAGE_GROUP = "sender_image"
RECEIVER_IMAGE_GROUP = "receiver_image"
DEFAULT_CNN_CFG: dict[str, object] = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "global_pool": "none",
}

CHANNEL_MODES: tuple[str, ...] = ("same_step", "delayed")
_CHANNEL_MODE_CODES = {mode: index for index, mode in enumerate(CHANNEL_MODES)}
MESSAGE_UNITS: tuple[str, ...] = ("tanh", "dru", "dru_st")
DISCRETE_MESSAGE_UNITS = frozenset({"dru", "dru_st"})
_MESSAGE_UNIT_CODES = {unit: index for index, unit in enumerate(MESSAGE_UNITS)}
# DIAL's training noise for the DRU; the paper uses sigma = 2. The tanh unit
# defaults to no noise, which is the behaviour every existing checkpoint has.
DEFAULT_DRU_NOISE_STD = 2.0
MESSAGE_NOISE_KEY = "message_noise"


def default_message_noise_std(message_unit: str) -> float:
    # The straight-through unit needs no noise to binarize; it is optional.
    return DEFAULT_DRU_NOISE_STD if message_unit == "dru" else 0.0


def _encode_image(cnn: CNN, image: torch.Tensor) -> torch.Tensor:
    """Run a CNN over any number of leading batch dimensions."""
    leading = image.shape[:-3]
    encoded = cnn(image.reshape(-1, *image.shape[-3:]))
    return encoded.reshape(*leading, -1)


def _split_hidden(
    hidden_state: HiddenState, sender_width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize the several shapes rsl-rl hands a hidden state back in.

    The rollout storage returns a ``list`` for a tuple-valued hidden state, so
    accepting only ``tuple`` would break during the policy update.
    """
    if isinstance(hidden_state, (tuple, list)):
        if len(hidden_state) != 2:
            raise ValueError(
                f"two-way hidden state must hold two tensors, got {len(hidden_state)}"
            )
        sender, receiver = hidden_state
    elif isinstance(hidden_state, torch.Tensor):
        if sender_width is None:
            if hidden_state.shape[-1] % 2:
                raise ValueError("packed hidden state must have an even last dimension")
            sender_width = hidden_state.shape[-1] // 2
        sender = hidden_state[..., :sender_width]
        receiver = hidden_state[..., sender_width:]
    else:
        raise TypeError(f"unsupported hidden state type: {type(hidden_state)!r}")
    return sender, receiver


class TwoWayDialActor(nn.Module):
    """Symmetric actor whose two bounded channels are optimized through reward."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        sender_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        sender_message_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        receiver_message_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        message_dim: int = 4,
        sender_message_dim: int | None = None,
        receiver_message_dim: int | None = None,
        channel_mode: str = "same_step",
        delayed_message_feedback: bool = True,
        message_unit: str = "tanh",
        message_noise_std: float | None = None,
        cnn_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        # The two channels may be sized independently: narrowing the sender is
        # what stops it from simply broadcasting every arrow direction and makes
        # the back-channel worth using.
        sender_message_dim = (
            message_dim if sender_message_dim is None else sender_message_dim
        )
        receiver_message_dim = (
            message_dim if receiver_message_dim is None else receiver_message_dim
        )
        if min(message_dim, sender_message_dim, receiver_message_dim) < 1:
            raise ValueError("message widths must be at least 1")
        if channel_mode not in CHANNEL_MODES:
            raise ValueError(
                f"unknown channel_mode {channel_mode!r}; use one of {CHANNEL_MODES}"
            )
        if message_unit not in MESSAGE_UNITS:
            raise ValueError(
                f"unknown message_unit {message_unit!r}; use one of {MESSAGE_UNITS}"
            )
        if message_noise_std is None:
            message_noise_std = default_message_noise_std(message_unit)
        if message_noise_std < 0.0:
            raise ValueError("message_noise_std must be non-negative")
        if not all(
            (
                hidden_dims,
                sender_hidden_dims,
                sender_message_hidden_dims,
                receiver_message_hidden_dims,
            )
        ):
            raise ValueError("hidden dimensions cannot be empty")
        active_groups = list(obs_groups[obs_set])
        vector_groups = {"sender", "receiver"}
        pixel_groups = vector_groups | {SENDER_IMAGE_GROUP, RECEIVER_IMAGE_GROUP}
        if set(active_groups) == vector_groups and len(active_groups) == 2:
            self.obs_mode = "vector"
        elif set(active_groups) == pixel_groups and len(active_groups) == 4:
            self.obs_mode = "pixel"
        else:
            raise ValueError(
                "TwoWayDialActor requires the sender and receiver groups, "
                "optionally plus sender_image and receiver_image"
            )
        if obs["sender"].ndim != 2 or obs["receiver"].ndim != 2:
            raise ValueError("TwoWayDialActor supports only flat private observations")
        if self.obs_mode == "pixel" and (
            obs[SENDER_IMAGE_GROUP].ndim != 4 or obs[RECEIVER_IMAGE_GROUP].ndim != 4
        ):
            raise ValueError("image observations must be (batch, channels, H, W)")
        if output_dim % 2:
            raise ValueError(
                f"output_dim must split evenly across two agents, got {output_dim}"
            )

        self.obs_groups = active_groups
        self.sender_dim = obs["sender"].shape[-1]
        self.receiver_dim = obs["receiver"].shape[-1]
        self.obs_dim = self.sender_dim + self.receiver_dim
        self.message_dim = message_dim
        self.sender_message_dim = sender_message_dim
        self.receiver_message_dim = receiver_message_dim
        self.agent_action_dim = output_dim // 2
        self.obs_normalization = obs_normalization
        self.channel_mode = channel_mode
        self.delayed_message_feedback = delayed_message_feedback
        self.message_unit = message_unit
        self.message_noise_std = float(message_noise_std)
        # Instance attribute shadows the class default; PPO reads it off the
        # constructed actor to pick the recurrent rollout storage.
        self.is_recurrent = channel_mode == "delayed"
        # The partner message reaches the message encoder only when the channel
        # is delayed; in same-step mode that would be a circular dependency.
        self._feedback = self.is_recurrent and delayed_message_feedback

        if obs_normalization:
            self.sender_normalizer = EmpiricalNormalization(self.sender_dim)
            self.receiver_normalizer = EmpiricalNormalization(self.receiver_dim)
        else:
            self.sender_normalizer = nn.Identity()
            self.receiver_normalizer = nn.Identity()

        # In pixel mode the private half of each observation is blank and the
        # arrows / door colour are only visible through the camera, so the CNN
        # output has to reach the MESSAGE encoder, not just the control path.
        if self.obs_mode == "pixel":
            cnn_cfg = dict(DEFAULT_CNN_CFG if cnn_cfg is None else cnn_cfg)
            self.sender_cnn = self._build_cnn(obs[SENDER_IMAGE_GROUP], cnn_cfg)
            self.receiver_cnn = self._build_cnn(obs[RECEIVER_IMAGE_GROUP], cnn_cfg)
            sender_feature_dim = self.sender_dim + self.sender_cnn.output_dim
            receiver_feature_dim = self.receiver_dim + self.receiver_cnn.output_dim
        else:
            self.sender_cnn = None
            self.receiver_cnn = None
            sender_feature_dim = self.sender_dim
            receiver_feature_dim = self.receiver_dim
        self.sender_feature_dim = sender_feature_dim
        self.receiver_feature_dim = receiver_feature_dim

        # With feedback each encoder additionally reads what its partner said.
        self.sender_message_encoder = MLP(
            sender_feature_dim + (receiver_message_dim if self._feedback else 0),
            sender_message_dim,
            sender_message_hidden_dims,
            activation,
        )
        self.receiver_message_encoder = MLP(
            receiver_feature_dim + (sender_message_dim if self._feedback else 0),
            receiver_message_dim,
            receiver_message_hidden_dims,
            activation,
        )
        sender_latent_dim = sender_hidden_dims[-1]
        self.sender_latent_encoder = MLP(
            sender_feature_dim,
            sender_latent_dim,
            sender_hidden_dims[:-1] or sender_hidden_dims,
            activation,
        )
        receiver_latent_dim = hidden_dims[-1]
        self.receiver_latent_encoder = MLP(
            receiver_feature_dim,
            receiver_latent_dim,
            hidden_dims[:-1] or hidden_dims,
            activation,
        )

        self.distribution: Distribution | None
        if distribution_cfg is not None:
            distribution_cfg = distribution_cfg.copy()
            dist_class: type[Distribution] = resolve_callable(
                distribution_cfg.pop("class_name")
            )
            self.distribution = dist_class(output_dim, **distribution_cfg)
            head_output_dim = self.distribution.input_dim
            if not isinstance(head_output_dim, int):
                raise ValueError("TwoWayDialActor requires a flat action distribution")
            if head_output_dim != output_dim:
                raise ValueError(
                    "TwoWayDialActor requires a distribution whose input width "
                    f"matches the action width, got {head_output_dim} != {output_dim}"
                )
        else:
            self.distribution = None
            head_output_dim = output_dim

        # Each head consumes the message its partner emitted.
        self.sender_head = nn.Linear(
            sender_latent_dim + receiver_message_dim, self.agent_action_dim
        )
        self.receiver_head = nn.Linear(
            receiver_latent_dim + sender_message_dim, self.agent_action_dim
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.sender_head)
            self.distribution.init_mlp_weights(self.receiver_head)

        # Persistent so a checkpoint records which channel it was trained with.
        # Without it a delayed checkpoint loads cleanly as same-step, because the
        # two modes share identical parameter shapes when feedback is disabled.
        self.register_buffer(
            "channel_mode_code",
            torch.tensor(_CHANNEL_MODE_CODES[channel_mode], dtype=torch.long),
            persistent=True,
        )
        # Same reason: tanh and DRU checkpoints have identical parameter shapes,
        # and the unit decides what the partner heads were trained to read.
        self.register_buffer(
            "message_unit_code",
            torch.tensor(_MESSAGE_UNIT_CODES[message_unit], dtype=torch.long),
            persistent=True,
        )
        # Sized from the constructor observations because PPO records
        # get_hidden_state() *before* the first forward, and the rollout storage
        # allocates its buffers from whatever shape that first record has.
        initial_batch = obs["sender"].shape[0]
        self.register_buffer(
            "_prev_sender_message",
            torch.zeros(1, initial_batch, sender_message_dim),
            persistent=False,
        )
        self.register_buffer(
            "_prev_receiver_message",
            torch.zeros(1, initial_batch, receiver_message_dim),
            persistent=False,
        )
        self.register_buffer(
            "_last_sender_message", torch.empty(0, sender_message_dim), persistent=False
        )
        self.register_buffer(
            "_last_receiver_message",
            torch.empty(0, receiver_message_dim),
            persistent=False,
        )

    # Diagnostics.

    @property
    def last_sender_message(self) -> torch.Tensor:
        """Message most recently emitted by the sender, detached."""
        return self._last_sender_message

    @property
    def last_receiver_message(self) -> torch.Tensor:
        """Message most recently emitted by the receiver, detached."""
        return self._last_receiver_message

    @property
    def last_message_logits(self) -> dict[str, torch.Tensor]:
        """Pre-unit encoder outputs from the latest encode, detached."""
        logits = getattr(self, "_last_logits", None)
        if logits is None:
            return {}
        return {"sender": logits[0], "receiver": logits[1]}

    @property
    def message_noise_dim(self) -> int:
        """Width of the noise an algorithm should record per step (0 = none)."""
        if self.message_noise_std == 0.0:
            return 0
        return self.sender_message_dim + self.receiver_message_dim

    @property
    def last_messages(self) -> dict[str, torch.Tensor]:
        return {
            "sender": self._last_sender_message,
            "receiver": self._last_receiver_message,
        }

    # Channel.

    @staticmethod
    def _build_cnn(image: torch.Tensor, cfg: dict) -> CNN:
        channels, height, width = image.shape[-3:]
        return CNN(
            input_dim=(height, width),
            input_channels=channels,
            flatten=True,
            **cfg,
        )

    def _features(
        self, obs: TensorDict | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-agent encoder input: normalized proprioception, plus pixels."""
        sender_obs, receiver_obs = self._obs_pair(obs)
        sender = self.sender_normalizer(sender_obs)
        receiver = self.receiver_normalizer(receiver_obs)
        if self.obs_mode == "pixel":
            if isinstance(obs, (tuple, list)):
                raise ValueError(
                    "pixel mode needs the full observation TensorDict, not a "
                    "sender/receiver pair"
                )
            sender = torch.cat(
                (sender, _encode_image(self.sender_cnn, obs[SENDER_IMAGE_GROUP])),
                dim=-1,
            )
            receiver = torch.cat(
                (receiver, _encode_image(self.receiver_cnn, obs[RECEIVER_IMAGE_GROUP])),
                dim=-1,
            )
        return sender, receiver

    @staticmethod
    def _obs_pair(
        obs: TensorDict | tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(obs, (tuple, list)):
            return obs[0], obs[1]
        return obs["sender"], obs["receiver"]

    def _bound(
        self, raw: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply the message unit: noisy and smooth in training, clean in eval."""
        if not self.training:
            if self.message_unit in DISCRETE_MESSAGE_UNITS:
                return (raw > 0.0).to(raw.dtype)
            return torch.tanh(raw)
        if self.message_noise_std > 0.0:
            if noise is None:
                noise = torch.randn_like(raw)
            raw = raw + noise * self.message_noise_std
        if self.message_unit == "dru":
            return torch.sigmoid(raw)
        if self.message_unit == "dru_st":
            soft = torch.sigmoid(raw)
            hard = (raw > 0.0).to(raw.dtype)
            # Forward value: exactly the hard bit (the bracket is exactly zero).
            # Gradient: the sigmoid's.
            return hard + (soft - soft.detach())
        return torch.tanh(raw)

    def _message_noise(
        self, obs: TensorDict | tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Recorded standard-normal draws, split per channel, if supplied."""
        if isinstance(obs, (tuple, list)) or MESSAGE_NOISE_KEY not in obs.keys():
            return None, None
        noise = obs[MESSAGE_NOISE_KEY]
        return (
            noise[..., : self.sender_message_dim],
            noise[..., self.sender_message_dim :],
        )

    def encode_messages(
        self,
        obs: TensorDict | tuple[torch.Tensor, torch.Tensor],
        prev: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sender_message, receiver_message)``, each bounded to [-1, 1].

        ``prev`` is the pair emitted one step earlier and is read only when the
        channel is delayed with feedback enabled.
        """
        sender, receiver = self._features(obs)
        sender_noise, receiver_noise = self._message_noise(obs)
        if self._feedback:
            if prev is None:
                raise ValueError(
                    "delayed feedback requires the previous message pair"
                )
            # Each agent hears what its partner said, not what it said itself.
            sender = torch.cat((sender, prev[1]), dim=-1)
            receiver = torch.cat((receiver, prev[0]), dim=-1)
        sender_logits = self.sender_message_encoder(sender)
        receiver_logits = self.receiver_message_encoder(receiver)
        self._last_logits = (sender_logits.detach(), receiver_logits.detach())
        return (
            self._bound(sender_logits, sender_noise),
            self._bound(receiver_logits, receiver_noise),
        )

    def _head_inputs(
        self,
        obs: TensorDict | tuple[torch.Tensor, torch.Tensor],
        sender_message: torch.Tensor,
        receiver_message: torch.Tensor,
    ) -> torch.Tensor:
        """Pre-distribution output, agent-major: sender controls then receiver."""
        sender_features, receiver_features = self._features(obs)
        sender_latent = self.sender_latent_encoder(sender_features)
        receiver_latent = self.receiver_latent_encoder(receiver_features)
        # A message emitted by the sender is consumed by the receiver's head.
        sender_action = self.sender_head(
            torch.cat((sender_latent, receiver_message), dim=-1)
        )
        receiver_action = self.receiver_head(
            torch.cat((receiver_latent, sender_message), dim=-1)
        )
        return torch.cat((sender_action, receiver_action), dim=-1)

    def action_from_messages(
        self,
        obs: TensorDict | tuple[torch.Tensor, torch.Tensor],
        sender_message: torch.Tensor,
        receiver_message: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic actions from externally supplied (possibly intervened)
        messages. Mode-agnostic: a delayed caller passes the previous pair."""
        head_output = self._head_inputs(obs, sender_message, receiver_message)
        if self.distribution is not None:
            return self.distribution.deterministic_output(head_output)
        return head_output

    # Hidden state (delayed mode only).

    def _ensure_prev(self, batch: int, reference: torch.Tensor) -> None:
        if self._prev_sender_message.shape[1] != batch:
            options = {"device": reference.device, "dtype": reference.dtype}
            self._prev_sender_message = torch.zeros(
                1, batch, self.sender_message_dim, **options
            )
            self._prev_receiver_message = torch.zeros(
                1, batch, self.receiver_message_dim, **options
            )

    def _prev_pair(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_prev(reference.shape[0], reference)
        return (
            self._prev_sender_message.squeeze(0),
            self._prev_receiver_message.squeeze(0),
        )

    def get_hidden_state(self) -> HiddenState:
        if not self.is_recurrent:
            return None
        return (self._prev_sender_message, self._prev_receiver_message)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        if not self.is_recurrent:
            return
        if dones is None:
            if hidden_state is None:
                self._prev_sender_message = torch.zeros_like(self._prev_sender_message)
                self._prev_receiver_message = torch.zeros_like(
                    self._prev_receiver_message
                )
            else:
                sender, receiver = _split_hidden(hidden_state, self.sender_message_dim)
                self._prev_sender_message = sender
                self._prev_receiver_message = receiver
            return
        if hidden_state is not None:
            raise NotImplementedError(
                "resetting a subset of worlds to a supplied hidden state is not "
                "supported"
            )
        if self._prev_sender_message.shape[1] != dones.shape[0]:
            self._ensure_prev(dones.shape[0], self._prev_sender_message)
        keep = (dones != 1).view(1, -1, 1)
        # Functional, not in-place: after a rollout step these buffers hold
        # inference tensors, which cannot be written in place from outside
        # inference mode.
        self._prev_sender_message = torch.where(
            keep, self._prev_sender_message, torch.zeros_like(self._prev_sender_message)
        )
        self._prev_receiver_message = torch.where(
            keep,
            self._prev_receiver_message,
            torch.zeros_like(self._prev_receiver_message),
        )

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones
        if not self.is_recurrent:
            return
        self._prev_sender_message = self._prev_sender_message.detach()
        self._prev_receiver_message = self._prev_receiver_message.detach()

    def _batched_delayed(
        self, obs: TensorDict, hidden_state: HiddenState
    ) -> torch.Tensor:
        """Replay a padded ``[L, n_traj]`` rollout slice with a one-step delay."""
        if hidden_state is None:
            raise ValueError(
                "delayed TwoWayDialActor requires a hidden state during the update"
            )
        sender_hidden, receiver_hidden = _split_hidden(
            hidden_state, self.sender_message_dim
        )
        # The storage allocates these inside the rollout's inference_mode block,
        # so they arrive as inference tensors and cannot be used in autograd.
        sender_hidden = sender_hidden.clone()
        receiver_hidden = receiver_hidden.clone()

        if not self._feedback:
            # Messages depend only on their own observation, so the whole delay
            # is a depth-one shift; no python loop and no deep BPTT.
            sender_message, receiver_message = self.encode_messages(obs)
            prev_sender = torch.cat((sender_hidden, sender_message[:-1]), dim=0)
            prev_receiver = torch.cat((receiver_hidden, receiver_message[:-1]), dim=0)
            return self._head_inputs(obs, prev_sender, prev_receiver)

        # With feedback the messages are genuinely recurrent, so the scan and its
        # backpropagation through time are unavoidable.
        prev = (sender_hidden.squeeze(0), receiver_hidden.squeeze(0))
        outputs = []
        for step in range(obs.batch_size[0]):
            obs_t = obs[step]
            outputs.append(self._head_inputs(obs_t, prev[0], prev[1]))
            prev = self.encode_messages(obs_t, prev)
        return torch.stack(outputs, dim=0)

    def _remember(
        self, sender_message: torch.Tensor, receiver_message: torch.Tensor
    ) -> None:
        self._last_sender_message = sender_message.detach()
        self._last_receiver_message = receiver_message.detach()

    # rsl-rl interface.

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if not self.is_recurrent:
            del hidden_state
            if masks is not None:
                obs = unpad_trajectories(obs, masks)
            sender_message, receiver_message = self.encode_messages(obs)
            self._remember(sender_message, receiver_message)
            head_output = self._head_inputs(obs, sender_message, receiver_message)
        elif masks is None:
            prev = self._prev_pair(obs["sender"])
            sender_message, receiver_message = self.encode_messages(obs, prev)
            # Consume the previous pair before overwriting it.
            head_output = self._head_inputs(obs, prev[0], prev[1])
            self._prev_sender_message = sender_message.detach().unsqueeze(0)
            self._prev_receiver_message = receiver_message.detach().unsqueeze(0)
            self._remember(sender_message, receiver_message)
        else:
            # Batched replay never touches the persistent buffers: n_traj differs
            # from num_envs and the next rollout must resume from its own state.
            head_output = unpad_trajectories(
                self._batched_delayed(obs, hidden_state), masks
            )
        if self.distribution is None:
            return head_output
        if stochastic_output:
            self.distribution.update(head_output)
            return self.distribution.sample()
        return self.distribution.deterministic_output(head_output)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        if self.is_recurrent:
            if hidden_state is not None:
                sender_message, receiver_message = _split_hidden(
                    hidden_state, self.sender_message_dim
                )
                sender_message = sender_message.squeeze(0)
                receiver_message = receiver_message.squeeze(0)
            else:
                sender_message, receiver_message = self._prev_pair(obs["sender"])
        else:
            sender_message, receiver_message = self.encode_messages(obs)
        sender_features, receiver_features = self._features(obs)
        sender_latent = self.sender_latent_encoder(sender_features)
        receiver_latent = self.receiver_latent_encoder(receiver_features)
        return torch.cat(
            (sender_latent, receiver_message, receiver_latent, sender_message), dim=-1
        )

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.sender_normalizer.update(obs["sender"])
            self.receiver_normalizer.update(obs["receiver"])

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        unit_key = prefix + "message_unit_code"
        if unit_key not in state_dict:
            # Checkpoints from before the DRU existed were all tanh.
            state_dict[unit_key] = torch.tensor(
                _MESSAGE_UNIT_CODES["tanh"], dtype=torch.long
            )
        stored_unit = int(state_dict[unit_key].reshape(-1)[0].item())
        current_unit = int(self.message_unit_code.item())
        if stored_unit != current_unit:
            error_msgs.append(
                f"message_unit mismatch: checkpoint was trained with "
                f"{MESSAGE_UNITS[stored_unit]!r} but this actor was built with "
                f"{MESSAGE_UNITS[current_unit]!r}"
            )
        key = prefix + "channel_mode_code"
        if key in state_dict:
            stored = int(state_dict[key].reshape(-1)[0].item())
            current = int(self.channel_mode_code.item())
            if stored != current:
                error_msgs.append(
                    f"channel_mode mismatch: checkpoint was trained with "
                    f"{CHANNEL_MODES[stored]!r} but this actor was built with "
                    f"{CHANNEL_MODES[current]!r}"
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def as_jit(self) -> nn.Module:
        if self.obs_mode == "pixel":
            raise NotImplementedError(
                "pixel-mode export would need the image tensors alongside the "
                "concatenated vector observation"
            )
        if self.is_recurrent:
            return _TwoWayDialDelayedExport(self)
        return _TwoWayDialExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        if self.obs_mode == "pixel":
            raise NotImplementedError(
                "pixel-mode export would need the image tensors alongside the "
                "concatenated vector observation"
            )
        if self.is_recurrent:
            return _TwoWayDialDelayedOnnxExport(self, verbose)
        return _TwoWayDialOnnxExport(self, verbose)


class _TwoWayDialExportBase(nn.Module):
    """Shared parameter copy for the same-step and delayed export modules."""

    def __init__(self, actor: TwoWayDialActor) -> None:
        super().__init__()
        self.sender_dim = actor.sender_dim
        self.sender_message_dim = actor.sender_message_dim
        self.receiver_message_dim = actor.receiver_message_dim
        self.sender_normalizer = copy.deepcopy(actor.sender_normalizer)
        self.receiver_normalizer = copy.deepcopy(actor.receiver_normalizer)
        self.sender_message_encoder = copy.deepcopy(actor.sender_message_encoder)
        self.receiver_message_encoder = copy.deepcopy(actor.receiver_message_encoder)
        self.sender_latent_encoder = copy.deepcopy(actor.sender_latent_encoder)
        self.receiver_latent_encoder = copy.deepcopy(actor.receiver_latent_encoder)
        self.sender_head = copy.deepcopy(actor.sender_head)
        self.receiver_head = copy.deepcopy(actor.receiver_head)
        # Exports are execution-time policies, so a DRU emits hard bits.
        self.discrete_messages = actor.message_unit in DISCRETE_MESSAGE_UNITS
        if actor.distribution is not None:
            self.deterministic_output = (
                actor.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def _emit(self, raw: torch.Tensor) -> torch.Tensor:
        if self.discrete_messages:
            return (raw > 0.0).to(raw.dtype)
        return torch.tanh(raw)

    def _heads(
        self,
        sender: torch.Tensor,
        receiver: torch.Tensor,
        sender_message: torch.Tensor,
        receiver_message: torch.Tensor,
    ) -> torch.Tensor:
        sender_action = self.sender_head(
            torch.cat((self.sender_latent_encoder(sender), receiver_message), dim=-1)
        )
        receiver_action = self.receiver_head(
            torch.cat((self.receiver_latent_encoder(receiver), sender_message), dim=-1)
        )
        return torch.cat((sender_action, receiver_action), dim=-1)


class _TwoWayDialExport(_TwoWayDialExportBase):
    """Stateless same-step export over one concatenated observation tensor."""

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sender = self.sender_normalizer(observations[:, : self.sender_dim])
        receiver = self.receiver_normalizer(observations[:, self.sender_dim :])
        sender_message = self._emit(self.sender_message_encoder(sender))
        receiver_message = self._emit(self.receiver_message_encoder(receiver))
        return self.deterministic_output(
            self._heads(sender, receiver, sender_message, receiver_message)
        )

    @torch.jit.export
    def reset(self) -> None:
        pass


class _TwoWayDialDelayedExport(_TwoWayDialExportBase):
    """Stateful delayed export; like rsl-rl's RNN exports it is batch-size one."""

    def __init__(self, actor: TwoWayDialActor) -> None:
        super().__init__(actor)
        self.feedback = actor._feedback
        self.register_buffer(
            "prev_sender_message", torch.zeros(1, actor.sender_message_dim)
        )
        self.register_buffer(
            "prev_receiver_message", torch.zeros(1, actor.receiver_message_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sender = self.sender_normalizer(observations[:, : self.sender_dim])
        receiver = self.receiver_normalizer(observations[:, self.sender_dim :])
        prev_sender = self.prev_sender_message
        prev_receiver = self.prev_receiver_message
        if self.feedback:
            sender_input = torch.cat((sender, prev_receiver), dim=-1)
            receiver_input = torch.cat((receiver, prev_sender), dim=-1)
        else:
            sender_input = sender
            receiver_input = receiver
        sender_message = self._emit(self.sender_message_encoder(sender_input))
        receiver_message = self._emit(self.receiver_message_encoder(receiver_input))
        output = self._heads(sender, receiver, prev_sender, prev_receiver)
        self.prev_sender_message[:] = sender_message
        self.prev_receiver_message[:] = receiver_message
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        self.prev_sender_message[:] = 0.0
        self.prev_receiver_message[:] = 0.0


class _TwoWayDialOnnxExport(_TwoWayDialExport):
    is_recurrent: bool = False

    def __init__(self, actor: TwoWayDialActor, verbose: bool) -> None:
        super().__init__(actor)
        self.verbose = verbose
        self.input_size = actor.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["sender_receiver_obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]


class _TwoWayDialDelayedOnnxExport(_TwoWayDialExportBase):
    """Stateless delayed export following rsl-rl's hidden-in/hidden-out shape."""

    is_recurrent: bool = True

    def __init__(self, actor: TwoWayDialActor, verbose: bool) -> None:
        super().__init__(actor)
        self.verbose = verbose
        self.feedback = actor._feedback
        self.input_size = actor.obs_dim

    def forward(
        self,
        observations: torch.Tensor,
        prev_sender_message: torch.Tensor,
        prev_receiver_message: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sender = self.sender_normalizer(observations[:, : self.sender_dim])
        receiver = self.receiver_normalizer(observations[:, self.sender_dim :])
        if self.feedback:
            sender_input = torch.cat((sender, prev_receiver_message), dim=-1)
            receiver_input = torch.cat((receiver, prev_sender_message), dim=-1)
        else:
            sender_input = sender
            receiver_input = receiver
        sender_message = self._emit(self.sender_message_encoder(sender_input))
        receiver_message = self._emit(self.receiver_message_encoder(receiver_input))
        actions = self.deterministic_output(
            self._heads(
                sender, receiver, prev_sender_message, prev_receiver_message
            )
        )
        return actions, sender_message, receiver_message

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(1, self.input_size),
            torch.zeros(1, self.sender_message_dim),
            torch.zeros(1, self.receiver_message_dim),
        )

    @property
    def input_names(self) -> list[str]:
        return ["sender_receiver_obs", "prev_sender_message", "prev_receiver_message"]

    @property
    def output_names(self) -> list[str]:
        return ["actions", "sender_message", "receiver_message"]
