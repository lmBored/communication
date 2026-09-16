"""DIAL-inspired direct differentiable sender-to-receiver actor."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class DirectDialActor(nn.Module):
    """One-way actor whose bounded message is optimized through PPO reward."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        sender_hidden_dims: tuple[int, ...] | list[int] = (64, 64),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        message_dim: int = 2,
    ) -> None:
        super().__init__()
        if message_dim < 1:
            raise ValueError("message_dim must be at least 1")
        if not hidden_dims or not sender_hidden_dims:
            raise ValueError("sender and receiver hidden dimensions cannot be empty")
        active_groups = list(obs_groups[obs_set])
        if set(active_groups) != {"sender", "receiver"} or len(active_groups) != 2:
            raise ValueError(
                "DirectDialActor requires exactly the sender and receiver groups"
            )
        if obs["sender"].ndim != 2 or obs["receiver"].ndim != 2:
            raise ValueError("DirectDialActor supports only flat private observations")

        self.obs_groups = active_groups
        self.sender_dim = obs["sender"].shape[-1]
        self.receiver_dim = obs["receiver"].shape[-1]
        self.obs_dim = self.sender_dim + self.receiver_dim
        self.message_dim = message_dim
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.sender_normalizer = EmpiricalNormalization(self.sender_dim)
            self.receiver_normalizer = EmpiricalNormalization(self.receiver_dim)
        else:
            self.sender_normalizer = nn.Identity()
            self.receiver_normalizer = nn.Identity()

        self.sender_encoder = MLP(
            self.sender_dim,
            message_dim,
            sender_hidden_dims,
            activation,
        )
        receiver_latent_dim = hidden_dims[-1]
        receiver_hidden_dims = hidden_dims[:-1] or hidden_dims
        self.receiver_encoder = MLP(
            self.receiver_dim,
            receiver_latent_dim,
            receiver_hidden_dims,
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
                raise ValueError("DirectDialActor requires a flat action distribution")
        else:
            self.distribution = None
            head_output_dim = output_dim
        self.action_head = nn.Linear(
            receiver_latent_dim + message_dim, head_output_dim
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.action_head)
        self.register_buffer(
            "_last_message", torch.empty(0, message_dim), persistent=False
        )

    @property
    def last_message(self) -> torch.Tensor:
        """Most recently emitted message, detached for diagnostics."""
        return self._last_message

    def encode_message(self, sender_obs: torch.Tensor | TensorDict) -> torch.Tensor:
        """Encode and bound the sender's private observation."""
        if isinstance(sender_obs, TensorDict):
            sender_obs = sender_obs["sender"]
        normalized = self.sender_normalizer(sender_obs)
        return torch.tanh(self.sender_encoder(normalized))

    def action_from_message(
        self, receiver_obs: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        """Compute deterministic actions from receiver state and a clamped message."""
        normalized = self.receiver_normalizer(receiver_obs)
        receiver_latent = self.receiver_encoder(normalized)
        head_output = self.action_head(torch.cat((receiver_latent, message), dim=-1))
        if self.distribution is not None:
            return self.distribution.deterministic_output(head_output)
        return head_output

    def _distribution_input(
        self, receiver_obs: torch.Tensor, message: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.receiver_normalizer(receiver_obs)
        receiver_latent = self.receiver_encoder(normalized)
        return self.action_head(torch.cat((receiver_latent, message), dim=-1))

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        message = self.encode_message(obs["sender"])
        self._last_message = message.detach()
        head_output = self._distribution_input(obs["receiver"], message)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(head_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(head_output)
        return head_output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        del hidden_state
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        message = self.encode_message(obs["sender"])
        receiver = self.receiver_encoder(self.receiver_normalizer(obs["receiver"]))
        return torch.cat((receiver, message), dim=-1)

    def reset(
        self,
        dones: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> None:
        del dones, hidden_state

    def get_hidden_state(self) -> HiddenState:
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        del dones

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

    def as_jit(self) -> nn.Module:
        return _DirectDialExport(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        return _DirectDialOnnxExport(self, verbose)


class _DirectDialExport(nn.Module):
    def __init__(self, actor: DirectDialActor) -> None:
        super().__init__()
        self.sender_dim = actor.sender_dim
        self.sender_normalizer = copy.deepcopy(actor.sender_normalizer)
        self.receiver_normalizer = copy.deepcopy(actor.receiver_normalizer)
        self.sender_encoder = copy.deepcopy(actor.sender_encoder)
        self.receiver_encoder = copy.deepcopy(actor.receiver_encoder)
        self.action_head = copy.deepcopy(actor.action_head)
        if actor.distribution is not None:
            self.deterministic_output = (
                actor.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sender = self.sender_normalizer(observations[:, : self.sender_dim])
        receiver = self.receiver_normalizer(observations[:, self.sender_dim :])
        message = torch.tanh(self.sender_encoder(sender))
        receiver_latent = self.receiver_encoder(receiver)
        output = self.action_head(torch.cat((receiver_latent, message), dim=-1))
        return self.deterministic_output(output)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _DirectDialOnnxExport(_DirectDialExport):
    is_recurrent: bool = False

    def __init__(self, actor: DirectDialActor, verbose: bool) -> None:
        super().__init__(actor)
        self.verbose = verbose
        self.input_size = actor.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["sender_receiver_obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]