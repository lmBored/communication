"""Multi-agent PPO (MAPPO) for the two-way communication task.

rsl-rl's PPO treats the six controls as one Gaussian, so it clips a single
joint ratio ``ratio_sender * ratio_receiver`` against a single team value. This
module implements MAPPO as described by Yu et al., "The Surprising
Effectiveness of PPO in Cooperative Multi-Agent Games" (NeurIPS 2022):

* a **per-agent** clipped surrogate: each agent's ratio is clipped on its own,
  and the policy loss is the mean over agents;
* a **centralized, agent-conditioned critic** ``V_i(s)``: the privileged global
  state plus a one-hot agent id, one shared network, one value per agent;
* **per-agent rewards, GAE and advantage normalization**, so a reward that only
  one agent caused (``reward_sharing="individual"``) only enters that agent's
  advantage;
* **ValueNorm** (debiased running mean/variance of the returns, per agent) and
  a clipped **Huber** value loss;
* **separate actor and critic optimizers**, each with its own gradient clip.

The actor stays decentralized at execution: each head reads only its own
latent and the message its partner sent. That message is a differentiable
activation, so the receiver's surrogate still back-propagates into the
sender's message encoder (and vice versa) -- DIAL's cross-agent gradient.

``DialPPO`` is the matching single-ratio baseline. Both record the DRU training
noise in the rollout (see ``model.MESSAGE_NOISE_KEY``), which is what keeps the
first-epoch ratio at exactly one.
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal, kl_divergence

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import (
    resolve_callable,
    resolve_obs_groups,
    resolve_optimizer,
    unpad_trajectories,
)

from escape_room.twowaycomm.model import MESSAGE_NOISE_KEY

AGENT_NAMES: tuple[str, ...] = ("sender", "receiver")
VALUE_LOSSES: tuple[str, ...] = ("huber", "mse")
# A message counts as saturated beyond these bounds. For a DRU this is the
# fraction of training messages that execution-time discretization would leave
# unchanged in effect, i.e. how ready the protocol is to be binarized.
_SATURATION_BOUNDS = {"dru": (0.1, 0.9), "dru_st": (0.1, 0.9), "tanh": (-0.9, 0.9)}


# Per-agent views of the factorized Gaussian policy.


def split_agents(values: torch.Tensor, num_agents: int) -> torch.Tensor:
    """``[..., num_agents * k] -> [..., num_agents, k]`` (agent-major layout)."""
    if values.shape[-1] % num_agents:
        raise ValueError(
            f"last dimension {values.shape[-1]} does not split across "
            f"{num_agents} agents"
        )
    return values.reshape(*values.shape[:-1], num_agents, -1)


def agent_log_prob(
    params: tuple[torch.Tensor, ...], actions: torch.Tensor, num_agents: int
) -> torch.Tensor:
    """Each agent's action log-probability, ``[..., num_agents]``.

    The Gaussian is independent per dimension, so the joint log-probability is
    exactly the sum of these.
    """
    mean, std = params
    return split_agents(Normal(mean, std).log_prob(actions), num_agents).sum(-1)


def agent_entropy(params: tuple[torch.Tensor, ...], num_agents: int) -> torch.Tensor:
    mean, std = params
    return split_agents(Normal(mean, std).entropy(), num_agents).sum(-1)


def agent_kl(
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
    num_agents: int,
) -> torch.Tensor:
    kl = kl_divergence(Normal(*old_params), Normal(*new_params))
    return split_agents(kl, num_agents).sum(-1)


def _per_agent_mean(values: torch.Tensor) -> torch.Tensor:
    """Average every leading dimension, keeping the trailing agent axis."""
    return values.reshape(-1, values.shape[-1]).mean(0)


def clipped_surrogate(
    ratio: torch.Tensor, advantages: torch.Tensor, clip_param: float
) -> torch.Tensor:
    """PPO's pessimistic clipped objective as a loss, one value per agent.

    ``ratio`` and ``advantages`` are ``[..., num_agents]``; each agent's ratio is
    clipped on its own, so one agent leaving the trust region does not switch
    off the other agent's gradient (as clipping the joint ratio would).
    """
    surrogate = -advantages * ratio
    surrogate_clipped = -advantages * torch.clamp(
        ratio, 1.0 - clip_param, 1.0 + clip_param
    )
    return _per_agent_mean(torch.max(surrogate, surrogate_clipped))


def huber(error: torch.Tensor, delta: float) -> torch.Tensor:
    """MAPPO's Huber loss: quadratic (``e^2 / 2``) inside ``delta``."""
    absolute = error.abs()
    return torch.where(
        absolute <= delta, 0.5 * error.pow(2), delta * (absolute - 0.5 * delta)
    )


# Critic and value normalization.


class MultiAgentCritic(MLPModel):
    """Centralized critic returning one value per agent, ``[..., num_agents]``.

    MAPPO's agent-specific global state: the shared privileged observation is
    tiled once per agent and tagged with a one-hot agent id before a single MLP.
    Inherits observation normalization, trajectory unpadding and the no-op
    hidden-state interface from ``MLPModel``.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int = 1,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        num_agents: int = 2,
    ) -> None:
        if output_dim != 1:
            raise ValueError("MultiAgentCritic produces one value per agent")
        if num_agents < 1:
            raise ValueError("num_agents must be at least 1")
        # Read by _get_latent_dim() from inside MLPModel.__init__.
        self.num_agents = num_agents
        super().__init__(
            obs, obs_groups, obs_set, 1, hidden_dims, activation, obs_normalization
        )
        self.register_buffer("agent_ids", torch.eye(num_agents), persistent=False)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.num_agents

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state, stochastic_output
        if masks is not None:
            obs = unpad_trajectories(obs, masks)
        state = self.get_latent(obs)
        leading = state.shape[:-1]
        tiled = state.unsqueeze(-2).expand(*leading, self.num_agents, state.shape[-1])
        ids = self.agent_ids.expand(*leading, self.num_agents, self.num_agents)
        return self.mlp(torch.cat((tiled, ids), dim=-1)).squeeze(-1)


class ValueNorm(nn.Module):
    """Per-agent debiased running mean/variance of the value targets.

    Mirrors ``onpolicy/utils/valuenorm.py`` from the MAPPO reference code.
    """

    def __init__(
        self, num_agents: int, beta: float = 0.99999, epsilon: float = 1.0e-5
    ) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.register_buffer("running_mean", torch.zeros(num_agents))
        self.register_buffer("running_mean_sq", torch.zeros(num_agents))
        self.register_buffer("debiasing_term", torch.zeros(()))

    def mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debias
        mean_sq = self.running_mean_sq / debias
        return mean, (mean_sq - mean.pow(2)).clamp(min=1.0e-2)

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        flat = values.reshape(-1, values.shape[-1]).to(self.running_mean)
        self.running_mean.mul_(self.beta).add_(flat.mean(0) * (1.0 - self.beta))
        self.running_mean_sq.mul_(self.beta).add_(
            flat.pow(2).mean(0) * (1.0 - self.beta)
        )
        self.debiasing_term.mul_(self.beta).add_(1.0 - self.beta)

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self.mean_var()
        return (values - mean) / var.sqrt()

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        mean, var = self.mean_var()
        return values * var.sqrt() + mean


# Storage.


class MultiAgentRolloutStorage(RolloutStorage):
    """Rollout storage whose RL scalars carry a trailing agent axis.

    Only allocation and ``add_transition`` differ: the stock minibatch
    generators slice ``[:, start:stop]`` and ``flatten(0, 1)``, which work for
    any trailing shape, so both the feed-forward and the recurrent (delayed
    channel) paths are reused unchanged. ``dones`` stays ``[T, N, 1]`` and
    broadcasts across agents, because the episode ends for both at once.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int, ...] | list[int],
        num_agents: int,
        device: str = "cpu",
    ) -> None:
        if training_type != "rl":
            raise ValueError("MultiAgentRolloutStorage only supports RL training")
        super().__init__(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, device
        )
        self.num_agents = num_agents
        shape = (num_transitions_per_env, num_envs, num_agents)
        self.rewards = torch.zeros(shape, device=device)
        self.values = torch.zeros(shape, device=device)
        self.actions_log_prob = torch.zeros(shape, device=device)
        self.returns = torch.zeros(shape, device=device)
        self.advantages = torch.zeros(shape, device=device)

    def add_transition(self, transition: RolloutStorage.Transition) -> None:
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding "
                "new transitions."
            )
        agents = self.num_agents
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, agents))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values.view(-1, agents))
        self.actions_log_prob[self.step].copy_(
            transition.actions_log_prob.view(-1, agents)
        )
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in transition.distribution_params
            )
        for index, param in enumerate(transition.distribution_params):
            self.distribution_params[index][self.step].copy_(param)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1


# Shared construction and message bookkeeping.


def with_message_noise(obs: TensorDict, noise_dim: int) -> TensorDict:
    """Return ``obs`` plus one standard-normal draw per world for the channel.

    A shallow copy, so the runner's observation object is left untouched.
    """
    if noise_dim == 0:
        return obs
    obs = obs.copy()
    obs[MESSAGE_NOISE_KEY] = torch.randn(
        obs.batch_size[0], noise_dim, device=obs.device
    )
    return obs


class _MessageChannelMixin:
    """Noise recording and message diagnostics shared by DialPPO and MAPPO."""

    def _init_message_stats(self) -> None:
        self._message_noise_dim = int(getattr(self.actor, "message_noise_dim", 0))
        self._message_unit = getattr(self.actor, "message_unit", None)
        self._message_stats: dict[str, torch.Tensor] = defaultdict(float)
        self._message_steps = 0

    def _with_message_noise(self, obs: TensorDict) -> TensorDict:
        # Only while training: an evaluation-mode actor ignores noise anyway.
        if not self.actor.training:
            return obs
        return with_message_noise(obs, self._message_noise_dim)

    def _track_messages(self) -> None:
        if self._message_unit not in _SATURATION_BOUNDS:
            return
        low, high = _SATURATION_BOUNDS[self._message_unit]
        for name, message in self.actor.last_messages.items():
            if message.numel() == 0:
                continue
            # Accumulated on device; converting here would sync every step.
            saturated = ((message < low) | (message > high)).float().mean()
            self._message_stats[f"Comm/message_saturation_{name}"] += saturated
            self._message_stats[f"Comm/message_abs_mean_{name}"] += message.abs().mean()
        # The sigmoid's gradient at the logit: how much learning signal a DRU
        # still passes. It decays to zero as the logits saturate.
        for name, logits in getattr(self.actor, "last_message_logits", {}).items():
            if logits.numel() == 0:
                continue
            sig = torch.sigmoid(logits)
            self._message_stats[f"Comm/logit_abs_mean_{name}"] += logits.abs().mean()
            self._message_stats[f"Comm/sigmoid_grad_mean_{name}"] += (sig * (1 - sig)).mean()
        self._message_steps += 1

    def _pop_message_stats(self) -> dict[str, float]:
        steps = max(self._message_steps, 1)
        stats = {
            key: float(value) / steps for key, value in self._message_stats.items()
        }
        self._message_stats = defaultdict(float)
        self._message_steps = 0
        return stats


def _resolve_models(
    obs: TensorDict, env: VecEnv, cfg: dict, device: str, critic_kwargs: dict
) -> tuple[type, nn.Module, nn.Module]:
    """The part of ``PPO.construct_algorithm`` that both algorithms share."""
    alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)
    actor = actor_class(
        obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
    ).to(device)
    print(f"Actor Model: {actor}")
    if cfg["algorithm"].pop("share_cnn_encoders", None):
        raise ValueError("sharing CNN encoders with the critic is not supported")
    critic = critic_class(
        obs, cfg["obs_groups"], "critic", 1, **cfg["critic"], **critic_kwargs
    ).to(device)
    print(f"Critic Model: {critic}")
    return alg_class, actor, critic


def _storage_obs(obs: TensorDict, actor: nn.Module) -> TensorDict:
    """Observation template whose keys the rollout storage will allocate."""
    noise_dim = int(getattr(actor, "message_noise_dim", 0))
    if noise_dim == 0:
        return obs
    obs = obs.copy()
    obs[MESSAGE_NOISE_KEY] = torch.zeros(
        obs.batch_size[0], noise_dim, device=obs.device
    )
    return obs


# Algorithms.


class DialPPO(_MessageChannelMixin, PPO):
    """rsl-rl PPO plus recorded channel noise; the single-ratio baseline.

    With no channel noise it adds no observation key and is numerically
    identical to ``PPO``.
    """

    def __init__(self, actor, critic, storage, **kwargs) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        self._init_message_stats()

    def act(self, obs: TensorDict) -> torch.Tensor:
        actions = super().act(self._with_message_noise(obs))
        self._track_messages()
        return actions

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        loss_dict.update(self._pop_message_stats())
        return loss_dict

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> DialPPO:
        alg_class, actor, critic = _resolve_models(obs, env, cfg, device, {})
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            _storage_obs(obs, actor),
            [env.num_actions],
            device,
        )
        alg = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg


class MAPPO(_MessageChannelMixin, PPO):
    """Per-agent clipped PPO with a centralized, agent-conditioned critic."""

    def __init__(
        self,
        actor,
        critic: MultiAgentCritic,
        storage: MultiAgentRolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.005,
        learning_rate: float = 3.0e-4,
        max_grad_norm: float = 10.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        num_agents: int = 2,
        critic_learning_rate: float | None = None,
        critic_max_grad_norm: float | None = None,
        use_value_norm: bool = True,
        value_loss: str = "huber",
        huber_delta: float = 10.0,
        max_kl: float | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise ValueError("MAPPO supports neither RND nor symmetry augmentation")
        if multi_gpu_cfg is not None:
            # The value normalizer and the second optimizer are not synchronized
            # across ranks.
            raise NotImplementedError("MAPPO does not support multi-GPU training")
        if value_loss not in VALUE_LOSSES:
            raise ValueError(f"unknown value_loss {value_loss!r}; use {VALUE_LOSSES}")
        if not isinstance(actor.distribution, GaussianDistribution):
            raise ValueError("MAPPO needs a factorized Gaussian action distribution")
        if getattr(critic, "num_agents", None) != num_agents:
            raise ValueError("the critic must produce one value per agent")
        if getattr(storage, "num_agents", None) != num_agents:
            raise ValueError("the storage must hold one value per agent")
        super().__init__(
            actor,
            critic,
            storage,
            num_learning_epochs=num_learning_epochs,
            num_mini_batches=num_mini_batches,
            clip_param=clip_param,
            gamma=gamma,
            lam=lam,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            optimizer=optimizer,
            use_clipped_value_loss=use_clipped_value_loss,
            schedule=schedule,
            desired_kl=desired_kl,
            normalize_advantage_per_mini_batch=normalize_advantage_per_mini_batch,
            device=device,
        )
        self.num_agents = num_agents
        # Separate optimizers, as in the MAPPO reference implementation; PPO's
        # single joint optimizer is replaced.
        optimizer_class = resolve_optimizer(optimizer)
        self.optimizer = optimizer_class(self.actor.parameters(), lr=learning_rate)
        self.critic_learning_rate = (
            learning_rate if critic_learning_rate is None else critic_learning_rate
        )
        self.critic_optimizer = optimizer_class(
            self.critic.parameters(), lr=self.critic_learning_rate
        )
        self.critic_max_grad_norm = (
            max_grad_norm if critic_max_grad_norm is None else critic_max_grad_norm
        )
        self.value_norm = ValueNorm(num_agents).to(device) if use_value_norm else None
        self.value_loss = value_loss
        self.huber_delta = huber_delta
        # Early stopping on policy drift (SB3's target_kl): the remaining
        # minibatches of an update are skipped once either agent's KL from the
        # rollout policy exceeds this. Unlike the adaptive schedule it also
        # works with a fixed learning rate, and it acts within the update that
        # is drifting rather than one minibatch later.
        if max_kl is not None and max_kl <= 0.0:
            raise ValueError("max_kl must be positive")
        self.max_kl = max_kl
        self._rollout_stats: dict[str, torch.Tensor] = {}
        self._init_message_stats()

    # Value scale.

    def _denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return values if self.value_norm is None else self.value_norm.denormalize(values)

    def _normalize(self, values: torch.Tensor) -> torch.Tensor:
        return values if self.value_norm is None else self.value_norm.normalize(values)

    def _value_error(self, error: torch.Tensor) -> torch.Tensor:
        if self.value_loss == "huber":
            return huber(error, self.huber_delta)
        return 0.5 * error.pow(2)

    # Rollout.

    def act(self, obs: TensorDict) -> torch.Tensor:
        obs = self._with_message_noise(obs)
        self.transition.hidden_states = (
            self.actor.get_hidden_state(),
            self.critic.get_hidden_state(),
        )
        actions = self.actor(obs, stochastic_output=True).detach()
        params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.actions = actions
        # Stored in real units, so GAE and timeout bootstrapping need no
        # knowledge of the normalizer.
        self.transition.values = self._denormalize(self.critic(obs)).detach()
        self.transition.actions_log_prob = agent_log_prob(
            params, actions, self.num_agents
        )
        self.transition.distribution_params = params
        self.transition.observations = obs
        self._track_messages()
        return actions

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        agent_rewards = extras.get("agent_rewards")
        if agent_rewards is None:
            agent_rewards = rewards.reshape(-1, 1).expand(-1, self.num_agents)
        self.transition.rewards = agent_rewards.to(self.device).clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += (
                self.gamma
                * self.transition.values
                * extras["time_outs"].reshape(-1, 1).to(self.device)
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        st = self.storage
        # A recurrent critic must not consume the next observation twice:
        # once here for bootstrapping and again at the next rollout's act().
        bootstrap = getattr(self.critic, "bootstrap_value", self.critic)
        last_values = self._denormalize(bootstrap(obs)).detach()
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            if step == st.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = st.values[step + 1]
            # [N, 1] against [N, num_agents]: one episode boundary for both.
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = (
                st.rewards[step]
                + next_is_not_terminal * self.gamma * next_values
                - st.values[step]
            )
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        flat = st.advantages.reshape(-1, self.num_agents)
        self._rollout_stats = {
            "return_mean": _per_agent_mean(st.returns),
            "advantage_std": flat.std(0),
        }
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - flat.mean(0)) / (flat.std(0) + 1.0e-8)

    # Optimization.

    def update(self) -> dict[str, float]:
        if self.value_norm is not None:
            self.value_norm.update(self.storage.returns)
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs
            )

        totals: dict[str, torch.Tensor] = defaultdict(float)
        num_updates = 0
        num_batches = self.num_learning_epochs * self.num_mini_batches
        stopped_early = False
        for batch in generator:
            advantages = batch.advantages
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    flat = advantages.reshape(-1, self.num_agents)
                    advantages = (advantages - flat.mean(0)) / (flat.std(0) + 1.0e-8)

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            params = self.actor.output_distribution_params
            log_prob = agent_log_prob(params, batch.actions, self.num_agents)
            entropy = _per_agent_mean(agent_entropy(params, self.num_agents))
            values = self.critic(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[1],
            )

            with torch.no_grad():
                kl = _per_agent_mean(
                    agent_kl(batch.old_distribution_params, params, self.num_agents)
                )
            if self.max_kl is not None and float(kl.max()) > self.max_kl:
                # This minibatch's forward already measures drift caused by the
                # previous steps; do not take another one.
                stopped_early = True
                break
            if self.desired_kl is not None and self.schedule == "adaptive":
                # The most-changed agent governs the step size for both.
                kl_max = float(kl.max())
                if kl_max > self.desired_kl * 2.0:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_max < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                for group in self.optimizer.param_groups:
                    group["lr"] = self.learning_rate

            # Per-agent clipped surrogate.
            ratio = torch.exp(log_prob - batch.old_actions_log_prob)
            surrogate_per_agent = clipped_surrogate(ratio, advantages, self.clip_param)
            surrogate_loss = surrogate_per_agent.mean()

            # Clipped value loss in normalized units. The old prediction is
            # re-expressed with the current statistics so that the clip range is
            # measured around the same real-valued estimate.
            targets = self._normalize(batch.returns)
            old_values = self._normalize(batch.values)
            value_error = self._value_error(values - targets)
            if self.use_clipped_value_loss:
                clipped = old_values + (values - old_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_error = torch.max(value_error, self._value_error(clipped - targets))
            value_per_agent = _per_agent_mean(value_error)
            value_loss = value_per_agent.mean()

            actor_loss = surrogate_loss - self.entropy_coef * entropy.mean()
            critic_loss = self.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # The actor and critic share no parameters, so one backward pass
            # yields exactly the two separate gradients.
            (actor_loss + critic_loss).backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_max_grad_norm)
            self.optimizer.step()
            self.critic_optimizer.step()

            with torch.no_grad():
                clip_fraction = _per_agent_mean(
                    ((ratio - 1.0).abs() > self.clip_param).float()
                )
            totals["surrogate"] += surrogate_loss.detach()
            totals["value"] += value_loss.detach()
            totals["entropy"] += entropy.mean().detach()
            for index, name in enumerate(AGENT_NAMES[: self.num_agents]):
                totals[f"MAPPO/surrogate_{name}"] += surrogate_per_agent[index].detach()
                totals[f"MAPPO/value_{name}"] += value_per_agent[index].detach()
                totals[f"MAPPO/entropy_{name}"] += entropy[index].detach()
                totals[f"MAPPO/kl_{name}"] += kl[index]
                totals[f"MAPPO/clip_fraction_{name}"] += clip_fraction[index]
            num_updates += 1

        loss_dict = {
            key: float(value) / max(num_updates, 1) for key, value in totals.items()
        }
        loss_dict["MAPPO/stopped_early"] = float(stopped_early)
        loss_dict["MAPPO/minibatch_updates_fraction"] = num_updates / num_batches
        for index, name in enumerate(AGENT_NAMES[: self.num_agents]):
            for stat, values in self._rollout_stats.items():
                loss_dict[f"MAPPO/{stat}_{name}"] = float(values[index])
        loss_dict["MAPPO/critic_learning_rate"] = self.critic_learning_rate
        loss_dict.update(self._pop_message_stats())
        self.storage.clear()
        return loss_dict

    # Checkpointing.

    def save(self) -> dict:
        saved = super().save()
        saved["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
        if self.value_norm is not None:
            saved["value_norm_state_dict"] = self.value_norm.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(
                loaded_dict["critic_state_dict"], strict=strict
            )
            # The normalizer defines what the critic's outputs mean.
            if self.value_norm is not None and "value_norm_state_dict" in loaded_dict:
                self.value_norm.load_state_dict(loaded_dict["value_norm_state_dict"])
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.learning_rate = self.optimizer.param_groups[0]["lr"]
            if "critic_optimizer_state_dict" in loaded_dict:
                self.critic_optimizer.load_state_dict(
                    loaded_dict["critic_optimizer_state_dict"]
                )
        return load_cfg.get("iteration", False)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> MAPPO:
        num_agents = cfg["algorithm"].get("num_agents", 2)
        if env.num_actions % num_agents:
            raise ValueError(
                f"{env.num_actions} actions do not split across {num_agents} agents"
            )
        alg_class, actor, critic = _resolve_models(
            obs, env, cfg, device, {"num_agents": num_agents}
        )
        storage = MultiAgentRolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            _storage_obs(obs, actor),
            [env.num_actions],
            num_agents,
            device,
        )
        alg = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg
