"""Centralized recurrent value model using both agents' actual observations."""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import CNN, MLP, EmpiricalNormalization
from rsl_rl.utils import unpad_trajectories
from escape_room.twowaycomm.model import DEFAULT_CNN_CFG, _encode_image


class PixelGruCritic(nn.Module):
    """One central GRU memory per world; agent-conditioned value outputs.

    Only RGB images, local proprioception and time are read. Neither the
    privileged ``critic`` group nor the private-label slots can reach values.
    Training replay is functional and never overwrites the live rollout memory.
    """
    is_recurrent = True

    def __init__(self, obs: TensorDict, obs_groups: dict, obs_set: str,
                 output_dim: int = 1, hidden_dims=(256, 256, 256),
                 activation="elu", obs_normalization=False, num_agents=2,
                 rnn_hidden_dim=256, rnn_num_layers=1, cnn_cfg=None, rnn_type="gru"):
        super().__init__()
        if output_dim != 1 or num_agents != 2:
            raise ValueError("PixelGruCritic requires two agents and scalar per-agent values")
        if rnn_type != "gru":
            raise ValueError("PixelGruCritic requires rnn_type='gru'")
        expected = {"sender", "receiver", "sender_image", "receiver_image"}
        if set(obs_groups[obs_set]) != expected:
            raise ValueError("pixel critic requires both local vectors and both images")
        self.num_agents = num_agents
        self.obs_normalization = obs_normalization
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        cfg = dict(DEFAULT_CNN_CFG if cnn_cfg is None else cnn_cfg)
        def build(image):
            channels, height, width = image.shape[-3:]
            return CNN(input_dim=(height, width), input_channels=channels, flatten=True, **cfg)
        self.sender_cnn = build(obs["sender_image"])
        self.receiver_cnn = build(obs["receiver_image"])
        # Seven pose/velocity fields per agent and one common remaining time.
        self.aux_normalizer = EmpiricalNormalization(15) if obs_normalization else nn.Identity()
        feature_dim = self.sender_cnn.output_dim + self.receiver_cnn.output_dim + 15
        self.projection = nn.Linear(feature_dim, rnn_hidden_dim)
        self.projection_activation = nn.ELU()
        self.gru = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers)
        self.value_head = MLP(rnn_hidden_dim + num_agents, 1, hidden_dims, activation)
        self.register_buffer("agent_ids", torch.eye(num_agents), persistent=False)
        self.register_buffer("_hidden", torch.zeros(rnn_num_layers, obs.batch_size[0], rnn_hidden_dim), persistent=False)

    @staticmethod
    def auxiliary(obs):
        return torch.cat((obs["sender"][..., :7], obs["receiver"][..., :7],
                          obs["sender"][..., 13:14]), dim=-1)

    def _encode_features(self, obs):
        return torch.cat((_encode_image(self.sender_cnn, obs["sender_image"]),
                          _encode_image(self.receiver_cnn, obs["receiver_image"]),
                          self.aux_normalizer(self.auxiliary(obs))), dim=-1)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output=False):
        del stochastic_output
        features = self._encode_features(obs)
        inputs = self.projection_activation(self.projection(features))
        replay = masks is not None
        if not replay:
            inputs = inputs.unsqueeze(0)
        initial = self._hidden if hidden_state is None else hidden_state
        # Rollout state can be an inference tensor; clone outside inference mode
        # before GRU autograd saves it during sequence replay.
        initial = initial.detach().clone()
        output, final = self.gru(inputs, initial)
        if replay:
            output = unpad_trajectories(output, masks)
        else:
            output = output.squeeze(0)
            if hidden_state is None:
                self._hidden = final.detach()
        leading = output.shape[:-1]
        tiled = output.unsqueeze(-2).expand(*leading, self.num_agents, output.shape[-1])
        ids = self.agent_ids.expand(*leading, self.num_agents, self.num_agents)
        return self.value_head(torch.cat((tiled, ids), dim=-1)).squeeze(-1)

    def bootstrap_value(self, obs):
        """Peek at the next value without consuming the next rollout frame."""
        return self.forward(obs, hidden_state=self.get_hidden_state())

    def get_hidden_state(self):
        return self._hidden

    def reset(self, dones=None, hidden_state=None):
        if hidden_state is not None:
            self._hidden = hidden_state.detach().clone()
        elif dones is None:
            self._hidden.zero_()
        else:
            self._hidden[:, dones.reshape(-1).bool(), :] = 0

    def detach_hidden_state(self, dones=None):
        del dones
        self._hidden = self._hidden.detach()

    def update_normalization(self, obs):
        if self.obs_normalization:
            self.aux_normalizer.update(self.auxiliary(obs))

    def as_jit(self):
        raise NotImplementedError("The training-only pixel critic is not exported")

    def as_onnx(self, verbose=False):
        raise NotImplementedError("The training-only pixel critic is not exported")


class VectorGruCritic(PixelGruCritic):
    """Privileged vector-state control with the identical GRU/value head.

    Actors still read only their local pixel/proprioceptive observations.
    Keeping recurrence, width and per-agent values isolates critic perception.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim=1,
                 hidden_dims=(256, 256, 256), activation="elu",
                 obs_normalization=False, num_agents=2, rnn_hidden_dim=256,
                 rnn_num_layers=1, cnn_cfg=None, rnn_type="gru"):
        nn.Module.__init__(self)
        if output_dim != 1 or num_agents != 2 or rnn_type != "gru":
            raise ValueError("VectorGruCritic requires two scalar values and GRU recurrence")
        if tuple(obs_groups[obs_set]) != ("critic",):
            raise ValueError("VectorGruCritic reads only the privileged critic state")
        self.num_agents = num_agents
        self.obs_normalization = obs_normalization
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_num_layers = rnn_num_layers
        width = obs["critic"].shape[-1]
        self.aux_normalizer = EmpiricalNormalization(width) if obs_normalization else nn.Identity()
        self.projection = nn.Linear(width, rnn_hidden_dim)
        self.projection_activation = nn.ELU()
        self.gru = nn.GRU(rnn_hidden_dim, rnn_hidden_dim, rnn_num_layers)
        self.value_head = MLP(rnn_hidden_dim + num_agents, 1, hidden_dims, activation)
        self.register_buffer("agent_ids", torch.eye(num_agents), persistent=False)
        self.register_buffer("_hidden", torch.zeros(rnn_num_layers, obs.batch_size[0], rnn_hidden_dim), persistent=False)

    def _encode_features(self, obs):
        return self.aux_normalizer(obs["critic"])

    def update_normalization(self, obs):
        if self.obs_normalization:
            self.aux_normalizer.update(obs["critic"])
