import copy

import torch
from tensordict import TensorDict

from escape_room.communication.env_cfg import (
    ACTION_DIM,
    RECEIVER_OBS_DIM,
    SENDER_OBS_DIM,
    communication_ppo_runner_cfg,
)
from escape_room.communication.model import DirectDialActor


def _observations(batch_size: int = 8) -> TensorDict:
    sender = torch.randn(batch_size, SENDER_OBS_DIM)
    sender[:, 0] = torch.where(sender[:, 0] < 0, -1.0, 1.0)
    receiver = torch.randn(batch_size, RECEIVER_OBS_DIM)
    return TensorDict(
        {
            "sender": sender,
            "receiver": receiver,
            "critic": torch.cat((sender, receiver), dim=-1),
        },
        batch_size=[batch_size],
    )


def _actor(message_dim: int = 2) -> DirectDialActor:
    obs = _observations()
    return DirectDialActor(
        obs=obs,
        obs_groups={"actor": ["sender", "receiver"], "critic": ["critic"]},
        obs_set="actor",
        output_dim=ACTION_DIM,
        hidden_dims=(32, 32, 32),
        sender_hidden_dims=(16, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "log",
        },
        message_dim=message_dim,
    )


def test_direct_dial_actor_output_distribution_and_message_contracts():
    obs = _observations()
    actor = _actor(message_dim=3)

    deterministic = actor(obs)
    stochastic = actor(obs, stochastic_output=True)

    assert deterministic.shape == (8, ACTION_DIM)
    assert stochastic.shape == (8, ACTION_DIM)
    assert actor.last_message.shape == (8, 3)
    assert torch.all(actor.last_message <= 1.0)
    assert torch.all(actor.last_message >= -1.0)
    assert actor.output_mean.shape == (8, ACTION_DIM)
    assert actor.output_std.shape == (8, ACTION_DIM)
    assert actor.output_entropy.shape == (8,)
    assert actor.get_output_log_prob(stochastic).shape == (8,)
    assert len(actor.output_distribution_params) == 2


def test_task_reward_gradient_reaches_sender_encoder():
    obs = _observations()
    actor = _actor()

    actor(obs).square().mean().backward()

    sender_gradient = sum(
        parameter.grad.abs().sum().item()
        for parameter in actor.sender_encoder.parameters()
        if parameter.grad is not None
    )
    assert sender_gradient > 0.0


def test_receiver_action_is_clue_invariant_when_message_is_clamped():
    obs = _observations()
    actor = _actor()
    fixed_message = torch.full((8, 2), 0.25)
    changed_clue = obs.clone()
    changed_clue["sender"][:, 0] *= -1

    first = actor.action_from_message(obs["receiver"], fixed_message)
    second = actor.action_from_message(changed_clue["receiver"], fixed_message)

    assert torch.allclose(first, second)


def test_direct_dial_state_dict_round_trip_and_deterministic_inference():
    obs = _observations()
    actor = _actor()
    expected = actor(obs)
    restored = _actor()
    restored.load_state_dict(copy.deepcopy(actor.state_dict()))

    assert torch.allclose(restored(obs), expected)
    assert torch.allclose(restored(obs), restored(obs))


def test_direct_dial_export_accepts_concatenated_private_observations():
    obs = _observations(batch_size=2)
    actor = _actor()
    actor.eval()
    exported = torch.jit.script(actor.as_jit())
    concatenated = torch.cat((obs["sender"], obs["receiver"]), dim=-1)

    assert exported(concatenated).shape == (2, ACTION_DIM)


def test_communication_runner_uses_direct_dial_and_centralized_critic():
    cfg = communication_ppo_runner_cfg(message_dim=5)

    assert cfg.actor.class_name == "escape_room.communication.model:DirectDialActor"
    assert cfg.actor.message_dim == 5
    assert cfg.obs_groups == {
        "actor": ("sender", "receiver"),
        "critic": ("critic",),
    }
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.experiment_name == "communication"