import copy

import pytest
import torch
from tensordict import TensorDict

from escape_room.twowaycomm.env_cfg import (
    ACTION_DIM,
    RECEIVER_OBS_DIM,
    SENDER_OBS_DIM,
    twowaycomm_ppo_runner_cfg,
)
from escape_room.twowaycomm.model import TwoWayDialActor

MESSAGE_DIM = 4
OBS_GROUPS = {"actor": ["sender", "receiver"], "critic": ["critic"]}


def _observations(batch_size: int = 8) -> TensorDict:
    """Observations with the always-zero private slots actually zeroed."""
    sender = torch.randn(batch_size, SENDER_OBS_DIM)
    sender[:, 7:10] = torch.where(
        torch.rand(batch_size, 3) < 0.5, -torch.ones(1), torch.ones(1)
    )
    sender[:, 10:13] = 0.0
    receiver = torch.randn(batch_size, RECEIVER_OBS_DIM)
    receiver[:, 7:10] = 0.0
    receiver[:, 10:13] = torch.eye(3)[torch.randint(0, 3, (batch_size,))]
    return TensorDict(
        {
            "sender": sender,
            "receiver": receiver,
            "critic": torch.cat((sender, receiver), dim=-1),
        },
        batch_size=[batch_size],
    )


def _trajectory(steps: int = 3, worlds: int = 2) -> TensorDict:
    sender = torch.randn(steps, worlds, SENDER_OBS_DIM)
    sender[..., 10:13] = 0.0
    receiver = torch.randn(steps, worlds, RECEIVER_OBS_DIM)
    receiver[..., 7:10] = 0.0
    return TensorDict(
        {"sender": sender, "receiver": receiver}, batch_size=[steps, worlds]
    )


def _actor(
    channel_mode: str = "same_step",
    delayed_message_feedback: bool = True,
    sender_message_dim: int | None = None,
    receiver_message_dim: int | None = None,
    batch_size: int = 8,
) -> TwoWayDialActor:
    return TwoWayDialActor(
        obs=_observations(batch_size),
        obs_groups=OBS_GROUPS,
        obs_set="actor",
        output_dim=ACTION_DIM,
        hidden_dims=(32, 32, 32),
        sender_hidden_dims=(32, 32, 32),
        sender_message_hidden_dims=(16, 16),
        receiver_message_hidden_dims=(16, 16),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "log",
        },
        message_dim=MESSAGE_DIM,
        sender_message_dim=sender_message_dim,
        receiver_message_dim=receiver_message_dim,
        channel_mode=channel_mode,
        delayed_message_feedback=delayed_message_feedback,
    )


@pytest.mark.parametrize("channel_mode", ["same_step", "delayed"])
def test_output_and_message_contracts(channel_mode):
    actor = _actor(channel_mode)
    obs = _observations()

    deterministic = actor(obs)
    stochastic = actor(obs, stochastic_output=True)

    assert deterministic.shape == (8, ACTION_DIM)
    assert stochastic.shape == (8, ACTION_DIM)
    assert actor.last_sender_message.shape == (8, MESSAGE_DIM)
    assert actor.last_receiver_message.shape == (8, MESSAGE_DIM)
    assert actor.last_sender_message.abs().max() <= 1.0
    assert actor.last_receiver_message.abs().max() <= 1.0
    assert set(actor.last_messages) == {"sender", "receiver"}
    assert actor.output_mean.shape == (8, ACTION_DIM)
    assert actor.output_std.shape == (8, ACTION_DIM)
    assert actor.output_entropy.shape == (8,)
    assert actor.get_output_log_prob(deterministic).shape == (8,)
    assert len(actor.output_distribution_params) == 2


def test_is_recurrent_matches_channel_mode():
    # This single attribute is what makes PPO pick the recurrent storage.
    assert _actor("same_step").is_recurrent is False
    assert _actor("delayed").is_recurrent is True


def test_task_loss_gradient_reaches_both_message_encoders_same_step():
    actor = _actor("same_step")

    actor(_observations()).square().mean().backward()

    sender_grad = sum(
        p.grad.abs().sum() for p in actor.sender_message_encoder.parameters()
    )
    receiver_grad = sum(
        p.grad.abs().sum() for p in actor.receiver_message_encoder.parameters()
    )
    assert sender_grad > 0.0
    assert receiver_grad > 0.0


@pytest.mark.parametrize("delayed_message_feedback", [True, False])
def test_task_loss_gradient_reaches_both_message_encoders_delayed(
    delayed_message_feedback,
):
    # The single-step delayed forward consumes only the previous (zero) pair, so
    # the gradient invariant only has meaning on the batched update path.
    actor = _actor("delayed", delayed_message_feedback, batch_size=2)
    obs = _trajectory(steps=3, worlds=2)
    masks = torch.ones(3, 2, dtype=torch.bool)
    hidden = (torch.zeros(1, 2, MESSAGE_DIM), torch.zeros(1, 2, MESSAGE_DIM))

    actor(obs, masks=masks, hidden_state=hidden).square().mean().backward()

    sender_grad = sum(
        p.grad.abs().sum() for p in actor.sender_message_encoder.parameters()
    )
    receiver_grad = sum(
        p.grad.abs().sum() for p in actor.receiver_message_encoder.parameters()
    )
    assert sender_grad > 0.0
    assert receiver_grad > 0.0


@pytest.mark.parametrize("delayed_message_feedback", [True, False])
def test_delay_is_exactly_one_step(delayed_message_feedback):
    actor = _actor("delayed", delayed_message_feedback, batch_size=2)
    obs = _trajectory(steps=2, worlds=2)
    masks = torch.ones(2, 2, dtype=torch.bool)
    hidden = (torch.zeros(1, 2, MESSAGE_DIM), torch.zeros(1, 2, MESSAGE_DIM))

    # Step 0 consumes only the zero hidden state, so no message encoder is on
    # its path; step 1 consumes the message emitted at step 0.
    actor.zero_grad()
    actor(obs, masks=masks, hidden_state=hidden)[0].square().mean().backward()
    first = sum(
        p.grad.abs().sum()
        for p in actor.sender_message_encoder.parameters()
        if p.grad is not None
    )
    actor.zero_grad()
    actor(obs, masks=masks, hidden_state=hidden)[1].square().mean().backward()
    second = sum(
        p.grad.abs().sum()
        for p in actor.sender_message_encoder.parameters()
        if p.grad is not None
    )

    assert float(first) == 0.0
    assert float(second) > 0.0


def test_batch_mode_leaves_the_rollout_hidden_state_alone():
    # n_traj differs from num_envs during the update, so writing the persistent
    # buffers there would corrupt the next rollout.
    actor = _actor("delayed", batch_size=5)
    actor(_observations(5))
    before = tuple(h.clone() for h in actor.get_hidden_state())

    actor(
        _trajectory(steps=2, worlds=3),
        masks=torch.ones(2, 3, dtype=torch.bool),
        hidden_state=(torch.zeros(1, 3, MESSAGE_DIM), torch.zeros(1, 3, MESSAGE_DIM)),
    )

    after = actor.get_hidden_state()
    assert all(torch.equal(a, b) for a, b in zip(before, after))


def test_delayed_batch_mode_requires_a_hidden_state():
    actor = _actor("delayed", batch_size=2)
    with pytest.raises(ValueError):
        actor(_trajectory(steps=2, worlds=2), masks=torch.ones(2, 2, dtype=torch.bool))


def test_reset_semantics():
    actor = _actor("delayed", batch_size=4)
    actor(_observations(4))
    assert actor.get_hidden_state()[0].abs().sum() > 0.0

    actor.reset(torch.tensor([1, 0, 1, 0]))
    zeroed = [
        float(actor.get_hidden_state()[0][0, i].abs().sum()) == 0.0 for i in range(4)
    ]
    assert zeroed == [True, False, True, False]

    actor.reset()
    assert actor.get_hidden_state()[0].abs().sum() == 0.0

    installed = (
        torch.full((1, 4, MESSAGE_DIM), 0.5),
        torch.full((1, 4, MESSAGE_DIM), -0.5),
    )
    actor.reset(hidden_state=installed)
    assert torch.equal(actor.get_hidden_state()[0], installed[0])

    with pytest.raises(NotImplementedError):
        actor.reset(torch.zeros(4), hidden_state=installed)


def test_hidden_state_round_trips_through_rollout_storage():
    """The contract that keeps delayed-mode PPO updates aligned."""
    from rsl_rl.storage.rollout_storage import RolloutStorage

    worlds, steps, minibatches = 6, 5, 2
    actor = _actor("delayed", batch_size=worlds)
    obs = _observations(worlds)
    storage = RolloutStorage(
        training_type="rl",
        num_envs=worlds,
        num_transitions_per_env=steps,
        obs=obs,
        actions_shape=[ACTION_DIM],
        device="cpu",
    )
    transition = RolloutStorage.Transition()
    for _ in range(steps):
        transition.hidden_states = (actor.get_hidden_state(), None)
        transition.observations = obs
        transition.actions = actor(obs, stochastic_output=True).detach()
        transition.actions_log_prob = actor.get_output_log_prob(
            transition.actions
        ).detach()
        transition.distribution_params = tuple(
            p.detach() for p in actor.output_distribution_params
        )
        transition.values = torch.zeros(worlds, 1)
        transition.rewards = torch.zeros(worlds, 1)
        transition.dones = torch.zeros(worlds, 1, dtype=torch.long)
        storage.add_transition(transition)
        transition.clear()

    batches = 0
    for batch in storage.recurrent_mini_batch_generator(minibatches, 1):
        output = actor(
            batch.observations,
            masks=batch.masks,
            hidden_state=batch.hidden_states[0],
            stochastic_output=True,
        )
        assert output.shape == (steps, worlds // minibatches, ACTION_DIM)
        assert actor.get_output_log_prob(batch.actions).shape == (
            steps,
            worlds // minibatches,
        )
        batches += 1
    assert batches == minibatches


def test_clamped_messages_block_private_information_from_crossing():
    actor = _actor("same_step")
    obs = _observations()
    sender_message = torch.full((8, MESSAGE_DIM), 0.25)
    receiver_message = torch.full((8, MESSAGE_DIM), -0.25)

    baseline = actor.action_from_messages(obs, sender_message, receiver_message)

    flipped = obs.clone()
    flipped["sender"][:, 7:10] = -flipped["sender"][:, 7:10]
    receiver_only = actor.action_from_messages(
        flipped, sender_message, receiver_message
    )
    # The receiver reads the arrows only through the message.
    assert torch.allclose(baseline[:, 3:], receiver_only[:, 3:])

    recoloured = obs.clone()
    recoloured["receiver"][:, 10:13] = recoloured["receiver"][:, 10:13].roll(1, dims=-1)
    sender_only = actor.action_from_messages(
        recoloured, sender_message, receiver_message
    )
    # The sender reads the door colour only through the message.
    assert torch.allclose(baseline[:, :3], sender_only[:, :3])


@pytest.mark.parametrize("channel_mode", ["same_step", "delayed"])
def test_state_dict_round_trip_and_deterministic_inference(channel_mode):
    actor = _actor(channel_mode)
    restored = _actor(channel_mode)
    restored.load_state_dict(copy.deepcopy(actor.state_dict()))
    actor.reset()
    restored.reset()
    obs = _observations()

    assert torch.allclose(actor(obs), restored(obs))


def test_channel_mode_mismatch_is_rejected_on_load():
    # Without the persistent code a delayed checkpoint would load cleanly as
    # same-step whenever the parameter shapes happen to agree.
    delayed = _actor("delayed", delayed_message_feedback=False)
    same_step = _actor("same_step")

    with pytest.raises(RuntimeError, match="channel_mode mismatch"):
        same_step.load_state_dict(delayed.state_dict(), strict=True)


@pytest.mark.parametrize("channel_mode", ["same_step", "delayed"])
def test_torchscript_export(channel_mode):
    actor = _actor(channel_mode)
    actor.eval()
    exported = torch.jit.script(actor.as_jit())

    if channel_mode == "same_step":
        assert exported(torch.zeros(2, SENDER_OBS_DIM + RECEIVER_OBS_DIM)).shape == (
            2,
            ACTION_DIM,
        )
    else:
        observation = torch.randn(1, SENDER_OBS_DIM + RECEIVER_OBS_DIM)
        first = exported(observation).clone()
        second = exported(observation).clone()
        assert not torch.allclose(first, second)  # stateful
        exported.reset()
        assert torch.allclose(exported(observation), first)


def test_asymmetric_message_widths():
    actor = _actor(sender_message_dim=1, receiver_message_dim=3)
    obs = _observations()

    actor(obs).square().mean().backward()

    assert actor.last_sender_message.shape == (8, 1)
    assert actor.last_receiver_message.shape == (8, 3)
    assert actor.receiver_head.in_features == 32 + 1
    assert actor.sender_head.in_features == 32 + 3
    assert sum(p.grad.abs().sum() for p in actor.sender_message_encoder.parameters()) > 0


@pytest.mark.parametrize("channel_mode", ["same_step", "delayed"])
def test_runner_cfg_contract(channel_mode):
    cfg = twowaycomm_ppo_runner_cfg(message_dim=5, channel_mode=channel_mode)

    assert cfg.actor.class_name == "escape_room.twowaycomm.model:TwoWayDialActor"
    assert cfg.actor.message_dim == 5
    assert cfg.actor.channel_mode == channel_mode
    # mjlab only strips the rnn_* kwargs when rnn_type is None; otherwise the
    # actor constructor is handed three arguments it does not accept.
    assert cfg.actor.rnn_type is None
    assert cfg.obs_groups == {"actor": ("sender", "receiver"), "critic": ("critic",)}
    assert cfg.critic.class_name == "MLPModel"
    assert cfg.experiment_name == "twowaycomm"


def test_pixel_runner_cfg_adds_image_groups_and_a_cnn():
    cfg = twowaycomm_ppo_runner_cfg(obs_mode="pixel")

    assert cfg.obs_groups["actor"] == (
        "sender",
        "receiver",
        "sender_image",
        "receiver_image",
    )
    assert cfg.actor.cnn_cfg is not None
    # Vector mode must register no CNN at all, so mjlab strips the kwarg.
    assert twowaycomm_ppo_runner_cfg().actor.cnn_cfg is None


def _pixel_observations(batch_size: int = 4, resolution: int = 16) -> TensorDict:
    obs = _observations(batch_size)
    # In pixel mode both private blocks are blank; the pixels carry them.
    obs["sender"][:, 7:10] = 0.0
    obs["receiver"][:, 10:13] = 0.0
    obs["sender_image"] = torch.rand(batch_size, 3, resolution, resolution)
    obs["receiver_image"] = torch.rand(batch_size, 3, resolution, resolution)
    return obs


def _pixel_actor(channel_mode: str = "same_step") -> TwoWayDialActor:
    obs = _pixel_observations()
    return TwoWayDialActor(
        obs=obs,
        obs_groups={
            "actor": ["sender", "receiver", "sender_image", "receiver_image"],
            "critic": ["critic"],
        },
        obs_set="actor",
        output_dim=ACTION_DIM,
        hidden_dims=(32, 32),
        sender_hidden_dims=(32, 32),
        sender_message_hidden_dims=(16,),
        receiver_message_hidden_dims=(16,),
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 0.5,
            "std_type": "log",
        },
        message_dim=MESSAGE_DIM,
        channel_mode=channel_mode,
    )


def test_pixel_actor_routes_images_into_both_message_encoders():
    actor = _pixel_actor()
    assert actor.obs_mode == "pixel"
    # The arrows are only visible in pixels, so the CNN must feed the message
    # encoder, not just the control path.
    assert actor.sender_feature_dim > actor.sender_dim

    actor(_pixel_observations()).square().mean().backward()

    assert sum(p.grad.abs().sum() for p in actor.sender_cnn.parameters()) > 0
    assert sum(p.grad.abs().sum() for p in actor.receiver_cnn.parameters()) > 0
    assert (
        sum(p.grad.abs().sum() for p in actor.sender_message_encoder.parameters()) > 0
    )
    assert (
        sum(p.grad.abs().sum() for p in actor.receiver_message_encoder.parameters()) > 0
    )


def test_pixel_actor_handles_padded_trajectory_batches():
    actor = _pixel_actor("delayed")
    steps, worlds, resolution = 3, 2, 16
    obs = _trajectory(steps, worlds)
    obs["sender_image"] = torch.rand(steps, worlds, 3, resolution, resolution)
    obs["receiver_image"] = torch.rand(steps, worlds, 3, resolution, resolution)

    output = actor(
        obs,
        masks=torch.ones(steps, worlds, dtype=torch.bool),
        hidden_state=(
            torch.zeros(1, worlds, MESSAGE_DIM),
            torch.zeros(1, worlds, MESSAGE_DIM),
        ),
    )

    assert output.shape == (steps, worlds, ACTION_DIM)


def test_pixel_export_is_refused_rather_than_silently_wrong():
    actor = _pixel_actor()
    with pytest.raises(NotImplementedError):
        actor.as_jit()
    with pytest.raises(NotImplementedError):
        actor.as_onnx(False)
