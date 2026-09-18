import pytest
import torch

from escape_room.twowaycomm.action import LEFT, RIGHT
from escape_room.twowaycomm.evaluate import (
    ABLATION_ARMS,
    NUM_CONDITIONS,
    apply_message_intervention,
    balanced_conditions,
    checkpoint_actor_architecture,
    decode_messages,
    default_probe_path,
    evaluate_probe,
    fit_nearest_centroid,
    load_message_probe,
    probe_accuracy,
    save_message_probe,
)


def test_balanced_conditions_cover_every_cell_equally():
    episodes = NUM_CONDITIONS * 8
    conditions = balanced_conditions(episodes, "cpu", seed=3)

    counts = torch.bincount(conditions.condition_id, minlength=NUM_CONDITIONS)
    assert counts.tolist() == [8] * NUM_CONDITIONS
    assert torch.bincount(conditions.colors, minlength=3).tolist() == [
        episodes // 3
    ] * 3
    for arrow in range(3):
        assert int((conditions.directions[:, arrow] == LEFT).sum()) == episodes // 2

    # The marginal that actually matters: each (colour, answer) cell equally
    # often, so chance-level door choice is exactly one half.
    for colour in range(3):
        selected = conditions.target_direction[conditions.colors == colour]
        assert int((selected == LEFT).sum()) == selected.numel() // 2

    with pytest.raises(ValueError):
        balanced_conditions(100, "cpu", seed=0)


def test_target_direction_follows_the_matching_colour():
    conditions = balanced_conditions(NUM_CONDITIONS, "cpu", seed=1)
    expected = conditions.directions[
        torch.arange(NUM_CONDITIONS), conditions.colors
    ]
    assert torch.equal(conditions.target_direction, expected)


def test_three_class_nearest_centroid_probe():
    messages = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ]
    )
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    head = fit_nearest_centroid(messages, labels, {0: "RED", 1: "GREEN", 2: "BLUE"})

    assert head["message_dim"] == 2
    assert set(head["centroids"]) == {"0", "1", "2"}
    assert head["labels"]["2"] == "BLUE"
    assert probe_accuracy(messages, labels, head) == 1.0

    _, held_out = evaluate_probe(messages, labels, calibration_fraction=0.5, seed=0)
    assert held_out == 1.0


def test_two_class_decode_matches_the_one_way_label_order():
    # The one-way scenario's probe files use the stringified LEFT/RIGHT keys;
    # decoding must stay compatible with that layout.
    head = {
        "message_dim": 2,
        "labels": {"-1": "LEFT", "1": "RIGHT"},
        "centroids": {"-1": [-1.0, 0.0], "1": [1.0, 0.0]},
    }
    decoded = decode_messages(torch.tensor([[-0.8, 0.2], [0.7, -0.1]]), head)
    assert decoded.tolist() == [LEFT, RIGHT]
    assert int(decode_messages(torch.tensor([0.9, 0.0]), head)[0]) == RIGHT


def test_probe_round_trip_and_checkpoint_adjacent_default(tmp_path):
    probe = {
        "version": 2,
        "method": "nearest-centroid",
        "channel_mode": "same_step",
        "channels": {
            "receiver": {
                "message_dim": 2,
                "probe_step": 0,
                "targets": {
                    "door_color": {
                        "message_dim": 2,
                        "labels": {"0": "RED", "1": "GREEN", "2": "BLUE"},
                        "centroids": {
                            "0": [1.0, 0.0],
                            "1": [0.0, 1.0],
                            "2": [-1.0, 0.0],
                        },
                    }
                },
            }
        },
    }
    assert default_probe_path(tmp_path / "model_9.pt") == (
        tmp_path / "model_9.message_probe.json"
    )

    path = tmp_path / "model_9.message_probe.json"
    save_message_probe(probe, path)
    assert load_message_probe(path) == probe


def test_channels_are_intervened_independently():
    sender = torch.arange(12.0).reshape(6, 2)
    receiver = torch.arange(12.0).reshape(6, 2) + 100.0
    generator = torch.Generator().manual_seed(0)

    assert apply_message_intervention(sender, "normal", generator) is sender
    assert apply_message_intervention(sender, "zero", generator).count_nonzero() == 0

    shuffled_sender = apply_message_intervention(sender, "shuffled", generator)
    shuffled_receiver = apply_message_intervention(receiver, "shuffled", generator)
    # Every row moved, and the two channels were permuted separately, so the
    # sender/receiver pairing is genuinely broken.
    assert not torch.any((shuffled_sender == sender).all(dim=-1))
    assert not torch.any((shuffled_receiver == receiver).all(dim=-1))
    sender_order = [int(row[0] // 2) for row in shuffled_sender]
    receiver_order = [int((row[0] - 100.0) // 2) for row in shuffled_receiver]
    assert sorted(sender_order) == list(range(6))
    assert sorted(receiver_order) == list(range(6))

    with pytest.raises(ValueError):
        apply_message_intervention(sender, "nonsense", generator)


def test_ablation_arms_cover_both_channels():
    assert ABLATION_ARMS["zero_sender"] == ("zero", "normal")
    assert ABLATION_ARMS["zero_receiver"] == ("normal", "zero")
    assert ABLATION_ARMS["zero_both"] == ("zero", "zero")


@pytest.mark.parametrize("channel_mode, code", [("same_step", 0), ("delayed", 1)])
def test_checkpoint_architecture_is_inferred(tmp_path, channel_mode, code):
    feedback = channel_mode == "delayed"
    state = {
        "sender_message_encoder.0.weight": torch.zeros(8, 14 + (3 if feedback else 0)),
        "sender_message_encoder.2.weight": torch.zeros(1, 8),
        "receiver_message_encoder.0.weight": torch.zeros(6, 14 + (1 if feedback else 0)),
        "receiver_message_encoder.2.weight": torch.zeros(3, 6),
        "sender_latent_encoder.0.weight": torch.zeros(64, 14),
        "sender_latent_encoder.2.weight": torch.zeros(32, 64),
        "receiver_latent_encoder.0.weight": torch.zeros(48, 14),
        "receiver_latent_encoder.2.weight": torch.zeros(24, 48),
        "channel_mode_code": torch.tensor(code),
    }
    path = tmp_path / "model_final.pt"
    torch.save({"actor_state_dict": state}, path)

    architecture = checkpoint_actor_architecture(path)

    assert architecture["sender_message_dim"] == 1
    assert architecture["receiver_message_dim"] == 3
    assert architecture["sender_message_hidden_dims"] == (8,)
    assert architecture["receiver_message_hidden_dims"] == (6,)
    assert architecture["sender_hidden_dims"] == (64, 32)
    assert architecture["hidden_dims"] == (48, 24)
    assert architecture["channel_mode"] == channel_mode
    assert architecture["delayed_message_feedback"] is feedback


def test_one_way_inference_rejects_a_twowaycomm_checkpoint(tmp_path):
    """The rename is a safety property: the old reader must not half-succeed."""
    from escape_room.communication.evaluate import (
        checkpoint_actor_architecture as one_way_architecture,
    )

    path = tmp_path / "model_final.pt"
    torch.save(
        {
            "actor_state_dict": {
                "sender_message_encoder.0.weight": torch.zeros(8, 14),
                "receiver_latent_encoder.0.weight": torch.zeros(32, 14),
            }
        },
        path,
    )
    with pytest.raises(ValueError):
        one_way_architecture(path)


def test_round_trip_from_a_real_actor_state_dict(tmp_path):
    from tensordict import TensorDict

    from escape_room.twowaycomm.model import TwoWayDialActor

    obs = TensorDict(
        {"sender": torch.zeros(2, 14), "receiver": torch.zeros(2, 14)},
        batch_size=[2],
    )
    actor = TwoWayDialActor(
        obs=obs,
        obs_groups={"actor": ["sender", "receiver"]},
        obs_set="actor",
        output_dim=6,
        hidden_dims=(48, 24),
        sender_hidden_dims=(64, 32),
        sender_message_hidden_dims=(8,),
        receiver_message_hidden_dims=(6,),
        sender_message_dim=1,
        receiver_message_dim=3,
        channel_mode="delayed",
    )
    path = tmp_path / "model_final.pt"
    torch.save({"actor_state_dict": actor.state_dict()}, path)

    architecture = checkpoint_actor_architecture(path)

    assert architecture["sender_message_dim"] == 1
    assert architecture["receiver_message_dim"] == 3
    assert architecture["hidden_dims"] == (48, 24)
    assert architecture["sender_hidden_dims"] == (64, 32)
    assert architecture["channel_mode"] == "delayed"
    assert architecture["delayed_message_feedback"] is True
