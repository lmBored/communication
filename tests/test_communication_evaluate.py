import json

import pytest
import torch

from escape_room.communication.evaluate import (
    apply_message_intervention,
    checkpoint_actor_architecture,
    default_probe_path,
    evaluate_probe,
    fit_message_probe,
    load_message_probe,
    probe_accuracy,
    save_message_probe,
)


def _separable_messages():
    left = torch.tensor(
        [[-1.0, -0.1], [-0.8, 0.1], [-1.1, 0.0], [-0.9, 0.05]]
    )
    right = torch.tensor(
        [[1.0, 0.1], [0.8, -0.1], [1.1, 0.0], [0.9, -0.05]]
    )
    return torch.cat((left, right)), torch.tensor([-1] * 4 + [1] * 4)


def test_nearest_centroid_probe_fits_and_scores_held_out_messages():
    messages, directions = _separable_messages()
    probe, held_out_accuracy = evaluate_probe(
        messages, directions, calibration_fraction=0.5, seed=7
    )

    assert probe["method"] == "nearest-centroid"
    assert probe["message_dim"] == 2
    assert held_out_accuracy == pytest.approx(1.0)
    assert probe_accuracy(messages, directions, probe) == pytest.approx(1.0)


def test_probe_round_trip_and_checkpoint_adjacent_default(tmp_path):
    messages, directions = _separable_messages()
    probe = fit_message_probe(messages, directions)
    checkpoint = tmp_path / "model_99.pt"
    path = default_probe_path(checkpoint)

    save_message_probe(probe, path)

    assert path == tmp_path / "model_99.message_probe.json"
    assert load_message_probe(path) == json.loads(path.read_text()) == probe


def test_zero_and_shuffled_message_interventions():
    messages = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    generator = torch.Generator().manual_seed(4)

    zero = apply_message_intervention(messages, "zero", generator)
    shuffled = apply_message_intervention(messages, "shuffled", generator)

    assert torch.count_nonzero(zero) == 0
    assert set(map(tuple, shuffled.tolist())) == set(map(tuple, messages.tolist()))
    assert not torch.any(torch.all(shuffled == messages, dim=-1))
    assert apply_message_intervention(messages, "normal", generator) is messages


def test_checkpoint_actor_architecture_is_inferred_for_strict_loading(tmp_path):
    checkpoint = tmp_path / "custom-width.pt"
    torch.save(
        {
            "actor_state_dict": {
                "sender_encoder.0.weight": torch.zeros(8, 2),
                "sender_encoder.2.weight": torch.zeros(8, 8),
                "sender_encoder.4.weight": torch.zeros(3, 8),
                "receiver_encoder.0.weight": torch.zeros(32, 12),
                "receiver_encoder.2.weight": torch.zeros(24, 32),
                "receiver_encoder.4.weight": torch.zeros(16, 24),
            }
        },
        checkpoint,
    )

    assert checkpoint_actor_architecture(checkpoint) == {
        "message_dim": 3,
        "sender_hidden_dims": (8, 8),
        "hidden_dims": (32, 24, 16),
    }
