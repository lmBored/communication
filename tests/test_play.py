import os
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

from escape_room.play import (
    COMMUNICATION_PANEL_HEIGHT,
    _checkpoint_uses_scalar_std,
    _communication_panel_lines,
    _compose_communication_frame,
    _needs_mjpython,
    _relaunch_with_mjpython,
    _render_communication_views,
    _viewer_kind,
    parse_args,
)


def test_interactive_macos_playback_relaunches_under_mjpython():
    args = Namespace(headless=False)

    assert _needs_mjpython(args, platform_name="Darwin", environ={})
    assert not _needs_mjpython(
        args,
        platform_name="Darwin",
        environ={"MJPYTHON_BIN": "/path/to/mjpython"},
    )


def test_headless_and_non_macos_playback_do_not_need_mjpython():
    assert not _needs_mjpython(
        Namespace(headless=True), platform_name="Darwin", environ={}
    )
    assert not _needs_mjpython(
        Namespace(headless=False), platform_name="Linux", environ={}
    )


def test_macos_uses_native_viewer_without_display_environment():
    assert _viewer_kind(platform_name="Darwin", environ={}) == "native"
    assert _viewer_kind(platform_name="Linux", environ={}) == "viser"
    assert _viewer_kind(
        platform_name="Linux", environ={"DISPLAY": ":0"}
    ) == "native"


def test_mjpython_relaunch_adds_uv_base_python_library_path(
    monkeypatch, tmp_path
):
    base_python = tmp_path / "uv" / "bin" / "python3"
    base_python.parent.mkdir(parents=True)
    base_python.touch()
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    python = venv_bin / "python"
    python.symlink_to(base_python)
    mjpython = venv_bin / "mjpython"
    mjpython.touch()
    launched = {}

    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.delenv("MJPYTHON_BIN", raising=False)
    monkeypatch.delenv("DYLD_FALLBACK_LIBRARY_PATH", raising=False)
    monkeypatch.setattr(
        os,
        "execv",
        lambda executable, argv: launched.update(executable=executable, argv=argv),
    )

    _relaunch_with_mjpython(Namespace(headless=False), ["--ckpt", "model_29.pt"])

    assert os.environ["DYLD_FALLBACK_LIBRARY_PATH"] == str(
        base_python.parent.parent / "lib"
    )
    assert launched == {
        "executable": str(mjpython),
        "argv": [
            str(mjpython),
            "-m",
            "escape_room.play",
            "--ckpt",
            "model_29.pt",
        ],
    }


def test_legacy_scalar_std_checkpoint_is_detected(tmp_path):
    scalar = tmp_path / "scalar.pt"
    log = tmp_path / "log.pt"
    torch.save(
        {"actor_state_dict": {"distribution.std_param": torch.ones(8)}},
        scalar,
    )
    torch.save(
        {"actor_state_dict": {"distribution.log_std_param": torch.zeros(8)}},
        log,
    )

    assert _checkpoint_uses_scalar_std(scalar)
    assert not _checkpoint_uses_scalar_std(log)


def test_communication_playback_parser_preserves_legacy_defaults():
    defaults = parse_args([])
    communication = parse_args(
        [
            "--task",
            "communication",
            "--policy",
            "checkpoint",
            "--ckpt",
            "model.pt",
            "--message-probe",
            "probe.json",
        ]
    )

    assert defaults.task == "escape-room"
    assert defaults.width == 480
    assert defaults.height == 960
    assert communication.task == "communication"
    assert communication.message_probe == "probe.json"


def test_communication_compositor_dimensions_and_labels():
    sender = np.full((24, 32, 3), (180, 40, 20), dtype=np.uint8)
    receiver = np.full((24, 32, 3), (20, 40, 180), dtype=np.uint8)
    probe = {
        "method": "nearest-centroid",
        "message_dim": 2,
        "centroids": {"-1": [-1.0, 0.0], "1": [1.0, 0.0]},
    }

    lines = _communication_panel_lines(np.array([0.8, 0.1]), probe)
    frame = _compose_communication_frame(sender, receiver, np.array([0.8, 0.1]), probe)

    assert lines[0] == "SENDER -> RECEIVER"
    assert "[0.800, 0.100]" in lines[1]
    assert lines[2] == "PROBE (diagnostic only): RIGHT"
    assert frame.shape == (24 + COMMUNICATION_PANEL_HEIGHT, 64, 3)
    assert np.any(frame[24:] != 0)


def test_communication_named_cameras_render_headlessly():
    if (
        sys.platform.startswith("linux")
        and not os.environ.get("DISPLAY")
        and os.environ.get("MUJOCO_GL") not in {"egl", "osmesa"}
    ):
        pytest.skip("headless Linux rendering requires MUJOCO_GL=egl or osmesa")
    from escape_room.communication.env import make_env

    env = make_env(
        num_envs=1,
        device="cpu",
        render_mode="rgb_array",
        play=True,
        viewer_width=64,
        viewer_height=48,
    )
    try:
        sender, receiver = _render_communication_views(env)
        assert sender.shape == receiver.shape == (48, 64, 3)
        assert not np.array_equal(sender, receiver)
    finally:
        env.close()