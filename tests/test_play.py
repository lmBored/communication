import os
import sys
from argparse import Namespace

from escape_room.play import _needs_mjpython, _relaunch_with_mjpython, _viewer_kind


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