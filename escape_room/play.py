"""
Real MuJoCo 3D playback and MP4 recording for the escape room environment.

Examples:
    # Heuristic demo in the native / Viser 3D viewer
    escape-room-play

    # Random agents
    escape-room-play --policy random

    # Roll out an rsl-rl checkpoint
    escape-room-play --ckpt ./ckpts/model_49.pt

    # Headless MuJoCo offscreen recording (servers / CI)
    escape-room-play --headless --record demos/escape_demo.mp4 --steps 400
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from escape_room.consts import (
    ACTION_DIM_PER_AGENT,
    DELTA_T,
    NUM_AGENTS,
    NUM_ROOMS,
    ROOM_LENGTH,
)

COMMUNICATION_PANEL_HEIGHT = 120
SENDER_VIEW_LABEL = "SENDER / CLUE HOLDER"
RECEIVER_VIEW_LABEL = "RECEIVER / DOOR AGENT"


class FFmpegVideoWriter:
    """Pipe raw RGB frames into ffmpeg (no imageio dependency)."""

    def __init__(self, path: Path, width: int, height: int, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found on PATH; install it to record video"
            )
        # Ensure even dimensions for yuv420p
        self._w = width - (width % 2)
        self._h = height - (height % 2)
        cmd = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self._w}x{self._h}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        rgb = np.asarray(frame, dtype=np.uint8)
        if rgb.shape[0] != self._h or rgb.shape[1] != self._w:
            rgb = rgb[: self._h, : self._w]
        self._proc.stdin.write(rgb.tobytes())
        self.frames += 1

    def close(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        ret = self._proc.wait(timeout=60)
        if ret != 0:
            err = b""
            if self._proc.stderr:
                err = self._proc.stderr.read()
            raise RuntimeError(
                f"ffmpeg failed (code={ret}): {err.decode('utf-8', 'ignore')}"
            )
        print(f"wrote {self.frames} frames → {self.path.resolve()}")


def _communication_panel_lines(
    message: np.ndarray | None, probe: dict | None
) -> tuple[str, str, str]:
    """Build the explanatory text shown below communication recordings."""
    if message is None:
        vector_line = "MESSAGE: unavailable"
        decoded = "unavailable"
    else:
        values = np.asarray(message, dtype=np.float32).reshape(-1)
        vector = ", ".join(f"{value:.3f}" for value in values)
        vector_line = f"MESSAGE: [{vector}]"
        if probe is None:
            decoded = "probe unavailable"
        else:
            from escape_room.communication.evaluate import decode_messages

            direction = int(decode_messages(torch.from_numpy(values), probe)[0])
            decoded = "LEFT" if direction < 0 else "RIGHT"
    return (
        "SENDER -> RECEIVER",
        vector_line,
        f"PROBE (diagnostic only): {decoded}",
    )


def _compose_communication_frame(
    sender_frame: np.ndarray,
    receiver_frame: np.ndarray,
    message: np.ndarray | None,
    probe: dict | None,
) -> np.ndarray:
    """Compose both first-person views over a communication message panel."""
    sender = Image.fromarray(np.asarray(sender_frame, dtype=np.uint8), mode="RGB")
    receiver = Image.fromarray(
        np.asarray(receiver_frame, dtype=np.uint8), mode="RGB"
    )
    if receiver.size != sender.size:
        receiver = receiver.resize(sender.size)
    width, height = sender.size
    canvas = Image.new("RGB", (width * 2, height + COMMUNICATION_PANEL_HEIGHT))
    canvas.paste(sender, (0, 0))
    canvas.paste(receiver, (width, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=16)
    draw.rectangle((0, 0, width, 30), fill=(12, 16, 22))
    draw.rectangle((width, 0, width * 2, 30), fill=(12, 16, 22))
    draw.text((8, 7), SENDER_VIEW_LABEL, fill=(255, 255, 255), font=font)
    draw.text((width + 8, 7), RECEIVER_VIEW_LABEL, fill=(255, 255, 255), font=font)
    draw.rectangle(
        (0, height, width * 2, height + COMMUNICATION_PANEL_HEIGHT),
        fill=(12, 16, 22),
    )
    for row, line in enumerate(_communication_panel_lines(message, probe)):
        draw.text(
            (16, height + 12 + row * 34),
            line,
            fill=(238, 241, 246),
            font=font,
        )
    return np.asarray(canvas)


def _render_communication_views(env) -> tuple[np.ndarray, np.ndarray]:
    """Render both named body cameras from the current simulation state."""
    from escape_room.communication.scene import (
        RECEIVER_NAME,
        RECEIVER_CAMERA_NAME,
        SENDER_NAME,
        SENDER_CAMERA_NAME,
    )

    renderer = env._offline_renderer
    if renderer is None:
        raise RuntimeError("communication camera rendering requires rgb_array mode")
    renderer.update(env.sim.data, camera=f"{SENDER_NAME}/{SENDER_CAMERA_NAME}")
    sender = renderer.render().copy()
    renderer.update(env.sim.data, camera=f"{RECEIVER_NAME}/{RECEIVER_CAMERA_NAME}")
    receiver = renderer.render().copy()
    return sender, receiver


def heuristic_policy(env):
    """Return a state-aware demo policy for the real MuJoCo environment."""
    from escape_room.mjlab_env import game_term

    game = game_term(env.unwrapped)

    def policy(_obs):
        pos, yaw = game._agent_pose()
        action = torch.zeros(
            (game.num_envs, NUM_AGENTS, ACTION_DIM_PER_AGENT),
            device=game.device,
        )
        for agent_idx in range(NUM_AGENTS):
            room = torch.clamp(
                (pos[:, agent_idx, 1] / ROOM_LENGTH).long(), 0, NUM_ROOMS - 1
            )
            env_ids = torch.arange(game.num_envs, device=game.device)
            is_open = game.door_open[env_ids, room]
            target = game.door_pos[env_ids, room, :2].clone()
            target[:, 1] += 1.5
            for room_idx in range(NUM_ROOMS):
                selected = (room == room_idx) & ~is_open
                if not selected.any():
                    continue
                buttons = game.door_button_mask[:, room_idx, :2]
                slot = torch.full(
                    (game.num_envs,), agent_idx, dtype=torch.long, device=game.device
                )
                one_button = buttons.sum(-1) == 1
                slot = torch.where(one_button, buttons.long().argmax(-1), slot)
                target[selected] = game.entity_pos[
                    env_ids[selected], room_idx, slot[selected], :2
                ]
            delta = target - pos[:, agent_idx, :2]
            distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.0e-6)
            world = delta / distance[:, None]
            world = torch.where(
                (distance < 0.35)[:, None], torch.zeros_like(world), world
            )
            c, s = torch.cos(yaw[:, agent_idx]), torch.sin(yaw[:, agent_idx])
            action[:, agent_idx, 0] = world[:, 0] * c + world[:, 1] * s
            action[:, agent_idx, 1] = -world[:, 0] * s + world[:, 1] * c
        return action.flatten(1)

    return policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Play the escape room in the MuJoCo 3D viewer, optionally to MP4"
    )
    p.add_argument(
        "--task",
        choices=["escape-room", "communication"],
        default="escape-room",
        help="scenario to play",
    )
    p.add_argument(
        "--policy",
        type=str,
        default="heuristic",
        choices=["random", "heuristic", "checkpoint"],
        help="Action source (checkpoint requires --ckpt)",
    )
    p.add_argument("--ckpt", type=str, default=None, help="Path to .pt checkpoint")
    p.add_argument(
        "--message-dim",
        type=int,
        default=None,
        help="override the message width inferred from a communication checkpoint",
    )
    p.add_argument(
        "--message-probe",
        type=str,
        default=None,
        help="nearest-centroid probe JSON (defaults beside checkpoint)",
    )
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400, help="Headless control steps")
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Vector env count (only env 0 is visualized)",
    )
    p.add_argument("--fps", type=float, default=1.0 / DELTA_T, help="Video FPS")
    p.add_argument(
        "--record",
        type=str,
        default=None,
        help="Output MP4 path (requires ffmpeg)",
    )
    p.add_argument(
        "--headless",
        action="store_true",
        help="No interactive viewer (use together with --record)",
    )
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=960)
    return p.parse_args(argv)


def _viewer_kind(
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    platform_name = platform.system() if platform_name is None else platform_name
    environ = os.environ if environ is None else environ
    has_display = bool(environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"))
    return "native" if platform_name == "Darwin" or has_display else "viser"


def _needs_mjpython(
    args: argparse.Namespace,
    platform_name: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    platform_name = platform.system() if platform_name is None else platform_name
    environ = os.environ if environ is None else environ
    return (
        platform_name == "Darwin"
        and not args.headless
        and "MJPYTHON_BIN" not in environ
    )


def _relaunch_with_mjpython(
    args: argparse.Namespace, argv: list[str] | None = None
) -> None:
    if not _needs_mjpython(args):
        return

    sibling = Path(sys.executable).with_name("mjpython")
    executable = str(sibling) if sibling.is_file() else shutil.which("mjpython")
    if executable is None:
        raise SystemExit(
            "Interactive MuJoCo playback on macOS requires mjpython, but it "
            "was not found beside the active Python interpreter or on PATH."
        )

    forwarded_args = sys.argv[1:] if argv is None else argv
    print("Restarting interactive playback with mjpython (required on macOS).", flush=True)
    libpython_dir = str(Path(sys.executable).resolve().parent.parent / "lib")
    fallback = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
    if fallback:
        libpython_dir = f"{libpython_dir}:{fallback}"
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = libpython_dir
    os.execv(
        executable,
        [executable, "-m", "escape_room.play", *forwarded_args],
    )


def _checkpoint_uses_scalar_std(path: str | Path) -> bool:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    actor_state = checkpoint.get("actor_state_dict", {})
    return "distribution.std_param" in actor_state


def main(argv: list[str] | None = None) -> None:
    """Run the real MuJoCo 3D viewer and optionally record its RGB output."""
    args = parse_args(argv)
    _relaunch_with_mjpython(args, argv)
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable; use --device cpu for playback")
    if args.policy == "checkpoint" and not args.ckpt:
        raise SystemExit("--policy checkpoint requires --ckpt")
    if args.task == "communication" and args.policy == "heuristic":
        raise SystemExit(
            "communication playback requires --policy checkpoint or --policy random"
        )
    if args.headless and not args.record:
        print("warning: --headless without --record produces no visual output")

    torch.manual_seed(args.seed)
    if args.task == "communication":
        from escape_room.communication.env import make_env
        from escape_room.communication.env_cfg import communication_ppo_runner_cfg
        from escape_room.communication.evaluate import checkpoint_actor_architecture

        message_dim = args.message_dim
        architecture = checkpoint_actor_architecture(args.ckpt) if args.ckpt else None
        if message_dim is None:
            message_dim = architecture["message_dim"] if architecture else 2
        runner_cfg = communication_ppo_runner_cfg(message_dim=message_dim)
        if architecture:
            runner_cfg.actor.hidden_dims = architecture["hidden_dims"]
            runner_cfg.actor.sender_hidden_dims = architecture["sender_hidden_dims"]
    else:
        from escape_room.env import make_env
        from escape_room.env_cfg import escape_room_ppo_runner_cfg

        runner_cfg = escape_room_ppo_runner_cfg()
    env = make_env(
        num_envs=args.num_envs,
        device=device,
        seed=args.seed,
        render_mode="rgb_array" if args.record else None,
        play=True,
        viewer_width=args.width,
        viewer_height=args.height,
    )
    if (
        args.task == "escape-room"
        and args.ckpt
        and _checkpoint_uses_scalar_std(args.ckpt)
    ):
        runner_cfg.actor.distribution_cfg["std_type"] = "scalar"
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.clip_actions)

    if args.ckpt:
        runner = MjlabOnPolicyRunner(wrapped, asdict(runner_cfg), device=device)
        runner.load(
            str(Path(args.ckpt)),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
    elif args.policy == "random":
        policy = lambda _obs: 2.0 * torch.rand(
            wrapped.unwrapped.action_space.shape, device=device
        ) - 1.0
    else:
        policy = heuristic_policy(wrapped)

    message_probe = None
    if args.task == "communication" and args.ckpt:
        from escape_room.communication.evaluate import (
            default_probe_path,
            load_message_probe,
        )

        probe_path = Path(args.message_probe or default_probe_path(args.ckpt))
        if probe_path.is_file():
            message_probe = load_message_probe(probe_path)
        else:
            print(f"warning: message probe not found at {probe_path}")

    if args.headless:
        obs = wrapped.get_observations()
        writer = None
        if args.record:
            width = args.width * 2 if args.task == "communication" else args.width
            height = (
                args.height + COMMUNICATION_PANEL_HEIGHT
                if args.task == "communication"
                else args.height
            )
            writer = FFmpegVideoWriter(
                Path(args.record), width, height, args.fps
            )
        try:
            for _ in range(args.steps):
                with torch.inference_mode():
                    action = policy(obs)
                    message = None
                    if args.task == "communication" and hasattr(
                        policy, "last_message"
                    ):
                        message = policy.last_message[0].cpu().numpy()
                if writer is not None and args.task == "communication":
                    sender_frame, receiver_frame = _render_communication_views(env)
                    writer.write(
                        _compose_communication_frame(
                            sender_frame,
                            receiver_frame,
                            message,
                            message_probe,
                        )
                    )
                with torch.inference_mode():
                    obs, _, _, _ = wrapped.step(action)
                if writer is not None and args.task != "communication":
                    writer.write(env.render())
        finally:
            if writer is not None:
                writer.close()
            env.close()
        return

    if _viewer_kind() == "native":
        NativeMujocoViewer(wrapped, policy).run()
    else:
        print("Opening the mjlab 3D Viser viewer; follow the URL shown below.")
        ViserPlayViewer(wrapped, policy).run()
    env.close()


if __name__ == "__main__":
    main()
