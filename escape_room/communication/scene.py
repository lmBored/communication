"""Physical scene for the isolated sender/receiver communication task."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

SENDER_NAME = "sender"
RECEIVER_NAME = "receiver"
CLUE_SIGN_NAME = "clue_sign"
SENDER_CAMERA_NAME = "sender_first_person"
RECEIVER_CAMERA_NAME = "receiver_first_person"

AGENT_RADIUS = 0.4
WALL_HEIGHT = 2.5
WALL_THICKNESS = 0.15
CAMERA_FOV_DEGREES = 120.0

SENDER_SPAWN = (-9.0, -1.5, 0.45)
RECEIVER_SPAWN = (3.0, 0.5, 0.45)
DECISION_CENTER_X = RECEIVER_SPAWN[0]
ENTRY_Y = 4.0
TERMINAL_THRESHOLD_Y = ENTRY_Y + 0.4
DOOR_CENTERS = ((0.5, ENTRY_Y), (5.5, ENTRY_Y))
CLUE_SIGN_POSITION = (-9.0, 2.7, 1.25)


def _spec_from_xml(xml: str):
    import mujoco

    return mujoco.MjSpec.from_string(xml)


def _wall(name: str, x: float, y: float, half_x: float, half_y: float) -> str:
    return (
        f'<geom name="{name}" type="box" pos="{x} {y} {WALL_HEIGHT * 0.5}" '
        f'size="{half_x} {half_y} {WALL_HEIGHT * 0.5}" '
        'rgba="0.24 0.27 0.32 1" friction="1 0.01 0.001"/>'
    )


def _arena_spec():
    t = WALL_THICKNESS
    walls = [
        # Sealed clue-holder room: x=[-12,-6], y=[-3,3].
        _wall("sender_back", -9.0, -3.0, 3.0 + t, t),
        _wall("sender_front", -9.0, 3.0, 3.0 + t, t),
        _wall("sender_left", -12.0, 0.0, t, 3.0),
        _wall("sender_right", -6.0, 0.0, t, 3.0),
        # Decision room and terminal rooms: x=[-2,8], y=[-4,9].
        _wall("decision_back", 3.0, -4.0, 5.0 + t, t),
        _wall("decision_left", -2.0, 2.5, t, 6.5),
        _wall("decision_right", 8.0, 2.5, t, 6.5),
        _wall("terminal_front", 3.0, 9.0, 5.0 + t, t),
        # The y=4 wall leaves equal wide openings centered at x=.5 and x=5.5.
        _wall("entry_left_segment", -1.5, ENTRY_Y, 0.5, t),
        _wall("entry_middle_segment", 3.0, ENTRY_Y, 1.0, t),
        _wall("entry_right_segment", 7.5, ENTRY_Y, 0.5, t),
        # A thick divider prevents crossing between terminal rooms after entry.
        _wall("terminal_divider", 3.0, 6.5, 0.5, 2.5),
    ]
    floor = (
        '<geom name="floor" type="box" pos="-2 2.5 -0.1" '
        'size="10.5 6.75 0.1" rgba="0.42 0.44 0.48 1" '
        'friction="1 0.01 0.001"/>'
    )
    doorway_markers = (
        '<geom name="left_marker" type="box" pos="0.5 4.08 2.15" '
        'size="1.5 0.05 0.12" contype="0" conaffinity="0" '
        'rgba="0.18 0.65 0.95 1"/>'
        '<geom name="right_marker" type="box" pos="5.5 4.08 2.15" '
        'size="1.5 0.05 0.12" contype="0" conaffinity="0" '
        'rgba="0.95 0.55 0.18 1"/>'
    )
    return _spec_from_xml(
        '<mujoco model="communication_arena"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        '<light name="overhead" directional="true" pos="0 0 8" '
        'dir="0 0 -1" diffuse="0.9 0.9 0.9"/>'
        + floor
        + "".join(walls)
        + doorway_markers
        + "</body></worldbody></mujoco>"
    )


def _sender_spec():
    x, y, z = SENDER_SPAWN
    return _spec_from_xml(
        '<mujoco model="sender"><compiler autolimits="true"/>'
        f'<worldbody><body name="body" pos="{x} {y} {z}">'
        f'<geom name="geom" type="cylinder" size="{AGENT_RADIUS} 0.45" '
        'rgba="0.95 0.35 0.10 1" friction="1 0.02 0.002"/>'
        f'<camera name="{SENDER_CAMERA_NAME}" pos="0 0.1 0.35" '
        f'xyaxes="1 0 0 0 0 1" fovy="{CAMERA_FOV_DEGREES}"/>'
        '</body></worldbody></mujoco>'
    )


def _receiver_spec():
    return _spec_from_xml(
        '<mujoco model="receiver"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        '<freejoint name="root"/>'
        f'<geom name="geom" type="cylinder" size="{AGENT_RADIUS} 0.45" '
        'mass="4" rgba="0.10 0.35 0.95 1" friction="1 0.02 0.002"/>'
        f'<camera name="{RECEIVER_CAMERA_NAME}" pos="0 0.1 0.35" '
        f'xyaxes="1 0 0 0 0 1" fovy="{CAMERA_FOV_DEGREES}"/>'
        '</body></worldbody></mujoco>'
    )


def _clue_sign_spec():
    x, y, z = CLUE_SIGN_POSITION
    return _spec_from_xml(
        '<mujoco model="clue_sign"><compiler autolimits="true" angle="degree"/>'
        f'<worldbody><body name="body" mocap="true" pos="{x} {y} {z}">'
        '<geom name="panel" type="box" size="1.15 0.06 0.65" '
        'contype="0" conaffinity="0" rgba="0.08 0.08 0.08 1"/>'
        '<geom name="arrow_shaft" type="box" pos="-0.15 -0.07 0" '
        'size="0.55 0.04 0.10" contype="0" conaffinity="0" '
        'rgba="0.95 0.92 0.18 1"/>'
        '<geom name="arrow_upper" type="box" pos="0.48 -0.07 0.20" '
        'size="0.34 0.04 0.10" euler="0 -45 0" '
        'contype="0" conaffinity="0" rgba="0.95 0.92 0.18 1"/>'
        '<geom name="arrow_lower" type="box" pos="0.48 -0.07 -0.20" '
        'size="0.34 0.04 0.10" euler="0 45 0" '
        'contype="0" conaffinity="0" rgba="0.95 0.92 0.18 1"/>'
        '<geom name="arrow_shaft_back" type="box" pos="-0.15 0.07 0" '
        'size="0.55 0.04 0.10" contype="0" conaffinity="0" '
        'rgba="0.95 0.92 0.18 1"/>'
        '<geom name="arrow_upper_back" type="box" pos="0.48 0.07 0.20" '
        'size="0.34 0.04 0.10" euler="0 -45 0" '
        'contype="0" conaffinity="0" rgba="0.95 0.92 0.18 1"/>'
        '<geom name="arrow_lower_back" type="box" pos="0.48 0.07 -0.20" '
        'size="0.34 0.04 0.10" euler="0 45 0" '
        'contype="0" conaffinity="0" rgba="0.95 0.92 0.18 1"/>'
        '</body></worldbody></mujoco>'
    )


def make_scene_entities() -> dict[str, "EntityCfg"]:
    """Create the separately batched physical entities for this task."""
    from mjlab.entity import EntityCfg

    return {
        "arena": EntityCfg(spec_fn=_arena_spec),
        SENDER_NAME: EntityCfg(spec_fn=_sender_spec),
        RECEIVER_NAME: EntityCfg(spec_fn=_receiver_spec),
        CLUE_SIGN_NAME: EntityCfg(spec_fn=_clue_sign_spec),
    }