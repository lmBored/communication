"""Scene composition for the two-way communication scenario.

Two sealed rooms. The sender stands in an arrow room holding three coloured
arrows, each pointing LEFT or RIGHT; the receiver stands in a room with two
doorways that are both tinted one of those three colours. Neither agent can see
the other's room, so solving the task requires the colour to travel one way and
the direction the other.

Arrow direction is carried entirely by each sign's mocap quaternion, reusing the
trick from the one-way scenario: the arrow always points along the panel's local
+x, and a 180-degree rotation about z flips it. The arrow geoms are duplicated on
both panel faces so a flipped sign still shows an arrow.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

ARENA_NAME = "arena"
SENDER_NAME = "sender"
RECEIVER_NAME = "receiver"
ARROW_SIGN_NAMES = ("arrow_sign_0", "arrow_sign_1", "arrow_sign_2")
SENDER_CAMERA_NAME = "sender_first_person"
RECEIVER_CAMERA_NAME = "receiver_first_person"

AGENT_RADIUS = 0.4
WALL_HEIGHT = 2.5
WALL_THICKNESS = 0.15
CAMERA_FOV_DEGREES = 120.0

NUM_COLORS = 3
COLOR_NAMES = ("red", "green", "blue")
# Single source of truth: arrow sign k bakes COLOR_RGBA[k] into its arrow geoms,
# and the doorway tint geoms are painted with the same tuple at runtime. All
# three read clearly against the wall grey (0.24 0.27 0.32) and floor grey.
COLOR_RGBA = (
    (0.90, 0.15, 0.12, 1.0),
    (0.10, 0.75, 0.25, 1.0),
    (0.15, 0.35, 0.95, 1.0),
)

ARROW_LAYOUTS = ("front", "scattered")
DEFAULT_ARROW_LAYOUT = "front"

# Receiver room: identical to the one-way scenario so the locomotion
# sub-problem, the entry threshold and the reward geometry stay comparable.
RECEIVER_SPAWN = (3.0, 0.5, 0.45)
DECISION_CENTER_X = RECEIVER_SPAWN[0]
ENTRY_Y = 4.0
TERMINAL_THRESHOLD_Y = ENTRY_Y + 0.4
DOOR_CENTERS = ((0.5, ENTRY_Y), (5.5, ENTRY_Y))

# Arrow room: 8x8, enlarged from the one-way scenario's 6x6 because the
# scattered layout needs standoff to three separate walls.
SENDER_ROOM_CENTER = (-10.0, 0.0)
SENDER_ROOM_HALF_X = 4.0
SENDER_ROOM_HALF_Y = 4.0

ARROW_HEIGHT = 1.25
ARROW_PANEL_HALF_WIDTH = 1.15

ARROW_POSITIONS: dict[str, tuple[tuple[float, float, float], ...]] = {
    "front": (
        (-12.6, 3.70, ARROW_HEIGHT),
        (-10.0, 3.70, ARROW_HEIGHT),
        (-7.4, 3.70, ARROW_HEIGHT),
    ),
    "scattered": (
        (-13.70, 0.0, ARROW_HEIGHT),
        (-10.0, 3.70, ARROW_HEIGHT),
        (-6.30, 0.0, ARROW_HEIGHT),
    ),
}

# Base yaw of each panel, chosen so that "the arrow points RIGHT" means the same
# world direction on every wall. For a viewer facing a panel whose inward normal
# is n, screen-right is (-n) x z_hat, so a panel on the left wall (n = +x) must
# draw its arrow along +y to read as RIGHT.
ARROW_BASE_YAW_DEG: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 0.0, 0.0),
    "scattered": (90.0, 0.0, -90.0),
}

# front wants a long standoff so all three panels fit one view; scattered wants
# the room centre so the three walls sit at -90/0/+90 degrees and no two panels
# can ever share a frame. Observation normalization does not depend on the
# spawn, so the two layouts differ only here.
SENDER_SPAWNS: dict[str, tuple[float, float, float]] = {
    "front": (-10.0, -2.5, 0.45),
    "scattered": (-10.0, 0.0, 0.45),
}

DOOR_TINT_GEOM_NAMES = (
    "door_tint_0",
    "door_tint_1",
    "door_tint_2",
    "door_tint_3",
    "door_tint_4",
    "door_tint_5",
)


def _quat_z(degrees: float) -> tuple[float, float, float, float]:
    """Rotation about z as a MuJoCo (w, x, y, z) quaternion."""
    half = math.radians(degrees) * 0.5
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def arrow_quaternions(
    layout: str,
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    """Per-arrow ``(left_quat, right_quat)`` for one layout.

    All rotations are about z so they commute with the base yaw; LEFT is simply
    the base yaw plus 180 degrees. For the ``front`` layout this reduces to the
    one-way scenario's identity / 180-degree pair.
    """
    return tuple(
        (_quat_z(yaw + 180.0), _quat_z(yaw)) for yaw in ARROW_BASE_YAW_DEG[layout]
    )


def _spec_from_xml(xml: str):
    import mujoco

    return mujoco.MjSpec.from_string(xml)


def _wall(name: str, x: float, y: float, half_x: float, half_y: float) -> str:
    return (
        f'<geom name="{name}" type="box" pos="{x} {y} {WALL_HEIGHT * 0.5}" '
        f'size="{half_x} {half_y} {WALL_HEIGHT * 0.5}" '
        'rgba="0.24 0.27 0.32 1" friction="1 0.01 0.001"/>'
    )


def _rgba_attr(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{component}" for component in rgba)


def _door_tint(
    name: str, x: float, y: float, z: float, half_x: float, half_y: float, half_z: float
) -> str:
    """Visual-only doorway trim whose colour is rewritten per world at runtime.

    Carries no material: ``geom_rgba`` only wins for geoms with ``matid == -1``.
    """
    return (
        f'<geom name="{name}" type="box" pos="{x} {y} {z}" '
        f'size="{half_x} {half_y} {half_z}" contype="0" conaffinity="0" '
        f'rgba="{_rgba_attr(COLOR_RGBA[0])}"/>'
    )


def _arena_spec():
    t = WALL_THICKNESS
    cx, cy = SENDER_ROOM_CENTER
    hx, hy = SENDER_ROOM_HALF_X, SENDER_ROOM_HALF_Y
    walls = [
        # Sealed arrow room: x=[-14,-6], y=[-4,4].
        _wall("arrow_back", cx, cy - hy, hx + t, t),
        _wall("arrow_front", cx, cy + hy, hx + t, t),
        _wall("arrow_left", cx - hx, cy, t, hy),
        _wall("arrow_right", cx + hx, cy, t, hy),
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
    # A lintel plus two jambs per doorway. The jambs sit inside the opening and
    # span the camera eye height, so the colour is readable from the room floor
    # rather than only from a high vantage point.
    tints = [
        _door_tint("door_tint_0", 0.5, ENTRY_Y, 2.15, 1.5, 0.22, 0.35),
        _door_tint("door_tint_1", -0.88, ENTRY_Y, 0.90, 0.12, 0.22, 0.90),
        _door_tint("door_tint_2", 1.88, ENTRY_Y, 0.90, 0.12, 0.22, 0.90),
        _door_tint("door_tint_3", 5.5, ENTRY_Y, 2.15, 1.5, 0.22, 0.35),
        _door_tint("door_tint_4", 4.12, ENTRY_Y, 0.90, 0.12, 0.22, 0.90),
        _door_tint("door_tint_5", 6.88, ENTRY_Y, 0.90, 0.12, 0.22, 0.90),
    ]
    floors = (
        f'<geom name="floor_arrow" type="box" pos="{cx} {cy} -0.1" '
        f'size="{hx + t} {hy + t} 0.1" rgba="0.42 0.44 0.48 1" '
        'friction="1 0.01 0.001"/>'
        '<geom name="floor_door" type="box" pos="3 2.5 -0.1" '
        f'size="{5.0 + t} {6.5 + t} 0.1" rgba="0.42 0.44 0.48 1" '
        'friction="1 0.01 0.001"/>'
    )
    lights = (
        '<light name="door_overhead" directional="true" pos="3 2.5 8" '
        'dir="0 0 -1" diffuse="0.9 0.9 0.9"/>'
        f'<light name="arrow_overhead" directional="true" pos="{cx} {cy} 8" '
        'dir="0 0 -1" diffuse="0.9 0.9 0.9" ambient="0.35 0.35 0.35"/>'
    )
    return _spec_from_xml(
        '<mujoco model="twowaycomm_arena"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        + lights
        + floors
        + "".join(walls)
        + "".join(tints)
        + "</body></worldbody></mujoco>"
    )


def _agent_spec(model_name: str, rgba: str, camera_name: str):
    return _spec_from_xml(
        f'<mujoco model="{model_name}"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        '<freejoint name="root"/>'
        f'<geom name="geom" type="cylinder" size="{AGENT_RADIUS} 0.45" '
        f'mass="4" rgba="{rgba}" friction="1 0.02 0.002"/>'
        f'<camera name="{camera_name}" pos="0 0.1 0.35" '
        f'xyaxes="1 0 0 0 0 1" fovy="{CAMERA_FOV_DEGREES}"/>'
        '</body></worldbody></mujoco>'
    )


def _sender_spec():
    return _agent_spec("sender", "0.95 0.35 0.10 1", SENDER_CAMERA_NAME)


def _receiver_spec():
    return _agent_spec("receiver", "0.10 0.35 0.95 1", RECEIVER_CAMERA_NAME)


def _arrow_sign_spec(index: int, layout: str):
    x, y, z = ARROW_POSITIONS[layout][index]
    yaw = ARROW_BASE_YAW_DEG[layout][index]
    colour = _rgba_attr(COLOR_RGBA[index])
    return _spec_from_xml(
        f'<mujoco model="arrow_sign_{index}">'
        '<compiler autolimits="true" angle="degree"/>'
        f'<worldbody><body name="body" mocap="true" pos="{x} {y} {z}" '
        f'euler="0 0 {yaw}">'
        f'<geom name="panel" type="box" size="{ARROW_PANEL_HALF_WIDTH} 0.06 0.65" '
        'contype="0" conaffinity="0" rgba="0.08 0.08 0.08 1"/>'
        f'<geom name="arrow_shaft" type="box" pos="-0.15 -0.07 0" '
        f'size="0.55 0.04 0.10" contype="0" conaffinity="0" rgba="{colour}"/>'
        f'<geom name="arrow_upper" type="box" pos="0.48 -0.07 0.20" '
        f'size="0.34 0.04 0.10" euler="0 -45 0" contype="0" conaffinity="0" '
        f'rgba="{colour}"/>'
        f'<geom name="arrow_lower" type="box" pos="0.48 -0.07 -0.20" '
        f'size="0.34 0.04 0.10" euler="0 45 0" contype="0" conaffinity="0" '
        f'rgba="{colour}"/>'
        f'<geom name="arrow_shaft_back" type="box" pos="-0.15 0.07 0" '
        f'size="0.55 0.04 0.10" contype="0" conaffinity="0" rgba="{colour}"/>'
        f'<geom name="arrow_upper_back" type="box" pos="0.48 0.07 0.20" '
        f'size="0.34 0.04 0.10" euler="0 -45 0" contype="0" conaffinity="0" '
        f'rgba="{colour}"/>'
        f'<geom name="arrow_lower_back" type="box" pos="0.48 0.07 -0.20" '
        f'size="0.34 0.04 0.10" euler="0 45 0" contype="0" conaffinity="0" '
        f'rgba="{colour}"/>'
        '</body></worldbody></mujoco>'
    )


def make_scene_entities(
    arrow_layout: str = DEFAULT_ARROW_LAYOUT,
) -> dict[str, "EntityCfg"]:
    """Create the separately batched physical entities for this task."""
    from mjlab.entity import EntityCfg

    if arrow_layout not in ARROW_LAYOUTS:
        raise ValueError(
            f"unknown arrow layout {arrow_layout!r}; choose from {ARROW_LAYOUTS}"
        )
    entities: dict[str, "EntityCfg"] = {
        ARENA_NAME: EntityCfg(spec_fn=_arena_spec),
        SENDER_NAME: EntityCfg(spec_fn=_sender_spec),
        RECEIVER_NAME: EntityCfg(spec_fn=_receiver_spec),
    }
    for index, name in enumerate(ARROW_SIGN_NAMES):
        entities[name] = EntityCfg(
            spec_fn=lambda i=index, layout=arrow_layout: _arrow_sign_spec(i, layout)
        )
    return entities
