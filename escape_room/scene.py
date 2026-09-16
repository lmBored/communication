"""Fixed-topology MuJoCo scene composition for mjlab."""

from __future__ import annotations

from typing import TYPE_CHECKING

from escape_room.consts import (
    AGENT_RADIUS,
    BUTTON_WIDTH,
    DOOR_WIDTH,
    MAX_ENTITIES_PER_ROOM,
    NUM_AGENTS,
    NUM_ROOMS,
    ROOM_LENGTH,
    WALL_WIDTH,
    WORLD_LENGTH,
    WORLD_WIDTH,
)

if TYPE_CHECKING:
    from mjlab.entity import EntityCfg

# Agent body names
AGENT_BODY_NAMES = [f"agent_{i}" for i in range(NUM_AGENTS)]
AGENT_SENSOR_NAMES = [f"agent_{i}_sensor" for i in range(NUM_AGENTS)]
AGENT_JOINT_NAMES = [f"agent_{i}_joint" for i in range(NUM_AGENTS)]
AGENT_GEOM_NAMES = [f"agent_{i}_geom" for i in range(NUM_AGENTS)]

# Border wall names
BORDER_NAMES = ["border_behind", "border_right", "border_left"]

# Room wall and door names
ROOM_WALL_LEFT_NAMES = [f"room{r}_wall_left" for r in range(NUM_ROOMS)]
ROOM_WALL_RIGHT_NAMES = [f"room{r}_wall_right" for r in range(NUM_ROOMS)]
ROOM_DOOR_NAMES = [f"room{r}_door" for r in range(NUM_ROOMS)]
ROOM_DOOR_GEOM_NAMES = [f"room{r}_door_geom" for r in range(NUM_ROOMS)]

# Entity slot names per room
ENTITY_SLOT_NAMES = []
for r in range(NUM_ROOMS):
    room_slots = [f"r{r}_ent{j}" for j in range(MAX_ENTITIES_PER_ROOM)]
    ENTITY_SLOT_NAMES.append(room_slots)

# All entity slot names flattened
ALL_ENTITY_SLOT_NAMES = [name for room in ENTITY_SLOT_NAMES for name in room]

# Geom names for entity slots
ENTITY_GEOM_NAMES = [f"{name}_geom" for name in ALL_ENTITY_SLOT_NAMES]

# Actuator names (per agent: force_x, force_y, torque_z)
ACTUATOR_NAMES = []
for i in range(NUM_AGENTS):
    ACTUATOR_NAMES.extend([
        f"agent{i}_force_x",
        f"agent{i}_force_y",
        f"agent{i}_torque_z",
    ])

# Lidar sensor names (30 per agent)
from escape_room.consts import NUM_LIDAR_SAMPLES
LIDAR_SENSOR_NAMES = []
for i in range(NUM_AGENTS):
    for j in range(NUM_LIDAR_SAMPLES):
        LIDAR_SENSOR_NAMES.append(f"agent{i}_lidar_{j}")

# Grab weld equality constraint names
GRAB_WELD_NAMES = [f"grab_weld_{i}" for i in range(NUM_AGENTS)]

# Floor geom name
FLOOR_GEOM_NAME = "floor"


def get_entity_slot_name(room_idx: int, slot_idx: int) -> str:
    """Get the body name for an entity slot."""
    return f"r{room_idx}_ent{slot_idx}"


def get_entity_geom_name(room_idx: int, slot_idx: int) -> str:
    """Get the geom name for an entity slot."""
    return f"r{room_idx}_ent{slot_idx}_geom"


def get_room_door_name(room_idx: int) -> str:
    """Get the door body name for a room."""
    return f"room{room_idx}_door"


def get_room_door_geom_name(room_idx: int) -> str:
    """Get the door geom name for a room."""
    return f"room{room_idx}_door_geom"


def get_agent_body_name(agent_idx: int) -> str:
    """Get the agent body name."""
    return f"agent_{agent_idx}"


def get_actuator_names(agent_idx: int) -> list:
    """Get actuator names for an agent."""
    return [
        f"agent{agent_idx}_force_x",
        f"agent{agent_idx}_force_y",
        f"agent{agent_idx}_torque_z",
    ]


BUTTON_ENTITY_NAMES = [
    f"button_{room}_{slot}" for room in range(NUM_ROOMS) for slot in range(2)
]
CUBE_ENTITY_NAMES = [
    f"cube_{room}_{slot}" for room in range(NUM_ROOMS) for slot in range(2, 6)
]
DOOR_ENTITY_NAMES = [f"door_{room}" for room in range(NUM_ROOMS)]


def _spec_from_xml(xml: str):
    import mujoco

    return mujoco.MjSpec.from_string(xml)


def _arena_spec():
    half_width = WORLD_WIDTH * 0.5
    half_door = DOOR_WIDTH * 0.5
    half_wall = WALL_WIDTH * 0.5
    wall_height = 1.75
    geoms = [
        f'<geom name="floor" type="box" pos="0 {WORLD_LENGTH * 0.5} -0.1" '
        f'size="{half_width} {WORLD_LENGTH * 0.5} 0.1" '
        'rgba="0.42 0.44 0.48 1" friction="1 0.01 0.001"/>',
        f'<geom name="border_left" type="box" pos="{-half_width - half_wall} '
        f'{WORLD_LENGTH * 0.5} {wall_height * 0.5}" '
        f'size="{half_wall} {WORLD_LENGTH * 0.5} {wall_height * 0.5}" '
        'rgba="0.30 0.32 0.36 1"/>',
        f'<geom name="border_right" type="box" pos="{half_width + half_wall} '
        f'{WORLD_LENGTH * 0.5} {wall_height * 0.5}" '
        f'size="{half_wall} {WORLD_LENGTH * 0.5} {wall_height * 0.5}" '
        'rgba="0.30 0.32 0.36 1"/>',
        f'<geom name="border_back" type="box" pos="0 {-half_wall} '
        f'{wall_height * 0.5}" size="{half_width + half_wall} {half_wall} '
        f'{wall_height * 0.5}" rgba="0.30 0.32 0.36 1"/>',
        f'<geom name="border_front" type="box" pos="0 {WORLD_LENGTH + half_wall} '
        f'{wall_height * 0.5}" size="{half_width + half_wall} {half_wall} '
        f'{wall_height * 0.5}" rgba="0.30 0.32 0.36 1"/>',
    ]
    left_center = (-half_width - half_door) * 0.5
    right_center = (half_width + half_door) * 0.5
    segment_half = (half_width - half_door) * 0.5
    for room in range(NUM_ROOMS):
        y = (room + 1) * ROOM_LENGTH
        for side, x in (("left", left_center), ("right", right_center)):
            geoms.append(
                f'<geom name="room{room}_{side}" type="box" '
                f'pos="{x} {y} {wall_height * 0.5}" '
                f'size="{segment_half} {half_wall} {wall_height * 0.5}" '
                'rgba="0.30 0.32 0.36 1"/>'
            )
    return _spec_from_xml(
        '<mujoco model="arena"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        + "".join(geoms)
        + "</body></worldbody></mujoco>"
    )


def _agent_spec(agent_idx: int):
    color = "0.95 0.35 0.10 1" if agent_idx == 0 else "0.10 0.35 0.95 1"
    return _spec_from_xml(
        '<mujoco model="agent"><compiler autolimits="true"/>'
        '<worldbody><body name="body">'
        '<freejoint name="root"/>'
        f'<geom name="geom" type="cylinder" size="{AGENT_RADIUS} 0.5" '
        f'mass="4" rgba="{color}" friction="1 0.02 0.002"/>'
        '<site name="heading" pos="0 0.8 0.35" size="0.10" '
        'rgba="1 1 1 1"/>'
        '</body></worldbody></mujoco>'
    )


def _door_spec():
    return _spec_from_xml(
        '<mujoco model="door"><worldbody><body name="body">'
        f'<geom name="geom" type="box" size="{DOOR_WIDTH * 0.5} '
        f'{WALL_WIDTH * 0.5} 0.875" rgba="0.75 0.18 0.12 1" '
        'friction="0.8 0.01 0.001"/>'
        '</body></worldbody></mujoco>'
    )


def _button_spec():
    return _spec_from_xml(
        '<mujoco model="button"><worldbody><body name="body">'
        f'<geom name="geom" type="cylinder" size="{BUTTON_WIDTH * 0.5} 0.10" '
        'contype="0" conaffinity="0" rgba="0.85 0.75 0.05 1"/>'
        '</body></worldbody></mujoco>'
    )


def _cube_spec(cube_idx: int):
    parking_x = -30.0 - 2.0 * cube_idx
    return _spec_from_xml(
        '<mujoco model="cube"><compiler autolimits="true"/>'
        f'<worldbody><body name="body" pos="{parking_x} 0 -10">'
        '<freejoint name="root"/>'
        '<geom name="geom" type="box" size="0.75 0.75 0.75" mass="2" '
        'rgba="0.12 0.28 0.85 1" friction="1.2 0.02 0.002"/>'
        '</body></worldbody></mujoco>'
    )


def make_scene_entities() -> dict[str, "EntityCfg"]:
    """Create the fixed set of independently batched mjlab entities."""
    from mjlab.entity import EntityCfg

    entities: dict[str, EntityCfg] = {
        "arena": EntityCfg(spec_fn=_arena_spec),
    }
    for agent_idx in range(NUM_AGENTS):
        entities[f"agent_{agent_idx}"] = EntityCfg(
            spec_fn=lambda idx=agent_idx: _agent_spec(idx)
        )
    for name in DOOR_ENTITY_NAMES:
        entities[name] = EntityCfg(spec_fn=_door_spec)
    for name in BUTTON_ENTITY_NAMES:
        entities[name] = EntityCfg(spec_fn=_button_spec)
    for cube_idx, name in enumerate(CUBE_ENTITY_NAMES):
        entities[name] = EntityCfg(
            spec_fn=lambda idx=cube_idx: _cube_spec(idx)
        )
    return entities
