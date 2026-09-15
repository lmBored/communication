"""
Procedural level generation for the escape room.

Ports the room generation logic from src/level_gen.cpp. Each room type
configures the fixed entity slots by repositioning and retyping them,
without rebuilding the MJCF topology.
"""

import math
import random
from typing import Dict, List, Tuple

from escape_room.consts import (
    NUM_ROOMS,
    MAX_ENTITIES_PER_ROOM,
    WORLD_WIDTH,
    WORLD_LENGTH,
    ROOM_LENGTH,
    WALL_WIDTH,
    BUTTON_WIDTH,
    DOOR_WIDTH,
    AGENT_RADIUS,
    RoomType,
    EntityType,
)


class EntitySlot:
    """Represents a fixed entity slot that can be repositioned and retyped."""

    def __init__(self, slot_idx: int):
        self.slot_idx = slot_idx
        self.type = EntityType.NONE
        self.active = False
        self.pos_x: float = 0.0
        self.pos_y: float = 0.0
        self.pos_z: float = 0.0
        self.scale: float = 1.0


class RoomConfig:
    """Configuration for a single room."""

    def __init__(self, room_idx: int):
        self.room_idx = room_idx
        self.entities: List[EntitySlot] = [
            EntitySlot(j) for j in range(MAX_ENTITIES_PER_ROOM)
        ]
        self.door_x: float = 0.0
        self.door_y: float = 0.0
        self.wall_left_len: float = 0.0
        self.wall_right_len: float = 0.0
        self.button_indices: List[int] = []
        self.is_persistent: bool = True


def rand_between(rng: random.Random, min_val: float, max_val: float) -> float:
    """Random float in [min_val, max_val)."""
    return rng.uniform(min_val, max_val)


def rand_centered(rng: random.Random, range_val: float) -> float:
    """Random float centered at 0 with given range."""
    return rng.uniform(-range_val / 2, range_val / 2)


def make_single_button_room(
    room: RoomConfig, rng: random.Random
) -> None:
    """
    A room with a single button that needs to be pressed.
    The door stays open once pressed (persistent).
    """
    y_min = room.room_idx * ROOM_LENGTH
    y_max = (room.room_idx + 1) * ROOM_LENGTH

    button_x = rand_centered(rng, WORLD_WIDTH / 2 - BUTTON_WIDTH)
    button_y = rand_between(
        rng, y_min + ROOM_LENGTH / 4, y_max - WALL_WIDTH - BUTTON_WIDTH / 2
    )

    slot = room.entities[0]
    slot.type = EntityType.BUTTON
    slot.active = True
    slot.pos_x = button_x
    slot.pos_y = button_y
    slot.pos_z = 0.0
    slot.scale = 1.0

    room.button_indices = [0]
    room.is_persistent = True

    # Disable remaining slots
    for i in range(1, MAX_ENTITIES_PER_ROOM):
        room.entities[i].active = False
        room.entities[i].type = EntityType.NONE


def make_double_button_room(
    room: RoomConfig, rng: random.Random
) -> None:
    """
    A room with two buttons that need to be pressed simultaneously.
    The door stays open once pressed (persistent).
    """
    y_min = room.room_idx * ROOM_LENGTH
    y_max = (room.room_idx + 1) * ROOM_LENGTH

    # Button A (left side)
    a_x = rand_between(
        rng, -WORLD_WIDTH / 2 + BUTTON_WIDTH, -BUTTON_WIDTH
    )
    a_y = rand_between(
        rng, y_min + ROOM_LENGTH / 4, y_max - WALL_WIDTH - BUTTON_WIDTH / 2
    )

    slot_a = room.entities[0]
    slot_a.type = EntityType.BUTTON
    slot_a.active = True
    slot_a.pos_x = a_x
    slot_a.pos_y = a_y
    slot_a.pos_z = 0.0
    slot_a.scale = 1.0

    # Button B (right side)
    b_x = rand_between(
        rng, BUTTON_WIDTH, WORLD_WIDTH / 2 - BUTTON_WIDTH
    )
    b_y = rand_between(
        rng, y_min + ROOM_LENGTH / 4, y_max - WALL_WIDTH - BUTTON_WIDTH / 2
    )

    slot_b = room.entities[1]
    slot_b.type = EntityType.BUTTON
    slot_b.active = True
    slot_b.pos_x = b_x
    slot_b.pos_y = b_y
    slot_b.pos_z = 0.0
    slot_b.scale = 1.0

    room.button_indices = [0, 1]
    room.is_persistent = True

    # Disable remaining slots
    for i in range(2, MAX_ENTITIES_PER_ROOM):
        room.entities[i].active = False
        room.entities[i].type = EntityType.NONE


def make_cube_blocking_room(
    room: RoomConfig, rng: random.Random
) -> None:
    """
    A room with 3 cubes blocking the door and 2 buttons.
    Agents can either pull cubes out of the way or open the door with buttons.
    The door stays open once pressed (persistent).
    """
    y_min = room.room_idx * ROOM_LENGTH
    y_max = (room.room_idx + 1) * ROOM_LENGTH

    # Button A (left side)
    button_a_x = rand_between(
        rng, -WORLD_WIDTH / 2 + BUTTON_WIDTH, -BUTTON_WIDTH - WORLD_WIDTH / 4
    )
    button_a_y = rand_between(
        rng, y_min + BUTTON_WIDTH, y_max - ROOM_LENGTH / 4
    )

    slot_a = room.entities[0]
    slot_a.type = EntityType.BUTTON
    slot_a.active = True
    slot_a.pos_x = button_a_x
    slot_a.pos_y = button_a_y
    slot_a.pos_z = 0.0
    slot_a.scale = 1.0

    # Button B (right side)
    button_b_x = rand_between(
        rng, BUTTON_WIDTH + WORLD_WIDTH / 4, WORLD_WIDTH / 2 - BUTTON_WIDTH
    )
    button_b_y = rand_between(
        rng, y_min + BUTTON_WIDTH, y_max - ROOM_LENGTH / 4
    )

    slot_b = room.entities[1]
    slot_b.type = EntityType.BUTTON
    slot_b.active = True
    slot_b.pos_x = button_b_x
    slot_b.pos_y = button_b_y
    slot_b.pos_z = 0.0
    slot_b.scale = 1.0

    # Door position for cube placement

    # Cube A (left of door)
    cube_a = room.entities[2]
    cube_a.type = EntityType.CUBE
    cube_a.active = True
    cube_a.pos_x = room.door_x - 3.0
    cube_a.pos_y = room.door_y - 2.0
    cube_a.pos_z = 0.75 * 1.5  # scale * 0.75
    cube_a.scale = 1.5

    # Cube B (center of door)
    cube_b = room.entities[3]
    cube_b.type = EntityType.CUBE
    cube_b.active = True
    cube_b.pos_x = room.door_x
    cube_b.pos_y = room.door_y - 2.0
    cube_b.pos_z = 0.75 * 1.5
    cube_b.scale = 1.5

    # Cube C (right of door)
    cube_c = room.entities[4]
    cube_c.type = EntityType.CUBE
    cube_c.active = True
    cube_c.pos_x = room.door_x + 3.0
    cube_c.pos_y = room.door_y - 2.0
    cube_c.pos_z = 0.75 * 1.5
    cube_c.scale = 1.5

    room.button_indices = [0, 1]
    room.is_persistent = True

    # Disable remaining slots
    for i in range(5, MAX_ENTITIES_PER_ROOM):
        room.entities[i].active = False
        room.entities[i].type = EntityType.NONE


def make_cube_buttons_room(
    room: RoomConfig, rng: random.Random
) -> None:
    """
    A room with 2 buttons and 2 cubes. Buttons must remain pressed for door
    to stay open (non-persistent). Agents push cubes onto buttons.
    """
    y_min = room.room_idx * ROOM_LENGTH
    y_max = (room.room_idx + 1) * ROOM_LENGTH

    # Button A (left side)
    button_a_x = rand_between(
        rng, -WORLD_WIDTH / 2 + BUTTON_WIDTH, -BUTTON_WIDTH - WORLD_WIDTH / 4
    )
    button_a_y = rand_between(
        rng, y_min + BUTTON_WIDTH, y_max - ROOM_LENGTH / 4
    )

    slot_a = room.entities[0]
    slot_a.type = EntityType.BUTTON
    slot_a.active = True
    slot_a.pos_x = button_a_x
    slot_a.pos_y = button_a_y
    slot_a.pos_z = 0.0
    slot_a.scale = 1.0

    # Button B (right side)
    button_b_x = rand_between(
        rng, BUTTON_WIDTH + WORLD_WIDTH / 4, WORLD_WIDTH / 2 - BUTTON_WIDTH
    )
    button_b_y = rand_between(
        rng, y_min + BUTTON_WIDTH, y_max - ROOM_LENGTH / 4
    )

    slot_b = room.entities[1]
    slot_b.type = EntityType.BUTTON
    slot_b.active = True
    slot_b.pos_x = button_b_x
    slot_b.pos_y = button_b_y
    slot_b.pos_z = 0.0
    slot_b.scale = 1.0

    # Cube A
    cube_a_x = rand_between(rng, -WORLD_WIDTH / 4, -1.5)
    cube_a_y = rand_between(rng, y_min + 2.0, y_max - WALL_WIDTH - 2.0)

    cube_a = room.entities[2]
    cube_a.type = EntityType.CUBE
    cube_a.active = True
    cube_a.pos_x = cube_a_x
    cube_a.pos_y = cube_a_y
    cube_a.pos_z = 0.75 * 1.5
    cube_a.scale = 1.5

    # Cube B
    cube_b_x = rand_between(rng, 1.5, WORLD_WIDTH / 4)
    cube_b_y = rand_between(rng, y_min + 2.0, y_max - WALL_WIDTH - 2.0)

    cube_b = room.entities[3]
    cube_b.type = EntityType.CUBE
    cube_b.active = True
    cube_b.pos_x = cube_b_x
    cube_b.pos_y = cube_b_y
    cube_b.pos_z = 0.75 * 1.5
    cube_b.scale = 1.5

    room.button_indices = [0, 1]
    room.is_persistent = False

    # Disable remaining slots
    for i in range(4, MAX_ENTITIES_PER_ROOM):
        room.entities[i].active = False
        room.entities[i].type = EntityType.NONE


def generate_door_position(
    room_idx: int, rng: random.Random
) -> Tuple[float, float, float, float]:
    """
    Generate door position and wall lengths for a room.
    Returns (door_x, door_y, wall_left_len, wall_right_len).
    """
    door_center = rand_between(
        rng, 0.75 * DOOR_WIDTH, WORLD_WIDTH - 0.75 * DOOR_WIDTH
    )
    door_x = door_center - WORLD_WIDTH / 2
    door_y = ROOM_LENGTH * (room_idx + 1) - WALL_WIDTH / 2

    left_len = door_center - 0.5 * DOOR_WIDTH
    right_len = WORLD_WIDTH - door_center - 0.5 * DOOR_WIDTH

    return door_x, door_y, left_len, right_len


def generate_level(rng: random.Random) -> List[RoomConfig]:
    """
    Generate a complete level with 3 rooms.
    Uses a fixed sequence: DoubleButton -> CubeBlocking -> CubeButtons.
    """
    rooms = []

    for room_idx in range(NUM_ROOMS):
        room = RoomConfig(room_idx)

        # Generate door position
        door_x, door_y, left_len, right_len = generate_door_position(
            room_idx, rng
        )
        room.door_x = door_x
        room.door_y = door_y
        room.wall_left_len = left_len
        room.wall_right_len = right_len

        # Select room type (fixed sequence for training)
        if room_idx == 0:
            make_double_button_room(room, rng)
        elif room_idx == 1:
            make_cube_blocking_room(room, rng)
        elif room_idx == 2:
            make_cube_buttons_room(room, rng)

        rooms.append(room)

    return rooms


def generate_random_level(rng: random.Random) -> List[RoomConfig]:
    """
    Generate a level with random room types (for evaluation/variety).
    """
    rooms = []

    for room_idx in range(NUM_ROOMS):
        room = RoomConfig(room_idx)

        door_x, door_y, left_len, right_len = generate_door_position(
            room_idx, rng
        )
        room.door_x = door_x
        room.door_y = door_y
        room.wall_left_len = left_len
        room.wall_right_len = right_len

        room_type = rng.randint(0, RoomType.NUM_TYPES - 1)

        if room_type == RoomType.SINGLE_BUTTON:
            make_single_button_room(room, rng)
        elif room_type == RoomType.DOUBLE_BUTTON:
            make_double_button_room(room, rng)
        elif room_type == RoomType.CUBE_BLOCKING:
            make_cube_blocking_room(room, rng)
        elif room_type == RoomType.CUBE_BUTTONS:
            make_cube_buttons_room(room, rng)

        rooms.append(room)

    return rooms
