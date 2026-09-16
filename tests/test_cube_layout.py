"""The cube layouts must cover every slot the level recipes can activate."""

import random

import pytest

from escape_room.consts import CUBE_SLOT_LAYOUTS, DEFAULT_CUBE_LAYOUT, EntityType
from escape_room.level_gen import (
    RoomConfig,
    generate_level,
    make_cube_blocking_room,
    make_cube_buttons_room,
    make_double_button_room,
    make_single_button_room,
)
from escape_room.scene import cube_entity_names, cube_slot_pairs, make_scene_entities


def _cube_slots_of(recipe, seeds: int = 64) -> set[int]:
    slots: set[int] = set()
    for seed in range(seeds):
        room = RoomConfig(0)
        recipe(room, random.Random(seed))
        slots.update(
            idx
            for idx, slot in enumerate(room.entities)
            if slot.active and int(slot.type) == int(EntityType.CUBE)
        )
    return slots


def test_fixed_layout_covers_the_generated_room_sequence():
    layout = set(cube_slot_pairs(DEFAULT_CUBE_LAYOUT))
    for seed in range(128):
        for room_idx, room in enumerate(generate_level(random.Random(seed))):
            for slot_idx, slot in enumerate(room.entities):
                if slot.active and int(slot.type) == int(EntityType.CUBE):
                    assert (room_idx, slot_idx) in layout


def test_default_layout_allocates_five_cube_bodies():
    assert len(cube_entity_names(DEFAULT_CUBE_LAYOUT)) == 5
    assert len(cube_entity_names("recipe")) == 9
    assert len(cube_entity_names("legacy")) == 12
    entities = make_scene_entities(DEFAULT_CUBE_LAYOUT)
    assert sum(name.startswith("cube_") for name in entities) == 5


@pytest.mark.parametrize(
    "recipe",
    [
        make_single_button_room,
        make_double_button_room,
        make_cube_blocking_room,
        make_cube_buttons_room,
    ],
)
def test_recipe_layout_covers_every_room_type(recipe):
    """``recipe`` is the layout to use when room types are randomized."""
    allowed = set(CUBE_SLOT_LAYOUTS["recipe"][0])
    assert _cube_slots_of(recipe) <= allowed


def test_unknown_layout_is_rejected_with_the_available_names():
    with pytest.raises(ValueError, match="unknown cube layout"):
        cube_entity_names("does-not-exist")
