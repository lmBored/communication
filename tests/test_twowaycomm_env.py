import math

import pytest
import torch

from escape_room.twowaycomm.action import LEFT, RIGHT
from escape_room.twowaycomm.env import TwoWayCommRlEnv, make_env
from escape_room.twowaycomm.env_cfg import (
    ACTION_DIM,
    CONTROL_DT,
    CRITIC_OBS_DIM,
    EPISODE_STEPS,
    RECEIVER_OBS_DIM,
    SENDER_OBS_DIM,
    twowaycomm_env_cfg,
)
from escape_room.twowaycomm.scene import (
    ARROW_LAYOUTS,
    ARROW_PANEL_HALF_WIDTH,
    ARROW_POSITIONS,
    ARROW_SIGN_NAMES,
    CAMERA_FOV_DEGREES,
    COLOR_RGBA,
    DECISION_CENTER_X,
    DOOR_CENTERS,
    ENTRY_Y,
    NUM_COLORS,
    RECEIVER_CAMERA_NAME,
    RECEIVER_SPAWN,
    SENDER_CAMERA_NAME,
    SENDER_ROOM_CENTER,
    SENDER_ROOM_HALF_X,
    SENDER_ROOM_HALF_Y,
    SENDER_SPAWNS,
    arrow_quaternions,
)


def _bearings(layout: str) -> list[float]:
    """Signed bearing to each arrow from the sender's spawn, facing +y."""
    spawn_x, spawn_y, _ = SENDER_SPAWNS[layout]
    return [
        math.degrees(math.atan2(x - spawn_x, y - spawn_y))
        for x, y, _ in ARROW_POSITIONS[layout]
    ]


def test_twowaycomm_config_publishes_three_private_groups_and_six_actions():
    cfg = twowaycomm_env_cfg(num_envs=4)

    assert tuple(cfg.observations) == ("sender", "receiver", "critic")
    assert all(group.nan_policy == "error" for group in cfg.observations.values())
    assert tuple(cfg.actions) == ("agents",)
    assert cfg.episode_length_s == pytest.approx(EPISODE_STEPS * CONTROL_DT)
    assert cfg.scale_rewards_by_dt
    assert cfg.is_finite_horizon
    # Default reward function is identical to the one-way scenario.
    assert {name: term.weight for name, term in cfg.rewards.items()} == {
        "step": -0.01,
        "correct": 1.0,
        "wrong": -10.0,
        "timeout": -20.0,
    }
    # Passing events overrides the dataclass default, so the stock reset must
    # still be present or entities stop returning to their initial state.
    assert "reset_scene_to_default" in cfg.events
    # The decorator on this term is what makes door colour per-world.
    assert cfg.events["door_color"].mode == "reset"
    assert "geom_rgba" in getattr(cfg.events["door_color"].func, "model_fields", ())


def test_sender_shaping_is_opt_in_and_gated_by_reward_sharing():
    assert "sender_alignment" not in twowaycomm_env_cfg(num_envs=2).rewards

    shared = twowaycomm_env_cfg(num_envs=2, sender_shaping_weight=0.05)
    assert shared.rewards["sender_alignment"].weight == pytest.approx(0.05)

    receiver_only = twowaycomm_env_cfg(
        num_envs=2, reward_sharing="receiver_only", sender_shaping_weight=0.05
    )
    assert "sender_alignment" not in receiver_only.rewards

    with pytest.raises(ValueError):
        twowaycomm_env_cfg(num_envs=2, reward_sharing="nonsense")
    with pytest.raises(ValueError):
        twowaycomm_env_cfg(num_envs=2, arrow_layout="nonsense")


def test_rooms_are_disjoint_and_arrows_fit_inside_the_arrow_room():
    assert SENDER_ROOM_CENTER[0] + SENDER_ROOM_HALF_X < -2.0

    for layout in ARROW_LAYOUTS:
        for x, y, _ in ARROW_POSITIONS[layout]:
            assert abs(x - SENDER_ROOM_CENTER[0]) <= SENDER_ROOM_HALF_X
            assert abs(y - SENDER_ROOM_CENTER[1]) <= SENDER_ROOM_HALF_Y
        # A panel rotated onto a side wall extends along y instead of x, so the
        # in-plane half width has to fit either way.
        for (x, y, _), yaw in zip(ARROW_POSITIONS[layout], _bearings(layout)):
            del yaw
            assert ARROW_PANEL_HALF_WIDTH < SENDER_ROOM_HALF_X


def test_front_layout_shows_every_arrow_and_scattered_shows_one():
    half_fov = CAMERA_FOV_DEGREES * 0.5

    front = _bearings("front")
    assert all(abs(bearing) < half_fov for bearing in front)
    # Panel edges, not just centres, must clear the frame.
    spawn_x, spawn_y, _ = SENDER_SPAWNS["front"]
    for x, y, _ in ARROW_POSITIONS["front"]:
        edge = math.degrees(
            math.atan2(abs(x - spawn_x) + ARROW_PANEL_HALF_WIDTH, y - spawn_y)
        )
        assert edge < half_fov

    scattered = _bearings("scattered")
    visible = [bearing for bearing in scattered if abs(bearing) < half_fov]
    assert len(visible) == 1, "scattered must force the sender to turn"
    separations = [
        abs(scattered[i] - scattered[j]) for i in range(3) for j in range(i + 1, 3)
    ]
    assert min(separations) >= 85.0


def test_doorways_stay_in_view_from_the_receiver_spawn():
    for door_x, door_y in DOOR_CENTERS:
        offset_x = door_x - RECEIVER_SPAWN[0]
        offset_y = door_y - RECEIVER_SPAWN[1]
        assert offset_y > 0
        assert abs(math.atan2(offset_x, offset_y)) < math.radians(
            CAMERA_FOV_DEGREES * 0.5
        )
        assert door_y == ENTRY_Y


def test_reset_exposes_shared_layout_with_asymmetric_content():
    env = make_env(num_envs=6, device="cpu", seed=7)
    try:
        term = env.action_manager.get_term("agents")
        obs, _ = env.reset()

        assert obs["sender"].shape == (6, SENDER_OBS_DIM)
        assert obs["receiver"].shape == (6, RECEIVER_OBS_DIM)
        # The critic is centralized and discarded at inference, so it is
        # privileged: it holds both halves in full, a superset of what either
        # actor branch sees. That is what keeps it usable in pixel mode.
        assert obs["critic"].shape == (6, CRITIC_OBS_DIM)
        assert torch.allclose(obs["critic"][:, :7], obs["sender"][:, :7])
        assert torch.allclose(obs["critic"][:, 14:21], obs["receiver"][:, :7])
        assert torch.equal(obs["critic"][:, 7:10].long(), term.arrow_direction)
        assert torch.equal(obs["critic"][:, 10:13].argmax(-1), term.door_color)

        # The whole point: neither agent can see the other's half.
        assert obs["sender"][:, 10:13].abs().sum() == 0.0
        assert obs["receiver"][:, 7:10].abs().sum() == 0.0
        assert torch.equal(obs["sender"][:, 7:10].long(), term.arrow_direction)
        assert torch.equal(obs["receiver"][:, 10:13].argmax(-1), term.door_color)
        assert torch.all(obs["receiver"][:, 10:13].sum(-1) == 1.0)

        # Balanced draws, per colour column and across door colours.
        for colour in range(NUM_COLORS):
            assert int((term.arrow_direction[:, colour] == LEFT).sum()) == 3
        assert torch.bincount(term.door_color, minlength=NUM_COLORS).tolist() == [2, 2, 2]

        assert env.scene["sender"].find_cameras(SENDER_CAMERA_NAME)[1]
        assert env.scene["receiver"].find_cameras(RECEIVER_CAMERA_NAME)[1]
        for name in ARROW_SIGN_NAMES:
            assert env.scene[name].data.indexing.mocap_id is not None
    finally:
        env.close()


def test_private_information_does_not_leak_across_rooms():
    env = make_env(num_envs=4, device="cpu", seed=3)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        sender_before = term.sender_observation().clone()
        receiver_before = term.receiver_observation().clone()

        term.set_arrow_directions(-term.arrow_direction)
        assert torch.allclose(term.receiver_observation(), receiver_before)
        assert not torch.allclose(term.sender_observation(), sender_before)
        assert torch.equal(term.sender_observation()[:, 7:10].long(), term.arrow_direction)

        sender_after_flip = term.sender_observation().clone()
        term.set_door_color((term.door_color + 1) % NUM_COLORS)
        assert torch.allclose(term.sender_observation(), sender_after_flip)
        assert torch.equal(term.receiver_observation()[:, 10:13].argmax(-1), term.door_color)
    finally:
        env.close()


def test_door_colour_is_written_per_world():
    env = make_env(num_envs=4, device="cpu", seed=11)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        term.set_door_color(torch.tensor([0, 1, 2, 1]))

        rgba = env.sim.model.geom_rgba
        assert rgba.shape[0] == 4
        # A stride-0 leading dimension would silently alias every world.
        assert rgba.stride(0) != 0
        for world, colour in enumerate([0, 1, 2, 1]):
            expected = torch.tensor(COLOR_RGBA[colour][:3])
            for geom_id in term._door_geom_ids.tolist():
                assert torch.allclose(rgba[world, geom_id, :3], expected)
    finally:
        env.close()


@pytest.mark.parametrize("layout", ARROW_LAYOUTS)
def test_arrow_sign_quaternions_encode_direction(layout):
    env = make_env(num_envs=4, device="cpu", seed=5, arrow_layout=layout)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        expected_quats = torch.tensor(arrow_quaternions(layout))

        for arrow in range(NUM_COLORS):
            mocap_id = term._arrow_mocap_ids[arrow]
            written = env.sim.data.mocap_quat[:, mocap_id]
            selection = (term.arrow_direction[:, arrow] == RIGHT).long()
            assert torch.allclose(written, expected_quats[arrow][selection], atol=1e-6)
            # LEFT and RIGHT differ by exactly half a turn about z.
            dot = float((expected_quats[arrow][0] * expected_quats[arrow][1]).sum())
            assert 2.0 * math.acos(min(abs(dot), 1.0)) == pytest.approx(math.pi, abs=1e-6)

        # reset_scene_to_default teleports mocap bodies, so positions must be
        # rewritten every reset.
        positions = torch.tensor(ARROW_POSITIONS[layout])
        for arrow in range(NUM_COLORS):
            written = env.sim.data.mocap_pos[:, term._arrow_mocap_ids[arrow]]
            assert torch.allclose(written, positions[arrow].expand_as(written), atol=1e-6)
    finally:
        env.close()


def test_six_wide_action_drives_both_agents_without_crosstalk():
    env = make_env(num_envs=2, device="cpu", seed=1)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        assert env.action_manager.total_action_dim == ACTION_DIM

        term.process_actions(
            torch.tensor([[1.0, 0.0, 0.5, 0.0, 1.0, -0.5], [0.0, 1.0, -0.5, 1.0, 0.0, 0.5]])
        )
        term.apply_actions()
        sender_velocity = term.sender_velocity()
        receiver_velocity = term.receiver_velocity()

        assert sender_velocity[0, 1] == pytest.approx(term.cfg.max_speed)
        assert sender_velocity[0, 5] == pytest.approx(0.5 * term.cfg.max_yaw_rate)
        assert receiver_velocity[0, 0] == pytest.approx(term.cfg.max_speed)
        assert receiver_velocity[0, 5] == pytest.approx(-0.5 * term.cfg.max_yaw_rate)
        assert sender_velocity[1, 0] == pytest.approx(term.cfg.max_speed)
        assert receiver_velocity[1, 1] == pytest.approx(term.cfg.max_speed)

        term.process_actions(torch.full((2, ACTION_DIM), 5.0))
        assert float(term._processed_flat.max()) == pytest.approx(1.0)

        # Driving the sender must leave the receiver where it is.
        env.reset()
        receiver_before = term.receiver_pose()[:, :2].clone()
        action = torch.zeros(2, ACTION_DIM)
        action[:, 0] = 1.0
        env.step(action)
        assert torch.allclose(term.receiver_pose()[:, :2], receiver_before, atol=1e-6)
    finally:
        env.close()


def test_partial_reset_changes_only_requested_worlds():
    env = make_env(num_envs=4, device="cpu", seed=2, auto_reset=False)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        arrows = term.arrow_direction.clone()
        colors = term.door_color.clone()
        rgba_before = env.sim.model.geom_rgba[:, term._door_geom_ids[0]].clone()
        term.set_receiver_pose(torch.tensor([4.0, 2.0, 0.45]), torch.tensor([2]))
        moved = term.receiver_pose()[2, :3].clone()

        env.reset(env_ids=torch.tensor([1, 3]))

        for keep in (0, 2):
            assert torch.equal(term.arrow_direction[keep], arrows[keep])
            assert int(term.door_color[keep]) == int(colors[keep])
            assert torch.allclose(
                env.sim.model.geom_rgba[keep, term._door_geom_ids[0]], rgba_before[keep]
            )
        assert torch.allclose(term.receiver_pose()[2, :3], moved, atol=1e-6)
        for reset_world in (1, 3):
            assert torch.allclose(
                term.receiver_pose()[reset_world, :3],
                torch.tensor(RECEIVER_SPAWN),
                atol=1e-6,
            )
    finally:
        env.close()


@pytest.mark.parametrize(
    "color, arrows, door_index, reward, correct",
    [
        (1, (RIGHT, LEFT, RIGHT), 0, 0.99, True),
        (1, (RIGHT, LEFT, RIGHT), 1, -10.01, False),
        (2, (LEFT, LEFT, RIGHT), 1, 0.99, True),
        # Two distractor arrows say RIGHT and the answer is still LEFT.
        (0, (LEFT, RIGHT, RIGHT), 0, 0.99, True),
    ],
)
def test_entry_reward_follows_the_matching_colour_arrow(
    color, arrows, door_index, reward, correct
):
    env = make_env(num_envs=1, device="cpu", seed=0, auto_reset=False)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        term.set_door_color(torch.tensor([color]))
        term.set_arrow_directions(torch.tensor(arrows))
        assert int(term.target_direction()[0]) == arrows[color]
        term.set_receiver_pose(
            torch.tensor([DOOR_CENTERS[door_index][0], ENTRY_Y + 0.75, 0.45])
        )
        env.sim.forward()

        _, rewards, terminated, truncated, _ = env.step(torch.zeros(1, ACTION_DIM))

        assert float(rewards[0]) == pytest.approx(reward, abs=1e-5)
        assert bool(term.correct_entry[0]) is correct
        assert bool(term.wrong_entry[0]) is (not correct)
        assert bool(terminated[0])
        assert not bool(truncated[0])
    finally:
        env.close()


def test_timeout_penalty_and_sender_motion_never_terminates():
    env = make_env(num_envs=1, device="cpu", seed=0, auto_reset=False)
    try:
        term = env.action_manager.get_term("agents")
        env.reset()
        env.episode_length_buf[:] = env.max_episode_length - 1

        _, rewards, terminated, truncated, _ = env.step(torch.zeros(1, ACTION_DIM))

        assert float(rewards[0]) == pytest.approx(-20.01, abs=1e-5)
        assert bool(truncated[0])
        assert not bool(terminated[0])
        assert bool(term.timeout[0])
        assert not bool(term.correct_entry[0]) and not bool(term.wrong_entry[0])

        env.reset()
        action = torch.zeros(1, ACTION_DIM)
        action[:, 0] = 1.0
        for _ in range(20):
            _, _, terminated, _, _ = env.step(action)
            assert not bool(terminated[0])
        # Also a contact-budget smoke test: too small an nconmax drops contacts
        # and the sender tunnels out of its sealed room.
        sender = term.sender_pose()[0]
        assert abs(float(sender[0]) - SENDER_ROOM_CENTER[0]) < SENDER_ROOM_HALF_X + 0.5
        assert abs(float(sender[1]) - SENDER_ROOM_CENTER[1]) < SENDER_ROOM_HALF_Y + 0.5
        assert float(sender[2]) > 0.2
    finally:
        env.close()


def test_env_is_registered_and_constructs():
    env = make_env(num_envs=1, device="cpu", seed=0, play=True)
    try:
        assert isinstance(env, TwoWayCommRlEnv)
        assert env.num_envs == 1
    finally:
        env.close()


def test_pixel_mode_blanks_the_private_slots_and_publishes_images():
    env = make_env(
        num_envs=4, device="cpu", seed=13, obs_mode="pixel",
        camera_width=32, camera_height=32,
        sender_camera_width=32, sender_camera_height=32,
    )
    try:
        obs, _ = env.reset()

        assert obs["sender_image"].shape == (4, 3, 32, 32)
        assert obs["receiver_image"].shape == (4, 3, 32, 32)
        assert float(obs["sender_image"].max()) <= 1.0
        # Both private blocks are blank: the only route to them is the camera.
        assert obs["sender"][:, 7:10].abs().sum() == 0.0
        assert obs["receiver"][:, 10:13].abs().sum() == 0.0
        # The centralized critic keeps the privileged halves regardless.
        assert obs["critic"][:, 7:10].abs().sum() > 0.0
        assert obs["critic"][:, 10:13].abs().sum() > 0.0
        assert obs["critic"].shape == (4, CRITIC_OBS_DIM)
    finally:
        env.close()


def test_pixel_mode_renders_the_door_colour_of_each_world():
    """The per-world geom_rgba write has to survive all the way to the pixels."""
    env = make_env(
        num_envs=6, device="cpu", seed=13, obs_mode="pixel",
        camera_width=32, camera_height=32,
        sender_camera_width=32, sender_camera_height=32,
    )
    try:
        term = env.action_manager.get_term("agents")
        obs, _ = env.reset()
        for world in range(6):
            dominant = int(obs["receiver_image"][world].mean(dim=(1, 2)).argmax())
            assert dominant == int(term.door_color[world])
    finally:
        env.close()


def test_vector_mode_registers_no_camera_sensors():
    # Registering any camera makes sim.sense() render every world every step.
    assert not twowaycomm_env_cfg(num_envs=2).scene.sensors
    assert len(twowaycomm_env_cfg(num_envs=2, obs_mode="pixel").scene.sensors) == 2


def test_sender_camera_defaults_to_a_higher_resolution_than_the_receiver():
    """The sender must resolve an arrow; the receiver only a flat colour.

    At 32x32 the arrow glyph is under two pixels wide and its direction is
    simply not in the image, so a shared low resolution makes the task
    unsolvable for the sender no matter how long it trains.
    """
    sensors = {
        sensor.name: sensor
        for sensor in twowaycomm_env_cfg(num_envs=2, obs_mode="pixel").scene.sensors
    }
    sender = sensors["sender_camera"]
    receiver = sensors["receiver_camera"]
    assert sender.width > receiver.width and sender.height > receiver.height
    assert sender.width >= 128
    # mujoco_warp requires these to agree across cameras; resolution may differ.
    for field in ("use_textures", "use_shadows", "enabled_geom_groups"):
        assert getattr(sender, field) == getattr(receiver, field)
