"""
Game systems layer for the escape room environment.

Ports all gameplay logic from src/sim.cpp into a Python module that
operates on MuJoCo model/data through mjlab. This module owns the puzzle
FSM, grab welds, and slot-based reset, while mjlab managers stay thin
adapters.
"""

import math
from typing import Optional, Tuple

import numpy as np
import torch

from escape_room.consts import (
    NUM_ROOMS,
    NUM_AGENTS,
    MAX_ENTITIES_PER_ROOM,
    WORLD_WIDTH,
    WORLD_LENGTH,
    ROOM_LENGTH,
    WALL_WIDTH,
    BUTTON_WIDTH,
    DOOR_WIDTH,
    AGENT_RADIUS,
    EPISODE_LEN,
    DELTA_T,
    NUM_PHYSICS_SUBSTEPS,
    NUM_LIDAR_SAMPLES,
    LIDAR_MAX_RANGE,
    DOOR_SPEED,
    DOOR_CLOSED_Z,
    DOOR_OPEN_Z,
    REWARD_PER_DIST,
    SLACK_REWARD,
    PARTNER_BONUS_MULT,
    PARTNER_CLOSE_THRESHOLD,
    MOVE_MAX_FORCE,
    TURN_MAX_TORQUE,
    GRAB_RAY_LENGTH,
    GRAB_OFFSET_FWD,
    GRAB_OFFSET_UP,
    ACTION_DIM_PER_AGENT,
    TOTAL_ACTION_DIM,
    EntityType,
    dist_obs,
    angle_obs,
)
from escape_room.level_gen import (
    generate_level,
    RoomConfig,
    EntitySlot,
)


class EscapeRoomState:
    """
    Owns per-environment game state and puzzle logic.

    This class maintains GPU tensors for all game state (progress, buttons,
    doors, grab, entity types) and provides methods that correspond to the
    Madrona ECS systems in sim.cpp.
    """

    def __init__(self, num_envs: int, device: str = "cuda:0"):
        self.num_envs = num_envs
        self.device = device

        # Per-env step counter
        self.steps_remaining = torch.full(
            (num_envs, NUM_AGENTS), EPISODE_LEN, device=device, dtype=torch.long
        )

        # Progress tracking (max Y achieved per agent)
        self.progress_max_y = torch.zeros(
            (num_envs, NUM_AGENTS), device=device, dtype=torch.float32
        )

        # Per-env rewards
        self.rewards = torch.zeros(
            (num_envs, NUM_AGENTS), device=device, dtype=torch.float32
        )

        # Done flags
        self.dones = torch.zeros(
            (num_envs, NUM_AGENTS), device=device, dtype=torch.bool
        )

        # Door state per room
        self.door_open = torch.zeros(
            (num_envs, NUM_ROOMS), device=device, dtype=torch.bool
        )
        self.door_persistent = torch.zeros(
            (num_envs, NUM_ROOMS), device=device, dtype=torch.bool
        )
        self.door_z = torch.zeros(
            (num_envs, NUM_ROOMS), device=device, dtype=torch.float32
        )

        # Button state per room per entity slot
        self.button_pressed = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=device,
            dtype=torch.bool,
        )

        # Entity types per room per slot
        self.entity_type = torch.full(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            EntityType.NONE,
            device=device,
            dtype=torch.long,
        )

        # Entity active flags
        self.entity_active = torch.zeros(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            device=device,
            dtype=torch.bool,
        )

        # Button indices per room (which slots are buttons linked to door)
        self.door_button_idx = torch.full(
            (num_envs, NUM_ROOMS, MAX_ENTITIES_PER_ROOM),
            -1,
            device=device,
            dtype=torch.long,
        )
        self.door_num_buttons = torch.zeros(
            (num_envs, NUM_ROOMS), device=device, dtype=torch.long
        )

        # Grab state per agent
        self.grab_target = torch.full(
            (num_envs, NUM_AGENTS), -1, device=device, dtype=torch.long
        )
        self.is_grabbing = torch.zeros(
            (num_envs, NUM_AGENTS), device=device, dtype=torch.bool
        )

        # Level configs (stored as Python objects for now, will be tensorized)
        self.level_configs: Optional[list] = None

        # RNG for level generation
        self.rng = np.random.default_rng(42)

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset all environments with optional seed."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reset step counters
        self.steps_remaining.fill_(EPISODE_LEN)
        self.progress_max_y.fill_(0.0)
        self.rewards.fill_(0.0)
        self.dones.fill_(False)
        self.door_open.fill_(False)
        self.door_persistent.fill_(False)
        self.door_z.fill_(DOOR_CLOSED_Z)
        self.button_pressed.fill_(False)
        self.entity_type.fill_(EntityType.NONE)
        self.entity_active.fill_(False)
        self.door_button_idx.fill_(-1)
        self.door_num_buttons.fill_(0)
        self.grab_target.fill_(-1)
        self.is_grabbing.fill_(False)

        # Generate new level configs
        self._generate_levels()

    def _generate_levels(self) -> None:
        """Generate procedural levels for all environments."""
        self.level_configs = []

        for env_idx in range(self.num_envs):
            rng = np.random.default_rng(
                self.rng.integers(0, 2**31)
            )
            rooms = generate_level(rng)
            self.level_configs.append(rooms)

            # Update entity state tensors
            for room_idx, room in enumerate(rooms):
                self.door_persistent[env_idx, room_idx] = room.is_persistent
                self.door_num_buttons[env_idx, room_idx] = len(
                    room.button_indices
                )

                for btn_idx, slot_idx in enumerate(room.button_indices):
                    self.door_button_idx[env_idx, room_idx, slot_idx] = btn_idx

                for slot_idx, slot in enumerate(room.entities):
                    self.entity_type[env_idx, room_idx, slot_idx] = slot.type
                    self.entity_active[env_idx, room_idx, slot_idx] = slot.active

    def apply_movement(
        self, actions: torch.Tensor, mujoco_model, mujoco_data
    ) -> None:
        """
        Apply continuous movement actions to agents.

        Actions shape: [num_envs, num_agents, action_dim]
        action_dim: (move_x, move_y, yaw_rate, grab)
        """
        # Map actions to forces/torques
        for agent_idx in range(NUM_AGENTS):
            move_x = actions[:, agent_idx, 0]  # [num_envs]
            move_y = actions[:, agent_idx, 1]
            yaw_rate = actions[:, agent_idx, 2]

            # Scale to force/torque ranges
            force_x = move_x * MOVE_MAX_FORCE
            force_y = move_y * MOVE_MAX_FORCE
            torque_z = yaw_rate * TURN_MAX_TORQUE

            # Apply to MuJoCo actuators
            # Actuator indices: agent_idx * 3 + [0, 1, 2]
            base_idx = agent_idx * 3
            mujoco_data.ctrl[base_idx] = force_x
            mujoco_data.ctrl[base_idx + 1] = force_y
            mujoco_data.ctrl[base_idx + 2] = torque_z

    def dampen_agents(self, mujoco_data) -> None:
        """
        Zero residual agent velocity for controllability.
        Port of agentZeroVelSystem.
        """
        # Agent body IDs need to be looked up
        # For now, this is a placeholder - will be filled with actual body IDs
        pass

    def apply_grab(
        self, actions: torch.Tensor, mujoco_model, mujoco_data
    ) -> None:
        """
        Handle grab toggle action.
        Port of grabSystem.
        """
        grab_action = actions[:, :, 3]  # [num_envs, num_agents]

        for agent_idx in range(NUM_AGENTS):
            grab = grab_action[:, agent_idx]

            for env_idx in range(self.num_envs):
                if grab[env_idx] > 0.5:  # Threshold for grab
                    if self.is_grabbing[env_idx, agent_idx]:
                        # Release
                        self.is_grabbing[env_idx, agent_idx] = False
                        self.grab_target[env_idx, agent_idx] = -1
                    else:
                        # Attempt grab - raycast forward
                        # This will be implemented with MuJoCo raycast
                        pass

    def update_buttons(self, mujoco_data) -> None:
        """
        Check if entities are standing on buttons.
        Port of buttonSystem.
        """
        # Check AABB overlap between agents/cubes and button positions
        # This will use MuJoCo contact data or position checks
        pass

    def update_doors(self) -> None:
        """
        Check if all linked buttons are pressed and open doors.
        Port of doorOpenSystem.
        """
        for room_idx in range(NUM_ROOMS):
            num_buttons = self.door_num_buttons[:, room_idx]

            for env_idx in range(self.num_envs):
                n = num_buttons[env_idx].item()
                if n == 0:
                    continue

                all_pressed = True
                for btn_idx in range(n):
                    slot_idx = self.door_button_idx[
                        env_idx, room_idx, btn_idx
                    ].item()
                    if slot_idx < 0:
                        continue
                    if not self.button_pressed[env_idx, room_idx, slot_idx]:
                        all_pressed = False
                        break

                if all_pressed:
                    self.door_open[env_idx, room_idx] = True
                elif not self.door_persistent[env_idx, room_idx]:
                    self.door_open[env_idx, room_idx] = False

    def animate_doors(self) -> None:
        """
        Animate door positions based on open state.
        Port of setDoorPositionSystem.
        """
        for room_idx in range(NUM_ROOMS):
            open_mask = self.door_open[:, room_idx]
            closed_mask = ~open_mask

            # Doors that should open: move down
            dz_open = -DOOR_SPEED * DELTA_T
            self.door_z[open_mask, room_idx] += dz_open
            self.door_z[open_mask, room_idx] = torch.clamp(
                self.door_z[open_mask, room_idx], DOOR_OPEN_Z, DOOR_CLOSED_Z
            )

            # Doors that should close: move up
            dz_close = DOOR_SPEED * DELTA_T
            self.door_z[closed_mask, room_idx] += dz_close
            self.door_z[closed_mask, room_idx] = torch.clamp(
                self.door_z[closed_mask, room_idx], DOOR_OPEN_Z, DOOR_CLOSED_Z
            )

    def compute_rewards(self, mujoco_data) -> None:
        """
        Compute progress-based rewards.
        Port of rewardSystem and bonusRewardSystem.
        """
        # Get agent positions
        # This will read from MuJoCo data

        # For now, placeholder reward computation
        for agent_idx in range(NUM_AGENTS):
            # Slack reward (default)
            self.rewards[:, agent_idx] = SLACK_REWARD

            # Progress reward (if agent moved forward)
            # This will be computed from actual positions

            # Partner bonus
            for env_idx in range(self.num_envs):
                if self.rewards[env_idx, agent_idx] > 0:
                    other_idx = 1 - agent_idx
                    y_diff = abs(
                        self.progress_max_y[env_idx, agent_idx]
                        - self.progress_max_y[env_idx, other_idx]
                    )
                    if y_diff <= PARTNER_CLOSE_THRESHOLD:
                        self.rewards[env_idx, agent_idx] *= PARTNER_BONUS_MULT

    def update_steps(self) -> None:
        """
        Decrement step counters and set done flags.
        Port of stepTrackerSystem.
        """
        self.steps_remaining -= 1
        self.dones = (self.steps_remaining <= 0)

    def collect_observations(
        self, mujoco_data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect all observations for all agents.
        Port of collectObservationsSystem and lidarSystem.

        Returns:
            self_obs: [num_envs, num_agents, 8]
            partner_obs: [num_envs, num_agents, 3]
            room_ent_obs: [num_envs, num_agents, max_entities, 3]
            door_obs: [num_envs, num_agents, 3]
            lidar: [num_envs, num_agents, num_lidar, 2]
            steps_remaining: [num_envs, num_agents, 1]
            agent_id: [num_envs, num_agents, 1]
        """
        batch_size = self.num_envs * NUM_AGENTS

        # Self observation: roomX, roomY, globalX, globalY, globalZ, maxY, theta, isGrabbing
        self_obs = torch.zeros(
            (self.num_envs, NUM_AGENTS, 8), device=self.device
        )

        # Partner observation: polar r, theta, isGrabbing
        partner_obs = torch.zeros(
            (self.num_envs, NUM_AGENTS, 3), device=self.device
        )

        # Room entity observation: polar r, theta, encoded_type
        room_ent_obs = torch.zeros(
            (self.num_envs, NUM_AGENTS, MAX_ENTITIES_PER_ROOM, 3),
            device=self.device,
        )

        # Door observation: polar r, theta, is_open
        door_obs = torch.zeros(
            (self.num_envs, NUM_AGENTS, 3), device=self.device
        )

        # Lidar: depth, encoded_type
        lidar = torch.zeros(
            (self.num_envs, NUM_AGENTS, NUM_LIDAR_SAMPLES, 2),
            device=self.device,
        )

        # Steps remaining
        steps_rem = self.steps_remaining.float().unsqueeze(-1)

        # Agent ID
        agent_id = torch.arange(
            NUM_AGENTS, device=self.device, dtype=torch.float32
        ).unsqueeze(0).expand(self.num_envs, -1).unsqueeze(-1)
        if NUM_AGENTS > 1:
            agent_id = agent_id / (NUM_AGENTS - 1)

        return (
            self_obs,
            partner_obs,
            room_ent_obs,
            door_obs,
            lidar,
            steps_rem,
            agent_id,
        )

    def step(self, actions: torch.Tensor, mujoco_model, mujoco_data) -> dict:
        """
        Execute one game step.

        Order of operations (matching sim.cpp task graph):
        1. Apply movement actions
        2. Animate doors
        3. Apply grab
        4. Physics step (handled by mjlab)
        5. Dampen agents
        6. Update buttons
        7. Update doors
        8. Compute rewards
        9. Update steps
        10. Collect observations
        """
        # 1. Apply movement
        self.apply_movement(actions, mujoco_model, mujoco_data)

        # 2. Animate doors
        self.animate_doors()

        # 3. Apply grab
        self.apply_grab(actions, mujoco_model, mujoco_data)

        # 4. Physics step - handled by mjlab's step()

        # 5. Dampen agents
        self.dampen_agents(mujoco_data)

        # 6. Update buttons
        self.update_buttons(mujoco_data)

        # 7. Update doors
        self.update_doors()

        # 8. Compute rewards
        self.compute_rewards(mujoco_data)

        # 9. Update steps
        self.update_steps()

        # 10. Collect observations
        obs = self.collect_observations(mujoco_data)

        return {
            "observations": obs,
            "rewards": self.rewards,
            "dones": self.dones,
        }
