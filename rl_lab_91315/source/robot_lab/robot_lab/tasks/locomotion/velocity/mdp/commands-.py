# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import MISSING
import torch
from typing import TYPE_CHECKING, Sequence

from jinja2.utils import missing

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.assets.articulation import Articulation
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
from .visual import GOAL_MARKER_CFG, POS_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()

@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """

class UniformVelocityCommandStand(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # -- command: x vel, y vel, yaw vel, heading stand
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.standing_up_env_ids = []
        self.normal_env_ids = []

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        # self.metrics["error_vel_xy"][self.normal_env_ids] += (
        #     torch.norm(self.vel_command_b[self.normal_env_ids, :2] - self.robot.data.root_lin_vel_b[self.normal_env_ids, :2], dim=-1) / max_command_step
        # )
        # self.metrics["error_vel_z_yaw"][self.normal_env_ids] += (
        #     torch.abs(self.vel_command_b[self.normal_env_ids, 2] - self.robot.data.root_ang_vel_b[self.normal_env_ids, 2]) / max_command_step
        # )
        #
        # self.metrics["error_vel_yz"][self.standing_up_env_ids] += (
        #         torch.norm(self.vel_command_b[self.standing_up_env_ids, :2] - self.robot.data.root_lin_vel_b[self.standing_up_env_ids, 1:], dim=-1) / max_command_step
        # )
        # self.metrics["error_vel_x_yaw"][self.standing_up_env_ids] += (
        #         torch.abs(self.vel_command_b[self.standing_up_env_ids, 2] - self.robot.data.root_ang_vel_b[self.standing_up_env_ids, 0]) / max_command_step
        # )
        self.metrics["error_vel_xy"] += (
                torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
                torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # -- stand sign 0 for no 1 for yes
        self.vel_command_b[env_ids, 3] = (r.uniform_(0, 1) > 0.5).float()
        self.standing_up_env_ids = self.vel_command_b[:, 3].nonzero(as_tuple=False).flatten()
        self.normal_env_ids = (self.vel_command_b[:, 3] == 0).nonzero(as_tuple=False).flatten()
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

@configclass
class UniformVelocityCommandStandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandStand


class UniformVelocityCommandSwitch(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityCommandSwitchCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # -- command: x vel, y vel, yaw vel, heading stand
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.normal_env_ids = []
        self.standing_up_env_ids = []
        self.normal_default_joint_pos = torch.tensor(cfg.normal_default_joint_pos, device=self.device, dtype=torch.float32)
        self.stand_default_joint_pos = torch.tensor(cfg.stand_default_joint_pos, device=self.device, dtype=torch.float32)
        self.default_joint_pos = torch.zeros(self.num_envs, self.normal_default_joint_pos.shape[0], device=self.device)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        # self.metrics["error_vel_xy"][self.normal_env_ids] += (
        #     torch.norm(self.vel_command_b[self.normal_env_ids, :2] - self.robot.data.root_lin_vel_b[self.normal_env_ids, :2], dim=-1) / max_command_step
        # )
        # self.metrics["error_vel_z_yaw"][self.normal_env_ids] += (
        #     torch.abs(self.vel_command_b[self.normal_env_ids, 2] - self.robot.data.root_ang_vel_b[self.normal_env_ids, 2]) / max_command_step
        # )
        #
        # self.metrics["error_vel_yz"][self.standing_up_env_ids] += (
        #         torch.norm(self.vel_command_b[self.standing_up_env_ids, :2] - self.robot.data.root_lin_vel_b[self.standing_up_env_ids, 1:], dim=-1) / max_command_step
        # )
        # self.metrics["error_vel_x_yaw"][self.standing_up_env_ids] += (
        #         torch.abs(self.vel_command_b[self.standing_up_env_ids, 2] - self.robot.data.root_ang_vel_b[self.standing_up_env_ids, 0]) / max_command_step
        # )
        self.metrics["error_vel_xy"] += (
                torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
                torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # -- stand sign 0 for no 1 for yes
        self.vel_command_b[env_ids, 3] = (r.uniform_(0, 1) > 0.5).float()

        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        self.standing_up_env_ids = self.vel_command_b[:, 3].nonzero(as_tuple=False).flatten()
        self.normal_env_ids = (self.vel_command_b[:, 3] == 0).nonzero(as_tuple=False).flatten()
        self.default_joint_pos[self.standing_up_env_ids, :] = self.stand_default_joint_pos
        self.default_joint_pos[self.normal_env_ids, :] = self.normal_default_joint_pos

@configclass
class UniformVelocityCommandSwitchCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandSwitch
    normal_default_joint_pos: list = MISSING
    stand_default_joint_pos: list = MISSING


class UniformVelocityAndPositionCommand(UniformVelocityCommand):
    def __init__(self, cfg: UniformVelocityAndPositionCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        # -- command: x vel, y vel, yaw vel, heading stand
        self.vel_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.vel_command_w = torch.zeros_like(self.vel_command_b, device=self.device)

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # transform command from base frame to simulation world frame
        _, self.vel_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            None,
            self.vel_command_b[:, 3:],
        )
        # compute the error
        _, rot_error = compute_pose_error(
            self.robot.data.root_pos_w,
            self.vel_command_w[:, 3:],
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
        )
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        euler_angles = torch.zeros_like(self.vel_command_b[env_ids, :3])
        # -- ang pos roll - rotation around x
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        # -- ang pos pitch - rotation around y
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.vel_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

@configclass
class UniformVelocityAndPositionCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityAndPositionCommand
    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """
        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""


