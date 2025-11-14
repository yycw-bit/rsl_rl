# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_error_switch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term("base_velocity")
    # print("wheel_asset_cfg.joint_ids ",asset_cfg.joint_ids)
    # print("wheel_asset_cfg.joint_names ",asset_cfg.joint_names)
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - cmd.default_joint_pos[:, asset_cfg.joint_ids]
    gravity_error = torch.square(asset.data.projected_gravity_b[:, 0] + 1)
    return (torch.sum(torch.square(angle), dim=1) *( -9*env.command_manager.get_command("base_velocity")[:, 3] +10)
            + ( 5 * gravity_error) * env.command_manager.get_command("base_velocity")[:, 3])

def joint_error_(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term("base_velocity")
    # print("wheel_asset_cfg.joint_ids ",asset_cfg.joint_ids)
    # print("wheel_asset_cfg.joint_names ",asset_cfg.joint_names)
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - cmd.default_joint_pos[:, asset_cfg.joint_ids]
    gravity_error = torch.square(asset.data.projected_gravity_b[:, 0] + 1)
    return torch.sum(torch.square(angle), dim=1) + ( 5 * gravity_error) * env.command_manager.get_command("base_velocity")[:, 3]



def standup_switch_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_error_x = torch.square(asset.data.projected_gravity_b[:, 0] + 1)
    gravity_error_z = torch.square(asset.data.projected_gravity_b[:, 2] + 1)

    return torch.exp(-gravity_error_x / std**2) * env.command_manager.get_command("base_velocity")[:, 3] \
         + torch.exp(-gravity_error_z / std**2) * (1- env.command_manager.get_command("base_velocity")[:, 3])


def standup_switch_mix(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_error_x = torch.square(asset.data.projected_gravity_b[:, 0] + 1)
    gravity_error_z = torch.square(asset.data.projected_gravity_b[:, 2] + 1)

    return -gravity_error_x * env.command_manager.get_command("base_velocity")[:, 3] \
         + torch.exp(-gravity_error_z / std**2) * (1- env.command_manager.get_command("base_velocity")[:, 3])

def standup_switch(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_error_x = torch.square(asset.data.projected_gravity_b[:, 0] + 1)
    gravity_error_z = torch.square(asset.data.projected_gravity_b[:, 2] + 1)

    return gravity_error_x * env.command_manager.get_command("base_velocity")[:, 3] \
         + gravity_error_z  * (1- env.command_manager.get_command("base_velocity")[:, 3])

def lin_vel_vertical_switch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0]) * env.command_manager.get_command("base_velocity")[:, 3] \
         + torch.square(asset.data.root_lin_vel_b[:, 2]) * (1- env.command_manager.get_command("base_velocity")[:, 3])


def ang_vel_horizontal_switch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, 1:]), dim=1) * env.command_manager.get_command("base_velocity")[:, 3] \
         + torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1) * (1- env.command_manager.get_command("base_velocity")[:, 3])


def flat_orientation_switch_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, 1:]), dim=1) * env.command_manager.get_command("base_velocity")[:, 3] \
        + 10*torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) * (1 - env.command_manager.get_command("base_velocity")[:, 3])


def track_lin_vel_switch_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, 1:]), dim=1) * env.command_manager.get_command("base_velocity")[:, 3] \
        + torch.sum(torch.square(env.command_manager.get_command(command_name)[:, [1,0]] - asset.data.root_lin_vel_b[:, :2]), dim=1) * (1 - env.command_manager.get_command("base_velocity")[:, 3])
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_switch_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 0]) * env.command_manager.get_command("base_velocity")[:, 3] \
        + torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2]) * (1 - env.command_manager.get_command("base_velocity")[:, 3])
    return torch.exp(-ang_vel_error / std**2)


def straight_shoulder_joints_switch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.norm(asset.data.joint_pos[:, asset_cfg.joint_ids], dim=1) * env.command_manager.get_command("base_velocity")[:, 3]


def straight_knee_joints_switch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.exp(-torch.norm(asset.data.joint_pos[:, asset_cfg.joint_ids], dim=1)) * env.command_manager.get_command("base_velocity")[:, 3]


def still_hand_joints_switch(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * env.command_manager.get_command("base_velocity")[:, 3]


def base_height_switch(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # if env.command_manager.get_command("base_velocity")[:, 3] == 0:
    #     if sensor_cfg is not None:
    #         sensor: RayCaster = env.scene[sensor_cfg.name]
    #         # Adjust the target height using the sensor data
    #         ray_hits = sensor.data.ray_hits_w[..., 2]
    #         if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
    #             adjusted_target_height = asset.data.root_link_pos_w[:, 2]
    #         else:
    #             adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    #     else:
    #         # Use the provided target height directly for flat terrain
    #         adjusted_target_height = target_height
    #     # Compute the L2 squared penalty
    #     return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height) * 0.5
    # else:
    #     if sensor_cfg is not None:
    #         sensor: RayCaster = env.scene[sensor_cfg.name]
    #         # Adjust the target height using the sensor data
    #         ray_hits = sensor.data.ray_hits_w[..., 2]
    #         if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
    #             return asset.data.root_pos_w[:, 2]
    #         else:
    #             return asset.data.root_pos_w[:, 2] - torch.mean(ray_hits, dim=1)
    #     else:
    #         # Use the provided target height directly for flat terrain
    #         return asset.data.root_pos_w[:, 2]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            return asset.data.root_pos_w[:, 2] *  env.command_manager.get_command("base_velocity")[:, 3]
        else:
            return (asset.data.root_pos_w[:, 2] - torch.mean(ray_hits, dim=1)) * env.command_manager.get_command("base_velocity")[:, 3]
    else:
        # Use the provided target height directly for flat terrain
        return asset.data.root_pos_w[:, 2] * env.command_manager.get_command("base_velocity")[:, 3]


def hand_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    return torch.sum(is_contact, dim=1) * env.command_manager.get_command("base_velocity")[:, 3]


def feet_contact_switch(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    return reward * (1-env.command_manager.get_command("base_velocity")[:, 3])


def wheel_stand_still_without_cmd_switch(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    diff_velocity = asset.data.joint_vel[:, wheel_asset_cfg.joint_ids]
    return (torch.sum(torch.abs(diff_angle), dim=1) + 0.2*torch.sum(torch.abs(diff_velocity), dim=1)) * (1-env.command_manager.get_command("base_velocity")[:, 3])

def wheel_stand_still_switch(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    diff_velocity = asset.data.joint_vel[:, wheel_asset_cfg.joint_ids]
    command = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) < 0.1
    return torch.sum(torch.abs(diff_velocity), dim=1) * (1-env.command_manager.get_command("base_velocity")[:, 3]) * command





