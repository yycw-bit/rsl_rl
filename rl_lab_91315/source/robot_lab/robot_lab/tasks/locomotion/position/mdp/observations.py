# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.sensors import ContactSensor, RayCaster


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_names[asset_cfg.joint_ids])
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return joint_pos_rel

def joint_pos_rel_with_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print(asset.data.joint_names[asset_cfg.joint_ids])
    joint_pos_rel = asset.data.joint_pos.clone()
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] =0
    joint_pos_rel = joint_pos_rel[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # indices=[3,7,11,15]
    # joint_pos_rel[:, indices] = 0
    # print(joint_pos_rel[0])
    return joint_pos_rel


def clock_inputs(env: ManagerBasedRLEnv, cycle_time: float, offsets:float, bounds:float, durations:float, phases:float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = torch.remainder(env.episode_length_buf * env.step_dt *3, cycle_time)
    phases = phases + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    offsets = offsets + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    bounds = bounds + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    foot_indices = [phase + phases + offsets + bounds,
                    phase + offsets,
                    phase + bounds,
                    phase + phases]
    foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), cycle_time)

    durations = durations + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    foot_indices=foot_indices.T
    for idxs in foot_indices:
        stance_idxs = torch.remainder(idxs, 1) < durations
        swing_idxs = torch.remainder(idxs, 1) > durations
        idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs]))
    import numpy as np
    clock_inputs = torch.zeros(env.num_envs, 4, dtype=torch.float, device=env.device, requires_grad=False)
    clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
    clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
    clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
    clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])
    return clock_inputs

def desired_contact(env: ManagerBasedRLEnv, cycle_time: float, offsets:float, bounds:float, durations:float, phases:float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = torch.remainder(env.episode_length_buf * env.step_dt *3, cycle_time)
    phases = phases + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    offsets = offsets + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    bounds = bounds + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    foot_indices = [phase + phases + offsets + bounds,
                    phase + offsets,
                    phase + bounds,
                    phase + phases]
    foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), cycle_time)
    durations = durations + torch.zeros(env.num_envs, dtype=torch.float, device=env.device, requires_grad=False)
    foot_indices = foot_indices.T
    for idxs in foot_indices:
        stance_idxs = torch.remainder(idxs, 1) < durations
        swing_idxs = torch.remainder(idxs, 1) > durations
        idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
        idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                0.5 / (1 - durations[swing_idxs]))

    kappa = 0.07
    smoothing_cdf_start = torch.distributions.normal.Normal(0,
                                                            kappa).cdf  # (x) + torch.distributions.normal.Normal(1, kappa).cdf(x)) / 2
    smoothing_multiplier_FL = (smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[0], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(foot_indices[0], 1.0) - 0.5 - 1)))
    smoothing_multiplier_FR = (smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[1], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(foot_indices[1], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RL = (smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[2], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(foot_indices[2], 1.0) - 0.5 - 1)))
    smoothing_multiplier_RR = (smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0)) * (
            1 - smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 0.5)) +
                               smoothing_cdf_start(torch.remainder(foot_indices[3], 1.0) - 1) * (
                                       1 - smoothing_cdf_start(
                                   torch.remainder(foot_indices[3], 1.0) - 0.5 - 1)))
    desired_contact = torch.zeros(env.num_envs, 4, dtype=torch.float, device=env.device, requires_grad=False, )
    desired_contact[:, 0] = smoothing_multiplier_FL
    desired_contact[:, 1] = smoothing_multiplier_FR
    desired_contact[:, 2] = smoothing_multiplier_RL
    desired_contact[:, 3] = smoothing_multiplier_RR
    return desired_contact

# def clock(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
#     if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
#         env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
#     phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    # phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    # return phase_tensor

def true_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Return feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact
    # contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]> 0.0
    return contact.float()

def obs_contact_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Return feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact
    forces_xyz = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2)
    return forces_xyz.float()

def direction_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, :2]

def mode_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return env.command_manager.get_command(command_name)[:, 3].unsqueeze(1)

def remain_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return ((env.max_episode_length - env.episode_length_buf)/env.max_episode_length).unsqueeze(1)

def friction_coefficient(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    print(materials[:, asset_cfg.body_ids, 0])
    print(materials[:, asset_cfg.body_ids, 0].shape)
    return materials[:, asset_cfg.body_ids, 0]