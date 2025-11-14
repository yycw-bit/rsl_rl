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


def track_position_xy(
    env: ManagerBasedRLEnv, command_name: str, time_remain_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    pos_error = torch.sum(
        torch.square(cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2]),
        dim=1,
    )
    time_remain = (env.max_episode_length - env.episode_length_buf) * env.step_dt
    reward = (1/(pos_error+1)) / time_remain_threshold
    return reward * (time_remain < time_remain_threshold)


def track_position_xy_2(
    env: ManagerBasedRLEnv, command_name: str, time_remain_threshold: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    pos_error = torch.norm(cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    time_remain = (env.max_episode_length - env.episode_length_buf) * env.step_dt
    reward = 1 - 0.5 * pos_error
    return reward * (time_remain < time_remain_threshold)


def face_goal(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    dir_vec = cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2]  # [num_envs, 2]
    goal_yaw = torch.atan2(dir_vec[:, 1], dir_vec[:, 0])
    yaw_error = torch.abs(asset.data.heading_w - goal_yaw)
    pos_error = torch.norm(cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1,) > 0.5
    reward = -yaw_error * pos_error
    # print("goal_yaw ",goal_yaw )
    # print("now: ", asset.data.heading_w)
    return reward


def position_bias_xy(
    env: ManagerBasedRLEnv, command_name: str, reward_threshold: float, time_remain_threshold: float, eps: float = 1e-6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    x = asset.data.root_pos_w[:, :2]  # [num_envs, 2]
    dx = asset.data.root_lin_vel_w[:, :2]  # [num_envs, 2]
    r_goal = cmd.pose_command_w[:, :2]  # [num_envs, 2]
    dir_vec = r_goal - x  # [num_envs, 2]
    dir_norm = torch.norm(dir_vec, dim=1)  # [num_envs]
    dx_norm = torch.norm(dx, dim=1)  # [num_envs]
    reward = torch.sum(dx * dir_vec, dim=1) / (dx_norm * dir_norm )  # + eps

    # # compute the error
    # pos_error = torch.sum(torch.square(dir_vec), dim=1)
    # time_remain = (env.max_episode_length - env.episode_length_buf) * env.step_dt
    # track_reward = (((1 / (pos_error + 1))/ time_remain_threshold) * (time_remain < time_remain_threshold)) < reward_threshold
    pos_error = torch.norm(cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1)
    time_remain = (env.max_episode_length - env.episode_length_buf) * env.step_dt
    track_reward = ( (1 - 0.5 * pos_error) * (time_remain < time_remain_threshold) ) < reward_threshold
    return reward * track_reward


def position_bias_xyz(
    env: ManagerBasedRLEnv, command_name: str, reward_threshold: float, time_remain_threshold: float, eps: float = 1e-6, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    x = asset.data.root_pos_w # [num_envs, 3]
    dx = asset.data.root_lin_vel_w  # [num_envs, 3]
    r_goal = cmd.pose_command_w[:, :3]  # [num_envs, 3]
    dir_vec = r_goal - x  # [num_envs, 3]
    dir_norm = torch.norm(dir_vec, dim=1)  # [num_envs]
    dx_norm = torch.norm(dx, dim=1)  # [num_envs]
    reward = torch.sum(dx * dir_vec, dim=1) / (dx_norm * dir_norm) # + eps

    # compute the error
    pos_error = torch.sum(torch.square(dir_vec[:, :2]), dim=1)
    time_remain = (env.max_episode_length - env.episode_length_buf) * env.step_dt
    track_reward = (((1 / (pos_error + 1))/ time_remain_threshold) * (time_remain < time_remain_threshold)) < reward_threshold
    return reward * track_reward


def do_not_wait(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute the error
    pos_error = torch.norm(cmd.pose_command_w[:, :3] - asset.data.root_pos_w, dim=1)  # [num_envs]
    lin_vel = torch.norm(asset.data.root_lin_vel_w, dim=1)  # [num_envs]
    reward = -1 * (pos_error > 0.5) * (lin_vel < 0.1)
    return reward


def wheel_stand_still_at_target(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one when no command."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_term(command_name)
    # compute out of limits constraints
    pos_error = torch.norm(cmd.pose_command_w[:, :2] - asset.data.root_pos_w[:, :2], dim=1) < 0.25
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (
        torch.sum(torch.abs(diff_angle), dim=1) * pos_error  # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    )


def body_ang_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the angular acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.norm(asset.data.body_ang_acc_w[:, asset_cfg.body_ids, :2], dim=-1), dim=1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward

def joint_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # print("wheel_asset_cfg.joint_ids ",asset_cfg.joint_ids)
    # print("wheel_asset_cfg.joint_names ",asset_cfg.joint_names)
    # compute out of limits constraints
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def joint_track_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the action."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    act_term = env.action_manager.get_term("joint_wheel_leg")
    # compute out of limits constraints
    angle = act_term.processed_actions[:, :-4] - asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def wheel_track_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the action."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    act_term = env.action_manager.get_term("joint_wheel_leg")
    # compute out of limits constraints
    angle = act_term.processed_actions[:, -4:]-asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(angle), dim=1)

def joint_track_error_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the action."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    track_error = torch.sum(torch.abs(env.action_manager.action[:, :12] * 0.5 + asset.data.default_joint_pos[:,asset_cfg.joint_ids] - asset.data.joint_pos[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-track_error / std**2)

def wheel_track_error_exp(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the action."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    track_error = torch.sum(torch.abs(env.action_manager.action[:, -4:] * 5 - asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    return torch.exp(-track_error / std**2)
    # reward = 1.0 - torch.clamp(track_error / (std ** 2), 0.0, 1.0)


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.1, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )  # * torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 1)
    # reward = torch.square(
    #     asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    # )
    # return torch.sum(reward, dim=1)


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # Penalize feet hitting vertical surfaces
    return torch.any(contact & (forces_xy > 2 * forces_z), dim=1)


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    return torch.exp(-torch.sum(stance_diff, dim=1) / std)


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    return torch.exp(-torch.sum(stance_diff, dim=1) / std)


def feet_height_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return torch.exp(-reward / std)


# def smoothness_1(env: ManagerBasedRLEnv) -> torch.Tensor:
#     # Penalize changes in actions
#     diff = torch.square(env.action_manager.action - env.action_manager.prev_action)
#     diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
#     return torch.sum(diff, dim=1)


def smoothness_2(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize changes in actions
    diff = torch.square(env.action_manager.action - 2 * env.action_manager.prev_action + env.action_manager.prev_prev_action)
    diff = diff * (env.action_manager.prev_action[:, :] != 0)  # ignore first step
    diff = diff * (env.action_manager.prev_prev_action[:, :] != 0)  # ignore second step
    return torch.sum(diff, dim=1)


def wheel_spin_in_air_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(in_air * joint_vel, dim=1)
    return reward

# def wheel_stuck_penalty(
#     env: ManagerBasedRLEnv,
#     sensor_cfg: SceneEntityCfg,
#     target_height: float,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     asset: Articulation = env.scene[asset_cfg.name]
#     contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
#
#     vel = asset.data.body_lin_vel_w[:, 0, :2]
#     print(vel.shape)
#     base_vel_expand = vel.unsqueeze(1).expand(-1, 4, -1)
#     force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2]
#     dot = torch.sum(base_vel_expand * force, dim=2)
#     norm_v = torch.norm(vel, dim=2) + 1e-6
#     norm_f = torch.norm(force, dim=2) + 1e-6
#     cos_theta = dot / (norm_v * norm_f)
#     is_leg_stuck = (torch.sum((cos_theta < -0.7).float(), dim=1) > 0).float()
#     is_stuck = (torch.sum(is_leg_stuck, dim=1) > 0).float()
#
#     foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
#     is_lifting_leg = (torch.sum((foot_z > target_height).float(), dim=1) > 0).float()
#
#     reward = is_stuck * is_lifting_leg - is_stuck * (1.0 - is_lifting_leg)
#     return reward

def upward(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_error = asset.data.projected_gravity_b[:, 2] - 1
    return torch.exp(-gravity_error / std**2)



def lin_vel_x_stand_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 0])


def feet_height_body_exp(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_rotate_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    height_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(height_error * foot_leteral_vel, dim=1)
    return torch.exp(-reward / std**2)


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)


def locomotion_distance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_base_position =  (asset.data.root_link_pos_w[:,0])
    forward_reward = torch.sub(current_base_position, env.last_base_pos)
    # calculate what max distance can be over last time interval
    max_dist = env.max_vel * env.step_dt
    # max_task_reward = env.high_boundry_vel * env.step_dt

    clipped_reward = torch.minimum(forward_reward, max_dist)
    temp_condition = torch.gt(forward_reward , clipped_reward)
    forward_reward = torch.where( temp_condition,torch.sub(clipped_reward ,  0.13*(torch.sub(forward_reward , clipped_reward))),clipped_reward)
    # print("clipped_reward", clipped_reward)
    # print("temp_condition", temp_condition)
    # print("forward_reward", forward_reward)
    return forward_reward

def track_pos_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    com = env.command_manager.get_term(command_name)
    # compute the error
    pos_error = torch.sum(
        torch.square(com.robot.data.body_state_w[:, com.body_idx, :3] - com.pose_command_w),
        dim=1)
    return torch.exp(-pos_error / std**2)
