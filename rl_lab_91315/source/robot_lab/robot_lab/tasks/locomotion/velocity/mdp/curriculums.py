# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
import numpy as np

# from isaaclab.assets import Articulation
# from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def commands_range_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], max_ranges_curriculum: float,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    command = env.command_manager.get_term("base_velocity")
    reward = env.reward_manager.get_term_cfg("track_lin_vel_xy_exp")

    # update commands range
    if ((torch.mean(env.reward_manager._episode_sums["track_lin_vel_xy_exp"][env_ids]) / env.max_episode_length_s) > 0.9 * reward.weight):
        old_min, old_max = command.cfg.ranges.lin_vel_x
        new_min = np.clip(old_min - 0.5, -max_ranges_curriculum, 0.)
        new_max = np.clip(old_max + 0.5, 0., max_ranges_curriculum)
        command.cfg.ranges.lin_vel_x = (new_min, new_max)
    elif ((torch.mean(env.reward_manager._episode_sums["track_lin_vel_xy_exp"][env_ids]) / env.max_episode_length_s) < 0.4 * reward.weight):
        old_min, old_max = command.cfg.ranges.lin_vel_x
        new_min = np.clip(old_min + 0.5, -max_ranges_curriculum, 0.)
        new_max = np.clip(old_max - 0.5, 0., max_ranges_curriculum)
        command.cfg.ranges.lin_vel_x = (new_min, new_max)
    # return the lin_vel_x
    return command.cfg.ranges.lin_vel_x[1]

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_range_x: (float,float),
    max_range_y: (float,float),
    reward_term_name: str = "track_lin_vel_xy_exp",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                max_range_x[0],
                max_range_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                max_range_y[0],
                max_range_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_range_a: (float,float),
    reward_term_name: str = "track_ang_vel_z_exp",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.9:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                max_range_a[0],
                max_range_a[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
