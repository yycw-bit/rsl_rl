# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import math
import numpy as np
import torch
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, ClassVar
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from rsl_rl.modules import CPG_RL

class ManagerBasedRLCPGEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        # initialize the base class to setup the scene.
        super().__init__(cfg, render_mode, **kwargs)
        self.robot_length = cfg.robotlength
        self._cpg = CPG_RL(time_step=self.physics_dt, num_envs=self.num_envs, device=self.device, **cfg.cpg.to_dict())
        if 'ALL' in cfg.cpg.rl_task_string and "OFFSETX" in cfg.cpg.rl_task_string:
            self.last_policy_action = torch.zeros(self.num_envs, 16, dtype=torch.float, device=self.device, requires_grad=False)
        elif "OFFSETX" in cfg.cpg.rl_task_string or 'ALL' in cfg.cpg.rl_task_string:
          self.last_policy_action = torch.zeros(self.num_envs, 12, dtype=torch.float, device=self.device, requires_grad=False)
        else:
            self.last_policy_action = torch.zeros(self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_base_pos = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # self.high_boundry_vel = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 1.23
        self.max_vel = torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) * 1.23
        self.policy_action_scale = cfg.cpg.policy_action_scale
        # self.last_action = torch.zeros_like(self.action_manager.action, device=self.device)
        self.foot_xs_list = []
        self.foot_ys_list = []
        self.foot_zs_list = []

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        self._cpg.reset(env_ids)
        self.last_policy_action[env_ids,:] = 0
        self.last_base_pos[env_ids] = 0


    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        des_joint_pos = torch.zeros_like(self.action_manager.action, device=self.device)
        foot_y = torch.ones(self.num_envs, device=self.device, requires_grad=False) * self.robot_length.hip_link_length_a1
        LEG_INDICES = np.array([1, 0, 3, 2])
        sideSign = np.array([-1, 1, -1, 1])
        # perform physics stepping
        for _ in range(self.cfg.decimation):
            # deal with the cpg actions
            # des_joint_pos = torch.zeros_like(self.action_manager.action, device=self.device)
            # print("action",action[0]*0.25)
            # print(action[28])
            xs, ys, zs = self._cpg.get_CPG_RL_actions(action * self.policy_action_scale)
            # sideSign = np.array([-1, 1, -1, 1])
            # sideSign = np.array([1, -1, 1, -1])
            # foot_y = torch.ones(self.num_envs, device=self.device,
            #                     requires_grad=False) * self.robot_length.hip_link_length_a1
            # LEG_INDICES = np.array([1, 0, 3, 2])
            # LEG_INDICES = np.array([0,1,2,3])
            for ig_idx, i in enumerate(LEG_INDICES):
                # x = xs[:, i]
                # z = zs[:, i]
                # y = sideSign[i] * foot_y + ys[:, i]
                x = xs[:, ig_idx]
                z = zs[:, ig_idx]
                y = sideSign[i] * foot_y + ys[:, ig_idx]
                # print(i, "x", x[0])
                # print(i, "y", y[0])
                # print(i, "z", z[0])
                des_joint_pos[:, 3 * ig_idx:3 * ig_idx + 3] = self._cpg.compute_inverse_kinematics(self.robot_length, i, x, y, z)
            self.dof_des_pos = des_joint_pos
            self.foot_xs_list.append(xs.cpu().numpy())
            self.foot_ys_list.append(ys.cpu().numpy())
            self.foot_zs_list.append(zs.cpu().numpy())
            # print("des_joint_pos",des_joint_pos[0])
            # print("xs, ys, zs",xs[0], ys[0], zs[0])
            # process actions
            self.action_manager.process_action(self.dof_des_pos.to(self.device))
            # print(des_joint_pos[0])
            # print(self.scene["robot"].data.computed_torque[0])
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # self._cpg.frequency_high[reset_env_ids, :] = torch.ones((len(reset_env_ids), 4), dtype=torch.float,
        #                                                    device=self.device, requires_grad=False) * 40
        # self._cpg.frequency_low[reset_env_ids, :] = torch.ones((len(reset_env_ids), 4), dtype=torch.float,
        #                                                   device=self.device, requires_grad=False) * -5
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()
        self.last_base_pos = self.scene["robot"].data.root_link_pos_w[:,0]

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras