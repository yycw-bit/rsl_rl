# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.position.mdp as mdp
from robot_lab.tasks.locomotion.position.postion_env_cfg import LocomotionPositionRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from robot_lab.assets.unitree import BITTER_CFG  # isort: skip


@configclass
class BITTERRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    joint_track_error = RewTerm(
        func=mdp.joint_track_error, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="", preserve_order=True)},
    )
    wheel_track_error = RewTerm(
        func=mdp.wheel_track_error, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="", preserve_order=True)}
    )


@configclass
class BITTERRoughEnvCfg(LocomotionPositionRoughEnvCfg):
    rewards: BITTERRewardsCfg = BITTERRewardsCfg()

    base_link_name = "base"
    leg_link_name = ["base",
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]
    foot_link_name = ".*_foot"
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint", "RL_foot_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RR_foot_joint"
    ]
    leg_joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    wheel_joint_names = ["FL_foot_joint", "FR_foot_joint", "RL_foot_joint", "RR_foot_joint"]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 0.005
        self.episode_length_s = 6

        # ------------------------------Sence------------------------------
        # switch robot to unitree go2w
        self.scene.num_envs = 4096
        BITTER_CFG.init_state.pos=(0.0, 0.0, 0.57)
        self.scene.robot = BITTER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner_base = None
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        # default q_wheel =0 ,return [env_nums, joint_nums]
        # self.observations.policy.base_lin_vel.scale = 2.0  # 2.0
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25  # 0.25
        self.observations.policy.position_commands.scale = (0.25, 0.5)
        self.observations.policy.mode_boolean.scale = 1.0
        self.observations.policy.joint_pos.scale = 1.0  # 1.0
        # self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        # self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_with_wheel
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot",joint_names=self.wheel_joint_names)
        self.observations.policy.joint_vel.scale = 0.05  # 0.05
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.actions.scale =1.0
        self.observations.policy.projected_gravity.scale = 0.5  # 0.5
        self.observations.policy.height_scan.scale = 5.0  # 0.5
        # self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.gaits = None


        self.observations.critic.base_lin_vel.scale = 2.0  # 2.0
        self.observations.critic.base_ang_vel.scale = 0.25  # 0.25
        self.observations.critic.position_commands.scale = (0.25, 0.5, 0.5, 1.0)
        self.observations.critic.joint_pos.scale = 1.0  # 1.0
        # self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        # self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_with_wheel
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
        self.observations.critic.joint_vel.scale = 0.05  # 0.05
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.actions.scale = 1.0
        self.observations.critic.projected_gravity.scale = 0.5  # 0.5
        self.observations.critic.height_scan.scale = 5.0  # 0.5
        # self.observations.critic.height_scan = None
        self.observations.critic.gaits = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = 0.25  # 0.5
        # self.actions.joint_vel.scale = 5.0   # 0.5
        self.actions.joint_wheel_leg.leg_scale = 0.5
        self.actions.joint_wheel_leg.wheel_scale = 5.0
        self.actions.joint_wheel_leg.leg_joint_names = self.leg_joint_names
        self.actions.joint_wheel_leg.wheel_joint_names = self.wheel_joint_names
        self.actions.joint_wheel_leg.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name] #self.leg_link_name
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name] #self.leg_link_name
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        # self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        # self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.5, 2.0)
        # self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.5, 2.0)

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05  # -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = -0.0 #-0.1
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_ang_acc_l2.weight = -0.00 #-0.01
        self.rewards.body_ang_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.5e-5  # -1.e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.e-5  # -1.e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -1.0e-4  # -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = -1.e-6  # -1.e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7 # -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-10  # -2.5e-7 # -2.5e-10
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0  # -1.
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = -0.001  # -0.05
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_error.weight = -1.0  #-3. #-1.0
        self.rewards.joint_error.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_track_error.weight = 0 # -1.e-3
        self.rewards.joint_track_error.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_track_error.weight = 0 # -1.e-4
        self.rewards.wheel_track_error.params["asset_cfg"].joint_names = self.wheel_joint_names

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01  # -0.01
        self.rewards.action_l2.weight = -0.00

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0  # -5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_position_xy.weight = 10.0  # 1.5
        self.rewards.track_position_xy.params["time_remain_threshold"] = 2.0
        self.rewards.position_bias_xyz.weight = 1.0
        self.rewards.do_not_wait.weight = 1.0

        # Others
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.0  # -5.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2.5e-5  # -5.e-6
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_spin_in_air_penalty.weight = 0
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.wheel_stuck_penalty.weight = -1.0
        # self.rewards.wheel_stuck_penalty.params["target_height"] = 0.1
        # self.rewards.wheel_stuck_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.wheel_stuck_penalty.params["asset_cfg"].body_names = [self.foot_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "BITTERRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip", ".*_thigh"]
        self.terminations.root_height_below_minimum.func = mdp.root_height_below_minimum_wl
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.15
        self.terminations.root_height_below_minimum.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.terminations.bad_orientation.params["limit_angle"] = 1.0

        # ------------------------------Commands------------------------------
        self.commands.base_position.body_name = self.base_link_name
        self.commands.base_position.ranges = mdp.RobotPoseCommandCfg.Ranges(
            pos_x=(-1.0, 1.0),
            pos_y=(-1.0, 1.0),
            pos_z=(BITTER_CFG.init_state.pos[2]-0.3, BITTER_CFG.init_state.pos[2]-0.1),
        )
        self.commands.base_position.stair_ranges = mdp.RobotPoseCommandCfg.Ranges(
            pos_x=(-0.2, 0.2),
            pos_y=(-1.0, 1.0),
            pos_z=(BITTER_CFG.init_state.pos[2]-0.3, BITTER_CFG.init_state.pos[2]-0.1),
        )

        # ------------------------------Curriculum------------------------------
        # self.curriculum.lin_vel_cmd_levels.params["max_range_x"] = (-1.5, 3.0)
        # self.curriculum.lin_vel_cmd_levels.params["max_range_y"] = (-1.0, 1.0)
        # self.curriculum.ang_vel_cmd_levels.params["max_range_a"] = (-1.5, 1.5)
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None