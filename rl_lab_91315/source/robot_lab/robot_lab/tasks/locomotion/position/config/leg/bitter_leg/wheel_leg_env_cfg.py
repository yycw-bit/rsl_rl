# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp

##
# Pre-defined configs
##
from robot_lab.assets.one_leg import BITTER_ONE_LEG_CFG  # isort: skip
from .leg_base_cfg import LegBaseEnvCfg, ActionsCfg, RewardsCfg


@configclass
class LegActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_wheel_leg = mdp.JointWheelAndLegActionCfg(
        asset_name="robot", leg_joint_names=[""], wheel_joint_names=[""], leg_scale=0.25, wheel_scale=5.0,
        use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class LegRewardsCfg(RewardsCfg):
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
    tracking_contacts_shaped = RewTerm(
        func=mdp.tracking_contacts_shaped, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "sensor_cfg": SceneEntityCfg("contact_forces")}
    )
    # wheel_stuck_penalty = RewTerm(
    #     func=mdp.wheel_stuck_penalty,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=""),
    #         "target_height": float,
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
    #     },
    # )


@configclass
class BITTEROneLegEnvCfg(LegBaseEnvCfg):
    actions: LegActionsCfg = LegActionsCfg()
    rewards: LegRewardsCfg = LegRewardsCfg()

    base_link_name = "base"
    leg_link_name = [ "base", "FL_hip", "FL_thigh", "FL_calf" ]
    foot_link_name = ".*_foot"
    joint_names = [ "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint" ]
    leg_joint_names = [ "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint" ]
    wheel_joint_names = [ "FL_foot_joint" ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree go2w
        self.scene.num_envs = 4096
        self.scene.robot = BITTER_ONE_LEG_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        # ------------------------------Observations------------------------------
        # default q_wheel =0 ,return [env_nums, joint_nums]
        self.observations.policy.pos_commands.scale = 2.0
        self.observations.policy.joint_pos.scale = 1.0  # 1.0
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_with_wheel
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot",joint_names=self.wheel_joint_names)
        self.observations.policy.joint_vel.scale = 0.05  # 0.05
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.actions.scale =1.0
        # self.observations.policy.height_scan.scale = 5.0  # 0.5
        self.observations.policy.height_scan = None


        self.observations.critic.pos_commands.scale = 2.0
        self.observations.critic.joint_pos.scale = 1.0  # 1.0s
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_with_wheel
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
        self.observations.critic.joint_vel.scale = 0.05  # 0.05
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.actions.scale = 1.0
        # self.observations.critic.height_scan.scale = 5.0  # 0.5
        # self.observations.critic.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        # self.actions.joint_pos.scale = 0.25  # 0.5
        # self.actions.joint_vel.scale = 5.0   # 0.5
        self.actions.joint_wheel_leg.class_type = mdp.JointOneLegAction
        self.actions.joint_wheel_leg.leg_scale = 0.5
        self.actions.joint_wheel_leg.wheel_scale = 5.0
        self.actions.joint_wheel_leg.leg_joint_names = self.leg_joint_names
        self.actions.joint_wheel_leg.wheel_joint_names = self.wheel_joint_names
        self.actions.joint_wheel_leg.clip = {".*": (-100.0, 100.0)}

        # ------------------------------Events------------------------------
        # self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name] #self.leg_link_name
        # self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name] #self.leg_link_name
        # self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        # self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.5, 2.0)
        # self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.5, 2.0)
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_com_positions = None

        # ------------------------------Rewards------------------------------
        self.rewards.track_pos_exp.weight = 3.0

        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.e-5  # -1.e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.e-5  # -1.e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -1.e-2  # -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = -1.e-4  # -1.e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-5 # -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-6  # -2.5e-7 # -2.5e-10
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0  # -1.
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = -0.001  # -0.05
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_error.weight = -0.0  #-2. #-1.0
        self.rewards.joint_error.params["asset_cfg"].joint_names = self.leg_joint_names

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.1  # -0.01
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -0.0  # -5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Others
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2e-3  # -5.e-6
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.weight = -0.  # 2
        self.rewards.joint_position_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 100
        self.rewards.feet_height_exp.weight = 0
        self.rewards.feet_height_exp.params["target_height"] = 0.1
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.weight = -500
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "BITTEROneLegEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------

        # ------------------------------Commands------------------------------
        self.commands.foot_pose.body_name = self.foot_link_name
        self.commands.foot_pose.ranges.pos_x = (-0.2, 0.2)
        self.commands.foot_pose.ranges.pos_y = (-0.01, 0.01)
        self.commands.foot_pose.ranges.pos_z = (-0.5, -0.3)

        # ------------------------------Curriculum------------------------------
