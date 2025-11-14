# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg, CommandsCfg

##
# Pre-defined configs
##
from robot_lab.assets.switch import UNITREE_GO2W_CFG, DOOR_CFG  # isort: skip



@configclass
class GO2WCommandsCfg(CommandsCfg):
    base_velocity = mdp.UniformVelocityCommandSwitchCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
        # FL FR RL RR
        normal_default_joint_pos = [ 0.0000, -0.0000,  0.0000, -0.0000, # hip
                                     0.8000,  0.8000,  0.8000,  0.8000, # thigh
                                    -1.5000, -1.5000, -1.5000, -1.5000, # calf
                                     0.0000,  0.0000,  0.0000,  0.0000  # wheel
                                   ],
        stand_default_joint_pos = [ 0.0000, -0.0000,  0.0000, -0.0000, # hip
                                    1.2000,  1.2000,  2.4000,  2.4000, # thigh
                                   -2.3000, -2.3000, -1.5000, -1.5000, # calf
                                    0.0000,  0.0000,  0.0000,  0.0000  # wheel
                                  ]
    )


@configclass
class GO2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_pos = None

    joint_wheel_leg = mdp.JointWheelAndLegActionCfg(
        asset_name="robot", leg_joint_names=[""], wheel_joint_names=[""], leg_scale=0.25, wheel_scale=5.0,
        use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class GO2WRewardsCfg(RewardsCfg):
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
    straight_shoulder_joints = RewTerm(
        func=mdp.straight_shoulder_joints_switch, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    straight_knee_joints = RewTerm(
        func=mdp.straight_knee_joints_switch, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    still_hand_joints = RewTerm(
        func=mdp.still_hand_joints_switch, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    hand_contact = RewTerm(
        func=mdp.hand_contact,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=""),
            "threshold": 1.0,
        },
    )


@configclass
class GO2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: GO2WActionsCfg = GO2WActionsCfg()
    rewards: GO2WRewardsCfg = GO2WRewardsCfg()
    commands: GO2WCommandsCfg = GO2WCommandsCfg()

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

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree go2w
        self.scene.num_envs = 4096
        # UNITREE_GO2W_CFG.init_state.joint_pos={
        #         ".*L_hip_joint": 0.0,  # 0.1
        #         ".*R_hip_joint": -0.0,  # -0.1
        #         "F.*_thigh_joint": 1.2,  # 0.8
        #         "R.*_thigh_joint": 2.4,  # 1.0
        #         "F.*_calf_joint": -2.3,  # -1.5
        #         "R.*_calf_joint": -1.5,
        #         ".*_foot_joint": 0.0,  # 0.0
        # }
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.door = DOOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Door")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base = None
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        # default q_wheel =0 ,return [env_nums, joint_nums]
        self.observations.policy.base_lin_vel.scale = 2.0  # 2.0
        self.observations.policy.base_ang_vel.scale = 0.25  # 0.25
        self.observations.policy.velocity_commands.scale = (2.0, 2.0, 0.25, 1)
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
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25, 1)
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
        self.events.randomize_reset_base.func = mdp.reset_root_state_switch
        self.events.randomize_reset_base.params["pose_range"]["pitch"] = (-1.6, -1.4)
        self.events.randomize_reset_base.params["pose_range"]["z"] = (0.05, 0.1)
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset_switch
        # self.events.randomize_push_robot.params["velocity_range"]["x"] = (-0.2, 0.2)
        # self.events.randomize_push_robot.params["velocity_range"]["z"] = (-0.2, 0.2)
        self.events.randomize_push_robot.params["velocity_range"]["pitch"] = (-0.1, 0.1)

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        # UNUESD self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.lin_vel_z_l2.func = mdp.lin_vel_vertical_switch_l2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.ang_vel_xy_l2.func = mdp.ang_vel_horizontal_switch_l2
        self.rewards.flat_orientation_l2.weight = -1.2
        self.rewards.flat_orientation_l2.func = mdp.flat_orientation_switch_l2
        self.rewards.body_lin_acc_l2.weight = -0.0 #-0.1
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_ang_acc_l2.weight = -0.00 #-0.01
        self.rewards.body_ang_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.0e-6  # -2.5e-6 # -1.e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -2.5e-6  # -2.5e-6 # -1.e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -1.e-4  # -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = -1.e-5  # -1.e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7 # -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9  # -2.5e-7 # -2.5e-10
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_error.weight = -0.1  #-2. #-1.0
        self.rewards.joint_error.func = mdp.joint_error_switch
        self.rewards.joint_error.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.straight_shoulder_joints.weight = -0.8
        self.rewards.straight_shoulder_joints.params["asset_cfg"].joint_names = [".*_hip_joint"]
        self.rewards.still_hand_joints.weight = -0.1
        self.rewards.still_hand_joints.params["asset_cfg"].joint_names = ["F.*_foot_joint"]
        # self.rewards.straight_knee_joints.weight = 0.5
        # self.rewards.straight_knee_joints.params["asset_cfg"].joint_names = ["FL_calf_joint", "FR_calf_joint",
        #                                                                      "RL_calf_joint", "RR_calf_joint"]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05  # -0.01
        self.rewards.action_l2.weight = -0.001 # -0.005

        # Contact sensor
        self.rewards.undesired_contacts.weight = -3.0  # -5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.hand_contact.weight = -1.0
        self.rewards.hand_contact.params["sensor_cfg"].body_names = ["FL_foot", "FR_foot"]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 1.5  # 1.5
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_switch_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.0  # 1.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_switch_exp
        self.rewards.base_height_l2.weight = 5.0  # -80.0
        self.rewards.base_height_l2.func = mdp.base_height_switch
        self.rewards.base_height_l2.params["target_height"] = 0.4
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.upward.weight = 2 # 1.8 # 2.5
        self.rewards.upward.func = mdp.standup_switch_exp
        # self.rewards.upward.weight = -50.0  # 2.5
        # self.rewards.upward.func = mdp.standup_switch

        # Others
        self.rewards.joint_power.weight = -2.5e-6  # -5.e-6
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -0.01  # -0.4
        self.rewards.stand_still_without_cmd.func = mdp.wheel_stand_still_switch
        self.rewards.stand_still_without_cmd.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 100
        self.rewards.wheel_spin_in_air_penalty.weight = -0.0 # -1.0
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.feet_contact.weight = -0.2
        self.rewards.feet_contact.func = mdp.feet_contact_switch
        self.rewards.feet_contact.params["expect_contact_num"] = 4
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "GO2WRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        # self.terminations.bad_orientation.func = mdp.bad_orientation_switch
        self.terminations.bad_orientation = None
        # self.terminations.root_height_below_minimum.func = mdp.root_height_below_minimum_wl
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.25
        # self.terminations.root_height_below_minimum.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_standing_envs = 0.05
        # self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
        #     lin_vel_x=(-1.0, 1.0),
        #     lin_vel_y=(-0.5, 0.5),
        #     ang_vel_z=(-0.5, 0.5),
        # )
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.0, 0.0),  # lat
            lin_vel_y=(-0.8, 0.8),  # forward
            ang_vel_z=(-1.0, 1.0),
        )

        # ------------------------------Curriculum------------------------------
        # self.curriculum.lin_vel_cmd_levels.params["max_range_x"] = (-2.0, 4.0)
        # self.curriculum.lin_vel_cmd_levels.params["max_range_y"] = (-1.5, 1.5)
        # self.curriculum.ang_vel_cmd_levels.params["max_range_a"] = (-1.5, 1.5)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None