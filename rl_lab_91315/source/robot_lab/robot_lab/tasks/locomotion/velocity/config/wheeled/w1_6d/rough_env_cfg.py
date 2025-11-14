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
from robot_lab.assets.bit import W1_CFG  # isort: skip
from .terrains.rough import ROUGH_TERRAINS_CFG, VERTICAL_WALL_TERRAINS_CFG


@configclass
class W1ActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_pos = None

    joint_wheel_leg = mdp.JointWheelAndLegActionCfg(
        asset_name="robot", leg_joint_names=[""], wheel_joint_names=[""], leg_scale=0.25, wheel_scale=5.0,
        use_default_offset=True, clip=None, preserve_order=True
    )

@configclass
class W1CommandsCfg(CommandsCfg):
    """Command specifications for the MDP."""
    base_velocity = mdp.Uniform6DVelocityCommandBITCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.Uniform6DVelocityCommandBITCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            ang_roll=(-1.0, 1.0),
            ang_pitch=(-1.0, 1.0),
            height_z=(-1.0, 1.0),
            default_height=0.4,
            heading=(-math.pi, math.pi),
        ),
    )

@configclass
class W1RewardsCfg(RewardsCfg):
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
    straight_shoulder_joints = RewTerm(
        func=mdp.straight_shoulder_joints, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )
    track_pose_exp = RewTerm(
        func=mdp.track_pose_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.025)},
    )
    track_height_exp = RewTerm(
        func=mdp.track_height_exp,
        weight=0.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.025)},
    )


@configclass
class W1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: W1ActionsCfg = W1ActionsCfg()
    rewards: W1RewardsCfg = W1RewardsCfg()
    commands: W1CommandsCfg = W1CommandsCfg()

    base_link_name = "base"
    leg_link_name = ["base",
        "FR_hip", "FR_thigh", "FR_calf",
        "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf",
        "RL_hip", "RL_thigh", "RL_calf",
    ]
    foot_link_name = ".*_foot"
    # wheel_joint_name = ".*_foot_joint"
    # fmt: off
    # joint_names = [
    #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    #     "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    # ]
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

        # ------------------------------Sence------------------------------
        # switch robot to unitree go2w
        self.scene.num_envs = 4096
        W1_CFG.init_state.pos=(0.0, 0.0, 0.55)
        self.scene.robot = W1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner_base = None
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        # default q_wheel =0 ,return [env_nums, joint_nums]
        self.observations.policy.base_lin_vel.scale = 2.0  # 2.0
        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.25  # 0.25
        self.observations.policy.velocity_commands.scale = 1.0
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
        self.observations.critic.velocity_commands.scale = 1.0
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

        self.observations.others = None

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
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 1.0)  # add
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [ f"^(?!.*{self.base_link_name}).*"]
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.95, 1.05)  # scale
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.2, 1.25)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.2, 1.2)
        # self.events.randomize_motor_strength = None
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains = None
        # self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05  # -0.05
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = -0.005 #-0.1
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_ang_acc_l2.weight = -0.0005 #-0.01
        self.rewards.body_ang_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.e-5  # -1.e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.e-5  # -1.e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -1.e-4  # -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0 # -1.e-5  # -1.e-4
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
        self.rewards.joint_track_error.weight = -0.001 #-1.e-3
        self.rewards.joint_track_error.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.wheel_track_error.weight = -0.0001 #-1.e-4
        self.rewards.wheel_track_error.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.straight_shoulder_joints.params["asset_cfg"].joint_names = [".*hip_joint"]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01  # -0.01
        self.rewards.action_l2.weight = -0.00

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0  # -5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5  # 1.0
        self.rewards.track_pose_exp.weight = 1.5
        self.rewards.track_height_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = -1.0  # -5.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2.e-5  # -5.e-6
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -1.0  # -0.4
        self.rewards.stand_still_without_cmd.func = mdp.wheel_stand_still_without_cmd
        self.rewards.stand_still_without_cmd.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.weight = -0.  # 2
        self.rewards.joint_position_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 100
        self.rewards.feet_height_exp.weight = 0
        self.rewards.feet_height_exp.params["target_height"] = 0.1
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body_exp.weight = 0
        self.rewards.feet_height_body_exp.params["target_height"] = -0.2
        self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.wheel_spin_in_air_penalty.weight = 0
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.wheel_stuck_penalty.weight = -1.0
        # self.rewards.wheel_stuck_penalty.params["target_height"] = 0.1
        # self.rewards.wheel_stuck_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        # self.rewards.wheel_stuck_penalty.params["asset_cfg"].body_names = [self.foot_link_name]

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "W1RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]
        self.terminations.root_height_below_minimum.func = mdp.root_height_below_minimum_wl
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.2
        self.terminations.root_height_below_minimum.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.terminations.bad_orientation.params["limit_angle"] = 0.9

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = True
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.ranges = mdp.Uniform6DVelocityCommandBITCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            ang_roll=(-0.5, 0.5),
            ang_pitch=(-0.5, 0.5),
            height_z=(-0.25, 0.70),
            default_height=0.4,
            heading=(-3.14, 3.14),
        )
        # ------------------------------Curriculum------------------------------
        # self.curriculum.lin_vel_cmd_levels.params["max_range_x"] = (-1.5, 3.0)
        # self.curriculum.lin_vel_cmd_levels.params["max_range_y"] = (-1.0, 1.0)
        # self.curriculum.ang_vel_cmd_levels.params["max_range_a"] = (-1.5, 1.5)
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None