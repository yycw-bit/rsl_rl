# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from .rough_env_cfg import  W1RoughEnvCfg


@configclass
class W1FlatEnvCfg(W1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.terminations.root_height_below_minimum.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.commands.base_velocity.heading_command = True
        # self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        )

        # Root penalties
        self.rewards.base_height_l2.weight = -00.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.body_lin_acc_l2.weight = -0.05  # -0.1
        self.rewards.body_ang_acc_l2.weight = -0.005  # -0.01
        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -2.5e-5  # -1.e-5
        self.rewards.joint_torques_wheel_l2.weight = -2.e-4   # -2.e-5 # -1.e-5
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -5.e-7 # -5.e-5  # -2.5e-4  # -1.e-4
        self.rewards.joint_vel_wheel_l2.weight = -2.e-5 # -1.e-5 # -1.e-4
        self.rewards.joint_acc_l2.weight = -1.e-5 # -5.e-6  # -2.5e-7
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-10  # -2.5e-7 # -2.5e-10
        self.rewards.joint_pos_limits.weight = -1.0  # -1.
        self.rewards.joint_vel_limits.weight = -0.001  # -0.05
        self.rewards.joint_error.weight = -0.60  # -3. #-1.0
        self.rewards.joint_track_error.weight = 0
        self.rewards.wheel_track_error.weight = 0
        # Action penalties
        self.rewards.action_rate_l2.weight = -0.02  # -0.01
        self.rewards.action_l2.weight = -0.00
        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0  # -5
        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5  # 1.0
        # Others
        self.rewards.feet_stumble.weight = -0.0
        self.rewards.joint_power.weight = -2.5e-5  # -5.e-6
        self.rewards.stand_still_without_cmd.weight = -1.0  # -0.4
        self.rewards.straight_shoulder_joints.weight = -0.5  #-0.1
        self.rewards.straight_shoulder_joints.params["asset_cfg"].joint_names = [".*_hip_joint"]


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "W1FlatEnvCfg":
            self.disable_zero_weight_rewards()
