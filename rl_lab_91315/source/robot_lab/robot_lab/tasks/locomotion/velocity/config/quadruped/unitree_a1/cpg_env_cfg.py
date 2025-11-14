# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_A1_CFG  # isort: skip
# use local assets
from robot_lab.assets.unitree import UNITREE_A1_CFG  # isort: skip
import robot_lab.tasks.locomotion.velocity.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg

from dataclasses import dataclass
@dataclass
class RobotLength:
    hip_link_length_a1: float
    thigh_link_length_a1: float
    calf_link_length_a1: float

@configclass
class Cpg:
    rl_task_string='CPG_ALL_OFFSETX'  # 'CPG', 'CPG_OFFSETX', 'CPG_ALL'
    # omega_swing = 8 * 2 * np.pi,
    # omega_stance = 2 * 2 * np.pi,
    gait = "TROT"
    couple = False
    coupling_strength = 1
    robot_height = 0.30
    des_step_len = 0.08
    ground_clearance = 0.07
    ground_penetration = 0.01
    mu_low = 1.0
    mu_up = 4.0
    frequency_low = -40.0
    frequency_up = 40.0
    fai_low = -10.0
    fai_up = 10.0
    max_step_len = 0.03
    policy_action_scale = 1.0


@configclass
class UnitreeA1RoughCpgEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "trunk" #"base"
    foot_link_name = ".*_foot"
    # fmt: off
    # joint_names = [
    #     "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    #     "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    #     "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    #     "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    # ]
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        # switch robot to unitree A1
        self.scene.num_envs =4096
        self.decimation = 10
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        self.scene.robot = UNITREE_A1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner = None
        # self.scene.height_scanner_base = None

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.ang_vel_cmd_levels = None
        # scale down the terrains because the robot is small
        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0  # 2.0
        self.observations.policy.base_ang_vel.scale = 0.25  # 0.25
        self.observations.policy.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.policy.joint_pos.scale = 1.0  # 1.0
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.scale = 0.05  # 0.05
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.projected_gravity.scale = 0.5  # 0.5
        # self.observations.policy.base_lin_vel = None
        self.observations.policy.actions = None
        self.observations.policy.height_scan = None
        self.observations.policy.gaits = None
        self.observations.policy.contact_boolean= ObsTerm(func=mdp.true_contact,params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                                                                              body_names=[self.foot_link_name], preserve_order=True)},
                                                         scale=1.0,clip=(-100.0, 100.0))
        # self.observations.policy.contact = ObsTerm(func=mdp.obs_contact_force ,
        #                                                    params={"sensor_cfg": SceneEntityCfg("contact_forces",
        #                                                                                         body_names=[
        #                                                                                             self.foot_link_name],
        #                                                                                         preserve_order=True)},
        #                                                    scale=1.0, clip=(-100.0, 100.0))


        self.observations.critic.base_lin_vel.scale = 2.0  # 2.0
        self.observations.critic.base_ang_vel.scale = 0.25  # 0.25
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.critic.joint_pos.scale = 1.0  # 1.0
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.scale = 0.05  # 0.05
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.actions.scale = 1.0
        self.observations.critic.projected_gravity.scale = 0.5  # 0.5
        self.observations.critic.height_scan.scale = 5.0  # 0.5
        self.observations.critic.actions = None
        self.observations.critic.height_scan = None
        self.observations.critic.gaits = None
        self.observations.critic.contact_boolean = ObsTerm(func=mdp.true_contact,
                                                           params={"sensor_cfg": SceneEntityCfg("contact_forces",
                                                                                                body_names=[self.foot_link_name],preserve_order=True)},
                                                           scale=1.0,clip=(-100.0, 100.0))
        self.observations.others = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = 1.0
        self.actions.joint_pos.use_default_offset = False
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        self.cpg=Cpg()
        self.robotlength = RobotLength(
            hip_link_length_a1=0.0838,
            thigh_link_length_a1=0.213,
            calf_link_length_a1=0.213
        )

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (-0.05, 0.05),"y": (-0.05, 0.05),"z": (-0.05, 0.05),
                "roll": (-0.05, 0.05),"pitch": (-0.05, 0.05),"yaw": (-0.05, 0.05),},
        }
        self.events.randomize_actuator_gains=None
        self.events.randomize_motor_strength = None
        # self.events.randomize_push_robot.interval_range_s=(3.0, 5.0)
        # self.events.randomize_push_robot.params["velocity_range"]= {"x": (-0.5, 0.5)}
        # self.events.randomize_actuator_gains= None
        # self.events.randomize_push_robot= None
        # self.events.randomize_com_positions = None
        # self.events.randomize_apply_external_force_torque = None


        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -1.0
        # self.rewards.lin_vel_z_l2.func = mdp.lin_vel_yz_l2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.3
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_ang_acc_l2.weight = 0
        self.rewards.body_ang_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = 0 #-2.5e-5
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = 0#-2.5e-8
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_error.weight = -1.0  # -2. #-1.0
        self.rewards.joint_error.params["asset_cfg"].joint_names = self.joint_names

        # Action penalties
        self.rewards.action_rate_l2.weight = 0 #-100.
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0 #-1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_x_exp
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.locomotion_distance = RewTerm(func=mdp.locomotion_distance, weight=0.0)
        self.rewards.locomotion_distance.weight = 00.0

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still_without_cmd.weight = -0.5
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names=self.joint_names
        self.rewards.joint_position_penalty.weight = 0
        self.rewards.feet_height_exp.weight = 0
        self.rewards.feet_height_exp.params["target_height"] = 0.05
        self.rewards.feet_height_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body_exp.weight = 0
        self.rewards.feet_height_body_exp.params["target_height"] = -0.2
        self.rewards.feet_height_body_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeA1RoughCpgEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.root_height_below_minimum.params["minimum_height"]=0.15
        self.terminations.terrain_out_of_bounds =None
        self.terminations.bad_orientation.params["limit_angle"] =0.7

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.heading_command = True
        # self.commands.base_velocity.rel_standing_envs = 0.02
        ranges = mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.2, 1.2),
            lin_vel_y=(-0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14, 3.14),
        ),