# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils import configclass

import robot_lab.tasks.locomotion.velocity.mdp as mdp
from robot_lab.tasks.locomotion.velocity.velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg, RewardsCfg

##
# Pre-defined configs
##
from robot_lab.assets.unitree import UNITREE_GO2W_CFG  # isort: skip

@configclass
class UnitreeGo2WObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_with_wheel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=None)
            },
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=1.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=".*", preserve_order=True
                )
            },
        )
        actions = ObsTerm(
            func=mdp.last_action,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=1.0, clip=(-100.0, 100.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=1.0, clip=(-100.0, 100.0))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, scale=1.0, clip=(-100.0, 100.0)
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel_with_wheel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                    "wheel_asset_cfg": SceneEntityCfg("robot", joint_names=None)},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
            scale=1.0,
            clip=(-100.0, 100.0),
        )
        actions = ObsTerm(func=mdp.last_action, scale=1.0, clip=(-100.0, 100.0))
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            scale=1.0,
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivateCfg(ObsGroup):
        # desired_contact = ObsTerm(func=mdp.desired_contact,
        #                           params={"cycle_time": 1., "offsets":0., "bounds":0., "durations":0.5, "phases":0.5},
        #                           scale=1.0,
        #                           clip=(-100.0, 100.0))
        body_mass = ObsTerm(
            func=mdp.body_mass,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(-100.0, 100.0)
        )
        body_inertia = ObsTerm(
            func=mdp.body_inertia,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(-100.0, 100.0)
        )
        friction_coefficient = ObsTerm(
            func=mdp.friction_coefficient,
            params={"asset_cfg": SceneEntityCfg("robot")},
            scale=1.0,
            clip=(-100.0, 100.0)
        )
        motor_strength = ObsTerm(
            func=mdp.motor_strength,
            scale=1.0,
            clip=(-100.0, 100.0)
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class HistoryCfg(ObsGroup):
        history_prop = ObsTerm(func=mdp.PropHistory,
                               params={"func_names": {"base_lin_vel": None,
                                                      "base_ang_vel": None,
                                                      "projected_gravity": None,
                                                      "joint_pos_rel_with_wheel": SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                                                      "joint_vel_rel": None,
                                                      "last_action": None,
                                                      },
                                       "func_scales":{"base_lin_vel": 0.0,
                                                      "base_ang_vel": 0.0,
                                                      "projected_gravity": 0.0,
                                                      "joint_pos_rel_with_wheel": 0.0,
                                                      "joint_vel_rel": 0.0,
                                                      "last_action": 0.0,
                                                     },
                                        "asset_cfg": SceneEntityCfg("robot", preserve_order=True)
                                       },
                               scale=1.0,
                               clip=(-100.0, 100.0),
                               history_length=10,)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    private: PrivateCfg = PrivateCfg()
    history: HistoryCfg = HistoryCfg()


@configclass
class UnitreeGo2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""
    joint_pos = None

    joint_wheel_leg = mdp.JointWheelAndLegActionCfg(
        asset_name="robot", leg_joint_names=[""], wheel_joint_names=[""], leg_scale=0.25, wheel_scale=5.0,
        use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class UnitreeGo2WRewardsCfg(RewardsCfg):
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


@configclass
class UnitreeGo2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()
    rewards: UnitreeGo2WRewardsCfg = UnitreeGo2WRewardsCfg()
    observations: UnitreeGo2WObservationsCfg = UnitreeGo2WObservationsCfg()

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

        # ------------------------------Sence------------------------------
        # switch robot to unitree go2w
        self.scene.num_envs = 19
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # scale down the terrains because the robot is small
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # self.scene.height_scanner_base=None
        # self.scene.height_scanner = None
        # ------------------------------Observations------------------------------
        # default q_wheel =0 ,return [env_nums, joint_nums]
        self.observations.policy.base_lin_vel.scale = 2.0  # 2.0
        self.observations.policy.base_ang_vel.scale = 0.25  # 0.25
        self.observations.policy.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.policy.joint_pos.scale = 1.0  # 1.0
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_pos.params["wheel_asset_cfg"].joint_names=self.wheel_joint_names
        self.observations.policy.joint_vel.scale = 0.05  # 0.05
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.actions.scale = 1.0
        self.observations.policy.projected_gravity.scale = 0.5  # 0.5
        self.observations.policy.height_scan.scale = 5.0  # 0.5
        # self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None

        self.observations.critic.base_lin_vel.scale = 2.0  # 2.0
        self.observations.critic.base_ang_vel.scale = 0.25  # 0.25
        self.observations.critic.velocity_commands.scale = (2.0, 2.0, 0.25)
        self.observations.critic.joint_pos.scale = 1.0  # 1.0
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_pos.params["wheel_asset_cfg"].joint_names=self.wheel_joint_names
        self.observations.critic.joint_vel.scale = 0.05  # 0.05
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.actions.scale = 1.0
        self.observations.critic.projected_gravity.scale = 0.5  # 0.5
        self.observations.critic.height_scan.scale = 5.0  # 0.5
        # self.observations.critic.height_scan = None

        self.observations.private.body_mass.scale = 0.5
        self.observations.private.body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.observations.private.body_inertia.scale = 0.5
        self.observations.private.body_inertia.params["asset_cfg"].body_names = [self.base_link_name]

        self.observations.private.friction_coefficient.params["asset_cfg"].body_names = [self.foot_link_name]

        self.observations.history.history_prop.history_length = 10
        self.observations.history.history_prop.params["asset_cfg"].joint_names = self.joint_names
        self.observations.history.history_prop.params["func_names"]["joint_pos_rel_with_wheel"].joint_names=self.wheel_joint_names
        self.observations.history.history_prop.params["func_scales"]["base_lin_vel"] = self.observations.policy.base_lin_vel.scale
        self.observations.history.history_prop.params["func_scales"]["base_ang_vel"] = self.observations.policy.base_ang_vel.scale
        self.observations.history.history_prop.params["func_scales"]["projected_gravity"] = self.observations.policy.projected_gravity.scale
        self.observations.history.history_prop.params["func_scales"]["joint_pos_rel_with_wheel"] =self.observations.policy.joint_pos.scale
        self.observations.history.history_prop.params["func_scales"]["joint_vel_rel"] = self.observations.policy.joint_vel.scale
        self.observations.history.history_prop.params["func_scales"]["last_action"] = self.observations.policy.actions.scale

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
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_reset_base.params["pose_range"]["yaw"] = (0.0,0.0)
        self.events.randomize_reset_base.params["velocity_range"]={"x": (-0.5, 0.5),"y": (-0.5, 0.5),"z": (-0.5, 0.5),
                                                                    "roll": (-0.5, 0.5),"pitch": (-0.5, 0.5),"yaw": (-0.5, 0.5)}
        self.events.randomize_actuator_gains = None

        # ------------------------------Rewards------------------------------
        # General
        # UNUESD self.rewards.is_alive.weight = 0
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -0.5
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -1.5e-5  # -1.e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_torques_wheel_l2.weight = -1.5e-5  # -1.e-5
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # UNUESD self.rewards.joint_vel_l1.weight = 0.0
        self.rewards.joint_vel_l2.weight = -1.e-4
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_wheel_l2.weight = 0  # -1.e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_acc_l2.weight = -2.5e-7  # -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-10  # -2.5e-7
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_l1", 0, [""])
        self.rewards.joint_pos_limits.weight = -5.0  # -1.
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_vel_limits.weight = 0  # -0.05
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        self.rewards.joint_error.weight = -0.2
        self.rewards.joint_error.params["asset_cfg"].joint_names = self.leg_joint_names

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01  # -0.01
        # UNUESD self.rewards.action_l2.weight = 0.0

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0  # -5
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0  # 1.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5  # 1.0

        # Others
        self.rewards.feet_stumble.weight = -0.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.joint_power.weight = -2.5e-5  # -5.e-6
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.stand_still_without_cmd.weight = -0.2  # -0.4
        self.rewards.stand_still_without_cmd.func = mdp.wheel_stand_still_without_cmd
        self.rewards.stand_still_without_cmd.params["wheel_asset_cfg"] = SceneEntityCfg("robot", joint_names=self.wheel_joint_names)
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = self.leg_joint_names
        self.rewards.joint_position_penalty.params["velocity_threshold"] = 100
        self.rewards.wheel_spin_in_air_penalty.weight = -1.0
        self.rewards.wheel_spin_in_air_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_spin_in_air_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip", ".*_thigh"]
        self.terminations.root_height_below_minimum.func = mdp.root_height_below_minimum_wl
        self.terminations.root_height_below_minimum.params["minimum_height"] = 0.15
        self.terminations.root_height_below_minimum.params["sensor_cfg"] = SceneEntityCfg("height_scanner_base")
        self.terminations.bad_orientation.params["limit_angle"] = 0.9

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x=(-0.5,1.0)
        self.commands.base_velocity.ranges.lin_vel_y=(-0.5,0.5)
        self.commands.base_velocity.ranges.heading=(0.,0.)
