# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Configuration for Unitree robots.

The following configurations are available:

* :obj:`UNITREE_A1_CFG`: Unitree A1 robot with DC motor model for the legs
* :obj:`G1_CFG`: G1 humanoid robot

Reference: https://github.com/unitreerobotics/unitree_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

##
# Configuration
##

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

BITTER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bit/Isaac_Home/robot_model/bitter/bitter_0825.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80), # 461.5mm
        joint_pos={
            ".*_hip_joint": 0.0,  
            "F.*_thigh_joint": 0.7854,
            "R.*_thigh_joint": -0.7854,
            "F.*_calf_joint": -1.5708,
            "R.*_calf_joint": 1.5708,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=70.0,
            saturation_effort=70.0,
            velocity_limit=11.99,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
        "legs": DCMotorCfg(
            joint_names_expr=[".*thigh_joint",".*calf_joint"],
            effort_limit=75.0,
            saturation_effort=75,
            velocity_limit=11.99,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=75.0,
            saturation_effort=75,
            velocity_limit=83.78,
            stiffness=0.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Bitter using DC-Motor actuator model."""


BITTER_Y__CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bit/robot_lab-main/source/robot_lab/data/Robots/BIT/Bitter/bitter.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.80), # 461.5mm
        joint_pos={
            ".*_hip_joint": 0.0,  # 0.1
            "F.*_thigh_joint": -0.7854,  # 0.8
            "R.*_thigh_joint": 0.7854,  # 1.0
            "F.*_calf_joint": 1.5708,  # -1.5
            "R.*_calf_joint": -1.5708,  # -1.5
            ".*_foot_joint": 0.0,  # 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=200.0,
            saturation_effort=45.0,
            velocity_limit=11.99,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
        "legs": DCMotorCfg(
            joint_names_expr=[".*thigh_joint",".*calf_joint"],
            effort_limit=200,
            saturation_effort=45,
            velocity_limit=11.99,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=35,
            saturation_effort=35,
            velocity_limit=83.78,
            stiffness=0.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Bitter using DC-Motor actuator model."""

DOOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bit/Desktop/ros2/door/door.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(1.0, 0.0, 0.0),
        rot = (0.7071, 0.0, 0.0, 0.7071),
        joint_pos={ "door_hinge": 0.0 },
        joint_vel={ "door_hinge": 0.0 },
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "door_hinge": IdealPDActuatorCfg(
            joint_names_expr=["door_hinge"],
            effort_limit=33.5,
            velocity_limit=21.0,
            stiffness=0,
            damping=0,
            friction=0.0,
        ),
    },
)
"""Configuration of Door using IdealPD actuator model."""


W1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/bit/ycw/model/w1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            # fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58), # 550mm
        joint_pos={
            ".*L_hip_joint": 0.,
            ".*R_hip_joint": -0.,
            ".*_thigh_joint": -0.6,   # -0.74,
            ".*_calf_joint": 0.6,    # 0.80,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[".*hip_joint"],
            effort_limit=70.0,
            saturation_effort=70.0,
            velocity_limit=20.0,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.001,
            # viscous_friction=0.05,
        ),
        "legs": DCMotorCfg(
            joint_names_expr=[".*thigh_joint",".*calf_joint"],
            effort_limit=80.0,
            saturation_effort=80,
            velocity_limit=20.0,
            stiffness=40.0,  # 20.0
            damping=0.5,
            friction=0.001,
            # viscous_friction=0.05,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=80.0,
            saturation_effort=80,
            velocity_limit=20.0,
            stiffness=0.0,  # 20.0
            damping=0.5,
            friction=0.005,
            # dynamic_friction=0.0001,
            # viscous_friction=0.0001,
        ),
    },
)
"""Configuration of W1 using DC-Motor actuator model."""
