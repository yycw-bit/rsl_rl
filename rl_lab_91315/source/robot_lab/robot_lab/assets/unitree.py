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
UNITREE_A1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/A1/a1.usd", #f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/A1/a1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4,
            solver_velocity_iteration_count=0, fix_root_link=False
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.35),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.785,
            "R.*_thigh_joint": 0.785,
            ".*_calf_joint": -1.57,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=100.0, #20.0,
            damping=2.0, #0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree A1 using DC motor.

Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
"""

UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/Go2/go2.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""

UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/Go2W/go2w.usd",
        usd_path=f"/home/bit/Downloads/go2w/go2w_new.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    # spawn=sim_utils.UrdfFileCfg(
    #     fix_base=False,
    #     merge_fixed_joints=True,
    #     replace_cylinders_with_capsules=False,
    #     asset_path=f"/home/bit/anaconda3/envs/go2w/go2w-main/legged_gym/resources/robots/go2w/go2w_description/urdf/go2w_description.urdf",
    #     activate_contact_sensors=True,
    #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #         disable_gravity=False,
    #         retain_accelerations=False,
    #         linear_damping=0.0,
    #         angular_damping=0.0,
    #         max_linear_velocity=1000.0,
    #         max_angular_velocity=1000.0,
    #         max_depenetration_velocity=1.0,
    #     ),
    #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
    #         enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
    #     ),
    #     joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
    #         gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    #     ),
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,       # 0.1
            ".*R_hip_joint": -0.0,      # -0.1
            "F.*_thigh_joint": 0.8,     # 0.8
            "R.*_thigh_joint": 0.8,     # 1.0
            ".*_calf_joint": -1.5,      # -1.5
            ".*_foot_joint": 0.0,       # 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
        "wheels": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=0.0,  # 20.0
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using DC motor.
"""

UNITREE_B2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/B2/b2.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=320.0,
            saturation_effort=320.0,
            velocity_limit=14.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B2 using DC motor.
"""


UNITREE_B2W_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/Unitree/B2W/b2w.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 1.0,#0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=320.0,
            saturation_effort=320.0,
            velocity_limit=14.0,
            stiffness=50.0,
            damping=1.0,
            friction=0.0,
        ),
        "wheel": DCMotorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit=20.0,
            saturation_effort=20.0,
            velocity_limit=50.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B2W using DC motor.
"""

        # "legs": DCMotorCfg(
        #     joint_names_expr=["^(?!.*_foot_joint).*"],
        #     effort_limit=50,
        #     saturation_effort=50,
        #     velocity_limit=30.0,
        #     stiffness=500.0,  # 20.0
        #     damping=50.0,
        #     friction=0.0,
        # ),

