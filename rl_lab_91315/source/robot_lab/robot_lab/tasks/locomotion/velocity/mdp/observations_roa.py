# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
import torchvision
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import ObservationTermCfg
from isaaclab.assets import Articulation, RigidObject
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.sensors import ContactSensor, RayCaster, TiledCamera


class PropHistory(ManagerTermBase):
    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        """Compute the history of terms.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        self.func_cfgs: dict[str, SceneEntityCfg | None] = cfg.params.get("func_names", {})
        self.func_scales: dict[str, float] = cfg.params.get("func_scales", {})

        # extract the used quantities (to enable type-hinting)
        self.obs_asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset = env.scene[self.obs_asset_cfg.name]
        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Observation term 'prop_history' not supported for asset: '{self.obs_asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

    def __call__(
        self,
        env: ManagerBasedEnv,
        func_names: dict[str, SceneEntityCfg | None] = None,
        func_scales: dict[str, float | None] = None,
        asset_cfg: SceneEntityCfg | None = None,
    ) -> torch.Tensor:
        outputs = []
        for name, asset_cfg in self.func_cfgs.items():
            func = getattr(self, name, None)
            if func is None:
                raise ValueError(f"Function '{name}' not found in PropHistory.")
            # 如果 asset_cfg=None，就只传 env
            result = func(env) if asset_cfg is None else func(env, asset_cfg)
            scale = self.func_scales.get(name, 1.0)
            result = result * scale
            outputs.append(result)

        return torch.cat(outputs, dim=-1)

    def base_lin_vel(self, env: ManagerBasedEnv,) -> torch.Tensor:
        """Root linear velocity in the asset's root frame."""
        # extract the used quantities (to enable type-hinting)
        return self.asset.data.root_lin_vel_b

    def base_ang_vel(self, env: ManagerBasedEnv,) -> torch.Tensor:
        """Root angular velocity in the asset's root frame."""
        # extract the used quantities (to enable type-hinting)
        return self.asset.data.root_ang_vel_b

    def projected_gravity(self, env: ManagerBasedEnv,) -> torch.Tensor:
        """Gravity projection on the asset's root frame."""
        # extract the used quantities (to enable type-hinting)
        return self.asset.data.projected_gravity_b

    def joint_pos_rel_with_wheel(self, env: ManagerBasedEnv, wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
        # extract the used quantities (to enable type-hinting)
        joint_pos_rel = self.asset.data.joint_pos.clone()
        joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
        joint_pos_rel = joint_pos_rel[:, self.obs_asset_cfg.joint_ids] - self.asset.data.default_joint_pos[:, self.obs_asset_cfg.joint_ids]
        return joint_pos_rel

    def joint_vel_rel(self, env: ManagerBasedEnv,):
        """The joint velocities of the asset w.r.t. the default joint velocities.

        Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
        """
        # extract the used quantities (to enable type-hinting)
        return self.asset.data.joint_vel[:, self.obs_asset_cfg.joint_ids] - self.asset.data.default_joint_vel[:, self.obs_asset_cfg.joint_ids]

    def last_action(self, env: ManagerBasedEnv) -> torch.Tensor:
        """The last input action to the environment.

        The name of the action term for which the action is required. If None, the
        entire action tensor is returned.
        """
        return env.action_manager.action


def true_contact(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Return feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact
    # contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]> 0.0
    return contact.float()


def obs_contact_force(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg
    ) -> torch.Tensor:
    """Return feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the contact
    forces_xyz = torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=2)
    return forces_xyz.float() - 25


def body_mass(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(env.device)
    return masses[:, asset_cfg.body_ids]


def body_inertia(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    inertia = asset.root_physx_view.get_inertias().to(env.device)
    return inertia[:, asset_cfg.body_ids].squeeze(dim=1)


def friction_coefficient(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties().to(env.device)
    return materials[:, asset_cfg.body_ids, 0]


def motor_strength(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    # extract the used quantities (to enable type-hinting)
    return env.action_manager.get_term("joint_wheel_leg")._motor_strength - 1


def camera_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, camera_type: str, resized_shape: tuple[float, float]) -> torch.Tensor:
    """Depth picture from the given sensor w.r.t. the sensor's frame.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera = env.scene.sensors[sensor_cfg.name]
    resize_transform = torchvision.transforms.Resize(resized_shape, interpolation=torchvision.transforms.InterpolationMode.BICUBIC)
    depth_image = sensor.data.output[camera_type].squeeze(-1)  # [n, H, W, 1] -> [n, H, W]
    depth_image = depth_image[:, :-2, 4:-4]
    depth_image = resize_transform(depth_image)
    depth_image = (depth_image - sensor.cfg.spawn.clipping_range[0]) / (sensor.cfg.spawn.clipping_range[1]- sensor.cfg.spawn.clipping_range[0]) - 0.5
    return depth_image

# Debug
# def camera_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, camera_type: str, resized_shape: tuple[float, float]) -> torch.Tensor:
#     """Depth picture from the given sensor w.r.t. the sensor's frame.
#     """
#     # extract the used quantities (to enable type-hinting)
#     return torch.ones([env.num_envs, 87, 58], device=env.device)