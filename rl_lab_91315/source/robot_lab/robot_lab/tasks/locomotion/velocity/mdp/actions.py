# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from dataclasses import MISSING
from collections.abc import Sequence
import omni.log
import isaaclab.utils.string as string_utils
from isaaclab.utils import configclass
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.envs import ManagerBasedEnv


##
# Joint actions.
##

class JointOneLegAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: JointWheelAndLegActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _leg_scale: torch.Tensor | float
    _wheel_scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _leg_mapping = [0,1,2]
    _wheel_mapping = [3]

    def __init__(self, cfg: JointWheelAndLegActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            self.cfg.leg_joint_names, preserve_order=self.cfg.preserve_order
        )
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            self.cfg.wheel_joint_names, preserve_order=self.cfg.preserve_order
        )

        self._num_joints = len(self._leg_joint_ids) + len(self._wheel_joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f"The first part leg_joint: {self._leg_joint_names} [{self._leg_joint_ids}]"
            f"The second part wheel_joint: {self._wheel_joint_names} [{self._wheel_joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.leg_scale, (float, int)) and isinstance(cfg.wheel_scale, (float, int)):
            self._leg_scale = float(cfg.leg_scale)
            self._wheel_scale = float(cfg.wheel_scale)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.leg_scale)}. Supported types are float.")
        # parse offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        else:
            raise ValueError(f"You must use default offset")

        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._leg_joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._wheel_joint_names)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions[:, :-1] = self._raw_actions[:,self._leg_mapping] * self._leg_scale + self._offset
        self._processed_actions[:, -1:] = self._raw_actions[:, self._wheel_mapping] * self._wheel_scale
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions[:, :-1], joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(self.processed_actions[:, -1:], joint_ids=self._wheel_joint_ids)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0


class JointWheelAndLegAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: JointWheelAndLegActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _leg_scale: torch.Tensor | float
    """The scaling factor applied to the input action for legs."""
    _wheel_scale: torch.Tensor | float
    """The scaling factor applied to the input action for wheels."""
    _leg_motor_strength: torch.Tensor | float
    """The motoe strength factor applied to legs."""
    _wheel_motor_strength: torch.Tensor | float
    """The motoe strength factor applied to wheels."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""
    _leg_mapping = [0,1,2,4,5,6,8,9,10,12,13,14]
    _wheel_mapping = [3,7,11,15]

    def __init__(self, cfg: JointWheelAndLegActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._leg_joint_ids, self._leg_joint_names = self._asset.find_joints(
            self.cfg.leg_joint_names, preserve_order=self.cfg.preserve_order
        )
        self._wheel_joint_ids, self._wheel_joint_names = self._asset.find_joints(
            self.cfg.wheel_joint_names, preserve_order=self.cfg.preserve_order
        )

        self._num_joints = len(self._leg_joint_ids) + len(self._wheel_joint_ids)
        # log the resolved joint names for debugging
        omni.log.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f"The first part leg_joint: {self._leg_joint_names} [{self._leg_joint_ids}]"
            f"The second part wheel_joint: {self._wheel_joint_names} [{self._wheel_joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.leg_scale, (float, int)) and isinstance(cfg.wheel_scale, (float, int)):
            self._leg_scale = float(cfg.leg_scale)
            self._wheel_scale = float(cfg.wheel_scale)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.leg_scale)}. Supported types are float.")
        # parse offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        else:
            raise ValueError(f"You must use default offset")

        self._motor_strength = torch.ones(self.num_envs, self.action_dim, device=self.device)

        # parse clip
        if self.cfg.leg_joint_clip is not None:
            if isinstance(cfg.leg_joint_clip, dict):
                self._leg_clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, 12, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.leg_joint_clip, self._leg_joint_names)
                self._leg_clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")
        if self.cfg.wheel_joint_clip is not None:
            if isinstance(cfg.wheel_joint_clip, dict):
                self._wheel_clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, 4, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.wheel_joint_clip, self._wheel_joint_names)
                self._wheel_clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions[:, :-4] = self._raw_actions[:, self._leg_mapping] * self._leg_scale * self._motor_strength[:, self._leg_mapping] + self._offset
        self._processed_actions[:, -4:] = self._raw_actions[:, self._wheel_mapping] * self._wheel_scale * self._motor_strength[:, self._wheel_mapping]

        # clip actions
        if self.cfg.leg_joint_clip is not None:
            self._processed_actions[:, :-4] = torch.clamp(
                self._processed_actions[:, :-4], min=self._leg_clip[:, :, 0], max=self._leg_clip[:, :, 1]
            )
        if self.cfg.wheel_joint_clip is not None:
            self._processed_actions[:, -4:] = torch.clamp(
                self._processed_actions[:, -4:], min=self._wheel_clip[:, :, 0], max=self._wheel_clip[:, :, 1]
            )
    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self.processed_actions[:, :-4], joint_ids=self._leg_joint_ids)
        self._asset.set_joint_velocity_target(self.processed_actions[:, -4:], joint_ids=self._wheel_joint_ids)


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0



@configclass
class JointWheelAndLegActionCfg(ActionTermCfg):
    """Configuration for the wheel-leg action term.

    See :class:`JointVelocityAction` for more details.
    """
    leg_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    leg_scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    wheel_joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    wheel_scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    leg_joint_clip: dict[str, tuple] | None = None
    """Clip range for the leg action (dict of regex expressions). Defaults to None."""
    wheel_joint_clip: dict[str, tuple] | None = None
    """Clip range for the wheel action (dict of regex expressions). Defaults to None."""

    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    class_type: type[ActionTerm] = JointWheelAndLegAction

    use_default_offset: bool = True
    """Whether to use default joint velocities configured in the articulation asset as offset.
    Defaults to True.

    This overrides the settings from :attr:`offset` if set to True.
    """


