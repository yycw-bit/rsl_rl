# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import MISSING
import torch
import numpy as np
from typing import TYPE_CHECKING, Sequence

from jinja2.utils import missing

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.envs.mdp.commands import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands import UniformPoseCommand
from isaaclab.assets.articulation import Articulation
from isaaclab.terrains import TerrainImporter
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from .visual import GOAL_MARKER_CFG, POS_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformFootPoseCommand(CommandTerm):

    cfg: UniformFootPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformFootPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.pose_command_w = torch.zeros_like(self.pose_command_b, device=self.device)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformFootPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        self.pose_command_w, _ = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b
        )
        pos_error =  self.robot.data.body_state_w[:, self.body_idx, :3] - self.pose_command_w
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose

        self.goal_pose_visualizer.visualize(self.pose_command_w)
        # -- current body pose
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx, :3]
        self.current_pose_visualizer.visualize(body_link_state_w)

@configclass
class UniformFootPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformFootPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""


    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to CUBOID_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = POS_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to CUBOID_MARKER_CFG."""


class RobotPoseCommand(UniformPoseCommand):

    cfg: RobotPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: RobotPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # create buffers
        # -- commands: (x, y, z) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.pose_command_w = torch.zeros_like(self.pose_command_b, device=self.device)
        self.stair_env_ids = []
        self.other_env_ids = []
        self.stair_type = 0
        # --terrains cfg
        self.terrain_gen_cfg = self._env.scene.terrain.cfg.terrain_generator
        # TODO fix the height problem
        if "up_stairs" in self.terrain_gen_cfg.sub_terrains:
            self.stair_type = self.terrain_gen_cfg.num_cols * self.terrain_gen_cfg.sub_terrains["up_stairs"].proportion
            platform_width = self.terrain_gen_cfg.sub_terrains["up_stairs"].platform_width
            border_width = self.terrain_gen_cfg.sub_terrains["up_stairs"].border_width
            _, grid_length = self.terrain_gen_cfg.size
            inner_length = grid_length - 2 * border_width

            platform_center_x = torch.full(
                (self.terrain_gen_cfg.num_rows, self.terrain_gen_cfg.num_cols),
                grid_length - border_width - platform_width / 2,
                device=self.device
            )
            platform_center_z = torch.zeros_like(platform_center_x, device=self.device)
            platform_center_y = torch.zeros_like(platform_center_x, device=self.device)
            platform_center = torch.stack(
                [platform_center_x, platform_center_y, platform_center_z],
                dim=-1
            )
            self.platform_pos = self._env.scene.terrain.terrain_origins + platform_center
            # vertices = self._env.scene.terrain.meshes["terrain"].vertices
            # points_2d = vertices[:, :2]
            # k = 20
            # for sub_row in range(self.terrain_gen_cfg.num_rows):
            #     for sub_col in range(self.terrain_gen_cfg.num_cols):
            #         query_point = np.array([self.platform_pos[sub_row, sub_col, 0].item(), self.platform_pos[sub_row, sub_col, 1].item()])
            #         dist = np.linalg.norm(points_2d - query_point, axis=1)
            #         # idx = np.argmin(dist)
            #         idx = np.argpartition(dist, k)[:k]
            #         self.platform_pos[sub_row, sub_col, 2] = vertices[idx, 2].max()
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        # self.metrics["orientation_error"] = None

        # lower, upper = self.terrain_gen_cfg.difficulty_range
        # for sub_col in range(self.terrain_gen_cfg.num_cols):
        #     for sub_row in range(self.terrain_gen_cfg.num_rows):
        #         difficulty = sub_row  / self.terrain_gen_cfg.num_rows
        #         difficulty = lower + (upper - lower) * difficulty
        #         step_height = self.terrain_gen_cfg.sub_terrains["up_stairs"].step_height_range[0] + difficulty * (
        #         self.terrain_gen_cfg.sub_terrains["up_stairs"].step_height_range[1] - self.terrain_gen_cfg.sub_terrains["up_stairs"].step_height_range[0]
        #         )
        #         # compute number of stairs
        #         num_steps = int((inner_length - self.terrain_gen_cfg.sub_terrains["up_stairs"].platform_width) // self.terrain_gen_cfg.sub_terrains["up_stairs"].step_width)
        #         platform_height = num_steps * step_height
        #         platform_center_z[sub_row, sub_col] = platform_height
        #
        # platform_center = torch.stack(
        #     [platform_center_x, platform_center_y, platform_center_z],
        #     dim=-1
        # )
        # self.platform_pos = self._env.scene.terrain.terrain_origins + platform_center
        # -- metrics
        # self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        # import numpy as np
        # direction = np.array([[0, 0, -1]])
        # origin = np.array([[0,0,0]])
        # locations, index_ray, index_tri =self._env.scene.terrain.meshes["terrain"].ray.intersects_location(origin, direction)
        # if len(locations) > 0:
        #     # 取最近的 z
        #     z = locations[:, 2].min()
        #     print(z)
        #     return z
        # else:
        #     return None  # 没碰到
        # points_2d = self._env.scene.terrain.meshes["terrain"].vertices[:, :2]  # x, y
        # query = np.array([0, 0])
        # # 计算到所有顶点的距离
        # dist = np.linalg.norm(points_2d - query, axis=1)
        # idx = np.argmin(dist)
        # print( self._env.scene.terrain.meshes["terrain"].vertices[idx, 2])

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # compute the error
        pos_error =  self.robot.data.body_state_w[:, self.body_idx, :3] - self.pose_command_w[:, :3]
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # update terrain boolean
        self.pose_command_b[env_ids, 3] = (self._env.scene.terrain.terrain_types[env_ids] < self.stair_type).float()
        mask = self.pose_command_b[env_ids, 3] != 0
        stair_env_ids = env_ids[mask]
        other_env_ids = env_ids[~mask]

        # sample new pose targets
        # -- position
        # other terrains
        r = torch.empty(len(other_env_ids), device=self.device)
        self.pose_command_w[other_env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x) + self._env.scene.env_origins[other_env_ids, 0]
        self.pose_command_w[other_env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y) + self._env.scene.env_origins[other_env_ids, 1]
        self.pose_command_w[other_env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z) + self._env.scene.env_origins[other_env_ids, 2]
        # stair terrains
        types = self._env.scene.terrain.terrain_types[stair_env_ids]
        level = self._env.scene.terrain.terrain_levels[stair_env_ids]
        r = torch.empty(len(stair_env_ids), device=self.device)
        self.pose_command_w[stair_env_ids, 0] = r.uniform_(*self.cfg.stair_ranges.pos_x) + self.platform_pos[level, types, 0]
        self.pose_command_w[stair_env_ids, 1] = r.uniform_(*self.cfg.stair_ranges.pos_y) + self.platform_pos[level, types, 1]
        self.pose_command_w[stair_env_ids, 2] = r.uniform_(*self.cfg.stair_ranges.pos_z) + self.platform_pos[level, types, 2]
        # print("env_ids: ", env_ids)
        # print("other_env_ids: ", other_env_ids)
        # print("stair_env_ids: ", stair_env_ids)

    def _update_command(self):
        self.pose_command_b[:, :3], _ = subtract_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_w[:, :3],
            None
        )

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose

        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3])
        # -- current body pose
        body_link_state_w = self.robot.data.body_state_w[:, self.body_idx, :3]
        self.current_pose_visualizer.visualize(body_link_state_w)

@configclass
class RobotPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = RobotPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""


    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

    stair_ranges: Ranges = MISSING
    """Ranges for the commands."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GOAL_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to CUBOID_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = POS_MARKER_CFG.replace(prim_path="/Visuals/Command/body_pose")
    """The configuration for the current pose visualization marker. Defaults to CUBOID_MARKER_CFG."""



