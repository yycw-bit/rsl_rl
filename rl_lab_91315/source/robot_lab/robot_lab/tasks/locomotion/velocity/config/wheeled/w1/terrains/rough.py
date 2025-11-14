# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=8.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    seed=1,
    curriculum = True,
    sub_terrains={
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.23),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "up_stairs": terrain_gen.MeshUpwardStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.23),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.4,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wall":terrain_gen.MeshBoxTerrainCfg(
        proportion=0., box_height_range=(0.05,0.4), platform_width=2.0, double_box=True
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3, noise_range=(0.02, 0.20), noise_step=0.02, border_width=0.25
        ),
        "curriculum_rough": terrain_gen.HfCurriculumUniformTerrainCfg(
            proportion=0.3, noise_range=(0.02, 0.20), noise_step=0.02, downsampled_scale=0.1,border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "plane":terrain_gen.MeshPlaneTerrainCfg(proportion=0.0,),
    },
)
"""Rough terrains configuration."""


VERTICAL_WALL_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(6.0, 6.0),
    border_width=8.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    seed=1,
    curriculum = True,
    sub_terrains={
        "stairs": terrain_gen.MeshStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.3),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "up_stairs": terrain_gen.MeshUpwardStairsTerrainCfg(
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.4,
            platform_width=1.5,
            border_width=1.0,
            holes=False,
        ),
        "vertical_wall": terrain_gen.MeshVerticalWallTerrainCfg(
            proportion=0.7,
            wall_height_range=(0.05, 0.30),
            wall_length=4.0,
            wall_width=4.0,
            border_width=1.0,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.0, noise_range=(0.02, 0.20), noise_step=0.02, border_width=0.25
        ),
        "curriculum_rough": terrain_gen.HfCurriculumUniformTerrainCfg(
            proportion=0.0, noise_range=(0.02, 0.20), noise_step=0.02, downsampled_scale=0.1,border_width=0.25
        ),
        "plane":terrain_gen.MeshPlaneTerrainCfg(proportion=0.1,),
    },
)
"""Rough terrains configuration."""