"""Observation terms for the blacknode SO-101 lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_position_b(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position expressed in the robot base frame, shape (num_envs, 3).

    World-frame positions are useless to the policy once environments are
    cloned across the grid, so we transform the cube into each robot's own
    base frame before feeding it to the network.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        cube.data.root_pos_w[:, :3],
    )
    return pos_b
