"""Termination terms for the blacknode SO-101 lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cube_out_of_reach(
    env: ManagerBasedRLEnv,
    max_dist: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """End the episode when the cube is batted beyond the arm's reach.

    Our flat infinite floor lacks the table edge of the upstream scene, where
    knocking the cube off ends the episode and forfeits all future reward.
    Without that consequence, flailing pays (random swings occasionally smack
    the cube upward past the lift threshold) and PPO inflates the action noise
    to farm it. This termination restores the table-edge dynamics: bat the
    cube away -> episode over.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    d_xy = torch.norm(cube.data.root_pos_w[:, :2] - robot.data.root_state_w[:, :2], dim=1)
    return d_xy > max_dist
