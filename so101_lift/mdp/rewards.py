"""Reward terms for the blacknode SO-101 lift task.

Three-stage shaping, each stage gated so the policy can't farm a later
stage without passing the earlier one:

1. reach  -- exponential kernel on gripper->cube distance,
2. lift   -- flat bonus once the cube clears a height threshold,
3. place  -- exponential kernel on cube->goal distance, paid only while lifted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_index(ee: FrameTransformer) -> int:
    """Index of the grasp-point frame -- never assume it is frame 0."""
    return ee.data.target_frame_names.index("end_effector")


def ee_cube_distance(
    env: ManagerBasedRLEnv,
    std: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """exp(-d/std) on the gripper->cube distance: 1 at contact, ->0 far away."""
    cube: RigidObject = env.scene[cube_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    d = torch.norm(cube.data.root_pos_w - ee.data.target_pos_w[..., _ee_index(ee), :], dim=1)
    return torch.exp(-d / std)


def gripper_open_on_approach(
    env: ManagerBasedRLEnv,
    open_pos: float = 0.8,
    near_dist: float = 0.04,
    far_dist: float = 0.15,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Reward OPEN jaws while approaching (between near_dist and far_dist).

    Complements ``gripper_closing_near_cube``: open on the way in, close once
    there. Without this, the policy may approach jaws-shut and just poke the
    cube, never discovering that it must straddle it first.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    robot = env.scene[robot_cfg.name]
    if isinstance(robot_cfg.joint_ids, slice):
        raise ValueError("gripper_open_on_approach: pass robot_cfg via the term's params")
    d = torch.norm(cube.data.root_pos_w - ee.data.target_pos_w[..., _ee_index(ee), :], dim=1)
    in_band = (d > near_dist) & (d < far_dist)
    grip = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
    openness = torch.clamp(grip / open_pos, 0.0, 1.0)  # 1 = fully open
    return in_band.float() * openness


def ee_pointing_at_cube(
    env: ManagerBasedRLEnv,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward orienting the gripper mouth toward the cube.

    The TCP's local +Z axis points out of the jaws (measured from the asset).
    Reward = cosine between that axis (in world) and the ee->cube direction:
    1 when the gripper faces the cube dead-on, 0 when sideways or away. This
    is what teaches the wrist joints to rotate into a graspable orientation.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    idx = _ee_index(ee)
    ee_pos = ee.data.target_pos_w[..., idx, :]
    ee_quat = ee.data.target_quat_w[..., idx, :]
    approach_axis = torch.tensor([0.0, 0.0, 1.0], device=ee_pos.device).expand(ee_pos.shape[0], 3)
    axis_w = quat_apply(ee_quat, approach_axis)
    to_cube = cube.data.root_pos_w - ee_pos
    to_cube = to_cube / torch.norm(to_cube, dim=1, keepdim=True).clamp(min=1e-6)
    return torch.clamp((axis_w * to_cube).sum(dim=1), 0.0, 1.0)


def gripper_closing_near_cube(
    env: ManagerBasedRLEnv,
    std: float,
    open_pos: float = 0.8,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["gripper"]),
) -> torch.Tensor:
    """Bridge reward between reach and lift: close the jaws *while* at the cube.

    Pure reach plateaus in a hover-at-the-cube local optimum, because the
    binary gripper has to fire at exactly the right moment for the lift bonus
    to ever pay out. This term hands out partial credit for that precursor:
    proximity (exp kernel) multiplied by how closed the gripper is.
    """
    cube: RigidObject = env.scene[cube_cfg.name]
    ee: FrameTransformer = env.scene[ee_cfg.name]
    robot = env.scene[robot_cfg.name]
    # robot_cfg must come through the term's params (the manager only resolves
    # SceneEntityCfg instances it finds there); guard against the unresolved default
    if isinstance(robot_cfg.joint_ids, slice):
        raise ValueError(
            "gripper_closing_near_cube: pass robot_cfg via the reward term's params, "
            'e.g. params={"robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"])}'
        )
    d = torch.norm(cube.data.root_pos_w - ee.data.target_pos_w[..., _ee_index(ee), :], dim=1)
    near = torch.exp(-d / std)
    grip = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
    closed = torch.clamp(1.0 - grip / open_pos, 0.0, 1.0)  # 1 = fully closed
    return near * closed


def cube_lifted(
    env: ManagerBasedRLEnv,
    min_height: float,
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1.0 while the cube's center is above ``min_height`` (world z), else 0."""
    cube: RigidObject = env.scene[cube_cfg.name]
    return (cube.data.root_pos_w[:, 2] > min_height).float()


def cube_to_goal(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """exp(-d/std) on cube->goal distance, gated on the cube being lifted.

    The goal is commanded in the robot base frame, so it is first transformed
    to world coordinates with the robot's root pose.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    cube: RigidObject = env.scene[cube_cfg.name]
    goal_b = env.command_manager.get_command(command_name)[:, :3]
    goal_w, _ = combine_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], goal_b
    )
    d = torch.norm(goal_w - cube.data.root_pos_w[:, :3], dim=1)
    lifted = cube.data.root_pos_w[:, 2] > min_height
    return lifted.float() * torch.exp(-d / std)
