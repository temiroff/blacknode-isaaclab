"""Concrete SO-101 cube-lift config: robot, cube and end-effector frame.

The cube is a plain spawned cuboid (no external asset dependency): 2.5 cm,
30 g -- the same size/weight we validated interactively in the GUI scene.
The end-effector frame uses the robot's own ``gripper_frame_link``, the
dedicated TCP frame that ships in the SO-101 kinematics -- no hand-tuned
offset needed.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from . import mdp
from .lift_env_cfg import CubeLiftEnvCfg
from .robot_cfg import SO101_CFG

CUBE_SIZE = 0.025  # m
CUBE_MASS = 0.03   # kg -- light enough for the STS3215 servos to lift


@configclass
class SO101LiftCubeEnvCfg(CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # robot
        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # actions: 5 arm joints as position targets, gripper as open/close
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": 0.8},
            close_command_expr={"gripper": 0.0},
        )

        # goal command is expressed relative to this body
        self.commands.object_pose.body_name = ["gripper_frame_link"]

        # cube: spawned primitive, resting on the ground in front of the base
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.22, 0.0, CUBE_SIZE / 2], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=5.0),
                mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.15)),
            ),
        )

        # end-effector frame: base_link -> gripper_frame_link (the SO-101 TCP)
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                    name="end_effector",
                    # gripper_frame_link sits at the JAW TIPS; pull the grasp
                    # point back along local -Z (into the jaws) so the reward
                    # optimum puts the cube between the fingers, not at the tip
                    offset=OffsetCfg(pos=[0.0, 0.0, -0.02]),
                ),
            ],
        )


@configclass
class SO101LiftCubeEnvCfg_PLAY(SO101LiftCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # small, deterministic scene for watching a trained policy
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
