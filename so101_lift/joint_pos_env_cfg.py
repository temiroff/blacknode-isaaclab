"""Concrete SO-101 cube-lift config: robot, cube and end-effector frame.

The cube is a plain spawned cuboid (no external asset dependency): 2.5 cm,
30 g -- the same size/weight we validated interactively in the GUI scene.
The end-effector frame uses the robot's own ``gripper_frame_link``, the
dedicated TCP frame that ships in the SO-101 kinematics -- no hand-tuned
offset needed.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
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
        # scale 1.0 with clip_actions=1.0 gives +-1 rad (+-57 deg) of authority
        # around the default "ready" pose -- wide enough to cover the grasp
        # envelope (scale 0.5 capped joints at +-28 deg and could not reach it)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_.*", "elbow_flex", "wrist_.*"],
            scale=1.0,
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

        # end-effector frame: the grasp point all distance rewards use.
        # debug vis kept ON but with the frame markers shrunk to invisible --
        # what remains is just the yellow connector line pinch -> cube,
        # a clean "this is my goal" indicator per robot. The SOURCE sits at
        # the pinch point itself (lines start at the source), so the line
        # begins at the jaws, not the wrist.
        ee_marker = FRAME_MARKER_CFG.copy()
        ee_marker.markers["frame"].scale = (0.0001, 0.0001, 0.0001)  # hide arrows, keep lines
        ee_marker.prim_path = "/Visuals/EEFrame"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
            source_frame_offset=OffsetCfg(pos=[-0.018, 0.0, -0.008]),
            debug_vis=True,
            visualizer_cfg=ee_marker,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                    name="end_effector",
                    # the upstream project's PROVEN grasp point, converted into
                    # the TCP frame: the pinch line sits ~1.8cm lateral (-X, the
                    # fixed jaw is off-axis) and ~0.8cm back from the tips (-Z;
                    # +Z is the out-of-jaws axis per the URDF's rpy=(0,pi,0))
                    offset=OffsetCfg(pos=[-0.018, 0.0, -0.008]),
                ),
                # tracked only so the debug line connects gripper -> cube
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Cube",
                    name="cube_center",
                ),
            ],
        )

        # visualization-only sensor: draws ONLY the frame arrows at the pinch
        # point. Source and target are the same prim with the same offset, so
        # the two drawn frames overlap into one set of arrows and the
        # connector line is zero-length (invisible).
        pinch_marker = FRAME_MARKER_CFG.copy()
        pinch_marker.markers["frame"].scale = (0.03, 0.03, 0.03)
        pinch_marker.prim_path = "/Visuals/PinchFrame"
        self.scene.ee_vis = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
            source_frame_offset=OffsetCfg(pos=[-0.018, 0.0, -0.008]),
            debug_vis=True,
            visualizer_cfg=pinch_marker,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/gripper_frame_link",
                    name="pinch_point",
                    offset=OffsetCfg(pos=[-0.018, 0.0, -0.008]),
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
