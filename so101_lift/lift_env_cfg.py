"""Manager-based RL environment for the blacknode SO-101 cube-lift task.

Abstract definition: the scene entities marked ``MISSING`` (robot, cube,
end-effector frame, action terms) are filled in by ``joint_pos_env_cfg.py``.

Reward design (three gated stages, see ``mdp/rewards.py``):
    reach (w=2) -> lift bonus (w=10) -> carry-to-goal (w=15, only while lifted)
plus small smoothness penalties. No curriculum: the penalties are mild enough
to leave on from the start.
"""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class CubeLiftSceneCfg(InteractiveSceneCfg):
    """Ground + light; robot, cube and ee-frame are set by the concrete cfg."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    ee_vis: FrameTransformerCfg = MISSING  # visualization-only: arrows at the pinch point
    object: RigidObjectCfg = MISSING

    # black glossy floor: a large static collision slab with a dark, low-
    # roughness, metallic surface (reflective under RTX). Top face at z=0.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.1]),
        spawn=sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.2),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # satin black: mirror settings (roughness ~0.1, metallic ~0.7)
            # produce a huge white specular pool + denoiser grain that reads
            # as a "second floor" -- keep the sheen subtle instead
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.03, 0.03, 0.03),
                roughness=0.35,
                metallic=0.25,
            ),
        ),
    )

    # dark dome: a bright white dome reflects off the glossy black floor and
    # washes it out white -- keep ambient dim and let the key light do the work
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.12, 0.12, 0.12), intensity=600.0),
    )

    # angled key light: illuminates the robots, reflects as a highlight
    # instead of a full-sky white wash
    key_light = AssetBaseCfg(
        prim_path="/World/keyLight",
        init_state=AssetBaseCfg.InitialStateCfg(rot=[0.924, 0.383, 0.0, 0.0]),
        spawn=sim_utils.DistantLightCfg(color=(1.0, 1.0, 1.0), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    """Where to carry the cube: a pose sampled in the robot base frame."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set by the concrete cfg
        resampling_time_range=(6.0, 6.0),  # one goal per episode
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.15, 0.30),
            pos_y=(-0.15, 0.15),
            pos_z=(0.12, 0.28),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cube_position = ObsTerm(func=mdp.cube_position_b)
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # new cube spot every episode -- the policy must generalize, not memorize
    randomize_cube = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.06, 0.08), "y": (-0.12, 0.12), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class RewardsCfg:
    # stage 1: get the gripper to the cube (sharp kernel, proven upstream)
    reach = RewTerm(func=mdp.ee_cube_distance, params={"std": 0.05}, weight=2.0)
    # stage 1a: face the cube -- teaches the wrist to rotate the jaws into a
    # graspable orientation instead of arriving sideways
    point_at_cube = RewTerm(func=mdp.ee_pointing_at_cube, weight=1.0)
    # stage 1b: jaws OPEN while approaching (5-15 cm out). Weight kept LOW:
    # at the handoff distance, closing must pay more than staying open, or
    # the policy hovers open-jawed forever (observed: lift never amplified)
    approach_open = RewTerm(
        func=mdp.gripper_open_on_approach,
        params={
            "open_pos": 0.5,
            "near_dist": 0.05,
            "far_dist": 0.15,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=0.25,
    )
    # stage 1.5: reward an ACTUAL hold -- the gripper joint physically
    # blocked partway by the cube. "Closed near cube" shaping failed twice:
    # too weak -> hover open forever; too strong -> approach shut and poke.
    # A blocked joint can only happen when the cube is truly between the
    # fingers, and physics itself enforces the open->around->close sequence.
    grasp = RewTerm(
        func=mdp.cube_between_jaws,
        params={
            "std": 0.03,
            "min_blocked": 0.06,
            "max_blocked": 0.6,
            "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
        },
        weight=4.0,
    )
    # stage 2: get the cube off the ground. LOW bar (cube center rests at
    # ~0.013 m; 0.03 = a ~1.7 cm raise): the proven upstream task pays lift
    # for barely a 1 cm hop, which gives exploration constant taste of the
    # bonus -- a 5 cm cliff was never discovered in our earlier runs
    lift = RewTerm(func=mdp.cube_lifted, params={"min_height": 0.03}, weight=10.0)
    # stage 3: carry the lifted cube to the commanded goal
    place = RewTerm(
        func=mdp.cube_to_goal,
        params={"std": 0.15, "min_height": 0.03, "command_name": "object_pose"},
        weight=15.0,
    )
    # smoothness
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0005, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cube_fell = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CubeLiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: CubeLiftSceneCfg = CubeLiftSceneCfg(num_envs=4096, env_spacing=1.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2          # policy at 50 Hz
        self.episode_length_s = 6.0
        self.viewer.eye = (1.5, 1.5, 1.0)
        self.sim.dt = 0.01           # physics at 100 Hz
        self.sim.render_interval = self.decimation

        # SDF jaw colliders x 4096 envs overflow the default GPU collision
        # stack (PhysX reported needing ~1.4 GB and DROPPED contacts, which
        # silently breaks grasping). Give it headroom; if this OOMs your GPU,
        # train with --num_envs 2048 instead.
        self.sim.physx.gpu_collision_stack_size = 2**31  # 2 GB
        # goal marker: small green sphere (orientation is irrelevant for this task)
        self.commands.object_pose.goal_pose_visualizer_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
                ),
            },
        )
        # the command's current-pose marker sits at the raw TCP (jaw tip) and
        # duplicates the ee_frame vis -- shrink it to nearly invisible
        self.commands.object_pose.current_pose_visualizer_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)