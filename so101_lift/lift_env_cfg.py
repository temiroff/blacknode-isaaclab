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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass

from . import mdp


@configclass
class CubeLiftSceneCfg(InteractiveSceneCfg):
    """Ground + light; robot, cube and ee-frame are set by the concrete cfg."""

    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
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
    # stage 1: get the gripper to the cube
    reach = RewTerm(func=mdp.ee_cube_distance, params={"std": 0.1}, weight=2.0)
    # stage 1.5: bridge -- close the jaws while at the cube, so the policy
    # discovers grasping instead of hovering forever in the reach optimum
    grasp = RewTerm(
        func=mdp.gripper_closing_near_cube,
        params={"std": 0.05, "robot_cfg": SceneEntityCfg("robot", joint_names=["gripper"])},
        weight=1.0,
    )
    # stage 2: get the cube off the ground (cube center starts at ~0.013 m)
    lift = RewTerm(func=mdp.cube_lifted, params={"min_height": 0.05}, weight=10.0)
    # stage 3: carry the lifted cube to the commanded goal
    place = RewTerm(
        func=mdp.cube_to_goal,
        params={"std": 0.15, "min_height": 0.05, "command_name": "object_pose"},
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
    scene: CubeLiftSceneCfg = CubeLiftSceneCfg(num_envs=4096, env_spacing=2.0)
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
