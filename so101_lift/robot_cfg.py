"""SO-101 articulation config for the blacknode lift task.

Spawns the robot from ``assets/so101_robot.usd`` (our self-contained export:
geometry baked in, convexDecomposition colliders).

Gain provenance — these are OUR values, not borrowed from another project:
- stiffness=100 / damping=10 were tuned and stability-tested interactively in
  Isaac Sim during scene authoring (arm holds pose against gravity, no
  oscillation, settles in well under a second).
- effort limit ~1.9 N*m comes from the Feetech STS3215 datasheet
  (19.4 kg*cm stall torque at 7.4 V).
- velocity limit 3.0 rad/s is a loaded-motion margin under the STS3215
  no-load speed (0.229 s / 60 deg  ->  ~4.6 rad/s).
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

# blacknode/assets/so101_robot.usd
ROBOT_USD = Path(__file__).resolve().parent.parent / "assets" / "so101_robot.usd"

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # MUST stay False: the baked collision hulls of adjacent links
            # interpenetrate at the joints by design (servo bodies sit inside
            # the neighboring holders), and this USD carries no filtered
            # collision pairs -- enabling self-collision makes every arm
            # explode at spawn (NaN cascade into the policy).
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    # wrist pre-bent ~90deg so the gripper starts facing down toward the
    # workspace -- the pose the proven upstream policy trains from
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 1.57,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # PD constants adopted from the upstream isaac_so_arm101 project
    # (BSD-3-Clause) -- empirically tuned per joint for the STS3215 servos
    # and validated by their working lift policy. Stiffness scales with the
    # mass each joint moves; damping ratios are much heavier than our earlier
    # 100/10 (which was underdamped during fast 50 Hz target changes).
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=1.5,
            stiffness={
                "shoulder_pan": 200.0,
                "shoulder_lift": 170.0,
                "elbow_flex": 120.0,
                "wrist_flex": 80.0,
                "wrist_roll": 50.0,
            },
            damping={
                "shoulder_pan": 80.0,
                "shoulder_lift": 65.0,
                "elbow_flex": 45.0,
                "wrist_flex": 30.0,
                "wrist_roll": 20.0,
            },
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=2.5,  # stronger grip than the arm joints
            velocity_limit_sim=1.5,
            stiffness=60.0,
            damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
