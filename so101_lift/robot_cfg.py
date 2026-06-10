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
    # the same neutral pose we verified stable on the ground plane in the GUI
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*", "elbow_flex", "wrist_.*"],
            effort_limit_sim=1.9,
            velocity_limit_sim=3.0,
            stiffness=100.0,
            damping=10.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=1.9,
            velocity_limit_sim=3.0,
            stiffness=100.0,
            damping=10.0,
        ),
    },
    soft_joint_pos_limit_factor=0.9,
)
