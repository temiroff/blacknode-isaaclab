# blacknode-isaaclab

Reinforcement learning tasks built on [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

**Currently inside — first robot, first task:**

> 🤖 [SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) robot arm · 🎯 **cube lift** — reach,
> grasp, and lift a cube to a randomized goal, trained with PPO (`rsl_rl`).

A **self-contained, manager-based RL task**: thousands of arm instances train in parallel on the GPU,
and the cube spawns at a random position every episode — the policy has to *learn* to reach it,
nothing is hardcoded. More tasks for the SO-101 (reach, pick-and-place) are the natural next steps;
the layout already accommodates them as sibling packages (`so101_reach/`, `so101_pickplace/`, …)
sharing the same robot asset and launchers.

---

## Table of contents
1. [How it works (the 60-second version)](#how-it-works-the-60-second-version)
2. [Prerequisites](#prerequisites)
3. [Quick start](#quick-start)
4. [Repository layout](#repository-layout)
5. [File-by-file walkthrough](#file-by-file-walkthrough)
6. [The MDP in detail](#the-mdp-in-detail)
7. [Command-line flags](#command-line-flags)
8. [Customizing the task](#customizing-the-task)
9. [How the robot asset was prepared](#how-the-robot-asset-was-prepared)
10. [Troubleshooting](#troubleshooting)
11. [Credits & license](#credits--license)

---

## How it works (the 60-second version)

Reinforcement learning trains a **policy** (a neural network) by trial and error. Each timestep:

```
        ┌─────────────┐   action (joint targets)   ┌──────────────┐
        │   Policy    │ ─────────────────────────▶ │ Environment  │
        │  (network)  │                            │ (Isaac Lab)  │
        └─────────────┘ ◀───────────────────────── └──────────────┘
              ▲           observation + reward
              └──────────────  PPO updates  ──────────────┘
```

- **Observation** — what the policy sees: joint angles/velocities, the cube's position, the goal, its last action.
- **Action** — what it outputs: target angles for the 5 arm joints + open/close for the gripper.
- **Reward** — the score it maximizes, in three gated stages: get the gripper **near the cube**, then
  **lift** the cube, then **carry** it to the goal. Each stage only pays out after the previous one,
  so the policy can't cheat.
- **Reset** — every episode the cube lands at a **random** spot, forcing generalization.

Isaac Lab runs thousands of copies of this loop simultaneously on the GPU.

The **manager-based** workflow means each part of the MDP (observations, rewards, terminations,
reset events) is a small declarative config assembled by Isaac Lab's managers — no monolithic env class.

---

## Prerequisites

- **OS:** Windows or Linux with an NVIDIA RTX GPU (CUDA).
- **Python 3.11** and [`uv`](https://docs.astral.sh/uv/).

Everything else (Isaac Lab 2.3.0, Isaac Sim, torch) is declared in `pyproject.toml` and installed
with one command from the repo root:

```bash
uv sync          # downloads ~10+ GB on first run (Isaac Sim wheels)
```

Verify the environment:

```bash
uv run python -c "import isaaclab, isaaclab_rl, rsl_rl; print('Isaac Lab OK')"
```

---

## Quick start

From the repo root:

```bash
# 0) one-time environment setup
uv sync

# 1) sanity check — a few envs, windowed, so you can watch
uv run python train.py --num_envs 64

# 2) real training — many envs, headless, fast
uv run python train.py --num_envs 4096 --headless

# 3) watch the trained policy
uv run python play.py

# 4) training curves
uv run tensorboard --logdir logs/rsl_rl/blacknode_so101_lift
```

Checkpoints and logs land in `logs/rsl_rl/blacknode_so101_lift/<timestamp>/`.

---

## Repository layout

```
blacknode/
├── assets/
│   └── so101_robot.usd            # robot-only USD (geometry baked in, no external refs)
├── so101_lift/                    # the RL task package
│   ├── __init__.py                # registers the Gym environment IDs
│   ├── robot_cfg.py               # SO-101 articulation: USD spawn + actuator gains
│   ├── lift_env_cfg.py            # the MDP: scene, obs, rewards, terminations, events
│   ├── joint_pos_env_cfg.py       # concrete config: robot + cube + end-effector frame
│   ├── mdp/
│   │   ├── __init__.py            # built-in Isaac Lab terms + the custom ones below
│   │   ├── observations.py        # cube position in the robot base frame
│   │   └── rewards.py             # reach / lift / carry-to-goal reward stages
│   └── agents/
│       └── rsl_rl_ppo_cfg.py      # PPO hyperparameters
├── pyproject.toml                 # environment definition (uv sync recreates it)
├── train.py                       # training entry point
├── play.py                        # evaluation entry point
└── README.md                      # this file
```

---

## File-by-file walkthrough

### `assets/so101_robot.usd`
A single self-contained USD with **only** the robot articulation: meshes baked in, colliders set to
`convexDecomposition`, no props. Exported from our cleaned authoring scene
(see [asset preparation](#how-the-robot-asset-was-prepared)), so the repo has no external mesh
dependencies.

### `so101_lift/robot_cfg.py` — the robot
Defines `SO101_CFG` (an `ArticulationCfg`):
- **`spawn`** — loads `assets/so101_robot.usd`.
- **`init_state`** — all joints at zero: the neutral pose we verified stable on the ground plane.
- **`actuators`** — PD gains **stiffness 100 / damping 10**, tuned and stability-tested interactively
  in Isaac Sim (holds pose against gravity, no oscillation). Effort limit 1.9 N·m and velocity limit
  3.0 rad/s are derived from the Feetech STS3215 servo datasheet (19.4 kg·cm stall @ 7.4 V,
  0.229 s/60° no-load). These override whatever is authored in the USD — tune RL behavior here.

### `so101_lift/lift_env_cfg.py` — the MDP (the heart of the task)
The abstract environment, as declarative `@configclass` managers:
- **`CubeLiftSceneCfg`** — ground plane + dome light + `MISSING` placeholders for robot/cube/ee-frame.
- **`CommandsCfg`** — samples one goal pose per episode (where to carry the cube), in the robot base frame.
- **`ActionsCfg`** — placeholders for the arm and gripper action terms.
- **`ObservationsCfg`** — the policy input vector (see [MDP detail](#the-mdp-in-detail)).
- **`EventCfg`** — on reset: restore the scene, then randomize the cube's x/y.
- **`RewardsCfg`** — the three-stage gated reward + smoothness penalties.
- **`TerminationsCfg`** — timeout, or cube fell below the floor.
- **`CubeLiftEnvCfg`** — bundles everything; physics at 100 Hz, policy at 50 Hz, 6 s episodes.

### `so101_lift/joint_pos_env_cfg.py` — the concrete task
Subclasses `CubeLiftEnvCfg` and fills in the blanks:
- robot = `SO101_CFG`,
- **arm action**: joint-position control of `shoulder_*`, `elbow_flex`, `wrist_*`; **gripper action**:
  binary open (0.8 rad) / close (0.0 rad),
- **cube**: a spawned 2.5 cm / 30 g cuboid primitive (`sim_utils.CuboidCfg`) — no asset download needed;
  size and mass match what we grasp-tested interactively,
- **end-effector frame**: the robot's own `gripper_frame_link` (the TCP frame in the SO-101
  kinematics), so no hand-tuned offset is needed.

Also defines `SO101LiftCubeEnvCfg_PLAY`: 16 envs, observation noise off — for watching a trained policy.

### `so101_lift/__init__.py` — registration
Registers the Gym IDs `Blacknode-SO101-Lift-v0` (train) and `Blacknode-SO101-Lift-Play-v0` (eval),
each pointing at an env config + the PPO agent config.

### `so101_lift/mdp/` — custom MDP terms
- **`observations.py`** — `cube_position_b`: the cube position transformed into each robot's base
  frame (world coordinates are meaningless to a policy cloned across a grid of environments).
- **`rewards.py`** — the three stages:
  `ee_cube_distance` (exp-kernel reach), `cube_lifted` (height bonus), `cube_to_goal`
  (exp-kernel goal tracking, paid only while lifted).
- **`__init__.py`** — re-exports Isaac Lab's built-in MDP terms and adds ours, so configs can
  uniformly write `mdp.<term>`.

### `so101_lift/agents/rsl_rl_ppo_cfg.py` — the learner
`SO101LiftPPORunnerCfg`: a `[256, 128]` ELU actor-critic, 32 steps/env per update, 2000 iterations,
adaptive LR from 3e-4, γ=0.99, λ=0.95.

### `train.py` / `play.py` — entry points
Small standalone launchers: boot Isaac Sim (`AppLauncher`), build the configs from the registry
(`parse_env_cfg` / `load_cfg_from_registry`), wrap the env for `rsl_rl`, then **learn** (train.py) or
**load the latest checkpoint and roll out** (play.py). Configs used for each run are dumped to
`logs/.../params/` for reproducibility.

---

## The MDP in detail

**Observation vector** (per environment, concatenated):
| Term | Meaning |
|------|---------|
| `joint_pos` | joint angles relative to defaults |
| `joint_vel` | joint velocities |
| `cube_position` | cube position in the robot base frame |
| `goal_pose` | the commanded goal pose |
| `last_action` | previous action |

**Action** (6 values): position targets for the 5 arm joints (scale 0.5, offset from defaults) +
binary gripper (open 0.8 / close 0.0 rad).

**Reward** (gated stages):
| Term | What it rewards | Weight |
|------|-----------------|--------|
| `reach` | `exp(-d/0.1)` on gripper→cube distance | +2 |
| `lift` | cube center above 5 cm | +10 |
| `place` | `exp(-d/0.15)` on cube→goal distance, **only while lifted** | +15 |
| `action_rate` | penalize jerky actions | −0.005 |
| `joint_vel` | penalize fast joint motion | −0.0005 |

**Reset:** scene restored, then the cube's x/y resampled uniformly
(x ∈ [−0.06, +0.08], y ∈ [−0.12, +0.12] around its nominal spot 22 cm in front of the base).

**Termination:** 6 s timeout, or the cube falls below the floor.

---

## Command-line flags

`train.py`:
| Flag | Default | Meaning |
|------|---------|---------|
| `--task` | `Blacknode-SO101-Lift-v0` | registered env to build |
| `--num_envs` | 4096 (from cfg) | parallel environments |
| `--headless` | off | no window — much faster |
| `--max_iterations` | 2000 (from cfg) | PPO iterations |
| `--seed` | 42 | RNG seed |
| `--resume` | off | continue from the latest checkpoint |
| `--run_name` | — | suffix for the log folder |

`play.py`: `--checkpoint` (a specific `model_*.pt`; default = latest run), `--num_envs`, `--real-time`.

Start with `--num_envs 64` windowed to confirm everything builds, then scale up headless.

---

## Customizing the task

- **Cube size/mass:** `CUBE_SIZE` / `CUBE_MASS` at the top of `joint_pos_env_cfg.py`.
- **Reward shaping:** weights and kernels in `RewardsCfg` (`lift_env_cfg.py`); new terms go in
  `mdp/rewards.py`.
- **Harder randomization:** widen `randomize_cube` ranges in `EventCfg`.
- **Robot gains:** `stiffness` / `damping` in `robot_cfg.py`.
- **Network / PPO:** `agents/rsl_rl_ppo_cfg.py`.
- **Cartesian control:** swap `JointPositionActionCfg` for
  `DifferentialInverseKinematicsActionCfg` in `joint_pos_env_cfg.py`.

---

## How the robot asset was prepared

`assets/so101_robot.usd` was produced from a hand-cleaned authoring scene:
1. Imported the SO-ARM101 from URDF and **flattened** the shared-mesh references so all geometry is
   baked into one file (no external refs, nothing breaks when the file moves).
2. Set collision approximation to **`convexDecomposition`** on all 17 collision meshes — accurate
   contact at a cost that scales to thousands of parallel envs (SDF colliders, which we use in the
   single-robot authoring scene for high-fidelity grasping, are too heavy at RL scale).
3. Stripped authoring-only props (ground, cube), keeping just the articulation.
4. Set `so101_new_calib` as the default prim.

---

## Troubleshooting

| Symptom | Fix |
|--------|-----|
| `ModuleNotFoundError: isaaclab` | Run `uv sync` once, then always launch via `uv run python train.py …` |
| First launch hangs / black window | Shader compilation + physics cooking; the first run is slow. Prefer `--headless`. |
| `gripper_frame_link` not found | The ee-frame and goal command reference this body; if your USD export renamed it, update `joint_pos_env_cfg.py`. |
| Out-of-memory at 4096 envs | Lower `--num_envs` (2048, 1024). |
| Robot unstable / NaNs | Lower actuator stiffness in `robot_cfg.py` or reduce `sim.dt`. |
| GPU contention | Close any interactive Isaac Sim GUI before launching training. |

---

## Credits & license

- **Task code** (everything under `blacknode/`): written from scratch for this project, using
  [NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3-Clause) as a framework dependency.
- **Robot model:** the SO-ARM101 design, URDF and meshes are by
  [The Robot Studio](https://github.com/TheRobotStudio/SO-ARM100) — `assets/so101_robot.usd` is a
  USD conversion of their robot description and remains subject to their license/attribution.
- Thanks to the broader SO-ARM + LeRobot community for making this hardware approachable.

The code in this repository is licensed under the [Apache License 2.0](LICENSE). The robot-model
attribution above applies to the USD asset.
