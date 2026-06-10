"""Train the blacknode SO-101 cube-lift policy with PPO (rsl_rl).

Usage (from the project root, inside the project venv):
    uv run python blacknode/train.py --num_envs 64                    # windowed sanity check
    uv run python blacknode/train.py --num_envs 4096 --headless       # full training
"""

import argparse
import os
import sys
from datetime import datetime

# make the task package importable regardless of where this is launched from
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the blacknode SO-101 cube-lift policy.")
parser.add_argument("--task", type=str, default="Blacknode-SO101-Lift-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=None, help="PPO iterations (default from agent cfg).")
parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
parser.add_argument("--run_name", type=str, default=None, help="Suffix for the log directory name.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# boot Isaac Sim before any isaaclab imports that need it
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import so101_lift  # noqa: F401  -- registers Blacknode-SO101-Lift-v0

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    # build configs from the registry
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    agent_cfg.seed = args.seed
    env_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.run_name is not None:
        agent_cfg.run_name = args.run_name

    # log directory: logs/rsl_rl/<experiment>/<timestamp>[_run_name]
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[blacknode] logging to: {log_dir}")

    # environment + rsl_rl wrapper
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args.resume:
        ckpt = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[blacknode] resuming from: {ckpt}")
        runner.load(ckpt)

    # keep a record of the exact configs used for this run
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
