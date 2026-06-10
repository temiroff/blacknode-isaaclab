"""Watch a trained blacknode SO-101 cube-lift policy.

Usage (from the project root, inside the project venv):
    uv run python blacknode/play.py                          # latest checkpoint
    uv run python blacknode/play.py --checkpoint path/to/model.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate the blacknode SO-101 cube-lift policy.")
parser.add_argument("--task", type=str, default="Blacknode-SO101-Lift-Play-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model .pt (default: latest run).")
parser.add_argument("--real-time", action="store_true", help="Throttle to wall-clock speed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg

import so101_lift  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ckpt = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[blacknode] loading checkpoint: {ckpt}")

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        if args.real_time:
            leftover = dt - (time.time() - t0)
            if leftover > 0:
                time.sleep(leftover)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
