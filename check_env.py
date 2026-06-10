"""Build the environment WITHOUT training, so you can inspect and tweak it.

The viewport is a full Isaac Sim session: pause it, dig through the Stage
tree (/World/envs/env_0/...), select prims, read the Property panel, toggle
physics overlays -- then port the numbers you like back into the cfg files.

Usage (from the project root):
    uv run python blacknode/check_env.py                     # 4 envs, zero actions
    uv run python blacknode/check_env.py --mode random       # random actions
    uv run python blacknode/check_env.py --num_envs 16
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect the blacknode SO-101 env without training.")
parser.add_argument("--task", type=str, default="Blacknode-SO101-Lift-v0", help="Registered task id.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments.")
parser.add_argument("--mode", choices=["zero", "random"], default="zero", help="Action source.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import so101_lift  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg)

    # the console above this line already shows the manager tables:
    # every observation / reward / termination / event term that got built
    obs, _ = env.reset()
    n_act = env.unwrapped.action_manager.total_action_dim
    device = env.unwrapped.device
    print(f"\n[check_env] obs shape: {obs['policy'].shape}  action dim: {n_act}")
    print(f"[check_env] mode={args.mode} -- stepping until you close the window\n")

    step = 0
    while simulation_app.is_running():
        if args.mode == "zero":
            actions = torch.zeros((args.num_envs, n_act), device=device)
        else:
            actions = 2.0 * torch.rand((args.num_envs, n_act), device=device) - 1.0
        obs, rew, terminated, truncated, info = env.step(actions)
        step += 1
        if step % 100 == 0:
            print(f"step {step:6d}  mean reward {rew.mean().item():+.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
