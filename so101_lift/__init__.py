"""blacknode SO-101 lift task — gym registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Blacknode-SO101-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SO101LiftCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Blacknode-SO101-Lift-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SO101LiftCubeEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SO101LiftPPORunnerCfg",
    },
    disable_env_checker=True,
)
