# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import torch
import numpy as np
import time

import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils import configclass


import sys
import os

# Step 1: Add 'exts/tim' directory to sys.path
current_dir = os.path.dirname(__file__)
target_path = os.path.abspath(os.path.join(current_dir, '..'))
if target_path not in sys.path:
    sys.path.append(target_path)

from exts.tim.tim.tasks.lift_n_toss.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg
# from omni.isaac.lab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import FrankaCubeStackEnvCfg

import random



def main():
    print(current_dir)
    env_cfg = FrankaCubeLiftEnvCfg()
    
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 6.0
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env.reset()

    robot_entity = env.scene["robot"]
    cube_entity = env.scene["object"]
    basket_entity = env.scene["basket"]

    ee_idx = robot_entity.body_names.index("panda_hand")


    count = 0
    ee_offset_z = 0.107  # Explicit EE offset (standard Franka EE offset)


    while simulation_app.is_running():
        with torch.inference_mode():
            if count % 300 == 0:
                env.reset()
                print("[INFO]: Resetting environment explicitly.")
                phase = 0
                count = 0
                save_pos = 0
                dur = 0

            # joint_efforts = torch.randn_like(env.action_manager.action)
            joint_efforts = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # step the environment
            obs, rew, terminated, truncated, info = env.step(joint_efforts)

            # robot_pos = robot_entity.data.root_pos_w[0].cpu().numpy

            ee_current_pos = robot_entity.data.body_state_w[0, ee_idx, :3].cpu().numpy()

            cube_position = cube_entity.data.root_pos_w[0].cpu().numpy()
                    # print("Randomized object position:", cube_position)
            cube_pos = cube_position.tolist()

            height = cube_entity.data.root_pos_w[:, 2]

            basket_position = basket_entity.data.root_pos_w[0].cpu().numpy()
                    # print("Randomized object position:", cube_position)
            basket_pos = basket_position.tolist()

            # print(
            #     f"ee_pos:   {ee_current_pos}\n"
            #     f"cube_pos: {cube_pos}\n"
            #     f"cube_height: {height}\n"
            #     f"basket_pos: {basket_pos}"
            # )
            
            # final_pos = env.scene["object"].data.root_state_w[:, :3]
            # goal_pos  = env.scene["basket"].data.root_state_w[:, :3]

            # print(torch.linalg.norm(final_pos - goal_pos))
        

            count += 1

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()


