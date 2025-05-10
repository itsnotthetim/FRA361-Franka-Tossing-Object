# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


## Tossing

def acc_term(env: ManagerBasedRLEnv, k_acc: float) -> torch.Tensor:
    """
    Accuracy term: −k_acc * ||p_final − p_goal||^2, but only applied at the final timestep.
    """
    # final object position in world frame
    final_pos = env.scene["object"].data.root_state_w[:, :3]
    # basket goal position in world frame
    goal_pos = env.scene["basket"].data.root_state_w[:, :3]
    # squared error
    err2 = torch.sum((final_pos - goal_pos) ** 2, dim=-1)

    # get a (N,) bool tensor which envs just terminated
    done = env.termination_manager.dones

    # apply the penalty only for envs where done==True
    return torch.where(done, -k_acc * err2, torch.zeros_like(err2))


def success_bonus(env: ManagerBasedRLEnv, eps: float) -> torch.Tensor:
    """
    Success bonus: if final object within eps of goal, only at terminal timestep.
    """
    final_pos = env.scene["object"].data.root_state_w[:, :3]
    goal_pos  = env.scene["basket"].data.root_state_w[:, :3]

    dist = torch.linalg.norm(final_pos - goal_pos, dim=-1)        # (N,)
    succ = (dist < eps).to(dist.dtype)                            # (N,) → 0.0 or 1.0
    done = env.termination_manager.dones.to(dist.dtype)           # (N,) → 0.0 or 1.0

    # only pay out the bonus at the end
    return succ * done

def energy_penalty(env: ManagerBasedRLEnv, alpha: float) -> torch.Tensor:
    v = env.scene["robot"].data.joint_vel  # (N, J)
    return -alpha * torch.sum(v**2, dim=-1)