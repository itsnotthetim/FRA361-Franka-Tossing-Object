# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b

## Tossing

def ee_to_goal(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
) -> torch.Tensor:
    """Δp_ee→goal = p_goal − p_ee in world frame, shape (N,3)."""
    robot = env.scene[robot_cfg.name]
    basket = env.scene[basket_cfg.name]

    ee_idx = robot.body_names.index("panda_hand")
    ee_pos = robot.data.body_state_w[:, ee_idx, :3]           # (N,3)
    goal_pos = basket.data.root_pos_w[:, :3]                  # (N,3)

    return goal_pos - ee_pos

def gripper_status(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """1.0 if gripper is closed (<0.07m gap), else 0.0.  Shape (N,1)."""
    robot = env.scene[robot_cfg.name]
    fingers = robot.data.joint_pos[:, -2:]                     # (N,2)
    gap = fingers.sum(dim=1)                                   # (N,)
    closed = (gap < 0.07).to(torch.float32)                    # (N,)

    return closed.unsqueeze(1)   


def basket_root_pos_w(
    env,
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
) -> torch.Tensor:
    """World-frame (x,y,z) of the basket centre, shape (N,3)."""
    basket: RigidObject = env.scene[basket_cfg.name]
    # root_pos_w holds (x,y,z, quat), select first 3
    return basket.data.root_pos_w[:, :3]
