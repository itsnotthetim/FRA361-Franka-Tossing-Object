# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations for the lift task.

The functions can be passed to the :class:`omni.isaac.lab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Termination condition for the object reaching the goal position.

    Args:
        env: The environment.
        command_name: The name of the command that is used to control the object.
        threshold: The threshold for the object to reach the goal position. Defaults to 0.02.
        robot_cfg: The robot configuration. Defaults to SceneEntityCfg("robot").
        object_cfg: The object configuration. Defaults to SceneEntityCfg("object").

    """
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
    return distance < threshold

def object_in_basket(
    env: ManagerBasedRLEnv,
    eps_xy: float                = 0.05,   # XY tolerance (m)
    rim_clearance: float         = 0.02,   # object must fall below rim − clearance
    object_cfg: SceneEntityCfg   = SceneEntityCfg("object"),
    basket_cfg: SceneEntityCfg   = SceneEntityCfg("basket"),
) -> torch.Tensor:
    """
    Termination condition: return True for each env where the cube has
    • fallen to (or below) basket-rim height minus `rim_clearance`, AND
    • is horizontally within `eps_xy` of the basket centre.

    Returns: Bool tensor of shape (N,) suitable for DoneTerm.
    """
    # world-frame positions
    p_obj = env.scene[object_cfg.name].data.root_pos_w[:, :3]   # (N,3)
    p_goal = env.scene[basket_cfg.name].data.root_pos_w[:, :3]  # (N,3)

    # 1) XY proximity
    horiz_dist = torch.linalg.norm(p_obj[:, :2] - p_goal[:, :2], dim=1)   # (N,)
    inside_xy  = horiz_dist < eps_xy                                      # bool (N,)

    # 2) Z below rim
    rim_height = p_goal[:, 2]                                             # assume USD origin at rim
    below_rim  = p_obj[:, 2] < (rim_height - rim_clearance)               # bool (N,)

    return inside_xy & below_rim            
