# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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


# def object_in_basket(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
#     threshold: float = 0.05,
# ) -> torch.Tensor:
#     """
#     Terminate when the object’s centre is within `threshold` meters of the basket’s centre.

#     Args:
#         env: the IsaacLab RL environment
#         object_cfg: SceneEntityCfg for the cube
#         basket_cfg: SceneEntityCfg for the basket prim
#         threshold: radius around basket centre signifying “in basket”
#     Returns:
#         mask of shape (num_envs,) where True indicates termination
#     """
#     obj: RigidObject    = env.scene[object_cfg.name]
#     basket: RigidObject = env.scene[basket_cfg.name]
#     # world‐frame centres, shape (N,3)
#     p_obj    = obj.data.root_pos_w[:, :3]
#     p_basket = basket.data.root_pos_w[:, :3]
#     # compute Euclidean distance
#     dist = torch.norm(p_obj - p_basket, dim=1)
#     # terminate when inside basket radius
#     return dist < threshold


def object_in_square_basket(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    half_extents: Tuple[float, float] = (0.10, 0.12),
) -> torch.Tensor:
    """
    Terminate when the object’s centre is within the square basket area.

    Args:
        env: The IsaacLab RL environment.
        object_cfg: SceneEntityCfg for the cube.
        basket_cfg: SceneEntityCfg for the basket prim.
        half_extents: (hx, hy) half-widths of the basket in its local x/y axes.
    Returns:
        mask of shape (num_envs,) where True indicates the object is inside the square.
    """
    # Fetch object & basket world positions
    obj: RigidObject    = env.scene[object_cfg.name]
    basket: RigidObject = env.scene[basket_cfg.name]
    p_obj    = obj.data.root_pos_w[:, :3]       # (N,3)
    p_basket = basket.data.root_pos_w[:, :3]    # (N,3)

    # Compute delta in world frame
    delta = p_obj - p_basket                     # (N,3)
    dx, dy = delta[:, 0], delta[:, 1]

    # Check square bounds: |dx| < hx AND |dy| < hy
    hx, hy = half_extents
    inside_x = torch.abs(dx) < hx
    inside_y = torch.abs(dy) < hy

    return inside_x & inside_y
