# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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

## which add not chasing reward
# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg    = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg  = SceneEntityCfg("ee_frame"),
#     gripper_cfg: SceneEntityCfg   = SceneEntityCfg("robot"),
#     finger_threshold: float       = 0.07,
# ) -> torch.Tensor:
#     """
#     Reward the agent for reaching the object using a tanh‐kernel,
#     but only while the gripper is still closed (i.e. pre‐launch).
#     """
#     # 1) Positions
#     obj:   RigidObject     = env.scene[object_cfg.name]
#     ee:    FrameTransformer = env.scene[ee_frame_cfg.name]
#     cube_pos_w = obj.data.root_pos_w            # (N,3)
#     ee_w       = ee.data.target_pos_w[..., 0, :] # (N,3)

#     # 2) Distance → tanh‐kernel reward
#     dist = torch.norm(cube_pos_w - ee_w, dim=1)          # (N,)
#     reach_reward = 1.0 - torch.tanh(dist / std)          # (N,)

#     # 3) Gripper‐closed mask
#     robot   = env.scene[gripper_cfg.name]
#     fingers = robot.data.joint_pos[:, -2:]               # (N,2)
#     gap     = fingers.sum(dim=1)                         # (N,)
#     closed  = (gap < finger_threshold).to(torch.float32) # (N,)

#     # 4) Gate by gripper state
#     return reach_reward * closed



# def object_goal_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     minimal_height: float,
#     command_name: str,
#     robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ) -> torch.Tensor:
#     """Reward the agent for tracking the goal pose using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     robot: RigidObject = env.scene[robot_cfg.name]
#     object: RigidObject = env.scene[object_cfg.name]
#     command = env.command_manager.get_command(command_name)
#     # compute the desired position in the world frame
#     des_pos_b = command[:, :3]
#     des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
#     # distance of the end-effector to the object: (num_envs,)
#     distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
#     # rewarded if the object is lifted above the threshold
#     return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


## Tossing

# def acc_term(env: ManagerBasedRLEnv, k_acc: float) -> torch.Tensor:
#     """
#     Accuracy term: −k_acc * ||p_final − p_goal||^2, but only applied at the final timestep.
#     """
#     # final object position in world frame
#     final_pos = env.scene["object"].data.root_state_w[:, :3]
#     # basket goal position in world frame
#     goal_pos = env.scene["basket"].data.root_state_w[:, :3]
#     # squared error
#     err2 = torch.sum((final_pos - goal_pos) ** 2, dim=-1)

#     # get a (N,) bool tensor which envs just terminated
#     done = env.termination_manager.dones

#     # apply the penalty only for envs where done==True
#     return torch.where(done, -k_acc * err2, torch.zeros_like(err2))

## done active
# def acc_term(
#     env: ManagerBasedRLEnv,
#     k_acc: float,
#     gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     object_cfg:  SceneEntityCfg  = SceneEntityCfg("object"),
#     basket_cfg:  SceneEntityCfg  = SceneEntityCfg("basket"),
#     minimal_height: float        = 0.04,
#     grasp_threshold: float       = 0.07,
# ) -> torch.Tensor:
#     """
#     Accuracy term: −k_acc * ||p_final − p_goal||^2, applied only if:
#       1) the episode just terminated,
#       2) the gripper was open at termination (i.e. object released),
#       3) the object was airborne (height > minimal_height) at termination.
#     Otherwise returns zero.
#     """
#     # final object & basket positions
#     final_pos = env.scene[object_cfg.name].data.root_state_w[:, :3]   # (N,3)
#     goal_pos  = env.scene[basket_cfg.name].data.root_state_w[:, :3]   # (N,3)

#     # squared distance error
#     err2 = torch.sum((final_pos - goal_pos) ** 2, dim=-1)             # (N,)

#     # episode done mask
#     done = env.termination_manager.dones                               # (N,)

#     # gripper-open mask
#     fingers    = env.scene[gripper_cfg.name].data.joint_pos[:, -2:]    # (N,2)
#     gap        = fingers.sum(dim=1)                                    # (N,)
#     open_grip  = gap > grasp_threshold                                # (N,)

#     # airborne mask
#     height     = final_pos[:, 2]                                       # (N,)
#     airborne   = height > minimal_height                              # (N,)

#     # only penalize true throws (done & open & airborne)
#     mask = done & open_grip & airborne                                # (N,)

#     # apply penalty where mask is true
#     return torch.where(mask, -k_acc * err2, torch.zeros_like(err2))

def acc_term(
    env: ManagerBasedRLEnv,
    k_acc: float,
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg : SceneEntityCfg = SceneEntityCfg("object"),
    basket_cfg : SceneEntityCfg = SceneEntityCfg("basket"),
    minimal_height: float       = 0.04,
    grasp_threshold: float      = 0.07,
) -> torch.Tensor:
    """
    Continuous accuracy penalty:  −k_acc · ||p_obj − p_goal||²
    active at every step **after** the object has been released
    (gripper open) and is airborne (height > minimal_height).
    """
    # positions
    p_obj  = env.scene[object_cfg.name ].data.root_state_w[:, :3]   # (N,3)
    p_goal = env.scene[basket_cfg.name].data.root_state_w[:, :3]    # (N,3)

    # squared error
    err2 = torch.sum((p_obj - p_goal)**2, dim=1)                    # (N,)

    # gripper-open mask
    fingers = env.scene[gripper_cfg.name].data.joint_pos[:, -2:]    # (N,2)
    gap     = fingers.sum(dim=1)
    open_grip = gap > grasp_threshold                               # (N,)

    # airborne mask
    airborne = p_obj[:, 2] > minimal_height                         # (N,)

    # activate penalty only if released & airborne
    mask = open_grip & airborne                                     # (N,)

    return -k_acc * err2 * mask.to(err2.dtype)                      # (N,)


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

# def energy_penalty(env: ManagerBasedRLEnv, alpha: float) -> torch.Tensor:
#     v = env.scene["robot"].data.joint_vel  # (N, J)
#     return -alpha * torch.sum(v**2, dim=-1)

def throw_prep(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimal_height: float       = 0.04
) -> torch.Tensor:
    """
    Reward the agent for bringing the object closer to the basket during the grasp and throw phases,
    only when the gripper is closed (i.e., holding the object).
    """
    num_envs = env.num_envs
    device   = env.scene["robot"].data.device  # Get the device from the robot (or any other object in the scene)

    # 1) Object position and distance to basket
    obj = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2]  # Object's height (z) to check if lifted
    p_obj = obj.data.root_pos_w[:, :3]  # (N,3)
    
    basket = env.scene[basket_cfg.name]
    p_basket = basket.data.root_pos_w[:, :3]  # (N,3)

    # Compute the distance to the basket
    dist = torch.norm(p_obj - p_basket, dim=1)  # (N,)

    # 2) Gripper status (closed or open)
    robot = env.scene[gripper_cfg.name]
    fingers = robot.data.joint_pos[:, -2:]  # (N,2) - the last two joints are fingers
    gap = fingers.sum(dim=1)  # (N,)
    gripper_closed = (gap < 0.07).to(torch.float32)  # (N,)

    # 3) Only reward if object is above minimal height (object is grasped and lifted)
    lifted = height > minimal_height  # (N,) - Only apply reward if object is lifted

    # 4) Final throw prep reward: move closer to the basket while holding the object
    throw_reward = -dist * gripper_closed * lifted  # Penalize distance if not holding object or not lifted

    return throw_reward


def release_bonus(
    env: ManagerBasedRLEnv,
    basket_cfg: SceneEntityCfg = SceneEntityCfg("basket"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    minimal_height: float = 0.04,
    grasp_threshold: float = 0.07,
    release_radius: float = 0.05,
) -> torch.Tensor:
    """
    Reward the agent for releasing the object near the basket:
      – Only at the terminal step (done)
      – Only if the gripper is open (gap > grasp_threshold)
      – Only if the object was airborne (height > minimal_height)
      – Only if the final object position is within release_radius of the basket
    Returns a (N,) tensor of 1.0 where all conditions are met, else 0.0.
    """
    # 1) Episode done
    done = env.termination_manager.dones  # (N,)

    # 2) Gripper-open mask (direct syntax)
    robot   = env.scene[gripper_cfg.name]
    fingers = robot.data.joint_pos[:, -2:]         # (N,2)
    gap     = fingers.sum(dim=1)                   # (N,)
    open_grip = gap > grasp_threshold              # (N,)

    # 3) Object airborne mask
    obj = env.scene["object"]
    pos = obj.data.root_pos_w[:, :3]               # (N,3)
    height = pos[:, 2]                             # (N,)
    airborne = height > minimal_height             # (N,)

    # 4) Near-basket mask
    basket = env.scene[basket_cfg.name]
    goal_pos = basket.data.root_pos_w[:, :3]       # (N,3)
    dist = torch.norm(pos - goal_pos, dim=1)       # (N,)
    near_basket = dist < release_radius            # (N,)

    # 5) Combine all conditions
    mask = done & open_grip & airborne & near_basket

    return mask.to(torch.float32)  


def hold_penalty(
    env: ManagerBasedRLEnv,
    beta: float = 0.01,                         # per-step cost
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg : SceneEntityCfg = SceneEntityCfg("object"),
    minimal_height: float = 0.04,
    grasp_threshold: float = 0.07,
) -> torch.Tensor:
    """
    Per-step negative reward if the robot is still holding the object
    (gripper closed) *and* the object has already been lifted.
    r = −beta   whenever  (gap < grasp_threshold)  &  (z > minimal_height)
    """
    # gripper closed?
    fingers = env.scene[gripper_cfg.name].data.joint_pos[:, -2:]   # (N,2)
    gap     = fingers.sum(dim=1)                                   # (N,)
    closed  = gap < grasp_threshold                                # bool (N,)

    # object lifted?
    z = env.scene[object_cfg.name].data.root_pos_w[:, 2]           # (N,)
    lifted = z > minimal_height                                    # bool (N,)

    # penalty mask
    mask = closed & lifted                                         # bool (N,)

    return (-beta) * mask.to(torch.float32)                        # (N,)