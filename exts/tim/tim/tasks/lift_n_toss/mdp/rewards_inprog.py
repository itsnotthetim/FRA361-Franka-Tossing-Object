from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors import FrameTransformer

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg       = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg     = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg      = SceneEntityCfg("robot"),
    minimal_height: float            = 0.04,
    finger_threshold: float          = 0.07,
    in_pinch_threshold: float        = 0.03,
) -> torch.Tensor:
    """
    Reward for Phase 1 (approach & lift) and Phase 3 (post-launch recapture):
    - Phase 1: encourage reaching until object truly held.
    - Phase 3: after launch, allow chasing and re-lift
    """
    # fetch positions
    obj      = env.scene[object_cfg.name]
    ee       = env.scene[ee_frame_cfg.name]
    cube_pos = obj.data.root_pos_w             # (N,3)
    ee_pos   = ee.data.target_pos_w[...,0,:]   # (N,3)

    # reach reward
    dist  = torch.norm(cube_pos - ee_pos, dim=1)    # (N,)
    reach = 1.0 - torch.tanh(dist / std)            # (N,)

    # held detection
    robot   = env.scene[gripper_cfg.name]
    fingers = robot.data.joint_pos[:, -2:]
    gap     = fingers.sum(dim=1)
    closed  = (gap < finger_threshold)
    lifted  = (cube_pos[:,2] > minimal_height)
    # palm position
    palm_pos  = ee.data.target_pos_w[...,0,:3]
    dist_palm = torch.norm(cube_pos - palm_pos, dim=1)
    in_pinch  = dist_palm < in_pinch_threshold
    held      = closed & lifted & in_pinch       # (N,)

    # init state trackers
    if not hasattr(env, '_prev_held'):
        env._prev_held = torch.zeros_like(held)
    if not hasattr(env, '_launched'):
        env._launched = torch.zeros_like(held)

    # detect launch event: held->not held
    new_launch = (~held) & env._prev_held
    # update launched flag: sticky until re-grasp
    env._launched = env._launched | new_launch
    # detect re-grasp: not held->held
    reheld = held & (~env._prev_held)
    env._launched = env._launched & (~reheld)
    # update prev_held
    env._prev_held = held

    # Phase masks
    phase1 = (~held) | env._launched           # approach or post-launch chase

    return reach * phase1.to(reach.dtype)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg       = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg     = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg      = SceneEntityCfg("robot"),
    finger_threshold: float          = 0.07,
    in_pinch_threshold: float        = 0.03,
) -> torch.Tensor:
    """
    Phase 1 lift reward and Phase 3 re-lift: +1 if lifted but not yet fully held,
    or after launch allow re-lift.
    """
    obj      = env.scene[object_cfg.name]
    height   = obj.data.root_pos_w[:,2]                # (N,)

    robot    = env.scene[gripper_cfg.name]
    fingers  = robot.data.joint_pos[:, -2:]
    gap      = fingers.sum(dim=1)
    closed   = (gap < finger_threshold)
    lifted   = (height > minimal_height)

    ee       = env.scene[ee_frame_cfg.name]
    palm_pos = ee.data.target_pos_w[...,0,:3]
    dist_p   = torch.norm(obj.data.root_pos_w[:,:3] - palm_pos, dim=1)
    in_pinch = dist_p < in_pinch_threshold

    held     = closed & lifted & in_pinch           # (N,)

    # init trackers (should already exist)
    if not hasattr(env, '_prev_held'):
        env._prev_held = torch.zeros_like(held)
    if not hasattr(env, '_launched'):
        env._launched = torch.zeros_like(held)
    # compute phase mask same as above
    phase1 = (~held) | env._launched

    # reward lift only during phase1
    return (lifted & ~held).to(torch.float32) * phase1.to(torch.float32)


def acc_term(env: ManagerBasedRLEnv, k_acc: float) -> torch.Tensor:
    """
    Track minimum squared-error to goal and apply at termination.
    """
    obj_pos  = env.scene["object"].data.root_state_w[:, :3]
    goal_pos = env.scene["basket"].data.root_state_w[:, :3]
    d2       = torch.sum((obj_pos - goal_pos)**2, dim=-1)

    if not hasattr(env, '_min_acc_err2'):
        env._min_acc_err2 = torch.full_like(d2, float('inf'))
    env._min_acc_err2 = torch.minimum(env._min_acc_err2, d2)

    done     = env.termination_manager.dones
    penalty  = torch.where(done, -k_acc * env._min_acc_err2, torch.zeros_like(d2))
    if done.any():
        env._min_acc_err2 = torch.where(done,
                                         torch.full_like(env._min_acc_err2, float('inf')),
                                         env._min_acc_err2)
    return penalty


def success_bonus(env: ManagerBasedRLEnv, eps: float) -> torch.Tensor:
    """
    +1 if final object within eps of goal at termination.
    """
    final_pos = env.scene["object"].data.root_state_w[:, :3]
    goal_pos  = env.scene["basket"].data.root_state_w[:, :3]

    dist = torch.linalg.norm(final_pos - goal_pos, dim=-1)
    succ = (dist < eps).to(dist.dtype)
    done = env.termination_manager.dones.to(dist.dtype)
    return succ * done


def throw_prep(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg    = SceneEntityCfg("object"),
    basket_cfg: SceneEntityCfg    = SceneEntityCfg("basket"),
    ee_frame_cfg: SceneEntityCfg  = SceneEntityCfg("ee_frame"),
    gripper_cfg: SceneEntityCfg   = SceneEntityCfg("robot"),
    minimal_height: float         = 0.04,
    finger_threshold: float       = 0.07,
    in_pinch_threshold: float     = 0.03,
) -> torch.Tensor:
    """
    Phase 2: once object truly held, penalize distance to basket to teach throw.
    """
    obj      = env.scene[object_cfg.name]
    basket   = env.scene[basket_cfg.name]
    p_obj    = obj.data.root_pos_w[:, :3]
    p_basket = basket.data.root_pos_w[:, :3]
    dist     = torch.norm(p_obj - p_basket, dim=1)

    robot    = env.scene[gripper_cfg.name]
    fingers  = robot.data.joint_pos[:, -2:]
    gap      = fingers.sum(dim=1)
    closed   = (gap < finger_threshold)
    lifted   = (p_obj[:, 2] > minimal_height)

    ee       = env.scene[ee_frame_cfg.name]
    palm_pos = ee.data.target_pos_w[...,0,:3]
    dist_p   = torch.norm(p_obj - palm_pos, dim=1)
    in_pinch = dist_p < in_pinch_threshold

    held     = closed & lifted & in_pinch

    if not hasattr(env, '_prev_held'):
        env._prev_held = torch.zeros_like(held)
    if not hasattr(env, '_launched'):
        env._launched = torch.zeros_like(held)
    new_launch     = (~held) & env._prev_held
    env._launched  = env._launched | new_launch
    reheld         = held & (~env._prev_held)
    env._launched  = env._launched & (~reheld)
    env._prev_held = held

    phase2 = held & (~env._launched)
    return -dist * phase2.to(torch.float32)
