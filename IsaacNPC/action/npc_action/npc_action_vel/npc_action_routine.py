from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, List, Tuple

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from .npc_action_vel import NPCActionVel, NPCActionVelCfg
from .velocity_planner_2d import BatchVelocityPlanner


class NPCActionRoutine(NPCActionVel):
    cfg: "NPCActionRoutineCfg"

    def __init__(self, cfg: "NPCActionRoutineCfg", env):
        # per-env waypoint pointer
        self.target_pos_ptr = torch.zeros(
            (env.num_envs,), device=env.device, dtype=torch.long
        )

        # planner is stateless, shared across envs
        self.planner = BatchVelocityPlanner(
            max_lin_vel = cfg.max_lin_vel,
            max_yaw_vel = cfg.max_yaw_vel,
            pos_tol     = cfg.pos_tol,
            yaw_tol     = cfg.yaw_tol,
        )

        # routine points: (K, 3) -> (x, y, yaw)
        self.routine_points = (
            torch.tensor(cfg.routine_points, dtype=torch.float32, device=env.device)
            .reshape(-1, 3)
        )
        
        self.total_rountine_points = self.routine_points.shape[0]
        
        super().__init__(cfg, env)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.target_pos_ptr[:] = 0
        else:
            self.target_pos_ptr[env_ids] = 0
        super().reset(env_ids)

    def _get_current_targets(self):
        """
        Returns:
            goal_pos: (N,2)
            goal_yaw: (N,)
        """
        # (N,) index into routine_points (K,3)
        idx = self.target_pos_ptr.clamp(max=self.total_rountine_points)

        # gather for each env
        goals = self.routine_points[idx]     # (N,3)
        goal_pos = goals[:, :2]              # (N,2)
        goal_yaw = goals[:, 2]               # (N,)
        return goal_pos, goal_yaw

    def vel_command(self):
        """
        Main interface called by IsaacLab action manager.
        Produces (N,3) velocity commands.
        """

        pos = self.root_pos_env()[:, :2]  # (N,2)
        quat = self.robot.data.root_quat_w
        _, _, yaw = math_utils.euler_xyz_from_quat(quat)  # (N,)
        goal_pos, goal_yaw = self._get_current_targets()
        arrived = self.planner.check_arrival(pos, yaw, goal_pos, goal_yaw)
        # update target index for arrived envs
        if arrived.any():
            self.target_pos_ptr[arrived] += 1
            self.target_pos_ptr %= self.total_rountine_points

        # recompute goals after updating pointers
        goal_pos, goal_yaw = self._get_current_targets()
        cmd, _ = self.planner.compute_cmd(pos, yaw, goal_pos, goal_yaw)
        
        return cmd

@configclass
class NPCActionRoutineCfg(NPCActionVelCfg):
    class_type          : type[NPCActionRoutine] = NPCActionRoutine
    routine_points      : List[Tuple[float, float, float]] = MISSING   # list of [x, y, yaw]
    max_lin_vel         : float = 0.5, 
    max_yaw_vel         : float = 0.5,
    pos_tol             : float = 0.1,
    yaw_tol             : float = 0.1
