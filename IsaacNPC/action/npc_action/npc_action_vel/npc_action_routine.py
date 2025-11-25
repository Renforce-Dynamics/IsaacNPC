from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from .npc_action_vel import NPCActionVel, NPCActionVelCfg
from .velocity_planner_2d import BatchVelocityPlanner


class NPCActionRoutine(NPCActionVel):
    cfg: "NPCActionRoutineCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        # planner is stateless, shared across envs
        self.planner = BatchVelocityPlanner(0.5, 0.5)

        # routine points: (K, 3) -> (x, y, yaw)
        self.routine_points = (
            torch.tensor(cfg.routine_points, dtype=torch.float32, device=self.device)
            .reshape(-1, 3)
        )

        # per-env waypoint pointer
        self.target_pos_ptr = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.long
        )
        
        self.total_rountine_points = self.routine_points.shape[0] - 1

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None:
            self.target_pos_ptr[:] = 0
        else:
            self.target_pos_ptr[env_ids] = 0

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

        pos = self.root_pos_env[:, :2]  # (N,2)
        quat = self.robot.data.root_quat_w
        yaw = math_utils.quat_to_euler_xyz(quat)[..., 2]  # (N,)
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


class NPCActionRoutineCfg(NPCActionVelCfg):
    class_type = NPCActionRoutine
    routine_points: list = MISSING   # list of [x, y, yaw]
