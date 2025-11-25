from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg, ObservationGroupCfg, ObservationManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import check_file_path, read_file

from ...null_action import NullAction, NullActionCfg
from ..npc_action_base import NPCActionBase, NPCActionBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from beyondMimic.mdp.commands import MotionLoader
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

class NPCActionFK(NPCActionBase):
    cfg: NPCActionFKCfg
    def __init__(self, cfg: NPCActionFKCfg, env):
        NullAction.__init__(self, cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.load_policy(cfg)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._counter = 0

    def reset(self, env_ids = None):
        # if env_ids is not None:
        #     self.time_steps[env_ids] = 0
        # else:
        #     self.time_steps[:] = 0
        return super().reset(env_ids)

    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self.write_to_sim()
            self._counter = 0
            self.time_steps += 1
            self.time_steps %= self.motion.time_step_total
        self._counter += 1

    def write_to_sim(self):
        # motion = self.motion
        # robot = self.robot
        root_states = self.robot.data.default_root_state.clone()
        root_states[:, :3]      =  self.motion.body_pos_w[self.time_steps][:, 0] + self._env.scene.env_origins[:, None, :]
        root_states[:, 3:7]     =  self.motion.body_quat_w[self.time_steps][:, 0]
        root_states[:, 7:10]    =  self.motion.body_lin_vel_w[self.time_steps][:, 0]
        root_states[:, 10:]     =  self.motion.body_ang_vel_w[self.time_steps][:, 0]

        self.robot.write_root_state_to_sim(root_states)
        self.robot.write_joint_state_to_sim(self.motion.joint_pos[self.time_steps], self.motion.joint_vel[self.time_steps])
        # self._env.scene.write_data_to_sim()

    def load_policy(self, cfg: NPCActionFKCfg):
        self.body_indexes, _ = self.robot.find_bodies(cfg.body_names)
        self.motion = MotionLoader(cfg.motion_file, self.body_indexes, device=self.device)

@configclass
class NPCActionFKCfg(NullActionCfg):
    class_type:         type[ActionTerm] = NPCActionFK
    low_level_decimation: int = 4
    policy_path:        str = None
    motion_file:        str = MISSING
    body_names:         list = MISSING
    anchor_body_name:   str =MISSING
