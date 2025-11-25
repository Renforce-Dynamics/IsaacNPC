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

from ..npc_action_base import NPCActionBase, NPCActionBaseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from beyondMimic.mdp.commands import MotionLoader
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

class NPCActionMimic(NPCActionBase):
    cfg: NPCActionMimicCfg
    def __init__(self, cfg: NPCActionMimicCfg, env):
        super().__init__(cfg, env)
        if hasattr(cfg.low_level_observations , "actions"):
            self.replace_obsterm_with_dummy_func("actions", lambda dummy_env: self.last_action())
        elif hasattr(cfg.low_level_observations , "last_action"):
            self.replace_obsterm_with_dummy_func("last_action", lambda dummy_env: self.last_action())
        else:
            raise NameError("no name for last action.")
        
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        
        self.motion = MotionLoader(cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        self.replace_obsterm_with_dummy_func("motion_anchor_ori_b", lambda dummy_env: self.motion_anchor_ori_b())
        self.replace_obsterm_with_dummy_func("command", lambda dummy_env: self.motion_cmd())
        
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)

    def motion_cmd(self) -> torch.Tensor:
        return torch.cat([
            self.motion.joint_pos[self.time_steps], 
            self.motion.joint_vel[self.time_steps]
        ], dim=1)

    def motion_anchor_ori_b(self) -> torch.Tensor:
        ref_pos = self.motion_anchor_pos_w[self.time_steps]
        ref_quat = self.motion_anchor_quat_w[self.time_steps]

        _, ori_b = subtract_frame_transforms(
            self.robot_anchor_pos_w,
            self.robot_anchor_quat_w,
            ref_pos,
            ref_quat,
        )
        mat = matrix_from_quat(ori_b)
        return mat[..., :2].reshape(mat.shape[0], -1)

    def process_actions(self, actions):
        self.time_steps += 1
        self.time_steps %= self.motion.time_step_total
        return super().process_actions(actions)

    @property
    def motion_anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def motion_anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def reset(self, env_ids = None):
        self._low_level_obs_manager.reset(env_ids=env_ids)
        return super().reset(env_ids)

    def _render_action(self):
        low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
        return self.policy(low_level_obs)

    def load_policy(self, cfg):
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = torch.jit.load(cfg.policy_path).to(self._env.device).eval()

@configclass
class NPCActionMimicCfg(NPCActionBaseCfg):
    class_type:         type[ActionTerm] = NPCActionMimic
    motion_file:        str = MISSING
    body_names:         list = MISSING
    anchor_body_name:   str =MISSING
