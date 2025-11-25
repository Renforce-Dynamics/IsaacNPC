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

class NPCActionVel(NPCActionBase):
    cfg: NPCActionVelCfg
    def __init__(self, cfg: NPCActionVelCfg, env):
        super().__init__(cfg, env)
        if hasattr(cfg.low_level_observations , "actions"):
            cfg.low_level_observations.actions.func = lambda dummy_env: self.last_action()
            cfg.low_level_observations.actions.params = dict()
        elif hasattr(cfg.low_level_observations , "last_action"):
            cfg.low_level_observations.last_action.func = lambda dummy_env: self.last_action()
            cfg.low_level_observations.last_action.params = dict()
        else:
            raise NameError("no name for last action.")
        cfg.low_level_observations.velocity_commands.func = lambda dummy_env: self.vel_command()
        cfg.low_level_observations.velocity_commands.params = dict()
        self._low_level_obs_manager = ObservationManager({"ll_policy": cfg.low_level_observations}, env)
        
    def reset(self, env_ids = None):
        self._low_level_obs_manager.reset(env_ids=env_ids)
        return super().reset(env_ids)
        
    def _render_action(self):
        low_level_obs = self._low_level_obs_manager.compute_group("ll_policy")
        return self.policy(low_level_obs)
    
    def vel_command(self):
        cmd = torch.zeros((self._env.num_envs, 3), device=self.device, dtype=torch.float32)
        cmd[:, 0] = 0.3 
        return cmd

@configclass
class NPCActionVelCfg(NPCActionBaseCfg):
    class_type: type[ActionTerm] = NPCActionVel
