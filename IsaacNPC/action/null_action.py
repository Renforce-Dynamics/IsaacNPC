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

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class NullAction(ActionTerm):
    cfg: NullActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((env.num_envs, 0), device=env.device)
        
    @property
    def action_dim(self) -> int:
        return 0
    
    @property
    def raw_actions(self):
        return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor:
        return self._raw_actions
    
    def process_actions(self, actions):
        return self._raw_actions

    def apply_actions(self):
        return self._raw_actions

class NullActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = NullAction
    debug_vis: bool = False