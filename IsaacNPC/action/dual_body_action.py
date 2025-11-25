from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg

class DualBodyActionBase(JointPositionAction):
    cfg: DualBodyActionBaseCfg
    def __init__(self, cfg: DualBodyActionBaseCfg, env):
        super().__init__(cfg, env)
        self.upper_body_action: JointPositionAction = cfg.upper_body_action_cfg.class_type(cfg.upper_body_action_cfg, env)
        
    def process_actions(self, actions):
        # Write your code for upper body joint poses calc
        uppper_actions = None
        self.upper_body_action.process_actions(uppper_actions)
        return super().process_actions(actions)
        
    def apply_actions(self):
        self.upper_body_action.apply_actions()
        return super().apply_actions()

@configclass
class DualBodyActionBaseCfg(JointPositionActionCfg):
    upper_body_action_cfg: JointPositionActionCfg = MISSING
    
# Example
# cfg = DualBodyActionBaseCfg(
#     asset_name="robot",
#     joint_names="*.upper_joints",
#     upper_body_action_cfg=DualBodyActionBaseCfg(
#         asset_name="robot",
#         joint_names="*.lower_joint_names"
#     )
# )