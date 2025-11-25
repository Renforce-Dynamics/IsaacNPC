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

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import mdp

from ..null_action import NullAction, NullActionCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class NPCActionBase(NullAction):
    """
    This action is responsible for env movement where at each apply action the env will move at the same time.
    """

    cfg: NPCActionBaseCfg
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.load_policy(cfg)

        # prepare low level actions
        self._low_level_action_term: ActionTerm = cfg.low_level_actions.class_type(cfg.low_level_actions, env)
        self.low_level_actions = torch.zeros(self.num_envs, self._low_level_action_term.action_dim, device=self.device)
        self._counter = 0

    def replace_obsterm_with_dummy_func(self, name, func:callable):
        if getattr(self.cfg.low_level_observations, name, None) is not None:
            term = getattr(self.cfg.low_level_observations, name)
            term.func = func
            term.params = dict()
        else:
            print(f"Skip term {name} setup for term not found.")

    def last_action(self):
        # reset the low level actions if the episode was reset
        if hasattr(self._env, "episode_length_buf"):
            self.low_level_actions[self._env.episode_length_buf == 0, :] = 0
        return self.low_level_actions

    def load_policy(self, cfg):
        if not check_file_path(cfg.policy_path):
            raise FileNotFoundError(f"Policy file '{cfg.policy_path}' does not exist.")
        self.policy = torch.jit.load(cfg.policy_path).to(self._env.device).eval()
        
    def _render_action(self):
        raise NotImplementedError("Not implemented for rendering actions.")
        
    def apply_actions(self):
        if self._counter % self.cfg.low_level_decimation == 0:
            self.low_level_actions[:] = self._render_action()
            self._low_level_action_term.process_actions(self.low_level_actions)
            self._counter = 0
        self._low_level_action_term.apply_actions()
        self._counter += 1
        
    def root_pos_env(self):
        return self.robot.data.root_pos_w - self.env.scene.env_origins
        
    @staticmethod
    def make_npc_determine_events(target_cfg, name):
        physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(name, body_names=".*"),
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.6, 0.6),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )
        
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(name),
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(name),
                "position_range": (1.0, 1.0),
                "velocity_range": (-1.0, 1.0),
            },
        )
        
        terms = ["physics_material", "reset_base", "reset_robot_joints"]
        
        for term in terms:
            setattr(target_cfg, f"{name}_{term}", locals()[term])
        
@configclass
class NPCActionBaseCfg(NullActionCfg):
    class_type: type[ActionTerm] = NPCActionBase
    
    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""
    policy_path: str = MISSING
    """Path to the low level policy (.pt files)."""

    low_level_decimation: int = 4
    """Decimation factor for the low level action term."""
    low_level_actions: ActionTermCfg = MISSING
    """Low level action configuration."""
    low_level_observations: ObservationGroupCfg = MISSING
    """Low level observation configuration."""
    

