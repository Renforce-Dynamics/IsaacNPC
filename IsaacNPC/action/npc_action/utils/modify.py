from __future__ import annotations
from typing import List, Dict, Any

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

def modify_obs_robot(obs_grp:ObsGroup, target_name:str, terms:List[str]):
    for term_name in terms:
        target_term:ObsTerm = getattr(obs_grp, term_name)
        if target_term is None:
            print(f"Term: {term_name} is None, skip.") 
            continue
        target_term.params["asset_cfg"] = SceneEntityCfg(name=target_name, joint_names=".*")
        setattr(obs_grp, term_name, target_term)
    return obs_grp

def modify_robot_event(source, target, name:str, params:Dict[str, SceneEntityCfg]):
    source_names = list(source.to_dict().keys())
    for key in source_names:
        # print(f"updated: {key}")
        term:EventTerm = getattr(source, key)
        if isinstance(term, EventTerm):
            term.params.update(params)
            attr_name = f"{name}_{key}"
            setattr(target, attr_name, term)