from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, ActionTerm, ManagerTermBaseCfg

def npc_make_robot_term(source, asset_name, target=None, is_del=False, term_type:type[ManagerTermBaseCfg]=ManagerTermBaseCfg):
    terms_to_add = []
    for key, value in vars(source).items():
        if isinstance(value, term_type):
            terms_to_add.append((f"{asset_name}_{key}", value.replace(asset_name=asset_name)))
        if is_del: delattr(source, key)
    for term in terms_to_add:
        k, v = term
        if target is None:
            setattr(source, k, v)
        else:
            setattr(target, k, v)
