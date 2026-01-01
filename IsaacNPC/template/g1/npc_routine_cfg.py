from isaaclab.utils import configclass
from isaaclab.envs import mdp

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg, ActionTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from IsaacNPC.action.npc_action.npc_action_vel import NPCActionRoutineCfg

@configclass
class G1NPCRoutineActionsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2), clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5), clip=(-100, 100))
        actions = ObsTerm(func=mdp.last_action, clip=(-12, 12))

        def __post_init__(self):
            # self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
    
    npc_routine_action:ActionTerm = NPCActionRoutineCfg(
            asset_name="asset_name",
            policy_path="data/ckpts/g1/g1_29d_loco_walk.pt",
            low_level_observations=PolicyCfg(),
            low_level_actions=mdp.JointPositionActionCfg(
                asset_name="asset_name", joint_names=[".*"], scale=0.25, use_default_offset=True
            ),
            routine_points=[
                (0, 0, 0),
                (10, 0, 0),
                (10, 10, 3.14/4),
                (0, 0, 0)
            ]
        )
        
@configclass
class G1NPCVelPolicyEventsCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("name", body_names=".*"),
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
            "asset_cfg": SceneEntityCfg("name"),
            "pose_range": {"x": (0.0, 0.0), "y": (0.05, 0.05), "yaw": (0, 0)},
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
            "asset_cfg": SceneEntityCfg("name"),
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )
