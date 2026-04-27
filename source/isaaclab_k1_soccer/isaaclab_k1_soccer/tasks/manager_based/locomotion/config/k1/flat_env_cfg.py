# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import K1RoughEnvCfg
import math
from .mdp.commands import DiscreteVelocityCommand, DiscreteVelocityCommandCfg


@configclass
class K1FlatEnvCfg(K1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Flat terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # No height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # No terrain curriculum
        self.curriculum.terrain_levels = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.2
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -1.0e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_Hip_.*", ".*_Ankle_.*"]
        )
        self.commands.base_velocity = DiscreteVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            heading_command=True,
            heading_control_stiffness=0.5,
            rel_standing_envs=0.02,
            rel_heading_envs=1.0,
            high_vel=1.0,
            low_vel_max=0.4,
            high_ang_vel=1.0,
            low_ang_vel_max=0.4,
            high_prob=0.5,
            ranges=DiscreteVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),   # _resample で上書きされるので実質不使用
                lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0),
                heading=(-math.pi, math.pi),
            ),
        )


class K1FlatEnvCfg_PLAY(K1FlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
