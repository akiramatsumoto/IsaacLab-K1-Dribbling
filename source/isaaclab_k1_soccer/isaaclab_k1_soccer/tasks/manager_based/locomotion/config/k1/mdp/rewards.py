# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import re
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, euler_xyz_from_quat, wrap_to_pi
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def minimum_height(env: ManagerBasedRLEnv, min_height: float = 0.47, 
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
                    sensor_cfg: SceneEntityCfg = None) -> torch.Tensor:
    """
    minimum heightよりもロボットの高さが低い場合にペナルティを与える報酬関数
    """
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None:  # これはRaycasterである必要あり.roughの場合に使う
        sensor = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_min_height = min_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_min_height = min_height
    # Compute the L2 squared penalty
    return torch.where(asset.data.root_pos_w[:, 2] < adjusted_min_height, torch.square(asset.data.root_pos_w[:, 2] - adjusted_min_height), torch.zeros_like(asset.data.root_pos_w[:, 2]))
