# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

# initial source: https://colab.research.google.com/drive/1HAqemP4cE81SQ6QO1-N85j5bF4C0qLs0?usp=sharing
# adapted to support loading from disk for faster initialization time

# Adapted from: https://github.com/stepjam/ARM/blob/main/arm/c2farm/launch_utils.py
import os
import torch
import pickle
import logging
import numpy as np
from typing import List

import clip
import peract_colab.arm.utils as utils

from peract_colab.rlbench.utils import get_stored_demo
from yarr.utils.observation_type import ObservationElement
from yarr.replay_buffer.replay_buffer import ReplayElement, ReplayBuffer
# from yarr.replay_buffer.uniform_replay_buffer import UniformReplayBuffer
from yarr.replay_buffer.uniform_replay_buffer_optimized import UniformReplayBufferOptimized
from rlbench.backend.observation import Observation
from rlbench.demo import Demo

from rvt.utils.peract_utils import LOW_DIM_SIZE, IMAGE_SIZE, CAMERAS
from rvt.libs.peract.helpers.demo_loading_utils import keypoint_discovery
from rvt.libs.peract.helpers.utils import extract_obs


def create_replay_optimized(
    batch_size: int,
    timesteps: int,
    disk_saving: bool,
    cameras: list,
    voxel_sizes,
    replay_size=3e5,
):

    observation_elements = []

    # action_ignore_collisions
    observation_elements.append(
        ReplayElement("action_ignore_collisions", (1,), np.float16)
    )

    # wpt_local
    observation_elements.append(
        ReplayElement("wpt_local", (3,), np.float16)
    )

    # low_dim_state
    # observation_elements.append(
    #     ReplayElement("low_dim_state", (1,4), np.float16)
    # )

    # rot_grip_action_indicies
    # observation_elements.append(
    #     ReplayElement("rot_grip_action_indicies", (1,4), np.float16)
    # )

    # ignore_collisions
    # observation_elements.append(
    #     ReplayElement("ignore_collisions", (1,1), np.float16)
    # )

    # gripper_pose
    # observation_elements.append(
    #     ReplayElement("gripper_pose", (1,7), np.float16)
    # )

    # lang_goal_embs
    observation_elements.append(
        ReplayElement("lang_goal_embs", (77,512), np.float16)
    )

    # img
    observation_elements.append(
        ReplayElement("img", (5,10,220,220), np.float16)
    )

    # proprio
    observation_elements.append(
        ReplayElement("proprio", (4,), np.float16)
    )

    # action_rot
    observation_elements.append(
        ReplayElement("action_rot", (4,), np.float16)
    )

    # action_grip
    observation_elements.append(
        ReplayElement("action_grip", (), np.float16)
    )

    extra_replay_elements = []

    replay_buffer = (
        UniformReplayBufferOptimized(  # all tuples in the buffer have equal sample weighting
            disk_saving=disk_saving,
            batch_size=batch_size,
            timesteps=timesteps,
            replay_capacity=int(replay_size),
            action_shape=(8,),  # 3 translation + 4 rotation quaternion + 1 gripper open
            action_dtype=np.float32,
            reward_shape=(),
            reward_dtype=np.float32,
            update_horizon=1,
            observation_elements=observation_elements,
            extra_replay_elements=extra_replay_elements,
            optimized_training=True,        # NOTE: This needs to be set to True to avoid loading incorrect terminal state for which the optimised replay buffer files do not exist
        )
    )
    return replay_buffer


def fill_replay_optimized(
    replay: ReplayBuffer,
    task: str,
    task_replay_storage_folder: str,
    start_idx: int,
    num_demos: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    cameras: List[str],
    rlbench_scene_bounds: List[float],  # AKA: DEPTH0_BOUNDS
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    data_path: str,
    episode_folder: str,
    variation_desriptions_pkl: str,
    clip_model=None,
    device="cpu",
):

    disk_exist = False
    if replay._disk_saving:
        if os.path.exists(task_replay_storage_folder):
            print(
                "[Info] Replay dataset already exists in the disk: {}".format(
                    task_replay_storage_folder
                ),
                flush=True,
            )
            disk_exist = True
        else:
            raise AssertionError("Optimized replay buffer NOT FOUND. Optimized Replay needs to be stored offline and cannot be generated on-the-fly !")
    if disk_exist:
        replay.recover_from_disk(task, task_replay_storage_folder)
    else:
        raise AssertionError("Optimized Replay needs to be stored offline and cannot be generated on-the-fly !")