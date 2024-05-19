# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import sys
import shutil
import torch
import clip

from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.rvt_utils import ForkedPdb
# from rvt.utils.dataset import create_replay, fill_replay
from rvt.utils.dataset_optimized import create_replay_optimized, fill_replay_optimized
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
    IMAGE_SIZE,
)
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer


def get_dataset(
    tasks,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_TEST,
    TRAIN_REPLAY_STORAGE_DIR,
    TEST_REPLAY_STORAGE_DIR,
    DATA_FOLDER,
    NUM_TRAIN,
    NUM_VAL,
    refresh_replay,
    device,
    num_workers,
    only_train,
    sample_distribution_mode="transition_uniform",
    sample_mode="random",
):

    train_replay_buffer = create_replay_optimized(
        batch_size=BATCH_SIZE_TRAIN,
        timesteps=1,
        disk_saving=True,
        cameras=CAMERAS,
        voxel_sizes=VOXEL_SIZES,
    )

    # load pre-trained language model
    try:
        clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
        clip_model = clip_model.to(device)
        clip_model.eval()
    except RuntimeError:
        print("WARNING: Setting Clip to None. Will not work if replay not on disk.")
        clip_model = None

    for task in tasks:  # for each task
        # print("---- Preparing the data for {} task ----".format(task), flush=True)
        EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
        EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"

        if IMAGE_SIZE == 128:
            EPISODES_FOLDER_TRAIN = f"train/{task}/all_variations/episodes"
            EPISODES_FOLDER_VAL = f"val/{task}/all_variations/episodes"
        elif IMAGE_SIZE == 256:
            EPISODES_FOLDER_TRAIN = f"train_256x256/{task}/all_variations/episodes"
            EPISODES_FOLDER_VAL = f"val_256x256/{task}/all_variations/episodes"
        else:
            # If IMAGE_SIZE is neither 128 nor 256, raise an AssertionError
            raise AssertionError("Invalid IMAGE_SIZE provided. Expected 128 or 256.")

        data_path_train = os.path.join(DATA_FOLDER, EPISODES_FOLDER_TRAIN)
        data_path_val = os.path.join(DATA_FOLDER, EPISODES_FOLDER_VAL)
        train_replay_storage_folder = f"{TRAIN_REPLAY_STORAGE_DIR}/{task}"
        test_replay_storage_folder = f"{TEST_REPLAY_STORAGE_DIR}/{task}"

        print("data path train: ", data_path_train)
        print("train replay storage: ", train_replay_storage_folder)

        # if refresh_replay, then remove the existing replay data folder
        # if refresh_replay:
        #     print("[Info] Remove exisitng replay dataset as requested.", flush=True)
        #     if os.path.exists(train_replay_storage_folder) and os.path.isdir(
        #         train_replay_storage_folder
        #     ):
        #         shutil.rmtree(train_replay_storage_folder)
        #         print(f"remove {train_replay_storage_folder}")
        #     if os.path.exists(test_replay_storage_folder) and os.path.isdir(
        #         test_replay_storage_folder
        #     ):
        #         shutil.rmtree(test_replay_storage_folder)
        #         print(f"remove {test_replay_storage_folder}")

        # print("----- Train Buffer -----")
        fill_replay_optimized(
            replay=train_replay_buffer,
            task=task,
            task_replay_storage_folder=train_replay_storage_folder,
            start_idx=0,
            num_demos=NUM_TRAIN,
            demo_augmentation=True,
            demo_augmentation_every_n=DEMO_AUGMENTATION_EVERY_N,
            cameras=CAMERAS,
            rlbench_scene_bounds=SCENE_BOUNDS,
            voxel_sizes=VOXEL_SIZES,
            rotation_resolution=ROTATION_RESOLUTION,
            crop_augmentation=False,
            data_path=data_path_train,
            episode_folder=EPISODE_FOLDER,
            variation_desriptions_pkl=VARIATION_DESCRIPTIONS_PKL,
            clip_model=clip_model,
            device=device,
        )

    # delete the CLIP model since we have already extracted language features
    del clip_model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    # wrap buffer with PyTorch dataset and make iterator
    train_wrapped_replay = PyTorchReplayBuffer(
        train_replay_buffer,
        sample_mode=sample_mode,
        num_workers=num_workers,
        sample_distribution_mode=sample_distribution_mode,
    )
    train_dataset = train_wrapped_replay.dataset()

    if only_train:
        test_dataset = None
    else:
        raise AssertionError("Testing during trainig not supported for optimized training runs !")

    return train_dataset, test_dataset