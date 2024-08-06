'''
Minimal script to understand how replay buffer is populated for RVT training using the raw demonstrations in RLBench
'''

from peract_colab.rlbench.utils import get_stored_demo
from rvt.utils.dataset import create_replay
import peract_colab.arm.utils as utils
from rvt.libs.peract.helpers.utils import extract_obs
from rvt.utils.dataset import _clip_encode_text

import os, pickle
from typing import List
import numpy as np
import clip
import torch

from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    EPISODE_FOLDER,
    VARIATION_DESCRIPTIONS_PKL,
    DEMO_AUGMENTATION_EVERY_N,
    ROTATION_RESOLUTION,
    VOXEL_SIZES,
)

def _is_stopped(demo, i, obs, stopped_buffer, delta=0.1):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = (
            i < (len(demo) - 2) and
            (obs.gripper_open == demo[i + 1].gripper_open and
             obs.gripper_open == demo[i - 1].gripper_open and
             demo[i - 2].gripper_open == demo[i - 1].gripper_open))
    small_delta = np.allclose(obs.joint_velocities, 0, atol=delta)
    stopped = (stopped_buffer <= 0 and small_delta and
               (not next_is_not_final) and gripper_state_no_change)
    return stopped


# discretize translation, rotation, gripper open, and ignore collision actions
def _get_action(
    obs_tp1,
    obs_tm1,
    rlbench_scene_bounds: List[float],  # metric 3D bounds of the scene
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
):
    quat = utils.normalize_quaternion(obs_tp1.gripper_pose[3:])
    if quat[-1] < 0:
        quat = -quat
    disc_rot = utils.quaternion_to_discrete_euler(quat, rotation_resolution)
    attention_coordinate = obs_tp1.gripper_pose[:3]
    trans_indicies, attention_coordinates = [], []
    bounds = np.array(rlbench_scene_bounds)
    ignore_collisions = int(obs_tm1.ignore_collisions)
    for depth, vox_size in enumerate(
        voxel_sizes
    ):  # only single voxelization-level is used in PerAct
        index = utils.point_to_voxel_index(obs_tp1.gripper_pose[:3], vox_size, bounds)
        trans_indicies.extend(index.tolist())
        res = (bounds[3:] - bounds[:3]) / vox_size
        attention_coordinate = bounds[:3] + res * index
        attention_coordinates.append(attention_coordinate)

    rot_and_grip_indicies = disc_rot.tolist()
    grip = float(obs_tp1.gripper_open)
    rot_and_grip_indicies.extend([int(obs_tp1.gripper_open)])
    return (
        trans_indicies,
        rot_and_grip_indicies,
        ignore_collisions,
        np.concatenate([obs_tp1.gripper_pose, np.array([grip])]),
        attention_coordinates,
    )

def _add_keypoints_to_replay(
    replay,
    task: str,
    task_replay_storage_folder: str,
    episode_idx: int,
    sample_frame: int,
    inital_obs,
    demo,
    episode_keypoints: List[int],
    cameras: List[str],
    rlbench_scene_bounds: List[float],
    voxel_sizes: List[int],
    rotation_resolution: int,
    crop_augmentation: bool,
    next_keypoint_idx: int,
    description: str = "",
    clip_model=None,
    device="cpu",
):
    prev_action = None
    obs = inital_obs
    for k in range(
        next_keypoint_idx, len(episode_keypoints)
    ):  # confused here, it seems that there are many similar samples in the replay
        keypoint = episode_keypoints[k]
        obs_tp1 = demo[keypoint]
        obs_tm1 = demo[max(0, keypoint - 1)]

        print("tm1: ", max(0, keypoint - 1), "       tp1: ", keypoint)
        (
            trans_indicies,
            rot_grip_indicies,
            ignore_collisions,
            action,
            attention_coordinates,
        ) = _get_action(
            obs_tp1,
            obs_tm1,
            rlbench_scene_bounds,
            voxel_sizes,
            rotation_resolution,
            crop_augmentation,
        )

        terminal = k == len(episode_keypoints) - 1
        reward = float(terminal) * 1.0 if terminal else 0

        # import pdb
        # pdb.set_trace()

        obs_dict = extract_obs(
            obs,
            CAMERAS,
            t=k - next_keypoint_idx,
            prev_action=prev_action,
            episode_length=25,
        )
        tokens = clip.tokenize([description]).numpy()
        token_tensor = torch.from_numpy(tokens).to(device)
        with torch.no_grad():
            lang_feats, lang_embs = _clip_encode_text(clip_model, token_tensor)
        obs_dict["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

        prev_action = np.copy(action)

        if k == 0:
            keypoint_frame = -1
        else:
            keypoint_frame = episode_keypoints[k - 1]
        others = {
            "demo": True,
            "keypoint_idx": k,
            "episode_idx": episode_idx,
            "keypoint_frame": keypoint_frame,
            "next_keypoint_frame": keypoint,
            "sample_frame": sample_frame,
        }
        final_obs = {
            "trans_action_indicies": trans_indicies,
            "rot_grip_action_indicies": rot_grip_indicies,
            "gripper_pose": obs_tp1.gripper_pose,
            "lang_goal": np.array([description], dtype=object),
        }

        print("adding buffer sample ------> keypoint_idx:", k,
              "     sample_frame: ", sample_frame,
              "     keypoint_frame: ", keypoint_frame,
              "     next_keypoint_frame: ", keypoint)

        others.update(final_obs)
        others.update(obs_dict)

        timeout = False
        replay.add(
            task,
            task_replay_storage_folder,
            action,
            reward,
            terminal,
            timeout,
            **others
        )
        obs = obs_tp1
        sample_frame = keypoint

    # final step
    obs_dict_tp1 = extract_obs(
        obs_tp1,
        CAMERAS,
        t=k + 1 - next_keypoint_idx,
        prev_action=prev_action,
        episode_length=25,
    )
    obs_dict_tp1["lang_goal_embs"] = lang_embs[0].float().detach().cpu().numpy()

    obs_dict_tp1.pop("wrist_world_to_cam", None)
    obs_dict_tp1.update(final_obs)
    replay.add_final(task, task_replay_storage_folder, **obs_dict_tp1)

    print(" --- " * 10)


# Create the replay buffer for training
train_replay_buffer = create_replay(
    batch_size=24,
    timesteps=1,
    disk_saving=True,
    cameras=CAMERAS,
    voxel_sizes=VOXEL_SIZES,
)

data_path = '/workspace/RVT/rvt/data/train/close_jar/all_variations/episodes'
d_idx = 0

print("reading demo....")
demo = get_stored_demo(data_path=data_path, index=d_idx)
print("loaded demo !")

# get language goal from disk
varation_descs_pkl_file = os.path.join(
    data_path, EPISODE_FOLDER % d_idx, VARIATION_DESCRIPTIONS_PKL
)
with open(varation_descs_pkl_file, "rb") as f:
    descs = pickle.load(f)

# Heuristic keypoint discovery
episode_keypoints = []
prev_gripper_open = demo[0].gripper_open
stopped_buffer = 0
stopping_delta = 0.1

for i, obs in enumerate(demo):
    stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
    stopped_buffer = 4 if stopped else stopped_buffer - 1
    # If change in gripper, or end of episode.
    last = i == (len(demo) - 1)
    if i != 0 and (obs.gripper_open != prev_gripper_open or
                    last or stopped):
        episode_keypoints.append(i)
    prev_gripper_open = obs.gripper_open
if len(episode_keypoints) > 1 and (episode_keypoints[-1] - 1) == \
        episode_keypoints[-2]:
    episode_keypoints.pop(-2)

print("episode keypoints: ", episode_keypoints)

print("augmenting demo and adding to replay buffer...")
demo_augmentation_every_n = 10
next_keypoint_idx = 0

clip_model, _ = clip.load("RN50", device="cpu")  # CLIP-ResNet50
clip_model = clip_model.to('cuda')
clip_model.eval()
task_replay_storage_folder = '/workspace/RVT/rvt/utils/minimal_replay/close_jar'
os.makedirs(task_replay_storage_folder, exist_ok=True)

for i in range(len(demo) - 1):
    if i % demo_augmentation_every_n != 0:  # choose only every n-th frame
        continue

    obs = demo[i]
    desc = descs[0]
    # if our starting point is past one of the keypoints, then remove it
    while (
        next_keypoint_idx < len(episode_keypoints)
        and i >= episode_keypoints[next_keypoint_idx]
    ):
        next_keypoint_idx += 1
    if next_keypoint_idx == len(episode_keypoints):
        break
    _add_keypoints_to_replay(
        train_replay_buffer,
        'close_jar',
        task_replay_storage_folder,
        d_idx,
        i,
        obs,
        demo,
        episode_keypoints,
        cameras=CAMERAS,
        rlbench_scene_bounds=SCENE_BOUNDS,
        voxel_sizes=VOXEL_SIZES,
        rotation_resolution=ROTATION_RESOLUTION,
        crop_augmentation=False,
        next_keypoint_idx=next_keypoint_idx,
        description=desc,
        clip_model=clip_model,
        device='cuda',
    )

print("filled replay buffer with demos !")