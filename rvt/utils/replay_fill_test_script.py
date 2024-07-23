from peract_colab.rlbench.utils import get_stored_demo
from peract_colab.rlbench.utils import get_stored_demo
from rvt.utils.dataset import create_replay

import os, pickle
import numpy as np

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


train_replay_buffer = create_replay(
    batch_size=24,
    timesteps=1,
    disk_saving=True,
    cameras=CAMERAS,
    voxel_sizes=VOXEL_SIZES,
)

data_path = '/workspace/RVT/rvt/data/train/close_jar/all_variations/episodes'
d_idx = 0

demo = get_stored_demo(data_path=data_path, index=d_idx)

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