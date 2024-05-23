import argparse
import os
import random
import sys
import time
import torch
from accelerate import Accelerator
from collections import defaultdict
import tqdm
from contextlib import redirect_stdout
import yaml

import config as exp_cfg_mod
import rvt.models.rvt_agent as rvt_agent
import rvt.mvt.config as mvt_cfg_mod

from rvt.mvt.mvt import MVT
from rvt.models.rvt_agent import print_eval_log, print_loss_log
from rvt.utils.get_dataset import get_dataset
from rvt.utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    load_agent,
    RLBENCH_TASKS,
)
from rvt.utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)


def train(agent, data_iter, training_iterations, accelerator, rank=0):
    agent.train()
    log = defaultdict(list)

    iter_command = range(training_iterations)

    for iteration in tqdm.tqdm(iter_command, disable=(rank != 0), position=0, leave=True):
        start_time = time.time()
        raw_batch = next(data_iter)
        print("Next batch received ! --- Time Cost: {} minutes".format((time.time() - start_time) / 60.0))

        keys_to_keep = ['action_ignore_collisions', 'wpt_local', 'low_dim_state', 'rot_grip_action_indicies',
                        'ignore_collisions', 'gripper_pose', 'lang_goal_embs', 'img', 'proprio', 'action_rot', 'action_grip']
        
        keys_to_delete = []
        for key, val in zip(raw_batch.keys(), raw_batch.values()):
            if (key not in keys_to_keep and 'rgb' not in key and 'point_cloud' not in key) or 'tp1' in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del raw_batch[key]

        batch = {}
        for k, v in raw_batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(accelerator.device).squeeze(1)
            else:
                batch[k] = v

        update_args = {
            "step": iteration,
            "replay_sample": batch,
            "ddp": False,
            "backprop": True,
            "reset_log": (iteration == 0),
            "eval_log": False,
        }
        agent.update_optimized(**update_args)

    if rank == 0:
        log = print_loss_log(agent)

    return log

def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched

    model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )

def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks

def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)

def experiment(rank, cmd_args, devices, port):
    print("Beginning experiment function execution.....")
    sys.stdout.flush()

    start_time = time.time()

    accelerator = Accelerator()
    device = accelerator.device

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    print("DDP SETUP ! --- Time Cost: {} minutes".format((time.time() - start_time) / 60.0))

    BATCH_SIZE_TRAIN = exp_cfg.bs
    NUM_TRAIN = 100
    TRAINING_ITERATIONS = int(10000 // (exp_cfg.bs * len(devices) / 16))
    EPOCHS = exp_cfg.epochs

    if IMAGE_SIZE == 128:
        TRAIN_REPLAY_STORAGE_DIR = "replay_optimized_fp32_all/replay_train"
        TEST_REPLAY_STORAGE_DIR = "replay_optimized_fp32_all/replay_val"
    else:
        raise AssertionError("Only IMAGE_SIZE 128 supported right now for optimized training")

    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("train replay storage: ", TRAIN_REPLAY_STORAGE_DIR)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()
    get_dataset_func = lambda: get_dataset(
        tasks,
        BATCH_SIZE_TRAIN,
        None,
        TRAIN_REPLAY_STORAGE_DIR,
        None,
        DATA_FOLDER,
        NUM_TRAIN,
        None,
        cmd_args.refresh_replay,
        device,
        num_workers=exp_cfg.num_workers,
        only_train=True,
        sample_distribution_mode=exp_cfg.sample_distribution_mode,
        sample_mode=exp_cfg.sample_mode,
    )
    train_dataset, _ = get_dataset_func()

    train_dataset = accelerator.prepare(train_dataset)
    train_dataset_iter = iter(train_dataset)
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    start_time = time.time()

    if exp_cfg.agent == "our":
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        torch.cuda.empty_cache()
        rvt = MVT(
            renderer_device=device,
            **mvt_cfg,
        ).to(device)

        agent = rvt_agent.RVTAgent(
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"
    
    agent._network, agent._optimizer, train_dataset_iter = accelerator.prepare(
        agent._network, agent._optimizer, train_dataset_iter
    )
    
    print("Agent SETUP ! --- Time Cost: {} minutes".format((time.time() - start_time) / 60.0))

    start_time = time.time()

    start_epoch = 0
    end_epoch = EPOCHS
    if exp_cfg.resume != "":
        agent_path = exp_cfg.resume
        print(f"Recovering model and checkpoint from {exp_cfg.resume}")
        epoch = load_agent(agent_path, agent, only_epoch=False)
        start_epoch = epoch + 1

    if rank == 0:
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(agent, train_dataset_iter, TRAINING_ITERATIONS, accelerator, rank)

        print(f"Rank [{rank}], Epoch [{i}]: Finished training")

        if rank == 0:
            t_start = time.time()

            tb.update("train", i, out)
            
            t_end = time.time()
            print("Updated tensorboard. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

        if rank == 0:
            t_start = time.time()
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            t_end = time.time()
            print("saved current agent ckpt. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

            t_start = time.time()
            save_agent(agent, f"{log_dir}/model_last.pth", i)
            t_end = time.time()
            print("saved last agent ckpt. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

        i += 1

    if rank == 0:
        tb.close()
        print("[Finish]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="")
    parser.add_argument("--exp_cfg_path", type=str, default="")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    port = (random.randint(0, 3000) % 3000) + 27000

    print("Starting training process....")
    sys.stdout.flush()

    experiment(0, cmd_args, devices, port)

    print("Training process completed!")
    sys.stdout.flush()
