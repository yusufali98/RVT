import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def find_event_files(root_dir, subfolders=True):
    """ Recursively find all TensorBoard event files in the given directory.
        Set `subfolders` to False to search only in the top-level directory. """
    event_files = []
    if subfolders:
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if 'tfevents' in filename:
                    event_files.append(os.path.join(dirpath, filename))
    else:
        for filename in os.listdir(root_dir):
            if 'tfevents' in filename:
                event_files.append(os.path.join(root_dir, filename))
    return event_files


def load_scalars(event_file, tag):
    """ Load scalar data from the given event file. If tag not found, return empty list. """
    ea = event_accumulator.EventAccumulator(event_file, size_guidance={'scalars': 0})
    ea.Reload()
    if tag in ea.Tags()['scalars']:
        return [(x.step, x.value) for x in ea.Scalars(tag)]
    return []  # Return an empty list if tag is not present

def get_exp_name(root_dir):
    
    # Get the last segment
    last_segment = os.path.basename(root_dir)

    # Move up to the parent directory to get the second last segment
    second_last_segment = os.path.basename(os.path.dirname(root_dir))

    # Combine the two segments
    last_two_segments = f"{second_last_segment}/{last_segment}"

    return last_two_segments


def plot_training_losses(directories, train_tag, val_tag):

    markers = ['o', 's', '^', 'v', '<', '>', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown', 'pink', 'lime', 'olive', 'gray', 'navy']

    # Plot training losses
    plt.figure(figsize=(15, 10))

    for idx, dir in enumerate(directories):
        train_event_files = find_event_files(dir, subfolders=True)
        exp_name = get_exp_name(dir)

        train_data = [load_scalars(event_file, train_tag) for event_file in train_event_files]
        train_data = [item for sublist in train_data for item in sublist]  # Flatten list
        train_data.sort()

        train_steps = [x[0] for x in train_data]
        train_values = [x[1] for x in train_data]
        
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        plt.plot(train_steps, train_values, marker=marker, color=color, linestyle='-', label=exp_name)
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xlim(0, 14)  # Set x-axis to show from 0 to 5
    plt.ylim(4.5,7)  # Set y-axis to show from 0 to 25
    plt.legend()
    plt.grid(True)
    plt.savefig('combined_training_loss_curves_220p.png', format='png', bbox_inches='tight', pad_inches=0.5)

    # # Plot validation success rates
    # plt.figure(figsize=(15, 10))

    # for idx, dir in enumerate(directories):
    #     val_event_files = find_event_files(dir, subfolders=False)
    #     exp_name = get_exp_name(dir)
        
    #     val_data = [load_scalars(event_file, val_tag) for event_file in val_event_files]
    #     val_data = [item for sublist in val_data for item in sublist]  # Flatten list
    #     val_data.sort()

    #     val_steps = [x[0] for x in val_data]
    #     val_values = [x[1] for x in val_data]
        
    #     marker = markers[idx % len(markers)]
    #     color = colors[idx % len(colors)]
    #     plt.plot(val_steps, val_values, marker=marker, color=color, linestyle='-', label=exp_name)

    # plt.title('Validation Success Rates')
    # plt.xlabel('Epochs')
    # plt.ylabel('Success Rate (%)')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('validation_success_rates.png', format='png', bbox_inches='tight', pad_inches=0.5)


# Usage
experiment_directories = [
    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/rvt_220p/all_tasks',
    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/rvt_330p/all_tasks',

    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/all_tasks_5e5_LR_orig',
    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_32',

    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adam_LR_1e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adam_LR_5e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adam_LR_6e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adam_LR_7e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adam_LR_8.5e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adamW_LR_2.5e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adamW_LR_5e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_adamW_LR_7e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p/all_tasks_LR_2.5e-5',

    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p_20L/all_tasks_lr_5e-5',
    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p_20L/bissm_all_tasks_LR_5e-5',

    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p_24L/all_tasks_LR_3.5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_2.5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/220p_ablations/mamba_220p_24L/all_tasks_LR_1.75e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_1e-5_2080_ti',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_2.5e-6',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_7.5e-6',

    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L_biSSM/all_tasks_LR_1e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/close_jar_5e5_fast_fp32',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/close_jar_5e5_fast_fp16',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/close_jar_5e5_orig',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/close_jar_5e5_slow',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/insert_onto_square_peg',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/place_cups',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/place_shape_in_shape_sorter',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_biSSM/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_biSSM_state_32/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_biSSM_state_64/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_64/all_tasks_LR_5e-5_2080_ti',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_patch_5/all_tasks_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_32/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_128/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_256/all_tasks_LR_5e-5',
    '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_biSSM/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_patch_5/all_tasks_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_550p/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_state_32/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_state_64/all_tasks_LR_5e-5',
    # '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_new_data/all_tasks_LR_5e-5',
    # Add more directories as needed
]
loss_tag = 'train_total_loss'  # Adjust this to match the actual tag used in your TensorBoard logs
val_success_rate_tag = 'Success_Rate'  # Adjust this to match the actual tag used in your TensorBoard logs
plot_training_losses(experiment_directories, loss_tag, val_success_rate_tag)