import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import os

def generate_tb_val_file(csv_file_path):

    # Set up the path to the CSV file
    # csv_file_path = '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p/all_tasks_LR_5e-5/average_success_rates.csv'
    event_file_directory = os.path.dirname(csv_file_path)  # Use the same directory as the CSV file

    # Read the CSV data from the file
    df = pd.read_csv(csv_file_path)

    # Initialize a writer to save the event file in the same directory as the CSV file
    writer = SummaryWriter(log_dir=event_file_directory)

    # Iterate over the DataFrame and log the success rate to TensorBoard
    for index, row in df.iterrows():
        model_id = row['Model ID']
        success_rate = row['average success rate']

        # Log the number of tasks over which the success rate was averaged
        num_tasks = row['number of tasks']
        
        if num_tasks == 18:
            checkpoint_idx = int(model_id.split('_')[-1])
            writer.add_scalar('Num_Tasks_Validated', num_tasks, checkpoint_idx)

            # Log the success rate using the model's numeric ID as the step value
            writer.add_scalar('Success_Rate', success_rate, checkpoint_idx)

    # Close the writer to flush/save the event file
    writer.close()


# paths = ['/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/rvt_220p/all_tasks/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/rvt_330p/all_tasks/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p/all_tasks_5e5_LR_orig/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p/all_tasks_LR_5e-5/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_550p/all_tasks_LR_5e-5/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_patch_5/all_tasks_5e-5/average_success_rates.csv',
#          '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_patch_5/all_tasks_5e-5/average_success_rates.csv']

paths =['/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_24L/all_tasks_LR_1e-5_2080_ti/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_biSSM/all_tasks_LR_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_patch_5/all_tasks_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_64/all_tasks_LR_5e-5_2080_ti/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_128/all_tasks_LR_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_220p_state_256/all_tasks_LR_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_biSSM/all_tasks_LR_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_patch_5/all_tasks_5e-5/average_success_rates.csv',
        '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_550p/all_tasks_LR_5e-5/average_success_rates.csv']

for path in paths:
    generate_tb_val_file(path)