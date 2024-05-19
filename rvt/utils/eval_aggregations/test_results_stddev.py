import os
import pandas as pd
import numpy as np

def find_csv_files(root_dir):
    """ Recursively find all CSV files in the given directory. """
    csv_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    return csv_files

def process_csv_files(csv_files):
    """ Process each CSV file to calculate averages and standard deviations. """
    all_data = []
    run_averages = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        avg_success_rate = df['success rate'].mean()
        run_averages.append(avg_success_rate)
        all_data.append(df)
    
    # Concatenate all dataframes for overall analysis
    combined_df = pd.concat(all_data)
    overall_average = np.mean(run_averages)
    task_average = combined_df.groupby('task')['success rate'].mean()
    task_std_dev = combined_df.groupby('task')['success rate'].std()

    return run_averages, overall_average, task_average, task_std_dev

def main(root_dir, task_order, files_to_consider):
    """ Main function to execute the processing steps. """
    csv_files = sorted(find_csv_files(root_dir))

    csv_files = csv_files[:files_to_consider]

    run_averages, overall_average, task_average, task_std_dev = process_csv_files(csv_files)

    print(f"Average success rate per run: {run_averages}")
    print(f"Overall average success rate across all runs: {overall_average:.2f}%")
    print("Average success rate per task across all runs:")
    print(task_average)
    print("Standard deviation of success rate per task across all runs:")
    print(task_std_dev)

    # Prepare LaTeX string in the specified task order
    latex_parts = []
    for task in task_order:
        avg = task_average.get(task, 0)
        std = task_std_dev.get(task, 0)
        latex_parts.append(f"${avg:.2f} \\pm {std:.2f}$")
    latex_string = " & ".join(latex_parts)
    print("LaTeX code for average success rate and standard deviation per task:")
    print(latex_string)

# Set the root directory of your test results
root_dir = '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_330p_new_data/all_tasks_LR_5e-5/rvt_tasks_all_NW_6_PA.lr_5e-5_E_15_RES_use_mamba_T_depth_16_IS_330/eval/test_alternate'

task_order = [
    "close_jar", "reach_and_drag", "insert_onto_square_peg", "meat_off_grill",
    "open_drawer", "place_cups", "place_wine_at_rack_location", "push_buttons",
    "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe", "light_bulb_in",
    "slide_block_to_color_target", "place_shape_in_shape_sorter", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]

main(root_dir, task_order, 5)