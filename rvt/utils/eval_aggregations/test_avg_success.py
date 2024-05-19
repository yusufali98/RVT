import pandas as pd
from io import StringIO

# Path to the CSV file
file_path = '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/high_precision_expts/mamba_550p/all_tasks_LR_5e-5/rvt_tasks_all_PA.lr_5e-5_E_15_RES_use_mamba_T_depth_16_IS_550/eval/test/model_14/model_14/eval_results.csv'

# Read the CSV data
df = pd.read_csv(file_path)

# Calculate the average success rate over all tasks
average_success_rate = df["success rate"].mean()

print("Test SR: ", average_success_rate)
print("Num tasks: ", len(df))

# We'll create a dictionary to map tasks to their success rates.
task_order = [
    "close_jar", "reach_and_drag", "insert_onto_square_peg", "meat_off_grill",
    "open_drawer", "place_cups", "place_wine_at_rack_location", "push_buttons",
    "put_groceries_in_cupboard", "put_item_in_drawer", "put_money_in_safe", "light_bulb_in",
    "slide_block_to_color_target", "place_shape_in_shape_sorter", "stack_blocks", "stack_cups",
    "sweep_to_dustpan_of_size", "turn_tap"
]

# Create a dictionary from the dataframe for quick lookup
success_rates = dict(zip(df['task'], df['success rate']))

# Now we'll create a string that has the success rates in the specified order
success_rate_string = ' & '.join(
    f"{success_rates.get(task)}" 
    for task in task_order
)

print("Latex format: \n")
print(success_rate_string)