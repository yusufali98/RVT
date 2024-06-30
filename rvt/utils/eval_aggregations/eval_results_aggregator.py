import os
import pandas as pd

def get_average_success_rate(top_level_dir):
    results = {}

    # List all subdirectories in the top level experiment directory
    root_dirs = [os.path.join(top_level_dir, d) for d in os.listdir(top_level_dir) if os.path.isdir(os.path.join(top_level_dir, d))]

    # Loop through all identified root directories
    for root_dir in root_dirs:
        # Path to the evaluation directory
        eval_path = os.path.join(root_dir, 'eval', 'val')

        # Check if the eval/val path exists
        if not os.path.exists(eval_path):
            print(f"Folder '{eval_path}' not present, skipping...")
            continue  # Skip this root directory and move to the next
        
        # Loop through all directories in the eval_path
        for folder in os.listdir(eval_path):
            if folder.startswith("model_"):
                model_dir = os.path.join(eval_path, folder)
                # Extract the model identifier (e.g., model_2 from model_2_remaining_tasks)
                model_id = '_'.join(folder.split('_')[:2])  # This will get something like 'model_2'
                subfolder_path = os.path.join(model_dir, model_id)
                csv_path = os.path.join(subfolder_path, 'eval_results.csv')

                # Assert that the CSV file exists
                assert os.path.exists(csv_path), f"CSV file not found at {csv_path}"

                # Read the CSV file
                data = pd.read_csv(csv_path)

                # If the model_id is already in the results, append the new data
                if model_id in results:
                    results[model_id]['data'] = pd.concat([results[model_id]['data'], data], ignore_index=True)
                else:
                    results[model_id] = {'data': data}

    # Compute average success rates for each model_id and count the number of tasks
    average_rates = {}
    for model_id, info in results.items():
        avg_success_rate = info['data']['success rate'].mean()
        num_tasks = len(info['data']['success rate'])
        average_rates[model_id] = {'average success rate': avg_success_rate, 'number of tasks': num_tasks}

    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame.from_dict(average_rates, orient='index')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Model ID'}, inplace=True)

    # Sort the DataFrame by the model checkpoint index
    df['Model Index'] = df['Model ID'].apply(lambda x: int(x.split('_')[1]))
    df.sort_values('Model Index', inplace=True)
    df.drop('Model Index', axis=1, inplace=True)

    # Save the DataFrame to a CSV file in the top-level directory
    output_csv_path = os.path.join(top_level_dir, 'average_success_rates.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

    return average_rates

# Provide the path to the top-level experiment directory
top_level_directory = '/srv/kira-lab/share4/yali30/rvt_mamba/dev/RVT/rvt/VMamba_expts/mamba_330p/with_z_all_tasks_LR_5e-5_no_pos/'
average_rates = get_average_success_rate(top_level_directory)


def print_results(results):
    print("Model Checkpoint Results:")
    # Sort the dictionary by model IDs, which are assumed to be strings like 'model_1', 'model_2', etc.
    for model_id in sorted(results, key=lambda x: int(x.split('_')[1])):
        data = results[model_id]
        if data['number of tasks'] == 18:
            print(f"Model ID: {model_id}")
            print(f"  Average Success Rate: {data['average success rate']:.2f}%")
            print(f"  Number of Tasks: {data['number of tasks']}")
            print()  # Adds a newline for better separation

from prettytable import PrettyTable

def print_results_table(results):
    table = PrettyTable()
    table.field_names = ["Model ID", "Average Success Rate", "Number of Tasks"]

    # Sort results by model ID
    for model_id in sorted(results, key=lambda x: int(x.split('_')[1])):
        data = results[model_id]
        table.add_row([model_id, f"{data['average success rate']:.2f}%", data['number of tasks']])

    print(table)

# Assuming the `average_rates` dictionary is filled as previous examples
print_results_table(average_rates)


# Assuming the `average_rates` dictionary is filled as previous examples
# print_results(average_rates)