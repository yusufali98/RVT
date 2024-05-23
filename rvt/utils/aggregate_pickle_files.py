import os
import pickle
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_single_pickle_file(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def load_pickle_files(directory):
    data = []
    pickle_files = glob.glob(os.path.join(directory, '*.replay'))
    total_files = len(pickle_files)
    print(f"Found {total_files} files to load.")

    with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers based on your system's capability
        futures = {executor.submit(load_single_pickle_file, file): file for file in pickle_files}
        
        for count, future in enumerate(as_completed(futures), 1):
            file = futures[future]
            try:
                content = future.result()
                data.append(content)  # Assuming the content is a list-like structure
                print(f"Loaded {count}/{total_files}: {file}")
            except Exception as exc:
                print(f"Error loading file {file}: {exc}")
    
    return data

def save_to_pickle(data, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved combined data to {output_file}")

def main():
    directory = '/workspace/RVT/rvt/replay/replay_train/close_jar'  # Replace with your directory path
    output_file = 'big_replay.pkl'  # Output file name
    data = load_pickle_files(directory)
    save_to_pickle(data, output_file)

if __name__ == "__main__":
    main()
