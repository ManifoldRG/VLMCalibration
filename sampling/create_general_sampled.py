import json
import os
import shutil
import random
from pathlib import Path

def load_json_data(file_path: str) -> list:
    """Load data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def sample_random_items(data: list, max_samples: int = 1500) -> list:
    """Sample random items from the data."""
    if len(data) <= max_samples:
        print(f"Data has {len(data)} items, returning all")
        return data
    
    sampled_data = random.sample(data, max_samples)
    print(f"Sampled {len(sampled_data)} items from {len(data)} total")
    return sampled_data

def create_sampled_directories(base_dirs: list, target_datasets: list, output_suffix: str = "_sampled") -> dict:
    """Create new directories for sampled experiments."""
    created_dirs = {}
    
    for base_dir in base_dirs:
        for dataset in target_datasets:
            dataset_dir = os.path.join(base_dir, dataset)
            
            if os.path.exists(dataset_dir):
                sampled_dir = dataset_dir + output_suffix
                
                # Create the sampled directory
                os.makedirs(sampled_dir, exist_ok=True)
                created_dirs[dataset_dir] = sampled_dir
                print(f"Created directory: {sampled_dir}")
                
                # Copy non-JSON files (like scripts, configs, etc.)
                for item in os.listdir(dataset_dir):
                    item_path = os.path.join(dataset_dir, item)
                    if os.path.isfile(item_path) and not item.endswith('.json'):
                        shutil.copy2(item_path, os.path.join(sampled_dir, item))
                        print(f"Copied: {item}")
            else:
                print(f"Warning: Directory {dataset_dir} does not exist")
    
    return created_dirs

def process_json_files(directory_mapping: dict, max_samples: int = 1500, seed: int = 42) -> None:
    """Process all JSON files in the specified directories."""
    random.seed(seed)
    
    for original_dir, sampled_dir in directory_mapping.items():
        if not os.path.exists(original_dir):
            continue
            
        print(f"\nProcessing directory: {original_dir}")
        
        # Find all JSON files, excluding experiment_details files
        for filename in os.listdir(original_dir):
            if (filename.endswith('.json') and 
                'experiment_details' not in filename.lower()):
                
                file_path = os.path.join(original_dir, filename)
                
                print(f"\nProcessing file: {filename}")
                
                try:
                    # Load and sample data
                    data = load_json_data(file_path)
                    
                    # Check if data is a list of items or needs to be accessed differently
                    if isinstance(data, dict):
                        print(f"Warning: File {filename} contains dict, skipping...")
                        continue
                    
                    if not data or not isinstance(data[0], dict):
                        print(f"Warning: File {filename} has unexpected format, skipping...")
                        continue
                    
                    sampled_data = sample_random_items(data, max_samples)
                    
                    # Save sampled data
                    output_path = os.path.join(sampled_dir, filename)
                    with open(output_path, 'w') as f:
                        json.dump(sampled_data, f, indent=2)
                    
                    print(f"Saved {len(sampled_data)} samples to: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

def main():
    """Main function to create sampled experiments for medmcqa and simpleqa."""
    # Define base directories to process
    base_directories = [
        "cot_exp",
        "zs_exp"
    ]
    
    # Define target datasets
    target_datasets = [
        "medmcqa",
        "simpleqa"
    ]
    
    # Set parameters
    max_samples = 1500
    random_seed = 42
    
    print("Creating sampled experiments for medmcqa and simpleqa...")
    print(f"Max samples per file: {max_samples}")
    print(f"Random seed: {random_seed}")
    print(f"Target datasets: {target_datasets}")
    
    # Create sampled directories
    directory_mapping = create_sampled_directories(base_directories, target_datasets)
    
    # Process all directories
    process_json_files(directory_mapping, max_samples, random_seed)
    
    print("\nSampling complete!")
    print("New directories created:")
    for original_dir, sampled_dir in directory_mapping.items():
        if os.path.exists(sampled_dir):
            print(f"  {sampled_dir}")

if __name__ == "__main__":
    main() 