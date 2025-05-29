import json
import os
import shutil
import random
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

def load_json_data(file_path: str) -> list:
    """Load data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def get_mmlu_broad_categories() -> dict:
    """Define broad categories for MMLU subjects."""
    return {
        "Quantitative & Formal Reasoning": [
            "abstract algebra", "elementary mathematics", "high school mathematics", 
            "college mathematics", "formal logic", "logical fallacies", "high school statistics"
        ],
        "Physical Sciences": [
            "astronomy", "conceptual physics", "high school physics", "college physics", 
            "high school chemistry", "college chemistry"
        ],
        "Life Sciences, Health & Medicine": [
            "anatomy", "high school biology", "college biology", "human aging", 
            "human sexuality", "virology", "nutrition", "college medicine", 
            "professional medicine", "clinical knowledge", "medical genetics"
        ],
        "Computing, Engineering & Technology": [
            "high school computer science", "college computer science", "machine learning", 
            "computer security", "electrical engineering"
        ],
        "Social Sciences & Human Behavior": [
            "high school microeconomics", "high school macroeconomics", "management", 
            "marketing", "business ethics", "professional accounting", "high school psychology", 
            "professional psychology", "sociology", "high school government and politics", 
            "us foreign policy", "security studies", "international law", "jurisprudence", 
            "public relations"
        ],
        "Humanities & Ethical Inquiry": [
            "philosophy", "moral disputes", "moral scenarios", "high school us history", 
            "high school world history", "high school european history", "prehistory", 
            "world religions", "global facts"
        ],
        "Miscellaneous": [
            "miscellaneous"
        ]
    }

def create_subject_to_category_mapping(broad_categories: dict) -> dict:
    """Create reverse mapping from subject to broad category."""
    subject_to_category = {}
    for category, subjects in broad_categories.items():
        for subject in subjects:
            subject_to_category[subject] = category
    return subject_to_category

def extract_subject_from_question(question: str) -> str:
    """Extract subject from MMLU question text."""
    subject_match = re.search(r'The following is a multiple choice question \(with choices\) about (.+?)\.', question)
    return subject_match.group(1) if subject_match else "unknown"

def group_mmlu_items_by_category(data: list, subject_to_category: dict) -> dict:
    """Group MMLU items by broad categories, returning indexes."""
    category_items = defaultdict(list)
    
    for item in data:
        if 'idx' not in item:
            continue
            
        question = item["question"]
        subject = extract_subject_from_question(question)
        category = subject_to_category.get(subject, "Miscellaneous")
        category_items[category].append(item['idx'])
    
    return category_items

def sample_mmlu_uniformly_across_categories(category_items: dict, max_samples: int = 1500) -> list:
    """Sample MMLU indexes uniformly across broad categories, ensuring exactly max_samples total."""
    num_categories = len(category_items)
    base_samples_per_category = max_samples // num_categories
    remainder = max_samples % num_categories
    
    print(f"Total broad categories: {num_categories}")
    print(f"Base samples per category: {base_samples_per_category}")
    print(f"Categories getting extra sample: {remainder}")
    print(f"Total target samples: {max_samples}")
    
    # First pass: allocate ideal samples per category
    ideal_allocation = {}
    category_list = list(category_items.items())
    
    for i, (category, indexes) in enumerate(category_list):
        # Some categories get one extra sample to reach exactly max_samples
        ideal_samples = base_samples_per_category + (1 if i < remainder else 0)
        ideal_allocation[category] = ideal_samples
    
    # Second pass: adjust for categories that don't have enough samples
    actual_allocation = {}
    shortfall = 0
    
    for category, indexes in category_items.items():
        ideal_samples = ideal_allocation[category]
        available_samples = len(indexes)
        
        if available_samples >= ideal_samples:
            actual_allocation[category] = ideal_samples
        else:
            actual_allocation[category] = available_samples
            shortfall += (ideal_samples - available_samples)
            print(f"Warning: {category} only has {available_samples} samples (wanted {ideal_samples})")
    
    # Third pass: redistribute shortfall to categories with extra capacity
    if shortfall > 0:
        print(f"Redistributing {shortfall} samples to categories with extra capacity...")
        
        # Sort categories by available extra capacity (descending)
        extra_capacity = []
        for category, indexes in category_items.items():
            allocated = actual_allocation[category]
            available = len(indexes)
            extra = available - allocated
            if extra > 0:
                extra_capacity.append((category, extra))
        
        extra_capacity.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute shortfall
        for category, extra in extra_capacity:
            if shortfall <= 0:
                break
            
            additional_samples = min(shortfall, extra)
            actual_allocation[category] += additional_samples
            shortfall -= additional_samples
            print(f"Added {additional_samples} extra samples to {category}")
    
    # Final sampling based on actual allocation
    sampled_indexes = []
    
    for category, indexes in category_items.items():
        sample_size = actual_allocation[category]
        
        if sample_size > 0:
            sampled_category_indexes = random.sample(indexes, sample_size)
            sampled_indexes.extend(sampled_category_indexes)
            print(f"{category}: {sample_size} samples (available: {len(indexes)})")
    
    print(f"Final total sampled: {len(sampled_indexes)} (target: {max_samples})")
    
    # If we still have a shortfall (shouldn't happen if we have enough total samples)
    if len(sampled_indexes) < max_samples:
        remaining_needed = max_samples - len(sampled_indexes)
        print(f"Warning: Still need {remaining_needed} more samples. This shouldn't happen if enough data is available.")
    
    return sampled_indexes

def get_all_indexes_for_subject(base_dirs: list, subject: str) -> list:
    """Get all available indexes for a given subject from any available file."""
    all_indexes = set()
    sample_data = None  # Store data for MMLU category analysis
    
    for base_dir in base_dirs:
        subject_dir = os.path.join(base_dir, subject)
        if not os.path.exists(subject_dir):
            continue
            
        # Find first JSON file (excluding experiment_details) to get indexes
        for filename in os.listdir(subject_dir):
            if (filename.endswith('.json') and 
                'experiment_details' not in filename.lower()):
                
                file_path = os.path.join(subject_dir, filename)
                try:
                    data = load_json_data(file_path)
                    
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        # Store data for MMLU analysis
                        if subject == "mmlu":
                            sample_data = data
                        
                        # Extract all idx values
                        for item in data:
                            if 'idx' in item:
                                all_indexes.add(item['idx'])
                        
                        print(f"Found {len(all_indexes)} indexes in {file_path}")
                        # Return after finding first valid file to avoid duplicates
                        return sorted(list(all_indexes)), sample_data
                        
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    continue
    
    return sorted(list(all_indexes)), sample_data

def plot_mmlu_distribution(category_items: dict, sampled_indexes: list, subject: str) -> None:
    """Plot the distribution of sampled MMLU items across broad categories."""
    # Count sampled items per category
    sampled_per_category = {}
    sampled_set = set(sampled_indexes)
    
    for category, indexes in category_items.items():
        sampled_count = len([idx for idx in indexes if idx in sampled_set])
        sampled_per_category[category] = sampled_count
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    categories = list(sampled_per_category.keys())
    counts = list(sampled_per_category.values())
    
    # Create bar plot
    bars = plt.bar(categories, counts, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Customize the plot
    plt.title(f'MMLU Broad Category Distribution After Sampling\n(Total: {len(sampled_indexes)} samples)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Broad Categories', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add grid for better readability
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plot_filename = f"{subject}_distribution.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plot to: {plot_filename}")
    
    # Show summary statistics
    print(f"\nMMlu Category Distribution Summary:")
    print(f"Total sampled: {len(sampled_indexes)}")
    print(f"Categories: {len(categories)}")
    print(f"Average per category: {len(sampled_indexes) / len(categories):.1f}")
    print(f"Min samples: {min(counts)}")
    print(f"Max samples: {max(counts)}")
    
    for category, count in sorted(sampled_per_category.items()):
        percentage = (count / len(sampled_indexes)) * 100
        print(f"  {category}: {count} samples ({percentage:.1f}%)")
    
    plt.show()

def create_subject_index_samples(base_dirs: list, target_datasets: list, max_samples: int = 1500, seed: int = 42) -> dict:
    """Create and save sampled indexes for each subject."""
    subject_indexes = {}
    
    # Set random seed for consistent sampling
    random.seed(seed)
    
    for subject in target_datasets:
        print(f"\nProcessing indexes for subject: {subject}")
        
        # Get all available indexes for this subject
        all_indexes, sample_data = get_all_indexes_for_subject(base_dirs, subject)
        
        if not all_indexes:
            print(f"Warning: No indexes found for subject {subject}")
            continue
            
        print(f"Total indexes available for {subject}: {len(all_indexes)}")
        
        # Special handling for MMLU with uniform category sampling
        if subject == "mmlu" and sample_data:
            print("Using uniform distribution across MMLU broad categories...")
            
            # Get MMLU categories and mapping
            broad_categories = get_mmlu_broad_categories()
            subject_to_category = create_subject_to_category_mapping(broad_categories)
            
            # Group items by category
            category_items = group_mmlu_items_by_category(sample_data, subject_to_category)
            
            # Sample uniformly across categories
            if len(all_indexes) <= max_samples:
                sampled_indexes = all_indexes
                print(f"Using all {len(sampled_indexes)} indexes for {subject}")
            else:
                sampled_indexes = sample_mmlu_uniformly_across_categories(category_items, max_samples)
                print(f"Sampled {len(sampled_indexes)} indexes uniformly across categories for {subject}")
            
            # Plot the distribution for MMLU
            plot_mmlu_distribution(category_items, sampled_indexes, subject)
        
        # Regular random sampling for other subjects
        else:
            if len(all_indexes) <= max_samples:
                sampled_indexes = all_indexes
                print(f"Using all {len(sampled_indexes)} indexes for {subject}")
            else:
                sampled_indexes = random.sample(all_indexes, max_samples)
                print(f"Sampled {len(sampled_indexes)} indexes from {len(all_indexes)} total for {subject}")
        
        # Store the sampled indexes
        subject_indexes[subject] = sorted(sampled_indexes)
        
        # Save to JSON file
        output_file = f"{subject}.json"
        with open(output_file, 'w') as f:
            json.dump(sorted(sampled_indexes), f, indent=2)
        
        print(f"Saved sampled indexes to: {output_file}")
    
    return subject_indexes

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

def process_json_files_with_indexes(directory_mapping: dict, subject_indexes: dict) -> None:
    """Process all JSON files using pre-selected indexes for each subject."""
    
    for original_dir, sampled_dir in directory_mapping.items():
        if not os.path.exists(original_dir):
            continue
            
        # Extract subject name from directory path
        subject = os.path.basename(original_dir)
        
        # Remove _sampled suffix if present
        if subject.endswith('_sampled'):
            subject = subject[:-8]
            
        if subject not in subject_indexes:
            print(f"Warning: No sampled indexes found for subject {subject}")
            continue
            
        selected_indexes = set(subject_indexes[subject])
        print(f"\nProcessing directory: {original_dir}")
        print(f"Using {len(selected_indexes)} pre-selected indexes for {subject}")
        
        # Find all JSON files, excluding experiment_details files
        for filename in os.listdir(original_dir):
            if (filename.endswith('.json') and 
                'experiment_details' not in filename.lower()):
                
                file_path = os.path.join(original_dir, filename)
                
                print(f"\nProcessing file: {filename}")
                
                try:
                    # Load data
                    data = load_json_data(file_path)
                    
                    # Check if data is a list of items
                    if isinstance(data, dict):
                        print(f"Warning: File {filename} contains dict, skipping...")
                        continue
                    
                    if not data or not isinstance(data[0], dict):
                        print(f"Warning: File {filename} has unexpected format, skipping...")
                        continue
                    
                    # Filter data based on pre-selected indexes
                    filtered_data = []
                    for item in data:
                        if 'idx' in item and item['idx'] in selected_indexes:
                            filtered_data.append(item)
                    
                    print(f"Filtered {len(filtered_data)} items from {len(data)} total using pre-selected indexes")
                    
                    # Save filtered data
                    output_path = os.path.join(sampled_dir, filename)
                    with open(output_path, 'w') as f:
                        json.dump(filtered_data, f, indent=2)
                    
                    print(f"Saved {len(filtered_data)} samples to: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

def main():
    """Main function to create sampled experiments with consistent indexing."""
    # Define base directories to process
    base_directories = [
        "cot_exp",
        "zs_exp"
    ]
    
    # Define target datasets
    target_datasets = [
        "mmlu",
        "medmcqa",
        "simpleqa"
    ]
    
    # Set parameters
    max_samples = 1500
    random_seed = 42
    
    print("Creating consistent index sampling for subjects...")
    print(f"Max samples per subject: {max_samples}")
    print(f"Random seed: {random_seed}")
    print(f"Target datasets: {target_datasets}")
    
    # Step 1: Create and save sampled indexes for each subject
    subject_indexes = create_subject_index_samples(base_directories, target_datasets, max_samples, random_seed)
    
    # Step 2: Create sampled directories
    directory_mapping = create_sampled_directories(base_directories, target_datasets)
    
    # Step 3: Process all directories using the pre-selected indexes
    process_json_files_with_indexes(directory_mapping, subject_indexes)
    
    print("\nSampling complete!")
    print("Index files created:")
    for subject in target_datasets:
        if subject in subject_indexes:
            print(f"  {subject}.json (with {len(subject_indexes[subject])} indexes)")
    
    print("\nNew directories created:")
    for original_dir, sampled_dir in directory_mapping.items():
        if os.path.exists(sampled_dir):
            print(f"  {sampled_dir}")

if __name__ == "__main__":
    main() 