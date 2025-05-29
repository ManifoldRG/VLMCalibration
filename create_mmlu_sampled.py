import json
import os
import shutil
import re
import random
from collections import defaultdict
from pathlib import Path

def load_mmlu_data(file_path: str) -> list:
    """Load MMLU data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

def get_broad_categories() -> dict:
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

def group_items_by_category(data: list, subject_to_category: dict) -> dict:
    """Group MMLU items by broad categories."""
    category_items = defaultdict(list)
    
    for idx, item in enumerate(data):
        question = item["question"]
        subject = extract_subject_from_question(question)
        category = subject_to_category.get(subject, "Miscellaneous")
        category_items[category].append((idx, item))
    
    return category_items

def sample_uniformly_across_categories(category_items: dict, max_samples: int = 1500) -> list:
    """Sample uniformly across broad categories."""
    num_categories = len(category_items)
    base_samples_per_category = max_samples // num_categories
    remainder = max_samples % num_categories
    
    print(f"Total broad categories: {num_categories}")
    print(f"Base samples per category: {base_samples_per_category}")
    print(f"Categories getting extra sample: {remainder}")
    print(f"Total target samples: {max_samples}")
    
    sampled_data = []
    category_list = list(category_items.items())
    
    for i, (category, items) in enumerate(category_list):
        # Some categories get one extra sample to reach exactly max_samples
        samples_for_this_category = base_samples_per_category + (1 if i < remainder else 0)
        
        # Sample up to the calculated amount from this category
        sample_size = min(samples_for_this_category, len(items))
        sampled_items = random.sample(items, sample_size)
        
        print(f"{category}: {sample_size} samples (available: {len(items)})")
        
        for idx, item in sampled_items:
            sampled_data.append(item)
    
    return sampled_data

def create_sampled_directories(base_dirs: list, output_suffix: str = "_sampled") -> list:
    """Create new directories for sampled experiments."""
    created_dirs = []
    
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            sampled_dir = base_dir + output_suffix
            
            # Create the sampled directory
            os.makedirs(sampled_dir, exist_ok=True)
            created_dirs.append(sampled_dir)
            print(f"Created directory: {sampled_dir}")
            
            # Copy non-JSON files (like scripts, configs, etc.)
            for item in os.listdir(base_dir):
                item_path = os.path.join(base_dir, item)
                if os.path.isfile(item_path) and not item.endswith('.json'):
                    shutil.copy2(item_path, os.path.join(sampled_dir, item))
                    print(f"Copied: {item}")
        else:
            print(f"Warning: Directory {base_dir} does not exist")
    
    return created_dirs

def process_mmlu_files(base_dirs: list, max_samples: int = 1500, seed: int = 42) -> None:
    """Process all MMLU JSON files in the specified directories."""
    random.seed(seed)
    
    broad_categories = get_broad_categories()
    subject_to_category = create_subject_to_category_mapping(broad_categories)
    
    # Create sampled directories
    sampled_dirs = create_sampled_directories(base_dirs)
    
    for base_dir, sampled_dir in zip(base_dirs, sampled_dirs):
        if not os.path.exists(base_dir):
            continue
            
        print(f"\nProcessing directory: {base_dir}")
        
        # Find all JSON files with MMLU data, excluding experiment_details files
        for filename in os.listdir(base_dir):
            if (filename.endswith('.json') and 
                'experiment_details' not in filename.lower()):
                
                file_path = os.path.join(base_dir, filename)
                
                print(f"\nProcessing file: {filename}")
                
                try:
                    # Load and sample data
                    data = load_mmlu_data(file_path)
                    
                    # Check if data is a list of items or needs to be accessed differently
                    if isinstance(data, dict):
                        print(f"Warning: File {filename} contains dict, skipping...")
                        continue
                    
                    if not data or not isinstance(data[0], dict):
                        print(f"Warning: File {filename} has unexpected format, skipping...")
                        continue
                    
                    category_items = group_items_by_category(data, subject_to_category)
                    sampled_data = sample_uniformly_across_categories(category_items, max_samples)
                    
                    # Save sampled data
                    output_path = os.path.join(sampled_dir, filename)
                    with open(output_path, 'w') as f:
                        json.dump(sampled_data, f, indent=2)
                    
                    print(f"Saved {len(sampled_data)} samples to: {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    continue

def main():
    """Main function to create sampled MMLU experiments."""
    # Define base directories to process
    base_directories = [
        "zs_exp/mmlu",
        "cot_exp/mmlu"
    ]
    
    # Set parameters
    max_samples = 1500
    random_seed = 42
    
    print("Creating sampled MMLU experiments...")
    print(f"Max samples per file: {max_samples}")
    print(f"Random seed: {random_seed}")
    
    # Process all directories
    process_mmlu_files(base_directories, max_samples, random_seed)
    
    print("\nSampling complete!")
    print("New directories created:")
    for base_dir in base_directories:
        sampled_dir = base_dir + "_sampled"
        if os.path.exists(sampled_dir):
            print(f"  {sampled_dir}")

if __name__ == "__main__":
    main() 