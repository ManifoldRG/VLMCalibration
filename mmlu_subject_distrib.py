import json
import matplotlib.pyplot as plt
import re
import random
from collections import defaultdict

path = r"E:\Manifold\VLMCalibration\zs_exp\mmlu\zs_exp_records_test_Qwen25-3B-Instruct.json"

with open(path, "r") as f:
    data = json.load(f)

# Define broad categories based on the image
BROAD_CATEGORIES = {
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

# Create reverse mapping from subject to broad category
subject_to_category = {}
for category, subjects in BROAD_CATEGORIES.items():
    for subject in subjects:
        subject_to_category[subject] = category

# Group items by subject
subject_items = defaultdict(list)
for idx, item in enumerate(data):
    question = item["question"]
    subject_match = re.search(r'The following is a multiple choice question \(with choices\) about (.+?)\.', question)
    if subject_match:
        subject = subject_match.group(1)
        subject_items[subject].append((idx, item))

# Group by broad categories
category_items = defaultdict(list)
for subject, items in subject_items.items():
    category = subject_to_category.get(subject, "Miscellaneous")
    category_items[category].extend(items)

# Calculate uniform sampling across broad categories
max_samples = 1500
num_categories = len(category_items)
base_samples_per_category = max_samples // num_categories
remainder = max_samples % num_categories

print(f"Total broad categories: {num_categories}")
print(f"Base samples per category: {base_samples_per_category}")
print(f"Categories getting extra sample: {remainder}")
print(f"Total samples: {max_samples}")

# Sample uniformly from each broad category
sampled_data = []
sampled_subjects = []
sampled_categories = []

# Convert to list for consistent ordering
category_list = list(category_items.items())

for i, (category, items) in enumerate(category_list):
    # Some categories get one extra sample to reach exactly 2000
    samples_for_this_category = base_samples_per_category + (1 if i < remainder else 0)
    
    # Sample up to the calculated amount from this category
    sample_size = min(samples_for_this_category, len(items))
    sampled_items = random.sample(items, sample_size)
    
    for idx, item in sampled_items:
        sampled_data.append(item)
        # Extract subject from the item
        question = item["question"]
        subject_match = re.search(r'The following is a multiple choice question \(with choices\) about (.+?)\.', question)
        if subject_match:
            subject = subject_match.group(1)
            sampled_subjects.append(subject)
            sampled_categories.append(category)

print(f"Actual total samples: {len(sampled_data)}")

# Create broad category distribution chart
category_counts = {}
for category in sampled_categories:
    category_counts[category] = category_counts.get(category, 0) + 1

plt.figure(figsize=(14, 8))
plt.bar(category_counts.keys(), category_counts.values())
plt.xticks(rotation=45, ha='right')
plt.title(f'MMLU Broad Category Distribution (Uniform Sampling, n={len(sampled_data)})')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Create detailed subject distribution chart within categories
subject_counts = {}
for subject in sampled_subjects:
    subject_counts[subject] = subject_counts.get(subject, 0) + 1

plt.figure(figsize=(16, 8))
plt.bar(subject_counts.keys(), subject_counts.values())
plt.xticks(rotation=90)
plt.title(f'MMLU Subject Distribution by Broad Category (Uniform Sampling, n={len(sampled_data)})')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Print distribution summary
print(f"\nBroad category distribution:")
for category, count in sorted(category_counts.items()):
    print(f"{category}: {count}")

print(f"\nDetailed subject distribution:")
for category, subjects in BROAD_CATEGORIES.items():
    if category in category_counts:
        print(f"\n{category} ({category_counts[category]} samples):")
        for subject in subjects:
            if subject in subject_counts:
                print(f"  {subject}: {subject_counts[subject]}")
