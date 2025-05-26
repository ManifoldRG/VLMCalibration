import json
import matplotlib.pyplot as plt
import re


path = r"E:\Manifold\VLMCalibration\zs_exp\mmlu\zs_exp_records_test_Qwen25-3B-Instruct.json"

with open(path, "r") as f:
    data = json.load(f)

# print(data)

subjects = []
for idx, item in enumerate(data):
    question = item["question"]
    subject_match = re.search(r'The following is a multiple choice question \(with choices\) about (.+?)\.', question)
    if subject_match:
        subjects.append(subject_match.group(1))

# Create subject distribution chart
subject_counts = {}
for subject in subjects:
    subject_counts[subject] = subject_counts.get(subject, 0) + 1

plt.figure(figsize=(12, 6))
plt.bar(subject_counts.keys(), subject_counts.values())
plt.xticks(rotation=90)
plt.title('MMLU Subject Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
