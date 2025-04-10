import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc
import os
import argparse

# Add argument parsing to match the server script
parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=["gpt-4o", "gpt-4o-mini"], help="Name of OpenAI model to use")
parser.add_argument("exp_type", choices=["cot_exp", "zs_exp"], help="Experiment type")
parser.add_argument("dataset", choices=["gsm8k"], help="Dataset to analyze")  # Currently only GSM8K supported
parser.add_argument("dataset_split", choices=["train", "test"], help="Dataset split to analyze")
args = parser.parse_args()

# Set up paths based on args
BASE_DIR = f"{args.exp_type}/{args.dataset}"
OUTPUT_DIR = os.path.join(BASE_DIR, args.model_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and process data
data_path = os.path.join(BASE_DIR, f"{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv")
df = pd.read_csv(data_path)

# Count and handle NaN values
na_count = df.isna().any(axis=1).sum()
print("Number of rows with NaN in CSV:", na_count)
df = df.dropna()

# Calculate and print statistics
mean_conf_correct = df[df['correct'] == True]['p_true'].mean()
mean_conf_incorrect = df[df['correct'] == False]['p_true'].mean()
print("Mean confidence (p_true) when correct:", mean_conf_correct)
print("Mean confidence (p_true) when incorrect:", mean_conf_incorrect)

# Plot confidence histograms
plt.figure(figsize=(6, 4))
bins = np.linspace(0, 1, 51)
plt.hist(df[df['correct'] == True]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Correct", edgecolor='black')
plt.hist(df[df['correct'] == False]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Incorrect", edgecolor='black')
plt.xlabel('Confidence (p_true)')
plt.ylabel('Density')
plt.title(f'Confidence Distribution for {args.dataset} ({args.model_name})')
plt.xlim(0, 1)
plt.legend()
plt.grid(True)

# Save confidence histogram
hist_path = os.path.join(OUTPUT_DIR, f'confidence_histogram_{args.dataset}_{args.model_name}.png')
plt.savefig(hist_path)
plt.close()

# Plot calibration curve
plt.figure(figsize=(6, 4))
df_grouped = df.groupby(pd.qcut(df['p_true'], 10))['correct'].agg(['mean', 'count']).reset_index()
df_grouped['p_true'] = df_grouped['p_true'].apply(lambda x: x.mid)

plt.scatter(df_grouped['p_true'], df_grouped['mean'], s=df_grouped['count']/30,
           alpha=0.6, label='Observed')
plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')

# Fit isotonic regression
ir = IsotonicRegression(out_of_bounds='clip')
ir.fit(df['p_true'], df['correct'])
plt.plot(np.linspace(0, 1, 100), 
         ir.predict(np.linspace(0, 1, 100)),
         'g-', label='Isotonic Regression')

plt.xlabel('Confidence (p_true)')
plt.ylabel('Empirical Accuracy')
plt.title(f'Calibration Curve for {args.dataset} ({args.model_name})')
plt.legend()
plt.grid(True)

# Save calibration curve
calib_path = os.path.join(OUTPUT_DIR, f'calibration_curve_{args.dataset}_{args.model_name}.png')
plt.savefig(calib_path)
plt.close()

# Plot ROC curve
fpr, tpr, _ = roc_curve(df['correct'], df['p_true'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve for {args.dataset} ({args.model_name})')
plt.legend(loc="lower right")
plt.grid(True)

# Save ROC curve
roc_path = os.path.join(OUTPUT_DIR, f'roc_curve_{args.dataset}_{args.model_name}.png')
plt.savefig(roc_path)
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print(f"Total samples: {len(df)}")
print(f"Accuracy: {df['correct'].mean():.3f}")
print(f"ROC AUC: {roc_auc:.3f}")

# Calculate and print calibration metrics
expected_accuracy = df['p_true'].mean()
actual_accuracy = df['correct'].mean()
print(f"Expected accuracy (mean confidence): {expected_accuracy:.3f}")
print(f"Actual accuracy: {actual_accuracy:.3f}")
print(f"Calibration error: {abs(expected_accuracy - actual_accuracy):.3f}") 