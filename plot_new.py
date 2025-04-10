import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc
import os
import argparse

# Add argument parsing to match the server script
parser = argparse.ArgumentParser()
parser.add_argument("model_name", choices=["llama", "qwen", "gemma"], help="Name of model to use")
parser.add_argument("exp_type", choices=["cot_exp", "zs_exp"], help="Experiment type")
parser.add_argument("dataset", choices=["simpleqa", "gsm8k", "mmlu", "medmcqa"], help="Dataset to analyze")
parser.add_argument("dataset_split", choices=["train", "val", "test"], help="Dataset split to analyze")
args = parser.parse_args()

# Set up paths based on args
BASE_DIR = f"{args.exp_type}/{args.dataset}"
OUTPUT_DIR = os.path.join(BASE_DIR, args.model_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and process training data
train_path = os.path.join(BASE_DIR, f"{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv")
train_df = pd.read_csv(train_path)

# Count and handle NaN values
na_train = train_df.isna().any(axis=1).sum()
print("Number of rows with NaN in CSV:", na_train)
train_df = train_df.dropna()

# Calculate and print statistics
mean_conf_correct = train_df[train_df['correct'] == True]['p_true'].mean()
mean_conf_incorrect = train_df[train_df['correct'] == False]['p_true'].mean()
print("Mean confidence (p_true) when correct:", mean_conf_correct)
print("Mean confidence (p_true) when incorrect:", mean_conf_incorrect)

# Plot confidence histograms
plt.figure(figsize=(6, 4))
bins = np.linspace(0, 1, 51)
plt.hist(train_df[train_df['correct'] == True]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Correct", edgecolor='black')
plt.hist(train_df[train_df['correct'] == False]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Incorrect", edgecolor='black')
plt.xlabel('Confidence (p_true)')
plt.ylabel('Density')
plt.title(f'Confidence Distribution for {args.dataset} ({args.model_name})')
plt.xlim(0, 1)
plt.legend()
plt.grid(True)

# Save plot with appropriate naming
plot_path = os.path.join(OUTPUT_DIR, f'confidence_histogram_{args.dataset}_{args.model_name}.png')
plt.savefig(plot_path)
plt.close()
