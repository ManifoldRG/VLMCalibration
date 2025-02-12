import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc
import os

# ---------------------------
# Load Data and Remove NaNs
# ---------------------------
# Load train.csv

MODEL_NAME = "cot/gsm8k/qwen"
os.makedirs(f'{MODEL_NAME}', exist_ok=True)


# train_df = pd.read_csv(f'train_results/records_train_full_{MODEL_NAME}.csv', index_col=0)
train_df = pd.read_csv(f'cot_records_train_full_qwen.csv', index_col=0)
# test_df = pd.read_csv(f'test_results/records_test_full_{MODEL_NAME}.csv', index_col=0)
# Count rows with any NaN values in train
na_train = train_df.isna().any(axis=1).sum()
print("Number of rows with NaN in train.csv:", na_train)
# Drop any rows with NaNs
train_df = train_df.dropna()

# # Load test.csv
# # Count rows with any NaN values in test
# na_test = test_df.isna().any(axis=1).sum()
# print("Number of rows with NaN in test.csv:", na_test)
# # Drop any rows with NaNs
# test_df = test_df.dropna()

# --------------------------------
# Print mean confidence for each group in train
# --------------------------------
mean_conf_correct = train_df[train_df['correct'] == True]['p_true'].mean()
mean_conf_incorrect = train_df[train_df['correct'] == False]['p_true'].mean()
print("Mean confidence (p_true) when correct:", mean_conf_correct)
print("Mean confidence (p_true) when incorrect:", mean_conf_incorrect)

# ===============================
# 2. Plot 1: Overlaid Histograms of Confidence on Train Set
# ===============================
plt.figure(figsize=(6, 4))
bins = np.linspace(0, 1, 51)
plt.hist(train_df[train_df['correct'] == True]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Correct", edgecolor='black')
plt.hist(train_df[train_df['correct'] == False]['p_true'], bins=bins, alpha=0.5, density=True,
         label="Confidence when Incorrect", edgecolor='black')
plt.xlabel('Confidence (p_true)')
plt.ylabel('Density')
plt.title('Histogram of p_true for Train Set (Correct vs Incorrect)')
plt.xlim(0, 1)
plt.legend()
plt.grid(True)
plt.savefig(f'{MODEL_NAME}/plot1_train_histograms.png')
plt.close()

# # ---------------------------------------------------------
# # Plot 2: Isotonic Regression Calibration on Train Set
# #         (a) using the full train set and (b) using a 10% calibration subset (cal_set)
# # ---------------------------------------------------------
# # Define bins for calibration
# bins = np.linspace(0, 1, 41)
# bin_centers = (bins[:-1] + bins[1:]) / 2

# # Helper function to compute binned accuracy
# def compute_binned_accuracy(df, bins):
#     bin_vals = []
#     for i in range(len(bins) - 1):
#         bin_mask = (df['p_true'] >= bins[i]) & (df['p_true'] < bins[i+1])
#         if df[bin_mask].empty:
#             bin_vals.append(np.nan)
#         else:
#             bin_vals.append(df[bin_mask]['correct'].mean())
#     return np.array(bin_vals)

# # (a) Full train set
# bin_values_full = compute_binned_accuracy(train_df, bins)
# valid_full = ~np.isnan(bin_values_full)
# bin_centers_full = bin_centers[valid_full]
# bin_values_full_valid = bin_values_full[valid_full]

# iso_reg_full = IsotonicRegression(out_of_bounds='clip')
# iso_reg_full.fit(bin_centers_full, bin_values_full_valid)
# y_pred_full = iso_reg_full.predict(bin_centers_full)

# # (b) Calibration set: 10% random subset of train set
# cal_set = train_df.sample(frac=0.1, random_state=42)
# bin_values_cal = compute_binned_accuracy(cal_set, bins)
# valid_cal = ~np.isnan(bin_values_cal)
# bin_centers_cal = bin_centers[valid_cal]
# bin_values_cal_valid = bin_values_cal[valid_cal]

# iso_reg_cal = IsotonicRegression(out_of_bounds='clip')
# iso_reg_cal.fit(bin_centers_cal, bin_values_cal_valid)
# y_pred_cal = iso_reg_cal.predict(bin_centers_cal)

# # Create subplots for the two regression fits
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# # Full train set plot
# axs[0].scatter(bin_centers_full, bin_values_full_valid, color='blue', label='Binned Accuracy')
# axs[0].plot(bin_centers_full, y_pred_full, color='red', label='Isotonic Fit')
# axs[0].set_xlabel('Confidence')
# axs[0].set_ylabel('Accuracy')
# axs[0].set_title('Isotonic Regression on Full Train Set')
# axs[0].grid(True)
# axs[0].legend()

# # Calibration set plot
# axs[1].scatter(bin_centers_cal, bin_values_cal_valid, color='green', label='Binned Accuracy')
# axs[1].plot(bin_centers_cal, y_pred_cal, color='red', label='Isotonic Fit')
# axs[1].set_xlabel('Confidence')
# axs[1].set_ylabel('Accuracy')
# axs[1].set_title('Isotonic Regression on Calibration Set (10% of Train)')
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.savefig(f'{MODEL_NAME}/plot2_isotonic_regression.png')
# plt.close()

# # ---------------------------------------------------------
# # Plot 3: Calibration Plot on Test Set: Uncorrected vs. Corrected Scores
# # ---------------------------------------------------------
# # For test set, bin the raw p_true values and compute average correctness
# bins_test = np.linspace(0, 1, 51)
# bin_centers_test = (bins_test[:-1] + bins_test[1:]) / 2
# bin_values_test_uncorr = compute_binned_accuracy(test_df, bins_test)
# valid_test = ~np.isnan(bin_values_test_uncorr)
# bin_centers_test_valid = bin_centers_test[valid_test]
# bin_values_test_uncorr_valid = bin_values_test_uncorr[valid_test]

# # For the corrected scores, use the isotonic regressor (from full train set) to transform the bin centers
# corrected_bin_centers = iso_reg_full.predict(bin_centers_test_valid)

# # Create subplots for uncorrected and corrected calibration on the test set
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# # Uncorrected calibration plot
# axs[0].scatter(bin_centers_test_valid, bin_values_test_uncorr_valid, color='purple', label='Binned Accuracy')
# axs[0].plot([0, 1], [0, 1], 'r--', label='Ideal')
# axs[0].set_xlabel('Raw Confidence (p_true)')
# axs[0].set_ylabel('Accuracy')
# axs[0].set_title('Test Set: Uncorrected Calibration')
# axs[0].grid(True)
# axs[0].legend()

# # Corrected calibration plot
# axs[1].scatter(corrected_bin_centers, bin_values_test_uncorr_valid, color='orange', label='Binned Accuracy (Corrected)')
# axs[1].plot([0, 1], [0, 1], 'r--', label='Ideal')
# axs[1].set_xlabel('Corrected Confidence')
# axs[1].set_ylabel('Accuracy')
# axs[1].set_title('Test Set: Corrected Calibration')
# axs[1].grid(True)
# axs[1].legend()

# plt.tight_layout()
# plt.savefig(f'{MODEL_NAME}/plot3_test_calibration.png')
# plt.close()

# # ---------------------------------------------------------
# # Plot 4: AUC-ROC Curve for Corrected Scores on Test Set
# # ---------------------------------------------------------
# # Compute corrected scores for each test sample using the isotonic regressor from the full train set.
# # (Assuming 'p_true' is the raw self-evaluation score.)
# test_df = test_df.copy()  # To avoid SettingWithCopyWarning
# test_df['corrected_score'] = iso_reg_full.predict(test_df['p_true'])

# # Compute ROC curve and AUC using the corrected scores as the estimator
# fpr, tpr, thresholds = roc_curve(test_df['correct'].astype(int), test_df['corrected_score'])
# roc_auc = auc(fpr, tpr)
# print(f'AUROC: {roc_auc:.4f}')

# plt.figure(figsize=(6, 6))
# plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})', color='darkblue')
# plt.plot([0, 1], [0, 1], linestyle='--', color='orange', label='Chance')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('AUC-ROC Curve for Corrected Scores (Test Set)')
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.savefig(f'{MODEL_NAME}/plot4_auc_roc_curve.png')
# plt.close()