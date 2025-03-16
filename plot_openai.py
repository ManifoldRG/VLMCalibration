import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc
import os
import seaborn as sns

# ---------------------------
# Configuration
# ---------------------------
MODEL_NAME = "gsm8k/openai"  # Directory for saving plots
os.makedirs(MODEL_NAME, exist_ok=True)


# ---------------------------
# Load Data and Remove NaNs
# ---------------------------
def load_and_clean_data(model_variant):
    # Load train data
    train_df = pd.read_csv(f"records_train_full_{model_variant}.csv", index_col=0)
    print(f"\nProcessing {model_variant}")
    print("Original train shape:", train_df.shape)

    # Count and remove NaN rows
    na_train = train_df.isna().any(axis=1).sum()
    print("Number of rows with NaN in train:", na_train)
    train_df = train_df.dropna()
    print("Clean train shape:", train_df.shape)

    return train_df


def analyze_confidence_stats(df):
    """Analyze and print confidence statistics."""
    mean_conf_correct = df[df["correct"] == True]["p_true"].mean()
    mean_conf_incorrect = df[df["correct"] == False]["p_true"].mean()

    print("\nConfidence Statistics:")
    print(f"Mean confidence when correct: {mean_conf_correct:.3f}")
    print(f"Mean confidence when incorrect: {mean_conf_incorrect:.3f}")

    # Additional statistics
    print("\nDetailed Statistics:")
    print(df.groupby("correct")["p_true"].describe())


def plot_confidence_histogram(df, model_name, save_path):
    """Plot histogram of confidence scores for correct vs incorrect predictions."""
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 51)

    # Plot histograms
    plt.hist(
        df[df["correct"] == True]["p_true"],
        bins=bins,
        alpha=0.5,
        density=True,
        label="Correct Predictions",
        color="green",
        edgecolor="black",
    )
    plt.hist(
        df[df["correct"] == False]["p_true"],
        bins=bins,
        alpha=0.5,
        density=True,
        label="Incorrect Predictions",
        color="red",
        edgecolor="black",
    )

    plt.xlabel("Confidence (p_true)")
    plt.ylabel("Density")
    plt.title(f"Distribution of Confidence Scores\n{model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_calibration_curve(df, model_name, save_path):
    """Plot calibration curve with isotonic regression."""
    # Create confidence bins
    bins = np.linspace(0, 1, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate actual accuracies for each bin
    bin_accuracies = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (df["p_true"] >= bins[i]) & (df["p_true"] < bins[i + 1])
        bin_acc = df[mask]["correct"].mean() if mask.sum() > 0 else np.nan
        bin_accuracies.append(bin_acc)
        bin_counts.append(mask.sum())

    bin_accuracies = np.array(bin_accuracies)
    bin_counts = np.array(bin_counts)

    # Fit isotonic regression
    valid_bins = ~np.isnan(bin_accuracies)
    if valid_bins.sum() > 1:  # Need at least 2 points for isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds="clip")
        iso_reg.fit(bin_centers[valid_bins], bin_accuracies[valid_bins])
        calibrated_probs = iso_reg.predict(bin_centers)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Calibration", alpha=0.5)

    # Plot raw calibration points
    plt.scatter(
        bin_centers,
        bin_accuracies,
        label="Observed Calibration",
        alpha=0.6,
        s=100 * np.array(bin_counts) / max(bin_counts),
    )

    # Plot isotonic regression line if available
    if valid_bins.sum() > 1:
        plt.plot(
            bin_centers, calibrated_probs, "g-", label="Isotonic Regression", alpha=0.7
        )

    plt.xlabel("Predicted Probability")
    plt.ylabel("Actual Accuracy")
    plt.title(f"Calibration Curve\n{model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def plot_reliability_diagram(df, model_name, save_path):
    """Plot reliability diagram showing expected vs actual confidence."""
    plt.figure(figsize=(10, 6))

    # Create confidence bins
    n_bins = 10
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate statistics for each bin
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(len(bins) - 1):
        mask = (df["p_true"] >= bins[i]) & (df["p_true"] < bins[i + 1])
        if mask.sum() > 0:
            bin_accuracies.append(df[mask]["correct"].mean())
            bin_confidences.append(df[mask]["p_true"].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accuracies.append(np.nan)
            bin_confidences.append(np.nan)
            bin_counts.append(0)

    # Plot
    plt.bar(
        bin_centers,
        np.abs(np.array(bin_confidences) - np.array(bin_accuracies)),
        width=1 / n_bins,
        alpha=0.3,
        color="red",
        label="Calibration Error",
    )

    plt.plot(bin_centers, bin_accuracies, "bo-", label="Actual Accuracy", alpha=0.7)
    plt.plot(
        bin_centers, bin_confidences, "go-", label="Predicted Confidence", alpha=0.7
    )

    plt.xlabel("Confidence Bin")
    plt.ylabel("Accuracy / Confidence")
    plt.title(f"Reliability Diagram\n{model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()


def main():
    # Process for each model variant
    model_variants = ["gpt-4o", "gpt-4o-mini"]

    for model_variant in model_variants:
        try:
            # Load and process data
            train_df = load_and_clean_data(model_variant)

            # Create model-specific directory
            model_dir = f"{MODEL_NAME}/{model_variant}"
            os.makedirs(model_dir, exist_ok=True)
            print(model_dir)

            # Generate analysis and plots
            analyze_confidence_stats(train_df)

            plot_confidence_histogram(
                train_df, model_variant, f"{model_dir}/confidence_histogram.png"
            )

            plot_calibration_curve(
                train_df, model_variant, f"{model_dir}/calibration_curve.png"
            )

            plot_reliability_diagram(
                train_df, model_variant, f"{model_dir}/reliability_diagram.png"
            )

            # Save processed data
            train_df.to_csv(f"{model_dir}/processed_data.csv")

        except Exception as e:
            print(f"Error processing {model_variant}: {e}")


if __name__ == "__main__":
    main()
